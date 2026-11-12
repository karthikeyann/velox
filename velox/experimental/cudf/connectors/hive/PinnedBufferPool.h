/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Bounded pool of equally sized page-locked host buffers for staging bytes
/// on their way to the device. Each buffer comes with a CUDA stream and event
/// of its own, so that a thread can copy its buffer to the device and wait for
/// just that copy. A buffer is page-locked the first time the pool needs it and
/// stays locked until the pool is destroyed, so steady-state users pay for
/// neither allocation nor page faults. Thread-safe.
class PinnedBufferPool {
 public:
  /// A page-locked buffer and the stream that copies it to the device.
  struct StagingBuffer {
    /// Start of the buffer, which is bufferSize() bytes long.
    uint8_t* data{nullptr};
    /// Stream used only for copies out of this buffer.
    cudaStream_t stream{nullptr};
    /// Event for waiting on work queued on 'stream'. Waiting on it blocks the
    /// thread, where waiting on the stream itself can spin.
    cudaEvent_t event{nullptr};
  };

  /// Returns a buffer to the pool it was checked out of.
  class Releaser {
   public:
    /// Creates a releaser for buffers of 'pool'.
    explicit Releaser(PinnedBufferPool* pool = nullptr) : pool_(pool) {}

    /// Returns 'buffer' to the pool.
    void operator()(StagingBuffer* buffer) const {
      pool_->release(buffer);
    }

   private:
    PinnedBufferPool* pool_;
  };

  /// A buffer checked out of a pool. Returns the buffer to its pool when
  /// destroyed, by which time no copy may still be running on its stream.
  using Buffer = std::unique_ptr<StagingBuffer, Releaser>;

  /// @param bufferSize Size of every buffer in bytes. Must be positive.
  /// @param capacity Upper bound on the bytes the pool page-locks. Rounded
  /// down to a whole number of buffers, but never below one buffer.
  PinnedBufferPool(size_t bufferSize, size_t capacity);

  /// Waits until every buffer has been returned, then unlocks and frees them.
  ~PinnedBufferPool();

  PinnedBufferPool(const PinnedBufferPool&) = delete;
  PinnedBufferPool& operator=(const PinnedBufferPool&) = delete;

  /// Returns the size of every buffer in bytes.
  size_t bufferSize() const {
    return bufferSize_;
  }

  /// Returns the number of buffers the pool may page-lock.
  size_t maxBuffers() const;

  /// Returns the number of buffers page-locked so far.
  size_t numAllocated() const;

  /// Checks a buffer out of the pool. When every buffer is checked out and
  /// the pool is at capacity, waits for one to be returned. A failure to
  /// page-lock more memory caps the pool at the buffers it already has; it
  /// throws only when the pool has none.
  Buffer acquire();

 private:
  // Page-locks a buffer and creates its stream and event. Returns nullptr,
  // with 'status' set, when any of that fails.
  std::unique_ptr<StagingBuffer> allocate(cudaError_t& status) const;

  // Puts 'buffer' back on the free list and wakes the waiters.
  void release(StagingBuffer* buffer);

  const size_t bufferSize_;

  mutable std::mutex mutex_;
  // Signalled when a buffer is returned or an allocation attempt fails.
  std::condition_variable available_;
  // Upper bound on 'numAllocated_'. Lowered when page-locking fails.
  size_t maxBuffers_;
  // Buffers page-locked or being page-locked.
  size_t numAllocated_{0};
  // Buffers page-locked and not checked out. Owned by the pool.
  std::vector<StagingBuffer*> free_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
