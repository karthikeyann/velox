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

#include "velox/experimental/cudf/connectors/hive/PinnedBufferPool.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"

#include <cuda_runtime.h>

#include <glog/logging.h>

#include <algorithm>
#include <exception>

namespace facebook::velox::cudf_velox::connector::hive {

PinnedBufferPool::PinnedBufferPool(size_t bufferSize, size_t capacity)
    : bufferSize_(bufferSize) {
  VELOX_CHECK_GT(bufferSize_, 0, "Pinned buffer size must be positive");
  maxBuffers_ = std::max<size_t>(1, capacity / bufferSize_);
}

PinnedBufferPool::~PinnedBufferPool() {
  std::unique_lock<std::mutex> lock(mutex_);
  available_.wait(lock, [&]() { return free_.size() == numAllocated_; });
  if (free_.empty()) {
    return;
  }
  bool hasContext{true};
  try {
    ensureCudaContextForThread();
  } catch (const std::exception&) {
    // The CUDA runtime is gone when the pool outlives it at process exit, and
    // it has released the buffers, streams and events itself.
    hasContext = false;
  }
  for (auto* buffer : free_) {
    if (hasContext) {
      cudaEventDestroy(buffer->event);
      cudaStreamDestroy(buffer->stream);
      cudaFreeHost(buffer->data);
    }
    delete buffer;
  }
  // Clear any error so that an unrelated CUDA call does not report it.
  cudaGetLastError();
}

size_t PinnedBufferPool::maxBuffers() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return maxBuffers_;
}

size_t PinnedBufferPool::numAllocated() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return numAllocated_;
}

std::unique_ptr<PinnedBufferPool::StagingBuffer> PinnedBufferPool::allocate(
    cudaError_t& status) const {
  ensureCudaContextForThread();
  auto buffer = std::make_unique<StagingBuffer>();
  void* data{nullptr};
  status = cudaHostAlloc(&data, bufferSize_, cudaHostAllocDefault);
  if (status == cudaSuccess) {
    buffer->data = static_cast<uint8_t*>(data);
    status = cudaStreamCreateWithFlags(&buffer->stream, cudaStreamNonBlocking);
    if (status == cudaSuccess) {
      status = cudaEventCreateWithFlags(
          &buffer->event, cudaEventBlockingSync | cudaEventDisableTiming);
      if (status == cudaSuccess) {
        return buffer;
      }
      cudaStreamDestroy(buffer->stream);
    }
    cudaFreeHost(data);
  }
  // Clear the error so that an unrelated CUDA call does not report it.
  cudaGetLastError();
  return nullptr;
}

PinnedBufferPool::Buffer PinnedBufferPool::acquire() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    available_.wait(
        lock, [&]() { return !free_.empty() || numAllocated_ < maxBuffers_; });
    if (!free_.empty()) {
      auto* buffer = free_.back();
      free_.pop_back();
      return Buffer(buffer, Releaser(this));
    }

    // Page-locking is slow, so it runs outside the lock. The slot is reserved
    // first to keep concurrent callers within the capacity.
    ++numAllocated_;
    lock.unlock();
    cudaError_t status{cudaSuccess};
    auto buffer = allocate(status);
    if (buffer != nullptr) {
      return Buffer(buffer.release(), Releaser(this));
    }

    lock.lock();
    --numAllocated_;
    // Callers waiting for the reserved slot must re-evaluate.
    available_.notify_all();
    VELOX_CHECK_GT(
        numAllocated_,
        0,
        "Failed to page-lock a host buffer: {}. Buffer size: {}",
        cudaGetErrorString(status),
        bufferSize_);
    if (maxBuffers_ > numAllocated_) {
      LOG(WARNING) << "Failed to page-lock a host buffer: "
                   << cudaGetErrorString(status) << ". Capping the pool at "
                   << numAllocated_ << " buffers of " << bufferSize_
                   << " bytes.";
      maxBuffers_ = numAllocated_;
    }
  }
}

void PinnedBufferPool::release(StagingBuffer* buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  free_.push_back(buffer);
  // Notifies under the lock: the destructor may free the pool as soon as the
  // lock is dropped.
  available_.notify_all();
}

} // namespace facebook::velox::cudf_velox::connector::hive
