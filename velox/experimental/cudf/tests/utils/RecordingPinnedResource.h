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

#include <rmm/aligned.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>

namespace facebook::velox::cudf_velox::test {

/// A pinned memory resource that records what is allocated from it, and can
/// fail an allocation. The member functions have the names cuda::mr resources
/// must have.
class RecordingPinnedResource {
 public:
  /// 'failingAllocation' is the number, from one, of the allocation that
  /// throws std::bad_alloc; zero for none.
  explicit RecordingPinnedResource(
      rmm::host_device_async_resource_ref upstream,
      size_t failingAllocation = 0)
      : upstream_(upstream), failingAllocation_(failingAllocation) {}

  void* allocate_sync(
      size_t bytes,
      size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      if (++state_->numRequests == failingAllocation_) {
        throw std::bad_alloc();
      }
    }
    auto* data = upstream_.allocate_sync(bytes, alignment);
    std::lock_guard<std::mutex> lock(state_->mutex);
    ++state_->numAllocations;
    ++state_->numAllocationsBySize[bytes];
    state_->bytesInUse += bytes;
    state_->maxBytesInUse = std::max(state_->maxBytesInUse, state_->bytesInUse);
    return data;
  }

  void* allocate(
      cuda::stream_ref /*stream*/,
      size_t bytes,
      size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) {
    return allocate_sync(bytes, alignment);
  }

  void deallocate_sync(
      void* data,
      size_t bytes,
      size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    upstream_.deallocate_sync(data, bytes, alignment);
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->bytesInUse -= bytes;
  }

  void deallocate(
      cuda::stream_ref /*stream*/,
      void* data,
      size_t bytes,
      size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept {
    deallocate_sync(data, bytes, alignment);
  }

  bool operator==(const RecordingPinnedResource& other) const {
    return state_ == other.state_;
  }

  bool operator!=(const RecordingPinnedResource& other) const {
    return !(*this == other);
  }

  size_t numAllocations() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->numAllocations;
  }

  /// Returns the number of allocations of exactly 'bytes'.
  size_t numAllocationsOf(size_t bytes) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto it = state_->numAllocationsBySize.find(bytes);
    return it == state_->numAllocationsBySize.end() ? 0 : it->second;
  }

  size_t bytesInUse() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->bytesInUse;
  }

  size_t maxBytesInUse() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->maxBytesInUse;
  }

 private:
  struct State {
    std::mutex mutex;
    size_t numRequests{0};
    size_t numAllocations{0};
    std::unordered_map<size_t, size_t> numAllocationsBySize;
    size_t bytesInUse{0};
    size_t maxBytesInUse{0};
  };

  rmm::host_device_async_resource_ref upstream_;
  size_t failingAllocation_;
  std::shared_ptr<State> state_{std::make_shared<State>()};
};

inline void get_property(
    const RecordingPinnedResource&,
    cuda::mr::device_accessible) noexcept {}

inline void get_property(
    const RecordingPinnedResource&,
    cuda::mr::host_accessible) noexcept {}

} // namespace facebook::velox::cudf_velox::test
