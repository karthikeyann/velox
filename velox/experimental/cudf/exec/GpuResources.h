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

#include "velox/experimental/cudf/exec/GpuMemoryTrace.h"

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Refers to stable owner and PlanNode records in the allocation ledger.
struct GpuMemoryOwnerHandle {
  /// Identifies one concrete operator instance.
  uint64_t ownerId{0};
  /// Identifies the task-local PlanNode aggregate.
  uint64_t planNodeId{0};

  bool operator==(const GpuMemoryOwnerHandle&) const = default;
};

/// Reports logical requested-byte counters for one allocation owner.
struct GpuMemoryOwnerSnapshot {
  /// Stable registered owner handle.
  GpuMemoryOwnerHandle handle;
  /// Full operator identity captured during registration.
  GpuMemoryOwner owner;
  /// Bytes held by live allocations.
  uint64_t currentBytes{0};
  /// Highest number of simultaneously live bytes.
  uint64_t peakBytes{0};
  /// Bytes allocated successfully over the tracked lifetime.
  uint64_t totalBytes{0};
  /// Number of live allocations.
  uint64_t currentAllocations{0};
  /// Number of successful allocations over the tracked lifetime.
  uint64_t totalAllocations{0};
};

/// Describes one live allocation and its allocation-time owner.
struct GpuMemoryAllocationSnapshot {
  /// Allocation address represented as an integer for stable copying.
  uintptr_t address{0};
  /// Requested allocation size in bytes.
  uint64_t bytes{0};
  /// Stable allocation-time owner.
  GpuMemoryOwnerHandle handle;
};

/// Contains a consistent process-wide logical-memory snapshot.
struct GpuMemorySnapshot {
  /// Bytes held by all live tracked allocations.
  uint64_t currentBytes{0};
  /// Process-wide high-water mark.
  uint64_t peakBytes{0};
  /// Bytes allocated successfully over the tracked lifetime.
  uint64_t totalBytes{0};
  /// Number of live allocations.
  uint64_t currentAllocations{0};
  /// Highest number of simultaneously live allocations.
  uint64_t peakAllocations{0};
  /// Number of successful allocations over the tracked lifetime.
  uint64_t totalAllocations{0};
  /// Number of serialized allocation and deallocation transitions.
  uint64_t sequence{0};
  /// Number of accounting events that could not be represented.
  uint64_t dataLossEvents{0};
  /// Per-owner counters.
  std::vector<GpuMemoryOwnerSnapshot> owners;
  /// Live allocations.
  std::vector<GpuMemoryAllocationSnapshot> allocations;
};

/// Serializes pointer ownership and process, PlanNode, and owner counters.
class GpuMemoryAllocationTracker {
 public:
  /// Creates an empty tracker with an explicit unattributed owner.
  GpuMemoryAllocationTracker();

  /// Releases tracker state.
  ~GpuMemoryAllocationTracker();

  GpuMemoryAllocationTracker(const GpuMemoryAllocationTracker&) = delete;
  GpuMemoryAllocationTracker& operator=(const GpuMemoryAllocationTracker&) =
      delete;

  /// Registers an owner and returns its stable handle.
  GpuMemoryOwnerHandle registerOwner(const GpuMemoryOwner& owner);

  /// Records a successful allocation.
  ///
  /// Returns the fully ordered counter transition, or no value for a null
  /// address. Duplicate live addresses are reported as diagnostic data loss.
  std::optional<GpuMemoryTraceUpdate> recordAllocation(
      void* address,
      std::size_t bytes,
      GpuMemoryOwnerHandle handle) noexcept;

  /// Removes a live allocation from its allocation-time owner.
  ///
  /// Returns the fully ordered counter transition. Unknown addresses are
  /// reported as diagnostic data loss.
  std::optional<GpuMemoryTraceUpdate> recordDeallocation(
      void* address) noexcept;

  /// Returns current counters for an allocation failure marker.
  GpuMemoryTraceUpdate currentState(GpuMemoryOwnerHandle handle) const noexcept;

  /// Returns a consistent point-in-time view of the ledger.
  [[nodiscard]] GpuMemorySnapshot snapshot() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/// Owns tracked wrappers for the default and explicit output resources.
///
/// Both wrappers feed one ledger. The distinction is intentionally absent
/// from the MVP trace because simultaneous logical ownership is the useful
/// OOM-debugging signal.
struct GpuMemoryResourcePair {
  cuda::mr::any_resource<cuda::mr::device_accessible> main;
  cuda::mr::any_resource<cuda::mr::device_accessible> output;
};

extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
    output_mr_;

/// Returns the memory resource designated for output vector allocations.
rmm::device_async_resource_ref get_output_mr();

/// Creates two wrappers backed by one allocation-ownership ledger.
[[nodiscard]] GpuMemoryResourcePair createGpuMemoryTrackingResources(
    cuda::mr::any_resource<cuda::mr::device_accessible> mainUpstream,
    cuda::mr::any_resource<cuda::mr::device_accessible> outputUpstream);

/// Clears the globally exposed tracker.
void resetGpuMemoryTracking();

/// Returns the current global diagnostic state or an empty snapshot.
[[nodiscard]] GpuMemorySnapshot getGpuMemorySnapshot();

namespace gpu_memory_detail {

/// Holds the thread-local attribution state replaced by an operator scope.
struct GpuMemoryActiveOwner {
  const void* tracker{nullptr};
  uint64_t ownerId{0};
};

/// Activates an operator and returns the previous thread-local owner.
GpuMemoryActiveOwner activateGpuMemoryOperator(exec::Operator* op) noexcept;

/// Returns the active thread-local attribution.
GpuMemoryActiveOwner activeGpuMemoryOwner() noexcept;

/// Restores a previously active thread-local owner.
void restoreGpuMemoryOwner(GpuMemoryActiveOwner owner) noexcept;

} // namespace gpu_memory_detail

/**
 * @brief Creates a memory resource based on the given mode.
 *
 * @param mode rmm::mr::pool_memory_resource mode.
 * @param percent The initial percent of GPU memory to allocate for memory
 * resource.
 */
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
createMemoryResource(std::string_view mode, int percent);

/// Returns the global CUDA stream pool used by cuDF.
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

} // namespace facebook::velox::cudf_velox
