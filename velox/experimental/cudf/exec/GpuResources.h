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

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox {
class BaseRuntimeStatWriter;
}

namespace facebook::velox::cudf_velox {

/// Identifies the logical purpose of a tracked GPU memory resource.
enum class GpuMemoryResourceKind {
  /// Identifies the main resource used for temporary allocations.
  kMain,
  /// Identifies the resource passed explicitly for output allocations.
  kOutput,
};

/// Identifies the operator that was active when an allocation succeeded.
struct GpuMemoryOwner {
  /// Universally unique Velox task identifier captured at allocation time.
  std::string taskUuid;
  /// Velox task identifier captured at allocation time.
  std::string taskId;
  /// Velox query identifier captured at allocation time.
  std::string queryId;
  /// Velox plan node identifier captured at allocation time.
  std::string planNodeId;
  /// Velox operator identifier captured at allocation time.
  int32_t operatorId{-1};
  /// Velox operator type captured at allocation time.
  std::string operatorType;

  /// Returns true when all stable owner fields match.
  bool operator==(const GpuMemoryOwner&) const = default;
};

/// Reports query-scoped high-water marks raised by one allocation.
struct GpuMemoryPeakUpdate {
  /// New combined query peak, or no value if the peak did not increase.
  std::optional<uint64_t> queryPeakBytes;
  /// New query peak for the allocation's resource kind.
  std::optional<uint64_t> queryResourcePeakBytes;
  /// New combined PlanNode peak, or no value if the peak did not increase.
  std::optional<uint64_t> planNodePeakBytes;
  /// New PlanNode peak for the allocation's resource kind.
  std::optional<uint64_t> planNodeResourcePeakBytes;
  /// New combined operator peak, or no value if the peak did not increase.
  std::optional<uint64_t> operatorPeakBytes;
  /// New operator peak for the allocation's resource kind.
  std::optional<uint64_t> operatorResourcePeakBytes;
};

/// Writes newly raised query-scoped peaks as byte-valued RuntimeStats.
///
/// A null writer and absent peak values are ignored.
void addGpuMemoryPeakRuntimeStats(
    BaseRuntimeStatWriter* writer,
    GpuMemoryResourceKind kind,
    const GpuMemoryPeakUpdate& update) noexcept;

/// Reports current and cumulative counters for one logical resource.
struct GpuMemoryResourceSnapshot {
  /// Logical resource represented by these counters.
  GpuMemoryResourceKind kind;
  /// Bytes held by live allocations.
  uint64_t currentBytes;
  /// Highest number of simultaneously live bytes.
  uint64_t peakBytes;
  /// Bytes allocated successfully over the resource lifetime.
  uint64_t totalBytes;
  /// Number of live allocations.
  uint64_t currentAllocations;
  /// Highest number of simultaneously live allocations.
  uint64_t peakAllocations;
  /// Number of successful allocations over the resource lifetime.
  uint64_t totalAllocations;
};

/// Reports allocation counters for one owner and logical resource.
struct GpuMemoryOwnerSnapshot {
  /// Stable allocation owner.
  GpuMemoryOwner owner;
  /// Logical resource used by the owner.
  GpuMemoryResourceKind kind;
  /// Bytes currently owned by live allocations.
  uint64_t currentBytes;
  /// Highest number of simultaneously owned bytes.
  uint64_t peakBytes;
  /// Bytes successfully allocated over the tracked lifetime.
  uint64_t totalBytes;
  /// Number of allocations currently owned.
  uint64_t currentAllocations;
  /// Highest number of simultaneously owned allocations.
  uint64_t peakAllocations;
  /// Number of successful allocations over the tracked lifetime.
  uint64_t totalAllocations;
};

/// Describes one live GPU allocation.
struct GpuMemoryAllocationSnapshot {
  /// Allocation address represented as an integer for stable copying.
  uintptr_t address;
  /// Requested allocation size in bytes.
  uint64_t bytes;
  /// Logical resource that owns the allocation.
  GpuMemoryResourceKind kind;
  /// Stable owner captured when the allocation succeeded.
  GpuMemoryOwner owner;
};

/// Contains a point-in-time view of tracked GPU memory.
struct GpuMemorySnapshot {
  /// Per-resource statistics.
  std::vector<GpuMemoryResourceSnapshot> resources;
  /// Per-owner statistics.
  std::vector<GpuMemoryOwnerSnapshot> owners;
  /// Live allocations.
  std::vector<GpuMemoryAllocationSnapshot> allocations;
};

/// Thread-safe allocation state used by the diagnostic resource wrappers.
class GpuMemoryAllocationTracker {
 public:
  /// Creates an empty tracker.
  GpuMemoryAllocationTracker();

  /// Releases tracker state.
  ~GpuMemoryAllocationTracker();

  /// Prevents copying tracker state and its synchronization primitive.
  GpuMemoryAllocationTracker(const GpuMemoryAllocationTracker&) = delete;

  /// Prevents replacing tracker state by copy assignment.
  GpuMemoryAllocationTracker& operator=(const GpuMemoryAllocationTracker&) =
      delete;

  /// Records a successful allocation and returns newly raised owner peaks.
  GpuMemoryPeakUpdate recordAllocation(
      void* address,
      std::size_t bytes,
      GpuMemoryResourceKind kind,
      const GpuMemoryOwner& owner) noexcept;

  /// Removes a live allocation from its allocation-time owner.
  void recordDeallocation(void* address) noexcept;

  /// Retires zero-live owner history for a completed task UUID.
  ///
  /// Owners that still have live allocations are removed by their final
  /// deallocation.
  void retireTask(std::string_view taskUuid);

  /// Retires query aggregates after the owning QueryCtx is destroyed.
  ///
  /// Aggregates with live allocations remain until final deallocation.
  void retireQuery(const std::string& queryId);

  /// Returns a consistent snapshot of resource, owner, and allocation state.
  [[nodiscard]] GpuMemorySnapshot snapshot() const;

 private:
  /// Holds synchronized implementation state.
  class Impl;

  /// Owns the synchronized implementation.
  std::unique_ptr<Impl> impl_;
};

/// Owns the two diagnostic resource wrapper chains.
struct GpuMemoryResourcePair {
  /// Main resource wrapper used as the cuDF default resource.
  cuda::mr::any_resource<cuda::mr::device_accessible> main;
  /// Separate output wrapper, even when both wrappers share one upstream.
  cuda::mr::any_resource<cuda::mr::device_accessible> output;
};

extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
    output_mr_;

/// Returns the memory resource designated for output vector allocations.
rmm::device_async_resource_ref get_output_mr();

/// Returns a stable name for a logical GPU memory resource.
std::string_view gpuMemoryResourceKindString(GpuMemoryResourceKind kind);

/// Creates separate tracked main and output wrappers and activates snapshots.
///
/// The returned wrappers own all callback state. The caller must retain them
/// while GPU allocations can use either resource.
[[nodiscard]] GpuMemoryResourcePair createGpuMemoryTrackingResources(
    cuda::mr::any_resource<cuda::mr::device_accessible> mainUpstream,
    cuda::mr::any_resource<cuda::mr::device_accessible> outputUpstream);

/// Clears the globally exposed diagnostic snapshot state.
void resetGpuMemoryTracking();

/// Returns the current global diagnostic state or an empty snapshot.
[[nodiscard]] GpuMemorySnapshot getGpuMemorySnapshot();

/// Retires zero-live owner history for a completed task UUID.
///
/// Owners with live allocations remain visible until final deallocation.
void retireGpuMemoryTask(std::string_view taskUuid);

/// Logs global counters and the largest live owners and allocations.
void logGpuMemorySnapshot(
    std::string_view context,
    std::size_t maxEntries = 10);

/**
 * @brief Creates a memory resource based on the given mode.
 *
 * @param mode rmm::mr::pool_memory_resource mode.
 * @param percent The initial percent of GPU memory to allocate for memory
 * resource.
 */
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
createMemoryResource(std::string_view mode, int percent);

/**
 * @brief Returns the global CUDA stream pool used by cudf.
 */
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

} // namespace facebook::velox::cudf_velox
