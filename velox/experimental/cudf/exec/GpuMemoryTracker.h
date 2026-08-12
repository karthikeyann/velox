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

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::exec {
class Operator;
}

namespace facebook::velox::memory {
class MemoryPool;
}

namespace facebook::velox::cudf_velox {

using GpuMemoryResource = cuda::mr::any_resource<cuda::mr::device_accessible>;

/// Stable identity for one physical operator instance. taskUuid distinguishes
/// repeated task IDs and driverId distinguishes parallel instances.
struct GpuMemoryOwner {
  std::string queryId;
  std::string taskId;
  std::string taskUuid;
  std::string planNodeId;
  int32_t pipelineId{-1};
  int32_t driverId{-1};
  int32_t operatorId{-1};
  std::string operatorType;

  bool operator==(const GpuMemoryOwner&) const = default;
};

struct GpuMemoryOwnerSnapshot {
  GpuMemoryOwner owner;
  uint64_t currentBytes{0};
  uint64_t peakBytes{0};
  uint64_t cumulativeRequestedBytes{0};
  uint64_t allocations{0};
  uint64_t frees{0};
};

struct GpuMemorySnapshot {
  uint64_t capacityBytes{0};
  uint64_t currentBytes{0};
  uint64_t peakBytes{0};
  uint64_t cumulativeRequestedBytes{0};
  std::vector<GpuMemoryOwnerSnapshot> owners;
};

struct GpuMemoryResourceRefs {
  rmm::device_async_resource_ref temp;
  rmm::device_async_resource_ref output;
};

struct GpuMemoryResourcePair {
  GpuMemoryResource temp;
  GpuMemoryResource output;
  std::shared_ptr<void> keepAlive;
};

/// Returns per-owner resources for the operator currently selected by
/// RuntimeStatWriterScopeGuard. The wrappers retain allocation-time identity,
/// so a later free on another thread remains charged to the right operator.
GpuMemoryResourceRefs gpuMemoryResourcesForCurrentOperator(
    GpuMemoryResource& tempUpstream,
    GpuMemoryResource& outputUpstream);

/// Looks up resources pre-bound to an operator's host pool. Connector data
/// sources use this on preload threads where no Driver scope is active.
std::optional<GpuMemoryResourceRefs> gpuMemoryResourcesForOperatorPool(
    memory::MemoryPool* operatorPool,
    GpuMemoryResource& tempUpstream,
    GpuMemoryResource& outputUpstream);

/// Pre-registers one operator so NVTX metadata is created off the allocation
/// path. Operators not pre-registered are still resolved lazily.
bool registerGpuMemoryOperator(exec::Operator* op) noexcept;

/// Invalidates the thread-local operator-to-resource cache. Call when a new
/// Driver has been built because allocator addresses can be reused.
void discardGpuMemoryOwnerCache() noexcept;

/// Test entry point using the same owning resource and accounting path as
/// production without requiring a Driver.
GpuMemoryResourcePair createGpuMemoryTrackingResources(
    GpuMemoryResource tempUpstream,
    GpuMemoryResource outputUpstream,
    const GpuMemoryOwner& owner);

GpuMemorySnapshot getGpuMemorySnapshot();

/// Retires the current profiling session when no tracked allocations remain.
/// Returns false and retains the session if a resource is still live.
bool resetGpuMemoryTracking() noexcept;

} // namespace facebook::velox::cudf_velox
