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

#include "velox/experimental/cudf/exec/GpuMemoryCapture.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace facebook::velox::exec {
class Operator;
}

namespace facebook::velox::cudf_velox {

/// Identifies one concrete operator instance that originated an allocation.
struct GpuMemoryOwner {
  /// Universally unique Velox task identifier.
  std::string taskUuid;
  /// User-visible Velox task identifier.
  std::string taskId;
  /// Velox query identifier.
  std::string queryId;
  /// Velox plan node identifier.
  std::string planNodeId;
  /// Display-only Velox PlanNode type. Does not affect attribution identity.
  std::string planNodeType;
  /// Pipeline containing the operator instance.
  int32_t pipelineId{-1};
  /// Driver executing the operator instance.
  int32_t driverId{-1};
  /// Operator position within the pipeline.
  int32_t operatorId{-1};
  /// User-visible operator implementation name.
  std::string operatorType;

  /// Compares stable allocation identity and ignores display-only metadata.
  bool operator==(const GpuMemoryOwner& other) const {
    return taskUuid == other.taskUuid && taskId == other.taskId &&
        queryId == other.queryId && planNodeId == other.planNodeId &&
        pipelineId == other.pipelineId && driverId == other.driverId &&
        operatorId == other.operatorId && operatorType == other.operatorType;
  }
};

/// Describes one fully ordered logical-memory transition.
struct GpuMemoryTraceUpdate {
  /// Strictly increasing timestamp shared by all counters for this transition.
  uint64_t timestampNs{0};
  /// Strictly increasing event number used for diagnostics.
  uint64_t sequence{0};
  /// Registered allocation-origin owner.
  uint64_t ownerId{0};
  /// Registered task-local PlanNode aggregate.
  uint64_t planNodeId{0};
  /// Process-wide live logical bytes after this transition.
  uint64_t globalCurrentBytes{0};
  /// Process-wide high-water mark after this transition.
  uint64_t globalPeakBytes{0};
  /// Query live logical bytes after this transition.
  uint64_t queryCurrentBytes{0};
  /// Task, or PlanFragment instance, live logical bytes after this transition.
  uint64_t taskCurrentBytes{0};
  /// PlanNode live logical bytes after this transition.
  uint64_t planNodeCurrentBytes{0};
  /// Operator-instance live logical bytes after this transition.
  uint64_t ownerCurrentBytes{0};
  /// Signed allocation delta applied by this transition.
  int64_t deltaBytes{0};
};

/// Starts a process-wide streamed Perfetto trace.
///
/// Replaces `%p` in 'pathPattern' with the process ID. Returns false when the
/// trace cannot be started; tracing failures never change query behavior.
/// Call once before creating the tracked resources or registering owners.
bool startGpuMemoryTrace(std::string_view pathPattern) noexcept;

/// Flushes and stops the trace after operator execution has quiesced.
void stopGpuMemoryTrace() noexcept;

/// Returns true while the process-wide Perfetto session is accepting events.
bool gpuMemoryTraceEnabled() noexcept;

/// Returns the resolved trace file path, or an empty string when disabled.
std::string gpuMemoryTracePath();

/// Adds an analysis-visible instant marker to the active operator lane.
void markGpuMemoryTrace(std::string_view name) noexcept;

/// Establishes allocation attribution and a Perfetto slice for one call.
class GpuMemoryOperatorCall {
 public:
  GpuMemoryOperatorCall(exec::Operator* op, std::string_view callName) noexcept;
  ~GpuMemoryOperatorCall();

  GpuMemoryOperatorCall(const GpuMemoryOperatorCall&) = delete;
  GpuMemoryOperatorCall& operator=(const GpuMemoryOperatorCall&) = delete;

 private:
  const void* previousTracker_{nullptr};
  uint64_t ownerId_{0};
  uint64_t previousOwnerId_{0};
  GpuMemoryCaptureCallHandle captureCall_;
  bool traceSliceStarted_{false};
};

namespace gpu_memory_detail {

/// Returns the transport-neutral source monotonic timestamp.
uint64_t gpuMemoryTraceNowNs() noexcept;

/// Registers stable hierarchy and metadata for one allocation owner.
void registerGpuMemoryTraceOwner(
    uint64_t ownerId,
    uint64_t planNodeId,
    const GpuMemoryOwner& owner) noexcept;

/// Emits synchronized process, query, task, PlanNode, and operator samples.
void emitGpuMemoryTraceUpdate(const GpuMemoryTraceUpdate& update) noexcept;

/// Ends and clears the native NVTX counter hierarchy.
void resetGpuMemoryNvtxCounters() noexcept;

/// Emits a factual allocation-failure marker and current logical state.
void emitGpuMemoryTraceOom(
    uint64_t ownerId,
    std::size_t requestedBytes,
    uint64_t globalCurrentBytes,
    uint64_t globalPeakBytes,
    uint64_t planNodeCurrentBytes,
    uint64_t ownerCurrentBytes,
    std::size_t cudaFreeBytes,
    std::size_t cudaTotalBytes,
    std::string_view cudaStatus) noexcept;

/// Emits a diagnostic marker when source accounting loses an event.
void emitGpuMemoryTraceDataLoss(
    std::string_view reason,
    uint64_t sequence) noexcept;

/// Begins an operator-call slice on its registered lane.
bool beginGpuMemoryOperatorCall(
    uint64_t ownerId,
    std::string_view callName) noexcept;

/// Ends an operator-call slice on its registered lane.
void endGpuMemoryOperatorCall(uint64_t ownerId) noexcept;

} // namespace gpu_memory_detail

} // namespace facebook::velox::cudf_velox
