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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox {

struct GpuMemoryOwner;
struct GpuMemoryOperatorCounts;
struct GpuMemorySnapshot;
struct GpuMemoryTraceUpdate;

/// Configures one-at-a-time, task-bounded GPU-memory captures.
struct GpuMemoryCaptureConfig {
  /// Writes the versioned raw capture after the selected task completes.
  std::string pathPattern;
  /// Selects a task whose query ID, task ID, or UUID contains this text.
  std::string queryFilter;
  /// Bounds high-volume memory updates and operator calls retained per capture.
  std::size_t maxEvents{250'000};
};

/// Identifies the Velox task selected for a capture.
struct GpuMemoryCaptureTask {
  /// Universally unique Velox task identifier.
  std::string taskUuid;
  /// User-visible Velox task identifier.
  std::string taskId;
  /// Velox query identifier.
  std::string queryId;
};

/// Describes one PlanNode in the selected task's plan.
struct GpuMemoryCapturePlanNode {
  /// Velox PlanNode identifier.
  std::string id;
  /// User-visible Velox PlanNode type.
  std::string type;
  /// Source PlanNode identifiers in plan order.
  std::vector<std::string> sourceIds;
};

/// Carries allocation-free state between one operator call's begin and end.
struct GpuMemoryCaptureCallHandle {
  /// Identifies the capture so late destructors cannot enter a later task.
  uint64_t captureId{0};
  /// Uniquely identifies the call within the worker process.
  uint64_t callId{0};
  /// Identifies the allocation owner active for the call.
  uint64_t ownerId{0};
  /// Uses the source monotonic clock shared by memory transitions.
  uint64_t startTimestampNs{0};
  /// Records the operating-system thread ID for profiler correlation.
  int64_t threadId{0};
  /// Identifies the preallocated active-call slot owned by this handle.
  uint32_t openSlot{0};
  /// Stores the bounded call name without allocating on the execution path.
  std::array<char, 32> callName{};
  /// Indicates that the begin event belongs to the active capture.
  bool active{false};
};

/// Starts the process-wide capture service. An empty path leaves it disabled.
bool startGpuMemoryCapture(const GpuMemoryCaptureConfig& config) noexcept;

/// Flushes completed captures and stops the background exporter.
void stopGpuMemoryCapture() noexcept;

/// Returns true while a task can be selected for capture.
bool gpuMemoryCaptureEnabled() noexcept;

/// Returns true when a task with this identity could still be recorded.
///
/// Lets a listener factory decline a task before any of its drivers pay for
/// observation, which is the difference between profiling one task and slowing
/// every task in the worker to profile one.
bool gpuMemoryCaptureWouldRecord(
    const std::string& taskId,
    const std::string& taskUuid) noexcept;

namespace gpu_memory_detail {

/// Returns true while one task capture is actively recording.
bool gpuMemoryCaptureActive() noexcept;

/// Returns monotonic nanoseconds independent of a visualization backend.
uint64_t gpuMemoryMonotonicTimeNs() noexcept;

/// Atomically selects a task and records its initial ledger snapshot.
bool tryBeginGpuMemoryCapture(
    const GpuMemoryCaptureTask& task,
    const std::vector<GpuMemoryCapturePlanNode>& planNodes,
    GpuMemorySnapshot initialSnapshot) noexcept;

/// Seals the matching task at its terminal ledger sequence watermark.
void finishGpuMemoryCapture(
    const std::string& taskUuid,
    const std::string& taskId,
    std::string_view taskState,
    bool cleanupComplete,
    GpuMemorySnapshot finalSnapshot) noexcept;

/// Seals an active capture as incomplete during worker shutdown.
void abortGpuMemoryCapture(
    std::string_view reason,
    GpuMemorySnapshot finalSnapshot) noexcept;

/// Registers owner metadata used by updates and call spans.
void registerGpuMemoryCaptureOwner(
    uint64_t ownerId,
    uint64_t planNodeId,
    const GpuMemoryOwner& owner) noexcept;

/// Records one process-wide logical-memory transition.
void recordGpuMemoryCaptureUpdate(const GpuMemoryTraceUpdate& update) noexcept;

/// Atomically records owner metadata with its logical-memory transition.
void recordGpuMemoryCaptureUpdate(
    const GpuMemoryTraceUpdate& update,
    const GpuMemoryOwner& owner) noexcept;

/// Records an operator-call begin and returns its allocation-free handle.
GpuMemoryCaptureCallHandle beginGpuMemoryCaptureOperatorCall(
    uint64_t ownerId,
    uint64_t planNodeId,
    const GpuMemoryOwner& owner,
    std::string_view callName) noexcept;

/// Records a completed operator-call span.
void endGpuMemoryCaptureOperatorCall(
    const GpuMemoryCaptureCallHandle& handle) noexcept;

/// Opens a blocked interval for one operator instance.
///
/// Keyed by owner rather than carried in a handle because a driver reports
/// blocked on one thread and resumes on whichever thread next picks it up.
void beginGpuMemoryCaptureBlockedSpan(
    uint64_t ownerId,
    uint64_t planNodeId,
    const GpuMemoryOwner& owner,
    std::string_view blockingReason) noexcept;

/// Closes the blocked interval opened for 'ownerId'.
void endGpuMemoryCaptureBlockedSpan(uint64_t ownerId) noexcept;

/// Records the row, byte and batch counts one operator instance finished with.
void recordGpuMemoryCaptureOperatorCounts(
    uint64_t ownerId,
    uint64_t planNodeId,
    const GpuMemoryOwner& owner,
    const GpuMemoryOperatorCounts& counts) noexcept;

/// Records a factual allocation-failure event.
void recordGpuMemoryCaptureOom(
    uint64_t timestampNs,
    uint64_t sourceSequence,
    uint64_t ownerId,
    uint64_t planNodeId,
    const GpuMemoryOwner& owner,
    std::size_t requestedBytes,
    uint64_t globalCurrentBytes,
    uint64_t globalPeakBytes,
    uint64_t planNodeCurrentBytes,
    uint64_t ownerCurrentBytes,
    std::size_t cudaFreeBytes,
    std::size_t cudaTotalBytes,
    std::string_view cudaStatus) noexcept;

/// Records a source-accounting or capture-integrity failure.
void recordGpuMemoryCaptureDataLoss(
    std::string_view reason,
    uint64_t sourceSequence) noexcept;

} // namespace gpu_memory_detail

} // namespace facebook::velox::cudf_velox
