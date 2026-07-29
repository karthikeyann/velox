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

#include "velox/experimental/cudf/exec/GpuMemoryCapture.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/NvtxGpuMemoryCounters.h"

#include <folly/dynamic.h>
#include <folly/json.h>
#include <folly/system/ThreadId.h>
#include <glog/logging.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace gpu_memory_detail {
/// Defined below. Declared here because it anchors replay into wall-clock time
/// at capture boundaries, which the recorders reach before the definition.
uint64_t gpuMemoryUnixTimeNs() noexcept;
} // namespace gpu_memory_detail

namespace {

constexpr std::string_view kCaptureFormat{"velox-cudf-gpu-memory-capture"};
constexpr uint64_t kCaptureVersion{1};
constexpr std::size_t kMaximumActiveCalls{4'096};
constexpr std::size_t kCriticalEventCapacity{64};

struct CapturedOwner {
  uint64_t ownerId{0};
  uint64_t planNodeId{0};
  GpuMemoryOwner owner;
};

struct CapturedMemoryUpdate {
  uint64_t eventSequence{0};
  GpuMemoryTraceUpdate update;
};

struct CapturedCallSpan {
  uint64_t eventSequence{0};
  uint64_t callId{0};
  uint64_t ownerId{0};
  int64_t threadId{0};
  uint64_t startTimestampNs{0};
  uint64_t endTimestampNs{0};
  std::array<char, 32> callName{};
  bool truncated{false};
};

struct ActiveCallSlot {
  GpuMemoryCaptureCallHandle handle;
  bool active{false};
};

struct CapturedOom {
  uint64_t eventSequence{0};
  uint64_t timestampNs{0};
  uint64_t sourceSequence{0};
  uint64_t ownerId{0};
  uint64_t requestedBytes{0};
  uint64_t globalCurrentBytes{0};
  uint64_t sourceLifetimeGlobalPeakBytes{0};
  uint64_t planNodeCurrentBytes{0};
  uint64_t ownerCurrentBytes{0};
  uint64_t cudaFreeBytes{0};
  uint64_t cudaTotalBytes{0};
  std::string cudaStatus;
};

struct CapturedDataLoss {
  uint64_t eventSequence{0};
  uint64_t timestampNs{0};
  uint64_t sourceSequence{0};
  std::string reason;
};

struct CaptureData {
  uint64_t captureId{0};
  GpuMemoryCaptureTask task;
  std::vector<GpuMemoryCapturePlanNode> planNodes;
  GpuMemorySnapshot initialSnapshot;
  GpuMemorySnapshot finalSnapshot;
  std::map<uint64_t, CapturedOwner> owners;
  std::set<uint64_t> requiredOwnerIds;
  std::vector<CapturedMemoryUpdate> memoryUpdates;
  std::vector<CapturedCallSpan> operatorCalls;
  std::vector<CapturedOom> oomEvents;
  std::vector<CapturedDataLoss> dataLossEvents;
  std::vector<ActiveCallSlot> activeCallSlots;
  std::vector<uint32_t> freeCallSlots;
  uint64_t startTimestampNs{0};
  uint64_t startUnixNs{0};
  uint64_t endTimestampNs{0};
  uint64_t endUnixNs{0};
  uint64_t nextEventSequence{0};
  uint64_t retainedEvents{0};
  uint64_t droppedEvents{0};
  uint64_t internalDataLossEvents{0};
  uint64_t observedMemoryUpdates{0};
  uint64_t lastObservedSourceSequence{0};
  uint64_t activeCalls{0};
  uint64_t openCallsAtEnd{0};
  uint64_t observedPeakBytes{0};
  uint64_t observedPeakTimestampNs{0};
  uint64_t observedPeakEventSequence{0};
  uint64_t observedPeakSourceSequence{0};
  std::size_t maxEvents{0};
  std::size_t ownerMetadataCapacity{0};
  std::size_t memoryUpdateCapacity{0};
  std::size_t operatorCallCapacity{0};
  std::string taskState;
  std::string endReason;
  bool captureOverflow{false};
  bool memoryUpdateOverflow{false};
  bool operatorCallOverflow{false};
  bool sourceSequenceGap{false};
  bool ownerMetadataComplete{true};
  bool complete{false};
  bool cleanupComplete{false};
};

std::string boundedString(std::string_view value, std::size_t maximumLength) {
  return std::string{value.substr(0, maximumLength)};
}

template <std::size_t Size>
void copyBounded(std::array<char, Size>& destination, std::string_view source) {
  const auto length = std::min(source.size(), Size - 1);
  std::copy_n(source.data(), length, destination.data());
  destination[length] = '\0';
}

template <std::size_t Size>
std::string fixedString(const std::array<char, Size>& value) {
  return std::string{value.data(), strnlen(value.data(), value.size())};
}

std::string sanitizePathToken(std::string_view value) {
  std::string result;
  result.reserve(std::min<std::size_t>(value.size(), 160));
  for (const char character : value) {
    if (result.size() == 160) {
      break;
    }
    if ((character >= 'a' && character <= 'z') ||
        (character >= 'A' && character <= 'Z') ||
        (character >= '0' && character <= '9') || character == '-' ||
        character == '_' || character == '.') {
      result.push_back(character);
    } else {
      result.push_back('_');
    }
  }
  return result.empty() ? "unknown" : result;
}

void replaceAll(
    std::string& value,
    std::string_view placeholder,
    std::string_view replacement) {
  std::size_t position{0};
  while ((position = value.find(placeholder, position)) != std::string::npos) {
    value.replace(position, placeholder.size(), replacement);
    position += replacement.size();
  }
}

std::string resolvePath(
    std::string_view pattern,
    const GpuMemoryCaptureTask& task) {
  std::string result{pattern};
  replaceAll(result, "%p", std::to_string(getpid()));
  replaceAll(result, "%q", sanitizePathToken(task.queryId));
  replaceAll(result, "%t", sanitizePathToken(task.taskId));
  replaceAll(result, "%u", sanitizePathToken(task.taskUuid));
  return result;
}

bool taskMatches(const GpuMemoryCaptureTask& task, std::string_view filter) {
  return filter.empty() || task.queryId.find(filter) != std::string::npos ||
      task.taskId.find(filter) != std::string::npos ||
      task.taskUuid.find(filter) != std::string::npos;
}

GpuMemoryOwner capturedUnattributedOwner() {
  return GpuMemoryOwner{
      "<unattributed>",
      "<unattributed>",
      "<unattributed>",
      "<unattributed>",
      "<unattributed>",
      -1,
      -1,
      -1,
      "<unattributed>"};
}

folly::dynamic ownerIdentity(const GpuMemoryOwner& owner) {
  return folly::dynamic::object("task_uuid", owner.taskUuid)(
      "task_id", owner.taskId)("query_id", owner.queryId)(
      "plan_node_id", owner.planNodeId)("plan_node_type", owner.planNodeType)(
      "pipeline_id", owner.pipelineId)("driver_id", owner.driverId)(
      "operator_id", owner.operatorId)("operator_type", owner.operatorType);
}

folly::dynamic ownerCounters(const GpuMemoryOwnerSnapshot& owner) {
  return folly::dynamic::object("owner_id", owner.handle.ownerId)(
      "plan_node_track_id", owner.handle.planNodeId)(
      "current_bytes", owner.currentBytes)(
      "source_lifetime_peak_bytes", owner.peakBytes)(
      "source_lifetime_total_bytes", owner.totalBytes)(
      "current_allocations", owner.currentAllocations)(
      "source_lifetime_total_allocations", owner.totalAllocations);
}

folly::dynamic snapshotJson(
    const GpuMemorySnapshot& snapshot,
    const std::set<uint64_t>& relevantOwnerIds) {
  auto owners = folly::dynamic::array();
  for (const auto& owner : snapshot.owners) {
    if (relevantOwnerIds.contains(owner.handle.ownerId)) {
      owners.push_back(ownerCounters(owner));
    }
  }

  return folly::dynamic::object("current_bytes", snapshot.currentBytes)(
      "source_lifetime_peak_bytes", snapshot.peakBytes)(
      "source_lifetime_total_bytes", snapshot.totalBytes)(
      "current_allocations", snapshot.currentAllocations)(
      "source_lifetime_peak_allocations", snapshot.peakAllocations)(
      "source_lifetime_total_allocations", snapshot.totalAllocations)(
      "source_sequence", snapshot.sequence)(
      "source_data_loss_events", snapshot.dataLossEvents)(
      "owners", std::move(owners));
}

struct CapturePeak {
  uint64_t bytes{0};
  uint64_t timestampNs{0};
  uint64_t eventSequence{0};
  uint64_t sourceSequence{0};
};

CapturePeak calculateCapturePeak(const CaptureData& capture) {
  return CapturePeak{
      capture.observedPeakBytes,
      capture.observedPeakTimestampNs,
      capture.observedPeakEventSequence,
      capture.observedPeakSourceSequence};
}

std::set<uint64_t> relevantOwnerIds(const CaptureData& capture) {
  std::set<uint64_t> result;
  const auto includeSnapshot = [&result](const GpuMemorySnapshot& snapshot) {
    for (const auto& owner : snapshot.owners) {
      if (owner.currentBytes > 0 || owner.currentAllocations > 0) {
        result.insert(owner.handle.ownerId);
      }
    }
    for (const auto& allocation : snapshot.allocations) {
      result.insert(allocation.handle.ownerId);
    }
  };
  includeSnapshot(capture.initialSnapshot);
  includeSnapshot(capture.finalSnapshot);

  for (const auto& event : capture.memoryUpdates) {
    result.insert(event.update.ownerId);
  }
  for (const auto& call : capture.operatorCalls) {
    result.insert(call.ownerId);
  }
  for (const auto& oom : capture.oomEvents) {
    result.insert(oom.ownerId);
  }
  return result;
}

bool hasCompleteSourceSequence(const CaptureData& capture) {
  if (capture.finalSnapshot.sequence < capture.initialSnapshot.sequence) {
    return false;
  }
  const auto expectedUpdates =
      capture.finalSnapshot.sequence - capture.initialSnapshot.sequence;
  if (capture.memoryUpdates.size() != expectedUpdates) {
    return false;
  }
  uint64_t expectedSequence = capture.initialSnapshot.sequence;
  for (const auto& event : capture.memoryUpdates) {
    if (event.update.sequence != ++expectedSequence) {
      return false;
    }
  }
  return expectedSequence == capture.finalSnapshot.sequence;
}

folly::dynamic captureJson(const CaptureData& capture) {
  const auto relevantOwners = relevantOwnerIds(capture);
  auto planNodes = folly::dynamic::array();
  for (const auto& node : capture.planNodes) {
    auto sources = folly::dynamic::array();
    for (const auto& source : node.sourceIds) {
      sources.push_back(source);
    }
    planNodes.push_back(
        folly::dynamic::object("id", node.id)("type", node.type)(
            "source_ids", std::move(sources)));
  }

  auto owners = folly::dynamic::array();
  for (const auto& [ownerId, owner] : capture.owners) {
    if (!relevantOwners.contains(ownerId)) {
      continue;
    }
    owners.push_back(
        folly::dynamic::object("owner_id", owner.ownerId)(
            "plan_node_track_id", owner.planNodeId)(
            "identity", ownerIdentity(owner.owner)));
  }

  auto updates = folly::dynamic::array();
  for (const auto& event : capture.memoryUpdates) {
    const auto& update = event.update;
    updates.push_back(
        folly::dynamic::object("event_sequence", event.eventSequence)(
            "timestamp_ns", update.timestampNs)(
            "source_sequence", update.sequence)("owner_id", update.ownerId)(
            "plan_node_track_id", update.planNodeId)(
            "global_current_bytes", update.globalCurrentBytes)(
            "source_lifetime_global_peak_bytes", update.globalPeakBytes)(
            "query_current_bytes", update.queryCurrentBytes)(
            "task_current_bytes", update.taskCurrentBytes)(
            "plan_node_current_bytes", update.planNodeCurrentBytes)(
            "owner_current_bytes", update.ownerCurrentBytes)(
            "delta_bytes", update.deltaBytes));
  }

  auto calls = folly::dynamic::array();
  for (const auto& call : capture.operatorCalls) {
    calls.push_back(
        folly::dynamic::object("event_sequence", call.eventSequence)(
            "call_id", call.callId)("owner_id", call.ownerId)(
            "thread_id", call.threadId)(
            "start_timestamp_ns", call.startTimestampNs)(
            "end_timestamp_ns", call.endTimestampNs)(
            "call_name", fixedString(call.callName))(
            "truncated", call.truncated));
  }

  auto oomEvents = folly::dynamic::array();
  for (const auto& oom : capture.oomEvents) {
    oomEvents.push_back(
        folly::dynamic::object("event_sequence", oom.eventSequence)(
            "timestamp_ns", oom.timestampNs)("owner_id", oom.ownerId)(
            "source_sequence", oom.sourceSequence)(
            "requested_bytes", oom.requestedBytes)(
            "global_current_bytes", oom.globalCurrentBytes)(
            "source_lifetime_global_peak_bytes",
            oom.sourceLifetimeGlobalPeakBytes)(
            "plan_node_current_bytes", oom.planNodeCurrentBytes)(
            "owner_current_bytes", oom.ownerCurrentBytes)(
            "cuda_free_bytes", oom.cudaFreeBytes)(
            "cuda_total_bytes", oom.cudaTotalBytes)(
            "cuda_status", oom.cudaStatus));
  }

  auto dataLossEvents = folly::dynamic::array();
  for (const auto& dataLoss : capture.dataLossEvents) {
    dataLossEvents.push_back(
        folly::dynamic::object("event_sequence", dataLoss.eventSequence)(
            "timestamp_ns", dataLoss.timestampNs)(
            "source_sequence", dataLoss.sourceSequence)(
            "reason", dataLoss.reason));
  }

  const auto peak = calculateCapturePeak(capture);
  const auto sourceDataLossEvents = capture.finalSnapshot.dataLossEvents >=
          capture.initialSnapshot.dataLossEvents
      ? capture.finalSnapshot.dataLossEvents -
          capture.initialSnapshot.dataLossEvents
      : 0;
  const bool exactMemoryTimeline = capture.complete &&
      capture.initialSnapshot.dataLossEvents == 0 &&
      !capture.memoryUpdateOverflow && capture.internalDataLossEvents == 0 &&
      sourceDataLossEvents == 0 && hasCompleteSourceSequence(capture);
  const bool captureLocalPeakExact = capture.complete &&
      capture.initialSnapshot.dataLossEvents == 0 &&
      capture.finalSnapshot.sequence >= capture.initialSnapshot.sequence &&
      capture.internalDataLossEvents == 0 && sourceDataLossEvents == 0 &&
      !capture.sourceSequenceGap &&
      capture.observedMemoryUpdates ==
          capture.finalSnapshot.sequence - capture.initialSnapshot.sequence &&
      capture.lastObservedSourceSequence == capture.finalSnapshot.sequence;
  const bool operatorCallsComplete = capture.complete &&
      capture.cleanupComplete && !capture.operatorCallOverflow &&
      capture.openCallsAtEnd == 0 && capture.internalDataLossEvents == 0;
  const bool exactTimeline =
      exactMemoryTimeline && operatorCallsComplete && !capture.captureOverflow;

  return folly::dynamic::object("format", kCaptureFormat)(
      "version", kCaptureVersion)(
      "capture",
      folly::dynamic::object("capture_id", capture.captureId)(
          "process_id", getpid())("query_id", capture.task.queryId)(
          "task_uuid", capture.task.taskUuid)("task_id", capture.task.taskId)(
          "task_state", capture.taskState)("complete", capture.complete)(
          "cleanup_complete", capture.cleanupComplete)(
          "end_reason", capture.endReason)(
          "start_timestamp_ns", capture.startTimestampNs)(
          "end_timestamp_ns", capture.endTimestampNs)(
          "start_unix_ns", capture.startUnixNs)(
          "end_unix_ns", capture.endUnixNs))(
      "clock",
      folly::dynamic::object("source", "steady_clock_monotonic_ns")(
          "replay_formula",
          "start_unix_ns + (timestamp_ns - start_timestamp_ns)"))(
      "plan_nodes", std::move(planNodes))("owners", std::move(owners))(
      "initial_snapshot",
      snapshotJson(capture.initialSnapshot, relevantOwners))(
      "memory_updates", std::move(updates))("operator_calls", std::move(calls))(
      "oom_events", std::move(oomEvents))(
      "data_loss_events", std::move(dataLossEvents))(
      "final_snapshot", snapshotJson(capture.finalSnapshot, relevantOwners))(
      "summary",
      folly::dynamic::object("capture_local_peak_bytes", peak.bytes)(
          "capture_local_peak_timestamp_ns", peak.timestampNs)(
          "capture_local_peak_event_sequence", peak.eventSequence)(
          "capture_local_peak_source_sequence", peak.sourceSequence)(
          "capture_local_peak_exact", captureLocalPeakExact))(
      "integrity",
      folly::dynamic::object("exact_timeline", exactTimeline)(
          "exact_memory_timeline", exactMemoryTimeline)(
          "operator_calls_complete", operatorCallsComplete)(
          "capture_overflow", capture.captureOverflow)(
          "memory_update_overflow", capture.memoryUpdateOverflow)(
          "operator_call_overflow", capture.operatorCallOverflow)(
          "retained_events", capture.retainedEvents)(
          "dropped_events", capture.droppedEvents)(
          "internal_data_loss_events", capture.internalDataLossEvents)(
          "observed_memory_updates", capture.observedMemoryUpdates)(
          "max_events", capture.maxEvents)(
          "memory_update_capacity", capture.memoryUpdateCapacity)(
          "operator_call_capacity", capture.operatorCallCapacity)(
          "critical_event_capacity", kCriticalEventCapacity)(
          "open_operator_calls_at_end", capture.openCallsAtEnd)(
          "start_source_sequence", capture.initialSnapshot.sequence)(
          "end_source_sequence", capture.finalSnapshot.sequence)(
          "source_data_loss_events", sourceDataLossEvents));
}

bool writeAtomically(
    const std::filesystem::path& path,
    std::string_view contents,
    std::string& error) {
  try {
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    const auto temporaryPath =
        path.string() + ".tmp." + std::to_string(getpid());
    {
      std::ofstream output{
          temporaryPath, std::ios::binary | std::ios::out | std::ios::trunc};
      if (!output) {
        error = "Cannot open temporary capture file: " + temporaryPath;
        return false;
      }
      output.write(
          contents.data(), static_cast<std::streamsize>(contents.size()));
      output.flush();
      if (!output) {
        error = "Cannot write temporary capture file: " + temporaryPath;
        return false;
      }
    }
    if (std::rename(temporaryPath.c_str(), path.c_str()) != 0) {
      error = "Cannot rename temporary capture file to: " + path.string();
      std::remove(temporaryPath.c_str());
      return false;
    }
    return true;
  } catch (const std::exception& exception) {
    error = exception.what();
    return false;
  }
}

class GpuMemoryCaptureController {
 public:
  bool start(const GpuMemoryCaptureConfig& config) {
    stop();
    if (config.pathPattern.empty()) {
      return true;
    }

    try {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = config;
        config_.maxEvents = std::max<std::size_t>(1, config.maxEvents);
        stopping_ = false;
        enabled_ = true;
        taskSelected_ = false;
      }
      worker_ = std::thread([this] { exportLoop(); });
      return true;
    } catch (const std::exception& exception) {
      LOG(ERROR) << "Cannot start GPU-memory capture exporter: "
                 << exception.what();
      std::lock_guard<std::mutex> lock(mutex_);
      enabled_ = false;
      stopping_ = true;
      return false;
    }
  }

  void stop() noexcept {
    try {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = false;
        activeCapture_.store(false, std::memory_order_release);
        if (active_ != nullptr) {
          LOG(WARNING)
              << "Discarding active GPU-memory capture because no terminal "
                 "ledger watermark is available";
          active_.reset();
        }
        stopping_ = true;
      }
      condition_.notify_all();
      if (worker_.joinable()) {
        worker_.join();
      }
      std::lock_guard<std::mutex> lock(mutex_);
      pending_.clear();
      stopping_ = false;
      config_ = {};
      taskSelected_ = false;
    } catch (...) {
      // Profiling cleanup must not alter worker shutdown.
    }
  }

  bool enabled() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_ && !taskSelected_;
  }

  bool active() const noexcept {
    return activeCapture_.load(std::memory_order_acquire);
  }

  std::string lastPath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lastPath_;
  }

  bool tryBegin(
      const GpuMemoryCaptureTask& task,
      const std::vector<GpuMemoryCapturePlanNode>& planNodes,
      GpuMemorySnapshot initialSnapshot) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!enabled_ || taskSelected_ || active_ != nullptr ||
          !taskMatches(task, config_.queryFilter)) {
        return false;
      }

      auto capture = std::make_unique<CaptureData>();
      capture->captureId = ++nextCaptureId_;
      capture->task = task;
      capture->planNodes = planNodes;
      capture->initialSnapshot = std::move(initialSnapshot);
      capture->startTimestampNs = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
      capture->startUnixNs = gpu_memory_detail::gpuMemoryUnixTimeNs();
      capture->maxEvents = config_.maxEvents;
      constexpr auto kOwnerMetadataSlack = 2 * kCriticalEventCapacity;
      const auto maximumSize = std::numeric_limits<std::size_t>::max();
      capture->ownerMetadataCapacity = capture->initialSnapshot.owners.size();
      const auto additionalCapacity =
          config_.maxEvents > maximumSize - kOwnerMetadataSlack
          ? maximumSize
          : config_.maxEvents + kOwnerMetadataSlack;
      capture->ownerMetadataCapacity =
          capture->ownerMetadataCapacity > maximumSize - additionalCapacity
          ? maximumSize
          : capture->ownerMetadataCapacity + additionalCapacity;
      capture->operatorCallCapacity = config_.maxEvents / 4;
      capture->memoryUpdateCapacity =
          config_.maxEvents - capture->operatorCallCapacity;
      capture->memoryUpdates.reserve(capture->memoryUpdateCapacity);
      capture->operatorCalls.reserve(capture->operatorCallCapacity);
      capture->oomEvents.reserve(kCriticalEventCapacity);
      capture->dataLossEvents.reserve(kCriticalEventCapacity);
      const auto activeCallCapacity =
          std::min(capture->operatorCallCapacity, kMaximumActiveCalls);
      capture->activeCallSlots.resize(activeCallCapacity);
      capture->freeCallSlots.reserve(activeCallCapacity);
      for (uint32_t slot = 0; slot < activeCallCapacity; ++slot) {
        capture->freeCallSlots.push_back(
            static_cast<uint32_t>(activeCallCapacity - slot - 1));
      }
      capture->observedPeakBytes = capture->initialSnapshot.currentBytes;
      capture->observedPeakTimestampNs = capture->startTimestampNs;
      capture->observedPeakSourceSequence = capture->initialSnapshot.sequence;
      capture->lastObservedSourceSequence = capture->initialSnapshot.sequence;
      retainOwnerMetadata(
          *capture, CapturedOwner{0, 0, capturedUnattributedOwner()}, true);
      mergeSnapshotOwners(*capture, capture->initialSnapshot);
      if (capture->initialSnapshot.dataLossEvents > 0) {
        ++capture->internalDataLossEvents;
        const auto eventSequence = retainEvent(*capture);
        capture->dataLossEvents.push_back(
            CapturedDataLoss{
                eventSequence,
                capture->startTimestampNs,
                capture->initialSnapshot.sequence,
                "source ledger baseline was compromised before capture"});
      }
      active_ = std::move(capture);
      activeCapture_.store(true, std::memory_order_release);
      taskSelected_ = true;
      LOG(INFO) << "GPU-memory capture selected query " << task.queryId
                << ", task " << task.taskId;
      return true;
    } catch (const std::exception& exception) {
      LOG(ERROR) << "Cannot begin GPU-memory capture: " << exception.what();
      return false;
    }
  }

  void finish(
      const std::string& taskUuid,
      const std::string& taskId,
      std::string_view taskState,
      bool cleanupComplete,
      GpuMemorySnapshot finalSnapshot) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr) {
        return;
      }
      const bool matchesTask = active_->task.taskUuid.empty()
          ? active_->task.taskId == taskId
          : active_->task.taskUuid == taskUuid;
      if (!matchesTask) {
        return;
      }
      activeCapture_.store(false, std::memory_order_release);
      active_->finalSnapshot = std::move(finalSnapshot);
      mergeSnapshotOwners(*active_, active_->finalSnapshot);
      verifyRequiredOwnerMetadata(*active_);
      active_->endTimestampNs = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
      active_->endUnixNs = gpu_memory_detail::gpuMemoryUnixTimeNs();
      active_->taskState = boundedString(taskState, 64);
      active_->endReason = "task_terminal";
      active_->complete = true;
      active_->cleanupComplete = cleanupComplete;
      clipOpenCalls(*active_, active_->endTimestampNs);
      pending_.push_back(std::move(active_));
      condition_.notify_one();
    } catch (...) {
      // Profiling must never change task-completion behavior.
    }
  }

  void abort(
      std::string_view reason,
      GpuMemorySnapshot finalSnapshot) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr) {
        return;
      }
      activeCapture_.store(false, std::memory_order_release);
      active_->finalSnapshot = std::move(finalSnapshot);
      mergeSnapshotOwners(*active_, active_->finalSnapshot);
      verifyRequiredOwnerMetadata(*active_);
      active_->endTimestampNs = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
      active_->endUnixNs = gpu_memory_detail::gpuMemoryUnixTimeNs();
      active_->taskState = "aborted";
      active_->endReason = boundedString(reason, 128);
      active_->complete = false;
      active_->cleanupComplete = false;
      clipOpenCalls(*active_, active_->endTimestampNs);
      pending_.push_back(std::move(active_));
      condition_.notify_one();
    } catch (...) {
      // Profiling must never change worker shutdown.
    }
  }

  void registerOwner(
      uint64_t ownerId,
      uint64_t planNodeId,
      const GpuMemoryOwner& owner) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr) {
        return;
      }
      retainOwnerMetadata(
          *active_, CapturedOwner{ownerId, planNodeId, owner}, false);
    } catch (...) {
      noteInternalDataLoss("owner metadata capture exception", 0);
    }
  }

  void recordUpdate(const GpuMemoryTraceUpdate& update) noexcept {
    recordUpdate(update, nullptr);
  }

  void recordUpdate(
      const GpuMemoryTraceUpdate& update,
      const GpuMemoryOwner& owner) noexcept {
    recordUpdate(update, &owner);
  }

 private:
  void recordUpdate(
      const GpuMemoryTraceUpdate& update,
      const GpuMemoryOwner* owner) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr ||
          update.sequence <= active_->initialSnapshot.sequence) {
        return;
      }
      if (update.sequence != active_->lastObservedSourceSequence + 1) {
        active_->sourceSequenceGap = true;
      }
      active_->lastObservedSourceSequence = update.sequence;
      ++active_->observedMemoryUpdates;
      const bool establishesPeak =
          update.globalCurrentBytes > active_->observedPeakBytes;
      if (establishesPeak) {
        active_->observedPeakBytes = update.globalCurrentBytes;
        active_->observedPeakTimestampNs = update.timestampNs;
        active_->observedPeakEventSequence = 0;
        active_->observedPeakSourceSequence = update.sequence;
      }
      if (active_->memoryUpdates.size() >= active_->memoryUpdateCapacity) {
        active_->memoryUpdateOverflow = true;
        noteOverflow(*active_);
        return;
      }
      const bool hasOwnerMetadata = owner == nullptr
          ? requireExistingOwnerMetadata(*active_, update.ownerId)
          : retainOwnerMetadata(
                *active_,
                CapturedOwner{update.ownerId, update.planNodeId, *owner},
                true);
      if (!hasOwnerMetadata) {
        active_->memoryUpdateOverflow = true;
        noteMissingOwnerMetadata(*active_);
        return;
      }
      const auto eventSequence = retainEvent(*active_);
      active_->memoryUpdates.push_back(
          CapturedMemoryUpdate{eventSequence, update});
      if (establishesPeak) {
        active_->observedPeakEventSequence = eventSequence;
      }
    } catch (...) {
      noteInternalDataLoss("memory update capture exception", update.sequence);
    }
  }

 public:
  GpuMemoryCaptureCallHandle beginCall(
      uint64_t ownerId,
      uint64_t planNodeId,
      const GpuMemoryOwner& owner,
      std::string_view callName) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr) {
        return {};
      }
      if (active_->operatorCalls.size() + active_->activeCalls >=
              active_->operatorCallCapacity ||
          active_->freeCallSlots.empty()) {
        active_->operatorCallOverflow = true;
        noteOverflow(*active_);
        return {};
      }
      if (!retainOwnerMetadata(
              *active_, CapturedOwner{ownerId, planNodeId, owner}, true)) {
        noteMissingOwnerMetadata(*active_);
        return {};
      }
      GpuMemoryCaptureCallHandle handle;
      handle.captureId = active_->captureId;
      handle.callId = ++nextCallId_;
      handle.ownerId = ownerId;
      handle.startTimestampNs = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
      handle.threadId = static_cast<int64_t>(folly::getOSThreadID());
      copyBounded(handle.callName, callName);
      handle.openSlot = active_->freeCallSlots.back();
      active_->freeCallSlots.pop_back();
      handle.active = true;
      active_->activeCallSlots.at(handle.openSlot) =
          ActiveCallSlot{handle, true};
      ++active_->activeCalls;
      return handle;
    } catch (...) {
      noteInternalDataLoss("operator call begin exception", 0);
      return {};
    }
  }

  void endCall(const GpuMemoryCaptureCallHandle& handle) noexcept {
    if (!handle.active) {
      return;
    }
    try {
      const auto endTimestampNs = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr || active_->captureId != handle.captureId) {
        return;
      }
      if (handle.openSlot >= active_->activeCallSlots.size()) {
        ++active_->internalDataLossEvents;
        return;
      }
      auto& slot = active_->activeCallSlots.at(handle.openSlot);
      if (!slot.active || slot.handle.callId != handle.callId) {
        ++active_->internalDataLossEvents;
        return;
      }
      slot.active = false;
      active_->freeCallSlots.push_back(handle.openSlot);
      --active_->activeCalls;
      const auto eventSequence = retainEvent(*active_);
      active_->operatorCalls.push_back(
          CapturedCallSpan{
              eventSequence,
              handle.callId,
              handle.ownerId,
              handle.threadId,
              handle.startTimestampNs,
              std::max(endTimestampNs, handle.startTimestampNs),
              handle.callName,
              false});
    } catch (...) {
      noteInternalDataLoss("operator call capture exception", 0);
    }
  }

  void recordOom(
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
      std::string_view cudaStatus) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr) {
        return;
      }
      if (active_->oomEvents.size() >= kCriticalEventCapacity) {
        noteCriticalEventLoss(*active_);
        return;
      }
      if (!retainOwnerMetadata(
              *active_, CapturedOwner{ownerId, planNodeId, owner}, true)) {
        noteMissingOwnerMetadata(*active_);
        return;
      }
      const auto eventSequence = retainEvent(*active_);
      active_->oomEvents.push_back(
          CapturedOom{
              eventSequence,
              timestampNs,
              sourceSequence,
              ownerId,
              static_cast<uint64_t>(requestedBytes),
              globalCurrentBytes,
              globalPeakBytes,
              planNodeCurrentBytes,
              ownerCurrentBytes,
              static_cast<uint64_t>(cudaFreeBytes),
              static_cast<uint64_t>(cudaTotalBytes),
              boundedString(cudaStatus, 128)});
    } catch (...) {
      noteInternalDataLoss("OOM event capture exception", 0);
    }
  }

  void recordDataLoss(
      std::string_view reason,
      uint64_t sourceSequence) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr) {
        return;
      }
      if (active_->dataLossEvents.size() >= kCriticalEventCapacity) {
        noteCriticalEventLoss(*active_);
        return;
      }
      const auto eventSequence = retainEvent(*active_);
      active_->dataLossEvents.push_back(
          CapturedDataLoss{
              eventSequence,
              gpu_memory_detail::gpuMemoryMonotonicTimeNs(),
              sourceSequence,
              boundedString(reason, 256)});
    } catch (...) {
      // A failed diagnostic cannot safely report another diagnostic.
    }
  }

 private:
  static void mergeSnapshotOwners(
      CaptureData& capture,
      const GpuMemorySnapshot& snapshot) {
    for (const auto& owner : snapshot.owners) {
      if (!retainOwnerMetadata(
              capture,
              CapturedOwner{
                  owner.handle.ownerId, owner.handle.planNodeId, owner.owner},
              true)) {
        noteMissingOwnerMetadata(capture);
      }
    }
  }

  static bool requireExistingOwnerMetadata(
      CaptureData& capture,
      uint64_t ownerId) {
    if (!capture.owners.contains(ownerId)) {
      return false;
    }
    capture.requiredOwnerIds.insert(ownerId);
    return true;
  }

  static bool retainOwnerMetadata(
      CaptureData& capture,
      CapturedOwner owner,
      bool required) {
    if (auto existing = capture.owners.find(owner.ownerId);
        existing != capture.owners.end()) {
      if (existing->second.owner.planNodeType.empty() &&
          !owner.owner.planNodeType.empty()) {
        existing->second = std::move(owner);
      }
      if (required) {
        capture.requiredOwnerIds.insert(existing->first);
      }
      return true;
    }

    if (capture.owners.size() >= capture.ownerMetadataCapacity) {
      const auto unused = std::find_if(
          capture.owners.begin(),
          capture.owners.end(),
          [&capture](const auto& entry) {
            return !capture.requiredOwnerIds.contains(entry.first);
          });
      if (unused == capture.owners.end()) {
        return false;
      }
      capture.owners.erase(unused);
    }
    const auto ownerId = owner.ownerId;
    capture.owners.emplace(ownerId, std::move(owner));
    if (required) {
      capture.requiredOwnerIds.insert(ownerId);
    }
    return true;
  }

  static void verifyRequiredOwnerMetadata(CaptureData& capture) {
    const bool complete = std::all_of(
        capture.requiredOwnerIds.begin(),
        capture.requiredOwnerIds.end(),
        [&capture](uint64_t ownerId) {
          return capture.owners.contains(ownerId);
        });
    if (complete) {
      return;
    }
    capture.ownerMetadataComplete = false;
    ++capture.internalDataLossEvents;
    noteOverflow(capture);
  }

  static void noteOverflow(CaptureData& capture) {
    capture.captureOverflow = true;
    ++capture.droppedEvents;
  }

  static void noteMissingOwnerMetadata(CaptureData& capture) {
    capture.ownerMetadataComplete = false;
    ++capture.internalDataLossEvents;
    noteOverflow(capture);
  }

  static void noteCriticalEventLoss(CaptureData& capture) {
    ++capture.internalDataLossEvents;
    noteOverflow(capture);
  }

  static uint64_t retainEvent(CaptureData& capture) {
    ++capture.retainedEvents;
    return ++capture.nextEventSequence;
  }

  static void clipOpenCalls(CaptureData& capture, uint64_t endTimestampNs) {
    capture.openCallsAtEnd = capture.activeCalls;
    for (auto& slot : capture.activeCallSlots) {
      if (!slot.active) {
        continue;
      }
      if (capture.operatorCalls.size() >= capture.operatorCallCapacity) {
        capture.operatorCallOverflow = true;
        noteCriticalEventLoss(capture);
        slot.active = false;
        continue;
      }
      const auto eventSequence = retainEvent(capture);
      const auto& handle = slot.handle;
      capture.operatorCalls.push_back(
          CapturedCallSpan{
              eventSequence,
              handle.callId,
              handle.ownerId,
              handle.threadId,
              handle.startTimestampNs,
              std::max(endTimestampNs, handle.startTimestampNs),
              handle.callName,
              true});
      slot.active = false;
    }
    capture.activeCalls = 0;
  }

  void noteInternalDataLoss(
      std::string_view reason,
      uint64_t sourceSequence) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr) {
        return;
      }
      ++active_->internalDataLossEvents;
      if (active_->dataLossEvents.size() >= kCriticalEventCapacity) {
        noteOverflow(*active_);
        return;
      }
      const auto eventSequence = retainEvent(*active_);
      active_->dataLossEvents.push_back(
          CapturedDataLoss{
              eventSequence,
              gpu_memory_detail::gpuMemoryMonotonicTimeNs(),
              sourceSequence,
              boundedString(reason, 256)});
    } catch (...) {
      // A failed diagnostic cannot safely report another diagnostic.
    }
  }

  void exportLoop() noexcept {
    while (true) {
      std::unique_ptr<CaptureData> capture;
      GpuMemoryCaptureConfig config;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(
            lock, [this] { return stopping_ || !pending_.empty(); });
        if (pending_.empty()) {
          if (stopping_) {
            return;
          }
          continue;
        }
        capture = std::move(pending_.front());
        pending_.pop_front();
        config = config_;
      }

      try {
        if (!capture->ownerMetadataComplete) {
          LOG(ERROR) << "Discarding GPU-memory capture because bounded owner "
                        "metadata could not describe every retained fact";
          continue;
        }
        const auto path = resolvePath(config.pathPattern, capture->task);
        const auto document = folly::toJson(captureJson(*capture));
        std::string error;
        if (!writeAtomically(path, document, error)) {
          LOG(ERROR) << "Cannot write GPU-memory capture: " << error;
          continue;
        }
        {
          std::lock_guard<std::mutex> lock(mutex_);
          lastPath_ = path;
        }
        LOG(INFO) << "GPU-memory capture written to " << path << " with "
                  << capture->retainedEvents << " retained events and "
                  << capture->droppedEvents << " dropped events";
      } catch (const std::exception& exception) {
        LOG(ERROR) << "GPU-memory capture export failed: " << exception.what();
      }
    }
  }

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  GpuMemoryCaptureConfig config_;
  std::unique_ptr<CaptureData> active_;
  std::deque<std::unique_ptr<CaptureData>> pending_;
  std::thread worker_;
  std::string lastPath_;
  uint64_t nextCaptureId_{0};
  uint64_t nextCallId_{0};
  std::atomic<bool> activeCapture_{false};
  bool enabled_{false};
  bool stopping_{false};
  bool taskSelected_{false};
};

GpuMemoryCaptureController& captureController() {
  static auto* controller = new GpuMemoryCaptureController;
  return *controller;
}

} // namespace

bool startGpuMemoryCapture(const GpuMemoryCaptureConfig& config) noexcept {
  try {
    return captureController().start(config);
  } catch (...) {
    return false;
  }
}

void stopGpuMemoryCapture() noexcept {
  captureController().stop();
}

bool gpuMemoryCaptureEnabled() noexcept {
  return captureController().enabled();
}

namespace gpu_memory_detail {

bool gpuMemoryCaptureActive() noexcept {
  return captureController().active();
}

uint64_t gpuMemoryMonotonicTimeNs() noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

uint64_t gpuMemoryUnixTimeNs() noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

bool tryBeginGpuMemoryCapture(
    const GpuMemoryCaptureTask& task,
    const std::vector<GpuMemoryCapturePlanNode>& planNodes,
    GpuMemorySnapshot initialSnapshot) noexcept {
  return captureController().tryBegin(
      task, planNodes, std::move(initialSnapshot));
}

void finishGpuMemoryCapture(
    const std::string& taskUuid,
    const std::string& taskId,
    std::string_view taskState,
    bool cleanupComplete,
    GpuMemorySnapshot finalSnapshot) noexcept {
  captureController().finish(
      taskUuid, taskId, taskState, cleanupComplete, std::move(finalSnapshot));
}

void abortGpuMemoryCapture(
    std::string_view reason,
    GpuMemorySnapshot finalSnapshot) noexcept {
  captureController().abort(reason, std::move(finalSnapshot));
}

void registerGpuMemoryCaptureOwner(
    uint64_t ownerId,
    uint64_t planNodeId,
    const GpuMemoryOwner& owner) noexcept {
  captureController().registerOwner(ownerId, planNodeId, owner);
}

void recordGpuMemoryCaptureUpdate(const GpuMemoryTraceUpdate& update) noexcept {
  captureController().recordUpdate(update);
}

void recordGpuMemoryCaptureUpdate(
    const GpuMemoryTraceUpdate& update,
    const GpuMemoryOwner& owner) noexcept {
  captureController().recordUpdate(update, owner);
}

GpuMemoryCaptureCallHandle beginGpuMemoryCaptureOperatorCall(
    uint64_t ownerId,
    uint64_t planNodeId,
    const GpuMemoryOwner& owner,
    std::string_view callName) noexcept {
  return captureController().beginCall(ownerId, planNodeId, owner, callName);
}

void endGpuMemoryCaptureOperatorCall(
    const GpuMemoryCaptureCallHandle& handle) noexcept {
  captureController().endCall(handle);
}

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
    std::string_view cudaStatus) noexcept {
  captureController().recordOom(
      timestampNs,
      sourceSequence,
      ownerId,
      planNodeId,
      owner,
      requestedBytes,
      globalCurrentBytes,
      globalPeakBytes,
      planNodeCurrentBytes,
      ownerCurrentBytes,
      cudaFreeBytes,
      cudaTotalBytes,
      cudaStatus);
}

void recordGpuMemoryCaptureDataLoss(
    std::string_view reason,
    uint64_t sourceSequence) noexcept {
  captureController().recordDataLoss(reason, sourceSequence);
}

} // namespace gpu_memory_detail

} // namespace facebook::velox::cudf_velox
