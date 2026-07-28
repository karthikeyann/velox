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

#include <dlfcn.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <glog/logging.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

constexpr std::string_view kCaptureFormat{"velox-cudf-gpu-memory-capture"};
constexpr uint64_t kCaptureVersion{1};
constexpr std::size_t kMaximumErrorLength{1'024};

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
  uint64_t threadId{0};
  uint64_t startTimestampNs{0};
  uint64_t endTimestampNs{0};
  std::array<char, 32> callName{};
};

struct CapturedMarker {
  uint64_t eventSequence{0};
  uint64_t timestampNs{0};
  uint64_t ownerId{0};
  std::string name;
};

struct CapturedOom {
  uint64_t eventSequence{0};
  uint64_t timestampNs{0};
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
  std::vector<CapturedMemoryUpdate> memoryUpdates;
  std::vector<CapturedCallSpan> operatorCalls;
  std::vector<CapturedMarker> markers;
  std::vector<CapturedOom> oomEvents;
  std::vector<CapturedDataLoss> dataLossEvents;
  uint64_t startTimestampNs{0};
  uint64_t startUnixNs{0};
  uint64_t endTimestampNs{0};
  uint64_t endUnixNs{0};
  uint64_t nextEventSequence{0};
  uint64_t retainedEvents{0};
  uint64_t droppedEvents{0};
  uint64_t activeCalls{0};
  uint64_t openCallsAtEnd{0};
  std::size_t maxEvents{0};
  std::string taskState;
  std::string endReason;
  bool captureOverflow{false};
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

folly::dynamic snapshotJson(const GpuMemorySnapshot& snapshot) {
  auto owners = folly::dynamic::array();
  for (const auto& owner : snapshot.owners) {
    owners.push_back(ownerCounters(owner));
  }

  auto allocations = folly::dynamic::array();
  for (const auto& allocation : snapshot.allocations) {
    std::ostringstream address;
    address << "0x" << std::hex << allocation.address;
    allocations.push_back(
        folly::dynamic::object("address", address.str())(
            "bytes", allocation.bytes)("owner_id", allocation.handle.ownerId)(
            "plan_node_track_id", allocation.handle.planNodeId));
  }

  return folly::dynamic::object("current_bytes", snapshot.currentBytes)(
      "source_lifetime_peak_bytes", snapshot.peakBytes)(
      "source_lifetime_total_bytes", snapshot.totalBytes)(
      "current_allocations", snapshot.currentAllocations)(
      "source_lifetime_peak_allocations", snapshot.peakAllocations)(
      "source_lifetime_total_allocations", snapshot.totalAllocations)(
      "source_sequence", snapshot.sequence)(
      "source_data_loss_events", snapshot.dataLossEvents)(
      "owners", std::move(owners))("allocations", std::move(allocations));
}

struct CapturePeak {
  uint64_t bytes{0};
  uint64_t timestampNs{0};
  uint64_t eventSequence{0};
};

CapturePeak calculateCapturePeak(const CaptureData& capture) {
  CapturePeak peak{
      capture.initialSnapshot.currentBytes, capture.startTimestampNs, 0};
  for (const auto& event : capture.memoryUpdates) {
    if (event.update.globalCurrentBytes > peak.bytes) {
      peak = CapturePeak{
          event.update.globalCurrentBytes,
          event.update.timestampNs,
          event.eventSequence};
    }
  }
  return peak;
}

folly::dynamic captureJson(const CaptureData& capture) {
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
  for (const auto& [_, owner] : capture.owners) {
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
            "call_name", fixedString(call.callName)));
  }

  auto markers = folly::dynamic::array();
  for (const auto& marker : capture.markers) {
    markers.push_back(
        folly::dynamic::object("event_sequence", marker.eventSequence)(
            "timestamp_ns", marker.timestampNs)("owner_id", marker.ownerId)(
            "name", marker.name));
  }

  auto oomEvents = folly::dynamic::array();
  for (const auto& oom : capture.oomEvents) {
    oomEvents.push_back(
        folly::dynamic::object("event_sequence", oom.eventSequence)(
            "timestamp_ns", oom.timestampNs)("owner_id", oom.ownerId)(
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
  const bool exactTimeline =
      !capture.captureOverflow && sourceDataLossEvents == 0;

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
      "initial_snapshot", snapshotJson(capture.initialSnapshot))(
      "memory_updates", std::move(updates))("operator_calls", std::move(calls))(
      "markers", std::move(markers))("oom_events", std::move(oomEvents))(
      "data_loss_events", std::move(dataLossEvents))(
      "final_snapshot", snapshotJson(capture.finalSnapshot))(
      "summary",
      folly::dynamic::object("capture_local_peak_bytes", peak.bytes)(
          "capture_local_peak_timestamp_ns", peak.timestampNs)(
          "capture_local_peak_event_sequence", peak.eventSequence))(
      "integrity",
      folly::dynamic::object("exact_timeline", exactTimeline)(
          "exact_memory_timeline", exactTimeline)(
          "operator_calls_complete",
          capture.openCallsAtEnd == 0 && capture.cleanupComplete)(
          "capture_overflow", capture.captureOverflow)(
          "retained_events", capture.retainedEvents)(
          "dropped_events", capture.droppedEvents)(
          "max_events", capture.maxEvents)(
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

using QuentReplayFunction = int (*)(
    const char* capturePath,
    const char* outputPath,
    char* error,
    std::size_t errorLength);

void replayIntoQuent(
    const GpuMemoryCaptureConfig& config,
    const GpuMemoryCaptureTask& task,
    const std::string& capturePath) {
  if (config.adapterPath.empty()) {
    return;
  }

  void* library = dlopen(config.adapterPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    LOG(ERROR) << "Cannot load Quent Velox replay adapter: " << dlerror();
    return;
  }

  dlerror();
  auto* replay = reinterpret_cast<QuentReplayFunction>(
      dlsym(library, "quent_velox_replay_v1"));
  if (const char* symbolError = dlerror()) {
    LOG(ERROR) << "Cannot resolve quent_velox_replay_v1: " << symbolError;
    dlclose(library);
    return;
  }

  std::array<char, kMaximumErrorLength> error{};
  const auto outputPath = resolvePath(config.adapterOutputPath, task);
  const int result = replay(
      capturePath.c_str(), outputPath.c_str(), error.data(), error.size());
  if (result != 0) {
    LOG(ERROR) << "Quent Velox replay failed: " << error.data();
  } else {
    LOG(INFO) << "Quent Velox profile written to " << outputPath;
  }
  dlclose(library);
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
        if (active_ != nullptr) {
          active_->complete = false;
          active_->cleanupComplete = false;
          active_->endReason = "capture_service_stopped";
          active_->taskState = "unknown";
          active_->endTimestampNs =
              gpu_memory_detail::gpuMemoryMonotonicTimeNs();
          active_->endUnixNs = gpu_memory_detail::gpuMemoryUnixTimeNs();
          active_->finalSnapshot = active_->initialSnapshot;
          active_->openCallsAtEnd = active_->activeCalls;
          pending_.push_back(std::move(active_));
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
    } catch (...) {
      // Profiling cleanup must not alter worker shutdown.
    }
  }

  bool enabled() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_ && active_ == nullptr;
  }

  std::string lastPath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lastPath_;
  }

  bool tryBegin(
      const GpuMemoryCaptureTask& task,
      const std::vector<GpuMemoryCapturePlanNode>& planNodes,
      const GpuMemorySnapshot& initialSnapshot) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!enabled_ || active_ != nullptr ||
          !taskMatches(task, config_.queryFilter)) {
        return false;
      }

      auto capture = std::make_unique<CaptureData>();
      capture->captureId = ++nextCaptureId_;
      capture->task = task;
      capture->planNodes = planNodes;
      capture->initialSnapshot = initialSnapshot;
      capture->finalSnapshot = initialSnapshot;
      capture->startTimestampNs = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
      capture->startUnixNs = gpu_memory_detail::gpuMemoryUnixTimeNs();
      capture->maxEvents = config_.maxEvents;

      capture->memoryUpdates.reserve(config_.maxEvents);
      capture->operatorCalls.reserve(config_.maxEvents);
      capture->markers.reserve(64);
      capture->oomEvents.reserve(16);
      capture->dataLossEvents.reserve(16);
      mergeSnapshotOwners(*capture, initialSnapshot);
      active_ = std::move(capture);
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
      const GpuMemorySnapshot& finalSnapshot) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr ||
          (active_->task.taskUuid != taskUuid &&
           active_->task.taskId != taskId)) {
        return;
      }
      active_->finalSnapshot = finalSnapshot;
      mergeSnapshotOwners(*active_, finalSnapshot);
      active_->endTimestampNs = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
      active_->endUnixNs = gpu_memory_detail::gpuMemoryUnixTimeNs();
      active_->taskState = boundedString(taskState, 64);
      active_->endReason = "task_terminal";
      active_->complete = true;
      active_->cleanupComplete = cleanupComplete;
      active_->openCallsAtEnd = active_->activeCalls;
      pending_.push_back(std::move(active_));
      condition_.notify_one();
    } catch (...) {
      // Profiling must never change task-completion behavior.
    }
  }

  void abort(
      std::string_view reason,
      const GpuMemorySnapshot& finalSnapshot) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr) {
        return;
      }
      active_->finalSnapshot = finalSnapshot;
      mergeSnapshotOwners(*active_, finalSnapshot);
      active_->endTimestampNs = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
      active_->endUnixNs = gpu_memory_detail::gpuMemoryUnixTimeNs();
      active_->taskState = "aborted";
      active_->endReason = boundedString(reason, 128);
      active_->complete = false;
      active_->cleanupComplete = false;
      active_->openCallsAtEnd = active_->activeCalls;
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
      active_->owners.try_emplace(
          ownerId, CapturedOwner{ownerId, planNodeId, owner});
    } catch (...) {
      noteInternalDataLoss("owner metadata capture exception", 0);
    }
  }

  void recordUpdate(const GpuMemoryTraceUpdate& update) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr ||
          update.sequence <= active_->initialSnapshot.sequence) {
        return;
      }
      if (!retainEvent(*active_)) {
        return;
      }
      active_->memoryUpdates.push_back(
          CapturedMemoryUpdate{active_->nextEventSequence, update});
    } catch (...) {
      noteInternalDataLoss("memory update capture exception", update.sequence);
    }
  }

  GpuMemoryCaptureCallHandle beginCall(
      uint64_t ownerId,
      std::string_view callName) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr ||
          active_->retainedEvents + active_->activeCalls >=
              active_->maxEvents) {
        if (active_ != nullptr) {
          noteOverflow(*active_);
        }
        return {};
      }
      GpuMemoryCaptureCallHandle handle;
      handle.captureId = active_->captureId;
      handle.callId = ++nextCallId_;
      handle.ownerId = ownerId;
      handle.startTimestampNs = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
      handle.threadId =
          std::hash<std::thread::id>{}(std::this_thread::get_id());
      copyBounded(handle.callName, callName);
      handle.active = true;
      ++active_->activeCalls;
      return handle;
    } catch (...) {
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
      if (active_->activeCalls > 0) {
        --active_->activeCalls;
      }
      if (!retainEvent(*active_)) {
        return;
      }
      active_->operatorCalls.push_back(
          CapturedCallSpan{
              active_->nextEventSequence,
              handle.callId,
              handle.ownerId,
              handle.threadId,
              handle.startTimestampNs,
              std::max(endTimestampNs, handle.startTimestampNs),
              handle.callName});
    } catch (...) {
      noteInternalDataLoss("operator call capture exception", 0);
    }
  }

  void recordMarker(uint64_t ownerId, std::string_view name) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr || !retainEvent(*active_)) {
        return;
      }
      active_->markers.push_back(
          CapturedMarker{
              active_->nextEventSequence,
              gpu_memory_detail::gpuMemoryMonotonicTimeNs(),
              ownerId,
              boundedString(name, 256)});
    } catch (...) {
      noteInternalDataLoss("marker capture exception", 0);
    }
  }

  void recordOom(
      uint64_t ownerId,
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
      if (active_ == nullptr || !retainEvent(*active_)) {
        return;
      }
      active_->oomEvents.push_back(
          CapturedOom{
              active_->nextEventSequence,
              gpu_memory_detail::gpuMemoryMonotonicTimeNs(),
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
      if (active_ == nullptr || !retainEvent(*active_)) {
        return;
      }
      active_->dataLossEvents.push_back(
          CapturedDataLoss{
              active_->nextEventSequence,
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
      capture.owners.try_emplace(
          owner.handle.ownerId,
          CapturedOwner{
              owner.handle.ownerId, owner.handle.planNodeId, owner.owner});
    }
  }

  static void noteOverflow(CaptureData& capture) {
    capture.captureOverflow = true;
    ++capture.droppedEvents;
  }

  static bool retainEvent(CaptureData& capture) {
    if (capture.retainedEvents >= capture.maxEvents) {
      noteOverflow(capture);
      return false;
    }
    ++capture.retainedEvents;
    ++capture.nextEventSequence;
    return true;
  }

  void noteInternalDataLoss(
      std::string_view reason,
      uint64_t sourceSequence) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (active_ == nullptr) {
        return;
      }
      if (!retainEvent(*active_)) {
        return;
      }
      active_->dataLossEvents.push_back(
          CapturedDataLoss{
              active_->nextEventSequence,
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
        replayIntoQuent(config, capture->task, path);
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
  bool enabled_{false};
  bool stopping_{false};
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

std::string gpuMemoryCaptureLastPath() {
  return captureController().lastPath();
}

void markGpuMemoryProfile(std::string_view name) noexcept {
  markGpuMemoryTrace(name);
}

namespace gpu_memory_detail {

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
    const GpuMemorySnapshot& initialSnapshot) noexcept {
  return captureController().tryBegin(task, planNodes, initialSnapshot);
}

void finishGpuMemoryCapture(
    const std::string& taskUuid,
    const std::string& taskId,
    std::string_view taskState,
    bool cleanupComplete,
    const GpuMemorySnapshot& finalSnapshot) noexcept {
  captureController().finish(
      taskUuid, taskId, taskState, cleanupComplete, finalSnapshot);
}

void abortGpuMemoryCapture(
    std::string_view reason,
    const GpuMemorySnapshot& finalSnapshot) noexcept {
  captureController().abort(reason, finalSnapshot);
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

GpuMemoryCaptureCallHandle beginGpuMemoryCaptureOperatorCall(
    uint64_t ownerId,
    std::string_view callName) noexcept {
  return captureController().beginCall(ownerId, callName);
}

void endGpuMemoryCaptureOperatorCall(
    const GpuMemoryCaptureCallHandle& handle) noexcept {
  captureController().endCall(handle);
}

void recordGpuMemoryCaptureOom(
    uint64_t ownerId,
    std::size_t requestedBytes,
    uint64_t globalCurrentBytes,
    uint64_t globalPeakBytes,
    uint64_t planNodeCurrentBytes,
    uint64_t ownerCurrentBytes,
    std::size_t cudaFreeBytes,
    std::size_t cudaTotalBytes,
    std::string_view cudaStatus) noexcept {
  captureController().recordOom(
      ownerId,
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

void recordGpuMemoryCaptureMarker(
    uint64_t ownerId,
    std::string_view name) noexcept {
  captureController().recordMarker(ownerId, name);
}

} // namespace gpu_memory_detail

} // namespace facebook::velox::cudf_velox
