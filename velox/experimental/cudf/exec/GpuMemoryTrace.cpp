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

#include "velox/experimental/cudf/exec/GpuMemoryTrace.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#include <perfetto.h>
#pragma GCC diagnostic pop

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCounters.h>
#include <nvtx3/nvToolsExtPayload.h>
#include <nvtx3/nvToolsExtSemanticsCounters.h>

#include <fcntl.h>
#include <glog/logging.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

PERFETTO_DEFINE_CATEGORIES_IN_NAMESPACE(
    velox_cudf_trace,
    perfetto::Category("velox.cudf.memory")
        .SetDescription("Velox-cuDF logical GPU allocation ownership"),
    perfetto::Category("velox.cudf.operator")
        .SetDescription("Velox-cuDF operator calls and diagnostic markers"));

PERFETTO_TRACK_EVENT_STATIC_STORAGE_IN_NAMESPACE(velox_cudf_trace);

namespace facebook::velox::cudf_velox {
namespace {

constexpr uint64_t kRootTrackId{0x56474d0000000001ULL};
constexpr uint64_t kMarkerTrackId{0x56474d0000000002ULL};
constexpr uint64_t kGlobalTrackId{0x56474d0000000003ULL};
constexpr uint64_t kGlobalPeakTrackId{0x56474d0000000004ULL};
constexpr uint64_t kMaximumTraceFileBytes{16ULL * 1024 * 1024 * 1024};
constexpr uint32_t kSharedMemorySizeKiB{32 * 1024};
constexpr uint32_t kSharedMemoryPageSizeKiB{32};

perfetto::NamedTrack rootTrack() {
  return perfetto::NamedTrack(
      "Velox-cuDF GPU memory", kRootTrackId, perfetto::ProcessTrack::Current());
}

perfetto::NamedTrack markerTrack() {
  return perfetto::NamedTrack(
      "Markers and allocation failures", kMarkerTrackId, rootTrack());
}

perfetto::CounterTrack globalTrack() {
  return perfetto::CounterTrack(
             "Overall RMM logical live bytes", kGlobalTrackId, rootTrack())
      .set_unit(perfetto::protos::pbzero::CounterDescriptor::UNIT_SIZE_BYTES)
      .set_category("velox.cudf.memory");
}

perfetto::CounterTrack globalPeakTrack() {
  return perfetto::CounterTrack(
             "Overall RMM logical peak bytes", kGlobalPeakTrackId, rootTrack())
      .set_unit(perfetto::protos::pbzero::CounterDescriptor::UNIT_SIZE_BYTES)
      .set_category("velox.cudf.memory");
}

std::string displayField(std::string_view value) {
  return value.empty() ? "<none>" : std::string{value};
}

std::string displayPlanNodeType(const GpuMemoryOwner& owner) {
  return owner.planNodeType.empty() ? "PlanNode" : owner.planNodeType;
}

const nvtxSemanticsCounter_t& nvtxByteCounterSemantics() {
  static const auto semantics = [] {
    nvtxSemanticsCounter_t result{};
    result.header.structSize = sizeof(nvtxSemanticsCounter_t);
    result.header.semanticId = NVTX_SEMANTIC_ID_COUNTERS_V1;
    result.header.version = NVTX_COUNTER_SEMANTIC_VERSION;
    result.flags = NVTX_COUNTER_FLAG_LIMIT_MIN |
        NVTX_COUNTER_FLAG_VALUETYPE_ABSOLUTE |
        NVTX_COUNTER_FLAG_INTERPOLATION_UNTIL_NEXT;
    result.unit = "bytes";
    result.unitScaleNumerator = 1;
    result.unitScaleDenominator = 1;
    result.limitType = NVTX_COUNTER_LIMIT_I64;
    result.min.i64 = 0;
    return result;
  }();
  return semantics;
}

uint64_t registerNvtxScope(std::string_view path, uint64_t parentScope) {
  nvtxScopeAttr_t attributes{};
  attributes.structSize = sizeof(nvtxScopeAttr_t);
  attributes.path = path.data();
  attributes.parentScope = parentScope;
  attributes.scopeId = NVTX_SCOPE_NONE;
  return nvtxScopeRegister(nullptr, &attributes);
}

uint64_t registerNvtxByteCounter(std::string_view name, uint64_t scopeId) {
  nvtxCounterAttr_t attributes{};
  attributes.structSize = sizeof(nvtxCounterAttr_t);
  attributes.schemaId = NVTX_PAYLOAD_ENTRY_TYPE_INT64;
  attributes.name = name.data();
  attributes.description =
      "Logical GPU allocation bytes attributed by Velox-cuDF.";
  attributes.scopeId = scopeId;
  attributes.semantics = &nvtxByteCounterSemantics().header;
  attributes.counterId = NVTX_COUNTER_ID_NONE;
  return nvtxCounterRegister(nullptr, &attributes);
}

int64_t nvtxByteValue(uint64_t bytes) {
  return static_cast<int64_t>(std::min(
      bytes, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
}

void sampleNvtxCounter(uint64_t counterId, uint64_t bytes) {
  if (counterId != NVTX_COUNTER_ID_NONE) {
    nvtxCounterSampleInt64(nullptr, counterId, nvtxByteValue(bytes));
  }
}

std::string nvtxScopeField(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const auto character : value) {
    if (character == '\\' || character == '/' || character == '[' ||
        character == ']') {
      result.push_back('\\');
    }
    result.push_back(character);
  }
  return result;
}

struct NvtxCounter {
  NvtxCounter(std::string counterName, uint64_t scopeId)
      : name(std::move(counterName)),
        id(registerNvtxByteCounter(name, scopeId)) {
    sampleNvtxCounter(id, 0);
  }

  std::string name;
  uint64_t id;
};

struct NvtxScope {
  NvtxScope(std::string scopeName, uint64_t parentScope)
      : name(std::move(scopeName)), id(registerNvtxScope(name, parentScope)) {}

  std::string name;
  uint64_t id;
};

struct NvtxScopeCounter {
  NvtxScopeCounter(
      std::string scopeName,
      std::string counterName,
      uint64_t parentScope)
      : scope(std::move(scopeName), parentScope),
        counter(std::move(counterName), scope.id) {}

  NvtxScope scope;
  NvtxCounter counter;
};

struct NvtxOwnerCounter {
  NvtxOwnerCounter(
      std::string name,
      uint64_t scopeId,
      uint64_t queryCounterId,
      uint64_t taskCounterId,
      uint64_t planNodeCounterId)
      : counter(std::move(name), scopeId),
        queryCounterId(queryCounterId),
        taskCounterId(taskCounterId),
        planNodeCounterId(planNodeCounterId) {}

  NvtxCounter counter;
  uint64_t queryCounterId;
  uint64_t taskCounterId;
  uint64_t planNodeCounterId;
};

struct NvtxCounterState {
  std::mutex mutex;
  bool initialized{false};
  uint64_t globalCounterId{NVTX_COUNTER_ID_NONE};
  uint64_t globalPeakCounterId{NVTX_COUNTER_ID_NONE};
  uint64_t lastGlobalPeakBytes{0};
  std::unordered_map<std::string, std::unique_ptr<NvtxScopeCounter>> queries;
  std::unordered_map<std::string, std::unique_ptr<NvtxScopeCounter>> tasks;
  std::unordered_map<uint64_t, std::unique_ptr<NvtxCounter>> planNodes;
  std::unordered_map<std::string, std::unique_ptr<NvtxScope>> drivers;
  std::unordered_map<uint64_t, std::unique_ptr<NvtxOwnerCounter>> owners;
};

NvtxCounterState& nvtxCounterState() {
  static auto* state = new NvtxCounterState;
  return *state;
}

void initializeNvtxCountersLocked(NvtxCounterState& state) {
  if (state.initialized) {
    return;
  }
  state.globalCounterId = registerNvtxByteCounter(
      "Velox-cuDF overall logical live bytes", NVTX_SCOPE_CURRENT_SW_PROCESS);
  state.globalPeakCounterId = registerNvtxByteCounter(
      "Velox-cuDF overall logical peak bytes", NVTX_SCOPE_CURRENT_SW_PROCESS);
  sampleNvtxCounter(state.globalCounterId, 0);
  sampleNvtxCounter(state.globalPeakCounterId, 0);
  state.initialized = true;
}

std::string nvtxDriverKey(const GpuMemoryOwner& owner) {
  return owner.taskUuid + '\x1f' + std::to_string(owner.pipelineId) + '\x1f' +
      std::to_string(owner.driverId);
}

void registerNvtxCounterOwner(
    uint64_t ownerId,
    uint64_t planNodeId,
    const GpuMemoryOwner& owner) noexcept {
  try {
    auto& state = nvtxCounterState();
    std::lock_guard<std::mutex> lock(state.mutex);
    initializeNvtxCountersLocked(state);
    if (state.owners.contains(ownerId)) {
      return;
    }

    if (ownerId == 0) {
      state.owners.emplace(
          ownerId,
          std::make_unique<NvtxOwnerCounter>(
              "Velox-cuDF unattributed logical live bytes",
              NVTX_SCOPE_CURRENT_SW_PROCESS,
              NVTX_COUNTER_ID_NONE,
              NVTX_COUNTER_ID_NONE,
              NVTX_COUNTER_ID_NONE));
      return;
    }

    auto query = state.queries.find(owner.queryId);
    if (query == state.queries.end()) {
      query = state.queries
                  .emplace(
                      owner.queryId,
                      std::make_unique<NvtxScopeCounter>(
                          "Plan | query=" +
                              nvtxScopeField(displayField(owner.queryId)),
                          "Plan logical live bytes",
                          NVTX_SCOPE_ROOT))
                  .first;
    }

    auto task = state.tasks.find(owner.taskUuid);
    if (task == state.tasks.end()) {
      task = state.tasks
                 .emplace(
                     owner.taskUuid,
                     std::make_unique<NvtxScopeCounter>(
                         "PlanFragment task | " +
                             nvtxScopeField(displayField(owner.taskId)) +
                             " | uuid=" +
                             nvtxScopeField(displayField(owner.taskUuid)),
                         "PlanFragment task logical live bytes",
                         query->second->scope.id))
                 .first;
    }

    auto planNode = state.planNodes.find(planNodeId);
    if (planNode == state.planNodes.end()) {
      planNode = state.planNodes
                     .emplace(
                         planNodeId,
                         std::make_unique<NvtxCounter>(
                             "PlanNode | " + displayPlanNodeType(owner) +
                                 " | plan=" + displayField(owner.planNodeId) +
                                 " | logical live bytes",
                             task->second->scope.id))
                     .first;
    }

    const auto driverKey = nvtxDriverKey(owner);
    auto driver = state.drivers.find(driverKey);
    if (driver == state.drivers.end()) {
      driver =
          state.drivers
              .emplace(
                  driverKey,
                  std::make_unique<NvtxScope>(
                      "Driver | pipeline=" + std::to_string(owner.pipelineId) +
                          " | driver=" + std::to_string(owner.driverId),
                      task->second->scope.id))
              .first;
    }

    state.owners.emplace(
        ownerId,
        std::make_unique<NvtxOwnerCounter>(
            "Operator | " + displayField(owner.operatorType) +
                " | operator=" + std::to_string(owner.operatorId) +
                " | plan=" + displayField(owner.planNodeId) + " " +
                displayPlanNodeType(owner) + " | logical live bytes",
            driver->second->id,
            query->second->counter.id,
            task->second->counter.id,
            planNode->second->id));
  } catch (...) {
    // Native profiler diagnostics must not alter query execution.
  }
}

void emitNvtxCounterUpdate(const GpuMemoryTraceUpdate& update) noexcept {
  try {
    auto& state = nvtxCounterState();
    std::lock_guard<std::mutex> lock(state.mutex);
    initializeNvtxCountersLocked(state);
    sampleNvtxCounter(state.globalCounterId, update.globalCurrentBytes);
    if (update.globalPeakBytes != state.lastGlobalPeakBytes) {
      sampleNvtxCounter(state.globalPeakCounterId, update.globalPeakBytes);
      state.lastGlobalPeakBytes = update.globalPeakBytes;
    }

    const auto owner = state.owners.find(update.ownerId);
    if (owner == state.owners.end()) {
      return;
    }
    sampleNvtxCounter(owner->second->queryCounterId, update.queryCurrentBytes);
    sampleNvtxCounter(owner->second->taskCounterId, update.taskCurrentBytes);
    sampleNvtxCounter(
        owner->second->planNodeCounterId, update.planNodeCurrentBytes);
    sampleNvtxCounter(owner->second->counter.id, update.ownerCurrentBytes);
  } catch (...) {
    // Native profiler diagnostics must not alter query execution.
  }
}

void emitNvtxMark(std::string_view name) noexcept {
  if (name.empty()) {
    return;
  }
  try {
    const std::string text{name};
    nvtxMarkA(text.c_str());
  } catch (...) {
    // Native profiler diagnostics must not alter query execution.
  }
}

void sealGpuMemoryTraceThread() noexcept {
  try {
    velox_cudf_trace::TrackEvent::Trace(
        [](velox_cudf_trace::TrackEvent::TraceContext context) {
          context.AddEmptyTracePacket();
        });
  } catch (...) {
    // Trace transport must not alter query execution.
  }
}

struct QueryTrace {
  explicit QueryTrace(const GpuMemoryOwner& owner)
      : name("Query | " + displayField(owner.queryId)),
        track(
            perfetto::DynamicString{name},
            std::hash<std::string>{}(owner.queryId),
            rootTrack()) {}

  std::string name;
  perfetto::NamedTrack track;
};

struct TaskTrace {
  TaskTrace(const GpuMemoryOwner& owner, perfetto::NamedTrack queryTrack)
      : name(
            "Task | " + displayField(owner.taskId) +
            " | uuid=" + displayField(owner.taskUuid)),
        track(
            perfetto::DynamicString{name},
            std::hash<std::string>{}(owner.taskUuid),
            queryTrack) {}

  std::string name;
  perfetto::NamedTrack track;
};

// Use ledger IDs only on hierarchy tracks. Perfetto XORs a child ID with its
// parent UUID, so repeating the parent's ID on a counter can cancel the
// identity and collide with a sibling whose names share the same suffix.
struct PlanNodeTrace {
  PlanNodeTrace(
      uint64_t id,
      const GpuMemoryOwner& owner,
      perfetto::NamedTrack taskTrack)
      : name(
            displayPlanNodeType(owner) + " | " +
            displayField(owner.planNodeId)),
        counterName(
            "PlanNode RMM logical live bytes | query=" +
            displayField(owner.queryId) +
            " | task=" + displayField(owner.taskUuid) +
            " | plan=" + displayField(owner.planNodeId) +
            " | type=" + displayPlanNodeType(owner)),
        track(perfetto::DynamicString{name}, id, taskTrack),
        counter(
            perfetto::CounterTrack(perfetto::DynamicString{counterName}, track)
                .set_unit(
                    perfetto::protos::pbzero::CounterDescriptor::
                        UNIT_SIZE_BYTES)
                .set_category("velox.cudf.memory")) {}

  std::string name;
  std::string counterName;
  perfetto::NamedTrack track;
  perfetto::CounterTrack counter;
};

struct OwnerTrace {
  OwnerTrace(
      uint64_t id,
      uint64_t planNodeId,
      GpuMemoryOwner owner,
      perfetto::NamedTrack planNodeTrack)
      : ownerId(id),
        planNodeId(planNodeId),
        owner(std::move(owner)),
        name(
            "Operator | " + displayField(this->owner.operatorType) +
            " | pipeline=" + std::to_string(this->owner.pipelineId) +
            " | driver=" + std::to_string(this->owner.driverId) +
            " | operator=" + std::to_string(this->owner.operatorId)),
        counterName(
            "Operator RMM logical live bytes | query=" +
            displayField(this->owner.queryId) +
            " | task=" + displayField(this->owner.taskUuid) +
            " | plan=" + displayField(this->owner.planNodeId) +
            " | pipeline=" + std::to_string(this->owner.pipelineId) +
            " | driver=" + std::to_string(this->owner.driverId) +
            " | operator=" + std::to_string(this->owner.operatorId) +
            " | type=" + displayField(this->owner.operatorType)),
        callsName("Calls | " + displayField(this->owner.operatorType)),
        track(perfetto::DynamicString{name}, id, planNodeTrack),
        counter(
            perfetto::CounterTrack(perfetto::DynamicString{counterName}, track)
                .set_unit(
                    perfetto::protos::pbzero::CounterDescriptor::
                        UNIT_SIZE_BYTES)
                .set_category("velox.cudf.memory")),
        calls(perfetto::DynamicString{callsName}, 0, track) {}

  uint64_t ownerId;
  uint64_t planNodeId;
  GpuMemoryOwner owner;
  std::string name;
  std::string counterName;
  std::string callsName;
  perfetto::NamedTrack track;
  perfetto::CounterTrack counter;
  perfetto::NamedTrack calls;
};

struct TraceState {
  std::mutex mutex;
  std::unique_ptr<perfetto::TracingSession> session;
  int fileDescriptor{-1};
  std::string path;
  std::unordered_map<std::string, std::unique_ptr<QueryTrace>> queries;
  std::unordered_map<std::string, std::unique_ptr<TaskTrace>> tasks;
  std::unordered_map<uint64_t, std::unique_ptr<PlanNodeTrace>> planNodes;
  std::unordered_map<uint64_t, std::unique_ptr<OwnerTrace>> owners;
};

TraceState& traceState() {
  static auto* state = new TraceState;
  return *state;
}

std::once_flag perfettoInitialization;
std::atomic<bool> tracingEnabled{false};

template <typename TrackType>
void registerTrackDescriptor(const TrackType& track) {
  velox_cudf_trace::TrackEvent::SetTrackDescriptor(track, track.Serialize());
}

void registerAllTrackDescriptorsLocked(const TraceState& state) {
  registerTrackDescriptor(rootTrack());
  registerTrackDescriptor(markerTrack());
  registerTrackDescriptor(globalTrack());
  registerTrackDescriptor(globalPeakTrack());
  for (const auto& [_, query] : state.queries) {
    registerTrackDescriptor(query->track);
  }
  for (const auto& [_, task] : state.tasks) {
    registerTrackDescriptor(task->track);
  }
  for (const auto& [_, planNode] : state.planNodes) {
    registerTrackDescriptor(planNode->track);
    registerTrackDescriptor(planNode->counter);
  }
  for (const auto& [_, owner] : state.owners) {
    registerTrackDescriptor(owner->track);
    registerTrackDescriptor(owner->counter);
    registerTrackDescriptor(owner->calls);
  }
}

std::string expandProcessId(std::string_view pattern) {
  std::string path{pattern};
  const std::string processId{std::to_string(::getpid())};
  std::size_t position{0};
  while ((position = path.find("%p", position)) != std::string::npos) {
    path.replace(position, 2, processId);
    position += processId.size();
  }
  return path;
}

void initializePerfetto() {
  std::call_once(perfettoInitialization, [] {
    perfetto::TracingInitArgs arguments;
    arguments.backends = perfetto::kInProcessBackend;
    // Absorb short allocation bursts from many drivers before Perfetto's
    // service thread drains producer pages. Track events otherwise use a
    // drop-on-exhaustion policy.
    arguments.shmem_size_hint_kb = kSharedMemorySizeKiB;
    arguments.shmem_page_size_hint_kb = kSharedMemoryPageSizeKiB;
    perfetto::Tracing::Initialize(arguments);
    velox_cudf_trace::TrackEvent::Register();
  });
}

OwnerTrace* findOwnerLocked(uint64_t ownerId) {
  auto& state = traceState();
  auto owner = state.owners.find(ownerId);
  if (owner == state.owners.end()) {
    owner = state.owners.find(0);
  }
  return owner == state.owners.end() ? nullptr : owner->second.get();
}

void emitOwnerMetadata(const OwnerTrace& ownerTrace) noexcept {
  PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(velox_cudf_trace);
  try {
    const auto& owner = ownerTrace.owner;
    TRACE_EVENT_INSTANT(
        "velox.cudf.operator",
        "Owner metadata",
        ownerTrace.calls,
        "owner_id",
        ownerTrace.ownerId,
        "plan_track_id",
        ownerTrace.planNodeId,
        "query_id",
        owner.queryId,
        "task_uuid",
        owner.taskUuid,
        "task_id",
        owner.taskId,
        "plan_node_id",
        owner.planNodeId,
        "plan_node_type",
        owner.planNodeType,
        "pipeline_id",
        owner.pipelineId,
        "driver_id",
        owner.driverId,
        "operator_id",
        owner.operatorId,
        "operator_type",
        owner.operatorType);
    sealGpuMemoryTraceThread();
  } catch (...) {
    // Trace metadata is best-effort and must not alter query execution.
  }
}

} // namespace

bool startGpuMemoryTrace(std::string_view pathPattern) noexcept {
  if (pathPattern.empty()) {
    return false;
  }

  try {
    initializePerfetto();
    auto& state = traceState();
    std::vector<const OwnerTrace*> registeredOwners;
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      if (state.session != nullptr) {
        return true;
      }

      state.path = expandProcessId(pathPattern);
      const std::filesystem::path outputPath{state.path};
      if (outputPath.has_parent_path()) {
        std::filesystem::create_directories(outputPath.parent_path());
      }
      state.fileDescriptor = ::open(
          state.path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
      if (state.fileDescriptor < 0) {
        LOG(ERROR) << "Could not open GPU-memory Perfetto trace: "
                   << state.path;
        state.path.clear();
        return false;
      }

      perfetto::TraceConfig configuration;
      configuration.add_buffers()->set_size_kb(64 * 1024);
      auto* dataSource = configuration.add_data_sources()->mutable_config();
      dataSource->set_name("track_event");
      perfetto::protos::gen::TrackEventConfig trackEventConfiguration;
      trackEventConfiguration.add_enabled_categories("velox.cudf.memory");
      trackEventConfiguration.add_enabled_categories("velox.cudf.operator");
      dataSource->set_track_event_config_raw(
          trackEventConfiguration.SerializeAsString());
      configuration.set_write_into_file(true);
      configuration.set_file_write_period_ms(1'000);
      configuration.set_max_file_size_bytes(kMaximumTraceFileBytes);

      state.session = perfetto::Tracing::NewTrace(perfetto::kInProcessBackend);
      state.session->Setup(configuration, state.fileDescriptor);
      state.session->StartBlocking();
      registerAllTrackDescriptorsLocked(state);
      tracingEnabled.store(true, std::memory_order_release);
      {
        PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(velox_cudf_trace);
        LOG(INFO) << "GPU-memory Perfetto categories enabled: memory="
                  << TRACE_EVENT_CATEGORY_ENABLED("velox.cudf.memory")
                  << " operator="
                  << TRACE_EVENT_CATEGORY_ENABLED("velox.cudf.operator");
      }

      registeredOwners.reserve(state.owners.size());
      for (const auto& [_, owner] : state.owners) {
        registeredOwners.push_back(owner.get());
      }
    }

    for (const auto* owner : registeredOwners) {
      emitOwnerMetadata(*owner);
    }
    LOG(INFO) << "GPU-memory Perfetto trace started: " << gpuMemoryTracePath();
    return true;
  } catch (const std::exception& error) {
    tracingEnabled.store(false, std::memory_order_release);
    LOG(ERROR) << "Could not start GPU-memory Perfetto trace: " << error.what();
    stopGpuMemoryTrace();
    return false;
  } catch (...) {
    tracingEnabled.store(false, std::memory_order_release);
    LOG(ERROR) << "Could not start GPU-memory Perfetto trace";
    stopGpuMemoryTrace();
    return false;
  }
}

void stopGpuMemoryTrace() noexcept {
  tracingEnabled.store(false, std::memory_order_release);
  try {
    auto& state = traceState();
    std::unique_ptr<perfetto::TracingSession> session;
    int fileDescriptor{-1};
    std::string path;
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      session = std::move(state.session);
      fileDescriptor = std::exchange(state.fileDescriptor, -1);
      path = std::exchange(state.path, std::string{});
    }
    if (session != nullptr) {
      velox_cudf_trace::TrackEvent::Flush();
      if (!session->FlushBlocking()) {
        LOG(WARNING) << "GPU-memory Perfetto session flush timed out";
      }
      session->StopBlocking();
    }
    if (fileDescriptor >= 0) {
      ::close(fileDescriptor);
    }
    if (!path.empty()) {
      LOG(INFO) << "GPU-memory Perfetto trace stopped: " << path;
    }
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.queries.clear();
      state.tasks.clear();
      state.planNodes.clear();
      state.owners.clear();
    }
  } catch (...) {
    LOG(ERROR) << "Could not cleanly stop GPU-memory Perfetto trace";
  }
}

bool gpuMemoryTraceEnabled() noexcept {
  return tracingEnabled.load(std::memory_order_acquire);
}

std::string gpuMemoryTracePath() {
  auto& state = traceState();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.path;
}

void markGpuMemoryTrace(std::string_view name) noexcept {
  if (name.empty()) {
    return;
  }
  emitNvtxMark(name);
  if (!gpuMemoryTraceEnabled()) {
    return;
  }

  PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(velox_cudf_trace);
  try {
    std::optional<perfetto::NamedTrack> track;
    track.emplace(markerTrack());
    const auto active = gpu_memory_detail::activeGpuMemoryOwner();
    {
      auto& state = traceState();
      std::lock_guard<std::mutex> lock(state.mutex);
      if (auto* owner = findOwnerLocked(active.ownerId)) {
        track.emplace(owner->calls);
      }
    }
    TRACE_EVENT_INSTANT(
        "velox.cudf.operator",
        perfetto::DynamicString(name.data(), name.size()),
        *track,
        "owner_id",
        active.ownerId);
    velox_cudf_trace::TrackEvent::Flush();
  } catch (...) {
    // Custom markers are diagnostics only.
  }
}

GpuMemoryOperatorCall::GpuMemoryOperatorCall(
    exec::Operator* op,
    std::string_view callName) noexcept {
  const auto previous = gpu_memory_detail::activateGpuMemoryOperator(op);
  previousTracker_ = previous.tracker;
  previousOwnerId_ = previous.ownerId;
  const auto current = gpu_memory_detail::activeGpuMemoryOwner();
  ownerId_ = current.ownerId;
  traceSliceStarted_ =
      gpu_memory_detail::beginGpuMemoryOperatorCall(ownerId_, callName);
}

GpuMemoryOperatorCall::~GpuMemoryOperatorCall() {
  if (traceSliceStarted_) {
    gpu_memory_detail::endGpuMemoryOperatorCall(ownerId_);
  }
  gpu_memory_detail::restoreGpuMemoryOwner(
      gpu_memory_detail::GpuMemoryActiveOwner{
          previousTracker_, previousOwnerId_});
}

namespace gpu_memory_detail {

uint64_t gpuMemoryTraceNowNs() noexcept {
  return velox_cudf_trace::TrackEvent::GetTraceTimeNs();
}

void registerGpuMemoryTraceOwner(
    uint64_t ownerId,
    uint64_t planNodeId,
    const GpuMemoryOwner& owner) noexcept {
  registerNvtxCounterOwner(ownerId, planNodeId, owner);
  if (!gpuMemoryTraceEnabled()) {
    return;
  }

  try {
    auto& state = traceState();
    const OwnerTrace* registeredOwner{nullptr};
    bool queryInserted{false};
    bool taskInserted{false};
    bool planNodeInserted{false};
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      if (!gpuMemoryTraceEnabled()) {
        return;
      }
      if (state.owners.contains(ownerId)) {
        return;
      }

      auto query = state.queries.find(owner.queryId);
      if (query == state.queries.end()) {
        query = state.queries
                    .emplace(owner.queryId, std::make_unique<QueryTrace>(owner))
                    .first;
        queryInserted = true;
      }

      auto task = state.tasks.find(owner.taskUuid);
      if (task == state.tasks.end()) {
        task = state.tasks
                   .emplace(
                       owner.taskUuid,
                       std::make_unique<TaskTrace>(owner, query->second->track))
                   .first;
        taskInserted = true;
      }

      auto planNode = state.planNodes.find(planNodeId);
      if (planNode == state.planNodes.end()) {
        planNode = state.planNodes
                       .emplace(
                           planNodeId,
                           std::make_unique<PlanNodeTrace>(
                               planNodeId, owner, task->second->track))
                       .first;
        planNodeInserted = true;
      }

      auto [ownerIt, _] = state.owners.emplace(
          ownerId,
          std::make_unique<OwnerTrace>(
              ownerId, planNodeId, owner, planNode->second->track));
      registeredOwner = ownerIt->second.get();
      if (queryInserted) {
        registerTrackDescriptor(query->second->track);
      }
      if (taskInserted) {
        registerTrackDescriptor(task->second->track);
      }
      if (planNodeInserted) {
        registerTrackDescriptor(planNode->second->track);
        registerTrackDescriptor(planNode->second->counter);
      }
      registerTrackDescriptor(registeredOwner->track);
      registerTrackDescriptor(registeredOwner->counter);
      registerTrackDescriptor(registeredOwner->calls);
    }
    emitOwnerMetadata(*registeredOwner);
  } catch (...) {
    emitGpuMemoryTraceDataLoss("owner trace registration exception", 0);
  }
}

void emitGpuMemoryTraceUpdate(const GpuMemoryTraceUpdate& update) noexcept {
  emitNvtxCounterUpdate(update);
  if (!gpuMemoryTraceEnabled()) {
    return;
  }

  PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(velox_cudf_trace);
  try {
    std::optional<perfetto::CounterTrack> planCounter;
    std::optional<perfetto::CounterTrack> ownerCounter;
    {
      auto& state = traceState();
      std::lock_guard<std::mutex> lock(state.mutex);
      const auto plan = state.planNodes.find(update.planNodeId);
      auto* owner = findOwnerLocked(update.ownerId);
      if (plan == state.planNodes.end() || owner == nullptr) {
        emitGpuMemoryTraceDataLoss(
            "counter update without registered trace owner", update.sequence);
        return;
      }
      planCounter.emplace(plan->second->counter);
      ownerCounter.emplace(owner->counter);
    }

    TRACE_COUNTER(
        "velox.cudf.memory",
        globalTrack(),
        update.timestampNs,
        static_cast<int64_t>(update.globalCurrentBytes));
    if (update.deltaBytes > 0 &&
        update.globalCurrentBytes == update.globalPeakBytes) {
      TRACE_COUNTER(
          "velox.cudf.memory",
          globalPeakTrack(),
          update.timestampNs,
          static_cast<int64_t>(update.globalPeakBytes));
    }
    TRACE_COUNTER(
        "velox.cudf.memory",
        *planCounter,
        update.timestampNs,
        static_cast<int64_t>(update.planNodeCurrentBytes));
    TRACE_COUNTER(
        "velox.cudf.memory",
        *ownerCounter,
        update.timestampNs,
        static_cast<int64_t>(update.ownerCurrentBytes));
    sealGpuMemoryTraceThread();
  } catch (...) {
    emitGpuMemoryTraceDataLoss(
        "counter trace emission exception", update.sequence);
  }
}

void resetGpuMemoryNvtxCounters() noexcept {
  try {
    auto& state = nvtxCounterState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
      return;
    }
    for (const auto& [_, owner] : state.owners) {
      sampleNvtxCounter(owner->counter.id, 0);
    }
    for (const auto& [_, planNode] : state.planNodes) {
      sampleNvtxCounter(planNode->id, 0);
    }
    for (const auto& [_, task] : state.tasks) {
      sampleNvtxCounter(task->counter.id, 0);
    }
    for (const auto& [_, query] : state.queries) {
      sampleNvtxCounter(query->counter.id, 0);
    }
    sampleNvtxCounter(state.globalCounterId, 0);
    sampleNvtxCounter(state.globalPeakCounterId, 0);

    state.owners.clear();
    state.drivers.clear();
    state.planNodes.clear();
    state.tasks.clear();
    state.queries.clear();
    state.globalCounterId = NVTX_COUNTER_ID_NONE;
    state.globalPeakCounterId = NVTX_COUNTER_ID_NONE;
    state.lastGlobalPeakBytes = 0;
    state.initialized = false;
  } catch (...) {
    // Native profiler diagnostics must not alter resource cleanup.
  }
}

void emitGpuMemoryTraceOom(
    uint64_t ownerId,
    std::size_t requestedBytes,
    uint64_t globalCurrentBytes,
    uint64_t globalPeakBytes,
    uint64_t planNodeCurrentBytes,
    uint64_t ownerCurrentBytes,
    std::size_t cudaFreeBytes,
    std::size_t cudaTotalBytes,
    std::string_view cudaStatus) noexcept {
  emitNvtxMark(
      "GPU allocation failed | owner=" + std::to_string(ownerId) +
      " | requested=" + std::to_string(requestedBytes) +
      " | logical_live=" + std::to_string(globalCurrentBytes) +
      " | logical_peak=" + std::to_string(globalPeakBytes));
  if (!gpuMemoryTraceEnabled()) {
    return;
  }

  PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(velox_cudf_trace);
  try {
    std::optional<perfetto::NamedTrack> track;
    track.emplace(markerTrack());
    {
      auto& state = traceState();
      std::lock_guard<std::mutex> lock(state.mutex);
      if (auto* owner = findOwnerLocked(ownerId)) {
        track.emplace(owner->calls);
      }
    }
    const std::string cudaStatusText{cudaStatus};
    TRACE_EVENT_INSTANT(
        "velox.cudf.memory",
        "GPU allocation failed",
        *track,
        "owner_id",
        ownerId,
        "requested_bytes",
        requestedBytes,
        "overall_logical_live_bytes",
        globalCurrentBytes,
        "overall_logical_peak_bytes",
        globalPeakBytes,
        "plan_node_logical_live_bytes",
        planNodeCurrentBytes,
        "operator_logical_live_bytes",
        ownerCurrentBytes,
        "cuda_free_bytes",
        cudaFreeBytes,
        "cuda_total_bytes",
        cudaTotalBytes,
        "cuda_status",
        cudaStatusText);
    velox_cudf_trace::TrackEvent::Flush();
  } catch (...) {
    // An allocation failure must be rethrown regardless of tracing health.
  }
}

void emitGpuMemoryTraceDataLoss(
    std::string_view reason,
    uint64_t sequence) noexcept {
  emitNvtxMark(
      "GPU memory trace data loss | sequence=" + std::to_string(sequence) +
      " | reason=" + std::string{reason});
  if (!gpuMemoryTraceEnabled()) {
    return;
  }

  PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(velox_cudf_trace);
  try {
    TRACE_EVENT_INSTANT(
        "velox.cudf.memory",
        "GPU memory trace data loss",
        markerTrack(),
        "reason",
        std::string{reason},
        "last_complete_sequence",
        sequence);
    velox_cudf_trace::TrackEvent::Flush();
  } catch (...) {
    // No additional recovery is possible for a failed diagnostic event.
  }
}

bool beginGpuMemoryOperatorCall(
    uint64_t ownerId,
    std::string_view callName) noexcept {
  if (!gpuMemoryTraceEnabled()) {
    return false;
  }

  PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(velox_cudf_trace);
  try {
    std::optional<perfetto::NamedTrack> calls;
    {
      auto& state = traceState();
      std::lock_guard<std::mutex> lock(state.mutex);
      auto* owner = findOwnerLocked(ownerId);
      if (owner == nullptr) {
        return false;
      }
      calls.emplace(owner->calls);
    }
    TRACE_EVENT_BEGIN(
        "velox.cudf.operator",
        perfetto::DynamicString(callName.data(), callName.size()),
        *calls,
        "owner_id",
        ownerId);
    return true;
  } catch (...) {
    return false;
  }
}

void endGpuMemoryOperatorCall(uint64_t ownerId) noexcept {
  if (!gpuMemoryTraceEnabled()) {
    return;
  }

  PERFETTO_USE_CATEGORIES_FROM_NAMESPACE_SCOPED(velox_cudf_trace);
  try {
    std::optional<perfetto::NamedTrack> calls;
    {
      auto& state = traceState();
      std::lock_guard<std::mutex> lock(state.mutex);
      auto* owner = findOwnerLocked(ownerId);
      if (owner == nullptr) {
        return;
      }
      calls.emplace(owner->calls);
    }
    TRACE_EVENT_END("velox.cudf.operator", *calls);
    sealGpuMemoryTraceThread();
  } catch (...) {
    // Operator execution must not depend on trace slice closure.
  }
}

} // namespace gpu_memory_detail
} // namespace facebook::velox::cudf_velox
