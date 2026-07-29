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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/NvtxGpuMemoryCounters.h"

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCounters.h>
#include <nvtx3/nvToolsExtPayload.h>
#include <nvtx3/nvToolsExtSemanticsCounters.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

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

} // namespace

namespace gpu_memory_detail {

void registerGpuMemoryNvtxOwner(
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

void emitGpuMemoryNvtxUpdate(const GpuMemoryTraceUpdate& update) noexcept {
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

} // namespace gpu_memory_detail

} // namespace facebook::velox::cudf_velox
