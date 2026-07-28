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

#include "velox/experimental/cudf/CudfDefaultStreamOverload.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Task.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/aligned.hpp>
#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <common/base/Exceptions.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace facebook::velox::cudf_velox {

namespace {

constexpr std::string_view kGpuAllocationBytes{"gpuMemoryAllocationBytes"};
constexpr std::string_view kGpuAllocationCount{"gpuMemoryAllocationCount"};
constexpr std::string_view kGpuFailedAllocationBytes{
    "gpuMemoryFailedAllocationBytes"};
constexpr std::string_view kGpuFailedAllocationCount{
    "gpuMemoryFailedAllocationCount"};
constexpr std::string_view kGpuQueryPeakLiveBytes{"gpuQueryPeakLiveBytes"};
constexpr std::string_view kGpuQueryMainPeakLiveBytes{
    "gpuQueryMainPeakLiveBytes"};
constexpr std::string_view kGpuQueryOutputPeakLiveBytes{
    "gpuQueryOutputPeakLiveBytes"};
constexpr std::string_view kGpuPlanNodePeakLiveBytes{
    "gpuPlanNodePeakLiveBytes"};
constexpr std::string_view kGpuPlanNodeMainPeakLiveBytes{
    "gpuPlanNodeMainPeakLiveBytes"};
constexpr std::string_view kGpuPlanNodeOutputPeakLiveBytes{
    "gpuPlanNodeOutputPeakLiveBytes"};
constexpr std::string_view kGpuOperatorPeakLiveBytes{
    "gpuOperatorPeakLiveBytes"};
constexpr std::string_view kGpuOperatorMainPeakLiveBytes{
    "gpuOperatorMainPeakLiveBytes"};
constexpr std::string_view kGpuOperatorOutputPeakLiveBytes{
    "gpuOperatorOutputPeakLiveBytes"};

size_t resourceIndex(GpuMemoryResourceKind kind) {
  return kind == GpuMemoryResourceKind::kMain ? 0 : 1;
}

struct OwnerKey {
  GpuMemoryOwner owner;
  GpuMemoryResourceKind kind;

  bool operator==(const OwnerKey&) const = default;
};

void hashCombine(size_t& seed, size_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

struct OwnerKeyHash {
  size_t operator()(const OwnerKey& key) const {
    size_t result = std::hash<std::string>{}(key.owner.taskUuid);
    hashCombine(result, std::hash<std::string>{}(key.owner.taskId));
    hashCombine(result, std::hash<std::string>{}(key.owner.queryId));
    hashCombine(result, std::hash<std::string>{}(key.owner.planNodeId));
    hashCombine(result, std::hash<int32_t>{}(key.owner.operatorId));
    hashCombine(result, std::hash<std::string>{}(key.owner.operatorType));
    hashCombine(result, std::hash<int>{}(static_cast<int>(key.kind)));
    return result;
  }
};

struct PlanNodeKey {
  std::string queryId;
  std::string planNodeId;

  bool operator==(const PlanNodeKey&) const = default;
};

struct PlanNodeKeyHash {
  size_t operator()(const PlanNodeKey& key) const {
    size_t result = std::hash<std::string>{}(key.queryId);
    hashCombine(result, std::hash<std::string>{}(key.planNodeId));
    return result;
  }
};

struct OperatorKey {
  std::string queryId;
  std::string planNodeId;
  int32_t operatorId;
  std::string operatorType;

  bool operator==(const OperatorKey&) const = default;
};

struct OperatorKeyHash {
  size_t operator()(const OperatorKey& key) const {
    size_t result = std::hash<std::string>{}(key.queryId);
    hashCombine(result, std::hash<std::string>{}(key.planNodeId));
    hashCombine(result, std::hash<int32_t>{}(key.operatorId));
    hashCombine(result, std::hash<std::string>{}(key.operatorType));
    return result;
  }
};

struct LiveBytes {
  uint64_t current{0};
  uint64_t peak{0};
};

struct AggregateLiveBytes {
  LiveBytes combined;
  std::array<LiveBytes, 2> resources;
  bool retired{false};
};

struct AggregatePeakUpdate {
  std::optional<uint64_t> combined;
  std::optional<uint64_t> resource;
};

AggregatePeakUpdate addLiveBytes(
    AggregateLiveBytes& aggregate,
    GpuMemoryResourceKind kind,
    uint64_t bytes) {
  AggregatePeakUpdate update;
  aggregate.combined.current += bytes;
  if (aggregate.combined.current > aggregate.combined.peak) {
    aggregate.combined.peak = aggregate.combined.current;
    update.combined = aggregate.combined.peak;
  }

  auto& resource = aggregate.resources[resourceIndex(kind)];
  resource.current += bytes;
  if (resource.current > resource.peak) {
    resource.peak = resource.current;
    update.resource = resource.peak;
  }
  return update;
}

void removeLiveBytes(
    AggregateLiveBytes& aggregate,
    GpuMemoryResourceKind kind,
    uint64_t bytes) {
  aggregate.combined.current -= std::min(aggregate.combined.current, bytes);
  auto& resource = aggregate.resources[resourceIndex(kind)];
  resource.current -= std::min(resource.current, bytes);
}

struct AllocationRecord {
  uint64_t bytes;
  OwnerKey ownerKey;
  AggregateLiveBytes* queryAggregate;
  AggregateLiveBytes* planNodeAggregate;
  AggregateLiveBytes* operatorAggregate;
};

struct OwnerRecord {
  GpuMemoryOwnerSnapshot snapshot;
  bool retired{false};
};

void addAllocation(
    uint64_t bytes,
    uint64_t& currentBytes,
    uint64_t& peakBytes,
    uint64_t& totalBytes,
    uint64_t& currentAllocations,
    uint64_t& peakAllocations,
    uint64_t& totalAllocations) {
  currentBytes += bytes;
  peakBytes = std::max(peakBytes, currentBytes);
  totalBytes += bytes;
  ++currentAllocations;
  peakAllocations = std::max(peakAllocations, currentAllocations);
  ++totalAllocations;
}

void removeAllocation(
    uint64_t bytes,
    uint64_t& currentBytes,
    uint64_t& currentAllocations) {
  currentBytes -= std::min(currentBytes, bytes);
  if (currentAllocations > 0) {
    --currentAllocations;
  }
}

} // namespace

class GpuMemoryAllocationTracker::Impl {
 public:
  Impl()
      : resources_{
            GpuMemoryResourceSnapshot{
                GpuMemoryResourceKind::kMain,
                0,
                0,
                0,
                0,
                0,
                0},
            GpuMemoryResourceSnapshot{
                GpuMemoryResourceKind::kOutput,
                0,
                0,
                0,
                0,
                0,
                0}} {}

  GpuMemoryPeakUpdate recordAllocation(
      void* address,
      std::size_t bytes,
      GpuMemoryResourceKind kind,
      const GpuMemoryOwner& owner) noexcept {
    if (address == nullptr) {
      return {};
    }

    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (allocations_.contains(address)) {
        return {};
      }

      OwnerKey ownerKey{owner, kind};
      auto [ownerIt, _] = owners_.try_emplace(
          ownerKey,
          OwnerRecord{GpuMemoryOwnerSnapshot{owner, kind, 0, 0, 0, 0, 0, 0}});

      AggregateLiveBytes* queryAggregate{nullptr};
      AggregateLiveBytes* planNodeAggregate{nullptr};
      AggregateLiveBytes* operatorAggregate{nullptr};
      if (!owner.queryId.empty() && !owner.planNodeId.empty()) {
        queryAggregate =
            &queryAggregates_.try_emplace(owner.queryId).first->second;

        const PlanNodeKey planNodeKey{owner.queryId, owner.planNodeId};
        planNodeAggregate =
            &planNodeAggregates_.try_emplace(planNodeKey).first->second;

        if (owner.operatorId >= 0 && !owner.operatorType.empty()) {
          const OperatorKey operatorKey{
              owner.queryId,
              owner.planNodeId,
              owner.operatorId,
              owner.operatorType};
          operatorAggregate =
              &operatorAggregates_.try_emplace(operatorKey).first->second;
        }
      }

      auto [allocationIt, inserted] = allocations_.try_emplace(
          address,
          AllocationRecord{
              static_cast<uint64_t>(bytes),
              ownerKey,
              queryAggregate,
              planNodeAggregate,
              operatorAggregate});
      if (!inserted) {
        return {};
      }

      auto& ownerStats = ownerIt->second.snapshot;
      addAllocation(
          bytes,
          ownerStats.currentBytes,
          ownerStats.peakBytes,
          ownerStats.totalBytes,
          ownerStats.currentAllocations,
          ownerStats.peakAllocations,
          ownerStats.totalAllocations);

      auto& resourceStats = resources_[resourceIndex(kind)];
      addAllocation(
          bytes,
          resourceStats.currentBytes,
          resourceStats.peakBytes,
          resourceStats.totalBytes,
          resourceStats.currentAllocations,
          resourceStats.peakAllocations,
          resourceStats.totalAllocations);

      if (queryAggregate == nullptr) {
        return {};
      }

      GpuMemoryPeakUpdate update;
      const auto queryUpdate = addLiveBytes(*queryAggregate, kind, bytes);
      update.queryPeakBytes = queryUpdate.combined;
      update.queryResourcePeakBytes = queryUpdate.resource;

      const auto planNodeUpdate = addLiveBytes(*planNodeAggregate, kind, bytes);
      update.planNodePeakBytes = planNodeUpdate.combined;
      update.planNodeResourcePeakBytes = planNodeUpdate.resource;

      if (operatorAggregate != nullptr) {
        const auto operatorUpdate =
            addLiveBytes(*operatorAggregate, kind, bytes);
        update.operatorPeakBytes = operatorUpdate.combined;
        update.operatorResourcePeakBytes = operatorUpdate.resource;
      }
      return update;
    } catch (...) {
      // Diagnostics must not turn a successful GPU allocation into a failure.
      return {};
    }
  }

  void recordDeallocation(void* address) noexcept {
    if (address == nullptr) {
      return;
    }

    try {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto allocationIt = allocations_.find(address);
      if (allocationIt == allocations_.end()) {
        return;
      }

      auto& allocation = allocationIt->second;
      const auto ownerIt = owners_.find(allocation.ownerKey);
      if (ownerIt != owners_.end()) {
        auto& ownerStats = ownerIt->second.snapshot;
        removeAllocation(
            allocation.bytes,
            ownerStats.currentBytes,
            ownerStats.currentAllocations);
        if (ownerStats.currentAllocations == 0 && ownerIt->second.retired) {
          owners_.erase(ownerIt);
        }
      }

      auto& resourceStats = resources_[resourceIndex(allocation.ownerKey.kind)];
      removeAllocation(
          allocation.bytes,
          resourceStats.currentBytes,
          resourceStats.currentAllocations);

      if (allocation.queryAggregate != nullptr) {
        removeLiveBytes(
            *allocation.queryAggregate,
            allocation.ownerKey.kind,
            allocation.bytes);
        removeLiveBytes(
            *allocation.planNodeAggregate,
            allocation.ownerKey.kind,
            allocation.bytes);
        if (allocation.operatorAggregate != nullptr) {
          removeLiveBytes(
              *allocation.operatorAggregate,
              allocation.ownerKey.kind,
              allocation.bytes);
        }
      }

      const bool queryRetired = allocation.queryAggregate != nullptr &&
          allocation.queryAggregate->retired;
      auto queryId = std::move(allocation.ownerKey.owner.queryId);
      allocations_.erase(allocationIt);
      if (queryRetired) {
        cleanupQuery(queryId);
      }
    } catch (...) {
      // Deallocation is noexcept and diagnostic bookkeeping is best effort.
    }
  }

  void retireTask(std::string_view taskUuid) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto ownerIt = owners_.begin(); ownerIt != owners_.end();) {
      auto& ownerRecord = ownerIt->second;
      if (ownerRecord.snapshot.owner.taskUuid != taskUuid) {
        ++ownerIt;
      } else if (ownerRecord.snapshot.currentAllocations == 0) {
        ownerIt = owners_.erase(ownerIt);
      } else {
        ownerRecord.retired = true;
        ++ownerIt;
      }
    }
  }

  void retireQuery(const std::string& queryId) {
    if (queryId.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto queryIt = queryAggregates_.find(queryId);
    if (queryIt != queryAggregates_.end()) {
      queryIt->second.retired = true;
    }
    cleanupQuery(queryId);
  }

  GpuMemorySnapshot snapshot() const {
    GpuMemorySnapshot result;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      result.resources.assign(resources_.begin(), resources_.end());
      result.owners.reserve(owners_.size());
      for (const auto& [_, owner] : owners_) {
        result.owners.push_back(owner.snapshot);
      }
      result.allocations.reserve(allocations_.size());
      for (const auto& [address, allocation] : allocations_) {
        result.allocations.push_back(GpuMemoryAllocationSnapshot{
            reinterpret_cast<uintptr_t>(address),
            allocation.bytes,
            allocation.ownerKey.kind,
            allocation.ownerKey.owner});
      }
    }

    std::sort(
        result.owners.begin(),
        result.owners.end(),
        [](const auto& left, const auto& right) {
          if (left.currentBytes != right.currentBytes) {
            return left.currentBytes > right.currentBytes;
          }
          if (left.owner.taskUuid != right.owner.taskUuid) {
            return left.owner.taskUuid < right.owner.taskUuid;
          }
          if (left.owner.taskId != right.owner.taskId) {
            return left.owner.taskId < right.owner.taskId;
          }
          if (left.owner.planNodeId != right.owner.planNodeId) {
            return left.owner.planNodeId < right.owner.planNodeId;
          }
          if (left.owner.operatorId != right.owner.operatorId) {
            return left.owner.operatorId < right.owner.operatorId;
          }
          if (left.owner.operatorType != right.owner.operatorType) {
            return left.owner.operatorType < right.owner.operatorType;
          }
          return left.kind < right.kind;
        });
    std::sort(
        result.allocations.begin(),
        result.allocations.end(),
        [](const auto& left, const auto& right) {
          if (left.bytes != right.bytes) {
            return left.bytes > right.bytes;
          }
          return left.address < right.address;
        });
    return result;
  }

 private:
  void cleanupQuery(const std::string& queryId) {
    if (queryId.empty()) {
      return;
    }

    const auto queryIt = queryAggregates_.find(queryId);
    if (queryIt != queryAggregates_.end() &&
        queryIt->second.combined.current != 0) {
      return;
    }
    if (queryIt != queryAggregates_.end()) {
      queryAggregates_.erase(queryIt);
    }

    for (auto it = planNodeAggregates_.begin();
         it != planNodeAggregates_.end();) {
      if (it->first.queryId == queryId && it->second.combined.current == 0) {
        it = planNodeAggregates_.erase(it);
      } else {
        ++it;
      }
    }

    for (auto it = operatorAggregates_.begin();
         it != operatorAggregates_.end();) {
      if (it->first.queryId == queryId && it->second.combined.current == 0) {
        it = operatorAggregates_.erase(it);
      } else {
        ++it;
      }
    }
  }

  mutable std::mutex mutex_;
  std::array<GpuMemoryResourceSnapshot, 2> resources_;
  std::unordered_map<OwnerKey, OwnerRecord, OwnerKeyHash> owners_;
  std::unordered_map<void*, AllocationRecord> allocations_;
  std::unordered_map<std::string, AggregateLiveBytes> queryAggregates_;
  std::unordered_map<PlanNodeKey, AggregateLiveBytes, PlanNodeKeyHash>
      planNodeAggregates_;
  std::unordered_map<OperatorKey, AggregateLiveBytes, OperatorKeyHash>
      operatorAggregates_;
};

GpuMemoryAllocationTracker::GpuMemoryAllocationTracker()
    : impl_(std::make_unique<Impl>()) {}

GpuMemoryAllocationTracker::~GpuMemoryAllocationTracker() = default;

GpuMemoryPeakUpdate GpuMemoryAllocationTracker::recordAllocation(
    void* address,
    std::size_t bytes,
    GpuMemoryResourceKind kind,
    const GpuMemoryOwner& owner) noexcept {
  return impl_->recordAllocation(address, bytes, kind, owner);
}

void GpuMemoryAllocationTracker::recordDeallocation(void* address) noexcept {
  impl_->recordDeallocation(address);
}

void GpuMemoryAllocationTracker::retireTask(std::string_view taskUuid) {
  impl_->retireTask(taskUuid);
}

void GpuMemoryAllocationTracker::retireQuery(const std::string& queryId) {
  impl_->retireQuery(queryId);
}

GpuMemorySnapshot GpuMemoryAllocationTracker::snapshot() const {
  return impl_->snapshot();
}

std::string_view gpuMemoryResourceKindString(GpuMemoryResourceKind kind) {
  switch (kind) {
    case GpuMemoryResourceKind::kMain:
      return "main";
    case GpuMemoryResourceKind::kOutput:
      return "output";
  }
  VELOX_UNREACHABLE();
}

namespace {

using StatisticsResource = rmm::mr::statistics_resource_adaptor;

struct GpuMemoryDiagnostics {
  std::shared_ptr<GpuMemoryAllocationTracker> tracker{
      std::make_shared<GpuMemoryAllocationTracker>()};
  std::optional<StatisticsResource> mainStatistics;
  std::optional<StatisticsResource> outputStatistics;
  mutable std::shared_mutex operationBarrier;
  std::mutex queryRegistrationMutex;
};

constexpr std::string_view kGpuMemoryQueryRegistration{
    "cudf.gpu-memory-query-registration"};

class GpuMemoryQueryRegistration {
 public:
  GpuMemoryQueryRegistration(
      std::shared_ptr<GpuMemoryDiagnostics> state,
      std::string queryId)
      : state_(std::move(state)), queryId_(std::move(queryId)) {}

  ~GpuMemoryQueryRegistration() {
    try {
      if (auto state = state_.lock()) {
        std::unique_lock<std::shared_mutex> operationLock(
            state->operationBarrier);
        state->tracker->retireQuery(queryId_);
      }
    } catch (...) {
      // Query destruction cannot fail because diagnostics could not retire.
    }
  }

  bool belongsTo(const std::shared_ptr<GpuMemoryDiagnostics>& state) const {
    return state_.lock() == state;
  }

 private:
  std::weak_ptr<GpuMemoryDiagnostics> state_;
  std::string queryId_;
};

exec::Operator* activeOperator();

GpuMemoryOwner captureOwner(exec::Operator* op);

void registerQuery(
    exec::Operator* op,
    const std::shared_ptr<GpuMemoryDiagnostics>& state);

void addOperatorMetric(
    exec::Operator* op,
    std::string_view name,
    int64_t value,
    RuntimeCounter::Unit unit = RuntimeCounter::Unit::kNone) noexcept;

class TrackingResourceImpl {
 public:
  TrackingResourceImpl(
      StatisticsResource statistics,
      std::shared_ptr<GpuMemoryDiagnostics> state,
      GpuMemoryResourceKind kind)
      : statistics_(std::move(statistics)),
        state_(std::move(state)),
        kind_(kind) {}

  void*
  allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
      return nullptr;
    }

    auto* op = activeOperator();
    GpuMemoryOwner owner;
    try {
      owner = captureOwner(op);
      registerQuery(op, state_);
    } catch (...) {
      // Query diagnostics cannot alter allocation behavior.
    }

    void* address;
    GpuMemoryPeakUpdate peakUpdate;
    {
      std::shared_lock<std::shared_mutex> operationLock(
          state_->operationBarrier);
      address = statistics_.allocate(stream, bytes, alignment);
      peakUpdate =
          state_->tracker->recordAllocation(address, bytes, kind_, owner);
    }
    addGpuMemoryPeakRuntimeStats(op, kind_, peakUpdate);
    addOperatorMetric(
        op,
        kGpuAllocationBytes,
        saturateCast(bytes),
        RuntimeCounter::Unit::kBytes);
    addOperatorMetric(op, kGpuAllocationCount, 1);
    return address;
  }

  void deallocate(
      cuda::stream_ref stream,
      void* address,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    if (address == nullptr || bytes == 0) {
      return;
    }

    std::shared_lock<std::shared_mutex> operationLock(state_->operationBarrier);
    state_->tracker->recordDeallocation(address);
    statistics_.deallocate(stream, address, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    return allocate(rmm::cuda_stream_default, bytes, alignment);
  }

  void deallocate_sync(
      void* address,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    deallocate(rmm::cuda_stream_default, address, bytes, alignment);
  }

  bool operator==(const TrackingResourceImpl& other) const noexcept {
    return this == &other;
  }

  bool operator!=(const TrackingResourceImpl& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const TrackingResourceImpl&,
      cuda::mr::device_accessible) noexcept {}

 private:
  StatisticsResource statistics_;
  std::shared_ptr<GpuMemoryDiagnostics> state_;
  GpuMemoryResourceKind kind_;
};

class TrackingResource final
    : public cuda::mr::shared_resource<TrackingResourceImpl> {
  using SharedBase = cuda::mr::shared_resource<TrackingResourceImpl>;

 public:
  TrackingResource(
      StatisticsResource statistics,
      std::shared_ptr<GpuMemoryDiagnostics> state,
      GpuMemoryResourceKind kind)
      : SharedBase(cuda::mr::make_shared_resource<TrackingResourceImpl>(
            std::move(statistics),
            std::move(state),
            kind)) {}

  friend void get_property(
      const TrackingResource&,
      cuda::mr::device_accessible) noexcept {}
};

static_assert(
    cuda::mr::resource_with<TrackingResource, cuda::mr::device_accessible>);

struct TrackedResource {
  cuda::mr::any_resource<cuda::mr::device_accessible> resource;
  StatisticsResource statistics;
};

std::mutex diagnosticsMutex;
std::shared_ptr<GpuMemoryDiagnostics> diagnostics;

exec::Operator* activeOperator() {
  return dynamic_cast<exec::Operator*>(getThreadLocalRunTimeStatWriter());
}

GpuMemoryOwner captureOwner(exec::Operator* op) {
  GpuMemoryOwner owner;
  if (op == nullptr) {
    return owner;
  }

  owner.taskUuid = op->operatorCtx()->task()->uuid();
  owner.taskId = op->taskId();
  owner.queryId = op->operatorCtx()->task()->queryCtx()->queryId();
  owner.planNodeId = op->planNodeId();
  owner.operatorId = op->operatorId();
  owner.operatorType = op->operatorType();
  return owner;
}

void registerQuery(
    exec::Operator* op,
    const std::shared_ptr<GpuMemoryDiagnostics>& state) {
  if (op == nullptr) {
    return;
  }

  auto queryCtx = op->operatorCtx()->task()->queryCtx();
  std::lock_guard<std::mutex> lock(state->queryRegistrationMutex);
  auto registration = queryCtx->registry<GpuMemoryQueryRegistration>(
      kGpuMemoryQueryRegistration);
  if (registration != nullptr && registration->belongsTo(state)) {
    return;
  }
  queryCtx->setRegistry(
      kGpuMemoryQueryRegistration,
      std::make_shared<GpuMemoryQueryRegistration>(state, queryCtx->queryId()),
      registration != nullptr);
}

std::string_view ownerField(std::string_view value) {
  return value.empty() ? "<none>" : value;
}

void addOperatorMetric(
    exec::Operator* op,
    std::string_view name,
    int64_t value,
    RuntimeCounter::Unit unit) noexcept {
  if (op == nullptr) {
    return;
  }

  try {
    op->addRuntimeStat(name, RuntimeCounter(value, unit));
  } catch (...) {
    // Runtime statistics are diagnostic and cannot affect GPU allocation.
  }
}

void addPeakMetric(
    BaseRuntimeStatWriter* writer,
    std::string_view name,
    const std::optional<uint64_t>& value) noexcept {
  if (writer == nullptr || !value.has_value()) {
    return;
  }
  try {
    writer->addRuntimeStat(
        name,
        RuntimeCounter(saturateCast(*value), RuntimeCounter::Unit::kBytes));
  } catch (...) {
    // Runtime statistics are diagnostic and cannot affect GPU allocation.
  }
}

uint64_t nonNegativeCounter(int64_t value) {
  return static_cast<uint64_t>(std::max<int64_t>(value, 0));
}

void replaceResourceStatistics(
    GpuMemorySnapshot& snapshot,
    GpuMemoryResourceKind kind,
    const StatisticsResource& statistics) {
  const auto bytes = statistics.get_bytes_counter();
  const auto allocations = statistics.get_allocations_counter();
  auto& resource = snapshot.resources[resourceIndex(kind)];
  resource = GpuMemoryResourceSnapshot{
      kind,
      nonNegativeCounter(bytes.value),
      nonNegativeCounter(bytes.peak),
      nonNegativeCounter(bytes.total),
      nonNegativeCounter(allocations.value),
      nonNegativeCounter(allocations.peak),
      nonNegativeCounter(allocations.total)};
}

GpuMemorySnapshot snapshotDiagnostics(
    const std::shared_ptr<GpuMemoryDiagnostics>& state) {
  if (state == nullptr) {
    return {};
  }

  std::unique_lock<std::shared_mutex> operationLock(state->operationBarrier);
  auto snapshot = state->tracker->snapshot();
  if (state->mainStatistics.has_value()) {
    replaceResourceStatistics(
        snapshot, GpuMemoryResourceKind::kMain, *state->mainStatistics);
  }
  if (state->outputStatistics.has_value()) {
    replaceResourceStatistics(
        snapshot, GpuMemoryResourceKind::kOutput, *state->outputStatistics);
  }
  return snapshot;
}

void logOom(
    std::size_t bytes,
    GpuMemoryResourceKind kind,
    const GpuMemoryOwner& owner,
    const std::shared_ptr<GpuMemoryDiagnostics>& state) noexcept {
  try {
    size_t freeBytes{0};
    size_t totalBytes{0};
    const auto cudaStatus = cudaMemGetInfo(&freeBytes, &totalBytes);
    const auto snapshot = snapshotDiagnostics(state);

    LOG(ERROR) << "GPU_MEMORY_OOM requested_bytes=" << bytes
               << " resource_kind=" << gpuMemoryResourceKindString(kind)
               << " task_uuid=" << ownerField(owner.taskUuid)
               << " task_id=" << ownerField(owner.taskId)
               << " query_id=" << ownerField(owner.queryId)
               << " plan_node_id=" << ownerField(owner.planNodeId)
               << " operator_id=" << owner.operatorId
               << " operator_type=" << ownerField(owner.operatorType)
               << " cuda_free_bytes=" << freeBytes
               << " cuda_total_bytes=" << totalBytes
               << " cuda_status=" << cudaGetErrorName(cudaStatus);

    for (const auto& resource : snapshot.resources) {
      LOG(ERROR) << "GPU_MEMORY_OOM resource_kind="
                 << gpuMemoryResourceKindString(resource.kind)
                 << " statistics_current_bytes=" << resource.currentBytes
                 << " statistics_peak_bytes=" << resource.peakBytes
                 << " statistics_total_bytes=" << resource.totalBytes
                 << " statistics_current_allocations="
                 << resource.currentAllocations
                 << " statistics_peak_allocations=" << resource.peakAllocations
                 << " statistics_total_allocations="
                 << resource.totalAllocations
                 << " cuda_free_bytes=" << freeBytes
                 << " cuda_total_bytes=" << totalBytes;
    }

    size_t ownerRank{0};
    for (const auto& ownerSnapshot : snapshot.owners) {
      if (ownerSnapshot.currentAllocations == 0 || ownerRank == 10) {
        continue;
      }
      ++ownerRank;
      LOG(ERROR) << "GPU_MEMORY_OWNER rank=" << ownerRank
                 << " live_bytes=" << ownerSnapshot.currentBytes
                 << " live_allocations=" << ownerSnapshot.currentAllocations
                 << " resource_kind="
                 << gpuMemoryResourceKindString(ownerSnapshot.kind)
                 << " task_uuid=" << ownerField(ownerSnapshot.owner.taskUuid)
                 << " task_id=" << ownerField(ownerSnapshot.owner.taskId)
                 << " query_id=" << ownerField(ownerSnapshot.owner.queryId)
                 << " plan_node_id="
                 << ownerField(ownerSnapshot.owner.planNodeId)
                 << " operator_id=" << ownerSnapshot.owner.operatorId
                 << " operator_type="
                 << ownerField(ownerSnapshot.owner.operatorType);
    }

    const auto allocationCount =
        std::min<size_t>(snapshot.allocations.size(), 10);
    for (size_t i = 0; i < allocationCount; ++i) {
      const auto& allocation = snapshot.allocations[i];
      LOG(ERROR) << "GPU_MEMORY_ALLOCATION rank=" << (i + 1) << " address="
                 << reinterpret_cast<const void*>(allocation.address)
                 << " bytes=" << allocation.bytes << " resource_kind="
                 << gpuMemoryResourceKindString(allocation.kind)
                 << " task_uuid=" << ownerField(allocation.owner.taskUuid)
                 << " task_id=" << ownerField(allocation.owner.taskId)
                 << " query_id=" << ownerField(allocation.owner.queryId)
                 << " plan_node_id=" << ownerField(allocation.owner.planNodeId)
                 << " operator_id=" << allocation.owner.operatorId
                 << " operator_type="
                 << ownerField(allocation.owner.operatorType);
    }
  } catch (...) {
    LOG(ERROR) << "GPU_MEMORY_OOM requested_bytes=" << bytes
               << " resource_kind=" << gpuMemoryResourceKindString(kind)
               << " task_uuid=" << ownerField(owner.taskUuid)
               << " task_id=" << ownerField(owner.taskId)
               << " query_id=" << ownerField(owner.queryId)
               << " plan_node_id=" << ownerField(owner.planNodeId)
               << " operator_id=" << owner.operatorId
               << " operator_type=" << ownerField(owner.operatorType)
               << " diagnostics_unavailable=true";
  }
}

TrackedResource makeTrackedResource(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    GpuMemoryResourceKind kind,
    const std::shared_ptr<GpuMemoryDiagnostics>& state) {
  StatisticsResource statistics(std::move(upstream));
  TrackingResource trackingResource(statistics, state, kind);

  rmm::mr::failure_callback_resource_adaptor failureResource(
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          std::move(trackingResource)},
      [state, kind](std::size_t bytes, void*) {
        auto* op = activeOperator();
        GpuMemoryOwner owner;
        try {
          owner = captureOwner(op);
        } catch (...) {
          // Continue with the owner fields captured before the copy failed.
        }
        addOperatorMetric(
            op,
            kGpuFailedAllocationBytes,
            saturateCast(bytes),
            RuntimeCounter::Unit::kBytes);
        addOperatorMetric(op, kGpuFailedAllocationCount, 1);
        logOom(bytes, kind, owner, state);
        return false;
      },
      nullptr);

  return {
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          std::move(failureResource)},
      std::move(statistics)};
}

} // namespace

void addGpuMemoryPeakRuntimeStats(
    BaseRuntimeStatWriter* writer,
    GpuMemoryResourceKind kind,
    const GpuMemoryPeakUpdate& update) noexcept {
  addPeakMetric(writer, kGpuQueryPeakLiveBytes, update.queryPeakBytes);
  addPeakMetric(writer, kGpuPlanNodePeakLiveBytes, update.planNodePeakBytes);
  addPeakMetric(writer, kGpuOperatorPeakLiveBytes, update.operatorPeakBytes);

  if (kind == GpuMemoryResourceKind::kMain) {
    addPeakMetric(
        writer, kGpuQueryMainPeakLiveBytes, update.queryResourcePeakBytes);
    addPeakMetric(
        writer,
        kGpuPlanNodeMainPeakLiveBytes,
        update.planNodeResourcePeakBytes);
    addPeakMetric(
        writer,
        kGpuOperatorMainPeakLiveBytes,
        update.operatorResourcePeakBytes);
  } else {
    addPeakMetric(
        writer, kGpuQueryOutputPeakLiveBytes, update.queryResourcePeakBytes);
    addPeakMetric(
        writer,
        kGpuPlanNodeOutputPeakLiveBytes,
        update.planNodeResourcePeakBytes);
    addPeakMetric(
        writer,
        kGpuOperatorOutputPeakLiveBytes,
        update.operatorResourcePeakBytes);
  }
}

GpuMemoryResourcePair createGpuMemoryTrackingResources(
    cuda::mr::any_resource<cuda::mr::device_accessible> mainUpstream,
    cuda::mr::any_resource<cuda::mr::device_accessible> outputUpstream) {
  auto state = std::make_shared<GpuMemoryDiagnostics>();
  auto main = makeTrackedResource(
      std::move(mainUpstream), GpuMemoryResourceKind::kMain, state);
  auto output = makeTrackedResource(
      std::move(outputUpstream), GpuMemoryResourceKind::kOutput, state);
  state->mainStatistics = main.statistics;
  state->outputStatistics = output.statistics;
  {
    std::lock_guard<std::mutex> lock(diagnosticsMutex);
    diagnostics = std::move(state);
  }
  return {std::move(main.resource), std::move(output.resource)};
}

void resetGpuMemoryTracking() {
  std::lock_guard<std::mutex> lock(diagnosticsMutex);
  diagnostics.reset();
}

GpuMemorySnapshot getGpuMemorySnapshot() {
  std::shared_ptr<GpuMemoryDiagnostics> state;
  {
    std::lock_guard<std::mutex> lock(diagnosticsMutex);
    state = diagnostics;
  }
  return snapshotDiagnostics(state);
}

void retireGpuMemoryTask(std::string_view taskUuid) {
  std::shared_ptr<GpuMemoryDiagnostics> state;
  {
    std::lock_guard<std::mutex> lock(diagnosticsMutex);
    state = diagnostics;
  }
  if (state != nullptr) {
    std::unique_lock<std::shared_mutex> operationLock(state->operationBarrier);
    state->tracker->retireTask(taskUuid);
  }
}

void logGpuMemorySnapshot(std::string_view context, std::size_t maxEntries) {
  const auto snapshot = getGpuMemorySnapshot();
  const auto liveOwnerCount = std::count_if(
      snapshot.owners.begin(), snapshot.owners.end(), [](const auto& owner) {
        return owner.currentAllocations > 0;
      });
  LOG(INFO) << "GPU_MEMORY_SNAPSHOT context=" << context
            << " resource_count=" << snapshot.resources.size()
            << " live_owner_count=" << liveOwnerCount
            << " live_allocation_count=" << snapshot.allocations.size();
  for (const auto& resource : snapshot.resources) {
    LOG(INFO) << "GPU_MEMORY_RESOURCE context=" << context
              << " resource_kind=" << gpuMemoryResourceKindString(resource.kind)
              << " current_bytes=" << resource.currentBytes
              << " peak_bytes=" << resource.peakBytes
              << " total_bytes=" << resource.totalBytes
              << " current_allocations=" << resource.currentAllocations
              << " peak_allocations=" << resource.peakAllocations
              << " total_allocations=" << resource.totalAllocations;
  }

  size_t ownerRank{0};
  for (const auto& owner : snapshot.owners) {
    if (owner.currentAllocations == 0 || ownerRank == maxEntries) {
      continue;
    }
    ++ownerRank;
    LOG(INFO) << "GPU_MEMORY_OWNER context=" << context << " rank=" << ownerRank
              << " live_bytes=" << owner.currentBytes
              << " live_allocations=" << owner.currentAllocations
              << " resource_kind=" << gpuMemoryResourceKindString(owner.kind)
              << " task_uuid=" << ownerField(owner.owner.taskUuid)
              << " task_id=" << ownerField(owner.owner.taskId)
              << " query_id=" << ownerField(owner.owner.queryId)
              << " plan_node_id=" << ownerField(owner.owner.planNodeId)
              << " operator_id=" << owner.owner.operatorId
              << " operator_type=" << ownerField(owner.owner.operatorType);
  }

  const auto allocationCount =
      std::min(maxEntries, snapshot.allocations.size());
  for (size_t i = 0; i < allocationCount; ++i) {
    const auto& allocation = snapshot.allocations[i];
    LOG(INFO) << "GPU_MEMORY_ALLOCATION context=" << context
              << " rank=" << (i + 1) << " address="
              << reinterpret_cast<const void*>(allocation.address)
              << " bytes=" << allocation.bytes << " resource_kind="
              << gpuMemoryResourceKindString(allocation.kind)
              << " task_uuid=" << ownerField(allocation.owner.taskUuid)
              << " task_id=" << ownerField(allocation.owner.taskId)
              << " query_id=" << ownerField(allocation.owner.queryId)
              << " plan_node_id=" << ownerField(allocation.owner.planNodeId)
              << " operator_id=" << allocation.owner.operatorId
              << " operator_type=" << ownerField(allocation.owner.operatorType);
  }
}

cuda::mr::any_resource<cuda::mr::device_accessible> createMemoryResource(
    std::string_view mode,
    int percent) {
  if (mode == "cuda") {
    return rmm::mr::cuda_memory_resource{};
  } else if (mode == "pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "async") {
    return rmm::mr::cuda_async_memory_resource{};
  } else if (mode == "arena") {
    return rmm::mr::arena_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed") {
    return rmm::mr::managed_memory_resource{};
  } else if (mode == "managed_pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::managed_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed_async") {
    return rmm::mr::cuda_async_managed_memory_resource{};
  } else if (mode == "prefetch_managed") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::managed_memory_resource{});
  } else if (mode == "prefetch_managed_pool") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(rmm::mr::pool_memory_resource(
        rmm::mr::managed_memory_resource{},
        rmm::percent_of_free_device_memory(percent)));
  } else if (mode == "prefetch_managed_async") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::cuda_async_managed_memory_resource{});
  }
  VELOX_FAIL(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, prefetch_managed, " +
      "managed_pool, prefetch_managed_pool, managed_async, prefetch_managed_async");
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> output_mr_;

rmm::device_async_resource_ref get_output_mr() {
  return output_mr_.value();
}

} // namespace facebook::velox::cudf_velox

// This must NOT be in a file that includes CudfNoDefaults.h, because
// CudfNoDefaults.h redeclares cudf::get_default_stream() with
// __attribute__((error)). The overload below calls the real function.
namespace cudf {

rmm::cuda_stream_view const get_default_stream(allow_default_stream_t) {
  return cudf::get_default_stream();
}

} // namespace cudf
