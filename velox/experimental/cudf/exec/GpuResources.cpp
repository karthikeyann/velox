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
#include "velox/core/PlanNode.h"
#include "velox/exec/Driver.h"
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

#include <cuda_runtime_api.h>

#include <common/base/Exceptions.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace facebook::velox::cudf_velox {

namespace {

void hashCombine(std::size_t& seed, std::size_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

struct GpuMemoryOwnerHash {
  std::size_t operator()(const GpuMemoryOwner& owner) const {
    std::size_t result = std::hash<std::string>{}(owner.taskUuid);
    hashCombine(result, std::hash<std::string>{}(owner.taskId));
    hashCombine(result, std::hash<std::string>{}(owner.queryId));
    hashCombine(result, std::hash<std::string>{}(owner.planNodeId));
    hashCombine(result, std::hash<int32_t>{}(owner.pipelineId));
    hashCombine(result, std::hash<int32_t>{}(owner.driverId));
    hashCombine(result, std::hash<int32_t>{}(owner.operatorId));
    hashCombine(result, std::hash<std::string>{}(owner.operatorType));
    return result;
  }
};

struct PlanNodeKey {
  std::string queryId;
  std::string taskUuid;
  std::string planNodeId;

  bool operator==(const PlanNodeKey&) const = default;
};

struct PlanNodeKeyHash {
  std::size_t operator()(const PlanNodeKey& key) const {
    std::size_t result = std::hash<std::string>{}(key.queryId);
    hashCombine(result, std::hash<std::string>{}(key.taskUuid));
    hashCombine(result, std::hash<std::string>{}(key.planNodeId));
    return result;
  }
};

struct LiveBytes {
  uint64_t current{0};
  uint64_t peak{0};
  uint64_t total{0};
};

struct OwnerRecord {
  GpuMemoryOwnerSnapshot snapshot;
};

struct AllocationRecord {
  uint64_t bytes{0};
  GpuMemoryOwnerHandle handle;
};

GpuMemoryOwner unattributedOwner() {
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

bool isEmptyOwner(const GpuMemoryOwner& owner) {
  return owner.taskUuid.empty() && owner.taskId.empty() &&
      owner.queryId.empty() && owner.planNodeId.empty() &&
      owner.pipelineId < 0 && owner.driverId < 0 && owner.operatorId < 0 &&
      owner.operatorType.empty();
}

} // namespace

class GpuMemoryAllocationTracker::Impl {
 public:
  Impl() {
    const GpuMemoryOwnerHandle handle{0, 0};
    owners_.emplace(
        0,
        OwnerRecord{GpuMemoryOwnerSnapshot{
            handle, unattributedOwner(), 0, 0, 0, 0, 0}});
    plans_.emplace(0, LiveBytes{});
  }

  GpuMemoryOwnerHandle registerOwner(const GpuMemoryOwner& owner) {
    if (isEmptyOwner(owner)) {
      return {};
    }

    GpuMemoryOwnerHandle handle;
    bool inserted{false};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto existing = ownerIds_.find(owner);
      if (existing != ownerIds_.end()) {
        return owners_.at(existing->second).snapshot.handle;
      }

      uint64_t planNodeId{0};
      if (!owner.queryId.empty() || !owner.taskUuid.empty() ||
          !owner.planNodeId.empty()) {
        const PlanNodeKey planNodeKey{
            owner.queryId, owner.taskUuid, owner.planNodeId};
        const auto planNode = planNodeIds_.find(planNodeKey);
        if (planNode != planNodeIds_.end()) {
          planNodeId = planNode->second;
        } else {
          planNodeId = nextPlanNodeId_++;
          planNodeIds_.emplace(planNodeKey, planNodeId);
          plans_.emplace(planNodeId, LiveBytes{});
        }
      }

      handle = GpuMemoryOwnerHandle{nextOwnerId_++, planNodeId};
      owners_.emplace(
          handle.ownerId,
          OwnerRecord{GpuMemoryOwnerSnapshot{handle, owner, 0, 0, 0, 0, 0}});
      ownerIds_.emplace(owner, handle.ownerId);
      inserted = true;
    }

    if (inserted) {
      gpu_memory_detail::registerGpuMemoryTraceOwner(
          handle.ownerId, handle.planNodeId, owner);
    }
    return handle;
  }

  std::optional<GpuMemoryOwnerHandle> findOwner(
      const GpuMemoryOwner& owner) const {
    if (isEmptyOwner(owner)) {
      return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = ownerIds_.find(owner);
    if (existing == ownerIds_.end()) {
      return std::nullopt;
    }
    return owners_.at(existing->second).snapshot.handle;
  }

  std::optional<GpuMemoryTraceUpdate> recordAllocation(
      void* address,
      std::size_t bytes,
      GpuMemoryOwnerHandle handle) noexcept {
    if (address == nullptr) {
      return std::nullopt;
    }

    try {
      GpuMemoryTraceUpdate update;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (allocations_.contains(address)) {
          noteDataLossLocked("duplicate live allocation address");
          return std::nullopt;
        }

        auto owner = owners_.find(handle.ownerId);
        if (owner == owners_.end()) {
          owner = owners_.find(0);
        }
        handle = owner->second.snapshot.handle;

        allocations_.emplace(
            address, AllocationRecord{static_cast<uint64_t>(bytes), handle});

        const auto byteCount = static_cast<uint64_t>(bytes);
        addBytes(global_, byteCount);
        ++currentAllocations_;
        ++totalAllocations_;
        peakAllocations_ = std::max(peakAllocations_, currentAllocations_);

        auto& ownerSnapshot = owner->second.snapshot;
        ownerSnapshot.currentBytes += byteCount;
        ownerSnapshot.peakBytes =
            std::max(ownerSnapshot.peakBytes, ownerSnapshot.currentBytes);
        ownerSnapshot.totalBytes += byteCount;
        ++ownerSnapshot.currentAllocations;
        ++ownerSnapshot.totalAllocations;

        auto& plan = plans_.at(handle.planNodeId);
        addBytes(plan, byteCount);

        update = makeUpdateLocked(
            handle,
            plan.current,
            ownerSnapshot.currentBytes,
            static_cast<int64_t>(byteCount));
      }
      if (update.ownerId == 0) {
        gpu_memory_detail::registerGpuMemoryTraceOwner(
            0, 0, unattributedOwner());
      }
      gpu_memory_detail::emitGpuMemoryTraceUpdate(update);
      return update;
    } catch (...) {
      noteDataLoss("allocation accounting exception");
      return std::nullopt;
    }
  }

  std::optional<GpuMemoryTraceUpdate> recordDeallocation(
      void* address) noexcept {
    if (address == nullptr) {
      return std::nullopt;
    }

    try {
      GpuMemoryTraceUpdate update;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto allocation = allocations_.find(address);
        if (allocation == allocations_.end()) {
          noteDataLossLocked("unknown deallocation address");
          return std::nullopt;
        }

        const auto bytes = allocation->second.bytes;
        const auto handle = allocation->second.handle;
        auto& ownerSnapshot = owners_.at(handle.ownerId).snapshot;
        auto& plan = plans_.at(handle.planNodeId);

        global_.current -= bytes;
        --currentAllocations_;
        ownerSnapshot.currentBytes -= bytes;
        --ownerSnapshot.currentAllocations;
        plan.current -= bytes;
        allocations_.erase(allocation);

        update = makeUpdateLocked(
            handle,
            plan.current,
            ownerSnapshot.currentBytes,
            -static_cast<int64_t>(bytes));
      }
      gpu_memory_detail::emitGpuMemoryTraceUpdate(update);
      return update;
    } catch (...) {
      noteDataLoss("deallocation accounting exception");
      return std::nullopt;
    }
  }

  GpuMemoryTraceUpdate currentState(
      GpuMemoryOwnerHandle handle) const noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      auto owner = owners_.find(handle.ownerId);
      if (owner == owners_.end()) {
        owner = owners_.find(0);
      }
      handle = owner->second.snapshot.handle;
      const auto& plan = plans_.at(handle.planNodeId);
      return GpuMemoryTraceUpdate{
          gpu_memory_detail::gpuMemoryTraceNowNs(),
          sequence_,
          handle.ownerId,
          handle.planNodeId,
          global_.current,
          global_.peak,
          plan.current,
          owner->second.snapshot.currentBytes,
          0};
    } catch (...) {
      return {};
    }
  }

  GpuMemorySnapshot snapshot() const {
    GpuMemorySnapshot result;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      result.currentBytes = global_.current;
      result.peakBytes = global_.peak;
      result.totalBytes = global_.total;
      result.currentAllocations = currentAllocations_;
      result.peakAllocations = peakAllocations_;
      result.totalAllocations = totalAllocations_;
      result.sequence = sequence_;
      result.dataLossEvents = dataLossEvents_;

      result.owners.reserve(owners_.size());
      for (const auto& [_, owner] : owners_) {
        result.owners.push_back(owner.snapshot);
      }

      result.allocations.reserve(allocations_.size());
      for (const auto& [address, allocation] : allocations_) {
        result.allocations.push_back(
            GpuMemoryAllocationSnapshot{
                reinterpret_cast<uintptr_t>(address),
                allocation.bytes,
                allocation.handle});
      }
    }

    std::sort(
        result.owners.begin(),
        result.owners.end(),
        [](const auto& left, const auto& right) {
          if (left.currentBytes != right.currentBytes) {
            return left.currentBytes > right.currentBytes;
          }
          return left.handle.ownerId < right.handle.ownerId;
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
  static void addBytes(LiveBytes& counter, uint64_t bytes) {
    counter.current += bytes;
    counter.peak = std::max(counter.peak, counter.current);
    counter.total += bytes;
  }

  GpuMemoryTraceUpdate makeUpdateLocked(
      GpuMemoryOwnerHandle handle,
      uint64_t planNodeCurrentBytes,
      uint64_t ownerCurrentBytes,
      int64_t deltaBytes) {
    const auto now = gpu_memory_detail::gpuMemoryTraceNowNs();
    lastTimestampNs_ = std::max(
        now,
        lastTimestampNs_ == std::numeric_limits<uint64_t>::max()
            ? lastTimestampNs_
            : lastTimestampNs_ + 1);
    return GpuMemoryTraceUpdate{
        lastTimestampNs_,
        ++sequence_,
        handle.ownerId,
        handle.planNodeId,
        global_.current,
        global_.peak,
        planNodeCurrentBytes,
        ownerCurrentBytes,
        deltaBytes};
  }

  void noteDataLossLocked(std::string_view reason) noexcept {
    ++dataLossEvents_;
    gpu_memory_detail::emitGpuMemoryTraceDataLoss(reason, sequence_);
  }

  void noteDataLoss(std::string_view reason) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      noteDataLossLocked(reason);
    } catch (...) {
      // Diagnostics must never change allocation or deallocation behavior.
    }
  }

  mutable std::mutex mutex_;
  LiveBytes global_;
  uint64_t currentAllocations_{0};
  uint64_t peakAllocations_{0};
  uint64_t totalAllocations_{0};
  uint64_t sequence_{0};
  uint64_t dataLossEvents_{0};
  uint64_t lastTimestampNs_{0};
  uint64_t nextOwnerId_{1};
  uint64_t nextPlanNodeId_{1};
  std::unordered_map<GpuMemoryOwner, uint64_t, GpuMemoryOwnerHash> ownerIds_;
  std::unordered_map<uint64_t, OwnerRecord> owners_;
  std::unordered_map<PlanNodeKey, uint64_t, PlanNodeKeyHash> planNodeIds_;
  std::unordered_map<uint64_t, LiveBytes> plans_;
  std::unordered_map<void*, AllocationRecord> allocations_;
};

GpuMemoryAllocationTracker::GpuMemoryAllocationTracker()
    : impl_(std::make_unique<Impl>()) {}

GpuMemoryAllocationTracker::~GpuMemoryAllocationTracker() = default;

GpuMemoryOwnerHandle GpuMemoryAllocationTracker::registerOwner(
    const GpuMemoryOwner& owner) {
  return impl_->registerOwner(owner);
}

GpuMemoryOwnerHandle GpuMemoryAllocationTracker::registerOperator(
    exec::Operator* op) {
  if (op == nullptr) {
    return {};
  }

  const auto* driverCtx = op->operatorCtx()->driverCtx();
  GpuMemoryOwner owner{
      op->operatorCtx()->task()->uuid(),
      op->taskId(),
      op->operatorCtx()->task()->queryCtx()->queryId(),
      op->planNodeId(),
      "",
      driverCtx->pipelineId,
      driverCtx->driverId,
      op->operatorId(),
      op->operatorType()};
  if (const auto existing = impl_->findOwner(owner)) {
    return *existing;
  }

  const auto& task = *op->operatorCtx()->task();
  const auto* planRoot = task.planFragment().planNode.get();
  auto planNodeId = std::string_view{op->planNodeId()};
  const auto findPlanNode = [&](std::string_view id) {
    return planRoot == nullptr
        ? nullptr
        : core::PlanNode::findNodeById(planRoot, core::PlanNodeId{id});
  };
  const auto* planNode = findPlanNode(planNodeId);
  if (planNode == nullptr) {
    // Resolve synthetic conversion operators through their source PlanNode.
    for (const auto suffix :
         {std::string_view{"-from-velox"}, std::string_view{"-to-velox"}}) {
      if (planNodeId.ends_with(suffix)) {
        planNodeId.remove_suffix(suffix.size());
        planNode = findPlanNode(planNodeId);
        break;
      }
    }
  }
  if (planNode != nullptr) {
    owner.planNodeType = planNode->name();
    if (!owner.planNodeType.ends_with("Node")) {
      owner.planNodeType += "Node";
    }
  }
  return impl_->registerOwner(owner);
}

std::optional<GpuMemoryTraceUpdate>
GpuMemoryAllocationTracker::recordAllocation(
    void* address,
    std::size_t bytes,
    GpuMemoryOwnerHandle handle) noexcept {
  return impl_->recordAllocation(address, bytes, handle);
}

std::optional<GpuMemoryTraceUpdate>
GpuMemoryAllocationTracker::recordDeallocation(void* address) noexcept {
  return impl_->recordDeallocation(address);
}

GpuMemoryTraceUpdate GpuMemoryAllocationTracker::currentState(
    GpuMemoryOwnerHandle handle) const noexcept {
  return impl_->currentState(handle);
}

GpuMemorySnapshot GpuMemoryAllocationTracker::snapshot() const {
  return impl_->snapshot();
}

namespace {

std::mutex diagnosticsMutex;
std::shared_ptr<GpuMemoryAllocationTracker> diagnostics;
thread_local gpu_memory_detail::GpuMemoryActiveOwner activeOwner;

exec::Operator* runtimeOperator() {
  return dynamic_cast<exec::Operator*>(getThreadLocalRunTimeStatWriter());
}

GpuMemoryOwnerHandle resolveOwner(
    const std::shared_ptr<GpuMemoryAllocationTracker>& tracker) noexcept {
  if (activeOwner.tracker == tracker.get()) {
    return GpuMemoryOwnerHandle{activeOwner.ownerId, 0};
  }

  try {
    auto* op = runtimeOperator();
    if (op != nullptr) {
      return tracker->registerOperator(op);
    }
  } catch (...) {
    // Fall through to the explicit unattributed owner.
  }
  return {};
}

class TrackingResourceImpl {
 public:
  TrackingResourceImpl(
      cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
      std::shared_ptr<GpuMemoryAllocationTracker> tracker)
      : upstream_(std::move(upstream)), tracker_(std::move(tracker)) {}

  void*
  allocate(cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
      return nullptr;
    }

    const auto owner = resolveOwner(tracker_);
    auto* address = upstream_.allocate(stream, bytes, alignment);
    tracker_->recordAllocation(address, bytes, owner);
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

    // Remove the address before calling upstream so an immediately reused
    // address cannot be erased by a delayed cross-thread deallocation.
    tracker_->recordDeallocation(address);
    upstream_.deallocate(stream, address, bytes, alignment);
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
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::shared_ptr<GpuMemoryAllocationTracker> tracker_;
};

class TrackingResource final
    : public cuda::mr::shared_resource<TrackingResourceImpl> {
  using SharedBase = cuda::mr::shared_resource<TrackingResourceImpl>;

 public:
  TrackingResource(
      cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
      std::shared_ptr<GpuMemoryAllocationTracker> tracker)
      : SharedBase(
            cuda::mr::make_shared_resource<TrackingResourceImpl>(
                std::move(upstream),
                std::move(tracker))) {}

  friend void get_property(
      const TrackingResource&,
      cuda::mr::device_accessible) noexcept {}
};

static_assert(
    cuda::mr::resource_with<TrackingResource, cuda::mr::device_accessible>);

void logAndTraceAllocationFailure(
    std::size_t bytes,
    const std::shared_ptr<GpuMemoryAllocationTracker>& tracker) noexcept {
  const auto owner = resolveOwner(tracker);
  const auto state = tracker->currentState(owner);

  std::size_t freeBytes{0};
  std::size_t totalBytes{0};
  const auto cudaStatus = cudaMemGetInfo(&freeBytes, &totalBytes);
  const std::string_view cudaStatusName{cudaGetErrorName(cudaStatus)};

  if (state.ownerId == 0) {
    gpu_memory_detail::registerGpuMemoryTraceOwner(0, 0, unattributedOwner());
  }
  gpu_memory_detail::emitGpuMemoryTraceOom(
      state.ownerId,
      bytes,
      state.globalCurrentBytes,
      state.globalPeakBytes,
      state.planNodeCurrentBytes,
      state.ownerCurrentBytes,
      freeBytes,
      totalBytes,
      cudaStatusName);
  LOG(ERROR) << "GPU_MEMORY_OOM requested_bytes=" << bytes
             << " logical_live_bytes=" << state.globalCurrentBytes
             << " owner_id=" << state.ownerId
             << " plan_node_track_id=" << state.planNodeId
             << " owner_live_bytes=" << state.ownerCurrentBytes
             << " plan_node_live_bytes=" << state.planNodeCurrentBytes
             << " cuda_free_bytes=" << freeBytes
             << " cuda_total_bytes=" << totalBytes
             << " cuda_status=" << cudaStatusName;
}

cuda::mr::any_resource<cuda::mr::device_accessible> makeTrackedResource(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    const std::shared_ptr<GpuMemoryAllocationTracker>& tracker) {
  TrackingResource trackingResource(std::move(upstream), tracker);
  rmm::mr::failure_callback_resource_adaptor failureResource(
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          std::move(trackingResource)},
      [tracker](std::size_t bytes, void*) {
        logAndTraceAllocationFailure(bytes, tracker);
        return false;
      },
      nullptr);
  return cuda::mr::any_resource<cuda::mr::device_accessible>{
      std::move(failureResource)};
}

} // namespace

GpuMemoryResourcePair createGpuMemoryTrackingResources(
    cuda::mr::any_resource<cuda::mr::device_accessible> mainUpstream,
    cuda::mr::any_resource<cuda::mr::device_accessible> outputUpstream) {
  auto tracker = std::make_shared<GpuMemoryAllocationTracker>();
  auto main = makeTrackedResource(std::move(mainUpstream), tracker);
  auto output = makeTrackedResource(std::move(outputUpstream), tracker);
  {
    std::lock_guard<std::mutex> lock(diagnosticsMutex);
    diagnostics = std::move(tracker);
  }
  return {std::move(main), std::move(output)};
}

void resetGpuMemoryTracking() {
  std::lock_guard<std::mutex> lock(diagnosticsMutex);
  diagnostics.reset();
}

GpuMemorySnapshot getGpuMemorySnapshot() {
  std::shared_ptr<GpuMemoryAllocationTracker> tracker;
  {
    std::lock_guard<std::mutex> lock(diagnosticsMutex);
    tracker = diagnostics;
  }
  return tracker == nullptr ? GpuMemorySnapshot{} : tracker->snapshot();
}

namespace gpu_memory_detail {

GpuMemoryActiveOwner activateGpuMemoryOperator(exec::Operator* op) noexcept {
  const auto previous = activeOwner;
  std::shared_ptr<GpuMemoryAllocationTracker> tracker;
  {
    std::lock_guard<std::mutex> lock(diagnosticsMutex);
    tracker = diagnostics;
  }

  if (tracker == nullptr || op == nullptr) {
    activeOwner = {};
    return previous;
  }

  uint64_t ownerId{0};
  try {
    ownerId = tracker->registerOperator(op).ownerId;
  } catch (...) {
    // Attribute to the explicit fallback owner if registration fails.
  }
  if (ownerId == 0) {
    gpu_memory_detail::registerGpuMemoryTraceOwner(0, 0, unattributedOwner());
  }
  activeOwner = GpuMemoryActiveOwner{tracker.get(), ownerId};
  return previous;
}

GpuMemoryActiveOwner activeGpuMemoryOwner() noexcept {
  return activeOwner;
}

void restoreGpuMemoryOwner(GpuMemoryActiveOwner owner) noexcept {
  activeOwner = owner;
}

} // namespace gpu_memory_detail

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
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::pool_memory_resource(
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
}

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
