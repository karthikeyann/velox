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

#include <rmm/error.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#include <perfetto.h>
#pragma GCC diagnostic pop

#include <folly/ScopeGuard.h>
#include <folly/json.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::test {
namespace {

struct RecordingResourceState {
  std::mutex mutex;
  std::vector<std::size_t> allocationAlignments;
  std::vector<std::size_t> deallocationAlignments;
  bool throwOnAllocation{false};
};

class RecordingResource {
 public:
  explicit RecordingResource(std::shared_ptr<RecordingResourceState> state)
      : state_(std::move(state)) {}

  void* allocate(
      cuda::stream_ref /*stream*/,
      std::size_t bytes,
      std::size_t alignment) {
    return allocateImpl(bytes, alignment);
  }

  void deallocate(
      cuda::stream_ref /*stream*/,
      void* address,
      std::size_t /*bytes*/,
      std::size_t alignment) noexcept {
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->deallocationAlignments.push_back(alignment);
    }
    std::free(address);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    return allocateImpl(bytes, alignment);
  }

  void deallocate_sync(
      void* address,
      std::size_t /*bytes*/,
      std::size_t alignment) noexcept {
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->deallocationAlignments.push_back(alignment);
    }
    std::free(address);
  }

  bool operator==(const RecordingResource& other) const noexcept {
    return state_ == other.state_;
  }

  bool operator!=(const RecordingResource& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const RecordingResource&,
      cuda::mr::device_accessible) noexcept {}

 private:
  void* allocateImpl(std::size_t bytes, std::size_t alignment) {
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->allocationAlignments.push_back(alignment);
      if (state_->throwOnAllocation) {
        throw rmm::out_of_memory("deterministic test allocation failure");
      }
    }

    const auto allocationSize =
        std::max(alignment, ((bytes + alignment - 1) / alignment) * alignment);
    auto* address = std::aligned_alloc(alignment, allocationSize);
    if (address == nullptr) {
      throw rmm::out_of_memory("host allocation failed in test resource");
    }
    return address;
  }

  std::shared_ptr<RecordingResourceState> state_;
};

static_assert(
    cuda::mr::resource_with<RecordingResource, cuda::mr::device_accessible>);

struct AddressReusingResourceState {
  std::mutex mutex;
  std::condition_variable condition;
  bool allocated{false};
  bool blockNextDeallocation{true};
  bool addressFreed{false};
  bool releaseDeallocation{false};
  alignas(256) std::array<std::byte, 256> storage;
};

class AddressReusingResource {
 public:
  explicit AddressReusingResource(
      std::shared_ptr<AddressReusingResourceState> state)
      : state_(std::move(state)) {}

  void* allocate(
      cuda::stream_ref /*stream*/,
      std::size_t /*bytes*/,
      std::size_t /*alignment*/) {
    return allocateImpl();
  }

  void deallocate(
      cuda::stream_ref /*stream*/,
      void* address,
      std::size_t /*bytes*/,
      std::size_t /*alignment*/) noexcept {
    deallocateImpl(address);
  }

  void* allocate_sync(std::size_t /*bytes*/, std::size_t /*alignment*/) {
    return allocateImpl();
  }

  void deallocate_sync(
      void* address,
      std::size_t /*bytes*/,
      std::size_t /*alignment*/) noexcept {
    deallocateImpl(address);
  }

  bool operator==(const AddressReusingResource& other) const noexcept {
    return state_ == other.state_;
  }

  bool operator!=(const AddressReusingResource& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const AddressReusingResource&,
      cuda::mr::device_accessible) noexcept {}

 private:
  void* allocateImpl() {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->condition.wait(lock, [&] { return !state_->allocated; });
    state_->allocated = true;
    return state_->storage.data();
  }

  void deallocateImpl(void* address) noexcept {
    std::unique_lock<std::mutex> lock(state_->mutex);
    if (address != state_->storage.data()) {
      std::terminate();
    }
    state_->allocated = false;
    if (state_->blockNextDeallocation) {
      state_->blockNextDeallocation = false;
      state_->addressFreed = true;
      state_->condition.notify_all();
      state_->condition.wait(lock, [&] { return state_->releaseDeallocation; });
    }
    state_->condition.notify_all();
  }

  std::shared_ptr<AddressReusingResourceState> state_;
};

static_assert(cuda::mr::resource_with<
              AddressReusingResource,
              cuda::mr::device_accessible>);

GpuMemoryOwner makeOwner(
    std::string taskSuffix,
    std::string planNodeId,
    int32_t pipelineId,
    int32_t driverId,
    int32_t operatorId) {
  return GpuMemoryOwner{
      .taskUuid = "task-" + taskSuffix + "-uuid",
      .taskId = "task-" + taskSuffix,
      .queryId = "query-" + taskSuffix,
      .planNodeId = std::move(planNodeId),
      .planNodeType = "TestPlanNode",
      .pipelineId = pipelineId,
      .driverId = driverId,
      .operatorId = operatorId,
      .operatorType = "TestOperator"};
}

const GpuMemoryOwnerSnapshot* findOwner(
    const GpuMemorySnapshot& snapshot,
    GpuMemoryOwnerHandle handle) {
  const auto it = std::find_if(
      snapshot.owners.begin(), snapshot.owners.end(), [&](const auto& owner) {
        return owner.handle == handle;
      });
  return it == snapshot.owners.end() ? nullptr : &*it;
}

bool isCoherent(const GpuMemorySnapshot& snapshot) {
  uint64_t ownerBytes{0};
  uint64_t ownerAllocations{0};
  for (const auto& owner : snapshot.owners) {
    ownerBytes += owner.currentBytes;
    ownerAllocations += owner.currentAllocations;
  }

  uint64_t allocationBytes{0};
  for (const auto& allocation : snapshot.allocations) {
    allocationBytes += allocation.bytes;
    if (findOwner(snapshot, allocation.handle) == nullptr) {
      return false;
    }
  }

  return snapshot.currentBytes == ownerBytes &&
      snapshot.currentBytes == allocationBytes &&
      snapshot.currentAllocations == ownerAllocations &&
      snapshot.currentAllocations == snapshot.allocations.size();
}

std::filesystem::path rawCapturePath(std::string_view suffix) {
  return std::filesystem::temp_directory_path() /
      ("velox-cudf-memory-capture-" + std::to_string(::getpid()) + "-" +
       std::string{suffix} + ".json");
}

GpuMemoryCaptureTask makeCaptureTask(std::string suffix) {
  return GpuMemoryCaptureTask{
      .taskUuid = "task-" + suffix + "-uuid",
      .taskId = "task-" + suffix,
      .queryId = "query-" + suffix};
}

GpuMemoryOwnerSnapshot makeCaptureOwner(
    GpuMemoryOwnerHandle handle,
    GpuMemoryOwner owner,
    uint64_t currentBytes,
    uint64_t sourceLifetimePeakBytes) {
  return GpuMemoryOwnerSnapshot{
      .handle = handle,
      .owner = std::move(owner),
      .currentBytes = currentBytes,
      .peakBytes = sourceLifetimePeakBytes,
      .totalBytes = sourceLifetimePeakBytes,
      .currentAllocations = currentBytes == 0 ? 0UL : 1UL,
      .totalAllocations = currentBytes == 0 ? 0UL : 1UL};
}

GpuMemorySnapshot makeCaptureSnapshot(
    uint64_t currentBytes,
    uint64_t sourceLifetimePeakBytes,
    uint64_t sourceSequence,
    std::vector<GpuMemoryOwnerSnapshot> owners) {
  uint64_t currentAllocations{0};
  for (const auto& owner : owners) {
    currentAllocations += owner.currentAllocations;
  }
  return GpuMemorySnapshot{
      .currentBytes = currentBytes,
      .peakBytes = sourceLifetimePeakBytes,
      .totalBytes = sourceLifetimePeakBytes,
      .currentAllocations = currentAllocations,
      .peakAllocations = currentAllocations,
      .totalAllocations = currentAllocations,
      .sequence = sourceSequence,
      .dataLossEvents = 0,
      .owners = std::move(owners),
      .allocations = {}};
}

GpuMemoryTraceUpdate makeCaptureUpdate(
    uint64_t sourceSequence,
    uint64_t timestampNs,
    GpuMemoryOwnerHandle handle,
    uint64_t globalCurrentBytes,
    uint64_t sourceLifetimeGlobalPeakBytes,
    uint64_t ownerCurrentBytes,
    int64_t deltaBytes) {
  return GpuMemoryTraceUpdate{
      .timestampNs = timestampNs,
      .sequence = sourceSequence,
      .ownerId = handle.ownerId,
      .planNodeId = handle.planNodeId,
      .globalCurrentBytes = globalCurrentBytes,
      .globalPeakBytes = sourceLifetimeGlobalPeakBytes,
      .queryCurrentBytes = ownerCurrentBytes,
      .taskCurrentBytes = ownerCurrentBytes,
      .planNodeCurrentBytes = ownerCurrentBytes,
      .ownerCurrentBytes = ownerCurrentBytes,
      .deltaBytes = deltaBytes};
}

folly::dynamic readRawCapture(const std::filesystem::path& path) {
  std::ifstream input{path, std::ios::binary};
  const std::string contents{
      std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  return folly::parseJson(contents);
}

std::vector<int64_t> jsonIntColumn(
    const folly::dynamic& rows,
    std::string_view field) {
  std::vector<int64_t> values;
  values.reserve(rows.size());
  for (const auto& row : rows) {
    values.push_back(row[std::string{field}].asInt());
  }
  return values;
}

} // namespace

TEST(GpuResourcesTest, TracksAllocationOriginAndOrderedTransitions) {
  GpuMemoryAllocationTracker tracker;
  const auto ownerA = makeOwner("shared", "plan-a", 1, 7, 2);
  auto ownerB = ownerA;
  ownerB.driverId = 8;

  const auto handleA = tracker.registerOwner(ownerA);
  const auto duplicateHandleA = tracker.registerOwner(ownerA);
  const auto handleB = tracker.registerOwner(ownerB);
  EXPECT_EQ(duplicateHandleA, handleA);
  EXPECT_NE(handleB.ownerId, handleA.ownerId);
  auto relabeledOwnerA = ownerA;
  relabeledOwnerA.planNodeType = "RelabeledPlanNode";
  EXPECT_EQ(tracker.registerOwner(relabeledOwnerA), handleA);
  EXPECT_EQ(handleB.planNodeId, handleA.planNodeId);

  auto otherTaskOwner = ownerA;
  otherTaskOwner.taskUuid = "other-task-uuid";
  otherTaskOwner.taskId = "other-task";
  const auto otherTaskHandle = tracker.registerOwner(otherTaskOwner);
  EXPECT_NE(otherTaskHandle.planNodeId, handleA.planNodeId);

  int allocationA;
  int allocationB;
  const auto first = tracker.recordAllocation(&allocationA, 100, handleA);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->sequence, 1);
  EXPECT_GT(first->timestampNs, 0);
  EXPECT_EQ(first->ownerId, handleA.ownerId);
  EXPECT_EQ(first->planNodeId, handleA.planNodeId);
  EXPECT_EQ(first->globalCurrentBytes, 100);
  EXPECT_EQ(first->globalPeakBytes, 100);
  EXPECT_EQ(first->queryCurrentBytes, 100);
  EXPECT_EQ(first->taskCurrentBytes, 100);
  EXPECT_EQ(first->planNodeCurrentBytes, 100);
  EXPECT_EQ(first->ownerCurrentBytes, 100);
  EXPECT_EQ(first->deltaBytes, 100);

  const auto second = tracker.recordAllocation(&allocationB, 60, handleB);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->sequence, 2);
  EXPECT_LT(first->timestampNs, second->timestampNs);
  EXPECT_EQ(second->globalCurrentBytes, 160);
  EXPECT_EQ(second->globalPeakBytes, 160);
  EXPECT_EQ(second->queryCurrentBytes, 160);
  EXPECT_EQ(second->taskCurrentBytes, 160);
  EXPECT_EQ(second->planNodeCurrentBytes, 160);
  EXPECT_EQ(second->ownerCurrentBytes, 60);
  EXPECT_EQ(second->deltaBytes, 60);

  std::optional<GpuMemoryTraceUpdate> deallocation;
  std::thread orphanDeallocator(
      [&] { deallocation = tracker.recordDeallocation(&allocationA); });
  orphanDeallocator.join();

  ASSERT_TRUE(deallocation.has_value());
  EXPECT_EQ(deallocation->sequence, 3);
  EXPECT_LT(second->timestampNs, deallocation->timestampNs);
  EXPECT_EQ(deallocation->ownerId, handleA.ownerId);
  EXPECT_EQ(deallocation->planNodeId, handleA.planNodeId);
  EXPECT_EQ(deallocation->globalCurrentBytes, 60);
  EXPECT_EQ(deallocation->globalPeakBytes, 160);
  EXPECT_EQ(deallocation->queryCurrentBytes, 60);
  EXPECT_EQ(deallocation->taskCurrentBytes, 60);
  EXPECT_EQ(deallocation->planNodeCurrentBytes, 60);
  EXPECT_EQ(deallocation->ownerCurrentBytes, 0);
  EXPECT_EQ(deallocation->deltaBytes, -100);

  const auto snapshot = tracker.snapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 60);
  EXPECT_EQ(snapshot.peakBytes, 160);
  EXPECT_EQ(snapshot.totalBytes, 160);
  EXPECT_EQ(snapshot.currentAllocations, 1);
  EXPECT_EQ(snapshot.peakAllocations, 2);
  EXPECT_EQ(snapshot.totalAllocations, 2);
  EXPECT_EQ(snapshot.sequence, 3);
  EXPECT_EQ(snapshot.dataLossEvents, 0);
  ASSERT_EQ(snapshot.allocations.size(), 1);
  EXPECT_EQ(snapshot.allocations.front().handle, handleB);

  const auto* ownerASnapshot = findOwner(snapshot, handleA);
  ASSERT_NE(ownerASnapshot, nullptr);
  EXPECT_EQ(ownerASnapshot->currentBytes, 0);
  EXPECT_EQ(ownerASnapshot->peakBytes, 100);
  EXPECT_EQ(ownerASnapshot->totalBytes, 100);
  EXPECT_EQ(ownerASnapshot->currentAllocations, 0);
  EXPECT_EQ(ownerASnapshot->totalAllocations, 1);

  const auto* ownerBSnapshot = findOwner(snapshot, handleB);
  ASSERT_NE(ownerBSnapshot, nullptr);
  EXPECT_EQ(ownerBSnapshot->currentBytes, 60);
  EXPECT_EQ(ownerBSnapshot->peakBytes, 60);
  EXPECT_EQ(ownerBSnapshot->totalBytes, 60);
  EXPECT_EQ(ownerBSnapshot->currentAllocations, 1);
  EXPECT_EQ(ownerBSnapshot->totalAllocations, 1);

  const auto final = tracker.recordDeallocation(&allocationB);
  ASSERT_TRUE(final.has_value());
  EXPECT_EQ(final->sequence, 4);
  EXPECT_LT(deallocation->timestampNs, final->timestampNs);
  EXPECT_EQ(final->globalCurrentBytes, 0);
  EXPECT_EQ(final->queryCurrentBytes, 0);
  EXPECT_EQ(final->taskCurrentBytes, 0);
  EXPECT_EQ(final->planNodeCurrentBytes, 0);
  EXPECT_EQ(final->ownerCurrentBytes, 0);
}

TEST(GpuResourcesTest, TracksQueryAndTaskAggregates) {
  GpuMemoryAllocationTracker tracker;
  const auto ownerA = makeOwner("shared", "plan-a", 1, 7, 2);
  auto ownerB = ownerA;
  ownerB.taskUuid = "task-other-uuid";
  ownerB.taskId = "task-other";
  ownerB.planNodeId = "plan-b";

  const auto handleA = tracker.registerOwner(ownerA);
  const auto handleB = tracker.registerOwner(ownerB);
  int allocationA;
  int allocationB;

  const auto first = tracker.recordAllocation(&allocationA, 100, handleA);
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->queryCurrentBytes, 100);
  EXPECT_EQ(first->taskCurrentBytes, 100);

  const auto second = tracker.recordAllocation(&allocationB, 60, handleB);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->queryCurrentBytes, 160);
  EXPECT_EQ(second->taskCurrentBytes, 60);
  EXPECT_EQ(second->planNodeCurrentBytes, 60);

  const auto firstDeallocation = tracker.recordDeallocation(&allocationA);
  ASSERT_TRUE(firstDeallocation.has_value());
  EXPECT_EQ(firstDeallocation->queryCurrentBytes, 60);
  EXPECT_EQ(firstDeallocation->taskCurrentBytes, 0);

  const auto secondDeallocation = tracker.recordDeallocation(&allocationB);
  ASSERT_TRUE(secondDeallocation.has_value());
  EXPECT_EQ(secondDeallocation->queryCurrentBytes, 0);
  EXPECT_EQ(secondDeallocation->taskCurrentBytes, 0);
}

TEST(GpuResourcesTest, InvalidPointerEventsDoNotCorruptLedger) {
  GpuMemoryAllocationTracker tracker;
  const auto handle =
      tracker.registerOwner(makeOwner("invalid", "plan-a", 1, 2, 3));

  int allocation;
  ASSERT_TRUE(tracker.recordAllocation(&allocation, 64, handle).has_value());
  EXPECT_FALSE(tracker.recordAllocation(&allocation, 128, handle).has_value());

  int unknownAllocation;
  EXPECT_FALSE(tracker.recordDeallocation(&unknownAllocation).has_value());

  auto snapshot = tracker.snapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 64);
  EXPECT_EQ(snapshot.currentAllocations, 1);
  EXPECT_EQ(snapshot.sequence, 1);
  EXPECT_EQ(snapshot.dataLossEvents, 2);

  ASSERT_TRUE(tracker.recordDeallocation(&allocation).has_value());
  snapshot = tracker.snapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.sequence, 2);
  EXPECT_EQ(snapshot.dataLossEvents, 2);
}

TEST(GpuResourcesTest, TrackedResourcesShareCombinedSnapshot) {
  auto mainState = std::make_shared<RecordingResourceState>();
  auto outputState = std::make_shared<RecordingResourceState>();
  auto resources = createGpuMemoryTrackingResources(
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{mainState}},
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{outputState}});
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  auto mainResource = rmm::device_async_resource_ref{resources.main};
  auto outputResource = rmm::device_async_resource_ref{resources.output};
  auto* mainAddress = mainResource.allocate(rmm::cuda_stream_default, 256, 256);
  auto* outputAddress =
      outputResource.allocate(rmm::cuda_stream_default, 512, 256);

  auto snapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 768);
  EXPECT_EQ(snapshot.peakBytes, 768);
  EXPECT_EQ(snapshot.totalBytes, 768);
  EXPECT_EQ(snapshot.currentAllocations, 2);
  EXPECT_EQ(snapshot.peakAllocations, 2);
  EXPECT_EQ(snapshot.totalAllocations, 2);
  EXPECT_EQ(snapshot.sequence, 2);
  ASSERT_EQ(snapshot.allocations.size(), 2);
  EXPECT_EQ(snapshot.allocations[0].handle, snapshot.allocations[1].handle);

  mainResource.deallocate(rmm::cuda_stream_default, mainAddress, 256, 256);
  outputResource.deallocate(rmm::cuda_stream_default, outputAddress, 512, 256);

  snapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.peakBytes, 768);
  EXPECT_EQ(snapshot.currentAllocations, 0);
  EXPECT_EQ(snapshot.sequence, 4);
}

TEST(GpuResourcesTest, PreservesAllocationAlignment) {
  auto state = std::make_shared<RecordingResourceState>();
  auto upstream = cuda::mr::any_resource<cuda::mr::device_accessible>{
      RecordingResource{state}};
  auto outputUpstream = upstream;
  auto resources = createGpuMemoryTrackingResources(
      std::move(upstream), std::move(outputUpstream));
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  constexpr std::size_t kAlignment = 4'096;
  auto resource = rmm::device_async_resource_ref{resources.main};
  auto* address = resource.allocate(rmm::cuda_stream_default, 64, kAlignment);
  resource.deallocate(rmm::cuda_stream_default, address, 64, kAlignment);

  std::lock_guard<std::mutex> lock(state->mutex);
  ASSERT_EQ(state->allocationAlignments.size(), 1);
  EXPECT_EQ(state->allocationAlignments.front(), kAlignment);
  ASSERT_EQ(state->deallocationAlignments.size(), 1);
  EXPECT_EQ(state->deallocationAlignments.front(), kAlignment);
}

TEST(GpuResourcesTest, ZeroSizeAllocationIsUntracked) {
  auto state = std::make_shared<RecordingResourceState>();
  auto upstream = cuda::mr::any_resource<cuda::mr::device_accessible>{
      RecordingResource{state}};
  auto outputUpstream = upstream;
  auto resources = createGpuMemoryTrackingResources(
      std::move(upstream), std::move(outputUpstream));
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  auto resource = rmm::device_async_resource_ref{resources.main};
  auto* address = resource.allocate(rmm::cuda_stream_default, 0, 256);
  EXPECT_EQ(address, nullptr);
  resource.deallocate(rmm::cuda_stream_default, address, 0, 256);

  const auto snapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.currentAllocations, 0);
  EXPECT_EQ(snapshot.totalAllocations, 0);
  EXPECT_EQ(snapshot.sequence, 0);
  EXPECT_EQ(snapshot.dataLossEvents, 0);

  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_TRUE(state->allocationAlignments.empty());
  EXPECT_TRUE(state->deallocationAlignments.empty());
}

TEST(GpuResourcesTest, ConcurrentSnapshotsAreCoherent) {
  auto mainState = std::make_shared<RecordingResourceState>();
  auto outputState = std::make_shared<RecordingResourceState>();
  auto resources = createGpuMemoryTrackingResources(
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{mainState}},
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{outputState}});
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  auto mainResource = rmm::device_async_resource_ref{resources.main};
  auto outputResource = rmm::device_async_resource_ref{resources.output};
  std::atomic<bool> start{false};
  std::atomic<bool> workersDone{false};
  std::atomic<bool> coherent{true};

  std::thread snapshotter([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (!workersDone.load(std::memory_order_acquire)) {
      if (!isCoherent(getGpuMemorySnapshot())) {
        coherent.store(false, std::memory_order_release);
        return;
      }
    }
  });

  constexpr int kWorkerCount = 4;
  constexpr int kIterations = 2'000;
  std::vector<std::thread> workers;
  workers.reserve(kWorkerCount);
  for (int worker = 0; worker < kWorkerCount; ++worker) {
    workers.emplace_back([&, worker] {
      start.store(true, std::memory_order_release);
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        auto resource =
            ((worker + iteration) % 2 == 0) ? mainResource : outputResource;
        const std::size_t bytes = 64 + (iteration % 4) * 64;
        auto* address = resource.allocate(rmm::cuda_stream_default, bytes, 256);
        resource.deallocate(rmm::cuda_stream_default, address, bytes, 256);
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }
  workersDone.store(true, std::memory_order_release);
  snapshotter.join();

  const auto snapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(coherent.load(std::memory_order_acquire));
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.currentAllocations, 0);
  EXPECT_EQ(
      snapshot.totalAllocations,
      static_cast<uint64_t>(kWorkerCount * kIterations));
  EXPECT_EQ(
      snapshot.sequence, static_cast<uint64_t>(2 * kWorkerCount * kIterations));
  EXPECT_EQ(snapshot.dataLossEvents, 0);
}

TEST(GpuResourcesTest, ReusedAddressKeepsReplacementAllocation) {
  auto state = std::make_shared<AddressReusingResourceState>();
  auto upstream = cuda::mr::any_resource<cuda::mr::device_accessible>{
      AddressReusingResource{state}};
  auto outputUpstream = upstream;
  auto resources = createGpuMemoryTrackingResources(
      std::move(upstream), std::move(outputUpstream));
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  auto mainResource = rmm::device_async_resource_ref{resources.main};
  auto outputResource = rmm::device_async_resource_ref{resources.output};
  auto* originalAddress =
      mainResource.allocate(rmm::cuda_stream_default, 64, 256);

  std::thread deallocator([&] {
    mainResource.deallocate(rmm::cuda_stream_default, originalAddress, 64, 256);
  });
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condition.wait(lock, [&] { return state->addressFreed; });
  }

  auto* replacementAddress =
      outputResource.allocate(rmm::cuda_stream_default, 128, 256);
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->releaseDeallocation = true;
  }
  state->condition.notify_all();
  deallocator.join();

  ASSERT_EQ(replacementAddress, originalAddress);
  const auto replacementSnapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(replacementSnapshot));
  EXPECT_EQ(replacementSnapshot.currentBytes, 128);
  EXPECT_EQ(replacementSnapshot.currentAllocations, 1);
  EXPECT_EQ(replacementSnapshot.sequence, 3);
  EXPECT_EQ(replacementSnapshot.dataLossEvents, 0);
  ASSERT_EQ(replacementSnapshot.allocations.size(), 1);
  EXPECT_EQ(
      replacementSnapshot.allocations.front().address,
      reinterpret_cast<uintptr_t>(replacementAddress));
  EXPECT_EQ(replacementSnapshot.allocations.front().bytes, 128);

  outputResource.deallocate(
      rmm::cuda_stream_default, replacementAddress, 128, 256);
  const auto finalSnapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(finalSnapshot));
  EXPECT_EQ(finalSnapshot.currentBytes, 0);
  EXPECT_EQ(finalSnapshot.currentAllocations, 0);
  EXPECT_EQ(finalSnapshot.sequence, 4);
  EXPECT_TRUE(finalSnapshot.allocations.empty());
}

TEST(GpuMemoryCaptureTest, BoundedCapacityReportsOverflow) {
  stopGpuMemoryCapture();
  const auto path = rawCapturePath("overflow");
  std::filesystem::remove(path);
  auto cleanup = folly::makeGuard([&] {
    stopGpuMemoryCapture();
    std::filesystem::remove(path);
  });

  GpuMemoryCaptureConfig config;
  config.pathPattern = path.string();
  config.maxEvents = 2;
  ASSERT_TRUE(startGpuMemoryCapture(config));
  ASSERT_TRUE(gpuMemoryCaptureEnabled());

  const auto task = makeCaptureTask("overflow");
  const GpuMemoryOwnerHandle handle{1, 1};
  const auto owner = makeOwner("overflow", "plan-a", 1, 2, 3);
  const auto initial =
      makeCaptureSnapshot(0, 0, 10, {makeCaptureOwner(handle, owner, 0, 0)});
  ASSERT_TRUE(
      gpu_memory_detail::tryBeginGpuMemoryCapture(
          task,
          {GpuMemoryCapturePlanNode{
              .id = "plan-a", .type = "TestPlanNode", .sourceIds = {}}},
          initial));

  const auto timestamp = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
  gpu_memory_detail::recordGpuMemoryCaptureUpdate(
      makeCaptureUpdate(11, timestamp, handle, 10, 10, 10, 10));
  gpu_memory_detail::recordGpuMemoryCaptureUpdate(
      makeCaptureUpdate(12, timestamp + 1, handle, 20, 20, 20, 10));
  gpu_memory_detail::recordGpuMemoryCaptureUpdate(
      makeCaptureUpdate(13, timestamp + 2, handle, 30, 30, 30, 10));

  const auto final = makeCaptureSnapshot(
      30, 30, 13, {makeCaptureOwner(handle, owner, 30, 30)});
  gpu_memory_detail::finishGpuMemoryCapture(
      task.taskUuid, task.taskId, "finished", true, final);
  stopGpuMemoryCapture();

  ASSERT_TRUE(std::filesystem::exists(path));
  const auto capture = readRawCapture(path);
  EXPECT_TRUE(capture["integrity"]["capture_overflow"].asBool());
  EXPECT_FALSE(capture["integrity"]["exact_timeline"].asBool());
  EXPECT_EQ(capture["integrity"]["retained_events"].asInt(), 2);
  EXPECT_EQ(capture["integrity"]["dropped_events"].asInt(), 1);
  EXPECT_EQ(capture["integrity"]["max_events"].asInt(), 2);
  EXPECT_THAT(
      jsonIntColumn(capture["memory_updates"], "source_sequence"),
      testing::ElementsAre(11, 12));
}

TEST(GpuMemoryCaptureTest, FiltersStartWatermarkAndUsesCaptureLocalPeak) {
  stopGpuMemoryCapture();
  const auto path = rawCapturePath("watermark-peak");
  std::filesystem::remove(path);
  auto cleanup = folly::makeGuard([&] {
    stopGpuMemoryCapture();
    std::filesystem::remove(path);
  });

  GpuMemoryCaptureConfig config;
  config.pathPattern = path.string();
  config.maxEvents = 8;
  ASSERT_TRUE(startGpuMemoryCapture(config));
  const auto task = makeCaptureTask("watermark");
  const GpuMemoryOwnerHandle handle{1, 1};
  const auto owner = makeOwner("watermark", "plan-a", 1, 2, 3);
  const auto initial = makeCaptureSnapshot(
      100, 1'000, 10, {makeCaptureOwner(handle, owner, 100, 1'000)});
  ASSERT_TRUE(
      gpu_memory_detail::tryBeginGpuMemoryCapture(
          task,
          {GpuMemoryCapturePlanNode{
              .id = "plan-a", .type = "TestPlanNode", .sourceIds = {}}},
          initial));

  const auto timestamp = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
  gpu_memory_detail::recordGpuMemoryCaptureUpdate(
      makeCaptureUpdate(10, timestamp, handle, 900, 1'000, 900, 800));
  gpu_memory_detail::recordGpuMemoryCaptureUpdate(
      makeCaptureUpdate(11, timestamp + 1, handle, 150, 1'000, 150, 50));
  gpu_memory_detail::recordGpuMemoryCaptureUpdate(
      makeCaptureUpdate(12, timestamp + 2, handle, 80, 1'000, 80, -70));

  const auto final = makeCaptureSnapshot(
      80, 1'000, 12, {makeCaptureOwner(handle, owner, 80, 1'000)});
  gpu_memory_detail::finishGpuMemoryCapture(
      task.taskUuid, task.taskId, "finished", true, final);
  stopGpuMemoryCapture();

  ASSERT_TRUE(std::filesystem::exists(path));
  const auto capture = readRawCapture(path);
  EXPECT_THAT(
      jsonIntColumn(capture["memory_updates"], "source_sequence"),
      testing::ElementsAre(11, 12));
  EXPECT_EQ(
      capture["initial_snapshot"]["source_lifetime_peak_bytes"].asInt(), 1'000);
  EXPECT_EQ(capture["summary"]["capture_local_peak_bytes"].asInt(), 150);
  EXPECT_EQ(
      capture["summary"]["capture_local_peak_timestamp_ns"].asInt(),
      timestamp + 1);
  EXPECT_EQ(capture["summary"]["capture_local_peak_event_sequence"].asInt(), 1);
  EXPECT_TRUE(capture["integrity"]["exact_timeline"].asBool());
  EXPECT_EQ(capture["integrity"]["start_source_sequence"].asInt(), 10);
  EXPECT_EQ(capture["integrity"]["end_source_sequence"].asInt(), 12);
}

TEST(GpuMemoryCaptureTest, RetainsConcurrentTaskOwners) {
  stopGpuMemoryCapture();
  const auto path = rawCapturePath("concurrent-owner");
  std::filesystem::remove(path);
  auto cleanup = folly::makeGuard([&] {
    stopGpuMemoryCapture();
    std::filesystem::remove(path);
  });

  GpuMemoryCaptureConfig config;
  config.pathPattern = path.string();
  config.maxEvents = 8;
  ASSERT_TRUE(startGpuMemoryCapture(config));
  const auto task = makeCaptureTask("selected");
  const GpuMemoryOwnerHandle selectedHandle{1, 1};
  const GpuMemoryOwnerHandle concurrentHandle{2, 2};
  const auto selectedOwner = makeOwner("selected", "plan-a", 1, 2, 3);
  const auto concurrentOwner = makeOwner("concurrent", "plan-b", 4, 5, 6);
  const auto initial = makeCaptureSnapshot(
      100,
      100,
      20,
      {
          makeCaptureOwner(selectedHandle, selectedOwner, 60, 60),
          makeCaptureOwner(concurrentHandle, concurrentOwner, 40, 40),
      });
  ASSERT_TRUE(
      gpu_memory_detail::tryBeginGpuMemoryCapture(
          task,
          {GpuMemoryCapturePlanNode{
              .id = "plan-a", .type = "TestPlanNode", .sourceIds = {}}},
          initial));

  const auto timestamp = gpu_memory_detail::gpuMemoryMonotonicTimeNs();
  gpu_memory_detail::recordGpuMemoryCaptureUpdate(
      makeCaptureUpdate(21, timestamp, concurrentHandle, 130, 130, 70, 30));
  const auto final = makeCaptureSnapshot(
      130,
      130,
      21,
      {
          makeCaptureOwner(selectedHandle, selectedOwner, 60, 60),
          makeCaptureOwner(concurrentHandle, concurrentOwner, 70, 70),
      });
  gpu_memory_detail::finishGpuMemoryCapture(
      task.taskUuid, task.taskId, "finished", true, final);
  stopGpuMemoryCapture();

  ASSERT_TRUE(std::filesystem::exists(path));
  const auto capture = readRawCapture(path);
  EXPECT_EQ(capture["capture"]["task_uuid"].asString(), task.taskUuid);
  ASSERT_EQ(capture["owners"].size(), 2);
  EXPECT_THAT(
      jsonIntColumn(capture["owners"], "owner_id"), testing::ElementsAre(1, 2));
  EXPECT_EQ(
      capture["owners"][0]["identity"]["task_uuid"].asString(),
      selectedOwner.taskUuid);
  EXPECT_EQ(
      capture["owners"][1]["identity"]["task_uuid"].asString(),
      concurrentOwner.taskUuid);
  ASSERT_EQ(capture["memory_updates"].size(), 1);
  EXPECT_EQ(capture["memory_updates"][0]["owner_id"].asInt(), 2);
  EXPECT_EQ(capture["memory_updates"][0]["global_current_bytes"].asInt(), 130);
  EXPECT_EQ(capture["summary"]["capture_local_peak_bytes"].asInt(), 130);
  EXPECT_EQ(capture["final_snapshot"]["owners"].size(), 2);
}

TEST(GpuMemoryCaptureTest, DisabledAndFinishedOperationsAreIdempotent) {
  stopGpuMemoryCapture();
  const auto path = rawCapturePath("idempotent");
  std::filesystem::remove(path);
  auto cleanup = folly::makeGuard([&] {
    stopGpuMemoryCapture();
    std::filesystem::remove(path);
  });

  ASSERT_TRUE(startGpuMemoryCapture(GpuMemoryCaptureConfig{}));
  EXPECT_FALSE(gpuMemoryCaptureEnabled());
  const GpuMemoryTraceUpdate ignoredUpdate{
      .timestampNs = gpu_memory_detail::gpuMemoryMonotonicTimeNs(),
      .sequence = 1,
      .globalCurrentBytes = 1,
      .globalPeakBytes = 1,
      .deltaBytes = 1};
  gpu_memory_detail::recordGpuMemoryCaptureUpdate(ignoredUpdate);
  EXPECT_FALSE(
      gpu_memory_detail::beginGpuMemoryCaptureOperatorCall(1, "ignored")
          .active);
  gpu_memory_detail::finishGpuMemoryCapture(
      "missing-uuid", "missing-task", "finished", true, {});

  GpuMemoryCaptureConfig config;
  config.pathPattern = path.string();
  config.maxEvents = 4;
  ASSERT_TRUE(startGpuMemoryCapture(config));
  const auto task = makeCaptureTask("idempotent");
  const auto initial = makeCaptureSnapshot(0, 0, 0, {});
  ASSERT_TRUE(gpu_memory_detail::tryBeginGpuMemoryCapture(task, {}, initial));
  gpu_memory_detail::finishGpuMemoryCapture(
      task.taskUuid, task.taskId, "finished", true, initial);
  auto replacementFinal = initial;
  replacementFinal.sequence = 99;
  gpu_memory_detail::finishGpuMemoryCapture(
      task.taskUuid, task.taskId, "failed", false, replacementFinal);
  stopGpuMemoryCapture();
  stopGpuMemoryCapture();

  EXPECT_FALSE(gpuMemoryCaptureEnabled());
  ASSERT_TRUE(std::filesystem::exists(path));
  EXPECT_EQ(gpuMemoryCaptureLastPath(), path.string());
  const auto capture = readRawCapture(path);
  EXPECT_EQ(capture["capture"]["task_state"].asString(), "finished");
  EXPECT_TRUE(capture["capture"]["complete"].asBool());
  EXPECT_TRUE(capture["capture"]["cleanup_complete"].asBool());
  EXPECT_EQ(capture["final_snapshot"]["source_sequence"].asInt(), 0);
  EXPECT_EQ(capture["memory_updates"].size(), 0);
  EXPECT_EQ(capture["integrity"]["retained_events"].asInt(), 0);
}

TEST(GpuResourcesTest, AllocationFailureRethrowsWithoutCounting) {
  const auto tracePath = std::filesystem::temp_directory_path() /
      ("velox-cudf-memory-oom-" + std::to_string(::getpid()) + ".pftrace");
  std::filesystem::remove(tracePath);
  ASSERT_TRUE(startGpuMemoryTrace(tracePath.string()));
  auto stopGuard = folly::makeGuard([] { stopGpuMemoryTrace(); });

  auto failingState = std::make_shared<RecordingResourceState>();
  failingState->throwOnAllocation = true;
  auto outputState = std::make_shared<RecordingResourceState>();
  auto resources = createGpuMemoryTrackingResources(
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{failingState}},
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{outputState}});
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  constexpr std::size_t kAlignment = 4'096;
  auto resource = rmm::device_async_resource_ref{resources.main};
  EXPECT_THROW(
      resource.allocate(rmm::cuda_stream_default, 1'234, kAlignment),
      rmm::out_of_memory);

  {
    std::lock_guard<std::mutex> lock(failingState->mutex);
    ASSERT_EQ(failingState->allocationAlignments.size(), 1);
    EXPECT_EQ(failingState->allocationAlignments.front(), kAlignment);
  }

  const auto snapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(snapshot));
  EXPECT_EQ(snapshot.currentBytes, 0);
  EXPECT_EQ(snapshot.peakBytes, 0);
  EXPECT_EQ(snapshot.totalBytes, 0);
  EXPECT_EQ(snapshot.currentAllocations, 0);
  EXPECT_EQ(snapshot.totalAllocations, 0);
  EXPECT_EQ(snapshot.sequence, 0);

  stopGpuMemoryTrace();
  stopGuard.dismiss();
  ASSERT_TRUE(std::filesystem::exists(tracePath));
  EXPECT_GT(std::filesystem::file_size(tracePath), 0);
  if (std::getenv("VELOX_CUDF_KEEP_PERFETTO_TEST_TRACE") == nullptr) {
    std::filesystem::remove(tracePath);
  } else {
    LOG(INFO) << "Kept GPU-memory OOM test trace: " << tracePath;
  }
}

TEST(GpuResourcesTest, StreamsPerfettoTrace) {
  const auto path = std::filesystem::temp_directory_path() /
      ("velox-cudf-memory-" + std::to_string(::getpid()) + ".pftrace");
  std::filesystem::remove(path);
  ASSERT_TRUE(startGpuMemoryTrace(path.string()));
  auto stopGuard = folly::makeGuard([] { stopGpuMemoryTrace(); });

  GpuMemoryAllocationTracker tracker;
  const auto firstHandle =
      tracker.registerOwner(makeOwner("trace", "1353", 2, 4, 6));
  const auto secondHandle =
      tracker.registerOwner(makeOwner("trace", "1354", 2, 4, 7));
  int firstAllocation;
  int secondAllocation;
  ASSERT_TRUE(tracker.recordAllocation(&firstAllocation, 1'024, firstHandle)
                  .has_value());
  ASSERT_TRUE(tracker.recordAllocation(&secondAllocation, 2'048, secondHandle)
                  .has_value());
  markGpuMemoryTrace("unit-test-marker");
  ASSERT_TRUE(tracker.recordDeallocation(&firstAllocation).has_value());
  ASSERT_TRUE(tracker.recordDeallocation(&secondAllocation).has_value());

  stopGpuMemoryTrace();
  stopGuard.dismiss();
  EXPECT_FALSE(gpuMemoryTraceEnabled());
  EXPECT_TRUE(gpuMemoryTracePath().empty());
  ASSERT_TRUE(std::filesystem::exists(path));
  EXPECT_GT(std::filesystem::file_size(path), 0);
  if (std::getenv("VELOX_CUDF_KEEP_PERFETTO_TEST_TRACE") == nullptr) {
    std::filesystem::remove(path);
  } else {
    LOG(INFO) << "Kept GPU-memory Perfetto test trace: " << path;
  }
}

TEST(GpuResourcesTest, SealsFinishedProducerThread) {
  const auto path = std::filesystem::temp_directory_path() /
      ("velox-cudf-memory-producer-" + std::to_string(::getpid()) + ".pftrace");
  std::filesystem::remove(path);
  ASSERT_TRUE(startGpuMemoryTrace(path.string()));
  auto stopGuard = folly::makeGuard([] { stopGpuMemoryTrace(); });

  std::thread producer([] {
    GpuMemoryAllocationTracker tracker;
    const auto handle =
        tracker.registerOwner(makeOwner("producer", "2468", 1, 3, 5));
    int allocation;
    EXPECT_TRUE(
        tracker.recordAllocation(&allocation, 4'096, handle).has_value());
    EXPECT_TRUE(tracker.recordDeallocation(&allocation).has_value());
    EXPECT_TRUE(
        gpu_memory_detail::beginGpuMemoryOperatorCall(
            handle.ownerId, "producer-call"));
    gpu_memory_detail::endGpuMemoryOperatorCall(handle.ownerId);
  });
  producer.join();

  stopGpuMemoryTrace();
  stopGuard.dismiss();
  ASSERT_TRUE(std::filesystem::exists(path));
  std::ifstream input(path, std::ios::binary);
  const std::string trace{
      std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  EXPECT_NE(trace.find("query-producer"), std::string::npos);
  EXPECT_NE(trace.find("TestPlanNode | 2468"), std::string::npos);

  std::vector<int64_t> counterValues;
  size_t numSliceBegins{0};
  size_t numSliceEnds{0};
  const perfetto::protos::pbzero::Trace::Decoder traceDecoder{trace};
  for (auto packetIterator = traceDecoder.packet(); packetIterator;
       ++packetIterator) {
    const perfetto::protos::pbzero::TracePacket::Decoder packet{
        *packetIterator};
    if (!packet.has_track_event()) {
      continue;
    }
    const perfetto::protos::pbzero::TrackEvent::Decoder trackEvent{
        packet.track_event()};
    if (trackEvent.has_counter_value()) {
      counterValues.push_back(trackEvent.counter_value());
    }
    if (trackEvent.has_type() &&
        trackEvent.type() ==
            perfetto::protos::pbzero::TrackEvent::TYPE_SLICE_BEGIN) {
      ++numSliceBegins;
    }
    if (trackEvent.has_type() &&
        trackEvent.type() ==
            perfetto::protos::pbzero::TrackEvent::TYPE_SLICE_END) {
      ++numSliceEnds;
    }
  }
  EXPECT_THAT(
      counterValues,
      testing::UnorderedElementsAre(0, 0, 0, 4'096, 4'096, 4'096, 4'096));
  EXPECT_EQ(numSliceBegins, 1);
  EXPECT_EQ(numSliceEnds, 1);
  if (std::getenv("VELOX_CUDF_KEEP_PERFETTO_TEST_TRACE") == nullptr) {
    std::filesystem::remove(path);
  } else {
    LOG(INFO) << "Kept GPU-memory producer-thread trace: " << path;
  }
}

} // namespace facebook::velox::cudf_velox::test
