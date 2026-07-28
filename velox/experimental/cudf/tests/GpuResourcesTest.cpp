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

#include "velox/experimental/cudf/exec/GpuResources.h"

#include <rmm/error.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#include <perfetto.h>
#pragma GCC diagnostic pop

#include <folly/ScopeGuard.h>
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
  EXPECT_EQ(first->planNodeCurrentBytes, 100);
  EXPECT_EQ(first->ownerCurrentBytes, 100);
  EXPECT_EQ(first->deltaBytes, 100);

  const auto second = tracker.recordAllocation(&allocationB, 60, handleB);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->sequence, 2);
  EXPECT_LT(first->timestampNs, second->timestampNs);
  EXPECT_EQ(second->globalCurrentBytes, 160);
  EXPECT_EQ(second->globalPeakBytes, 160);
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
  EXPECT_EQ(final->planNodeCurrentBytes, 0);
  EXPECT_EQ(final->ownerCurrentBytes, 0);
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
