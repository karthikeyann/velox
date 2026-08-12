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

#include "velox/experimental/cudf/exec/GpuMemoryTracker.h"

#include "velox/common/memory/Memory.h"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/error.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <new>
#include <thread>

namespace facebook::velox::cudf_velox::test {
namespace {

struct HostResourceState {
  std::atomic<bool> failNextAllocation{false};
  std::atomic<size_t> lastAllocationAlignment{0};
  std::atomic<size_t> lastDeallocationAlignment{0};
};

class HostBackedDeviceResource {
 public:
  explicit HostBackedDeviceResource(std::shared_ptr<HostResourceState> state)
      : state_(std::move(state)) {}

  void* allocate_sync(size_t bytes, size_t alignment) {
    if (state_->failNextAllocation.exchange(false)) {
      throw rmm::out_of_memory("injected allocation failure");
    }
    state_->lastAllocationAlignment = alignment;
    return ::operator new(bytes, std::align_val_t(alignment));
  }

  void
  deallocate_sync(void* pointer, size_t /*bytes*/, size_t alignment) noexcept {
    state_->lastDeallocationAlignment = alignment;
    ::operator delete(pointer, std::align_val_t(alignment));
  }

  void* allocate(cuda::stream_ref, size_t bytes, size_t alignment) {
    return allocate_sync(bytes, alignment);
  }

  void deallocate(
      cuda::stream_ref,
      void* pointer,
      size_t bytes,
      size_t alignment) noexcept {
    deallocate_sync(pointer, bytes, alignment);
  }

  bool operator==(const HostBackedDeviceResource& other) const noexcept {
    return state_ == other.state_;
  }

  bool operator!=(const HostBackedDeviceResource& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const HostBackedDeviceResource&,
      cuda::mr::device_accessible) noexcept {}

 private:
  std::shared_ptr<HostResourceState> state_;
};

static_assert(cuda::mr::resource_with<
              HostBackedDeviceResource,
              cuda::mr::device_accessible>);

GpuMemoryResource resource(const std::shared_ptr<HostResourceState>& state) {
  return HostBackedDeviceResource{state};
}

GpuMemoryOwner owner(std::string suffix, int32_t driverId, int32_t operatorId) {
  return {
      "query-" + suffix,
      "task-" + suffix,
      "task-" + suffix + "-uuid",
      "plan-" + suffix,
      2,
      driverId,
      operatorId,
      "TestOperator"};
}

const GpuMemoryOwnerSnapshot* findOwner(
    const GpuMemorySnapshot& snapshot,
    const GpuMemoryOwner& expected) {
  const auto it = std::find_if(
      snapshot.owners.begin(), snapshot.owners.end(), [&](const auto& item) {
        return item.owner == expected;
      });
  return it == snapshot.owners.end() ? nullptr : std::addressof(*it);
}

class GpuMemoryTrackerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::initialize({});
  }

  void SetUp() override {
    ASSERT_TRUE(resetGpuMemoryTracking());
  }

  void TearDown() override {
    EXPECT_TRUE(resetGpuMemoryTracking());
  }
};

TEST_F(GpuMemoryTrackerTest, TracksAcrossResourcesAndThreads) {
  auto tempState = std::make_shared<HostResourceState>();
  auto outputState = std::make_shared<HostResourceState>();
  const auto expectedOwner = owner("shared", 7, 3);
  auto resources = createGpuMemoryTrackingResources(
      resource(tempState), resource(outputState), expectedOwner);

  void* tempAllocation{nullptr};
  std::thread allocateTemp([&] {
    tempAllocation = resources.temp.allocate(
        rmm::cuda_stream_default, 256, alignof(std::max_align_t));
  });
  allocateTemp.join();
  auto* outputAllocation =
      resources.output.allocate_sync(128, alignof(std::max_align_t));

  auto snapshot = getGpuMemorySnapshot();
  ASSERT_EQ(snapshot.currentBytes, 384);
  const auto* trackedOwner = findOwner(snapshot, expectedOwner);
  ASSERT_NE(trackedOwner, nullptr);
  EXPECT_EQ(trackedOwner->currentBytes, 384);
  EXPECT_GE(trackedOwner->peakBytes, 384);

  std::thread freeTemp([&] {
    resources.temp.deallocate(
        rmm::cuda_stream_default,
        tempAllocation,
        256,
        alignof(std::max_align_t));
  });
  freeTemp.join();
  resources.output.deallocate_sync(
      outputAllocation, 128, alignof(std::max_align_t));

  snapshot = getGpuMemorySnapshot();
  EXPECT_EQ(snapshot.currentBytes, 0);
  trackedOwner = findOwner(snapshot, expectedOwner);
  ASSERT_NE(trackedOwner, nullptr);
  EXPECT_EQ(trackedOwner->currentBytes, 0);
  EXPECT_EQ(trackedOwner->allocations, 2);
  EXPECT_EQ(trackedOwner->frees, 2);
}

TEST_F(GpuMemoryTrackerTest, RollsBackFailedAllocation) {
  auto state = std::make_shared<HostResourceState>();
  const auto expectedOwner = owner("failure", 1, 2);
  auto resources = createGpuMemoryTrackingResources(
      resource(state), resource(state), expectedOwner);

  state->failNextAllocation = true;
  EXPECT_THROW(
      resources.temp.allocate(
          rmm::cuda_stream_default, 512, alignof(std::max_align_t)),
      rmm::out_of_memory);

  const auto snapshot = getGpuMemorySnapshot();
  EXPECT_EQ(snapshot.currentBytes, 0);
  const auto* trackedOwner = findOwner(snapshot, expectedOwner);
  ASSERT_NE(trackedOwner, nullptr);
  EXPECT_EQ(trackedOwner->currentBytes, 0);
  EXPECT_EQ(trackedOwner->allocations, trackedOwner->frees);
}

TEST_F(GpuMemoryTrackerTest, KeepsOwnersSeparateAndReconciles) {
  auto state = std::make_shared<HostResourceState>();
  const auto ownerA = owner("query", 1, 1);
  auto ownerB = ownerA;
  ownerB.driverId = 2;
  auto resourcesA = createGpuMemoryTrackingResources(
      resource(state), resource(state), ownerA);
  auto resourcesB = createGpuMemoryTrackingResources(
      resource(state), resource(state), ownerB);

  auto* allocationA = resourcesA.temp.allocate_sync(64, 64);
  auto* allocationB = resourcesB.temp.allocate_sync(96, 64);
  auto snapshot = getGpuMemorySnapshot();
  ASSERT_EQ(snapshot.currentBytes, 160);
  const auto* trackedOwnerA = findOwner(snapshot, ownerA);
  const auto* trackedOwnerB = findOwner(snapshot, ownerB);
  ASSERT_NE(trackedOwnerA, nullptr);
  ASSERT_NE(trackedOwnerB, nullptr);
  EXPECT_EQ(trackedOwnerA->currentBytes, 64);
  EXPECT_EQ(trackedOwnerB->currentBytes, 96);

  resourcesB.temp.deallocate_sync(allocationB, 96, 64);
  resourcesA.temp.deallocate_sync(allocationA, 64, 64);
  snapshot = getGpuMemorySnapshot();
  EXPECT_EQ(snapshot.currentBytes, 0);
  uint64_t ownerBytes{0};
  for (const auto& item : snapshot.owners) {
    ownerBytes += item.currentBytes;
  }
  EXPECT_EQ(snapshot.currentBytes, ownerBytes);
}

TEST_F(GpuMemoryTrackerTest, PreservesAlignmentAndIgnoresZeroBytes) {
  auto state = std::make_shared<HostResourceState>();
  auto resources = createGpuMemoryTrackingResources(
      resource(state), resource(state), owner("alignment", 0, 0));

  constexpr size_t kAlignment{256};
  auto* allocation = resources.temp.allocate_sync(256, kAlignment);
  resources.temp.deallocate_sync(allocation, 256, kAlignment);
  EXPECT_EQ(state->lastAllocationAlignment, kAlignment);
  EXPECT_EQ(state->lastDeallocationAlignment, kAlignment);

  const auto before = getGpuMemorySnapshot();
  auto* zero = resources.temp.allocate_sync(0, alignof(std::max_align_t));
  resources.temp.deallocate_sync(zero, 0, alignof(std::max_align_t));
  const auto after = getGpuMemorySnapshot();
  EXPECT_EQ(after.currentBytes, before.currentBytes);
  EXPECT_EQ(after.cumulativeRequestedBytes, before.cumulativeRequestedBytes);
}

TEST_F(GpuMemoryTrackerTest, RefusesResetWhileAllocationIsLive) {
  auto state = std::make_shared<HostResourceState>();
  {
    auto resources = createGpuMemoryTrackingResources(
        resource(state), resource(state), owner("reset", 0, 0));
    auto* allocation = resources.temp.allocate_sync(128, 64);
    EXPECT_FALSE(resetGpuMemoryTracking());
    EXPECT_EQ(getGpuMemorySnapshot().currentBytes, 128);
    resources.temp.deallocate_sync(allocation, 128, 64);

    // Borrowed resource handles remain valid until their registry retires, so
    // a zero-byte session still cannot reset while the pair is in scope.
    EXPECT_FALSE(resetGpuMemoryTracking());
  }
  EXPECT_TRUE(resetGpuMemoryTracking());

  // A new session after reset must not reuse a retired registry or NVTX epoch.
  auto resources = createGpuMemoryTrackingResources(
      resource(state), resource(state), owner("after-reset", 0, 0));
  auto* allocation = resources.temp.allocate_sync(64, 64);
  EXPECT_EQ(getGpuMemorySnapshot().currentBytes, 64);
  resources.temp.deallocate_sync(allocation, 64, 64);
}

TEST_F(GpuMemoryTrackerTest, LiveLeaseBlocksConcurrentReset) {
  auto state = std::make_shared<HostResourceState>();
  auto resources = createGpuMemoryTrackingResources(
      resource(state), resource(state), owner("concurrent-reset", 0, 0));
  std::atomic<bool> resetSucceeded{false};

  std::thread allocator([&] {
    for (int32_t i = 0; i < 100; ++i) {
      auto* allocation = resources.temp.allocate_sync(64, 64);
      resources.temp.deallocate_sync(allocation, 64, 64);
    }
  });
  std::thread resetter([&] {
    for (int32_t i = 0; i < 100; ++i) {
      if (resetGpuMemoryTracking()) {
        resetSucceeded = true;
      }
    }
  });
  allocator.join();
  resetter.join();

  EXPECT_FALSE(resetSucceeded);
  const auto snapshot = getGpuMemorySnapshot();
  EXPECT_EQ(snapshot.currentBytes, 0);
  const auto* trackedOwner =
      findOwner(snapshot, owner("concurrent-reset", 0, 0));
  ASSERT_NE(trackedOwner, nullptr);
  EXPECT_EQ(trackedOwner->allocations, 100);
  EXPECT_EQ(trackedOwner->frees, 100);
  EXPECT_EQ(trackedOwner->cumulativeRequestedBytes, 6'400);
}

TEST_F(GpuMemoryTrackerTest, ConcurrentSnapshotsReconcile) {
  auto state = std::make_shared<HostResourceState>();
  auto resourcesA = createGpuMemoryTrackingResources(
      resource(state), resource(state), owner("snapshot-a", 0, 0));
  auto resourcesB = createGpuMemoryTrackingResources(
      resource(state), resource(state), owner("snapshot-b", 1, 1));
  std::atomic<bool> start{false};
  std::atomic<int32_t> completed{0};
  std::atomic<bool> mismatch{false};

  auto worker = [&](GpuMemoryResource& resource, size_t bytes) {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (int32_t i = 0; i < 100; ++i) {
      auto* allocation = resource.allocate_sync(bytes, 64);
      std::this_thread::yield();
      resource.deallocate_sync(allocation, bytes, 64);
    }
    completed.fetch_add(1, std::memory_order_release);
  };

  std::thread workerA(worker, std::ref(resourcesA.temp), 64);
  std::thread workerB(worker, std::ref(resourcesB.temp), 96);
  start.store(true, std::memory_order_release);
  do {
    const auto snapshot = getGpuMemorySnapshot();
    uint64_t ownerBytes{0};
    for (const auto& item : snapshot.owners) {
      ownerBytes += item.currentBytes;
    }
    if (ownerBytes != snapshot.currentBytes) {
      mismatch = true;
    }
  } while (completed.load(std::memory_order_acquire) != 2);
  workerA.join();
  workerB.join();

  EXPECT_FALSE(mismatch);
  EXPECT_EQ(getGpuMemorySnapshot().currentBytes, 0);
}

TEST_F(GpuMemoryTrackerTest, RetainsCompletedOwnerStatistics) {
  auto state = std::make_shared<HostResourceState>();
  const auto completedOwner = owner("completed", 0, 0);
  {
    auto resources = createGpuMemoryTrackingResources(
        resource(state), resource(state), completedOwner);
    auto* allocation = resources.temp.allocate_sync(192, 64);
    resources.temp.deallocate_sync(allocation, 192, 64);
  }

  const auto snapshot = getGpuMemorySnapshot();
  const auto* trackedOwner = findOwner(snapshot, completedOwner);
  ASSERT_NE(trackedOwner, nullptr);
  EXPECT_EQ(trackedOwner->currentBytes, 0);
  EXPECT_GE(trackedOwner->peakBytes, 192);
  EXPECT_EQ(trackedOwner->cumulativeRequestedBytes, 192);
}

} // namespace
} // namespace facebook::velox::cudf_velox::test
