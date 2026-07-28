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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"

#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <folly/ScopeGuard.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
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

class LogCapture final : public google::LogSink {
 public:
  LogCapture() {
    google::AddLogSink(this);
  }

  ~LogCapture() override {
    google::RemoveLogSink(this);
  }

  LogCapture(const LogCapture&) = delete;
  LogCapture& operator=(const LogCapture&) = delete;

  void send(
      google::LogSeverity /*severity*/,
      const char* /*fullFilename*/,
      const char* /*baseFilename*/,
      int /*line*/,
      const struct ::tm* /*time*/,
      const char* message,
      std::size_t messageLength) override {
    std::lock_guard<std::mutex> lock(mutex_);
    captured_.append(message, messageLength);
  }

  std::string captured() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return captured_;
  }

 private:
  mutable std::mutex mutex_;
  std::string captured_;
};

class RecordingRuntimeStatWriter final : public BaseRuntimeStatWriter {
 public:
  void addRuntimeStat(std::string_view name, const RuntimeCounter& value)
      override {
    metrics.emplace_back(name, value);
  }

  std::vector<std::pair<std::string, RuntimeCounter>> metrics;
};

const GpuMemoryResourceSnapshot& findResource(
    const GpuMemorySnapshot& snapshot,
    GpuMemoryResourceKind kind) {
  const auto it = std::find_if(
      snapshot.resources.begin(),
      snapshot.resources.end(),
      [kind](const auto& resource) { return resource.kind == kind; });
  VELOX_CHECK(
      it != snapshot.resources.end(), "Missing GPU memory resource snapshot");
  return *it;
}

const GpuMemoryOwnerSnapshot& findOwner(
    const GpuMemorySnapshot& snapshot,
    const GpuMemoryOwner& owner,
    GpuMemoryResourceKind kind) {
  const auto it = std::find_if(
      snapshot.owners.begin(),
      snapshot.owners.end(),
      [&](const auto& ownerSnapshot) {
        return ownerSnapshot.owner == owner && ownerSnapshot.kind == kind;
      });
  VELOX_CHECK(it != snapshot.owners.end(), "Missing GPU memory owner snapshot");
  return *it;
}

bool hasOwner(
    const GpuMemorySnapshot& snapshot,
    const GpuMemoryOwner& owner,
    GpuMemoryResourceKind kind) {
  return std::any_of(
      snapshot.owners.begin(),
      snapshot.owners.end(),
      [&](const auto& ownerSnapshot) {
        return ownerSnapshot.owner == owner && ownerSnapshot.kind == kind;
      });
}

bool isCoherent(const GpuMemorySnapshot& snapshot) {
  for (const auto& resource : snapshot.resources) {
    uint64_t ownerBytes{0};
    uint64_t ownerAllocations{0};
    for (const auto& owner : snapshot.owners) {
      if (owner.kind == resource.kind) {
        ownerBytes += owner.currentBytes;
        ownerAllocations += owner.currentAllocations;
      }
    }

    uint64_t allocationBytes{0};
    uint64_t allocationCount{0};
    for (const auto& allocation : snapshot.allocations) {
      if (allocation.kind == resource.kind) {
        allocationBytes += allocation.bytes;
        ++allocationCount;
      }
    }

    if (resource.currentBytes != ownerBytes ||
        resource.currentBytes != allocationBytes ||
        resource.currentAllocations != ownerAllocations ||
        resource.currentAllocations != allocationCount) {
      return false;
    }
  }
  return true;
}

} // namespace

TEST(GpuResourcesTest, DeallocationUsesOriginalAllocationOwner) {
  GpuMemoryAllocationTracker tracker;
  const GpuMemoryOwner ownerA{
      .taskUuid = "task-a-uuid",
      .taskId = "task-a",
      .queryId = "query-a",
      .planNodeId = "plan-a",
      .operatorId = 7,
      .operatorType = "CudfGroupby"};
  const GpuMemoryOwner ownerB{
      .taskUuid = "task-b-uuid",
      .taskId = "task-b",
      .queryId = "query-b",
      .planNodeId = "plan-b",
      .operatorId = 9,
      .operatorType = "CudfHashJoinBuild"};

  int mainAllocation;
  int ownerAOutputAllocation;
  int ownerBOutputAllocation;
  tracker.recordAllocation(
      &mainAllocation, 128, GpuMemoryResourceKind::kMain, ownerA);
  tracker.recordAllocation(
      &ownerAOutputAllocation, 32, GpuMemoryResourceKind::kOutput, ownerA);
  tracker.recordAllocation(
      &ownerBOutputAllocation, 64, GpuMemoryResourceKind::kOutput, ownerB);

  std::thread orphanDeallocator(
      [&] { tracker.recordDeallocation(&mainAllocation); });
  orphanDeallocator.join();

  int replacementMainAllocation;
  const auto replacementUpdate = tracker.recordAllocation(
      &replacementMainAllocation, 64, GpuMemoryResourceKind::kMain, ownerA);
  EXPECT_FALSE(replacementUpdate.queryPeakBytes.has_value());
  EXPECT_FALSE(replacementUpdate.queryResourcePeakBytes.has_value());
  EXPECT_FALSE(replacementUpdate.planNodePeakBytes.has_value());
  EXPECT_FALSE(replacementUpdate.planNodeResourcePeakBytes.has_value());
  EXPECT_FALSE(replacementUpdate.operatorPeakBytes.has_value());
  EXPECT_FALSE(replacementUpdate.operatorResourcePeakBytes.has_value());

  const auto snapshot = tracker.snapshot();
  ASSERT_EQ(snapshot.allocations.size(), 3);

  const auto& main = findResource(snapshot, GpuMemoryResourceKind::kMain);
  EXPECT_EQ(main.currentBytes, 64);
  EXPECT_EQ(main.peakBytes, 128);
  EXPECT_EQ(main.totalBytes, 192);
  EXPECT_EQ(main.currentAllocations, 1);
  EXPECT_EQ(main.peakAllocations, 1);
  EXPECT_EQ(main.totalAllocations, 2);

  const auto& output = findResource(snapshot, GpuMemoryResourceKind::kOutput);
  EXPECT_EQ(output.currentBytes, 96);
  EXPECT_EQ(output.peakBytes, 96);
  EXPECT_EQ(output.totalBytes, 96);
  EXPECT_EQ(output.currentAllocations, 2);
  EXPECT_EQ(output.peakAllocations, 2);
  EXPECT_EQ(output.totalAllocations, 2);

  const auto& ownerAMain =
      findOwner(snapshot, ownerA, GpuMemoryResourceKind::kMain);
  EXPECT_EQ(ownerAMain.currentBytes, 64);
  EXPECT_EQ(ownerAMain.totalBytes, 192);
  EXPECT_EQ(ownerAMain.currentAllocations, 1);
  EXPECT_EQ(ownerAMain.totalAllocations, 2);

  const auto& ownerAOutput =
      findOwner(snapshot, ownerA, GpuMemoryResourceKind::kOutput);
  EXPECT_EQ(ownerAOutput.currentBytes, 32);
  EXPECT_EQ(ownerAOutput.currentAllocations, 1);

  const auto& ownerBOutput =
      findOwner(snapshot, ownerB, GpuMemoryResourceKind::kOutput);
  EXPECT_EQ(ownerBOutput.currentBytes, 64);
  EXPECT_EQ(ownerBOutput.currentAllocations, 1);

  tracker.recordDeallocation(&replacementMainAllocation);
}

TEST(GpuResourcesTest, QueryScopedPeaksTrackOverlapAndResourceSplits) {
  GpuMemoryAllocationTracker tracker;
  const GpuMemoryOwner ownerA{
      .taskUuid = "task-a-uuid",
      .taskId = "task-a",
      .queryId = "query-1",
      .planNodeId = "plan-a",
      .operatorId = 7,
      .operatorType = "CudfHashJoinBuild"};
  const GpuMemoryOwner ownerB{
      .taskUuid = "task-b-uuid",
      .taskId = "task-b",
      .queryId = "query-1",
      .planNodeId = "plan-a",
      .operatorId = 7,
      .operatorType = "CudfHashJoinBuild"};
  const GpuMemoryOwner ownerC{
      .taskUuid = "task-c-uuid",
      .taskId = "task-c",
      .queryId = "query-1",
      .planNodeId = "plan-b",
      .operatorId = 9,
      .operatorType = "CudfGroupby"};
  const GpuMemoryOwner otherQueryOwner{
      .taskUuid = "task-d-uuid",
      .taskId = "task-d",
      .queryId = "query-2",
      .planNodeId = "plan-a",
      .operatorId = 7,
      .operatorType = "CudfHashJoinBuild"};

  int ownerAMain;
  int ownerBOutput;
  int ownerCMain;
  int ownerBSecondOutput;
  int otherQueryMain;

  auto update = tracker.recordAllocation(
      &ownerAMain, 100, GpuMemoryResourceKind::kMain, ownerA);
  EXPECT_EQ(update.queryPeakBytes, 100);
  EXPECT_EQ(update.queryResourcePeakBytes, 100);
  EXPECT_EQ(update.planNodePeakBytes, 100);
  EXPECT_EQ(update.planNodeResourcePeakBytes, 100);
  EXPECT_EQ(update.operatorPeakBytes, 100);
  EXPECT_EQ(update.operatorResourcePeakBytes, 100);

  update = tracker.recordAllocation(
      &ownerBOutput, 60, GpuMemoryResourceKind::kOutput, ownerB);
  EXPECT_EQ(update.queryPeakBytes, 160);
  EXPECT_EQ(update.queryResourcePeakBytes, 60);
  EXPECT_EQ(update.planNodePeakBytes, 160);
  EXPECT_EQ(update.planNodeResourcePeakBytes, 60);
  EXPECT_EQ(update.operatorPeakBytes, 160);
  EXPECT_EQ(update.operatorResourcePeakBytes, 60);

  update = tracker.recordAllocation(
      &ownerCMain, 80, GpuMemoryResourceKind::kMain, ownerC);
  EXPECT_EQ(update.queryPeakBytes, 240);
  EXPECT_EQ(update.queryResourcePeakBytes, 180);
  EXPECT_EQ(update.planNodePeakBytes, 80);
  EXPECT_EQ(update.planNodeResourcePeakBytes, 80);
  EXPECT_EQ(update.operatorPeakBytes, 80);
  EXPECT_EQ(update.operatorResourcePeakBytes, 80);

  tracker.recordDeallocation(&ownerAMain);
  update = tracker.recordAllocation(
      &ownerBSecondOutput, 90, GpuMemoryResourceKind::kOutput, ownerB);
  EXPECT_FALSE(update.queryPeakBytes.has_value());
  EXPECT_EQ(update.queryResourcePeakBytes, 150);
  EXPECT_FALSE(update.planNodePeakBytes.has_value());
  EXPECT_EQ(update.planNodeResourcePeakBytes, 150);
  EXPECT_FALSE(update.operatorPeakBytes.has_value());
  EXPECT_EQ(update.operatorResourcePeakBytes, 150);

  update = tracker.recordAllocation(
      &otherQueryMain, 500, GpuMemoryResourceKind::kMain, otherQueryOwner);
  EXPECT_EQ(update.queryPeakBytes, 500);
  EXPECT_EQ(update.queryResourcePeakBytes, 500);
  EXPECT_EQ(update.planNodePeakBytes, 500);
  EXPECT_EQ(update.planNodeResourcePeakBytes, 500);
  EXPECT_EQ(update.operatorPeakBytes, 500);
  EXPECT_EQ(update.operatorResourcePeakBytes, 500);

  tracker.recordDeallocation(&ownerBOutput);
  tracker.recordDeallocation(&ownerCMain);
  tracker.recordDeallocation(&ownerBSecondOutput);
  tracker.recordDeallocation(&otherQueryMain);
}

TEST(GpuResourcesTest, QueryPeakPersistsUntilQueryRetirement) {
  GpuMemoryAllocationTracker tracker;
  const GpuMemoryOwner firstTask{
      .taskUuid = "first-task-uuid",
      .taskId = "first-task",
      .queryId = "query-1",
      .planNodeId = "plan-a",
      .operatorId = 7,
      .operatorType = "CudfHashJoinBuild"};
  const GpuMemoryOwner secondTask{
      .taskUuid = "second-task-uuid",
      .taskId = "second-task",
      .queryId = "query-1",
      .planNodeId = "plan-a",
      .operatorId = 7,
      .operatorType = "CudfHashJoinBuild"};

  int firstAllocation;
  auto update = tracker.recordAllocation(
      &firstAllocation, 100, GpuMemoryResourceKind::kMain, firstTask);
  EXPECT_EQ(update.queryPeakBytes, 100);
  tracker.recordDeallocation(&firstAllocation);
  tracker.retireTask(firstTask.taskUuid);

  int secondAllocation;
  update = tracker.recordAllocation(
      &secondAllocation, 50, GpuMemoryResourceKind::kMain, secondTask);
  EXPECT_FALSE(update.queryPeakBytes.has_value());
  EXPECT_FALSE(update.queryResourcePeakBytes.has_value());
  EXPECT_FALSE(update.planNodePeakBytes.has_value());
  EXPECT_FALSE(update.planNodeResourcePeakBytes.has_value());
  EXPECT_FALSE(update.operatorPeakBytes.has_value());
  EXPECT_FALSE(update.operatorResourcePeakBytes.has_value());
  tracker.recordDeallocation(&secondAllocation);
  tracker.retireTask(secondTask.taskUuid);

  tracker.retireQuery("query-1");

  int nextQueryAllocation;
  update = tracker.recordAllocation(
      &nextQueryAllocation, 25, GpuMemoryResourceKind::kMain, secondTask);
  EXPECT_EQ(update.queryPeakBytes, 25);
  EXPECT_EQ(update.queryResourcePeakBytes, 25);
  EXPECT_EQ(update.planNodePeakBytes, 25);
  EXPECT_EQ(update.planNodeResourcePeakBytes, 25);
  EXPECT_EQ(update.operatorPeakBytes, 25);
  EXPECT_EQ(update.operatorResourcePeakBytes, 25);
  tracker.recordDeallocation(&nextQueryAllocation);
}

TEST(GpuResourcesTest, PeakUpdatesUseByteRuntimeStats) {
  RecordingRuntimeStatWriter writer;
  const GpuMemoryPeakUpdate mainUpdate{
      .queryPeakBytes = 900,
      .queryResourcePeakBytes = 700,
      .planNodePeakBytes = 500,
      .planNodeResourcePeakBytes = 400,
      .operatorPeakBytes = 300,
      .operatorResourcePeakBytes = 200};

  addGpuMemoryPeakRuntimeStats(
      &writer, GpuMemoryResourceKind::kMain, mainUpdate);

  const std::vector<std::pair<std::string, int64_t>> expected{
      {"gpuQueryPeakLiveBytes", 900},
      {"gpuPlanNodePeakLiveBytes", 500},
      {"gpuOperatorPeakLiveBytes", 300},
      {"gpuQueryMainPeakLiveBytes", 700},
      {"gpuPlanNodeMainPeakLiveBytes", 400},
      {"gpuOperatorMainPeakLiveBytes", 200}};
  ASSERT_EQ(writer.metrics.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(writer.metrics[i].first, expected[i].first);
    EXPECT_EQ(writer.metrics[i].second.value, expected[i].second);
    EXPECT_EQ(writer.metrics[i].second.unit, RuntimeCounter::Unit::kBytes);
  }

  addGpuMemoryPeakRuntimeStats(
      &writer, GpuMemoryResourceKind::kOutput, GpuMemoryPeakUpdate{});
  EXPECT_EQ(writer.metrics.size(), expected.size());

  RecordingRuntimeStatWriter outputWriter;
  const GpuMemoryPeakUpdate outputUpdate{
      .queryPeakBytes = std::nullopt,
      .queryResourcePeakBytes = 70,
      .planNodePeakBytes = std::nullopt,
      .planNodeResourcePeakBytes = 40,
      .operatorPeakBytes = std::nullopt,
      .operatorResourcePeakBytes = 20};
  addGpuMemoryPeakRuntimeStats(
      &outputWriter, GpuMemoryResourceKind::kOutput, outputUpdate);
  const std::vector<std::pair<std::string, int64_t>> expectedOutput{
      {"gpuQueryOutputPeakLiveBytes", 70},
      {"gpuPlanNodeOutputPeakLiveBytes", 40},
      {"gpuOperatorOutputPeakLiveBytes", 20}};
  ASSERT_EQ(outputWriter.metrics.size(), expectedOutput.size());
  for (size_t i = 0; i < expectedOutput.size(); ++i) {
    EXPECT_EQ(outputWriter.metrics[i].first, expectedOutput[i].first);
    EXPECT_EQ(outputWriter.metrics[i].second.value, expectedOutput[i].second);
    EXPECT_EQ(
        outputWriter.metrics[i].second.unit, RuntimeCounter::Unit::kBytes);
  }
}

TEST(GpuResourcesTest, SharedUpstreamHasSeparateResourceCounters) {
  auto mainUpstream = createMemoryResource("cuda", 0);
  auto outputUpstream = mainUpstream;
  auto resources = createGpuMemoryTrackingResources(
      std::move(mainUpstream), std::move(outputUpstream));
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  const auto mainResource = rmm::device_async_resource_ref{resources.main};
  const auto outputResource = rmm::device_async_resource_ref{resources.output};
  {
    rmm::device_buffer mainBuffer{256, rmm::cuda_stream_default, mainResource};
    rmm::device_buffer outputBuffer{
        512, rmm::cuda_stream_default, outputResource};

    const auto snapshot = getGpuMemorySnapshot();
    const auto& main = findResource(snapshot, GpuMemoryResourceKind::kMain);
    EXPECT_EQ(main.currentBytes, 256);
    EXPECT_EQ(main.currentAllocations, 1);
    const auto& output = findResource(snapshot, GpuMemoryResourceKind::kOutput);
    EXPECT_EQ(output.currentBytes, 512);
    EXPECT_EQ(output.currentAllocations, 1);
    ASSERT_EQ(snapshot.allocations.size(), 2);
    EXPECT_NE(snapshot.allocations[0].kind, snapshot.allocations[1].kind);
  }

  const auto snapshot = getGpuMemorySnapshot();
  EXPECT_EQ(
      findResource(snapshot, GpuMemoryResourceKind::kMain).currentBytes, 0);
  EXPECT_EQ(
      findResource(snapshot, GpuMemoryResourceKind::kOutput).currentBytes, 0);
}

TEST(GpuResourcesTest, PreservesAllocationAlignment) {
  auto state = std::make_shared<RecordingResourceState>();
  auto upstream = cuda::mr::any_resource<cuda::mr::device_accessible>{
      RecordingResource{state}};
  auto outputUpstream = upstream;
  auto resources = createGpuMemoryTrackingResources(
      std::move(upstream), std::move(outputUpstream));
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  constexpr std::size_t kAlignment = 4096;
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
  EXPECT_TRUE(snapshot.owners.empty());
  EXPECT_TRUE(snapshot.allocations.empty());
  const auto& main = findResource(snapshot, GpuMemoryResourceKind::kMain);
  EXPECT_EQ(main.currentBytes, 0);
  EXPECT_EQ(main.currentAllocations, 0);
  EXPECT_EQ(main.totalAllocations, 0);

  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_TRUE(state->allocationAlignments.empty());
  EXPECT_TRUE(state->deallocationAlignments.empty());
}

TEST(GpuResourcesTest, TrackedOutputResourceSurvivesUnregister) {
  auto& config = CudfConfig::getInstance();
  const auto previousTrackingEnabled = config.memoryTrackingEnabled;
  config.memoryTrackingEnabled = true;
  auto cleanupGuard = folly::makeGuard([&] {
    if (cudfIsRegistered()) {
      unregisterCudf();
    }
    config.memoryTrackingEnabled = previousTrackingEnabled;
  });

  registerCudf();
  auto outputResource = get_output_mr();
  auto* address = outputResource.allocate(rmm::cuda_stream_default, 64, 256);
  unregisterCudf();

  outputResource.deallocate(rmm::cuda_stream_default, address, 64, 256);
  EXPECT_FALSE(cudfIsRegistered());
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
    }
    while (!workersDone.load(std::memory_order_acquire)) {
      if (!isCoherent(getGpuMemorySnapshot())) {
        coherent.store(false, std::memory_order_release);
        return;
      }
    }
  });

  constexpr int kWorkerCount = 4;
  constexpr int kIterations = 5'000;
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

  EXPECT_TRUE(coherent.load(std::memory_order_acquire));
  EXPECT_TRUE(isCoherent(getGpuMemorySnapshot()));
}

TEST(GpuResourcesTest, ReusedAddressKeepsReplacementAllocation) {
  auto state = std::make_shared<AddressReusingResourceState>();
  auto resources = createGpuMemoryTrackingResources(
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          AddressReusingResource{state}},
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          AddressReusingResource{state}});
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
  ASSERT_EQ(replacementSnapshot.allocations.size(), 1);
  EXPECT_EQ(
      replacementSnapshot.allocations.front().address,
      reinterpret_cast<uintptr_t>(replacementAddress));
  EXPECT_EQ(replacementSnapshot.allocations.front().bytes, 128);
  EXPECT_EQ(
      replacementSnapshot.allocations.front().kind,
      GpuMemoryResourceKind::kOutput);
  EXPECT_EQ(replacementSnapshot.allocations.front().owner, GpuMemoryOwner{});
  EXPECT_TRUE(isCoherent(replacementSnapshot));

  outputResource.deallocate(
      rmm::cuda_stream_default, replacementAddress, 128, 256);
  const auto finalSnapshot = getGpuMemorySnapshot();
  EXPECT_TRUE(isCoherent(finalSnapshot));
  EXPECT_TRUE(finalSnapshot.allocations.empty());
  EXPECT_EQ(
      findResource(finalSnapshot, GpuMemoryResourceKind::kMain)
          .currentAllocations,
      0);
  EXPECT_EQ(
      findResource(finalSnapshot, GpuMemoryResourceKind::kOutput)
          .currentAllocations,
      0);
}

TEST(GpuResourcesTest, RetiredTaskHistoryFollowsLiveAllocations) {
  GpuMemoryAllocationTracker tracker;
  const GpuMemoryOwner liveOwner{
      .taskUuid = "retired-task-uuid",
      .taskId = "shared-task-id",
      .queryId = "",
      .planNodeId = "live-plan",
      .operatorId = 1,
      .operatorType = "LiveOperator"};
  const GpuMemoryOwner historicalOwner{
      .taskUuid = "retired-task-uuid",
      .taskId = "shared-task-id",
      .queryId = "",
      .planNodeId = "historical-plan",
      .operatorId = 2,
      .operatorType = "HistoricalOperator"};
  const GpuMemoryOwner futureOwner{
      .taskUuid = "future-task-uuid",
      .taskId = "shared-task-id",
      .queryId = "",
      .planNodeId = "future-plan",
      .operatorId = 3,
      .operatorType = "FutureOperator"};
  const GpuMemoryOwner otherTaskOwner{
      .taskUuid = "other-task-uuid",
      .taskId = "shared-task-id",
      .queryId = "",
      .planNodeId = "live-plan",
      .operatorId = 1,
      .operatorType = "LiveOperator"};

  int liveAllocation;
  int historicalAllocation;
  int otherTaskAllocation;
  tracker.recordAllocation(
      &liveAllocation, 64, GpuMemoryResourceKind::kMain, liveOwner);
  tracker.recordAllocation(
      &historicalAllocation, 32, GpuMemoryResourceKind::kMain, historicalOwner);
  tracker.recordAllocation(
      &otherTaskAllocation, 48, GpuMemoryResourceKind::kMain, otherTaskOwner);
  tracker.recordDeallocation(&historicalAllocation);

  tracker.retireTask("retired-task-uuid");
  auto snapshot = tracker.snapshot();
  EXPECT_TRUE(hasOwner(snapshot, liveOwner, GpuMemoryResourceKind::kMain));
  EXPECT_FALSE(
      hasOwner(snapshot, historicalOwner, GpuMemoryResourceKind::kMain));
  EXPECT_TRUE(hasOwner(snapshot, otherTaskOwner, GpuMemoryResourceKind::kMain));

  tracker.recordDeallocation(&liveAllocation);
  snapshot = tracker.snapshot();
  EXPECT_FALSE(hasOwner(snapshot, liveOwner, GpuMemoryResourceKind::kMain));
  EXPECT_TRUE(hasOwner(snapshot, otherTaskOwner, GpuMemoryResourceKind::kMain));

  tracker.retireTask("future-task-uuid");
  int futureAllocation;
  tracker.recordAllocation(
      &futureAllocation, 16, GpuMemoryResourceKind::kMain, futureOwner);
  tracker.recordDeallocation(&futureAllocation);
  EXPECT_TRUE(
      hasOwner(tracker.snapshot(), futureOwner, GpuMemoryResourceKind::kMain));

  tracker.recordDeallocation(&otherTaskAllocation);
}

TEST(GpuResourcesTest, AllocationFailureLogsAndRethrows) {
  auto failingState = std::make_shared<RecordingResourceState>();
  failingState->throwOnAllocation = true;
  auto outputState = std::make_shared<RecordingResourceState>();
  auto resources = createGpuMemoryTrackingResources(
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{failingState}},
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          RecordingResource{outputState}});
  auto resetGuard = folly::makeGuard([] { resetGpuMemoryTracking(); });

  LogCapture logs;
  constexpr std::size_t kAlignment = 4096;
  auto resource = rmm::device_async_resource_ref{resources.main};
  EXPECT_THROW(
      resource.allocate(rmm::cuda_stream_default, 1234, kAlignment),
      rmm::out_of_memory);

  {
    std::lock_guard<std::mutex> lock(failingState->mutex);
    ASSERT_EQ(failingState->allocationAlignments.size(), 1);
    EXPECT_EQ(failingState->allocationAlignments.front(), kAlignment);
  }

  const auto captured = logs.captured();
  EXPECT_NE(
      captured.find("GPU_MEMORY_OOM requested_bytes=1234"), std::string::npos);
  EXPECT_NE(captured.find("resource_kind=main"), std::string::npos);
  EXPECT_NE(captured.find("task_uuid=<none>"), std::string::npos);
  EXPECT_NE(captured.find("statistics_current_bytes=0"), std::string::npos);
}

} // namespace facebook::velox::cudf_velox::test
