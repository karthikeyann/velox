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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/FetchScheduler.h"
#include "velox/experimental/cudf/connectors/hive/PinnedBufferPool.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <gmock/gmock.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

constexpr size_t kBufferSize{4'096};

// Lets fills through one at a time or all at once, and counts those inside.
class Gate {
 public:
  // Blocks the calling fill until the gate opens.
  void pass() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++numInside_;
    maxInside_ = std::max(maxInside_, numInside_);
    changed_.notify_all();
    changed_.wait(lock, [&]() { return open_; });
    --numInside_;
  }

  void open() {
    std::lock_guard<std::mutex> lock(mutex_);
    open_ = true;
    changed_.notify_all();
  }

  // Waits until 'numInside' fills are blocked at the gate.
  void waitForInside(size_t numInside) {
    std::unique_lock<std::mutex> lock(mutex_);
    ASSERT_TRUE(changed_.wait_for(lock, std::chrono::seconds(30), [&]() {
      return numInside_ >= numInside;
    }));
  }

  size_t maxInside() {
    std::lock_guard<std::mutex> lock(mutex_);
    return maxInside_;
  }

  size_t numInside() {
    std::lock_guard<std::mutex> lock(mutex_);
    return numInside_;
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool open_{false};
  size_t numInside_{0};
  size_t maxInside_{0};
};

class FetchSchedulerTest : public ::facebook::velox::cudf_velox::exec::test::
                               CudfHiveConnectorTestBase {
 protected:
  using Options = FetchScheduler::Options;
  using FillOrder = FetchScheduler::FillOrder;
  using CopyMode = FetchScheduler::CopyMode;

  // The two ways of waiting for copies, each pinned.
  static constexpr std::array<CopyMode, 2> kCopyModes{
      CopyMode::kWait,
      CopyMode::kQueue,
  };

  static std::string makeContent(size_t size) {
    std::string content(size, '\0');
    for (size_t i = 0; i < size; ++i) {
      content[i] = static_cast<char>((i * 131 + i / 4'096) % 251);
    }
    return content;
  }

  // Options that keep the number of readers at 'numReaders'.
  static Options fixedReaders(size_t numReaders) {
    Options options;
    options.minReaders = numReaders;
    options.maxReaders = numReaders;
    options.initialReaders = numReaders;
    options.copyMode = CopyMode::kWait;
    return options;
  }

  // Default options with the way of waiting for copies pinned.
  static Options withCopyMode(CopyMode copyMode) {
    Options options;
    options.copyMode = copyMode;
    return options;
  }

  static std::unique_ptr<FetchScheduler> makeScheduler(
      const Options& options,
      size_t numBuffers) {
    return std::make_unique<FetchScheduler>(
        std::make_shared<PinnedBufferPool>(
            kBufferSize, numBuffers * kBufferSize),
        options);
  }

  // Returns a fill that serves the bytes of 'content' from 'offset' on.
  static FetchScheduler::Fill fillFrom(
      const std::string& content,
      size_t offset) {
    return [&content, offset](
               uint8_t* buffer, size_t pieceOffset, size_t pieceSize) {
      std::memcpy(buffer, content.data() + offset + pieceOffset, pieceSize);
    };
  }

  static rmm::device_buffer makeDeviceBuffer(size_t size) {
    return rmm::device_buffer(
        size, cudfGlobalStreamPool().get_stream(), get_temp_mr());
  }

  static std::string toHost(const rmm::device_buffer& device) {
    auto stream = cudfGlobalStreamPool().get_stream();
    std::string result(device.size(), '\0');
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        result.data(),
        device.data(),
        device.size(),
        cudaMemcpyDeviceToHost,
        stream.get()));
    stream.sync();
    return result;
  }

  // Waits until 'condition' holds, for a bounded time.
  template <typename Condition>
  static void waitFor(Condition condition) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!condition() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(condition());
  }
};

TEST_F(FetchSchedulerTest, copiesRanges) {
  const auto content = makeContent(100'000);
  for (const auto order : {FillOrder::kAny, FillOrder::kSequential}) {
    for (const auto copyMode : kCopyModes) {
      SCOPED_TRACE(order == FillOrder::kAny ? "any" : "sequential");
      SCOPED_TRACE(copyMode == CopyMode::kWait ? "wait" : "queue");
      auto scheduler = makeScheduler(withCopyMode(copyMode), 2);
      auto group = scheduler->makeGroup();
      uint64_t numBytes{0};
      uint64_t numPieces{0};
      // Empty, within a buffer, exactly one buffer, a multiple of it, and many
      // more buffers than the pool has with a partial last one.
      for (const size_t size : {0, 1, 1'000, 4'096, 8'192, 99'999}) {
        SCOPED_TRACE(size);
        auto device = makeDeviceBuffer(size);
        auto read = scheduler->submit(
            group,
            size,
            static_cast<uint8_t*>(device.data()),
            order,
            fillFrom(content, 1));
        const auto cost = scheduler->wait(*read);
        EXPECT_EQ(cost.bytes, size);
        EXPECT_EQ(toHost(device), content.substr(1, size));
        numBytes += size;
        numPieces += (size + kBufferSize - 1) / kBufferSize;
      }
      const auto stats = scheduler->stats();
      EXPECT_EQ(stats.bytes, numBytes);
      EXPECT_EQ(stats.numPieces, numPieces);
      EXPECT_LE(stats.maxBuffersInUse, 2);
      EXPECT_LE(scheduler->pool()->numAllocated(), 2);
      // The reader of a sequential range waits for its copies either way.
      EXPECT_EQ(
          stats.numQueuedCopies,
          copyMode == CopyMode::kQueue && order == FillOrder::kAny ? numPieces
                                                                   : 0);
      EXPECT_EQ(stats.isCopyQueued, copyMode == CopyMode::kQueue);
      EXPECT_EQ(stats.numCopyModeChanges, 0);
    }
  }
}

// A posted task runs on a thread of the scheduler, also while every reader is
// busy until one is free, and the tasks still waiting run when the scheduler
// is destroyed.
TEST_F(FetchSchedulerTest, post) {
  const auto content = makeContent(kBufferSize);
  auto scheduler = makeScheduler(fixedReaders(1), 4);
  std::promise<std::thread::id> ran;
  scheduler->post([&]() { ran.set_value(std::this_thread::get_id()); });
  EXPECT_NE(ran.get_future().get(), std::this_thread::get_id());

  // The only reader stands at the gate, so the tasks wait.
  Gate gate;
  auto device = makeDeviceBuffer(kBufferSize);
  auto read = scheduler->submit(
      scheduler->makeGroup(),
      kBufferSize,
      static_cast<uint8_t*>(device.data()),
      FillOrder::kAny,
      [&](uint8_t* buffer, size_t offset, size_t size) {
        gate.pass();
        std::memcpy(buffer, content.data() + offset, size);
      });
  gate.waitForInside(1);
  std::atomic_int32_t numRan{0};
  for (int i = 0; i < 3; ++i) {
    scheduler->post([&]() { ++numRan; });
  }
  EXPECT_EQ(numRan, 0);
  gate.open();
  scheduler->wait(*read);
  scheduler.reset();
  EXPECT_EQ(numRan, 3);
}

// The number of fills that run at once is the scheduler's, whichever thread
// submits and however many ranges are queued.
TEST_F(FetchSchedulerTest, admission) {
  const auto content = makeContent(64 * kBufferSize);
  auto scheduler = makeScheduler(fixedReaders(2), 16);
  Gate gate;
  const auto fill = [&](uint8_t* buffer, size_t offset, size_t size) {
    gate.pass();
    std::memcpy(buffer, content.data() + offset, size);
  };

  auto device = makeDeviceBuffer(content.size());
  auto* dst = static_cast<uint8_t*>(device.data());
  const auto half = content.size() / 2;
  auto group = scheduler->makeGroup();
  auto first = scheduler->submit(group, half, dst, FillOrder::kAny, fill);
  // The second half is submitted from another thread, under another group.
  std::shared_ptr<FetchScheduler::Read> second;
  std::thread([&]() {
    second = scheduler->submit(
        scheduler->makeGroup(),
        half,
        dst + half,
        FillOrder::kAny,
        [&](uint8_t* buffer, size_t offset, size_t size) {
          fill(buffer, half + offset, size);
        });
  }).join();

  gate.waitForInside(2);
  // More fills would have reached the gate by now if they were allowed to.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(gate.maxInside(), 2);
  gate.open();
  scheduler->wait(*first);
  scheduler->wait(*second);

  EXPECT_EQ(toHost(device), content);
  const auto stats = scheduler->stats();
  EXPECT_EQ(stats.maxRunningReaders, 2);
  EXPECT_EQ(stats.numReaders, 2);
  // A reader holds one staging buffer.
  EXPECT_LE(stats.maxBuffersInUse, 2);
}

// Ranges are served group by group in the order the groups asked, except
// that a group somebody waits for goes first.
TEST_F(FetchSchedulerTest, ordering) {
  const auto content = makeContent(kBufferSize);
  for (const bool waitForLast : {false, true}) {
    SCOPED_TRACE(waitForLast);
    auto scheduler = makeScheduler(fixedReaders(1), 4);
    Gate gate;
    std::mutex mutex;
    std::vector<std::string> order;
    const auto fillAs = [&](const std::string& name) {
      return [&, name](uint8_t* buffer, size_t offset, size_t size) {
        {
          std::lock_guard<std::mutex> lock(mutex);
          order.push_back(name);
        }
        std::memcpy(buffer, content.data() + offset, size);
      };
    };

    auto device = makeDeviceBuffer(8 * kBufferSize);
    auto* dst = static_cast<uint8_t*>(device.data());
    std::vector<std::shared_ptr<FetchScheduler::Read>> reads;
    // Keeps the only reader busy while the other ranges queue up.
    reads.push_back(scheduler->submit(
        scheduler->makeGroup(),
        kBufferSize,
        dst,
        FillOrder::kAny,
        [&](uint8_t* buffer, size_t offset, size_t size) {
          gate.pass();
          std::memcpy(buffer, content.data() + offset, size);
        }));
    gate.waitForInside(1);

    auto groupA = scheduler->makeGroup();
    auto groupB = scheduler->makeGroup();
    size_t next{1};
    const auto submit = [&](const auto& group, const std::string& name) {
      reads.push_back(scheduler->submit(
          group,
          kBufferSize,
          dst + next++ * kBufferSize,
          FillOrder::kAny,
          fillAs(name)));
    };
    // The groups alternate, as the ranges of two splits do when two drivers
    // submit at the same time.
    submit(groupA, "a1");
    submit(groupB, "b1");
    submit(groupA, "a2");
    submit(groupB, "b2");

    std::thread waiter;
    if (waitForLast) {
      waiter = std::thread([&]() { scheduler->wait(*reads.back()); });
      waitFor([&]() { return scheduler->stats().numWaiters == 1; });
    }
    gate.open();
    // Waiting for a range puts its group first, so the ranges are collected
    // only when all of them have been served.
    waitFor([&]() {
      std::lock_guard<std::mutex> lock(mutex);
      return order.size() == 4;
    });
    if (waiter.joinable()) {
      waiter.join();
    }
    for (auto& read : reads) {
      scheduler->wait(*read);
    }
    if (waitForLast) {
      EXPECT_THAT(order, testing::ElementsAre("b1", "b2", "a1", "a2"));
    } else {
      EXPECT_THAT(order, testing::ElementsAre("a1", "a2", "b1", "b2"));
    }
  }
}

// Among the groups that consumers wait for, the one with the fewest bytes
// left is served first, whichever asked first.
TEST_F(FetchSchedulerTest, shortestWaitedGroupFirst) {
  const auto content = makeContent(kBufferSize);
  auto scheduler = makeScheduler(fixedReaders(1), 4);
  Gate gate;
  std::mutex mutex;
  std::vector<std::string> order;
  const auto fillAs = [&](const std::string& name) {
    return [&, name](uint8_t* buffer, size_t offset, size_t size) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        order.push_back(name);
      }
      std::memcpy(buffer, content.data() + offset, size);
    };
  };
  auto device = makeDeviceBuffer(8 * kBufferSize);
  auto* dst = static_cast<uint8_t*>(device.data());
  // Keeps the only reader busy while the other ranges queue up.
  auto first = scheduler->submit(
      scheduler->makeGroup(),
      kBufferSize,
      dst,
      FillOrder::kAny,
      [&](uint8_t* buffer, size_t offset, size_t size) {
        gate.pass();
        std::memcpy(buffer, content.data() + offset, size);
      });
  gate.waitForInside(1);

  auto large = scheduler->makeGroup();
  auto small = scheduler->makeGroup();
  std::vector<std::shared_ptr<FetchScheduler::Read>> reads;
  size_t next{1};
  const auto submit = [&](const auto& group, const std::string& name) {
    reads.push_back(scheduler->submit(
        group,
        kBufferSize,
        dst + next++ * kBufferSize,
        FillOrder::kAny,
        fillAs(name)));
  };
  submit(large, "large1");
  submit(large, "large2");
  submit(large, "large3");
  submit(small, "small1");
  submit(small, "small2");
  std::thread largeWaiter([&]() { scheduler->wait(*reads[2]); });
  std::thread smallWaiter([&]() { scheduler->wait(*reads[4]); });
  waitFor([&]() { return scheduler->stats().numWaiters == 2; });
  gate.open();
  largeWaiter.join();
  smallWaiter.join();
  scheduler->wait(*first);
  for (auto& read : reads) {
    scheduler->wait(*read);
  }
  // Once the small group is down to one piece the large one cannot overtake
  // it; after it, the large group is served in order.
  EXPECT_THAT(
      order,
      testing::ElementsAre("small1", "small2", "large1", "large2", "large3"));
}

TEST_F(FetchSchedulerTest, fillFailure) {
  const auto content = makeContent(32 * kBufferSize);
  for (const auto order : {FillOrder::kAny, FillOrder::kSequential}) {
    for (const auto copyMode : kCopyModes) {
      SCOPED_TRACE(order == FillOrder::kAny ? "any" : "sequential");
      SCOPED_TRACE(copyMode == CopyMode::kWait ? "wait" : "queue");
      auto scheduler = makeScheduler(withCopyMode(copyMode), 2);
      auto group = scheduler->makeGroup();
      auto device = makeDeviceBuffer(content.size());
      auto* dst = static_cast<uint8_t*>(device.data());
      std::atomic_int32_t numFills{0};

      auto failing = scheduler->submit(
          group,
          content.size(),
          dst,
          order,
          [&](uint8_t* buffer, size_t offset, size_t size) {
            ++numFills;
            if (offset >= 4 * kBufferSize) {
              VELOX_FAIL("Injected fill failure at offset: {}", offset);
            }
            std::memcpy(buffer, content.data() + offset, size);
          });
      VELOX_ASSERT_THROW(
          scheduler->wait(*failing), "Injected fill failure at offset");
      // The pieces after the failure were dropped.
      EXPECT_LT(numFills, 32);
      const auto numFillsAtFailure = numFills.load();

      // The failed range gave its staging buffers back: the pool has only two.
      auto read = scheduler->submit(
          group, content.size(), dst, order, fillFrom(content, 0));
      scheduler->wait(*read);
      EXPECT_EQ(toHost(device), content);
      EXPECT_EQ(numFills, numFillsAtFailure);
    }
  }
}

// Cancelling drops the pieces no reader has started and returns only when
// the piece in flight is done with the destination.
TEST_F(FetchSchedulerTest, cancel) {
  const auto content = makeContent(16 * kBufferSize);
  for (const auto order : {FillOrder::kAny, FillOrder::kSequential}) {
    for (const auto copyMode : kCopyModes) {
      SCOPED_TRACE(order == FillOrder::kAny ? "any" : "sequential");
      SCOPED_TRACE(copyMode == CopyMode::kWait ? "wait" : "queue");
      auto options = fixedReaders(1);
      options.copyMode = copyMode;
      auto scheduler = makeScheduler(options, 4);
      Gate gate;
      std::atomic_int32_t numFills{0};
      auto group = scheduler->makeGroup();
      auto device = makeDeviceBuffer(content.size());
      auto* dst = static_cast<uint8_t*>(device.data());

      auto started = scheduler->submit(
          group,
          content.size(),
          dst,
          order,
          [&](uint8_t* buffer, size_t offset, size_t size) {
            ++numFills;
            gate.pass();
            std::memcpy(buffer, content.data() + offset, size);
          });
      auto queued = scheduler->submit(
          group, content.size(), dst, order, [&](uint8_t*, size_t, size_t) {
            ++numFills;
          });
      gate.waitForInside(1);

      // Nothing of the queued range has started, so there is nothing to wait
      // for.
      scheduler->cancel(*queued);

      std::atomic_bool cancelled{false};
      std::thread canceller([&]() {
        scheduler->cancel(*started);
        cancelled = true;
      });
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      EXPECT_FALSE(cancelled);
      gate.open();
      canceller.join();
      EXPECT_EQ(numFills, 1);

      // The first piece was copied before the cancellation returned.
      EXPECT_EQ(
          toHost(device).substr(0, kBufferSize),
          content.substr(0, kBufferSize));
      // A cancelled range can be cancelled again, and the scheduler goes on.
      scheduler->cancel(*started);
      auto read = scheduler->submit(
          group, content.size(), dst, order, fillFrom(content, 0));
      scheduler->wait(*read);
      EXPECT_EQ(toHost(device), content);
    }
  }
}

// Destroying the scheduler waits for the piece in flight and fails the
// ranges that are still queued instead of running them.
TEST_F(FetchSchedulerTest, teardownWithReadsInFlight) {
  const auto content = makeContent(4 * kBufferSize);
  for (const auto order : {FillOrder::kAny, FillOrder::kSequential}) {
    for (const auto copyMode : kCopyModes) {
      SCOPED_TRACE(order == FillOrder::kAny ? "any" : "sequential");
      SCOPED_TRACE(copyMode == CopyMode::kWait ? "wait" : "queue");
      auto options = fixedReaders(1);
      options.copyMode = copyMode;
      auto scheduler = makeScheduler(options, 4);
      Gate gate;
      std::atomic_int32_t numFills{0};
      auto device = makeDeviceBuffer(content.size());
      auto* dst = static_cast<uint8_t*>(device.data());
      auto group = scheduler->makeGroup();

      auto inFlight = scheduler->submit(
          group,
          kBufferSize,
          dst,
          order,
          [&](uint8_t* buffer, size_t offset, size_t size) {
            ++numFills;
            gate.pass();
            std::memcpy(buffer, content.data() + offset, size);
          });
      auto queued = scheduler->submit(
          group, content.size(), dst, order, [&](uint8_t*, size_t, size_t) {
            ++numFills;
          });
      gate.waitForInside(1);

      std::atomic_bool destroyed{false};
      std::thread destroyer([&]() {
        scheduler.reset();
        destroyed = true;
      });
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      EXPECT_FALSE(destroyed);
      gate.open();
      destroyer.join();

      EXPECT_EQ(numFills, 1);
      EXPECT_EQ(
          toHost(device).substr(0, kBufferSize),
          content.substr(0, kBufferSize));
    }
  }
}

TEST_F(FetchSchedulerTest, concurrentSubmitters) {
  const auto content = makeContent(300'000);
  Options options;
  options.maxReaders = 4;
  options.controlPeriod = std::chrono::microseconds(200);
  auto scheduler = makeScheduler(options, 4);
  constexpr size_t kNumThreads{8};
  constexpr size_t kNumRanges{40};
  std::atomic_int32_t numMismatches{0};
  std::atomic_uint64_t numBytes{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (size_t thread = 0; thread < kNumThreads; ++thread) {
    threads.emplace_back([&, thread]() {
      auto group = scheduler->makeGroup();
      for (size_t i = 0; i < kNumRanges; ++i) {
        const auto offset = (thread * 7'919 + i * 104'729) % 200'000;
        const auto size = 1 + (thread * 31 + i * 9'973) % 90'000;
        const auto order =
            (thread + i) % 2 == 0 ? FillOrder::kAny : FillOrder::kSequential;
        auto device = makeDeviceBuffer(size);
        auto read = scheduler->submit(
            group,
            size,
            static_cast<uint8_t*>(device.data()),
            order,
            fillFrom(content, offset));
        // Every third range is cancelled instead of waited for.
        if (i % 3 == 2) {
          scheduler->cancel(*read);
          continue;
        }
        scheduler->wait(*read);
        numBytes += size;
        if (toHost(device) != content.substr(offset, size)) {
          ++numMismatches;
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(numMismatches, 0);
  const auto stats = scheduler->stats();
  EXPECT_GE(stats.bytes, numBytes);
  EXPECT_LE(stats.maxRunningReaders, 4);
  EXPECT_LE(stats.maxBuffersInUse, 4);
  EXPECT_LE(scheduler->pool()->numAllocated(), 4);
}

// One measurement of 'bytesPerSecond' over a millisecond.
FetchScheduler::ControlWindow measured(double bytesPerSecond) {
  return {
      .saturatedNanos = 1'000'000,
      .bytes = static_cast<uint64_t>(bytesPerSecond / 1'000)};
}

TEST_F(FetchSchedulerTest, nextControlState) {
  using State = FetchScheduler::ControlState;
  using Phase = State::Phase;
  Options options;
  options.minReaders = 2;
  options.maxReaders = 20;
  options.numHoldPeriods = 2;
  options.copyMode = CopyMode::kWait;
  const auto holding = [](size_t numReaders, double baseRate) {
    return State{
        .numReaders = numReaders,
        .numHeldReaders = numReaders,
        .baseRate = baseRate};
  };
  const auto trying = [](Phase phase, size_t numHeld, size_t numReaders) {
    return State{
        .numReaders = numReaders,
        .numHeldReaders = numHeld,
        .phase = phase,
        .baseRate = 1'000e6};
  };

  // A measurement of 'bytesPerSecond' during which the readers waited for
  // copies half of their time.
  const auto copyLimited = [](double bytesPerSecond) {
    auto window = measured(bytesPerSecond);
    window.readerBusyNanos = 1'000;
    window.copyWaitNanos = 500;
    return window;
  };

  // Nothing measured: nothing changes.
  auto next = FetchScheduler::nextControlState(options, holding(4, 0), {});
  EXPECT_EQ(next.numReaders, 4);
  EXPECT_EQ(next.phase, Phase::kHold);
  EXPECT_EQ(next.baseRate, 0);

  // While it holds, the controller averages the rate and counts down.
  auto state = holding(4, 1'000e6);
  state.numHoldsLeft = 2;
  next = FetchScheduler::nextControlState(options, state, measured(2'000e6));
  EXPECT_EQ(next.numReaders, 4);
  EXPECT_EQ(next.phase, Phase::kHold);
  EXPECT_EQ(next.numHoldsLeft, 1);
  EXPECT_DOUBLE_EQ(next.baseRate, 1'250e6);

  // After the hold it tries a quarter more, at least one.
  next = FetchScheduler::nextControlState(
      options, holding(4, 1'000e6), measured(1'000e6));
  EXPECT_EQ(next.numReaders, 5);
  EXPECT_EQ(next.numHeldReaders, 4);
  EXPECT_EQ(next.phase, Phase::kTryMore);
  next = FetchScheduler::nextControlState(
      options, holding(12, 1'000e6), measured(1'000e6));
  EXPECT_EQ(next.numReaders, 15);
  EXPECT_EQ(next.numHeldReaders, 12);

  // Three more readers on top of twelve are kept when the rate rises by an
  // eighth, half of their share of a quarter, and more are tried right away.
  state = trying(Phase::kTryMore, 12, 15);
  next = FetchScheduler::nextControlState(options, state, measured(1'130e6));
  EXPECT_EQ(next.numHeldReaders, 15);
  EXPECT_EQ(next.numReaders, 18);
  EXPECT_EQ(next.phase, Phase::kTryMore);
  EXPECT_DOUBLE_EQ(next.baseRate, 1'130e6);
  // Otherwise the controller goes back to twelve and holds.
  next = FetchScheduler::nextControlState(options, state, measured(1'120e6));
  EXPECT_EQ(next.numReaders, 12);
  EXPECT_EQ(next.numHeldReaders, 12);
  EXPECT_EQ(next.phase, Phase::kHold);
  EXPECT_EQ(next.numHoldsLeft, 2);
  EXPECT_DOUBLE_EQ(next.baseRate, 1'000e6);

  // Fewer readers are tried only while the readers wait for copies half of
  // their time: an eighth fewer, at least one. A rate that does not depend
  // on the number of readers is no reason to give readers up.
  state = next;
  state.numHoldsLeft = 0;
  next = FetchScheduler::nextControlState(options, state, measured(1'000e6));
  EXPECT_EQ(next.numReaders, 15);
  EXPECT_EQ(next.phase, Phase::kTryMore);
  next = FetchScheduler::nextControlState(options, state, copyLimited(1'000e6));
  EXPECT_EQ(next.numReaders, 11);
  EXPECT_EQ(next.numHeldReaders, 12);
  EXPECT_EQ(next.phase, Phase::kTryFewer);
  next = FetchScheduler::nextControlState(
      options, holding(16, 1'000e6), copyLimited(1'000e6));
  EXPECT_EQ(next.numReaders, 14);

  // Fourteen of sixteen readers are kept when the rate falls by less than a
  // thirty-second, a quarter of the share of the two left out. Fewer still
  // are tried right away while the copies stay the limit.
  state = trying(Phase::kTryFewer, 16, 14);
  next = FetchScheduler::nextControlState(options, state, copyLimited(970e6));
  EXPECT_EQ(next.numHeldReaders, 14);
  EXPECT_EQ(next.numReaders, 13);
  EXPECT_EQ(next.phase, Phase::kTryFewer);
  EXPECT_DOUBLE_EQ(next.baseRate, 970e6);
  next = FetchScheduler::nextControlState(options, state, measured(970e6));
  EXPECT_EQ(next.numHeldReaders, 14);
  EXPECT_EQ(next.numReaders, 14);
  EXPECT_EQ(next.phase, Phase::kHold);
  EXPECT_DOUBLE_EQ(next.baseRate, 970e6);
  // Otherwise the controller goes back to sixteen and holds.
  next = FetchScheduler::nextControlState(options, state, copyLimited(960e6));
  EXPECT_EQ(next.numReaders, 16);
  EXPECT_EQ(next.phase, Phase::kHold);

  // The bounds end a run of kept trials and limit a trial.
  next = FetchScheduler::nextControlState(
      options, holding(18, 1'000e6), measured(1'000e6));
  EXPECT_EQ(next.numReaders, 20);
  state = trying(Phase::kTryMore, 18, 20);
  next = FetchScheduler::nextControlState(options, state, measured(2'000e6));
  EXPECT_EQ(next.numReaders, 20);
  EXPECT_EQ(next.phase, Phase::kHold);
  state = trying(Phase::kTryFewer, 3, 2);
  next = FetchScheduler::nextControlState(options, state, measured(1'000e6));
  EXPECT_EQ(next.numReaders, 2);
  EXPECT_EQ(next.phase, Phase::kHold);
  // At a bound nothing is tried in its direction.
  next = FetchScheduler::nextControlState(
      options, holding(20, 1'000e6), measured(1'000e6));
  EXPECT_EQ(next.numReaders, 20);
  EXPECT_EQ(next.phase, Phase::kHold);
  next = FetchScheduler::nextControlState(
      options, holding(2, 1'000e6), copyLimited(1'000e6));
  EXPECT_EQ(next.numReaders, 2);
  EXPECT_EQ(next.phase, Phase::kHold);

  // Readers that wait for copies half of their time are not given company:
  // a trial of more is not kept whatever the rate.
  next = FetchScheduler::nextControlState(
      options, trying(Phase::kTryMore, 8, 10), copyLimited(2'000e6));
  EXPECT_EQ(next.numReaders, 8);
  EXPECT_EQ(next.phase, Phase::kHold);
  auto almostLimited = copyLimited(2'000e6);
  almostLimited.copyWaitNanos = 499;
  next = FetchScheduler::nextControlState(
      options, trying(Phase::kTryMore, 8, 10), almostLimited);
  EXPECT_EQ(next.numReaders, 12);
  EXPECT_EQ(next.phase, Phase::kTryMore);

  // Readers kept from their next piece for lack of a staging buffer half
  // of the time have more copies queued than the device takes: more readers
  // are not tried, and none are given up.
  auto bufferLimited = measured(1'000e6);
  bufferLimited.bufferLimitedNanos = bufferLimited.saturatedNanos / 2;
  next = FetchScheduler::nextControlState(
      options, holding(8, 1'000e6), bufferLimited);
  EXPECT_EQ(next.numReaders, 8);
  EXPECT_EQ(next.phase, Phase::kHold);
  --bufferLimited.bufferLimitedNanos;
  next = FetchScheduler::nextControlState(
      options, holding(8, 1'000e6), bufferLimited);
  EXPECT_EQ(next.numReaders, 10);
  EXPECT_EQ(next.phase, Phase::kTryMore);

  // With nothing delivered at all, fewer readers are never kept.
  state = trying(Phase::kTryFewer, 4, 3);
  state.baseRate = 0;
  next = FetchScheduler::nextControlState(options, state, measured(0));
  EXPECT_EQ(next.numReaders, 4);
  EXPECT_EQ(next.phase, Phase::kHold);

  VELOX_ASSERT_THROW(
      FetchScheduler::nextControlState(options, holding(0, 0), {}),
      "(0 vs. 0)");
}

// From below the controller settles near the fewest readers that deliver
// what the copies can take. From above it comes down as far as the readers
// wait for copies half of their time, and no further. It ends at the upper
// bound when the copies are never the limit.
TEST_F(FetchSchedulerTest, nextControlStateConverges) {
  using State = FetchScheduler::ControlState;
  Options options;
  options.minReaders = 1;
  options.maxReaders = 32;
  options.numHoldPeriods = 3;
  options.copyMode = CopyMode::kWait;
  // What the controller measures with 'numReaders' readers that fill
  // 'readRate' bytes per second each, in front of copies that run one after
  // the other at 'copyRate', when ranges are always queued: a reader fills
  // its buffer and then waits for the copies ahead of its own.
  const auto model = [](size_t numReaders, double readRate, double copyRate) {
    const double readTime = 1 / readRate;
    const double copyTime = 1 / copyRate;
    const double cycle = std::max(readTime + copyTime, numReaders * copyTime);
    auto window = measured(numReaders / cycle);
    window.readerBusyNanos = window.saturatedNanos * numReaders;
    window.copyWaitNanos = static_cast<uint64_t>(
        window.readerBusyNanos * (cycle - readTime) / cycle);
    return window;
  };
  // Returns the number of readers the controller holds after many
  // measurements.
  const auto settle = [&](size_t numReaders, double readRate, double copyRate) {
    State state{.numReaders = numReaders, .numHeldReaders = numReaders};
    std::vector<size_t> held;
    for (int i = 0; i < 400; ++i) {
      state = FetchScheduler::nextControlState(
          options, state, model(state.numReaders, readRate, copyRate));
      if (i >= 300 && state.phase == State::Phase::kHold) {
        held.push_back(state.numReaders);
      }
    }
    EXPECT_FALSE(held.empty());
    // It holds one number in the end.
    EXPECT_THAT(held, testing::Each(held.front()));
    return held.front();
  };

  // Readers of 4 GB/s in front of a 23 GB/s link: seven keep it busy, and
  // from twelve on they wait for copies half of their time.
  EXPECT_EQ(settle(1, 4e9, 23e9), 7);
  EXPECT_EQ(settle(8, 4e9, 23e9), 8);
  EXPECT_EQ(settle(32, 4e9, 23e9), 11);
  // Readers of 15 GB/s in front of a 310 GB/s link: twenty-two keep it busy,
  // and the controller, which moves several readers at a time there, stays
  // within one of that. Thirty-two do not wait half of their time yet.
  EXPECT_THAT(
      settle(8, 15e9, 310e9), testing::AllOf(testing::Ge(21), testing::Le(23)));
  EXPECT_EQ(settle(32, 15e9, 310e9), 32);
  // Readers that never fill the link.
  EXPECT_EQ(settle(8, 4e9, 300e9), 32);
}

// With nothing pinned, the controller measures the other way of waiting for
// copies and the way it came from in turn, keeps the other way when it
// delivered more each time, and otherwise tries it a quarter as often.
TEST_F(FetchSchedulerTest, nextControlStateCopyMode) {
  using State = FetchScheduler::ControlState;
  using Phase = State::Phase;
  Options options;
  options.minReaders = 2;
  options.maxReaders = 20;
  options.numHoldPeriods = 2;
  const auto step = [&](const State& state, double bytesPerSecond) {
    return FetchScheduler::nextControlState(
        options, state, measured(bytesPerSecond));
  };
  for (const bool isCopyQueued : {false, true}) {
    SCOPED_TRACE(isCopyQueued);
    const State held{
        .numReaders = 8,
        .numHeldReaders = 8,
        .baseRate = 900e6,
        .isCopyQueued = isCopyQueued};
    // Checks that 'state' went back to the way the trial came from, holds,
    // and has the next such trial further away.
    const auto expectNotKept = [&](const State& state) {
      EXPECT_EQ(state.phase, Phase::kHold);
      EXPECT_EQ(state.isCopyQueued, isCopyQueued);
      EXPECT_EQ(state.numReaders, 8);
      EXPECT_EQ(state.numHoldsLeft, 2);
      EXPECT_EQ(state.copyTrialInterval, 4);
      EXPECT_EQ(state.numTrialsToCopyTrial, 4);
    };

    // The first trial after a hold is of the other way, with the same
    // readers. What it has to beat is the latest measurement, not the
    // average, which still knows how the scan started.
    const auto trial = step(held, 1'000e6);
    EXPECT_EQ(trial.phase, Phase::kTryOtherCopy);
    EXPECT_EQ(trial.isCopyQueued, !isCopyQueued);
    EXPECT_EQ(trial.numReaders, 8);
    EXPECT_EQ(trial.numHeldReaders, 8);

    // A trial that does not deliver more ends there.
    auto next = step(trial, 1'020e6);
    expectNotKept(next);
    // Until the next one, the trials are of numbers of readers.
    next.numHoldsLeft = 0;
    next = step(next, 1'000e6);
    EXPECT_EQ(next.phase, Phase::kTryMore);
    EXPECT_EQ(next.isCopyQueued, isCopyQueued);
    EXPECT_EQ(next.numTrialsToCopyTrial, 3);

    // A trial that delivers more is followed by a second measurement of the
    // way it came from.
    const auto check = step(trial, 1'040e6);
    EXPECT_EQ(check.phase, Phase::kCheckCopyTrial);
    EXPECT_EQ(check.isCopyQueued, isCopyQueued);
    EXPECT_EQ(check.numReaders, 8);
    // The trial is over when that delivers as much: the scan got faster.
    expectNotKept(step(check, 1'020e6));

    // Otherwise the other way is measured once more.
    const auto confirm = step(check, 1'000e6);
    EXPECT_EQ(confirm.phase, Phase::kConfirmOtherCopy);
    EXPECT_EQ(confirm.isCopyQueued, !isCopyQueued);
    EXPECT_EQ(confirm.numReaders, 8);
    expectNotKept(step(confirm, 1'020e6));
    // It is kept when it delivers more again.
    next = step(confirm, 1'050e6);
    EXPECT_EQ(next.phase, Phase::kHold);
    EXPECT_EQ(next.isCopyQueued, !isCopyQueued);
    EXPECT_EQ(next.numReaders, 8);
    EXPECT_EQ(next.numHoldsLeft, 2);
    EXPECT_DOUBLE_EQ(next.baseRate, 1'050e6);
    EXPECT_EQ(next.copyTrialInterval, 1);
    EXPECT_EQ(next.numTrialsToCopyTrial, 1);

    // The distance between such trials is bounded.
    auto longest = trial;
    longest.copyTrialInterval = FetchScheduler::kMaxCopyTrialInterval;
    next = step(longest, 1e6);
    EXPECT_EQ(next.isCopyQueued, isCopyQueued);
    EXPECT_EQ(next.copyTrialInterval, FetchScheduler::kMaxCopyTrialInterval);
    EXPECT_EQ(next.numTrialsToCopyTrial, FetchScheduler::kMaxCopyTrialInterval);

    // A pinned way of waiting is never put on trial.
    auto pinned = options;
    pinned.copyMode = isCopyQueued ? CopyMode::kQueue : CopyMode::kWait;
    next = FetchScheduler::nextControlState(pinned, held, measured(1'000e6));
    EXPECT_EQ(next.phase, Phase::kTryMore);
    EXPECT_EQ(next.isCopyQueued, isCopyQueued);
  }
}

// Whichever way it starts, the controller ends with readers that wait for
// their copies on a machine where fills slow the copies that run meanwhile,
// and with readers that go on where waiting only keeps them idle.
TEST_F(FetchSchedulerTest, nextControlStateSettlesCopyMode) {
  using State = FetchScheduler::ControlState;
  Options options;
  options.minReaders = 1;
  options.maxReaders = 32;
  options.numHoldPeriods = 3;
  // What the controller measures with 'numReaders' readers that fill
  // 'readRate' bytes per second each when ranges are always queued. Readers
  // that wait fill a buffer and then wait for the copies ahead of their own,
  // which run one after the other at 'waitedCopyRate'. Readers that go on
  // never wait for a copy, only for a buffer, and their copies run at
  // 'queuedCopyRate'.
  const auto model = [](const State& state,
                        double readRate,
                        double waitedCopyRate,
                        double queuedCopyRate) {
    const auto numReaders = state.numReaders;
    if (state.isCopyQueued) {
      const double filled = numReaders * readRate;
      auto window = measured(std::min(filled, queuedCopyRate));
      window.readerBusyNanos = window.saturatedNanos * numReaders;
      // Readers that fill more than the copies take run out of buffers.
      if (filled > queuedCopyRate) {
        window.bufferLimitedNanos = static_cast<uint64_t>(
            window.saturatedNanos * (1 - queuedCopyRate / filled));
      }
      return window;
    }
    const double readTime = 1 / readRate;
    const double copyTime = 1 / waitedCopyRate;
    const double cycle = std::max(readTime + copyTime, numReaders * copyTime);
    auto window = measured(numReaders / cycle);
    window.readerBusyNanos = window.saturatedNanos * numReaders;
    window.copyWaitNanos = static_cast<uint64_t>(
        window.readerBusyNanos * (cycle - readTime) / cycle);
    return window;
  };
  // Returns the share of the measurements that readers went on in, among
  // those of the second thousand that were no trials.
  const auto queuedShare = [&](bool startsQueued,
                               double readRate,
                               double waitedCopyRate,
                               double queuedCopyRate) {
    State state{
        .numReaders = 16, .numHeldReaders = 16, .isCopyQueued = startsQueued};
    size_t numHeld{0};
    size_t numQueued{0};
    for (int i = 0; i < 2'000; ++i) {
      state = FetchScheduler::nextControlState(
          options,
          state,
          model(state, readRate, waitedCopyRate, queuedCopyRate));
      if (i >= 1'000 && state.phase == State::Phase::kHold) {
        ++numHeld;
        numQueued += state.isCopyQueued ? 1 : 0;
      }
    }
    EXPECT_GT(numHeld, 500);
    return static_cast<double>(numQueued) / numHeld;
  };
  for (const bool startsQueued : {false, true}) {
    SCOPED_TRACE(startsQueued);
    // Readers of 4 GB/s in front of a link that takes 23 GB/s from readers
    // that wait and 17 GB/s while buffers are being filled.
    EXPECT_EQ(queuedShare(startsQueued, 4e9, 23e9, 17e9), 0);
    // Readers of 9 GB/s whose wait for a copy holds the link to 30 GB/s,
    // where it takes 54 GB/s from readers that go on.
    EXPECT_EQ(queuedShare(startsQueued, 9e9, 30e9, 54e9), 1);
    // No difference: it stays as it started.
    EXPECT_EQ(
        queuedShare(startsQueued, 4e9, 300e9, 300e9), startsQueued ? 1 : 0);
  }
}

// While ranges wait for a reader and nothing is delivered, the controller
// adds readers up to its bound. The fills stand at a gate, so the waiting is
// a state and not a matter of timing.
TEST_F(FetchSchedulerTest, raisesReadersWhileRangesWait) {
  const auto content = makeContent(kBufferSize);
  Options options;
  options.minReaders = 1;
  options.maxReaders = 4;
  options.initialReaders = 1;
  options.copyMode = CopyMode::kWait;
  options.controlPeriod = std::chrono::microseconds(100);
  // No further trial once the bound is reached.
  options.numHoldPeriods = 1'000'000;
  auto scheduler = makeScheduler(options, 16);
  Gate gate;
  constexpr size_t kNumPieces{32};
  auto device = makeDeviceBuffer(kNumPieces * kBufferSize);
  auto read = scheduler->submit(
      scheduler->makeGroup(),
      kNumPieces * kBufferSize,
      static_cast<uint8_t*>(device.data()),
      FillOrder::kAny,
      [&](uint8_t* buffer, size_t /*offset*/, size_t size) {
        gate.pass();
        std::memcpy(buffer, content.data(), size);
      });
  // Reading the counters lets the controller look at the time that passed.
  waitFor([&]() {
    scheduler->stats();
    return gate.numInside() == 4;
  });
  auto stats = scheduler->stats();
  EXPECT_EQ(stats.numReaders, 4);
  EXPECT_EQ(stats.numRaises, 3);
  EXPECT_GT(stats.saturatedNanos, 0);
  EXPECT_EQ(stats.numLowers, 0);
  EXPECT_EQ(gate.maxInside(), 4);

  gate.open();
  const auto cost = scheduler->wait(*read);
  EXPECT_EQ(cost.bytes, kNumPieces * kBufferSize);
  EXPECT_EQ(scheduler->stats().maxRunningReaders, 4);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
