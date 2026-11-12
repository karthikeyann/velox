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
    SCOPED_TRACE(order == FillOrder::kAny ? "any" : "sequential");
    auto scheduler = makeScheduler(Options{}, 2);
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
  }
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
  // Two readers get one buffer to fill and one being copied each, plus one
  // per copy stream.
  EXPECT_LE(stats.maxBuffersInUse, 6);
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

TEST_F(FetchSchedulerTest, fillFailure) {
  const auto content = makeContent(32 * kBufferSize);
  for (const auto order : {FillOrder::kAny, FillOrder::kSequential}) {
    SCOPED_TRACE(order == FillOrder::kAny ? "any" : "sequential");
    auto scheduler = makeScheduler(Options{}, 2);
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

// Cancelling drops the pieces no reader has started and returns only when
// the piece in flight is done with the destination.
TEST_F(FetchSchedulerTest, cancel) {
  const auto content = makeContent(16 * kBufferSize);
  for (const auto order : {FillOrder::kAny, FillOrder::kSequential}) {
    SCOPED_TRACE(order == FillOrder::kAny ? "any" : "sequential");
    auto scheduler = makeScheduler(fixedReaders(1), 4);
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
        toHost(device).substr(0, kBufferSize), content.substr(0, kBufferSize));
    // A cancelled range can be cancelled again, and the scheduler goes on.
    scheduler->cancel(*started);
    auto read = scheduler->submit(
        group, content.size(), dst, order, fillFrom(content, 0));
    scheduler->wait(*read);
    EXPECT_EQ(toHost(device), content);
  }
}

// Destroying the scheduler waits for the piece in flight and fails the
// ranges that are still queued instead of running them.
TEST_F(FetchSchedulerTest, teardownWithReadsInFlight) {
  const auto content = makeContent(4 * kBufferSize);
  for (const auto order : {FillOrder::kAny, FillOrder::kSequential}) {
    SCOPED_TRACE(order == FillOrder::kAny ? "any" : "sequential");
    auto scheduler = makeScheduler(fixedReaders(1), 4);
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
        toHost(device).substr(0, kBufferSize), content.substr(0, kBufferSize));
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

TEST_F(FetchSchedulerTest, nextNumReaders) {
  Options options;
  options.minReaders = 2;
  options.maxReaders = 8;
  using Window = FetchScheduler::ControlWindow;

  // Nothing measured.
  EXPECT_EQ(FetchScheduler::nextNumReaders(options, 4, Window{}), 4);
  // Ranges waited for a reader while the copy streams were mostly idle.
  EXPECT_EQ(
      FetchScheduler::nextNumReaders(
          options,
          4,
          Window{
              .activeNanos = 100,
              .copyBusyNanos = 50,
              .readerLimitedNanos = 90}),
      5);
  EXPECT_EQ(
      FetchScheduler::nextNumReaders(
          options,
          8,
          Window{
              .activeNanos = 100,
              .copyBusyNanos = 50,
              .readerLimitedNanos = 90}),
      8);
  // Ranges waited for a reader, but the copy streams had no time left.
  EXPECT_EQ(
      FetchScheduler::nextNumReaders(
          options,
          4,
          Window{
              .activeNanos = 100,
              .copyBusyNanos = 95,
              .readerLimitedNanos = 90}),
      4);
  // The copy streams were idle, but no range waited for a reader.
  EXPECT_EQ(
      FetchScheduler::nextNumReaders(
          options, 4, Window{.activeNanos = 100, .copyBusyNanos = 10}),
      4);
  // Ranges waited for staging buffers: the copies cannot keep up.
  EXPECT_EQ(
      FetchScheduler::nextNumReaders(
          options,
          4,
          Window{
              .activeNanos = 100,
              .copyBusyNanos = 100,
              .bufferLimitedNanos = 30}),
      3);
  EXPECT_EQ(
      FetchScheduler::nextNumReaders(
          options,
          2,
          Window{
              .activeNanos = 100,
              .copyBusyNanos = 100,
              .bufferLimitedNanos = 30}),
      2); // The count of readers is never zero.
  VELOX_ASSERT_THROW(
      FetchScheduler::nextNumReaders(options, 0, Window{}), "(0 vs. 0)");
}

// The rule settles where the readers deliver what the copies can take, from
// either side, and at the upper bound when the copies are never the limit.
TEST_F(FetchSchedulerTest, nextNumReadersConverges) {
  Options options;
  options.minReaders = 1;
  options.maxReaders = 16;
  // What the controller measures with 'numReaders' readers of 'readRate'
  // each in front of copies of 'copyRate', when ranges are always queued.
  const auto measure = [](size_t numReaders, double readRate, double copyRate) {
    FetchScheduler::ControlWindow window;
    window.activeNanos = 1'000'000;
    const auto delivered = numReaders * readRate;
    if (delivered > copyRate) {
      // The copies never stop, and the readers stand by for staging buffers
      // for the share of the time they are ahead.
      window.copyBusyNanos = window.activeNanos;
      window.bufferLimitedNanos = static_cast<uint64_t>(
          window.activeNanos * (1 - copyRate / delivered));
      window.readerLimitedNanos =
          window.activeNanos - window.bufferLimitedNanos;
    } else {
      window.copyBusyNanos =
          static_cast<uint64_t>(window.activeNanos * (delivered / copyRate));
      window.readerLimitedNanos = window.activeNanos;
    }
    return window;
  };
  const auto settle = [&](size_t numReaders, double readRate, double copyRate) {
    for (int i = 0; i < 100; ++i) {
      numReaders = FetchScheduler::nextNumReaders(
          options, numReaders, measure(numReaders, readRate, copyRate));
    }
    return numReaders;
  };

  // Page-cache reads of 3.6 GB/s per reader in front of a 22 GB/s link: six
  // readers deliver 21.6 GB/s.
  EXPECT_THAT(
      settle(1, 3.6, 22), testing::AllOf(testing::Ge(6), testing::Le(8)));
  EXPECT_THAT(
      settle(16, 3.6, 22), testing::AllOf(testing::Ge(6), testing::Le(8)));
  // The same readers in front of a 146 GB/s link, and a drive that gives
  // sixteen readers 6 GB/s between them in front of the 22 GB/s link: the
  // copies always wait.
  EXPECT_EQ(settle(1, 3.6, 146), 16);
  EXPECT_EQ(settle(4, 0.375, 22), 16);
}

// While ranges wait for a reader and none waits for a staging buffer, the
// controller adds readers up to its bound. The fills stand at a gate, so the
// waiting is a state and not a matter of timing.
TEST_F(FetchSchedulerTest, raisesReadersWhileRangesWait) {
  const auto content = makeContent(kBufferSize);
  Options options;
  options.minReaders = 1;
  options.maxReaders = 4;
  options.initialReaders = 1;
  options.controlPeriod = std::chrono::microseconds(100);
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
  EXPECT_GT(stats.readerLimitedNanos, 0);
  // The bound holds however long the ranges wait.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(scheduler->stats().numReaders, 4);
  EXPECT_EQ(gate.maxInside(), 4);

  gate.open();
  const auto cost = scheduler->wait(*read);
  EXPECT_EQ(cost.bytes, kNumPieces * kBufferSize);
  EXPECT_EQ(scheduler->stats().maxRunningReaders, 4);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
