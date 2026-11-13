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

#include "velox/experimental/cudf/connectors/hive/FetchAheadBudget.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/synchronization/Baton.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// Records the order in which fetches are started.
class StartLog {
 public:
  void add(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    names_.push_back(name);
  }

  std::vector<std::string> names() {
    std::lock_guard<std::mutex> lock(mutex_);
    return names_;
  }

 private:
  std::mutex mutex_;
  std::vector<std::string> names_;
};

// A fetch that logs its start, or declines it as a fetch does whose consumer
// has started it already.
class LoggedFetch : public FetchAheadBudget::Candidate {
 public:
  LoggedFetch(StartLog& log, std::string name, bool declines)
      : log_(log), name_(std::move(name)), declines_(declines) {}

  bool start() override {
    if (declines_) {
      return false;
    }
    log_.add(name_);
    return true;
  }

 private:
  StartLog& log_;
  const std::string name_;
  const bool declines_;
};

class FetchAheadBudgetTest : public testing::Test {
 protected:
  static constexpr auto kNext = FetchAheadBudget::Distance::kNext;
  static constexpr auto kLater = FetchAheadBudget::Distance::kLater;
  static constexpr auto kAwaited = FetchAheadBudget::Stage::kAwaited;
  static constexpr auto kReceived = FetchAheadBudget::Stage::kReceived;

  // A consumer of 'scan' starts a fetch of 'bytes' and receives them.
  static void
  holdReceived(FetchAheadBudget& budget, uint64_t bytes, uint64_t scan) {
    budget.hold(bytes, scan);
    budget.receive(scan);
  }

  // The consumer of a fetch of 'scan' that was started ahead asks for it and
  // receives its bytes.
  static void claimReceived(FetchAheadBudget& budget, uint64_t scan) {
    budget.claim(scan);
    budget.receive(scan);
  }

  std::shared_ptr<LoggedFetch> fetch(const std::string& name) {
    return std::make_shared<LoggedFetch>(log_, name, false);
  }

  std::shared_ptr<LoggedFetch> declinedFetch(const std::string& name) {
    return std::make_shared<LoggedFetch>(log_, name, true);
  }

  StartLog log_;
};

// Fetches offered for a scan start once a consumer of the scan has received
// bytes, while the bytes held stay within the capacity; the rest start in
// the order of the offers as bytes are released.
TEST_F(FetchAheadBudgetTest, startsAheadWithinCapacity) {
  FetchAheadBudget budget(100);
  budget.offer(fetch("a"), 40, kLater, 0);
  // No consumer of the scan has asked for bytes yet.
  EXPECT_TRUE(log_.names().empty());
  // Bytes another scan's consumer has do not count for this one.
  holdReceived(budget, 0, 1);
  EXPECT_TRUE(log_.names().empty());
  // Nor do bytes the scan's consumer has asked for and does not have yet.
  budget.hold(10, 0);
  EXPECT_TRUE(log_.names().empty());
  budget.receive(0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a"));
  budget.offer(fetch("b"), 40, kLater, 0);
  budget.offer(fetch("c"), 40, kLater, 0);
  budget.offer(fetch("d"), 10, kLater, 0);
  // "d" would fit, but "c" was offered first.
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b"));
  auto stats = budget.stats();
  EXPECT_EQ(stats.heldBytes, 90);
  EXPECT_EQ(stats.numOffered, 4);
  EXPECT_EQ(stats.numStartedAhead, 2);
  EXPECT_EQ(stats.numWaiting, 2);

  // The consumer of "a" asks for it, decodes it and gives its bytes back.
  claimReceived(budget, 0);
  budget.release(40, 0, kReceived);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b", "c", "d"));
  stats = budget.stats();
  EXPECT_EQ(stats.heldBytes, 100);
  EXPECT_EQ(stats.maxHeldBytes, 100);
  EXPECT_EQ(stats.numWaiting, 0);
  EXPECT_EQ(stats.numAhead, 3);

  for (const uint64_t bytes : {40, 40, 10}) {
    claimReceived(budget, 0);
    budget.release(bytes, 0, kReceived);
  }
  budget.release(10, 0, kReceived);
  budget.release(0, 1, kReceived);
  stats = budget.stats();
  EXPECT_EQ(stats.heldBytes, 0);
  EXPECT_EQ(stats.numAhead, 0);
}

// A fetch its consumer started is counted whatever the capacity. One fetch
// is ahead of its consumer even then; the next waits for room.
TEST_F(FetchAheadBudgetTest, heldByConsumers) {
  FetchAheadBudget budget(100);
  holdReceived(budget, 70, 0);
  holdReceived(budget, 70, 0);
  holdReceived(budget, 0, 0);
  EXPECT_EQ(budget.stats().heldBytes, 140);

  budget.offer(fetch("a"), 20, kLater, 0);
  budget.offer(fetch("b"), 20, kLater, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a"));
  EXPECT_EQ(budget.stats().heldBytes, 160);
  budget.release(70, 0, kReceived);
  // 90 + 20 do not fit, and "a" is still ahead.
  EXPECT_THAT(log_.names(), testing::ElementsAre("a"));
  budget.release(70, 0, kReceived);
  // 20 + 20 fit.
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b"));
  EXPECT_EQ(budget.stats().heldBytes, 40);
  EXPECT_EQ(budget.stats().maxHeldBytes, 160);
}

// One fetch starts ahead even when it is larger than the capacity and its
// consumers hold more than the capacity already. The next starts when the
// consumer of the first has it.
TEST_F(FetchAheadBudgetTest, oneFetchAlwaysAhead) {
  FetchAheadBudget budget(10);
  holdReceived(budget, 50, 0);
  budget.offer(fetch("a"), 50, kLater, 0);
  budget.offer(fetch("b"), 50, kLater, 0);
  budget.offer(fetch("c"), 50, kLater, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a"));
  claimReceived(budget, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b"));
  // Giving bytes back makes no room while a fetch is ahead and the capacity
  // is exceeded.
  budget.release(50, 0, kReceived);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b"));
  claimReceived(budget, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b", "c"));
  claimReceived(budget, 0);
  for (int i = 0; i < 3; ++i) {
    budget.release(50, 0, kReceived);
  }
  EXPECT_EQ(budget.stats().heldBytes, 0);
}

// Every scan may have one fetch ahead, whatever is held and whatever waits
// in front of it.
TEST_F(FetchAheadBudgetTest, oneFetchAheadPerScan) {
  FetchAheadBudget budget(100);
  holdReceived(budget, 50, 1);
  holdReceived(budget, 50, 2);
  budget.offer(fetch("a1"), 50, kLater, 1);
  budget.offer(fetch("a2"), 50, kLater, 1);
  // The second scan has nothing ahead, so its first fetch does not wait
  // behind the first scan's second.
  budget.offer(fetch("b1"), 50, kLater, 2);
  budget.offer(fetch("b2"), 50, kLater, 2);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a1", "b1"));
  EXPECT_EQ(budget.stats().numAhead, 2);
  EXPECT_EQ(budget.stats().numWaiting, 2);
  // Its consumer takes the first scan's fetch: the next one may be ahead.
  claimReceived(budget, 1);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a1", "b1", "a2"));
  VELOX_ASSERT_THROW(budget.claim(3), "No fetch of the scan is ahead: 3");
  claimReceived(budget, 1);
  claimReceived(budget, 2);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a1", "b1", "a2", "b2"));
  claimReceived(budget, 2);
  for (const uint64_t scan : {1, 2}) {
    for (int i = 0; i < 3; ++i) {
      budget.release(50, scan, kReceived);
    }
  }
  VELOX_ASSERT_THROW(
      budget.release(0, 1, kReceived), "No fetch of the scan is received: 1");
  EXPECT_EQ(budget.stats().heldBytes, 0);
  EXPECT_EQ(budget.stats().numAhead, 0);
}

// An offer taken back does not start and does not keep later offers waiting.
TEST_F(FetchAheadBudgetTest, withdrawnOffer) {
  FetchAheadBudget budget(100);
  holdReceived(budget, 80, 0);
  budget.offer(fetch("a"), 10, kLater, 0);
  auto large = fetch("b");
  budget.offer(large, 80, kLater, 0);
  // Fits, and waits behind the offer that does not.
  budget.offer(fetch("c"), 10, kLater, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a"));
  EXPECT_EQ(budget.stats().numWaiting, 2);
  // With the large offer gone the one behind it starts at once.
  budget.withdraw(large.get());
  EXPECT_EQ(budget.stats().numWaiting, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "c"));
  EXPECT_EQ(budget.stats().heldBytes, 100);
  // Taking back an offer that is not waiting does nothing.
  budget.withdraw(large.get());
  budget.release(80, 0, kReceived);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "c"));
  EXPECT_EQ(budget.stats().heldBytes, 20);
}

// An offer for what a consumer gets to next goes ahead of the offers for
// what consumers get to later, and behind the earlier ones of its kind.
TEST_F(FetchAheadBudgetTest, nextBeforeLater) {
  FetchAheadBudget budget(100);
  holdReceived(budget, 100, 0);
  holdReceived(budget, 0, 0);
  budget.offer(fetch("first"), 10, kLater, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("first"));
  budget.offer(fetch("later1"), 10, kLater, 0);
  budget.offer(fetch("next1"), 10, kNext, 0);
  budget.offer(fetch("later2"), 10, kLater, 0);
  budget.offer(fetch("next2"), 10, kNext, 0);
  EXPECT_EQ(budget.stats().numWaiting, 4);
  budget.release(100, 0, kReceived);
  EXPECT_THAT(
      log_.names(),
      testing::ElementsAre("first", "next1", "next2", "later1", "later2"));
}

// No fetch of any scan starts ahead while a consumer is blocked waiting for
// bytes; the offers made meanwhile start, in order, when the last waiting
// consumer has its bytes.
TEST_F(FetchAheadBudgetTest, consumersWaitingForBytes) {
  FetchAheadBudget budget(100);
  holdReceived(budget, 0, 0);
  holdReceived(budget, 0, 1);
  budget.waitBegin();
  budget.waitBegin();
  budget.offer(fetch("a"), 10, kLater, 0);
  budget.offer(fetch("b"), 10, kLater, 1);
  // Room appearing does not start them either.
  holdReceived(budget, 50, 0);
  budget.release(50, 0, kReceived);
  EXPECT_TRUE(log_.names().empty());
  EXPECT_EQ(budget.stats().numWaiting, 2);
  EXPECT_EQ(budget.stats().numWaitingConsumers, 2);
  budget.waitEnd();
  EXPECT_TRUE(log_.names().empty());
  budget.waitEnd();
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b"));
  EXPECT_EQ(budget.stats().numWaitingConsumers, 0);
}

// No fetch of a scan starts ahead while a consumer of the scan awaits a
// fetch, whether it started that fetch or claimed it, until the consumer has
// the bytes or gives the fetch up. Other scans are not held back.
TEST_F(FetchAheadBudgetTest, awaitedFetches) {
  FetchAheadBudget budget(100);
  holdReceived(budget, 0, 0);
  holdReceived(budget, 0, 1);
  budget.hold(10, 0);
  budget.offer(fetch("a"), 10, kLater, 0);
  budget.offer(fetch("b"), 10, kLater, 1);
  EXPECT_THAT(log_.names(), testing::ElementsAre("b"));
  budget.receive(0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("b", "a"));

  budget.claim(0);
  budget.offer(fetch("c"), 10, kLater, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("b", "a"));
  budget.receive(0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("b", "a", "c"));

  budget.hold(10, 0);
  budget.offer(fetch("d"), 10, kLater, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("b", "a", "c"));
  budget.release(10, 0, kAwaited);
  EXPECT_THAT(log_.names(), testing::ElementsAre("b", "a", "c", "d"));

  VELOX_ASSERT_THROW(budget.receive(2), "No fetch of the scan is awaited: 2");
}

// A scan whose consumers hold nothing between two fetches still starts what
// it has on offer: they are busy with what they received.
TEST_F(FetchAheadBudgetTest, betweenFetches) {
  FetchAheadBudget budget(100);
  holdReceived(budget, 10, 0);
  budget.waitBegin();
  budget.offer(fetch("a"), 10, kLater, 0);
  budget.release(10, 0, kReceived);
  EXPECT_TRUE(log_.names().empty());
  budget.waitEnd();
  EXPECT_THAT(log_.names(), testing::ElementsAre("a"));
  // Once the scan holds and offers nothing, it starts over: an offer waits
  // for a consumer to receive bytes again.
  claimReceived(budget, 0);
  budget.release(10, 0, kReceived);
  budget.offer(fetch("b"), 10, kLater, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a"));
  holdReceived(budget, 10, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b"));

  // While a reader of the scan exists, the scan does not start over.
  budget.attach(1);
  holdReceived(budget, 10, 1);
  budget.release(10, 1, kReceived);
  budget.offer(fetch("c"), 10, kLater, 1);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b", "c"));
  budget.detach(1);
}

// The offer for what a consumer gets to next, made before the consumer says
// that it has its bytes, starts before the later offers that were waiting.
TEST_F(FetchAheadBudgetTest, nextOfferedBeforeReceive) {
  FetchAheadBudget budget(100);
  budget.hold(60, 0);
  budget.offer(fetch("later"), 50, kLater, 0);
  budget.offer(fetch("next"), 30, kNext, 0);
  EXPECT_TRUE(log_.names().empty());
  budget.receive(0);
  // 60 + 30 fit; 50 more do not, and a fetch of the scan is ahead.
  EXPECT_THAT(log_.names(), testing::ElementsAre("next"));
}

// A fetch that has nothing to start holds no bytes and does not keep the
// ones behind it waiting.
TEST_F(FetchAheadBudgetTest, declinedFetch) {
  FetchAheadBudget budget(100);
  holdReceived(budget, 80, 0);
  budget.offer(declinedFetch("a"), 60, kLater, 0);
  budget.offer(fetch("b"), 60, kLater, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("b"));
  const auto stats = budget.stats();
  EXPECT_EQ(stats.heldBytes, 140);
  EXPECT_EQ(stats.numStartedAhead, 1);
  EXPECT_EQ(stats.numAhead, 1);
  EXPECT_EQ(stats.numWaiting, 0);
}

TEST_F(FetchAheadBudgetTest, releaseOfBytesNotHeld) {
  FetchAheadBudget budget(100);
  budget.hold(10, 0);
  VELOX_ASSERT_THROW(budget.release(20, 0, kAwaited), "(10 vs. 20)");
}

// A posted start runs once on the thread it is posted to. Until it begins
// the consumer can take it back, and a start that cannot be posted or that
// fails leaves the offer and the budget as if it had not been made.
TEST_F(FetchAheadBudgetTest, postedStart) {
  std::vector<std::function<void()>> tasks;
  bool postFails{false};
  bool startFails{false};
  int numStarts{0};
  FetchAheadBudget budget(100);
  // A consumer of the scan holds a fetch of no bytes throughout.
  holdReceived(budget, 0, 0);
  const auto offer = [&]() {
    auto start = std::make_shared<PostedFetchStart>(
        &budget,
        10,
        /*scan=*/0,
        [&](std::function<void()> task) {
          VELOX_CHECK(!postFails, "Injected post failure");
          tasks.push_back(std::move(task));
        },
        [&]() {
          VELOX_CHECK(!startFails, "Injected start failure");
          ++numStarts;
        });
    budget.offer(start, 10, kLater, 0);
    return start;
  };
  const auto expectNothingHeld = [&]() {
    EXPECT_EQ(budget.stats().heldBytes, 0);
    EXPECT_EQ(budget.stats().numAhead, 0);
    EXPECT_EQ(budget.stats().numWaiting, 0);
  };
  {
    // Posted, then run: the fetch is ahead until its consumer claims it.
    auto start = offer();
    ASSERT_EQ(tasks.size(), 1);
    EXPECT_EQ(numStarts, 0);
    EXPECT_EQ(budget.stats().heldBytes, 10);
    tasks[0]();
    EXPECT_EQ(numStarts, 1);
    start->withdraw();
    EXPECT_EQ(budget.stats().heldBytes, 10);
    EXPECT_EQ(budget.stats().numAhead, 1);
    budget.claim(0);
    budget.release(10, 0, kAwaited);
    tasks.clear();
  }
  {
    // Taken back before it begins: it never runs, and holds nothing.
    auto start = offer();
    ASSERT_EQ(tasks.size(), 1);
    EXPECT_EQ(budget.stats().heldBytes, 10);
    start->withdraw();
    expectNothingHeld();
    tasks[0]();
    EXPECT_EQ(numStarts, 1);
    expectNothingHeld();
    tasks.clear();
  }
  {
    // A start that cannot be posted leaves the offer free to be taken back.
    postFails = true;
    auto start = offer();
    postFails = false;
    EXPECT_TRUE(tasks.empty());
    expectNothingHeld();
    start->withdraw();
    expectNothingHeld();
    EXPECT_EQ(numStarts, 1);
  }
  {
    // A start that fails gives its bytes back.
    startFails = true;
    auto start = offer();
    ASSERT_EQ(tasks.size(), 1);
    tasks[0]();
    startFails = false;
    expectNothingHeld();
    start->withdraw();
    expectNothingHeld();
    EXPECT_EQ(numStarts, 1);
    tasks.clear();
  }
}

// withdraw() of a posted start that is running returns after the start.
TEST_F(FetchAheadBudgetTest, withdrawOfRunningStart) {
  FetchAheadBudget budget(100);
  holdReceived(budget, 0, 0);
  folly::Baton<> began;
  folly::Baton<> finish;
  std::atomic_bool startDone{false};
  std::thread startThread;
  auto start = std::make_shared<PostedFetchStart>(
      &budget,
      10,
      /*scan=*/0,
      [&](std::function<void()> task) {
        startThread = std::thread(std::move(task));
      },
      [&]() {
        began.post();
        finish.wait();
        startDone = true;
      });
  budget.offer(start, 10, kLater, 0);
  began.wait();
  folly::Baton<> withdrawing;
  bool sawStartDone{false};
  std::thread withdrawThread([&]() {
    withdrawing.post();
    start->withdraw();
    sawStartDone = startDone;
  });
  withdrawing.wait();
  finish.post();
  withdrawThread.join();
  startThread.join();
  EXPECT_TRUE(sawStartDone);
  // The start ran: the fetch is ahead until its consumer claims it.
  EXPECT_EQ(budget.stats().numAhead, 1);
  EXPECT_EQ(budget.stats().numClaimed, 0);
  budget.claim(0);
  EXPECT_EQ(budget.stats().numClaimed, 1);
  budget.release(10, 0, kAwaited);
}

// A task posted from a thread of an IO executor runs on the context's own
// thread, after the tasks posted before it.
TEST_F(FetchAheadBudgetTest, contextStartThread) {
  FetchAheadContext context(100);
  EXPECT_EQ(context.budget().capacity(), 100);
  folly::IOThreadPoolExecutor ioExecutor(2);
  std::thread::id ioThread;
  std::vector<std::thread::id> startThreads;
  folly::Baton<> done;
  ioExecutor.add([&]() {
    ioThread = std::this_thread::get_id();
    context.post([&]() { startThreads.push_back(std::this_thread::get_id()); });
    context.post([&]() {
      startThreads.push_back(std::this_thread::get_id());
      done.post();
    });
  });
  done.wait();
  ASSERT_EQ(startThreads.size(), 2);
  EXPECT_EQ(startThreads[0], startThreads[1]);
  EXPECT_NE(startThreads[0], ioThread);
  EXPECT_NE(startThreads[0], std::this_thread::get_id());
}

// Threads that hold, offer and release at once leave nothing held and start
// every offered fetch exactly once.
TEST_F(FetchAheadBudgetTest, concurrentUse) {
  FetchAheadBudget budget(1'000);
  std::atomic_int32_t numStarted{0};

  // A fetch that counts its start and gives its bytes back right away.
  class CountedFetch : public FetchAheadBudget::Candidate {
   public:
    explicit CountedFetch(std::atomic_int32_t& numStarted)
        : numStarted_(numStarted) {}

    bool start() override {
      ++numStarted_;
      return true;
    }

   private:
    std::atomic_int32_t& numStarted_;
  };

  constexpr int32_t kNumThreads{8};
  constexpr int32_t kNumRounds{500};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int32_t thread = 0; thread < kNumThreads; ++thread) {
    threads.emplace_back([&]() {
      for (int32_t i = 0; i < kNumRounds; ++i) {
        holdReceived(budget, 300, 0);
        budget.offer(
            std::make_shared<CountedFetch>(numStarted), 100, kLater, 0);
        budget.release(300, 0, kReceived);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  // The offers still waiting start while a consumer holds bytes, one after
  // the other as the fetches ahead are claimed and released below.
  holdReceived(budget, 0, 0);
  auto stats = budget.stats();
  while (stats.numWaiting > 0 || stats.numAhead > 0) {
    ASSERT_GT(stats.numAhead, 0);
    claimReceived(budget, 0);
    budget.release(100, 0, kReceived);
    stats = budget.stats();
  }
  budget.release(0, 0, kReceived);
  stats = budget.stats();
  EXPECT_EQ(stats.heldBytes, 0);
  EXPECT_EQ(numStarted, kNumThreads * kNumRounds);
  EXPECT_EQ(stats.numStartedAhead, kNumThreads * kNumRounds);
  // What the threads hold, the capacity, and the one fetch that may be ahead
  // beyond it.
  EXPECT_LE(stats.maxHeldBytes, 8 * 300 + 1'000 + 100);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
