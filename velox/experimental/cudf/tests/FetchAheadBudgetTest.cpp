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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
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
  std::shared_ptr<LoggedFetch> fetch(const std::string& name) {
    return std::make_shared<LoggedFetch>(log_, name, false);
  }

  std::shared_ptr<LoggedFetch> declinedFetch(const std::string& name) {
    return std::make_shared<LoggedFetch>(log_, name, true);
  }

  StartLog log_;
};

// Offered fetches start while the bytes held stay within the capacity, and
// the rest start in the order of the offers as bytes are released.
TEST_F(FetchAheadBudgetTest, startsAheadWithinCapacity) {
  FetchAheadBudget budget(100);
  budget.offer(fetch("a"), 40);
  budget.offer(fetch("b"), 40);
  budget.offer(fetch("c"), 40);
  budget.offer(fetch("d"), 10);
  // "d" would fit, but "c" was offered first.
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b"));
  auto stats = budget.stats();
  EXPECT_EQ(stats.heldBytes, 80);
  EXPECT_EQ(stats.numOffered, 4);
  EXPECT_EQ(stats.numStartedAhead, 2);
  EXPECT_EQ(stats.numWaiting, 2);

  // The consumer of "a" asks for it, decodes it and gives its bytes back.
  budget.claim();
  budget.release(40);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b", "c", "d"));
  stats = budget.stats();
  EXPECT_EQ(stats.heldBytes, 90);
  EXPECT_EQ(stats.maxHeldBytes, 90);
  EXPECT_EQ(stats.numWaiting, 0);
  EXPECT_EQ(stats.numAhead, 3);

  for (int i = 0; i < 3; ++i) {
    budget.claim();
  }
  budget.release(90);
  stats = budget.stats();
  EXPECT_EQ(stats.heldBytes, 0);
  EXPECT_EQ(stats.numAhead, 0);
}

// A fetch its consumer started is counted whatever the capacity. One fetch
// is ahead of its consumer even then; the next waits for room.
TEST_F(FetchAheadBudgetTest, heldByConsumers) {
  FetchAheadBudget budget(100);
  budget.hold(70);
  budget.hold(70);
  EXPECT_EQ(budget.stats().heldBytes, 140);

  budget.offer(fetch("a"), 20);
  budget.offer(fetch("b"), 20);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a"));
  EXPECT_EQ(budget.stats().heldBytes, 160);
  budget.release(70);
  // 90 + 20 do not fit, and "a" is still ahead.
  EXPECT_THAT(log_.names(), testing::ElementsAre("a"));
  budget.release(70);
  // 20 + 20 fit.
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b"));
  EXPECT_EQ(budget.stats().heldBytes, 40);
  EXPECT_EQ(budget.stats().maxHeldBytes, 160);
}

// One fetch starts ahead even when it is larger than the capacity and its
// consumers hold more than the capacity already. The next starts when the
// consumer of the first asks for it.
TEST_F(FetchAheadBudgetTest, oneFetchAlwaysAhead) {
  FetchAheadBudget budget(10);
  budget.hold(50);
  budget.offer(fetch("a"), 50);
  budget.offer(fetch("b"), 50);
  budget.offer(fetch("c"), 50);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a"));
  budget.claim();
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b"));
  // Giving bytes back makes no room while a fetch is ahead and the capacity
  // is exceeded.
  budget.release(50);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b"));
  budget.claim();
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "b", "c"));
  budget.claim();
  budget.release(150);
  EXPECT_EQ(budget.stats().heldBytes, 0);
}

// An offer taken back does not start and does not keep later offers waiting.
TEST_F(FetchAheadBudgetTest, withdrawnOffer) {
  FetchAheadBudget budget(100);
  budget.hold(80);
  budget.offer(fetch("a"), 10);
  auto large = fetch("b");
  budget.offer(large, 80);
  // Fits, and waits behind the offer that does not.
  budget.offer(fetch("c"), 10);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a"));
  EXPECT_EQ(budget.stats().numWaiting, 2);
  // With the large offer gone the one behind it starts at once.
  budget.withdraw(large.get());
  EXPECT_EQ(budget.stats().numWaiting, 0);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "c"));
  EXPECT_EQ(budget.stats().heldBytes, 100);
  // Taking back an offer that is not waiting does nothing.
  budget.withdraw(large.get());
  budget.release(80);
  EXPECT_THAT(log_.names(), testing::ElementsAre("a", "c"));
  EXPECT_EQ(budget.stats().heldBytes, 20);
}

// A fetch that has nothing to start holds no bytes and does not keep the
// ones behind it waiting.
TEST_F(FetchAheadBudgetTest, declinedFetch) {
  FetchAheadBudget budget(100);
  budget.hold(80);
  budget.offer(declinedFetch("a"), 60);
  budget.offer(fetch("b"), 60);
  EXPECT_THAT(log_.names(), testing::ElementsAre("b"));
  const auto stats = budget.stats();
  EXPECT_EQ(stats.heldBytes, 140);
  EXPECT_EQ(stats.numStartedAhead, 1);
  EXPECT_EQ(stats.numAhead, 1);
  EXPECT_EQ(stats.numWaiting, 0);
}

TEST_F(FetchAheadBudgetTest, releaseOfBytesNotHeld) {
  FetchAheadBudget budget(100);
  budget.hold(10);
  VELOX_ASSERT_THROW(budget.release(20), "(10 vs. 20)");
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
        budget.hold(300);
        budget.offer(std::make_shared<CountedFetch>(numStarted), 100);
        budget.release(300);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  // Every offer has started by now: the last release found room for the
  // ones still waiting, one after the other as their bytes are released
  // below.
  auto stats = budget.stats();
  while (stats.numWaiting > 0 || stats.heldBytes > 0) {
    ASSERT_GE(stats.heldBytes, 100);
    ASSERT_GT(stats.numAhead, 0);
    budget.claim();
    budget.release(100);
    stats = budget.stats();
  }
  EXPECT_EQ(numStarted, kNumThreads * kNumRounds);
  EXPECT_EQ(stats.numStartedAhead, kNumThreads * kNumRounds);
  EXPECT_LE(stats.maxHeldBytes, 8 * 300 + 1'000);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
