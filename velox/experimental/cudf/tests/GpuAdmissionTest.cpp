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

#include "velox/experimental/cudf/exec/GpuAdmission.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

class GpuAdmissionTest : public ::testing::Test {
 protected:
  void TearDown() override {
    GpuAdmission::setThresholdBytes(0);
  }
};

TEST_F(GpuAdmissionTest, disabledByDefaultAndAdmitsEveryone) {
  // The default has to be inert: this changes when work runs, so a deployment
  // that has not asked for it must not get it.
  EXPECT_EQ(GpuAdmission::thresholdBytes(), 0);
  auto a = GpuAdmission::acquire(1ULL << 40);
  auto b = GpuAdmission::acquire(1ULL << 40);
  EXPECT_FALSE(a.held());
  EXPECT_FALSE(b.held());
}

TEST_F(GpuAdmissionTest, admitsFreelyBelowTheThreshold) {
  // A request that fits alongside what is already out should not queue, and
  // should not even contend on the lock. Two at once prove it did not.
  GpuAdmission::setThresholdBytes(1ULL << 40);
  auto a = GpuAdmission::acquire(1024);
  auto b = GpuAdmission::acquire(1024);
  EXPECT_FALSE(a.held());
  EXPECT_FALSE(b.held());
}

TEST_F(GpuAdmissionTest, serialisesOnlyOncePressureIsReal) {
  // A threshold of one byte makes every request look like pressure, which is
  // the condition the mechanism exists for. The second caller must wait for
  // the first, and must be let through when the first finishes.
  GpuAdmission::setThresholdBytes(1);
  const auto waitsBefore = GpuAdmission::waitCount();

  std::atomic<bool> secondHasAdmission{false};
  std::atomic<bool> firstHasReleased{false};
  std::thread second;
  {
    auto first = GpuAdmission::acquire(1 << 20);
    ASSERT_TRUE(first.held());

    second = std::thread([&] {
      auto guard = GpuAdmission::acquire(1 << 20);
      // Reaching here at all means the first one let go first.
      EXPECT_TRUE(guard.held());
      EXPECT_TRUE(firstHasReleased.load());
      secondHasAdmission = true;
    });

    // Long enough that the other thread would have got in were it able to.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(secondHasAdmission.load());
    firstHasReleased = true;
  }
  second.join();
  EXPECT_TRUE(secondHasAdmission.load());
  EXPECT_GT(GpuAdmission::waitCount(), waitsBefore);
}

TEST_F(GpuAdmissionTest, manyCallersAllMakeProgress) {
  // The property that makes this safe is that a holder always finishes and
  // always gives it back, so a crowd drains rather than deadlocks.
  GpuAdmission::setThresholdBytes(1);
  std::atomic<int> completed{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      for (int n = 0; n < 20; ++n) {
        auto guard = GpuAdmission::acquire(1 << 20);
        ++completed;
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  EXPECT_EQ(completed.load(), 8 * 20);
}

} // namespace
} // namespace facebook::velox::cudf_velox
