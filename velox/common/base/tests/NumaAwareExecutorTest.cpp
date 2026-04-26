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

#include "velox/common/base/NumaAwareExecutor.h"

#include <filesystem>
#include <fstream>

#include <folly/synchronization/Latch.h>
#include <gtest/gtest.h>

#ifdef __linux__
#include <sched.h>
#endif

namespace facebook::velox {
namespace {

class NumaAwareExecutorTest : public testing::Test {
 protected:
  void SetUp() override {
    testDir_ = std::filesystem::temp_directory_path() / "numa_exec_test";
    std::filesystem::remove_all(testDir_);
    std::filesystem::create_directories(testDir_);
  }

  void TearDown() override {
    std::filesystem::remove_all(testDir_);
  }

  void createNode(int nodeId, const std::string& cpuList) {
    auto nodeDir = testDir_ / ("node" + std::to_string(nodeId));
    std::filesystem::create_directories(nodeDir);
    {
      std::ofstream f(nodeDir / "cpulist");
      f << cpuList;
    }
    {
      std::ofstream f(nodeDir / "meminfo");
      f << "Node " << nodeId << " MemTotal:    16000000 kB\n";
    }
  }

  std::filesystem::path testDir_;
};

TEST_F(NumaAwareExecutorTest, singleNodeFallback) {
  createNode(0, "0-3");
  auto config = process::NumaConfig::fromSysFs(testDir_.string());

  NumaAwareExecutor executor(
      NumaAwareExecutor::Options{.threadsPerNode = 2, .pinThreads = false},
      config);

  EXPECT_EQ(executor.numNodes(), 1);

  folly::Latch latch(1);
  executor.add([&latch]() { latch.count_down(); });
  latch.wait();
}

TEST_F(NumaAwareExecutorTest, multiNodeDispatch) {
  createNode(0, "0-3");
  createNode(1, "4-7");
  auto config = process::NumaConfig::fromSysFs(testDir_.string());

  NumaAwareExecutor executor(
      NumaAwareExecutor::Options{.threadsPerNode = 2, .pinThreads = false},
      config);

  EXPECT_EQ(executor.numNodes(), 2);

  constexpr int kTasks = 20;
  folly::Latch latch(kTasks);
  std::atomic<int> completed{0};

  for (int i = 0; i < kTasks; ++i) {
    executor.addOnNode(
        [&latch, &completed]() {
          completed.fetch_add(1, std::memory_order_relaxed);
          latch.count_down();
        },
        i % 2);
  }

  latch.wait();
  EXPECT_EQ(completed.load(), kTasks);
}

TEST_F(NumaAwareExecutorTest, addRoundRobins) {
  createNode(0, "0-1");
  createNode(1, "2-3");
  auto config = process::NumaConfig::fromSysFs(testDir_.string());

  NumaAwareExecutor executor(
      NumaAwareExecutor::Options{.threadsPerNode = 1, .pinThreads = false},
      config);

  constexpr int kTasks = 10;
  folly::Latch latch(kTasks);

  for (int i = 0; i < kTasks; ++i) {
    executor.add([&latch]() { latch.count_down(); });
  }

  latch.wait();
}

TEST_F(NumaAwareExecutorTest, addOnNodeOutOfRange) {
  createNode(0, "0-3");
  createNode(1, "4-7");
  auto config = process::NumaConfig::fromSysFs(testDir_.string());

  NumaAwareExecutor executor(
      NumaAwareExecutor::Options{.threadsPerNode = 1, .pinThreads = false},
      config);

  // Out-of-range node should fall back to round-robin, not crash.
  folly::Latch latch(2);
  executor.addOnNode([&latch]() { latch.count_down(); }, -1);
  executor.addOnNode([&latch]() { latch.count_down(); }, 99);
  latch.wait();
}

#ifdef __linux__
TEST_F(NumaAwareExecutorTest, threadAffinityIsPinned) {
  // Use the real system NUMA config.
  const auto& config = process::NumaConfig::instance();
  if (!config.isAvailable()) {
    GTEST_SKIP() << "NUMA not available on this system";
  }

  NumaAwareExecutor executor(
      NumaAwareExecutor::Options{.threadsPerNode = 1, .pinThreads = true},
      config);

  const auto& expectedCpus = config.node(0).cpuIds;

  folly::Latch latch(1);
  bool affinityCorrect = false;

  executor.addOnNode(
      [&]() {
        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);
        sched_getaffinity(0, sizeof(cpuSet), &cpuSet);
        affinityCorrect = true;
        for (int cpu : expectedCpus) {
          if (!CPU_ISSET(cpu, &cpuSet)) {
            affinityCorrect = false;
            break;
          }
        }
        latch.count_down();
      },
      0);

  latch.wait();
  EXPECT_TRUE(affinityCorrect);
}
#endif

} // namespace
} // namespace facebook::velox
