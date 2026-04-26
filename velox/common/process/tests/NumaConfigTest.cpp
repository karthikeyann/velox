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

#include "velox/common/process/NumaConfig.h"

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

namespace facebook::velox::process {
namespace {

class NumaConfigTest : public testing::Test {
 protected:
  void SetUp() override {
    testDir_ = std::filesystem::temp_directory_path() / "numa_config_test";
    std::filesystem::remove_all(testDir_);
    std::filesystem::create_directories(testDir_);
  }

  void TearDown() override {
    std::filesystem::remove_all(testDir_);
  }

  void createNode(
      int nodeId,
      const std::string& cpuList,
      uint64_t memTotalKb = 16000000) {
    auto nodeDir = testDir_ / ("node" + std::to_string(nodeId));
    std::filesystem::create_directories(nodeDir);

    {
      std::ofstream f(nodeDir / "cpulist");
      f << cpuList;
    }
    {
      std::ofstream f(nodeDir / "meminfo");
      f << "Node " << nodeId << " MemTotal:    " << memTotalKb << " kB\n";
      f << "Node " << nodeId << " MemFree:     8000000 kB\n";
    }
  }

  std::filesystem::path testDir_;
};

TEST_F(NumaConfigTest, nonExistentSysFs) {
  auto config = NumaConfig::fromSysFs("/no/such/path");
  EXPECT_FALSE(config.isAvailable());
  EXPECT_EQ(config.numNodes(), 1);
  EXPECT_EQ(config.assignNode(0, 0), 0);
  EXPECT_EQ(config.assignNode(42, 7), 0);
}

TEST_F(NumaConfigTest, singleNode) {
  createNode(0, "0-7");
  auto config = NumaConfig::fromSysFs(testDir_.string());
  EXPECT_FALSE(config.isAvailable());
  EXPECT_EQ(config.numNodes(), 1);
  EXPECT_EQ(config.node(0).cpuIds.size(), 8);
  EXPECT_EQ(config.assignNode(0, 0), 0);
}

TEST_F(NumaConfigTest, twoNodes) {
  createNode(0, "0-3,8-11");
  createNode(1, "4-7,12-15");

  auto config = NumaConfig::fromSysFs(testDir_.string());
  EXPECT_TRUE(config.isAvailable());
  EXPECT_EQ(config.numNodes(), 2);

  // node0: 0,1,2,3,8,9,10,11
  EXPECT_EQ(config.node(0).cpuIds.size(), 8);
  EXPECT_EQ(config.node(0).cpuIds[0], 0);
  EXPECT_EQ(config.node(0).cpuIds[4], 8);

  // node1: 4,5,6,7,12,13,14,15
  EXPECT_EQ(config.node(1).cpuIds.size(), 8);
  EXPECT_EQ(config.node(1).cpuIds[0], 4);

  EXPECT_GT(config.node(0).totalMemoryBytes, 0);
}

TEST_F(NumaConfigTest, fourNodes) {
  createNode(0, "0-3");
  createNode(1, "4-7");
  createNode(2, "8-11");
  createNode(3, "12-15");

  auto config = NumaConfig::fromSysFs(testDir_.string());
  EXPECT_TRUE(config.isAvailable());
  EXPECT_EQ(config.numNodes(), 4);
}

TEST_F(NumaConfigTest, assignNodeRoundRobin) {
  createNode(0, "0-3");
  createNode(1, "4-7");
  auto config = NumaConfig::fromSysFs(testDir_.string());

  // (splitGroupId + pipelineId) % numNodes
  EXPECT_EQ(config.assignNode(0, 0), 0);
  EXPECT_EQ(config.assignNode(0, 1), 1);
  EXPECT_EQ(config.assignNode(1, 0), 1);
  EXPECT_EQ(config.assignNode(1, 1), 0);
  EXPECT_EQ(config.assignNode(2, 0), 0);
  EXPECT_EQ(config.assignNode(2, 1), 1);
}

TEST_F(NumaConfigTest, parseCpuListEdgeCases) {
  createNode(0, "0");
  auto config = NumaConfig::fromSysFs(testDir_.string());
  EXPECT_EQ(config.node(0).cpuIds.size(), 1);
  EXPECT_EQ(config.node(0).cpuIds[0], 0);
}

TEST_F(NumaConfigTest, globalInstance) {
  const auto& config = NumaConfig::instance();
  // On any Linux system, should get at least 1 node.
  EXPECT_GE(config.numNodes(), 1);
}

} // namespace
} // namespace facebook::velox::process
