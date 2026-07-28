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

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::test {

TEST(ConfigTest, MemoryTrackingDisabledByDefault) {
  CudfConfig config;
  EXPECT_FALSE(config.memoryTrackingEnabled);
  EXPECT_TRUE(config.perfettoMemoryTracePath.empty());
  EXPECT_TRUE(config.quentMemoryProfilePath.empty());
  EXPECT_EQ(config.quentMaxEvents, 250'000);
  EXPECT_FALSE(config.gpuMemoryTrackingEnabled());
}

TEST(ConfigTest, CudfConfig) {
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfEnabled, "false"},
      {CudfConfig::kCudfDebugEnabled, "true"},
      {CudfConfig::kCudfMemoryTrackingEnabled, "true"},
      {CudfConfig::kCudfPerfettoMemoryTracePath, "/tmp/gpu-memory-%p.pftrace"},
      {CudfConfig::kCudfQuentMemoryProfilePath, "/tmp/gpu-memory-%q-%t.json"},
      {CudfConfig::kCudfQuentQueryFilter, "query-42"},
      {CudfConfig::kCudfQuentMaxEvents, "250000"},
      {CudfConfig::kCudfQuentAdapterPath, "/opt/quent/lib/libquent.so"},
      {CudfConfig::kCudfQuentOutputPath, "/tmp/quent-%q"},
      {CudfConfig::kCudfMemoryResource, "arena"},
      {CudfConfig::kCudfMemoryPercent, "25"},
      {CudfConfig::kCudfFunctionNamePrefix, "presto"},
      {CudfConfig::kCudfAllowCpuFallback, "false"}};

  CudfConfig config;
  config.initialize(std::move(options));
  EXPECT_FALSE(config.enabled);
  EXPECT_TRUE(config.debugEnabled);
  EXPECT_TRUE(config.memoryTrackingEnabled);
  EXPECT_EQ(config.perfettoMemoryTracePath, "/tmp/gpu-memory-%p.pftrace");
  EXPECT_EQ(config.quentMemoryProfilePath, "/tmp/gpu-memory-%q-%t.json");
  EXPECT_EQ(config.quentQueryFilter, "query-42");
  EXPECT_EQ(config.quentMaxEvents, 250'000);
  EXPECT_EQ(config.quentAdapterPath, "/opt/quent/lib/libquent.so");
  EXPECT_EQ(config.quentOutputPath, "/tmp/quent-%q");
  EXPECT_TRUE(config.gpuMemoryTrackingEnabled());
  EXPECT_EQ(config.memoryResource, "arena");
  EXPECT_EQ(config.memoryPercent, 25);
  EXPECT_EQ(config.functionNamePrefix, "presto");
  EXPECT_FALSE(config.allowCpuFallback);
}

TEST(ConfigTest, PerfettoPathEnablesTracking) {
  CudfConfig config;
  config.initialize(
      {{CudfConfig::kCudfPerfettoMemoryTracePath, "/tmp/gpu.pftrace"}});
  EXPECT_FALSE(config.memoryTrackingEnabled);
  EXPECT_TRUE(config.gpuMemoryTrackingEnabled());
}

TEST(ConfigTest, QuentPathEnablesTracking) {
  CudfConfig config;
  config.initialize(
      {{CudfConfig::kCudfQuentMemoryProfilePath, "/tmp/gpu.json"}});
  EXPECT_FALSE(config.memoryTrackingEnabled);
  EXPECT_TRUE(config.gpuMemoryTrackingEnabled());
}

} // namespace facebook::velox::cudf_velox::test
