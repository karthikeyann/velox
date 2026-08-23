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

#include "velox/experimental/cudf/benchmarks/KvikioReadPlan.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sstream>

using namespace facebook::velox::cudf_velox;

namespace {

std::vector<TargetInfo> twoTargets() {
  return {TargetInfo{"s3://bucket/a", 10}, TargetInfo{"s3://bucket/b", 5}};
}

// Returns each task as a (targetIndex, offset, size) tuple so gmock can
// compare whole plans in one assertion.
std::vector<std::tuple<size_t, uint64_t, uint64_t>> asTuples(
    const std::vector<ReadTask>& plan) {
  std::vector<std::tuple<size_t, uint64_t, uint64_t>> result;
  result.reserve(plan.size());
  for (const auto& task : plan) {
    result.emplace_back(task.targetIndex, task.offset, task.size);
  }
  return result;
}

uint64_t totalBytes(const std::vector<ReadTask>& plan) {
  uint64_t total{0};
  for (const auto& task : plan) {
    total += task.size;
  }
  return total;
}

} // namespace

TEST(KvikioReadPlanTest, parseManifestSkipsBlanksAndComments) {
  std::istringstream in(
      "s3://bucket/a\n"
      "\n"
      "# a comment\n"
      "   \n"
      "  s3://bucket/b  \n"
      "   # indented comment\n");

  EXPECT_THAT(
      parseManifest(in),
      testing::ElementsAre("s3://bucket/a", "s3://bucket/b"));
}

TEST(KvikioReadPlanTest, coldPlanCoversEachByteAtMostOnce) {
  const ReadPlanOptions options{
      .mode = ReadMode::kCold,
      .requestBytes = 4,
      .measurementBytes = 15,
      .seed = 0,
  };

  const auto plan = makeReadPlan(twoTargets(), options);

  // Target 0 is 10 bytes and target 1 is 5, so a 4-byte request size yields
  // three tasks then two, with the trailing task of each truncated.
  EXPECT_THAT(
      asTuples(plan),
      testing::ElementsAre(
          std::make_tuple(0UL, 0UL, 4UL),
          std::make_tuple(0UL, 4UL, 4UL),
          std::make_tuple(0UL, 8UL, 2UL),
          std::make_tuple(1UL, 0UL, 4UL),
          std::make_tuple(1UL, 4UL, 1UL)));
  EXPECT_EQ(totalBytes(plan), 15);
}

TEST(KvikioReadPlanTest, coldPlanRespectsMeasurementCap) {
  const ReadPlanOptions options{
      .mode = ReadMode::kCold,
      .requestBytes = 4,
      .measurementBytes = 6,
      .seed = 0,
  };

  const auto plan = makeReadPlan(twoTargets(), options);

  EXPECT_THAT(
      asTuples(plan),
      testing::ElementsAre(
          std::make_tuple(0UL, 0UL, 4UL), std::make_tuple(0UL, 4UL, 2UL)));
  EXPECT_EQ(totalBytes(plan), 6);
}

TEST(KvikioReadPlanTest, coldPlanRejectsMeasurementLargerThanManifest) {
  const ReadPlanOptions options{
      .mode = ReadMode::kCold,
      .requestBytes = 4,
      .measurementBytes = 16,
      .seed = 0,
  };

  VELOX_ASSERT_THROW(
      makeReadPlan(twoTargets(), options),
      "Cold mode cannot read more bytes than the manifest holds");
}

TEST(KvikioReadPlanTest, warmPlanProducesRequestedVolume) {
  const ReadPlanOptions options{
      .mode = ReadMode::kWarm,
      .requestBytes = 4,
      .measurementBytes = 100,
      .seed = 42,
  };

  const auto plan = makeReadPlan(twoTargets(), options);

  // Warm mode re-reads, so it is allowed to exceed the manifest size.
  EXPECT_EQ(totalBytes(plan), 100);
  for (const auto& task : plan) {
    ASSERT_LT(task.targetIndex, 2);
    const uint64_t targetSize = task.targetIndex == 0 ? 10 : 5;
    EXPECT_LE(task.offset + task.size, targetSize);
  }
}

TEST(KvikioReadPlanTest, rejectsEmptyTargetList) {
  const ReadPlanOptions options{
      .mode = ReadMode::kCold,
      .requestBytes = 4,
      .measurementBytes = 4,
      .seed = 0,
  };

  VELOX_ASSERT_THROW(
      makeReadPlan({}, options), "Manifest contains no readable bytes");
}
