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
#include "velox/exec/DynamicFilterCollector.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::exec {
namespace {

using BuildDistribution = DynamicFilterCollector::BuildDistribution;

class DynamicFilterCollectorTest : public testing::Test {
 protected:
  static common::FilterPtr values(const std::vector<int64_t>& values) {
    return common::createBigintValues(values, false);
  }

  static common::FilterPtr range(int64_t lower, int64_t upper) {
    return std::make_shared<common::BigintRange>(lower, upper, false);
  }

  static common::FilterPtr alwaysFalse() {
    return std::make_shared<common::AlwaysFalse>();
  }

  // Returns the values in [0, 100) 'filter' passes.
  static std::vector<int64_t> passing(const common::Filter& filter) {
    std::vector<int64_t> passing;
    for (int64_t i = 0; i < 100; ++i) {
      if (filter.testInt64(i)) {
        passing.push_back(i);
      }
    }
    return passing;
  }
};

TEST_F(DynamicFilterCollectorTest, partitioned) {
  DynamicFilterCollector collector(BuildDistribution::kPartitioned, 3, {});
  EXPECT_FALSE(collector.done());
  VELOX_ASSERT_THROW(
      collector.result(), "Dynamic filters of some tasks are missing");

  EXPECT_FALSE(collector.add(values({1, 5, 60})));
  EXPECT_FALSE(collector.add(values({5, 7})));
  EXPECT_TRUE(collector.add(values({90})));
  ASSERT_TRUE(collector.done());
  EXPECT_EQ(
      passing(*collector.result()), std::vector<int64_t>({1, 5, 7, 60, 90}));
  EXPECT_FALSE(collector.result()->testNull());

  VELOX_ASSERT_THROW(
      collector.add(values({3})), "More dynamic filters added than tasks make");
}

TEST_F(DynamicFilterCollectorTest, emptyBuildPartition) {
  // A task without build-side keys does not narrow or disable the union.
  DynamicFilterCollector collector(BuildDistribution::kPartitioned, 3, {});
  collector.add(alwaysFalse());
  collector.add(values({4, 9}));
  ASSERT_TRUE(collector.add(alwaysFalse()));
  EXPECT_EQ(passing(*collector.result()), std::vector<int64_t>({4, 9}));

  // No task has build-side keys.
  DynamicFilterCollector empty(BuildDistribution::kPartitioned, 2, {});
  empty.add(alwaysFalse());
  ASSERT_TRUE(empty.add(alwaysFalse()));
  EXPECT_EQ(empty.result()->kind(), common::FilterKind::kAlwaysFalse);
}

TEST_F(DynamicFilterCollectorTest, optOut) {
  // One task that cannot describe its keys disables the filter right away.
  DynamicFilterCollector collector(BuildDistribution::kPartitioned, 3, {});
  EXPECT_FALSE(collector.add(values({1, 2})));
  EXPECT_TRUE(collector.add(nullptr));
  EXPECT_EQ(collector.result(), nullptr);
  EXPECT_TRUE(collector.add(values({3})));
  EXPECT_EQ(collector.result(), nullptr);
}

TEST_F(DynamicFilterCollectorTest, broadcast) {
  DynamicFilterCollector collector(BuildDistribution::kBroadcast, 3, {});
  ASSERT_TRUE(collector.add(values({1, 5})));
  EXPECT_EQ(passing(*collector.result()), std::vector<int64_t>({1, 5}));

  DynamicFilterCollector optOut(BuildDistribution::kBroadcast, 3, {});
  ASSERT_TRUE(optOut.add(nullptr));
  EXPECT_EQ(optOut.result(), nullptr);
}

TEST_F(DynamicFilterCollectorTest, unionOf) {
  const DynamicFilterCollector::Options options{.maxValues = 6};
  {
    // Sets within the limit stay exact.
    auto filter = DynamicFilterCollector::unionOf(
        values({1, 3, 50}), values({3, 70}), options);
    EXPECT_EQ(passing(*filter), std::vector<int64_t>({1, 3, 50, 70}));
  }
  {
    // A larger union becomes the range of the values.
    auto filter = DynamicFilterCollector::unionOf(
        values({10, 12, 14, 16}), values({11, 13, 15}), options);
    ASSERT_EQ(filter->kind(), common::FilterKind::kBigintRange);
    EXPECT_TRUE(filter->testInt64(10));
    EXPECT_TRUE(filter->testInt64(16));
    EXPECT_FALSE(filter->testInt64(9));
    EXPECT_FALSE(filter->testInt64(17));
  }
  {
    auto filter =
        DynamicFilterCollector::unionOf(range(10, 20), range(40, 50), options);
    ASSERT_EQ(filter->kind(), common::FilterKind::kBigintRange);
    EXPECT_TRUE(filter->testInt64(30));
    EXPECT_FALSE(filter->testInt64(9));
    EXPECT_FALSE(filter->testInt64(51));
  }
  {
    auto filter = DynamicFilterCollector::unionOf(
        range(10, 20), values({1, 30}), options);
    ASSERT_EQ(filter->kind(), common::FilterKind::kBigintRange);
    EXPECT_TRUE(filter->testInt64(1));
    EXPECT_TRUE(filter->testInt64(30));
    EXPECT_FALSE(filter->testInt64(31));
  }
  {
    // A single value is a set of one.
    auto filter =
        DynamicFilterCollector::unionOf(range(7, 7), values({1, 30}), options);
    EXPECT_EQ(passing(*filter), std::vector<int64_t>({1, 7, 30}));
  }
  {
    auto left = std::make_shared<common::BytesValues>(
        std::vector<std::string>{"a", "b"}, false);
    auto right = std::make_shared<common::BytesValues>(
        std::vector<std::string>{"b", "c"}, false);
    auto filter = DynamicFilterCollector::unionOf(left, right, options);
    ASSERT_EQ(filter->kind(), common::FilterKind::kBytesValues);
    EXPECT_TRUE(filter->testBytes("a", 1));
    EXPECT_TRUE(filter->testBytes("c", 1));
    EXPECT_FALSE(filter->testBytes("d", 1));
  }
  {
    // Kinds without a union disable the filter.
    auto bloom =
        std::make_shared<common::BigintValuesUsingBloomFilter>(1'000, false);
    EXPECT_EQ(
        DynamicFilterCollector::unionOf(bloom, values({1, 2}), options),
        nullptr);
    EXPECT_EQ(DynamicFilterCollector::unionOf(bloom, bloom, options), nullptr);
    // The filter of a task without keys still drops out.
    EXPECT_EQ(
        DynamicFilterCollector::unionOf(bloom, alwaysFalse(), options), bloom);
  }
}

} // namespace
} // namespace facebook::velox::exec
