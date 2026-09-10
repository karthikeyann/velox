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
#include "velox/experimental/cudf/exec/AggregationRegistry.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/PrestoAggregateFunctions.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/Timestamp.h"

#include <gflags/gflags.h>

#include <cmath>

DECLARE_bool(cudf_groupby_detect_sorted_keys);
DECLARE_int64(cudf_groupby_detect_sorted_min_rows);
DECLARE_bool(cudf_groupby_share_nonnull_counts);
DECLARE_bool(cudf_groupby_complete_batches);
DECLARE_bool(cudf_groupby_stream_raw_single);
DECLARE_bool(cudf_groupby_pack_short_string_keys);
DECLARE_int64(cudf_groupby_pack_short_string_min_rows);
DECLARE_uint64(cudf_dense_integer_sum_max_range);
DECLARE_int64(cudf_dense_integer_sum_min_rows);
DECLARE_bool(cudf_dense_integer_count_rows);
DECLARE_uint64(cudf_dense_integer_count_32_max_rows);
DECLARE_bool(cudf_final_groupby_unique_batches);
DECLARE_uint64(cudf_final_groupby_unique_max_bytes);
DECLARE_int64(cudf_final_groupby_unique_min_rows);

namespace facebook::velox::exec::test {

using core::QueryConfig;
using facebook::velox::test::BatchMaker;
using namespace common::testutil;

class AggregationTest : public OperatorTestBase {
 public:
  enum class AggSteps { kSingle, kPartialFinal, kPartialIntermediateFinal };

 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    TestValue::enable();
  }

  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    savedStreamingGroupbyEnabled_ =
        cudf_velox::CudfConfig::getInstance().streamingGroupbyEnabled;
    cudf_velox::CudfConfig::getInstance().streamingGroupbyEnabled = false;
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
    cudf_velox::registerPrestoAggregateFunctions("");
  }

  void TearDown() override {
    cudf_velox::CudfConfig::getInstance().streamingGroupbyEnabled =
        savedStreamingGroupbyEnabled_;
    cudf_velox::unregisterCudf();
    cudf_velox::unregisterAggregateFunctions();
    OperatorTestBase::TearDown();
  }

  std::vector<RowVectorPtr>
  makeVectors(const RowTypePtr& rowType, size_t size, int numVectors) {
    std::vector<RowVectorPtr> vectors;
    VectorFuzzer fuzzer({.vectorSize = size}, pool());
    for (int32_t i = 0; i < numVectors; ++i) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType));
    }
    return vectors;
  }

  template <typename T>
  void testSingleKey(
      const std::vector<RowVectorPtr>& vectors,
      const std::string& keyName,
      bool ignoreNullKeys,
      bool distinct) {
    std::vector<std::string> aggregates;
    if (!distinct) {
      // TODO (dm): "sum(15)", "sum(0.1)",  "min(15)",  "min(0.1)", "max(15)",
      // "max(0.1)",
      aggregates = {
          "sum(c1)",
          "sum(c2)",
          "sum(c4)",
          "sum(c5)",
          "min(c1)",
          "min(c2)",
          "min(c3)",
          "min(c4)",
          "min(c5)",
          "max(c1)",
          "max(c2)",
          "max(c3)",
          "max(c4)",
          "max(c5)"};
    }

    auto op = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {keyName},
                      aggregates,
                      {},
                      core::AggregationNode::Step::kPartial,
                      ignoreNullKeys)
                  .planNode();

    std::string fromClause = "FROM tmp";
    if (ignoreNullKeys) {
      fromClause += " WHERE " + keyName + " IS NOT NULL";
    }
    if (distinct) {
      assertQuery(op, "SELECT distinct " + keyName + " " + fromClause);
    } else {
      // TODO (dm): sum(15), sum(cast(0.1 as double)), min(15), min(0.1),
      // max(15), max(0.1),
      assertQuery(
          op,
          "SELECT " + keyName +
              ", sum(c1), sum(c2), sum(c4), sum(c5) , min(c1), min(c2), min(c3), min(c4), min(c5), max(c1), max(c2), max(c3), max(c4), max(c5) " +
              fromClause + " GROUP BY " + keyName);
    }
  }

  void testMultiKey(
      const std::vector<RowVectorPtr>& vectors,
      bool ignoreNullKeys,
      bool distinct) {
    std::vector<std::string> aggregates;
    // TODO (dm): "sum(15)", "sum(0.1)",  "min(15)",  "min(0.1)", "max(15)",
    // "max(0.1)"
    if (!distinct) {
      aggregates = {
          "sum(c4)",
          "sum(c5)",
          "min(c3)",
          "min(c4)",
          "min(c5)",
          "max(c3)",
          "max(c4)",
          "max(c5)"};
    }
    auto op = PlanBuilder()
                  .values(vectors)
                  .aggregation(
                      {"c0", "c1", "c6"},
                      aggregates,
                      {},
                      core::AggregationNode::Step::kPartial,
                      ignoreNullKeys)
                  .planNode();

    std::string fromClause = "FROM tmp";
    if (ignoreNullKeys) {
      fromClause +=
          " WHERE c0 IS NOT NULL AND c1 IS NOT NULL AND c6 IS NOT NULL";
    }
    if (distinct) {
      assertQuery(op, "SELECT distinct c0, c1, c6 " + fromClause);
    } else {
      // TODO (dm): sum(15), sum(cast(0.1 as double)), min(15), min(0.1),
      // max(15), max(0.1),, sum(1)
      assertQuery(
          op,
          "SELECT c0, c1, c6, sum(c4), sum(c5), min(c3), min(c4), min(c5),  max(c3), max(c4), max(c5) " +
              fromClause + " GROUP BY c0, c1, c6");
    }
  }

  void testAggregation(
      const std::vector<RowVectorPtr>& data,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::string& expectedSql,
      AggSteps steps) {
    auto builder = PlanBuilder().values(data);
    switch (steps) {
      case AggSteps::kSingle:
        builder.singleAggregation(groupingKeys, aggregates);
        break;
      case AggSteps::kPartialFinal:
        builder.partialAggregation(groupingKeys, aggregates).finalAggregation();
        break;
      case AggSteps::kPartialIntermediateFinal:
        builder.partialAggregation(groupingKeys, aggregates)
            .intermediateAggregation()
            .finalAggregation();
        break;
    }
    assertQuery(builder.planNode(), expectedSql);
  }

  void testGlobalCountStarZeroColumns(AggSteps steps) {
    auto data = makeRowVector({
        makeFlatVector<int64_t>({1, 2, 3, 4}),
    });
    createDuckDbTable({data});

    auto builder = PlanBuilder().values({data}).filter("c0 > 0").project({});
    switch (steps) {
      case AggSteps::kSingle:
        builder.singleAggregation({}, {"count(*)"});
        break;
      case AggSteps::kPartialFinal:
        builder.partialAggregation({}, {"count(*)"}).finalAggregation();
        break;
      case AggSteps::kPartialIntermediateFinal:
        builder.partialAggregation({}, {"count(*)"})
            .intermediateAggregation()
            .finalAggregation();
        break;
    }
    assertQuery(builder.planNode(), "SELECT count(*) FROM tmp WHERE c0 > 0");
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {BIGINT(),
           SMALLINT(),
           INTEGER(),
           BIGINT(),
           DOUBLE(), // DM: This used to be REAL() but we don't support that
           DOUBLE(),
           VARCHAR()})};
  bool savedStreamingGroupbyEnabled_{false};
};

class StreamingGroupbyAggregationTest : public AggregationTest {
 protected:
  void SetUp() override {
    AggregationTest::SetUp();
    auto& config = cudf_velox::CudfConfig::getInstance();
    savedOutputMemoryResource_ = config.outputMemoryResource;
    savedCapacityMultiplier_ = config.streamingGroupbyCapacityMultiplier;
    config.streamingGroupbyEnabled = true;
  }

  void TearDown() override {
    cudf_velox::CudfConfig::getInstance().outputMemoryResource =
        savedOutputMemoryResource_;
    cudf_velox::CudfConfig::getInstance().streamingGroupbyCapacityMultiplier =
        savedCapacityMultiplier_;
    AggregationTest::TearDown();
  }

  std::vector<RowVectorPtr> makeHighCardinalityBatches(
      int32_t batchRows,
      int32_t numBatches) {
    std::vector<RowVectorPtr> vectors;
    vectors.reserve(numBatches);
    for (int32_t batch = 0; batch < numBatches; ++batch) {
      const auto offset = static_cast<int64_t>(batch) * batchRows;
      vectors.push_back(makeRowVector({
          makeFlatVector<int64_t>(
              batchRows, [offset](auto row) { return offset + row; }),
          makeFlatVector<int64_t>(batchRows, [](auto /*row*/) { return 1; }),
      }));
    }
    return vectors;
  }

 private:
  std::string savedOutputMemoryResource_;
  double savedCapacityMultiplier_{2.0};
};

bool hasStreamingGroupbyStat(
    const std::shared_ptr<exec::Task>& task,
    const core::PlanNodeId& planNodeId,
    std::string_view name) {
  const auto planStats = toPlanStats(task->taskStats());
  const auto it = planStats.find(planNodeId);
  return it != planStats.end() &&
      it->second.customStats.count(std::string{name}) > 0;
}

int64_t streamingGroupbyStatSum(
    const std::shared_ptr<exec::Task>& task,
    const core::PlanNodeId& planNodeId,
    std::string_view name) {
  const auto planStats = toPlanStats(task->taskStats());
  const auto planIt = planStats.find(planNodeId);
  if (planIt == planStats.end()) {
    return 0;
  }
  const auto statIt = planIt->second.customStats.find(std::string{name});
  return statIt == planIt->second.customStats.end() ? 0 : statIt->second.sum;
}

TEST_F(StreamingGroupbyAggregationTest, uniqueFinalIntegerMinMaxBatches) {
  gflags::FlagSaver restore;
  FLAGS_cudf_final_groupby_unique_batches = true;
  FLAGS_cudf_final_groupby_unique_min_rows = 0;
  const auto run = [&]<typename Key>() {
    std::vector<RowVectorPtr> vectors;
    for (const auto offset :
         {std::numeric_limits<Key>::min(),
          Key(std::numeric_limits<Key>::max() - 2),
          Key{0}}) {
      vectors.push_back(makeRowVector(
          {"v", "junk", "k"},
          {makeNullableFlatVector<int64_t>(
               {std::numeric_limits<int64_t>::min(),
                std::nullopt,
                std::numeric_limits<int64_t>::max()}),
           makeFlatVector<int64_t>({1, 2, 3}),
           makeFlatVector<Key>({offset, Key(offset + 1), Key(offset + 2)})}));
    }
    createDuckDbTable(vectors);
    core::PlanNodeId id;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 3)
            .plan(PlanBuilder()
                      .values(vectors)
                      .finalAggregation(
                          {"k"}, {"min(v)", "max(v)"}, {{BIGINT()}, {BIGINT()}})
                      .capturePlanNodeId(id)
                      .planNode())
            .assertResults("SELECT k,min(v),max(v) FROM tmp GROUP BY k");
    EXPECT_EQ(streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyRows"), 9);
    EXPECT_EQ(
        streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyFallbacks"), 0);
    EXPECT_FALSE(hasStreamingGroupbyStat(
        task, id, cudf_velox::kStreamingGroupbyUsedStat));
  };
  run.template operator()<int32_t>();
  run.template operator()<int64_t>();
}

TEST_F(
    StreamingGroupbyAggregationTest,
    uniqueFinalFallsBackWithoutEarlyOutput) {
  gflags::FlagSaver restore;
  FLAGS_cudf_final_groupby_unique_batches = true;
  FLAGS_cudf_final_groupby_unique_min_rows = 0;
  const std::vector<std::vector<std::optional<int64_t>>> laterKeys{
      {6, 6, 7}, {6, 7, 6}, {6, std::nullopt, 8}, {4, 5, 6}, {1, 3, 5}};
  for (bool streaming : {false, true}) {
    cudf_velox::CudfConfig::getInstance().streamingGroupbyEnabled = streaming;
    for (const auto& keys : laterKeys) {
      std::vector<RowVectorPtr> vectors{
          makeRowVector(
              {makeFlatVector<int64_t>({0, 2, 4}),
               makeNullableFlatVector<int64_t>({1, std::nullopt, 3})}),
          makeRowVector(
              {makeNullableFlatVector<int64_t>(keys),
               makeNullableFlatVector<int64_t>({-2, 5, std::nullopt})})};
      createDuckDbTable(vectors);
      core::PlanNodeId id;
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 3)
              .plan(PlanBuilder()
                        .values(vectors)
                        .finalAggregation(
                            {"c0"},
                            {"min(c1)", "max(c1)"},
                            {{BIGINT()}, {BIGINT()}})
                        .capturePlanNodeId(id)
                        .planNode())
              .assertResults("SELECT c0,min(c1),max(c1) FROM tmp GROUP BY c0");
      EXPECT_EQ(
          streamingGroupbyStatSum(
              task, id, "uniqueFinalGroupbyBufferedBatches"),
          1);
      EXPECT_EQ(
          streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyFallbacks"), 1);
      EXPECT_EQ(streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyRows"), 0);
    }
  }
}

TEST_F(StreamingGroupbyAggregationTest, uniqueFinalUnsortedDistinctBatches) {
  gflags::FlagSaver restore;
  FLAGS_cudf_final_groupby_unique_batches = true;
  FLAGS_cudf_final_groupby_unique_min_rows = 0;
  std::vector<RowVectorPtr> vectors;
  for (int64_t offset : {0, 3}) {
    vectors.push_back(makeRowVector(
        {makeFlatVector<int64_t>({offset + 2, offset, offset + 1}),
         makeNullableFlatVector<int64_t>({7, std::nullopt, -9})}));
  }
  createDuckDbTable(vectors);
  core::PlanNodeId id;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 3)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .finalAggregation(
                      {"c0"}, {"min(c1)", "max(c1)"}, {{BIGINT()}, {BIGINT()}})
                  .capturePlanNodeId(id)
                  .planNode())
          .assertResults("SELECT c0,min(c1),max(c1) FROM tmp GROUP BY c0");
  EXPECT_EQ(streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyRows"), 6);
  EXPECT_EQ(
      streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyUnsortedBatches"),
      2);
  EXPECT_EQ(
      streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyFallbacks"), 0);
}

TEST_F(StreamingGroupbyAggregationTest, uniqueFinalRetainedByteLimit) {
  gflags::FlagSaver restore;
  FLAGS_cudf_final_groupby_unique_batches = true;
  FLAGS_cudf_final_groupby_unique_min_rows = 0;
  FLAGS_cudf_final_groupby_unique_max_bytes = 1 << 20;
  auto vectors = makeHighCardinalityBatches(3, 2);
  core::PlanNodeId id;
  const auto run = [&](const std::vector<RowVectorPtr>& input) {
    createDuckDbTable(input);
    return AssertQueryBuilder(duckDbQueryRunner_)
        .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 3)
        .plan(PlanBuilder()
                  .values(input)
                  .finalAggregation({"c0"}, {"min(c1)"}, {{BIGINT()}})
                  .capturePlanNodeId(id)
                  .planNode())
        .assertResults("SELECT c0,min(c1) FROM tmp GROUP BY c0");
  };
  // Use the actual retained allocation size, including alignment and masks,
  // rather than assuming a packed sizeof(value) * row count allocation.
  auto first = run({vectors.front()});
  FLAGS_cudf_final_groupby_unique_max_bytes =
      streamingGroupbyStatSum(first, id, "uniqueFinalGroupbyBufferedBytes");
  ASSERT_GT(FLAGS_cudf_final_groupby_unique_max_bytes, 0);
  auto task = run(vectors);
  EXPECT_EQ(
      streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyBufferedBatches"),
      1);
  EXPECT_EQ(
      streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyFallbacks"), 1);
  EXPECT_EQ(streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyRows"), 0);
}

TEST_F(StreamingGroupbyAggregationTest, uniqueFinalUnsupportedAggregates) {
  gflags::FlagSaver restore;
  FLAGS_cudf_final_groupby_unique_batches = true;
  FLAGS_cudf_final_groupby_unique_min_rows = 0;
  auto vectors = makeHighCardinalityBatches(3, 2);
  createDuckDbTable(vectors);
  core::PlanNodeId id;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 3)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(PlanBuilder()
                    .values(vectors)
                    .partialAggregation(
                        {"c0"}, {"min(c1)", "sum(c1)", "count(c1)", "avg(c1)"})
                    .finalAggregation()
                    .capturePlanNodeId(id)
                    .planNode())
          .assertResults(
              "SELECT c0,min(c1),sum(c1),count(c1),avg(c1) FROM tmp GROUP BY c0");
  EXPECT_EQ(
      streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyBufferedBatches"),
      0);
  EXPECT_EQ(streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyRows"), 0);
}

TEST_F(StreamingGroupbyAggregationTest, uniqueFinalPartitionedDrivers) {
  gflags::FlagSaver restore;
  FLAGS_cudf_final_groupby_unique_batches = true;
  FLAGS_cudf_final_groupby_unique_min_rows = 0;
  auto vectors = makeHighCardinalityBatches(1024, 3);
  createDuckDbTable(vectors);
  core::PlanNodeId id;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .maxDrivers(3)
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 1024)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .localPartition({"c0"})
                  .finalAggregation(
                      {"c0"}, {"min(c1)", "max(c1)"}, {{BIGINT()}, {BIGINT()}})
                  .capturePlanNodeId(id)
                  .planNode())
          .assertResults("SELECT c0,min(c1),max(c1) FROM tmp GROUP BY c0");
  EXPECT_EQ(streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyRows"), 3072);
  EXPECT_EQ(
      streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyFallbacks"), 0);
}

TEST_F(StreamingGroupbyAggregationTest, uniqueFinalSmallOrEmptyInput) {
  gflags::FlagSaver restore;
  FLAGS_cudf_final_groupby_unique_batches = true;
  FLAGS_cudf_final_groupby_unique_min_rows = 4;
  for (const int32_t rows : {0, 3}) {
    auto vectors = makeHighCardinalityBatches(rows, 1);
    createDuckDbTable(vectors);
    core::PlanNodeId id;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 3)
            .plan(PlanBuilder()
                      .values(vectors)
                      .finalAggregation({"c0"}, {"min(c1)"}, {{BIGINT()}})
                      .capturePlanNodeId(id)
                      .planNode())
            .assertResults("SELECT c0,min(c1) FROM tmp GROUP BY c0");
    EXPECT_EQ(streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyRows"), 0);
    EXPECT_EQ(
        streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyFallbacks"),
        rows == 0 ? 0 : 1);
  }
}

TEST_F(
    StreamingGroupbyAggregationTest,
    uniqueFinalFloatingKeyOrStateFallsBack) {
  gflags::FlagSaver restore;
  FLAGS_cudf_final_groupby_unique_batches = true;
  FLAGS_cudf_final_groupby_unique_min_rows = 0;
  for (bool floatingKey : {false, true}) {
    auto vector = makeRowVector(
        {makeFlatVector<double>({1.0, 2.0, 3.0}),
         makeFlatVector<int64_t>({1, 2, 3})});
    createDuckDbTable({vector});
    const auto key = floatingKey ? "c0" : "c1";
    const auto value = floatingKey ? "c1" : "c0";
    core::PlanNodeId id;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .plan(PlanBuilder()
                      .values({vector})
                      .finalAggregation(
                          {key},
                          {fmt::format("min({})", value)},
                          {{vector->childAt(floatingKey ? 1 : 0)->type()}})
                      .capturePlanNodeId(id)
                      .planNode())
            .assertResults(fmt::format(
                "SELECT {},min({}) FROM tmp GROUP BY {}", key, value, key));
    EXPECT_EQ(
        streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyBufferedBatches"),
        0);
    EXPECT_EQ(streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyRows"), 0);
  }
}

TEST_F(
    StreamingGroupbyAggregationTest,
    uniqueFinalLogicalIntegerTypesFallBack) {
  gflags::FlagSaver restore;
  FLAGS_cudf_final_groupby_unique_batches = true;
  FLAGS_cudf_final_groupby_unique_min_rows = 0;
  std::vector<RowVectorPtr> vectors{
      makeRowVector(
          {makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(12, 2)),
           makeFlatVector<int64_t>({1, 2, 3})}),
      makeRowVector(
          {makeFlatVector<int32_t>({1, 2, 3}, DATE()),
           makeFlatVector<int64_t>({1, 2, 3})}),
      makeRowVector(
          {makeFlatVector<int64_t>({1, 2, 3}),
           makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(12, 2))})};
  for (const auto& vector : vectors) {
    createDuckDbTable({vector});
    core::PlanNodeId id;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .plan(PlanBuilder()
                      .values({vector})
                      .finalAggregation(
                          {"c0"}, {"min(c1)"}, {{vector->childAt(1)->type()}})
                      .capturePlanNodeId(id)
                      .planNode())
            .assertResults("SELECT c0,min(c1) FROM tmp GROUP BY c0");
    EXPECT_EQ(
        streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyBufferedBatches"),
        0);
    EXPECT_EQ(streamingGroupbyStatSum(task, id, "uniqueFinalGroupbyRows"), 0);
  }
}

TEST_F(AggregationTest, verifiedSortedIntegerKeysAndUnsortedFallback) {
  // Keep conversion batches separate so input and compaction checks both run.
  gflags::FlagSaver flagSaver;
  FLAGS_cudf_groupby_detect_sorted_keys = true;
  FLAGS_cudf_groupby_detect_sorted_min_rows = 0;
  for (bool sorted : {false, true}) {
    for (bool masked : {false, true}) {
      for (bool ignoreNullKeys : {false, true}) {
        SCOPED_TRACE(
            fmt::format(
                "sorted={}, masked={}, ignoreNull={}",
                sorted,
                masked,
                ignoreNullKeys));
        std::vector<std::optional<int64_t>> keys = sorted
            ? std::vector<std::optional<
                  int64_t>>{std::nullopt, std::nullopt, 1, 1, 2, 3, 3}
            : std::vector<std::optional<int64_t>>{
                  2, std::nullopt, 1, 3, 1, std::nullopt, 3};
        auto data = makeRowVector(
            {"k", "v", "m"},
            {makeNullableFlatVector<int64_t>(keys),
             makeNullableFlatVector<double>(
                 {4, std::nullopt, 5, 6, std::nullopt, 8, std::nullopt}),
             makeNullableFlatVector<bool>(
                 {true, true, false, true, true, std::nullopt, true})});
        // Two individually sorted batches are not necessarily sorted when
        // concatenated during partial/intermediate aggregation.
        createDuckDbTable({data, data});
        core::PlanNodeId partialId;
        auto plan =
            PlanBuilder()
                .values({data, data})
                .aggregation(
                    {"k"},
                    {"sum(v)", "avg(v)", "count(v)", "min(v)", "max(v)"},
                    masked ? std::vector<std::string>{"m", "", "m", "m", "m"}
                           : std::vector<std::string>{},
                    core::AggregationNode::Step::kPartial,
                    ignoreNullKeys)
                .capturePlanNodeId(partialId)
                .intermediateAggregation()
                .finalAggregation()
                .planNode();
        const std::string mask = masked ? " FILTER (WHERE m)" : "";
        auto task =
            AssertQueryBuilder(plan, duckDbQueryRunner_)
                .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "7")
                .assertResults(
                    "SELECT k, sum(v)" + mask + ", avg(v), count(v)" + mask +
                    ", min(v)" + mask + ", max(v)" + mask + " FROM tmp" +
                    (ignoreNullKeys ? " WHERE k IS NOT NULL" : "") +
                    " GROUP BY k");
        EXPECT_GE(
            streamingGroupbyStatSum(task, partialId, "sortedGroupbyChecks"), 2);
        if (sorted) {
          EXPECT_GE(
              streamingGroupbyStatSum(task, partialId, "sortedGroupbyBatches"),
              2);
        }
      }
    }
  }
}

TEST_F(AggregationTest, sharedNonnullCountsChangingNullability) {
  gflags::FlagSaver flagSaver;
  FLAGS_cudf_groupby_share_nonnull_counts = true;
  auto first = makeRowVector(
      {"k", "x", "y", "m"},
      {makeNullableFlatVector<int64_t>({std::nullopt, 1, 1, 2}),
       makeFlatVector<int64_t>({1, 2, 3, 4}),
       makeFlatVector<double>({1.0, 2.0, 3.0, 4.0}),
       makeNullableFlatVector<bool>({true, false, std::nullopt, true})});
  auto second = makeRowVector(
      {"k", "x", "y", "m"},
      {makeNullableFlatVector<int64_t>({std::nullopt, 1, 1, 2}),
       makeNullableFlatVector<int64_t>({std::nullopt, 2, std::nullopt, 4}),
       makeNullableFlatVector<double>({1.0, std::nullopt, 3.0, 4.0}),
       makeNullableFlatVector<bool>({true, true, true, std::nullopt})});
  createDuckDbTable({first, second, first});
  for (bool partial : {false, true}) {
    core::PlanNodeId aggregateId;
    auto builder = PlanBuilder().values({first, second, first});
    builder.aggregation(
        {"k"},
        {"count(*)",
         "count(x)",
         "count(y)",
         "count(null)",
         "count(x)",
         "sum(x)"},
        {"", "", "", "", "m", ""},
        partial ? core::AggregationNode::Step::kPartial
                : core::AggregationNode::Step::kSingle,
        false);
    builder.capturePlanNodeId(aggregateId);
    if (partial) {
      builder.intermediateAggregation().finalAggregation();
    }
    auto task =
        AssertQueryBuilder(builder.planNode(), duckDbQueryRunner_)
            .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "4")
            .assertResults(
                "SELECT k, count(*), count(x), count(y), count(null), "
                "count(x) FILTER (WHERE m), sum(x) FROM tmp GROUP BY k");
    EXPECT_GT(
        streamingGroupbyStatSum(
            task, aggregateId, "groupbyRequestedAggregates"),
        streamingGroupbyStatSum(task, aggregateId, "groupbySubmittedRequests"));
  }
}

TEST_F(AggregationTest, completeGroupBatchesValidated) {
  gflags::FlagSaver flagSaver;
  FLAGS_cudf_groupby_complete_batches = true;
  FLAGS_cudf_groupby_detect_sorted_keys = true;
  FLAGS_cudf_groupby_detect_sorted_min_rows = 0;
  auto high = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int64_t>({10, 10, 12, 13}),
       makeNullableFlatVector<double>({1, std::nullopt, 3, std::nullopt})});
  auto low = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int64_t>({-4, -4, -2, -1}),
       makeNullableFlatVector<double>({2, 4, std::nullopt, 7})});
  // Ranges can arrive out of order; aggregate values can be null.
  createDuckDbTable({high, low});
  core::PlanNodeId aggregateId;
  auto plan =
      PlanBuilder()
          .values({high, low})
          .singleAggregation(
              {"k"}, {"sum(v)", "count(v)", "count(*)", "avg(v)"})
          .addNode([](std::string, core::PlanNodePtr node) {
            auto aggregation =
                std::dynamic_pointer_cast<const core::AggregationNode>(node);
            return core::AggregationNode::Builder(*aggregation)
                .preGroupedKeys(aggregation->groupingKeys())
                .noGroupsSpanBatches(true)
                .build();
          })
          .capturePlanNodeId(aggregateId)
          .planNode();
  auto task =
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "4")
          .assertResults(
              "SELECT k, sum(v), count(v), count(*), avg(v) FROM tmp GROUP BY k");
  EXPECT_EQ(
      streamingGroupbyStatSum(task, aggregateId, "completeGroupBatches"), 2);
}

TEST_F(AggregationTest, completeGroupBatchesRejectOverlapAndNull) {
  gflags::FlagSaver flagSaver;
  FLAGS_cudf_groupby_complete_batches = true;
  auto first = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({1, 1, 2, 3}),
       makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto overlapping = makeRowVector(
      {"k", "v"},
      {makeFlatVector<int32_t>({3, 4, 5, 6}),
       makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto nullable = makeRowVector(
      {"k", "v"},
      {makeNullableFlatVector<int32_t>({std::nullopt, 4, 5, 6}),
       makeFlatVector<int64_t>({1, 2, 3, 4})});
  for (const auto& second : {overlapping, nullable}) {
    auto plan =
        PlanBuilder()
            .values({first, second})
            .singleAggregation({"k"}, {"sum(v)"})
            .addNode([](std::string, core::PlanNodePtr node) {
              auto aggregation =
                  std::dynamic_pointer_cast<const core::AggregationNode>(node);
              return core::AggregationNode::Builder(*aggregation)
                  .preGroupedKeys(aggregation->groupingKeys())
                  .noGroupsSpanBatches(true)
                  .build();
            })
            .planNode();
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(plan)
            .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "4")
            .copyResults(pool()),
        second == nullable ? "requires non-null keys"
                           : "overlapping integer key ranges");
  }
  // Parallel Values gives each driver the same batch. Validate across drivers,
  // not only successive batches within one operator instance.
  auto plan =
      PlanBuilder()
          .values({first}, true)
          .singleAggregation({"k"}, {"sum(v)"})
          .addNode([](std::string, core::PlanNodePtr node) {
            auto aggregation =
                std::dynamic_pointer_cast<const core::AggregationNode>(node);
            return core::AggregationNode::Builder(*aggregation)
                .preGroupedKeys(aggregation->groupingKeys())
                .noGroupsSpanBatches(true)
                .build();
          })
          .planNode();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).maxDrivers(3).copyResults(pool()),
      "overlapping integer key ranges");
}

TEST_F(AggregationTest, losslessShortStringGroupingKeys) {
  gflags::FlagSaver flagSaver;
  FLAGS_cudf_groupby_pack_short_string_keys = true;
  FLAGS_cudf_groupby_pack_short_string_min_rows = 0;
  const std::string embeddedNull("x\0y", 3);
  for (bool twoKeys : {false, true}) {
    for (int fallback : {0, 1, 2}) {
      std::vector<std::optional<std::string>> keys{
          "", "A", "AB", "ABC", "é", "界", embeddedNull, "A"};
      if (fallback == 1) {
        keys[2] = std::nullopt;
      } else if (fallback == 2) {
        keys[2] = "ABCD";
      }
      auto data = makeRowVector(
          {"k", "k2", "v"},
          {makeNullableFlatVector<std::string>(keys),
           makeFlatVector<std::string>({"X", "X", "Y", "Y", "X", "Y", "", "X"}),
           makeNullableFlatVector<double>(
               {1, 2, 3, std::nullopt, 5, 6, 7, 8})});
      createDuckDbTable({data, data});
      core::PlanNodeId partialId;
      auto plan = PlanBuilder()
                      .values({data, data})
                      .partialAggregation(
                          twoKeys ? std::vector<std::string>{"k", "k2"}
                                  : std::vector<std::string>{"k"},
                          {"sum(v)", "count(v)", "avg(v)"})
                      .capturePlanNodeId(partialId)
                      .finalAggregation()
                      .planNode();
      const std::string groupKeys = twoKeys ? "k, k2" : "k";
      auto task =
          AssertQueryBuilder(plan, duckDbQueryRunner_)
              .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, "8")
              .assertResults(
                  "SELECT " + groupKeys +
                  ", sum(v), count(v), avg(v) FROM tmp GROUP BY " + groupKeys);
      EXPECT_EQ(
          streamingGroupbyStatSum(task, partialId, "packedStringGroupBatches") >
              0,
          fallback == 0);
    }
  }
}

TEST_F(AggregationTest, global) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // DM: removed "sum(15)","min(15)","max(15)",
  auto op = PlanBuilder()
                .values(vectors)
                .aggregation(
                    {},
                    {"sum(c1)",
                     "sum(c2)",
                     "sum(c4)",
                     "sum(c5)",

                     "min(c1)",
                     "min(c2)",
                     "min(c3)",
                     "min(c4)",
                     "min(c5)",

                     "max(c1)",
                     "max(c2)",
                     "max(c3)",
                     "max(c4)",
                     "max(c5)"},
                    {},
                    core::AggregationNode::Step::kPartial,
                    false)
                .planNode();

  // DM: removed sum(15), min(15), max(15),
  assertQuery(
      op,
      "SELECT sum(c1), sum(c2), sum(c4), sum(c5), "
      "min(c1), min(c2), min(c3), min(c4), min(c5), "
      "max(c1), max(c2), max(c3), max(c4), max(c5) FROM tmp");
}

TEST_F(AggregationTest, minMaxTimestampGlobal) {
  std::vector<std::optional<Timestamp>> timestamps = {
      Timestamp(1609459200, 0), // 2021-01-01 00:00:00
      Timestamp(1609459200, 500000000), // 2021-01-01 00:00:00.500
      Timestamp(1609545600, 0), // 2021-01-02 00:00:00
      std::nullopt,
      Timestamp(1609459199, 900000000) // 2020-12-31 23:59:59.900
  };

  auto data = makeRowVector(
      {makeNullableFlatVector<Timestamp>(timestamps, TIMESTAMP())});
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"min(c0)", "max(c0)"})
                  .planNode();

  assertQuery(plan, "SELECT min(c0), max(c0) FROM tmp");
}

TEST_F(AggregationTest, minMaxTimestampGroupBy) {
  std::vector<std::optional<Timestamp>> timestamps = {
      Timestamp(1609459200, 0), // 2021-01-01 00:00:00
      std::nullopt,
      Timestamp(1609545600, 0), // 2021-01-02 00:00:00
      Timestamp(1609459199, 0), // 2020-12-31 23:59:59
      Timestamp(1609632000, 0) // 2021-01-03 00:00:00
  };

  auto data = makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 2, 2, 2}),
       makeNullableFlatVector<Timestamp>(timestamps, TIMESTAMP())});
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({"c0"}, {"min(c1)", "max(c1)"})
                  .planNode();

  assertQuery(plan, "SELECT c0, min(c1), max(c1) FROM tmp GROUP BY c0");
}

TEST_F(AggregationTest, minMaxDateGlobal) {
  // cuDF represents DATE as TIMESTAMP_DAYS, a distinct type from TIMESTAMP, so
  // exercise min/max on it directly.
  std::vector<std::optional<int32_t>> dates = {
      DATE()->toDays("2021-01-01"),
      DATE()->toDays("2021-01-02"),
      std::nullopt,
      DATE()->toDays("2020-12-31"),
      DATE()->toDays("2021-01-03"),
  };

  auto data = makeRowVector({makeNullableFlatVector<int32_t>(dates, DATE())});
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"min(c0)", "max(c0)"})
                  .planNode();

  assertQuery(plan, "SELECT min(c0), max(c0) FROM tmp");
}

TEST_F(AggregationTest, minMaxDateGroupBy) {
  std::vector<std::optional<int32_t>> dates = {
      DATE()->toDays("2021-01-01"),
      std::nullopt,
      DATE()->toDays("2021-01-02"),
      DATE()->toDays("2020-12-31"),
      DATE()->toDays("2021-01-03"),
  };

  auto data = makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 2, 2, 2}),
       makeNullableFlatVector<int32_t>(dates, DATE())});
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({"c0"}, {"min(c1)", "max(c1)"})
                  .planNode();

  assertQuery(plan, "SELECT c0, min(c1), max(c1) FROM tmp GROUP BY c0");
}

TEST_F(AggregationTest, singleBigintKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<int64_t>(vectors, "c0", false, false);
  testSingleKey<int64_t>(vectors, "c0", true, false);
}

TEST_F(AggregationTest, singleBigintKeyDistinct) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<int64_t>(vectors, "c0", false, true);
  testSingleKey<int64_t>(vectors, "c0", true, true);
}

TEST_F(AggregationTest, singleStringKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<StringView>(vectors, "c6", false, false);
  testSingleKey<StringView>(vectors, "c6", true, false);
}

TEST_F(AggregationTest, singleStringKeyDistinct) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testSingleKey<StringView>(vectors, "c6", false, true);
  testSingleKey<StringView>(vectors, "c6", true, true);
}

TEST_F(AggregationTest, multiKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testMultiKey(vectors, false, false);
  testMultiKey(vectors, true, false);
}

TEST_F(AggregationTest, multiKeyDistinct) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testMultiKey(vectors, false, true);
  testMultiKey(vectors, true, true);
}

TEST_F(AggregationTest, aggregateOfNulls) {
  auto rowVector = makeRowVector({
      BatchMaker::createVector<TypeKind::BIGINT>(
          rowType_->childAt(0), 100, *pool_),
      makeNullConstant(TypeKind::SMALLINT, 100),
  });

  auto vectors = {rowVector};
  createDuckDbTable(vectors);

  auto op = PlanBuilder()
                .values(vectors)
                .aggregation(
                    {"c0"},
                    {"sum(c1)", "min(c1)", "max(c1)"},
                    {},
                    core::AggregationNode::Step::kPartial,
                    false)
                .planNode();

  assertQuery(op, "SELECT c0, sum(c1), min(c1), max(c1) FROM tmp GROUP BY c0");

  // global aggregation
  op = PlanBuilder()
           .values(vectors)
           .aggregation(
               {},
               {"sum(c1)", "min(c1)", "max(c1)"},
               {},
               core::AggregationNode::Step::kPartial,
               false)
           .planNode();

  assertQuery(op, "SELECT sum(c1), min(c1), max(c1) FROM tmp");
}

TEST_F(AggregationTest, varcharMinMax) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // Groupby with varchar min/max.
  auto op = PlanBuilder()
                .values(vectors)
                .aggregation(
                    {"c0"},
                    {"min(c6)", "max(c6)"},
                    {},
                    core::AggregationNode::Step::kPartial,
                    false)
                .planNode();

  assertQuery(op, "SELECT c0, min(c6), max(c6) FROM tmp GROUP BY c0");

  // Global aggregation with varchar min/max.
  op = PlanBuilder()
           .values(vectors)
           .aggregation(
               {},
               {"min(c6)", "max(c6)"},
               {},
               core::AggregationNode::Step::kPartial,
               false)
           .planNode();

  assertQuery(op, "SELECT min(c6), max(c6) FROM tmp");
}

TEST_F(AggregationTest, allKeyTypes) {
  // Covers different key types. Unlike the integer/string tests, the
  // hash table begins life in the generic mode, not array or
  // normalized key. Add types here as they become supported.
  auto rowType = ROW(
      {"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
      {DOUBLE(), REAL(), BIGINT(), INTEGER(), BOOLEAN(), VARCHAR(), DOUBLE()});

  std::vector<RowVectorPtr> batches;
  for (auto i = 0; i < 10; ++i) {
    batches.push_back(
        std::static_pointer_cast<RowVector>(
            BatchMaker::createBatch(rowType, 100, *pool_)));
  }
  createDuckDbTable(batches);
  auto op =
      PlanBuilder()
          .values(batches)
          .singleAggregation({"c0", "c1", "c2", "c3", "c4", "c5"}, {"sum(c6)"})
          .planNode();

  // DM: Instead of sum(c6), this was sum(1) but we don't yet support constants
  assertQuery(
      op,
      "SELECT c0, c1, c2, c3, c4, c5, sum(c6) FROM tmp "
      " GROUP BY c0, c1, c2, c3, c4, c5");
}

TEST_F(AggregationTest, ignoreNullKeys) {
  // Some keys are null.
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, 1, std::nullopt, 2, std::nullopt, 1, 2}),
      makeFlatVector<int32_t>({-1, 1, -2, 2, -3, 3, 4}),
  });

  auto makePlan = [&](bool ignoreNullKeys) {
    return PlanBuilder()
        .values({data})
        .aggregation(
            {"c0"},
            {"sum(c1)"},
            {},
            core::AggregationNode::Step::kPartial,
            ignoreNullKeys)
        .planNode();
  };

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int64_t>({4, 6}),
  });
  AssertQueryBuilder(makePlan(true)).assertResults(expected);

  expected = makeRowVector({
      makeNullableFlatVector<int32_t>({std::nullopt, 1, 2}),
      makeFlatVector<int64_t>({-6, 4, 6}),
  });
  AssertQueryBuilder(makePlan(false)).assertResults(expected);

  // All keys are null.
  data = makeRowVector({
      makeAllNullFlatVector<int32_t>(3),
      makeFlatVector<int32_t>({1, 2, 3}),
  });

  AssertQueryBuilder(makePlan(true)).assertEmptyResults();
}

TEST_F(AggregationTest, avgSingleGrouped) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // DM: removed avg(c3). We're having overflow issues with int64_t.
  std::vector<std::string> aggregates = {
      "avg(c1)", "avg(c2)", "avg(c4)", "avg(c5)"};

  std::string keyName = "c0";
  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({keyName}, aggregates)
                .planNode();

  assertQuery(
      op,
      "SELECT " + keyName + ", avg(c1), avg(c2), avg(c4), avg(c5) " +
          "FROM tmp GROUP BY " + keyName);
}

TEST_F(AggregationTest, avgPartialFinalGrouped) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  // DM: removed avg(c3). We're having overflow issues with int64_t.
  std::vector<std::string> aggregates = {
      "avg(c1)", "avg(c2)", "avg(c4)", "avg(c5)"};

  std::string keyName = "c0";
  auto op = PlanBuilder()
                .values(vectors)
                .partialAggregation({keyName}, aggregates)
                .finalAggregation()
                .planNode();

  assertQuery(
      op,
      "SELECT " + keyName + ", avg(c1), avg(c2), avg(c4), avg(c5) " +
          "FROM tmp GROUP BY " + keyName);
}

TEST_F(AggregationTest, avgSingleGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::vector<std::string> aggregates = {
      "avg(c1)", "avg(c2)", "avg(c4)", "avg(c5)"};
  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({}, aggregates)
                .planNode();

  assertQuery(op, "SELECT avg(c1), avg(c2), avg(c4), avg(c5) FROM tmp");
}

TEST_F(AggregationTest, avgPartialFinalGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::vector<std::string> aggregates = {
      "avg(c1)", "avg(c2)", "avg(c4)", "avg(c5)"};

  auto op = PlanBuilder()
                .values(vectors)
                .partialAggregation({}, aggregates)
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT avg(c1), avg(c2), avg(c4), avg(c5) FROM tmp");
}

TEST_F(AggregationTest, countStarGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);

  createDuckDbTable(vectors);

  auto op = PlanBuilder()
                .values(vectors)
                .filter("c0 > 10")
                .project({})
                .partialAggregation({}, {"count(*)"})
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT count(*) FROM tmp WHERE c0 > 10");
}

TEST_F(AggregationTest, countStarGlobalNonZeroRowsColumns) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .partialAggregation({}, {"count(*)"})
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT count(*) FROM tmp");
}

TEST_F(AggregationTest, countStarGlobalZeroRows) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .filter("c0 > 10")
                .partialAggregation({}, {"count(*)"})
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT count(*) FROM tmp WHERE c0 > 10");
}

TEST_F(AggregationTest, countStarGlobalPartialFinalZeroColumnsLocalPartition) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .filter("c0 > 0")
                  .project({})
                  .partialAggregation({}, {"count(*)"})
                  .localPartitionRoundRobin()
                  .finalAggregation()
                  .planNode();

  AssertQueryBuilder(duckDbQueryRunner_)
      .config(core::QueryConfig::kMaxLocalExchangePartitionCount, "2")
      .plan(plan)
      .assertResults("SELECT count(*) FROM tmp WHERE c0 > 0");
}

TEST_F(AggregationTest, countConstantSingleGroupByNonZeroKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c2"},
      {"count(1)"},
      "SELECT c2, count(1) FROM tmp GROUP BY c2",
      AggSteps::kSingle);
}

TEST_F(AggregationTest, countConstantPartialFinalGroupByNonZeroKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c2"},
      {"count(1)"},
      "SELECT c2, count(1) FROM tmp GROUP BY c2",
      AggSteps::kPartialFinal);
}

// Parameterized fixture that runs each count-aggregation scenario across
// single, partial+final, and partial+intermediate+final steps.
class CountAggregationStepsTest
    : public AggregationTest,
      public testing::WithParamInterface<AggregationTest::AggSteps> {};

TEST_P(CountAggregationStepsTest, countStarGlobalZeroColumns) {
  testGlobalCountStarZeroColumns(GetParam());
}

TEST_P(CountAggregationStepsTest, countStarVsCountColumnGlobalNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 2, std::nullopt}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data},
      {},
      {"count(*)", "count(c0)"},
      "SELECT count(*), count(c0) FROM tmp",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countGroupBy) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c0"},
      {"count(0)"},
      "SELECT c0, count(*) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countConstantGroupBy) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c0"},
      {"count(1)"},
      "SELECT c0, count(1) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors, {}, {"count(0)"}, "SELECT count(*) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countStarGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors, {}, {"count(*)"}, "SELECT count(*) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countStarGroupBy) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c0"},
      {"count(*)"},
      "SELECT c0, count(*) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countColumnGlobal) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors, {}, {"count(c0)"}, "SELECT count(c0) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countColumnGlobalNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 2, std::nullopt}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data}, {}, {"count(c0)"}, "SELECT count(c0) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countColumnGroupBy) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  testAggregation(
      vectors,
      {"c0"},
      {"count(c3)"},
      "SELECT c0, count(c3) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countColumnGroupByNulls) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
      makeNullableFlatVector<int64_t>(
          {10, std::nullopt, 20, std::nullopt, std::nullopt, std::nullopt}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data},
      {"c0"},
      {"count(c1)"},
      "SELECT c0, count(c1) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countStarVsCountColumnGroupByNulls) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
      makeNullableFlatVector<int64_t>(
          {10, std::nullopt, 20, std::nullopt, std::nullopt, std::nullopt}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data},
      {"c0"},
      {"count(*)", "count(c1)"},
      "SELECT c0, count(*), count(c1) FROM tmp GROUP BY c0",
      GetParam());
}

TEST_P(CountAggregationStepsTest, countNullConstantMarkerForIntersectShape) {
  auto data = makeRowVector({
      makeFlatVector<StringView>({"left_only", "left_only", "both"}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .project({
                      "true AS left_marker",
                      "cast(null AS boolean) AS right_marker",
                      "c0 AS key",
                  })
                  .partialAggregation(
                      {"key"}, {"count(left_marker)", "count(right_marker)"})
                  .finalAggregation()
                  .filter("a0 >= 1 AND a1 = 0")
                  .project({"key", "a0"})
                  .planNode();

  auto expected = makeRowVector({
      makeFlatVector<StringView>({"left_only", "both"}),
      makeFlatVector<int64_t>({2, 1}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}

TEST_P(CountAggregationStepsTest, countConstantGlobalNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, std::nullopt, 2, std::nullopt}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data}, {}, {"count(1)"}, "SELECT count(1) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countNullGlobal) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data}, {}, {"count(null)"}, "SELECT count(null) FROM tmp", GetParam());
}

TEST_P(CountAggregationStepsTest, countNullGroupBy) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
  });
  createDuckDbTable({data});
  testAggregation(
      {data},
      {"c0"},
      {"count(null)"},
      "SELECT c0, count(null) FROM tmp GROUP BY c0",
      GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    CountAggregation,
    CountAggregationStepsTest,
    testing::Values(
        AggregationTest::AggSteps::kSingle,
        AggregationTest::AggSteps::kPartialFinal,
        AggregationTest::AggSteps::kPartialIntermediateFinal),
    [](const testing::TestParamInfo<AggregationTest::AggSteps>& info)
        -> std::string {
      switch (info.param) {
        case AggregationTest::AggSteps::kSingle:
          return "Single";
        case AggregationTest::AggSteps::kPartialFinal:
          return "PartialFinal";
        case AggregationTest::AggSteps::kPartialIntermediateFinal:
          return "PartialIntermediateFinal";
      }
      return "Unknown";
    });

/// Tests the spark scenario of having different types of aggs in the same
/// planNode Specific example being tested is
/// https://github.com/facebookincubator/velox/issues/12830#issuecomment-2783340233
TEST_F(AggregationTest, companionAggs) {
  std::vector<int64_t> keys0{1, 1, 1, 2, 1, 1, 2, 2};
  std::vector<int64_t> keys1{1, 2, 1, 2, 1, 2, 1, 2};
  std::vector<int64_t> values{1, 2, 3, 4, 5, 6, 7, 8};
  auto rowVector = makeRowVector(
      {makeFlatVector<int64_t>(keys0),
       makeFlatVector<int64_t>(keys1),
       makeFlatVector<int64_t>(values)});

  createDuckDbTable({rowVector});

  auto op =
      PlanBuilder()
          .values({rowVector})
          .singleAggregation({"c2", "c0"}, {"count_partial(c1)"})
          .localPartition({"c2", "c0"})
          .singleAggregation({"c0"}, {"count_merge(a0)", "count_partial(c2)"})
          .localPartition({"c0"})
          .singleAggregation({"c0"}, {"count_merge(a0)", "count_merge(a1)"})
          .planNode();
  assertQuery(
      op, "SELECT c0, count(c1), count(distinct c2) FROM tmp GROUP BY c0");
}

TEST_F(AggregationTest, partialAggregationMemoryLimit) {
  auto vectors = {
      makeRowVector({makeFlatVector<int32_t>(
          100, [](auto row) { return row; }, nullEvery(5))}),
      makeRowVector({makeFlatVector<int32_t>(
          110, [](auto row) { return row + 29; }, nullEvery(7))}),
      makeRowVector({makeFlatVector<int32_t>(
          90, [](auto row) { return row - 71; }, nullEvery(7))}),
  };

  createDuckDbTable(vectors);

  // Set an artificially low limit on the amount of data to accumulate in
  // the partial aggregation.

  // Distinct aggregation.
  core::PlanNodeId aggNodeId;
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(QueryConfig::kMaxPartialAggregationMemory, 100)
                  .plan(
                      PlanBuilder()
                          .values(vectors)
                          .partialAggregation({"c0"}, {})
                          .capturePlanNodeId(aggNodeId)
                          .finalAggregation()
                          .planNode())
                  .assertResults("SELECT distinct c0 FROM tmp");

  auto rowFlushStats = toPlanStats(task->taskStats())
                           .at(aggNodeId)
                           .customStats.at("flushRowCount");
  EXPECT_GT(rowFlushStats.sum, 0);
  EXPECT_GT(rowFlushStats.max, 0);

  // Count aggregation.
  task = AssertQueryBuilder(duckDbQueryRunner_)
             .config(QueryConfig::kMaxPartialAggregationMemory, 1)
             .plan(
                 PlanBuilder()
                     .values(vectors)
                     .partialAggregation({"c0"}, {"count(1)"})
                     .capturePlanNodeId(aggNodeId)
                     .finalAggregation()
                     .planNode())
             .assertResults("SELECT c0, count(1) FROM tmp GROUP BY 1");

  rowFlushStats = toPlanStats(task->taskStats())
                      .at(aggNodeId)
                      .customStats.at("flushRowCount");
  EXPECT_GT(rowFlushStats.sum, 0);
  EXPECT_GT(rowFlushStats.max, 0);

  // Global aggregation.
  task = AssertQueryBuilder(duckDbQueryRunner_)
             .config(QueryConfig::kMaxPartialAggregationMemory, 1)
             .plan(
                 PlanBuilder()
                     .values(vectors)
                     .partialAggregation({}, {"sum(c0)"})
                     .capturePlanNodeId(aggNodeId)
                     .finalAggregation()
                     .planNode())
             .assertResults("SELECT sum(c0) FROM tmp");
  EXPECT_EQ(
      0,
      toPlanStats(task->taskStats())
          .at(aggNodeId)
          .customStats.count("flushRowCount"));
}

TEST_F(AggregationTest, finalAggregationStreamsOnAddInput) {
  auto vectors = {
      makeRowVector({makeFlatVector<int32_t>(
          100, [](auto row) { return row; }, nullEvery(5))}),
      makeRowVector({makeFlatVector<int32_t>(
          110, [](auto row) { return row + 29; }, nullEvery(7))}),
      makeRowVector({makeFlatVector<int32_t>(
          90, [](auto row) { return row - 71; }, nullEvery(7))}),
  };

  createDuckDbTable(vectors);

  // Force the final aggregation to see multiple addInput() calls by setting an
  // artificially low limit on the amount of data to accumulate in the partial
  // aggregation.
  core::PlanNodeId partialAggId;
  core::PlanNodeId finalAggId;
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(QueryConfig::kMaxPartialAggregationMemory, 1)
                  .plan(
                      PlanBuilder()
                          .values(vectors)
                          .partialAggregation({"c0"}, {"sum(c0)"})
                          .capturePlanNodeId(partialAggId)
                          .finalAggregation()
                          .capturePlanNodeId(finalAggId)
                          .planNode())
                  .assertResults("SELECT c0, sum(c0) FROM tmp GROUP BY 1");

  const auto planStats = toPlanStats(task->taskStats());
  EXPECT_GT(planStats.at(partialAggId).customStats.at("flushRowCount").sum, 0);
  EXPECT_GT(planStats.at(finalAggId).outputRows, 0);
}

TEST_F(AggregationTest, finalAggregationStreamingMixedAggs) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation(
                      {"c0"},
                      {"sum(c2)", "count(0)", "min(c3)", "max(c5)", "avg(c4)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .assertResults(
              "SELECT c0, sum(c2), count(*), min(c3), max(c5), avg(c4) FROM tmp GROUP BY c0");

  const auto planStats = toPlanStats(task->taskStats());
  EXPECT_GT(planStats.at(finalAggId).outputRows, 0);
}

TEST_F(AggregationTest, finalAggregationStreamingMultiKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation(
                      {"c0", "c1", "c6"},
                      {"sum(c4)", "count(0)", "avg(c5)", "max(c3)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .assertResults(
              "SELECT c0, c1, c6, sum(c4), count(*), avg(c5), max(c3) FROM tmp GROUP BY c0, c1, c6");

  const auto planStats = toPlanStats(task->taskStats());
  EXPECT_GT(planStats.at(finalAggId).outputRows, 0);
}

TEST_F(
    StreamingGroupbyAggregationTest,
    partialFinalUsesStreamingGroupbyForSupportedAggregates) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  core::PlanNodeId partialAggId;
  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation(
                      {"c0", "c6"},
                      {"sum(c2)", "count(c1)", "min(c3)", "max(c5)", "avg(c4)"})
                  .capturePlanNodeId(partialAggId)
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .assertResults(
              "SELECT c0, c6, sum(c2), count(c1), min(c3), max(c5), "
              "avg(c4) FROM tmp GROUP BY c0, c6");

  EXPECT_FALSE(hasStreamingGroupbyStat(
      task, partialAggId, cudf_velox::kStreamingGroupbyUsedStat));
  EXPECT_TRUE(hasStreamingGroupbyStat(
      task, finalAggId, cudf_velox::kStreamingGroupbyUsedStat));
}

TEST_F(StreamingGroupbyAggregationTest, rawSingleSupportedAndFallback) {
  gflags::FlagSaver restore;
  FLAGS_cudf_groupby_stream_raw_single = true;
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);
  for (const bool supported : {true, false}) {
    SCOPED_TRACE(
        supported ? "raw supported fields" : "fallback count and average");
    core::PlanNodeId aggregateId;
    const auto aggregates = supported
        ? std::vector<std::string>{"sum(c2)", "min(c3)", "max(c5)"}
        : std::vector<std::string>{"count(c1)", "avg(c4)", "count(0)"};
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 100)
            .plan(
                PlanBuilder()
                    .values(vectors)
                    .singleAggregation({"c0", "c6"}, aggregates)
                    .capturePlanNodeId(aggregateId)
                    .planNode())
            .assertResults(
                supported
                    ? "SELECT c0, c6, sum(c2), min(c3), max(c5) FROM tmp GROUP BY c0,c6"
                    : "SELECT c0, c6, count(c1), avg(c4), count(0) FROM tmp GROUP BY c0,c6");
    EXPECT_EQ(
        hasStreamingGroupbyStat(
            task, aggregateId, cudf_velox::kStreamingGroupbyUsedStat),
        supported);
  }
}

TEST_F(StreamingGroupbyAggregationTest, denseIntegerSumNullsAndPermutation) {
  gflags::FlagSaver restore;
  FLAGS_cudf_dense_integer_sum_max_range = 1024;
  FLAGS_cudf_dense_integer_sum_min_rows = 0;
  const auto run = [&]<typename Key>() {
    std::vector<RowVectorPtr> vectors;
    vectors.push_back(makeRowVector(
        {"junk", "v", "k"},
        {makeFlatVector<int64_t>({0, 0, 0, 0, 0, 0, 0}),
         makeNullableFlatVector<int64_t>(
             {1, std::nullopt, 3, 0, -2, std::nullopt, 5}),
         makeNullableFlatVector<Key>({-2, -1, std::nullopt, 1, 1, 2, 5})}));
    vectors.push_back(makeRowVector(
        {"junk", "v", "k"},
        {makeFlatVector<int64_t>({0, 0, 0, 0, 0, 0, 0}),
         makeNullableFlatVector<int64_t>(
             {4, std::nullopt, std::nullopt, 2, -5, 10, std::nullopt}),
         makeNullableFlatVector<Key>(
             {-2, -1, std::nullopt, 1, 5, 7, std::nullopt})}));
    createDuckDbTable(vectors);
    for (bool ignoreNullKeys : {false, true}) {
      core::PlanNodeId id;
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 7)
              .plan(
                  PlanBuilder()
                      .values(vectors)
                      .aggregation(
                          {"k"},
                          {"sum(v)"},
                          {},
                          core::AggregationNode::Step::kSingle,
                          ignoreNullKeys)
                      .capturePlanNodeId(id)
                      .planNode())
              .assertResults(
                  ignoreNullKeys
                      ? "SELECT k,sum(v) FROM tmp WHERE k IS NOT NULL GROUP BY k"
                      : "SELECT k,sum(v) FROM tmp GROUP BY k");
      EXPECT_GT(streamingGroupbyStatSum(task, id, "denseIntegerSumBatches"), 0);
      EXPECT_EQ(
          streamingGroupbyStatSum(task, id, "denseIntegerSumFallbacks"), 0);
    }
  };
  run.template operator()<int32_t>();
  run.template operator()<int64_t>();
}

TEST_F(
    StreamingGroupbyAggregationTest,
    denseIntegerSumFallsBackAfterPriorBatches) {
  gflags::FlagSaver restore;
  FLAGS_cudf_dense_integer_sum_max_range = 16;
  FLAGS_cudf_dense_integer_sum_min_rows = 0;
  for (int scenario = 0; scenario < 3; ++scenario) {
    std::vector<RowVectorPtr> vectors;
    const bool negative = scenario == 2;
    vectors.push_back(makeRowVector(
        {makeFlatVector<int64_t>({0, 1}),
         makeFlatVector<int64_t>(
             negative ? std::vector<int64_t>{-2, -3}
                      : std::vector<int64_t>{2, 3})}));
    vectors.push_back(makeRowVector(
        {makeFlatVector<int64_t>({scenario == 0 ? 10000 : 2}),
         makeFlatVector<int64_t>(
             {scenario == 0  ? 9
                  : negative ? std::numeric_limits<int64_t>::min()
                             : std::numeric_limits<int64_t>::max()})}));
    createDuckDbTable(vectors);
    core::PlanNodeId id;
    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 2)
                    .plan(
                        PlanBuilder()
                            .values(vectors)
                            .singleAggregation({"c0"}, {"sum(c1)"})
                            .capturePlanNodeId(id)
                            .planNode())
                    .assertResults("SELECT c0,sum(c1) FROM tmp GROUP BY c0");
    EXPECT_GT(streamingGroupbyStatSum(task, id, "denseIntegerSumBatches"), 0);
    EXPECT_EQ(streamingGroupbyStatSum(task, id, "denseIntegerSumFallbacks"), 1);
  }
}

TEST_F(
    StreamingGroupbyAggregationTest,
    denseIntegerSumSignedLimitsAndNullFirst) {
  gflags::FlagSaver restore;
  FLAGS_cudf_dense_integer_sum_max_range = 64;
  FLAGS_cudf_dense_integer_sum_min_rows = 0;
  for (auto base :
       {std::numeric_limits<int64_t>::min(),
        std::numeric_limits<int64_t>::max() - 10,
        int64_t{-3}}) {
    std::vector<RowVectorPtr> vectors{
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}),
             makeNullableFlatVector<int64_t>({std::nullopt, 3})}),
        makeRowVector(
            {makeNullableFlatVector<int64_t>({base, base + 1, std::nullopt}),
             makeNullableFlatVector<int64_t>({4, std::nullopt, -1})}),
        makeRowVector(
            {makeNullableFlatVector<int64_t>({base + 2, base}),
             makeNullableFlatVector<int64_t>({6, 5})})};
    createDuckDbTable(vectors);
    core::PlanNodeId id;
    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 2)
                    .plan(
                        PlanBuilder()
                            .values(vectors)
                            .singleAggregation({"c0"}, {"sum(c1)"})
                            .capturePlanNodeId(id)
                            .planNode())
                    .assertResults("SELECT c0,sum(c1) FROM tmp GROUP BY c0");
    EXPECT_GT(streamingGroupbyStatSum(task, id, "denseIntegerSumBatches"), 0);
    EXPECT_EQ(streamingGroupbyStatSum(task, id, "denseIntegerSumFallbacks"), 0);
  }
}

TEST_F(StreamingGroupbyAggregationTest, denseIntegerSumAllNullKeys) {
  gflags::FlagSaver restore;
  FLAGS_cudf_dense_integer_sum_max_range = 64;
  FLAGS_cudf_dense_integer_sum_min_rows = 0;
  auto input = makeRowVector(
      {makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}),
       makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt})});
  auto largeIgnored = makeRowVector(
      {makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}),
       makeNullableFlatVector<int64_t>(
           {std::numeric_limits<int64_t>::max(), std::nullopt})});
  createDuckDbTable({input, largeIgnored});
  for (bool ignore : {false, true}) {
    core::PlanNodeId id;
    auto task =
        AssertQueryBuilder(duckDbQueryRunner_)
            .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 2)
            .plan(
                PlanBuilder()
                    .values({input, largeIgnored})
                    .aggregation(
                        {"c0"},
                        {"sum(c1)"},
                        {},
                        core::AggregationNode::Step::kSingle,
                        ignore)
                    .capturePlanNodeId(id)
                    .planNode())
            .assertResults(
                ignore
                    ? "SELECT c0,sum(c1) FROM tmp WHERE c0 IS NOT NULL GROUP BY c0"
                    : "SELECT c0,sum(c1) FROM tmp GROUP BY c0");
    EXPECT_GT(streamingGroupbyStatSum(task, id, "denseIntegerSumBatches"), 0);
    EXPECT_EQ(
        streamingGroupbyStatSum(task, id, "denseIntegerSumFallbacks"),
        ignore ? 0 : 1);
  }
}

TEST_F(StreamingGroupbyAggregationTest, denseIntegerCountRowsAndNullModes) {
  gflags::FlagSaver restore;
  FLAGS_cudf_dense_integer_sum_max_range = 64;
  FLAGS_cudf_dense_integer_sum_min_rows = 0;
  FLAGS_cudf_dense_integer_count_rows = true;
  const auto run = [&]<typename Key>() {
    std::vector<RowVectorPtr> vectors{
        makeRowVector(
            {makeNullableFlatVector<Key>(
                 {std::nullopt, std::nullopt, std::nullopt, std::nullopt}),
             makeNullableFlatVector<int64_t>({0, std::nullopt, 4, 5})}),
        makeRowVector(
            {makeNullableFlatVector<Key>({-2, 0, 0, std::nullopt}),
             makeNullableFlatVector<int64_t>({std::nullopt, 2, 3, 4})}),
        makeRowVector(
            {makeNullableFlatVector<Key>({-2, 4, 7, 7}),
             makeNullableFlatVector<int64_t>({1, 2, std::nullopt, 4})})};
    createDuckDbTable(vectors);
    for (const std::string aggregate :
         {"count(*)", "count(0)", "count(c1)", "count(cast(null as bigint))"}) {
      for (bool ignore : {false, true}) {
        core::PlanNodeId id;
        auto task =
            AssertQueryBuilder(duckDbQueryRunner_)
                .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 4)
                .plan(
                    PlanBuilder()
                        .values(vectors)
                        .aggregation(
                            {"c0"},
                            {aggregate},
                            {},
                            core::AggregationNode::Step::kSingle,
                            ignore)
                        .capturePlanNodeId(id)
                        .planNode())
                .assertResults(
                    "SELECT c0," + aggregate + " FROM tmp " +
                    (ignore ? "WHERE c0 IS NOT NULL " : "") + "GROUP BY c0");
        EXPECT_EQ(
            streamingGroupbyStatSum(task, id, "denseIntegerCountBatches") > 0,
            aggregate == "count(*)" || aggregate == "count(0)");
        EXPECT_EQ(
            streamingGroupbyStatSum(task, id, "denseIntegerSumFallbacks"), 0);
      }
    }
  };
  for (uint64_t limit : {uint64_t{0}, uint64_t{0xffffffff}}) {
    FLAGS_cudf_dense_integer_count_32_max_rows = limit;
    run.template operator()<int32_t>();
    run.template operator()<int64_t>();
  }
}

TEST_F(StreamingGroupbyAggregationTest, denseIntegerCountRangeFallback) {
  gflags::FlagSaver restore;
  FLAGS_cudf_dense_integer_sum_max_range = 32;
  FLAGS_cudf_dense_integer_sum_min_rows = 0;
  FLAGS_cudf_dense_integer_count_rows = true;
  for (uint64_t limit : {uint64_t{0}, uint64_t{0xffffffff}}) {
    FLAGS_cudf_dense_integer_count_32_max_rows = limit;
    for (bool ignore : {false, true}) {
      std::vector<RowVectorPtr> vectors{
          makeRowVector(
              {makeNullableFlatVector<int64_t>({0, 1, 1, std::nullopt})}),
          makeRowVector({makeNullableFlatVector<int64_t>(
              {1, std::numeric_limits<int64_t>::max(), std::nullopt, 1})}),
          makeRowVector(
              {makeNullableFlatVector<int64_t>({0, 1, 2, std::nullopt})})};
      createDuckDbTable(vectors);
      core::PlanNodeId id;
      auto task =
          AssertQueryBuilder(duckDbQueryRunner_)
              .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 4)
              .plan(
                  PlanBuilder()
                      .values(vectors)
                      .aggregation(
                          {"c0"},
                          {"count(*)"},
                          {},
                          core::AggregationNode::Step::kSingle,
                          ignore)
                      .capturePlanNodeId(id)
                      .planNode())
              .assertResults(
                  ignore
                      ? "SELECT c0,count(*) FROM tmp WHERE c0 IS NOT NULL GROUP BY c0"
                      : "SELECT c0,count(*) FROM tmp GROUP BY c0");
      EXPECT_GT(
          streamingGroupbyStatSum(task, id, "denseIntegerCountBatches"), 0);
      EXPECT_EQ(
          streamingGroupbyStatSum(task, id, "denseIntegerSumFallbacks"), 1);
    }
  }
}

TEST_F(StreamingGroupbyAggregationTest, denseIntegerCount32RowBoundFallback) {
  gflags::FlagSaver restore;
  FLAGS_cudf_dense_integer_sum_max_range = 32;
  FLAGS_cudf_dense_integer_sum_min_rows = 0;
  FLAGS_cudf_dense_integer_count_rows = true;
  // Small limits exercise the same host-side overflow guard without billions
  // of test rows. Equality is accepted; the next batch must fall back before
  // any counters are modified. NULL counts and growth survive conversion.
  for (uint64_t limit : {uint64_t{4}, uint64_t{8}}) {
    FLAGS_cudf_dense_integer_count_32_max_rows = limit;
    std::vector<RowVectorPtr> vectors{
        makeRowVector(
            {makeNullableFlatVector<int64_t>({0, 1, 1, std::nullopt})}),
        makeRowVector(
            {makeNullableFlatVector<int64_t>({-5, 1, std::nullopt, 1})}),
        makeRowVector(
            {makeNullableFlatVector<int64_t>({0, 1, 2, std::nullopt})})};
    createDuckDbTable(vectors);
    core::PlanNodeId id;
    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 4)
                    .plan(
                        PlanBuilder()
                            .values(vectors)
                            .singleAggregation({"c0"}, {"count(*)"})
                            .capturePlanNodeId(id)
                            .planNode())
                    .assertResults("SELECT c0,count(*) FROM tmp GROUP BY c0");
    EXPECT_EQ(
        streamingGroupbyStatSum(task, id, "denseIntegerCount32Batches"),
        limit / 4);
    EXPECT_EQ(streamingGroupbyStatSum(task, id, "denseIntegerSumFallbacks"), 1);
  }
}

TEST_F(StreamingGroupbyAggregationTest, rawSingleGrowthAndNullableValues) {
  gflags::FlagSaver restore;
  FLAGS_cudf_groupby_stream_raw_single = true;
  auto vectors = makeHighCardinalityBatches(8, 8);
  vectors.push_back(makeRowVector(
      {makeNullableFlatVector<int64_t>(
           {0, 1, 2, std::nullopt, std::nullopt, 64, 65, 66}),
       makeNullableFlatVector<int64_t>(
           {std::nullopt, 4, -2, 3, std::nullopt, std::nullopt, 7, 8})}));
  createDuckDbTable(vectors);
  core::PlanNodeId aggregateId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 8)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"c0"}, {"sum(c1)", "min(c1)", "max(c1)"})
                  .capturePlanNodeId(aggregateId)
                  .planNode())
          .assertResults(
              "SELECT c0, sum(c1), min(c1), max(c1) FROM tmp GROUP BY c0");
  EXPECT_TRUE(hasStreamingGroupbyStat(
      task, aggregateId, cudf_velox::kStreamingGroupbyUsedStat));
  EXPECT_GT(
      streamingGroupbyStatSum(
          task, aggregateId, cudf_velox::kStreamingGroupbyRebuildsStat),
      0);
}

TEST_F(
    StreamingGroupbyAggregationTest,
    defaultCapacityMultiplierCoversTwoHighCardinalityBatches) {
  constexpr int32_t kBatchRows = 8;
  auto vectors = makeHighCardinalityBatches(kBatchRows, 2);
  createDuckDbTable(vectors);

  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, kBatchRows)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"c0"}, {"sum(c1)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .assertResults("SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_TRUE(hasStreamingGroupbyStat(
      task, finalAggId, cudf_velox::kStreamingGroupbyUsedStat));
  EXPECT_EQ(
      streamingGroupbyStatSum(
          task, finalAggId, cudf_velox::kStreamingGroupbyRebuildsStat),
      0);
}

TEST_F(
    StreamingGroupbyAggregationTest,
    growsBeforeInsertingAnotherHighCardinalityBatch) {
  constexpr int32_t kBatchRows = 8;
  constexpr int32_t kNumBatches = 8;
  auto vectors = makeHighCardinalityBatches(kBatchRows, kNumBatches);
  createDuckDbTable(vectors);

  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, kBatchRows)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"c0"}, {"sum(c1)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .assertResults("SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_TRUE(hasStreamingGroupbyStat(
      task, finalAggId, cudf_velox::kStreamingGroupbyUsedStat));
  EXPECT_GT(
      streamingGroupbyStatSum(
          task, finalAggId, cudf_velox::kStreamingGroupbyRebuildsStat),
      0);
}

TEST_F(
    StreamingGroupbyAggregationTest,
    configuredCapacityMultiplierAvoidsRebuilds) {
  constexpr int32_t kBatchRows = 8;
  constexpr int32_t kNumBatches = 8;
  auto vectors = makeHighCardinalityBatches(kBatchRows, kNumBatches);
  createDuckDbTable(vectors);
  cudf_velox::CudfConfig::getInstance().streamingGroupbyCapacityMultiplier =
      kNumBatches;

  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, kBatchRows)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"c0"}, {"sum(c1)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .assertResults("SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_TRUE(hasStreamingGroupbyStat(
      task, finalAggId, cudf_velox::kStreamingGroupbyUsedStat));
  EXPECT_EQ(
      streamingGroupbyStatSum(
          task, finalAggId, cudf_velox::kStreamingGroupbyRebuildsStat),
      0);
}

TEST_F(
    StreamingGroupbyAggregationTest,
    configuredCapacityMultiplierControlsGrowth) {
  constexpr int32_t kBatchRows = 8;
  constexpr int32_t kNumBatches = 5;
  auto vectors = makeHighCardinalityBatches(kBatchRows, kNumBatches);
  createDuckDbTable(vectors);
  cudf_velox::CudfConfig::getInstance().streamingGroupbyCapacityMultiplier =
      3.0;

  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, kBatchRows)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"c0"}, {"sum(c1)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .assertResults("SELECT c0, sum(c1) FROM tmp GROUP BY c0");

  EXPECT_TRUE(hasStreamingGroupbyStat(
      task, finalAggId, cudf_velox::kStreamingGroupbyUsedStat));
  EXPECT_EQ(
      streamingGroupbyStatSum(
          task, finalAggId, cudf_velox::kStreamingGroupbyRebuildsStat),
      1);
}

TEST_F(
    StreamingGroupbyAggregationTest,
    unsupportedVariableWidthMinUsesExistingGroupby) {
  auto vectors = makeVectors(rowType_, 10, 20);
  createDuckDbTable(vectors);

  core::PlanNodeId finalAggId;
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(QueryConfig::kMaxPartialAggregationMemory, 1)
                  .plan(
                      PlanBuilder()
                          .values(vectors)
                          .partialAggregation({"c0"}, {"min(c6)"})
                          .finalAggregation()
                          .capturePlanNodeId(finalAggId)
                          .planNode())
                  .assertResults("SELECT c0, min(c6) FROM tmp GROUP BY c0");

  EXPECT_FALSE(hasStreamingGroupbyStat(
      task, finalAggId, cudf_velox::kStreamingGroupbyUsedStat));
}

TEST_F(
    StreamingGroupbyAggregationTest,
    mixedSupportedAndUnsupportedAggregatesUseExistingGroupby) {
  auto vectors = makeVectors(rowType_, 10, 20);
  createDuckDbTable(vectors);

  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"c0"}, {"sum(c2)", "min(c6)"})
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .assertResults("SELECT c0, sum(c2), min(c6) FROM tmp GROUP BY c0");

  EXPECT_FALSE(hasStreamingGroupbyStat(
      task, finalAggId, cudf_velox::kStreamingGroupbyUsedStat));
}

TEST_F(
    StreamingGroupbyAggregationTest,
    separateOutputMemoryResourceUsesExistingGroupby) {
  auto vectors = makeVectors(rowType_, 10, 20);
  createDuckDbTable(vectors);
  auto& config = cudf_velox::CudfConfig::getInstance();
  config.outputMemoryResource =
      config.memoryResource == "cuda" ? "async" : "cuda";

  core::PlanNodeId finalAggId;
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(QueryConfig::kMaxPartialAggregationMemory, 1)
                  .plan(
                      PlanBuilder()
                          .values(vectors)
                          .partialAggregation({"c0"}, {"sum(c2)"})
                          .finalAggregation()
                          .capturePlanNodeId(finalAggId)
                          .planNode())
                  .assertResults("SELECT c0, sum(c2) FROM tmp GROUP BY c0");

  EXPECT_FALSE(hasStreamingGroupbyStat(
      task, finalAggId, cudf_velox::kStreamingGroupbyUsedStat));
}

TEST_F(StreamingGroupbyAggregationTest, ignoreNullKeys) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, 1, std::nullopt, 2, std::nullopt, 1, 2}),
      makeFlatVector<int32_t>({-1, 1, -2, 2, -3, 3, 4}),
  });
  createDuckDbTable({data});

  core::PlanNodeId finalAggId;
  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 2)
          .config(QueryConfig::kMaxPartialAggregationMemory, 1)
          .plan(
              PlanBuilder()
                  .values({data})
                  .aggregation(
                      {"c0"},
                      {"sum(c1)"},
                      {},
                      core::AggregationNode::Step::kPartial,
                      true)
                  .finalAggregation()
                  .capturePlanNodeId(finalAggId)
                  .planNode())
          .assertResults(
              "SELECT c0, sum(c1) FROM tmp WHERE c0 IS NOT NULL GROUP BY c0");

  EXPECT_TRUE(hasStreamingGroupbyStat(
      task, finalAggId, cudf_velox::kStreamingGroupbyUsedStat));
}

TEST_F(StreamingGroupbyAggregationTest, partialFinalAveragePreservesNaN) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 1, 1, 2, 2, 3, 3, 3}),
      makeNullableFlatVector<double>(
          {std::nan(""),
           1.0,
           3.0,
           5.0,
           std::nullopt,
           std::nullopt,
           std::nan(""),
           std::nullopt,
           7.0}),
  });
  createDuckDbTable({data});

  core::PlanNodeId finalAggId;
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 2)
                  .config(QueryConfig::kMaxPartialAggregationMemory, 1)
                  .plan(
                      PlanBuilder()
                          .values({data})
                          .partialAggregation({"c0"}, {"avg(c1)"})
                          .finalAggregation()
                          .capturePlanNodeId(finalAggId)
                          .planNode())
                  .assertResults("SELECT c0, avg(c1) FROM tmp GROUP BY c0");

  EXPECT_TRUE(hasStreamingGroupbyStat(
      task, finalAggId, cudf_velox::kStreamingGroupbyUsedStat));
}

TEST_F(
    StreamingGroupbyAggregationTest,
    partialFinalAverageOfAllNullGroupIsNull) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 2, 2}),
      makeNullableFlatVector<double>({std::nullopt, std::nullopt, 4.0, 8.0}),
  });
  createDuckDbTable({data});

  core::PlanNodeId finalAggId;
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .config(cudf_velox::CudfFromVelox::kGpuBatchSizeRows, 2)
                  .config(QueryConfig::kMaxPartialAggregationMemory, 1)
                  .plan(
                      PlanBuilder()
                          .values({data})
                          .partialAggregation({"c0"}, {"avg(c1)"})
                          .finalAggregation()
                          .capturePlanNodeId(finalAggId)
                          .planNode())
                  .assertResults("SELECT c0, avg(c1) FROM tmp GROUP BY c0");

  EXPECT_TRUE(hasStreamingGroupbyStat(
      task, finalAggId, cudf_velox::kStreamingGroupbyUsedStat));
}

class EmptyInputAggregationTest : public AggregationTest {
 protected:
  void SetUp() override {
    AggregationTest::SetUp();

    // Common test data setup
    data_ = makeRowVector({
        makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
        makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
        makeFlatVector<std::string>({"a", "b", "c", "d", "e"}),
    });

    createDuckDbTable({data_});
    filter_ = "c0 > 10"; // This filter eliminates all rows
  }

  void TearDown() override {
    // Need to clear data before plan destruction to keep memory pools happy
    data_.reset();
    plan_.reset();
    filter_.clear();
    AggregationTest::TearDown();
  }

  RowVectorPtr data_;
  core::PlanNodePtr plan_;
  std::string filter_;
};

TEST_F(EmptyInputAggregationTest, groupedSingleAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for
  // grouped aggregation
  plan_ = PlanBuilder()
              .values({data_})
              .filter(filter_)
              .singleAggregation(
                  {"c2"}, {"sum(c0)", "count(c1)", "max(c1)", "avg(c1)"})
              .planNode();

  // should return empty result for grouped aggregation
  assertQuery(
      plan_,
      "SELECT c2, sum(c0), count(c1), max(c1), avg(c1) FROM tmp WHERE c0 > 10 GROUP BY c2");
}

TEST_F(EmptyInputAggregationTest, globalSingleAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for global
  // aggregation
  plan_ =
      PlanBuilder()
          .values({data_})
          .filter(filter_)
          .singleAggregation({}, {"sum(c0)", "count(c1)", "max(c1)", "avg(c1)"})
          .planNode();

  // global aggregation should return one row with null/zero values
  assertQuery(
      plan_,
      "SELECT sum(c0), count(c1), max(c1), avg(c1) FROM tmp WHERE c0 > 10");
}

TEST_F(EmptyInputAggregationTest, distinctSingleAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for
  // distinct aggregation
  plan_ = PlanBuilder()
              .values({data_})
              .filter(filter_)
              .singleAggregation({"c2"}, {})
              .planNode();

  // should return empty result for distinct aggregation
  assertQuery(plan_, "SELECT DISTINCT c2 FROM tmp WHERE c0 > 10");
}

TEST_F(EmptyInputAggregationTest, distinctPartialFinalAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for
  // distinct partial-final aggregation
  plan_ = PlanBuilder()
              .values({data_})
              .filter(filter_)
              .partialAggregation({"c2"}, {})
              .finalAggregation()
              .planNode();

  // should return empty result for distinct aggregation
  assertQuery(plan_, "SELECT DISTINCT c2 FROM tmp WHERE c0 > 10");
}

TEST_F(EmptyInputAggregationTest, groupedPartialFinalAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for
  // partial-final aggregation
  plan_ = PlanBuilder()
              .values({data_})
              .filter(filter_)
              .partialAggregation(
                  {"c2"}, {"sum(c0)", "count(c1)", "max(c1)", "avg(c1)"})
              .finalAggregation()
              .planNode();

  // should return empty result for partial-final aggregation
  assertQuery(
      plan_,
      "SELECT c2, sum(c0), count(c1), max(c1), avg(c1) FROM tmp WHERE c0 > 10 GROUP BY c2");
}

TEST_F(EmptyInputAggregationTest, globalPartialFinalAggregation) {
  // Test case where CUDF aggregation operator receives no input rows for global
  // partial-final aggregation
  plan_ = PlanBuilder()
              .values({data_})
              .filter(filter_)
              .partialAggregation(
                  {}, {"sum(c0)", "count(c1)", "max(c1)", "avg(c1)"})
              .finalAggregation()
              .planNode();

  // global partial-final aggregation should return 1 row with null/zero values
  assertQuery(
      plan_,
      "SELECT sum(c0), count(c1), max(c1), avg(c1) FROM tmp WHERE c0 > 10");
}

TEST_F(AggregationTest, singleAggregationStreamingSumMinMax) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::string keyName = "c0";
  std::vector<std::string> aggregates = {
      "sum(c1)",
      "sum(c2)",
      "sum(c4)",
      "sum(c5)",
      "min(c1)",
      "min(c2)",
      "min(c3)",
      "min(c4)",
      "min(c5)",
      "max(c1)",
      "max(c2)",
      "max(c3)",
      "max(c4)",
      "max(c5)"};

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({keyName}, aggregates)
                .planNode();

  assertQuery(
      op,
      "SELECT " + keyName +
          ", sum(c1), sum(c2), sum(c4), sum(c5)"
          ", min(c1), min(c2), min(c3), min(c4), min(c5)"
          ", max(c1), max(c2), max(c3), max(c4), max(c5)"
          " FROM tmp GROUP BY " +
          keyName);
}

TEST_F(AggregationTest, singleAggregationStreamingAvg) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::string keyName = "c0";
  std::vector<std::string> aggregates = {
      "avg(c1)", "avg(c2)", "avg(c4)", "avg(c5)"};

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({keyName}, aggregates)
                .planNode();

  assertQuery(
      op,
      "SELECT " + keyName + ", avg(c1), avg(c2), avg(c4), avg(c5) " +
          "FROM tmp GROUP BY " + keyName);
}

TEST_F(AggregationTest, singleAggregationStreamingCount) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::string keyName = "c0";
  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({keyName}, {"count(0)"})
                .planNode();

  assertQuery(
      op, "SELECT " + keyName + ", count(*) FROM tmp GROUP BY " + keyName);
}

TEST_F(AggregationTest, singleAggregationStreamingMultiKey) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::vector<std::string> aggregates = {
      "sum(c4)",
      "sum(c5)",
      "min(c3)",
      "min(c4)",
      "min(c5)",
      "max(c3)",
      "max(c4)",
      "max(c5)"};

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({"c0", "c1", "c6"}, aggregates)
                .planNode();

  assertQuery(
      op,
      "SELECT c0, c1, c6, sum(c4), sum(c5), min(c3), min(c4), min(c5),"
      " max(c3), max(c4), max(c5) FROM tmp GROUP BY c0, c1, c6");
}

TEST_F(AggregationTest, singleAggregationStreamingMixedAggs) {
  auto vectors = makeVectors(rowType_, 10, 100);
  createDuckDbTable(vectors);

  std::string keyName = "c0";
  std::vector<std::string> aggregates = {
      "sum(c2)", "count(0)", "min(c3)", "max(c5)", "avg(c4)"};

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation({keyName}, aggregates)
                .planNode();

  assertQuery(
      op,
      "SELECT " + keyName +
          ", sum(c2), count(*), min(c3), max(c5), avg(c4)"
          " FROM tmp GROUP BY " +
          keyName);
}

TEST_F(AggregationTest, singleAggregationStreamingWithNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, 1, std::nullopt, 2, std::nullopt, 1, 2}),
      makeFlatVector<int32_t>({-1, 1, -2, 2, -3, 3, 4}),
  });

  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"sum(c1)", "min(c1)", "max(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, sum(c1), min(c1), max(c1) FROM tmp GROUP BY c0");
}

TEST_F(AggregationTest, singleAggregationStreamingIgnoreNullKeys) {
  auto data = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, 1, std::nullopt, 2, std::nullopt, 1, 2}),
      makeFlatVector<int32_t>({-1, 1, -2, 2, -3, 3, 4}),
  });

  auto op = PlanBuilder()
                .values({data})
                .aggregation(
                    {"c0"},
                    {"sum(c1)"},
                    {},
                    core::AggregationNode::Step::kSingle,
                    true)
                .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({1, 2}),
      makeFlatVector<int64_t>({4, 6}),
  });
  AssertQueryBuilder(op).assertResults(expected);
}

TEST_F(AggregationTest, singleAggregationStreamingIgnoreNullKeysAcrossBatches) {
  auto batch1 = makeRowVector({
      makeNullableFlatVector<int32_t>({1, 2, 1}),
      makeFlatVector<int32_t>({10, 20, 30}),
  });
  auto batch2 = makeRowVector({
      makeNullableFlatVector<int32_t>(
          {std::nullopt, std::nullopt, std::nullopt}),
      makeFlatVector<int32_t>({7, 8, 9}),
  });
  std::vector<RowVectorPtr> vectors{batch1, batch2};

  createDuckDbTable(vectors);

  auto op = PlanBuilder()
                .values(vectors)
                .aggregation(
                    {"c0"},
                    {"sum(c1)", "count(0)"},
                    {},
                    core::AggregationNode::Step::kSingle,
                    true)
                .planNode();

  assertQuery(
      op,
      "SELECT c0, sum(c1), count(*) FROM tmp WHERE c0 IS NOT NULL GROUP BY c0");
}

TEST_F(AggregationTest, globalApproxDistinct) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 1, 2, 3, 4, 5}),
      makeFlatVector<int32_t>({10, 20, 30, 40, 50, 10, 20, 30, 40, 50}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation(
                      {}, {"approx_distinct(c0)", "approx_distinct(c1)"})
                  .finalAggregation()
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto c0_estimate = result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);
  auto c1_estimate = result->childAt(1)->as<FlatVector<int64_t>>()->valueAt(0);

  EXPECT_GE(c0_estimate, 4);
  EXPECT_LE(c0_estimate, 6);

  EXPECT_GE(c1_estimate, 4);
  EXPECT_LE(c1_estimate, 6);
}

TEST_F(AggregationTest, globalApproxDistinctWithNulls) {
  auto data = makeRowVector({
      makeNullableFlatVector<int64_t>({1, 2, std::nullopt, 3, 4, 5, 1, 2, 3}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation({}, {"approx_distinct(c0)"})
                  .finalAggregation()
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto estimate = result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);

  EXPECT_GE(estimate, 4);
  EXPECT_LE(estimate, 6);
}

TEST_F(AggregationTest, globalApproxDistinctHighCardinality) {
  std::vector<int64_t> values;
  for (int64_t i = 0; i < 10000; ++i) {
    values.push_back(i);
  }

  auto data = makeRowVector({
      makeFlatVector<int64_t>(values),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation({}, {"approx_distinct(c0)"})
                  .finalAggregation()
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto estimate = result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);

  double error_rate =
      std::abs(static_cast<double>(estimate) - 10000.0) / 10000.0;
  EXPECT_LT(error_rate, 0.05);
}

TEST_F(AggregationTest, globalApproxDistinctEmpty) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>(std::vector<int64_t>{}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation({}, {"approx_distinct(c0)"})
                  .finalAggregation()
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto estimate = result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);

  EXPECT_EQ(estimate, 0);
}

TEST_F(AggregationTest, globalApproxDistinctPartialIntermediateFinal) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5, 1, 2, 3, 4, 5}),
      makeFlatVector<int32_t>({10, 20, 30, 40, 50, 10, 20, 30, 40, 50}),
  });

  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation(
                      {}, {"approx_distinct(c0)", "approx_distinct(c1)"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  auto result = AssertQueryBuilder(plan).copyResults(pool());

  ASSERT_EQ(result->size(), 1);
  auto c0_estimate = result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);
  auto c1_estimate = result->childAt(1)->as<FlatVector<int64_t>>()->valueAt(0);

  EXPECT_GE(c0_estimate, 4);
  EXPECT_LE(c0_estimate, 6);

  EXPECT_GE(c1_estimate, 4);
  EXPECT_LE(c1_estimate, 6);
}

TEST_F(AggregationTest, globalApproxDistinctWithNaN) {
  auto data = makeRowVector({
      makeFlatVector<double>({1.0, 2.0, std::nan(""), 4.0, std::nan(""), 1.0}),
  });

  auto planCudf = PlanBuilder()
                      .values({data})
                      .partialAggregation({}, {"approx_distinct(c0)"})
                      .finalAggregation()
                      .planNode();

  auto cudfResult = AssertQueryBuilder(planCudf).copyResults(pool());
  ASSERT_EQ(cudfResult->size(), 1);
  auto cudfEstimate =
      cudfResult->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);

  cudf_velox::unregisterCudf();
  auto planVelox = PlanBuilder()
                       .values({data})
                       .partialAggregation({}, {"approx_distinct(c0)"})
                       .finalAggregation()
                       .planNode();

  auto veloxResult = AssertQueryBuilder(planVelox).copyResults(pool());
  ASSERT_EQ(veloxResult->size(), 1);
  auto veloxEstimate =
      veloxResult->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);
  cudf_velox::registerCudf();

  EXPECT_EQ(cudfEstimate, veloxEstimate)
      << "CUDF and Velox should produce the same result for NaN values. "
      << "Expected distinct count: 3 (1.0, 2.0, 4.0) plus NaN as distinct. "
      << "CUDF result: " << cudfEstimate << ", Velox result: " << veloxEstimate;

  EXPECT_GE(cudfEstimate, 3);
  EXPECT_LE(cudfEstimate, 5);
}

// Test stddev_samp with kSingle step and grouped aggregation
TEST_F(AggregationTest, stddevSampSingleGrouped) {
  // Hand-crafted data with known expected results
  // Group 0: [1, 2, 3] -> stddev_samp = 1.0
  // Group 1: [4, 6] -> stddev_samp = sqrt(2) ≈ 1.414
  // Group 2: [10, 20, 30, 40] -> stddev_samp = sqrt(500/3) ≈ 12.909
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 0, 1, 1, 2, 2, 2, 2}), // c0 - key
      makeFlatVector<int64_t>({1, 2, 3, 4, 6, 10, 20, 30, 40}), // c1 - bigint
      makeFlatVector<double>(
          {1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0, 40.0}), // c2 - double
  });
  createDuckDbTable({data});

  // Test with bigint input
  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"stddev_samp(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  // Test with double input
  auto op2 = PlanBuilder()
                 .values({data})
                 .singleAggregation({"c0"}, {"stddev_samp(c2)"})
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Test stddev_samp with kPartial + kFinal (two-stage distributed)
TEST_F(AggregationTest, stddevSampPartialFinalGrouped) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 0, 1, 1, 2, 2, 2, 2}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 6, 10, 20, 30, 40}),
      makeFlatVector<double>({1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0, 40.0}),
  });
  createDuckDbTable({data});

  // Test with bigint input
  auto op = PlanBuilder()
                .values({data})
                .partialAggregation({"c0"}, {"stddev_samp(c1)"})
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  // Test with double input
  auto op2 = PlanBuilder()
                 .values({data})
                 .partialAggregation({"c0"}, {"stddev_samp(c2)"})
                 .finalAggregation()
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Test stddev_samp with kPartial + kIntermediate + kFinal (three-stage)
TEST_F(AggregationTest, stddevSampPartialIntermediateFinalGrouped) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 0, 1, 1, 2, 2, 2, 2}),
      makeFlatVector<int64_t>({1, 2, 3, 4, 6, 10, 20, 30, 40}),
      makeFlatVector<double>({1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0, 40.0}),
  });
  createDuckDbTable({data});

  // Test with bigint input
  auto op = PlanBuilder()
                .values({data})
                .partialAggregation({"c0"}, {"stddev_samp(c1)"})
                .intermediateAggregation()
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  // Test with double input
  auto op2 = PlanBuilder()
                 .values({data})
                 .partialAggregation({"c0"}, {"stddev_samp(c2)"})
                 .intermediateAggregation()
                 .finalAggregation()
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Test stddev_samp with NULL values in input
TEST_F(AggregationTest, stddevSampWithNulls) {
  // Group 0: [1, NULL, 3] -> should compute stddev of [1, 3] = sqrt(2) ≈ 1.414
  // Group 1: [4, 6, NULL] -> should compute stddev of [4, 6] = sqrt(2) ≈ 1.414
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 0, 1, 1, 1}),
      makeNullableFlatVector<int64_t>({1, std::nullopt, 3, 4, 6, std::nullopt}),
      makeNullableFlatVector<double>(
          {1.0, std::nullopt, 3.0, 4.0, 6.0, std::nullopt}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"stddev_samp(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  auto op2 = PlanBuilder()
                 .values({data})
                 .singleAggregation({"c0"}, {"stddev_samp(c2)"})
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Test stddev_samp with single value per group (should return NULL)
TEST_F(AggregationTest, stddevSampSingleValueGroup) {
  // Each group has only one value -> stddev_samp should return NULL
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2}),
      makeFlatVector<int64_t>({10, 20, 30}),
      makeFlatVector<double>({10.0, 20.0, 30.0}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"stddev_samp(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  auto op2 = PlanBuilder()
                 .values({data})
                 .singleAggregation({"c0"}, {"stddev_samp(c2)"})
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Test stddev_samp with all NULL input (should return NULL)
TEST_F(AggregationTest, stddevSampAllNulls) {
  // Group 0: all NULLs -> stddev_samp should return NULL
  // Group 1: has values -> should compute normally
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 1, 1}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 1, 2}),
      makeNullableFlatVector<double>({std::nullopt, std::nullopt, 1.0, 2.0}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"stddev_samp(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, stddev_samp(c1) FROM tmp GROUP BY c0");

  auto op2 = PlanBuilder()
                 .values({data})
                 .singleAggregation({"c0"}, {"stddev_samp(c2)"})
                 .planNode();

  assertQuery(op2, "SELECT c0, stddev_samp(c2) FROM tmp GROUP BY c0");
}

// Masked count(col) + unmasked sum, single-stage (the reproducer shape).
TEST_F(AggregationTest, maskedCountAndSumGrouped) {
  auto data = makeRowVector(
      {"k", "v", "m"},
      {makeFlatVector<int64_t>({1, 1, 2, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
       makeFlatVector<bool>({true, false, true, true, false})});
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({"k"}, {"count(v)", "sum(v)"}, {"m"})
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT k, count(v) FILTER (WHERE m), sum(v) FROM tmp GROUP BY k");
}

// Multi-stage masked count (partial + final).
TEST_F(AggregationTest, maskedCountMultiStage) {
  auto data = makeRowVector(
      {"k", "v", "m"},
      {makeFlatVector<int64_t>({1, 1, 2, 2, 3}),
       makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
       makeFlatVector<bool>({true, false, true, true, false})});
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation({"k"}, {"count(v)", "sum(v)"}, {"m"})
                  .finalAggregation()
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT k, count(v) FILTER (WHERE m), sum(v) FROM tmp GROUP BY k");
}

// Masked count(*) counts mask-true rows; includes a null mask entry.
TEST_F(AggregationTest, maskedCountStar) {
  auto data = makeRowVector(
      {"k", "m"},
      {makeFlatVector<int64_t>({1, 1, 1, 2, 2}),
       makeNullableFlatVector<bool>(
           {true, false, std::nullopt, false, false})});
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({"k"}, {"count(1)"}, {"m"})
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT k, count(*) FILTER (WHERE m) FROM tmp GROUP BY k");
}

// Global masked count(*) (no GROUP BY) -> reduce count-all branch with a mask,
// including a null mask entry (treated as false).
TEST_F(AggregationTest, maskedCountStarGlobal) {
  auto data = makeRowVector(
      {"m"},
      {makeNullableFlatVector<bool>({true, false, std::nullopt, true, false})});
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"count(1)"}, {"m"})
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT count(*) FILTER (WHERE m) FROM tmp");
}

// Global masked count(*), multi-stage (partial + final), no GROUP BY.
TEST_F(AggregationTest, maskedCountStarGlobalMultiStage) {
  auto data = makeRowVector(
      {"m"},
      {makeNullableFlatVector<bool>({true, false, std::nullopt, true, false})});
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation({}, {"count(1)"}, {"m"})
                  .finalAggregation()
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT count(*) FILTER (WHERE m) FROM tmp");
}

// Fully-masked-out group -> sum NULL, count 0.
TEST_F(AggregationTest, maskedAllExcludedGroup) {
  auto data = makeRowVector(
      {"k", "v", "m"},
      {makeFlatVector<int64_t>({1, 1, 2, 2}),
       makeFlatVector<int64_t>({10, 20, 30, 40}),
       makeFlatVector<bool>({false, false, true, true})});
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({"k"}, {"sum(v)", "count(v)"}, {"m", "m"})
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT k, sum(v) FILTER (WHERE m), count(v) FILTER (WHERE m) "
          "FROM tmp GROUP BY k");
}

// NULL mask values behave as false (excluded), on count(v) + sum(v).
TEST_F(AggregationTest, maskedNullMaskExcludes) {
  auto data = makeRowVector(
      {"k", "v", "m"},
      {makeFlatVector<int64_t>({1, 1, 1}),
       makeFlatVector<int64_t>({10, 20, 30}),
       makeNullableFlatVector<bool>({true, std::nullopt, false})});
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({"k"}, {"sum(v)", "count(v)"}, {"m", "m"})
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT k, sum(v) FILTER (WHERE m), count(v) FILTER (WHERE m) "
          "FROM tmp GROUP BY k");
}

// Global (reduce) masked sum/count incl. all-excluded -> sum NULL, count 0.
TEST_F(AggregationTest, maskedGlobalReduce) {
  auto data = makeRowVector(
      {"v", "m"},
      {makeFlatVector<int64_t>({10, 20, 30}),
       makeFlatVector<bool>({false, false, false})});
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"sum(v)", "count(v)"}, {"m", "m"})
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT sum(v) FILTER (WHERE m), count(v) FILTER (WHERE m) FROM tmp");
}

// Masked min/max incl. varchar and a fully-masked group.
TEST_F(AggregationTest, maskedMinMaxVarchar) {
  auto data = makeRowVector(
      {"k", "s", "m"},
      {makeFlatVector<int64_t>({1, 1, 2, 2}),
       makeFlatVector<std::string>({"b", "a", "c", "d"}),
       makeFlatVector<bool>({true, false, false, false})});
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({"k"}, {"min(s)", "max(s)"}, {"m", "m"})
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT k, min(s) FILTER (WHERE m), max(s) FILTER (WHERE m) "
          "FROM tmp GROUP BY k");
}

// Masked groupby across multiple input batches, so the cross-batch reuse of
// maskedValues_/maskedCount_ actually runs: groups and mask true/false/null
// rows straddle batch boundaries.
TEST_F(AggregationTest, maskedGroupbyMultiBatch) {
  auto batch1 = makeRowVector(
      {"k", "v", "m"},
      {makeFlatVector<int64_t>({1, 2, 1}),
       makeFlatVector<int64_t>({10, 20, 30}),
       makeNullableFlatVector<bool>({true, false, std::nullopt})});
  auto batch2 = makeRowVector(
      {"k", "v", "m"},
      {makeFlatVector<int64_t>({2, 1, 2}),
       makeFlatVector<int64_t>({40, 50, 60}),
       makeFlatVector<bool>({true, true, false})});
  createDuckDbTable({batch1, batch2});
  auto plan = PlanBuilder()
                  .values({batch1, batch2})
                  .singleAggregation({"k"}, {"sum(v)", "count(v)"}, {"m", "m"})
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT k, sum(v) FILTER (WHERE m), count(v) FILTER (WHERE m) "
          "FROM tmp GROUP BY k");
}

// Global masked min/max (no GROUP BY) with an all-false mask: every row is
// excluded, so both aggregates return NULL (the all-excluded reduce path).
TEST_F(AggregationTest, maskedMinMaxGlobalAllExcluded) {
  auto data = makeRowVector(
      {"v", "m"},
      {makeFlatVector<int64_t>({10, 20, 30, 40}),
       makeFlatVector<bool>({false, false, false, false})});
  createDuckDbTable({data});
  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"min(v)", "max(v)"}, {"m", "m"})
                  .planNode();
  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT min(v) FILTER (WHERE m), max(v) FILTER (WHERE m) FROM tmp");
}

// Test avg with all NULL input (should return NULL, not NaN)
TEST_F(AggregationTest, avgAllNulls) {
  // Group 0: all NULLs -> avg should return NULL
  // Group 1: has values -> should compute normally
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 1, 1}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 4, 6}),
      makeNullableFlatVector<double>({std::nullopt, std::nullopt, 4.0, 6.0}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"avg(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, avg(c1) FROM tmp GROUP BY c0");

  auto op2 = PlanBuilder()
                 .values({data})
                 .singleAggregation({"c0"}, {"avg(c2)"})
                 .planNode();

  assertQuery(op2, "SELECT c0, avg(c2) FROM tmp GROUP BY c0");
}

// Test avg with all NULL input using partial + final (distributed) aggregation
TEST_F(AggregationTest, avgAllNullsPartialFinal) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 1, 1}),
      makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt, 4, 6}),
      makeNullableFlatVector<double>({std::nullopt, std::nullopt, 4.0, 6.0}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .partialAggregation({"c0"}, {"avg(c1)"})
                .finalAggregation()
                .planNode();

  assertQuery(op, "SELECT c0, avg(c1) FROM tmp GROUP BY c0");

  auto op2 = PlanBuilder()
                 .values({data})
                 .partialAggregation({"c0"}, {"avg(c2)"})
                 .finalAggregation()
                 .planNode();

  assertQuery(op2, "SELECT c0, avg(c2) FROM tmp GROUP BY c0");
}

// Test avg with NaN inputs preserves NaN (does not convert to NULL)
TEST_F(AggregationTest, avgNaNInputs) {
  // Group 0: NaN only -> avg should be NaN
  // Group 1: normal values -> avg should compute normally
  // Group 2: all NULLs (count == 0) -> avg should be NULL
  // Group 3: NaN + NULL + normal -> avg should be NaN
  auto data = makeRowVector({
      makeFlatVector<int64_t>({0, 0, 1, 1, 2, 2, 3, 3, 3}),
      makeNullableFlatVector<double>(
          {std::nan(""),
           1.0,
           3.0,
           5.0,
           std::nullopt,
           std::nullopt,
           std::nan(""),
           std::nullopt,
           7.0}),
  });
  createDuckDbTable({data});

  auto op = PlanBuilder()
                .values({data})
                .singleAggregation({"c0"}, {"avg(c1)"})
                .planNode();

  assertQuery(op, "SELECT c0, avg(c1) FROM tmp GROUP BY c0");
}

// Test that zero-column rows flow correctly through CudfFromVelox.
// project({}) produces zero-column output; localPartitionRoundRobin is a CPU
// operator that forces CudfFromVelox insertion before the GPU aggregation.
// Without the zero-column fix in CudfFromVelox, this crashes with:
//   "Operator::getOutput() must return nullptr or a non-empty vector"
// because toCudfTable loses the row count for zero-column tables.
TEST_F(AggregationTest, zeroColumnThroughCudfFromVelox) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4}),
  });
  createDuckDbTable({data});

  auto plan = PlanBuilder()
                  .values({data})
                  .filter("c0 > 0")
                  .project({})
                  .localPartitionRoundRobin()
                  .singleAggregation({}, {"count(*)"})
                  .planNode();

  AssertQueryBuilder(duckDbQueryRunner_)
      .config(core::QueryConfig::kMaxLocalExchangePartitionCount, "2")
      .plan(plan)
      .assertResults("SELECT count(*) FROM tmp WHERE c0 > 0");
}

} // namespace facebook::velox::exec::test
