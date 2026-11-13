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
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <folly/stop_watch.h>

namespace facebook::velox::cudf_velox::exec::test {
namespace {

using facebook::velox::exec::DriverStats;
using facebook::velox::exec::PlanNodeStats;
using facebook::velox::exec::TableScan;
using facebook::velox::exec::toPlanStats;
using facebook::velox::exec::test::AssertQueryBuilder;
using facebook::velox::exec::test::PlanBuilder;

// Tests dynamic filters that a cuDF hash join makes for a cuDF table scan
// outside of its pipeline.
class TaskDynamicFilterTest : public CudfHiveConnectorTestBase {
 protected:
  static constexpr int32_t kNumFiles = 4;
  static constexpr int32_t kRowsPerFile = 1'000;
  static constexpr int32_t kNumRows = kNumFiles * kRowsPerFile;
  static constexpr std::string_view kWaitForever = "600000";

  static void SetUpTestCase() {
    CudfHiveConnectorTestBase::SetUpTestCase();
    parquet::registerParquetReaderFactory();
  }

  static void TearDownTestCase() {
    parquet::unregisterParquetReaderFactory();
    CudfHiveConnectorTestBase::TearDownTestCase();
  }

  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
    cudf_velox::registerPrestoFunctions(
        cudf_velox::CudfConfig::getInstance().functionNamePrefix);

    // File i has the keys [i * 1'000, (i + 1) * 1'000) in 'k', twice the key
    // in 'v' and the key as a string in 's'.
    std::vector<RowVectorPtr> vectors;
    for (int32_t i = 0; i < kNumFiles; ++i) {
      const int64_t start = i * kRowsPerFile;
      vectors.push_back(makeRowVector(
          {"k", "v", "s"},
          {
              makeFlatVector<int64_t>(
                  kRowsPerFile, [&](auto row) { return start + row; }),
              makeFlatVector<int64_t>(
                  kRowsPerFile, [&](auto row) { return 2 * (start + row); }),
              makeFlatVector<std::string>(
                  kRowsPerFile,
                  [&](auto row) { return std::to_string(start + row); }),
          }));
      files_.push_back(TempFilePath::create());
      writeToFile(files_.back()->getPath(), vectors.back());
    }
    createDuckDbTable("probe", vectors);
  }

  // A join whose probe side is a cuDF scan of the probe table behind a local
  // exchange, so that the scan is not in the pipeline of the join.
  struct JoinPlan {
    core::PlanNodePtr plan;
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
  };

  JoinPlan makeJoinPlan(
      const RowVectorPtr& build,
      core::JoinType joinType = core::JoinType::kInner,
      const std::string& probeKey = "k",
      const std::string& buildKey = "bk") {
    createDuckDbTable("build", {build});
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    JoinPlan plan;
    plan.plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .startTableScan()
                    .connectorId(kCudfHiveConnectorId)
                    .outputType(probeType_)
                    .dataColumns(probeType_)
                    .assignments(
                        facebook::velox::exec::test::HiveConnectorTestBase::
                            allRegularColumns(probeType_))
                    .endTableScan()
                    .capturePlanNodeId(plan.scanId)
                    .localPartition({probeKey})
                    .hashJoin(
                        {probeKey},
                        {buildKey},
                        PlanBuilder(planNodeIdGenerator, pool_.get())
                            .values({build})
                            .planNode(),
                        "",
                        {"k", "v"},
                        joinType)
                    .capturePlanNodeId(plan.joinId)
                    .planNode();
    return plan;
  }

  // Runs 'plan' with the filters the task derives from it if 'enabled', and
  // returns the statistics by plan node.
  std::unordered_map<core::PlanNodeId, PlanNodeStats>
  run(const JoinPlan& plan, const std::string& duckDbSql, bool enabled = true) {
    auto task = AssertQueryBuilder(plan.plan, duckDbQueryRunner_)
                    .maxDrivers(4)
                    .config(
                        core::QueryConfig::kCrossPipelineDynamicFilterEnabled,
                        enabled ? "true" : "false")
                    .config(
                        core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
                        std::string(kWaitForever))
                    .splits(plan.scanId, makeCudfHiveConnectorSplits(files_))
                    .assertResults(duckDbSql);
    auto stats = toPlanStats(task->taskStats());
    // The join under test is the cuDF one.
    EXPECT_EQ(
        stats.at(plan.joinId).operatorStats.count("CudfHashJoinBuild"), 1);
    return stats;
  }

  static int64_t statSum(const PlanNodeStats& stats, std::string_view name) {
    const auto it = stats.customStats.find(std::string(name));
    return it == stats.customStats.end() ? 0 : it->second.sum;
  }

  RowVectorPtr makeBuild(const std::vector<std::optional<int64_t>>& keys) {
    return makeRowVector({"bk"}, {makeNullableFlatVector<int64_t>(keys)});
  }

  const RowTypePtr probeType_{
      ROW({{"k", BIGINT()}, {"v", BIGINT()}, {"s", VARCHAR()}})};
  const std::string joinSql_{"SELECT k, v FROM probe, build WHERE k = bk"};
  std::vector<std::shared_ptr<TempFilePath>> files_;
};

TEST_F(TaskDynamicFilterTest, localExchange) {
  auto plan = makeJoinPlan(makeBuild({5, 1'500, 1'501, 3'999, 10'000}));
  {
    // The filters of a pipeline stop at the local exchange.
    const auto stats = run(plan, joinSql_, false);
    EXPECT_EQ(stats.at(plan.scanId).outputRows, kNumRows);
  }
  {
    const auto stats = run(plan, joinSql_);
    const auto& scanStats = stats.at(plan.scanId);
    EXPECT_EQ(scanStats.outputRows, 4);
    EXPECT_GT(statSum(scanStats, DriverStats::kDynamicFiltersAccepted), 0);
    EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 0);
    EXPECT_EQ(
        statSum(stats.at(plan.joinId), DriverStats::kDynamicFiltersProduced),
        1);
  }
}

TEST_F(TaskDynamicFilterTest, semiJoin) {
  auto plan =
      makeJoinPlan(makeBuild({5, 5, 1'500}), core::JoinType::kLeftSemiFilter);
  const auto stats =
      run(plan, "SELECT k, v FROM probe WHERE k IN (SELECT bk FROM build)");
  EXPECT_EQ(stats.at(plan.scanId).outputRows, 2);
  EXPECT_EQ(
      statSum(stats.at(plan.scanId), TableScan::kDynamicFiltersTimedOut), 0);
}

TEST_F(TaskDynamicFilterTest, emptyBuildAndNullKeys) {
  {
    // No build key matches anything.
    auto plan = makeJoinPlan(makeBuild({std::nullopt, std::nullopt}));
    const auto stats = run(plan, joinSql_);
    EXPECT_EQ(stats.at(plan.scanId).outputRows, 0);
  }
  {
    auto plan = makeJoinPlan(makeBuild({7, std::nullopt, 2'500}));
    const auto stats = run(plan, joinSql_);
    EXPECT_EQ(stats.at(plan.scanId).outputRows, 2);
  }
}

TEST_F(TaskDynamicFilterTest, largeBuildWithFewKeys) {
  // More build rows than an exact set of values holds, but few distinct keys:
  // the scan gets the exact set, [1'000, 1'999].
  constexpr int32_t kNumBuildRows = 150'000;
  auto build = makeRowVector(
      {"bk"}, {makeFlatVector<int64_t>(kNumBuildRows, [](auto row) {
        return 1'000 + row % 1'000;
      })});
  auto plan = makeJoinPlan(build, core::JoinType::kLeftSemiFilter);
  const auto stats =
      run(plan, "SELECT k, v FROM probe WHERE k IN (SELECT bk FROM build)");
  const auto& scanStats = stats.at(plan.scanId);
  EXPECT_EQ(scanStats.outputRows, 1'000);
  EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 0);
}

TEST_F(TaskDynamicFilterTest, largeBuildWithDenseKeys) {
  // More distinct keys than an exact set of values holds, and too dense for a
  // bitmap to leave much out: every number below 150'000. Only the bounds of
  // the keys are known, which the join does not send. The scan is released
  // and reads everything.
  constexpr int32_t kNumBuildRows = 150'000;
  auto build = makeRowVector(
      {"bk"},
      {makeFlatVector<int64_t>(kNumBuildRows, [](auto row) { return row; })});
  auto plan = makeJoinPlan(build);
  const auto stats = run(plan, joinSql_);
  const auto& scanStats = stats.at(plan.scanId);
  EXPECT_EQ(scanStats.outputRows, kNumRows);
  EXPECT_EQ(statSum(scanStats, DriverStats::kDynamicFiltersAccepted), 0);
  EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 0);
}

TEST_F(TaskDynamicFilterTest, largeBuildWithSparseKeys) {
  // More distinct keys than an exact set of values holds: the even numbers
  // below 300'000. The range of the keys covers the whole probe table. The
  // bitmap of the keys that goes with it on the device leaves the odd keys
  // out.
  constexpr int32_t kNumBuildRows = 150'000;
  auto build = makeRowVector(
      {"bk"}, {makeFlatVector<int64_t>(kNumBuildRows, [](auto row) {
        return 2 * row;
      })});
  auto plan = makeJoinPlan(build);
  const auto stats = run(plan, joinSql_);
  const auto& scanStats = stats.at(plan.scanId);
  EXPECT_EQ(scanStats.outputRows, kNumRows / 2);
  EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 0);
}

TEST_F(TaskDynamicFilterTest, joinsWithoutFilter) {
  // The scan must not wait out its limit for a filter the join does not make.
  const folly::stop_watch<std::chrono::seconds> watch;
  {
    // A left join keeps the probe rows without a match. Declared by hand: the
    // planner does not place a filter here.
    auto plan = makeJoinPlan(makeBuild({5, 1'500}), core::JoinType::kLeft);
    auto task =
        AssertQueryBuilder(plan.plan, duckDbQueryRunner_)
            .maxDrivers(4)
            .dynamicFilters(
                {{.filterId = "df", .joinNodeId = plan.joinId, .keyIndex = 0}},
                {{.filterId = "df",
                  .scanNodeId = plan.scanId,
                  .outputChannel = 0}})
            .config(
                core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
                std::string(kWaitForever))
            .splits(plan.scanId, makeCudfHiveConnectorSplits(files_))
            .assertResults("SELECT k, v FROM probe LEFT JOIN build ON k = bk");
    const auto stats = toPlanStats(task->taskStats());
    const auto& scanStats = stats.at(plan.scanId);
    EXPECT_EQ(scanStats.outputRows, kNumRows);
    EXPECT_EQ(statSum(scanStats, DriverStats::kDynamicFiltersAccepted), 0);
    EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 0);
  }
  {
    // A string key. Also declared by hand.
    auto build = makeRowVector(
        {"bs"}, {makeFlatVector<std::string>({"5", "1500", "no"})});
    auto plan = makeJoinPlan(build, core::JoinType::kInner, "s", "bs");
    auto task =
        AssertQueryBuilder(plan.plan, duckDbQueryRunner_)
            .maxDrivers(4)
            .dynamicFilters(
                {{.filterId = "df", .joinNodeId = plan.joinId, .keyIndex = 0}},
                {{.filterId = "df",
                  .scanNodeId = plan.scanId,
                  .outputChannel = 2}})
            .config(
                core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
                std::string(kWaitForever))
            .splits(plan.scanId, makeCudfHiveConnectorSplits(files_))
            .assertResults("SELECT k, v FROM probe, build WHERE s = bs");
    const auto stats = toPlanStats(task->taskStats());
    const auto& scanStats = stats.at(plan.scanId);
    EXPECT_EQ(scanStats.outputRows, kNumRows);
    EXPECT_EQ(statSum(scanStats, DriverStats::kDynamicFiltersAccepted), 0);
    EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 0);
  }
  EXPECT_LT(watch.elapsed().count(), 120);
}

TEST_F(TaskDynamicFilterTest, groupedExecution) {
  // Each split group builds its own tables from a part of the build side. The
  // join opts out and the scans read everything.
  auto build = makeBuild({5, 1'500, 10'000});
  createDuckDbTable("build", {build});
  auto buildFile = TempFilePath::create();
  writeToFile(buildFile->getPath(), build);
  const auto buildType = asRowType(build->type());

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId scanId;
  core::PlanNodeId buildScanId;
  core::PlanNodeId joinId;
  const auto cudfScan = [&](const RowTypePtr& type, core::PlanNodeId& id) {
    return PlanBuilder(planNodeIdGenerator, pool_.get())
        .startTableScan()
        .connectorId(kCudfHiveConnectorId)
        .outputType(type)
        .dataColumns(type)
        .assignments(
            facebook::velox::exec::test::HiveConnectorTestBase::
                allRegularColumns(type))
        .endTableScan()
        .capturePlanNodeId(id);
  };
  auto plan = cudfScan(probeType_, scanId)
                  .localPartition({"k"})
                  .hashJoin(
                      {"k"},
                      {"bk"},
                      cudfScan(buildType, buildScanId).planNode(),
                      "",
                      {"k", "v"})
                  .capturePlanNodeId(joinId)
                  .planNode();

  constexpr int32_t kNumGroups = 2;
  core::PlanFragment fragment{plan};
  fragment.executionStrategy = core::ExecutionStrategy::kGrouped;
  fragment.groupedExecutionLeafNodeIds = {scanId, buildScanId};
  fragment.numSplitGroups = kNumGroups;
  fragment.dynamicFilterSources = {
      {.filterId = "df", .joinNodeId = joinId, .keyIndex = 0}};
  fragment.dynamicFilterTargets = {
      {.filterId = "df", .scanNodeId = scanId, .outputChannel = 0}};

  std::atomic<vector_size_t> numRows{0};
  auto queryCtx = core::QueryCtx::create(
      executor_.get(),
      core::QueryConfig(
          {{core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
            std::string(kWaitForever)}}));
  auto task = facebook::velox::exec::Task::create(
      "grouped",
      std::move(fragment),
      0,
      std::move(queryCtx),
      facebook::velox::exec::Task::ExecutionMode::kParallel,
      [&](RowVectorPtr vector, bool, ContinueFuture*) {
        if (vector != nullptr) {
          numRows += vector->size();
        }
        return facebook::velox::exec::BlockingReason::kNotBlocked;
      });
  task->start(2, 1);
  for (int32_t group = 0; group < kNumGroups; ++group) {
    for (auto& split : makeCudfHiveConnectorSplits(files_)) {
      task->addSplit(
          scanId, facebook::velox::exec::Split(std::move(split), group));
    }
    for (auto& split : makeCudfHiveConnectorSplits({buildFile})) {
      task->addSplit(
          buildScanId, facebook::velox::exec::Split(std::move(split), group));
    }
    task->noMoreSplitsForGroup(scanId, group);
    task->noMoreSplitsForGroup(buildScanId, group);
  }
  task->noMoreSplits(scanId);
  task->noMoreSplits(buildScanId);
  ASSERT_TRUE(
      facebook::velox::exec::test::waitForTaskCompletion(
          task.get(), 60'000'000));

  // Keys 5 and 1'500 match in each group.
  EXPECT_EQ(numRows, 2 * kNumGroups);
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(stats.at(joinId).operatorStats.count("CudfHashJoinBuild"), 1);
  const auto& scanStats = stats.at(scanId);
  EXPECT_EQ(scanStats.outputRows, kNumRows * kNumGroups);
  EXPECT_EQ(statSum(scanStats, DriverStats::kDynamicFiltersAccepted), 0);
  EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 0);
}

DEBUG_ONLY_TEST_F(TaskDynamicFilterTest, failureWhileBuilding) {
  // The join build fails once its input is complete. The filter is resolved
  // on the way out and the query fails with the error, without the scan
  // waiting out its limit.
  SCOPED_TESTVALUE_SET(
      "facebook::velox::cudf_velox::CudfHashJoinBuild::buildAndPublish",
      std::function<void(facebook::velox::exec::Operator*)>(
          [](facebook::velox::exec::Operator*) {
            VELOX_FAIL("Injected build failure");
          }));
  auto plan = makeJoinPlan(makeBuild({5, 1'500}));
  const folly::stop_watch<std::chrono::seconds> watch;
  VELOX_ASSERT_THROW(run(plan, joinSql_), "Injected build failure");
  EXPECT_LT(watch.elapsed().count(), 60);
}

TEST_F(TaskDynamicFilterTest, filterOfWrongType) {
  // A filter delivered from outside of the task that does not fit the column
  // fails for the one who delivers it, before it reaches the scan.
  core::PlanNodeId scanId;
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(probeType_)
                  .dataColumns(probeType_)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(probeType_))
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .project({"k", "v"})
                  .planNode();
  auto task =
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .maxDrivers(2)
          .dynamicFilters(
              {},
              {{.filterId = "wrong", .scanNodeId = scanId, .outputChannel = 0},
               {.filterId = "right", .scanNodeId = scanId, .outputChannel = 0}})
          .config(
              core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
              std::string(kWaitForever))
          .beforeTaskStart([](facebook::velox::exec::Task& task) {
            VELOX_ASSERT_USER_THROW(
                task.addDynamicFilter(
                    "wrong",
                    std::make_shared<common::BytesValues>(
                        std::vector<std::string>{"a"}, false)),
                "Dynamic filter does not fit the type of the scan column");
            task.addDynamicFilter("wrong", nullptr);
            task.addDynamicFilter(
                "right",
                std::make_shared<common::BigintRange>(100, 199, false));
          })
          .splits(scanId, makeCudfHiveConnectorSplits(files_))
          .assertResults("SELECT k, v FROM probe WHERE k BETWEEN 100 AND 199");
  EXPECT_EQ(toPlanStats(task->taskStats()).at(scanId).outputRows, 100);
}

} // namespace
} // namespace facebook::velox::cudf_velox::exec::test
