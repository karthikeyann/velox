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
#include <folly/stop_watch.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/DynamicFilterCollector.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/HashProbe.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {
namespace {

using BuildDistribution = DynamicFilterCollector::BuildDistribution;

// Tests dynamic filters that travel outside of a pipeline: from a hash join
// to a table scan behind a local exchange in the same task, and to table scans
// in other tasks, with the test delivering the filters between the tasks.
class TaskDynamicFilterTest : public HiveConnectorTestBase {
 protected:
  static constexpr int32_t kNumFiles = 4;
  static constexpr int32_t kRowsPerFile = 1'000;
  static constexpr int32_t kNumRows = kNumFiles * kRowsPerFile;
  static constexpr std::string_view kWaitForever = "600000";

  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    exec::ExchangeSource::factories().clear();
    exec::ExchangeSource::registerFactory(createLocalExchangeSource);

    // File i has the keys [i * 1'000, (i + 1) * 1'000) in 'k' and twice the
    // key in 'v'.
    for (int32_t i = 0; i < kNumFiles; ++i) {
      const int64_t start = i * kRowsPerFile;
      probeVectors_.push_back(makeRowVector(
          {"k", "v"},
          {
              makeFlatVector<int64_t>(
                  kRowsPerFile, [&](auto row) { return start + row; }),
              makeFlatVector<int64_t>(
                  kRowsPerFile, [&](auto row) { return 2 * (start + row); }),
          }));
      files_.push_back(TempFilePath::create());
      writeToFile(files_.back()->getPath(), probeVectors_.back());
    }
    createDuckDbTable("probe", probeVectors_);
  }

  void TearDown() override {
    probeVectors_.clear();
    HiveConnectorTestBase::TearDown();
  }

  std::vector<std::shared_ptr<connector::ConnectorSplit>> probeSplits(
      int32_t firstFile = 0,
      int32_t numFiles = kNumFiles) const {
    std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
    for (auto i = firstFile; i < firstFile + numFiles; ++i) {
      splits.push_back(makeHiveConnectorSplit(files_[i]->getPath()));
    }
    return splits;
  }

  RowVectorPtr makeBuild(const std::vector<std::optional<int64_t>>& keys) {
    auto build = makeRowVector({"bk"}, {makeNullableFlatVector<int64_t>(keys)});
    createDuckDbTable("build", {build});
    return build;
  }

  static common::FilterPtr range(int64_t lower, int64_t upper) {
    return std::make_shared<common::BigintRange>(lower, upper, false);
  }

  static int64_t statSum(const PlanNodeStats& stats, std::string_view name) {
    const auto it = stats.customStats.find(std::string(name));
    return it == stats.customStats.end() ? 0 : it->second.sum;
  }

  // A join whose probe side is a scan of the probe table behind a local
  // exchange, so that the scan is not in the pipeline of the join.
  struct JoinPlan {
    core::PlanNodePtr plan;
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;

    std::vector<core::DynamicFilterSource> sources() const {
      return {{.filterId = "df", .joinNodeId = joinId, .keyIndex = 0}};
    }

    std::vector<core::DynamicFilterTarget> targets() const {
      return {{.filterId = "df", .scanNodeId = scanId, .outputChannel = 0}};
    }
  };

  JoinPlan makeJoinPlan(
      const core::PlanNodePtr& build,
      const std::shared_ptr<core::PlanNodeIdGenerator>& planNodeIdGenerator,
      core::JoinType joinType = core::JoinType::kInner) {
    JoinPlan plan;
    plan.plan = PlanBuilder(planNodeIdGenerator)
                    .tableScan(probeType_)
                    .capturePlanNodeId(plan.scanId)
                    .localPartition({"k"})
                    .hashJoin({"k"}, {"bk"}, build, "", {"k", "v"}, joinType)
                    .capturePlanNodeId(plan.joinId)
                    .planNode();
    return plan;
  }

  JoinPlan makeJoinPlan(
      const RowVectorPtr& build,
      core::JoinType joinType = core::JoinType::kInner) {
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    return makeJoinPlan(
        PlanBuilder(planNodeIdGenerator).values({build}).planNode(),
        planNodeIdGenerator,
        joinType);
  }

  // Runs 'plan' with its dynamic filter and returns the statistics by plan
  // node.
  std::unordered_map<core::PlanNodeId, PlanNodeStats> runJoin(
      const JoinPlan& plan,
      const std::string& duckDbSql,
      int32_t numDrivers = 4,
      const std::unordered_map<std::string, std::string>& configs = {}) {
    AssertQueryBuilder builder(plan.plan, duckDbQueryRunner_);
    builder.maxDrivers(numDrivers)
        .dynamicFilters(plan.sources(), plan.targets())
        .config(
            core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
            std::string(kWaitForever))
        .splits(plan.scanId, probeSplits());
    for (const auto& [name, value] : configs) {
      builder.config(name, value);
    }
    auto task = builder.assertResults(duckDbSql);
    return toPlanStats(task->taskStats());
  }

  // Runs a scan of the probe table that waits for the filters 'filterIds' on
  // 'k' like a task of a multi-task query, with 'deliver' called on another
  // thread once the task has started. Returns the statistics of the scan.
  PlanNodeStats runScan(
      const std::vector<std::string>& filterIds,
      const std::function<void(Task&)>& deliver,
      const std::string& duckDbSql,
      const std::string& maxWaitMs = std::string(kWaitForever),
      int32_t numDrivers = 2,
      const std::vector<std::string>& subfieldFilters = {}) {
    core::PlanNodeId scanId;
    auto plan = PlanBuilder()
                    .tableScan(probeType_, subfieldFilters)
                    .capturePlanNodeId(scanId)
                    .planNode();
    std::vector<core::DynamicFilterTarget> targets;
    for (const auto& filterId : filterIds) {
      targets.push_back(
          {.filterId = filterId, .scanNodeId = scanId, .outputChannel = 0});
    }
    std::thread coordinator;
    auto task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .maxDrivers(numDrivers)
            .dynamicFilters({}, targets)
            .config(
                core::QueryConfig::kTableScanDynamicFilterMaxWaitMs, maxWaitMs)
            .beforeTaskStart([&](Task& task) {
              coordinator =
                  std::thread([&deliver, task = task.shared_from_this()]() {
                    deliver(*task);
                  });
            })
            .splits(probeSplits())
            .assertResults(duckDbSql);
    coordinator.join();
    return std::move(toPlanStats(task->taskStats()).at(scanId));
  }

  std::shared_ptr<Task> makeTask(
      const std::string& taskId,
      core::PlanFragment fragment,
      int32_t destination,
      const std::unordered_map<std::string, std::string>& configs = {}) {
    auto allConfigs = configs;
    allConfigs.emplace(
        core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
        std::string(kWaitForever));
    auto queryCtx = core::QueryCtx::create(
        executor_.get(), core::QueryConfig(std::move(allConfigs)));
    return Task::create(
        taskId,
        std::move(fragment),
        destination,
        std::move(queryCtx),
        Task::ExecutionMode::kParallel);
  }

  // Waits until 'numDrivers' drivers of 'task' are blocked waiting for a
  // dynamic filter.
  static void waitForDriversBlockedOnFilter(Task& task, int32_t numDrivers) {
    const auto numWaiting = [&]() {
      const auto counts = task.driverCounts().numBlockedDrivers;
      const auto it = counts.find(BlockingReason::kWaitForDynamicFilter);
      return it == counts.end() ? 0 : it->second;
    };
    const folly::stop_watch<std::chrono::seconds> watch;
    while (numWaiting() < numDrivers && watch.elapsed().count() < 60) {
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(numWaiting(), numDrivers);
  }

  static std::string makeTaskId(const std::string& prefix, int32_t num) {
    return fmt::format("local://{}-{}", prefix, num);
  }

  static exec::Split remoteSplit(const std::string& taskId) {
    return exec::Split(std::make_shared<RemoteConnectorSplit>(taskId));
  }

  // A query of three stages: 'kNumScanTasks' tasks scan the probe table and
  // partition it on the join key, 'kNumJoinTasks' tasks join it with the
  // build side, and the test reads the result of the join tasks. The test
  // plays the part of whoever runs the tasks: it collects the filters of the
  // join tasks and delivers their combination to the scan tasks.
  struct MultiTaskQuery {
    std::vector<std::shared_ptr<Task>> scanTasks;
    std::vector<std::shared_ptr<Task>> joinTasks;
    core::PlanNodeId scanId;
    RowTypePtr joinOutputType;
  };

  static constexpr int32_t kNumScanTasks = 2;
  static constexpr int32_t kNumJoinTasks = 2;

  // Starts the scan and join tasks. With a partitioned build, one more task
  // partitions 'build' on its key for the join tasks; with a broadcast build
  // every join task builds from all of 'build'.
  MultiTaskQuery startQuery(
      const RowVectorPtr& build,
      BuildDistribution distribution,
      int32_t numJoinDrivers = 2) {
    MultiTaskQuery query;
    for (int32_t i = 0; i < kNumScanTasks; ++i) {
      core::PlanFragment fragment{PlanBuilder()
                                      .tableScan(probeType_)
                                      .capturePlanNodeId(query.scanId)
                                      .partitionedOutput({"k"}, kNumJoinTasks)
                                      .planNode()};
      fragment.dynamicFilterTargets = {
          {.filterId = "df", .scanNodeId = query.scanId, .outputChannel = 0}};
      auto task = makeTask(makeTaskId("scan", i), std::move(fragment), 0);
      const auto filesPerTask = kNumFiles / kNumScanTasks;
      for (auto& split : probeSplits(i * filesPerTask, filesPerTask)) {
        task->addSplit(query.scanId, exec::Split(std::move(split)));
      }
      task->noMoreSplits(query.scanId);
      task->start(2);
      query.scanTasks.push_back(std::move(task));
    }

    std::shared_ptr<Task> buildTask;
    if (distribution == BuildDistribution::kPartitioned) {
      core::PlanFragment fragment{PlanBuilder()
                                      .values({build})
                                      .partitionedOutput({"bk"}, kNumJoinTasks)
                                      .planNode()};
      buildTask = makeTask(makeTaskId("build", 0), std::move(fragment), 0);
      buildTask->start(1);
    }

    for (int32_t i = 0; i < kNumJoinTasks; ++i) {
      auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
      core::PlanNodeId probeExchangeId;
      core::PlanNodeId buildExchangeId;
      core::PlanNodeId joinId;
      auto buildSide = PlanBuilder(planNodeIdGenerator);
      if (buildTask != nullptr) {
        buildSide.exchange(asRowType(build->type()), "Presto")
            .capturePlanNodeId(buildExchangeId);
      } else {
        buildSide.values({build});
      }
      core::PlanFragment fragment{
          PlanBuilder(planNodeIdGenerator)
              .exchange(probeType_, "Presto")
              .capturePlanNodeId(probeExchangeId)
              .hashJoin({"k"}, {"bk"}, buildSide.planNode(), "", {"k", "v"})
              .capturePlanNodeId(joinId)
              .partitionedOutput({}, 1)
              .planNode()};
      query.joinOutputType = fragment.planNode->outputType();
      fragment.dynamicFilterSources = {
          {.filterId = "df", .joinNodeId = joinId, .keyIndex = 0}};
      auto task = makeTask(makeTaskId("join", i), std::move(fragment), i);
      for (const auto& scanTask : query.scanTasks) {
        task->addSplit(probeExchangeId, remoteSplit(scanTask->taskId()));
      }
      task->noMoreSplits(probeExchangeId);
      if (buildTask != nullptr) {
        task->addSplit(buildExchangeId, remoteSplit(buildTask->taskId()));
        task->noMoreSplits(buildExchangeId);
      }
      task->start(numJoinDrivers);
      query.joinTasks.push_back(std::move(task));
    }
    return query;
  }

  // Collects the filters of the join tasks, combines them and delivers the
  // result to the scan tasks. Returns the delivered filter.
  static common::FilterPtr deliverFilters(
      const MultiTaskQuery& query,
      BuildDistribution distribution) {
    DynamicFilterCollector collector(distribution, query.joinTasks.size(), {});
    for (const auto& task : query.joinTasks) {
      if (collector.add(
              task->dynamicFilterFuture("df").get(std::chrono::seconds(60)))) {
        break;
      }
    }
    for (const auto& task : query.scanTasks) {
      task->addDynamicFilter("df", collector.result());
    }
    return collector.result();
  }

  // Checks the result of the join tasks and returns the number of rows the
  // scans of the scan tasks produced.
  uint64_t finishQuery(
      const MultiTaskQuery& query,
      const std::string& duckDbSql) {
    std::vector<exec::Split> splits;
    for (const auto& task : query.joinTasks) {
      splits.push_back(remoteSplit(task->taskId()));
    }
    AssertQueryBuilder(
        PlanBuilder().exchange(query.joinOutputType, "Presto").planNode(),
        duckDbQueryRunner_)
        .splits(std::move(splits))
        .assertResults(duckDbSql);

    uint64_t numScanRows{0};
    for (const auto& task : query.scanTasks) {
      EXPECT_TRUE(waitForTaskCompletion(task.get(), 60'000'000));
      numScanRows += toPlanStats(task->taskStats()).at(query.scanId).outputRows;
    }
    for (const auto& task : query.joinTasks) {
      EXPECT_TRUE(waitForTaskCompletion(task.get(), 60'000'000));
    }
    return numScanRows;
  }

  const RowTypePtr probeType_{ROW({"k", "v"}, BIGINT())};
  const std::string joinSql_{"SELECT k, v FROM probe, build WHERE k = bk"};
  std::vector<RowVectorPtr> probeVectors_;
  std::vector<std::shared_ptr<TempFilePath>> files_;
};

TEST_F(TaskDynamicFilterTest, localExchange) {
  auto plan = makeJoinPlan(makeBuild({5, 1'500, 1'501, 3'999, 10'000}));

  for (const auto numDrivers : {1, 4}) {
    SCOPED_TRACE(fmt::format("numDrivers {}", numDrivers));
    {
      // The filters of a pipeline stop at the local exchange.
      auto task = AssertQueryBuilder(plan.plan, duckDbQueryRunner_)
                      .maxDrivers(numDrivers)
                      .splits(plan.scanId, probeSplits())
                      .assertResults(joinSql_);
      const auto stats = toPlanStats(task->taskStats());
      EXPECT_EQ(stats.at(plan.scanId).outputRows, kNumRows);
    }
    {
      const auto stats = runJoin(plan, joinSql_, numDrivers);
      const auto& scanStats = stats.at(plan.scanId);
      EXPECT_EQ(scanStats.outputRows, 4);
      EXPECT_GT(statSum(scanStats, DriverStats::kDynamicFiltersAccepted), 0);
      EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 0);
      EXPECT_EQ(
          scanStats.dynamicFilterStats.producerNodeIds.count(plan.joinId), 1);
      EXPECT_EQ(
          statSum(stats.at(plan.joinId), DriverStats::kDynamicFiltersProduced),
          1);
    }
  }
}

TEST_F(TaskDynamicFilterTest, semiAndRightJoins) {
  auto build = makeBuild({5, 5, 1'500, 10'000});
  {
    auto plan = makeJoinPlan(build, core::JoinType::kLeftSemiFilter);
    const auto stats = runJoin(
        plan, "SELECT k, v FROM probe WHERE k IN (SELECT bk FROM build)");
    EXPECT_EQ(stats.at(plan.scanId).outputRows, 2);
  }
  {
    auto plan = makeJoinPlan(build, core::JoinType::kRight);
    const auto stats =
        runJoin(plan, "SELECT k, v FROM probe RIGHT JOIN build ON k = bk");
    EXPECT_EQ(stats.at(plan.scanId).outputRows, 2);
  }
}

TEST_F(TaskDynamicFilterTest, joinThatCannotFilterProbe) {
  // A left join returns the probe rows without a match, so the join opts out
  // and the scan reads everything.
  auto plan = makeJoinPlan(makeBuild({5, 1'500}), core::JoinType::kLeft);
  const auto stats =
      runJoin(plan, "SELECT k, v FROM probe LEFT JOIN build ON k = bk");
  const auto& scanStats = stats.at(plan.scanId);
  EXPECT_EQ(scanStats.outputRows, kNumRows);
  EXPECT_EQ(statSum(scanStats, DriverStats::kDynamicFiltersAccepted), 0);
  EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 0);
}

TEST_F(TaskDynamicFilterTest, emptyBuild) {
  {
    // The build side has no rows.
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan = makeJoinPlan(
        PlanBuilder(planNodeIdGenerator)
            .values({makeBuild({1, 2})})
            .filter("bk > 10")
            .planNode(),
        planNodeIdGenerator);
    const auto stats = runJoin(plan, "SELECT k, v FROM probe WHERE false");
    EXPECT_EQ(stats.at(plan.scanId).outputRows, 0);
  }
  {
    // The build side has only null keys.
    auto plan = makeJoinPlan(makeBuild({std::nullopt, std::nullopt}));
    const auto stats = runJoin(plan, joinSql_);
    EXPECT_EQ(stats.at(plan.scanId).outputRows, 0);
  }
}

TEST_F(TaskDynamicFilterTest, nullKeys) {
  // Null keys on either side match nothing and are not part of the filter.
  auto nulls = makeRowVector(
      {"k", "v"},
      {
          makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}),
          makeFlatVector<int64_t>({-1, -2}),
      });
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), nulls);
  auto probeVectors = probeVectors_;
  probeVectors.push_back(nulls);
  createDuckDbTable("probe", probeVectors);

  auto plan = makeJoinPlan(makeBuild({7, std::nullopt, 2'500}));
  auto splits = probeSplits();
  splits.push_back(makeHiveConnectorSplit(file->getPath()));
  auto task = AssertQueryBuilder(plan.plan, duckDbQueryRunner_)
                  .maxDrivers(4)
                  .dynamicFilters(plan.sources(), plan.targets())
                  .config(
                      core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
                      std::string(kWaitForever))
                  .splits(plan.scanId, splits)
                  .assertResults(joinSql_);
  EXPECT_EQ(toPlanStats(task->taskStats()).at(plan.scanId).outputRows, 2);
}

TEST_F(TaskDynamicFilterTest, buildWithTooManyKeys) {
  // More distinct keys than a set of values holds, spread too far for a
  // bitmask: 0, 1'000, 2'000, ...
  constexpr int32_t kNumKeys = 110'000;
  auto build =
      makeRowVector({"bk"}, {makeFlatVector<int64_t>(kNumKeys, [](auto row) {
                      return row * 1'000LL;
                    })});
  createDuckDbTable("build", {build});
  auto plan = makeJoinPlan(build);
  {
    // The join opts out, which releases the scan.
    const auto stats = runJoin(plan, joinSql_);
    const auto& scanStats = stats.at(plan.scanId);
    EXPECT_EQ(scanStats.outputRows, kNumRows);
    EXPECT_EQ(statSum(scanStats, DriverStats::kDynamicFiltersAccepted), 0);
    EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 0);
  }
  {
    // With a bloom filter the join can describe its keys.
    const auto stats = runJoin(
        plan,
        joinSql_,
        4,
        {{core::QueryConfig::kHashProbeBloomFilterPushdownMaxSize, "1048576"}});
    const auto& scanStats = stats.at(plan.scanId);
    EXPECT_GE(scanStats.outputRows, 4);
    EXPECT_LT(scanStats.outputRows, kNumRows / 4);
    EXPECT_GT(statSum(scanStats, DriverStats::kDynamicFiltersAccepted), 0);
  }
}

TEST_F(TaskDynamicFilterTest, groupedExecution) {
  // Each split group builds its own table from a part of the build side. The
  // join opts out when it starts and the scans read everything.
  auto buildFile = TempFilePath::create();
  writeToFile(buildFile->getPath(), makeBuild({5, 1'500, 10'000}));

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId buildScanId;
  auto plan = makeJoinPlan(
      PlanBuilder(planNodeIdGenerator)
          .tableScan(ROW("bk", BIGINT()))
          .capturePlanNodeId(buildScanId)
          .planNode(),
      planNodeIdGenerator);

  constexpr int32_t kNumGroups = 2;
  core::PlanFragment fragment{plan.plan};
  fragment.executionStrategy = core::ExecutionStrategy::kGrouped;
  fragment.groupedExecutionLeafNodeIds = {plan.scanId, buildScanId};
  fragment.numSplitGroups = kNumGroups;
  fragment.dynamicFilterSources = plan.sources();
  fragment.dynamicFilterTargets = plan.targets();

  std::atomic<vector_size_t> numRows{0};
  auto queryCtx = core::QueryCtx::create(
      executor_.get(),
      core::QueryConfig(
          {{core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
            std::string(kWaitForever)}}));
  auto task = Task::create(
      "grouped",
      std::move(fragment),
      0,
      std::move(queryCtx),
      Task::ExecutionMode::kParallel,
      [&](RowVectorPtr vector, bool, ContinueFuture*) {
        if (vector != nullptr) {
          numRows += vector->size();
        }
        return BlockingReason::kNotBlocked;
      });
  task->start(2, 1);
  for (int32_t group = 0; group < kNumGroups; ++group) {
    for (auto& split : probeSplits()) {
      task->addSplit(plan.scanId, exec::Split(std::move(split), group));
    }
    task->addSplit(
        buildScanId,
        exec::Split(makeHiveConnectorSplit(buildFile->getPath()), group));
    task->noMoreSplitsForGroup(plan.scanId, group);
    task->noMoreSplitsForGroup(buildScanId, group);
  }
  task->noMoreSplits(plan.scanId);
  task->noMoreSplits(buildScanId);
  ASSERT_TRUE(waitForTaskCompletion(task.get(), 60'000'000));

  // Keys 5 and 1'500 match in each group.
  EXPECT_EQ(numRows, 2 * kNumGroups);
  const auto stats = toPlanStats(task->taskStats());
  const auto& scanStats = stats.at(plan.scanId);
  EXPECT_EQ(scanStats.outputRows, kNumRows * kNumGroups);
  EXPECT_EQ(statSum(scanStats, DriverStats::kDynamicFiltersAccepted), 0);
  EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 0);
}

TEST_F(TaskDynamicFilterTest, failingBuild) {
  // The build side fails while the scan waits for its filter. The query fails
  // with the error of the build side and does not wait out the limit.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = makeJoinPlan(
      PlanBuilder(planNodeIdGenerator)
          .values({makeBuild({5, 1'500})})
          .project({"bk / 0 AS bk"})
          .planNode(),
      planNodeIdGenerator);
  const folly::stop_watch<std::chrono::seconds> watch;
  VELOX_ASSERT_THROW(runJoin(plan, joinSql_), "division by zero");
  EXPECT_LT(watch.elapsed().count(), 60);
}

DEBUG_ONLY_TEST_F(TaskDynamicFilterTest, failureWhileFinishingBuild) {
  // The last build driver fails after the build side is complete and before
  // it publishes. The filter is resolved on the way out and the query fails
  // with the error.
  SCOPED_TESTVALUE_SET(
      "facebook::velox::exec::HashBuild::finishHashBuild",
      std::function<void(Operator*)>(
          [](Operator*) { VELOX_FAIL("Injected build failure"); }));
  auto plan = makeJoinPlan(makeBuild({5, 1'500}));
  const folly::stop_watch<std::chrono::seconds> watch;
  VELOX_ASSERT_THROW(runJoin(plan, joinSql_), "Injected build failure");
  EXPECT_LT(watch.elapsed().count(), 60);
}

TEST_F(TaskDynamicFilterTest, rightJoinWithNullStringKeys) {
  // The build side of a right join keeps its rows with null keys. A string
  // key without a single value makes an always-false filter, both for the
  // scan in the pipeline of the join and for the scan behind a local
  // exchange.
  auto data = makeRowVector(
      {"s"}, {makeFlatVector<std::string>({"apple", "banana", "cherry"})});
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), data);
  auto build = makeRowVector(
      {"bs"},
      {makeNullableFlatVector<std::string>({std::nullopt, std::nullopt})});
  const auto expected = makeRowVector(
      {makeNullableFlatVector<std::string>({std::nullopt, std::nullopt})});

  for (const bool localExchange : {false, true}) {
    SCOPED_TRACE(fmt::format("localExchange {}", localExchange));
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    PlanBuilder builder(planNodeIdGenerator);
    builder.tableScan(ROW("s", VARCHAR())).capturePlanNodeId(scanId);
    if (localExchange) {
      builder.localPartition({"s"});
    }
    auto plan =
        builder
            .hashJoin(
                {"s"},
                {"bs"},
                PlanBuilder(planNodeIdGenerator).values({build}).planNode(),
                "",
                {"s"},
                core::JoinType::kRight)
            .capturePlanNodeId(joinId)
            .planNode();
    AssertQueryBuilder queryBuilder(plan);
    queryBuilder
        .config(
            core::QueryConfig::kHashProbeStringDynamicFilterPushdownEnabled,
            "true")
        .config(
            core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
            std::string(kWaitForever))
        .split(scanId, makeHiveConnectorSplit(file->getPath()));
    if (localExchange) {
      queryBuilder.dynamicFilters(
          {{.filterId = "df", .joinNodeId = joinId, .keyIndex = 0}},
          {{.filterId = "df", .scanNodeId = scanId, .outputChannel = 0}});
    }
    auto task = queryBuilder.assertResults(expected);
    EXPECT_EQ(toPlanStats(task->taskStats()).at(scanId).outputRows, 0);
  }
}

TEST_F(TaskDynamicFilterTest, scanFeedingTheBuildSide) {
  // The filter of a join targets the scan of the join's own build side. The
  // filter cannot exist before that scan is done, so the scan must not wait
  // for it.
  auto buildFile = TempFilePath::create();
  writeToFile(buildFile->getPath(), makeBuild({5, 1'500, 10'000}));

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId buildScanId;
  auto plan = makeJoinPlan(
      PlanBuilder(planNodeIdGenerator)
          .tableScan(ROW("bk", BIGINT()))
          .capturePlanNodeId(buildScanId)
          .planNode(),
      planNodeIdGenerator);

  const folly::stop_watch<std::chrono::seconds> watch;
  auto task =
      AssertQueryBuilder(plan.plan, duckDbQueryRunner_)
          .maxDrivers(2)
          .dynamicFilters(
              plan.sources(),
              {{.filterId = "df",
                .scanNodeId = buildScanId,
                .outputChannel = 0}})
          .config(
              core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
              std::string(kWaitForever))
          .splits(plan.scanId, probeSplits())
          .split(buildScanId, makeHiveConnectorSplit(buildFile->getPath()))
          .assertResults(joinSql_);
  EXPECT_LT(watch.elapsed().count(), 60);
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(stats.at(buildScanId).outputRows, 3);
  EXPECT_EQ(
      statSum(stats.at(buildScanId), "blockedWaitForDynamicFilterTimes"), 0);
}

TEST_F(TaskDynamicFilterTest, deliveredBeforeAndWhileWaiting) {
  {
    const auto stats = runScan(
        {"df"},
        [](Task& task) { task.addDynamicFilter("df", range(100, 199)); },
        "SELECT * FROM probe WHERE k BETWEEN 100 AND 199");
    EXPECT_EQ(stats.outputRows, 100);
  }
  {
    // The scan reads nothing until the filter arrives.
    const auto stats = runScan(
        {"df"},
        [&](Task& task) {
          waitForDriversBlockedOnFilter(task, 2);
          task.addDynamicFilter("df", range(100, 199));
        },
        "SELECT * FROM probe WHERE k BETWEEN 100 AND 199");
    EXPECT_EQ(stats.outputRows, 100);
    EXPECT_GT(statSum(stats, "blockedWaitForDynamicFilterTimes"), 0);
    EXPECT_GT(statSum(stats, TableScan::kDynamicFilterWaitWallNanos), 0);
    EXPECT_EQ(statSum(stats, TableScan::kDynamicFiltersTimedOut), 0);
  }
}

TEST_F(TaskDynamicFilterTest, disabledAndAlwaysFalse) {
  {
    // A disabled filter releases the scan, which reads everything.
    const auto stats = runScan(
        {"df"},
        [](Task& task) { task.addDynamicFilter("df", nullptr); },
        "SELECT * FROM probe");
    EXPECT_EQ(stats.outputRows, kNumRows);
    EXPECT_EQ(statSum(stats, DriverStats::kDynamicFiltersAccepted), 0);
  }
  {
    const auto stats = runScan(
        {"df"},
        [](Task& task) {
          task.addDynamicFilter("df", std::make_shared<common::AlwaysFalse>());
        },
        "SELECT * FROM probe WHERE false");
    EXPECT_EQ(stats.outputRows, 0);
  }
}

TEST_F(TaskDynamicFilterTest, filterKinds) {
  const std::string sql = "SELECT * FROM probe WHERE k IN (3, 1500, 1999)";
  const std::vector<int64_t> keys{3, 1'500, 1'999};
  {
    auto values = common::createBigintValues(keys, false);
    ASSERT_EQ(values->kind(), common::FilterKind::kBigintValuesUsingBitmask);
    const auto stats = runScan(
        {"df"},
        [&](Task& task) { task.addDynamicFilter("df", std::move(values)); },
        sql);
    EXPECT_EQ(stats.outputRows, 3);
  }
  {
    auto values =
        common::createBigintValues({3, 1'500, 1'999, 1'000'000'000}, false);
    ASSERT_EQ(values->kind(), common::FilterKind::kBigintValuesUsingHashTable);
    const auto stats = runScan(
        {"df"},
        [&](Task& task) { task.addDynamicFilter("df", std::move(values)); },
        sql);
    EXPECT_EQ(stats.outputRows, 3);
  }
  {
    auto bloom =
        std::make_shared<common::BigintValuesUsingBloomFilter>(1'000, false);
    for (auto key : keys) {
      bloom->insert(key);
    }
    // None of the other keys passes this bloom filter.
    const auto stats = runScan(
        {"df"}, [&](Task& task) { task.addDynamicFilter("df", bloom); }, sql);
    EXPECT_EQ(stats.outputRows, 3);
  }
}

TEST_F(TaskDynamicFilterTest, stringColumn) {
  auto data = makeRowVector(
      {"s"}, {makeFlatVector<std::string>({"apple", "banana", "cherry"})});
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), data);

  core::PlanNodeId scanId;
  auto plan = PlanBuilder()
                  .tableScan(ROW("s", VARCHAR()))
                  .capturePlanNodeId(scanId)
                  .planNode();
  auto task =
      AssertQueryBuilder(plan)
          .dynamicFilters(
              {},
              {{.filterId = "df", .scanNodeId = scanId, .outputChannel = 0}})
          .config(
              core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
              std::string(kWaitForever))
          .beforeTaskStart([](Task& task) {
            task.addDynamicFilter(
                "df",
                std::make_shared<common::BytesValues>(
                    std::vector<std::string>{"banana", "date"}, false));
          })
          .split(makeHiveConnectorSplit(file->getPath()))
          .assertResults(
              makeRowVector({makeFlatVector<std::string>({"banana"})}));
  EXPECT_EQ(toPlanStats(task->taskStats()).at(scanId).outputRows, 1);

  // The scan's own filter on the column is combined with the delivered one
  // when the first split is opened.
  core::PlanNodeId filteredScanId;
  auto filteredPlan = PlanBuilder()
                          .tableScan(ROW("s", VARCHAR()), {"s >= 'b'"})
                          .capturePlanNodeId(filteredScanId)
                          .planNode();
  auto filteredTask =
      AssertQueryBuilder(filteredPlan)
          .dynamicFilters(
              {},
              {{.filterId = "df",
                .scanNodeId = filteredScanId,
                .outputChannel = 0}})
          .config(
              core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
              std::string(kWaitForever))
          .beforeTaskStart([](Task& task) {
            task.addDynamicFilter(
                "df",
                std::make_shared<common::BytesValues>(
                    std::vector<std::string>{"apple", "banana"}, false));
          })
          .split(makeHiveConnectorSplit(file->getPath()))
          .assertResults(
              makeRowVector({makeFlatVector<std::string>({"banana"})}));
  EXPECT_EQ(
      toPlanStats(filteredTask->taskStats()).at(filteredScanId).outputRows, 1);
}

TEST_F(TaskDynamicFilterTest, filterOfWrongType) {
  // The delivery fails for the one who delivers. The query is not affected.
  const auto stats = runScan(
      {"df"},
      [](Task& task) {
        SCOPE_EXIT {
          task.addDynamicFilter("df", nullptr);
        };
        VELOX_ASSERT_USER_THROW(
            task.addDynamicFilter(
                "df",
                std::make_shared<common::BytesValues>(
                    std::vector<std::string>{"a"}, false)),
            "Dynamic filter does not fit the type of the scan column");
        // The members of a filter that combines others are not looked into.
        std::vector<std::unique_ptr<common::Filter>> ranges;
        ranges.push_back(std::make_unique<common::BigintRange>(0, 10, false));
        ranges.push_back(std::make_unique<common::BigintRange>(20, 30, false));
        VELOX_ASSERT_USER_THROW(
            task.addDynamicFilter(
                "df",
                std::make_shared<common::MultiRange>(std::move(ranges), false)),
            "Dynamic filter does not fit the type of the scan column");
      },
      "SELECT * FROM probe");
  EXPECT_EQ(stats.outputRows, kNumRows);
}

TEST_F(TaskDynamicFilterTest, staticFilterOnTheColumn) {
  const auto stats = runScan(
      {"df"},
      [](Task& task) { task.addDynamicFilter("df", range(100, 199)); },
      "SELECT * FROM probe WHERE k BETWEEN 150 AND 199",
      std::string(kWaitForever),
      2,
      {"k >= 150"});
  EXPECT_EQ(stats.outputRows, 50);
}

TEST_F(TaskDynamicFilterTest, severalFiltersOnOneScan) {
  // The filters of one column intersect. A disabled one drops out. Only the
  // first delivery of a filter counts.
  const auto stats = runScan(
      {"a", "b", "c"},
      [](Task& task) {
        task.addDynamicFilter("a", range(100, 299));
        task.addDynamicFilter("c", nullptr);
        task.addDynamicFilter("b", range(200, 399));
        task.addDynamicFilter("b", range(0, 0));
      },
      "SELECT * FROM probe WHERE k BETWEEN 200 AND 299");
  EXPECT_EQ(stats.outputRows, 100);
}

TEST_F(TaskDynamicFilterTest, waitLimit) {
  // The filter never comes. The scan reads everything after the limit.
  const auto stats =
      runScan({"df"}, [](Task&) {}, "SELECT * FROM probe", "200");
  EXPECT_EQ(stats.outputRows, kNumRows);
  EXPECT_GT(statSum(stats, TableScan::kDynamicFiltersTimedOut), 0);
  EXPECT_GE(
      statSum(stats, TableScan::kDynamicFilterWaitWallNanos), 200'000'000);
}

TEST_F(TaskDynamicFilterTest, lateFilter) {
  // The scan does not wait. The filter arrives after the first split is read
  // and applies to the splits read afterwards.
  core::PlanNodeId scanId;
  CursorParameters params;
  params.planNode =
      PlanBuilder().tableScan(probeType_).capturePlanNodeId(scanId).planNode();
  params.dynamicFilterTargets = {
      {.filterId = "df", .scanNodeId = scanId, .outputChannel = 0}};
  params.queryConfigs = {
      {core::QueryConfig::kTableScanDynamicFilterMaxWaitMs, "0"}};
  auto cursor = TaskCursor::create(params);
  cursor->start();
  auto* task = cursor->task().get();
  auto splits = probeSplits();
  task->addSplit(scanId, exec::Split(std::move(splits[0])));

  vector_size_t numRows{0};
  while (numRows < kRowsPerFile) {
    ASSERT_TRUE(cursor->moveNext());
    numRows += cursor->current()->size();
  }
  ASSERT_EQ(numRows, kRowsPerFile);

  task->addDynamicFilter("df", range(1'100, 1'199));
  for (auto i = 1; i < kNumFiles; ++i) {
    task->addSplit(scanId, exec::Split(std::move(splits[i])));
  }
  task->noMoreSplits(scanId);
  while (cursor->moveNext()) {
    numRows += cursor->current()->size();
  }
  EXPECT_EQ(numRows, kRowsPerFile + 100);

  ASSERT_TRUE(waitForTaskCompletion(task));
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(statSum(stats.at(scanId), DriverStats::kDynamicFiltersAccepted), 1);
  EXPECT_EQ(statSum(stats.at(scanId), TableScan::kDynamicFiltersTimedOut), 1);
}

TEST_F(TaskDynamicFilterTest, lateFilterWhileDriversRead) {
  // Four drivers read many splits without waiting. The filter arrives while
  // they read. A driver that cannot apply it without stopping the others
  // applies it at a later split.
  constexpr int32_t kNumCopies = 50;
  core::PlanNodeId scanId;
  CursorParameters params;
  params.planNode =
      PlanBuilder().tableScan(probeType_).capturePlanNodeId(scanId).planNode();
  params.maxDrivers = 4;
  params.dynamicFilterTargets = {
      {.filterId = "df", .scanNodeId = scanId, .outputChannel = 0}};
  params.queryConfigs = {
      {core::QueryConfig::kTableScanDynamicFilterMaxWaitMs, "0"}};
  auto cursor = TaskCursor::create(params);
  cursor->start();
  auto* task = cursor->task().get();
  for (auto i = 0; i < kNumCopies; ++i) {
    for (auto& split : probeSplits()) {
      task->addSplit(scanId, exec::Split(std::move(split)));
    }
  }
  task->noMoreSplits(scanId);

  // Every split returns all of its 1'000 rows or the rows that pass.
  vector_size_t numRows{0};
  ASSERT_TRUE(cursor->moveNext());
  numRows += cursor->current()->size();
  task->addDynamicFilter("df", range(100, 199));
  while (cursor->moveNext()) {
    numRows += cursor->current()->size();
  }
  EXPECT_GE(numRows, kNumCopies * 100);
  EXPECT_LE(numRows, kNumCopies * kNumRows);
  EXPECT_EQ(numRows % 100, 0);
  ASSERT_TRUE(waitForTaskCompletion(task));
}

TEST_F(TaskDynamicFilterTest, waitLimitOncePerScan) {
  // Once a driver of a scan has waited out the limit, the drivers of the scan
  // that start later, e.g. those of another split group, do not wait again.
  core::PlanNodeId scanId;
  core::PlanFragment fragment{
      PlanBuilder().tableScan(probeType_).capturePlanNodeId(scanId).planNode()};
  fragment.dynamicFilterTargets = {
      {.filterId = "a", .scanNodeId = scanId, .outputChannel = 0},
      {.filterId = "b", .scanNodeId = scanId, .outputChannel = 1}};
  TaskDynamicFilters filters(fragment);

  uint32_t numSeen{0};
  std::vector<TaskDynamicFilters::ScanFilter> arrived;
  ContinueFuture future = ContinueFuture::makeEmpty();
  EXPECT_EQ(filters.poll(scanId, numSeen, arrived, &future), 2);
  EXPECT_FALSE(future.isReady());

  filters.add("a", range(1, 2));
  EXPECT_TRUE(future.isReady());
  EXPECT_EQ(filters.poll(scanId, numSeen, arrived, &future), 1);
  EXPECT_EQ(arrived.size(), 1);
  EXPECT_FALSE(future.isReady());

  filters.stopWaiting(scanId);
  EXPECT_TRUE(future.isReady());
  EXPECT_EQ(filters.poll(scanId, numSeen, arrived, nullptr), 0);

  // A filter that arrives after that still reaches the scan.
  filters.add("b", range(3, 4));
  EXPECT_EQ(filters.numArrived(scanId), 2);
  EXPECT_EQ(filters.poll(scanId, numSeen, arrived, nullptr), 0);
  ASSERT_EQ(arrived.size(), 2);
  EXPECT_EQ(arrived[1].filterId, "b");
  EXPECT_EQ(arrived[1].outputChannel, 1);
}

TEST_F(TaskDynamicFilterTest, joinReplacedByItsFilter) {
  // The lower join pushes its keys down its own pipeline and, having unique
  // keys and no build column in its output, lets the scan do its work. The
  // scan must then apply that filter together with the one of the upper join
  // from behind the local exchange: with the upper join's filter alone, rows
  // would get past the lower join that are not in its build side.
  auto lowerBuild = makeRowVector(
      {"ak"}, {makeFlatVector<int64_t>({5, 1'500, 1'501, 3'999})});
  auto upperBuild =
      makeRowVector({"bk"}, {makeFlatVector<int64_t>({1'500, 1'501, 2'000})});
  createDuckDbTable("lower", {lowerBuild});
  createDuckDbTable("build", {upperBuild});
  const std::string sql =
      "SELECT k, v FROM probe, lower, build WHERE k = ak AND k = bk";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId scanId;
  core::PlanNodeId lowerJoinId;
  core::PlanNodeId upperJoinId;
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .tableScan(probeType_)
          .capturePlanNodeId(scanId)
          .hashJoin(
              {"k"},
              {"ak"},
              PlanBuilder(planNodeIdGenerator).values({lowerBuild}).planNode(),
              "",
              {"k", "v"})
          .capturePlanNodeId(lowerJoinId)
          .localPartition({"k"})
          .hashJoin(
              {"k"},
              {"bk"},
              PlanBuilder(planNodeIdGenerator).values({upperBuild}).planNode(),
              "",
              {"k", "v"})
          .capturePlanNodeId(upperJoinId)
          .planNode();

  // Ten splits per file, so that drivers open splits before and after the
  // filter arrives and, with preload, ahead of reading them.
  const auto makeSplits = [&]() {
    std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
    for (auto i = 0; i < 10; ++i) {
      for (auto& split : probeSplits()) {
        splits.push_back(std::move(split));
      }
    }
    return splits;
  };
  createDuckDbTable("probe", [&]() {
    std::vector<RowVectorPtr> vectors;
    for (auto i = 0; i < 10; ++i) {
      vectors.insert(vectors.end(), probeVectors_.begin(), probeVectors_.end());
    }
    return vectors;
  }());

  struct Variant {
    std::string name;
    std::string maxWaitMs;
    std::string maxSplitPreloadPerDriver;
  };
  const std::vector<Variant> variants = {
      {"filter before the first split", std::string(kWaitForever), "0"},
      {"filter while reading", "0", "0"},
      {"filter before the first split, preload",
       std::string(kWaitForever),
       "4"},
      {"filter while reading, preload", "0", "4"},
  };
  for (const auto& variant : variants) {
    SCOPED_TRACE(variant.name);
    auto task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .maxDrivers(4)
            .dynamicFilters(
                {{.filterId = "df", .joinNodeId = upperJoinId, .keyIndex = 0}},
                {{.filterId = "df", .scanNodeId = scanId, .outputChannel = 0}})
            .config(
                core::QueryConfig::kTableScanDynamicFilterMaxWaitMs,
                variant.maxWaitMs)
            .config(
                core::QueryConfig::kMaxSplitPreloadPerDriver,
                variant.maxSplitPreloadPerDriver)
            .splits(scanId, makeSplits())
            .assertResults(sql);
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_GT(
        statSum(
            stats.at(lowerJoinId), HashProbe::kReplacedWithDynamicFilterRows),
        0);
    if (variant.maxWaitMs != "0") {
      // Both filters reached every split.
      EXPECT_EQ(stats.at(scanId).outputRows, 2 * 10);
    }
  }
}

TEST_F(TaskDynamicFilterTest, waitLimitWithSplitGroups) {
  // The filter never comes. The driver of the first split group waits out the
  // limit. The driver of the second group, which starts afterwards, does not
  // wait again.
  constexpr int32_t kNumGroups = 2;
  core::PlanNodeId scanId;
  core::PlanFragment fragment{
      PlanBuilder().tableScan(probeType_).capturePlanNodeId(scanId).planNode()};
  fragment.executionStrategy = core::ExecutionStrategy::kGrouped;
  fragment.groupedExecutionLeafNodeIds = {scanId};
  fragment.numSplitGroups = kNumGroups;
  fragment.dynamicFilterTargets = {
      {.filterId = "df", .scanNodeId = scanId, .outputChannel = 0}};

  std::atomic<vector_size_t> numRows{0};
  auto queryCtx = core::QueryCtx::create(
      executor_.get(),
      core::QueryConfig(
          {{core::QueryConfig::kTableScanDynamicFilterMaxWaitMs, "300"}}));
  auto task = Task::create(
      "groups",
      std::move(fragment),
      0,
      std::move(queryCtx),
      Task::ExecutionMode::kParallel,
      [&](RowVectorPtr vector, bool, ContinueFuture*) {
        if (vector != nullptr) {
          numRows += vector->size();
        }
        return BlockingReason::kNotBlocked;
      });
  task->start(1, 1);
  for (int32_t group = 0; group < kNumGroups; ++group) {
    for (auto& split : probeSplits()) {
      task->addSplit(scanId, exec::Split(std::move(split), group));
    }
    task->noMoreSplitsForGroup(scanId, group);
  }
  task->noMoreSplits(scanId);
  ASSERT_TRUE(waitForTaskCompletion(task.get(), 60'000'000));

  EXPECT_EQ(numRows, kNumRows * kNumGroups);
  const auto stats = toPlanStats(task->taskStats());
  const auto& scanStats = stats.at(scanId);
  // One driver per group, and only the first one found the filter missing
  // when it stopped waiting.
  EXPECT_EQ(statSum(scanStats, TableScan::kDynamicFiltersTimedOut), 1);
  EXPECT_GE(
      statSum(scanStats, TableScan::kDynamicFilterWaitWallNanos), 300'000'000);
  EXPECT_LT(
      statSum(scanStats, TableScan::kDynamicFilterWaitWallNanos), 600'000'000);
}

TEST_F(TaskDynamicFilterTest, abortWhileWaiting) {
  core::PlanNodeId scanId;
  core::PlanFragment fragment{PlanBuilder()
                                  .tableScan(probeType_)
                                  .capturePlanNodeId(scanId)
                                  .partitionedOutput({}, 1)
                                  .planNode()};
  fragment.dynamicFilterTargets = {
      {.filterId = "df", .scanNodeId = scanId, .outputChannel = 0}};
  auto task = makeTask(makeTaskId("scan", 0), std::move(fragment), 0);
  for (auto& split : probeSplits()) {
    task->addSplit(scanId, exec::Split(std::move(split)));
  }
  task->noMoreSplits(scanId);
  task->start(2);

  auto filterFuture = task->dynamicFilterFuture("df");
  waitForDriversBlockedOnFilter(*task, 2);
  EXPECT_FALSE(filterFuture.isReady());

  task->requestAbort();
  ASSERT_TRUE(waitForTaskAborted(task.get(), 60'000'000));
  // Whoever collects the filter of a task that ended gets a disabled filter.
  EXPECT_EQ(std::move(filterFuture).get(std::chrono::seconds(60)), nullptr);
  EXPECT_EQ(task->dynamicFilterFuture("df").get(), nullptr);
  // A delivery after the end has no effect.
  task->addDynamicFilter("df", range(0, 10));
}

TEST_F(TaskDynamicFilterTest, invalidDeclarations) {
  auto plan = makeJoinPlan(makeBuild({1}));
  const auto makeQueryTask =
      [&](std::vector<core::DynamicFilterSource> sources,
          std::vector<core::DynamicFilterTarget> targets) {
        core::PlanFragment fragment{plan.plan};
        fragment.dynamicFilterSources = std::move(sources);
        fragment.dynamicFilterTargets = std::move(targets);
        return makeTask("task", std::move(fragment), 0);
      };

  VELOX_ASSERT_THROW(
      makeQueryTask({{"df", plan.scanId, 0}}, {}),
      "Plan node of a dynamic filter source must be a hash join");
  VELOX_ASSERT_THROW(
      makeQueryTask({{"df", plan.joinId, 1}}, {}),
      "Key of a dynamic filter source is not a key of the join");
  VELOX_ASSERT_THROW(
      makeQueryTask({{"df", plan.joinId, 0}, {"df", plan.joinId, 0}}, {}),
      "Dynamic filter must have one source in a plan fragment");
  VELOX_ASSERT_THROW(
      makeQueryTask({}, {{"df", plan.joinId, 0}}),
      "Plan node of a dynamic filter target must be a table scan");
  VELOX_ASSERT_THROW(
      makeQueryTask({}, {{"df", plan.scanId, 2}}),
      "Column of a dynamic filter target is not an output of the scan");
  VELOX_ASSERT_THROW(
      makeQueryTask({}, {{"df", "no-such-node", 0}}),
      "Plan node of a dynamic filter target is not in the plan fragment");
  VELOX_ASSERT_THROW(
      makeQueryTask({}, {{"df", plan.scanId, 0}, {"df", plan.scanId, 1}}),
      "Dynamic filter must have one target on a table scan");

  auto task = makeQueryTask(plan.sources(), plan.targets());
  VELOX_ASSERT_THROW(
      task->addDynamicFilter("df", range(0, 1)),
      "Dynamic filter is made by a join of the task itself");
  VELOX_ASSERT_THROW(
      task->publishDynamicFilter("other", range(0, 1)),
      "Task is not a source of the dynamic filter");
  VELOX_ASSERT_THROW(
      task->dynamicFilterFuture("other"),
      "Task has no source or target for the dynamic filter");
}

TEST_F(TaskDynamicFilterTest, partitionedBuildAcrossTasks) {
  auto query = startQuery(
      makeBuild({5, 77, 1'500, 1'501, 3'999, 10'000}),
      BuildDistribution::kPartitioned);
  const auto filter = deliverFilters(query, BuildDistribution::kPartitioned);
  ASSERT_NE(filter, nullptr);
  EXPECT_TRUE(filter->testInt64(77));
  EXPECT_FALSE(filter->testInt64(78));
  EXPECT_EQ(finishQuery(query, joinSql_), 5);
}

TEST_F(TaskDynamicFilterTest, emptyBuildPartitionAcrossTasks) {
  // One of the join tasks gets no build rows. Its always-false filter must
  // not narrow or disable the filter of the other task.
  auto query = startQuery(makeBuild({1'500}), BuildDistribution::kPartitioned);
  const auto filter = deliverFilters(query, BuildDistribution::kPartitioned);
  ASSERT_NE(filter, nullptr);
  EXPECT_TRUE(filter->testInt64(1'500));
  EXPECT_EQ(finishQuery(query, joinSql_), 1);
}

TEST_F(TaskDynamicFilterTest, broadcastBuildAcrossTasks) {
  auto query = startQuery(
      makeBuild({5, 77, 1'500, 1'501, 3'999, 10'000}),
      BuildDistribution::kBroadcast);
  const auto filter = deliverFilters(query, BuildDistribution::kBroadcast);
  ASSERT_NE(filter, nullptr);
  EXPECT_EQ(finishQuery(query, joinSql_), 5);
}

TEST_F(TaskDynamicFilterTest, optOutAcrossTasks) {
  // Each join task gets more distinct keys than a set of values holds, so the
  // tasks opt out. The disabled filter releases the scans. With one build
  // driver per task, that driver sees all the keys of the task.
  constexpr int32_t kNumKeys = 250'000;
  auto build =
      makeRowVector({"bk"}, {makeFlatVector<int64_t>(kNumKeys, [](auto row) {
                      return row * 1'000LL;
                    })});
  createDuckDbTable("build", {build});
  auto query = startQuery(build, BuildDistribution::kPartitioned, 1);
  EXPECT_EQ(deliverFilters(query, BuildDistribution::kPartitioned), nullptr);
  EXPECT_EQ(finishQuery(query, joinSql_), kNumRows);
}

} // namespace
} // namespace facebook::velox::exec::test
