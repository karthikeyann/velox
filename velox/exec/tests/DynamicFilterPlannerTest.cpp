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
#include "velox/exec/DynamicFilterPlanner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"

namespace facebook::velox::exec::test {
namespace {

using testing::ElementsAre;
using testing::IsEmpty;
using testing::UnorderedElementsAre;

class DynamicFilterPlannerTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    window::prestosql::registerAllWindowFunctions();
    planNodeIdGenerator_ = std::make_shared<core::PlanNodeIdGenerator>();
  }

  PlanBuilder builder() {
    return PlanBuilder(planNodeIdGenerator_);
  }

  // Returns a scan of columns {prefix}k and {prefix}v and sets 'scanId'.
  PlanBuilder scan(const std::string& prefix, core::PlanNodeId& scanId) {
    return builder()
        .tableScan(ROW({prefix + "k", prefix + "v"}, BIGINT()))
        .capturePlanNodeId(scanId);
  }

  // "<filter id> from <join id> key <key index>" for every source.
  static std::vector<std::string> sources(
      const DynamicFilterPlanner::Result& result) {
    std::vector<std::string> sources;
    for (const auto& source : result.sources) {
      sources.push_back(
          fmt::format(
              "{} from {} key {}",
              source.filterId,
              source.joinNodeId,
              source.keyIndex));
    }
    return sources;
  }

  // "<filter id> to <scan id> column <output channel>" for every target.
  static std::vector<std::string> targets(
      const DynamicFilterPlanner::Result& result) {
    std::vector<std::string> targets;
    for (const auto& target : result.targets) {
      targets.push_back(
          fmt::format(
              "{} to {} column {}",
              target.filterId,
              target.scanNodeId,
              target.outputChannel));
    }
    return targets;
  }

  // Places filters whether or not the build side has a filter.
  static DynamicFilterPlanner::Result plan(const core::PlanNodePtr& plan) {
    return DynamicFilterPlanner::plan(
        *plan,
        {.selectiveBuildOnly = false, .skipScansFilteredInPipeline = false});
  }

  // Same, including the scans in the pipeline of the join.
  static DynamicFilterPlanner::Result planWithProbePipeline(
      const core::PlanNodePtr& plan) {
    return DynamicFilterPlanner::plan(
        *plan,
        {.includeProbePipeline = true,
         .selectiveBuildOnly = false,
         .skipScansFilteredInPipeline = false});
  }

  std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator_;
};

TEST_F(DynamicFilterPlannerTest, probePipeline) {
  // The scan is in the pipeline of the join, which pushes its filter down by
  // itself.
  core::PlanNodeId probeId;
  core::PlanNodeId buildId;
  core::PlanNodeId joinId;
  auto root =
      scan("p", probeId)
          .filter("pv > 0")
          .hashJoin({"pk"}, {"bk"}, scan("b", buildId).planNode(), "", {"pv"})
          .capturePlanNodeId(joinId)
          .planNode();
  {
    auto result = plan(root);
    EXPECT_THAT(result.sources, IsEmpty());
    EXPECT_THAT(result.targets, IsEmpty());
  }
  {
    auto result = planWithProbePipeline(root);
    const auto filterId = joinId + ":0";
    EXPECT_THAT(
        sources(result),
        ElementsAre(fmt::format("{} from {} key 0", filterId, joinId)));
    EXPECT_THAT(
        targets(result),
        ElementsAre(fmt::format("{} to {} column 0", filterId, probeId)));
  }
}

TEST_F(DynamicFilterPlannerTest, localExchange) {
  core::PlanNodeId probeId;
  core::PlanNodeId buildId;
  core::PlanNodeId joinId;
  // The key is renamed and is the second column of the scan.
  auto root =
      scan("p", probeId)
          .project({"pv AS key", "pk AS payload"})
          .filter("payload > 0")
          .localPartition({"key"})
          .hashJoin(
              {"key"}, {"bk"}, scan("b", buildId).planNode(), "", {"payload"})
          .capturePlanNodeId(joinId)
          .planNode();
  auto result = plan(root);
  const auto filterId = joinId + ":0";
  EXPECT_THAT(
      sources(result),
      ElementsAre(fmt::format("{} from {} key 0", filterId, joinId)));
  EXPECT_THAT(
      targets(result),
      ElementsAre(fmt::format("{} to {} column 1", filterId, probeId)));
}

TEST_F(DynamicFilterPlannerTest, computedKey) {
  core::PlanNodeId probeId;
  core::PlanNodeId buildId;
  auto root =
      scan("p", probeId)
          .project({"pk + 1 AS key", "pv"})
          .localPartition({"key"})
          .hashJoin({"key"}, {"bk"}, scan("b", buildId).planNode(), "", {"pv"})
          .planNode();
  EXPECT_THAT(plan(root).sources, IsEmpty());
}

TEST_F(DynamicFilterPlannerTest, severalSourcesOfLocalExchange) {
  core::PlanNodeId firstId;
  core::PlanNodeId secondId;
  core::PlanNodeId buildId;
  core::PlanNodeId joinId;
  auto root =
      builder()
          .localPartition(
              {"pk"},
              {scan("p", firstId).planNode(), scan("p", secondId).planNode()})
          .hashJoin({"pk"}, {"bk"}, scan("b", buildId).planNode(), "", {"pv"})
          .capturePlanNodeId(joinId)
          .planNode();
  auto result = plan(root);
  const auto filterId = joinId + ":0";
  EXPECT_THAT(
      sources(result),
      ElementsAre(fmt::format("{} from {} key 0", filterId, joinId)));
  EXPECT_THAT(
      targets(result),
      UnorderedElementsAre(
          fmt::format("{} to {} column 0", filterId, firstId),
          fmt::format("{} to {} column 0", filterId, secondId)));
}

TEST_F(DynamicFilterPlannerTest, aggregation) {
  core::PlanNodeId probeId;
  core::PlanNodeId buildId;
  const auto makePlan = [&](const std::string& joinKey) {
    return scan("p", probeId)
        .partialAggregation({"pk"}, {"sum(pv) AS total"})
        .localPartition({"pk"})
        .finalAggregation()
        .hashJoin({joinKey}, {"bk"}, scan("b", buildId).planNode(), "", {"pk"})
        .planNode();
  };
  {
    // The key is a grouping key of both aggregations.
    auto result = plan(makePlan("pk"));
    ASSERT_EQ(result.targets.size(), 1);
    EXPECT_EQ(result.targets[0].scanNodeId, probeId);
    EXPECT_EQ(result.targets[0].outputChannel, 0);
  }
  {
    // The key is the result of an aggregate.
    EXPECT_THAT(plan(makePlan("total")).sources, IsEmpty());
  }
}

TEST_F(DynamicFilterPlannerTest, throughJoins) {
  core::PlanNodeId probeId;
  core::PlanNodeId firstBuildId;
  core::PlanNodeId secondBuildId;
  core::PlanNodeId firstJoinId;
  core::PlanNodeId secondJoinId;
  // The second join's key 'xv' comes from the build side of the first join,
  // whose scan runs in another pipeline. Its key 'pv' comes from the probe
  // side of the first join, in the pipeline of both joins.
  const auto makePlan = [&](core::JoinType firstJoinType,
                            const std::string& secondJoinKey) {
    return scan("p", probeId)
        .hashJoin(
            {"pk"},
            {"xk"},
            scan("x", firstBuildId).planNode(),
            "",
            {"pv", "xv"},
            firstJoinType)
        .capturePlanNodeId(firstJoinId)
        .hashJoin(
            {secondJoinKey},
            {"yk"},
            scan("y", secondBuildId).planNode(),
            "",
            {"pv"})
        .capturePlanNodeId(secondJoinId)
        .planNode();
  };
  {
    auto result = plan(makePlan(core::JoinType::kInner, "xv"));
    const auto filterId = secondJoinId + ":0";
    EXPECT_THAT(
        sources(result),
        ElementsAre(fmt::format("{} from {} key 0", filterId, secondJoinId)));
    EXPECT_THAT(
        targets(result),
        ElementsAre(fmt::format("{} to {} column 1", filterId, firstBuildId)));
  }
  {
    // The first join returns nulls for the build columns of probe rows
    // without a match, so a filter on a build column does not commute.
    EXPECT_THAT(plan(makePlan(core::JoinType::kLeft, "xv")).sources, IsEmpty());
  }
  {
    EXPECT_THAT(
        plan(makePlan(core::JoinType::kInner, "pv")).sources, IsEmpty());
    // The key goes through the probe side of the first join to the scan, in
    // the pipeline of both joins. The first join itself makes a filter too.
    auto result = planWithProbePipeline(makePlan(core::JoinType::kInner, "pv"));
    EXPECT_THAT(
        targets(result),
        UnorderedElementsAre(
            fmt::format("{}:0 to {} column 0", firstJoinId, probeId),
            fmt::format("{}:0 to {} column 1", secondJoinId, probeId)));
  }
}

TEST_F(DynamicFilterPlannerTest, nodesThatEndTheSearch) {
  // The search goes through the nodes it knows to commute with a filter and
  // stops at every other one. What most of these return for a row depends on
  // the other rows they get.
  const std::vector<std::pair<std::string, std::function<void(PlanBuilder&)>>>
      nodes = {
          {"limit", [](auto& builder) { builder.limit(0, 10, false); }},
          {"topN", [](auto& builder) { builder.topN({"pv"}, 10, false); }},
          {"orderBy", [](auto& builder) { builder.orderBy({"pv"}, false); }},
          {"markDistinct",
           [](auto& builder) { builder.markDistinct("marker", {"pv"}); }},
          {"window",
           [](auto& builder) {
             builder.window(
                 {"row_number() over (partition by pv order by pk)"});
           }},
          {"rowNumber", [](auto& builder) { builder.rowNumber({"pv"}); }},
          {"topNRowNumber",
           [](auto& builder) {
             builder.topNRowNumber({"pv"}, {"pk"}, 3, false);
           }},
          {"enforceSingleRow",
           [](auto& builder) { builder.enforceSingleRow(); }},
          {"assignUniqueId", [](auto& builder) { builder.assignUniqueId(); }},
          {"localMerge",
           [](auto& builder) {
             builder.orderBy({"pk"}, true).localMerge({"pk"});
           }},
      };
  for (const auto& [name, addNode] : nodes) {
    SCOPED_TRACE(name);
    core::PlanNodeId probeId;
    core::PlanNodeId buildId;
    auto probe = scan("p", probeId);
    addNode(probe);
    auto root =
        probe.localPartition({"pk"})
            .hashJoin({"pk"}, {"bk"}, scan("b", buildId).planNode(), "", {"pv"})
            .planNode();
    auto result = planWithProbePipeline(root);
    EXPECT_THAT(result.sources, IsEmpty());
    EXPECT_THAT(result.targets, IsEmpty());
  }
}

TEST_F(DynamicFilterPlannerTest, lowerJoinTypes) {
  core::PlanNodeId probeId;
  core::PlanNodeId firstBuildId;
  core::PlanNodeId secondBuildId;
  // The key of the upper join is 'pv' of the lower join's probe side or 'xv'
  // of its build side.
  const auto numFilters = [&](core::JoinType lowerJoinType,
                              const std::string& upperJoinKey) {
    core::PlanNodeId upperJoinId;
    auto root = scan("p", probeId)
                    .localPartition({"pk"})
                    .hashJoin(
                        {"pk"},
                        {"xk"},
                        scan("x", firstBuildId).planNode(),
                        "",
                        {upperJoinKey},
                        lowerJoinType)
                    .hashJoin(
                        {upperJoinKey},
                        {"yk"},
                        scan("y", secondBuildId).planNode(),
                        "",
                        {upperJoinKey})
                    .capturePlanNodeId(upperJoinId)
                    .planNode();
    const auto result = plan(root);
    return std::count_if(
        result.sources.begin(), result.sources.end(), [&](const auto& source) {
          return source.joinNodeId == upperJoinId;
        });
  };
  // A probe row comes out of these joins as it went in, or not at all.
  EXPECT_EQ(numFilters(core::JoinType::kInner, "pv"), 1);
  EXPECT_EQ(numFilters(core::JoinType::kLeft, "pv"), 1);
  EXPECT_EQ(numFilters(core::JoinType::kLeftSemiFilter, "pv"), 1);
  EXPECT_EQ(numFilters(core::JoinType::kAnti, "pv"), 1);
  // These add rows with null probe columns for build rows without a match.
  EXPECT_EQ(numFilters(core::JoinType::kRight, "pv"), 0);
  EXPECT_EQ(numFilters(core::JoinType::kFull, "pv"), 0);
  // Likewise for the build side.
  EXPECT_EQ(numFilters(core::JoinType::kInner, "xv"), 1);
  EXPECT_EQ(numFilters(core::JoinType::kRight, "xv"), 1);
  EXPECT_EQ(numFilters(core::JoinType::kLeft, "xv"), 0);
  EXPECT_EQ(numFilters(core::JoinType::kFull, "xv"), 0);
}

TEST_F(DynamicFilterPlannerTest, twoJoinsOnOneColumn) {
  core::PlanNodeId probeId;
  core::PlanNodeId firstBuildId;
  core::PlanNodeId secondBuildId;
  core::PlanNodeId firstJoinId;
  core::PlanNodeId secondJoinId;
  auto root =
      scan("p", probeId)
          .localPartition({"pk"})
          .hashJoin(
              {"pk"}, {"xk"}, scan("x", firstBuildId).planNode(), "", {"pk"})
          .capturePlanNodeId(firstJoinId)
          .hashJoin(
              {"pk"}, {"yk"}, scan("y", secondBuildId).planNode(), "", {"pk"})
          .capturePlanNodeId(secondJoinId)
          .planNode();
  EXPECT_THAT(
      targets(plan(root)),
      UnorderedElementsAre(
          fmt::format("{}:0 to {} column 0", firstJoinId, probeId),
          fmt::format("{}:0 to {} column 0", secondJoinId, probeId)));
}

TEST_F(DynamicFilterPlannerTest, selectiveBuildOnly) {
  core::PlanNodeId probeId;
  core::PlanNodeId buildId;
  core::PlanNodeId dimensionId;
  const auto numFilters = [&](const core::PlanNodePtr& build) {
    auto root = scan("p", probeId)
                    .localPartition({"pk"})
                    .hashJoin({"pk"}, {"bk"}, build, "", {"pv"})
                    .planNode();
    return DynamicFilterPlanner::plan(*root, {}).sources.size();
  };
  // The build side is a whole table: its keys are likely all the keys.
  EXPECT_EQ(numFilters(scan("b", buildId).planNode()), 0);
  EXPECT_EQ(
      numFilters(scan("b", buildId).project({"bk", "bv + 1"}).planNode()), 0);
  // A filter in the scan or after it.
  EXPECT_EQ(
      numFilters(
          builder()
              .tableScan(ROW({"bk", "bv"}, BIGINT()), {"bv < 10"})
              .planNode()),
      1);
  EXPECT_EQ(numFilters(scan("b", buildId).filter("bv % 2 = 0").planNode()), 1);
  // A join with a filtered table removes rows, a join with a whole table is
  // taken to keep them.
  const auto joinWithDimension = [&](const core::PlanNodePtr& dimension) {
    return scan("b", buildId)
        .hashJoin({"bv"}, {"dk"}, dimension, "", {"bk"})
        .planNode();
  };
  EXPECT_EQ(
      numFilters(joinWithDimension(scan("d", dimensionId).planNode())), 0);
  EXPECT_EQ(
      numFilters(joinWithDimension(
          scan("d", dimensionId).filter("dv = 1").planNode())),
      1);
  // Unknown input.
  EXPECT_EQ(
      numFilters(
          builder()
              .values(
                  {makeRowVector({"bk"}, {makeFlatVector<int64_t>({1, 2, 3})})})
              .planNode()),
      1);
}

TEST_F(DynamicFilterPlannerTest, scanFilteredInItsPipeline) {
  core::PlanNodeId probeId;
  core::PlanNodeId lowerBuildId;
  core::PlanNodeId upperBuildId;
  core::PlanNodeId upperJoinId;
  // The upper join's key reaches the scan through an aggregation, a local
  // exchange and the probe side of the lower join, which is in the pipeline of
  // the scan.
  const auto makePlan = [&](bool lowerBuildFiltered) {
    auto lowerBuild = scan("x", lowerBuildId);
    if (lowerBuildFiltered) {
      lowerBuild.filter("xv > 0");
    }
    return scan("p", probeId)
        .hashJoin({"pk"}, {"xk"}, lowerBuild.planNode(), "", {"pk", "pv"})
        .partialAggregation({"pk"}, {"sum(pv) AS total"})
        .localPartition({"pk"})
        .finalAggregation()
        .hashJoin(
            {"pk"},
            {"yk"},
            scan("y", upperBuildId).filter("yv > 0").planNode(),
            "",
            {"pk", "total"})
        .capturePlanNodeId(upperJoinId)
        .planNode();
  };
  const auto numUpperFilters =
      [&](const core::PlanNodePtr& root,
          const DynamicFilterPlanner::Options& options) {
        const auto result = DynamicFilterPlanner::plan(*root, options);
        return std::count_if(
            result.sources.begin(),
            result.sources.end(),
            [&](const auto& source) {
              return source.joinNodeId == upperJoinId;
            });
      };
  // The lower join returns every probe row, so the upper join's filter is the
  // only one for the scan.
  EXPECT_EQ(numUpperFilters(makePlan(false), {}), 1);
  // The lower join already filters the scan with its own keys.
  EXPECT_EQ(numUpperFilters(makePlan(true), {}), 0);
  EXPECT_EQ(
      numUpperFilters(makePlan(true), {.skipScansFilteredInPipeline = false}),
      1);
}

TEST_F(DynamicFilterPlannerTest, joinTypes) {
  core::PlanNodeId probeId;
  core::PlanNodeId buildId;
  const auto numFilters = [&](core::JoinType joinType,
                              bool nullAware = false,
                              bool nullAsValue = false) {
    std::vector<std::string> output{"pv"};
    if (isRightSemiFilterJoin(joinType)) {
      output = {"bv"};
    }
    auto root = scan("p", probeId)
                    .localPartition({"pk"})
                    .hashJoin(
                        {"pk"},
                        {"bk"},
                        scan("b", buildId).planNode(),
                        "",
                        output,
                        joinType,
                        nullAware,
                        nullAsValue)
                    .planNode();
    return plan(root).sources.size();
  };
  EXPECT_EQ(numFilters(core::JoinType::kInner), 1);
  EXPECT_EQ(numFilters(core::JoinType::kLeftSemiFilter), 1);
  EXPECT_EQ(numFilters(core::JoinType::kRightSemiFilter), 1);
  EXPECT_EQ(numFilters(core::JoinType::kRight), 1);
  // These return probe rows without a match.
  EXPECT_EQ(numFilters(core::JoinType::kLeft), 0);
  EXPECT_EQ(numFilters(core::JoinType::kFull), 0);
  EXPECT_EQ(numFilters(core::JoinType::kAnti), 0);
  // A null key decides the result of these.
  EXPECT_EQ(numFilters(core::JoinType::kAnti, true), 0);
  EXPECT_EQ(numFilters(core::JoinType::kInner, false, true), 0);
}

TEST_F(DynamicFilterPlannerTest, keys) {
  core::PlanNodeId probeId;
  core::PlanNodeId buildId;
  core::PlanNodeId joinId;
  const auto probeType = ROW({
      {"a", BIGINT()},
      {"b", VARCHAR()},
      {"c", DATE()},
      {"d", DECIMAL(10, 2)},
      {"e", INTEGER()},
  });
  const auto buildType = ROW({
      {"u", BIGINT()},
      {"v", VARCHAR()},
      {"w", DATE()},
      {"x", DECIMAL(10, 2)},
      {"y", INTEGER()},
  });
  auto root = builder()
                  .tableScan(probeType)
                  .capturePlanNodeId(probeId)
                  .localPartition({"a"})
                  .hashJoin(
                      {"a", "b", "c", "d", "e"},
                      {"u", "v", "w", "x", "y"},
                      builder()
                          .tableScan(buildType)
                          .capturePlanNodeId(buildId)
                          .planNode(),
                      "",
                      {"a"})
                  .capturePlanNodeId(joinId)
                  .planNode();
  // One filter for each integer and date key.
  EXPECT_THAT(
      targets(plan(root)),
      ElementsAre(
          fmt::format("{}:0 to {} column 0", joinId, probeId),
          fmt::format("{}:2 to {} column 2", joinId, probeId),
          fmt::format("{}:4 to {} column 4", joinId, probeId)));
}

} // namespace
} // namespace facebook::velox::exec::test
