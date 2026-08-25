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

#include "velox/experimental/cudf/benchmarks/CudfBenchmarkDiscard.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TaskStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox;

// Test fixture that registers cuDF (with the BenchmarkDiscard adapter) before
// each test and unregisters afterwards.
class CudfBenchmarkDiscardTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    registerCudf();
    registerCudfBenchmarkDiscard();
  }

  void TearDown() override {
    unregisterCudf();
    OperatorTestBase::TearDown();
  }
};

// Test fixture that does NOT register cuDF, so BenchmarkDiscard receives
// plain RowVectors and its dynamic type check fires.
class CudfBenchmarkDiscardNoGpuTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    // Register only the plan-node translator so the plan is valid,
    // but do not register the cuDF adapter so no CudfFromVelox is inserted.
    registerCudfBenchmarkDiscard();
  }

  void TearDown() override {
    OperatorTestBase::TearDown();
  }
};

// --- tests ---

// Multiple input batches must be aggregated into a single set of runtime
// counters.
TEST_F(CudfBenchmarkDiscardTest, aggregatesCountersAcrossBatches) {
  // Two batches of 3 rows each => 6 total rows, 2 batches.
  auto batch1 = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3})});
  auto batch2 = makeRowVector({"c0"}, {makeFlatVector<int32_t>({4, 5, 6})});

  // Build the plan and capture the discard node ID from the actual node.
  core::PlanNodeId discardId;
  auto plan =
      PlanBuilder()
          .values({batch1, batch2})
          .addNode([&](std::string /*lambdaId*/, core::PlanNodePtr src) {
            auto node = addBenchmarkDiscard(std::move(src));
            discardId = node->id();
            return node;
          })
          .planNode();

  std::shared_ptr<exec::Task> task;
  AssertQueryBuilder(plan).copyResults(pool(), task);

  auto planStats = toPlanStats(task->taskStats());
  ASSERT_TRUE(planStats.count(discardId))
      << "discardId=" << discardId
      << " not found in planStats (size=" << planStats.size() << ")";
  const auto& nodeStats = planStats.at(discardId);
  const auto& cs = nodeStats.customStats;

  ASSERT_TRUE(cs.count(std::string(kDiscardedRows)));
  ASSERT_TRUE(cs.count(std::string(kDiscardedBatches)));
  ASSERT_TRUE(cs.count(std::string(kDiscardedBytes)));

  // The GPU pipeline may coalesce the two CPU batches into one CudfVector, so
  // assert rows=6 and batches>=1 rather than batches==2.
  EXPECT_EQ(cs.at(std::string(kDiscardedRows)).sum, 6);
  EXPECT_GE(cs.at(std::string(kDiscardedBatches)).sum, 1);
  // Bytes > 0 because CudfVector has device storage.
  EXPECT_GT(cs.at(std::string(kDiscardedBytes)).sum, 0);
}

// The cursor/task result must be empty (null or zero-row vector).
TEST_F(CudfBenchmarkDiscardTest, cursorResultIsEmpty) {
  auto batch = makeRowVector({"c0"}, {makeFlatVector<int32_t>({10, 20})});
  auto plan = PlanBuilder()
                  .values({batch})
                  .addNode([](std::string id, core::PlanNodePtr src) {
                    return addBenchmarkDiscard(std::move(src));
                  })
                  .planNode();

  auto results = AssertQueryBuilder(plan).copyResults(pool());
  // BenchmarkDiscard produces no output; copyResults returns null or empty.
  EXPECT_TRUE(results == nullptr || results->size() == 0);
}

// CudfToVelox must be absent from pipeline operator stats.
TEST_F(CudfBenchmarkDiscardTest, cudfToVeloxAbsentFromPipeline) {
  auto batch = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1})});
  auto plan = PlanBuilder()
                  .values({batch})
                  .addNode([](std::string id, core::PlanNodePtr src) {
                    return addBenchmarkDiscard(std::move(src));
                  })
                  .planNode();

  std::shared_ptr<exec::Task> task;
  AssertQueryBuilder(plan).copyResults(pool(), task);

  // Walk every operator in every pipeline and assert CudfToVelox is absent.
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& opStats : pipeline.operatorStats) {
      EXPECT_NE(opStats.operatorType, "CudfToVelox")
          << "CudfToVelox should not appear after BenchmarkDiscard";
    }
  }
}

// Receiving a non-CudfVector must produce a clear user-facing error.
TEST_F(CudfBenchmarkDiscardNoGpuTest, nonCudfVectorThrowsClearError) {
  auto batch = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3})});
  auto plan = PlanBuilder()
                  .values({batch})
                  .addNode([](std::string id, core::PlanNodePtr src) {
                    return addBenchmarkDiscard(std::move(src));
                  })
                  .planNode();

  EXPECT_THROW(AssertQueryBuilder(plan).copyResults(pool()), std::exception);
}
