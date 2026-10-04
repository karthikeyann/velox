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

// Tests for the GPU -> CPU boundary of the self-describing decimal aggregate
// state (design doc sections 6.2 and 6.4): CudfToVelox packs a VARBINARY
// column that is physically a decimal state STRUCT into the 32-byte blob, and
// GPU expressions refuse to compute over such a column while pass-through
// operators (identity projection, filter on another column, limit) carry it
// untouched.

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <cudf/table/table.hpp>

#include <cuda_runtime_api.h>

#include <cstring>
#include <optional>

namespace facebook::velox::cudf_velox {
namespace {

using exec::test::AssertQueryBuilder;
using exec::test::PlanBuilder;

constexpr const char* kGuardMessage =
    "expression over an unmaterialized decimal aggregate state column is not "
    "supported";

/// Decoded 32-byte decimal state blob: count int64, overflow int64, sum low
/// uint64, sum high int64 (little endian).
struct DecodedState {
  int64_t count;
  int64_t overflow;
  int128_t sum;
};

DecodedState decodeState(const StringView& blob) {
  VELOX_CHECK_EQ(blob.size(), 32, "decimal state blob must be 32 bytes");
  DecodedState state;
  uint64_t low;
  int64_t high;
  std::memcpy(&state.count, blob.data(), sizeof(int64_t));
  std::memcpy(&state.overflow, blob.data() + 8, sizeof(int64_t));
  std::memcpy(&low, blob.data() + 16, sizeof(uint64_t));
  std::memcpy(&high, blob.data() + 24, sizeof(int64_t));
  state.sum = (static_cast<int128_t>(high) << 64) | static_cast<int128_t>(low);
  return state;
}

std::string shapeName(DecimalStateShape shape) {
  switch (shape) {
    case DecimalStateShape::kSum64:
      return "kSum64";
    case DecimalStateShape::kSum128:
      return "kSum128";
    case DecimalStateShape::kAvg64:
      return "kAvg64";
    case DecimalStateShape::kAvg128:
      return "kAvg128";
  }
  return "unknown";
}

/// Test-only GPU operator. Consumes (k, sum DECIMAL(38, s), count BIGINT,
/// overflow BIGINT) and emits (k, state) where `state` is a decimal state
/// STRUCT of the configured shape under the plan's VARBINARY logical type,
/// exactly as a GPU partial aggregation would.
class MakeDecimalStateOperator : public CudfOperatorBase {
 public:
  MakeDecimalStateOperator(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      RowTypePtr outputType,
      const core::PlanNodeId& planNodeId,
      DecimalStateShape shape)
      : CudfOperatorBase(
            operatorId,
            driverCtx,
            std::move(outputType),
            planNodeId,
            "MakeDecimalState"),
        shape_(shape) {}

  bool needsInput() const override {
    return !input_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && !input_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }

  RowVectorPtr doGetOutput() override {
    if (!input_) {
      return nullptr;
    }
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
    input_.reset();
    VELOX_CHECK_NOT_NULL(cudfInput);
    const auto stream = cudfInput->stream();
    const auto size = cudfInput->size();
    auto columns = cudfInput->release()->release();
    VELOX_CHECK_EQ(columns.size(), 4);

    DecimalStateColumns flat;
    flat.sum = std::move(columns[1]);
    flat.count = std::move(columns[2]);
    flat.overflow = std::move(columns[3]);

    std::vector<std::unique_ptr<cudf::column>> output;
    output.push_back(std::move(columns[0]));
    output.push_back(
        wrapDecimalState(std::move(flat), shape_, stream, get_output_mr()));
    VELOX_CHECK(isDecimalStateStruct(output.back()->view()));
    return std::make_shared<CudfVector>(
        pool(),
        outputType_,
        size,
        std::make_unique<cudf::table>(std::move(output)),
        stream);
  }

 private:
  const DecimalStateShape shape_;
};

/// Replaces the CPU operator of one plan node with MakeDecimalStateOperator.
class MakeDecimalStateAdapter : public OperatorAdapter {
 public:
  MakeDecimalStateAdapter(core::PlanNodeId planNodeId, DecimalStateShape shape)
      : OperatorAdapter("MakeDecimalState"),
        planNodeId_(std::move(planNodeId)),
        shape_(shape) {}

  bool canHandle(const exec::Operator* op) const override {
    return op->planNodeId() == planNodeId_;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return true;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    std::vector<std::unique_ptr<exec::Operator>> replacements;
    replacements.push_back(
        std::make_unique<MakeDecimalStateOperator>(
            operatorId, ctx, planNode->outputType(), planNode->id(), shape_));
    return replacements;
  }

 private:
  const core::PlanNodeId planNodeId_;
  const DecimalStateShape shape_;
};

class DecimalStateConversionTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No usable CUDA device";
    }
    VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
    savedCpuFallback_ = CudfConfig::getInstance().allowCpuFallback;
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    CudfConfig::getInstance().allowCpuFallback = savedCpuFallback_;
    exec::test::OperatorTestBase::TearDown();
  }

  /// Flat state fields, row i has key i. Row 2 has a null sum (null state).
  struct StateRows {
    std::vector<std::optional<int128_t>> sums;
    std::vector<int64_t> counts;
    std::vector<int64_t> overflows;
  };

  static StateRows stateRows() {
    const int128_t big = static_cast<int128_t>(1) << 100;
    return {
        {12'345, -2'500, std::nullopt, big, -big, 0},
        {3, 1, 7, 42, 5, 9},
        {0, 2, 0, -1, 3, 0},
    };
  }

  /// Splits the rows across two batches so CudfToVelox sees several GPU
  /// inputs (exercising the device concat in non-passthrough mode).
  std::vector<RowVectorPtr> stateInput(const StateRows& rows) {
    std::vector<RowVectorPtr> batches;
    const size_t half = rows.sums.size() / 2;
    for (auto [begin, end] :
         {std::pair<size_t, size_t>{0, half},
          std::pair<size_t, size_t>{half, rows.sums.size()}}) {
      std::vector<int32_t> keys;
      std::vector<std::optional<int128_t>> sums;
      std::vector<int64_t> counts;
      std::vector<int64_t> overflows;
      for (size_t i = begin; i < end; ++i) {
        keys.push_back(static_cast<int32_t>(i));
        sums.push_back(rows.sums[i]);
        counts.push_back(rows.counts[i]);
        overflows.push_back(rows.overflows[i]);
      }
      batches.push_back(makeRowVector(
          {"k", "sum", "cnt", "ovf"},
          {makeFlatVector<int32_t>(keys),
           makeNullableFlatVector<int128_t>(sums, DECIMAL(38, 2)),
           makeFlatVector<int64_t>(counts),
           makeFlatVector<int64_t>(overflows)}));
    }
    return batches;
  }

  /// Values -> (replaced) project producing (k INTEGER, s VARBINARY) where s
  /// is physically a decimal state STRUCT of `shape` on the GPU.
  PlanBuilder stateSource(
      const std::vector<RowVectorPtr>& input,
      DecimalStateShape shape) {
    core::PlanNodeId markerId;
    auto builder = PlanBuilder(planNodeIdGenerator_)
                       .values(input)
                       .project({"k", "to_utf8(cast(cnt AS varchar)) AS s"})
                       .capturePlanNodeId(markerId);
    OperatorAdapterRegistry::getInstance().registerAdapterFront(
        std::make_unique<MakeDecimalStateAdapter>(markerId, shape));
    return builder;
  }

  RowVectorPtr run(const core::PlanNodePtr& plan, bool passthrough) {
    return AssertQueryBuilder(plan)
        .config(CudfToVelox::kPassthroughMode, passthrough ? "true" : "false")
        .config(CudfFromVelox::kGpuBatchSizeRows, "1")
        .copyResults(pool());
  }

  /// Checks that column `stateName` of `result` holds the packed blobs for
  /// `rows` (keyed by column "k"), with absent fields filled as count=1,
  /// overflow=0.
  void verifyStates(
      const RowVectorPtr& result,
      const StateRows& rows,
      DecimalStateShape shape,
      const std::string& stateName = "s") {
    const auto& rowType = result->type()->asRow();
    auto keys =
        result->childAt(rowType.getChildIdx("k"))->asFlatVector<int32_t>();
    auto states = result->childAt(rowType.getChildIdx(stateName));
    ASSERT_EQ(states->type()->kind(), TypeKind::VARBINARY);
    auto flatStates = states->asFlatVector<StringView>();
    ASSERT_NE(flatStates, nullptr);
    ASSERT_NE(keys, nullptr);
    for (vector_size_t row = 0; row < result->size(); ++row) {
      const auto key = keys->valueAt(row);
      SCOPED_TRACE(fmt::format("shape {} key {}", shapeName(shape), key));
      const auto& expectedSum = rows.sums.at(key);
      if (!expectedSum.has_value()) {
        EXPECT_TRUE(flatStates->isNullAt(row));
        continue;
      }
      ASSERT_FALSE(flatStates->isNullAt(row));
      ASSERT_EQ(flatStates->valueAt(row).size(), 32);
      const auto decoded = decodeState(flatStates->valueAt(row));
      EXPECT_EQ(decoded.sum, *expectedSum);
      EXPECT_EQ(
          decoded.count, decimalStateHasCount(shape) ? rows.counts.at(key) : 1);
      EXPECT_EQ(
          decoded.overflow,
          decimalStateHasOverflow(shape) ? rows.overflows.at(key) : 0);
    }
  }

  std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator_{
      std::make_shared<core::PlanNodeIdGenerator>()};
  bool savedCpuFallback_{false};
};

constexpr DecimalStateShape kAllShapes[] = {
    DecimalStateShape::kSum64,
    DecimalStateShape::kSum128,
    DecimalStateShape::kAvg64,
    DecimalStateShape::kAvg128,
};

// (a) + (b): a VARBINARY column that is physically a decimal state STRUCT of
// every shape converts through CudfToVelox into 32-byte blobs, on both the
// passthrough path and the device-concat path.
TEST_F(DecimalStateConversionTest, structStateToVeloxAllShapes) {
  const auto rows = stateRows();
  const auto input = stateInput(rows);
  for (const auto shape : kAllShapes) {
    for (const bool passthrough : {true, false}) {
      SCOPED_TRACE(
          fmt::format(
              "shape {} passthrough {}", shapeName(shape), passthrough));
      OperatorAdapterRegistry::getInstance().clear();
      unregisterCudf();
      registerCudf();
      auto plan = stateSource(input, shape).planNode();
      auto result = run(plan, passthrough);
      ASSERT_EQ(result->size(), rows.sums.size());
      verifyStates(result, rows, shape);
    }
  }
}

// (d) Pass-through operators must carry the struct untouched: identity
// projection that reorders and renames, a filter and a computed projection on
// another column, and a limit.
TEST_F(DecimalStateConversionTest, passThroughOperatorsDoNotTriggerGuard) {
  const auto rows = stateRows();
  const auto input = stateInput(rows);
  const auto shape = DecimalStateShape::kAvg128;

  {
    SCOPED_TRACE("identity reorder and rename");
    auto plan =
        stateSource(input, shape).project({"s AS state", "k"}).planNode();
    auto result = run(plan, false);
    ASSERT_EQ(result->size(), rows.sums.size());
    verifyStates(result, rows, shape, "state");
  }
  unregisterCudf();
  registerCudf();
  {
    SCOPED_TRACE("filter and computed projection on another column");
    auto plan = stateSource(input, shape)
                    .filter("k % 2 = 0")
                    .project({"k", "s", "k + 1 AS k1"})
                    .planNode();
    auto result = run(plan, true);
    ASSERT_EQ(result->size(), 3);
    verifyStates(result, rows, shape);
  }
  unregisterCudf();
  registerCudf();
  {
    SCOPED_TRACE("limit");
    auto plan = stateSource(input, shape).limit(0, 2, false).planNode();
    auto result = run(plan, true);
    ASSERT_EQ(result->size(), 2);
    verifyStates(result, rows, shape);
  }
}

// (d) A computed expression over the struct-typed VARBINARY column fails with
// a clear message, in a projection and in a filter.
TEST_F(DecimalStateConversionTest, computedExpressionOverStateFails) {
  const auto rows = stateRows();
  const auto input = stateInput(rows);
  const auto shape = DecimalStateShape::kSum64;

  {
    auto plan =
        stateSource(input, shape).project({"k", "s IS NULL AS n"}).planNode();
    VELOX_ASSERT_THROW(run(plan, true), kGuardMessage);
  }
  unregisterCudf();
  registerCudf();
  {
    auto plan = stateSource(input, shape).filter("s IS NULL").planNode();
    VELOX_ASSERT_THROW(run(plan, true), kGuardMessage);
  }
}

// (c) GPU partial aggregation feeding a CPU final aggregation (the HTTP
// exchange and CPU-fallback shape) returns the same results as pure CPU for
// SUM and AVG over short and long decimals, grouped and global, including
// all-null groups and a null key.
TEST_F(DecimalStateConversionTest, gpuPartialCpuFinalMatchesCpu) {
  const int128_t big = static_cast<int128_t>(1) << 70;
  auto makeBatch = [&](std::vector<std::optional<int32_t>> keys,
                       std::vector<std::optional<int64_t>> shortValues,
                       std::vector<std::optional<int128_t>> longValues) {
    return makeRowVector(
        {"k", "d64", "d128"},
        {makeNullableFlatVector<int32_t>(keys),
         makeNullableFlatVector<int64_t>(shortValues, DECIMAL(12, 2)),
         makeNullableFlatVector<int128_t>(longValues, DECIMAL(25, 2))});
  };
  std::vector<RowVectorPtr> input = {
      makeBatch(
          {1, 1, 2, 2, 3, std::nullopt},
          {100, -250, std::nullopt, std::nullopt, 999'999'999'999, 7},
          {big, -3, std::nullopt, std::nullopt, 12'345, 8}),
      makeBatch(
          {1, 3, 4, std::nullopt, 2},
          {33, -999'999'999'999, 5, 9, std::nullopt},
          {big, -big, 1, std::nullopt, std::nullopt}),
  };

  const std::vector<std::string> partialAggs = {
      "sum(d64) AS s64",
      "avg(d64) AS a64",
      "sum(d128) AS s128",
      "avg(d128) AS a128"};
  const std::vector<std::string> finalAggs = {
      "sum(s64) AS s64",
      "avg(a64) AS a64",
      "sum(s128) AS s128",
      "avg(a128) AS a128"};
  const std::vector<std::vector<TypePtr>> rawInputTypes = {
      {DECIMAL(12, 2)}, {DECIMAL(12, 2)}, {DECIMAL(25, 2)}, {DECIMAL(25, 2)}};

  for (const bool grouped : {true, false}) {
    for (const bool passthrough : {true, false}) {
      SCOPED_TRACE(
          fmt::format("grouped {} passthrough {}", grouped, passthrough));
      const std::vector<std::string> keys =
          grouped ? std::vector<std::string>{"k"} : std::vector<std::string>{};

      // GPU partial aggregation; CudfToVelox at the task boundary converts
      // the intermediate states to CPU VARBINARY.
      registerCudf();
      auto partialPlan = PlanBuilder()
                             .values(input)
                             .partialAggregation(keys, partialAggs)
                             .planNode();
      auto intermediate = run(partialPlan, passthrough);
      for (const auto& name : {"s64", "a64", "s128", "a128"}) {
        ASSERT_EQ(
            intermediate->childAt(name)->type()->kind(), TypeKind::VARBINARY);
      }

      // CPU final aggregation over the GPU-produced states versus a pure CPU
      // single aggregation.
      unregisterCudf();
      auto finalPlan = PlanBuilder()
                           .values({intermediate})
                           .finalAggregation(keys, finalAggs, rawInputTypes)
                           .planNode();
      auto actual = AssertQueryBuilder(finalPlan).copyResults(pool());
      auto expectedPlan = PlanBuilder()
                              .values(input)
                              .singleAggregation(keys, partialAggs)
                              .planNode();
      auto expected = AssertQueryBuilder(expectedPlan).copyResults(pool());
      ASSERT_EQ(actual->size(), expected->size());
      EXPECT_TRUE(exec::test::assertEqualResults({expected}, {actual}));
    }
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox
