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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/CudfReduce.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/tests/DecimalStateTestColumns.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/DecimalAggregate.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/DecimalUtil.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <folly/String.h>

#include <array>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <type_traits>

namespace facebook::velox::cudf_velox::test {
namespace {

int64_t computeAvgRaw(const std::vector<int64_t>& values) {
  int128_t sum = 0;
  for (auto value : values) {
    sum += value;
  }
  int128_t avg = 0;
  facebook::velox::DecimalUtil::computeAverage(avg, sum, values.size(), 0);
  return static_cast<int64_t>(avg);
}

// computeDecimalAverage for a state without carries: the overflow field is an
// all-zero INT64 column.
std::unique_ptr<cudf::column> computeDecimalAverageWithoutOverflow(
    const cudf::column_view& sum,
    const cudf::column_view& count,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto overflow =
      makeInt64Column(std::vector<int64_t>(sum.size(), 0), nullptr, stream);
  return computeDecimalAverage(sum, count, overflow->view(), stream, mr);
}

class CudfDecimalTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    CudfConfig::getInstance().allowCpuFallback = false;
    // Ensure a CUDA device is selected and initialized (RMM asserts otherwise).
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
      GTEST_SKIP() << "cudaGetDeviceCount failed: " << static_cast<int>(status)
                   << " (" << cudaGetErrorString(status) << ")";
    }
    if (deviceCount == 0) {
      GTEST_SKIP() << "No CUDA devices visible (check CUDA_VISIBLE_DEVICES)";
    }
    VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
    VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(nullptr)));
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    exec::test::OperatorTestBase::TearDown();
  }

  bool hasStreamingGroupbyStat(
      const std::shared_ptr<exec::Task>& task,
      const core::PlanNodeId& planNodeId) {
    const auto planStats = exec::toPlanStats(task->taskStats());
    const auto it = planStats.find(planNodeId);
    return it != planStats.end() &&
        it->second.customStats.count(std::string{kStreamingGroupbyUsedStat}) >
        0;
  }

  // Grouped decimal input covering positive, negative, cancelling, partially
  // null and all-null groups plus DECIMAL64 extremes (for scale-0 DECIMAL(18)
  // the extremes are +/-(10^18 - 1)). Returned as several small batches.
  std::vector<RowVectorPtr> makeGroupedDecimalBatches(const TypePtr& type) {
    const int64_t extreme = 999'999'999'999'999'999;
    const std::vector<int32_t> keys{1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6};
    const std::vector<std::optional<int64_t>> values{
        12345,
        250,
        -12345,
        -250,
        700,
        -700,
        0,
        100,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        extreme,
        extreme,
        -extreme};
    std::vector<RowVectorPtr> batches;
    for (int32_t batch = 0; batch < 3; ++batch) {
      // Rotate so each batch has a different group mix.
      std::vector<int32_t> batchKeys;
      std::vector<std::optional<int64_t>> batchValues;
      for (size_t i = 0; i < keys.size(); ++i) {
        const auto index = (i + batch * 5) % keys.size();
        batchKeys.push_back(keys[index]);
        batchValues.push_back(values[index]);
      }
      VectorPtr valueVector;
      if (type->isShortDecimal()) {
        valueVector = makeNullableFlatVector<int64_t>(batchValues, type);
      } else {
        std::vector<std::optional<int128_t>> wide;
        for (const auto& value : batchValues) {
          wide.push_back(
              value.has_value() ? std::make_optional<int128_t>(*value)
                                : std::nullopt);
        }
        valueVector = makeNullableFlatVector<int128_t>(wide, type);
      }
      batches.push_back(makeRowVector(
          {"k", "d"}, {makeFlatVector<int32_t>(batchKeys), valueVector}));
    }
    return batches;
  }
};

TEST_F(CudfDecimalTest, mixedWidthDecimalDivision) {
  const auto rowType = ROW({
      {"short_decimal", DECIMAL(7, 2)},
      {"long_decimal", DECIMAL(20, 3)},
  });
  auto queryCtx = core::QueryCtx::create();
  core::ExecCtx execCtx(pool(), queryCtx.get());
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  const std::vector<bool> shortValid{true, true, true, false, true};
  const std::vector<bool> longValid{true, true, true, true, false};
  // Narrowing this denominator to int64_t would turn it into zero.
  const int128_t largeDenominator = int128_t{1} << 64;
  auto shortDecimal = makeDecimalColumn<int64_t>(
      {1'000, -1'000, 1'000, 1'000, 0}, 2, &shortValid, stream);
  auto longDecimal = makeDecimalColumn<int128_t>(
      {2'000, 2'000, largeDenominator, 2'000, 2'000}, 3, &longValid, stream);
  const std::vector<cudf::column_view> inputs{
      shortDecimal->view(), longDecimal->view()};
  // short_decimal's zero is only reachable as a divisor when the other operand
  // is a scalar: every column operand is null on that row. Value checks for
  // that shape use a divisor column without the zero.
  auto zeroFreeShortDecimal = makeDecimalColumn<int64_t>(
      {1'000, -1'000, 1'000, 1'000, 500}, 2, &shortValid, stream);
  auto zeroFreeLongDecimal = makeDecimalColumn<int128_t>(
      {2'000, 2'000, largeDenominator, 2'000, 2'000}, 3, &longValid, stream);
  const std::vector<cudf::column_view> zeroFreeInputs{
      zeroFreeShortDecimal->view(), longDecimal->view()};

  auto assertDivision =
      [&](const std::string& sql,
          const TypePtr& expectedType,
          const std::vector<std::optional<int128_t>>& expected,
          const std::vector<cudf::column_view>& evalInputs) {
        SCOPED_TRACE(sql);
        auto expression = test_utils::optimizeTypedExpr(
            sql, rowType, queryCtx.get(), &execCtx);
        ASSERT_TRUE(expression->type()->equivalent(*expectedType));
        auto evaluator = createCudfExpression(expression, rowType, pool());
        auto result = evaluator->eval(evalInputs, stream, mr);
        const auto view = asView(result);
        ASSERT_EQ(view.type(), veloxToCudfDataType(expectedType));
        ASSERT_EQ(view.size(), expected.size());
        auto values = [&]() -> std::vector<int128_t> {
          if (expectedType->isShortDecimal()) {
            const auto shortValues = copyColumnData<int64_t>(view, stream);
            return {shortValues.begin(), shortValues.end()};
          }
          return copyColumnData<int128_t>(view, stream);
        }();
        const auto nullMask = copyNullMask(view, stream);
        cudf::size_type expectedNulls = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
          ASSERT_EQ(isValidAt(nullMask, i), expected[i].has_value());
          if (expected[i]) {
            EXPECT_EQ(values[i], *expected[i]);
          } else {
            ++expectedNulls;
          }
        }
        EXPECT_EQ(view.null_count(), expectedNulls);
      };

  assertDivision(
      "short_decimal / long_decimal",
      DECIMAL(11, 3),
      {5'000, -5'000, 0, std::nullopt, std::nullopt},
      inputs);
  // The last row divides by short_decimal's zero, but long_decimal is null
  // so the row is discarded before the divisor is inspected.
  assertDivision(
      "long_decimal / short_decimal",
      DECIMAL(22, 3),
      {200, -200, 1'844'674'407'370'955'162, std::nullopt, std::nullopt},
      inputs);
  assertDivision(
      "short_decimal / CAST('2.000' AS DECIMAL(20, 3))",
      DECIMAL(11, 3),
      {5'000, -5'000, 5'000, std::nullopt, 0},
      inputs);
  assertDivision(
      "long_decimal / CAST('10.00' AS DECIMAL(7, 2))",
      DECIMAL(22, 3),
      {200, 200, 1'844'674'407'370'955'162, 200, std::nullopt},
      inputs);
  assertDivision(
      "CAST('10.00' AS DECIMAL(7, 2)) / long_decimal",
      DECIMAL(11, 3),
      {5'000, 5'000, 0, 5'000, std::nullopt},
      inputs);
  assertDivision(
      "CAST('2.000' AS DECIMAL(20, 3)) / short_decimal",
      DECIMAL(22, 3),
      {200, -200, 200, std::nullopt, 400},
      zeroFreeInputs);
  assertDivision(
      "short_decimal / CAST('18446744073709551.616' AS DECIMAL(20, 3))",
      DECIMAL(11, 3),
      {0, 0, 0, std::nullopt, 0},
      zeroFreeInputs);
  // A zero divisor on a row that survives the null stencil fails the whole
  // expression, as decimal divide does on the CPU.
  {
    const std::string sql = "CAST('2.000' AS DECIMAL(20, 3)) / short_decimal";
    SCOPED_TRACE(sql);
    auto expression =
        test_utils::optimizeTypedExpr(sql, rowType, queryCtx.get(), &execCtx);
    auto evaluator = createCudfExpression(expression, rowType, pool());
    VELOX_ASSERT_USER_THROW(
        evaluator->eval(inputs, stream, mr), "Division by zero");
  }
}

TEST_F(CudfDecimalTest, decimalAvgDecimalInput) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>(
          {100, 200, 300, 400}, // 1.00, 2.00, 3.00, 4.00
          DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS avg_d"})
                  .planNode();

  auto expected = makeRowVector(
      {"avg_d"}, {makeFlatVector<int64_t>({250}, DECIMAL(12, 2))}); // 2.50

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgDecimalInputRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  // Sum = 1.60, count = 7 => 0.22857..., rounds to 0.23 at scale 2.
  std::vector<int64_t> rawValues = {100, 10, 10, 10, 10, 10, 10};
  auto input = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>(rawValues, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS avg_d"})
                  .planNode();

  auto expected = makeRowVector(
      {"avg_d"},
      {makeFlatVector<int64_t>({computeAvgRaw(rawValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgPartialFinalVarbinaryRounds) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  std::vector<int32_t> keys = {1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3};
  std::vector<int64_t> values = {100, 10, 10, 10, 10, 10, 10, 100, 1, -100, -1};

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  std::vector<std::pair<int32_t, std::vector<int64_t>>> groups = {
      {1, {100, 10, 10, 10, 10, 10, 10}},
      {2, {100, 1}},
      {3, {-100, -1}},
  };

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>(
              {computeAvgRaw(groups[0].second),
               computeAvgRaw(groups[1].second),
               computeAvgRaw(groups[2].second)},
              DECIMAL(12, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgIntermediateVarbinaryRounds) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 3}),
          makeFlatVector<int64_t>({100, 10, 100, -100}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 1, 1, 1, 2, 3}),
          makeFlatVector<int64_t>({10, 10, 10, 10, 10, 1, -1}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  std::vector<std::pair<int32_t, std::vector<int64_t>>> groups = {
      {1, {100, 10, 10, 10, 10, 10, 10}},
      {2, {100, 1}},
      {3, {-100, -1}},
  };

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>(
              {computeAvgRaw(groups[0].second),
               computeAvgRaw(groups[1].second),
               computeAvgRaw(groups[2].second)},
              DECIMAL(12, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalPartialFinalVarbinaryRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 10, 10}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .finalAggregation()
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>({computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalIntermediateVarbinaryRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 10, 10}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>({computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalSingleRounds) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>({100, 10, 10, 10, 10, 10, 10}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS a"})
                  .planNode();

  std::vector<int64_t> allValues = {100, 10, 10, 10, 10, 10, 10};
  auto expected = makeRowVector(
      {"a"},
      {makeFlatVector<int64_t>({computeAvgRaw(allValues)}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalSingleDecimal64Overflow) {
  // 12 values of 9e17 (DECIMAL(18,0)) sum to 1.08e19, past 2^63. The sum must
  // accumulate in 128 bits or a DECIMAL64 accumulator wraps; avg is 9e17.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int64_t> values(kNumRows, kBig);

  auto input =
      makeRowVector({"d"}, {makeFlatVector<int64_t>(values, DECIMAL(18, 0))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS a"})
                  .planNode();

  auto expected =
      makeRowVector({"a"}, {makeFlatVector<int64_t>({kBig}, DECIMAL(18, 0))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGroupbySingleDecimal64Overflow) {
  // Same overflow within a single group, exercising the groupby raw sum path.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int32_t> keys(kNumRows, 1);
  std::vector<int64_t> values(kNumRows, kBig);

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values, DECIMAL(18, 0)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"k"}, {"avg(d) AS a"})
                  .planNode();

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1}),
          makeFlatVector<int64_t>({kBig}, DECIMAL(18, 0)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalSingleAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"avg(d) AS a"})
                  .planNode();

  auto expected = makeRowVector(
      {"a"}, {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalPartialFinalVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"a"}, {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgGlobalIntermediateVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"a"}, {makeNullableFlatVector<int64_t>({std::nullopt}, DECIMAL(12, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgPartialFinalVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
          makeNullableFlatVector<int64_t>(
              {100, 200, std::nullopt, std::nullopt, 400, std::nullopt},
              DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {150, std::nullopt, 400}, DECIMAL(12, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalAvgIntermediateVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, 400}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {200, std::nullopt, std::nullopt}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"avg(d) AS a"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "a"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {150, std::nullopt, 400}, DECIMAL(12, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumPartialFinalVarbinary) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 2}),
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  auto task =
      facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
          .assertResults("SELECT k, sum(d) AS s FROM tmp GROUP BY k");
  const auto stats = exec::toPlanStats(task->taskStats());
  EXPECT_GT(
      stats.at(plan->id())
          .customStats.count(std::string{kDirectGroupbyFinalizationStat}),
      0);
}

TEST_F(CudfDecimalTest, decimalGroupbyReleasesRequestTemporaries) {
  rmm::cuda_stream streamOwner;
  const cuda::stream_ref stream = streamOwner;
  rmm::mr::tracking_resource_adaptor tracking{
      rmm::mr::cuda_async_memory_resource{}};
  auto previousResource = cudf::set_current_device_resource(tracking);
  SCOPE_EXIT {
    stream.sync();
    cudf::set_current_device_resource(std::move(previousResource));
  };
  const rmm::device_async_resource_ref mr{tracking};
  const auto decimalType = DECIMAL(12, 2);
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({0}),
       makeFlatVector<int64_t>({100}, decimalType)});

  for (const auto& function : {"sum", "avg"}) {
    SCOPED_TRACE(function);
    auto builder = exec::test::PlanBuilder().values({input}).partialAggregation(
        {"c0"}, {fmt::format("{}(c1)", function)});
    auto partialNode = std::dynamic_pointer_cast<const core::AggregationNode>(
        builder.planNode());
    auto finalNode = std::dynamic_pointer_cast<const core::AggregationNode>(
        builder.finalAggregation().planNode());
    auto partial = toGroupbyAggregators(
        *partialNode,
        partialNode->step(),
        partialNode->outputType(),
        {nullptr},
        {});
    auto final = toGroupbyAggregators(
        *finalNode, finalNode->step(), finalNode->outputType(), {nullptr}, {});

    // Reuse both adapters, but do not let the next request hide a retained
    // temporary by replacing it. Check allocation ownership after each release.
    for (int64_t batch = 1; batch <= 2; ++batch) {
      SCOPED_TRACE(batch);
      {
        auto keys = makeInt64Column({0, 0, 0}, nullptr, stream);
        const std::vector<bool> valid{true, false, true};
        auto values = makeDecimalColumn<int64_t>(
            {100 * batch, 0, 300 * batch}, 2, &valid, stream);
        const cudf::table_view rawInput{{keys->view(), values->view()}};
        cudf::groupby::groupby partialGroupby(cudf::table_view{{keys->view()}});
        std::vector<cudf::groupby::aggregation_request> requests;
        partial[0]->addGroupbyRequest(rawInput, requests, stream, mr);
        auto* castData =
            const_cast<int128_t*>(requests[0].values.data<int128_t>());
        ASSERT_EQ(tracking.get_outstanding_allocations().count(castData), 1);
        auto [partialKeys, partialResults] =
            partialGroupby.aggregate(requests, stream, mr);
        requests.clear();
        partial[0]->releaseInput();
        EXPECT_EQ(tracking.get_outstanding_allocations().count(castData), 0);
        auto state = partial[0]->makeOutputColumn(partialResults, stream, mr);

        // The partial state is a self-describing struct; the final step's
        // request views alias its children, so no decode temporary exists and
        // nothing may be left behind after releaseInput().
        ASSERT_EQ(state->type().id(), cudf::type_id::STRUCT);
        std::set<const void*> stateChildData;
        for (cudf::size_type child = 0; child < state->num_children();
             ++child) {
          stateChildData.insert(state->view().child(child).head());
        }
        const cudf::table_view stateInput{
            {partialKeys->view().column(0), state->view()}};
        cudf::groupby::groupby finalGroupby(partialKeys->view());
        final[0]->addGroupbyRequest(stateInput, requests, stream, mr);
        std::vector<void*> requestData;
        for (const auto& request : requests) {
          requestData.push_back(const_cast<void*>(request.values.head()));
          ASSERT_EQ(
              tracking.get_outstanding_allocations().count(requestData.back()),
              1);
          EXPECT_EQ(stateChildData.count(requestData.back()), 1);
        }
        auto [finalKeys, finalResults] =
            finalGroupby.aggregate(requests, stream, mr);
        requests.clear();
        final[0]->releaseInput();
        for (auto* data : requestData) {
          if (stateChildData.count(data) == 0) {
            EXPECT_EQ(tracking.get_outstanding_allocations().count(data), 0);
          }
        }
        auto result = final[0]->makeOutputColumn(finalResults, stream, mr);
        ASSERT_EQ(result->size(), 1);
        EXPECT_EQ(result->null_count(), 0);
        if (std::string_view(function) == "sum") {
          EXPECT_EQ(
              copyColumnData<int128_t>(result->view(), stream),
              std::vector<int128_t>{400 * batch});
        } else {
          EXPECT_EQ(
              copyColumnData<int64_t>(result->view(), stream),
              std::vector<int64_t>{200 * batch});
        }
      }
      EXPECT_EQ(tracking.get_allocated_bytes(), 0);
    }
  }
}

TEST_F(CudfDecimalTest, streamingDecimalSumReleasesDecodedInput) {
  rmm::cuda_stream streamOwner;
  const cuda::stream_ref stream = streamOwner;
  rmm::mr::tracking_resource_adaptor tracking{
      rmm::mr::cuda_async_memory_resource{}};
  auto previousResource = cudf::set_current_device_resource(tracking);
  SCOPE_EXIT {
    stream.sync();
    cudf::set_current_device_resource(std::move(previousResource));
  };
  const rmm::device_async_resource_ref mr{tracking};
  auto input = makeRowVector(
      {makeFlatVector<int64_t>({0}),
       makeFlatVector<int64_t>({100}, DECIMAL(12, 2))});
  const auto plan = exec::test::PlanBuilder()
                        .values({input})
                        .partialAggregation({"c0"}, {"sum(c1)"})
                        .finalAggregation()
                        .planNode();
  const auto node =
      std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  auto adapters = toStreamingGroupbyAggregators(
      *node,
      node->sources()[0]->outputType(),
      {0, 1},
      node->outputType(),
      {nullptr},
      {});
  ASSERT_TRUE(adapters.has_value());
  auto& adapter = *adapters->at(0);
  std::unique_ptr<cudf::groupby::streaming_groupby> groupby;
  for (int64_t batch = 1; batch <= 2; ++batch) {
    auto keys = makeInt64Column({0}, nullptr, stream);
    auto sums = makeDecimalColumn<int128_t>({100 * batch}, 2, nullptr, stream);
    auto counts = makeInt64Column({1}, nullptr, stream);
    auto state =
        serializeDecimalSumState(sums->view(), counts->view(), stream, mr);
    const cudf::table_view stateInput{{keys->view(), state->view()}};
    std::vector<cudf::column_view> prepared{keys->view()};
    adapter.prepareInput(stateInput, prepared, stream);
    auto* decodedData = const_cast<void*>(prepared.back().head());
    ASSERT_EQ(tracking.get_outstanding_allocations().count(decodedData), 1);
    if (!groupby) {
      std::vector<cudf::groupby::streaming_aggregation_request> requests;
      adapter.addStreamingRequest(requests);
      groupby = std::make_unique<cudf::groupby::streaming_groupby>(
          std::vector<cudf::size_type>{0},
          requests,
          4096,
          cudf::null_policy::INCLUDE,
          mr);
    }
    groupby->aggregate(cudf::table_view{prepared}, stream);
    prepared.clear();
    adapter.releaseInput();
    EXPECT_EQ(tracking.get_outstanding_allocations().count(decodedData), 0);
  }
  const auto bytesBeforeFinalize = tracking.get_allocated_bytes();
  bool allocatedOutput = false;
  auto outputUpstream = mr;
  rmm::mr::callback_memory_resource outputResource{
      [&](std::size_t bytes, cuda::stream_ref outputStream, void*) {
        // Key locations (8 bytes/slot) and hash slots (4 bytes at load factor
        // 0.5) must be released before the first output allocation.
        if (!allocatedOutput) {
          EXPECT_LE(
              tracking.get_allocated_bytes() + 4096 * 16, bytesBeforeFinalize);
          allocatedOutput = true;
        }
        return outputUpstream.allocate(
            outputStream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      },
      [&](void* ptr, std::size_t bytes, cuda::stream_ref outputStream, void*) {
        outputUpstream.deallocate(
            outputStream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      }};
  auto [keys, results] = groupby->finalize_and_release(stream, outputResource);
  EXPECT_TRUE(allocatedOutput);
  EXPECT_EQ(groupby->distinct_keys(), 0);
  EXPECT_THROW(groupby->finalize(stream, mr), cudf::logic_error);
  EXPECT_THROW(groupby->finalize_and_release(stream, mr), cudf::logic_error);
  auto result = adapter.makeOutputColumn(results, stream, mr);
  EXPECT_EQ(
      copyColumnData<int128_t>(result->view(), stream),
      std::vector<int128_t>{300});
}

TEST_F(CudfDecimalTest, decimalSumFinalUsesStreamingGroupby) {
  auto& config = CudfConfig::getInstance();
  const auto savedStreamingGroupbyEnabled = config.streamingGroupbyEnabled;
  const auto savedCapacityMultiplier =
      config.streamingGroupbyCapacityMultiplier;
  const auto savedConcatEnabled = config.concatOptimizationEnabled;
  const auto savedBatchSizeMin = config.batchSizeMinThreshold;
  config.streamingGroupbyEnabled = true;
  config.streamingGroupbyCapacityMultiplier = 2.0;
  config.concatOptimizationEnabled = true;
  config.batchSizeMinThreshold = 1;
  SCOPE_EXIT {
    config.streamingGroupbyEnabled = savedStreamingGroupbyEnabled;
    config.streamingGroupbyCapacityMultiplier = savedCapacityMultiplier;
    config.concatOptimizationEnabled = savedConcatEnabled;
    config.batchSizeMinThreshold = savedBatchSizeMin;
  };

  const auto decimalType = DECIMAL(18, 2);
  std::vector<RowVectorPtr> batches;
  for (int32_t batch = 0; batch < 8; ++batch) {
    batches.push_back(makeRowVector(
        {"k", "d"},
        {makeNullableFlatVector<int32_t>(
             {0, 2 * batch + 1, 2 * batch + 2, std::nullopt}),
         makeNullableFlatVector<int64_t>(
             {100, -200, std::nullopt, 400}, decimalType)}));
  }
  auto builder = exec::test::PlanBuilder()
                     .values(batches, true)
                     .partialAggregation({"k"}, {"sum(d) AS s"});
  const auto plan = builder.finalAggregation().planNode();
  const auto finalAggregationId = plan->id();

  unregisterCudf();
  auto expected =
      exec::test::AssertQueryBuilder(plan).maxDrivers(2).copyResults(pool());
  registerCudf();
  auto task = exec::test::AssertQueryBuilder(plan)
                  .maxDrivers(2)
                  .config(CudfFromVelox::kGpuBatchSizeRows, "4")
                  .config(core::QueryConfig::kMaxPartialAggregationMemory, "1")
                  .assertResults(expected);
  EXPECT_TRUE(hasStreamingGroupbyStat(task, finalAggregationId));
  const auto stats = exec::toPlanStats(task->taskStats());
  EXPECT_GT(
      stats.at(finalAggregationId)
          .customStats.at(std::string{kStreamingGroupbyRebuildsStat})
          .sum,
      0);
}

TEST_F(CudfDecimalTest, decimalPartialSumVarbinaryToVeloxRoundTrip) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .planNode();

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  VELOX_CHECK_NOT_NULL(result);
  ASSERT_GT(result->size(), 0);
  ASSERT_EQ(result->childAt(0)->type()->kind(), TypeKind::VARBINARY);
}

TEST_F(CudfDecimalTest, decimalSumPartialFinalEmptyInput) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>({100, 200, 300}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .filter("k < 0")
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT k, sum(d) AS s FROM tmp WHERE k < 0 GROUP BY k");
}

TEST_F(CudfDecimalTest, decimalSumIntermediateVarbinary) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2}),
          makeFlatVector<int64_t>({12345, -2500, 10000}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({2, 3}),
          makeFlatVector<int64_t>({200, -300}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT k, sum(d) AS s FROM tmp GROUP BY k");
}

TEST_F(CudfDecimalTest, decimalSumGlobalPartialFinalVarbinary) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({12345, -2500, 10000}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({200, -300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSumGlobalIntermediateVarbinary) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({12345, -2500, 10000}, DECIMAL(12, 2))});
  auto input2 = makeRowVector(
      {"d"}, {makeFlatVector<int64_t>({200, -300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input1, input2};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSumGlobalSingle) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeFlatVector<int64_t>(
          {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"sum(d) AS s"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) AS s FROM tmp");
}

// Masked groupby sum: the mask null-injects raw input so cuDF sum excludes
// masked rows. Group 3 is fully masked out -> NULL. Runs on GPU (fallback off).
TEST_F(CudfDecimalTest, decimalSumMaskedGroupbySingle) {
  auto input = makeRowVector(
      {"k", "d", "m"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3}),
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeFlatVector<bool>({true, false, true, true, false}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"k"}, {"sum(d) AS s"}, {"m"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT k, sum(d) FILTER (WHERE m) AS s FROM tmp GROUP BY k");
}

// Masked groupby sum across partial + final steps: the mask applies only at the
// raw partial step and propagates through the serialized intermediate state.
TEST_F(CudfDecimalTest, decimalSumMaskedPartialFinal) {
  auto input = makeRowVector(
      {"k", "d", "m"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3}),
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeFlatVector<bool>({true, false, true, true, false}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"}, {"m"})
                  .finalAggregation()
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT k, sum(d) FILTER (WHERE m) AS s FROM tmp GROUP BY k");
}

// Masked global (reduce) sum, including a NULL mask entry which is excluded.
TEST_F(CudfDecimalTest, decimalSumMaskedGlobalSingle) {
  auto input = makeRowVector(
      {"d", "m"},
      {
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeNullableFlatVector<bool>({true, false, true, std::nullopt, true}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"sum(d) AS s"}, {"m"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) FILTER (WHERE m) AS s FROM tmp");
}

// Masked global (reduce) sum where every row is masked out: the input reduces
// to the empty set, so the result is NULL. Runs on GPU (fallback off).
TEST_F(CudfDecimalTest, decimalSumMaskedGlobalAllMasked) {
  auto input = makeRowVector(
      {"d", "m"},
      {
          makeFlatVector<int64_t>(
              {12345, -2500, 10000, 200, -300}, DECIMAL(12, 2)),
          makeFlatVector<bool>({false, false, false, false, false}),
      });

  std::vector<RowVectorPtr> vectors = {input};
  createDuckDbTable(vectors);

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({}, {"sum(d) AS s"}, {"m"})
                  .planNode();

  facebook::velox::exec::test::AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults("SELECT sum(d) FILTER (WHERE m) AS s FROM tmp");
}

TEST_F(CudfDecimalTest, decimalSumGroupbySingleDecimal64Overflow) {
  // One group of 12 values of 9e17 (DECIMAL(18,0)) sums to 1.08e19, past 2^63.
  // sum(decimal(18,0)) -> decimal(38,0), computed in 128 bits, no wrap.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int32_t> keys(kNumRows, 1);
  std::vector<int64_t> values(kNumRows, kBig);

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values, DECIMAL(18, 0)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  const int128_t expectedSum = static_cast<int128_t>(kBig) * kNumRows;
  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .singleAggregation({"k"}, {"sum(d) AS s"})
                  .planNode();

  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int32_t>({1}),
          makeFlatVector<int128_t>({expectedSum}, DECIMAL(38, 0)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumGlobalPartialFinalDecimal64Overflow) {
  // Global SUM whose total overflows DECIMAL64; exercises the partial raw sum
  // (serialized to VARBINARY) and the final merge, both in 128 bits.
  constexpr int64_t kBig = 900'000'000'000'000'000;
  constexpr int kNumRows = 12;
  std::vector<int64_t> values(kNumRows, kBig);

  auto input =
      makeRowVector({"d"}, {makeFlatVector<int64_t>(values, DECIMAL(18, 0))});

  std::vector<RowVectorPtr> vectors = {input};

  const int128_t expectedSum = static_cast<int128_t>(kBig) * kNumRows;
  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"s"}, {makeFlatVector<int128_t>({expectedSum}, DECIMAL(38, 0))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumPartialFinalVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 1, 2, 2, 3, 3}),
          makeNullableFlatVector<int64_t>(
              {100, 200, std::nullopt, std::nullopt, 400, std::nullopt},
              DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int128_t>(
              {static_cast<int128_t>(300),
               std::nullopt,
               static_cast<int128_t>(400)},
              DECIMAL(38, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumIntermediateVarbinaryNullGroup) {
  auto rowType = ROW({
      {"k", INTEGER()},
      {"d", DECIMAL(12, 2)},
  });

  auto input1 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {100, std::nullopt, 400}, DECIMAL(12, 2)),
      });
  auto input2 = makeRowVector(
      {"k", "d"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int64_t>(
              {200, std::nullopt, std::nullopt}, DECIMAL(12, 2)),
      });

  std::vector<RowVectorPtr> vectors = {input1, input2};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({"k"}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto expected = makeRowVector(
      {"k", "s"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeNullableFlatVector<int128_t>(
              {static_cast<int128_t>(300),
               std::nullopt,
               static_cast<int128_t>(400)},
              DECIMAL(38, 2)),
      });

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumGlobalPartialFinalVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"s"},
      {makeNullableFlatVector<int128_t>({std::nullopt}, DECIMAL(38, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalSumGlobalIntermediateVarbinaryAllNulls) {
  auto rowType = ROW({
      {"d", DECIMAL(12, 2)},
  });

  auto input = makeRowVector(
      {"d"},
      {makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(12, 2))});

  std::vector<RowVectorPtr> vectors = {input};

  auto plan = exec::test::PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(d) AS s"})
                  .intermediateAggregation()
                  .finalAggregation()
                  .planNode();

  auto expected = makeRowVector(
      {"s"},
      {makeNullableFlatVector<int128_t>({std::nullopt}, DECIMAL(38, 2))});

  auto result =
      facebook::velox::exec::test::AssertQueryBuilder(plan).copyResults(pool());
  facebook::velox::test::assertEqualVectors(expected, result);
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateDecimal64) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, -200, 300};
  std::vector<int64_t> counts = {1, 2, 0};
  std::vector<bool> sumValid = {true, false, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 2, stream);
  auto stateMask = copyNullMask(stateCol->view(), stream);
  auto sumMask = copyNullMask(sumAndCount.sum->view(), stream);
  EXPECT_EQ(stateMask, sumMask);

  auto outSum = copyColumnData<__int128_t>(sumAndCount.sum->view(), stream);
  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(sumMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], static_cast<__int128_t>(sums[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateDecimal128) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {2, 1, 0};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, false, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 3, stream);
  auto stateMask = copyNullMask(stateCol->view(), stream);
  auto sumMask = copyNullMask(sumAndCount.sum->view(), stream);
  EXPECT_EQ(stateMask, sumMask);

  auto outSum = copyColumnData<__int128_t>(sumAndCount.sum->view(), stream);
  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(sumMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], sums[i]);
    }
  }
}

// Reproduces the Q18 failure scenario: serializes a decimal sum state with
// partial nulls, round-trips through Arrow (which compacts null rows to 0-byte
// payloads), then deserializes. Without the null-count fix in
// deserializeDecimalSumState, the payload size check would fire because
// chars_size == (numRows - nullCount) * 32, not numRows * 32.
TEST_F(CudfDecimalTest, decimalDeserializeSumStatePartialNullCompact) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // Serialize a 3-row state where row 1 is null (sum-null, count == 0).
  std::vector<int64_t> sums = {100, 0, 300};
  std::vector<int64_t> counts = {1, 0, 2};
  std::vector<bool> sumValid = {true, false, true};
  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  // Round-trip through Arrow (cuDF -> Velox VARBINARY -> cuDF STRING).
  // Arrow stores null rows with 0-byte payloads, so the resulting cuDF STRING
  // column has chars_size == (numRows - nullCount) * 32.
  auto expectedType = ROW({{"s", VARBINARY()}});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{stateCol->view()}},
      pool(),
      expectedType,
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);
  auto compactStateView = compactTable->view().column(0);

  // Verify the column now has the compact layout.
  cudf::strings_column_view strings(compactStateView);
  EXPECT_LT(
      strings.chars_size(stream),
      static_cast<int64_t>(sums.size()) * 32); // 32 == kDecimalSumStateSize

  auto result = deserializeDecimalSumState(compactStateView, 2, stream);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_TRUE(isValidAt(outMask, 0));
  EXPECT_FALSE(isValidAt(outMask, 1));
  EXPECT_TRUE(isValidAt(outMask, 2));
  EXPECT_EQ(outSum[0], static_cast<__int128_t>(100));
  EXPECT_EQ(outCount[0], 1);
  EXPECT_EQ(outSum[2], static_cast<__int128_t>(300));
  EXPECT_EQ(outCount[2], 2);
}

// Reproduces the TPC-DS Q1 failure in CudfGroupbyFINAL: the streaming final
// aggregation concatenates its buffered result -- which came straight from
// serializeDecimalSumState and so keeps a 32-byte payload for null rows -- with
// each newly arrived batch, which was round tripped through velox and so has
// its null rows compacted to 0 bytes. The concatenated state column mixes both
// encodings, so its payload size is neither numRows * 32 nor
// (numRows - nullCount) * 32 and a two-way equality check rejects it.
TEST_F(CudfDecimalTest, decimalDeserializeSumStateMixedNullEncodings) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  auto serialize = [&](const std::vector<int64_t>& sums,
                       const std::vector<int64_t>& counts,
                       const std::vector<bool>& sumValid) {
    auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
    auto countCol = makeInt64Column(counts, nullptr, stream);
    return serializeDecimalSumState(
        sumCol->view(), countCol->view(), stream, mr);
  };

  // Buffered side: row 1 is null and keeps its 32-byte payload.
  auto fullState = serialize({100, 0, 300}, {1, 0, 2}, {true, false, true});

  // Incoming side: same shape, but the velox round trip compacts its null row
  // to a 0-byte payload.
  auto incomingState = serialize({400, 0, 600}, {3, 0, 4}, {true, false, true});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{incomingState->view()}},
      pool(),
      ROW({{"s", VARBINARY()}}),
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);

  auto mixed = cudf::concatenate(
      std::vector<cudf::column_view>{
          fullState->view(), compactTable->view().column(0)},
      stream,
      mr);

  // Four non-null rows, plus the one null payload the buffered side kept: the
  // payload size sits strictly between the compact and full extremes.
  cudf::strings_column_view mixedStrings(mixed->view());
  auto const numRows = static_cast<int64_t>(mixed->size());
  auto const nullCount = static_cast<int64_t>(mixed->null_count());
  EXPECT_EQ(numRows, 6);
  EXPECT_EQ(nullCount, 2);
  EXPECT_GT(
      mixedStrings.chars_size(stream),
      (numRows - nullCount) * 32); // 32 == kDecimalSumStateSize
  EXPECT_LT(mixedStrings.chars_size(stream), numRows * 32);

  auto result = deserializeDecimalSumState(mixed->view(), 2, stream);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_FALSE(isValidAt(outMask, 1));
  EXPECT_FALSE(isValidAt(outMask, 4));
  for (auto row : {0, 2, 3, 5}) {
    EXPECT_TRUE(isValidAt(outMask, row));
  }
  EXPECT_EQ(outSum[0], static_cast<__int128_t>(100));
  EXPECT_EQ(outCount[0], 1);
  EXPECT_EQ(outSum[2], static_cast<__int128_t>(300));
  EXPECT_EQ(outCount[2], 2);
  EXPECT_EQ(outSum[3], static_cast<__int128_t>(400));
  EXPECT_EQ(outCount[3], 3);
  EXPECT_EQ(outSum[5], static_cast<__int128_t>(600));
  EXPECT_EQ(outCount[5], 4);
}

// Trailing null: the offset for the last row equals chars_size, so the kernel
// would read 32 bytes past the buffer end without the null-mask guard.
TEST_F(CudfDecimalTest, decimalDeserializeSumStateTrailingNullCompact) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // 3-row state: [valid, valid, null].
  std::vector<int64_t> sums = {100, 200, 0};
  std::vector<int64_t> counts = {1, 2, 0};
  std::vector<bool> sumValid = {true, true, false};
  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  auto expectedType = ROW({{"s", VARBINARY()}});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{stateCol->view()}},
      pool(),
      expectedType,
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);
  auto compactStateView = compactTable->view().column(0);

  cudf::strings_column_view strings(compactStateView);
  EXPECT_LT(strings.chars_size(stream), static_cast<int64_t>(sums.size()) * 32);

  auto result = deserializeDecimalSumState(compactStateView, 2, stream);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_TRUE(isValidAt(outMask, 0));
  EXPECT_TRUE(isValidAt(outMask, 1));
  EXPECT_FALSE(isValidAt(outMask, 2));
  EXPECT_EQ(outSum[0], static_cast<__int128_t>(100));
  EXPECT_EQ(outCount[0], 1);
  EXPECT_EQ(outSum[1], static_cast<__int128_t>(200));
  EXPECT_EQ(outCount[1], 2);
}

// Leading null: offset 0 overlaps the next valid row's data so the read is
// in-bounds, but verify the null mask propagates correctly.
TEST_F(CudfDecimalTest, decimalDeserializeSumStateLeadingNullCompact) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // 3-row state: [null, valid, valid].
  std::vector<int64_t> sums = {0, 200, 300};
  std::vector<int64_t> counts = {0, 2, 3};
  std::vector<bool> sumValid = {false, true, true};
  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  auto expectedType = ROW({{"s", VARBINARY()}});
  auto veloxRow = with_arrow::toVeloxColumn(
      cudf::table_view{{stateCol->view()}},
      pool(),
      expectedType,
      "s",
      stream,
      mr);
  auto compactTable = with_arrow::toCudfTable(veloxRow, pool(), stream, mr);
  auto compactStateView = compactTable->view().column(0);

  cudf::strings_column_view strings(compactStateView);
  EXPECT_LT(strings.chars_size(stream), static_cast<int64_t>(sums.size()) * 32);

  auto result = deserializeDecimalSumState(compactStateView, 2, stream);

  auto outSum = copyColumnData<__int128_t>(result.sum->view(), stream);
  auto outCount = copyColumnData<int64_t>(result.count->view(), stream);
  auto outMask = copyNullMask(result.sum->view(), stream);

  EXPECT_FALSE(isValidAt(outMask, 0));
  EXPECT_TRUE(isValidAt(outMask, 1));
  EXPECT_TRUE(isValidAt(outMask, 2));
  EXPECT_EQ(outSum[1], static_cast<__int128_t>(200));
  EXPECT_EQ(outCount[1], 2);
  EXPECT_EQ(outSum[2], static_cast<__int128_t>(300));
  EXPECT_EQ(outCount[2], 3);
}

TEST_F(CudfDecimalTest, decimalDeserializeSumStateAllNull) {
  auto stream = cudf::get_default_stream();
  constexpr cudf::size_type numRows = 4;

  auto offsetsCol = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32},
      numRows + 1,
      cudf::mask_state::UNALLOCATED,
      stream);
  auto* offsetsPtr = offsetsCol->mutable_view().data<int32_t>();
  auto status = cudaMemsetAsync(
      offsetsPtr,
      0,
      static_cast<size_t>(numRows + 1) * sizeof(int32_t),
      stream.get());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.sync();

  std::vector<bool> valid(numRows, false);
  auto [nullMask, nullCount] = makeNullMask(valid, stream);
  rmm::device_buffer charsBuf(0, stream);
  auto stateCol = cudf::make_strings_column(
      numRows,
      std::move(offsetsCol),
      std::move(charsBuf),
      nullCount,
      std::move(nullMask));

  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 2, stream);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();

  EXPECT_EQ(outSumView.size(), numRows);
  EXPECT_EQ(outCountView.size(), numRows);
  EXPECT_EQ(outSumView.null_count(), numRows);
  EXPECT_EQ(outCountView.null_count(), numRows);

  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);
  for (size_t i = 0; i < static_cast<size_t>(numRows); ++i) {
    EXPECT_FALSE(isValidAt(outSumMask, i));
    EXPECT_FALSE(isValidAt(outCountMask, i));
  }
}

TEST_F(CudfDecimalTest, decimalSerializeSumStateUsesInt64OffsetsWhenEnabled) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  ScopedEnvVar enableLargeStrings("LIBCUDF_LARGE_STRINGS_ENABLED", "1");
  ScopedEnvVar threshold("LIBCUDF_LARGE_STRINGS_THRESHOLD", "1");

  std::vector<int64_t> sums = {100, -200};
  std::vector<int64_t> counts = {1, 1};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, nullptr, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  cudf::strings_column_view strings(stateCol->view());
  EXPECT_EQ(strings.offsets().type().id(), cudf::type_id::INT64);
}

TEST_F(CudfDecimalTest, decimalSumStateRoundTripUsesInt64Offsets) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  ScopedEnvVar enableLargeStrings("LIBCUDF_LARGE_STRINGS_ENABLED", "1");
  ScopedEnvVar threshold("LIBCUDF_LARGE_STRINGS_THRESHOLD", "1");

  std::vector<int64_t> sums = {100, -200, 300, 400};
  std::vector<int64_t> counts = {1, 0, 2, 3};
  std::vector<bool> sumValid = {true, true, false, true};
  std::vector<bool> countValid = {true, true, true, false};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);

  cudf::strings_column_view strings(stateCol->view());
  EXPECT_EQ(strings.offsets().type().id(), cudf::type_id::INT64);

  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 2, stream);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();
  auto outSum = copyColumnData<__int128_t>(outSumView, stream);
  auto outCount = copyColumnData<int64_t>(outCountView, stream);
  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);

  EXPECT_EQ(outSumView.size(), sums.size());
  EXPECT_EQ(outCountView.size(), counts.size());
  EXPECT_EQ(outSumMask, outCountMask);

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(outSumMask, i), expectedValid);
    EXPECT_EQ(isValidAt(outCountMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], static_cast<__int128_t>(sums[i]));
      EXPECT_EQ(outCount[i], counts[i]);
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, 105, 250, -125};
  std::vector<int64_t> counts = {4, 2, 0, 2};
  std::vector<bool> sumValid = {true, true, true, true};
  std::vector<bool> countValid = {true, false, true, true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol = computeDecimalAverageWithoutOverflow(
      sumCol->view(), countCol->view(), stream, mr);

  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return static_cast<int64_t>(out);
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(avgMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {3, 2, 0};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol = computeDecimalAverageWithoutOverflow(
      sumCol->view(), countCol->view(), stream, mr);

  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return out;
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(avgMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i]));
    }
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64MostNegativeSum) {
  // Negating INT64_MIN in the signed type is overflow UB; the magnitude and
  // sign must be handled in the unsigned domain. avg of one INT64_MIN is
  // itself.
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  std::vector<int64_t> sums = {kMin};
  std::vector<int64_t> counts = {1};
  std::vector<bool> valid = {true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 0, &valid, stream);
  auto countCol = makeInt64Column(counts, &valid, stream);
  auto avgCol = computeDecimalAverageWithoutOverflow(
      sumCol->view(), countCol->view(), stream, mr);

  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);
  EXPECT_EQ(outAvg[0], kMin);
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128MostNegativeSum) {
  // Same regression at the __int128 boundary. avg of one -2^127 is itself.
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  const __int128_t kMin =
      static_cast<__int128_t>(static_cast<unsigned __int128>(1) << 127);
  std::vector<__int128_t> sums = {kMin};
  std::vector<int64_t> counts = {1};
  std::vector<bool> valid = {true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 0, &valid, stream);
  auto countCol = makeInt64Column(counts, &valid, stream);
  auto avgCol = computeDecimalAverageWithoutOverflow(
      sumCol->view(), countCol->view(), stream, mr);

  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);
  EXPECT_EQ(outAvg[0], kMin);
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64AllValid) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, 200, -150};
  std::vector<int64_t> counts = {4, 5, 3};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol = computeDecimalAverageWithoutOverflow(
      sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return static_cast<int64_t>(out);
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128AllValid) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(90000),
      static_cast<__int128_t>(-5000),
      static_cast<__int128_t>(1),
  };
  std::vector<int64_t> counts = {3, 5, 1};
  std::vector<bool> sumValid = {true, true, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto avgCol = computeDecimalAverageWithoutOverflow(
      sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return out;
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal64NonNullableInputs) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {80, -40, 1000};
  std::vector<int64_t> counts = {2, 4, 10};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, nullptr, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  ASSERT_FALSE(sumCol->nullable());
  ASSERT_FALSE(countCol->nullable());

  auto avgCol = computeDecimalAverageWithoutOverflow(
      sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<int64_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return static_cast<int64_t>(out);
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalComputeAverageDecimal128NonNullableInputs) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(600),
      static_cast<__int128_t>(-99),
  };
  std::vector<int64_t> counts = {3, 9};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 4, nullptr, stream);
  auto countCol = makeInt64Column(counts, nullptr, stream);
  ASSERT_FALSE(sumCol->nullable());
  ASSERT_FALSE(countCol->nullable());

  auto avgCol = computeDecimalAverageWithoutOverflow(
      sumCol->view(), countCol->view(), stream, mr);

  EXPECT_EQ(avgCol->view().null_count(), 0);
  auto avgMask = copyNullMask(avgCol->view(), stream);
  auto outAvg = copyColumnData<__int128_t>(avgCol->view(), stream);

  auto avgUnscaled = [](int128_t sum, int64_t count) {
    __int128_t out = 0;
    facebook::velox::DecimalUtil::
        divideWithRoundUp<__int128_t, __int128_t, int64_t>(
            out, sum, count, false, 0, 0);
    return out;
  };

  for (size_t i = 0; i < sums.size(); ++i) {
    if (avgCol->view().nullable()) {
      EXPECT_TRUE(isValidAt(avgMask, i)) << "row " << i;
    }
    EXPECT_EQ(outAvg[i], avgUnscaled(sums[i], counts[i])) << "row " << i;
  }
}

TEST_F(CudfDecimalTest, decimalSumStateRoundTripDecimal64) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<int64_t> sums = {100, -200, 300, 400};
  std::vector<int64_t> counts = {1, 0, 2, 3};
  std::vector<bool> sumValid = {true, true, false, true};
  std::vector<bool> countValid = {true, true, true, false};

  auto sumCol = makeDecimalColumn<int64_t>(sums, 2, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto stateMask = copyNullMask(stateCol->view(), stream);

  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 2, stream);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();
  auto outSum = copyColumnData<__int128_t>(outSumView, stream);
  auto outCount = copyColumnData<int64_t>(outCountView, stream);
  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);

  EXPECT_EQ(stateMask, outSumMask);
  EXPECT_EQ(stateMask, outCountMask);

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(stateMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], static_cast<__int128_t>(sums[i]));
      EXPECT_EQ(outCount[i], counts[i]);
    }
  }
}

TEST_F(CudfDecimalTest, decimalSumStateRoundTripDecimal128) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  std::vector<__int128_t> sums = {
      static_cast<__int128_t>(123450),
      static_cast<__int128_t>(-25000),
      static_cast<__int128_t>(100000),
  };
  std::vector<int64_t> counts = {2, 1, 0};
  std::vector<bool> sumValid = {true, false, true};
  std::vector<bool> countValid = {true, true, true};

  auto sumCol = makeDecimalColumn<__int128_t>(sums, 3, &sumValid, stream);
  auto countCol = makeInt64Column(counts, &countValid, stream);
  auto stateCol =
      serializeDecimalSumState(sumCol->view(), countCol->view(), stream, mr);
  auto stateMask = copyNullMask(stateCol->view(), stream);

  auto sumAndCount = deserializeDecimalSumState(stateCol->view(), 3, stream);
  auto outSumView = sumAndCount.sum->view();
  auto outCountView = sumAndCount.count->view();
  auto outSum = copyColumnData<__int128_t>(outSumView, stream);
  auto outCount = copyColumnData<int64_t>(outCountView, stream);
  auto outSumMask = copyNullMask(outSumView, stream);
  auto outCountMask = copyNullMask(outCountView, stream);

  EXPECT_EQ(stateMask, outSumMask);
  EXPECT_EQ(stateMask, outCountMask);

  for (size_t i = 0; i < sums.size(); ++i) {
    bool expectedValid = sumValid[i] && countValid[i] && counts[i] != 0;
    EXPECT_EQ(isValidAt(stateMask, i), expectedValid);
    if (expectedValid) {
      EXPECT_EQ(outSum[i], sums[i]);
      EXPECT_EQ(outCount[i], counts[i]);
    }
  }
}

TEST_F(CudfDecimalTest, cudfVarbinaryArrowRoundTrip) {
  auto input = makeRowVector(
      {"bin"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")},
          VARBINARY())});

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, pool(), stream, mr);
  auto roundTrip =
      with_arrow::toVeloxColumn(cudfTable->view(), pool(), "rt_", stream, mr);

  ASSERT_EQ(roundTrip->childAt(0)->type()->kind(), TypeKind::VARCHAR);
  VELOX_ASSERT_THROW(
      roundTrip->setType(ROW({{"rt_0", VARBINARY()}})),
      "Cannot change vector type");

  auto expected = makeRowVector(
      {"rt_0"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")}, VARCHAR())});

  facebook::velox::test::assertEqualVectors(expected, roundTrip);
}

TEST_F(CudfDecimalTest, cudfVarbinaryArrowRoundTripWithExpectedType) {
  auto input = makeRowVector(
      {"bin"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")},
          VARBINARY())});

  auto expectedType = ROW({{"bin", VARBINARY()}});

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, pool(), stream, mr);
  auto roundTrip = with_arrow::toVeloxColumn(
      cudfTable->view(), pool(), expectedType, "rt_", stream, mr);

  ASSERT_EQ(roundTrip->childAt(0)->type()->kind(), TypeKind::VARBINARY);

  auto expected = makeRowVector(
      {"rt_0"},
      {makeNullableFlatVector<std::string>(
          {std::string("abc"), std::nullopt, std::string("xyz")},
          VARBINARY())});

  facebook::velox::test::assertEqualVectors(expected, roundTrip);
}

TEST_F(CudfDecimalTest, cudfVarbinaryRowTypeMismatch) {
  auto input = makeRowVector(
      {"l_returnflag",
       "l_linestatus",
       "avg_51",
       "avg_52",
       "avg_53",
       "count_54",
       "sum_47",
       "sum_48",
       "sum_49",
       "sum_50"},
      {makeFlatVector<std::string>({"A", "B"}, VARCHAR()),
       makeFlatVector<std::string>({"F", "O"}, VARCHAR()),
       makeFlatVector<std::string>({"x", "y"}, VARBINARY()),
       makeFlatVector<std::string>({"p", "q"}, VARBINARY()),
       makeFlatVector<std::string>({"m", "n"}, VARBINARY()),
       makeFlatVector<int64_t>({10, 20}, BIGINT()),
       makeFlatVector<std::string>({"u", "v"}, VARBINARY()),
       makeFlatVector<std::string>({"r", "s"}, VARBINARY()),
       makeFlatVector<std::string>({"t", "w"}, VARBINARY()),
       makeFlatVector<std::string>({"c", "d"}, VARBINARY())});

  auto expectedType = ROW({
      {"l_returnflag", VARCHAR()},
      {"l_linestatus", VARCHAR()},
      {"avg_51", VARBINARY()},
      {"avg_52", VARBINARY()},
      {"avg_53", VARBINARY()},
      {"count_54", BIGINT()},
      {"sum_47", VARBINARY()},
      {"sum_48", VARBINARY()},
      {"sum_49", VARBINARY()},
      {"sum_50", VARBINARY()},
  });

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, pool(), stream, mr);
  auto roundTrip =
      with_arrow::toVeloxColumn(cudfTable->view(), pool(), "rt_", stream, mr);

  ASSERT_EQ(roundTrip->childAt(2)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(3)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(4)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(6)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(7)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(8)->type()->kind(), TypeKind::VARCHAR);
  ASSERT_EQ(roundTrip->childAt(9)->type()->kind(), TypeKind::VARCHAR);

  VELOX_ASSERT_THROW(
      roundTrip->setType(expectedType), "Cannot change vector type");
}

// Builds the aggregation node for 'aggregate' over ROW(k BIGINT, d 'type') at
// the given step (grouped by k, or global when 'grouped' is false).
std::shared_ptr<const core::AggregationNode> makeDecimalAggregationNode(
    const std::string& aggregate,
    const TypePtr& type,
    core::AggregationNode::Step step,
    bool grouped,
    memory::MemoryPool* pool) {
  auto keys = BaseVector::create<FlatVector<int64_t>>(BIGINT(), 1, pool);
  auto values = BaseVector::create(type, 1, pool);
  auto input = std::make_shared<RowVector>(
      pool,
      ROW({"k", "d"}, {BIGINT(), type}),
      nullptr,
      1,
      std::vector<VectorPtr>{keys, values});
  const std::vector<std::string> groupingKeys =
      grouped ? std::vector<std::string>{"k"} : std::vector<std::string>{};
  auto builder = exec::test::PlanBuilder().values({input}).partialAggregation(
      groupingKeys, {aggregate});
  if (step == core::AggregationNode::Step::kIntermediate) {
    builder.intermediateAggregation();
  } else if (step == core::AggregationNode::Step::kFinal) {
    builder.finalAggregation();
  } else {
    VELOX_CHECK(step == core::AggregationNode::Step::kPartial);
  }
  auto node = std::dynamic_pointer_cast<const core::AggregationNode>(
      builder.planNode());
  VELOX_CHECK_NOT_NULL(node);
  VELOX_CHECK(node->step() == step);
  return node;
}

std::unique_ptr<GroupbyAggregator> makeDecimalGroupbyAggregator(
    const std::string& aggregate,
    const TypePtr& type,
    core::AggregationNode::Step step,
    memory::MemoryPool* pool) {
  auto node = makeDecimalAggregationNode(aggregate, type, step, true, pool);
  auto aggregators = toGroupbyAggregators(
      *node, node->step(), node->outputType(), {nullptr}, {});
  VELOX_CHECK_EQ(aggregators.size(), 1);
  return std::move(aggregators[0]);
}

struct ReduceAggregatorWithType {
  std::unique_ptr<ReduceAggregator> aggregator;
  TypePtr outputType;
};

ReduceAggregatorWithType makeDecimalReduceAggregator(
    const std::string& aggregate,
    const TypePtr& type,
    core::AggregationNode::Step step,
    memory::MemoryPool* pool) {
  auto node = makeDecimalAggregationNode(aggregate, type, step, false, pool);
  auto aggregators = toReduceAggregators(
      *node, node->step(), node->outputType(), {nullptr}, {});
  VELOX_CHECK_EQ(aggregators.size(), 1);
  return {std::move(aggregators[0]), node->outputType()->childAt(0)};
}

// Runs one group-by aggregator over a [keys, values] table and returns the
// group keys (INT64) with the aggregate output column.
struct GroupbyRun {
  std::unique_ptr<cudf::table> keys;
  std::unique_ptr<cudf::column> output;
};

GroupbyRun runGroupbyAggregator(
    GroupbyAggregator& aggregator,
    cudf::column_view keys,
    cudf::column_view values,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const cudf::table_view input{{keys, values}};
  cudf::groupby::groupby groupby(
      cudf::table_view{{keys}}, cudf::null_policy::INCLUDE);
  std::vector<cudf::groupby::aggregation_request> requests;
  aggregator.addGroupbyRequest(input, requests, stream, mr);
  auto [groupKeys, results] = groupby.aggregate(requests, stream, mr);
  requests.clear();
  aggregator.releaseInput();
  return {
      std::move(groupKeys), aggregator.makeOutputColumn(results, stream, mr)};
}

using DecimalValues = std::vector<std::optional<int128_t>>;
using DecimalGroups = std::map<int64_t, std::optional<int128_t>>;

// Copies a DECIMAL64, DECIMAL128 or INT64 column to the host, widened to
// int128_t; null rows are std::nullopt.
DecimalValues hostDecimalValues(
    const cudf::column_view& values,
    cuda::stream_ref stream) {
  std::vector<int128_t> wide;
  if (values.type().id() == cudf::type_id::DECIMAL128) {
    wide = copyColumnData<int128_t>(values, stream);
  } else {
    for (auto value : copyColumnData<int64_t>(values, stream)) {
      wide.push_back(value);
    }
  }
  const auto mask = copyNullMask(values, stream);
  DecimalValues result;
  for (size_t row = 0; row < wide.size(); ++row) {
    result.push_back(
        isValidAt(mask, row) ? std::make_optional(wide[row]) : std::nullopt);
  }
  return result;
}

// Copies a group-by result to the host as key -> value.
DecimalGroups hostDecimalGroups(
    const cudf::table_view& keys,
    const cudf::column_view& values,
    cuda::stream_ref stream) {
  const auto hostKeys = copyColumnData<int64_t>(keys.column(0), stream);
  const auto hostValues = hostDecimalValues(values, stream);
  DecimalGroups result;
  for (size_t row = 0; row < hostKeys.size(); ++row) {
    result[hostKeys[row]] = hostValues[row];
  }
  return result;
}

struct ShapeCase {
  std::string aggregate;
  TypePtr type;
  DecimalStateShape shape;
};

std::vector<ShapeCase> decimalShapeCases() {
  return {
      {"sum(d)", DECIMAL(18, 2), DecimalStateShape::kSum64},
      {"avg(d)", DECIMAL(18, 2), DecimalStateShape::kAvg64},
      {"sum(d)", DECIMAL(38, 2), DecimalStateShape::kSum128},
      {"avg(d)", DECIMAL(38, 2), DecimalStateShape::kAvg128},
  };
}

// Flattens a state struct of the case's shape (decimalStateShapeOf checks the
// child types) whose sum carries scale 2.
FlatDecimalState expectDecimalStateStruct(
    const cudf::column_view& column,
    const ShapeCase& shapeCase) {
  EXPECT_TRUE(isDecimalStateStruct(column));
  EXPECT_EQ(decimalStateShapeOf(column), shapeCase.shape);
  EXPECT_EQ(decimalStateScaleOf(column), 2);
  return flattenDecimalState(
      column,
      2,
      true,
      true,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());
}

// Raw input column of 'type' for keys {0, 0, 1, 1}: group 0 holds {100, null},
// group 1 is all null.
std::unique_ptr<cudf::column> makeRawDecimalInput(
    const TypePtr& type,
    cuda::stream_ref stream) {
  const std::vector<bool> valid{true, false, false, false};
  if (type->isShortDecimal()) {
    return makeDecimalColumn<int64_t>({100, 0, 0, 0}, 2, &valid, stream);
  }
  return makeDecimalColumn<int128_t>({100, 0, 0, 0}, 2, &valid, stream);
}

// The CPU FINAL over decimal partial states.
struct CpuDecimalFinal {
  // Whether the merged total (sum + overflow * 2^127) fits int128_t, i.e.
  // DecimalUtil::adjustSumForOverflow accepts the merged state.
  bool totalFitsInt128{false};
  // Unset where the CPU SUM raises "Decimal overflow".
  std::optional<int128_t> sum;
  // Unset where the CPU AVG is not a DECIMAL(38) value.
  std::optional<int128_t> avg;
};

// Merges `rows` in order with the CPU accumulator itself
// (LongDecimalWithOverflowState::mergeWith, which carries past +/-2^127 into
// the overflow field with DecimalUtil::addWithOverflow) and finalizes the
// merged state as DecimalSumAggregate and DecimalAverageAggregateBase do.
CpuDecimalFinal cpuDecimalFinal(const std::vector<HostDecimalState>& rows) {
  functions::aggregate::LongDecimalWithOverflowState accumulator;
  std::array<
      char,
      functions::aggregate::LongDecimalWithOverflowState::serializedSize()>
      bytes;
  for (const auto& row : rows) {
    encodeDecimalStateRow(row, bytes.data());
    accumulator.mergeWith(
        StringView(bytes.data(), static_cast<int32_t>(bytes.size())));
  }
  CpuDecimalFinal result;
  const auto sum =
      DecimalUtil::adjustSumForOverflow(accumulator.sum, accumulator.overflow);
  result.totalFitsInt128 = sum.has_value();
  if (sum.has_value() && DecimalUtil::valueInPrecisionRange(*sum, 38)) {
    result.sum = *sum;
  }
  int128_t average{0};
  DecimalUtil::computeAverage(
      average, accumulator.sum, accumulator.count, accumulator.overflow);
  if (DecimalUtil::valueInPrecisionRange(average, 38)) {
    result.avg = average;
  }
  return result;
}

} // namespace

// Every shape: a partial (group-by and global) emits the struct of the plan
// shape, an intermediate re-emits it, and a final over either agrees.
TEST_F(CudfDecimalTest, decimalPartialEmitsStateStruct) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  for (const auto& shapeCase : decimalShapeCases()) {
    SCOPED_TRACE(shapeCase.aggregate + " " + shapeCase.type->toString());
    auto groupbyFor = [&](core::AggregationNode::Step step) {
      return makeDecimalGroupbyAggregator(
          shapeCase.aggregate, shapeCase.type, step, pool());
    };
    auto keys = makeInt64Column({0, 0, 1, 1}, nullptr, stream);
    auto values = makeRawDecimalInput(shapeCase.type, stream);
    auto partial = runGroupbyAggregator(
        *groupbyFor(core::AggregationNode::Step::kPartial),
        keys->view(),
        values->view(),
        stream,
        mr);
    ASSERT_EQ(partial.output->size(), 2);
    const auto flat =
        expectDecimalStateStruct(partial.output->view(), shapeCase);
    const auto groupKeys = partial.keys->view();
    const DecimalGroups expectedSums{{0, 100}, {1, std::nullopt}};
    EXPECT_EQ(hostDecimalGroups(groupKeys, flat.sum, stream), expectedSums);
    if (decimalStateHasCount(shapeCase.shape)) {
      EXPECT_EQ(
          hostDecimalGroups(groupKeys, flat.count, stream),
          (DecimalGroups{{0, 1}, {1, 0}}));
    }
    if (decimalStateHasOverflow(shapeCase.shape)) {
      EXPECT_EQ(
          hostDecimalGroups(groupKeys, flat.overflow, stream),
          (DecimalGroups{{0, 0}, {1, 0}}));
    }

    // The struct feeds an INTERMEDIATE step, which re-emits the shape, and a
    // FINAL step; both the partial and the merged struct finalize alike.
    auto intermediate = runGroupbyAggregator(
        *groupbyFor(core::AggregationNode::Step::kIntermediate),
        groupKeys.column(0),
        partial.output->view(),
        stream,
        mr);
    expectDecimalStateStruct(intermediate.output->view(), shapeCase);
    for (const auto* state : {&partial, &intermediate}) {
      auto final = runGroupbyAggregator(
          *groupbyFor(core::AggregationNode::Step::kFinal),
          state->keys->view().column(0),
          state->output->view(),
          stream,
          mr);
      EXPECT_EQ(
          hostDecimalGroups(final.keys->view(), final.output->view(), stream),
          expectedSums);
    }

    // The global (CudfReduce) partial emits a one-row struct of the shape.
    auto reduce = makeDecimalReduceAggregator(
        shapeCase.aggregate,
        shapeCase.type,
        core::AggregationNode::Step::kPartial,
        pool());
    auto state = reduce.aggregator->doReduce(
        cudf::table_view{{values->view()}},
        reduce.outputType,
        values->size(),
        stream,
        mr);
    ASSERT_EQ(state->size(), 1);
    EXPECT_EQ(
        hostDecimalValues(
            expectDecimalStateStruct(state->view(), shapeCase).sum, stream),
        (DecimalValues{100}));
  }
}

// A blob from a CPU partial feeds the GPU FINAL (group-by, global and direct
// finalization) and INTERMEDIATE steps of every shape.
TEST_F(CudfDecimalTest, decimalFinalAcceptsCpuBlobState) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  // Four state rows from a CPU partial: group 0 has two rows (counts 2 and
  // 1), group 1 one row, group 2 an all-null row.
  const std::vector<bool> valid{true, true, true, false};
  auto keys = makeInt64Column({0, 0, 1, 2}, nullptr, stream);
  auto blob = makeDecimalStateBlob(
      {{2, 0, 100}, {1, 0, 50}, {1, 0, -30}, {0, 0, 0}}, &valid, false, stream);

  for (const auto& shapeCase : decimalShapeCases()) {
    SCOPED_TRACE(shapeCase.aggregate + " " + shapeCase.type->toString());
    const bool isSum = shapeCase.aggregate == "sum(d)";
    auto groupbyFor = [&](core::AggregationNode::Step step) {
      return makeDecimalGroupbyAggregator(
          shapeCase.aggregate, shapeCase.type, step, pool());
    };
    // Group 0: SUM 100 + 50 = 150, AVG 150 / (2 + 1) = 50.
    const DecimalGroups expected{
        {0, isSum ? 150 : 50}, {1, -30}, {2, std::nullopt}};
    auto final = groupbyFor(core::AggregationNode::Step::kFinal);
    auto run =
        runGroupbyAggregator(*final, keys->view(), blob->view(), stream, mr);
    EXPECT_EQ(
        hostDecimalGroups(run.keys->view(), run.output->view(), stream),
        expected);

    // The blob through an INTERMEDIATE step re-emits the plan shape, and a
    // FINAL over that agrees.
    auto intermediate = runGroupbyAggregator(
        *groupbyFor(core::AggregationNode::Step::kIntermediate),
        keys->view(),
        blob->view(),
        stream,
        mr);
    expectDecimalStateStruct(intermediate.output->view(), shapeCase);
    auto rerun = runGroupbyAggregator(
        *final,
        intermediate.keys->view().column(0),
        intermediate.output->view(),
        stream,
        mr);
    EXPECT_EQ(
        hostDecimalGroups(rerun.keys->view(), rerun.output->view(), stream),
        expected);

    // Direct finalization (the incremental final path) accepts the blob.
    ASSERT_TRUE(final->supportsDirectFinalization());
    auto fromBlob = final->finalize(
        std::make_unique<cudf::column>(blob->view(), stream, mr), stream, mr);
    EXPECT_EQ(
        hostDecimalValues(fromBlob->view(), stream),
        (DecimalValues{isSum ? 100 : 50, 50, -30, std::nullopt}));

    // Global (CudfReduce) FINAL over the blob: SUM 100 + 50 - 30 = 120, AVG
    // 120 / (2 + 1 + 1) = 30.
    auto reduce = makeDecimalReduceAggregator(
        shapeCase.aggregate,
        shapeCase.type,
        core::AggregationNode::Step::kFinal,
        pool());
    auto result = reduce.aggregator->doReduce(
        cudf::table_view{{blob->view()}}, reduce.outputType, 4, stream, mr);
    EXPECT_EQ(
        hostDecimalValues(result->view(), stream),
        (DecimalValues{isSum ? 120 : 30}));
  }
}

// A CPU partial state carries the number of 2^127 carries in its overflow
// field. The expected values come from cpuDecimalFinal, which runs the CPU
// accumulator's own merge and finalization in row order. The GPU merges the
// sum children with a wrapping cuDF SUM, so its merged (sum, overflow) pair can
// differ from the CPU's while denoting the same total. GPU SUM equals CPU SUM
// for every total inside int128. GPU AVG equals CPU AVG except that when the
// merged pairs differ (one side carried past 2^127 and the other did not) the
// result may differ by one unit in the last place at an exact-half quotient,
// because the CPU's own result depends on accumulation order there; such a
// case sets `gpuAvg`. Each case runs through the group-by FINAL, the global
// (CudfReduce) FINAL and direct finalization of the state the group-by
// INTERMEDIATE merged, as the incremental FINAL does. Totals beyond int128 are
// a documented limitation (the GPU cannot tell them from their alias modulo
// 2^128; see DecimalAggregationState.h) and are not exercised here.
TEST_F(CudfDecimalTest, decimalFinalFoldsBlobOverflowLikeCpu) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  const int128_t tenPow35 = DecimalUtil::kPowersOfTen[35];
  const int128_t tenPow36 = DecimalUtil::kPowersOfTen[36];
  const int128_t tenPow37 = DecimalUtil::kPowersOfTen[37];
  // The blob sum that, with `overflow` carries of 2^127, represents `value`:
  // value - overflow * 2^127, computed modulo 2^128.
  auto wrappedSum = [](int128_t value, int64_t overflow) {
    return static_cast<int128_t>(
        static_cast<uint128_t>(value) -
        static_cast<uint128_t>(overflow) * (static_cast<uint128_t>(1) << 127));
  };

  struct OverflowCase {
    std::string name;
    std::vector<HostDecimalState> rows;
    // Hand-derived CPU results, checked against cpuDecimalFinal.
    CpuDecimalFinal expected;
    // Set where the GPU AVG is known to differ from the CPU AVG by one ulp.
    std::optional<int128_t> gpuAvg{};
  };
  const std::vector<OverflowCase> cases = {
      // overflow 1 with a negative sum: the value is sum + 2^127 = 10^37.
      {"positiveCarry",
       {{1, 1, wrappedSum(tenPow37, 1)}},
       {true, tenPow37, tenPow37}},
      // overflow -1 with a positive sum: sum - 2^127 = -10^37.
      {"negativeCarry",
       {{1, -1, wrappedSum(-tenPow37, -1)}},
       {true, -tenPow37, -tenPow37}},
      // Merged: count 2, overflow 1, sum 10^37 + 7 - 2^127. computeAverage
      // divides 2^127 and the sum separately; the sum part's remainder -1
      // rounds away from zero on its own, so the CPU AVG is 5 * 10^36 + 3
      // although (10^37 + 7) / 2 = 5 * 10^36 + 3.5.
      {"mergedCarryRounding",
       {{1, 1, wrappedSum(tenPow37, 1)}, {1, 0, 7}},
       {true, tenPow37 + 7, 5 * tenPow36 + 3}},
      // A = 10^37 (count 3, overflow 1), B = -9.5 * 10^37. The CPU merge adds
      // two negative sums whose magnitudes reach 2^127 and carries -1 to
      // (-8.5 * 10^37, overflow 0); the GPU merge wraps the sum to a positive
      // value with overflow 1. Both denote -8.5 * 10^37.
      {"mergeCarriesPastTwoPow127",
       {{3, 1, wrappedSum(tenPow37, 1)}, {1, 0, -95 * tenPow36}},
       {true, -85 * tenPow36, -2125 * (tenPow36 / 100)}},
      // Known one-ulp AVG divergence. Four overflow-free partials with total
      // T = 10^37 + 6 and count 4: the CPU merge carries on 9 * 10^37 +
      // 9 * 10^37 and ends with the canonical pair (T - 2^127, 1), whose
      // split division gives 2.5 * 10^36 + 1; the GPU sum wraps back to
      // (T, 0) and T / 4 rounds half up to 2.5 * 10^36 + 2, which the CPU
      // also returns when merging in an order that never carries.
      {"cpuCarryExactHalfAvg",
       {{1, 0, 9 * tenPow37},
        {1, 0, 9 * tenPow37},
        {1, 0, -9 * tenPow37},
        {1, 0, -8 * tenPow37 + 6}},
       {true, tenPow37 + 6, 25 * tenPow35 + 1},
       25 * tenPow35 + 2},
      // Folds to 15 * 10^37 > 10^38 - 1: out of the DECIMAL(38) range.
      {"foldedOutOfRange",
       {{1, 1, wrappedSum(15 * tenPow37, 1)}},
       {true, std::nullopt, std::nullopt}},
  };

  for (const auto& overflowCase : cases) {
    const auto cpu = cpuDecimalFinal(overflowCase.rows);
    ASSERT_TRUE(cpu.totalFitsInt128) << overflowCase.name;
    ASSERT_EQ(cpu.sum, overflowCase.expected.sum) << overflowCase.name;
    ASSERT_EQ(cpu.avg, overflowCase.expected.avg) << overflowCase.name;

    std::vector<int64_t> zeroKeys(overflowCase.rows.size(), 0);
    auto keys = makeInt64Column(zeroKeys, nullptr, stream);
    auto blob = makeDecimalStateBlob(overflowCase.rows, nullptr, false, stream);
    for (const bool isSum : {true, false}) {
      const auto& expected = isSum
          ? cpu.sum
          : (overflowCase.gpuAvg.has_value() ? overflowCase.gpuAvg : cpu.avg);
      if (!isSum && !expected.has_value()) {
        // Like the CPU, AVG applies no range check.
        continue;
      }
      const std::string aggregate = isSum ? "sum(d)" : "avg(d)";
      SCOPED_TRACE(overflowCase.name + " " + aggregate);
      auto makeGroupby = [&](core::AggregationNode::Step step) {
        return makeDecimalGroupbyAggregator(
            aggregate, DECIMAL(38, 2), step, pool());
      };
      auto final = makeGroupby(core::AggregationNode::Step::kFinal);
      auto reduce = makeDecimalReduceAggregator(
          aggregate,
          DECIMAL(38, 2),
          core::AggregationNode::Step::kFinal,
          pool());
      auto merged = runGroupbyAggregator(
          *makeGroupby(core::AggregationNode::Step::kIntermediate),
          keys->view(),
          blob->view(),
          stream,
          mr);
      ASSERT_EQ(merged.output->size(), 1);
      const std::vector<std::function<std::unique_ptr<cudf::column>()>> runs{
          [&] {
            return std::move(runGroupbyAggregator(
                                 *final, keys->view(), blob->view(), stream, mr)
                                 .output);
          },
          [&] {
            return reduce.aggregator->doReduce(
                cudf::table_view{{blob->view()}},
                reduce.outputType,
                blob->size(),
                stream,
                mr);
          },
          [&] {
            return final->finalize(
                std::make_unique<cudf::column>(
                    merged.output->view(), stream, mr),
                stream,
                mr);
          },
      };
      for (const auto& run : runs) {
        if (expected.has_value()) {
          EXPECT_EQ(
              hostDecimalValues(run()->view(), stream),
              (DecimalValues{*expected}));
        } else {
          VELOX_ASSERT_THROW(run(), "Decimal overflow");
        }
      }
    }
  }
}

// GPU partial states (structs, kSum128) whose total exceeds DECIMAL(38) across
// batches fail in the GPU FINAL as on the CPU: 6 * 10^37 + 6 * 10^37 =
// 1.2 * 10^38 > 10^38 - 1. The total stays below 2^127, so no step wraps.
TEST_F(CudfDecimalTest, decimalStructFinalRangeCheck) {
  const int128_t value = 6 * DecimalUtil::kPowersOfTen[37];
  std::vector<RowVectorPtr> batches;
  for (int batch = 0; batch < 2; ++batch) {
    batches.push_back(makeRowVector(
        {"k", "d"},
        {makeFlatVector<int32_t>({1, 2}),
         makeFlatVector<int128_t>({value, 1}, DECIMAL(38, 0))}));
  }
  for (const bool grouped : {true, false}) {
    SCOPED_TRACE(fmt::format("grouped {}", grouped));
    const std::vector<std::string> keys =
        grouped ? std::vector<std::string>{"k"} : std::vector<std::string>{};
    const auto plan = exec::test::PlanBuilder()
                          .values(batches)
                          .partialAggregation(keys, {"sum(d) AS s"})
                          .finalAggregation()
                          .planNode();
    // Flush the partial after every batch so the FINAL merges two structs.
    const std::unordered_map<std::string, std::string> configs{
        {CudfFromVelox::kGpuBatchSizeRows, "1"},
        {core::QueryConfig::kMaxPartialAggregationMemory, "1"}};
    unregisterCudf();
    VELOX_ASSERT_THROW(
        exec::test::AssertQueryBuilder(plan).configs(configs).copyResults(
            pool()),
        "Decimal overflow");
    registerCudf();
    VELOX_ASSERT_THROW(
        exec::test::AssertQueryBuilder(plan).configs(configs).copyResults(
            pool()),
        "Decimal overflow");
  }
}

// Partial -> local exchange -> [intermediate -> local exchange ->] final, with
// the streaming final group-by on and off, matches the CPU. DECIMAL(18, 0)
// exercises the DECIMAL64 extremes (kSum64 / kAvg64), DECIMAL(38, 2) the
// kSum128 / kAvg128 shapes.
TEST_F(CudfDecimalTest, decimalPartialFinalMatchesCpu) {
  auto& config = CudfConfig::getInstance();
  const auto savedStreamingGroupbyEnabled = config.streamingGroupbyEnabled;
  SCOPE_EXIT {
    config.streamingGroupbyEnabled = savedStreamingGroupbyEnabled;
  };
  const std::unordered_map<std::string, std::string> configs{
      {CudfFromVelox::kGpuBatchSizeRows, "3"},
      {core::QueryConfig::kMaxPartialAggregationMemory, "1"}};
  for (const auto& type : {DECIMAL(18, 0), DECIMAL(38, 2)}) {
    const auto batches = makeGroupedDecimalBatches(type);
    for (const auto& aggregates : std::vector<std::vector<std::string>>{
             {"sum(d) AS s"},
             {"avg(d) AS a"},
             {"sum(d) AS s", "avg(d) AS a"}}) {
      for (const bool withIntermediate : {false, true}) {
        auto builder = exec::test::PlanBuilder()
                           .values(batches, true)
                           .partialAggregation({"k"}, aggregates)
                           .localPartition({"k"});
        if (withIntermediate) {
          builder.intermediateAggregation().localPartition({"k"});
        }
        const auto plan = builder.finalAggregation().planNode();
        for (const bool streaming : {false, true}) {
          SCOPED_TRACE(
              fmt::format(
                  "{} {} intermediate={} streaming={}",
                  type->toString(),
                  folly::join(",", aggregates),
                  withIntermediate,
                  streaming));
          config.streamingGroupbyEnabled = streaming;
          assertCudfMatchesCpu(plan, pool(), configs, 2);
        }
      }
    }
  }
}

TEST_F(CudfDecimalTest, decimalGlobalSumAvgMatchesCpu) {
  for (const auto& type : {DECIMAL(18, 2), DECIMAL(38, 2)}) {
    std::vector<RowVectorPtr> batches;
    for (const auto& batch : makeGroupedDecimalBatches(type)) {
      batches.push_back(makeRowVector({"d"}, {batch->childAt(1)}));
    }
    // An all-null batch set exercises the null state row.
    const std::vector<RowVectorPtr> allNull{makeRowVector(
        {"d"},
        {type->isShortDecimal()
             ? VectorPtr(
                   makeNullableFlatVector<int64_t>(
                       {std::nullopt, std::nullopt}, type))
             : VectorPtr(
                   makeNullableFlatVector<int128_t>(
                       {std::nullopt, std::nullopt}, type))})};
    for (const auto& input : {batches, allNull}) {
      for (const bool withIntermediate : {false, true}) {
        SCOPED_TRACE(
            fmt::format(
                "{} rows={} intermediate={}",
                type->toString(),
                input.size(),
                withIntermediate));
        auto builder =
            exec::test::PlanBuilder().values(input).partialAggregation(
                {}, {"sum(d) AS s", "avg(d) AS a"});
        if (withIntermediate) {
          builder.intermediateAggregation();
        }
        assertCudfMatchesCpu(
            builder.finalAggregation().planNode(),
            pool(),
            {{CudfFromVelox::kGpuBatchSizeRows, "3"}},
            1);
      }
    }
  }
}

} // namespace facebook::velox::cudf_velox::test
