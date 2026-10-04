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
// state: CudfToVelox packs the decimal state STRUCT a GPU partial aggregation
// emits under a VARBINARY column into the 32-byte blob, and GPU expressions
// refuse to compute over such a column while pass-through operators (identity
// projection, filter on another column, limit) carry it untouched.

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/tests/DecimalStateTestColumns.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <limits>
#include <map>
#include <optional>
#include <string_view>

namespace facebook::velox::cudf_velox::test {
namespace {

using exec::test::AssertQueryBuilder;
using exec::test::PlanBuilder;

constexpr std::string_view kGuardMessage{
    "over an unmaterialized decimal aggregate state column"};

constexpr int128_t kBig = static_cast<int128_t>(1) << 100;

// Raw input per key: two batches with disjoint keys, so every key has exactly
// one partial state row whether or not the partial aggregation flushes
// between batches. Key 2 is all null and yields a null state.
const std::vector<std::pair<int32_t, std::vector<std::optional<int128_t>>>>&
inputGroups() {
  static const std::vector<
      std::pair<int32_t, std::vector<std::optional<int128_t>>>>
      groups = {
          {0, {12'345, -2'500, 7}},
          {1, {-2'500}},
          {2, {std::nullopt, std::nullopt}},
          {3, {kBig, 42, std::nullopt}},
          {4, {-kBig, 5}},
          {5, {0, 9, -9}},
      };
  return groups;
}

// Aggregate producing each state shape over the columns of input().
std::string aggregateFor(DecimalStateShape shape) {
  switch (shape) {
    case DecimalStateShape::kSum64:
      return "sum(d64) AS s";
    case DecimalStateShape::kSum128:
      return "sum(d128) AS s";
    case DecimalStateShape::kAvg64:
      return "avg(d64) AS s";
    case DecimalStateShape::kAvg128:
      return "avg(d128) AS s";
  }
  VELOX_UNREACHABLE();
}

class DecimalStateConversionTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    if (!initCudaDevice()) {
      GTEST_SKIP() << "No usable CUDA device";
    }
    savedCpuFallback_ = CudfConfig::getInstance().allowCpuFallback;
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
  }

  void TearDown() override {
    unregisterCudf();
    CudfConfig::getInstance().allowCpuFallback = savedCpuFallback_;
    exec::test::OperatorTestBase::TearDown();
  }

  // ROW(k INTEGER, d64 DECIMAL(12, 2), d128 DECIMAL(38, 2)) in two batches.
  // The DECIMAL64 column holds the values that fit 64 bits and is null
  // otherwise.
  std::vector<RowVectorPtr> input() {
    std::vector<RowVectorPtr> batches;
    const auto& groups = inputGroups();
    const size_t half = groups.size() / 2;
    for (auto [begin, end] :
         {std::pair<size_t, size_t>{0, half},
          std::pair<size_t, size_t>{half, groups.size()}}) {
      std::vector<int32_t> keys;
      std::vector<std::optional<int64_t>> shortValues;
      std::vector<std::optional<int128_t>> longValues;
      for (size_t group = begin; group < end; ++group) {
        for (const auto& value : groups[group].second) {
          keys.push_back(groups[group].first);
          const bool fits =
              value.has_value() && *value < kBig && *value > -kBig;
          shortValues.push_back(
              fits ? std::make_optional(static_cast<int64_t>(*value))
                   : std::nullopt);
          longValues.push_back(value);
        }
      }
      batches.push_back(makeRowVector(
          {"k", "d64", "d128"},
          {makeFlatVector<int32_t>(keys),
           makeNullableFlatVector<int64_t>(shortValues, DECIMAL(12, 2)),
           makeNullableFlatVector<int128_t>(longValues, DECIMAL(38, 2))}));
    }
    return batches;
  }

  // GPU partial aggregation (k, s VARBINARY): s is physically a decimal state
  // STRUCT of `shape` until CudfToVelox packs it.
  PlanBuilder statePartial(DecimalStateShape shape) {
    return PlanBuilder().values(input()).partialAggregation(
        {"k"}, {aggregateFor(shape)});
  }

  // Each input batch flushes its own partial output, so CudfToVelox sees
  // several GPU batches (the device concat in non-passthrough mode).
  RowVectorPtr run(const core::PlanNodePtr& plan, bool passthrough) {
    return AssertQueryBuilder(plan)
        .config(CudfToVelox::kPassthroughMode, passthrough ? "true" : "false")
        .config(CudfFromVelox::kGpuBatchSizeRows, "1")
        .config(core::QueryConfig::kMaxPartialAggregationMemory, "1")
        .copyResults(pool());
  }

  // Checks that column `stateName` of `result` holds, per key, the blob of the
  // shape's partial state: the sum of the key's non-null values, the count
  // (1 for SUM shapes), overflow 0, and null for an all-null key.
  void verifyStates(
      const RowVectorPtr& result,
      DecimalStateShape shape,
      const std::string& stateName = "s") {
    const bool shortInput = shape == DecimalStateShape::kSum64 ||
        shape == DecimalStateShape::kAvg64;
    std::map<int32_t, HostDecimalState> expected;
    for (const auto& [key, values] : inputGroups()) {
      HostDecimalState state{0, 0, 0};
      for (const auto& value : values) {
        if (value.has_value() &&
            (!shortInput || (*value < kBig && *value > -kBig))) {
          state.sum += *value;
          ++state.count;
        }
      }
      expected[key] = state;
    }
    const auto& rowType = result->type()->asRow();
    auto keys =
        result->childAt(rowType.getChildIdx("k"))->asFlatVector<int32_t>();
    auto states = result->childAt(rowType.getChildIdx(stateName))
                      ->asFlatVector<StringView>();
    ASSERT_NE(keys, nullptr);
    ASSERT_NE(states, nullptr);
    ASSERT_EQ(states->type()->kind(), TypeKind::VARBINARY);
    for (vector_size_t row = 0; row < result->size(); ++row) {
      const auto key = keys->valueAt(row);
      SCOPED_TRACE(
          fmt::format(
              "shape {} key {}", DecimalStateShapeName::toName(shape), key));
      const auto& state = expected.at(key);
      if (state.count == 0) {
        EXPECT_TRUE(states->isNullAt(row));
        continue;
      }
      ASSERT_FALSE(states->isNullAt(row));
      ASSERT_EQ(states->valueAt(row).size(), detail::kDecimalSumStateSize);
      const auto decoded = decodeDecimalStateRow(states->valueAt(row).data());
      EXPECT_TRUE(decoded.sum == state.sum);
      EXPECT_EQ(
          decoded.count, decimalStateFields(shape).hasCount ? state.count : 1);
      EXPECT_EQ(decoded.overflow, 0);
    }
  }

  bool savedCpuFallback_{false};
};

class DecimalStateConversionShapeTest
    : public DecimalStateConversionTest,
      public ::testing::WithParamInterface<DecimalStateShape> {};

// A GPU partial state of every shape converts through CudfToVelox into
// 32-byte blobs, on both the passthrough path and the device-concat path.
TEST_P(DecimalStateConversionShapeTest, structStateToVelox) {
  const auto shape = GetParam();
  for (const bool passthrough : {true, false}) {
    SCOPED_TRACE(fmt::format("passthrough {}", passthrough));
    auto result = run(statePartial(shape).planNode(), passthrough);
    ASSERT_EQ(result->size(), inputGroups().size());
    verifyStates(result, shape);
  }
}

INSTANTIATE_TEST_SUITE_P(
    DecimalStateShapes,
    DecimalStateConversionShapeTest,
    ::testing::ValuesIn(kAllDecimalStateShapes),
    [](const auto& info) {
      return std::string(DecimalStateShapeName::toName(info.param));
    });

// Same as structStateToVelox, with libcudf forced to give the packed blob
// column INT64 (large string) offsets. cuDF then exports it to Arrow as
// large_utf8 ("U") and the import side has to keep the 64-bit offset width
// when it rewrites the format to VARBINARY. Reproduces, at a small size, what
// TPC-H Q17 at SF1000 hits when the partial state column exceeds 2^31 bytes.
TEST_P(DecimalStateConversionShapeTest, structStateToVeloxWithLargeOffsets) {
  ScopedEnvVar enableLargeStrings("LIBCUDF_LARGE_STRINGS_ENABLED", "1");
  ScopedEnvVar threshold("LIBCUDF_LARGE_STRINGS_THRESHOLD", "1");
  const auto shape = GetParam();
  for (const bool passthrough : {true, false}) {
    SCOPED_TRACE(fmt::format("passthrough {}", passthrough));
    auto result = run(statePartial(shape).planNode(), passthrough);
    ASSERT_EQ(result->size(), inputGroups().size());
    verifyStates(result, shape);
  }
}

// A kAvg64 state whose packed chars exceed 2^31 bytes converts through
// packDecimalState and with_arrow::toVeloxColumn (the exportToVelox call
// pattern) into 70 million 32-byte blobs that decode back to their fields.
// This is the real large-offsets path with libcudf's default threshold.
// NOTE: allocates several GB of device memory (about 4.5 GB: 70M x 16 B sums,
// 70M x 8 B counts, 70M x 8 B offsets and 2.24 GB of chars) and about 4 GB of
// host memory for the Arrow buffers and the StringView vector.
TEST_F(DecimalStateConversionTest, packedStateBeyondInt32CharsToVelox) {
  constexpr cudf::size_type kRows = 70'000'000;
  static_assert(
      static_cast<int64_t>(kRows) * detail::kDecimalSumStateSize >
      std::numeric_limits<int32_t>::max());
  if (!cudf::strings::is_large_strings_enabled()) {
    GTEST_SKIP() << "libcudf large strings are disabled";
  }
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();

  // sum(i) = 3 * i - 1'000'000 (DECIMAL128 with scale 0, so the unscaled
  // value the blob carries equals the integer), count(i) = i + 1.
  constexpr int64_t kSumBase = -1'000'000;
  constexpr int64_t kSumStep = 3;
  auto sums = cudf::cast(
      cudf::sequence(
          kRows,
          cudf::numeric_scalar<int64_t>(kSumBase, true, stream),
          cudf::numeric_scalar<int64_t>(kSumStep, true, stream),
          stream,
          mr)
          ->view(),
      cudf::data_type{cudf::type_id::DECIMAL128, 0},
      stream,
      mr);
  auto counts = cudf::sequence(
      kRows,
      cudf::numeric_scalar<int64_t>(1, true, stream),
      cudf::numeric_scalar<int64_t>(1, true, stream),
      stream,
      mr);
  DecimalStateColumns flat;
  flat.sum = std::move(sums);
  flat.count = std::move(counts);
  auto state =
      wrapDecimalState(std::move(flat), DecimalStateShape::kAvg64, stream, mr);

  auto packed = packDecimalState(state->view(), stream, mr);
  state.reset();
  ASSERT_EQ(packed->type().id(), cudf::type_id::STRING);
  cudf::strings_column_view strings(packed->view());
  ASSERT_EQ(strings.offsets().type().id(), cudf::type_id::INT64);
  ASSERT_EQ(
      strings.chars_size(stream),
      static_cast<int64_t>(kRows) * detail::kDecimalSumStateSize);

  auto result = with_arrow::toVeloxColumn(
      cudf::table_view({packed->view()}),
      pool(),
      ROW({"s"}, {VARBINARY()}),
      "",
      stream,
      mr);
  stream.sync();
  packed.reset();

  ASSERT_EQ(result->size(), kRows);
  auto states = result->childAt(0)->asFlatVector<StringView>();
  ASSERT_NE(states, nullptr);
  ASSERT_EQ(states->type()->kind(), TypeKind::VARBINARY);
  ASSERT_EQ(states->getNullCount().value_or(0), 0);
  for (vector_size_t row = 0; row < kRows; ++row) {
    const auto value = states->valueAt(row);
    ASSERT_EQ(value.size(), detail::kDecimalSumStateSize) << "row " << row;
    const auto decoded = decodeDecimalStateRow(value.data());
    ASSERT_TRUE(
        decoded.sum ==
        static_cast<int128_t>(kSumBase + kSumStep * static_cast<int64_t>(row)))
        << "row " << row;
    ASSERT_EQ(decoded.count, static_cast<int64_t>(row) + 1) << "row " << row;
    ASSERT_EQ(decoded.overflow, 0) << "row " << row;
  }
}

// Pass-through operators carry the struct untouched: an identity projection
// that reorders and renames, a filter and a computed projection on another
// column, and a limit.
TEST_F(DecimalStateConversionTest, passThroughOperatorsDoNotTriggerGuard) {
  const auto shape = DecimalStateShape::kAvg128;
  {
    SCOPED_TRACE("identity reorder and rename");
    auto plan = statePartial(shape).project({"s AS state", "k"}).planNode();
    auto result = run(plan, false);
    ASSERT_EQ(result->size(), inputGroups().size());
    verifyStates(result, shape, "state");
  }
  {
    SCOPED_TRACE("filter and computed projection on another column");
    auto plan = statePartial(shape)
                    .filter("k % 2 = 0")
                    .project({"k", "s", "k + 1 AS k1"})
                    .planNode();
    auto result = run(plan, true);
    ASSERT_EQ(result->size(), 3);
    verifyStates(result, shape);
  }
  {
    SCOPED_TRACE("limit");
    auto plan = statePartial(shape).limit(0, 2, false).planNode();
    auto result = run(plan, true);
    ASSERT_EQ(result->size(), 2);
    verifyStates(result, shape);
  }
}

// A computed expression over the struct-typed VARBINARY column fails with a
// clear message, in a projection and in a filter.
TEST_F(DecimalStateConversionTest, computedExpressionOverStateFails) {
  const auto shape = DecimalStateShape::kSum64;
  VELOX_ASSERT_THROW(
      run(statePartial(shape).project({"k", "s IS NULL AS n"}).planNode(),
          true),
      kGuardMessage);
  VELOX_ASSERT_THROW(
      run(statePartial(shape).filter("s IS NULL").planNode(), true),
      kGuardMessage);
}

// GPU partial aggregation feeding a CPU final aggregation (the HTTP exchange
// and CPU-fallback shape) returns the same results as pure CPU for SUM and AVG
// over short and long decimals, grouped and global, including all-null groups
// and a null key.
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
      auto intermediate =
          run(PlanBuilder()
                  .values(input)
                  .partialAggregation(keys, partialAggs)
                  .planNode(),
              passthrough);
      for (const auto& name : {"s64", "a64", "s128", "a128"}) {
        ASSERT_EQ(
            intermediate->childAt(name)->type()->kind(), TypeKind::VARBINARY);
      }

      // CPU final aggregation over the GPU-produced states versus a pure CPU
      // single aggregation.
      unregisterCudf();
      auto actual = AssertQueryBuilder(
                        PlanBuilder()
                            .values({intermediate})
                            .finalAggregation(keys, finalAggs, rawInputTypes)
                            .planNode())
                        .copyResults(pool());
      auto expected = AssertQueryBuilder(
                          PlanBuilder()
                              .values(input)
                              .singleAggregation(keys, partialAggs)
                              .planNode())
                          .copyResults(pool());
      ASSERT_EQ(actual->size(), expected->size());
      EXPECT_TRUE(exec::test::assertEqualResults({expected}, {actual}));
    }
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::test
