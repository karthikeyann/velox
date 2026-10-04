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

// Unit tests for the self-describing decimal aggregate state helpers declared
// in DecimalAggregationState.h: wrap, flatten, pack and unpack over the four
// struct shapes and the STRING blob.

#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/tests/DecimalStateTestColumns.h"

#include "velox/common/base/tests/GTestUtils.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

int numOwnedColumns(const DecimalStateColumns& owned) {
  return (owned.sum ? 1 : 0) + (owned.count ? 1 : 0) + (owned.overflow ? 1 : 0);
}

using Shape = DecimalStateShape;

constexpr int32_t kScale = 2;
constexpr int64_t kDecimal64Max = 999'999'999'999'999'999;

// Positive, negative, zero and cancelling sums, DECIMAL64 extremes, sums
// beyond 64 bits with a nonzero overflow, and one null row.
DecimalStateRows defaultRows() {
  const int128_t big = (static_cast<int128_t>(1) << 100) + 7;
  DecimalStateRows rows;
  rows.sums = {
      12'345,
      -67'890,
      0,
      500 - 500,
      kDecimal64Max,
      -kDecimal64Max,
      big,
      -big,
      999,
  };
  rows.counts = {1, 2, 3, 2, 1, 1, 4, 5, 1};
  rows.overflows = {0, 0, 0, 0, 0, 0, 1, -1, 0};
  rows.valid = {true, true, true, true, true, true, true, true, false};
  return rows;
}

DecimalStateRows uniformRows(size_t numRows, bool valid) {
  DecimalStateRows rows;
  for (size_t row = 0; row < numRows; ++row) {
    rows.sums.push_back(static_cast<int128_t>(row + 1));
    rows.counts.push_back(0);
    rows.overflows.push_back(0);
    rows.valid.push_back(valid);
  }
  return rows;
}

std::unique_ptr<cudf::column> makeNonStateColumn(
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32},
      3,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
}

// A struct whose children are not one of the four shapes.
std::unique_ptr<cudf::column> makeBadStruct(
    int numChildren,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<std::unique_ptr<cudf::column>> children;
  for (int i = 0; i < numChildren; ++i) {
    children.push_back(cudf::make_empty_column(cudf::type_id::INT64));
  }
  return cudf::make_structs_column(
      0,
      std::move(children),
      0,
      cuda::device_buffer<std::byte>{stream, mr},
      stream,
      mr);
}

// Shared by the shape-parameterized and the shape-independent fixtures.
class DecimalStateFixture {
 protected:
  std::unique_ptr<cudf::column> makeStruct(
      const DecimalStateRows& rows,
      Shape shape) {
    return makeDecimalStateStruct(rows, shape, kScale, stream_, mr_);
  }

  // The blob the CPU would hand over for `rows` (every field as given).
  std::unique_ptr<cudf::column> makeBlob(const DecimalStateRows& rows) {
    return makeDecimalStateBlob(rows, stream_);
  }

  void expectStructShape(const cudf::column_view& view, Shape shape) {
    ASSERT_TRUE(isDecimalStateStruct(view)) << decimalStateShapeLabel(shape);
    EXPECT_EQ(decimalStateShapeOf(view), shape);
    EXPECT_FALSE(view.nullable()) << "struct parent must not carry a mask";
  }

  void expectFlatMatches(
      const FlatDecimalState& flat,
      const DecimalStateRows& expected) {
    EXPECT_EQ(flat.sum.type().scale(), -kScale);
    expectDecimalStateRowsEqual(expected, copyFlatDecimalState(flat, stream_));
  }

  DecimalStateRows readBack(const cudf::column_view& column) {
    return readDecimalState(column, kScale, stream_, mr_);
  }

  cuda::stream_ref stream_{cudf::get_default_stream()};
  rmm::device_async_resource_ref mr_{cudf::get_current_device_resource_ref()};
};

class DecimalStateShapeTest : public ::testing::Test,
                              public DecimalStateFixture {
 protected:
  void SetUp() override {
    if (!initCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }
};

class DecimalStateShapeParamTest : public ::testing::TestWithParam<Shape>,
                                   public DecimalStateFixture {
 protected:
  void SetUp() override {
    if (!initCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }
};

TEST_F(DecimalStateShapeTest, shapeFor) {
  EXPECT_EQ(decimalStateShapeFor(false, false), Shape::kSum64);
  EXPECT_EQ(decimalStateShapeFor(false, true), Shape::kSum128);
  EXPECT_EQ(decimalStateShapeFor(true, false), Shape::kAvg64);
  EXPECT_EQ(decimalStateShapeFor(true, true), Shape::kAvg128);
}

TEST_F(DecimalStateShapeTest, nonStateColumnsAreRejected) {
  auto int32Column = makeNonStateColumn(stream_, mr_);
  EXPECT_FALSE(isDecimalStateStruct(int32Column->view()));
  EXPECT_FALSE(isDecimalStateColumn(int32Column->view()));
  VELOX_ASSERT_THROW(
      decimalStateShapeOf(int32Column->view()),
      "not a decimal aggregate state struct");
  VELOX_ASSERT_THROW(
      packDecimalState(int32Column->view(), stream_, mr_),
      "not a decimal aggregate state struct");
  VELOX_ASSERT_THROW(
      validateIntermediateColumnType(int32Column->view()),
      "Expected decimal aggregation state");

  auto emptyBlob = cudf::make_empty_column(cudf::type_id::STRING);
  EXPECT_FALSE(isDecimalStateStruct(emptyBlob->view()));
  EXPECT_TRUE(isDecimalStateColumn(emptyBlob->view()));
  EXPECT_NO_THROW(validateIntermediateColumnType(emptyBlob->view()));

  // Structs with the arity of a shape but the wrong child types.
  for (int numChildren : {1, 2}) {
    auto badStruct = makeBadStruct(numChildren, stream_, mr_);
    EXPECT_FALSE(isDecimalStateStruct(badStruct->view()));
    EXPECT_FALSE(isDecimalStateColumn(badStruct->view()));
    VELOX_ASSERT_THROW(
        validateIntermediateColumnType(badStruct->view()),
        "Expected decimal aggregation state");
  }
}

TEST_F(DecimalStateShapeTest, wrapCastsDecimal64SumAndInt32Count) {
  DecimalStateColumns flat;
  flat.sum = makeDecimalColumn<int64_t>(
      {kDecimal64Max, -kDecimal64Max, 0}, kScale, nullptr, stream_);
  flat.count = makeFixedWidthColumn<int32_t>(
      cudf::data_type{cudf::type_id::INT32}, {7, 8, 9}, nullptr, stream_);
  auto structColumn =
      wrapDecimalState(std::move(flat), Shape::kAvg64, stream_, mr_);
  expectStructShape(structColumn->view(), Shape::kAvg64);
  DecimalStateRows expected;
  expected.sums = {kDecimal64Max, -kDecimal64Max, 0};
  expected.counts = {7, 8, 9};
  expected.overflows = {0, 0, 0};
  expected.valid = {true, true, true};
  expectDecimalStateRowsEqual(expected, readBack(structColumn->view()));
}

TEST_F(DecimalStateShapeTest, wrapRequiresSumAndCountAndSynthesizesOverflow) {
  const auto rows = defaultRows();
  {
    DecimalStateColumns noSum;
    VELOX_ASSERT_THROW(
        wrapDecimalState(std::move(noSum), Shape::kSum64, stream_, mr_), "sum");
  }
  {
    DecimalStateColumns noCount;
    noCount.sum =
        makeDecimalColumn<int128_t>(rows.sums, kScale, &rows.valid, stream_);
    VELOX_ASSERT_THROW(
        wrapDecimalState(std::move(noCount), Shape::kAvg64, stream_, mr_),
        "count");
  }
  // A missing overflow child is the documented default: zeros.
  for (auto shape : {Shape::kSum128, Shape::kAvg128}) {
    SCOPED_TRACE(decimalStateShapeLabel(shape));
    DecimalStateColumns noOverflow;
    noOverflow.sum =
        makeDecimalColumn<int128_t>(rows.sums, kScale, &rows.valid, stream_);
    if (decimalStateHasCount(shape)) {
      noOverflow.count = makeInt64Column(rows.counts, nullptr, stream_);
    }
    auto structColumn =
        wrapDecimalState(std::move(noOverflow), shape, stream_, mr_);
    expectStructShape(structColumn->view(), shape);
    auto expected = rows.as(shape);
    expected.overflows.assign(rows.size(), 0);
    expectDecimalStateRowsEqual(expected, readBack(structColumn->view()));
  }
}

// flatten owns exactly the columns it decodes or synthesizes.
TEST_F(DecimalStateShapeTest, flattenSynthesizesOnlyOnDemand) {
  const auto rows = defaultRows();
  const auto numRows = static_cast<cudf::size_type>(rows.size());
  auto sumOnly = makeStruct(rows, Shape::kSum64);
  auto blob = makeBlob(rows);
  struct Case {
    cudf::column_view state;
    bool needCount;
    bool needOverflow;
    size_t numOwned;
  };
  for (const auto& [state, needCount, needOverflow, numOwned] :
       std::vector<Case>{
           {sumOnly->view(), false, false, 0},
           {sumOnly->view(), true, false, 1},
           {sumOnly->view(), true, true, 2},
           {blob->view(), false, false, 1},
           {blob->view(), true, true, 3},
       }) {
    SCOPED_TRACE(
        fmt::format(
            "{} needCount {} needOverflow {}",
            cudf::type_to_name(state.type()),
            needCount,
            needOverflow));
    auto flat = flattenDecimalState(
        state, kScale, needCount, needOverflow, stream_, mr_);
    EXPECT_EQ(numOwnedColumns(flat.owned), numOwned);
    EXPECT_EQ(flat.sum.size(), numRows);
    EXPECT_EQ(flat.count.size(), needCount ? numRows : 0);
    EXPECT_EQ(flat.overflow.size(), needOverflow ? numRows : 0);
    EXPECT_EQ(validityOf(flat.sum, stream_), rows.valid);
  }
}

TEST_F(DecimalStateShapeTest, packValidityFollowsSumOrCountRule) {
  DecimalStateRows rows;
  rows.sums = {1, 2, 3};
  rows.counts = {0, 0, 0};
  rows.overflows = {0, 0, 0};
  rows.valid = {true, false, true};

  // No count child: the mask is the sum mask.
  auto sumOnly = makeStruct(rows, Shape::kSum64);
  EXPECT_EQ(
      validityOf(
          packDecimalState(sumOnly->view(), stream_, mr_)->view(), stream_),
      rows.valid);

  // A zero count for a valid sum follows buildStateValidityMask (as
  // serializeDecimalSumState does), so that row is null.
  auto zeroCount = makeStruct(rows, Shape::kAvg64);
  EXPECT_EQ(
      validityOf(
          packDecimalState(zeroCount->view(), stream_, mr_)->view(), stream_),
      (std::vector<bool>{false, false, false}));

  // A null count for a valid sum is null too.
  DecimalStateColumns flat;
  flat.sum =
      makeDecimalColumn<int128_t>(rows.sums, kScale, &rows.valid, stream_);
  const std::vector<bool> countValid{true, true, false};
  flat.count = makeInt64Column({5, 5, 5}, &countValid, stream_);
  auto nullCount =
      wrapDecimalState(std::move(flat), Shape::kAvg64, stream_, mr_);
  EXPECT_EQ(
      validityOf(
          packDecimalState(nullCount->view(), stream_, mr_)->view(), stream_),
      (std::vector<bool>{true, false, false}));
}

// Unpacking a blob to a shape that drops fields loses exactly those fields.
TEST_F(DecimalStateShapeTest, unpackToNarrowerShapeDropsFields) {
  const auto rows = defaultRows();
  auto blob = makeBlob(rows);
  auto narrow =
      unpackDecimalState(blob->view(), Shape::kSum64, kScale, stream_, mr_);
  expectStructShape(narrow->view(), Shape::kSum64);
  expectDecimalStateRowsEqual(rows.as(Shape::kSum64), readBack(narrow->view()));
}

// A blob that went through Velox or Arrow has 0-byte payloads for null rows.
TEST_F(DecimalStateShapeTest, nullRowsSurviveBlobWithCompactedPayload) {
  const std::vector<bool> valid{true, false, true};
  auto blob = makeDecimalStateBlob(
      {{3, 0, 777}, {0, 0, 0}, {1, -2, -888}}, &valid, true, stream_);

  DecimalStateRows expected;
  expected.sums = {777, 0, -888};
  expected.counts = {3, 0, 1};
  expected.overflows = {0, 0, -2};
  expected.valid = valid;
  expectDecimalStateRowsEqual(expected, readBack(blob->view()));

  for (auto shape : kAllDecimalStateShapes) {
    SCOPED_TRACE(decimalStateShapeLabel(shape));
    auto unpacked =
        unpackDecimalState(blob->view(), shape, kScale, stream_, mr_);
    expectStructShape(unpacked->view(), shape);
    // Read the sum through flatten: its child index depends on the shape.
    auto flat = flattenDecimalState(
        unpacked->view(), kScale, false, false, stream_, mr_);
    EXPECT_EQ(flat.sum.null_count(), 1);
    expectDecimalStateRowsEqual(expected.as(shape), readBack(unpacked->view()));
  }
}

TEST_F(DecimalStateShapeTest, zeroRowBlob) {
  auto emptyBlob = cudf::make_empty_column(cudf::type_id::STRING);
  auto flat =
      flattenDecimalState(emptyBlob->view(), kScale, true, true, stream_, mr_);
  EXPECT_EQ(flat.sum.size(), 0);
  EXPECT_EQ(flat.sum.type().id(), cudf::type_id::DECIMAL128);
  EXPECT_EQ(flat.count.type().id(), cudf::type_id::INT64);
  EXPECT_EQ(flat.overflow.type().id(), cudf::type_id::INT64);
}

TEST_P(DecimalStateShapeParamTest, wrapFlattenRoundTrip) {
  const auto shape = GetParam();
  const auto rows = defaultRows();
  auto structColumn = makeStruct(rows, shape);
  const auto view = structColumn->view();
  expectStructShape(view, shape);
  EXPECT_TRUE(isDecimalStateColumn(view));

  auto flat = flattenDecimalState(view, kScale, true, true, stream_, mr_);
  expectFlatMatches(flat, rows.as(shape));

  // Fields the struct carries alias its children (no copy); the others are
  // synthesized into `owned`.
  auto aliasesChild = [&](const cudf::column_view& field) {
    for (int i = 0; i < view.num_children(); ++i) {
      if (view.child(i).head() == field.head()) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(aliasesChild(flat.sum));
  EXPECT_EQ(aliasesChild(flat.count), decimalStateHasCount(shape));
  EXPECT_EQ(aliasesChild(flat.overflow), decimalStateHasOverflow(shape));
  EXPECT_EQ(
      numOwnedColumns(flat.owned),
      (decimalStateHasCount(shape) ? 0 : 1) +
          (decimalStateHasOverflow(shape) ? 0 : 1));
}

// pack writes every field the shape carries and fills in the rest, the blob
// flattens back to the same fields (including overflow), and unpack(pack(x))
// re-packs to the same bytes.
TEST_P(DecimalStateShapeParamTest, packUnpackRoundTrip) {
  const auto shape = GetParam();
  const auto rows = defaultRows();
  const auto expected = rows.as(shape);
  const auto expectedStates = expected.states();
  auto packed = packDecimalState(makeStruct(rows, shape)->view(), stream_, mr_);
  ASSERT_EQ(packed->type().id(), cudf::type_id::STRING);
  const auto packedRows = readDecimalStateBlob(packed->view(), stream_);
  ASSERT_EQ(packedRows.size(), rows.size());
  for (size_t row = 0; row < rows.size(); ++row) {
    ASSERT_EQ(packedRows[row].has_value(), rows.valid[row]) << "row " << row;
    if (rows.valid[row]) {
      EXPECT_TRUE(*packedRows[row] == expectedStates[row]) << "row " << row;
    }
  }
  expectFlatMatches(
      flattenDecimalState(packed->view(), kScale, true, true, stream_, mr_),
      expected);

  auto unpacked =
      unpackDecimalState(packed->view(), shape, kScale, stream_, mr_);
  expectStructShape(unpacked->view(), shape);
  expectDecimalStateRowsEqual(expected, readBack(unpacked->view()));
  EXPECT_EQ(
      readDecimalStateBlob(
          packDecimalState(unpacked->view(), stream_, mr_)->view(), stream_),
      packedRows);
}

// With zero overflow, pack is byte- and mask-equal to serializeDecimalSumState
// for the same sum and the count the shape implies.
TEST_P(DecimalStateShapeParamTest, packMatchesSerialize) {
  const auto shape = GetParam();
  auto rows = defaultRows();
  rows.overflows.assign(rows.size(), 0);
  const auto expected = rows.as(shape);
  auto packed = packDecimalState(makeStruct(rows, shape)->view(), stream_, mr_);
  auto sum =
      makeDecimalColumn<int128_t>(rows.sums, kScale, &rows.valid, stream_);
  auto count = makeInt64Column(expected.counts, nullptr, stream_);
  auto reference =
      serializeDecimalSumState(sum->view(), count->view(), stream_, mr_);
  EXPECT_EQ(
      readDecimalStateBlob(packed->view(), stream_),
      readDecimalStateBlob(reference->view(), stream_));
}

TEST_P(DecimalStateShapeParamTest, allNullRows) {
  const auto shape = GetParam();
  const auto rows = uniformRows(4, false);
  auto packed = packDecimalState(makeStruct(rows, shape)->view(), stream_, mr_);
  EXPECT_EQ(packed->size(), 4);
  EXPECT_EQ(packed->null_count(), 4);

  auto flat =
      flattenDecimalState(packed->view(), kScale, true, true, stream_, mr_);
  EXPECT_EQ(flat.sum.null_count(), 4);
  EXPECT_EQ(flat.count.null_count(), 4);
  EXPECT_EQ(flat.overflow.null_count(), 4);

  auto unpacked =
      unpackDecimalState(packed->view(), shape, kScale, stream_, mr_);
  expectStructShape(unpacked->view(), shape);
  EXPECT_EQ(
      flattenDecimalState(unpacked->view(), kScale, false, false, stream_, mr_)
          .sum.null_count(),
      4);
}

TEST_P(DecimalStateShapeParamTest, zeroRows) {
  const auto shape = GetParam();
  auto structColumn = makeStruct(DecimalStateRows{}, shape);
  EXPECT_EQ(structColumn->size(), 0);
  expectStructShape(structColumn->view(), shape);
  EXPECT_EQ(readBack(structColumn->view()).size(), 0);

  auto packed = packDecimalState(structColumn->view(), stream_, mr_);
  EXPECT_EQ(packed->type().id(), cudf::type_id::STRING);
  EXPECT_EQ(packed->size(), 0);

  auto unpacked =
      unpackDecimalState(packed->view(), shape, kScale, stream_, mr_);
  EXPECT_EQ(unpacked->size(), 0);
  expectStructShape(unpacked->view(), shape);
}

TEST_P(DecimalStateShapeParamTest, validateIntermediateColumnTypeAcceptsForms) {
  auto structColumn = makeStruct(defaultRows(), GetParam());
  EXPECT_NO_THROW(validateIntermediateColumnType(structColumn->view()));
  auto packed = packDecimalState(structColumn->view(), stream_, mr_);
  EXPECT_NO_THROW(validateIntermediateColumnType(packed->view()));
}

INSTANTIATE_TEST_SUITE_P(
    DecimalStateShapes,
    DecimalStateShapeParamTest,
    ::testing::ValuesIn(kAllDecimalStateShapes),
    [](const auto& info) { return decimalStateShapeLabel(info.param); });

} // namespace
} // namespace facebook::velox::cudf_velox
