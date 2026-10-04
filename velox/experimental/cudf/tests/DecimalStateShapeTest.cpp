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
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::test {
namespace {

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

// Rows [begin, all.size()) of `all`.
DecimalStateRows rowsFrom(const DecimalStateRows& all, size_t begin) {
  DecimalStateRows rows;
  for (size_t row = begin; row < all.size(); ++row) {
    rows.sums.push_back(all.sums[row]);
    rows.counts.push_back(all.counts[row]);
    rows.overflows.push_back(all.overflows[row]);
    rows.valid.push_back(all.valid[row]);
  }
  return rows;
}

// Shape-independent tests use this fixture directly; the per-shape tests use
// DecimalStateShapeTest below.
class DecimalStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!initCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }

  std::unique_ptr<cudf::column> makeStruct(
      const DecimalStateRows& rows,
      Shape shape) {
    return makeDecimalStateStruct(rows, shape, kScale, stream_, mr_);
  }

  void expectStructShape(const cudf::column_view& view, Shape shape) {
    ASSERT_TRUE(isDecimalStateStruct(view))
        << DecimalStateShapeName::toName(shape);
    EXPECT_EQ(decimalStateShapeOf(view), shape);
    EXPECT_EQ(decimalStateScaleOf(view), kScale);
    EXPECT_FALSE(view.nullable()) << "struct parent must not carry a mask";
  }

  void expectFlatMatches(
      const FlatDecimalState& flat,
      const DecimalStateRows& expected) {
    EXPECT_EQ(flat.sum.type().scale(), -kScale);
    expectDecimalStateRowsEqual(expected, copyFlatDecimalState(flat, stream_));
  }

  // Reads a state column of either form through flattenDecimalState.
  DecimalStateRows readBack(const cudf::column_view& column) {
    return readDecimalState(column, kScale, stream_, mr_);
  }

  cuda::stream_ref stream_{cudf::get_default_stream()};
  rmm::device_async_resource_ref mr_{cudf::get_current_device_resource_ref()};
};

class DecimalStateShapeTest : public DecimalStateTest,
                              public ::testing::WithParamInterface<Shape> {};

TEST_F(DecimalStateTest, nonStateColumnsAreRejected) {
  auto int32Column =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32}, 3);
  EXPECT_FALSE(isDecimalStateStruct(int32Column->view()));
  EXPECT_FALSE(isDecimalStateColumn(int32Column->view()));
  VELOX_ASSERT_THROW(
      decimalStateShapeOf(int32Column->view()),
      "not a decimal aggregate state struct");
  VELOX_ASSERT_THROW(
      validateIntermediateColumnType(int32Column->view()),
      "Expected decimal aggregation state");
  // A struct with the arity of a shape but the wrong child types.
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(cudf::make_empty_column(cudf::type_id::INT64));
  auto badStruct = cudf::make_structs_column(
      0, std::move(children), 0, cuda::device_buffer<std::byte>{stream_, mr_});
  EXPECT_FALSE(isDecimalStateColumn(badStruct->view()));
  auto emptyBlob = cudf::make_empty_column(cudf::type_id::STRING);
  EXPECT_TRUE(isDecimalStateColumn(emptyBlob->view()));
}

TEST_P(DecimalStateShapeTest, wrapRequiresSumAndCountAndSynthesizesOverflow) {
  const auto shape = GetParam();
  const auto rows = defaultRows();
  DecimalStateColumns noSum;
  VELOX_ASSERT_THROW(
      wrapDecimalState(std::move(noSum), shape, stream_, mr_), "sum");
  // Only the sum: an error for the AVG shapes, and the documented default
  // (overflow 0) for a SUM shape that carries the field.
  DecimalStateColumns sumOnly;
  sumOnly.sum =
      makeDecimalColumn<int128_t>(rows.sums, kScale, &rows.valid, stream_);
  if (decimalStateHasCount(shape)) {
    VELOX_ASSERT_THROW(
        wrapDecimalState(std::move(sumOnly), shape, stream_, mr_), "count");
    return;
  }
  auto structColumn = wrapDecimalState(std::move(sumOnly), shape, stream_, mr_);
  expectStructShape(structColumn->view(), shape);
  auto expected = rows.as(shape);
  expected.overflows.assign(rows.size(), 0);
  expectDecimalStateRowsEqual(expected, readBack(structColumn->view()));
}

TEST_F(DecimalStateTest, packValidityFollowsSumOrCountRule) {
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
}

// A blob that went through Velox or Arrow has 0-byte payloads for null rows.
TEST_F(DecimalStateTest, nullRowsSurviveBlobWithCompactedPayload) {
  const std::vector<bool> valid{true, false, true};
  auto blob = makeDecimalStateBlob(
      {{3, 0, 777}, {0, 0, 0}, {1, -2, -888}}, &valid, true, stream_);
  DecimalStateRows expected;
  expected.sums = {777, 0, -888};
  expected.counts = {3, 0, 1};
  expected.overflows = {0, 0, -2};
  expected.valid = valid;
  expectDecimalStateRowsEqual(expected, readBack(blob->view()));
}

// A blob view with a nonzero offset (cudf::slice) flattens and unpacks to the
// rows of the slice.
TEST_F(DecimalStateTest, slicedBlob) {
  const auto rows = defaultRows();
  auto blob = makeDecimalStateBlob(rows, stream_);
  // Rows [3, 9) include the DECIMAL64 extremes, both 2^100 sums with their
  // overflows, and the null row.
  const cudf::size_type begin = 3;
  const auto end = static_cast<cudf::size_type>(rows.size());
  const auto sliced = cudf::slice(blob->view(), {begin, end})[0];
  ASSERT_EQ(sliced.offset(), begin);
  expectDecimalStateRowsEqual(rowsFrom(rows, begin), readBack(sliced));
}

// A struct view with a nonzero offset flattens and packs to the rows of the
// slice.
TEST_P(DecimalStateShapeTest, slicedStruct) {
  const auto shape = GetParam();
  const auto rows = defaultRows();
  auto structColumn = makeStruct(rows, shape);
  const cudf::size_type begin = 2;
  const auto end = static_cast<cudf::size_type>(rows.size());
  const auto sliced = cudf::slice(structColumn->view(), {begin, end})[0];
  ASSERT_EQ(sliced.offset(), begin);
  const auto expected = rowsFrom(rows, begin).as(shape);
  expectFlatMatches(
      flattenDecimalState(sliced, kScale, true, true, stream_, mr_), expected);
  expectPackedStateEquals(
      packDecimalState(sliced, stream_, mr_)->view(), expected, stream_);
}

// A struct whose parent carries a null mask, as after a nullifying gather: a
// row the parent marks null reads as null in every carried field even where
// the children are valid, through flatten and pack. Built two ways: a view
// with an explicit parent mask over valid children, and cudf::gather with an
// out-of-bounds index under NULLIFY.
TEST_P(DecimalStateShapeTest, parentNullMask) {
  const auto shape = GetParam();
  const auto rows = defaultRows();
  auto structColumn = makeStruct(rows, shape);
  const auto view = structColumn->view();
  const auto numRows = static_cast<cudf::size_type>(rows.size());

  // Parent nulls on valid sums (rows 1 and 6) and on the null-sum row 8.
  std::vector<bool> parentValid(rows.size(), true);
  parentValid[1] = false;
  parentValid[6] = false;
  parentValid[8] = false;
  auto expected = rows.as(shape);
  for (size_t row = 0; row < rows.size(); ++row) {
    expected.valid[row] = expected.valid[row] && parentValid[row];
  }

  auto check = [&](const cudf::column_view& state) {
    ASSERT_EQ(state.null_count(), 3);
    auto flat = flattenDecimalState(state, kScale, true, true, stream_, mr_);
    expectFlatMatches(flat, expected);
    if (decimalStateHasCount(shape)) {
      EXPECT_EQ(validityOf(flat.count, stream_), expected.valid);
    }
    if (decimalStateHasOverflow(shape)) {
      EXPECT_EQ(validityOf(flat.overflow, stream_), expected.valid);
    }
    expectPackedStateEquals(
        packDecimalState(state, stream_, mr_)->view(), expected, stream_);
  };

  {
    SCOPED_TRACE("explicit parent mask");
    auto [parentMask, parentNullCount] = makeNullMask(parentValid, stream_);
    std::vector<cudf::column_view> children;
    for (cudf::size_type i = 0; i < view.num_children(); ++i) {
      children.push_back(view.child(i));
    }
    check(
        cudf::column_view(
            view.type(),
            numRows,
            nullptr,
            reinterpret_cast<const cudf::bitmask_type*>(parentMask.data()),
            parentNullCount,
            0,
            children));
  }
  {
    SCOPED_TRACE("nullifying gather");
    // Gather rows 0..n-1 in order, with an out-of-bounds index in place of
    // the rows the parent mask above nulls.
    std::vector<int32_t> indices(rows.size());
    for (cudf::size_type row = 0; row < numRows; ++row) {
      indices[row] = parentValid[row] ? row : numRows;
    }
    auto gatherMap = makeFixedWidthColumn<int32_t>(
        cudf::data_type{cudf::type_id::INT32}, indices, nullptr, stream_);
    auto gathered = cudf::gather(
        cudf::table_view{{view}},
        gatherMap->view(),
        cudf::out_of_bounds_policy::NULLIFY,
        stream_,
        mr_);
    check(gathered->view().column(0));
  }
}

TEST_P(DecimalStateShapeTest, wrapFlattenRoundTrip) {
  const auto shape = GetParam();
  const auto rows = defaultRows();
  auto structColumn = makeStruct(rows, shape);
  const auto view = structColumn->view();
  expectStructShape(view, shape);
  EXPECT_TRUE(isDecimalStateColumn(view));
  EXPECT_NO_THROW(validateIntermediateColumnType(view));

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
  EXPECT_EQ(flat.owned.sum, nullptr);
  EXPECT_EQ(flat.owned.count != nullptr, !decimalStateHasCount(shape));
  EXPECT_EQ(flat.owned.overflow != nullptr, !decimalStateHasOverflow(shape));
}

// pack writes every field the shape carries and fills in the rest, the blob
// flattens back to the same fields, and unpack rebuilds the same struct.
TEST_P(DecimalStateShapeTest, packRoundTrip) {
  const auto shape = GetParam();
  const auto rows = defaultRows();
  const auto expected = rows.as(shape);
  auto packed = packDecimalState(makeStruct(rows, shape)->view(), stream_, mr_);
  expectPackedStateEquals(packed->view(), expected, stream_);
  EXPECT_NO_THROW(validateIntermediateColumnType(packed->view()));
  expectFlatMatches(
      flattenDecimalState(packed->view(), kScale, true, true, stream_, mr_),
      expected);
  auto unpacked =
      unpackDecimalState(packed->view(), shape, kScale, stream_, mr_);
  expectStructShape(unpacked->view(), shape);
  expectDecimalStateRowsEqual(expected, readBack(unpacked->view()));
}

// With zero overflow, pack is byte- and mask-equal to serializeDecimalSumState
// for the same sum and the count the shape implies.
TEST_P(DecimalStateShapeTest, packMatchesSerialize) {
  const auto shape = GetParam();
  auto rows = defaultRows();
  rows.overflows.assign(rows.size(), 0);
  auto packed = packDecimalState(makeStruct(rows, shape)->view(), stream_, mr_);
  auto sum =
      makeDecimalColumn<int128_t>(rows.sums, kScale, &rows.valid, stream_);
  auto count = makeInt64Column(rows.as(shape).counts, nullptr, stream_);
  auto reference =
      serializeDecimalSumState(sum->view(), count->view(), stream_, mr_);
  EXPECT_EQ(
      readDecimalStateBlob(packed->view(), stream_),
      readDecimalStateBlob(reference->view(), stream_));
}

TEST_P(DecimalStateShapeTest, allNullAndZeroRows) {
  const auto shape = GetParam();
  DecimalStateRows rows;
  rows.sums = {1, 2, 3, 4};
  rows.counts = {0, 0, 0, 0};
  rows.overflows = {0, 0, 0, 0};
  rows.valid = {false, false, false, false};
  auto packed = packDecimalState(makeStruct(rows, shape)->view(), stream_, mr_);
  EXPECT_EQ(packed->null_count(), 4);
  auto flat =
      flattenDecimalState(packed->view(), kScale, true, true, stream_, mr_);
  EXPECT_EQ(flat.sum.null_count(), 4);
  EXPECT_EQ(flat.count.null_count(), 4);
  EXPECT_EQ(flat.overflow.null_count(), 4);

  auto empty = makeStruct(DecimalStateRows{}, shape);
  expectStructShape(empty->view(), shape);
  EXPECT_EQ(readBack(empty->view()).size(), 0);
  EXPECT_EQ(packDecimalState(empty->view(), stream_, mr_)->size(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    DecimalStateShapes,
    DecimalStateShapeTest,
    ::testing::ValuesIn(kAllDecimalStateShapes),
    [](const auto& info) {
      return std::string(DecimalStateShapeName::toName(info.param));
    });

} // namespace
} // namespace facebook::velox::cudf_velox::test
