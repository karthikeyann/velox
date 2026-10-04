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

// Concat-funnel normalization of self-describing decimal aggregate state
// (design doc section 6.3): a logical VARBINARY column may be physically a
// STRING blob or a STRUCT, and getConcatenatedTable,
// getConcatenatedTableBatched and getConcatenatedCudfVectorsBatched must bring
// every batch to one form before cudf::concatenate.

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <optional>
#include <type_traits>

namespace facebook::velox::cudf_velox {
namespace {

constexpr int32_t kScale = 2;
constexpr int kBitsPerWord = 8 * sizeof(cudf::bitmask_type);

rmm::device_async_resource_ref testMr() {
  return cudf::get_current_device_resource_ref();
}

// ---------------------------------------------------------------------------
// Column helpers (adapted from DecimalAggregationTest.cpp).
// ---------------------------------------------------------------------------

std::pair<cuda::device_buffer<std::byte>, cudf::size_type> makeNullMask(
    const std::vector<bool>& valid,
    cuda::stream_ref stream) {
  auto numBits = static_cast<cudf::size_type>(valid.size());
  if (numBits == 0) {
    return {cuda::device_buffer<std::byte>{stream, testMr()}, 0};
  }
  auto numWords =
      cudf::bitmask_allocation_size_bytes(numBits) / sizeof(cudf::bitmask_type);
  std::vector<cudf::bitmask_type> host(numWords, 0);
  cudf::size_type nullCount = 0;
  for (cudf::size_type i = 0; i < numBits; ++i) {
    if (valid[i]) {
      host[i / kBitsPerWord] |= (cudf::bitmask_type{1} << (i % kBitsPerWord));
    } else {
      ++nullCount;
    }
  }
  auto mask =
      cudf::create_null_mask(numBits, cudf::mask_state::UNINITIALIZED, stream);
  VELOX_CHECK_EQ(
      0,
      static_cast<int>(cudaMemcpyAsync(
          mask.data(),
          host.data(),
          host.size() * sizeof(cudf::bitmask_type),
          cudaMemcpyHostToDevice,
          stream.get())));
  stream.sync();
  return {std::move(mask), nullCount};
}

template <typename T>
std::unique_ptr<cudf::column> makeFixedWidthColumn(
    cudf::data_type type,
    const std::vector<T>& values,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  auto col = cudf::make_fixed_width_column(
      type,
      static_cast<cudf::size_type>(values.size()),
      cudf::mask_state::UNALLOCATED,
      stream);
  if (!values.empty()) {
    VELOX_CHECK_EQ(
        0,
        static_cast<int>(cudaMemcpyAsync(
            col->mutable_view().data<T>(),
            values.data(),
            values.size() * sizeof(T),
            cudaMemcpyHostToDevice,
            stream.get())));
    stream.sync();
  }
  if (valid) {
    auto [mask, nullCount] = makeNullMask(*valid, stream);
    col->set_null_mask(std::move(mask), nullCount);
  }
  return col;
}

std::unique_ptr<cudf::column> makeDecimal128Column(
    const std::vector<int128_t>& values,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  return makeFixedWidthColumn(
      cudf::data_type{cudf::type_id::DECIMAL128, -kScale},
      values,
      valid,
      stream);
}

std::unique_ptr<cudf::column> makeInt64Column(
    const std::vector<int64_t>& values,
    cuda::stream_ref stream) {
  return makeFixedWidthColumn(
      cudf::data_type{cudf::type_id::INT64}, values, nullptr, stream);
}

template <typename T>
std::vector<T> copyColumnData(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  std::vector<T> host(view.size());
  if (view.size() == 0) {
    return host;
  }
  VELOX_CHECK_EQ(
      0,
      static_cast<int>(cudaMemcpyAsync(
          host.data(),
          view.data<T>(),
          view.size() * sizeof(T),
          cudaMemcpyDeviceToHost,
          stream.get())));
  stream.sync();
  return host;
}

std::vector<bool> copyValidity(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  std::vector<bool> valid(view.size(), true);
  if (!view.nullable() || view.size() == 0) {
    return valid;
  }
  std::vector<cudf::bitmask_type> host(cudf::num_bitmask_words(view.size()));
  VELOX_CHECK_EQ(view.offset(), 0);
  VELOX_CHECK_EQ(
      0,
      static_cast<int>(cudaMemcpyAsync(
          host.data(),
          view.null_mask(),
          host.size() * sizeof(cudf::bitmask_type),
          cudaMemcpyDeviceToHost,
          stream.get())));
  stream.sync();
  for (cudf::size_type i = 0; i < view.size(); ++i) {
    valid[i] = (host[i / kBitsPerWord] >> (i % kBitsPerWord)) & 1;
  }
  return valid;
}

// ---------------------------------------------------------------------------
// Decimal state rows: the logical content of a state column, independent of
// its physical form.
// ---------------------------------------------------------------------------

struct StateRows {
  std::vector<int128_t> sums;
  std::vector<int64_t> counts;
  std::vector<int64_t> overflows;
  std::vector<bool> valid;

  size_t size() const {
    return sums.size();
  }

  void append(const StateRows& other) {
    sums.insert(sums.end(), other.sums.begin(), other.sums.end());
    counts.insert(counts.end(), other.counts.begin(), other.counts.end());
    overflows.insert(
        overflows.end(), other.overflows.begin(), other.overflows.end());
    valid.insert(valid.end(), other.valid.begin(), other.valid.end());
  }

  // The rows as a column of `shape` would report them: fields the shape
  // lacks read back as count 1 and overflow 0.
  StateRows as(DecimalStateShape shape) const {
    StateRows out = *this;
    for (size_t i = 0; i < size(); ++i) {
      if (!decimalStateHasCount(shape)) {
        out.counts[i] = 1;
      }
      if (!decimalStateHasOverflow(shape)) {
        out.overflows[i] = 0;
      }
    }
    return out;
  }
};

// Deterministic rows; every fourth row (from `nullPhase`) is null.
StateRows makeRows(int64_t base, size_t n, size_t nullPhase = 3) {
  StateRows rows;
  for (size_t i = 0; i < n; ++i) {
    const auto v = base + static_cast<int64_t>(i);
    rows.sums.push_back(static_cast<int128_t>(v) * 1'000'003 - 77);
    rows.counts.push_back(2 + (v % 5));
    rows.overflows.push_back(v % 3);
    rows.valid.push_back((i % 4) != nullPhase);
  }
  return rows;
}

// Blob of `rows`. serializeDecimalSumState encodes sum and count only, so
// the blob's overflow is 0 whatever `rows.overflows` says.
std::unique_ptr<cudf::column> makeBlobColumn(
    const StateRows& rows,
    cuda::stream_ref stream) {
  auto sum = makeDecimal128Column(rows.sums, &rows.valid, stream);
  auto count = makeInt64Column(rows.counts, stream);
  auto blob =
      serializeDecimalSumState(sum->view(), count->view(), stream, testMr());
  stream.sync();
  return blob;
}

std::unique_ptr<cudf::column> makeStructColumn(
    const StateRows& rows,
    DecimalStateShape shape,
    cuda::stream_ref stream) {
  std::vector<std::unique_ptr<cudf::column>> children;
  auto sum = makeDecimal128Column(rows.sums, &rows.valid, stream);
  switch (shape) {
    case DecimalStateShape::kSum64:
      children.push_back(std::move(sum));
      break;
    case DecimalStateShape::kSum128:
      children.push_back(makeInt64Column(rows.overflows, stream));
      children.push_back(std::move(sum));
      break;
    case DecimalStateShape::kAvg64:
      children.push_back(std::move(sum));
      children.push_back(makeInt64Column(rows.counts, stream));
      break;
    case DecimalStateShape::kAvg128:
      children.push_back(std::move(sum));
      children.push_back(makeInt64Column(rows.counts, stream));
      children.push_back(makeInt64Column(rows.overflows, stream));
      break;
  }
  const auto size = static_cast<cudf::size_type>(rows.size());
  // No parent null mask: validity lives on the sum child.
  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::STRUCT},
      size,
      rmm::device_buffer{},
      cuda::device_buffer<std::byte>{stream, testMr()},
      0,
      std::move(children));
}

// Reads any physical form back into StateRows.
StateRows readState(const cudf::column_view& column, cuda::stream_ref stream) {
  StateRows rows;
  if (column.type().id() == cudf::type_id::STRING) {
    auto flat = deserializeDecimalSumState(column, kScale, stream);
    rows.sums = copyColumnData<int128_t>(flat.sum->view(), stream);
    rows.counts = copyColumnData<int64_t>(flat.count->view(), stream);
    rows.overflows.assign(rows.sums.size(), 0);
    rows.valid = copyValidity(flat.sum->view(), stream);
    return rows;
  }
  VELOX_CHECK(isDecimalStateStruct(column));
  const auto shape = decimalStateShapeOf(column);
  int sumIdx = 0;
  int countIdx = -1;
  int overflowIdx = -1;
  switch (shape) {
    case DecimalStateShape::kSum64:
      break;
    case DecimalStateShape::kSum128:
      overflowIdx = 0;
      sumIdx = 1;
      break;
    case DecimalStateShape::kAvg64:
      countIdx = 1;
      break;
    case DecimalStateShape::kAvg128:
      countIdx = 1;
      overflowIdx = 2;
      break;
  }
  const auto sum = column.child(sumIdx);
  EXPECT_EQ(sum.type().scale(), -kScale);
  rows.sums = copyColumnData<int128_t>(sum, stream);
  rows.valid = copyValidity(sum, stream);
  rows.counts = countIdx >= 0
      ? copyColumnData<int64_t>(column.child(countIdx), stream)
      : std::vector<int64_t>(rows.sums.size(), 1);
  rows.overflows = overflowIdx >= 0
      ? copyColumnData<int64_t>(column.child(overflowIdx), stream)
      : std::vector<int64_t>(rows.sums.size(), 0);
  return rows;
}

void expectRowsEqual(const StateRows& expected, const StateRows& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected.valid[i], actual.valid[i]) << "row " << i;
    if (!expected.valid[i]) {
      continue;
    }
    EXPECT_TRUE(expected.sums[i] == actual.sums[i]) << "sum at row " << i;
    EXPECT_EQ(expected.counts[i], actual.counts[i]) << "count at row " << i;
    EXPECT_EQ(expected.overflows[i], actual.overflows[i])
        << "overflow at row " << i;
  }
}

// Physical form of one batch's state column.
struct Form {
  bool blob;
  DecimalStateShape shape;
};

constexpr Form kBlob{true, DecimalStateShape::kSum64};
constexpr Form structOf(DecimalStateShape shape) {
  return Form{false, shape};
}

// The rows as a column of `form` reports them.
StateRows observedAs(const StateRows& rows, Form form) {
  if (!form.blob) {
    return rows.as(form.shape);
  }
  StateRows out = rows;
  out.overflows.assign(rows.size(), 0);
  return out;
}

std::unique_ptr<cudf::column>
makeStateColumn(const StateRows& rows, Form form, cuda::stream_ref stream) {
  return form.blob ? makeBlobColumn(rows, stream)
                   : makeStructColumn(rows, form.shape, stream);
}

// ---------------------------------------------------------------------------
// Fixture.
// ---------------------------------------------------------------------------

class DecimalStateConcatTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    savedConfig_ = CudfConfig::getInstance();
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
  }

  void TearDown() override {
    if (savedConfig_.has_value()) {
      unregisterCudf();
      CudfConfig::getInstance() = *savedConfig_;
    }
    exec::test::OperatorTestBase::TearDown();
  }

  // ROW(k BIGINT, s VARBINARY).
  RowTypePtr keyStateType() const {
    return ROW({"k", "s"}, {BIGINT(), VARBINARY()});
  }

  // One batch of ROW(k BIGINT, s VARBINARY) with keys [keyBase, keyBase+n).
  CudfVectorPtr
  makeKeyStateBatch(int64_t keyBase, const StateRows& rows, Form form) {
    std::vector<int64_t> keys(rows.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      keys[i] = keyBase + static_cast<int64_t>(i);
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(makeInt64Column(keys, stream_));
    columns.push_back(makeStateColumn(rows, form, stream_));
    return makeVector(keyStateType(), std::move(columns));
  }

  CudfVectorPtr makeVector(
      const RowTypePtr& type,
      std::vector<std::unique_ptr<cudf::column>> columns) {
    auto table = std::make_unique<cudf::table>(std::move(columns));
    const auto size = static_cast<vector_size_t>(table->num_rows());
    return std::make_shared<CudfVector>(
        pool(), type, size, std::move(table), stream_);
  }

  CudfVectorPtr makeEmptyBatch(const RowTypePtr& type) {
    auto table = makeEmptyTable(type);
    return std::make_shared<CudfVector>(
        pool(), type, 0, std::move(table), stream_);
  }

  // Runs `batches` through all three funnel entry points and returns, for
  // each, the single output table's view owner.
  enum class Funnel { kTable, kTableBatched, kCudfVectorsBatched };

  std::vector<std::unique_ptr<cudf::table>> runFunnel(
      Funnel funnel,
      std::vector<CudfVectorPtr> batches,
      const RowTypePtr& type) {
    std::vector<std::unique_ptr<cudf::table>> out;
    switch (funnel) {
      case Funnel::kTable:
        out.push_back(
            getConcatenatedTable(std::move(batches), type, stream_, testMr()));
        break;
      case Funnel::kTableBatched:
        out = getConcatenatedTableBatched(
            std::move(batches), type, stream_, testMr());
        break;
      case Funnel::kCudfVectorsBatched: {
        auto vectors = getConcatenatedCudfVectorsBatched(
            pool(), std::move(batches), type, stream_, testMr());
        for (auto& vector : vectors) {
          EXPECT_EQ(vector->size(), vector->getTableView().num_rows());
          out.push_back(vector->release());
        }
        break;
      }
    }
    stream_.sync();
    return out;
  }

  static std::vector<Funnel> allFunnels() {
    return {Funnel::kTable, Funnel::kTableBatched, Funnel::kCudfVectorsBatched};
  }

  // Builds key/state batches from (rows, form) pairs, runs each funnel and
  // checks keys (order), state values and nulls, and the result form.
  void checkMix(
      const std::vector<std::pair<StateRows, Form>>& inputs,
      std::optional<cudf::type_id> expectedTypeId,
      std::optional<DecimalStateShape> expectedShape) {
    for (auto funnel : allFunnels()) {
      SCOPED_TRACE(static_cast<int>(funnel));
      std::vector<CudfVectorPtr> batches;
      StateRows expected;
      std::vector<int64_t> expectedKeys;
      int64_t keyBase = 0;
      for (const auto& [rows, form] : inputs) {
        batches.push_back(makeKeyStateBatch(keyBase, rows, form));
        expected.append(observedAs(rows, form));
        for (size_t i = 0; i < rows.size(); ++i) {
          expectedKeys.push_back(keyBase + static_cast<int64_t>(i));
        }
        keyBase += 1000;
      }
      auto tables = runFunnel(funnel, std::move(batches), keyStateType());
      ASSERT_EQ(tables.size(), 1);
      const auto view = tables[0]->view();
      ASSERT_EQ(view.num_columns(), 2);
      EXPECT_EQ(view.column(0).type().id(), cudf::type_id::INT64);
      EXPECT_EQ(copyColumnData<int64_t>(view.column(0), stream_), expectedKeys);
      const auto state = view.column(1);
      if (expectedTypeId) {
        EXPECT_EQ(state.type().id(), *expectedTypeId);
      }
      if (expectedShape) {
        ASSERT_TRUE(isDecimalStateStruct(state));
        EXPECT_EQ(decimalStateShapeOf(state), *expectedShape);
        expectRowsEqual(expected.as(*expectedShape), readState(state, stream_));
      } else {
        expectRowsEqual(expected, readState(state, stream_));
      }
    }
  }

  std::optional<CudfConfig> savedConfig_;
  cuda::stream_ref stream_{cudf::get_default_stream()};
};

// ---------------------------------------------------------------------------
// Funnel tests.
// ---------------------------------------------------------------------------

TEST_F(DecimalStateConcatTest, allBlobsUnchanged) {
  checkMix(
      {{makeRows(0, 5), kBlob}, {makeRows(10, 7), kBlob}},
      cudf::type_id::STRING,
      std::nullopt);
}

TEST_F(DecimalStateConcatTest, allStructSameShape) {
  for (auto shape :
       {DecimalStateShape::kSum64,
        DecimalStateShape::kSum128,
        DecimalStateShape::kAvg64,
        DecimalStateShape::kAvg128}) {
    SCOPED_TRACE(static_cast<int>(shape));
    checkMix(
        {{makeRows(0, 5), structOf(shape)},
         {makeRows(10, 6), structOf(shape)},
         {makeRows(20, 3), structOf(shape)}},
        cudf::type_id::STRUCT,
        shape);
  }
}

TEST_F(DecimalStateConcatTest, blobAndStructMixToStruct) {
  // The blob carries count; a sum-only struct target reads count back as 1.
  // Blob rows are compared as the target shape would report them.
  for (auto shape :
       {DecimalStateShape::kSum64,
        DecimalStateShape::kSum128,
        DecimalStateShape::kAvg64,
        DecimalStateShape::kAvg128}) {
    SCOPED_TRACE(static_cast<int>(shape));
    checkMix(
        {{makeRows(0, 5), kBlob},
         {makeRows(10, 6), structOf(shape)},
         {makeRows(20, 4, 0), kBlob},
         {makeRows(30, 2), structOf(shape)}},
        cudf::type_id::STRUCT,
        shape);
  }
}

TEST_F(DecimalStateConcatTest, twoStructShapesWidened) {
  checkMix(
      {{makeRows(0, 5), structOf(DecimalStateShape::kSum64)},
       {makeRows(10, 6), structOf(DecimalStateShape::kAvg64)},
       {makeRows(20, 3), structOf(DecimalStateShape::kSum128)}},
      cudf::type_id::STRUCT,
      DecimalStateShape::kAvg128);
}

TEST_F(DecimalStateConcatTest, blobAndTwoStructShapesWidened) {
  checkMix(
      {{makeRows(0, 5), kBlob},
       {makeRows(10, 6), structOf(DecimalStateShape::kSum64)},
       {makeRows(20, 3), structOf(DecimalStateShape::kAvg128)}},
      cudf::type_id::STRUCT,
      DecimalStateShape::kAvg128);
}

TEST_F(DecimalStateConcatTest, zeroRowBlobWithStructBatches) {
  // makeEmptyTable builds STRING for VARBINARY. It must not reach
  // cudf::concatenate next to struct batches.
  for (auto funnel : allFunnels()) {
    SCOPED_TRACE(static_cast<int>(funnel));
    const auto a = makeRows(0, 5);
    const auto b = makeRows(10, 3);
    std::vector<CudfVectorPtr> batches;
    batches.push_back(makeEmptyBatch(keyStateType()));
    batches.push_back(
        makeKeyStateBatch(0, a, structOf(DecimalStateShape::kAvg64)));
    batches.push_back(makeEmptyBatch(keyStateType()));
    batches.push_back(
        makeKeyStateBatch(1000, b, structOf(DecimalStateShape::kAvg64)));
    batches.push_back(makeEmptyBatch(keyStateType()));
    ASSERT_EQ(
        batches[0]->getTableView().column(1).type().id(),
        cudf::type_id::STRING);

    auto tables = runFunnel(funnel, std::move(batches), keyStateType());
    ASSERT_EQ(tables.size(), 1);
    const auto state = tables[0]->view().column(1);
    ASSERT_TRUE(isDecimalStateStruct(state));
    EXPECT_EQ(decimalStateShapeOf(state), DecimalStateShape::kAvg64);
    StateRows expected = a.as(DecimalStateShape::kAvg64);
    expected.append(b.as(DecimalStateShape::kAvg64));
    expectRowsEqual(expected, readState(state, stream_));
  }
}

TEST_F(DecimalStateConcatTest, zeroRowStructWithBlobBatches) {
  checkMix(
      {{makeRows(0, 0), structOf(DecimalStateShape::kSum128)},
       {makeRows(0, 5), kBlob},
       {makeRows(10, 4), kBlob},
       {makeRows(0, 0), structOf(DecimalStateShape::kAvg64)}},
      cudf::type_id::STRING,
      std::nullopt);
}

TEST_F(DecimalStateConcatTest, allEmptyInputs) {
  for (auto funnel : allFunnels()) {
    SCOPED_TRACE(static_cast<int>(funnel));
    // No inputs at all: a typed empty table.
    {
      auto tables = runFunnel(funnel, {}, keyStateType());
      ASSERT_EQ(tables.size(), 1);
      EXPECT_EQ(tables[0]->num_rows(), 0);
      EXPECT_EQ(tables[0]->num_columns(), 2);
    }
    // Zero-row inputs of both forms: exactly one zero-row output.
    {
      std::vector<CudfVectorPtr> batches;
      batches.push_back(makeEmptyBatch(keyStateType()));
      batches.push_back(makeKeyStateBatch(
          0, makeRows(0, 0), structOf(DecimalStateShape::kAvg128)));
      batches.push_back(makeEmptyBatch(keyStateType()));
      auto tables = runFunnel(funnel, std::move(batches), keyStateType());
      ASSERT_EQ(tables.size(), 1);
      EXPECT_EQ(tables[0]->num_rows(), 0);
      ASSERT_EQ(tables[0]->num_columns(), 2);
      EXPECT_TRUE(isDecimalStateColumn(tables[0]->view().column(1)));
    }
  }
}

TEST_F(DecimalStateConcatTest, onlyMixedVarbinaryColumnIsTouched) {
  // ROW(k BIGINT, s1 VARBINARY, s2 VARBINARY): s1 mixes forms, s2 is all
  // blobs and must stay STRING with identical content.
  const auto type =
      ROW({"k", "s1", "s2"}, {BIGINT(), VARBINARY(), VARBINARY()});
  for (auto funnel : allFunnels()) {
    SCOPED_TRACE(static_cast<int>(funnel));
    const auto a1 = makeRows(0, 5);
    const auto a2 = makeRows(100, 5, 1);
    const auto b1 = makeRows(10, 6);
    const auto b2 = makeRows(200, 6, 2);
    std::vector<CudfVectorPtr> batches;
    {
      std::vector<std::unique_ptr<cudf::column>> cols;
      cols.push_back(makeInt64Column({0, 1, 2, 3, 4}, stream_));
      cols.push_back(makeBlobColumn(a1, stream_));
      cols.push_back(makeBlobColumn(a2, stream_));
      batches.push_back(makeVector(type, std::move(cols)));
    }
    {
      std::vector<std::unique_ptr<cudf::column>> cols;
      cols.push_back(makeInt64Column({5, 6, 7, 8, 9, 10}, stream_));
      cols.push_back(makeStructColumn(b1, DecimalStateShape::kSum64, stream_));
      cols.push_back(makeBlobColumn(b2, stream_));
      batches.push_back(makeVector(type, std::move(cols)));
    }
    auto tables = runFunnel(funnel, std::move(batches), type);
    ASSERT_EQ(tables.size(), 1);
    const auto view = tables[0]->view();
    EXPECT_EQ(
        copyColumnData<int64_t>(view.column(0), stream_),
        (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    StateRows expected1 = observedAs(a1, kBlob);
    expected1.append(observedAs(b1, structOf(DecimalStateShape::kSum64)));
    ASSERT_TRUE(isDecimalStateStruct(view.column(1)));
    EXPECT_EQ(decimalStateShapeOf(view.column(1)), DecimalStateShape::kSum64);
    expectRowsEqual(
        expected1.as(DecimalStateShape::kSum64),
        readState(view.column(1), stream_));

    StateRows expected2 = observedAs(a2, kBlob);
    expected2.append(observedAs(b2, kBlob));
    EXPECT_EQ(view.column(2).type().id(), cudf::type_id::STRING);
    expectRowsEqual(expected2, readState(view.column(2), stream_));
  }
}

TEST_F(DecimalStateConcatTest, batchedFunnelNormalizesPerOutputBatch) {
  // Force two output batches of 4 rows; each mixes a blob with a struct.
  CudfConfig::getInstance().batchSizeMaxThreshold = 4;
  for (auto funnel : {Funnel::kTableBatched, Funnel::kCudfVectorsBatched}) {
    SCOPED_TRACE(static_cast<int>(funnel));
    const auto r0 = makeRows(0, 2);
    const auto r1 = makeRows(10, 2);
    const auto r2 = makeRows(20, 2, 1);
    const auto r3 = makeRows(30, 2, 0);
    std::vector<CudfVectorPtr> batches;
    batches.push_back(makeKeyStateBatch(0, r0, kBlob));
    batches.push_back(
        makeKeyStateBatch(2, r1, structOf(DecimalStateShape::kSum64)));
    batches.push_back(makeKeyStateBatch(4, r2, kBlob));
    batches.push_back(
        makeKeyStateBatch(6, r3, structOf(DecimalStateShape::kSum64)));
    auto tables = runFunnel(funnel, std::move(batches), keyStateType());
    ASSERT_EQ(tables.size(), 2);
    for (size_t t = 0; t < 2; ++t) {
      const auto view = tables[t]->view();
      std::vector<int64_t> keys;
      for (int64_t k = 4 * t; k < 4 * t + 4; ++k) {
        keys.push_back(k);
      }
      EXPECT_EQ(copyColumnData<int64_t>(view.column(0), stream_), keys);
      StateRows expected = observedAs(t == 0 ? r0 : r2, kBlob);
      expected.append(
          observedAs(t == 0 ? r1 : r3, structOf(DecimalStateShape::kSum64)));
      ASSERT_TRUE(isDecimalStateStruct(view.column(1)));
      expectRowsEqual(
          expected.as(DecimalStateShape::kSum64),
          readState(view.column(1), stream_));
    }
  }
}

// ---------------------------------------------------------------------------
// normalizeDecimalStateTableViews directly: identity of untouched columns.
// ---------------------------------------------------------------------------

TEST_F(DecimalStateConcatTest, normalizeLeavesOtherColumnsIdentical) {
  const auto type =
      ROW({"k", "s1", "s2"}, {BIGINT(), VARBINARY(), VARBINARY()});
  std::vector<std::unique_ptr<cudf::table>> owners;
  auto makeTable = [&](std::vector<int64_t> keys,
                       const StateRows& s1,
                       Form f1,
                       const StateRows& s2) {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(makeInt64Column(keys, stream_));
    cols.push_back(makeStateColumn(s1, f1, stream_));
    cols.push_back(makeBlobColumn(s2, stream_));
    owners.push_back(std::make_unique<cudf::table>(std::move(cols)));
    return owners.back()->view();
  };
  std::vector<cudf::table_view> views{
      makeTable({0, 1, 2}, makeRows(0, 3), kBlob, makeRows(50, 3)),
      makeTable(
          {3, 4},
          makeRows(10, 2),
          structOf(DecimalStateShape::kAvg64),
          makeRows(60, 2))};
  const auto original = views;

  auto replacements =
      normalizeDecimalStateTableViews(views, type, stream_, testMr());
  stream_.sync();
  ASSERT_EQ(views.size(), 2);
  // The blob batch of s1 was unpacked into a replacement column.
  EXPECT_FALSE(replacements.empty());
  for (size_t b = 0; b < views.size(); ++b) {
    // BIGINT and the uniform VARBINARY column keep the same device memory.
    EXPECT_EQ(views[b].column(0).head(), original[b].column(0).head());
    EXPECT_EQ(views[b].column(2).head(), original[b].column(2).head());
    EXPECT_EQ(views[b].column(2).type(), original[b].column(2).type());
    ASSERT_TRUE(isDecimalStateStruct(views[b].column(1)));
    EXPECT_EQ(
        decimalStateShapeOf(views[b].column(1)), DecimalStateShape::kAvg64);
  }
  // The struct batch is left as is (no copy).
  EXPECT_EQ(
      views[1].column(1).child(0).head(),
      original[1].column(1).child(0).head());
}

TEST_F(DecimalStateConcatTest, normalizeIsNoOpWithoutMix) {
  const auto type = keyStateType();
  std::vector<std::unique_ptr<cudf::table>> owners;
  owners.push_back(makeEmptyTable(type));
  for (int i = 0; i < 2; ++i) {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(makeInt64Column({1, 2}, stream_));
    cols.push_back(makeBlobColumn(makeRows(i, 2), stream_));
    owners.push_back(std::make_unique<cudf::table>(std::move(cols)));
  }
  std::vector<cudf::table_view> views;
  for (const auto& owner : owners) {
    views.push_back(owner->view());
  }
  auto replacements =
      normalizeDecimalStateTableViews(views, type, stream_, testMr());
  EXPECT_TRUE(replacements.empty());
  // Zero-row views are only dropped when a mix must be resolved.
  ASSERT_EQ(views.size(), 3);
  for (size_t b = 0; b < views.size(); ++b) {
    EXPECT_EQ(views[b].column(1).head(), owners[b]->view().column(1).head());
  }

  // A non-VARBINARY column with differing physical types is never inspected
  // (the type says BIGINT; only VARBINARY columns are normalized).
  {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(makeInt64Column({3, 4}, stream_));
    cols.push_back(
        makeStructColumn(makeRows(5, 2), DecimalStateShape::kSum64, stream_));
    owners.push_back(std::make_unique<cudf::table>(std::move(cols)));
  }
  const auto bigintType = ROW({"k", "s"}, {BIGINT(), BIGINT()});
  std::vector<cudf::table_view> mixed{owners[1]->view(), owners[3]->view()};
  EXPECT_TRUE(
      normalizeDecimalStateTableViews(mixed, bigintType, stream_, testMr())
          .empty());
  EXPECT_EQ(mixed[0].column(1).type().id(), cudf::type_id::STRING);
  EXPECT_EQ(mixed[1].column(1).type().id(), cudf::type_id::STRUCT);
}

// ---------------------------------------------------------------------------
// End to end: GPU partial aggregation and CPU-produced blob states meet in
// CudfBatchConcat in front of the GPU final aggregation.
// ---------------------------------------------------------------------------

TEST_F(DecimalStateConcatTest, partialToBatchConcatToFinalMatchesCpu) {
  auto& config = CudfConfig::getInstance();
  config.concatOptimizationEnabled = true;
  config.batchSizeMinThreshold = 1'000'000;

  auto makeInput = [&](int32_t seed) {
    return makeRowVector(
        {"k", "d64", "d128"},
        {
            makeFlatVector<int32_t>(64, [](auto row) { return row % 7; }),
            makeFlatVector<int64_t>(
                64,
                [seed](auto row) { return (row * 37 + seed) % 10'000 - 5'000; },
                [](auto row) { return row % 11 == 0; },
                DECIMAL(12, 2)),
            makeFlatVector<int128_t>(
                64,
                [seed](auto row) {
                  return static_cast<int128_t>(row * 1'234'567 + seed) *
                      1'000'000'007;
                },
                [](auto row) { return row % 13 == 0; },
                DECIMAL(30, 3)),
        });
  };
  const std::vector<std::string> aggregates = {
      "sum(d64) AS s64",
      "avg(d64) AS a64",
      "sum(d128) AS s128",
      "avg(d128) AS a128"};

  std::vector<RowVectorPtr> gpuInputs = {
      makeInput(1), makeInput(2), makeInput(3)};
  std::vector<RowVectorPtr> cpuInputs = {makeInput(4), makeInput(5)};

  // CPU-produced partial states: VARBINARY blobs, as from an HTTP exchange or
  // a CPU-fallback producer. They enter the GPU plan through CudfFromVelox as
  // STRING columns.
  unregisterCudf();
  auto cpuPartial = exec::test::AssertQueryBuilder(
                        exec::test::PlanBuilder()
                            .values(cpuInputs)
                            .partialAggregation({"k"}, aggregates)
                            .planNode())
                        .copyResults(pool());
  std::vector<RowVectorPtr> allInputs = gpuInputs;
  allInputs.insert(allInputs.end(), cpuInputs.begin(), cpuInputs.end());
  auto expected = exec::test::AssertQueryBuilder(
                      exec::test::PlanBuilder()
                          .values(allInputs)
                          .singleAggregation({"k"}, aggregates)
                          .orderBy({"k"}, false)
                          .planNode())
                      .copyResults(pool());
  registerCudf();

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<core::PlanNodePtr> sources;
  // The first source must be the GPU partial aggregation: finalAggregation()
  // derives the final node from it.
  for (const auto& input : gpuInputs) {
    sources.push_back(
        exec::test::PlanBuilder(generator)
            .values({input})
            .partialAggregation({"k"}, aggregates)
            .planNode());
  }
  sources.push_back(
      exec::test::PlanBuilder(generator).values({cpuPartial}).planNode());
  auto plan = exec::test::PlanBuilder(generator)
                  .localPartitionRoundRobin(sources)
                  .finalAggregation()
                  .orderBy({"k"}, false)
                  .planNode();

  auto result =
      exec::test::AssertQueryBuilder(plan).maxDrivers(1).copyResults(pool());
  velox::test::assertEqualVectors(expected, result);
}

} // namespace
} // namespace facebook::velox::cudf_velox
