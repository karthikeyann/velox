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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/DecimalAggregationDevice.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/EnumDefine.h"
#include "velox/common/base/Exceptions.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>

namespace facebook::velox::cudf_velox {
namespace {

const auto& decimalStateShapeNames() {
  static const folly::F14FastMap<DecimalStateShape, std::string_view> kNames = {
      {DecimalStateShape::kSum64, "SUM64"},
      {DecimalStateShape::kSum128, "SUM128"},
      {DecimalStateShape::kAvg64, "AVG64"},
      {DecimalStateShape::kAvg128, "AVG128"},
  };
  return kNames;
}

std::optional<DecimalStateShape> tryDecimalStateShapeOf(
    const cudf::column_view& column) {
  if (column.type().id() != cudf::type_id::STRUCT) {
    return std::nullopt;
  }
  auto const isDecimal = [&](int i) {
    return column.child(i).type().id() == cudf::type_id::DECIMAL128;
  };
  auto const isInt64 = [&](int i) {
    return column.child(i).type().id() == cudf::type_id::INT64;
  };
  switch (column.num_children()) {
    case 1:
      if (isDecimal(0)) {
        return DecimalStateShape::kSum64;
      }
      break;
    case 2:
      if (isInt64(0) && isDecimal(1)) {
        return DecimalStateShape::kSum128;
      }
      if (isDecimal(0) && isInt64(1)) {
        return DecimalStateShape::kAvg64;
      }
      break;
    case 3:
      if (isDecimal(0) && isInt64(1) && isInt64(2)) {
        return DecimalStateShape::kAvg128;
      }
      break;
    default:
      break;
  }
  return std::nullopt;
}

// Returns a view of child `index` restricted to the parent's row range. For an
// unsliced parent this is exactly cudf::column_view::child (no device work);
// for a sliced parent it is cudf::structs_column_view::get_sliced_child, which
// aliases the same device memory and only recomputes the null count.
cudf::column_view stateChild(
    const cudf::column_view& parent,
    int index,
    cuda::stream_ref stream) {
  auto child = parent.child(index);
  if (parent.offset() == 0 && parent.size() == child.size()) {
    return child;
  }
  return cudf::structs_column_view(parent).get_sliced_child(index, stream);
}

// Velox decimal scale (non-negative) of a cuDF decimal column.
int32_t veloxScaleOf(const cudf::column_view& decimal) {
  return -decimal.type().scale();
}

std::unique_ptr<cudf::column> makeConstantInt64Column(
    int64_t value,
    cudf::size_type numRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  cudf::numeric_scalar<int64_t> scalar(value, true, stream, mr);
  return cudf::make_column_from_scalar(scalar, numRows, stream, mr);
}

std::unique_ptr<cudf::column> makeEmptyDecimalStateStruct(
    DecimalStateShape shape,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto const layout = decimalStateLayout(shape);
  std::vector<std::unique_ptr<cudf::column>> children(layout.numChildren);
  children[layout.sum] = cudf::make_empty_column(
      cudf::data_type{cudf::type_id::DECIMAL128, -scale});
  if (layout.hasCount()) {
    children[layout.count] = cudf::make_empty_column(cudf::type_id::INT64);
  }
  if (layout.hasOverflow()) {
    children[layout.overflow] = cudf::make_empty_column(cudf::type_id::INT64);
  }
  return cudf::make_structs_column(
      0,
      std::move(children),
      0,
      cuda::device_buffer<std::byte>{stream, mr},
      stream,
      mr);
}

// Decodes a STRING blob into flat columns. `count` is only produced when
// `withCount` is set and `overflow` only when `withOverflow` is set (the
// unpack kernel needs a count buffer either way, so an unrequested count is a
// temporary). All produced columns carry a copy of the blob's null mask when
// it has one.
DecimalStateColumns deserializeDecimalStateImpl(
    const cudf::column_view& stateCol,
    int32_t scale,
    bool withCount,
    bool withOverflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      stateCol.type().id() == cudf::type_id::STRING,
      "Decimal sum state requires a STRING (VARBINARY) column: {}",
      cudf::type_to_name(stateCol.type()));
  if (stateCol.offset() != 0) {
    // The unpack kernel addresses offsets, chars and the null mask from row 0.
    // Sliced blob views never occur on operator paths, so a contiguous copy is
    // the simplest correct handling.
    auto contiguous =
        std::make_unique<cudf::column>(stateCol, stream, get_temp_mr());
    return deserializeDecimalStateImpl(
        contiguous->view(), scale, withCount, withOverflow, stream, mr);
  }
  auto const numRows = stateCol.size();
  auto const sumType = cudf::data_type{cudf::type_id::DECIMAL128, -scale};
  auto const int64Type = cudf::data_type{cudf::type_id::INT64};

  auto makeAll = [&](cudf::mask_state state) {
    DecimalStateColumns columns;
    columns.sum =
        cudf::make_fixed_width_column(sumType, numRows, state, stream, mr);
    columns.count =
        cudf::make_fixed_width_column(int64Type, numRows, state, stream, mr);
    if (withOverflow) {
      columns.overflow =
          cudf::make_fixed_width_column(int64Type, numRows, state, stream, mr);
    }
    return columns;
  };
  auto finish = [&](DecimalStateColumns columns) {
    if (!withCount) {
      columns.count.reset();
    }
    return columns;
  };

  if (numRows == 0) {
    return finish(makeAll(cudf::mask_state::UNALLOCATED));
  }

  // For fully-null state columns there is nothing to deserialize. Avoid
  // launching unpack kernels over string payload buffers that may be empty.
  if (stateCol.nullable() && stateCol.null_count() == numRows) {
    return finish(makeAll(cudf::mask_state::ALL_NULL));
  }

  cudf::strings_column_view strings(stateCol);

  auto const nullCount = stateCol.nullable() ? stateCol.null_count() : 0;
  auto const payloadSize = strings.chars_size(stream);
  // A null row's payload width is path dependent. serializeDecimalSumState
  // writes kDecimalSumStateSize bytes for every row including nulls, while a
  // velox/Arrow round trip compacts null rows to 0 bytes. Both encodings can
  // occur in the same column: the streaming final aggregation concatenates its
  // buffered (serialized) result with each newly arrived (round tripped) batch,
  // so the payload size lands anywhere between the two extremes. Non-null rows
  // are always kDecimalSumStateSize wide, and that is all
  // unpackDecimalSumState requires -- it skips null rows and addresses payloads
  // through the per-row offsets rather than a fixed stride.
  auto const fullPayloadSize =
      static_cast<int64_t>(numRows) * detail::kDecimalSumStateSize;
  auto const compactPayloadSize =
      static_cast<int64_t>(numRows - nullCount) * detail::kDecimalSumStateSize;
  VELOX_CHECK(
      payloadSize >= compactPayloadSize && payloadSize <= fullPayloadSize &&
          payloadSize % detail::kDecimalSumStateSize == 0,
      "Decimal sum state has an invalid payload size: expected a multiple of {} in [{}, {}], got {}",
      detail::kDecimalSumStateSize,
      compactPayloadSize,
      fullPayloadSize,
      payloadSize);

  auto offsetsView = strings.offsets();
  auto charsPtr = reinterpret_cast<const uint8_t*>(strings.chars_begin(stream));

  auto result = makeAll(cudf::mask_state::UNALLOCATED);

  auto const offsetsType = offsetsView.type().id();
  VELOX_CHECK(
      offsetsType == cudf::type_id::INT32 ||
          offsetsType == cudf::type_id::INT64,
      "Decimal sum state requires INT32 or INT64 offsets: {}",
      cudf::type_to_name(offsetsView.type()));
  detail::unpackDecimalSumState(
      offsetsType,
      offsetsView,
      charsPtr,
      result.sum->mutable_view(),
      result.count->mutable_view(),
      withOverflow ? result.overflow->mutable_view().data<int64_t>() : nullptr,
      numRows,
      stateCol.null_mask(),
      stream);

  if (stateCol.nullable()) {
    result.sum->set_null_mask(
        cudf::copy_bitmask(stateCol, stream, mr), nullCount);
    result.count->set_null_mask(
        cudf::copy_bitmask(stateCol, stream, mr), nullCount);
    if (withOverflow) {
      result.overflow->set_null_mask(
          cudf::copy_bitmask(stateCol, stream, mr), nullCount);
    }
  }
  return finish(std::move(result));
}

// Encodes sum plus optional count/overflow columns into the 32-byte blob.
// Null mask: buildStateValidityMask(sum, count) when `countCol` is given
// (null sum, null count or zero count -> null row), otherwise a copy of the
// sum's mask.
std::unique_ptr<cudf::column> serializeDecimalStateImpl(
    const cudf::column_view& sumCol,
    const cudf::column_view* countCol,
    const cudf::column_view* overflowCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto const numRows = sumCol.size();
  if (countCol) {
    VELOX_CHECK(
        countCol->type().id() == cudf::type_id::INT64,
        "Decimal sum state requires an INT64 count column: {}",
        cudf::type_to_name(countCol->type()));
    VELOX_CHECK_EQ(
        numRows,
        countCol->size(),
        "Decimal sum state requires sum and count of the same size");
  }
  if (overflowCol) {
    VELOX_CHECK(
        overflowCol->type().id() == cudf::type_id::INT64,
        "Decimal sum state requires an INT64 overflow column: {}",
        cudf::type_to_name(overflowCol->type()));
    VELOX_CHECK_EQ(
        numRows,
        overflowCol->size(),
        "Decimal sum state requires sum and overflow of the same size");
  }
  VELOX_CHECK_LE(
      numRows,
      static_cast<cudf::size_type>(std::numeric_limits<int32_t>::max()),
      "Too many rows to serialize decimal sum state");

  if (numRows == 0) {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

  auto const rowCount = static_cast<int32_t>(numRows);

  auto const charsBytes =
      static_cast<int64_t>(numRows) * detail::kDecimalSumStateSize;
  auto const threshold = cudf::strings::get_offset64_threshold();
  auto const useLargeOffsets = charsBytes >= threshold;
  VELOX_CHECK(
      !useLargeOffsets || cudf::strings::is_large_strings_enabled(),
      "Size of output ({}) exceeds the column size limit ({})",
      charsBytes,
      threshold);

  auto const offsetsType =
      useLargeOffsets ? cudf::type_id::INT64 : cudf::type_id::INT32;
  auto offsetsCol = cudf::make_fixed_width_column(
      cudf::data_type{offsetsType},
      numRows + 1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto offsetsView = offsetsCol->mutable_view();

  rmm::device_buffer charsBuf(
      static_cast<size_t>(numRows) * detail::kDecimalSumStateSize, stream, mr);

  auto charsPtr = reinterpret_cast<uint8_t*>(charsBuf.data());
  detail::fillOffsetsForDecimalSumState(
      offsetsType, offsetsView, rowCount, stream);

  const auto sumType = sumCol.type().id();
  VELOX_CHECK(
      sumType == cudf::type_id::DECIMAL64 ||
          sumType == cudf::type_id::DECIMAL128,
      "Unsupported decimal sum column type: {}",
      cudf::type_to_name(sumCol.type()));
  detail::packDecimalSumState(
      sumType,
      offsetsType,
      sumCol,
      countCol ? countCol->data<int64_t>() : nullptr,
      overflowCol ? overflowCol->data<int64_t>() : nullptr,
      offsetsView,
      charsPtr,
      rowCount,
      stream);

  cuda::device_buffer<std::byte> nullMask{stream, mr};
  cudf::size_type nullCount = 0;
  if (countCol) {
    std::tie(nullMask, nullCount) =
        detail::buildStateValidityMask(sumCol, *countCol, stream, mr);
  } else if (sumCol.nullable()) {
    nullMask = cudf::copy_bitmask(sumCol, stream, mr);
    nullCount = sumCol.null_count();
  }
  return cudf::make_strings_column(
      static_cast<cudf::size_type>(numRows),
      std::move(offsetsCol),
      std::move(charsBuf),
      nullCount,
      std::move(nullMask));
}


} // namespace

std::unique_ptr<cudf::column> unpackDecimalState(
    const cudf::column_view& blobColumn,
    DecimalStateShape shape,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto flat = deserializeDecimalStateImpl(
      blobColumn,
      scale,
      decimalStateHasCount(shape),
      decimalStateHasOverflow(shape),
      stream,
      mr);
  return wrapDecimalState(std::move(flat), shape, stream, mr);
}

VELOX_DEFINE_ENUM_NAME(DecimalStateShape, decimalStateShapeNames);

DecimalStateLayout decimalStateLayout(DecimalStateShape shape) {
  constexpr int kAbsent = DecimalStateLayout::kAbsent;
  switch (shape) {
    case DecimalStateShape::kSum64:
      return {0, kAbsent, kAbsent, 1};
    case DecimalStateShape::kSum128:
      return {1, kAbsent, 0, 2};
    case DecimalStateShape::kAvg64:
      return {0, 1, kAbsent, 2};
    case DecimalStateShape::kAvg128:
      return {0, 1, 2, 3};
  }
  VELOX_UNREACHABLE();
}

bool decimalStateHasCount(DecimalStateShape shape) {
  return decimalStateLayout(shape).hasCount();
}

bool decimalStateHasOverflow(DecimalStateShape shape) {
  return decimalStateLayout(shape).hasOverflow();
}

DecimalStateShape decimalStateShapeFor(
    bool isAverage,
    bool rawInputIsDecimal128) {
  if (isAverage) {
    return rawInputIsDecimal128 ? DecimalStateShape::kAvg128
                                : DecimalStateShape::kAvg64;
  }
  return rawInputIsDecimal128 ? DecimalStateShape::kSum128
                              : DecimalStateShape::kSum64;
}

DecimalSumStateColumns reduceDecimal64SumCount(
    const cudf::column_view& input,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  DecimalSumStateColumns result;
  result.sum = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::DECIMAL128, input.type().scale()},
      1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  result.count = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT64},
      1,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  detail::reduceDecimal64SumCount(
      input,
      result.sum->mutable_view(),
      result.count->mutable_view(),
      stream,
      mr);
  auto [nullMask, nullCount] = detail::buildStateValidityMask(
      result.sum->view(), result.count->view(), stream, mr);
  result.sum->set_null_mask(std::move(nullMask), nullCount);
  return result;
}

DecimalSumStateColumns deserializeDecimalSumState(
    const cudf::column_view& stateCol,
    int32_t scale,
    cuda::stream_ref stream) {
  // The decoded sum/count columns are consumed by the next groupby/reduce and
  // never leave the operator, so they use the temporary memory resource.
  return deserializeDecimalStateImpl(
      stateCol,
      scale,
      /*withCount=*/true,
      /*withOverflow=*/false,
      stream,
      get_temp_mr());
}

std::unique_ptr<cudf::column> serializeDecimalSumState(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return serializeDecimalStateImpl(
      sumCol, &countCol, /*overflowCol=*/nullptr, stream, mr);
}

std::unique_ptr<cudf::column> computeDecimalAverage(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    const cudf::column_view& overflowCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      countCol.type().id() == cudf::type_id::INT64,
      "Decimal average requires an INT64 count column: {}",
      cudf::type_to_name(countCol.type()));
  VELOX_CHECK(
      sumCol.type().id() == cudf::type_id::DECIMAL64 ||
          sumCol.type().id() == cudf::type_id::DECIMAL128,
      "Decimal average requires a DECIMAL64 or DECIMAL128 sum column: {}",
      cudf::type_to_name(sumCol.type()));
  VELOX_CHECK_EQ(
      sumCol.size(),
      countCol.size(),
      "Decimal average requires sum and count of the same size");
  bool const withOverflow = overflowCol.size() > 0;
  if (withOverflow) {
    VELOX_CHECK(
        overflowCol.type().id() == cudf::type_id::INT64,
        "Decimal average requires an INT64 overflow column: {}",
        cudf::type_to_name(overflowCol.type()));
    VELOX_CHECK_EQ(
        sumCol.size(),
        overflowCol.size(),
        "Decimal average requires sum and overflow of the same size");
  }

  auto numRows = sumCol.size();
  auto average = cudf::make_fixed_width_column(
      sumCol.type(), numRows, cudf::mask_state::UNALLOCATED, stream, mr);

  if (numRows > 0) {
    detail::averageRoundDecimalSum(
        sumCol.type().id(),
        sumCol,
        countCol.data<int64_t>(),
        withOverflow ? overflowCol.data<int64_t>() : nullptr,
        average->mutable_view(),
        static_cast<int32_t>(numRows),
        stream);
  }

  auto [nullMask, nullCount] =
      detail::buildStateValidityMask(sumCol, countCol, stream, mr);
  if (nullCount > 0) {
    average->set_null_mask(std::move(nullMask), nullCount);
  }
  return average;
}

std::unique_ptr<cudf::column> computeDecimalAverage(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return computeDecimalAverage(
      sumCol, countCol, cudf::column_view{}, stream, mr);
}

std::unique_ptr<cudf::column> wrapDecimalState(
    DecimalStateColumns flat,
    DecimalStateShape shape,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_NOT_NULL(flat.sum, "Decimal state requires a sum column");
  auto const sumTypeId = flat.sum->type().id();
  VELOX_CHECK(
      sumTypeId == cudf::type_id::DECIMAL64 ||
          sumTypeId == cudf::type_id::DECIMAL128,
      "Decimal state sum must be DECIMAL64 or DECIMAL128: {}",
      cudf::type_to_name(flat.sum->type()));
  if (sumTypeId == cudf::type_id::DECIMAL64) {
    flat.sum = cudf::cast(
        flat.sum->view(),
        cudf::data_type{cudf::type_id::DECIMAL128, flat.sum->type().scale()},
        stream,
        mr);
  }
  auto const numRows = flat.sum->size();

  auto takeInt64 = [&](std::unique_ptr<cudf::column>& column,
                       std::string_view name) {
    VELOX_CHECK_NOT_NULL(
        column,
        "Decimal state shape {} requires a column: {}",
        DecimalStateShapeName::toName(shape),
        name);
    VELOX_CHECK_EQ(
        column->size(),
        numRows,
        "Decimal state column size must match the sum size: {}",
        name);
    if (column->type().id() != cudf::type_id::INT64) {
      column = cudf::cast(
          column->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
    }
    return std::move(column);
  };

  auto const layout = decimalStateLayout(shape);
  std::vector<std::unique_ptr<cudf::column>> children(layout.numChildren);
  children[layout.sum] = std::move(flat.sum);
  if (layout.hasCount()) {
    children[layout.count] = takeInt64(flat.count, "count");
  }
  if (layout.hasOverflow()) {
    if (!flat.overflow) {
      // GPU producers never track carries, so a state that did not carry the
      // field has overflow 0 by definition.
      flat.overflow = makeConstantInt64Column(0, numRows, stream, mr);
    }
    children[layout.overflow] = takeInt64(flat.overflow, "overflow");
  }
  // No parent null mask: validity lives on the sum child only, and an empty
  // mask makes make_structs_column adopt the children without copying them.
  return cudf::make_structs_column(
      numRows,
      std::move(children),
      0,
      cuda::device_buffer<std::byte>{stream, mr},
      stream,
      mr);
}

DecimalStateColumns unwrapDecimalState(
    std::unique_ptr<cudf::column> structColumn) {
  VELOX_CHECK_NOT_NULL(structColumn);
  auto const shape = decimalStateShapeOf(structColumn->view());
  VELOX_CHECK_EQ(
      structColumn->null_count(),
      0,
      "Decimal state struct must not carry a parent null mask; validity lives on the sum child");
  auto const layout = decimalStateLayout(shape);
  auto children = structColumn->release().children;
  DecimalStateColumns flat;
  flat.sum = std::move(children[layout.sum]);
  if (layout.hasCount()) {
    flat.count = std::move(children[layout.count]);
  }
  if (layout.hasOverflow()) {
    flat.overflow = std::move(children[layout.overflow]);
  }
  return flat;
}

bool isDecimalStateStruct(const cudf::column_view& column) {
  return tryDecimalStateShapeOf(column).has_value();
}

DecimalStateShape decimalStateShapeOf(const cudf::column_view& column) {
  auto shape = tryDecimalStateShapeOf(column);
  VELOX_CHECK(
      shape.has_value(),
      "Column is not a decimal aggregate state struct: {} with {} children",
      cudf::type_to_name(column.type()),
      column.num_children());
  return *shape;
}

bool isDecimalStateColumn(const cudf::column_view& column) {
  return column.type().id() == cudf::type_id::STRING ||
      isDecimalStateStruct(column);
}

FlatDecimalState flattenDecimalState(
    const cudf::column_view& state,
    int32_t scale,
    bool needCount,
    bool needOverflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  FlatDecimalState flat;
  if (state.type().id() == cudf::type_id::STRING) {
    flat.owned = deserializeDecimalStateImpl(
        state, scale, needCount, needOverflow, stream, mr);
    flat.sum = flat.owned.sum->view();
    if (needCount) {
      flat.count = flat.owned.count->view();
    }
    if (needOverflow) {
      flat.overflow = flat.owned.overflow->view();
    }
    return flat;
  }

  auto const shape = decimalStateShapeOf(state);
  auto const layout = decimalStateLayout(shape);
  auto const numRows = state.size();

  flat.sum = stateChild(state, layout.sum, stream);
  if (state.null_count() > 0) {
    // Producers never emit a parent mask, but a nullifying gather (for example
    // an outer join probe) adds one. A row is null when the parent or the sum
    // child is null, so fold the parent mask into an owned copy of the sum.
    auto [mask, nullCount] =
        cudf::bitmask_and(cudf::table_view{{state, flat.sum}}, stream, mr);
    flat.owned.sum = std::make_unique<cudf::column>(flat.sum, stream, mr);
    flat.owned.sum->set_null_mask(std::move(mask), nullCount);
    flat.sum = flat.owned.sum->view();
  }
  if (layout.hasCount()) {
    flat.count = stateChild(state, layout.count, stream);
  } else if (needCount) {
    flat.owned.count = makeConstantInt64Column(1, numRows, stream, mr);
    flat.count = flat.owned.count->view();
  }
  if (layout.hasOverflow()) {
    flat.overflow = stateChild(state, layout.overflow, stream);
  } else if (needOverflow) {
    flat.owned.overflow = makeConstantInt64Column(0, numRows, stream, mr);
    flat.overflow = flat.owned.overflow->view();
  }
  return flat;
}

std::unique_ptr<cudf::column> packDecimalState(
    const cudf::column_view& structColumn,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      isDecimalStateStruct(structColumn),
      "Column is not a decimal aggregate state struct: {} with {} children",
      cudf::type_to_name(structColumn.type()),
      structColumn.num_children());
  // The scale argument is unused for a struct; fields the struct carries are
  // exposed without synthesis.
  auto flat = flattenDecimalState(
      structColumn,
      /*scale=*/0,
      /*needCount=*/false,
      /*needOverflow=*/false,
      stream,
      get_temp_mr());
  auto const layout = decimalStateLayout(decimalStateShapeOf(structColumn));
  return serializeDecimalStateImpl(
      flat.sum,
      layout.hasCount() ? &flat.count : nullptr,
      layout.hasOverflow() ? &flat.overflow : nullptr,
      stream,
      mr);
}

std::vector<std::unique_ptr<cudf::column>> normalizeDecimalStateBatches(
    std::vector<cudf::column_view>& views,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<std::unique_ptr<cudf::column>> owned;
  if (views.empty()) {
    return owned;
  }

  // Classify every batch. `shapes[i]` is nullopt for a STRING batch.
  std::vector<std::optional<DecimalStateShape>> shapes(views.size());
  for (size_t i = 0; i < views.size(); ++i) {
    shapes[i] = tryDecimalStateShapeOf(views[i]);
    VELOX_CHECK(
        shapes[i].has_value() || views[i].type().id() == cudf::type_id::STRING,
        "Batch {} is not a decimal aggregate state column: {}",
        i,
        cudf::type_to_name(views[i].type()));
  }

  // Pick the target form from the non-empty batches; zero-row batches are
  // wildcards. If every batch is empty, consider all of them so that a mixed
  // set of empties still ends up uniform.
  bool const allEmpty =
      std::all_of(views.begin(), views.end(), [](const auto& view) {
        return view.size() == 0;
      });
  std::optional<DecimalStateShape> target;
  std::optional<int32_t> targetScale;
  bool mixedShapes = false;
  for (size_t i = 0; i < views.size(); ++i) {
    if (!shapes[i] || (!allEmpty && views[i].size() == 0)) {
      continue;
    }
    if (!targetScale) {
      targetScale =
          veloxScaleOf(views[i].child(decimalStateLayout(*shapes[i]).sum));
    }
    if (!target) {
      target = *shapes[i];
    } else if (*target != *shapes[i]) {
      mixedShapes = true;
    }
  }
  if (mixedShapes) {
    target = DecimalStateShape::kAvg128;
  }

  if (!target) {
    // Every batch that matters is a STRING blob. Only zero-row struct batches
    // (if any) need re-typing.
    for (auto& view : views) {
      if (view.type().id() != cudf::type_id::STRING) {
        VELOX_DCHECK_EQ(view.size(), 0);
        owned.push_back(cudf::make_empty_column(cudf::type_id::STRING));
        view = owned.back()->view();
      }
    }
    return owned;
  }

  auto const targetShape = *target;
  auto const targetLayout = decimalStateLayout(targetShape);

  for (size_t i = 0; i < views.size(); ++i) {
    auto& view = views[i];
    if (shapes[i] && *shapes[i] == targetShape) {
      continue;
    }
    if (view.size() == 0) {
      owned.push_back(
          makeEmptyDecimalStateStruct(targetShape, *targetScale, stream, mr));
    } else if (!shapes[i]) {
      owned.push_back(
          unpackDecimalState(view, targetShape, *targetScale, stream, mr));
    } else {
      // Widen a struct of another shape to the target (kAvg128). Children
      // must be owned by the new struct, so those aliasing the input are
      // copied here; this path is only reached on version skew between
      // producers.
      auto flat = flattenDecimalState(
          view,
          *targetScale,
          targetLayout.hasCount(),
          targetLayout.hasOverflow(),
          stream,
          mr);
      auto own = [&](std::unique_ptr<cudf::column>& storage,
                     const cudf::column_view& field) {
        return storage ? std::move(storage)
                       : std::make_unique<cudf::column>(field, stream, mr);
      };
      DecimalStateColumns columns;
      columns.sum = own(flat.owned.sum, flat.sum);
      if (targetLayout.hasCount()) {
        columns.count = own(flat.owned.count, flat.count);
      }
      if (targetLayout.hasOverflow()) {
        columns.overflow = own(flat.owned.overflow, flat.overflow);
      }
      owned.push_back(
          wrapDecimalState(std::move(columns), targetShape, stream, mr));
    }
    view = owned.back()->view();
  }
  return owned;
}

} // namespace facebook::velox::cudf_velox
