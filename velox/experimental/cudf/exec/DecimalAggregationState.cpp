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

#include "velox/common/base/Exceptions.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>

namespace facebook::velox::cudf_velox {
namespace {

constexpr cudf::type_id kInt64Id = cudf::type_id::INT64;
constexpr cudf::type_id kDecimal128Id = cudf::type_id::DECIMAL128;

// Child index of each state field for a shape; -1 when the shape omits it.
struct ShapeLayout {
  int sum;
  int count;
  int overflow;
  int numChildren;
};

constexpr ShapeLayout layoutOf(DecimalStateShape shape) {
  switch (shape) {
    case DecimalStateShape::kSum64:
      return {0, -1, -1, 1};
    case DecimalStateShape::kSum128:
      return {1, -1, 0, 2};
    case DecimalStateShape::kAvg64:
      return {0, 1, -1, 2};
    case DecimalStateShape::kAvg128:
      return {0, 1, 2, 3};
  }
  return {0, -1, -1, 1};
}

const char* shapeName(DecimalStateShape shape) {
  switch (shape) {
    case DecimalStateShape::kSum64:
      return "kSum64[sum]";
    case DecimalStateShape::kSum128:
      return "kSum128[overflow,sum]";
    case DecimalStateShape::kAvg64:
      return "kAvg64[sum,count]";
    case DecimalStateShape::kAvg128:
      return "kAvg128[sum,count,overflow]";
  }
  return "unknown";
}

std::optional<DecimalStateShape> tryDecimalStateShapeOf(
    cudf::column_view const& column) {
  if (column.type().id() != cudf::type_id::STRUCT) {
    return std::nullopt;
  }
  auto const isDecimal = [&](int i) {
    return column.child(i).type().id() == kDecimal128Id;
  };
  auto const isInt64 = [&](int i) {
    return column.child(i).type().id() == kInt64Id;
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
    cudf::column_view const& parent,
    int index,
    cuda::stream_ref stream) {
  auto child = parent.child(index);
  if (parent.offset() == 0 && parent.size() == child.size()) {
    return child;
  }
  return cudf::structs_column_view(parent).get_sliced_child(index, stream);
}

// Velox decimal scale (non-negative) of a cuDF decimal column.
int32_t veloxScaleOf(cudf::column_view const& decimal) {
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
  auto const layout = layoutOf(shape);
  std::vector<std::unique_ptr<cudf::column>> children(layout.numChildren);
  children[layout.sum] =
      cudf::make_empty_column(cudf::data_type{kDecimal128Id, -scale});
  if (layout.count >= 0) {
    children[layout.count] = cudf::make_empty_column(kInt64Id);
  }
  if (layout.overflow >= 0) {
    children[layout.overflow] = cudf::make_empty_column(kInt64Id);
  }
  return cudf::make_structs_column(
      0,
      std::move(children),
      0,
      cuda::device_buffer<std::byte>{stream, mr},
      stream,
      mr);
}

// Decodes a STRING blob into flat columns. `overflow` is only produced when
// `withOverflow` is set. All produced columns carry a copy of the blob's null
// mask when it has one.
DecimalStateColumns deserializeDecimalStateImpl(
    const cudf::column_view& stateCol,
    int32_t scale,
    bool withOverflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      stateCol.type().id() == cudf::type_id::STRING,
      "Decimal sum state requires STRING/VARBINARY column (type is {})",
      cudf::type_to_name(stateCol.type()));
  auto const numRows = stateCol.size();
  auto const sumType = cudf::data_type{kDecimal128Id, -scale};
  auto const int64Type = cudf::data_type{kInt64Id};

  auto makeAll = [&](cudf::mask_state state) {
    DecimalStateColumns out;
    out.sum = cudf::make_fixed_width_column(sumType, numRows, state, stream, mr);
    out.count =
        cudf::make_fixed_width_column(int64Type, numRows, state, stream, mr);
    if (withOverflow) {
      out.overflow =
          cudf::make_fixed_width_column(int64Type, numRows, state, stream, mr);
    }
    return out;
  };

  if (numRows == 0) {
    return makeAll(cudf::mask_state::UNALLOCATED);
  }

  // For fully-null state columns there is nothing to deserialize. Avoid
  // launching unpack kernels over string payload buffers that may be empty.
  if (stateCol.nullable() && stateCol.null_count() == numRows) {
    return makeAll(cudf::mask_state::ALL_NULL);
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
      "Decimal sum state requires INT32 or INT64 offsets (offset type is {})",
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
  return result;
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
        countCol->type().id() == kInt64Id,
        "Decimal sum state requires INT64 count column (type is {})",
        cudf::type_to_name(countCol->type()));
    VELOX_CHECK_EQ(
        numRows,
        countCol->size(),
        "Decimal sum state requires sum and count to be same size (sum size is {}, count size is {})",
        sumCol.size(),
        countCol->size());
  }
  if (overflowCol) {
    VELOX_CHECK(
        overflowCol->type().id() == kInt64Id,
        "Decimal sum state requires INT64 overflow column (type is {})",
        cudf::type_to_name(overflowCol->type()));
    VELOX_CHECK_EQ(
        numRows,
        overflowCol->size(),
        "Decimal sum state requires sum and overflow to be same size (sum size is {}, overflow size is {})",
        sumCol.size(),
        overflowCol->size());
  }
  VELOX_CHECK_LE(
      numRows,
      static_cast<cudf::size_type>(std::numeric_limits<int32_t>::max()),
      "Too many rows to serialize decimal sum state (row count is {})",
      numRows);

  if (numRows == 0) {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

  auto const rowCount = static_cast<int32_t>(numRows);

  auto const charsBytes =
      static_cast<int64_t>(numRows) * detail::kDecimalSumStateSize;
  auto const threshold = cudf::strings::get_offset64_threshold();
  auto const useLargeOffsets = charsBytes >= threshold;
  // Previously this guard threw std::overflow_error; Velox uses
  // VeloxRuntimeError for this guard.
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
      sumType == cudf::type_id::DECIMAL64 || sumType == kDecimal128Id,
      "Unsupported decimal sum column type (type is {})",
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
  // never leave the operator and should use the temporary memory resource.
  auto flat = deserializeDecimalStateImpl(
      stateCol, scale, /*withOverflow=*/false, stream, get_temp_mr());
  DecimalSumStateColumns result;
  result.sum = std::move(flat.sum);
  result.count = std::move(flat.count);
  return result;
}

DecimalStateColumns deserializeDecimalSumStateWithOverflow(
    cudf::column_view const& stateCol,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return deserializeDecimalStateImpl(
      stateCol, scale, /*withOverflow=*/true, stream, mr);
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
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      countCol.type().id() == cudf::type_id::INT64,
      "Decimal average requires INT64 count column (type is {})",
      cudf::type_to_name(countCol.type()));
  VELOX_CHECK(
      sumCol.type().id() == cudf::type_id::DECIMAL64 ||
          sumCol.type().id() == cudf::type_id::DECIMAL128,
      "Decimal average requires DECIMAL64 or DECIMAL128 sum column (type is {})",
      cudf::type_to_name(sumCol.type()));
  VELOX_CHECK_EQ(
      sumCol.size(),
      countCol.size(),
      "Decimal average requires sum and count to be same size (sum size is {}, count size is {})",
      sumCol.size(),
      countCol.size());

  auto numRows = sumCol.size();
  auto out = cudf::make_fixed_width_column(
      sumCol.type(), numRows, cudf::mask_state::UNALLOCATED, stream, mr);

  if (numRows > 0) {
    auto const rowCount = static_cast<int32_t>(numRows);
    const auto sumType = sumCol.type().id();
    detail::averageRoundDecimalSum(
        sumType,
        sumCol,
        countCol.data<int64_t>(),
        out->mutable_view(),
        rowCount,
        stream);
  }

  auto [nullMask, nullCount] =
      detail::buildStateValidityMask(sumCol, countCol, stream, mr);
  if (nullCount > 0) {
    out->set_null_mask(std::move(nullMask), nullCount);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Self-describing decimal aggregate state.
// ---------------------------------------------------------------------------

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

std::unique_ptr<cudf::column> wrapDecimalState(
    DecimalStateColumns&& flat,
    DecimalStateShape shape,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_NOT_NULL(flat.sum, "Decimal state requires sum column");
  auto const sumTypeId = flat.sum->type().id();
  VELOX_CHECK(
      sumTypeId == cudf::type_id::DECIMAL64 || sumTypeId == kDecimal128Id,
      "Decimal state sum must be DECIMAL64 or DECIMAL128 (type is {})",
      cudf::type_to_name(flat.sum->type()));
  if (sumTypeId == cudf::type_id::DECIMAL64) {
    flat.sum = cudf::cast(
        flat.sum->view(),
        cudf::data_type{kDecimal128Id, flat.sum->type().scale()},
        stream,
        mr);
  }
  auto const numRows = flat.sum->size();

  auto takeInt64 = [&](std::unique_ptr<cudf::column>& col, const char* name) {
    VELOX_CHECK_NOT_NULL(
        col,
        "Decimal state shape requires {} column: {}",
        name,
        shapeName(shape));
    VELOX_CHECK_EQ(
        col->size(),
        numRows,
        "Decimal state {} column size must match sum size",
        name);
    if (col->type().id() != kInt64Id) {
      col = cudf::cast(col->view(), cudf::data_type{kInt64Id}, stream, mr);
    }
    return std::move(col);
  };

  auto const layout = layoutOf(shape);
  std::vector<std::unique_ptr<cudf::column>> children(layout.numChildren);
  children[layout.sum] = std::move(flat.sum);
  if (layout.count >= 0) {
    children[layout.count] = takeInt64(flat.count, "count");
  }
  if (layout.overflow >= 0) {
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

bool isDecimalStateStruct(cudf::column_view const& column) {
  return tryDecimalStateShapeOf(column).has_value();
}

DecimalStateShape decimalStateShapeOf(cudf::column_view const& column) {
  auto shape = tryDecimalStateShapeOf(column);
  VELOX_CHECK(
      shape.has_value(),
      "Column is not a decimal aggregate state struct (type is {}, {} children)",
      cudf::type_to_name(column.type()),
      column.num_children());
  return *shape;
}

bool isDecimalStateColumn(cudf::column_view const& column) {
  return column.type().id() == cudf::type_id::STRING ||
      isDecimalStateStruct(column);
}

FlatDecimalState flattenDecimalState(
    cudf::column_view const& state,
    int32_t scale,
    bool needCount,
    bool needOverflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  FlatDecimalState out;
  if (state.type().id() == cudf::type_id::STRING) {
    auto decoded =
        deserializeDecimalStateImpl(state, scale, needOverflow, stream, mr);
    out.sum = decoded.sum->view();
    out.owned.push_back(std::move(decoded.sum));
    if (needCount) {
      out.count = decoded.count->view();
      out.owned.push_back(std::move(decoded.count));
    }
    if (needOverflow) {
      out.overflow = decoded.overflow->view();
      out.owned.push_back(std::move(decoded.overflow));
    }
    return out;
  }

  auto const shape = decimalStateShapeOf(state);
  VELOX_CHECK_EQ(
      state.null_count(),
      0,
      "Decimal state struct must not carry a parent null mask; validity lives on the sum child");
  auto const layout = layoutOf(shape);
  auto const numRows = state.size();

  out.sum = stateChild(state, layout.sum, stream);
  if (layout.count >= 0) {
    out.count = stateChild(state, layout.count, stream);
  } else if (needCount) {
    out.owned.push_back(makeConstantInt64Column(1, numRows, stream, mr));
    out.count = out.owned.back()->view();
  }
  if (layout.overflow >= 0) {
    out.overflow = stateChild(state, layout.overflow, stream);
  } else if (needOverflow) {
    out.owned.push_back(makeConstantInt64Column(0, numRows, stream, mr));
    out.overflow = out.owned.back()->view();
  }
  return out;
}

std::unique_ptr<cudf::column> packDecimalState(
    cudf::column_view const& structColumn,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto const shape = decimalStateShapeOf(structColumn);
  VELOX_CHECK_EQ(
      structColumn.null_count(),
      0,
      "Decimal state struct must not carry a parent null mask; validity lives on the sum child");
  auto const layout = layoutOf(shape);
  auto const sum = stateChild(structColumn, layout.sum, stream);
  std::optional<cudf::column_view> count;
  std::optional<cudf::column_view> overflow;
  if (layout.count >= 0) {
    count = stateChild(structColumn, layout.count, stream);
  }
  if (layout.overflow >= 0) {
    overflow = stateChild(structColumn, layout.overflow, stream);
  }
  return serializeDecimalStateImpl(
      sum,
      count ? &*count : nullptr,
      overflow ? &*overflow : nullptr,
      stream,
      mr);
}

std::unique_ptr<cudf::column> unpackDecimalState(
    cudf::column_view const& blobColumn,
    DecimalStateShape shape,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto flat = deserializeDecimalStateImpl(
      blobColumn, scale, decimalStateHasOverflow(shape), stream, mr);
  if (!decimalStateHasCount(shape)) {
    flat.count.reset();
  }
  return wrapDecimalState(std::move(flat), shape, stream, mr);
}

std::unique_ptr<cudf::column> makeEmptyDecimalStateLike(
    cudf::column_view const& like,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (like.type().id() == cudf::type_id::STRING) {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }
  auto const shape = decimalStateShapeOf(like);
  auto const sumScale = veloxScaleOf(like.child(layoutOf(shape).sum));
  return makeEmptyDecimalStateStruct(shape, sumScale, stream, mr);
}

std::vector<std::unique_ptr<cudf::column>> normalizeDecimalStateBatches(
    std::vector<cudf::column_view>& views,
    int32_t scale,
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
        "Batch {} is not a decimal aggregate state column (type is {})",
        i,
        cudf::type_to_name(views[i].type()));
  }

  // Pick the target form from the non-empty batches; zero-row batches are
  // wildcards. If every batch is empty, fall back to considering all of them
  // so that a mixed set of empties still ends up uniform.
  auto chooseTarget = [&](bool includeEmpty)
      -> std::pair<std::optional<DecimalStateShape>, std::optional<int32_t>> {
    std::optional<DecimalStateShape> target;
    std::optional<int32_t> targetScale;
    bool mixedShapes = false;
    bool sawAny = false;
    for (size_t i = 0; i < views.size(); ++i) {
      if (!includeEmpty && views[i].size() == 0) {
        continue;
      }
      sawAny = true;
      if (!shapes[i]) {
        continue;
      }
      if (!targetScale) {
        targetScale = veloxScaleOf(views[i].child(layoutOf(*shapes[i]).sum));
      }
      if (!target) {
        target = *shapes[i];
      } else if (*target != *shapes[i]) {
        mixedShapes = true;
      }
    }
    if (!sawAny) {
      return {std::nullopt, std::nullopt};
    }
    if (mixedShapes) {
      target = DecimalStateShape::kAvg128;
    }
    return {target, targetScale};
  };

  bool const allEmpty = std::all_of(
      views.begin(), views.end(), [](auto& v) { return v.size() == 0; });
  auto const chosen = chooseTarget(/*includeEmpty=*/allEmpty);
  auto const& target = chosen.first;
  auto const& targetScale = chosen.second;

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
  auto const unpackScale = targetScale.value_or(scale);
  auto const targetLayout = layoutOf(targetShape);

  for (size_t i = 0; i < views.size(); ++i) {
    auto& view = views[i];
    bool const isTargetShape = shapes[i] && *shapes[i] == targetShape;
    if (isTargetShape) {
      continue;
    }
    if (view.size() == 0) {
      owned.push_back(
          makeEmptyDecimalStateStruct(targetShape, unpackScale, stream, mr));
    } else if (!shapes[i]) {
      owned.push_back(
          unpackDecimalState(view, targetShape, unpackScale, stream, mr));
    } else {
      // Widen a struct of another shape to the target (kAvg128). Children
      // must be owned by the new struct, so they are copied here; this path
      // is only reached on version skew between producers.
      auto const srcLayout = layoutOf(*shapes[i]);
      DecimalStateColumns cols;
      cols.sum = std::make_unique<cudf::column>(
          stateChild(view, srcLayout.sum, stream), stream, mr);
      if (targetLayout.count >= 0) {
        cols.count = srcLayout.count >= 0
            ? std::make_unique<cudf::column>(
                  stateChild(view, srcLayout.count, stream), stream, mr)
            : makeConstantInt64Column(1, view.size(), stream, mr);
      }
      if (targetLayout.overflow >= 0) {
        cols.overflow = srcLayout.overflow >= 0
            ? std::make_unique<cudf::column>(
                  stateChild(view, srcLayout.overflow, stream), stream, mr)
            : makeConstantInt64Column(0, view.size(), stream, mr);
      }
      owned.push_back(
          wrapDecimalState(std::move(cols), targetShape, stream, mr));
    }
    view = owned.back()->view();
  }
  return owned;
}

} // namespace facebook::velox::cudf_velox
