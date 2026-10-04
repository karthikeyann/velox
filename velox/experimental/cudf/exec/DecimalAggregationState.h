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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <cuda/stream>

#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

struct DecimalSumStateColumns {
  std::unique_ptr<cudf::column> sum;
  std::unique_ptr<cudf::column> count;
};

/** Directly reduces DECIMAL64 input into one DECIMAL128 sum and INT64 count. */
DecimalSumStateColumns reduceDecimal64SumCount(
    const cudf::column_view& input,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Decodes intermediate decimal SUM aggregate state stored as a cuDF STRING
 * column (fixed-size packed bytes per row, converted from Velox VARBINARY) into
 * two device columns: a DECIMAL128 sum with scale -scale (matching Velox
 * intermediate state) and an INT64 partial row count. Handles empty input,
 * all-null state without touching payload buffers, and propagates the source
 * null mask to both outputs when present.
 *
 * @param stateCol STRING column of packed sum/count payloads.
 * @param scale decimal scale used to set the output sum column's scale to
 *        -scale.
 * @param stream CUDA stream for device work.
 * @return decoded sum and count columns.
 */
DecimalSumStateColumns deserializeDecimalSumState(
    const cudf::column_view& stateCol,
    int32_t scale,
    cuda::stream_ref stream);

/**
 * Encodes partial decimal SUM state (DECIMAL64 or DECIMAL128 sums plus INT64
 * counts) into a single STRING column (later converted to Velox VARBINARY):
 * per-row fixed-width payloads and string offsets (INT32 or INT64 depending on
 * total char size and cuDF large-strings settings). The output null mask
 * matches buildStateValidityMask: a row is invalid if the sum or count is null,
 * or the count is zero.
 *
 * @param sumCol per-row partial sums (DECIMAL64 or DECIMAL128).
 * @param countCol per-row INT64 partial row counts.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for allocated columns.
 * @return STRING column of serialized state.
 */
std::unique_ptr<cudf::column> serializeDecimalSumState(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/**
 * Finalizes AVG from intermediate SUM state: divides each sum by its count on
 * device with decimal-specific rounding (see averageRoundDecimalSum),
 * producing a column of the same decimal type as the sum. Rows are null where
 * buildStateValidityMask marks them invalid (null sum/count or zero count),
 * matching serializeDecimalSumState.
 *
 * @param sumCol per-row partial sums (DECIMAL64 or DECIMAL128).
 * @param countCol per-row INT64 partial row counts.
 * @param stream CUDA stream for device work.
 * @param mr memory resource for allocated columns.
 * @return per-row decimal average column.
 */
std::unique_ptr<cudf::column> computeDecimalAverage(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);


// ---------------------------------------------------------------------------
// Self-describing decimal aggregate state.
//
// A decimal SUM/AVG aggregate state whose logical Velox type is VARBINARY may
// be carried on the GPU in one of two physical forms:
//
//   * the 32-byte STRING blob (count int64, overflow int64, sum low uint64,
//     sum high int64) that CPU Velox uses and that CudfFromVelox produces, or
//   * a cuDF STRUCT whose children are a subset of {sum DECIMAL128,
//     count INT64, overflow INT64}, chosen by the producer from two facts it
//     knows locally: whether the aggregate is AVG and whether the raw input is
//     DECIMAL64 or DECIMAL128.
//
// Struct shapes (child order matters; the shape is recoverable from the child
// types alone, which is what makes the column self-describing):
//
//   kSum64  : [sum]                          fields missing: count=1, ovf=0
//   kSum128 : [overflow, sum]                fields missing: count=1
//   kAvg64  : [sum, count]                   fields missing: ovf=0
//   kAvg128 : [sum, count, overflow]         nothing missing
//
// The DECIMAL128 child at index 1 means "overflow precedes it" (kSum128); at
// index 0 the children that follow are count and then overflow. The struct
// parent never carries a null mask: a state row is null exactly when `sum` is
// null. See docs/designs/cudf-self-describing-decimal-aggregate-state.md.
// ---------------------------------------------------------------------------

enum class DecimalStateShape : uint8_t {
  kSum64, // [sum]
  kSum128, // [overflow, sum]
  kAvg64, // [sum, count]
  kAvg128, // [sum, count, overflow]
};

/// True if the shape carries a count child.
constexpr bool decimalStateHasCount(DecimalStateShape shape) {
  return shape == DecimalStateShape::kAvg64 ||
      shape == DecimalStateShape::kAvg128;
}

/// True if the shape carries an overflow child.
constexpr bool decimalStateHasOverflow(DecimalStateShape shape) {
  return shape == DecimalStateShape::kSum128 ||
      shape == DecimalStateShape::kAvg128;
}

/// Shape a producer must emit for an aggregate. `isAverage` distinguishes
/// AVG from SUM; `rawInputIsDecimal128` is true when the aggregate's raw
/// (pre-aggregation) input type is a long decimal (precision > 18).
DecimalStateShape decimalStateShapeFor(
    bool isAverage,
    bool rawInputIsDecimal128);

/// Flat (un-wrapped) state columns. `count` and `overflow` may be null
/// pointers when the shape does not carry them.
struct DecimalStateColumns {
  std::unique_ptr<cudf::column> sum; // DECIMAL128, never null
  std::unique_ptr<cudf::column> count; // INT64 or nullptr
  std::unique_ptr<cudf::column> overflow; // INT64 or nullptr
};

/// Producer side. Moves the flat columns into a STRUCT of the given shape.
/// Children the shape omits are dropped; children the shape requires must be
/// present (VELOX_CHECK). Casts `sum` to DECIMAL128 (keeping scale) and
/// `count`/`overflow` to INT64 if needed. The struct is built with
/// cudf::make_structs_column and NO parent null mask, so no child is copied.
std::unique_ptr<cudf::column> wrapDecimalState(
    DecimalStateColumns&& flat,
    DecimalStateShape shape,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// True if `column` is a STRUCT in one of the four documented shapes.
/// False for STRING and for any other type. Never throws.
bool isDecimalStateStruct(cudf::column_view const& column);

/// Returns the shape of a struct accepted by isDecimalStateStruct. Throws
/// VeloxRuntimeError otherwise.
DecimalStateShape decimalStateShapeOf(cudf::column_view const& column);

/// True if `column` is either the STRING blob or a decimal state struct,
/// i.e. anything flattenDecimalState accepts.
bool isDecimalStateColumn(cudf::column_view const& column);

/// Consumer side. Non-owning views over the three state fields plus the
/// storage that keeps any decoded or synthesized column alive. Views into a
/// struct input alias that input's children (no copy); views for fields the
/// input did not carry, and all views for a STRING input, point into `owned`.
struct FlatDecimalState {
  cudf::column_view sum; // DECIMAL128 with the state's scale
  cudf::column_view count; // INT64; synthesized 1s if absent
  cudf::column_view overflow; // INT64; synthesized 0s if absent
  std::vector<std::unique_ptr<cudf::column>> owned;
};

/// Flattens a STRING blob (via deserializeDecimalSumState) or any struct shape
/// into flat fields. `scale` is the decimal scale of the sum and is used only
/// for STRING input (a struct already carries it on the sum child). Missing
/// fields are synthesized (count=1, overflow=0) only when the corresponding
/// `need*` flag is true; otherwise that view is left default-constructed
/// (size 0) so callers that never read count or overflow pay nothing.
FlatDecimalState flattenDecimalState(
    cudf::column_view const& state,
    int32_t scale,
    bool needCount,
    bool needOverflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Decodes a STRING blob into sum, count AND overflow (deserializeDecimalSumState
/// drops the overflow field). All three outputs carry a copy of the blob's
/// null mask when it has one. Allocates from `mr`.
DecimalStateColumns deserializeDecimalSumStateWithOverflow(
    cudf::column_view const& stateCol,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Packs a decimal state struct of any shape into the 32-byte STRING blob
/// (count/overflow filled in as 1/0 when the shape lacks them). Used by
/// CudfToVelox before Arrow export.
///
/// Validity: when the shape has no count child the output null mask is a copy
/// of the `sum` child's mask. When it has one, the output follows the same
/// rule as serializeDecimalSumState (buildStateValidityMask): a row is null if
/// `sum` or `count` is null or `count` is zero. Both rules agree for every
/// struct a producer emits (count is non-null and non-zero exactly where sum
/// is non-null), and the second keeps packDecimalState byte- and mask-equal to
/// serializeDecimalSumState for the same sum/count inputs.
std::unique_ptr<cudf::column> packDecimalState(
    cudf::column_view const& structColumn,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Unpacks a STRING blob into a struct of the requested shape (used when a
/// blob batch must be concatenated with struct batches). `scale` is the
/// decimal scale of the sum.
std::unique_ptr<cudf::column> unpackDecimalState(
    cudf::column_view const& blobColumn,
    DecimalStateShape shape,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Builds a zero-row column in the physical form of `like` (STRING or any
/// struct shape). Used to re-type empty batches before concatenation.
std::unique_ptr<cudf::column> makeEmptyDecimalStateLike(
    cudf::column_view const& like,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Concat-funnel normalization for one logical VARBINARY column across
/// batches. On return every view in `views` has the same physical type:
///   * all STRING                       -> unchanged, returns empty vector;
///   * all one struct shape             -> unchanged, returns empty vector;
///   * STRING mixed with struct         -> STRING batches are unpacked to the
///                                         struct shape (toward the smaller
///                                         form, never toward the blob);
///   * several struct shapes            -> all widened to kAvg128;
///   * zero-row batches of either form  -> re-typed to the chosen form.
/// Replacement columns are returned and the corresponding entries of `views`
/// are rebound to them; the caller must keep the returned vector alive until
/// after cudf::concatenate. Struct batches carry their scale on the `sum`
/// child and that scale is the one blobs are unpacked to (blobs carry none);
/// `scale` is used only when no struct batch is present (which makes the call
/// a no-op), so callers that decode VARBINARY at scale 0 are still correct.
std::vector<std::unique_ptr<cudf::column>> normalizeDecimalStateBatches(
    std::vector<cudf::column_view>& views,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
