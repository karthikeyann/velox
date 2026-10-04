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

#include "velox/common/EnumDeclare.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// A decimal SUM/AVG aggregate state whose logical Velox type is VARBINARY
/// may be carried on the GPU in one of two physical forms:
///
///   * the 32-byte STRING blob (count int64, overflow int64, sum low uint64,
///     sum high int64) that CPU Velox uses and that CudfFromVelox produces, or
///   * a cuDF STRUCT whose children are a subset of {sum DECIMAL128,
///     count INT64, overflow INT64}, chosen by the producer from two facts it
///     knows locally: whether the aggregate is AVG and whether the raw input
///     is DECIMAL64 or DECIMAL128.
///
/// Struct shapes (child order matters; the shape is recoverable from the
/// child types alone, which is what makes the column self-describing):
///
///   kSum64  : [sum]                          fields missing: count=1, ovf=0
///   kSum128 : [overflow, sum]                fields missing: count=1
///   kAvg64  : [sum, count]                   fields missing: ovf=0
///   kAvg128 : [sum, count, overflow]         nothing missing
///
/// The struct parent carries no null mask when a producer emits it: a state
/// row is null exactly when `sum` is null. flattenDecimalState and
/// packDecimalState nevertheless tolerate a parent mask (for example after a
/// nullifying gather) and fold it into every child they expose, so a row the
/// parent marks null is null in sum, count and overflow alike.
/// unwrapDecimalState does not: it only sees structs the group-by buffered
/// itself.
///
/// Known limitation: GPU producers do not track int128 carries (cuDF's
/// DECIMAL128 SUM wraps modulo 2^128), so the overflow child they emit is
/// always 0. A nonzero overflow can only come from a CPU-produced blob. On
/// merge the overflow children are summed exactly while the sum children wrap
/// modulo 2^128, so the merged pair is congruent to the true total modulo
/// 2^128 but is not in the CPU's canonical form; FINAL therefore folds it as
/// sum + overflow * 2^127 modulo 2^128 (see finalizeDecimalSum and
/// finalizeDecimalAverage in DecimalAggregationHostOps.h). GPU SUM then equals
/// CPU SUM for every total inside int128. GPU AVG equals CPU AVG except that
/// when the merged pair is non-canonical the result may differ from the CPU's
/// by one unit in the last place, because the CPU's own result depends on
/// accumulation order in exact-half cases.
enum class DecimalStateShape : uint8_t {
  kSum64,
  kSum128,
  kAvg64,
  kAvg128,
};

VELOX_DECLARE_ENUM_NAME(DecimalStateShape);

/// True if the shape carries a count child.
bool decimalStateHasCount(DecimalStateShape shape);

/// True if the shape carries an overflow child.
bool decimalStateHasOverflow(DecimalStateShape shape);

/// Shape a producer must emit for an aggregate. `isAverage` distinguishes
/// AVG from SUM; `rawInputIsDecimal128` is true when the aggregate's raw
/// (pre-aggregation) input type is a long decimal (precision > 18).
DecimalStateShape decimalStateShapeFor(
    bool isAverage,
    bool rawInputIsDecimal128);

/// Describes a decimal SUM/AVG aggregate whose intermediate column (logical
/// VARBINARY) is carried on the GPU as a self-describing state: the
/// plan-determined struct shape the aggregate emits and the decimal scale of
/// the state's sum (the raw input scale), used to decode a scale-less STRING
/// blob.
struct DecimalStateInfo {
  DecimalStateShape shape;
  int32_t scale;
};

/// Owned flat state columns. `count` and `overflow` are null pointers when
/// the state does not carry them.
struct DecimalStateColumns {
  std::unique_ptr<cudf::column> sum; // DECIMAL128 (DECIMAL64 before wrapping)
  std::unique_ptr<cudf::column> count; // INT64 or nullptr
  std::unique_ptr<cudf::column> overflow; // INT64 or nullptr
};

/// Kept for the sum/count producers and the CPU-interchange codec below.
using DecimalSumStateColumns = DecimalStateColumns;

/// Directly reduces DECIMAL64 input into one DECIMAL128 sum and INT64 count.
DecimalSumStateColumns reduceDecimal64SumCount(
    const cudf::column_view& input,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// CPU-interchange codec, blob to flat columns. Decodes the 32-byte STRING
/// blob into a DECIMAL128 sum with scale -scale and an INT64 count allocated
/// from the temporary memory resource (the overflow field is dropped).
/// Handles empty input, all-null input without touching payload buffers, and
/// propagates the blob's null mask to both outputs. Production code goes
/// through flattenDecimalState; this entry point is kept for tests that pin
/// the blob layout.
DecimalSumStateColumns deserializeDecimalSumState(
    const cudf::column_view& stateCol,
    int32_t scale,
    cuda::stream_ref stream);

/// CPU-interchange codec, flat columns to blob. Encodes per-row sums
/// (DECIMAL64 or DECIMAL128) and INT64 counts into the 32-byte STRING blob
/// with overflow 0. A row is null if the sum or count is null or the count is
/// zero (buildStateValidityMask). Production code goes through
/// packDecimalState; this entry point is kept for tests that pin the blob
/// layout.
std::unique_ptr<cudf::column> serializeDecimalSumState(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Finalizes AVG from flat state: divides each sum by its count with the
/// CPU's rounding (DecimalUtil::computeAverage for a canonical nonzero
/// `overflow`, the folded total otherwise; see detail::averageRoundDecimalSum)
/// and produces a column of the sum's decimal type. Rows are null where the
/// sum or count is null or the count is zero. `overflow` must be a
/// default-constructed (size 0) view when the state does not carry the field.
std::unique_ptr<cudf::column> computeDecimalAverage(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    const cudf::column_view& overflowCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Producer side. Moves the flat columns into a STRUCT of the given shape.
/// Children the shape omits are dropped. A missing count is an error; a
/// missing overflow is synthesized as zeros (GPU producers never track
/// carries). Casts `sum` to DECIMAL128 (keeping scale) and `count`/`overflow`
/// to INT64 if needed. The struct is built with cudf::make_structs_column and
/// no parent null mask, so no child is copied.
std::unique_ptr<cudf::column> wrapDecimalState(
    DecimalStateColumns flat,
    DecimalStateShape shape,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Inverse of wrapDecimalState for an owned struct: releases the children and
/// places them by layout without copying. The struct must not carry a parent
/// null mask (producers never emit one).
DecimalStateColumns unwrapDecimalState(
    std::unique_ptr<cudf::column> structColumn);

/// True if `column` is a STRUCT in one of the four documented shapes.
/// False for STRING and for any other type. Never throws.
bool isDecimalStateStruct(const cudf::column_view& column);

/// Returns the shape of a struct accepted by isDecimalStateStruct. Throws
/// VeloxRuntimeError otherwise.
DecimalStateShape decimalStateShapeOf(const cudf::column_view& column);

/// True if `column` is either the STRING blob or a decimal state struct,
/// i.e. anything flattenDecimalState accepts.
bool isDecimalStateColumn(const cudf::column_view& column);

/// Consumer side. Non-owning views over the three state fields plus the
/// storage that keeps any decoded or synthesized column alive. Views into a
/// struct input alias that input's children (no copy); views for fields the
/// input did not carry, and all views for a STRING input, point into `owned`.
struct FlatDecimalState {
  cudf::column_view sum; // DECIMAL128 with the state's scale
  cudf::column_view count; // INT64; synthesized 1s if absent and requested
  cudf::column_view overflow; // INT64; synthesized 0s if absent and requested
  // Storage behind the views above when they do not alias the input: every
  // field for a STRING input, synthesized fields for a struct input, and
  // every carried field when a struct parent carried a null mask.
  DecimalStateColumns owned;
};

/// Flattens a STRING blob or any struct shape into flat fields. `scale` is the
/// decimal scale of the sum and is used only for STRING input (a struct
/// already carries it on the sum child). Missing fields are synthesized
/// (count=1, overflow=0) only when the corresponding `need*` flag is true;
/// otherwise that view is left default-constructed (size 0) so callers that
/// never read count or overflow pay nothing. Fields a struct carries are
/// always exposed. Columns are allocated from `mr`.
FlatDecimalState flattenDecimalState(
    const cudf::column_view& state,
    int32_t scale,
    bool needCount,
    bool needOverflow,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Unpacks a STRING blob into a struct of the requested shape; the inverse of
/// packDecimalState. `scale` is the decimal scale of the sum. The only
/// production caller is normalizeDecimalStateBatches (a blob batch meeting
/// struct batches); it is public so the blob-to-struct direction can be
/// tested per shape independently of the batch normalization policy.
std::unique_ptr<cudf::column> unpackDecimalState(
    const cudf::column_view& blobColumn,
    DecimalStateShape shape,
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
    const cudf::column_view& structColumn,
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
/// Zero-row batches do not take part in choosing the target form unless every
/// batch is empty. Blobs are unpacked at the scale carried by the struct
/// batches' sum child. Replacement columns are returned and the corresponding
/// entries of `views` are rebound to them; the caller must keep the returned
/// vector alive until after cudf::concatenate.
std::vector<std::unique_ptr<cudf::column>> normalizeDecimalStateBatches(
    std::vector<cudf::column_view>& views,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
