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

#include "velox/experimental/cudf/exec/DecimalAggregationState.h"

#include "velox/core/PlanNode.h"
#include "velox/type/Type.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <cuda/stream>

#include <memory>

namespace facebook::velox::cudf_velox {

/// Asserts that a column holds decimal aggregate state under a Velox VARBINARY
/// logical type: either the cuDF STRING blob or a self-describing decimal
/// state STRUCT (see DecimalAggregationState.h). The blob does not carry
/// scale, so it is decoded at the scale the plan provides; a struct carries
/// its scale on the sum child.
void validateIntermediateColumnType(const cudf::column_view& column);

/// True if a column whose Velox type is `type` is physically an unpacked
/// decimal aggregate state struct, i.e. the logical type is VARBINARY and the
/// cuDF column is a STRUCT. Operators that would read the bytes of such a
/// column (as a key, in an expression, through Arrow) use this to guard or to
/// pack first.
bool isDecimalStateUnderVarbinary(
    const TypePtr& type,
    const cudf::column_view& column);

/// Resolves the state shape and sum scale of a decimal SUM or AVG aggregate
/// from `aggregate.rawInputTypes[0]`. Throws unless the plan carries exactly
/// one raw input type and it is a DECIMAL. The operators only classify an
/// aggregate as a decimal SUM/AVG when that holds (resolveAggregateInfos).
DecimalStateInfo decimalStateInfoFor(
    bool isAverage,
    const core::AggregationNode::Aggregate& aggregate);

/// Casts a DECIMAL64 column up to DECIMAL128 (scale preserved) so a subsequent
/// SUM accumulates in 128 bits instead of wrapping. Allocates the casted
/// column from the temporary memory resource into `holder` and returns its
/// view, valid only while `holder` is alive. Returns `inputCol` unchanged
/// when it is not DECIMAL64.
cudf::column_view castDecimal64InputToDecimal128(
    const cudf::column_view& inputCol,
    std::unique_ptr<cudf::column>& holder,
    cuda::stream_ref stream);

/// Casts `column` to the cuDF type of the Velox `type` when the two differ and
/// moves it through unchanged otherwise.
std::unique_ptr<cudf::column> castToVeloxType(
    std::unique_ptr<cudf::column> column,
    const TypePtr& type,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// FINAL step for decimal SUM, matching the CPU DecimalSumAggregate: folds the
/// merged `overflow` field into `sum` as sum + overflow * 2^127 modulo 2^128
/// (the exact total of a state produced by DecimalUtil::addWithOverflow,
/// applied without adjustSumForOverflow's sign predicate because a GPU merge
/// wraps the sum children; see foldDecimalSumOverflow), raises the CPU's
/// "Decimal overflow" user error where the folded sum leaves the DECIMAL(38)
/// range, and casts to `resultType`. A nullptr `overflow` means the field was
/// not tracked and is zero. One device pass and one host sync for the
/// validation.
std::unique_ptr<cudf::column> finalizeDecimalSum(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> overflow,
    const TypePtr& resultType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// FINAL step for decimal AVG, following the CPU DecimalAverageAggregateBase:
/// normalizes `count` and `overflow` to INT64, divides with the rounding of
/// DecimalUtil::computeAverage for a canonical nonzero `overflow` and with the
/// folded total otherwise (nullptr means the field was not tracked), then
/// casts to `resultType`. Like the CPU, no range check is applied to the
/// average. The result equals the CPU's except that when the merged pair is
/// non-canonical it may differ by one unit in the last place, because the
/// CPU's own result depends on accumulation order in exact-half cases.
std::unique_ptr<cudf::column> finalizeDecimalAverage(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    std::unique_ptr<cudf::column> overflow,
    const TypePtr& resultType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// FINAL step over the merged flat fields of a decimal state of `shape`:
/// finalizeDecimalSum for the SUM shapes (a count, if present, is ignored) and
/// finalizeDecimalAverage for the AVG shapes. `count` and `overflow` may be
/// null pointers when the state did not carry them. Shared by CudfGroupby
/// (group-by FINAL and direct finalization) and CudfReduce.
std::unique_ptr<cudf::column> finalizeDecimalState(
    DecimalStateColumns state,
    DecimalStateShape shape,
    const TypePtr& resultType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
