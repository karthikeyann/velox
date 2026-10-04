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
#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace facebook::velox::cudf_velox {

void validateIntermediateColumnType(const cudf::column_view& column) {
  VELOX_CHECK(
      isDecimalStateColumn(column),
      "Expected decimal aggregation state (a cuDF STRING blob or a decimal state STRUCT under Velox VARBINARY): {} with {} children",
      cudf::type_to_name(column.type()),
      column.num_children());
}

bool isDecimalStateUnderVarbinary(
    const TypePtr& type,
    const cudf::column_view& column) {
  return type->kind() == TypeKind::VARBINARY &&
      column.type().id() == cudf::type_id::STRUCT;
}

DecimalStateInfo decimalStateInfoFor(
    bool isAverage,
    const core::AggregationNode::Aggregate& aggregate) {
  VELOX_CHECK_EQ(
      aggregate.rawInputTypes.size(),
      1,
      "Decimal aggregate requires exactly one raw input type in the plan: {}",
      aggregate.call->toString());
  const auto& rawInputType = aggregate.rawInputTypes[0];
  VELOX_CHECK(
      rawInputType->isDecimal(),
      "Decimal aggregate requires a DECIMAL raw input: {}",
      rawInputType->toString());
  return DecimalStateInfo{
      decimalStateShapeFor(isAverage, rawInputType->isLongDecimal()),
      getDecimalPrecisionScale(*rawInputType).second};
}

cudf::column_view castDecimal64InputToDecimal128(
    const cudf::column_view& inputCol,
    std::unique_ptr<cudf::column>& holder,
    cuda::stream_ref stream) {
  if (inputCol.type().id() != cudf::type_id::DECIMAL64) {
    return inputCol;
  }
  holder = cudf::cast(
      inputCol,
      cudf::data_type{cudf::type_id::DECIMAL128, inputCol.type().scale()},
      stream,
      get_temp_mr());
  return holder->view();
}

namespace {

// Casts an INT64 state field (count or overflow) that arrived in another
// integer type. The result is consumed internally, so it uses the temporary
// memory resource.
std::unique_ptr<cudf::column> castToInt64(
    std::unique_ptr<cudf::column> column,
    cuda::stream_ref stream) {
  if (column->type().id() != cudf::type_id::INT64) {
    column = cudf::cast(
        *column, cudf::data_type{cudf::type_id::INT64}, stream, get_temp_mr());
  }
  return column;
}

// Raises the CPU's "Decimal overflow" user error for a device check outcome.
void raiseDecimalSumCheck(detail::DecimalSumCheck check) {
  switch (check) {
    case detail::DecimalSumCheck::kOk:
      return;
    case detail::DecimalSumCheck::kOutOfRange:
      VELOX_USER_FAIL(
          "Decimal overflow. Sum is not in the range of Decimal Type");
  }
  VELOX_UNREACHABLE();
}

} // namespace

std::unique_ptr<cudf::column> finalizeDecimalSum(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> overflow,
    const TypePtr& resultType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(
      sum->type().id() == cudf::type_id::DECIMAL128,
      "Decimal sum result requires a DECIMAL128 column: {}",
      cudf::type_to_name(sum->type()));
  if (overflow) {
    overflow = castToInt64(std::move(overflow), stream);
  }
  raiseDecimalSumCheck(
      overflow ? detail::foldDecimalSumOverflow(
                     sum->mutable_view(), overflow->view(), stream)
               : detail::checkDecimalSumRange(sum->view(), stream));
  auto const cudfOutType = veloxToCudfDataType(resultType);
  if (sum->type() != cudfOutType) {
    return cudf::cast(sum->view(), cudfOutType, stream, mr);
  }
  return sum;
}

std::unique_ptr<cudf::column> finalizeDecimalAverage(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    std::unique_ptr<cudf::column> overflow,
    const TypePtr& resultType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  count = castToInt64(std::move(count), stream);
  if (overflow) {
    overflow = castToInt64(std::move(overflow), stream);
  }
  auto average = computeDecimalAverage(
      sum->view(),
      count->view(),
      overflow ? overflow->view() : cudf::column_view{},
      stream,
      mr);
  auto const cudfOutType = veloxToCudfDataType(resultType);
  if (average->type() != cudfOutType) {
    average = cudf::cast(average->view(), cudfOutType, stream, mr);
  }
  return average;
}

} // namespace facebook::velox::cudf_velox
