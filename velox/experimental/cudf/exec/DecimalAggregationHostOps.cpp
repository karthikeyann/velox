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
#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace facebook::velox::cudf_velox {

void validateIntermediateColumnType(cudf::column_view const& column) {
  if (isDecimalStateColumn(column)) {
    return;
  }
  VELOX_FAIL(
      "Expected decimal aggregation state: Velox VARBINARY represented as a cuDF STRING blob or a decimal state STRUCT ([sum], [overflow, sum], [sum, count] or [sum, count, overflow]); got {} with {} children",
      cudf::type_to_name(column.type()),
      column.num_children());
}

cudf::column_view castDecimal64InputToDecimal128(
    cudf::column_view inputCol,
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

std::unique_ptr<cudf::column> castCountColumnToInt64(
    std::unique_ptr<cudf::column> count,
    cuda::stream_ref stream) {
  if (count->type().id() != cudf::type_id::INT64) {
    count = cudf::cast(
        *count, cudf::data_type{cudf::type_id::INT64}, stream, get_temp_mr());
  }
  return count;
}

std::unique_ptr<cudf::column> serializeDecimalPartialOrIntermediateState(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  count = castCountColumnToInt64(std::move(count), stream);
  return serializeDecimalSumState(sum->view(), count->view(), stream, mr);
}

std::unique_ptr<cudf::column> finalizeDecimalAverage(
    std::unique_ptr<cudf::column> sum,
    std::unique_ptr<cudf::column> count,
    const TypePtr& resultType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  count = castCountColumnToInt64(std::move(count), stream);
  auto avgCol = computeDecimalAverage(sum->view(), count->view(), stream, mr);
  auto const cudfOutType = veloxToCudfDataType(resultType);
  if (avgCol->type() != cudfOutType) {
    avgCol = cudf::cast(avgCol->view(), cudfOutType, stream, mr);
  }
  return avgCol;
}

} // namespace facebook::velox::cudf_velox
