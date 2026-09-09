/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "velox/experimental/cudf/exec/TopKSortKeys.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>

namespace facebook::velox::cudf_velox {
namespace {

__global__ void
encodeDoubleKeys(const double* input, uint64_t* output, cudf::size_type size) {
  constexpr uint64_t sign = uint64_t{1} << 63;
  constexpr uint64_t infinity = 0x7ff0000000000000ULL;
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < size;
       i += int64_t{blockDim.x} * gridDim.x) {
    auto bits = static_cast<uint64_t>(__double_as_longlong(input[i]));
    auto magnitude = bits & ~sign;
    if (magnitude > infinity) {
      output[i] = ~uint64_t{0};
    } else {
      if (magnitude == 0) {
        bits = 0;
      }
      output[i] = (bits & sign) ? ~bits : bits ^ sign;
    }
  }
}

} // namespace

std::unique_ptr<cudf::column> makeDoubleTopKSortKeys(
    cudf::column_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      input.type().id() == cudf::type_id::FLOAT64 && input.null_count() == 0,
      "Expected non-null DOUBLE sort keys");
  auto output = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::UINT64},
      input.size(),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (input.size() != 0) {
    auto blocks = static_cast<int>(
        std::min<int64_t>((int64_t{input.size()} + 255) / 256, 4096));
    encodeDoubleKeys<<<blocks, 256, 0, stream.value()>>>(
        input.data<double>(),
        output->mutable_view().data<uint64_t>(),
        input.size());
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  return output;
}

} // namespace facebook::velox::cudf_velox
