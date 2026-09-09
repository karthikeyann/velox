/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/experimental/cudf/exec/ShortStringGroupKeys.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace facebook::velox::cudf_velox {
namespace {

__global__ void
packKeys(cudf::table_device_view keys, uint64_t* output, int* invalid) {
  for (int64_t row = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
       row < keys.num_rows();
       row += int64_t{blockDim.x} * gridDim.x) {
    uint64_t packed = 0;
    for (int key = 0; key < keys.num_columns(); ++key) {
      auto string = keys.column(key).element<cudf::string_view>(row);
      auto length = string.size_bytes();
      if (length > 3) {
        atomicExch(invalid, 1);
        continue;
      }
      uint64_t field = static_cast<uint64_t>(length) << 24;
      for (int i = 0; i < length; ++i) {
        field |= uint64_t{static_cast<unsigned char>(string.data()[i])}
            << (8 * i);
      }
      packed |= field << (32 * key);
    }
    output[row] = packed;
  }
}

using StringPair = cuda::std::pair<const char*, cudf::size_type>;

__global__ void makeStringPairs(
    const uint64_t* packed,
    cudf::size_type size,
    int key,
    StringPair* pairs) {
  for (int64_t row = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; row < size;
       row += int64_t{blockDim.x} * gridDim.x) {
    // CUDA devices are little-endian. The first three bytes of each 32-bit
    // field contain the original bytes, including embedded NUL and UTF-8.
    pairs[row] = {
        reinterpret_cast<const char*>(packed + row) + 4 * key,
        static_cast<cudf::size_type>((packed[row] >> (24 + 32 * key)) & 3)};
  }
}

int gridSize(cudf::size_type rows) {
  return static_cast<int>(std::min<int64_t>((int64_t{rows} + 255) / 256, 4096));
}

} // namespace

std::unique_ptr<cudf::column> tryPackShortStringKeys(
    cudf::table_view keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (keys.num_rows() == 0 || keys.num_columns() < 1 ||
      keys.num_columns() > 2) {
    return nullptr;
  }
  for (const auto& key : keys) {
    if (key.type().id() != cudf::type_id::STRING || key.null_count() != 0) {
      return nullptr;
    }
  }
  auto packed = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64},
      keys.num_rows(),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto deviceKeys = cudf::table_device_view::create(keys, stream);
  rmm::device_scalar<int> invalid(0, stream, mr);
  packKeys<<<gridSize(keys.num_rows()), 256, 0, stream.value()>>>(
      *deviceKeys,
      reinterpret_cast<uint64_t*>(packed->mutable_view().data<int64_t>()),
      invalid.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  if (invalid.value(stream)) {
    return nullptr;
  }
  return packed;
}

std::vector<std::unique_ptr<cudf::column>> unpackShortStringKeys(
    cudf::column_view packed,
    int numKeys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      numKeys >= 1 && numKeys <= 2, "Expected one or two packed string keys");
  std::vector<rmm::device_uvector<StringPair>> pairs;
  std::vector<cudf::device_span<const StringPair>> inputs;
  pairs.reserve(numKeys);
  inputs.reserve(numKeys);
  for (int key = 0; key < numKeys; ++key) {
    pairs.emplace_back(packed.size(), stream, mr);
    if (packed.size() != 0) {
      makeStringPairs<<<gridSize(packed.size()), 256, 0, stream.value()>>>(
          reinterpret_cast<const uint64_t*>(packed.data<int64_t>()),
          packed.size(),
          key,
          pairs.back().data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    inputs.emplace_back(pairs.back().data(), pairs.back().size());
  }
  return cudf::make_strings_column_batch(inputs, stream, mr);
}

} // namespace facebook::velox::cudf_velox
