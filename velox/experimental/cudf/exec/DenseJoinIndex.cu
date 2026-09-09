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

#include "velox/experimental/cudf/exec/DenseJoinIndex.h"

#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <limits>

namespace facebook::velox::cudf_velox {
namespace {

template <typename Key>
__global__ void buildIndex(
    const Key* keys,
    cudf::size_type size,
    int64_t minimum,
    cudf::size_type* rows) {
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < size;
       i += int64_t{blockDim.x} * gridDim.x) {
    // Uniqueness was checked exactly before this kernel: no write races.
    rows[static_cast<int64_t>(keys[i]) - minimum] = i;
  }
}

template <typename Key>
__global__ void lookupIndex(
    const Key* keys,
    const cudf::bitmask_type* nullMask,
    cudf::size_type offset,
    cudf::size_type size,
    int64_t minimum,
    int64_t range,
    const cudf::size_type* index,
    cudf::size_type* matches) {
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < size;
       i += int64_t{blockDim.x} * gridDim.x) {
    if (nullMask && !cudf::bit_is_set(nullMask, i + offset)) {
      matches[i] = -1;
      continue;
    }
    auto const key = static_cast<int64_t>(keys[i]);
    // Test bounds before subtraction: INT64_MIN probes must not overflow.
    bool valid = key >= minimum && key <= minimum + range - 1;
    matches[i] = valid ? index[key - minimum] : -1;
  }
}

struct IsMatch {
  __device__ bool operator()(cudf::size_type row) const {
    return row >= 0;
  }
};

int gridSize(cudf::size_type rows) {
  return static_cast<int>(std::min<int64_t>((int64_t{rows} + 255) / 256, 4096));
}

} // namespace

std::unique_ptr<DenseJoinIndex> tryMakeDenseJoinIndex(
    cudf::column_view keys,
    uint64_t maxRange,
    double minDensity,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (keys.size() == 0 || keys.null_count() != 0 ||
      (keys.type().id() != cudf::type_id::INT32 &&
       keys.type().id() != cudf::type_id::INT64)) {
    return nullptr;
  }
  auto [low, high] = cudf::minmax(keys, stream, mr);
  auto scalarValue = [&](const cudf::scalar& scalar) -> int64_t {
    return keys.type().id() == cudf::type_id::INT32
        ? static_cast<const cudf::numeric_scalar<int32_t>&>(scalar).value(
              stream)
        : static_cast<const cudf::numeric_scalar<int64_t>&>(scalar).value(
              stream);
  };
  const auto minimum = scalarValue(*low);
  const auto maximum = scalarValue(*high);
  // Keep range arithmetic bounded independently of arbitrary INT64 inputs.
  if (minimum < 0 || maximum > std::numeric_limits<int32_t>::max()) {
    return nullptr;
  }
  const auto range = static_cast<uint64_t>(maximum - minimum) + 1;
  if (range > maxRange ||
      static_cast<double>(keys.size()) / range < minDensity) {
    return nullptr;
  }
  auto result = std::make_unique<DenseJoinIndex>(minimum, range, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(
      result->rows.data(),
      0xff,
      range * sizeof(cudf::size_type),
      stream.value()));
  if (keys.type().id() == cudf::type_id::INT32) {
    buildIndex<<<gridSize(keys.size()), 256, 0, stream.value()>>>(
        keys.data<int32_t>(), keys.size(), minimum, result->rows.data());
  } else {
    buildIndex<<<gridSize(keys.size()), 256, 0, stream.value()>>>(
        keys.data<int64_t>(), keys.size(), minimum, result->rows.data());
  }
  CUDF_CUDA_TRY(cudaGetLastError());
  return result;
}

std::pair<
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
probeDenseJoinIndex(
    const DenseJoinIndex& index,
    cudf::column_view keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto left = std::make_unique<rmm::device_uvector<cudf::size_type>>(
      keys.size(), stream, mr);
  auto right = std::make_unique<rmm::device_uvector<cudf::size_type>>(
      keys.size(), stream, mr);
  if (keys.size() == 0) {
    return {std::move(left), std::move(right)};
  }
  rmm::device_uvector<cudf::size_type> matches(keys.size(), stream, mr);
  if (keys.type().id() == cudf::type_id::INT32) {
    lookupIndex<<<gridSize(keys.size()), 256, 0, stream.value()>>>(
        keys.data<int32_t>(),
        keys.null_mask(),
        keys.offset(),
        keys.size(),
        index.minimum,
        index.rows.size(),
        index.rows.data(),
        matches.data());
  } else {
    CUDF_EXPECTS(
        keys.type().id() == cudf::type_id::INT64,
        "Expected integer probe keys");
    lookupIndex<<<gridSize(keys.size()), 256, 0, stream.value()>>>(
        keys.data<int64_t>(),
        keys.null_mask(),
        keys.offset(),
        keys.size(),
        index.minimum,
        index.rows.size(),
        index.rows.data(),
        matches.data());
  }
  CUDF_CUDA_TRY(cudaGetLastError());
  auto input = thrust::make_zip_iterator(
      thrust::make_tuple(
          thrust::make_counting_iterator<cudf::size_type>(0), matches.begin()));
  auto output = thrust::make_zip_iterator(
      thrust::make_tuple(left->begin(), right->begin()));
  auto end = thrust::copy_if(
      rmm::exec_policy(stream),
      input,
      input + keys.size(),
      matches.begin(),
      output,
      IsMatch{});
  const auto size = end - output;
  left->resize(size, stream);
  right->resize(size, stream);
  return {std::move(left), std::move(right)};
}

} // namespace facebook::velox::cudf_velox
