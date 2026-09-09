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
#include "velox/experimental/cudf/exec/JoinBloomFilter.h"

#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <limits>

namespace facebook::velox::cudf_velox {
namespace {

__device__ uint64_t mixKey(uint64_t x) {
  // Unsigned arithmetic handles the complete signed INT32/INT64 domain.
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ uint64_t bitPair(uint64_t hash) {
  return (uint64_t{1} << (hash & 63)) | (uint64_t{1} << ((hash >> 6) & 63));
}

int gridSize(cudf::size_type rows) {
  return static_cast<int>(std::min<int64_t>((int64_t{rows} + 255) / 256, 4096));
}

template <typename Key, bool Exact>
__global__ void buildFilter(
    const Key* keys,
    const cudf::bitmask_type* nullMask,
    cudf::size_type offset,
    cudf::size_type size,
    uint64_t* words,
    uint64_t wordMask,
    int64_t minimum) {
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < size;
       i += int64_t{blockDim.x} * gridDim.x) {
    if (nullMask && !cudf::bit_is_set(nullMask, i + offset)) {
      continue;
    }
    const auto key = static_cast<uint64_t>(static_cast<int64_t>(keys[i]));
    const auto hash =
        Exact ? key - static_cast<uint64_t>(minimum) : mixKey(key);
    const auto word = Exact ? hash >> 6 : (hash >> 12) & wordMask;
    const auto bits = Exact ? uint64_t{1} << (hash & 63) : bitPair(hash);
    atomicOr(
        reinterpret_cast<unsigned long long*>(words + word),
        static_cast<unsigned long long>(bits));
  }
}

template <typename Key, bool Exact>
struct IsCandidate {
  const Key* keys;
  const cudf::bitmask_type* nullMask;
  cudf::size_type offset;
  const uint64_t* words;
  uint64_t wordMask;
  int64_t minimum;
  uint64_t range;

  __device__ bool operator()(cudf::size_type row) const {
    if (nullMask && !cudf::bit_is_set(nullMask, row + offset)) {
      return false;
    }
    const auto key = static_cast<uint64_t>(static_cast<int64_t>(keys[row]));
    if constexpr (Exact) {
      // Unsigned subtraction also safely rejects values outside narrow ranges
      // near either signed limit; no signed overflow is possible.
      const auto index = key - static_cast<uint64_t>(minimum);
      return index < range &&
          (words[index >> 6] & (uint64_t{1} << (index & 63))) != 0;
    }
    auto hash = mixKey(key);
    auto mask = bitPair(hash);
    return (words[(hash >> 12) & wordMask] & mask) == mask;
  }
};

template <typename Key, bool Exact>
cudf::size_type selectCandidates(
    const JoinBloomFilter& filter,
    cudf::column_view keys,
    cudf::size_type* output,
    rmm::cuda_stream_view stream) {
  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end = thrust::copy_if(
      rmm::exec_policy(stream),
      begin,
      begin + keys.size(),
      output,
      IsCandidate<Key, Exact>{
          keys.data<Key>(),
          keys.null_mask(),
          keys.offset(),
          filter.words.data(),
          filter.words.size() - 1,
          filter.minimum,
          filter.range});
  return end - output;
}

__global__ void remapIndices(
    cudf::size_type* joined,
    cudf::size_type size,
    const cudf::size_type* candidates) {
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; i < size;
       i += int64_t{blockDim.x} * gridDim.x) {
    joined[i] = candidates[joined[i]];
  }
}

} // namespace

std::unique_ptr<JoinBloomFilter> makeJoinBloomFilter(
    cudf::column_view keys,
    uint64_t bitsPerRow,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    uint64_t maxExactRange,
    uint64_t maxExactToBloomRatio) {
  CUDF_EXPECTS(
      keys.size() > 0 && bitsPerRow > 0 && bitsPerRow <= 64 &&
          maxExactToBloomRatio > 0 && maxExactToBloomRatio <= 1024,
      "Invalid join Bloom filter dimensions");
  auto needed =
      (uint64_t{static_cast<uint32_t>(keys.size())} * bitsPerRow + 63) / 64;
  size_t words = 1;
  while (words < needed) {
    words *= 2;
  }
  CUDF_EXPECTS(
      keys.type().id() == cudf::type_id::INT32 ||
          keys.type().id() == cudf::type_id::INT64,
      "Expected integer build keys");
  int64_t minimum = 0;
  uint64_t range = 0;
  if (maxExactRange > 0 && keys.null_count() < keys.size()) {
    auto [low, high] = cudf::minmax(keys, stream, mr);
    const auto value = [&](const cudf::scalar& scalar) -> int64_t {
      return keys.type().id() == cudf::type_id::INT32
          ? static_cast<const cudf::numeric_scalar<int32_t>&>(scalar).value(
                stream)
          : static_cast<const cudf::numeric_scalar<int64_t>&>(scalar).value(
                stream);
    };
    minimum = value(*low);
    const auto difference =
        static_cast<uint64_t>(value(*high)) - static_cast<uint64_t>(minimum);
    // Check before +1 to reject the entire INT64 domain without overflow.
    if (difference < maxExactRange &&
        difference < std::numeric_limits<uint64_t>::max() - 63) {
      const auto exactWords = (difference + 1 + 63) / 64;
      if (exactWords <= words * maxExactToBloomRatio) {
        words = exactWords;
        range = difference + 1;
      }
    }
  }
  auto filter = std::make_unique<JoinBloomFilter>(words, stream, mr);
  filter->minimum = minimum;
  filter->range = range;
  CUDF_CUDA_TRY(cudaMemsetAsync(
      filter->words.data(), 0, words * sizeof(uint64_t), stream.value()));
  const auto launch = [&]<typename Key, bool Exact>() {
    buildFilter<Key, Exact><<<gridSize(keys.size()), 256, 0, stream.value()>>>(
        keys.data<Key>(),
        keys.null_mask(),
        keys.offset(),
        keys.size(),
        filter->words.data(),
        words - 1,
        minimum);
  };
  if (keys.type().id() == cudf::type_id::INT32) {
    if (range > 0) {
      launch.template operator()<int32_t, true>();
    } else {
      launch.template operator()<int32_t, false>();
    }
  } else {
    CUDF_EXPECTS(
        keys.type().id() == cudf::type_id::INT64,
        "Expected integer build keys");
    if (range > 0) {
      launch.template operator()<int64_t, true>();
    } else {
      launch.template operator()<int64_t, false>();
    }
  }
  CUDF_CUDA_TRY(cudaGetLastError());
  return filter;
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> filterJoinProbeKeys(
    const JoinBloomFilter& filter,
    cudf::column_view keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto rows = std::make_unique<rmm::device_uvector<cudf::size_type>>(
      keys.size(), stream, mr);
  if (keys.size() == 0) {
    return rows;
  }
  auto count = keys.type().id() == cudf::type_id::INT32
      ? (filter.range > 0 ? selectCandidates<int32_t, true>(
                                filter, keys, rows->data(), stream)
                          : selectCandidates<int32_t, false>(
                                filter, keys, rows->data(), stream))
      : (filter.range > 0 ? selectCandidates<int64_t, true>(
                                filter, keys, rows->data(), stream)
                          : selectCandidates<int64_t, false>(
                                filter, keys, rows->data(), stream));
  rows->resize(count, stream);
  return rows;
}

void remapJoinProbeIndices(
    rmm::device_uvector<cudf::size_type>& joinedIndices,
    const rmm::device_uvector<cudf::size_type>& candidateIndices,
    rmm::cuda_stream_view stream) {
  if (joinedIndices.is_empty()) {
    return;
  }
  CUDF_EXPECTS(
      joinedIndices.size() <= std::numeric_limits<cudf::size_type>::max(),
      "Join output exceeds supported row count");
  remapIndices<<<gridSize(joinedIndices.size()), 256, 0, stream.value()>>>(
      joinedIndices.data(), joinedIndices.size(), candidateIndices.data());
  CUDF_CUDA_TRY(cudaGetLastError());
}

} // namespace facebook::velox::cudf_velox
