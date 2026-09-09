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
#include "velox/experimental/cudf/exec/DenseIntegerSum.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <limits>

namespace facebook::velox::cudf_velox {
namespace {
int gridSize(int64_t rows) {
  return static_cast<int>(std::min<int64_t>((rows + 255) / 256, 4096));
}

__global__ void copyState(
    const uint64_t* oldSums,
    const uint32_t* oldFlags,
    int64_t oldRange,
    uint64_t* sums,
    uint32_t* flags,
    int64_t range,
    int64_t offset,
    bool count32) {
  for (int64_t row = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
       row <= oldRange;
       row += int64_t{blockDim.x} * gridDim.x) {
    const auto target = row == oldRange ? range : row + offset;
    if (count32) {
      reinterpret_cast<uint32_t*>(sums)[target] =
          reinterpret_cast<const uint32_t*>(oldSums)[row];
    } else {
      sums[target] = oldSums[row];
    }
    if (flags) {
      flags[target] = oldFlags[row];
    }
  }
}

template <typename Key>
__global__ void addRows(
    const Key* keys,
    const cudf::bitmask_type* keyMask,
    int32_t keyOffset,
    const int64_t* values,
    const cudf::bitmask_type* valueMask,
    int32_t valueOffset,
    int32_t rows,
    int64_t minimum,
    uint64_t range,
    bool ignoreNullKeys,
    bool countRows,
    bool count32,
    uint64_t* sums,
    uint32_t* flags) {
  for (int64_t row = int64_t{blockIdx.x} * blockDim.x + threadIdx.x; row < rows;
       row += int64_t{blockDim.x} * gridDim.x) {
    const bool validKey =
        !keyMask || cudf::bit_is_set(keyMask, row + keyOffset);
    if (!validKey && ignoreNullKeys) {
      continue;
    }
    const auto index = validKey
        ? static_cast<uint64_t>(static_cast<int64_t>(keys[row])) -
            static_cast<uint64_t>(minimum)
        : range;
    const bool validValue = countRows || !valueMask ||
        cudf::bit_is_set(valueMask, row + valueOffset);
    if (!countRows) {
      atomicOr(flags + index, validValue ? 3u : 1u);
    }
    if (validValue) {
      if (count32) {
        atomicAdd(reinterpret_cast<unsigned int*>(sums) + index, 1u);
      } else {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(sums + index),
            countRows ? 1ULL : static_cast<unsigned long long>(values[row]));
      }
    }
  }
}

struct IsPresent {
  const uint32_t* flags;
  const uint64_t* counts;
  bool count32;
  __device__ bool operator()(int32_t index) const {
    if (count32) {
      return reinterpret_cast<const uint32_t*>(counts)[index] != 0;
    }
    return counts ? counts[index] != 0 : (flags[index] & 1) != 0;
  }
};

__device__ void clearValidity(cudf::bitmask_type* mask, int32_t row) {
  atomicAnd(mask + row / 32, ~(uint32_t{1} << (row % 32)));
}

template <typename Key>
__global__ void writeOutput(
    const int32_t* indices,
    int32_t count,
    int64_t minimum,
    int64_t range,
    const uint64_t* sums,
    const uint32_t* flags,
    bool count32,
    Key* outputKeys,
    cudf::bitmask_type* keyMask,
    int64_t* outputValues,
    cudf::bitmask_type* valueMask) {
  for (int64_t row = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
       row < count;
       row += int64_t{blockDim.x} * gridDim.x) {
    const auto index = indices[row];
    const bool nullKey = index == range;
    outputKeys[row] = nullKey ? Key{0} : static_cast<Key>(minimum + index);
    outputValues[row] = count32
        ? static_cast<int64_t>(reinterpret_cast<const uint32_t*>(sums)[index])
        : static_cast<int64_t>(sums[index]);
    if (nullKey) {
      clearValidity(keyMask, row);
    }
    if (flags && (flags[index] & 2) == 0) {
      clearValidity(valueMask, row);
    }
  }
}
} // namespace

struct DenseIntegerSum::Impl {
  cudf::data_type keyType;
  uint64_t maxRange;
  bool ignoreNullKeys;
  bool countRows;
  uint64_t count32MaxRows;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
  int64_t minimum{0};
  uint64_t range{0};
  __int128 positiveBound{0};
  __int128 negativeBound{0};
  uint64_t inputRows{0};
  rmm::device_uvector<uint64_t> sums;
  rmm::device_uvector<uint32_t> flags;

  Impl(
      cudf::data_type type,
      uint64_t limit,
      bool ignore,
      rmm::cuda_stream_view inputStream,
      rmm::device_async_resource_ref resource,
      bool count,
      uint64_t count32Limit)
      : keyType(type),
        maxRange(limit),
        ignoreNullKeys(ignore),
        countRows(count),
        count32MaxRows(count ? count32Limit : 0),
        stream(inputStream),
        mr(resource),
        sums(1, stream, mr),
        flags(1, stream, mr) {
    CUDF_EXPECTS(
        limit > 0 && limit < std::numeric_limits<int32_t>::max(),
        "Invalid dense SUM range limit");
    CUDF_EXPECTS(
        count32Limit <= std::numeric_limits<uint32_t>::max(),
        "Invalid dense COUNT32 row limit");
    CUDF_CUDA_TRY(
        cudaMemsetAsync(sums.data(), 0, sizeof(uint64_t), stream.value()));
    CUDF_CUDA_TRY(
        cudaMemsetAsync(flags.data(), 0, sizeof(uint32_t), stream.value()));
  }
};

DenseIntegerSum::DenseIntegerSum(
    cudf::data_type type,
    uint64_t limit,
    bool ignore,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool countRows,
    uint64_t count32MaxRows)
    : impl_(
          std::make_unique<Impl>(
              type,
              limit,
              ignore,
              stream,
              mr,
              countRows,
              count32MaxRows)) {}
DenseIntegerSum::~DenseIntegerSum() = default;
uint64_t DenseIntegerSum::range() const {
  return impl_->range;
}
uint64_t DenseIntegerSum::bytes() const {
  return impl_->sums.size() * sizeof(uint64_t) +
      impl_->flags.size() * sizeof(uint32_t);
}
rmm::cuda_stream_view DenseIntegerSum::stream() const {
  return impl_->stream;
}

bool DenseIntegerSum::add(cudf::column_view keys, cudf::column_view values) {
  auto& state = *impl_;
  CUDF_EXPECTS(
      keys.type() == state.keyType &&
          (state.countRows ||
           (values.type().id() == cudf::type_id::INT64 &&
            keys.size() == values.size())),
      "Invalid dense integer SUM input");
  if (keys.size() == 0 ||
      (state.ignoreNullKeys && keys.null_count() == keys.size())) {
    return true;
  }
  auto positive = state.positiveBound;
  auto negative = state.negativeBound;
  if (state.countRows) {
    positive += keys.size();
    // This bounds every individual counter, including the NULL-key bucket.
    // Reject before changing state; ordinary BIGINT partial COUNT takes over.
    if (state.count32MaxRows && positive > state.count32MaxRows) {
      return false;
    }
  } else if (values.null_count() < values.size()) {
    auto [low, high] = cudf::minmax(values, state.stream, state.mr);
    const auto minimum =
        static_cast<cudf::numeric_scalar<int64_t>&>(*low).value(state.stream);
    const auto maximum =
        static_cast<cudf::numeric_scalar<int64_t>&>(*high).value(state.stream);
    positive +=
        static_cast<__int128>(std::max<int64_t>(maximum, 0)) * values.size();
    negative +=
        -static_cast<__int128>(std::min<int64_t>(minimum, 0)) * values.size();
  }
  if (positive > std::numeric_limits<int64_t>::max() ||
      negative > -static_cast<__int128>(std::numeric_limits<int64_t>::min())) {
    return false;
  }
  int64_t proposedMinimum = state.minimum;
  uint64_t proposedRange = state.range;
  if (keys.null_count() < keys.size()) {
    auto [low, high] = cudf::minmax(keys, state.stream, state.mr);
    const auto value = [&](const cudf::scalar& scalar) -> int64_t {
      return keys.type().id() == cudf::type_id::INT32
          ? static_cast<const cudf::numeric_scalar<int32_t>&>(scalar).value(
                state.stream)
          : static_cast<const cudf::numeric_scalar<int64_t>&>(scalar).value(
                state.stream);
    };
    const auto minimum =
        state.range == 0 ? value(*low) : std::min(value(*low), state.minimum);
    const auto maximum = state.range == 0
        ? value(*high)
        : std::max(
              value(*high),
              state.minimum + static_cast<int64_t>(state.range - 1));
    const auto difference =
        static_cast<uint64_t>(maximum) - static_cast<uint64_t>(minimum);
    if (difference >= state.maxRange ||
        (state.inputRows == 0 &&
         difference + 1 > uint64_t{static_cast<uint32_t>(keys.size())} * 16)) {
      return false;
    }
    if (state.range == 0 || minimum < state.minimum ||
        difference + 1 > state.range) {
      // Modest padding avoids repeated whole-state reallocations as later
      // batches discover a few extra boundary keys. Use 128-bit host bounds.
      const auto padding = std::min<uint64_t>(
          (state.maxRange - difference - 1) / 2,
          std::max<uint64_t>(1024, (difference + 1) / 64));
      const auto paddedLow = std::max<__int128>(
          std::numeric_limits<int64_t>::min(),
          static_cast<__int128>(minimum) - padding);
      const auto paddedHigh = std::min<__int128>(
          std::numeric_limits<int64_t>::max(),
          static_cast<__int128>(maximum) + padding);
      proposedMinimum = static_cast<int64_t>(paddedLow);
      proposedRange = static_cast<uint64_t>(paddedHigh - paddedLow + 1);
    }
  }
  if (proposedRange != state.range || proposedMinimum != state.minimum) {
    rmm::device_uvector<uint64_t> sums(
        state.count32MaxRows ? (proposedRange + 2) / 2 : proposedRange + 1,
        state.stream,
        state.mr);
    rmm::device_uvector<uint32_t> flags(
        state.countRows ? 1 : proposedRange + 1, state.stream, state.mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(
        sums.data(), 0, sums.size() * sizeof(uint64_t), state.stream.value()));
    CUDF_CUDA_TRY(cudaMemsetAsync(
        flags.data(),
        0,
        flags.size() * sizeof(uint32_t),
        state.stream.value()));
    const auto offset = state.range == 0
        ? 0
        : static_cast<int64_t>(
              static_cast<uint64_t>(state.minimum) -
              static_cast<uint64_t>(proposedMinimum));
    copyState<<<gridSize(state.range + 1), 256, 0, state.stream.value()>>>(
        state.sums.data(),
        state.countRows ? nullptr : state.flags.data(),
        state.range,
        sums.data(),
        state.countRows ? nullptr : flags.data(),
        proposedRange,
        offset,
        state.count32MaxRows != 0);
    CUDF_CUDA_TRY(cudaGetLastError());
    state.sums = std::move(sums);
    state.flags = std::move(flags);
    state.minimum = proposedMinimum;
    state.range = proposedRange;
  }
  const auto launch = [&]<typename Key>() {
    addRows<<<gridSize(keys.size()), 256, 0, state.stream.value()>>>(
        keys.data<Key>(),
        keys.null_mask(),
        keys.offset(),
        state.countRows ? nullptr : values.data<int64_t>(),
        state.countRows ? nullptr : values.null_mask(),
        values.offset(),
        keys.size(),
        state.minimum,
        state.range,
        state.ignoreNullKeys,
        state.countRows,
        state.count32MaxRows != 0,
        state.sums.data(),
        state.flags.data());
  };
  if (keys.type().id() == cudf::type_id::INT32) {
    launch.template operator()<int32_t>();
  } else {
    launch.template operator()<int64_t>();
  }
  CUDF_CUDA_TRY(cudaGetLastError());
  state.positiveBound = positive;
  state.negativeBound = negative;
  state.inputRows += keys.size();
  return true;
}

std::unique_ptr<cudf::table> DenseIntegerSum::finalize() {
  auto& state = *impl_;
  rmm::device_uvector<int32_t> indices(state.range + 1, state.stream, state.mr);
  auto begin = thrust::make_counting_iterator<int32_t>(0);
  auto end = thrust::copy_if(
      rmm::exec_policy(state.stream),
      begin,
      begin + static_cast<int32_t>(state.range + 1),
      indices.begin(),
      IsPresent{
          state.flags.data(),
          state.countRows ? state.sums.data() : nullptr,
          state.count32MaxRows != 0});
  const auto count = static_cast<int32_t>(end - indices.begin());
  auto keys = cudf::make_numeric_column(
      state.keyType,
      count,
      cudf::mask_state::ALL_VALID,
      state.stream,
      state.mr);
  auto values = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64},
      count,
      cudf::mask_state::ALL_VALID,
      state.stream,
      state.mr);
  if (count > 0) {
    auto keyView = keys->mutable_view();
    auto valueView = values->mutable_view();
    const auto launch = [&]<typename Key>() {
      writeOutput<<<gridSize(count), 256, 0, state.stream.value()>>>(
          indices.data(),
          count,
          state.minimum,
          state.range,
          state.sums.data(),
          state.countRows ? nullptr : state.flags.data(),
          state.count32MaxRows != 0,
          keyView.data<Key>(),
          keyView.null_mask(),
          valueView.data<int64_t>(),
          valueView.null_mask());
    };
    if (state.keyType.id() == cudf::type_id::INT32) {
      launch.template operator()<int32_t>();
    } else {
      launch.template operator()<int64_t>();
    }
    CUDF_CUDA_TRY(cudaGetLastError());
    keys->set_null_count(
        cudf::null_count(keyView.null_mask(), 0, count, state.stream));
    values->set_null_count(
        cudf::null_count(valueView.null_mask(), 0, count, state.stream));
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(keys));
  columns.push_back(std::move(values));
  return std::make_unique<cudf::table>(std::move(columns));
}
} // namespace facebook::velox::cudf_velox
