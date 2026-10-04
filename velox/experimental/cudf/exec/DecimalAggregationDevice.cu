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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_for.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda/iterator>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda_runtime.h>
#include <thrust/transform.h>

#include <concepts>
#include <cstdint>

namespace facebook::velox::cudf_velox {
namespace {

// Mirrors the CPU LongDecimalWithOverflowState layout so serialized SUM state
// is interchangeable between CPU and GPU aggregation.
// TODO: Track int128 carries as the CPU does (DecimalUtil::addWithOverflow).
// cuDF's DECIMAL128 SUM wraps modulo 2^128 without counting carries, so every
// GPU producer writes overflow = 0 and the field is meaningful only when it
// came from a CPU-produced state; see DecimalAggregationState.h.
struct DecimalSumState {
  int64_t count; // count of non-null input rows aggregated
  int64_t overflow; // net int128 carries; always 0 when produced on the GPU
  uint64_t lower; // lower 64 bits of the decimal sum
  int64_t upper; // upper 64 bits of the decimal sum (signed)
};

// 2^127, the weight of one unit of the overflow field
// (DecimalUtil::kOverflowMultiplier).
constexpr __uint128_t kOverflowMultiplier = static_cast<__uint128_t>(1) << 127;

// 10^38: DECIMAL(38) values lie strictly within (-kDecimal38Limit,
// kDecimal38Limit).
constexpr __int128_t kDecimal38Limit =
    static_cast<__int128_t>(10'000'000'000'000'000'000ULL) *
    static_cast<__int128_t>(10'000'000'000'000'000'000ULL);

struct DecimalSumCount {
  __int128_t sum;
  int64_t count;
};

struct Decimal64ToSumCount {
  int64_t const* values;
  cudf::bitmask_type const* nullMask;

  __device__ DecimalSumCount operator()(cudf::size_type idx) const {
    return nullMask && !cudf::bit_is_set(nullMask, idx)
        ? DecimalSumCount{0, 0}
        : DecimalSumCount{static_cast<__int128_t>(values[idx]), 1};
  }
};

struct AddDecimalSumCount {
  __device__ DecimalSumCount
  operator()(DecimalSumCount lhs, DecimalSumCount rhs) const {
    return {lhs.sum + rhs.sum, lhs.count + rhs.count};
  }
};

struct StoreDecimalSumCount {
  DecimalSumCount const* result;
  __int128_t* sum;
  int64_t* count;

  __device__ void operator()(cudf::size_type) const {
    *sum = result->sum;
    *count = result->count;
  }
};

static_assert(sizeof(DecimalSumState) == detail::kDecimalSumStateSize);

__device__ __forceinline__ void
splitToWords(int64_t value, int64_t& upper, uint64_t& lower) {
  lower = static_cast<uint64_t>(value);
  upper = value < 0 ? -1 : 0;
}

__device__ __forceinline__ void
splitToWords(__int128_t value, int64_t& upper, uint64_t& lower) {
  lower = static_cast<uint64_t>(value);
  upper = static_cast<int64_t>(value >> 64);
}

template <typename OffsetT>
struct FillOffsetsFunctor {
  cuda::std::span<OffsetT> offsets;

  __device__ void operator()(cudf::size_type idx) const {
    int64_t offset = static_cast<int64_t>(idx) * detail::kDecimalSumStateSize;
    offsets[idx] = static_cast<OffsetT>(offset);
  }
};

// `counts` and `overflows` may be nullptr, in which case the blob fields are
// filled with 1 and 0 respectively (the values a state that does not carry
// them is defined to have).
template <typename SumT, typename OffsetT>
struct PackStateFunctor {
  cuda::std::span<const SumT> sums;
  const int64_t* counts;
  const int64_t* overflows;
  cuda::std::span<const OffsetT> offsets;
  uint8_t* chars;

  __device__ void operator()(cudf::size_type idx) const {
    int64_t offset = static_cast<int64_t>(offsets[idx]);
    auto* state = reinterpret_cast<DecimalSumState*>(chars + offset);
    int64_t upper;
    uint64_t lower;
    splitToWords(sums[idx], upper, lower);
    state->count = counts ? counts[idx] : int64_t{1};
    state->overflow = overflows ? overflows[idx] : int64_t{0};
    state->lower = lower;
    state->upper = upper;
  }
};

template <typename OffsetT>
struct UnpackStateFunctor {
  cuda::std::span<const OffsetT> offsets;
  const uint8_t* chars;
  cuda::std::span<__int128_t> sums;
  cuda::std::span<int64_t> counts;
  int64_t* overflows; // may be nullptr
  cudf::bitmask_type const* nullMask;

  __device__ void operator()(cudf::size_type idx) const {
    if (nullMask && !cudf::bit_is_set(nullMask, idx)) {
      return;
    }
    assert(
        offsets[idx + 1] - offsets[idx] ==
        static_cast<OffsetT>(detail::kDecimalSumStateSize));
    int64_t offset = static_cast<int64_t>(offsets[idx]);
    auto* state = reinterpret_cast<const DecimalSumState*>(chars + offset);
    counts[idx] = state->count;
    if (overflows) {
      overflows[idx] = state->overflow;
    }
    sums[idx] = (static_cast<__int128_t>(state->upper) << 64) | state->lower;
  }
};

// Half-up `value / count` for count > 0: rounds the magnitude up when the
// remainder is at least half the divisor, as DecimalUtil::divideWithRoundUp
// does with noRoundUp == false.
template <typename T>
__device__ __forceinline__ T divideHalfUp(T value, int64_t count) {
  using U = cuda::std::make_unsigned_t<T>;
  U magnitude = value < 0 ? -static_cast<U>(value) : static_cast<U>(value);
  U half = static_cast<U>(count / 2);
  U rounded = (magnitude + half) / static_cast<U>(count);
  // `U{0} - rounded` avoids signed overflow for the most negative value.
  return static_cast<T>(value < 0 ? U{0} - rounded : rounded);
}

// Exact total of a (sum, overflow) pair modulo 2^128, interpreted as int128.
// On the CPU one unit of overflow is worth 2^127 (DecimalUtil::addWithOverflow
// keeps the low 127 bits of each same-sign add and counts the carry), so the
// true total is T = sum + overflow * 2^127 as an exact integer. A GPU merge
// adds the sum children with cuDF's wrapping DECIMAL128 SUM and the overflow
// children exactly, so the merged pair is congruent to T modulo 2^128 but is
// not necessarily in the CPU's canonical form (|sum| < 2^127 after every add).
// Folding modulo 2^128 recovers T exactly whenever |T| < 2^127, which covers
// the whole DECIMAL(38) range; a total beyond int128 aliases and cannot be
// detected from the pair (the wrap count is lost, see the TODO above).
__device__ __forceinline__ __int128_t
foldOverflow(__int128_t sum, int64_t overflow) {
  return static_cast<__int128_t>(
      static_cast<__uint128_t>(sum) +
      static_cast<__uint128_t>(overflow) * kOverflowMultiplier);
}

// True when the pair is in the form DecimalUtil::adjustSumForOverflow
// accepts: the CPU's canonical representation of a total that fits in int128
// after exactly one carry.
__device__ __forceinline__ bool isCanonicalOverflow(
    __int128_t sum,
    int64_t overflow) {
  return (overflow == 1 && sum < 0) || (overflow == -1 && sum > 0);
}

// (sum + overflow * 2^127) / count without widening past 128 bits, mirroring
// DecimalUtil::computeAverage: divide 2^127 and sum by count separately, scale
// the first quotient and remainder by overflow, and round only the combined
// remainder. Bit-identical to the CPU for the same (sum, count, overflow).
__device__ __forceinline__ __int128_t
averageWithSplitOverflow(__int128_t sum, int64_t count, int64_t overflow) {
  auto const unsignedCount = static_cast<__uint128_t>(count);
  auto const unsignedOverflow = static_cast<__uint128_t>(overflow);
  __uint128_t quotientMultiplier = kOverflowMultiplier / unsignedCount;
  __uint128_t remainderMultiplier = kOverflowMultiplier % unsignedCount;
  quotientMultiplier *= unsignedOverflow;
  remainderMultiplier *= unsignedOverflow;
  // C++ truncating division matches divideWithRoundUp with noRoundUp == true:
  // the remainder carries the dividend's sign.
  __int128_t const quotientSum = sum / count;
  __int128_t const remainderSum = sum % count;
  auto const remainderTotal = divideHalfUp(
      static_cast<__int128_t>(
          remainderMultiplier + static_cast<__uint128_t>(remainderSum)),
      count);
  return static_cast<__int128_t>(
      quotientMultiplier + static_cast<__uint128_t>(quotientSum) +
      static_cast<__uint128_t>(remainderTotal));
}

// AVG of a (sum, count, overflow) triple with overflow != 0. A canonical pair
// takes the CPU's split division, which is bit-exact with computeAverage. A
// non-canonical pair (a GPU merge that crossed +-2^127) is folded first and
// then divided half-up. The CPU itself is path dependent by one ulp here: the
// same inputs accumulated in an order that carries give a different canonical
// pair, and the split and folded divisions differ by one ulp in exact-half
// cases where the quotient and remainder terms have opposite signs.
__device__ __forceinline__ __int128_t
averageWithOverflow(__int128_t sum, int64_t count, int64_t overflow) {
  if (isCanonicalOverflow(sum, overflow)) {
    return averageWithSplitOverflow(sum, count, overflow);
  }
  return divideHalfUp(foldOverflow(sum, overflow), count);
}

// Per-row AVG with the CPU's rounding; `overflows` may be nullptr.
template <typename SumT>
struct AvgRoundFunctor {
  cuda::std::span<const SumT> sums;
  cuda::std::span<const int64_t> counts;
  const int64_t* overflows;
  cuda::std::span<SumT> out;

  __device__ void operator()(cudf::size_type idx) const {
    auto count = counts[idx];
    if (count == 0) {
      out[idx] = SumT{0};
      return;
    }
    int64_t const overflow = overflows ? overflows[idx] : int64_t{0};
    if (overflow == 0) {
      out[idx] = divideHalfUp(sums[idx], count);
      return;
    }
    out[idx] = static_cast<SumT>(averageWithOverflow(
        static_cast<__int128_t>(sums[idx]), count, overflow));
  }
};

// Per-row FINAL check of a merged DECIMAL128 sum. When `kFold` is set the
// overflow field is folded into the sum in place (foldOverflow); the sum is
// then range-checked against DECIMAL(38). The worst outcome over all rows is
// accumulated in `*result`.
template <bool kFold>
struct CheckDecimalSumFunctor {
  using SumPointer =
      cuda::std::conditional_t<kFold, __int128_t*, const __int128_t*>;
  SumPointer sums; // already advanced by the sum view's offset
  cudf::bitmask_type const* sumMask; // may be nullptr
  cudf::size_type sumOffset;
  const int64_t* overflows; // used only when kFold
  cudf::bitmask_type const* overflowMask; // may be nullptr
  cudf::size_type overflowOffset;
  int32_t* result;

  __device__ void operator()(cudf::size_type idx) const {
    if (sumMask && !cudf::bit_is_set(sumMask, idx + sumOffset)) {
      return;
    }
    __int128_t sum = sums[idx];
    if constexpr (kFold) {
      auto const overflowIdx = idx + overflowOffset;
      int64_t const overflow =
          overflowMask && !cudf::bit_is_set(overflowMask, overflowIdx)
          ? int64_t{0}
          : overflows[overflowIdx];
      if (overflow != 0) {
        sum = foldOverflow(sum, overflow);
        sums[idx] = sum;
      }
    }
    if (sum <= -kDecimal38Limit || sum >= kDecimal38Limit) {
      atomicMax(
          result, static_cast<int32_t>(detail::DecimalSumCheck::kOutOfRange));
    }
  }
};

template <typename BuildOp>
void launchDeviceFor(
    cudf::size_type size,
    BuildOp buildOp,
    cuda::stream_ref stream) {
  if (size == 0) {
    return;
  }
  auto op = buildOp();
  cub::DeviceFor::ForEachN(
      cuda::counting_iterator<cudf::size_type>{0}, size, op, stream.get());
  CUDF_CUDA_TRY(cudaGetLastError());
}

struct StateValidPredicate {
  cudf::column_device_view sum;
  cudf::column_device_view count;

  __device__ bool operator()(cudf::size_type idx) const {
    if (sum.is_null(idx) || count.is_null(idx)) {
      return false;
    }
    return count.element<int64_t>(idx) != 0;
  }
};

std::pair<cuda::device_buffer<std::byte>, cudf::size_type>
buildStateValidityMaskImpl(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto numRows = sumCol.size();
  if (numRows == 0) {
    return {
        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
        0};
  }
  // The device views are scratch for the kernel below and never leave this
  // call.
  auto sumDeviceView =
      cudf::column_device_view::create(sumCol, stream, get_temp_mr());
  auto countDeviceView =
      cudf::column_device_view::create(countCol, stream, get_temp_mr());
  StateValidPredicate pred{*sumDeviceView, *countDeviceView};
  // Build a BOOL8 column of per-row validity, then convert via the public API.
  auto bools = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::BOOL8},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  auto iter = cuda::counting_iterator{0};
  thrust::transform(
      rmm::exec_policy(stream),
      iter,
      iter + numRows,
      bools->mutable_view().begin<bool>(),
      pred);
  auto [mask, nullCount] = cudf::bools_to_mask(bools->view(), stream, mr);
  return {std::move(*mask), nullCount};
}

} // namespace

namespace detail {

void reduceDecimal64SumCount(
    cudf::column_view input,
    cudf::mutable_column_view sum,
    cudf::mutable_column_view count,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      input.type().id() == cudf::type_id::DECIMAL64,
      "Direct decimal reduction requires DECIMAL64 input");
  CUDF_EXPECTS(
      sum.type().id() == cudf::type_id::DECIMAL128 && sum.size() == 1,
      "Direct decimal reduction requires one DECIMAL128 sum output");
  CUDF_EXPECTS(
      count.type().id() == cudf::type_id::INT64 && count.size() == 1,
      "Direct decimal reduction requires one INT64 count output");

  auto indices = cuda::counting_iterator<cudf::size_type>{0};
  auto transform = Decimal64ToSumCount{
      input.data<int64_t>(), input.nullable() ? input.null_mask() : nullptr};
  auto result = rmm::device_uvector<DecimalSumCount>(1, stream, mr);
  size_t tempStorageBytes = 0;
  CUDF_CUDA_TRY(
      cub::DeviceReduce::TransformReduce(
          nullptr,
          tempStorageBytes,
          indices,
          result.data(),
          input.size(),
          AddDecimalSumCount{},
          transform,
          DecimalSumCount{0, 0},
          stream.get()));
  auto tempStorage = rmm::device_buffer(tempStorageBytes, stream, mr);
  CUDF_CUDA_TRY(
      cub::DeviceReduce::TransformReduce(
          tempStorage.data(),
          tempStorageBytes,
          indices,
          result.data(),
          input.size(),
          AddDecimalSumCount{},
          transform,
          DecimalSumCount{0, 0},
          stream.get()));
  cub::DeviceFor::ForEachN(
      cuda::counting_iterator<cudf::size_type>{0},
      1,
      StoreDecimalSumCount{
          result.data(), sum.data<__int128_t>(), count.data<int64_t>()},
      stream.get());
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename T>
concept OffsetStorageType =
    std::same_as<T, int32_t> || std::same_as<T, int64_t>;

template <typename T>
concept DecimalSumStorageType =
    std::same_as<T, int64_t> || std::same_as<T, __int128_t>;

template <typename SumT, typename OffsetT>
concept ValidDecimalPackStorageTypes =
    DecimalSumStorageType<SumT> && OffsetStorageType<OffsetT>;

struct fillOffsetsForDecimalSumStateKernel {
  cudf::mutable_column_view offsetsView;
  cudf::size_type numRows;
  cuda::stream_ref stream;

  template <typename OffsetT>
    requires OffsetStorageType<OffsetT>
  void operator()() const {
    launchDeviceFor(
        numRows + 1,
        [&] {
          return FillOffsetsFunctor<OffsetT>{cuda::std::span<OffsetT>{
              offsetsView.data<OffsetT>(), static_cast<size_t>(numRows) + 1}};
        },
        stream);
  }

  template <typename OffsetT>
    requires(!OffsetStorageType<OffsetT>)
  void operator()() const {
    CUDF_FAIL("Invalid offset type for decimal sum state");
  }
};

struct unpackDecimalSumStateKernel {
  cudf::column_view offsetsView;
  const uint8_t* chars;
  cudf::mutable_column_view sumView;
  cudf::mutable_column_view countView;
  int64_t* overflows;
  cudf::size_type numRows;
  cudf::bitmask_type const* nullMask;
  cuda::stream_ref stream;

  template <typename OffsetT>
    requires OffsetStorageType<OffsetT>
  void operator()() const {
    auto const n = static_cast<size_t>(numRows);
    launchDeviceFor(
        numRows,
        [&] {
          return UnpackStateFunctor<OffsetT>{
              cuda::std::span<const OffsetT>{
                  offsetsView.data<OffsetT>(), n + 1},
              chars,
              cuda::std::span<__int128_t>{sumView.data<__int128_t>(), n},
              cuda::std::span<int64_t>{countView.data<int64_t>(), n},
              overflows,
              nullMask};
        },
        stream);
  }

  template <typename OffsetT>
    requires(!OffsetStorageType<OffsetT>)
  void operator()() const {
    CUDF_FAIL("Invalid offset type for decimal sum state");
  }
};

struct averageRoundDecimalSumKernel {
  cudf::column_view sumCol;
  const int64_t* counts;
  const int64_t* overflows;
  cudf::mutable_column_view outView;
  cudf::size_type numRows;
  cuda::stream_ref stream;

  template <typename SumT>
    requires DecimalSumStorageType<SumT>
  void operator()() const {
    auto const n = static_cast<size_t>(numRows);
    launchDeviceFor(
        numRows,
        [&] {
          return AvgRoundFunctor<SumT>{
              cuda::std::span<const SumT>{sumCol.data<SumT>(), n},
              cuda::std::span<const int64_t>{counts, n},
              overflows,
              cuda::std::span<SumT>{outView.data<SumT>(), n}};
        },
        stream);
  }

  template <typename SumT>
    requires(!DecimalSumStorageType<SumT>)
  void operator()() const {
    CUDF_FAIL("Invalid sum type for decimal average");
  }
};

struct packDecimalSumStateKernel {
  cudf::column_view sumCol;
  const int64_t* counts;
  const int64_t* overflows;
  cudf::column_view offsetsView;
  uint8_t* chars;
  cudf::size_type numRows;
  cuda::stream_ref stream;

  template <typename SumT, typename OffsetT>
    requires ValidDecimalPackStorageTypes<SumT, OffsetT>
  void operator()() const {
    auto const n = static_cast<size_t>(numRows);
    auto const sums = sumCol.data<SumT>();
    launchDeviceFor(
        numRows,
        [&] {
          return PackStateFunctor<SumT, OffsetT>{
              cuda::std::span<const SumT>{sums, n},
              counts,
              overflows,
              cuda::std::span<const OffsetT>{offsetsView.data<OffsetT>(), n},
              chars};
        },
        stream);
  }

  template <typename SumT, typename OffsetT>
    requires(!ValidDecimalPackStorageTypes<SumT, OffsetT>)
  void operator()() const {
    CUDF_FAIL("Invalid types for decimal sum state pack");
  }
};

void packDecimalSumState(
    cudf::type_id sumType,
    cudf::type_id offsetType,
    cudf::column_view sumCol,
    const int64_t* counts,
    const int64_t* overflows,
    cudf::column_view offsetsView,
    uint8_t* chars,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
  cudf::double_type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{sumType},
      cudf::data_type{offsetType},
      packDecimalSumStateKernel{
          sumCol, counts, overflows, offsetsView, chars, numRows, stream});
}

void fillOffsetsForDecimalSumState(
    cudf::type_id offsetType,
    cudf::mutable_column_view offsetsView,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
  cudf::type_dispatcher(
      cudf::data_type{offsetType},
      fillOffsetsForDecimalSumStateKernel{offsetsView, numRows, stream});
}

void unpackDecimalSumState(
    cudf::type_id offsetType,
    cudf::column_view offsetsView,
    const uint8_t* chars,
    cudf::mutable_column_view sumView,
    cudf::mutable_column_view countView,
    int64_t* overflows,
    cudf::size_type numRows,
    cudf::bitmask_type const* nullMask,
    cuda::stream_ref stream) {
  cudf::type_dispatcher(
      cudf::data_type{offsetType},
      unpackDecimalSumStateKernel{
          offsetsView,
          chars,
          sumView,
          countView,
          overflows,
          numRows,
          nullMask,
          stream});
}

void averageRoundDecimalSum(
    cudf::type_id sumType,
    cudf::column_view sumCol,
    const int64_t* counts,
    const int64_t* overflows,
    cudf::mutable_column_view outView,
    cudf::size_type numRows,
    cuda::stream_ref stream) {
  cudf::type_dispatcher<cudf::dispatch_storage_type>(
      cudf::data_type{sumType},
      averageRoundDecimalSumKernel{
          sumCol, counts, overflows, outView, numRows, stream});
}

DecimalSumCheck checkDecimalSumRange(
    cudf::column_view sum,
    cuda::stream_ref stream) {
  CUDF_EXPECTS(
      sum.type().id() == cudf::type_id::DECIMAL128,
      "Decimal sum range check requires a DECIMAL128 sum column");
  auto const numRows = sum.size();
  if (numRows == 0 || sum.null_count() == numRows) {
    return DecimalSumCheck::kOk;
  }
  int32_t const ok = static_cast<int32_t>(DecimalSumCheck::kOk);
  rmm::device_scalar<int32_t> result(ok, stream, get_temp_mr());
  launchDeviceFor(
      numRows,
      [&] {
        return CheckDecimalSumFunctor<false>{
            sum.data<__int128_t>(),
            sum.nullable() ? sum.null_mask() : nullptr,
            sum.offset(),
            nullptr,
            nullptr,
            0,
            result.data()};
      },
      stream);
  return static_cast<DecimalSumCheck>(result.value(stream));
}

DecimalSumCheck foldDecimalSumOverflow(
    cudf::mutable_column_view sum,
    cudf::column_view overflow,
    cuda::stream_ref stream) {
  CUDF_EXPECTS(
      sum.type().id() == cudf::type_id::DECIMAL128 && sum.offset() == 0,
      "Decimal overflow fold requires an unsliced DECIMAL128 sum column");
  auto const numRows = sum.size();
  CUDF_EXPECTS(
      overflow.type().id() == cudf::type_id::INT64 &&
          overflow.size() == numRows,
      "Decimal overflow fold requires an INT64 overflow column of the sum's size");
  if (numRows == 0 || sum.null_count() == numRows) {
    return DecimalSumCheck::kOk;
  }
  int32_t const ok = static_cast<int32_t>(DecimalSumCheck::kOk);
  rmm::device_scalar<int32_t> result(ok, stream, get_temp_mr());
  launchDeviceFor(
      numRows,
      [&] {
        return CheckDecimalSumFunctor<true>{
            sum.data<__int128_t>(),
            sum.nullable() ? sum.null_mask() : nullptr,
            0,
            overflow.head<int64_t>(),
            overflow.nullable() ? overflow.null_mask() : nullptr,
            overflow.offset(),
            result.data()};
      },
      stream);
  return static_cast<DecimalSumCheck>(result.value(stream));
}

std::pair<cuda::device_buffer<std::byte>, cudf::size_type>
buildStateValidityMask(
    const cudf::column_view& sumCol,
    const cudf::column_view& countCol,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  return buildStateValidityMaskImpl(sumCol, countCol, stream, mr);
}

} // namespace detail
} // namespace facebook::velox::cudf_velox
