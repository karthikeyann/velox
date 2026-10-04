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

// Test columns for the decimal aggregation state tests: host <-> device
// column builders and readers, the CPU 32-byte state blob encoder and decoder,
// builders and readers for state columns of any physical form, and the
// GPU-versus-CPU plan comparison those columns feed.

#include "velox/experimental/cudf/exec/DecimalAggregationDevice.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/base/Exceptions.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/type/HugeInt.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::test {

inline constexpr int kBitsPerWord = 8 * sizeof(cudf::bitmask_type);

inline constexpr DecimalStateShape kAllDecimalStateShapes[] = {
    DecimalStateShape::kSum64,
    DecimalStateShape::kSum128,
    DecimalStateShape::kAvg64,
    DecimalStateShape::kAvg128,
};

/// The fields a struct of `shape` carries besides the sum, per the shape
/// table in DecimalAggregationState.h. Kept here, independent of the
/// production layout code, so the tests pin the documented layout.
struct DecimalStateFields {
  bool hasCount;
  bool hasOverflow;
};

inline DecimalStateFields decimalStateFields(DecimalStateShape shape) {
  switch (shape) {
    case DecimalStateShape::kSum64:
      return {false, false};
    case DecimalStateShape::kSum128:
      return {false, true};
    case DecimalStateShape::kAvg64:
      return {true, false};
    case DecimalStateShape::kAvg128:
      return {true, true};
  }
  VELOX_UNREACHABLE();
}

/// Selects device 0. Returns false when no CUDA device is usable.
inline bool initCudaDevice() {
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
    return false;
  }
  VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
  VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(nullptr)));
  return true;
}

/// Synchronous copies; no-ops for zero bytes.
inline void copyHostToDevice(
    void* destination,
    const void* source,
    size_t bytes,
    cuda::stream_ref stream) {
  if (bytes == 0) {
    return;
  }
  VELOX_CHECK_EQ(
      0,
      static_cast<int>(cudaMemcpyAsync(
          destination, source, bytes, cudaMemcpyHostToDevice, stream.get())));
  stream.sync();
}

inline void copyDeviceToHost(
    void* destination,
    const void* source,
    size_t bytes,
    cuda::stream_ref stream) {
  if (bytes == 0) {
    return;
  }
  VELOX_CHECK_EQ(
      0,
      static_cast<int>(cudaMemcpyAsync(
          destination, source, bytes, cudaMemcpyDeviceToHost, stream.get())));
  stream.sync();
}

/// Device null mask and null count for per-row validity.
inline std::pair<cuda::device_buffer<std::byte>, cudf::size_type> makeNullMask(
    const std::vector<bool>& valid,
    cuda::stream_ref stream) {
  const auto numBits = static_cast<cudf::size_type>(valid.size());
  if (numBits == 0) {
    return {
        cuda::device_buffer<std::byte>{
            stream, cudf::get_current_device_resource_ref()},
        0};
  }
  std::vector<cudf::bitmask_type> host(cudf::num_bitmask_words(numBits), 0);
  cudf::size_type nullCount = 0;
  for (cudf::size_type i = 0; i < numBits; ++i) {
    if (valid[i]) {
      host[i / kBitsPerWord] |= cudf::bitmask_type{1} << (i % kBitsPerWord);
    } else {
      ++nullCount;
    }
  }
  auto mask =
      cudf::create_null_mask(numBits, cudf::mask_state::UNINITIALIZED, stream);
  copyHostToDevice(
      mask.data(),
      host.data(),
      host.size() * sizeof(cudf::bitmask_type),
      stream);
  return {std::move(mask), nullCount};
}

/// Fixed-width column of `values`; nullable only when `valid` is given.
template <typename T>
std::unique_ptr<cudf::column> makeFixedWidthColumn(
    cudf::data_type type,
    const std::vector<T>& values,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  auto column = cudf::make_fixed_width_column(
      type,
      static_cast<cudf::size_type>(values.size()),
      cudf::mask_state::UNALLOCATED,
      stream);
  copyHostToDevice(
      column->mutable_view().template data<T>(),
      values.data(),
      values.size() * sizeof(T),
      stream);
  if (valid) {
    auto [mask, nullCount] = makeNullMask(*valid, stream);
    column->set_null_mask(std::move(mask), nullCount);
  }
  return column;
}

/// DECIMAL64 for int64_t values, DECIMAL128 for int128_t values.
template <typename T>
std::unique_ptr<cudf::column> makeDecimalColumn(
    const std::vector<T>& values,
    int32_t scale,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  const auto typeId = std::is_same_v<T, int64_t> ? cudf::type_id::DECIMAL64
                                                 : cudf::type_id::DECIMAL128;
  return makeFixedWidthColumn(
      cudf::data_type{typeId, -scale}, values, valid, stream);
}

/// Non-decimal INT64 column (keys, counts, overflows).
inline std::unique_ptr<cudf::column> makeInt64Column(
    const std::vector<int64_t>& values,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  return makeFixedWidthColumn(
      cudf::data_type{cudf::type_id::INT64}, values, valid, stream);
}

/// Copies the values of a fixed-width column to the host.
template <typename T>
std::vector<T> copyColumnData(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  std::vector<T> host(view.size());
  copyDeviceToHost(
      host.data(), view.template data<T>(), host.size() * sizeof(T), stream);
  return host;
}

/// Null mask words for rows [0, size) of `view` (re-based when the view is
/// sliced). Empty for a non-nullable view, which isValidAt reads as all valid.
inline std::vector<cudf::bitmask_type> copyNullMask(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  if (!view.nullable() || view.size() == 0) {
    return {};
  }
  std::vector<cudf::bitmask_type> words(
      cudf::num_bitmask_words(view.offset() + view.size()), 0);
  copyDeviceToHost(
      words.data(),
      view.null_mask(),
      words.size() * sizeof(cudf::bitmask_type),
      stream);
  if (view.offset() == 0) {
    words.resize(cudf::num_bitmask_words(view.size()));
    return words;
  }
  std::vector<cudf::bitmask_type> mask(cudf::num_bitmask_words(view.size()), 0);
  for (cudf::size_type i = 0; i < view.size(); ++i) {
    const auto source = view.offset() + i;
    if ((words[source / kBitsPerWord] >> (source % kBitsPerWord)) & 1) {
      mask[i / kBitsPerWord] |= cudf::bitmask_type{1} << (i % kBitsPerWord);
    }
  }
  return mask;
}

/// True if `row` is valid in a mask from copyNullMask.
inline bool isValidAt(const std::vector<cudf::bitmask_type>& mask, size_t row) {
  if (mask.empty()) {
    return true;
  }
  return (mask[row / kBitsPerWord] >> (row % kBitsPerWord)) & 1;
}

/// Per-row validity; all true for a non-nullable view.
inline std::vector<bool> validityOf(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  const auto mask = copyNullMask(view, stream);
  std::vector<bool> valid(view.size());
  for (size_t row = 0; row < valid.size(); ++row) {
    valid[row] = isValidAt(mask, row);
  }
  return valid;
}

/// One decoded row of the CPU decimal state blob.
struct HostDecimalState {
  int64_t count;
  int64_t overflow;
  int128_t sum;

  bool operator==(const HostDecimalState&) const = default;
};

/// Decodes one 32-byte blob row.
inline HostDecimalState decodeDecimalStateRow(const char* bytes) {
  HostDecimalState state;
  uint64_t low;
  int64_t high;
  std::memcpy(&state.count, bytes, sizeof(int64_t));
  std::memcpy(&state.overflow, bytes + 8, sizeof(int64_t));
  std::memcpy(&low, bytes + 16, sizeof(uint64_t));
  std::memcpy(&high, bytes + 24, sizeof(int64_t));
  state.sum = HugeInt::build(static_cast<uint64_t>(high), low);
  return state;
}

/// Encodes one 32-byte blob row.
inline void encodeDecimalStateRow(const HostDecimalState& state, char* bytes) {
  const uint64_t low = HugeInt::lower(state.sum);
  const int64_t high = static_cast<int64_t>(HugeInt::upper(state.sum));
  std::memcpy(bytes, &state.count, sizeof(int64_t));
  std::memcpy(bytes + 8, &state.overflow, sizeof(int64_t));
  std::memcpy(bytes + 16, &low, sizeof(uint64_t));
  std::memcpy(bytes + 24, &high, sizeof(int64_t));
}

/// Builds the STRING blob column CudfFromVelox produces from a CPU partial's
/// VARBINARY output. With `compactNullRows` null rows have 0-byte payloads, as
/// after a round trip through Velox or Arrow.
inline std::unique_ptr<cudf::column> makeDecimalStateBlob(
    const std::vector<HostDecimalState>& rows,
    const std::vector<bool>* valid,
    bool compactNullRows,
    cuda::stream_ref stream) {
  const auto numRows = static_cast<cudf::size_type>(rows.size());
  std::vector<int32_t> offsets{0};
  std::vector<char> chars;
  for (cudf::size_type row = 0; row < numRows; ++row) {
    if (!compactNullRows || valid == nullptr || (*valid)[row]) {
      chars.resize(chars.size() + detail::kDecimalSumStateSize);
      encodeDecimalStateRow(
          rows[row],
          chars.data() + chars.size() - detail::kDecimalSumStateSize);
    }
    offsets.push_back(static_cast<int32_t>(chars.size()));
  }
  auto offsetsColumn = makeFixedWidthColumn(
      cudf::data_type{cudf::type_id::INT32}, offsets, nullptr, stream);
  rmm::device_buffer charsBuffer(chars.size(), stream);
  copyHostToDevice(charsBuffer.data(), chars.data(), chars.size(), stream);
  cuda::device_buffer<std::byte> mask{
      stream, cudf::get_current_device_resource_ref()};
  cudf::size_type nullCount = 0;
  if (valid) {
    std::tie(mask, nullCount) = makeNullMask(*valid, stream);
  }
  return cudf::make_strings_column(
      numRows,
      std::move(offsetsColumn),
      std::move(charsBuffer),
      nullCount,
      std::move(mask));
}

/// Decodes a STRING blob column on the host; null rows are std::nullopt.
inline std::vector<std::optional<HostDecimalState>> readDecimalStateBlob(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  VELOX_CHECK(view.type().id() == cudf::type_id::STRING);
  std::vector<std::optional<HostDecimalState>> rows(view.size());
  if (view.size() == 0) {
    return rows;
  }
  const cudf::strings_column_view strings(view);
  std::vector<int64_t> offsets;
  if (strings.offsets().type().id() == cudf::type_id::INT32) {
    for (auto offset : copyColumnData<int32_t>(strings.offsets(), stream)) {
      offsets.push_back(offset);
    }
  } else {
    offsets = copyColumnData<int64_t>(strings.offsets(), stream);
  }
  std::vector<char> chars(strings.chars_size(stream));
  copyDeviceToHost(
      chars.data(), strings.chars_begin(stream), chars.size(), stream);
  const auto mask = copyNullMask(view, stream);
  for (cudf::size_type row = 0; row < view.size(); ++row) {
    if (!isValidAt(mask, row)) {
      continue;
    }
    const auto begin = offsets[view.offset() + row];
    VELOX_CHECK_EQ(
        static_cast<size_t>(offsets[view.offset() + row + 1] - begin),
        detail::kDecimalSumStateSize);
    rows[row] = decodeDecimalStateRow(chars.data() + begin);
  }
  return rows;
}

/// The logical content of a decimal state column, independent of its physical
/// form. A row is null exactly when `valid` is false.
struct DecimalStateRows {
  std::vector<int128_t> sums;
  std::vector<int64_t> counts;
  std::vector<int64_t> overflows;
  std::vector<bool> valid;

  size_t size() const {
    return sums.size();
  }

  void append(const DecimalStateRows& other) {
    sums.insert(sums.end(), other.sums.begin(), other.sums.end());
    counts.insert(counts.end(), other.counts.begin(), other.counts.end());
    overflows.insert(
        overflows.end(), other.overflows.begin(), other.overflows.end());
    valid.insert(valid.end(), other.valid.begin(), other.valid.end());
  }

  /// The rows as a column of `shape` reports them: fields the shape lacks read
  /// back as count 1 and overflow 0.
  DecimalStateRows as(DecimalStateShape shape) const {
    auto rows = *this;
    if (!decimalStateFields(shape).hasCount) {
      rows.counts.assign(size(), 1);
    }
    if (!decimalStateFields(shape).hasOverflow) {
      rows.overflows.assign(size(), 0);
    }
    return rows;
  }

  /// The blob rows; null rows keep their payload.
  std::vector<HostDecimalState> states() const {
    std::vector<HostDecimalState> states;
    for (size_t row = 0; row < size(); ++row) {
      states.push_back({counts[row], overflows[row], sums[row]});
    }
    return states;
  }
};

/// A STRING blob carrying every field of `rows`.
inline std::unique_ptr<cudf::column> makeDecimalStateBlob(
    const DecimalStateRows& rows,
    cuda::stream_ref stream) {
  return makeDecimalStateBlob(rows.states(), &rows.valid, false, stream);
}

/// A state struct of `shape` built by the production wrapDecimalState. The
/// validity goes on the sum; count and overflow are non-nullable, as cuDF
/// group-by COUNT and SUM of INT64 produce them.
inline std::unique_ptr<cudf::column> makeDecimalStateStruct(
    const DecimalStateRows& rows,
    DecimalStateShape shape,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  DecimalStateColumns flat;
  flat.sum = makeDecimalColumn<int128_t>(rows.sums, scale, &rows.valid, stream);
  if (decimalStateFields(shape).hasCount) {
    flat.count = makeInt64Column(rows.counts, nullptr, stream);
  }
  if (decimalStateFields(shape).hasOverflow) {
    flat.overflow = makeInt64Column(rows.overflows, nullptr, stream);
  }
  return wrapDecimalState(std::move(flat), shape, stream, mr);
}

/// Copies flattened state fields (count and overflow must be present) to the
/// host.
inline DecimalStateRows copyFlatDecimalState(
    const FlatDecimalState& flat,
    cuda::stream_ref stream) {
  EXPECT_EQ(flat.sum.type().id(), cudf::type_id::DECIMAL128);
  EXPECT_EQ(flat.count.type().id(), cudf::type_id::INT64);
  EXPECT_EQ(flat.overflow.type().id(), cudf::type_id::INT64);
  EXPECT_EQ(flat.count.size(), flat.sum.size());
  EXPECT_EQ(flat.overflow.size(), flat.sum.size());
  DecimalStateRows rows;
  rows.sums = copyColumnData<int128_t>(flat.sum, stream);
  rows.counts = copyColumnData<int64_t>(flat.count, stream);
  rows.overflows = copyColumnData<int64_t>(flat.overflow, stream);
  rows.valid = validityOf(flat.sum, stream);
  return rows;
}

/// Reads a state column of any physical form through the production
/// flattenDecimalState and checks the sum carries `scale`.
inline DecimalStateRows readDecimalState(
    const cudf::column_view& column,
    int32_t scale,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto flat = flattenDecimalState(column, scale, true, true, stream, mr);
  EXPECT_EQ(flat.sum.type().scale(), -scale);
  return copyFlatDecimalState(flat, stream);
}

/// Compares valid rows field by field; null rows only by validity.
inline void expectDecimalStateRowsEqual(
    const DecimalStateRows& expected,
    const DecimalStateRows& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t row = 0; row < expected.size(); ++row) {
    ASSERT_EQ(expected.valid[row], actual.valid[row]) << "row " << row;
    if (!expected.valid[row]) {
      continue;
    }
    EXPECT_TRUE(expected.sums[row] == actual.sums[row]) << "sum at row " << row;
    EXPECT_EQ(expected.counts[row], actual.counts[row])
        << "count at row " << row;
    EXPECT_EQ(expected.overflows[row], actual.overflows[row])
        << "overflow at row " << row;
  }
}

/// Runs `plan` on the CPU, then with cuDF registered, and asserts both return
/// the same rows. Leaves cuDF registered.
inline void assertCudfMatchesCpu(
    const core::PlanNodePtr& plan,
    memory::MemoryPool* pool,
    const std::unordered_map<std::string, std::string>& configs,
    int32_t maxDrivers) {
  unregisterCudf();
  auto expected = exec::test::AssertQueryBuilder(plan)
                      .maxDrivers(maxDrivers)
                      .configs(configs)
                      .copyResults(pool);
  registerCudf();
  exec::test::AssertQueryBuilder(plan)
      .maxDrivers(maxDrivers)
      .configs(configs)
      .assertResults(expected);
}

} // namespace facebook::velox::cudf_velox::test
