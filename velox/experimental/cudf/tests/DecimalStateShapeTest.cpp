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

// Unit tests for the self-describing decimal aggregate state helpers declared
// in DecimalAggregationState.h (see
// docs/designs/cudf-self-describing-decimal-aggregate-state.md).

#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

using Shape = DecimalStateShape;

constexpr int kBitsPerWord = 8 * sizeof(cudf::bitmask_type);
constexpr int32_t kScale = 2;
constexpr int64_t kDecimal64Max = 999999999999999999LL;

const std::vector<Shape> kAllShapes = {
    Shape::kSum64,
    Shape::kSum128,
    Shape::kAvg64,
    Shape::kAvg128,
};

std::string shapeToString(Shape shape) {
  switch (shape) {
    case Shape::kSum64:
      return "kSum64";
    case Shape::kSum128:
      return "kSum128";
    case Shape::kAvg64:
      return "kAvg64";
    case Shape::kAvg128:
      return "kAvg128";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// Device <-> host helpers (mirrors the helpers in DecimalAggregationTest.cpp).
// ---------------------------------------------------------------------------

std::pair<cuda::device_buffer<std::byte>, cudf::size_type> makeNullMask(
    const std::vector<bool>& valid,
    cuda::stream_ref stream) {
  auto numBits = static_cast<cudf::size_type>(valid.size());
  if (numBits == 0) {
    return {
        cuda::device_buffer<std::byte>{
            stream, cudf::get_current_device_resource_ref()},
        0};
  }
  auto maskBytes = cudf::bitmask_allocation_size_bytes(numBits);
  auto numWords = maskBytes / sizeof(cudf::bitmask_type);
  std::vector<cudf::bitmask_type> host(numWords, 0);
  cudf::size_type nullCount = 0;
  for (cudf::size_type i = 0; i < numBits; ++i) {
    if (valid[i]) {
      host[i / kBitsPerWord] |= (cudf::bitmask_type{1} << (i % kBitsPerWord));
    } else {
      ++nullCount;
    }
  }
  auto mask =
      cudf::create_null_mask(numBits, cudf::mask_state::UNINITIALIZED, stream);
  auto status = cudaMemcpyAsync(
      mask.data(),
      host.data(),
      host.size() * sizeof(cudf::bitmask_type),
      cudaMemcpyHostToDevice,
      stream.get());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.sync();
  return {std::move(mask), nullCount};
}

template <typename T>
std::unique_ptr<cudf::column> makeFixedWidthColumn(
    cudf::data_type type,
    const std::vector<T>& values,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  auto col = cudf::make_fixed_width_column(
      type,
      static_cast<cudf::size_type>(values.size()),
      cudf::mask_state::UNALLOCATED,
      stream);
  if (!values.empty()) {
    auto status = cudaMemcpyAsync(
        col->mutable_view().data<T>(),
        values.data(),
        values.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        stream.get());
    VELOX_CHECK_EQ(0, static_cast<int>(status));
    stream.sync();
  }
  if (valid) {
    auto [mask, nullCount] = makeNullMask(*valid, stream);
    col->set_null_mask(std::move(mask), nullCount);
  }
  return col;
}

template <typename T>
std::unique_ptr<cudf::column> makeDecimalColumn(
    const std::vector<T>& values,
    int32_t scale,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  cudf::type_id typeId = std::is_same_v<T, int64_t> ? cudf::type_id::DECIMAL64
                                                    : cudf::type_id::DECIMAL128;
  return makeFixedWidthColumn(
      cudf::data_type{typeId, -scale}, values, valid, stream);
}

std::unique_ptr<cudf::column> makeInt64Column(
    const std::vector<int64_t>& values,
    const std::vector<bool>* valid,
    cuda::stream_ref stream) {
  return makeFixedWidthColumn(
      cudf::data_type{cudf::type_id::INT64}, values, valid, stream);
}

template <typename T>
std::vector<T> copyColumnData(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  std::vector<T> host(view.size());
  if (view.size() == 0) {
    return host;
  }
  auto status = cudaMemcpyAsync(
      host.data(),
      view.data<T>(),
      view.size() * sizeof(T),
      cudaMemcpyDeviceToHost,
      stream.get());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.sync();
  return host;
}

std::vector<cudf::bitmask_type> copyNullMask(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  auto numWords = cudf::num_bitmask_words(view.size());
  std::vector<cudf::bitmask_type> host(numWords, 0);
  if (!view.nullable() || numWords == 0) {
    return host;
  }
  // Sliced views: copy the words covering [offset, offset + size).
  std::vector<cudf::bitmask_type> words(
      cudf::num_bitmask_words(view.offset() + view.size()), 0);
  auto status = cudaMemcpyAsync(
      words.data(),
      view.null_mask(),
      words.size() * sizeof(cudf::bitmask_type),
      cudaMemcpyDeviceToHost,
      stream.get());
  VELOX_CHECK_EQ(0, static_cast<int>(status));
  stream.sync();
  for (cudf::size_type i = 0; i < view.size(); ++i) {
    auto src = view.offset() + i;
    if ((words[src / kBitsPerWord] >> (src % kBitsPerWord)) & 1) {
      host[i / kBitsPerWord] |= cudf::bitmask_type{1} << (i % kBitsPerWord);
    }
  }
  return host;
}

// Per-row validity as a vector<bool>; all true for a non-nullable view.
std::vector<bool> validityOf(
    const cudf::column_view& view,
    cuda::stream_ref stream) {
  std::vector<bool> out(view.size(), true);
  if (!view.nullable()) {
    return out;
  }
  auto mask = copyNullMask(view, stream);
  for (cudf::size_type i = 0; i < view.size(); ++i) {
    out[i] = (mask[i / kBitsPerWord] >> (i % kBitsPerWord)) & 1;
  }
  return out;
}

struct HostBlob {
  std::vector<uint8_t> chars;
  std::vector<int64_t> offsets;
  std::vector<bool> valid;
};

HostBlob copyBlob(const cudf::column_view& view, cuda::stream_ref stream) {
  HostBlob out;
  out.valid = validityOf(view, stream);
  if (view.size() == 0) {
    return out;
  }
  cudf::strings_column_view strings(view);
  auto offsets = strings.offsets();
  if (offsets.type().id() == cudf::type_id::INT32) {
    for (auto o : copyColumnData<int32_t>(offsets, stream)) {
      out.offsets.push_back(o);
    }
  } else {
    out.offsets = copyColumnData<int64_t>(offsets, stream);
  }
  auto charsSize = strings.chars_size(stream);
  out.chars.resize(charsSize);
  if (charsSize > 0) {
    auto status = cudaMemcpyAsync(
        out.chars.data(),
        strings.chars_begin(stream),
        charsSize,
        cudaMemcpyDeviceToHost,
        stream.get());
    VELOX_CHECK_EQ(0, static_cast<int>(status));
    stream.sync();
  }
  return out;
}

// Decodes a 32-byte blob row on the host (little endian, CPU layout).
struct HostState {
  int64_t count;
  int64_t overflow;
  __int128_t sum;
};

HostState decodeRow(const HostBlob& blob, size_t row) {
  HostState s{};
  auto base = blob.chars.data() + blob.offsets[row];
  uint64_t lower;
  int64_t upper;
  memcpy(&s.count, base, 8);
  memcpy(&s.overflow, base + 8, 8);
  memcpy(&lower, base + 16, 8);
  memcpy(&upper, base + 24, 8);
  s.sum = (static_cast<__int128_t>(upper) << 64) | lower;
  return s;
}

// ---------------------------------------------------------------------------
// Fixture.
// ---------------------------------------------------------------------------

struct TestData {
  std::vector<__int128_t> sums;
  std::vector<int64_t> counts;
  std::vector<int64_t> overflows;
  std::vector<bool> valid;
};

// Positive, negative, zero and cancelling sums, DECIMAL64 extremes, one null.
TestData defaultData() {
  TestData d;
  d.sums = {
      12345,
      -67890,
      0,
      static_cast<__int128_t>(500) + static_cast<__int128_t>(-500),
      kDecimal64Max,
      -kDecimal64Max,
      (static_cast<__int128_t>(1) << 100) + 7,
      -((static_cast<__int128_t>(1) << 100) + 7),
      999, // null row
  };
  d.counts = {1, 2, 3, 2, 1, 1, 4, 5, 1};
  d.overflows = {0, 0, 0, 0, 0, 0, 1, -1, 0};
  d.valid = {true, true, true, true, true, true, true, true, false};
  return d;
}

class DecimalStateShapeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    auto status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    VELOX_CHECK_EQ(0, static_cast<int>(cudaSetDevice(0)));
    VELOX_CHECK_EQ(0, static_cast<int>(cudaFree(nullptr)));
    stream_ = cudf::get_default_stream();
    mr_ = cudf::get_current_device_resource_ref();
  }

  // Null mask layout: `valid` is applied to sum (authoritative), count and
  // overflow are left non-nullable, as cuDF group-by COUNT/SUM-of-INT64
  // produce them for a group whose sum is null (count 0, overflow 0).
  DecimalStateColumns makeFlat(
      const TestData& d,
      bool withCount,
      bool withOverflow,
      bool sumAsDecimal64 = false) {
    DecimalStateColumns flat;
    if (sumAsDecimal64) {
      std::vector<int64_t> sums64;
      for (auto s : d.sums) {
        sums64.push_back(static_cast<int64_t>(s));
      }
      flat.sum = makeDecimalColumn<int64_t>(sums64, kScale, &d.valid, stream_);
    } else {
      flat.sum = makeDecimalColumn<__int128_t>(d.sums, kScale, &d.valid, stream_);
    }
    if (withCount) {
      flat.count = makeInt64Column(d.counts, nullptr, stream_);
    }
    if (withOverflow) {
      flat.overflow = makeInt64Column(d.overflows, nullptr, stream_);
    }
    return flat;
  }

  std::unique_ptr<cudf::column> makeStruct(const TestData& d, Shape shape) {
    return wrapDecimalState(
        makeFlat(
            d, decimalStateHasCount(shape), decimalStateHasOverflow(shape)),
        shape,
        stream_,
        mr_);
  }

  // Blob the CPU would hand us for `d` (count/overflow as given).
  std::unique_ptr<cudf::column> makeBlob(
      const TestData& d,
      bool withOverflow) {
    auto flat = makeFlat(d, true, withOverflow);
    auto structCol = wrapDecimalState(
        std::move(flat),
        withOverflow ? Shape::kAvg128 : Shape::kAvg64,
        stream_,
        mr_);
    return packDecimalState(structCol->view(), stream_, mr_);
  }

  // Expected count field in a blob packed from `shape`.
  static int64_t expectedCount(const TestData& d, Shape shape, size_t i) {
    return decimalStateHasCount(shape) ? d.counts[i] : 1;
  }

  static int64_t expectedOverflow(const TestData& d, Shape shape, size_t i) {
    return decimalStateHasOverflow(shape) ? d.overflows[i] : 0;
  }

  void expectStructShape(const cudf::column_view& view, Shape shape) {
    ASSERT_EQ(view.type().id(), cudf::type_id::STRUCT);
    ASSERT_TRUE(isDecimalStateStruct(view)) << shapeToString(shape);
    EXPECT_EQ(decimalStateShapeOf(view), shape);
    EXPECT_TRUE(isDecimalStateColumn(view));
    EXPECT_FALSE(view.nullable()) << "struct parent must not carry a mask";
    EXPECT_EQ(view.null_count(), 0);
    int expectedChildren = 1 + decimalStateHasCount(shape) +
        decimalStateHasOverflow(shape);
    EXPECT_EQ(view.num_children(), expectedChildren);
  }

  void expectBlobMatches(
      const cudf::column_view& blobView,
      const TestData& d,
      Shape shape) {
    ASSERT_EQ(blobView.type().id(), cudf::type_id::STRING);
    auto blob = copyBlob(blobView, stream_);
    ASSERT_EQ(blob.valid.size(), d.sums.size());
    for (size_t i = 0; i < d.sums.size(); ++i) {
      EXPECT_EQ(blob.valid[i], d.valid[i]) << "row " << i;
      if (!d.valid[i]) {
        continue;
      }
      auto s = decodeRow(blob, i);
      EXPECT_EQ(s.sum, d.sums[i]) << "row " << i;
      EXPECT_EQ(s.count, expectedCount(d, shape, i)) << "row " << i;
      EXPECT_EQ(s.overflow, expectedOverflow(d, shape, i)) << "row " << i;
    }
  }

  // Checks the flat fields against `d`, with shape-dependent synthesized
  // count/overflow.
  void expectFlatMatches(
      const FlatDecimalState& flat,
      const TestData& d,
      Shape shape) {
    ASSERT_EQ(flat.sum.type().id(), cudf::type_id::DECIMAL128);
    EXPECT_EQ(flat.sum.type().scale(), -kScale);
    ASSERT_EQ(flat.sum.size(), static_cast<cudf::size_type>(d.sums.size()));
    ASSERT_EQ(flat.count.size(), flat.sum.size());
    ASSERT_EQ(flat.overflow.size(), flat.sum.size());
    EXPECT_EQ(flat.count.type().id(), cudf::type_id::INT64);
    EXPECT_EQ(flat.overflow.type().id(), cudf::type_id::INT64);
    auto sums = copyColumnData<__int128_t>(flat.sum, stream_);
    auto counts = copyColumnData<int64_t>(flat.count, stream_);
    auto overflows = copyColumnData<int64_t>(flat.overflow, stream_);
    auto valid = validityOf(flat.sum, stream_);
    for (size_t i = 0; i < d.sums.size(); ++i) {
      EXPECT_EQ(valid[i], d.valid[i]) << "row " << i;
      if (!d.valid[i]) {
        continue;
      }
      EXPECT_EQ(sums[i], d.sums[i]) << "row " << i;
      EXPECT_EQ(counts[i], expectedCount(d, shape, i)) << "row " << i;
      EXPECT_EQ(overflows[i], expectedOverflow(d, shape, i)) << "row " << i;
    }
  }

  cuda::stream_ref stream_{cudf::get_default_stream()};
  rmm::device_async_resource_ref mr_{cudf::get_current_device_resource_ref()};
};

// ---------------------------------------------------------------------------
// Shape selection and recognition.
// ---------------------------------------------------------------------------

TEST_F(DecimalStateShapeTest, shapeFor) {
  EXPECT_EQ(decimalStateShapeFor(false, false), Shape::kSum64);
  EXPECT_EQ(decimalStateShapeFor(false, true), Shape::kSum128);
  EXPECT_EQ(decimalStateShapeFor(true, false), Shape::kAvg64);
  EXPECT_EQ(decimalStateShapeFor(true, true), Shape::kAvg128);

  EXPECT_FALSE(decimalStateHasCount(Shape::kSum64));
  EXPECT_FALSE(decimalStateHasCount(Shape::kSum128));
  EXPECT_TRUE(decimalStateHasCount(Shape::kAvg64));
  EXPECT_TRUE(decimalStateHasCount(Shape::kAvg128));
  EXPECT_FALSE(decimalStateHasOverflow(Shape::kSum64));
  EXPECT_TRUE(decimalStateHasOverflow(Shape::kSum128));
  EXPECT_FALSE(decimalStateHasOverflow(Shape::kAvg64));
  EXPECT_TRUE(decimalStateHasOverflow(Shape::kAvg128));
}

TEST_F(DecimalStateShapeTest, recognizeNonStateColumns) {
  auto int32Col = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED,
      stream_, mr_);
  EXPECT_FALSE(isDecimalStateStruct(int32Col->view()));
  EXPECT_FALSE(isDecimalStateColumn(int32Col->view()));
  VELOX_ASSERT_THROW(
      decimalStateShapeOf(int32Col->view()),
      "not a decimal aggregate state struct");

  auto blob = cudf::make_empty_column(cudf::type_id::STRING);
  EXPECT_FALSE(isDecimalStateStruct(blob->view()));
  EXPECT_TRUE(isDecimalStateColumn(blob->view()));

  // A struct with the right arity but wrong child types is not a state.
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(cudf::make_empty_column(cudf::type_id::INT64));
  children.push_back(cudf::make_empty_column(cudf::type_id::INT64));
  auto badStruct = cudf::make_structs_column(
      0, std::move(children), 0, cuda::device_buffer<std::byte>{stream_, mr_},
      stream_, mr_);
  EXPECT_FALSE(isDecimalStateStruct(badStruct->view()));
  EXPECT_FALSE(isDecimalStateColumn(badStruct->view()));
}

// ---------------------------------------------------------------------------
// wrap -> flatten.
// ---------------------------------------------------------------------------

TEST_F(DecimalStateShapeTest, wrapFlattenRoundTripAllShapes) {
  auto d = defaultData();
  for (auto shape : kAllShapes) {
    SCOPED_TRACE(shapeToString(shape));
    auto structCol = makeStruct(d, shape);
    expectStructShape(structCol->view(), shape);

    auto flat = flattenDecimalState(
        structCol->view(), kScale, true, true, stream_, mr_);
    expectFlatMatches(flat, d, shape);

    // Fields the struct carries must alias its children (no copy).
    auto view = structCol->view();
    auto childPtrs = std::vector<const void*>{};
    for (int i = 0; i < view.num_children(); ++i) {
      childPtrs.push_back(view.child(i).head());
    }
    auto aliases = [&](const cudf::column_view& v) {
      for (auto p : childPtrs) {
        if (p == v.head()) {
          return true;
        }
      }
      return false;
    };
    EXPECT_TRUE(aliases(flat.sum));
    EXPECT_EQ(aliases(flat.count), decimalStateHasCount(shape));
    EXPECT_EQ(aliases(flat.overflow), decimalStateHasOverflow(shape));
    size_t expectedOwned = (decimalStateHasCount(shape) ? 0 : 1) +
        (decimalStateHasOverflow(shape) ? 0 : 1);
    EXPECT_EQ(flat.owned.size(), expectedOwned);
  }
}

TEST_F(DecimalStateShapeTest, flattenSynthesizesOnlyOnDemand) {
  auto d = defaultData();
  auto structCol = makeStruct(d, Shape::kSum64);
  auto flat = flattenDecimalState(
      structCol->view(), kScale, false, false, stream_, mr_);
  EXPECT_EQ(flat.owned.size(), 0);
  EXPECT_EQ(flat.count.size(), 0);
  EXPECT_EQ(flat.overflow.size(), 0);
  EXPECT_EQ(flat.sum.size(), static_cast<cudf::size_type>(d.sums.size()));

  auto onlyCount = flattenDecimalState(
      structCol->view(), kScale, true, false, stream_, mr_);
  EXPECT_EQ(onlyCount.owned.size(), 1);
  EXPECT_EQ(onlyCount.count.size(), flat.sum.size());
  EXPECT_EQ(onlyCount.overflow.size(), 0);
}

TEST_F(DecimalStateShapeTest, wrapCastsDecimal64SumAndInt32Count) {
  TestData d;
  d.sums = {kDecimal64Max, -kDecimal64Max, 0};
  d.counts = {1, 1, 1};
  d.overflows = {0, 0, 0};
  d.valid = {true, true, true};
  auto flat = makeFlat(d, false, false, /*sumAsDecimal64=*/true);
  std::vector<int32_t> counts32 = {7, 8, 9};
  flat.count = makeFixedWidthColumn(
      cudf::data_type{cudf::type_id::INT32}, counts32, nullptr, stream_);
  auto structCol =
      wrapDecimalState(std::move(flat), Shape::kAvg64, stream_, mr_);
  expectStructShape(structCol->view(), Shape::kAvg64);
  auto sum = structCol->view().child(0);
  EXPECT_EQ(sum.type().id(), cudf::type_id::DECIMAL128);
  EXPECT_EQ(sum.type().scale(), -kScale);
  auto sums = copyColumnData<__int128_t>(sum, stream_);
  EXPECT_EQ(sums[0], kDecimal64Max);
  EXPECT_EQ(sums[1], -kDecimal64Max);
  EXPECT_EQ(sums[2], 0);
  auto counts = copyColumnData<int64_t>(structCol->view().child(1), stream_);
  EXPECT_EQ(counts, (std::vector<int64_t>{7, 8, 9}));
}

TEST_F(DecimalStateShapeTest, wrapRejectsMissingRequiredChild) {
  auto d = defaultData();
  VELOX_ASSERT_THROW(
      wrapDecimalState(makeFlat(d, false, false), Shape::kAvg64, stream_, mr_),
      "requires count column");
  VELOX_ASSERT_THROW(
      wrapDecimalState(makeFlat(d, true, false), Shape::kAvg128, stream_, mr_),
      "requires overflow column");
  DecimalStateColumns noSum;
  VELOX_ASSERT_THROW(
      wrapDecimalState(std::move(noSum), Shape::kSum64, stream_, mr_),
      "requires sum column");
}

// ---------------------------------------------------------------------------
// pack.
// ---------------------------------------------------------------------------

TEST_F(DecimalStateShapeTest, packByteEqualsSerializeAllShapes) {
  auto d = defaultData();
  // Overflow is zero everywhere so the blob is comparable with
  // serializeDecimalSumState, which always writes overflow 0.
  std::fill(d.overflows.begin(), d.overflows.end(), 0);
  for (auto shape : kAllShapes) {
    SCOPED_TRACE(shapeToString(shape));
    auto structCol = makeStruct(d, shape);
    auto packed = packDecimalState(structCol->view(), stream_, mr_);
    expectBlobMatches(packed->view(), d, shape);

    // Reference: serializeDecimalSumState for the same sum and the count the
    // shape implies (1 when absent).
    auto sumCol = makeDecimalColumn<__int128_t>(d.sums, kScale, &d.valid, stream_);
    std::vector<int64_t> counts(d.sums.size());
    for (size_t i = 0; i < counts.size(); ++i) {
      counts[i] = expectedCount(d, shape, i);
    }
    auto countCol = makeInt64Column(counts, nullptr, stream_);
    auto reference =
        serializeDecimalSumState(sumCol->view(), countCol->view(), stream_, mr_);

    auto a = copyBlob(packed->view(), stream_);
    auto b = copyBlob(reference->view(), stream_);
    EXPECT_EQ(a.offsets, b.offsets);
    EXPECT_EQ(a.valid, b.valid);
    ASSERT_EQ(a.chars.size(), b.chars.size());
    // Null rows hold unspecified bytes in both; compare valid rows only.
    for (size_t i = 0; i < d.sums.size(); ++i) {
      if (!d.valid[i]) {
        continue;
      }
      EXPECT_TRUE(std::equal(
          a.chars.begin() + a.offsets[i],
          a.chars.begin() + a.offsets[i + 1],
          b.chars.begin() + b.offsets[i]))
          << "row " << i;
    }
  }
}

TEST_F(DecimalStateShapeTest, packCarriesOverflowAndStringFlattenRecoversIt) {
  auto d = defaultData();
  for (auto shape : {Shape::kSum128, Shape::kAvg128}) {
    SCOPED_TRACE(shapeToString(shape));
    auto structCol = makeStruct(d, shape);
    auto packed = packDecimalState(structCol->view(), stream_, mr_);
    expectBlobMatches(packed->view(), d, shape);

    // flatten(STRING, needOverflow) must return the blob's overflow, not 0.
    auto flat =
        flattenDecimalState(packed->view(), kScale, true, true, stream_, mr_);
    EXPECT_EQ(flat.owned.size(), 3);
    expectFlatMatches(flat, d, shape);

    auto decoded = deserializeDecimalSumStateWithOverflow(
        packed->view(), kScale, stream_, mr_);
    ASSERT_NE(decoded.overflow, nullptr);
    auto overflows = copyColumnData<int64_t>(decoded.overflow->view(), stream_);
    for (size_t i = 0; i < d.sums.size(); ++i) {
      if (d.valid[i]) {
        EXPECT_EQ(overflows[i], d.overflows[i]) << "row " << i;
      }
    }
    // The pre-existing API still drops overflow and still agrees on sum/count.
    auto legacy = deserializeDecimalSumState(packed->view(), kScale, stream_);
    EXPECT_EQ(
        copyColumnData<__int128_t>(legacy.sum->view(), stream_),
        copyColumnData<__int128_t>(decoded.sum->view(), stream_));
    EXPECT_EQ(
        copyColumnData<int64_t>(legacy.count->view(), stream_),
        copyColumnData<int64_t>(decoded.count->view(), stream_));
  }
}

TEST_F(DecimalStateShapeTest, stringFlattenWithoutNeedFlagsOwnsOnlySum) {
  auto d = defaultData();
  auto blob = makeBlob(d, true);
  auto flat =
      flattenDecimalState(blob->view(), kScale, false, false, stream_, mr_);
  EXPECT_EQ(flat.owned.size(), 1);
  EXPECT_EQ(flat.count.size(), 0);
  EXPECT_EQ(flat.overflow.size(), 0);
  auto sums = copyColumnData<__int128_t>(flat.sum, stream_);
  auto valid = validityOf(flat.sum, stream_);
  for (size_t i = 0; i < d.sums.size(); ++i) {
    EXPECT_EQ(valid[i], d.valid[i]);
    if (d.valid[i]) {
      EXPECT_EQ(sums[i], d.sums[i]);
    }
  }
}

TEST_F(DecimalStateShapeTest, packValidityFollowsSumOrCountRule) {
  // kSum64: no count child -> mask is the sum mask, nothing else matters.
  TestData d;
  d.sums = {1, 2, 3};
  d.counts = {0, 0, 0};
  d.overflows = {0, 0, 0};
  d.valid = {true, false, true};
  auto sumOnly = makeStruct(d, Shape::kSum64);
  auto packed = packDecimalState(sumOnly->view(), stream_, mr_);
  EXPECT_EQ(validityOf(packed->view(), stream_), d.valid);

  // kAvg64 with a zero count for a valid sum: follows buildStateValidityMask
  // (same as serializeDecimalSumState), so that row is null.
  auto avg = makeStruct(d, Shape::kAvg64);
  auto packedAvg = packDecimalState(avg->view(), stream_, mr_);
  EXPECT_EQ(
      validityOf(packedAvg->view(), stream_),
      (std::vector<bool>{false, false, false}));

  // kAvg64 with a null count for a valid sum: also null.
  DecimalStateColumns flat;
  flat.sum = makeDecimalColumn<__int128_t>(d.sums, kScale, &d.valid, stream_);
  std::vector<bool> countValid = {true, true, false};
  flat.count = makeInt64Column({5, 5, 5}, &countValid, stream_);
  auto avgNullCount =
      wrapDecimalState(std::move(flat), Shape::kAvg64, stream_, mr_);
  auto packedNullCount = packDecimalState(avgNullCount->view(), stream_, mr_);
  EXPECT_EQ(
      validityOf(packedNullCount->view(), stream_),
      (std::vector<bool>{true, false, false}));
}

TEST_F(DecimalStateShapeTest, packRejectsNonStateColumn) {
  auto int32Col = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED,
      stream_, mr_);
  VELOX_ASSERT_THROW(
      packDecimalState(int32Col->view(), stream_, mr_),
      "not a decimal aggregate state struct");
}

// ---------------------------------------------------------------------------
// unpack(pack(x)) == x.
// ---------------------------------------------------------------------------

TEST_F(DecimalStateShapeTest, unpackPackRoundTripAllShapes) {
  auto d = defaultData();
  for (auto shape : kAllShapes) {
    SCOPED_TRACE(shapeToString(shape));
    auto original = makeStruct(d, shape);
    auto packed = packDecimalState(original->view(), stream_, mr_);
    auto unpacked = unpackDecimalState(packed->view(), shape, kScale, stream_, mr_);
    expectStructShape(unpacked->view(), shape);

    auto flatA = flattenDecimalState(
        original->view(), kScale, true, true, stream_, mr_);
    auto flatB = flattenDecimalState(
        unpacked->view(), kScale, true, true, stream_, mr_);
    EXPECT_EQ(validityOf(flatA.sum, stream_), validityOf(flatB.sum, stream_));
    auto sumsA = copyColumnData<__int128_t>(flatA.sum, stream_);
    auto sumsB = copyColumnData<__int128_t>(flatB.sum, stream_);
    auto countsA = copyColumnData<int64_t>(flatA.count, stream_);
    auto countsB = copyColumnData<int64_t>(flatB.count, stream_);
    auto ovfA = copyColumnData<int64_t>(flatA.overflow, stream_);
    auto ovfB = copyColumnData<int64_t>(flatB.overflow, stream_);
    for (size_t i = 0; i < d.sums.size(); ++i) {
      if (!d.valid[i]) {
        continue;
      }
      EXPECT_EQ(sumsA[i], sumsB[i]) << "row " << i;
      EXPECT_EQ(countsA[i], countsB[i]) << "row " << i;
      EXPECT_EQ(ovfA[i], ovfB[i]) << "row " << i;
    }

    // And packing again is byte-identical on valid rows.
    auto repacked = packDecimalState(unpacked->view(), stream_, mr_);
    expectBlobMatches(repacked->view(), d, shape);
  }
}

// Unpacking a blob to a shape that drops fields loses exactly those fields.
TEST_F(DecimalStateShapeTest, unpackToNarrowerShapeDropsFields) {
  auto d = defaultData();
  auto blob = makeBlob(d, true); // count and overflow both non-trivial
  auto narrow =
      unpackDecimalState(blob->view(), Shape::kSum64, kScale, stream_, mr_);
  expectStructShape(narrow->view(), Shape::kSum64);
  auto flat =
      flattenDecimalState(narrow->view(), kScale, true, true, stream_, mr_);
  expectFlatMatches(flat, d, Shape::kSum64); // count 1, overflow 0
}

// ---------------------------------------------------------------------------
// Nulls.
// ---------------------------------------------------------------------------

TEST_F(DecimalStateShapeTest, allNullRows) {
  TestData d;
  d.sums = {1, 2, 3, 4};
  d.counts = {0, 0, 0, 0};
  d.overflows = {0, 0, 0, 0};
  d.valid = {false, false, false, false};
  for (auto shape : kAllShapes) {
    SCOPED_TRACE(shapeToString(shape));
    auto structCol = makeStruct(d, shape);
    auto packed = packDecimalState(structCol->view(), stream_, mr_);
    EXPECT_EQ(packed->size(), 4);
    EXPECT_EQ(packed->null_count(), 4);

    auto flat =
        flattenDecimalState(packed->view(), kScale, true, true, stream_, mr_);
    EXPECT_EQ(flat.sum.null_count(), 4);
    EXPECT_EQ(flat.count.null_count(), 4);
    EXPECT_EQ(flat.overflow.null_count(), 4);

    auto unpacked =
        unpackDecimalState(packed->view(), shape, kScale, stream_, mr_);
    expectStructShape(unpacked->view(), shape);
    auto flatU = flattenDecimalState(
        unpacked->view(), kScale, false, false, stream_, mr_);
    EXPECT_EQ(flatU.sum.null_count(), 4);
  }
}

TEST_F(DecimalStateShapeTest, nullRowsSurviveBlobWithCompactedPayload) {
  // A blob that went through Velox/Arrow has 0-byte payloads for null rows.
  // Build one by slicing offsets manually: valid rows 0 and 2, null row 1.
  std::vector<uint8_t> chars(2 * 32, 0);
  HostState rows[2] = {{3, 0, 777}, {1, -2, -888}};
  for (int r = 0; r < 2; ++r) {
    auto base = chars.data() + r * 32;
    uint64_t lower = static_cast<uint64_t>(rows[r].sum);
    int64_t upper = static_cast<int64_t>(rows[r].sum >> 64);
    memcpy(base, &rows[r].count, 8);
    memcpy(base + 8, &rows[r].overflow, 8);
    memcpy(base + 16, &lower, 8);
    memcpy(base + 24, &upper, 8);
  }
  std::vector<int32_t> offsets = {0, 32, 32, 64};
  auto offsetsCol = makeFixedWidthColumn(
      cudf::data_type{cudf::type_id::INT32}, offsets, nullptr, stream_);
  rmm::device_buffer charsBuf(chars.data(), chars.size(), stream_, mr_);
  std::vector<bool> valid = {true, false, true};
  auto [mask, nullCount] = makeNullMask(valid, stream_);
  auto blob = cudf::make_strings_column(
      3, std::move(offsetsCol), std::move(charsBuf), nullCount, std::move(mask));

  auto flat =
      flattenDecimalState(blob->view(), kScale, true, true, stream_, mr_);
  EXPECT_EQ(validityOf(flat.sum, stream_), valid);
  auto sums = copyColumnData<__int128_t>(flat.sum, stream_);
  auto counts = copyColumnData<int64_t>(flat.count, stream_);
  auto ovf = copyColumnData<int64_t>(flat.overflow, stream_);
  EXPECT_EQ(sums[0], 777);
  EXPECT_EQ(counts[0], 3);
  EXPECT_EQ(ovf[0], 0);
  EXPECT_EQ(sums[2], -888);
  EXPECT_EQ(counts[2], 1);
  EXPECT_EQ(ovf[2], -2);

  for (auto shape : kAllShapes) {
    auto unpacked =
        unpackDecimalState(blob->view(), shape, kScale, stream_, mr_);
    expectStructShape(unpacked->view(), shape);
    EXPECT_EQ(unpacked->view().child(0).null_count(), 1);
  }
}

// ---------------------------------------------------------------------------
// Zero-row inputs.
// ---------------------------------------------------------------------------

TEST_F(DecimalStateShapeTest, zeroRowInputsEveryForm) {
  TestData empty;
  for (auto shape : kAllShapes) {
    SCOPED_TRACE(shapeToString(shape));
    auto structCol = makeStruct(empty, shape);
    EXPECT_EQ(structCol->size(), 0);
    expectStructShape(structCol->view(), shape);

    auto flat = flattenDecimalState(
        structCol->view(), kScale, true, true, stream_, mr_);
    EXPECT_EQ(flat.sum.size(), 0);
    EXPECT_EQ(flat.count.size(), 0);
    EXPECT_EQ(flat.overflow.size(), 0);

    auto packed = packDecimalState(structCol->view(), stream_, mr_);
    EXPECT_EQ(packed->type().id(), cudf::type_id::STRING);
    EXPECT_EQ(packed->size(), 0);

    auto unpacked =
        unpackDecimalState(packed->view(), shape, kScale, stream_, mr_);
    EXPECT_EQ(unpacked->size(), 0);
    expectStructShape(unpacked->view(), shape);

    auto like = makeEmptyDecimalStateLike(structCol->view(), stream_, mr_);
    EXPECT_EQ(like->size(), 0);
    expectStructShape(like->view(), shape);
    auto sumIdx = shape == Shape::kSum128 ? 1 : 0;
    EXPECT_EQ(like->view().child(sumIdx).type().scale(), -kScale);
  }

  auto emptyBlob = cudf::make_empty_column(cudf::type_id::STRING);
  auto flat =
      flattenDecimalState(emptyBlob->view(), kScale, true, true, stream_, mr_);
  EXPECT_EQ(flat.sum.size(), 0);
  EXPECT_EQ(flat.sum.type().id(), cudf::type_id::DECIMAL128);
  EXPECT_EQ(flat.count.type().id(), cudf::type_id::INT64);
  EXPECT_EQ(flat.overflow.type().id(), cudf::type_id::INT64);
  auto likeBlob = makeEmptyDecimalStateLike(emptyBlob->view(), stream_, mr_);
  EXPECT_EQ(likeBlob->type().id(), cudf::type_id::STRING);
  EXPECT_EQ(likeBlob->size(), 0);
}

// ---------------------------------------------------------------------------
// normalizeDecimalStateBatches.
// ---------------------------------------------------------------------------

class NormalizeTest : public DecimalStateShapeTest {
 protected:
  // Concatenates `views` and checks the result decodes to the row-wise
  // concatenation of `datas` (as kAvg128-equivalent fields for the shape each
  // batch had).
  void expectConcatMatches(
      const std::vector<cudf::column_view>& views,
      const std::vector<std::pair<TestData, Shape>>& datas) {
    std::unique_ptr<cudf::column> concat;
    ASSERT_NO_THROW(concat = cudf::concatenate(views, stream_, mr_));
    ASSERT_TRUE(isDecimalStateColumn(concat->view()));
    auto flat = flattenDecimalState(
        concat->view(), kScale, true, true, stream_, mr_);
    auto sums = copyColumnData<__int128_t>(flat.sum, stream_);
    auto counts = copyColumnData<int64_t>(flat.count, stream_);
    auto ovf = copyColumnData<int64_t>(flat.overflow, stream_);
    auto valid = validityOf(flat.sum, stream_);
    size_t row = 0;
    for (auto& [d, shape] : datas) {
      for (size_t i = 0; i < d.sums.size(); ++i, ++row) {
        ASSERT_LT(row, sums.size());
        EXPECT_EQ(valid[row], d.valid[i]) << "row " << row;
        if (!d.valid[i]) {
          continue;
        }
        EXPECT_EQ(sums[row], d.sums[i]) << "row " << row;
        EXPECT_EQ(counts[row], expectedCount(d, shape, i)) << "row " << row;
        EXPECT_EQ(ovf[row], expectedOverflow(d, shape, i)) << "row " << row;
      }
    }
    EXPECT_EQ(row, sums.size());
  }

  TestData secondData() {
    TestData d;
    d.sums = {-1, 1, 42};
    d.counts = {9, 9, 9};
    d.overflows = {0, 0, 0};
    d.valid = {true, true, false};
    return d;
  }
};

TEST_F(NormalizeTest, allStringIsNoOp) {
  auto d1 = defaultData();
  auto d2 = secondData();
  auto b1 = makeBlob(d1, false);
  auto b2 = makeBlob(d2, false);
  std::vector<cudf::column_view> views = {b1->view(), b2->view()};
  auto owned = normalizeDecimalStateBatches(views, kScale, stream_, mr_);
  EXPECT_TRUE(owned.empty());
  EXPECT_EQ(views[0].head(), b1->view().head());
  EXPECT_EQ(views[1].head(), b2->view().head());
  expectConcatMatches(views, {{d1, Shape::kAvg64}, {d2, Shape::kAvg64}});
}

TEST_F(NormalizeTest, sameStructShapeIsNoOp) {
  auto d1 = defaultData();
  auto d2 = secondData();
  for (auto shape : kAllShapes) {
    SCOPED_TRACE(shapeToString(shape));
    auto s1 = makeStruct(d1, shape);
    auto s2 = makeStruct(d2, shape);
    std::vector<cudf::column_view> views = {s1->view(), s2->view()};
    auto owned = normalizeDecimalStateBatches(views, kScale, stream_, mr_);
    EXPECT_TRUE(owned.empty());
    expectConcatMatches(views, {{d1, shape}, {d2, shape}});
  }
}

TEST_F(NormalizeTest, stringMixedWithStructUnpacksBlob) {
  auto d1 = defaultData();
  auto d2 = secondData();
  for (auto shape : kAllShapes) {
    SCOPED_TRACE(shapeToString(shape));
    auto s1 = makeStruct(d1, shape);
    // The blob carries the fields the shape implies so the comparison below is
    // exact after unpacking to that shape.
    auto b2 = packDecimalState(makeStruct(d2, shape)->view(), stream_, mr_);
    std::vector<cudf::column_view> views = {s1->view(), b2->view()};
    // Pass a different scale than the structs carry: the struct scale wins.
    auto owned = normalizeDecimalStateBatches(views, 0, stream_, mr_);
    ASSERT_EQ(owned.size(), 1);
    EXPECT_EQ(views[0].head(), s1->view().head()) << "struct batch untouched";
    expectStructShape(views[1], shape);
    EXPECT_EQ(views[1].child(shape == Shape::kSum128 ? 1 : 0).type().scale(),
              -kScale);
    expectConcatMatches(views, {{d1, shape}, {d2, shape}});
  }
}

TEST_F(NormalizeTest, differentStructShapesWidenToAvg128) {
  auto d1 = defaultData();
  auto d2 = secondData();
  auto d3 = secondData();
  auto s1 = makeStruct(d1, Shape::kSum64);
  auto s2 = makeStruct(d2, Shape::kAvg64);
  auto s3 = makeStruct(d3, Shape::kSum128);
  std::vector<cudf::column_view> views = {s1->view(), s2->view(), s3->view()};
  auto owned = normalizeDecimalStateBatches(views, kScale, stream_, mr_);
  EXPECT_EQ(owned.size(), 3);
  for (auto& v : views) {
    expectStructShape(v, Shape::kAvg128);
  }
  expectConcatMatches(
      views,
      {{d1, Shape::kSum64}, {d2, Shape::kAvg64}, {d3, Shape::kSum128}});

  // Mixed shapes plus a blob: the blob is unpacked to kAvg128 too.
  auto b4 = makeBlob(secondData(), true);
  std::vector<cudf::column_view> views2 = {
      s1->view(), b4->view(), s2->view()};
  auto owned2 = normalizeDecimalStateBatches(views2, kScale, stream_, mr_);
  EXPECT_EQ(owned2.size(), 3);
  for (auto& v : views2) {
    expectStructShape(v, Shape::kAvg128);
  }
  expectConcatMatches(
      views2,
      {{d1, Shape::kSum64}, {secondData(), Shape::kAvg128},
       {d2, Shape::kAvg64}});
}

TEST_F(NormalizeTest, zeroRowStringWithStructBatches) {
  auto d1 = defaultData();
  auto emptyBlob = cudf::make_empty_column(cudf::type_id::STRING);
  for (auto shape : kAllShapes) {
    SCOPED_TRACE(shapeToString(shape));
    auto s1 = makeStruct(d1, shape);
    std::vector<cudf::column_view> views = {emptyBlob->view(), s1->view()};
    // Without normalization this is a cuDF type mismatch.
    EXPECT_ANY_THROW(cudf::concatenate(views, stream_, mr_));
    auto owned = normalizeDecimalStateBatches(views, kScale, stream_, mr_);
    ASSERT_EQ(owned.size(), 1);
    EXPECT_EQ(views[0].size(), 0);
    expectStructShape(views[0], shape);
    EXPECT_EQ(views[1].head(), s1->view().head());
    expectConcatMatches(views, {{TestData{}, shape}, {d1, shape}});
  }
}

TEST_F(NormalizeTest, zeroRowStructWithStringBatches) {
  auto d1 = defaultData();
  auto b1 = makeBlob(d1, false);
  for (auto shape : kAllShapes) {
    SCOPED_TRACE(shapeToString(shape));
    auto emptyStruct = makeStruct(TestData{}, shape);
    std::vector<cudf::column_view> views = {
        b1->view(), emptyStruct->view(), b1->view()};
    EXPECT_ANY_THROW(cudf::concatenate(views, stream_, mr_));
    auto owned = normalizeDecimalStateBatches(views, kScale, stream_, mr_);
    ASSERT_EQ(owned.size(), 1);
    EXPECT_EQ(views[1].type().id(), cudf::type_id::STRING);
    EXPECT_EQ(views[1].size(), 0);
    expectConcatMatches(
        views, {{d1, Shape::kAvg64}, {TestData{}, shape}, {d1, Shape::kAvg64}});
  }
}

TEST_F(NormalizeTest, zeroRowStructOfOtherShapeIsRetyped) {
  auto d1 = defaultData();
  auto s1 = makeStruct(d1, Shape::kSum64);
  auto emptyAvg = makeStruct(TestData{}, Shape::kAvg128);
  std::vector<cudf::column_view> views = {emptyAvg->view(), s1->view()};
  auto owned = normalizeDecimalStateBatches(views, kScale, stream_, mr_);
  ASSERT_EQ(owned.size(), 1);
  // Empties are wildcards: the only non-empty shape wins, no widening.
  expectStructShape(views[0], Shape::kSum64);
  expectStructShape(views[1], Shape::kSum64);
  expectConcatMatches(views, {{TestData{}, Shape::kSum64}, {d1, Shape::kSum64}});
}

TEST_F(NormalizeTest, allZeroRowMixedForms) {
  auto emptyBlob = cudf::make_empty_column(cudf::type_id::STRING);
  auto emptySum = makeStruct(TestData{}, Shape::kSum64);
  auto emptyAvg = makeStruct(TestData{}, Shape::kAvg64);
  std::vector<cudf::column_view> views = {
      emptyBlob->view(), emptySum->view(), emptyAvg->view()};
  auto owned = normalizeDecimalStateBatches(views, kScale, stream_, mr_);
  for (auto& v : views) {
    expectStructShape(v, Shape::kAvg128);
  }
  std::unique_ptr<cudf::column> concat;
  ASSERT_NO_THROW(concat = cudf::concatenate(views, stream_, mr_));
  EXPECT_EQ(concat->size(), 0);

  // Uniform empties stay untouched.
  std::vector<cudf::column_view> blobs = {emptyBlob->view(), emptyBlob->view()};
  EXPECT_TRUE(normalizeDecimalStateBatches(blobs, kScale, stream_, mr_).empty());
  std::vector<cudf::column_view> none;
  EXPECT_TRUE(normalizeDecimalStateBatches(none, kScale, stream_, mr_).empty());
}

TEST_F(NormalizeTest, rejectsNonStateBatch) {
  auto int32Col = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED,
      stream_, mr_);
  auto s1 = makeStruct(defaultData(), Shape::kSum64);
  std::vector<cudf::column_view> views = {s1->view(), int32Col->view()};
  VELOX_ASSERT_THROW(
      normalizeDecimalStateBatches(views, kScale, stream_, mr_),
      "not a decimal aggregate state column");
}

// ---------------------------------------------------------------------------
// validateIntermediateColumnType.
// ---------------------------------------------------------------------------

TEST_F(DecimalStateShapeTest, validateIntermediateColumnTypeAcceptsAllForms) {
  auto d = defaultData();
  for (auto shape : kAllShapes) {
    SCOPED_TRACE(shapeToString(shape));
    auto structCol = makeStruct(d, shape);
    EXPECT_NO_THROW(validateIntermediateColumnType(structCol->view()));
    auto packed = packDecimalState(structCol->view(), stream_, mr_);
    EXPECT_NO_THROW(validateIntermediateColumnType(packed->view()));
  }
  auto emptyBlob = cudf::make_empty_column(cudf::type_id::STRING);
  EXPECT_NO_THROW(validateIntermediateColumnType(emptyBlob->view()));

  auto int32Col = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32}, 3, cudf::mask_state::UNALLOCATED,
      stream_, mr_);
  VELOX_ASSERT_THROW(
      validateIntermediateColumnType(int32Col->view()),
      "Expected decimal aggregation state");

  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(cudf::make_empty_column(cudf::type_id::INT64));
  auto badStruct = cudf::make_structs_column(
      0, std::move(children), 0, cuda::device_buffer<std::byte>{stream_, mr_},
      stream_, mr_);
  VELOX_ASSERT_THROW(
      validateIntermediateColumnType(badStruct->view()),
      "Expected decimal aggregation state");
}

} // namespace
} // namespace facebook::velox::cudf_velox
