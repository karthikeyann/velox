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

#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/caching/FileHandle.h"
#include "velox/common/config/Config.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/error.hpp>

#include <gflags/gflags.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

DECLARE_bool(cudf_decoded_cache_skip_full_range_slice);
DECLARE_bool(cudf_gpu_cache_store_packed);
DECLARE_bool(cudf_gpu_cache_scaled_float);
DECLARE_bool(cudf_cache_restore_packed_views);
DECLARE_bool(cudf_cache_stream_ordered_release);
DECLARE_string(cudf_gpu_cache_packed_types);
DECLARE_uint64(cudf_gpu_cache_raw_prefix_bytes);
DECLARE_double(cudf_gpu_cache_min_host_fraction);

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

class MetadataOnlySplitReader final : public CudfSplitReader {
 public:
  using CudfSplitReader::CudfSplitReader;

  cudf::ast::expression const* logicalFilter() const {
    return subfieldFilter();
  }

  cudf::ast::expression const* splitFilter() const {
    return pushdownFilter();
  }

  bool hasSplitFilter() const {
    return hasSplitSpecificPushdownFilter();
  }

 protected:
  void prepareSplitInternal(
      dwio::common::RuntimeStats& /*runtimeStats*/) override {
    fileMetaDatas();
    // Metadata caching must not rebuild the filter during one preparation.
    fileMetaDatas();
  }
};

class CudfSplitReaderTest : public ::facebook::velox::cudf_velox::exec::test::
                                CudfHiveConnectorTestBase {};

TEST_F(CudfSplitReaderTest, buildsPushdownFilterForEachSplitPreparation) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3})}));

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  ::facebook::velox::connector::ConnectorQueryCtx connectorQueryCtx(
      pool_.get(),
      pool_.get(),
      properties.get(),
      nullptr,
      common::PrefixSortConfig{},
      nullptr,
      nullptr,
      "query.CudfSplitReaderTest",
      "task.CudfSplitReaderTest",
      "plan.CudfSplitReaderTest",
      0,
      "");
  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto split =
      CudfHiveConnectorSplitBuilder(dataFile->getPath())
          .connectorId(
              ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
          .build();

  cudf::ast::column_reference logicalFilter{0};
  cudf::ast::column_reference firstSplitFilter{0};
  cudf::ast::column_reference secondSplitFilter{0};
  MetadataOnlySplitReader reader(
      std::move(split),
      ::facebook::velox::cudf_velox::exec::test::CudfHiveConnectorTestBase::
          makeTableHandle("parquet_table", rowType),
      rowType,
      {"c0"},
      &fileHandleFactory,
      ioExecutor_.get(),
      &connectorQueryCtx,
      std::make_shared<CudfHiveConfig>(properties),
      std::make_shared<io::IoStatistics>(),
      std::make_shared<IoStats>(),
      false,
      &logicalFilter);

  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &logicalFilter);
  EXPECT_FALSE(reader.hasSplitFilter());

  size_t builderCalls = 0;
  std::vector<size_t> schemaSizes;
  reader.setPushdownFilterBuilder(
      [&](const cudf::io::parquet::FileMetaData& metadata) {
        schemaSizes.push_back(metadata.schema.size());
        return builderCalls++ == 0
            ? static_cast<cudf::ast::expression const*>(&firstSplitFilter)
            : static_cast<cudf::ast::expression const*>(&secondSplitFilter);
      });

  // Installing a builder does not change the filter until split metadata is
  // available.
  EXPECT_EQ(reader.splitFilter(), &logicalFilter);
  EXPECT_FALSE(reader.hasSplitFilter());

  dwio::common::RuntimeStats runtimeStats;
  reader.prepareSplit(runtimeStats);
  EXPECT_EQ(builderCalls, 1);
  ASSERT_EQ(schemaSizes.size(), 1);
  EXPECT_GT(schemaSizes.front(), 1);
  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &firstSplitFilter);
  EXPECT_TRUE(reader.hasSplitFilter());

  // Preparing again resets the previous split filter and rebuilds it from the
  // footer without replacing the logical filter.
  reader.prepareSplit(runtimeStats);
  EXPECT_EQ(builderCalls, 2);
  ASSERT_EQ(schemaSizes.size(), 2);
  EXPECT_GT(schemaSizes.back(), 1);
  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &secondSplitFilter);
  EXPECT_TRUE(reader.hasSplitFilter());
  EXPECT_EQ(runtimeStats.processedSplits, 2);
}

TEST_F(CudfSplitReaderTest, pinnedRangeCacheAssemblesOverlaps) {
  auto input =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(100, folly::identity)});
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, input->pool(), stream, mr);
  auto ranges = cudf::slice(cudfTable->view(), {0, 50, 50, 100}, stream);
  ASSERT_EQ(ranges.size(), 2);

  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  CudfDecodedColumnCache::ColumnKey key{
      .file = {.connectorId = "test", .filePath = "overlapping-ranges"},
      .deviceId = deviceId,
      .columnName = "c0",
      .veloxType = BIGINT()->toString(),
      .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
      .usePandasMetadata = true,
      .useArrowSchema = true,
      .allowMismatchedSchemas = false,
  };

  auto& cache = CudfDecodedColumnCache::instance();
  EXPECT_EQ(cache.pinnedBytes(), 0);
  EXPECT_EQ(CudfDecodedColumnCache::kMaxPinnedBytes, 70ULL << 30);
  EXPECT_EQ(cache.maxPinnedBytes(), CudfDecodedColumnCache::kMaxPinnedBytes);
  ASSERT_TRUE(cache.insertColumnRangeIfAbsent(
      key,
      0,
      50,
      ranges[0].column(0),
      stream,
      mr,
      CudfDecodedColumnCache::CompressionMode::kNone));
  ASSERT_TRUE(cache.insertColumnRangeIfAbsent(
      key,
      50,
      100,
      ranges[1].column(0),
      stream,
      mr,
      CudfDecodedColumnCache::CompressionMode::kNone));
  EXPECT_GT(cache.pinnedBytes(), 0);
  EXPECT_LE(cache.pinnedBytes(), cache.maxPinnedBytes());

  auto coverage = cache.findColumnRanges(key, 25, 75);
  ASSERT_TRUE(coverage.has_value());
  ASSERT_EQ(coverage->size(), 2);
  EXPECT_EQ(coverage->at(0).firstRow, 25);
  EXPECT_EQ(coverage->at(0).lastRow, 50);
  EXPECT_EQ(coverage->at(1).firstRow, 50);
  EXPECT_EQ(coverage->at(1).lastRow, 75);
  for (const auto& range : *coverage) {
    cudaPointerAttributes attributes{};
    CUDF_CUDA_TRY(
        cudaPointerGetAttributes(&attributes, range.chunk->pinnedData()));
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
  }

  auto assembled = cache.materializeColumnRange(key, 25, 75, stream, mr, mr);
  ASSERT_NE(assembled, nullptr);
  ASSERT_EQ(assembled->size(), 50);
  std::vector<int64_t> actual(assembled->size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      actual.data(),
      assembled->view().data<int64_t>(),
      actual.size() * sizeof(int64_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], i + 25);
  }

  EXPECT_EQ(
      cache.materializeColumnRange(key, 25, 125, stream, mr, mr), nullptr);
  const auto pinnedBytes = cache.pinnedBytes();
  EXPECT_FALSE(cache.insertColumnRangeIfAbsent(
      key,
      0,
      50,
      ranges[0].column(0),
      stream,
      mr,
      CudfDecodedColumnCache::CompressionMode::kNone));
  EXPECT_EQ(cache.pinnedBytes(), pinnedBytes);
}

TEST_F(CudfSplitReaderTest, gpuRangeCacheAssemblesOverlaps) {
  auto input =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(100, folly::identity)});
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, input->pool(), stream, mr);
  auto ranges = cudf::slice(cudfTable->view(), {0, 50, 50, 100}, stream);
  ASSERT_EQ(ranges.size(), 2);

  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  CudfDecodedColumnCache::ColumnKey key{
      .file = {.connectorId = "test", .filePath = "gpu-overlapping-ranges"},
      .deviceId = deviceId,
      .columnName = "c0",
      .veloxType = BIGINT()->toString(),
      .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
      .usePandasMetadata = true,
      .useArrowSchema = true,
      .allowMismatchedSchemas = false,
  };

  auto& cache = CudfDecodedColumnCache::instance();
  const auto gpuBytesBefore = cache.gpuBytes();
  ASSERT_TRUE(cache.insertGpuColumnRangeIfAbsent(
      key, 0, 50, ranges[0].column(0), 50 * sizeof(int64_t), stream, mr));
  ASSERT_TRUE(cache.insertGpuColumnRangeIfAbsent(
      key, 50, 100, ranges[1].column(0), 50 * sizeof(int64_t), stream, mr));
  EXPECT_GT(cache.gpuBytes(), gpuBytesBefore);
  EXPECT_LE(cache.gpuBytes(), cache.maxGpuBytes());

  const auto statsBefore = cache.stats();
  std::vector<CudfDecodedColumnCache::ColumnRangeRequest> requests{
      {key, {{25, 75}}}};
  auto assembled = cache.materializeGpuColumnRanges(requests, stream, mr);
  ASSERT_EQ(assembled.size(), 1);
  ASSERT_NE(assembled.front(), nullptr);
  ASSERT_EQ(assembled.front()->size(), 50);
  std::vector<int64_t> actual(assembled.front()->size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      actual.data(),
      assembled.front()->view().data<int64_t>(),
      actual.size() * sizeof(int64_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], i + 25);
  }

  const auto statsAfter = cache.stats();
  EXPECT_EQ(statsAfter.gpuRestoreCalls - statsBefore.gpuRestoreCalls, 2);
  EXPECT_EQ(statsAfter.gpuRestoreBatches - statsBefore.gpuRestoreBatches, 1);

  requests.front().ranges = {{25, 125}};
  auto missing = cache.materializeGpuColumnRanges(requests, stream, mr);
  ASSERT_EQ(missing.size(), 1);
  EXPECT_EQ(missing.front(), nullptr);

  const auto gpuBytes = cache.gpuBytes();
  EXPECT_FALSE(cache.insertGpuColumnRangeIfAbsent(
      key, 0, 50, ranges[0].column(0), 50 * sizeof(int64_t), stream, mr));
  EXPECT_EQ(cache.gpuBytes(), gpuBytes);

  auto overCapKey = key;
  overCapKey.file.filePath = "gpu-over-cap";
  const auto rejectedBefore = cache.stats().gpuAdmissionRejectedRanges;
  EXPECT_FALSE(cache.insertGpuColumnRangeIfAbsent(
      std::move(overCapKey),
      0,
      50,
      ranges[0].column(0),
      cache.maxGpuBytes() + 1,
      stream,
      mr));
  EXPECT_EQ(cache.stats().gpuAdmissionRejectedRanges - rejectedBefore, 1);
}

TEST_F(CudfSplitReaderTest, compressedPinnedRangeCacheRoundTrip) {
  constexpr vector_size_t kRows = 1 << 16;
  auto input =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(kRows, [](auto row) {
                      return static_cast<int64_t>(row);
                    })});
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto cudfTable = with_arrow::toCudfTable(input, input->pool(), stream, mr);

  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  CudfDecodedColumnCache::ColumnKey key{
      .file = {.connectorId = "test", .filePath = "compressed-range"},
      .deviceId = deviceId,
      .columnName = "c0",
      .veloxType = BIGINT()->toString(),
      .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
      .usePandasMetadata = true,
      .useArrowSchema = true,
      .allowMismatchedSchemas = false,
  };

  auto& cache = CudfDecodedColumnCache::instance();
  ASSERT_TRUE(cache.insertColumnRangeIfAbsent(
      key,
      0,
      kRows,
      cudfTable->view().column(0),
      stream,
      mr,
      CudfDecodedColumnCache::CompressionMode::kColumnAdvanced));

  const auto afterInsert = cache.stats();
  EXPECT_EQ(afterInsert.insertedCompressedRanges, 1);
  EXPECT_EQ(afterInsert.insertedRawRanges, 0);
  EXPECT_EQ(afterInsert.compressionAttempts, 1);
  EXPECT_LT(
      afterInsert.insertedStoredBytes, afterInsert.insertedUncompressedBytes);

  auto restored = cache.materializeColumnRange(key, 0, kRows, stream, mr, mr);
  ASSERT_NE(restored, nullptr);
  ASSERT_EQ(restored->size(), kRows);
  std::vector<int64_t> actual(kRows);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      actual.data(),
      restored->view().data<int64_t>(),
      actual.size() * sizeof(int64_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  for (vector_size_t i = 0; i < kRows; ++i) {
    EXPECT_EQ(actual[i], i);
  }

  const auto afterRestore = cache.stats();
  EXPECT_EQ(afterRestore.restoreCalls, 1);
  EXPECT_EQ(afterRestore.restoredStoredBytes, afterInsert.insertedStoredBytes);
  EXPECT_EQ(
      afterRestore.restoredUncompressedBytes,
      afterInsert.insertedUncompressedBytes);
  EXPECT_GT(afterRestore.decompressionNanos, 0);
}

TEST_F(CudfSplitReaderTest, packedGpuCacheNullableMultiRangeAndHybrid) {
  gflags::FlagSaver flagSaver;
  FLAGS_cudf_gpu_cache_store_packed = true;
  FLAGS_cudf_decoded_cache_skip_full_range_slice = true;
  constexpr int32_t kRows = 8192;
  std::vector<std::optional<int64_t>> numbers;
  std::vector<std::optional<StringView>> strings;
  std::vector<std::optional<double>> doubles;
  for (int i = 0; i < kRows; ++i) {
    numbers.push_back(
        i % 13 == 0 ? std::nullopt : std::optional<int64_t>{i % 7});
    strings.push_back(
        i % 17 == 0
            ? std::nullopt
            : std::optional<StringView>{i % 2 ? "tiny" : "a longer string"});
    doubles.push_back(
        i % 19 == 0 ? std::nullopt : std::optional<double>{i % 11 * 1.01});
  }
  auto input = makeRowVector(
      {"numbers", "strings", "doubles", "smallstrings", "unique", "rare"},
      {makeNullableFlatVector<int64_t>(numbers),
       makeNullableFlatVector<StringView>(strings),
       makeNullableFlatVector<double>(doubles),
       makeFlatVector<std::string>(
           kRows,
           [](auto row) {
             return row % 3 == 0 ? std::string("\0", 1)
                 : row % 3 == 1  ? std::string("é")
                                 : std::string("");
           },
           nullEvery(23)),
       makeFlatVector<std::string>(
           kRows, [](auto row) { return std::to_string(row); }),
       makeFlatVector<std::string>(kRows, [](auto row) {
         return row % 4096 == 2048 ? std::string("outside-initial-sample")
                                   : std::string("common-long-label");
       })});
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto table = with_arrow::toCudfTable(input, input->pool(), stream, mr);
  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  auto& cache = CudfDecodedColumnCache::instance();
  for (const std::string types :
       {"all", "nonfloating", "integral", "smallstrings", "lowcardstrings"}) {
    FLAGS_cudf_gpu_cache_packed_types = types;
    for (uint64_t rawPrefix : {uint64_t{0}, uint64_t{8192}}) {
      cache.clearForTesting();
      FLAGS_cudf_gpu_cache_raw_prefix_bytes = rawPrefix;
      for (int columnIndex = 0; columnIndex < table->num_columns();
           ++columnIndex) {
        const auto packedRestoresBefore = cache.stats().gpuPackedRestoreCalls;
        auto column = table->view().column(columnIndex);
        CudfDecodedColumnCache::ColumnKey key{
            .file = {.connectorId = "test", .filePath = "packed-gpu-roundtrip"},
            .deviceId = deviceId,
            .columnName = std::to_string(columnIndex),
            .veloxType = input->childAt(columnIndex)->type()->toString(),
            .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
            .usePandasMetadata = true,
            .useArrowSchema = true,
            .allowMismatchedSchemas = false};
        for (int32_t first : {0, kRows / 2}) {
          const auto last = first + kRows / 2;
          auto piece = cudf::slice(column, {first, last}, stream).front();
          ASSERT_TRUE(cache.insertColumnRangeIfAbsent(
              key,
              first,
              last,
              piece,
              stream,
              mr,
              CudfDecodedColumnCache::CompressionMode::kColumn,
              mr));
        }
        for (const auto& ranges :
             std::vector<std::vector<std::pair<int64_t, int64_t>>>{
                 {{0, kRows}}, {{123, 6012}}, {{4000, 5000}, {10, 20}}}) {
          std::vector<cudf::column_view> expectedViews;
          for (const auto& [first, last] : ranges) {
            expectedViews.push_back(
                cudf::slice(
                    column,
                    {static_cast<cudf::size_type>(first),
                     static_cast<cudf::size_type>(last)},
                    stream)
                    .front());
          }
          auto expectedColumn = cudf::concatenate(expectedViews, stream, mr);
          auto expected = with_arrow::toVeloxColumn(
              cudf::table_view({expectedColumn->view()}),
              pool_.get(),
              "c",
              stream,
              mr);
          auto restored =
              cache.materializeGpuColumnRanges({{key, ranges}}, stream, mr);
          ASSERT_EQ(restored.size(), 1);
          ASSERT_NE(restored[0], nullptr);
          auto actual = with_arrow::toVeloxColumn(
              cudf::table_view({restored[0]->view()}),
              pool_.get(),
              "c",
              stream,
              mr);
          facebook::velox::test::assertEqualVectors(expected, actual);
        }
        if ((types == "nonfloating" && columnIndex == 2) ||
            (types == "integral" && columnIndex > 0) ||
            (types == "smallstrings" && columnIndex != 3) ||
            (types == "lowcardstrings" && columnIndex != 1 &&
             columnIndex != 3)) {
          EXPECT_EQ(cache.stats().gpuPackedRestoreCalls, packedRestoresBefore);
        }
      }
      const auto stats = cache.stats();
      EXPECT_GT(stats.gpuPackedInsertedBytes, 0);
      EXPECT_GT(stats.gpuPackedRestoreCalls, 0);
      EXPECT_GT(stats.gpuPackedDecompressionNanos, 0);
      EXPECT_LT(stats.gpuPackedInsertedBytes, stats.insertedUncompressedBytes);
      if (rawPrefix != 0) {
        EXPECT_GT(stats.gpuBytes, stats.gpuPackedInsertedBytes);
      }
      stream.synchronize();
    }
  }
}

TEST_F(CudfSplitReaderTest, fullRangeCacheViewsPreserveNullsAndStrings) {
  gflags::FlagSaver flagSaver;
  FLAGS_cudf_decoded_cache_skip_full_range_slice = true;
  auto input = makeRowVector(
      {"numbers", "strings"},
      {makeNullableFlatVector<int64_t>({1, std::nullopt, 3, 4, std::nullopt}),
       makeNullableFlatVector<StringView>(
           {"one", std::nullopt, "", "four", "a longer string"})});
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto table = with_arrow::toCudfTable(input, input->pool(), stream, mr);
  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  auto& cache = CudfDecodedColumnCache::instance();
  for (int columnIndex = 0; columnIndex < table->num_columns(); ++columnIndex) {
    auto column = table->view().column(columnIndex);
    CudfDecodedColumnCache::ColumnKey key{
        .file = {.connectorId = "test", .filePath = "full-range-nullable"},
        .deviceId = deviceId,
        .columnName = std::to_string(columnIndex),
        .veloxType = input->childAt(columnIndex)->type()->toString(),
        .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
        .usePandasMetadata = true,
        .useArrowSchema = true,
        .allowMismatchedSchemas = false,
    };
    ASSERT_TRUE(cache.insertColumnRangeIfAbsent(
        key,
        0,
        column.size(),
        column,
        stream,
        mr,
        CudfDecodedColumnCache::CompressionMode::kColumnAdvanced));
    ASSERT_TRUE(cache.insertGpuColumnRangeIfAbsent(
        key,
        0,
        column.size(),
        column,
        table->get_column(columnIndex).alloc_size(),
        stream,
        mr));
    for (const auto& [first, last] :
         std::vector<std::pair<int64_t, int64_t>>{{0, 5}, {1, 4}}) {
      auto expectedView = cudf::slice(
                              column,
                              {static_cast<cudf::size_type>(first),
                               static_cast<cudf::size_type>(last)},
                              stream)
                              .front();
      auto expected = with_arrow::toVeloxColumn(
          cudf::table_view({expectedView}), pool_.get(), "c", stream, mr);
      const auto check = [&](const cudf::column& restored) {
        auto actual = with_arrow::toVeloxColumn(
            cudf::table_view({restored.view()}), pool_.get(), "c", stream, mr);
        facebook::velox::test::assertEqualVectors(expected, actual);
      };
      auto single =
          cache.materializeColumnRange(key, first, last, stream, mr, mr);
      ASSERT_NE(single, nullptr);
      check(*single);
      std::vector<CudfDecodedColumnCache::ColumnRangeRequest> requests{
          {key, {{first, last}}}};
      auto batched =
          cache.materializeColumnRanges(requests, stream, stream, mr, mr);
      ASSERT_TRUE(batched.has_value());
      ASSERT_EQ(batched->size(), 1);
      check(*batched->front());
      auto gpu = cache.materializeGpuColumnRanges(requests, stream, mr);
      ASSERT_EQ(gpu.size(), 1);
      ASSERT_NE(gpu.front(), nullptr);
      check(*gpu.front());
      auto borrowed = cache.borrowGpuColumnRanges(requests, stream);
      ASSERT_NE(borrowed, nullptr);
      ASSERT_EQ(borrowed->views.size(), 1);
      auto borrowedActual = with_arrow::toVeloxColumn(
          cudf::table_view(borrowed->views), pool_.get(), "c", stream, mr);
      facebook::velox::test::assertEqualVectors(expected, borrowedActual);
    }
  }
}

TEST_F(CudfSplitReaderTest, scaledFloatGpuCacheIsBitExact) {
  gflags::FlagSaver restore;
  FLAGS_cudf_gpu_cache_scaled_float = true;
  FLAGS_cudf_gpu_cache_store_packed = false;
  FLAGS_cudf_decoded_cache_skip_full_range_slice = true;
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto inf = std::numeric_limits<double>::infinity();
  const std::vector<std::vector<std::optional<double>>> values{
      {1, 2, std::nullopt, 50, -4},
      {.01, .1, .07, .08, std::nullopt},
      {12345.67, 99999.99, -12345.67, std::nullopt, 0},
      {.125, 2.5, -.375, std::nullopt, 0},
      {std::sqrt(2.0), 3.141592653589793, 1e-15, std::nullopt, 1.0 / 3},
      {-0.0, 0.0, 1.0, std::nullopt, -1.0},
      {nan, inf, -inf, std::nullopt, 1.0},
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt}};
  const std::vector<bool> expectEncoded{
      true, true, true, true, false, false, false, false};
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  auto& cache = CudfDecodedColumnCache::instance();
  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  for (size_t index = 0; index < values.size(); ++index) {
    SCOPED_TRACE(index);
    auto input = makeRowVector({makeNullableFlatVector<double>(values[index])});
    auto table = with_arrow::toCudfTable(input, pool_.get(), stream, mr);
    CudfDecodedColumnCache::ColumnKey key{
        .file = {.connectorId = "test", .filePath = "scaled-float-bitexact"},
        .deviceId = deviceId,
        .columnName = std::to_string(index),
        .veloxType = "DOUBLE",
        .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
        .usePandasMetadata = true,
        .useArrowSchema = true,
        .allowMismatchedSchemas = false};
    const auto before = cache.stats();
    ASSERT_TRUE(cache.insertGpuColumnRangeIfAbsent(
        key,
        0,
        5,
        table->view().column(0),
        table->get_column(0).alloc_size(),
        stream,
        mr));
    EXPECT_EQ(
        cache.stats().gpuScaledInsertedRanges - before.gpuScaledInsertedRanges,
        expectEncoded[index] ? 1 : 0);
    auto borrowed = cache.borrowGpuColumnRanges({{key, {{0, 5}}}}, stream);
    EXPECT_EQ(borrowed == nullptr, expectEncoded[index]);
    auto mixedLease =
        cache.borrowGpuColumnRanges({{key, {{1, 4}}}}, stream, mr);
    ASSERT_NE(mixedLease, nullptr);
    ASSERT_EQ(mixedLease->views.size(), 1);
    EXPECT_EQ(mixedLease->decodedColumns.size(), expectEncoded[index] ? 1 : 0);
    auto mixedActual = with_arrow::toVeloxColumn(
        cudf::table_view(mixedLease->views), pool_.get(), "c", stream, mr);
    auto mixedFlat = mixedActual->childAt(0)->as<FlatVector<double>>();
    for (int row = 1; row < 4; ++row) {
      EXPECT_EQ(mixedFlat->isNullAt(row - 1), !values[index][row].has_value());
      if (values[index][row]) {
        EXPECT_EQ(
            std::bit_cast<uint64_t>(mixedFlat->valueAt(row - 1)),
            std::bit_cast<uint64_t>(*values[index][row]));
      }
    }
    for (const std::vector<std::pair<int64_t, int64_t>>& ranges :
         {std::vector<std::pair<int64_t, int64_t>>{{0, 5}},
          std::vector<std::pair<int64_t, int64_t>>{{1, 4}},
          std::vector<std::pair<int64_t, int64_t>>{{0, 2}, {3, 5}}}) {
      auto restored =
          cache.materializeGpuColumnRanges({{key, ranges}}, stream, mr);
      ASSERT_EQ(restored.size(), 1);
      ASSERT_NE(restored.front(), nullptr);
      EXPECT_EQ(restored.front()->type().id(), cudf::type_id::FLOAT64);
      auto actual = with_arrow::toVeloxColumn(
          cudf::table_view({restored.front()->view()}),
          pool_.get(),
          "c",
          stream,
          mr);
      auto flat = actual->childAt(0)->as<FlatVector<double>>();
      ASSERT_NE(flat, nullptr);
      int outputRow = 0;
      for (const auto& [first, last] : ranges) {
        for (int64_t row = first; row < last; ++row, ++outputRow) {
          EXPECT_EQ(flat->isNullAt(outputRow), !values[index][row].has_value());
          if (values[index][row]) {
            EXPECT_EQ(
                std::bit_cast<uint64_t>(flat->valueAt(outputRow)),
                std::bit_cast<uint64_t>(*values[index][row]));
          }
        }
      }
    }
  }
}

TEST_F(CudfSplitReaderTest, borrowedGpuCacheLeaseSurvivesClear) {
  gflags::FlagSaver flags;
  FLAGS_cudf_gpu_cache_store_packed = false;
  auto input =
      makeRowVector({makeNullableFlatVector<int64_t>({1, std::nullopt, 3})});
  rmm::cuda_stream consumer;
  auto stream = consumer.view();
  auto mr = cudf::get_current_device_resource_ref();
  auto table = with_arrow::toCudfTable(input, pool_.get(), stream, mr);
  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  CudfDecodedColumnCache::ColumnKey key{
      .file = {.connectorId = "test", .filePath = "borrowed-lease-clear"},
      .deviceId = deviceId,
      .columnName = "c0",
      .veloxType = "BIGINT",
      .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
      .usePandasMetadata = true,
      .useArrowSchema = true,
      .allowMismatchedSchemas = false};
  auto& cache = CudfDecodedColumnCache::instance();
  ASSERT_TRUE(cache.insertGpuColumnRangeIfAbsent(
      key,
      0,
      3,
      table->view().column(0),
      table->get_column(0).alloc_size(),
      stream,
      mr));
  EXPECT_EQ(cache.borrowGpuColumnRanges({{key, {{0, 4}}}}, stream), nullptr);
  EXPECT_EQ(
      cache.borrowGpuColumnRanges({{key, {{0, 1}, {2, 3}}}}, stream), nullptr);
  auto lease = cache.borrowGpuColumnRanges({{key, {{0, 3}}}}, stream);
  ASSERT_NE(lease, nullptr);
  auto leaseAgain = cache.borrowGpuColumnRanges({{key, {{0, 3}}}}, stream);
  ASSERT_NE(leaseAgain, nullptr);
  EXPECT_EQ(
      lease->views[0].head<int64_t>(), leaseAgain->views[0].head<int64_t>());
  leaseAgain.reset();
  cache.clearForTesting();
  auto copy = std::make_unique<cudf::column>(lease->views[0], stream, mr);
  // No explicit synchronization: lease destruction must fence the queued copy
  // before releasing the last reference to the cached allocation.
  lease.reset();
  auto actual = with_arrow::toVeloxColumn(
      cudf::table_view({copy->view()}), pool_.get(), "c", stream, mr);
  facebook::velox::test::assertEqualVectors(input, actual);
}

TEST_F(CudfSplitReaderTest, streamOrderedRawCacheLeaseSurvivesClear) {
  gflags::FlagSaver flags;
  FLAGS_cudf_gpu_cache_store_packed = false;
  FLAGS_cudf_gpu_cache_scaled_float = false;
  FLAGS_cudf_cache_stream_ordered_release = true;
  constexpr vector_size_t kRows = 65536;
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, folly::identity, nullEvery(11))});
  rmm::cuda_stream allocation;
  rmm::cuda_stream consumer;
  const auto stream = consumer.view();
  const auto mr = cudf::get_current_device_resource_ref();
  auto table =
      with_arrow::toCudfTable(input, pool_.get(), allocation.view(), mr);
  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  CudfDecodedColumnCache::ColumnKey key{
      .file = {.connectorId = "test", .filePath = "stream-ordered-lease"},
      .deviceId = deviceId,
      .columnName = "c0",
      .veloxType = "BIGINT",
      .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
      .usePandasMetadata = true,
      .useArrowSchema = true,
      .allowMismatchedSchemas = false};
  auto& cache = CudfDecodedColumnCache::instance();
  ASSERT_TRUE(cache.insertGpuColumnRangeIfAbsent(
      key,
      0,
      kRows,
      table->view().column(0),
      table->get_column(0).alloc_size(),
      allocation.view(),
      mr));
  auto lease = cache.borrowGpuColumnRanges({{key, {{0, kRows}}}}, stream);
  ASSERT_NE(lease, nullptr);
  ASSERT_TRUE(lease->orderRelease);
  cache.clearForTesting();
  auto copy = std::make_unique<cudf::column>(lease->views[0], stream, mr);
  lease.reset();
  // Event ordering, not a host-side fence, must protect the outstanding copy.
  auto actual = with_arrow::toVeloxColumn(
      cudf::table_view({copy->view()}), pool_.get(), "c", stream, mr);
  facebook::velox::test::assertEqualVectors(input, actual);
  allocation.view().synchronize();
}

TEST_F(CudfSplitReaderTest, compositeGpuHostCacheLease) {
  gflags::FlagSaver restore;
  FLAGS_cudf_gpu_cache_scaled_float = true;
  FLAGS_cudf_gpu_cache_store_packed = false;
  auto input = makeRowVector(
      {makeNullableFlatVector<int64_t>({1, std::nullopt, 3, 4, 5, 6}),
       makeNullableFlatVector<double>({.01, .10, std::nullopt, .50, .07, -.02}),
       makeNullableFlatVector<std::string>(
           {"a",
            std::nullopt,
            "b",
            "c",
            std::string("long\0tail", 9),
            "last"})});
  rmm::cuda_stream consumer;
  rmm::cuda_stream transfer;
  const auto stream = consumer.view();
  const auto mr = cudf::get_current_device_resource_ref();
  auto table = with_arrow::toCudfTable(input, pool_.get(), stream, mr);
  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  auto& cache = CudfDecodedColumnCache::instance();
  std::vector<CudfDecodedColumnCache::ColumnKey> keys;
  for (int index = 0; index < 3; ++index) {
    keys.push_back(
        {.file = {.connectorId = "test", .filePath = "composite-lease"},
         .deviceId = deviceId,
         .columnName = std::to_string(index),
         .veloxType = input->type()->childAt(index)->toString(),
         .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
         .usePandasMetadata = true,
         .useArrowSchema = true,
         .allowMismatchedSchemas = false});
    if (index < 2) {
      ASSERT_TRUE(cache.insertGpuColumnRangeIfAbsent(
          keys.back(),
          0,
          6,
          table->view().column(index),
          table->get_column(index).alloc_size(),
          stream,
          mr));
    } else {
      ASSERT_TRUE(cache.insertColumnRangeIfAbsent(
          keys.back(),
          0,
          6,
          table->view().column(index),
          stream,
          mr,
          CudfDecodedColumnCache::CompressionMode::kColumn));
    }
  }
  EXPECT_EQ(
      cache.borrowOrRestoreColumnRanges(
          {{keys[2], {{0, 7}}}}, stream, transfer.view(), mr, mr),
      nullptr);
  std::unique_ptr<CudfDecodedColumnCache::BorrowedGpuColumns> lease;
  for (const std::vector<std::pair<int64_t, int64_t>>& ranges :
       {std::vector<std::pair<int64_t, int64_t>>{{0, 6}},
        std::vector<std::pair<int64_t, int64_t>>{{1, 5}},
        std::vector<std::pair<int64_t, int64_t>>{{0, 2}, {4, 6}}}) {
    std::vector<CudfDecodedColumnCache::ColumnRangeRequest> requests;
    for (const auto& key : keys) {
      requests.push_back({key, ranges});
    }
    lease = cache.borrowOrRestoreColumnRanges(
        requests, stream, transfer.view(), mr, mr);
    ASSERT_NE(lease, nullptr);
    EXPECT_EQ(lease->gpuColumns, 2);
    EXPECT_EQ(lease->decodedColumns.size(), ranges.size() == 1 ? 2 : 3);
    if (ranges.size() == 1) {
      auto raw = cache.borrowGpuColumnRanges({requests[0]}, stream);
      ASSERT_NE(raw, nullptr);
      EXPECT_EQ(lease->views[0].head<int64_t>(), raw->views[0].head<int64_t>());
    }
    std::vector<cudf::table_view> slices;
    for (const auto& [first, last] : ranges) {
      slices.push_back(
          cudf::slice(
              table->view(),
              {static_cast<cudf::size_type>(first),
               static_cast<cudf::size_type>(last)},
              stream)
              .front());
    }
    auto expectedTable = cudf::concatenate(slices, stream, mr);
    auto expected = with_arrow::toVeloxColumn(
        expectedTable->view(),
        pool_.get(),
        asRowType(input->type()),
        stream,
        mr);
    auto actual = with_arrow::toVeloxColumn(
        cudf::table_view(lease->views),
        pool_.get(),
        asRowType(input->type()),
        stream,
        mr);
    facebook::velox::test::assertEqualVectors(expected, actual);
  }
  cache.clearForTesting();
  auto afterClear =
      std::make_unique<cudf::table>(cudf::table_view(lease->views), stream, mr);
  lease.reset(); // Last-owner fence protects this outstanding read on the
                 // stream.
  EXPECT_EQ(afterClear->num_rows(), 4);
}

TEST_F(CudfSplitReaderTest, restoredPackedViewsLease) {
  gflags::FlagSaver restore;
  FLAGS_cudf_cache_restore_packed_views = true;
  constexpr vector_size_t kRows = 8192;
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(
           kRows, [](auto row) { return row % 7; }, nullEvery(11)),
       makeFlatVector<double>(
           kRows, [](auto row) { return (row % 10) * .01; }, nullEvery(13)),
       makeFlatVector<std::string>(
           kRows,
           [](auto row) {
             return row % 3 ? std::string("abc\0def", 7)
                            : std::string(100, 'x');
           },
           nullEvery(17))});
  rmm::cuda_stream consumer;
  rmm::cuda_stream transfer;
  const auto stream = consumer.view();
  const auto mr = cudf::get_current_device_resource_ref();
  auto table = with_arrow::toCudfTable(input, pool_.get(), stream, mr);
  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  auto& cache = CudfDecodedColumnCache::instance();
  for (auto mode :
       {CudfDecodedColumnCache::CompressionMode::kNone,
        CudfDecodedColumnCache::CompressionMode::kColumn}) {
    cache.clearForTesting();
    std::vector<CudfDecodedColumnCache::ColumnKey> keys;
    for (int index = 0; index < 3; ++index) {
      keys.push_back(
          {.file = {.connectorId = "test", .filePath = "restored-views"},
           .deviceId = deviceId,
           .columnName = std::to_string(index),
           .veloxType = input->type()->childAt(index)->toString(),
           .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
           .usePandasMetadata = true,
           .useArrowSchema = true,
           .allowMismatchedSchemas = false});
      ASSERT_TRUE(cache.insertColumnRangeIfAbsent(
          keys.back(),
          0,
          kRows,
          table->view().column(index),
          stream,
          mr,
          mode));
    }
    EXPECT_EQ(
        cache.restoreColumnRangeViews(
            {{keys[0], {{0, kRows + 1}}}}, stream, transfer.view(), mr, mr),
        nullptr);
    EXPECT_EQ(
        cache.restoreColumnRangeViews(
            {{keys[0], {}}}, stream, transfer.view(), mr, mr),
        nullptr);
    for (const auto& ranges :
         {std::vector<std::pair<int64_t, int64_t>>{{0, kRows}},
          std::vector<std::pair<int64_t, int64_t>>{{1, kRows - 1}},
          std::vector<std::pair<int64_t, int64_t>>{{0, 100}, {1000, 1400}}}) {
      std::vector<CudfDecodedColumnCache::ColumnRangeRequest> requests;
      for (const auto& key : keys) {
        requests.push_back({key, ranges});
      }
      auto lease = cache.borrowOrRestoreColumnRanges(
          requests, stream, transfer.view(), mr, mr);
      ASSERT_NE(lease, nullptr);
      EXPECT_EQ(lease->gpuColumns, 0);
      EXPECT_EQ(lease->decodedBuffers.size(), ranges.size() == 1 ? 3 : 0);
      EXPECT_EQ(lease->decodedColumns.size(), ranges.size() == 1 ? 0 : 3);
      std::vector<cudf::table_view> slices;
      for (const auto& [first, last] : ranges) {
        slices.push_back(
            cudf::slice(
                table->view(),
                {static_cast<cudf::size_type>(first),
                 static_cast<cudf::size_type>(last)},
                stream)
                .front());
      }
      auto expectedTable = cudf::concatenate(slices, stream, mr);
      auto expected = with_arrow::toVeloxColumn(
          expectedTable->view(),
          pool_.get(),
          asRowType(input->type()),
          stream,
          mr);
      auto actual = with_arrow::toVeloxColumn(
          cudf::table_view(lease->views),
          pool_.get(),
          asRowType(input->type()),
          stream,
          mr);
      facebook::velox::test::assertEqualVectors(expected, actual);
    }
    auto lease = cache.borrowOrRestoreColumnRanges(
        {{keys[0], {{1, kRows - 1}}},
         {keys[1], {{1, kRows - 1}}},
         {keys[2], {{1, kRows - 1}}}},
        stream,
        transfer.view(),
        mr,
        mr);
    ASSERT_NE(lease, nullptr);
    cache.clearForTesting();
    auto afterClear = std::make_unique<cudf::table>(
        cudf::table_view(lease->views), stream, mr);
    lease.reset();
    auto actual = with_arrow::toVeloxColumn(
        afterClear->view(), pool_.get(), asRowType(input->type()), stream, mr);
    auto expected = with_arrow::toVeloxColumn(
        cudf::slice(table->view(), {1, kRows - 1}, stream).front(),
        pool_.get(),
        asRowType(input->type()),
        stream,
        mr);
    facebook::velox::test::assertEqualVectors(expected, actual);
  }
}

TEST_F(CudfSplitReaderTest, gpuAdmissionHostFractionPreservesFallback) {
  gflags::FlagSaver restore;
  FLAGS_cudf_gpu_cache_store_packed = false;
  FLAGS_cudf_gpu_cache_scaled_float = false;
  constexpr vector_size_t kRows = 65536;
  auto input = makeRowVector({makeFlatVector<int64_t>(
      kRows, [](auto row) { return row % 7; }, nullEvery(11))});
  // Unspecified NULL payloads must not leak into neighboring valid values.
  // Poison them deterministically instead of relying on allocator contents.
  auto* rawValues =
      input->childAt(0)->asFlatVector<int64_t>()->mutableRawValues();
  for (vector_size_t row = 0; row < kRows; ++row) {
    if (input->childAt(0)->isNullAt(row)) {
      rawValues[row] = 0x7878787878787878LL;
    }
  }
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  auto table = with_arrow::toCudfTable(input, pool_.get(), stream, mr);
  auto imported = with_arrow::toVeloxColumn(
      table->view(), pool_.get(), asRowType(input->type()), stream, mr);
  facebook::velox::test::assertEqualVectors(input, imported);
  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  CudfDecodedColumnCache::ColumnKey key{
      .file = {.connectorId = "test", .filePath = "host-fraction-policy"},
      .deviceId = deviceId,
      .columnName = "c0",
      .veloxType = BIGINT()->toString(),
      .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
      .usePandasMetadata = true,
      .useArrowSchema = true,
      .allowMismatchedSchemas = false};
  auto& cache = CudfDecodedColumnCache::instance();
  for (const auto mode :
       {CudfDecodedColumnCache::CompressionMode::kNone,
        CudfDecodedColumnCache::CompressionMode::kColumn}) {
    for (const double fraction : {0.0, 0.75}) {
      SCOPED_TRACE(
          fmt::format("mode={} fraction={}", static_cast<int>(mode), fraction));
      cache.clearForTesting();
      FLAGS_cudf_gpu_cache_min_host_fraction = fraction;
      ASSERT_TRUE(cache.insertColumnRangeIfAbsent(
          key, 0, kRows, table->view().column(0), stream, mr, mode, mr));
      const auto host = cache.findColumnRanges(key, 0, kRows);
      ASSERT_TRUE(host.has_value());
      ASSERT_EQ(host->size(), 1);
      const bool skip =
          mode == CudfDecodedColumnCache::CompressionMode::kColumn &&
          fraction > 0;
      if (mode == CudfDecodedColumnCache::CompressionMode::kColumn) {
        ASSERT_TRUE(host->front().chunk->compressed());
        ASSERT_LT(
            host->front().chunk->packedSize(),
            host->front().chunk->uncompressedPackedSize() * 0.75);
      }
      auto gpu =
          cache.materializeGpuColumnRanges({{key, {{0, kRows}}}}, stream, mr);
      ASSERT_EQ(gpu.size(), 1);
      EXPECT_EQ(gpu[0] == nullptr, skip);
      if (gpu[0]) {
        auto gpuActual = with_arrow::toVeloxColumn(
            cudf::table_view({gpu[0]->view()}),
            pool_.get(),
            asRowType(input->type()),
            stream,
            mr);
        facebook::velox::test::assertEqualVectors(input, gpuActual);
      }
      EXPECT_EQ(cache.stats().gpuAdmissionPolicySkippedRanges, skip ? 1 : 0);
      EXPECT_EQ(cache.stats().gpuAdmissionRejectedRanges, 0);
      auto restored =
          cache.materializeColumnRange(key, 0, kRows, stream, mr, mr);
      ASSERT_NE(restored, nullptr);
      auto actual = with_arrow::toVeloxColumn(
          cudf::table_view({restored->view()}),
          pool_.get(),
          asRowType(input->type()),
          stream,
          mr);
      facebook::velox::test::assertEqualVectors(input, actual);

      if (skip) {
        // Host coverage for a larger range is not an exact metadata pair.
        const auto half =
            cudf::slice(table->view().column(0), {0, kRows / 2}, stream)
                .front();
        EXPECT_TRUE(cache.insertGpuColumnRangeIfAbsent(
            key, 0, kRows / 2, half, kRows * sizeof(int64_t) / 2, stream, mr));
      }
      auto standalone = key;
      standalone.columnName = "standalone";
      EXPECT_TRUE(cache.insertGpuColumnRangeIfAbsent(
          standalone,
          0,
          kRows,
          table->view().column(0),
          kRows * sizeof(int64_t),
          stream,
          mr));
      EXPECT_EQ(cache.stats().gpuAdmissionPolicySkippedRanges, skip ? 1 : 0);
    }
  }
  cache.clearForTesting();
  FLAGS_cudf_gpu_cache_min_host_fraction = 1.01;
  EXPECT_THROW(
      cache.insertGpuColumnRangeIfAbsent(
          key,
          0,
          kRows,
          table->view().column(0),
          kRows * sizeof(int64_t),
          stream,
          mr),
      VeloxUserError);
}

TEST_F(CudfSplitReaderTest, compressedIntegerWideDomainRoundTrip) {
  gflags::FlagSaver restore;
  FLAGS_cudf_gpu_cache_min_host_fraction = 0;
  constexpr vector_size_t kRows = 65536;
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  int deviceId = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&deviceId));
  auto& cache = CudfDecodedColumnCache::instance();
  const auto run = [&]<typename Key>() {
    for (int pattern : {0, 1}) {
      SCOPED_TRACE(fmt::format("keyBytes={} pattern={}", sizeof(Key), pattern));
      auto input =
          makeRowVector({makeFlatVector<Key>(kRows, [=](auto row) -> Key {
            const auto low = std::numeric_limits<Key>::min();
            const auto high = std::numeric_limits<Key>::max();
            if (pattern == 1) {
              // A short modulo ramp across the signed boundary selects delta
              // compression and must reconstruct without signed overflow.
              return static_cast<Key>(
                  static_cast<uint64_t>(high) - 31 + row % 64);
            }
            switch (row % 6) {
              case 0:
                return low;
              case 1:
                return high;
              case 2:
                return -1;
              case 3:
                return 0;
              case 4:
                return 1;
              default:
                return static_cast<Key>(0x7878787878787878ULL);
            }
          })});
      auto table = with_arrow::toCudfTable(input, pool_.get(), stream, mr);
      for (const auto mode :
           {CudfDecodedColumnCache::CompressionMode::kColumn,
            CudfDecodedColumnCache::CompressionMode::kColumnAdvanced}) {
        SCOPED_TRACE(fmt::format("mode={}", static_cast<int>(mode)));
        cache.clearForTesting();
        CudfDecodedColumnCache::ColumnKey key{
            .file = {.connectorId = "test", .filePath = "wide-integer-codec"},
            .deviceId = deviceId,
            .columnName = "c0",
            .veloxType = input->type()->childAt(0)->toString(),
            .timestampType = cudf::type_id::TIMESTAMP_MILLISECONDS,
            .usePandasMetadata = true,
            .useArrowSchema = true,
            .allowMismatchedSchemas = false};
        ASSERT_TRUE(cache.insertColumnRangeIfAbsent(
            key, 0, kRows, table->view().column(0), stream, mr, mode));
        const auto coverage = cache.findColumnRanges(key, 0, kRows);
        ASSERT_TRUE(coverage && coverage->size() == 1);
        EXPECT_TRUE(coverage->front().chunk->compressed());
        auto column =
            cache.materializeColumnRange(key, 0, kRows, stream, mr, mr);
        ASSERT_NE(column, nullptr);
        auto actual = with_arrow::toVeloxColumn(
            cudf::table_view({column->view()}),
            pool_.get(),
            asRowType(input->type()),
            stream,
            mr);
        facebook::velox::test::assertEqualVectors(input, actual);
        auto batched = cache.materializeColumnRanges(
            {{key, {{0, kRows}}}}, stream, stream, mr, mr);
        ASSERT_TRUE(batched && batched->size() == 1);
        actual = with_arrow::toVeloxColumn(
            cudf::table_view({batched->front()->view()}),
            pool_.get(),
            asRowType(input->type()),
            stream,
            mr);
        facebook::velox::test::assertEqualVectors(input, actual);
      }
    }
  };
  run.template operator()<int32_t>();
  run.template operator()<int64_t>();
  cache.clearForTesting();
}

TEST_F(CudfSplitReaderTest, readerPrefersGpuTierOverPinnedTier) {
  constexpr vector_size_t kRows = 32;
  auto fileRowType = ROW({"c0"}, {BIGINT()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(kRows, folly::identity)}));

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {CudfHiveConfig::kUseExperimentalCudfReader, "true"},
          {CudfHiveConfig::kExperimentalDecodedColumnCacheEnabled, "true"},
          {CudfHiveConfig::kExperimentalDecodedColumnGpuCacheEnabled, "true"},
          {CudfHiveConfig::kImmutableFiles, "true"},
      });
  ::facebook::velox::connector::ConnectorQueryCtx connectorQueryCtx(
      pool_.get(),
      pool_.get(),
      properties.get(),
      nullptr,
      common::PrefixSortConfig{},
      nullptr,
      nullptr,
      "query.CudfSplitReaderTest",
      "task.CudfSplitReaderTest",
      "plan.CudfSplitReaderTest",
      0,
      "");
  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto tableHandle =
      CudfHiveConnectorTestBase::makeTableHandle("parquet_table", fileRowType);

  struct ReadResult {
    uint64_t hits;
    uint64_t gpuHits;
    uint64_t misses;
    uint64_t decodeCalls;
    uint64_t cpuRestoreBatches;
    uint64_t gpuRestoreBatches;
    size_t rows;
  };
  auto read = [&]() {
    auto split =
        CudfHiveConnectorSplitBuilder(dataFile->getPath())
            .connectorId(
                ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
            .build();
    CudfSplitReader reader(
        std::move(split),
        tableHandle,
        fileRowType,
        {"c0"},
        &fileHandleFactory,
        ioExecutor_.get(),
        &connectorQueryCtx,
        std::make_shared<CudfHiveConfig>(properties),
        std::make_shared<io::IoStatistics>(),
        std::make_shared<IoStats>(),
        true,
        nullptr);
    dwio::common::RuntimeStats runtimeStats;
    reader.prepareSplit(runtimeStats);
    const auto statsBefore = CudfDecodedColumnCache::instance().stats();
    size_t rows = 0;
    while (auto chunk = reader.next(0)) {
      rows += chunk.value()->num_rows();
      reader.stream().synchronize();
    }
    const auto statsAfter = CudfDecodedColumnCache::instance().stats();
    return ReadResult{
        .hits = reader.decodedColumnCacheHits(),
        .gpuHits = reader.decodedColumnGpuCacheHits(),
        .misses = reader.decodedColumnCacheMisses(),
        .decodeCalls = reader.decodedColumnCacheDecodeCalls(),
        .cpuRestoreBatches = statsAfter.pipelinedRestoreBatches -
            statsBefore.pipelinedRestoreBatches,
        .gpuRestoreBatches =
            statsAfter.gpuRestoreBatches - statsBefore.gpuRestoreBatches,
        .rows = rows,
    };
  };

  const auto first = read();
  EXPECT_EQ(first.hits, 0);
  EXPECT_EQ(first.gpuHits, 0);
  EXPECT_EQ(first.misses, 1);
  EXPECT_EQ(first.decodeCalls, 1);
  EXPECT_EQ(first.rows, kRows);
  EXPECT_GT(CudfDecodedColumnCache::instance().gpuBytes(), 0);

  ASSERT_TRUE(std::filesystem::remove(dataFile->getPath()));
  const auto second = read();
  EXPECT_EQ(second.hits, 1);
  EXPECT_EQ(second.gpuHits, 1);
  EXPECT_EQ(second.misses, 0);
  EXPECT_EQ(second.decodeCalls, 0);
  EXPECT_EQ(second.cpuRestoreBatches, 0);
  EXPECT_EQ(second.gpuRestoreBatches, 1);
  EXPECT_EQ(second.rows, kRows);
}

TEST_F(CudfSplitReaderTest, batchesDecodedColumnsAcrossFileRowGroups) {
  constexpr cudf::size_type kRowsPerRowGroup = 4;
  constexpr cudf::size_type kNumRowGroups = 3;
  auto fileRowType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), BIGINT()});
  auto input = makeRowVector(
      {"c0", "c1", "c2"},
      {makeFlatVector<int64_t>(12, folly::identity),
       makeFlatVector<int64_t>(12, [](auto row) { return 100 + row; }),
       makeFlatVector<int64_t>(12, [](auto row) { return 200 + row; })});
  auto dataFile = common::testutil::TempFilePath::create();

  auto stream = cudf::get_default_stream();
  auto cudfTable = with_arrow::toCudfTable(
      input, input->pool(), stream, cudf::get_current_device_resource_ref());
  cudf::io::table_input_metadata metadata(cudfTable->view());
  for (size_t i = 0; i < fileRowType->size(); ++i) {
    metadata.column_metadata[i].set_name(fileRowType->nameOf(i));
  }
  auto writerOptions =
      cudf::io::parquet_writer_options::builder(
          cudf::io::sink_info{dataFile->getPath()}, cudfTable->view())
          .metadata(std::move(metadata))
          .row_group_size_rows(kRowsPerRowGroup)
          .max_page_size_rows(kRowsPerRowGroup)
          .max_page_fragment_size(kRowsPerRowGroup)
          .build();
  cudf::io::write_parquet(writerOptions, stream);
  stream.synchronize();

  auto sources =
      cudf::io::make_datasources(cudf::io::source_info{dataFile->getPath()});
  auto fileMetadata = cudf::io::read_parquet_footers(sources);
  ASSERT_EQ(fileMetadata.size(), 1);
  ASSERT_EQ(fileMetadata.front().row_groups.size(), kNumRowGroups);
  auto rowGroupOffset = [](const cudf::io::parquet::RowGroup& rowGroup) {
    if (rowGroup.file_offset.has_value()) {
      return static_cast<uint64_t>(rowGroup.file_offset.value());
    }
    if (rowGroup.columns.front().file_offset != 0) {
      return static_cast<uint64_t>(rowGroup.columns.front().file_offset);
    }
    const auto& columnMetadata = rowGroup.columns.front().meta_data;
    return static_cast<uint64_t>(
        columnMetadata.dictionary_page_offset != 0
            ? std::min(
                  columnMetadata.dictionary_page_offset,
                  columnMetadata.data_page_offset)
            : columnMetadata.data_page_offset);
  };
  const auto middleRowGroupOffset =
      rowGroupOffset(fileMetadata.front().row_groups[1]);
  sources.clear();
  fileMetadata.clear();

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {CudfHiveConfig::kUseExperimentalCudfReader, "true"},
          {CudfHiveConfig::kExperimentalDecodedColumnCacheEnabled, "true"},
          {CudfHiveConfig::kImmutableFiles, "true"},
      });
  ::facebook::velox::connector::ConnectorQueryCtx connectorQueryCtx(
      pool_.get(),
      pool_.get(),
      properties.get(),
      nullptr,
      common::PrefixSortConfig{},
      nullptr,
      nullptr,
      "query.CudfSplitReaderTest",
      "task.CudfSplitReaderTest",
      "plan.CudfSplitReaderTest",
      0,
      "");
  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto tableHandle =
      CudfHiveConnectorTestBase::makeTableHandle("parquet_table", fileRowType);

  struct ReadResult {
    uint64_t hits;
    uint64_t misses;
    uint64_t decodeCalls;
    size_t chunks;
    size_t rows;
    bool fullyCached;
    bool metadataFastPath;
    uint64_t restoreBatches;
  };
  auto read = [&](std::vector<std::string> names,
                  RowTypePtr outputType,
                  uint64_t start,
                  uint64_t length) {
    auto split =
        CudfHiveConnectorSplitBuilder(dataFile->getPath())
            .connectorId(
                ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
            .start(start)
            .length(length)
            .build();
    CudfSplitReader reader(
        std::move(split),
        tableHandle,
        outputType,
        names,
        &fileHandleFactory,
        ioExecutor_.get(),
        &connectorQueryCtx,
        std::make_shared<CudfHiveConfig>(properties),
        std::make_shared<io::IoStatistics>(),
        std::make_shared<IoStats>(),
        true,
        nullptr);
    dwio::common::RuntimeStats runtimeStats;
    reader.prepareSplit(runtimeStats);
    const auto fullyCached = reader.isFullyDecodedColumnCacheHit();
    const auto metadataFastPath =
        reader.usedDecodedColumnCacheMetadataFastPath();
    const auto statsBefore = CudfDecodedColumnCache::instance().stats();
    size_t chunks = 0;
    size_t rows = 0;
    while (auto chunk = reader.next(0)) {
      ++chunks;
      rows += chunk.value()->num_rows();
      reader.stream().synchronize();
    }
    const auto statsAfter = CudfDecodedColumnCache::instance().stats();
    return ReadResult{
        reader.decodedColumnCacheHits(),
        reader.decodedColumnCacheMisses(),
        reader.decodedColumnCacheDecodeCalls(),
        chunks,
        rows,
        fullyCached,
        metadataFastPath,
        statsAfter.pipelinedRestoreBatches -
            statsBefore.pipelinedRestoreBatches};
  };

  // Warm c0 and c1 together across all three row groups. The cold read must
  // issue one multi-column, multi-row-group cuDF decode.
  auto first = read(
      {"c0", "c1"},
      ROW({"c0", "c1"}, {BIGINT(), BIGINT()}),
      0,
      std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(first.hits, 0);
  EXPECT_EQ(first.misses, 2);
  EXPECT_EQ(first.decodeCalls, 1);
  EXPECT_EQ(first.chunks, 1);
  EXPECT_EQ(first.rows, kRowsPerRowGroup * kNumRowGroups);
  EXPECT_FALSE(first.fullyCached);
  EXPECT_TRUE(first.metadataFastPath);
  EXPECT_EQ(first.restoreBatches, 0);

  // Select only the middle row group and overlap on c1. c1 hits while c2 is
  // decoded and cached, demonstrating independent column and row-group reuse.
  auto second = read(
      {"c1", "c2"},
      ROW({"c1", "c2"}, {BIGINT(), BIGINT()}),
      middleRowGroupOffset,
      1);
  EXPECT_EQ(second.hits, 1);
  EXPECT_EQ(second.misses, 1);
  EXPECT_EQ(second.decodeCalls, 1);
  EXPECT_EQ(second.chunks, 1);
  EXPECT_EQ(second.rows, kRowsPerRowGroup);
  EXPECT_FALSE(second.fullyCached);
  EXPECT_FALSE(second.metadataFastPath);
  EXPECT_EQ(second.restoreBatches, 1);

  // A full hit must not open the file for its footer or column data.
  ASSERT_TRUE(std::filesystem::remove(dataFile->getPath()));
  auto third = read(
      {"c0", "c1"},
      ROW({"c0", "c1"}, {BIGINT(), BIGINT()}),
      0,
      std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(third.hits, 2);
  EXPECT_EQ(third.misses, 0);
  EXPECT_EQ(third.decodeCalls, 0);
  EXPECT_EQ(third.chunks, 1);
  EXPECT_EQ(third.rows, kRowsPerRowGroup * kNumRowGroups);
  EXPECT_TRUE(third.fullyCached);
  EXPECT_TRUE(third.metadataFastPath);
  EXPECT_EQ(third.restoreBatches, 1);

  // A cached byte-range hit retains the hybrid-reader selection path.
  auto fourth = read(
      {"c1", "c2"},
      ROW({"c1", "c2"}, {BIGINT(), BIGINT()}),
      middleRowGroupOffset,
      1);
  EXPECT_EQ(fourth.hits, 2);
  EXPECT_EQ(fourth.misses, 0);
  EXPECT_EQ(fourth.decodeCalls, 0);
  EXPECT_EQ(fourth.chunks, 1);
  EXPECT_EQ(fourth.rows, kRowsPerRowGroup);
  EXPECT_TRUE(fourth.fullyCached);
  EXPECT_FALSE(fourth.metadataFastPath);
  EXPECT_EQ(fourth.restoreBatches, 1);
}

TEST_F(CudfSplitReaderTest, cachesStatsPrunedNonContiguousRowGroups) {
  constexpr cudf::size_type kRowsPerRowGroup = 4;
  auto fileRowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto input = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>(
           12,
           [](auto row) {
             if (row < 4) {
               return static_cast<int64_t>(row);
             }
             if (row < 8) {
               return static_cast<int64_t>(100 + row);
             }
             return static_cast<int64_t>(row - 4);
           }),
       makeFlatVector<int64_t>(
           12, [](auto row) { return static_cast<int64_t>(1'000 + row); })});
  auto dataFile = common::testutil::TempFilePath::create();

  auto stream = cudf::get_default_stream();
  auto cudfTable = with_arrow::toCudfTable(
      input, input->pool(), stream, cudf::get_current_device_resource_ref());
  cudf::io::table_input_metadata metadata(cudfTable->view());
  for (size_t i = 0; i < fileRowType->size(); ++i) {
    metadata.column_metadata[i].set_name(fileRowType->nameOf(i));
  }
  auto writerOptions =
      cudf::io::parquet_writer_options::builder(
          cudf::io::sink_info{dataFile->getPath()}, cudfTable->view())
          .metadata(std::move(metadata))
          .row_group_size_rows(kRowsPerRowGroup)
          .max_page_size_rows(kRowsPerRowGroup)
          .max_page_fragment_size(kRowsPerRowGroup)
          .build();
  cudf::io::write_parquet(writerOptions, stream);
  stream.synchronize();

  auto subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add("c0", std::make_unique<common::BigintRange>(0, 9, false))
          .build();
  cudf::ast::tree filterTree;
  std::vector<std::unique_ptr<cudf::scalar>> filterScalars;
  const auto& filterExpr = createAstFromSubfieldFilters(
      subfieldFilters, filterTree, filterScalars, fileRowType);

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {CudfHiveConfig::kUseExperimentalCudfReader, "true"},
          {CudfHiveConfig::kExperimentalDecodedColumnCacheEnabled, "true"},
          {CudfHiveConfig::kImmutableFiles, "true"},
      });
  ::facebook::velox::connector::ConnectorQueryCtx connectorQueryCtx(
      pool_.get(),
      pool_.get(),
      properties.get(),
      nullptr,
      common::PrefixSortConfig{},
      nullptr,
      nullptr,
      "query.CudfSplitReaderTest",
      "task.CudfSplitReaderTest",
      "plan.CudfSplitReaderTest",
      0,
      "");
  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto tableHandle =
      CudfHiveConnectorTestBase::makeTableHandle("parquet_table", fileRowType);

  struct ReadResult {
    uint64_t hits;
    uint64_t misses;
    uint64_t decodeCalls;
    size_t chunks;
    size_t rows;
    bool fullyCached;
    bool metadataFastPath;
  };
  auto read = [&]() {
    auto split =
        CudfHiveConnectorSplitBuilder(dataFile->getPath())
            .connectorId(
                ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
            .build();
    CudfSplitReader reader(
        std::move(split),
        tableHandle,
        fileRowType,
        {"c0", "c1"},
        &fileHandleFactory,
        ioExecutor_.get(),
        &connectorQueryCtx,
        std::make_shared<CudfHiveConfig>(properties),
        std::make_shared<io::IoStatistics>(),
        std::make_shared<IoStats>(),
        true,
        &filterExpr,
        "c0=0..9");
    dwio::common::RuntimeStats runtimeStats;
    reader.prepareSplit(runtimeStats);
    const auto fullyCached = reader.isFullyDecodedColumnCacheHit();
    const auto metadataFastPath =
        reader.usedDecodedColumnCacheMetadataFastPath();
    size_t chunks = 0;
    size_t rows = 0;
    while (auto chunk = reader.next(0)) {
      ++chunks;
      rows += chunk.value()->num_rows();
      reader.stream().synchronize();
    }
    return ReadResult{
        reader.decodedColumnCacheHits(),
        reader.decodedColumnCacheMisses(),
        reader.decodedColumnCacheDecodeCalls(),
        chunks,
        rows,
        fullyCached,
        metadataFastPath};
  };

  // Footer statistics prune the middle row group. The two surviving,
  // non-contiguous groups are decoded together and stored as two source-row
  // runs.
  auto cold = read();
  EXPECT_EQ(cold.hits, 0);
  EXPECT_EQ(cold.misses, 2);
  EXPECT_EQ(cold.decodeCalls, 1);
  EXPECT_EQ(cold.chunks, 1);
  EXPECT_EQ(cold.rows, 8);
  EXPECT_FALSE(cold.fullyCached);
  EXPECT_FALSE(cold.metadataFastPath);

  ASSERT_TRUE(std::filesystem::remove(dataFile->getPath()));
  auto hot = read();
  EXPECT_EQ(hot.hits, 2);
  EXPECT_EQ(hot.misses, 0);
  EXPECT_EQ(hot.decodeCalls, 0);
  EXPECT_EQ(hot.chunks, 1);
  EXPECT_EQ(hot.rows, 8);
  EXPECT_TRUE(hot.fullyCached);
  EXPECT_TRUE(hot.metadataFastPath);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
