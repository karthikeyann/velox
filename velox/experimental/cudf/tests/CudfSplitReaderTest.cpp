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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/common/config/Config.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <gmock/gmock.h>

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

class MetadataOnlySplitReader final : public CudfSplitReader {
 public:
  using CudfSplitReader::CudfSplitReader;

  cudf::ast::expression const* logicalFilter() const {
    return subfieldFilterAst();
  }

  cudf::ast::expression const* splitFilter() const {
    return pushdownFilter();
  }

  bool hasSplitFilter() const {
    return hasSplitSpecificPushdownFilter();
  }

 protected:
  void prepareSplitInternal(
      dwio::common::RuntimeStats& /*runtimeStats*/) override {}
};

// Exposes the cuDF reader options a prepared split reads with.
class ReaderOptionsSplitReader final : public CudfSplitReader {
 public:
  using CudfSplitReader::CudfSplitReader;

  bool useTileDecode() const {
    return readerOptions().is_enabled_use_tile_decode();
  }
};

class CudfSplitReaderTest : public ::facebook::velox::cudf_velox::exec::test::
                                CudfHiveConnectorTestBase {};

TEST_F(CudfSplitReaderTest, dynamicFilterChecksParsedIntegerSchema) {
  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  ::facebook::velox::connector::ConnectorQueryCtx context(
      pool_.get(),
      pool_.get(),
      properties.get(),
      nullptr,
      common::PrefixSortConfig{},
      nullptr,
      nullptr,
      "query",
      "task",
      "scan",
      0,
      "");
  FileHandleFactory files(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto check = [&](const VectorPtr& values,
                   const TypePtr& declaredType,
                   bool matches,
                   bool statistics = true) {
    SCOPED_TRACE(
        fmt::format(
            "file: {}, declared: {}",
            values->type()->toString(),
            declaredType->toString()));
    auto file = common::testutil::TempFilePath::create();
    auto data = makeRowVector({"c0"}, {values});
    if (statistics) {
      writeToFile(file->getPath(), data);
    } else {
      auto stream = cudf::get_default_stream();
      auto table = with_arrow::toCudfTable(
          data, pool_.get(), stream, cudf::get_current_device_resource_ref());
      cudf::io::table_input_metadata metadata(table->view());
      metadata.column_metadata[0].set_name("c0");
      auto options =
          cudf::io::parquet_writer_options::builder(
              cudf::io::sink_info(file->getPath()), table->view())
              .metadata(metadata)
              .stats_level(cudf::io::statistics_freq::STATISTICS_NONE)
              .build();
      cudf::io::write_parquet(options, stream);
    }
    auto rowType = ROW({"c0"}, {declaredType});
    auto split =
        CudfHiveConnectorSplitBuilder(file->getPath())
            .connectorId(
                ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
            .build();
    common::SubfieldFilters filters;
    filters.emplace(
        common::Subfield("c0"),
        std::make_shared<common::BigintRange>(1, 3, false));
    cudf::ast::column_reference staticFilter{0};
    MetadataOnlySplitReader reader(
        split,
        ::facebook::velox::cudf_velox::exec::test::CudfHiveConnectorTestBase::
            makeTableHandle("parquet_table", rowType),
        rowType,
        {"c0"},
        &files,
        ioExecutor_.get(),
        &context,
        std::make_shared<CudfHiveConfig>(properties),
        /*fetchAheadContext=*/nullptr,
        std::make_shared<io::IoStatistics>(),
        std::make_shared<IoStats>(),
        &staticFilter,
        &filters);
    dwio::common::RuntimeStats stats;
    reader.prepareSplit(stats);
    EXPECT_EQ(reader.dynamicFiltersSatisfied(), matches);
  };
  std::vector<VectorPtr> values{
      makeFlatVector<int8_t>({1, 2, 3}),
      makeFlatVector<int16_t>({1, 2, 3}),
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<int64_t>({1, 2, 3})};
  for (const auto& fileValues : values) {
    for (const auto& declared : values) {
      check(
          fileValues, declared->type(), fileValues->type() == declared->type());
    }
  }
  auto date = makeFlatVector<int32_t>({1, 2, 3});
  date->setType(DATE());
  check(date, INTEGER(), false);
  auto decimal = makeFlatVector<int64_t>({1, 2, 3});
  decimal->setType(DECIMAL(9, 2));
  check(decimal, BIGINT(), false);
  check(makeFlatVector<int64_t>({1, 2, 3}), BIGINT(), false, false);
}

TEST_F(CudfSplitReaderTest, setsPushdownFilterAfterReadingMetadata) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeNamedRowVector("c0", makeFlatVector<int64_t>({1, 2, 3})));

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto connectorQueryCtx =
      ::facebook::velox::connector::ConnectorQueryCtx::Builder()
          .operatorPool(pool_.get())
          .connectorPool(pool_.get())
          .sessionProperties(properties.get())
          .queryId("query.CudfSplitReaderTest")
          .taskId("task.CudfSplitReaderTest")
          .planNodeId("plan.CudfSplitReaderTest")
          .build();
  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto split =
      CudfHiveConnectorSplitBuilder(dataFile->getPath())
          .connectorId(
              ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
          .build();

  cudf::ast::column_reference logicalFilter{0};
  cudf::ast::column_reference splitFilter{0};
  MetadataOnlySplitReader reader(
      std::move(split),
      ::facebook::velox::cudf_velox::exec::test::CudfHiveConnectorTestBase::
          makeTableHandle("parquet_table", rowType),
      rowType,
      {"c0"},
      &fileHandleFactory,
      ioExecutor_.get(),
      connectorQueryCtx.get(),
      std::make_shared<CudfHiveConfig>(properties),
      /*fetchAheadContext=*/nullptr,
      std::make_shared<io::IoStatistics>(),
      std::make_shared<IoStats>(),
      &logicalFilter,
      nullptr);

  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &logicalFilter);
  EXPECT_FALSE(reader.hasSplitFilter());

  const auto& fileMetadatas = reader.fileMetaDatas();
  ASSERT_FALSE(fileMetadatas.empty());
  EXPECT_GT(fileMetadatas.front().schema.size(), 1);
  EXPECT_EQ(&reader.fileMetaDatas(), &fileMetadatas);

  reader.setPushdownFilter(&splitFilter);
  dwio::common::RuntimeStats runtimeStats;
  reader.prepareSplit(runtimeStats);
  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &splitFilter);
  EXPECT_TRUE(reader.hasSplitFilter());
  EXPECT_EQ(runtimeStats.processedSplits, 1);
}

// The reader casts file timestamps to the session's
// parquet.reader.timestamp_type when the session sets it, and to the
// connector's parquet.reader.timestamp-type otherwise.
TEST_F(CudfSplitReaderTest, timestampTypeFollowsSession) {
  auto rowType = ROW({"c0"}, {TIMESTAMP()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeNamedRowVector(
          "c0", makeFlatVector<Timestamp>({Timestamp(1, 0), Timestamp(2, 0)})));
  // Both settings take the enumerator's integer value.
  const auto seconds =
      std::to_string(static_cast<int32_t>(cudf::type_id::TIMESTAMP_SECONDS));
  const auto microseconds = std::to_string(
      static_cast<int32_t>(cudf::type_id::TIMESTAMP_MICROSECONDS));

  // Reads the file with the connector config 'connectorConfig' under
  // 'session' and returns the cuDF type of its column.
  const auto readTimestampType =
      [&](const std::unordered_map<std::string, std::string>& connectorConfig,
          const std::unordered_map<std::string, std::string>& session) {
        auto sessionProperties = std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>(session));
        auto connectorQueryCtx =
            ::facebook::velox::connector::ConnectorQueryCtx::Builder()
                .operatorPool(pool_.get())
                .connectorPool(pool_.get())
                .sessionProperties(sessionProperties.get())
                .queryId("query.CudfSplitReaderTest")
                .taskId("task.CudfSplitReaderTest")
                .planNodeId("plan.CudfSplitReaderTest")
                .build();
        FileHandleFactory fileHandleFactory(
            std::make_unique<FileHandleCache>(1000),
            std::make_unique<FileHandleGenerator>());
        auto split = CudfHiveConnectorSplitBuilder(dataFile->getPath())
                         .connectorId(
                             ::facebook::velox::cudf_velox::exec::test::
                                 kCudfHiveConnectorId)
                         .build();
        CudfSplitReader reader(
            std::move(split),
            ::facebook::velox::cudf_velox::exec::test::
                CudfHiveConnectorTestBase::makeTableHandle(
                    "parquet_table", rowType),
            rowType,
            {"c0"},
            &fileHandleFactory,
            ioExecutor_.get(),
            connectorQueryCtx.get(),
            std::make_shared<CudfHiveConfig>(
                std::make_shared<config::ConfigBase>(
                    std::unordered_map<std::string, std::string>(
                        connectorConfig))),
            /*fetchAheadContext=*/nullptr,
            std::make_shared<io::IoStatistics>(),
            std::make_shared<IoStats>(),
            nullptr,
            nullptr);
        dwio::common::RuntimeStats runtimeStats;
        reader.prepareSplit(runtimeStats);
        auto chunk = reader.next(std::numeric_limits<uint64_t>::max(), {});
        reader.stream().sync();
        if (!chunk.has_value()) {
          ADD_FAILURE() << "The reader returned no chunk";
          return cudf::type_id::EMPTY;
        }
        return chunk->table->view().column(0).type().id();
      };

  EXPECT_EQ(readTimestampType({}, {}), cudf::type_id::TIMESTAMP_MILLISECONDS);
  EXPECT_EQ(
      readTimestampType({{CudfHiveConfig::kTimestampType, seconds}}, {}),
      cudf::type_id::TIMESTAMP_SECONDS);
  EXPECT_EQ(
      readTimestampType(
          {{CudfHiveConfig::kTimestampType, seconds}},
          {{CudfHiveConfig::kTimestampTypeSession, microseconds}}),
      cudf::type_id::TIMESTAMP_MICROSECONDS);
}

// The cuDF reader options take tile decode from the session's
// parquet.reader.use_tile_decode when the session sets it, and from the
// connector's parquet.reader.use-tile-decode otherwise.
TEST_F(CudfSplitReaderTest, tileDecodeFollowsSession) {
  auto rowType = ROW("c0", BIGINT());
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeNamedRowVector("c0", makeFlatVector<int64_t>({1, 2, 3})));

  // Prepares a split of the file with the connector config 'connectorConfig'
  // under 'session' and returns whether its reader options use tile decode.
  const auto useTileDecode =
      [&](const std::unordered_map<std::string, std::string>& connectorConfig,
          const std::unordered_map<std::string, std::string>& session) {
        auto sessionProperties = std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>(session));
        auto connectorQueryCtx =
            ::facebook::velox::connector::ConnectorQueryCtx::Builder()
                .operatorPool(pool_.get())
                .connectorPool(pool_.get())
                .sessionProperties(sessionProperties.get())
                .queryId("query.CudfSplitReaderTest")
                .taskId("task.CudfSplitReaderTest")
                .planNodeId("plan.CudfSplitReaderTest")
                .build();
        FileHandleFactory fileHandleFactory(
            std::make_unique<FileHandleCache>(1000),
            std::make_unique<FileHandleGenerator>());
        auto split = CudfHiveConnectorSplitBuilder(dataFile->getPath())
                         .connectorId(
                             ::facebook::velox::cudf_velox::exec::test::
                                 kCudfHiveConnectorId)
                         .build();
        ReaderOptionsSplitReader reader(
            std::move(split),
            ::facebook::velox::cudf_velox::exec::test::
                CudfHiveConnectorTestBase::makeTableHandle(
                    "parquet_table", rowType),
            rowType,
            {"c0"},
            &fileHandleFactory,
            ioExecutor_.get(),
            connectorQueryCtx.get(),
            std::make_shared<CudfHiveConfig>(
                std::make_shared<config::ConfigBase>(
                    std::unordered_map<std::string, std::string>(
                        connectorConfig))),
            /*fetchAheadContext=*/nullptr,
            std::make_shared<io::IoStatistics>(),
            std::make_shared<IoStats>(),
            nullptr,
            nullptr);
        dwio::common::RuntimeStats runtimeStats;
        reader.prepareSplit(runtimeStats);
        return reader.useTileDecode();
      };

  EXPECT_FALSE(useTileDecode({}, {}));
  EXPECT_TRUE(useTileDecode({{CudfHiveConfig::kUseTileDecode, "true"}}, {}));
  EXPECT_TRUE(
      useTileDecode({}, {{CudfHiveConfig::kUseTileDecodeSession, "true"}}));
  EXPECT_FALSE(useTileDecode(
      {{CudfHiveConfig::kUseTileDecode, "true"}},
      {{CudfHiveConfig::kUseTileDecodeSession, "false"}}));
}

// A split prepared with a filter-first filter is read filter columns first and
// returns the rows, in order, of a reader that only has the pushdown filter.
TEST_F(CudfSplitReaderTest, filterFirst) {
  constexpr vector_size_t kRowsPerGroup{50'000};
  constexpr int32_t kNumGroups{3};
  const auto rowType =
      ROW({{"c0", BIGINT()}, {"c1", INTEGER()}, {"c2", DOUBLE()}});
  // c1 is the row number in the file. The payload column c0 and the filter
  // column c1 have nulls.
  std::vector<RowVectorPtr> groups;
  for (int32_t group = 0; group < kNumGroups; ++group) {
    const auto base = group * kRowsPerGroup;
    groups.push_back(makeRowVector(
        rowType->names(),
        {
            makeFlatVector<int64_t>(
                kRowsPerGroup,
                [&](auto row) { return 10 * (base + row); },
                nullEvery(7)),
            makeFlatVector<int32_t>(
                kRowsPerGroup,
                [&](auto row) { return base + row; },
                nullEvery(5)),
            makeFlatVector<double>(
                kRowsPerGroup, [&](auto row) { return 0.25 * row; }),
        }));
  }
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(dataFile->getPath(), groups);

  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  const auto makeContext = [&](const config::ConfigBase& session) {
    return ::facebook::velox::connector::ConnectorQueryCtx::Builder()
        .operatorPool(pool_.get())
        .connectorPool(pool_.get())
        .sessionProperties(&session)
        .queryId("query.CudfSplitReaderTest")
        .taskId("task.CudfSplitReaderTest")
        .planNodeId("plan.CudfSplitReaderTest")
        .build();
  };
  // Makes a reader of 'readColumns' with the pushdown filter 'pushdownFilter',
  // which refers to them by index.
  const auto makeReader = [&](const ::facebook::velox::connector::
                                  ConnectorQueryCtx& context,
                              const std::vector<std::string>& readColumns,
                              const cudf::ast::expression& pushdownFilter) {
    std::vector<TypePtr> types;
    for (const auto& name : readColumns) {
      types.push_back(rowType->findChild(name));
    }
    return std::make_unique<CudfSplitReader>(
        CudfHiveConnectorSplitBuilder(dataFile->getPath())
            .connectorId(
                ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
            .build(),
        ::facebook::velox::cudf_velox::exec::test::CudfHiveConnectorTestBase::
            makeTableHandle("parquet_table", rowType),
        ROW(std::vector<std::string>(readColumns), std::move(types)),
        readColumns,
        &fileHandleFactory,
        ioExecutor_.get(),
        &context,
        std::make_shared<CudfHiveConfig>(std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{})),
        /*fetchAheadContext=*/nullptr,
        std::make_shared<io::IoStatistics>(),
        std::make_shared<IoStats>(),
        &pushdownFilter,
        nullptr);
  };
  // What a reader returned for the prepared split.
  struct Read {
    RowVectorPtr rows;
    int32_t numChunks{0};
    // Chunks whose rows the row filter tested.
    int32_t numRowFilteredChunks{0};
  };
  const auto readAll = [&](CudfSplitReader& reader,
                           const CudfSplitReader::RowFilter& rowFilter) {
    Read read;
    read.rows = BaseVector::create<RowVector>(rowType, 0, pool_.get());
    while (auto chunk =
               reader.next(std::numeric_limits<uint64_t>::max(), rowFilter)) {
      ++read.numChunks;
      read.numRowFilteredChunks += chunk->rowFilterApplied;
      read.rows->append(
          with_arrow::toVeloxColumn(
              chunk->table->view(),
              pool_.get(),
              rowType,
              reader.stream(),
              cudf::get_current_device_resource_ref())
              .get());
      reader.stream().sync();
    }
    return read;
  };
  const std::vector<std::string> allColumns{"c0", "c1", "c2"};
  dwio::common::RuntimeStats runtimeStats;

  // A range of c1 and the rows of it c1 has a value for.
  struct Range {
    std::string name;
    int64_t lower;
    int64_t upper;
    int64_t numKeptRows;
    // Row groups the statistics of c1 keep.
    int32_t numGroups;
  };
  const std::vector<Range> ranges{
      {"rows of every row group", 25'000, 124'999, 80'000, 3},
      {"rows at the start of the first row group only", 0, 999, 800, 1},
      {"rows at the end of the first row group only", 49'000, 49'999, 800, 1},
  };
  struct Limits {
    std::string chunkReadLimit;
    std::string passReadLimit;
  };
  // The limits of the second case make every page of a row group a chunk, so
  // the two one-row-group ranges have chunks without a row before or after the
  // chunk with rows.
  for (const auto& limits : {Limits{"0", "0"}, Limits{"100000", "500000"}}) {
    const config::ConfigBase session(
        std::unordered_map<std::string, std::string>{
            {CudfHiveConfig::kMaxChunkReadLimitSession, limits.chunkReadLimit},
            {CudfHiveConfig::kMaxPassReadLimitSession, limits.passReadLimit},
        });
    const auto context = makeContext(session);
    for (const auto& range : ranges) {
      SCOPED_TRACE(
          fmt::format(
              "{}, chunk read limit {}, pass read limit {}",
              range.name,
              limits.chunkReadLimit,
              limits.passReadLimit));
      const auto filters = common::test::SubfieldFiltersBuilder()
                               .add(
                                   "c1",
                                   std::make_unique<common::BigintRange>(
                                       range.lower, range.upper, false))
                               .build();
      cudf::ast::tree tree;
      std::vector<std::unique_ptr<cudf::scalar>> scalars;
      const auto& filter =
          createAstFromSubfieldFilters(filters, tree, scalars, rowType);

      auto oneStepReader = makeReader(*context, allColumns, filter);
      oneStepReader->prepareSplit(runtimeStats);
      EXPECT_FALSE(oneStepReader->readsFilterColumnsFirst());
      const auto expected = readAll(*oneStepReader, {});
      ASSERT_EQ(expected.rows->size(), range.numKeptRows);
      EXPECT_EQ(expected.numRowFilteredChunks, 0);
      {
        const auto stats = oneStepReader->takeReadStats();
        EXPECT_GT(stats.numOneStepPasses, 0);
        EXPECT_EQ(stats.numFilterFirstPasses, 0);
        EXPECT_EQ(stats.numFilterFirstRows, 0);
      }

      {
        SCOPED_TRACE("filter columns first");
        auto reader = makeReader(*context, allColumns, filter);
        reader->setFilterFirstFilter(&filter, {"c1"});
        reader->prepareSplit(runtimeStats);
        ASSERT_TRUE(reader->readsFilterColumnsFirst());
        const auto actual = readAll(*reader, {});
        facebook::velox::test::assertEqualVectors(expected.rows, actual.rows);
        EXPECT_EQ(actual.numRowFilteredChunks, 0);
        if (limits.chunkReadLimit != "0") {
          EXPECT_GT(actual.numChunks, range.numGroups);
        }
        const auto stats = reader->takeReadStats();
        EXPECT_EQ(stats.numOneStepPasses, 0);
        EXPECT_GT(stats.numFilterFirstPasses, 0);
        EXPECT_EQ(stats.numFilterFirstRows, range.numGroups * kRowsPerGroup);
        EXPECT_EQ(stats.numFilterFirstKeptRows, range.numKeptRows);
      }

      {
        SCOPED_TRACE("row filter");
        // Keeps the rows with an even c1 of those the reader's filter kept.
        std::vector<size_t> filterReadColumns;
        int64_t numFilteredRows{0};
        const CudfSplitReader::RowFilter keepEven =
            [&](const cudf::table_view& filterColumns,
                std::span<const size_t> readColumns,
                cuda::stream_ref stream) {
              filterReadColumns.assign(readColumns.begin(), readColumns.end());
              numFilteredRows += filterColumns.num_rows();
              const auto values = with_arrow::toVeloxColumn(
                  filterColumns,
                  pool_.get(),
                  ROW("c1", INTEGER()),
                  stream,
                  cudf::get_current_device_resource_ref());
              stream.sync();
              DecodedVector c1(*values->childAt(0));
              std::vector<uint8_t> keep(c1.size());
              for (vector_size_t row = 0; row < c1.size(); ++row) {
                keep[row] = c1.valueAt<int32_t>(row) % 2 == 0;
              }
              auto mask = cudf::make_fixed_width_column(
                  cudf::data_type{cudf::type_id::BOOL8},
                  keep.size(),
                  cudf::mask_state::UNALLOCATED,
                  stream,
                  cudf::get_current_device_resource_ref());
              CUDF_CUDA_TRY(cudaMemcpyAsync(
                  mask->mutable_view().data<uint8_t>(),
                  keep.data(),
                  keep.size(),
                  cudaMemcpyHostToDevice,
                  stream.get()));
              // 'keep' has to outlive the copy.
              stream.sync();
              return mask;
            };
        auto reader = makeReader(*context, allColumns, filter);
        reader->setFilterFirstFilter(&filter, {"c1"});
        reader->prepareSplit(runtimeStats);
        const auto actual = readAll(*reader, keepEven);

        std::vector<vector_size_t> evenRows;
        DecodedVector c1(*expected.rows->childAt(1));
        for (vector_size_t row = 0; row < expected.rows->size(); ++row) {
          if (c1.valueAt<int32_t>(row) % 2 == 0) {
            evenRows.push_back(row);
          }
        }
        facebook::velox::test::assertEqualVectors(
            wrapInDictionary(makeIndices(evenRows), expected.rows),
            actual.rows);
        EXPECT_EQ(actual.numRowFilteredChunks, actual.numChunks);
        EXPECT_THAT(filterReadColumns, testing::ElementsAre(1));
        EXPECT_EQ(numFilteredRows, range.numKeptRows);
        EXPECT_EQ(
            reader->takeReadStats().numFilterFirstKeptRows, evenRows.size());

        // The one-step reader has no rows for a row filter to test.
        auto unfilteredReader = makeReader(*context, allColumns, filter);
        unfilteredReader->prepareSplit(runtimeStats);
        const auto unfiltered = readAll(*unfilteredReader, keepEven);
        facebook::velox::test::assertEqualVectors(
            expected.rows, unfiltered.rows);
        EXPECT_EQ(unfiltered.numRowFilteredChunks, 0);
      }

      {
        SCOPED_TRACE("row filter that tests no row");
        const CudfSplitReader::RowFilter testNothing =
            [](const cudf::table_view& /*filterColumns*/,
               std::span<const size_t> /*readColumns*/,
               cuda::stream_ref /*stream*/) { return nullptr; };
        auto reader = makeReader(*context, allColumns, filter);
        reader->setFilterFirstFilter(&filter, {"c1"});
        reader->prepareSplit(runtimeStats);
        const auto actual = readAll(*reader, testNothing);
        facebook::velox::test::assertEqualVectors(expected.rows, actual.rows);
        EXPECT_EQ(actual.numRowFilteredChunks, 0);
      }

      {
        SCOPED_TRACE("row filter that keeps no row");
        const CudfSplitReader::RowFilter keepNothing =
            [](const cudf::table_view& filterColumns,
               std::span<const size_t> /*readColumns*/,
               cuda::stream_ref stream) {
              return cudf::make_column_from_scalar(
                  cudf::numeric_scalar<bool>(
                      false,
                      true,
                      stream,
                      cudf::get_current_device_resource_ref()),
                  filterColumns.num_rows(),
                  stream,
                  cudf::get_current_device_resource_ref());
            };
        auto reader = makeReader(*context, allColumns, filter);
        reader->setFilterFirstFilter(&filter, {"c1"});
        reader->prepareSplit(runtimeStats);
        EXPECT_EQ(readAll(*reader, keepNothing).numChunks, 0);
        const auto stats = reader->takeReadStats();
        EXPECT_EQ(stats.numFilterFirstRows, range.numGroups * kRowsPerGroup);
        EXPECT_EQ(stats.numFilterFirstKeptRows, 0);
      }
    }
  }

  const config::ConfigBase session(
      std::unordered_map<std::string, std::string>{});
  const auto context = makeContext(session);
  const auto filters =
      common::test::SubfieldFiltersBuilder()
          .add("c1", std::make_unique<common::BigintRange>(0, 999, false))
          .build();
  {
    SCOPED_TRACE("nothing to read after the filter columns");
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> scalars;
    const auto filterColumnType = ROW("c1", INTEGER());
    const auto& filter =
        createAstFromSubfieldFilters(filters, tree, scalars, filterColumnType);
    auto reader = makeReader(*context, {"c1"}, filter);
    reader->setFilterFirstFilter(&filter, {"c1"});
    reader->prepareSplit(runtimeStats);
    EXPECT_FALSE(reader->readsFilterColumnsFirst());
    int64_t numRows{0};
    while (auto chunk =
               reader->next(std::numeric_limits<uint64_t>::max(), {})) {
      numRows += chunk->numRows;
    }
    EXPECT_EQ(numRows, 800);
  }
  {
    cudf::ast::tree tree;
    std::vector<std::unique_ptr<cudf::scalar>> scalars;
    const auto& filter =
        createAstFromSubfieldFilters(filters, tree, scalars, rowType);
    auto reader = makeReader(*context, allColumns, filter);
    VELOX_ASSERT_THROW(
        reader->setFilterFirstFilter(&filter, {}),
        "A filter must reference a column");
    reader->prepareSplit(runtimeStats);
    VELOX_ASSERT_THROW(
        reader->setFilterFirstFilter(&filter, {"c1"}),
        "The filter of a prepared split cannot be replaced");
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
