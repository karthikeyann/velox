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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfDynamicFilter.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/connectors/hive/CudfIntegerMembership.h"
#include "velox/experimental/cudf/connectors/hive/FetchAheadBudget.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"
#include "velox/experimental/cudf/tests/utils/RecordingPinnedResource.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/Casts.h"
#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/file/tests/FaultyFile.h"
#include "velox/common/file/tests/FaultyFileSystem.h"
#include "velox/common/io/IoStatisticsRuntimeStats.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/VectorHasher.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/type/TimestampConversion.h"
#include "velox/type/Type.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda_runtime_api.h>

#include <fmt/ranges.h>
#include <folly/ScopeGuard.h>
#include <folly/synchronization/Baton.h>
#include <folly/synchronization/Latch.h>
#include <gmock/gmock.h>

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <limits>
#include <thread>
#include <unordered_set>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::connector;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::test;
using namespace facebook::velox::tests::utils;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::exec;
using namespace facebook::velox::cudf_velox::exec::test;

namespace {
struct StatsFilterMetrics {
  cudf::size_type inputRowGroups{0};
  std::optional<cudf::size_type> rowGroupsAfterStats;
  cudf::size_type outputRows{0};
};

StatsFilterMetrics readParquetWithStatsFilter(
    const std::string& filePath,
    const RowTypePtr& rowType,
    const common::SubfieldFilters& filters,
    bool useJitFilter) {
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  auto const& expr =
      createAstFromSubfieldFilters(filters, tree, scalars, rowType);

  auto options =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(filePath))
          .use_jit_filter(useJitFilter)
          .build();
  options.set_filter(expr);

  auto result = cudf::io::read_parquet(options);
  return {
      result.metadata.num_input_row_groups,
      result.metadata.num_row_groups_after_stats_filter,
      result.tbl->num_rows()};
}
// Reads what is left of the current split of 'source' into host vectors.
std::vector<RowVectorPtr> readSplit(
    facebook::velox::connector::DataSource& source,
    const RowTypePtr& type,
    memory::MemoryPool* pool) {
  std::vector<RowVectorPtr> rows;
  ContinueFuture future;
  while (auto next = source.next(1'000, future)) {
    if (*next == nullptr) {
      break;
    }
    auto gpu = std::dynamic_pointer_cast<CudfVector>(*next);
    VELOX_CHECK_NOT_NULL(gpu);
    rows.push_back(
        with_arrow::toVeloxColumn(
            gpu->getTableView(),
            pool,
            type,
            gpu->stream(),
            cudf::get_current_device_resource_ref()));
    gpu->stream().sync();
  }
  return rows;
}

// Counts the rows left in the current split of 'source' without copying them
// to the host.
vector_size_t countSplitRows(facebook::velox::connector::DataSource& source) {
  vector_size_t total{0};
  ContinueFuture future;
  while (auto next = source.next(1'000, future)) {
    if (*next == nullptr) {
      break;
    }
    total += (*next)->size();
  }
  return total;
}

vector_size_t numRows(const std::vector<RowVectorPtr>& vectors) {
  vector_size_t total{0};
  for (const auto& vector : vectors) {
    total += vector->size();
  }
  return total;
}
} // namespace

class TableScanTest : public virtual CudfHiveConnectorTestBase {
 protected:
  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
    cudf_velox::registerPrestoFunctions(
        cudf_velox::CudfConfig::getInstance().functionNamePrefix);
    ExchangeSource::factories().clear();
    ExchangeSource::registerFactory(createLocalExchangeSource);
  }

  static void SetUpTestCase() {
    CudfHiveConnectorTestBase::SetUpTestCase();
    parquet::registerParquetReaderFactory();
  }

  static void TearDownTestCase() {
    parquet::unregisterParquetReaderFactory();
    CudfHiveConnectorTestBase::TearDownTestCase();
  }

  // Writes 'data' to 'filePath' and returns a cuDF Hive scan of it with
  // 'remainingFilter'.
  core::PlanNodePtr makeRemainingFilterPlan(
      const RowVectorPtr& data,
      const std::string& remainingFilter,
      const std::shared_ptr<TempFilePath>& filePath) {
    writeToFile(filePath->getPath(), data);
    const auto rowType = asRowType(data->type());
    return PlanBuilder(pool_.get())
        .startTableScan()
        .connectorId(kCudfHiveConnectorId)
        .outputType(rowType)
        .dataColumns(rowType)
        .assignments(
            facebook::velox::exec::test::HiveConnectorTestBase::
                allRegularColumns(rowType))
        .remainingFilter(remainingFilter)
        .endTableScan()
        .planNode();
  }

  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const RowTypePtr& rowType = nullptr) {
    auto inputs = rowType ? rowType : rowType_;
    return CudfHiveConnectorTestBase::makeVectors(inputs, count, rowsPerVector);
  }

  std::shared_ptr<TempFilePath> writeUnsupportedMapInput() {
    auto input = makeRowVector(
        {"k", "m"},
        {makeFlatVector<int64_t>({1, 2}),
         makeMapVector<int64_t, int64_t>({{{10, 100}}, {{20, 200}}})});
    auto filePath = TempFilePath::create();
    auto fs = filesystems::getFileSystem(filePath->getPath(), {});
    auto writeFile = fs->openFileForWrite(
        filePath->getPath(),
        {.shouldCreateParentDirectories = true,
         .shouldThrowOnFileAlreadyExists = false});
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::move(writeFile), filePath->getPath());
    auto writerPool =
        rootPool_->addAggregateChild("TableScanTest.UnsupportedMapWriter");
    dwio::common::WriterOptions options;
    options.memoryPool = writerPool.get();
    parquet::Writer writer(
        std::move(sink), options, writerPool, asRowType(input->type()));
    writer.write(input);
    writer.close();
    return filePath;
  }

  Split makeCudfHiveSplit(std::string path, int64_t splitWeight = 0) {
    return Split(makeCudfHiveConnectorSplit(std::move(path), splitWeight));
  }

  // Returns a split of 'filePath' that carries 'partitionKeys'.
  static std::shared_ptr<facebook::velox::connector::ConnectorSplit>
  makePartitionedSplit(
      const std::string& filePath,
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys) {
    auto builder = hive::HiveConnectorSplitBuilder(filePath)
                       .connectorId(kCudfHiveConnectorId)
                       .fileFormat(dwio::common::FileFormat::PARQUET);
    for (const auto& [name, value] : partitionKeys) {
      builder.partitionKey(name, value);
    }
    return builder.build();
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::shared_ptr<facebook::velox::connector::ConnectorSplit>&
          parquetSplit,
      const std::string& duckDbSql) {
    return OperatorTestBase::assertQuery(plan, {parquetSplit}, duckDbSql);
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const Split&& split,
      const std::string& duckDbSql) {
    return OperatorTestBase::assertQuery(plan, {split}, duckDbSql);
  }

  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::vector<std::shared_ptr<TempFilePath>>& filePaths,
      const std::string& duckDbSql) {
    return CudfHiveConnectorTestBase::assertQuery(plan, filePaths, duckDbSql);
  }

  // Run query with spill enabled.
  std::shared_ptr<Task> assertQuery(
      const PlanNodePtr& plan,
      const std::vector<std::shared_ptr<TempFilePath>>& filePaths,
      const std::string& spillDirectory,
      const std::string& duckDbSql) {
    return AssertQueryBuilder(plan, duckDbQueryRunner_)
        .spillDirectory(spillDirectory)
        .config(core::QueryConfig::kSpillEnabled, false)
        .config(core::QueryConfig::kAggregationSpillEnabled, false)
        .splits(makeCudfHiveConnectorSplits(filePaths))
        .assertResults(duckDbSql);
  }

  core::PlanNodePtr tableScanNode() {
    return tableScanNode(rowType_);
  }

  core::PlanNodePtr tableScanNode(const RowTypePtr& outputType) {
    auto tableHandle = makeTableHandle();
    return PlanBuilder(pool_.get())
        .startTableScan()
        .outputType(outputType)
        .tableHandle(tableHandle)
        .endTableScan()
        .planNode();
  }

  static PlanNodeStats getTableScanStats(const std::shared_ptr<Task>& task) {
    auto planStats = toPlanStats(task->taskStats());
    return std::move(planStats.at("0"));
  }

  // Returns whether the scan ran on the GPU, and expects the table scan adapter
  // and the connector to agree on it. The adapter adds a CudfToVelox operator
  // after a GPU scan, and only the cuDF data source reports
  // kDwioStorageReadBytes. It reports it only for bytes read from storage, so
  // each scan reads a file no earlier scan read, which is not yet cached.
  static bool wasCudfScanUsed(
      const std::shared_ptr<Task>& task,
      const core::PlanNodeId& planNodeId) {
    const auto planStats = toPlanStats(task->taskStats());
    const auto& scanStats = planStats.at(planNodeId);
    const bool adapterUsedCudf =
        scanStats.operatorStats.count("CudfToVelox") > 0;
    const bool connectorUsedCudf =
        scanStats.customStats.count(
            std::string(
                cudf_velox::connector::hive::CudfHiveDataSource::
                    kDwioStorageReadBytes)) > 0;
    EXPECT_EQ(adapterUsedCudf, connectorUsedCudf);
    return adapterUsedCudf;
  }

  static std::unordered_map<std::string, RuntimeMetric>
  getTableScanRuntimeStats(const std::shared_ptr<Task>& task) {
    return task->taskStats().pipelineStats[0].operatorStats[0].runtimeStats;
  }

  // Runtime stats omit zero-valued counters, so a missing stat reads as zero.
  static int64_t getTableScanRuntimeStat(
      const std::shared_ptr<Task>& task,
      std::string_view name) {
    const auto runtimeStats = getTableScanRuntimeStats(task);
    const auto it = runtimeStats.find(std::string(name));
    return it == runtimeStats.end() ? 0 : it->second.sum;
  }

  // Verifies I/O is bounded by one footer and one data read of the unique file.
  static void assertStorageReadStats(
      const std::unordered_map<std::string, RuntimeMetric>& runtimeStats,
      int64_t fileSize) {
    for (const auto key : {
             io::kStorageReadBytes,
             cudf_velox::connector::hive::CudfHiveDataSource::
                 kDwioStorageReadBytes,
         }) {
      const auto& metric = runtimeStats.at(std::string(key));
      EXPECT_GT(metric.sum, 0);
      EXPECT_LE(metric.sum, 2 * fileSize);
    }
  }

  static int64_t getSkippedStridesStat(const std::shared_ptr<Task>& task) {
    VELOX_NYI(
        "RuntimeStats not yet implemented for the cudf CudfHiveConnector");
    // return getTableScanRuntimeStats(task)["skippedStrides"].sum;
  }

  static int64_t getSkippedSplitsStat(const std::shared_ptr<Task>& task) {
    VELOX_NYI(
        "RuntimeStats not yet implemented for the cudf CudfHiveConnector");
    // return getTableScanRuntimeStats(task)["skippedSplits"].sum;
  }

  static void waitForFinishedDrivers(
      const std::shared_ptr<Task>& task,
      uint32_t n) {
    // Limit wait to 10 seconds.
    size_t iteration{0};
    while (task->numFinishedDrivers() < n and iteration < 100) {
      /* sleep override */
      usleep(100'000); // 0.1 second.
      ++iteration;
    }
    ASSERT_EQ(n, task->numFinishedDrivers());
  }

  void writeCompactDecimalParquet(
      const std::shared_ptr<TempFilePath>& filePath,
      const RowVectorPtr& vector) {
    auto fs = filesystems::getFileSystem(filePath->getPath(), {});
    auto writeFile = fs->openFileForWrite(
        filePath->getPath(),
        {.shouldCreateParentDirectories = true,
         .shouldThrowOnFileAlreadyExists = false});
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::move(writeFile), filePath->getPath());
    auto writerPool =
        rootPool_->addAggregateChild("TableScanTest.ParquetWriter");
    dwio::common::WriterOptions options;
    options.memoryPool = writerPool.get();
    auto parquetOptions = std::make_shared<parquet::ParquetWriterOptions>();
    parquetOptions->enableStoreDecimalAsInteger = true;
    options.formatSpecificOptions = std::move(parquetOptions);
    parquet::Writer writer(
        std::move(sink), options, writerPool, vector->rowType());
    writer.write(vector);
    writer.close();
  }

  void assertDecimalScanRoundTrip(
      const RowVectorPtr& vector,
      const RowTypePtr& rowType) {
    auto filePath = TempFilePath::create();
    writeCompactDecimalParquet(filePath, vector);
    createDuckDbTable({vector});

    auto assignments =
        facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
            rowType);
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .connectorId(kCudfHiveConnectorId)
                    .outputType(rowType)
                    .dataColumns(rowType)
                    .assignments(assignments)
                    .endTableScan()
                    .planNode();

    AssertQueryBuilder(plan, duckDbQueryRunner_)
        .splits(makeCudfHiveConnectorSplits({filePath}))
        .assertResults("SELECT * FROM tmp");
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {INTEGER(),
           VARCHAR(),
           TINYINT(),
           DOUBLE(),
           BIGINT(),
           VARCHAR(),
           REAL()})};
};

class TableScanTestParameterized : public TableScanTest,
                                   public testing::WithParamInterface<bool> {};

// Returns the cuDF Hive connector that is registered.
std::shared_ptr<cudf_velox::connector::hive::CudfHiveConnector>
cudfHiveConnector() {
  auto connector =
      std::dynamic_pointer_cast<cudf_velox::connector::hive::CudfHiveConnector>(
          facebook::velox::connector::ConnectorRegistry::tryGet(
              kCudfHiveConnectorId));
  VELOX_CHECK_NOT_NULL(connector);
  return connector;
}

// Returns the counters of the registered connector's fetch-ahead budget.
cudf_velox::connector::hive::FetchAheadBudget::Stats fetchAheadStats() {
  return cudfHiveConnector()->fetchAheadContext()->budget().stats();
}

TEST_P(TableScanTestParameterized, ignoreMissingFiles) {
  using cudf_velox::connector::hive::CudfHiveConfig;
  using hive::HiveConfig;
  auto directory = TempDirectoryPath::create();
  const auto missingPath = directory->getPath() + "/missing.parquet";
  auto missingSplit = [&]() {
    return hive::HiveConnectorSplitBuilder(missingPath)
        .connectorId(kCudfHiveConnectorId)
        .fileFormat(dwio::common::FileFormat::PARQUET)
        .length(123)
        .build();
  };
  auto vectors = makeVectors(1, 100);
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), vectors);
  createDuckDbTable(vectors);
  auto plan = tableScanNode();

  const auto useBufferedInput = GetParam() ? "true" : "false";
  EXPECT_THROW(
      AssertQueryBuilder(plan)
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              CudfHiveConfig::kUseBufferedInputSession,
              useBufferedInput)
          .split(missingSplit())
          .assertEmptyResults(),
      std::exception);

  for (const auto preload : {0, 4}) {
    std::vector<std::shared_ptr<ConnectorSplit>> splits{missingSplit()};
    auto existingSplits = makeCudfHiveConnectorSplits({file});
    splits.insert(splits.end(), existingSplits.begin(), existingSplits.end());
    splits.push_back(missingSplit());
    auto moreExistingSplits = makeCudfHiveConnectorSplits({file});
    splits.insert(
        splits.end(), moreExistingSplits.begin(), moreExistingSplits.end());
    auto task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .maxDrivers(1)
            .config(
                QueryConfig::kMaxSplitPreloadPerDriver, std::to_string(preload))
            .connectorSessionProperty(
                kCudfHiveConnectorId,
                CudfHiveConfig::kUseBufferedInputSession,
                useBufferedInput)
            .connectorSessionProperty(
                kCudfHiveConnectorId,
                HiveConfig::kIgnoreMissingFilesSession,
                "true")
            .splits(splits)
            .assertResults("SELECT * FROM tmp UNION ALL SELECT * FROM tmp");
    const auto stats = getTableScanRuntimeStats(task);
    EXPECT_EQ(stats.count("skippedSplits"), 0);
    EXPECT_EQ(stats.count("skippedSplitBytes"), 0);
    EXPECT_EQ(stats.at("processedSplits").sum, 2);
  }
}

TEST_F(TableScanTest, ignoreMissingFilesPreservesReadErrors) {
  // Preserve the actual open failure instead of retrying through KvikIO.
  filesystems::registerFileSystem(
      [](std::string_view path) { return path.starts_with("denied:"); },
      [](std::shared_ptr<const config::ConfigBase>,
         std::string_view) -> std::shared_ptr<filesystems::FileSystem> {
        VELOX_FAIL("Injected permission denied at open");
      });
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(tableScanNode())
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              cudf_velox::connector::hive::CudfHiveConfig::
                  kUseBufferedInputSession,
              "true")
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              hive::HiveConfig::kIgnoreMissingFilesSession,
              "true")
          .split(makeCudfHiveConnectorSplit("denied:/missing.parquet"))
          .assertEmptyResults(),
      "Injected permission denied at open");
}

TEST_F(TableScanTest, ignoreMissingFilesPreservesCorruption) {
  // An existing empty file is invalid Parquet, not a missing split.
  auto file = TempFilePath::create();
  EXPECT_THROW(
      AssertQueryBuilder(tableScanNode())
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              hive::HiveConfig::kIgnoreMissingFilesSession,
              "true")
          .splits(makeCudfHiveConnectorSplits({file}))
          .assertEmptyResults(),
      std::exception);
}

TEST_P(TableScanTestParameterized, allColumns) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  createDuckDbTable(vectors);
  auto plan = tableScanNode();

  const std::string duckDbSql = "SELECT * FROM tmp";

  // Helper to test scan all columns for the given splits
  auto testScanAllColumns =
      [&](const std::vector<std::shared_ptr<
              facebook::velox::connector::ConnectorSplit>>& splits) {
        auto task = AssertQueryBuilder(duckDbQueryRunner_)
                        .plan(plan)
                        .splits(splits)
                        .assertResults(duckDbSql);

        // A quick sanity check for memory usage reporting. Check that peak
        // total memory usage for the project node is > 0.
        auto planStats = toPlanStats(task->taskStats());
        auto scanNodeId = plan->id();
        auto it = planStats.find(scanNodeId);
        ASSERT_TRUE(it != planStats.end());
        // TODO (dm): enable this test once we start to track gpu memory
        // ASSERT_TRUE(it->second.peakMemoryBytes > 0);

        //  Verifies there is no dynamic filter stats.
        ASSERT_TRUE(it->second.dynamicFilterStats.empty());

        // TODO: We are not writing any customStats yet so disable this check
        // ASSERT_LT(0, it->second.customStats.at("ioWaitWallNanos").sum);
      };

  const bool useBufferedInput = GetParam();
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseBufferedInput,
       useBufferedInput ? "true" : "false"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));

  // Test scan all columns with CudfHiveConnectorSplits
  {
    auto splits = makeCudfHiveConnectorSplits({filePath});
    testScanAllColumns(splits);
  }

  // Test scan all columns with HiveConnectorSplits
  {
    std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
        splits;
    splits.push_back(
        facebook::velox::connector::hive::HiveConnectorSplitBuilder(
            filePath->getPath())
            .connectorId(kCudfHiveConnectorId)
            .fileFormat(dwio::common::FileFormat::PARQUET)
            .build());
    testScanAllColumns(splits);
  }
}

// Reads several splits of a multi-row-group file with chunk and pass read
// limits small enough that each split is read as multiple row group passes,
// each yielding multiple table chunks.
TEST_P(TableScanTestParameterized, allColumnsWithRowGroupPasses) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  createDuckDbTable(vectors);
  const std::string duckDbSql =
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp UNION ALL "
      "SELECT * FROM tmp";

  auto splits = makeCudfHiveConnectorSplits(
      {filePath, filePath, filePath, filePath, filePath});

  const bool useBufferedInput = GetParam();
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kMaxChunkReadLimit,
       "8192"},
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kMaxPassReadLimit,
       "32768"},
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseBufferedInput,
       useBufferedInput ? "true" : "false"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));

  auto plan = tableScanNode();
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .splits(splits)
                  .assertResults(duckDbSql);

  auto planStats = toPlanStats(task->taskStats());
  auto it = planStats.find(plan->id());
  ASSERT_TRUE(it != planStats.end());

  // Reading in chunks must not change the number of rows returned.
  const auto& scanStats = it->second.operatorStatsFor("TableScan");
  EXPECT_EQ(scanStats.outputRows, 5 * 10 * 1'000);

  // Splitting the read into chunks must produce more than one output vector
  // per split.
  EXPECT_GT(scanStats.outputVectors, splits.size());

  //  Verifies there is no dynamic filter stats.
  ASSERT_TRUE(it->second.dynamicFilterStats.empty());
}

// One driver reads many small files. The column chunks of the splits
// prepared ahead of it are fetched ahead too, within the fetch-ahead budget,
// when the session asks for it; either way the scan returns the same rows and
// gives back every byte it held.
TEST_P(TableScanTestParameterized, fetchAhead) {
  using cudf_velox::connector::hive::CudfHiveConfig;
  using cudf_velox::connector::hive::FetchAheadBudget;
  constexpr int32_t kNumFiles{12};
  auto filePaths = makeFilePaths(kNumFiles);
  auto vectors = makeVectors(kNumFiles, 5'000);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);
  auto plan = tableScanNode();
  const auto limitPlan = [&](int64_t limit) {
    return PlanBuilder(pool_.get())
        .startTableScan()
        .outputType(rowType_)
        .tableHandle(makeTableHandle())
        .endTableScan()
        .limit(0, limit, false)
        .planNode();
  };

  // What a scan is run with.
  struct Scan {
    // Capacity of the budget, or zero for none.
    uint64_t capacity;
    bool preloadColumnChunks{false};
    int32_t numPreloadedSplits{4};
    // Rows to read, or zero for every row.
    int64_t limit{0};
  };
  // Returns the counters of the connector's budget after the scan. Every
  // scan has a connector, and so a budget, of its own.
  const auto run = [&](const Scan& scan) {
    resetCudfHiveConnector(
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{
                {CudfHiveConfig::kFetchAheadBytes,
                 std::to_string(scan.capacity)},
                {CudfHiveConfig::kUseBufferedInput,
                 GetParam() ? "true" : "false"},
            }));
    EXPECT_EQ(
        cudfHiveConnector()->fetchAheadContext() != nullptr, scan.capacity > 0);
    AssertQueryBuilder builder(
        scan.limit > 0 ? limitPlan(scan.limit) : plan, duckDbQueryRunner_);
    builder.maxDrivers(1)
        .config(
            core::QueryConfig::kMaxSplitPreloadPerDriver,
            std::to_string(scan.numPreloadedSplits))
        .connectorSessionProperty(
            kCudfHiveConnectorId,
            CudfHiveConfig::kPreloadColumnChunksSession,
            scan.preloadColumnChunks ? "true" : "false")
        .splits(makeCudfHiveConnectorSplits(filePaths));
    if (scan.limit > 0) {
      auto result = builder.copyResults(pool_.get());
      EXPECT_EQ(result->size(), scan.limit);
      result.reset();
    } else {
      builder.assertResults("SELECT * FROM tmp");
    }
    // The splits the scan did not get to are given up with their task.
    waitForAllTasksToBeDeleted();
    return scan.capacity > 0 ? fetchAheadStats() : FetchAheadBudget::Stats{};
  };
  constexpr uint64_t kRoomy{1ULL << 30};

  {
    // The default: no budget. The scan returns the same rows.
    SCOPED_TRACE("off");
    run({.capacity = 0});
  }
  // With no split prepared ahead the one driver holds the column chunks of
  // one split at a time, so the most it held is the largest split.
  uint64_t largestSplit{0};
  {
    SCOPED_TRACE("no split prepared ahead");
    const auto stats = run({.capacity = kRoomy, .numPreloadedSplits = 0});
    EXPECT_EQ(stats.numOffered, 0);
    EXPECT_EQ(stats.heldBytes, 0);
    EXPECT_GT(stats.maxHeldBytes, 0);
    largestSplit = stats.maxHeldBytes;
  }
  {
    SCOPED_TRACE("room for every split");
    const auto stats = run({.capacity = kRoomy});
    EXPECT_GT(stats.numOffered, 0);
    EXPECT_EQ(stats.numWaiting, 0);
    EXPECT_EQ(stats.numAhead, 0);
    EXPECT_EQ(stats.heldBytes, 0);
    EXPECT_LE(stats.maxHeldBytes, kNumFiles * largestSplit);
  }
  {
    SCOPED_TRACE("room for one split ahead");
    // A capacity of one byte lets a fetch start ahead only when no other is
    // ahead, so at most one split is held ahead of the one being read.
    const auto stats = run({.capacity = 1});
    EXPECT_GT(stats.numOffered, 0);
    EXPECT_EQ(stats.heldBytes, 0);
    EXPECT_LE(stats.maxHeldBytes, 2 * largestSplit);
  }
  {
    SCOPED_TRACE("preload-column-chunks");
    // With fetch-ahead on, asking for the column chunks of prepared splits
    // does not fetch them beyond the budget.
    const auto stats = run({.capacity = 2, .preloadColumnChunks = true});
    EXPECT_GT(stats.numOffered, 0);
    EXPECT_EQ(stats.heldBytes, 0);
    EXPECT_LE(stats.maxHeldBytes, 2 * largestSplit);
  }
  {
    SCOPED_TRACE("scan that stops early");
    // The scan ends after its first rows, with fetches offered and started
    // ahead that no driver asked for. Every byte comes back.
    const auto stats = run({.capacity = kRoomy, .limit = 10});
    EXPECT_GT(stats.numOffered, 0);
    EXPECT_EQ(stats.heldBytes, 0);
    EXPECT_EQ(stats.numAhead, 0);
    EXPECT_EQ(stats.numWaiting, 0);
  }
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{}));
}

// One driver reads one file in several row group passes. With fetch-ahead
// the pass after the one being decoded is offered to the budget, so the
// reader holds at most two passes; the rows are the same and every byte is
// given back, also when the scan stops early.
TEST_P(TableScanTestParameterized, fetchAheadWithinSplit) {
  using cudf_velox::connector::hive::CudfHiveConfig;
  using cudf_velox::connector::hive::FetchAheadBudget;
  constexpr int32_t kNumRowGroups{8};
  // One row group per vector.
  auto vectors = makeVectors(kNumRowGroups, 5'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);
  auto plan = tableScanNode();
  auto limitPlan = PlanBuilder(pool_.get())
                       .startTableScan()
                       .outputType(rowType_)
                       .tableHandle(makeTableHandle())
                       .endTableScan()
                       .limit(0, 10, false)
                       .planNode();

  // Returns the counters of the connector's budget after a scan that reads
  // every row, or its first rows.
  const auto run = [&](uint64_t capacity, bool stopsEarly) {
    resetCudfHiveConnector(
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{
                {CudfHiveConfig::kFetchAheadBytes, std::to_string(capacity)},
                {CudfHiveConfig::kUseBufferedInput,
                 GetParam() ? "true" : "false"},
            }));
    AssertQueryBuilder builder(
        stopsEarly ? limitPlan : plan, duckDbQueryRunner_);
    builder
        .maxDrivers(1)
        // Every row group is a pass of its own.
        .connectorSessionProperty(
            kCudfHiveConnectorId, CudfHiveConfig::kMaxPassReadLimitSession, "1")
        .split(makeCudfHiveConnectorSplit(filePath->getPath()));
    if (stopsEarly) {
      auto result = builder.copyResults(pool_.get());
      EXPECT_EQ(result->size(), 10);
      result.reset();
    } else {
      builder.assertResults("SELECT * FROM tmp");
    }
    waitForAllTasksToBeDeleted();
    return capacity > 0 ? fetchAheadStats() : FetchAheadBudget::Stats{};
  };
  constexpr uint64_t kRoomy{1ULL << 30};

  {
    // The default: no budget. The scan returns the same rows.
    SCOPED_TRACE("off");
    run(0, false);
  }
  // The reader offers one pass at a time, and one fetch may always be ahead,
  // so with or without room it holds the pass being decoded and the next:
  // two of the eight, which are of about one size.
  const auto fileSize = static_cast<uint64_t>(filePath->fileSize());
  for (const uint64_t capacity : {uint64_t{3}, kRoomy}) {
    SCOPED_TRACE(capacity);
    const auto stats = run(capacity, false);
    // Every pass but the first is offered while the one before it is
    // decoded. A pass holds at least one row group.
    EXPECT_GE(stats.numOffered, kNumRowGroups - 1);
    EXPECT_EQ(stats.numWaiting, 0);
    EXPECT_EQ(stats.numAhead, 0);
    EXPECT_EQ(stats.heldBytes, 0);
    EXPECT_GT(stats.maxHeldBytes, 0);
    EXPECT_LE(stats.maxHeldBytes, fileSize / 2);
  }
  {
    SCOPED_TRACE("scan that stops early");
    const auto stats = run(kRoomy, true);
    EXPECT_GT(stats.numOffered, 0);
    EXPECT_EQ(stats.numWaiting, 0);
    EXPECT_EQ(stats.numAhead, 0);
    EXPECT_EQ(stats.heldBytes, 0);
  }
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{}));
}

// A driver that is held back after its first batch finds column chunks of
// the splits prepared ahead on the device: their fetches start while the
// driver is busy with what it has, and the driver claims them when it gets
// to those splits. The scan reports how long the driver waited for bytes.
TEST_P(TableScanTestParameterized, fetchAheadClaimed) {
  using cudf_velox::connector::hive::CudfHiveConfig;
  using cudf_velox::connector::hive::CudfSplitReader;
  constexpr int32_t kNumFiles{4};
  constexpr int32_t kNumRows{5'000};
  auto filePaths = makeFilePaths(kNumFiles);
  auto vectors = makeVectors(kNumFiles, kNumRows);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{
              {CudfHiveConfig::kFetchAheadBytes, std::to_string(1ULL << 30)},
              {CudfHiveConfig::kUseBufferedInput,
               GetParam() ? "true" : "false"},
          }));
  auto plan = tableScanNode();
  {
    CursorParameters params;
    params.planNode = plan;
    params.maxDrivers = 1;
    // The driver blocks after every batch until the batch is taken.
    params.bufferedBytes = 2;
    auto cursor = TaskCursor::create(params);
    cursor->start();
    for (auto& split : makeCudfHiveConnectorSplits(filePaths)) {
      cursor->task()->addSplit(
          plan->id(), facebook::velox::exec::Split(std::move(split)));
    }
    cursor->task()->noMoreSplits(plan->id());

    // The driver has decoded the first split and waits for its batch to be
    // taken. A split prepared ahead offers its fetch, which starts.
    while (fetchAheadStats().numStartedAhead == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // The start thread runs the starts in the order they were posted.
    folly::Baton<> started;
    cudfHiveConnector()->fetchAheadContext()->post([&]() { started.post(); });
    started.wait();

    int64_t numRows{0};
    while (cursor->moveNext()) {
      numRows += cursor->current()->size();
    }
    EXPECT_EQ(numRows, kNumFiles * kNumRows);

    auto planStats = toPlanStats(cursor->task()->taskStats());
    const auto& scanStats = planStats.at(plan->id());
    const auto fetchWait = scanStats.customStats.find(
        std::string(CudfSplitReader::kFetchWaitWallNanos));
    ASSERT_TRUE(fetchWait != scanStats.customStats.end());
    // One pass in every file.
    EXPECT_EQ(fetchWait->second.count, kNumFiles);
  }
  waitForAllTasksToBeDeleted();
  const auto stats = fetchAheadStats();
  EXPECT_GE(stats.numClaimed, 1);
  EXPECT_EQ(stats.numAhead, 0);
  EXPECT_EQ(stats.numWaiting, 0);
  EXPECT_EQ(stats.heldBytes, 0);
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{}));
}

// A scan with the buffered-input reader copies its column chunks through
// page-locked pieces far smaller than the chunks, on the threads of the IO
// executor, and returns the same rows, alone and with fetch-ahead.
TEST_F(TableScanTest, bufferedInputPinnedPieces) {
  using cudf_velox::connector::hive::CudfHiveConfig;
  // A size that nothing else allocates.
  constexpr uint64_t kPieceBytes{2'500};
  constexpr int32_t kNumFiles{4};
  auto filePaths = makeFilePaths(kNumFiles);
  auto vectors = makeVectors(kNumFiles, 5'000);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);
  for (const uint64_t fetchAheadBytes : {uint64_t{0}, uint64_t{1} << 30}) {
    SCOPED_TRACE(fetchAheadBytes);
    cudf_velox::test::RecordingPinnedResource pinned{
        cudf::get_pinned_memory_resource()};
    const auto previous = cudf::set_pinned_memory_resource(pinned);
    SCOPE_EXIT {
      cudf::set_pinned_memory_resource(previous);
    };
    resetCudfHiveConnector(
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{
                {CudfHiveConfig::kUseBufferedInput, "true"},
                {CudfHiveConfig::kBufferedInputPinnedPieceBytes,
                 std::to_string(kPieceBytes)},
                {CudfHiveConfig::kFetchAheadBytes,
                 std::to_string(fetchAheadBytes)},
            }));
    AssertQueryBuilder(tableScanNode(), duckDbQueryRunner_)
        .maxDrivers(2)
        .splits(makeCudfHiveConnectorSplits(filePaths))
        .assertResults("SELECT * FROM tmp");
    waitForAllTasksToBeDeleted();
    resetCudfHiveConnector(
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{}));
    EXPECT_GT(pinned.numAllocationsOf(kPieceBytes), 0);
    EXPECT_EQ(pinned.bytesInUse(), 0);
  }
}

// Splits prepared in the background by the preloader must produce the same
// results as splits prepared on the driver thread.
TEST_F(TableScanTest, preloadSplits) {
  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "10")
                  .splits(makeCudfHiveConnectorSplits(filePaths))
                  .assertResults("SELECT * FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  const auto& customStats = planStats.at(plan->id()).customStats;
  ASSERT_EQ(customStats.count(std::string(TableScan::kPreloadedSplits)), 1);
  EXPECT_EQ(
      customStats.at(std::string(TableScan::kPreloadedSplits)).sum,
      filePaths.size());
}

// A busy IO thread pool never runs the queued preload tasks, so every split is
// prepared inline by the driver that comes to read it.
TEST_F(TableScanTest, preloadingSplitClose) {
  auto filePaths = makeFilePaths(20);
  auto vectors = makeVectors(20, 100);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  auto* ioExecutor = ioExecutor_.get();
  folly::Latch latch(ioExecutor->numThreads());
  std::vector<folly::Baton<>> batons(ioExecutor->numThreads());
  // Simulate a busy IO thread pool by blocking all its threads.
  for (auto& baton : batons) {
    ioExecutor->add([&]() {
      baton.wait();
      latch.count_down();
    });
  }

  ASSERT_EQ(Task::numRunningTasks(), 0);
  auto plan = tableScanNode();
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "4")
                  .splits(makeCudfHiveConnectorSplits(filePaths))
                  .assertResults("SELECT * FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  EXPECT_GT(
      planStats.at(plan->id())
          .customStats.at(std::string(TableScan::kPreloadedSplits))
          .sum,
      1);

  task.reset();
  // Once all task references are cleared, all the tasks should be destroyed.
  ASSERT_EQ(Task::numRunningTasks(), 0);
  // Unblock the IO thread pool.
  for (auto& baton : batons) {
    baton.post();
  }
  latch.wait();
}

// A query that stops early leaves the splits the preloader has already prepared
// unread, so their readers are destroyed with a payload fetch outstanding.
TEST_F(TableScanTest, abandonPreloadedSplits) {
  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  // Consume all but one IO threads so the preloader can post column chunk fetch
  // tasks queuing behind the blockers and stay pending. Driver reads complete
  // inline via work-stealing.
  auto* ioExecutor = ioExecutor_.get();
  const auto numBlocked = ioExecutor->numThreads() - 1;
  ASSERT_GE(numBlocked, 1);
  folly::Latch latch(numBlocked);
  std::vector<folly::Baton<>> batons(numBlocked);
  for (auto& baton : batons) {
    ioExecutor->add([&]() {
      baton.wait();
      latch.count_down();
    });
  }

  // The limit is reached partway into the second split, so the splits the
  // preloader prepared behind it are never read. Which rows are returned is
  // unspecified, so only the row count can be asserted.
  constexpr int32_t kLimit = 1'500;
  core::PlanNodeId scanNodeId;
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(rowType_)
                  .tableHandle(makeTableHandle())
                  .endTableScan()
                  .capturePlanNodeId(scanNodeId)
                  .limit(0, kLimit, false)
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "8")
                    // Enable column chunk fetch during preload
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        cudf_velox::connector::hive::CudfHiveConfig::
                            kPreloadColumnChunksSession,
                        "true")
                    .splits(makeCudfHiveConnectorSplits(filePaths))
                    .copyResults(pool_.get(), task);
  EXPECT_EQ(result->size(), kLimit);

  // The first split is read before the preloader runs, so only the splits
  // after it are preloaded. The stat counts the preloaded splits that were
  // read, so it confirms preloading was on but cannot measure how many were
  // abandoned.
  auto planStats = toPlanStats(task->taskStats());
  const auto& customStats = planStats.at(scanNodeId).customStats;
  ASSERT_EQ(customStats.count(std::string(TableScan::kPreloadedSplits)), 1);
  EXPECT_GE(customStats.at(std::string(TableScan::kPreloadedSplits)).sum, 1);

  // Tear down while abandoned splits' payload fetches are still queued.
  task.reset();
  ASSERT_EQ(Task::numRunningTasks(), 0);

  // Unblock the IO thread pool.
  for (auto& baton : batons) {
    baton.post();
  }
  latch.wait();
}

// A filter that no row group can satisfy prunes every row group of the split,
// leaving no row group passes to read.
TEST_F(TableScanTest, filterPrunesAllRowGroups) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  // One row group per vector, all holding values well below the filter bound.
  std::vector<RowVectorPtr> vectors = {
      makeNamedRowVector("c0", makeFlatVector<int64_t>({1, 2, 3})),
      makeNamedRowVector("c0", makeFlatVector<int64_t>({4, 5, 6})),
      makeNamedRowVector("c0", makeFlatVector<int64_t>({7, 8, 9})),
  };
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  constexpr int64_t kUnmatchedValue = 1'000;
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  kUnmatchedValue, kUnmatchedValue, false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  auto task =
      AssertQueryBuilder(duckDbQueryRunner_)
          .plan(plan)
          .splits(makeCudfHiveConnectorSplits({filePath}))
          .assertResults(
              fmt::format("SELECT c0 FROM tmp WHERE c0 = {}", kUnmatchedValue));

  auto planStats = toPlanStats(task->taskStats());
  EXPECT_EQ(planStats.at(plan->id()).outputRows, 0);
}

// Table schemas use lowercase names while the file keeps mixed-case names. The
// filter-only column must still be read so the pushed-down filter can find it.
TEST_F(TableScanTest, mixedCaseFileColumnNames) {
  auto vector = makeRowVector(
      {"Filter_Col", "Value_Col"},
      {makeFlatVector<std::string>({"a", "b", "c", "d"}),
       makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), {vector});
  createDuckDbTable({vector});

  auto rowType = ROW({"filter_col", "value_col"}, {VARCHAR(), BIGINT()});
  auto outputType = ROW({"value_col"}, {BIGINT()});
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "filter_col",
              std::make_unique<common::BytesRange>(
                  "b",
                  /*lowerUnbounded*/ false,
                  /*lowerExclusive*/ false,
                  "",
                  /*upperUnbounded*/ true,
                  /*upperExclusive*/ false,
                  /*nullAllowed*/ false))
          .build();
  auto plan =
      PlanBuilder()
          .startTableScan()
          .outputType(outputType)
          .tableHandle(makeTableHandle(
              "parquet_table", rowType, std::move(subfieldFilters), nullptr))
          .assignments(
              facebook::velox::exec::test::HiveConnectorTestBase::
                  allRegularColumns(outputType))
          .endTableScan()
          .planNode();

  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .connectorSessionProperty(
          kCudfHiveConnectorId,
          facebook::velox::connector::hive::HiveConfig::
              kFileColumnNamesReadAsLowerCaseSession,
          "true")
      .splits(makeCudfHiveConnectorSplits({filePath}))
      .assertResults("SELECT Value_Col FROM tmp WHERE Filter_Col >= 'b'");
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TableScanTestParameterized,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "BufferedInput" : "FileDataSource";
    });

TEST_F(TableScanTest, mixedCaseDynamicIntegerFilter) {
  auto probe = makeRowVector(
      {"Key_Col"},
      {makeNullableFlatVector<int64_t>({1, 2, 100000, std::nullopt})});
  // A second row group without a build key.
  auto otherProbe =
      makeRowVector({"Key_Col"}, {makeFlatVector<int64_t>({5'000, 6'000})});
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), {probe, otherProbe});
  auto build = makeRowVector({"b"}, {makeFlatVector<int64_t>({1, 100000})});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildPlan = PlanBuilder(ids, pool_.get()).values({build}).planNode();
  auto rowType = ROW({"key_col"}, {BIGINT()});
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan =
      PlanBuilder(ids, pool_.get())
          .startTableScan()
          .connectorId(kCudfHiveConnectorId)
          .tableHandle(makeTableHandle("t", rowType))
          .outputType(rowType)
          .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
          .endTableScan()
          .capturePlanNodeId(scanId)
          .hashJoin(
              {"key_col"},
              {"b"},
              buildPlan,
              "",
              {"key_col"},
              core::JoinType::kInner)
          .capturePlanNodeId(joinId)
          .planNode();
  auto expected =
      makeRowVector({"key_col"}, {makeFlatVector<int64_t>({1, 100000})});
  auto task = AssertQueryBuilder(plan)
                  .config("hash_probe_dynamic_filter_pushdown_enabled", "true")
                  .connectorSessionProperty(
                      kCudfHiveConnectorId,
                      facebook::velox::connector::hive::HiveConfig::
                          kFileColumnNamesReadAsLowerCaseSession,
                      "true")
                  .maxDrivers(1)
                  .splits(scanId, makeCudfHiveConnectorSplits({file}))
                  .assertResults(expected);
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(stats.at(scanId).outputRows, 2);
  EXPECT_EQ(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
  // The statistics of the file column are found under the table's name.
  EXPECT_EQ(stats.at(scanId).customStats.at("skippedStrides").sum, 1);
  EXPECT_EQ(
      stats.at(joinId).customStats.at("replacedWithDynamicFilterRows").sum, 2);
}

TEST_F(TableScanTest, directBufferInputRawInputBytes) {
  constexpr int kSize = 10;
  auto vector = makeRowVector({
      makeFlatIdentityVector<int64_t>(kSize),
      makeFlatIdentityVector<int64_t>(kSize),
      makeFlatIdentityVector<int64_t>(kSize),
  });
  auto filePath = TempFilePath::create();
  createDuckDbTable({vector});
  writeToFile(filePath->getPath(), {vector});

  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .outputType(ROW({"c0", "c2"}, {BIGINT(), BIGINT()}))
                  .endTableScan()
                  .planNode();

  std::unordered_map<std::string, std::string> config;
  std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>
      connectorConfigs = {};
  auto queryCtx = core::QueryCtx::create(
      executor_.get(),
      core::QueryConfig(std::move(config)),
      connectorConfigs,
      nullptr);

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .splits(makeCudfHiveConnectorSplits({filePath}))
                  .queryCtx(queryCtx)
                  .assertResults("SELECT c0, c2 FROM tmp");

  // A quick sanity check for memory usage reporting. Check that peak total
  // memory usage for the project node is > 0.
  auto planStats = toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
  auto rawInputBytes = it->second.rawInputBytes;
  // Reduced from 500 to 400 as cudf CudfHive writer seems to be writing smaller
  // files.
  ASSERT_GE(rawInputBytes, 400);

  const auto runtimeStats = getTableScanRuntimeStats(task);
  assertStorageReadStats(runtimeStats, filePath->fileSize());
  ASSERT_GT(runtimeStats.at("totalScanTime").sum, 0);
  ASSERT_GT(runtimeStats.at("ioWaitWallNanos").sum, 0);
}

// The split's size differs from the file's so that the assertion tells which
// one the scan reports. Local opens take the size from the file itself.
TEST_F(TableScanTest, rawInputBytesFromSplitFileSize) {
  using cudf_velox::connector::hive::CudfHiveConfig;
  auto vectors = makeVectors(2, 100);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);
  const int64_t fileSize = filePath->fileSize();
  const int64_t splitFileSize = fileSize + 4'096;

  struct FileSizeSource {
    std::string name;
    std::function<void(hive::HiveConnectorSplitBuilder&)> setFileSize;
    int64_t expectedRawInputBytes;
  };
  const std::vector<FileSizeSource> fileSizeSources = {
      {"none", [](auto&) {}, fileSize},
      {"$file_size",
       [&](auto& builder) {
         builder.infoColumn("$file_size", std::to_string(splitFileSize));
       },
       splitFileSize},
      {"properties",
       [&](auto& builder) {
         builder.fileProperties(FileProperties{.fileSize = splitFileSize});
       },
       splitFileSize},
  };

  for (const auto* useBufferedInput : {"true", "false"}) {
    for (const auto& source : fileSizeSources) {
      SCOPED_TRACE(
          fmt::format(
              "use_buffered_input={} file size from {}",
              useBufferedInput,
              source.name));
      // A scan consumes its split, so every case builds a fresh one.
      hive::HiveConnectorSplitBuilder builder(filePath->getPath());
      builder.connectorId(kCudfHiveConnectorId)
          .fileFormat(dwio::common::FileFormat::PARQUET);
      source.setFileSize(builder);
      auto task = AssertQueryBuilder(tableScanNode(), duckDbQueryRunner_)
                      .connectorSessionProperty(
                          kCudfHiveConnectorId,
                          CudfHiveConfig::kUseBufferedInputSession,
                          useBufferedInput)
                      .split(builder.build())
                      .assertResults("SELECT * FROM tmp");
      EXPECT_EQ(
          getTableScanStats(task).rawInputBytes, source.expectedRawInputBytes);
    }
  }
}

TEST_P(TableScanTestParameterized, scanOnNonDefaultGpu) {
  int deviceCount = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    GTEST_SKIP() << "Requires two GPUs";
  }

  int originalDevice = -1;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&originalDevice));
  const int selectedDevice = originalDevice == 0 ? 1 : 0;
  ensureCudaContextForThread();
  unregisterCudf();
  SCOPE_EXIT {
    unregisterCudf();
    EXPECT_EQ(cudaSuccess, cudaSetDevice(originalDevice));
    registerCudf();
  };
  ASSERT_EQ(cudaSuccess, cudaSetDevice(selectedDevice));
  registerCudf();
  // Verify re-registration rebinds a thread still using the previous device.
  ASSERT_EQ(cudaSuccess, cudaSetDevice(originalDevice));
  ensureCudaContextForThread();
  int boundDevice = -1;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&boundDevice));
  ASSERT_EQ(selectedDevice, boundDevice);
  auto config = std::unordered_map<std::string, std::string>{
      {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
           kUseBufferedInput,
       GetParam() ? "true" : "false"}};
  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(std::move(config)));

  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000);
  for (auto i = 0; i < vectors.size(); ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "10")
                  .splits(makeCudfHiveConnectorSplits(filePaths))
                  .assertResults("SELECT * FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  const auto& customStats = planStats.at(plan->id()).customStats;
  ASSERT_EQ(customStats.count(std::string(TableScan::kPreloadedSplits)), 1);
  EXPECT_EQ(
      customStats.at(std::string(TableScan::kPreloadedSplits)).sum,
      filePaths.size());
}

TEST_F(TableScanTest, columnAliases) {
  auto vectors = makeVectors(1, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  std::string tableName = "t";
  std::unordered_map<std::string, std::string> aliases = {{"a", "c0"}};
  auto outputType = ROW({"a"}, {INTEGER()});
  auto tableHandle = makeTableHandle();
  auto op = PlanBuilder(pool_.get())
                .startTableScan()
                .tableHandle(tableHandle)
                .tableName(tableName)
                .outputType(outputType)
                .columnAliases(aliases)
                .endTableScan()
                .planNode();
  assertQuery(op, {filePath}, "SELECT c0 FROM tmp");
}

// Partition keys are not in the data file: the scan materializes each from
// the split's partition value, typed per its column handle, wherever the
// output puts it.
TEST_F(TableScanTest, partitionedTable) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});
  auto vectors = makeVectors(10, 1'000, rowType);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  const auto scan =
      [&](const RowTypePtr& outputType,
          const facebook::velox::connector::ColumnHandleMap& assignments) {
        return PlanBuilder(pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .outputType(outputType)
            .assignments(assignments)
            .endTableScan()
            .planNode();
      };

  // Scans a split in the partition 'partitionValue' of a key of
  // 'partitionType', expecting 'expectedValueSql' for the key. A split is
  // consumed by the task it is added to, so each scan gets its own.
  const auto assertPartitionedScan =
      [&](const TypePtr& partitionType,
          const std::optional<std::string>& partitionValue,
          const std::string& expectedValueSql) {
        SCOPED_TRACE(
            fmt::format(
                "{} key {}",
                partitionType->toString(),
                partitionValue.value_or("null")));
        const auto makeSplit = [&]() {
          return makePartitionedSplit(
              filePath->getPath(), {{"pkey", partitionValue}});
        };
        const facebook::velox::connector::ColumnHandleMap assignments = {
            {"pkey",
             HiveConnectorTestBase::partitionKey("pkey", partitionType)},
            {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
            {"c1", HiveConnectorTestBase::regularColumn("c1", DOUBLE())}};

        // The key at every output position.
        assertQuery(
            scan(
                ROW({"pkey", "c0", "c1"}, {partitionType, BIGINT(), DOUBLE()}),
                assignments),
            makeSplit(),
            fmt::format("SELECT {}, c0, c1 FROM tmp", expectedValueSql));
        assertQuery(
            scan(
                ROW({"c0", "pkey", "c1"}, {BIGINT(), partitionType, DOUBLE()}),
                assignments),
            makeSplit(),
            fmt::format("SELECT c0, {}, c1 FROM tmp", expectedValueSql));
        assertQuery(
            scan(
                ROW({"c0", "c1", "pkey"}, {BIGINT(), DOUBLE(), partitionType}),
                assignments),
            makeSplit(),
            fmt::format("SELECT c0, c1, {} FROM tmp", expectedValueSql));
        // The key alone: nothing is read from the file but its row count.
        assertQuery(
            scan(ROW({"pkey"}, {partitionType}), assignments),
            makeSplit(),
            fmt::format("SELECT {} FROM tmp", expectedValueSql));
      };

  // A value is parsed per the key's type.
  assertPartitionedScan(VARCHAR(), "2020-11-01", "'2020-11-01'");
  assertPartitionedScan(BIGINT(), "123456789123456789", "'123456789123456789'");
  assertPartitionedScan(INTEGER(), "123456789", "'123456789'");
  assertPartitionedScan(SMALLINT(), "1", "'1'");
  assertPartitionedScan(TINYINT(), "1", "'1'");
  assertPartitionedScan(BOOLEAN(), "0", "'0'");
  assertPartitionedScan(REAL(), "3.5", "'3.5'");
  assertPartitionedScan(DOUBLE(), "3.5", "'3.5'");
  assertPartitionedScan(
      DECIMAL(10, 2), "1234.56", "CAST(1234.56 AS DECIMAL(10, 2))");
  assertPartitionedScan(DATE(), "2023-10-27", "'2023-10-27'");
  // A NULL value, whatever the type.
  for (const auto& type :
       std::vector<TypePtr>{VARCHAR(), BIGINT(), DATE(), TIMESTAMP()}) {
    assertPartitionedScan(type, std::nullopt, "null");
  }

  {
    // A split read in several batches carries the key in each of them.
    SCOPED_TRACE("several batches");
    auto plan = scan(
        ROW({"pkey", "c0", "c1"}, {VARCHAR(), BIGINT(), DOUBLE()}),
        {{"pkey", HiveConnectorTestBase::partitionKey("pkey", VARCHAR())},
         {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
         {"c1", HiveConnectorTestBase::regularColumn("c1", DOUBLE())}});
    auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        cudf_velox::connector::hive::CudfHiveConfig::
                            kMaxChunkReadLimitSession,
                        "8192")
                    .connectorSessionProperty(
                        kCudfHiveConnectorId,
                        cudf_velox::connector::hive::CudfHiveConfig::
                            kMaxPassReadLimitSession,
                        "32768")
                    .splits({makePartitionedSplit(
                        filePath->getPath(), {{"pkey", "a"}})})
                    .assertResults("SELECT 'a', c0, c1 FROM tmp");
    const auto planStats = toPlanStats(task->taskStats());
    EXPECT_GT(
        planStats.at(plan->id()).operatorStatsFor("TableScan").outputVectors,
        1);
  }

  {
    // A DATE value is a count of days since the epoch when its handle says so.
    SCOPED_TRACE("days since epoch");
    const facebook::velox::connector::ColumnHandleMap assignments = {
        {"pkey",
         std::make_shared<hive::HiveColumnHandle>(
             "pkey",
             hive::HiveColumnHandle::ColumnType::kPartitionKey,
             DATE(),
             DATE(),
             std::vector<common::Subfield>{},
             hive::HiveColumnHandle::ColumnParseParameters{
                 hive::HiveColumnHandle::ColumnParseParameters::
                     kDaysSinceEpoch})},
        {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())}};
    assertQuery(
        scan(ROW({"pkey", "c0"}, {DATE(), BIGINT()}), assignments),
        makePartitionedSplit(filePath->getPath(), {{"pkey", "19657"}}),
        "SELECT '2023-10-27', c0 FROM tmp");
  }

  {
    // A TIMESTAMP value is local time or UTC, as the session says.
    SCOPED_TRACE("local time");
    const std::string partitionValue = "2023-10-27 00:12:35";
    auto plan = scan(
        ROW({"pkey", "c0"}, {TIMESTAMP(), BIGINT()}),
        {{"pkey", HiveConnectorTestBase::partitionKey("pkey", TIMESTAMP())},
         {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())}});
    const auto utcTimestamp =
        util::fromTimestampString(
            StringView(partitionValue), util::TimestampParseMode::kPrestoCast)
            .thenOrThrow(std::identity{}, [&](const Status& status) {
              VELOX_USER_FAIL("{}", status.message());
            });
    auto localTimestamp = utcTimestamp;
    localTimestamp.toGMT(Timestamp::defaultTimezone());
    for (const auto& [asLocalTime, expected] :
         {std::pair{false, utcTimestamp}, std::pair{true, localTimestamp}}) {
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .connectorSessionProperty(
              kCudfHiveConnectorId,
              hive::HiveConfig::kReadTimestampPartitionValueAsLocalTimeSession,
              asLocalTime ? "true" : "false")
          .splits({makePartitionedSplit(
              filePath->getPath(), {{"pkey", partitionValue}})})
          .assertResults(
              fmt::format("SELECT '{}', c0 FROM tmp", expected.toString()));
    }
  }

  {
    // An output name need not be the key's name, a key may be projected under
    // several names, and a key the split carries but no handle requests is
    // ignored.
    SCOPED_TRACE("alias");
    const auto makeSplit = [&]() {
      return makePartitionedSplit(filePath->getPath(), {{"ds", "2021-12-02"}});
    };
    assertQuery(
        scan(
            ROW({"a", "ds_alias"}, {BIGINT(), VARCHAR()}),
            {{"a", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
             {"ds_alias",
              HiveConnectorTestBase::partitionKey("ds", VARCHAR())}}),
        makeSplit(),
        "SELECT c0, '2021-12-02' FROM tmp");
    assertQuery(
        scan(
            ROW({"ds_alias", "a", "ds"}, {VARCHAR(), BIGINT(), VARCHAR()}),
            {{"a", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
             {"ds_alias", HiveConnectorTestBase::partitionKey("ds", VARCHAR())},
             {"ds", HiveConnectorTestBase::partitionKey("ds", VARCHAR())}}),
        makeSplit(),
        "SELECT '2021-12-02', c0, '2021-12-02' FROM tmp");
    assertQuery(
        scan(
            ROW({"a"}, {BIGINT()}),
            {{"a", HiveConnectorTestBase::regularColumn("c0", BIGINT())}}),
        makeSplit(),
        "SELECT c0 FROM tmp");
  }
}

// A TIMESTAMP partition key takes the type the reader gives file timestamps,
// whatever unit cuDF expressions default to, so an expression can combine the
// two.
TEST_F(TableScanTest, partitionTimestampMixedWithFileTimestamp) {
  auto& config = CudfConfig::getInstance();
  const auto originalUnit = config.timestampUnit;
  SCOPE_EXIT {
    config.timestampUnit = originalUnit;
  };

  auto input = makeRowVector(
      {"flag", "file_ts"},
      {makeFlatVector<bool>({true, false}),
       makeFlatVector<Timestamp>(
           {Timestamp(1'700'000'000, 0), Timestamp(1'700'000'001, 0)})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), input);
  const auto outputType =
      ROW({"flag", "file_ts", "partition_ts"},
          {BOOLEAN(), TIMESTAMP(), TIMESTAMP()});
  const auto fileValues = makeRowVector({input->childAt(1)});
  const auto caseValues = makeRowVector({makeFlatVector<Timestamp>(
      {Timestamp(1'577'836'800, 0), Timestamp(1'700'000'001, 0)})});

  // The reader's timestamp type is pinned to a unit distinct from both default
  // units below, so each scan proves the key follows the reader rather than a
  // fixed unit. The connector and session settings take the enumerator's
  // integer value.
  const auto readerTimestampType = std::to_string(
      static_cast<int32_t>(cudf::type_id::TIMESTAMP_MICROSECONDS));

  // Projects 'projection' over the file and partition timestamps of a split
  // in the partition 'partitionValue', expecting 'expected'. The session pins
  // the reader's timestamp type to 'sessionTimestampType' when that is set.
  const auto check = [&](const std::string& projection,
                         const std::optional<std::string>& partitionValue,
                         const RowVectorPtr& expected,
                         const std::optional<std::string>&
                             sessionTimestampType) {
    SCOPED_TRACE(projection);
    auto plan =
        PlanBuilder(pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .outputType(outputType)
            .assignments(
                {{"flag",
                  HiveConnectorTestBase::regularColumn("flag", BOOLEAN())},
                 {"file_ts",
                  HiveConnectorTestBase::regularColumn("file_ts", TIMESTAMP())},
                 {"partition_ts",
                  HiveConnectorTestBase::partitionKey(
                      "partition_ts", TIMESTAMP())}})
            .endTableScan()
            .project({projection})
            .planNode();
    AssertQueryBuilder builder(plan, duckDbQueryRunner_);
    builder.connectorSessionProperty(
        kCudfHiveConnectorId,
        hive::HiveConfig::kReadTimestampPartitionValueAsLocalTimeSession,
        "false");
    if (sessionTimestampType.has_value()) {
      builder.connectorSessionProperty(
          kCudfHiveConnectorId,
          cudf_velox::connector::hive::CudfHiveConfig::kTimestampTypeSession,
          *sessionTimestampType);
    }
    builder
        .splits({makePartitionedSplit(
            filePath->getPath(), {{"partition_ts", partitionValue}})})
        .assertResults(expected);
  };

  for (const auto& [unit, label] :
       {std::pair{cudf::type_id::TIMESTAMP_MILLISECONDS, "ms default unit"},
        std::pair{cudf::type_id::TIMESTAMP_NANOSECONDS, "ns default unit"}}) {
    SCOPED_TRACE(label);
    config.timestampUnit = unit;
    check("file_ts", std::nullopt, fileValues, readerTimestampType);
    check(
        "coalesce(partition_ts, file_ts)",
        std::nullopt,
        fileValues,
        readerTimestampType);
    check(
        "CASE WHEN flag THEN partition_ts ELSE file_ts END",
        "2020-01-01 00:00:00",
        caseValues,
        readerTimestampType);
  }

  // Without a session timestamp type, the key follows the connector's
  // parquet.reader.timestamp-type.
  {
    SCOPED_TRACE("connector timestamp type");
    config.timestampUnit = cudf::type_id::TIMESTAMP_MILLISECONDS;
    resetCudfHiveConnector(
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{
                {cudf_velox::connector::hive::CudfHiveConfig::kTimestampType,
                 readerTimestampType}}));
    check(
        "coalesce(partition_ts, file_ts)",
        /*partitionValue=*/std::nullopt,
        fileValues,
        /*sessionTimestampType=*/std::nullopt);
  }
}

// A pushed-down filter on a partition key is decided once per split from the
// split's partition value: a split it rejects is skipped without being read.
TEST_F(TableScanTest, partitionKeyFilter) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});
  auto vectors = makeVectors(5, 1'000, rowType);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  using PartitionKeys =
      std::unordered_map<std::string, std::optional<std::string>>;
  // The same file in the partitions 'a', 'b' and NULL. A split is consumed by
  // the task it is added to, so each scan gets fresh ones.
  std::vector<PartitionKeys> partitions = {
      PartitionKeys{{"ds", "a"}},
      PartitionKeys{{"ds", "b"}},
      PartitionKeys{{"ds", std::nullopt}}};
  const auto makeSplits = [&]() {
    std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
        splits;
    for (const auto& partition : partitions) {
      splits.push_back(makePartitionedSplit(filePath->getPath(), partition));
    }
    return splits;
  };
  facebook::velox::connector::ColumnHandleMap assignments = {
      {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
      {"ds", HiveConnectorTestBase::partitionKey("ds", VARCHAR())}};

  // Scans the partitions with 'filter' on the key, checking the rows and how
  // many splits were skipped. The table schema lists the file columns only.
  const auto assertFilteredScan = [&](std::unique_ptr<common::Filter> filter,
                                      const RowTypePtr& outputType,
                                      const std::string& expectedSql,
                                      int64_t expectedSkippedSplits) {
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .outputType(outputType)
                    .tableHandle(makeTableHandle(
                        "parquet_table",
                        rowType,
                        common::test::SubfieldFiltersBuilder()
                            .add("ds", std::move(filter))
                            .build()))
                    .assignments(assignments)
                    .endTableScan()
                    .planNode();
    auto task = OperatorTestBase::assertQuery(plan, makeSplits(), expectedSql);
    EXPECT_EQ(
        getTableScanRuntimeStat(task, "skippedSplits"), expectedSkippedSplits);
    EXPECT_EQ(
        getTableScanRuntimeStat(task, "processedSplits"),
        static_cast<int64_t>(partitions.size()) - expectedSkippedSplits);
  };

  {
    SCOPED_TRACE("ds = 'a'");
    assertFilteredScan(
        std::make_unique<common::BytesValues>(
            std::vector<std::string>{"a"}, false),
        ROW({"c0", "ds"}, {BIGINT(), VARCHAR()}),
        "SELECT c0, 'a' FROM tmp",
        2);
  }
  {
    SCOPED_TRACE("ds IS NULL");
    assertFilteredScan(
        std::make_unique<common::IsNull>(),
        ROW({"c0", "ds"}, {BIGINT(), VARCHAR()}),
        "SELECT c0, null FROM tmp",
        2);
  }
  {
    // The key need not be projected.
    SCOPED_TRACE("ds IS NOT NULL");
    assertFilteredScan(
        std::make_unique<common::IsNotNull>(),
        ROW({"c0"}, {BIGINT()}),
        "SELECT c0 FROM tmp UNION ALL SELECT c0 FROM tmp",
        1);
  }
  {
    SCOPED_TRACE("ds = 'b' OR ds IS NULL");
    assertFilteredScan(
        std::make_unique<common::BytesValues>(
            std::vector<std::string>{"b"}, true),
        ROW({"c0"}, {BIGINT()}),
        "SELECT c0 FROM tmp UNION ALL SELECT c0 FROM tmp",
        1);
  }
  {
    // A key the split does not carry is NULL.
    SCOPED_TRACE("missing key");
    partitions = {PartitionKeys{}};
    assertFilteredScan(
        std::make_unique<common::IsNull>(),
        ROW({"c0"}, {BIGINT()}),
        "SELECT c0 FROM tmp",
        0);
    assertFilteredScan(
        std::make_unique<common::BytesValues>(
            std::vector<std::string>{"a"}, false),
        ROW({"c0"}, {BIGINT()}),
        "SELECT c0 FROM tmp WHERE false",
        1);
  }
  {
    // A typed key with a range filter: ds > DATE '2020-09-01' over day counts.
    SCOPED_TRACE("date range");
    partitions = {
        PartitionKeys{{"ds", "19657"}}, PartitionKeys{{"ds", "18000"}}};
    assignments["ds"] = std::make_shared<hive::HiveColumnHandle>(
        "ds",
        hive::HiveColumnHandle::ColumnType::kPartitionKey,
        DATE(),
        DATE(),
        std::vector<common::Subfield>{},
        hive::HiveColumnHandle::ColumnParseParameters{
            hive::HiveColumnHandle::ColumnParseParameters::kDaysSinceEpoch});
    assertFilteredScan(
        std::make_unique<common::BigintRange>(
            18506, std::numeric_limits<int64_t>::max(), false),
        ROW({"c0", "ds"}, {BIGINT(), DATE()}),
        "SELECT c0, '2023-10-27' FROM tmp",
        1);
  }
  {
    // A key only the filter references comes with the table handle's filter
    // column handles.
    SCOPED_TRACE("filter column handle");
    partitions = {PartitionKeys{{"ds", "a"}}, PartitionKeys{{"ds", "b"}}};
    auto tableHandle = std::make_shared<hive::HiveTableHandle>(
        kCudfHiveConnectorId,
        "parquet_table",
        common::test::SubfieldFiltersBuilder()
            .add(
                "ds",
                std::make_unique<common::BytesValues>(
                    std::vector<std::string>{"a"}, false))
            .build(),
        nullptr,
        rowType,
        std::vector<std::string>{},
        std::unordered_map<std::string, std::string>{},
        std::vector<hive::HiveColumnHandlePtr>{
            HiveConnectorTestBase::partitionKey("ds", VARCHAR())});
    auto plan =
        PlanBuilder(pool_.get())
            .startTableScan()
            .outputType(ROW({"c0"}, {BIGINT()}))
            .tableHandle(tableHandle)
            .assignments(
                {{"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())}})
            .endTableScan()
            .planNode();
    auto task =
        OperatorTestBase::assertQuery(plan, makeSplits(), "SELECT c0 FROM tmp");
    EXPECT_EQ(getTableScanRuntimeStat(task, "skippedSplits"), 1);
  }
  {
    // The decision on a split prepared ahead of time carries over to the
    // driver that adopts it.
    SCOPED_TRACE("preloaded splits");
    partitions.clear();
    for (int i = 0; i < 3; ++i) {
      partitions.insert(
          partitions.end(),
          {PartitionKeys{{"ds", "a"}},
           PartitionKeys{{"ds", "b"}},
           PartitionKeys{{"ds", std::nullopt}}});
    }
    assignments["ds"] = HiveConnectorTestBase::partitionKey("ds", VARCHAR());
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .outputType(ROW({"c0", "ds"}, {BIGINT(), VARCHAR()}))
                    .tableHandle(makeTableHandle(
                        "parquet_table",
                        rowType,
                        common::test::SubfieldFiltersBuilder()
                            .add(
                                "ds",
                                std::make_unique<common::BytesValues>(
                                    std::vector<std::string>{"a"}, false))
                            .build()))
                    .assignments(assignments)
                    .endTableScan()
                    .singleAggregation({"ds"}, {"count(1)"})
                    .planNode();
    auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                    .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "4")
                    .splits(makeSplits())
                    .assertResults("SELECT 'a', 3 * count(*) FROM tmp");
    EXPECT_EQ(getTableScanRuntimeStat(task, "skippedSplits"), 6);
    EXPECT_EQ(getTableScanRuntimeStat(task, "processedSplits"), 3);
  }
}

// The remaining filter sees a partition key as a constant column of the split,
// whether or not the key is projected and whether or not a file column is
// read along with it.
TEST_F(TableScanTest, partitionKeyRemainingFilter) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});
  auto vectors = makeVectors(5, 1'000, rowType);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  // The same file in the partitions 'a' and 'bb'. A split is consumed by the
  // task it is added to, so each scan gets fresh ones.
  const auto makeSplits = [&]() {
    return std::vector<
        std::shared_ptr<facebook::velox::connector::ConnectorSplit>>{
        makePartitionedSplit(filePath->getPath(), {{"ds", "a"}}),
        makePartitionedSplit(filePath->getPath(), {{"ds", "bb"}})};
  };
  const facebook::velox::connector::ColumnHandleMap assignments = {
      {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())},
      {"c1", HiveConnectorTestBase::regularColumn("c1", DOUBLE())},
      {"ds", HiveConnectorTestBase::partitionKey("ds", VARCHAR())}};
  // The filter is parsed against the table schema, which lists the key here.
  const auto dataColumns =
      ROW({"c0", "c1", "ds"}, {BIGINT(), DOUBLE(), VARCHAR()});

  const auto assertFilteredScan = [&](const RowTypePtr& outputType,
                                      const std::string& remainingFilter,
                                      const std::string& expectedSql,
                                      int64_t expectedSkippedSplits) {
    SCOPED_TRACE(remainingFilter);
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .connectorId(kCudfHiveConnectorId)
                    .outputType(outputType)
                    .dataColumns(dataColumns)
                    .assignments(assignments)
                    .remainingFilter(remainingFilter)
                    .endTableScan()
                    .planNode();
    auto task = OperatorTestBase::assertQuery(plan, makeSplits(), expectedSql);
    EXPECT_EQ(
        getTableScanRuntimeStat(task, "skippedSplits"), expectedSkippedSplits);
  };

  // A filter over the key and a file column runs after the read, with the key
  // projected or not.
  assertFilteredScan(
      ROW({"c0", "ds"}, {BIGINT(), VARCHAR()}),
      "ds = 'a' OR c0 > 500",
      "SELECT c0, 'a' FROM tmp UNION ALL "
      "SELECT c0, 'bb' FROM tmp WHERE c0 > 500",
      0);
  assertFilteredScan(
      ROW({"c1"}, {DOUBLE()}),
      "ds = 'a' OR c0 > 500",
      "SELECT c1 FROM tmp UNION ALL SELECT c1 FROM tmp WHERE c0 > 500",
      0);
  // A comparison of the key alone is decided per split.
  assertFilteredScan(
      ROW({"c0"}, {BIGINT()}), "ds = 'bb'", "SELECT c0 FROM tmp", 1);
  // Any other filter on the key alone runs over the materialized key, even
  // when no file column is read.
  assertFilteredScan(
      ROW({"c0"}, {BIGINT()}), "length(ds) = 2", "SELECT c0 FROM tmp", 0);
  assertFilteredScan(
      ROW({"ds"}, {VARCHAR()}), "length(ds) = 2", "SELECT 'bb' FROM tmp", 0);
}

// A split round-trips through its serialized form with its partition keys,
// NULL ones included.
TEST_F(TableScanTest, partitionedSplitSerde) {
  auto split = CudfHiveConnectorSplitBuilder("/path/to/file.parquet")
                   .infoColumn("$file_size", "123")
                   .partitionKey("ds", "2021-12-02")
                   .partitionKey("region", std::nullopt)
                   .build();
  auto copy = cudf_velox::connector::hive::CudfHiveConnectorSplit::create(
      split->serialize());
  EXPECT_EQ(copy->filePath, split->filePath);
  EXPECT_EQ(copy->infoColumns, split->infoColumns);
  EXPECT_EQ(copy->partitionKeys, split->partitionKeys);
}

TEST_F(TableScanTest, dynamicFilterUsesPhysicalNamesWithoutDataColumns) {
  auto probe = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({9})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);

  auto build = makeRowVector({"u_key"}, {makeFlatVector<int64_t>({1})});
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide =
      PlanBuilder(planNodeIdGenerator, pool_.get()).values({build}).planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto outputType = ROW({"c1", "c0"}, {BIGINT(), BIGINT()});
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .tableHandle(makeTableHandle())
                  .outputType(outputType)
                  .columnAliases({{"c1", "c0"}, {"c0", "c1"}})
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .hashJoin(
                      {"c1"},
                      {"u_key"},
                      buildSide,
                      "",
                      {"c1", "c0"},
                      core::JoinType::kInner)
                  .capturePlanNodeId(joinId)
                  .planNode();
  auto expected = makeRowVector(
      {"c1", "c0"},
      {makeFlatVector<int64_t>({1}), makeFlatVector<int64_t>({9})});
  auto task = AssertQueryBuilder(plan)
                  .config(CudfConfig::kCudfEnabled, "true")
                  .maxDrivers(1)
                  .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                  .assertResults(expected);
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 1);
  EXPECT_EQ(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
}

TEST_F(TableScanTest, filterPushdown) {
  auto rowType =
      ROW({"c0", "c1", "c2", "c3"}, {TINYINT(), BIGINT(), DOUBLE(), BOOLEAN()});
  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000, rowType);
  for (int32_t i = 0; i < vectors.size(); i++) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  // c1 >= 0 or null and c3 is true
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c1",
              std::make_unique<common::BigintRange>(
                  int64_t(0), std::numeric_limits<int64_t>::max(), true))
          .add("c3", std::make_unique<common::BoolValue>(true, false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto task = assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({"c1", "c3", "c0"}, {BIGINT(), BOOLEAN(), TINYINT()}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .planNode(),
      filePaths,
      "SELECT c1, c3, c0 FROM tmp WHERE (c1 >= 0 OR c1 IS NULL) AND c3");

  auto tableScanStats = getTableScanStats(task);
  // EXPECT_EQ(tableScanStats.rawInputRows, 10'000);
  // EXPECT_LT(tableScanStats.inputRows, tableScanStats.rawInputRows);
  EXPECT_EQ(tableScanStats.inputRows, tableScanStats.outputRows);

  // Repeat the same but do not project out the filtered columns.
  assignments.clear();
  assignments["c0"] =
      facebook::velox::exec::test::HiveConnectorTestBase::regularColumn(
          "c0", TINYINT());
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({"c0"}, {TINYINT()}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .planNode(),
      filePaths,
      "SELECT c0 FROM tmp WHERE (c1 >= 0 OR c1 IS NULL) AND c3");

  // Count with no projections, however the filter columns c1 and c3 are still
  // read.
  assignments.clear();
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({}, {}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .singleAggregation({}, {"count(1)"})
          .planNode(),
      filePaths,
      "SELECT count(*) FROM tmp WHERE (c1 >= 0 OR c1 IS NULL) AND c3");

  // Do the same for count, no filter, no projections. Only the footer is read.
  assignments.clear();
  tableHandle = makeTableHandle("parquet_table", rowType);
  assertQuery(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({}, {}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .singleAggregation({}, {"count(1)"})
          .planNode(),
      filePaths,
      "SELECT count(*) FROM tmp");
}

// The tile decode kernels are switched by a session property. The scan returns
// the same rows with it on and off.
TEST_F(TableScanTest, tileDecode) {
  using cudf_velox::connector::hive::CudfHiveConfig;
  auto rowType = ROW({{"c0", INTEGER()}, {"c1", BIGINT()}, {"c2", BIGINT()}});
  auto filePaths = makeFilePaths(4);
  auto vectors = makeVectors(4, 1'000, rowType);
  for (int32_t i = 0; i < vectors.size(); i++) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  // c1 >= 0 and c2 < 0, nulls rejected.
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c1",
              std::make_unique<common::BigintRange>(
                  int64_t(0), std::numeric_limits<int64_t>::max(), false))
          .add(
              "c2",
              std::make_unique<common::BigintRange>(
                  std::numeric_limits<int64_t>::min(), int64_t(-1), false))
          .build();
  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);
  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(ROW({{"c0", INTEGER()}, {"c1", BIGINT()}}))
                  .tableHandle(tableHandle)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  for (const auto* tileDecode : {"false", "true"}) {
    SCOPED_TRACE(fmt::format("tile decode {}", tileDecode));
    AssertQueryBuilder(plan, duckDbQueryRunner_)
        .connectorSessionProperty(
            kCudfHiveConnectorId,
            CudfHiveConfig::kUseTileDecodeSession,
            tileDecode)
        .splits(makeCudfHiveConnectorSplits(filePaths))
        .assertResults("SELECT c0, c1 FROM tmp WHERE c1 >= 0 AND c2 < 0");
  }
}

// Presto pushes an equality on a column into the scan and re-creates the
// column above it as a constant projection. When the scan reads no other
// column, that projection sits over a zero-column input and still runs on the
// GPU, filled to the row count of each batch.
TEST_F(TableScanTest, constantProjectionOverZeroColumnScan) {
  auto rowType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto data = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
       makeFlatVector<std::string>({"a", "b", "a", "c", "a"})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), data);

  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c1",
              std::make_unique<common::BytesValues>(
                  std::vector<std::string>{"a"}, false))
          .build();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(ROW({}, {}))
                  .tableHandle(makeTableHandle(
                      "parquet_table", rowType, std::move(subfieldFilters)))
                  .endTableScan()
                  .project({"'a' AS c1"})
                  .singleAggregation({"c1"}, {"count(1)"})
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .split(makeCudfHiveSplit(filePath->getPath()))
                    .copyResults(pool_.get(), task);
  assertEqualResults(
      {makeRowVector(
          {makeFlatVector<std::string>({"a"}), makeFlatVector<int64_t>({3})})},
      {result});

  bool cudfFilterProjectUsed = false;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      cudfFilterProjectUsed |= op.operatorType == "CudfFilterProject";
    }
  }
  EXPECT_TRUE(cudfFilterProjectUsed);
}

TEST_F(TableScanTest, decimalFilterPushdown) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(12, 2), DECIMAL(20, 2)});

  auto vector = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int64_t>(
              {123, 500, -250, 300, 400, 200}, DECIMAL(12, 2)),
          makeFlatVector<int128_t>(
              {int128_t{200},
               int128_t{200},
               int128_t{700},
               int128_t{700},
               int128_t{900},
               int128_t{-100}},
              DECIMAL(20, 2)),
      });

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  // c0 between 1.00 and 4.00 and c1 in (2.00, 7.00)
  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{100}, int64_t{400}, /*nullAllowed*/ false))
          .add(
              "c1",
              common::createHugeintValues(
                  {int128_t{200}, int128_t{700}}, /*nullAllowed*/ false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT c0, c1 FROM tmp "
      "WHERE c0 BETWEEN CAST('1.00' AS DECIMAL(12, 2)) "
      "AND CAST('4.00' AS DECIMAL(12, 2)) "
      "AND c1 IN (CAST('2.00' AS DECIMAL(20, 2)), "
      "CAST('7.00' AS DECIMAL(20, 2)))");
}

TEST_F(TableScanTest, decimalStatsFilterIoPruning) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(12, 2), DECIMAL(20, 2)});
  auto vec0 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, 200}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{1000}, int128_t{2000}}, DECIMAL(20, 2))});
  auto vec1 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({300, 400}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{3000}, int128_t{4000}}, DECIMAL(20, 2))});
  auto vec2 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({500, 600}, DECIMAL(12, 2)),
       makeFlatVector<int128_t>(
           {int128_t{5000}, int128_t{6000}}, DECIMAL(20, 2))});

  std::vector<RowVectorPtr> vectors = {vec0, vec1, vec2};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  common::SubfieldFilters filters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{300}, int64_t{400}, /*nullAllowed*/ false))
          .add(
              "c1",
              std::make_unique<common::HugeintRange>(
                  int128_t{3000}, int128_t{4000}, /*nullAllowed*/ false))
          .build();

  auto metrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, filters, /*useJitFilter*/ true);
  EXPECT_EQ(metrics.inputRowGroups, 3);
  ASSERT_TRUE(metrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(metrics.rowGroupsAfterStats.value(), 1);
  EXPECT_EQ(metrics.outputRows, 2);
}

TEST_F(TableScanTest, doubleStatsFilterIoPruning) {
  auto rowType = ROW({"c0", "c1"}, {DOUBLE(), DOUBLE()});
  auto vec0 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({1.0, 2.0}),
       makeFlatVector<double>({10.0, 20.0})});
  auto vec1 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({3.0, 4.0}),
       makeFlatVector<double>({30.0, 40.0})});
  auto vec2 = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<double>({5.0, 6.0}),
       makeFlatVector<double>({50.0, 60.0})});

  std::vector<RowVectorPtr> vectors = {vec0, vec1, vec2};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  common::SubfieldFilters filters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::DoubleRange>(
                  3.0,
                  /*lowerUnbounded*/ false,
                  /*lowerExclusive*/ false,
                  4.0,
                  /*upperUnbounded*/ false,
                  /*upperExclusive*/ false,
                  /*nullAllowed*/ false))
          .add(
              "c1",
              std::make_unique<common::DoubleRange>(
                  30.0,
                  /*lowerUnbounded*/ false,
                  /*lowerExclusive*/ false,
                  40.0,
                  /*upperUnbounded*/ false,
                  /*upperExclusive*/ false,
                  /*nullAllowed*/ false))
          .build();

  auto metrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, filters, /*useJitFilter*/ true);
  EXPECT_EQ(metrics.inputRowGroups, 3);
  ASSERT_TRUE(metrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(metrics.rowGroupsAfterStats.value(), 1);
  EXPECT_EQ(metrics.outputRows, 2);
}

TEST_F(TableScanTest, splitOffsetAndLength) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  // Note that the number of row groups selected within `halfFileSize` may
  // change in the future and this test may start failing. In such a case,
  // just adjust the duckdb sql string accordingly.
  const auto halfFileSize = fs::file_size(filePath->getPath()) / 2;

  // First half of file - OFFSET 0 LIMIT 6000
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), 0, halfFileSize),
      "SELECT * FROM tmp OFFSET 0 LIMIT 6000");

  // Second half of file - OFFSET 6000 LIMIT 4000
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), halfFileSize),
      "SELECT * FROM tmp OFFSET 6000 LIMIT 4000");

  const auto fileSize = fs::file_size(filePath->getPath());

  // All row groups
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), 0, fileSize),
      "SELECT * FROM tmp");

  // No row groups
  assertQuery(
      tableScanNode(),
      makeCudfHiveConnectorSplit(filePath->getPath(), fileSize),
      "SELECT * FROM tmp LIMIT 0");
}

// A split carries the file size in its properties or, as Presto sends it, in
// the $file_size info column; both survive a serialization round trip.
TEST_F(TableScanTest, splitCarriesFileSize) {
  using cudf_velox::connector::hive::CudfHiveConnectorSplit;
  auto split =
      CudfHiveConnectorSplitBuilder("/data/file.parquet")
          .fileProperties(
              FileProperties{.fileSize = 1'234, .modificationTime = 5'678})
          .build();
  auto copy = CudfHiveConnectorSplit::create(split->serialize());
  ASSERT_TRUE(copy->properties.has_value());
  EXPECT_EQ(copy->properties->fileSize, 1'234);
  EXPECT_EQ(copy->properties->modificationTime, 5'678);
  EXPECT_EQ(copy->fileSize(), 1'234);

  auto fromInfoColumn = CudfHiveConnectorSplitBuilder("/data/file.parquet")
                            .infoColumn("$file_size", "1234")
                            .build();
  EXPECT_FALSE(fromInfoColumn->properties.has_value());
  EXPECT_EQ(fromInfoColumn->fileSize(), 1'234);
  EXPECT_EQ(
      CudfHiveConnectorSplit::create(fromInfoColumn->serialize())->fileSize(),
      1'234);

  auto bare = CudfHiveConnectorSplitBuilder("/data/file.parquet").build();
  EXPECT_FALSE(bare->fileSize().has_value());
  EXPECT_FALSE(
      CudfHiveConnectorSplit::create(bare->serialize())
          ->properties.has_value());

  // A negative size in the properties is no size, so the info column stands
  // in for it, and a valid size in the properties wins over the info column.
  EXPECT_EQ(
      CudfHiveConnectorSplitBuilder("/data/file.parquet")
          .fileProperties(FileProperties{.fileSize = -1})
          .infoColumn("$file_size", "1234")
          .build()
          ->fileSize(),
      1'234);
  EXPECT_FALSE(CudfHiveConnectorSplitBuilder("/data/file.parquet")
                   .fileProperties(FileProperties{.fileSize = -1})
                   .build()
                   ->fileSize()
                   .has_value());
  EXPECT_EQ(
      CudfHiveConnectorSplitBuilder("/data/file.parquet")
          .fileProperties(FileProperties{.fileSize = 1'234})
          .infoColumn("$file_size", "5678")
          .build()
          ->fileSize(),
      1'234);
  for (const auto* notASize : {"", "abc", "-1", "12.5"}) {
    EXPECT_FALSE(CudfHiveConnectorSplitBuilder("/data/file.parquet")
                     .infoColumn("$file_size", notASize)
                     .build()
                     ->fileSize()
                     .has_value())
        << notASize;
  }
}

// Verify that a `count(*)` scan with no projected columns returns the correct
// row count from the Parquet footer
TEST_F(TableScanTest, countStarNoProjectedColumns) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  // Scans no columns, the shape a `SELECT count(*)` plan produces.
  auto countStarPlan = PlanBuilder(pool_.get())
                           .startTableScan()
                           .outputType(ROW({}, {}))
                           .tableHandle(makeTableHandle())
                           .endTableScan()
                           .singleAggregation({}, {"count(1)"})
                           .planNode();

  // Counts the same rows through a decoded column, the oracle for the
  // footer-derived counts below.
  auto countColumnPlan = PlanBuilder(pool_.get())
                             .startTableScan()
                             .outputType(ROW({"c0"}, {INTEGER()}))
                             .tableHandle(makeTableHandle())
                             .endTableScan()
                             .singleAggregation({}, {"count(1)"})
                             .planNode();

  const auto fileSize = fs::file_size(filePath->getPath());
  const auto halfFileSize = fileSize / 2;

  const auto countRows = [&](const core::PlanNodePtr& plan,
                             uint64_t start,
                             uint64_t length) {
    std::shared_ptr<Task> task;
    auto result = AssertQueryBuilder(plan)
                      .split(Split(makeCudfHiveConnectorSplit(
                          filePath->getPath(), start, length)))
                      .copyResults(pool_.get(), task);
    EXPECT_EQ(result->size(), 1);
    // A footer-only scan reads no column chunks.
    if (plan == countStarPlan) {
      EXPECT_LT(
          getTableScanRuntimeStat(task, io::kStorageReadBytes), fileSize / 4);
    }
    return result->childAt(0)->as<SimpleVector<int64_t>>()->valueAt(0);
  };

  // A split owns only the row groups that start inside its byte range, so a
  // whole-file footer count would over-count either half.
  const auto firstHalf = countRows(countStarPlan, 0, halfFileSize);
  const auto secondHalf =
      countRows(countStarPlan, halfFileSize, fileSize - halfFileSize);
  EXPECT_EQ(firstHalf, countRows(countColumnPlan, 0, halfFileSize));
  EXPECT_EQ(
      secondHalf,
      countRows(countColumnPlan, halfFileSize, fileSize - halfFileSize));
  EXPECT_EQ(firstHalf + secondHalf, 10'000);

  // A split that starts past the last row group owns no rows.
  EXPECT_EQ(countRows(countStarPlan, fileSize, 1), 0);
}

// A remaining filter that folds to a constant must either keep the row count
// from the footer or skip the split without reading any columns.
TEST_F(TableScanTest, constantRemainingFilter) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);

  using CountAndSkippedSplits = std::pair<int64_t, int64_t>;
  const auto countRows = [&](const core::TypedExprPtr& remainingFilter) {
    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .outputType(ROW({}, {}))
                    .tableHandle(makeTableHandle(
                        "parquet_table", rowType_, {}, remainingFilter))
                    .endTableScan()
                    .singleAggregation({}, {"count(1)"})
                    .planNode();
    std::shared_ptr<Task> task;
    auto result = AssertQueryBuilder(plan)
                      .split(makeCudfHiveSplit(filePath->getPath()))
                      .copyResults(pool_.get(), task);
    EXPECT_EQ(result->size(), 1);
    return CountAndSkippedSplits{
        result->childAt(0)->as<SimpleVector<int64_t>>()->valueAt(0),
        getTableScanRuntimeStat(task, "skippedSplits")};
  };

  EXPECT_EQ(
      countRows(std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), true)),
      CountAndSkippedSplits(10'000, 0));
  EXPECT_EQ(
      countRows(std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), false)),
      CountAndSkippedSplits(0, 1));
  EXPECT_EQ(
      countRows(
          std::make_shared<core::ConstantTypedExpr>(
              BOOLEAN(), Variant::null(TypeKind::BOOLEAN))),
      CountAndSkippedSplits(0, 1));

  // An unsupported non-constant filter that references no column falls back
  // to the CPU data source.
  auto randomFilter = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::CallTypedExpr>(
              DOUBLE(), std::vector<core::TypedExprPtr>{}, "rand"),
          std::make_shared<core::ConstantTypedExpr>(DOUBLE(), -1.0),
      },
      "gt");
  EXPECT_EQ(countRows(randomFilter), CountAndSkippedSplits(10'000, 0));
}

// Verify that extractFiltersFromRemainingFilter extracts simple single-column
// filters from the remaining filter into subfield filters for pushdown.
// When a filter like "c0 = 1" is fully extracted,
// cudfRemainingFilterExpression_ is null and totalRemainingFilterWallNanos is
// 0. Without extraction, the filter runs post-read on the GPU and the stat is
// > 0.
TEST_F(TableScanTest, remainingFilterExtraction) {
  auto rowType = ROW({"c0", "c1", "c2"}, {BIGINT(), BIGINT(), DOUBLE()});
  auto vectors = makeVectors(5, 1'000, rowType);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  // "c0 = 1" is a single-column equality that should be fully extracted into
  // a subfield filter, leaving no remaining filter to evaluate post-read.
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(assignments)
                  .remainingFilter("c0 = 1")
                  .endTableScan()
                  .planNode();

  auto task = assertQuery(plan, {filePath}, "SELECT * FROM tmp WHERE c0 = 1");

  // Verify the filter was fully extracted: no post-read remaining filter ran.
  auto planStats = toPlanStats(task->taskStats());
  const auto& scanStats = planStats.at(plan->id());
  auto it = scanStats.customStats.find("totalRemainingFilterWallNanos");
  ASSERT_NE(it, scanStats.customStats.end());
  EXPECT_EQ(it->second.sum, 0)
      << "Expected no remaining filter time when filter is fully extracted";
}

TEST_F(TableScanTest, remainingFilterEligibility) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto data = makeNamedRowVector("c0", makeFlatVector<int64_t>({1, 2, 3, 4}));
  const auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto testFilter = [&](const std::string& filter,
                        const RowVectorPtr& expected,
                        bool expectCudfScan) {
    SCOPED_TRACE(filter);

    auto scanPlan = PlanBuilder(pool_.get())
                        .startTableScan()
                        .connectorId(kCudfHiveConnectorId)
                        .outputType(rowType)
                        .dataColumns(rowType)
                        .assignments(assignments)
                        .remainingFilter(filter)
                        .endTableScan()
                        .planNode();

    // A file no earlier scan read; see wasCudfScanUsed().
    const auto filePath = TempFilePath::create();
    writeToFile(filePath->getPath(), data);

    std::shared_ptr<Task> task;
    auto actual = AssertQueryBuilder(scanPlan)
                      .config(CudfConfig::kCudfEnabled, true)
                      .splits(makeCudfHiveConnectorSplits({filePath}))
                      .copyResults(pool(), task);

    assertEqualResults({expected}, {actual});
    EXPECT_EQ(wasCudfScanUsed(task, scanPlan->id()), expectCudfScan);
  };

  // A supported, non-extractable remaining filter keeps the scan on GPU.
  testFilter(
      "c0 % 2 = 0",
      makeNamedRowVector("c0", makeFlatVector<int64_t>({2, 4})),
      true);

  // An unsupported nested function falls back to the CPU Hive data source.
  testFilter(
      "to_big_endian_64(c0) = to_big_endian_64(CAST(1 AS BIGINT))",
      makeNamedRowVector("c0", makeFlatVector<int64_t>({1})),
      false);

  // The data source extracts this filter into a subfield filter the cuDF AST
  // cannot express, so the scan falls back to the CPU Hive data source.
  testFilter(
      "c0 not in (1, 3)",
      makeNamedRowVector("c0", makeFlatVector<int64_t>({2, 4})),
      false);

  // The data source extracts this filter into a sample rate. The cuDF data
  // source does not sample, so the scan falls back to the CPU Hive data source.
  {
    SCOPED_TRACE("rand() < 0.5");
    const auto sampledFilePath = TempFilePath::create();
    auto scanPlan =
        makeRemainingFilterPlan(data, "rand() < 0.5", sampledFilePath);
    std::shared_ptr<Task> task;
    auto actual = AssertQueryBuilder(scanPlan)
                      .config(CudfConfig::kCudfEnabled, true)
                      .splits(makeCudfHiveConnectorSplits({sampledFilePath}))
                      .copyResults(pool(), task);
    EXPECT_LE(actual->size(), data->size());
    EXPECT_FALSE(wasCudfScanUsed(task, scanPlan->id()));
  }

  // The connector picks a data source again for each split it preloads, so
  // these scans read several splits.
  const auto testPreloadedSplits = [&](const std::string& filter,
                                       bool expectCudfScan) {
    SCOPED_TRACE(fmt::format("Preloaded splits: {}", filter));
    const auto scanPlan = PlanBuilder(pool_.get())
                              .startTableScan()
                              .connectorId(kCudfHiveConnectorId)
                              .outputType(rowType)
                              .dataColumns(rowType)
                              .assignments(assignments)
                              .remainingFilter(filter)
                              .endTableScan()
                              .planNode();

    // Files no earlier scan read; see wasCudfScanUsed().
    const auto filePaths = makeFilePaths(3);
    for (const auto& filePath : filePaths) {
      writeToFile(filePath->getPath(), data);
    }

    std::shared_ptr<Task> task;
    const auto actual =
        AssertQueryBuilder(scanPlan)
            .config(CudfConfig::kCudfEnabled, true)
            .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "2")
            .splits(makeCudfHiveConnectorSplits(filePaths))
            .copyResults(pool(), task);

    assertEqualResults(
        {makeNamedRowVector("c0", makeFlatVector<int64_t>({2, 4, 2, 4, 2, 4}))},
        {actual});
    EXPECT_EQ(wasCudfScanUsed(task, scanPlan->id()), expectCudfScan);
    const auto planStats = toPlanStats(task->taskStats());
    const auto& customStats = planStats.at(scanPlan->id()).customStats;
    ASSERT_EQ(customStats.count(std::string(TableScan::kPreloadedSplits)), 1);
    EXPECT_GT(customStats.at(std::string(TableScan::kPreloadedSplits)).sum, 0);
  };
  testPreloadedSplits("c0 % 2 = 0", true);
  testPreloadedSplits("c0 not in (1, 3)", false);
}

TEST_F(TableScanTest, subfieldFilterEligibility) {
  auto rowType = ROW("c0", BIGINT());
  auto data = makeNamedRowVector("c0", makeFlatVector<int64_t>({1, 2, 3, 4}));
  const auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto testFilter = [&](const std::string& filter,
                        const RowVectorPtr& expected,
                        bool expectCudfScan) {
    SCOPED_TRACE(filter);

    auto scanPlan = PlanBuilder(pool_.get())
                        .startTableScan()
                        .connectorId(kCudfHiveConnectorId)
                        .outputType(rowType)
                        .dataColumns(rowType)
                        .assignments(assignments)
                        .subfieldFilter(filter)
                        .endTableScan()
                        .planNode();

    // A file no earlier scan read; see wasCudfScanUsed().
    const auto filePath = TempFilePath::create();
    writeToFile(filePath->getPath(), data);

    std::shared_ptr<Task> task;
    auto actual = AssertQueryBuilder(scanPlan)
                      .config(CudfConfig::kCudfEnabled, true)
                      .splits(makeCudfHiveConnectorSplits({filePath}))
                      .copyResults(pool(), task);

    assertEqualResults({expected}, {actual});
    EXPECT_EQ(wasCudfScanUsed(task, scanPlan->id()), expectCudfScan);
  };

  testFilter(
      "c0 in (1, 3)",
      makeNamedRowVector("c0", makeFlatVector<int64_t>({1, 3})),
      true);

  // The cuDF AST has no negated list of values, so the scan falls back to the
  // CPU Hive data source.
  testFilter(
      "c0 not in (1, 3)",
      makeNamedRowVector("c0", makeFlatVector<int64_t>({2, 4})),
      false);

  // The cuDF data source reads top-level columns only, so a filter on a nested
  // field falls back to the CPU Hive data source.
  {
    SCOPED_TRACE("s.a = 1");
    const auto nestedType = ROW({{"s", ROW("a", BIGINT())}, {"c1", BIGINT()}});
    const auto nestedFilePath = TempFilePath::create();
    writeToFile(
        nestedFilePath->getPath(),
        makeRowVector(
            {"s", "c1"},
            {makeNamedRowVector("a", makeFlatVector<int64_t>({1, 2})),
             makeFlatVector<int64_t>({10, 20})}));
    auto scanPlan = PlanBuilder(pool_.get())
                        .startTableScan()
                        .connectorId(kCudfHiveConnectorId)
                        .outputType(ROW("c1", BIGINT()))
                        .dataColumns(nestedType)
                        .subfieldFilter("s.a = 1")
                        .endTableScan()
                        .planNode();
    std::shared_ptr<Task> task;
    auto actual = AssertQueryBuilder(scanPlan)
                      .config(CudfConfig::kCudfEnabled, true)
                      .splits(makeCudfHiveConnectorSplits({nestedFilePath}))
                      .copyResults(pool(), task);
    assertEqualResults(
        {makeNamedRowVector("c1", makeFlatVector<int64_t>({10}))}, {actual});
    EXPECT_FALSE(wasCudfScanUsed(task, scanPlan->id()));
  }
}

TEST_F(TableScanTest, remainingFilterTimestampTimezoneEligibility) {
  auto rowType = ROW({"event_ts"}, {TIMESTAMP()});
  auto data = makeNamedRowVector(
      "event_ts",
      makeFlatVector<Timestamp>(
          {
              Timestamp(1767297600, 0), // 2026-01-01 20:00:00 UTC.
              Timestamp(1767258000, 0), // 2026-01-01 09:00:00 UTC.
          },
          TIMESTAMP()));
  const auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto testFilter = [&](const std::string& filter,
                        bool adjustTimestampToTimezone,
                        const RowVectorPtr& expected,
                        bool expectCudfScan) {
    SCOPED_TRACE(filter);

    auto scanPlan = PlanBuilder(pool_.get())
                        .startTableScan()
                        .connectorId(kCudfHiveConnectorId)
                        .outputType(rowType)
                        .dataColumns(rowType)
                        .assignments(assignments)
                        .remainingFilter(filter)
                        .endTableScan()
                        .planNode();

    // A file no earlier scan read; see wasCudfScanUsed().
    const auto filePath = TempFilePath::create();
    writeToFile(filePath->getPath(), data);

    std::shared_ptr<Task> task;
    auto actual = AssertQueryBuilder(scanPlan)
                      .config(CudfConfig::kCudfEnabled, true)
                      .config(QueryConfig::kSessionTimezone, "Asia/Kolkata")
                      .config(
                          QueryConfig::kAdjustTimestampToTimezone,
                          adjustTimestampToTimezone ? "true" : "false")
                      .splits(makeCudfHiveConnectorSplits({filePath}))
                      .copyResults(pool(), task);

    assertEqualResults({expected}, {actual});
    EXPECT_EQ(wasCudfScanUsed(task, scanPlan->id()), expectCudfScan);
  };

  auto firstRow = makeNamedRowVector(
      "event_ts",
      makeFlatVector<Timestamp>(
          {Timestamp(1767297600, 0)},
          TIMESTAMP())); // 2026-01-01 20:00:00 UTC.
  auto noRows = makeRowVector(rowType, 0);

  // Without session-timezone adjustment, day truncation is safe on GPU.
  testFilter(
      "date_trunc('day', event_ts) = TIMESTAMP '2026-01-01 00:00:00'",
      false,
      data,
      true);

  // Day truncation must use CPU session-timezone semantics when adjustment is
  // enabled.
  testFilter(
      "date_trunc('day', event_ts) = TIMESTAMP '2026-01-01 18:30:00'",
      true,
      noRows,
      false);

  // Minute truncation is timezone-insensitive and remains eligible. With
  // adjustment enabled the literal is Asia/Kolkata time, so 01:30 on January 2
  // there is the first row's 20:00 UTC on January 1.
  testFilter(
      "date_trunc('minute', event_ts) = TIMESTAMP '2026-01-02 01:30:00'",
      true,
      firstRow,
      true);
}

TEST_F(TableScanTest, remainingFilterCastFromVarcharEligibility) {
  auto rowType = ROW({{"c0", VARCHAR()}, {"c1", BIGINT()}});
  auto data = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<std::string>({"1.5", "x", "inf", "0.5"}),
       makeFlatVector<int64_t>({1, 2, 3, 4})});
  const auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto testFilter =
      [&](const std::string& filter,
          const std::unordered_map<std::string, std::string>& sessionConfig,
          const RowVectorPtr& expected,
          bool expectCudfScan) {
        SCOPED_TRACE(filter);

        auto scanPlan = PlanBuilder(pool_.get())
                            .startTableScan()
                            .connectorId(kCudfHiveConnectorId)
                            .outputType(rowType)
                            .dataColumns(rowType)
                            .assignments(assignments)
                            .remainingFilter(filter)
                            .endTableScan()
                            .planNode();

        // A file no earlier scan read; see wasCudfScanUsed().
        const auto filePath = TempFilePath::create();
        writeToFile(filePath->getPath(), data);

        std::shared_ptr<Task> task;
        auto actual = AssertQueryBuilder(scanPlan)
                          .config(CudfConfig::kCudfEnabled, true)
                          .configs(sessionConfig)
                          .splits(makeCudfHiveConnectorSplits({filePath}))
                          .copyResults(pool(), task);

        assertEqualResults({expected}, {actual});
        EXPECT_EQ(wasCudfScanUsed(task, scanPlan->id()), expectCudfScan);
      };

  const std::string tryCastFilter = "try_cast(c0 as double) > 1.0";
  auto firstRow = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<std::string>({"1.5"}), makeFlatVector<int64_t>({1})});

  // The connector does not see the session's cast settings, so a cast from
  // VARCHAR stays on the CPU data source in every session.
  testFilter(tryCastFilter, {}, firstRow, false);

  // The GPU cast would follow Presto's strict rules and drop 'inf', which
  // this session accepts.
  testFilter(
      tryCastFilter,
      {{core::QueryConfig::kCastStringToFloatingPointLenientInfNan, "true"}},
      makeRowVector(
          {"c0", "c1"},
          {makeFlatVector<std::string>({"1.5", "inf"}),
           makeFlatVector<int64_t>({1, 3})}),
      false);

  testFilter(
      tryCastFilter,
      {{core::QueryConfig::kLegacyCast, "true"}},
      firstRow,
      false);

  // A cast of a literal folds away before the check, so the scan stays on the
  // GPU.
  testFilter(
      "c1 % 2 = cast('0' as bigint)",
      {},
      makeRowVector(
          {"c0", "c1"},
          {makeFlatVector<std::string>({"x", "0.5"}),
           makeFlatVector<int64_t>({2, 4})}),
      true);

  // Under legacy_cast the CPU reads a TIMESTAMP literal in UTC rather than the
  // session time zone. The literal is under an OR, so it stays in the remaining
  // filter, which the data source would fold without legacy_cast and miss the
  // row.
  // TODO: Expect the row instead of the error once the data source folds with
  // the session QueryConfig (facebookincubator/velox#19476).
  {
    const auto timestampData = makeRowVector(
        {"c0", "c1"},
        {makeFlatVector<Timestamp>({Timestamp(1'767'314'700, 0)}),
         makeFlatVector<std::string>({"x"})});
    const auto timestampFilePath = TempFilePath::create();
    const auto scanPlan = makeRemainingFilterPlan(
        timestampData,
        "c1 = 'y' or c0 = TIMESTAMP '2026-01-02 00:45:00'",
        timestampFilePath);
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(scanPlan)
            .config(CudfConfig::kCudfEnabled, true)
            .config(core::QueryConfig::kLegacyCast, "true")
            .config(core::QueryConfig::kAdjustTimestampToTimezone, "true")
            .config(core::QueryConfig::kSessionTimezone, "Asia/Kolkata")
            .splits(makeCudfHiveConnectorSplits({timestampFilePath}))
            .copyResults(pool()),
        "Remaining filter with a cast that legacy_cast may change is not supported by the cuDF data source");
  }
}

// The data source folds the remaining filter without legacy_cast, so under
// legacy_cast a scan stays on the GPU only when every cast in its remaining
// filter is on the list legacy_cast leaves as is. Each filter is under an OR
// with another column, so it stays in the remaining filter.
// TODO: Expect every filter on the GPU once the data source folds with the
// session QueryConfig (facebookincubator/velox#19476).
TEST_F(TableScanTest, remainingFilterLegacyCastEligibility) {
  const auto data = makeRowVector(
      {"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
      {
          makeFlatVector<int32_t>({1, 2}),
          makeFlatVector<double>({1.0, 3.0}),
          makeFlatVector<bool>({true, false}),
          makeFlatVector<int32_t>({20'455, 20'456}, DATE()),
          makeFlatVector<int64_t>({150, 250}, DECIMAL(18, 2)),
          makeFlatVector<std::string>({"10000000.0", "x"}),
          makeFlatVector<std::string>({"x", "x"}),
      });
  const auto filePath = TempFilePath::create();
  const auto rowType = asRowType(data->type());

  // Expects a GPU scan to return the rows a CPU scan returns under
  // legacy_cast.
  const auto testParity = [&](const core::PlanNodePtr& scanPlan) {
    SCOPED_TRACE(scanPlan->toString(/*detailed=*/true));
    // Files no earlier scan read; see wasCudfScanUsed().
    const auto cpuFilePath = TempFilePath::create();
    writeToFile(cpuFilePath->getPath(), data);
    const auto gpuFilePath = TempFilePath::create();
    writeToFile(gpuFilePath->getPath(), data);

    RowVectorPtr expected;
    {
      unregisterCudf();
      SCOPE_EXIT {
        registerCudf();
      };
      expected = AssertQueryBuilder(scanPlan)
                     .config(core::QueryConfig::kLegacyCast, "true")
                     .splits(makeCudfHiveConnectorSplits({cpuFilePath}))
                     .copyResults(pool());
    }
    ASSERT_EQ(expected->size(), 1);

    std::shared_ptr<Task> task;
    const auto actual = AssertQueryBuilder(scanPlan)
                            .config(CudfConfig::kCudfEnabled, true)
                            .config(core::QueryConfig::kLegacyCast, "true")
                            .splits(makeCudfHiveConnectorSplits({gpuFilePath}))
                            .copyResults(pool(), task);
    assertEqualResults({expected}, {actual});
    EXPECT_TRUE(wasCudfScanUsed(task, scanPlan->id()));
  };

  // Returns a scan with 'filter' under an OR with another column.
  const auto makePlan = [&](const std::string& filter) {
    return makeRemainingFilterPlan(data, "c6 = 'y' or " + filter, filePath);
  };

  // Casts between BOOLEAN, integer, REAL, DOUBLE, DECIMAL and DATE types.
  testParity(makePlan("cast(c0 as bigint) = 1"));
  testParity(makePlan("cast(c0 as double) = c1"));
  testParity(makePlan("cast(c1 as bigint) = cast(c0 as bigint)"));
  testParity(makePlan("cast(c2 as bigint) = cast(c0 as bigint)"));
  testParity(makePlan("cast(c4 as double) = c1 + 0.5"));
  // Casts from VARCHAR to DECIMAL and DATE.
  testParity(makePlan("c4 = cast('1.50' as decimal(18, 2))"));
  // The parser folds a cast of a VARCHAR literal to DATE, so this filter is
  // built as expressions.
  {
    const auto varcharConstant = [](const std::string& value) {
      return std::make_shared<core::ConstantTypedExpr>(VARCHAR(), value);
    };
    const auto equals = [](core::TypedExprPtr left, core::TypedExprPtr right) {
      return std::make_shared<core::CallTypedExpr>(
          BOOLEAN(),
          std::vector<core::TypedExprPtr>{std::move(left), std::move(right)},
          "eq");
    };
    const auto filter = std::make_shared<core::CallTypedExpr>(
        BOOLEAN(),
        std::vector<core::TypedExprPtr>{
            equals(
                std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c6"),
                varcharConstant("y")),
            equals(
                std::make_shared<core::FieldAccessTypedExpr>(DATE(), "c3"),
                std::make_shared<core::CastTypedExpr>(
                    DATE(),
                    varcharConstant("2026-01-02"),
                    /*isTryCast=*/false)),
        },
        "or");
    testParity(
        PlanBuilder(pool_.get())
            .startTableScan()
            .outputType(rowType)
            .tableHandle(makeTableHandle("parquet_table", rowType, {}, filter))
            .endTableScan()
            .planNode());
  }

  // Under legacy_cast the CPU prints 10000000.0 as '10000000.0' rather than
  // '1.0E7', which the data source would fold to and miss the row.
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(
          makeRemainingFilterPlan(
              data, "c6 = 'y' or c5 = cast(10000000.0 as varchar)", filePath))
          .config(CudfConfig::kCudfEnabled, true)
          .config(core::QueryConfig::kLegacyCast, "true")
          .splits(makeCudfHiveConnectorSplits({filePath}))
          .copyResults(pool()),
      "Remaining filter with a cast that legacy_cast may change is not supported by the cuDF data source");
}

// The connector does not see legacy_cast, under which the CPU prints a
// TIMESTAMP with a 'T' separator, so a remaining filter that casts a TIMESTAMP
// to VARCHAR stays on the CPU data source. A constant beside a column becomes a
// subfield filter through the session's evaluator, so the constant cast here is
// under an OR, where the data source would fold it without legacy_cast.
TEST_F(TableScanTest, remainingFilterCastTimestampToVarcharEligibility) {
  const auto data = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<Timestamp>(
           {Timestamp(0, 0), Timestamp(1'767'314'700, 0)}),
       makeFlatVector<std::string>({"1970-01-01 00:00:00.000", "x"})});

  auto testFilter =
      [&](const std::string& filter,
          const std::unordered_map<std::string, std::string>& sessionConfig,
          const RowVectorPtr& expected) {
        SCOPED_TRACE(filter);

        const auto filePath = TempFilePath::create();
        auto scanPlan = makeRemainingFilterPlan(data, filter, filePath);
        std::shared_ptr<Task> task;
        auto actual = AssertQueryBuilder(scanPlan)
                          .config(CudfConfig::kCudfEnabled, true)
                          .configs(sessionConfig)
                          .splits(makeCudfHiveConnectorSplits({filePath}))
                          .copyResults(pool(), task);

        assertEqualResults({expected}, {actual});
        EXPECT_FALSE(wasCudfScanUsed(task, scanPlan->id()));
      };

  const auto secondRow = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<Timestamp>({Timestamp(1'767'314'700, 0)}),
       makeFlatVector<std::string>({"x"})});

  testFilter("cast(c0 as varchar) like '2026-%'", {}, secondRow);

  const std::string constantCastFilter =
      "c1 = cast(TIMESTAMP '1970-01-01 00:00:00' as varchar) "
      "or c0 = TIMESTAMP '2026-01-02 00:45:00'";
  testFilter(constantCastFilter, {}, data);

  // Under legacy_cast the constant reads '1970-01-01T00:00:00.000', so only the
  // second row passes.
  testFilter(
      constantCastFilter,
      {{core::QueryConfig::kLegacyCast, "true"}},
      secondRow);
}

TEST_F(TableScanTest, dynamicFilterPrunesReaderRowGroups) {
  auto rowType = ROW({"c0", "c1"}, BIGINT());
  std::vector<RowVectorPtr> vectors;
  for (int group = 0; group < 3; ++group) {
    vectors.push_back(makeRowVector(
        {"c0", "c1"},
        {makeFlatVector<int64_t>(
             1'000,
             [group](auto row) { return group * 10'000 + row * 5; },
             nullEvery(13)),
         makeFlatVector<int64_t>(
             1'000, [group](auto row) { return group * 10'000 + row; })}));
  }
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable("t", vectors);
  common::SubfieldFilters staticFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{12505}, int64_t{12505}, false))
          .build();
  auto staticMetrics = readParquetWithStatsFilter(
      filePath->getPath(), rowType, staticFilters, true);
  EXPECT_EQ(staticMetrics.inputRowGroups, 3);
  ASSERT_TRUE(staticMetrics.rowGroupsAfterStats.has_value());
  EXPECT_EQ(*staticMetrics.rowGroupsAfterStats, 1);

  int32_t numFileRowGroups{3};
  auto runTest = [&](const std::vector<int64_t>& values,
                     common::FilterKind expectedKind,
                     int32_t expectedRowGroups = 1,
                     bool includeBuildKey = false) {
    SCOPED_TRACE(fmt::format("values: {}", fmt::join(values, ", ")));
    ASSERT_EQ(common::createBigintValues(values, false)->kind(), expectedKind);
    auto build = makeRowVector({"c0"}, {makeFlatVector<int64_t>(values)});
    createDuckDbTable("u", {build});
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({build})
                         .project({"c0 AS u_key"})
                         .planNode();
    core::PlanNodeId scanId;
    auto scanType = ROW({"c1", "c0"}, BIGINT());
    auto plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .outputType(scanType)
            .dataColumns(rowType)
            .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
            .endTableScan()
            .capturePlanNodeId(scanId)
            .hashJoin(
                {"c0"},
                {"u_key"},
                buildSide,
                "",
                includeBuildKey ? std::vector<std::string>{"c0", "c1", "u_key"}
                                : std::vector<std::string>{"c0", "c1"},
                core::JoinType::kInner)
            .planNode();
    std::atomic<int32_t> selected{-1};
    const bool testValuesWereEnabled = common::testutil::TestValue::enabled();
    common::testutil::TestValue::enable();
    SCOPE_EXIT {
      if (!testValuesWereEnabled) {
        common::testutil::TestValue::disable();
      }
    };
    common::testutil::ScopedTestValue selectedCounter(
        "facebook::velox::cudf_velox::connector::hive::CudfSplitReader::selectedRowGroups",
        std::function<void(void*)>(
            [&](void* value) { selected = *static_cast<size_t*>(value); }));
    auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                    .maxDrivers(1)
                    .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                    .assertResults(
                        fmt::format(
                            "SELECT t.c0, t.c1{} FROM t JOIN u ON t.c0 = u.c0",
                            includeBuildKey ? ", u.c0" : ""));
#ifndef NDEBUG
    EXPECT_EQ(selected, expectedRowGroups);
#endif
    // The scan drops the rows without a build key whether or not the join
    // remains.
    const auto stats = toPlanStats(task->taskStats());
    const auto& scanStats = stats.at(scanId);
    EXPECT_EQ(scanStats.outputRows, expectedRowGroups == 0 ? 0 : values.size());
    EXPECT_EQ(
        scanStats.customStats.at("skippedStrides").sum,
        numFileRowGroups - expectedRowGroups);
    if (expectedRowGroups > 0) {
      EXPECT_EQ(
          scanStats.customStats.at("processedStrides").sum, expectedRowGroups);
      EXPECT_EQ(
          scanStats.customStats.at("dynamicFilterDroppedRows").sum,
          expectedRowGroups * 1'000 - values.size());
    }
  };
  for (bool includeBuildKey : {false, true}) {
    runTest({12505}, common::FilterKind::kBigintRange, 1, includeBuildKey);
    runTest(
        {10005, 10015},
        common::FilterKind::kBigintValuesUsingBitmask,
        1,
        includeBuildKey);
    runTest(
        {10005, 14005},
        common::FilterKind::kBigintValuesUsingHashTable,
        1,
        includeBuildKey);
    // The middle row group lies between the two values: a set prunes by
    // membership, not by its bounds.
    runTest(
        {5, 20'005},
        common::FilterKind::kBigintValuesUsingHashTable,
        2,
        includeBuildKey);
  }
  auto nullKeys = makeRowVector(
      {"c0", "c1"},
      {makeNullableFlatVector<int64_t>({std::nullopt, std::nullopt}),
       makeFlatVector<int64_t>({1, 2})});
  filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), nullKeys);
  createDuckDbTable("t", {nullKeys});
  numFileRowGroups = 1;
  runTest({1}, common::FilterKind::kBigintRange, 0);
}

TEST_F(TableScanTest, integerDynamicFilterFromHashJoin) {
  constexpr vector_size_t kNumBuildRows{
      ::facebook::velox::exec::VectorHasher::kMaxDistinct + 1};
  auto probeType = ROW({{"c0", INTEGER()}, {"c1", BIGINT()}});
  auto probe = makeRowVector(
      {makeFlatVector<int32_t>(1'000, folly::identity),
       makeFlatVector<int64_t>(1'000, [](auto row) {
         // Both per-column filters accept this row, but the pair must
         // still be checked by the join.
         return static_cast<int64_t>(row == 150 ? 151 : row) * 10;
       })});
  auto buildKey = makeFlatVector<int32_t>(
      kNumBuildRows, [](auto row) { return 100 + row; });
  auto secondBuildKey = makeFlatVector<int64_t>(kNumBuildRows, [](auto row) {
    return static_cast<int64_t>(100 + row) * 10;
  });
  for (vector_size_t row = 100; row < kNumBuildRows; ++row) {
    buildKey->setNull(row, true);
    secondBuildKey->setNull(row, true);
  }
  auto build = makeRowVector({buildKey, secondBuildKey});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({build})
                       .project({"c0 AS u_key0", "c1 AS u_key1"})
                       .planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(probeType)
                  .dataColumns(probeType)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(probeType))
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .project({"c1 AS key1", "c0 AS key0"})
                  .hashJoin(
                      {"key0", "key1"},
                      {"u_key0", "u_key1"},
                      buildSide,
                      "",
                      {"key0", "key1"},
                      core::JoinType::kInner)
                  .capturePlanNodeId(joinId)
                  .planNode();

  std::atomic<int32_t> numFiltersBuilt{0};
  const bool testValuesWereEnabled = common::testutil::TestValue::enabled();
  common::testutil::TestValue::enable();
  SCOPE_EXIT {
    if (!testValuesWereEnabled) {
      common::testutil::TestValue::disable();
    }
  };
  common::testutil::ScopedTestValue filterBuildCounter(
      "facebook::velox::cudf_velox::CudfHashJoinProbe::makeDynamicFilter",
      std::function<void(void*)>([&](void*) { ++numFiltersBuilt; }));
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .maxDrivers(4)
                  .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
                  .assertResults(
                      "SELECT t.c0, t.c1 FROM t JOIN u "
                      "ON t.c0 = u.c0 AND t.c1 = u.c1");
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_GE(stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 2);
  EXPECT_GE(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 2);
  // Each filter keeps the 100 rows of its key. Row 150 passes both and fails
  // the join.
  EXPECT_EQ(stats.at(scanId).outputRows, 100);
  EXPECT_EQ(
      stats.at(scanId).dynamicFilterStats.producerNodeIds,
      std::unordered_set<core::PlanNodeId>{joinId});
#ifndef NDEBUG
  EXPECT_EQ(numFiltersBuilt, 2);
#endif

  auto disabledTask =
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .config("hash_probe_dynamic_filter_pushdown_enabled", "false")
          .maxDrivers(4)
          .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
          .assertResults(
              "SELECT t.c0, t.c1 FROM t JOIN u "
              "ON t.c0 = u.c0 AND t.c1 = u.c1");
  const auto disabledStats = toPlanStats(disabledTask->taskStats());
  EXPECT_EQ(disabledStats.at(scanId).outputRows, probe->size());
  EXPECT_EQ(
      disabledStats.at(joinId).customStats.count("dynamicFiltersProduced"), 0);
}

// The filters of a join reach the scan of every driver and the splits that
// were prepared ahead of the driver that reads them.
TEST_F(TableScanTest, dynamicFilterWithPreloadedSplits) {
  constexpr int32_t kNumFiles{8};
  constexpr vector_size_t kRowsPerFile{2'000};
  constexpr vector_size_t kKeysPerFile{20};
  std::vector<RowVectorPtr> vectors;
  std::vector<std::shared_ptr<TempFilePath>> files;
  for (auto i = 0; i < kNumFiles; ++i) {
    vectors.push_back(makeRowVector(
        {"c0", "c1"},
        {makeFlatVector<int64_t>(
             kRowsPerFile, [i](auto row) { return i * kRowsPerFile + row; }),
         makeFlatVector<int64_t>(
             kRowsPerFile, [](auto row) { return row % 7; })}));
    files.push_back(TempFilePath::create());
    writeToFile(files.back()->getPath(), vectors.back());
  }
  createDuckDbTable("t", vectors);
  // Every 100th probe key.
  auto build = makeRowVector(
      {"c0", "b"},
      {makeFlatVector<int64_t>(
           kNumFiles * kKeysPerFile, [](auto row) { return row * 100; }),
       makeFlatVector<int64_t>(kNumFiles * kKeysPerFile, folly::identity)});
  createDuckDbTable("u", {build});
  auto rowType = asRowType(vectors.front()->type());

  for (bool returnBuildColumn : {false, true}) {
    SCOPED_TRACE(fmt::format("returnBuildColumn: {}", returnBuildColumn));
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({build})
                         .project({"c0 AS u_c0", "b"})
                         .planNode();
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    auto plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .outputType(rowType)
            .dataColumns(rowType)
            .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
            .endTableScan()
            .capturePlanNodeId(scanId)
            .hashJoin(
                {"c0"},
                {"u_c0"},
                buildSide,
                "",
                returnBuildColumn ? std::vector<std::string>{"c0", "c1", "b"}
                                  : std::vector<std::string>{"c0", "c1"},
                core::JoinType::kInner)
            .capturePlanNodeId(joinId)
            .planNode();
    auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                    .maxDrivers(4)
                    .config(core::QueryConfig::kMaxSplitPreloadPerDriver, "2")
                    .splits(scanId, makeCudfHiveConnectorSplits(files))
                    .assertResults(
                        fmt::format(
                            "SELECT t.c0, t.c1{} FROM t JOIN u ON t.c0 = u.c0",
                            returnBuildColumn ? ", u.b" : ""));
    const auto stats = toPlanStats(task->taskStats());
    const auto& scanStats = stats.at(scanId);
    EXPECT_EQ(scanStats.outputRows, kNumFiles * kKeysPerFile);
    EXPECT_EQ(scanStats.customStats.at("processedSplits").sum, kNumFiles);
    EXPECT_EQ(
        scanStats.customStats.at("dynamicFilterDroppedRows").sum,
        kNumFiles * (kRowsPerFile - kKeysPerFile));
    EXPECT_EQ(
        stats.at(joinId).customStats.count("replacedWithDynamicFilterRows"),
        returnBuildColumn ? 0 : 1);
  }
}

// One cuDF option switches the join's filters and the scan's acceptance off.
TEST_F(TableScanTest, dynamicFilterDisabled) {
  auto probe =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(1'000, folly::identity)});
  auto build = makeRowVector({"b"}, {makeFlatVector<int64_t>({3, 7})});
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), probe);
  auto rowType = asRowType(probe->type());
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildPlan = PlanBuilder(ids, pool_.get()).values({build}).planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan =
      PlanBuilder(ids, pool_.get())
          .startTableScan()
          .connectorId(kCudfHiveConnectorId)
          .tableHandle(makeTableHandle("t", rowType))
          .outputType(rowType)
          .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
          .endTableScan()
          .capturePlanNodeId(scanId)
          .hashJoin(
              {"c0"}, {"b"}, buildPlan, "", {"c0"}, core::JoinType::kInner)
          .capturePlanNodeId(joinId)
          .planNode();
  auto expected = makeRowVector({"c0"}, {makeFlatVector<int64_t>({3, 7})});

  auto& config = CudfConfig::getInstance();
  const auto enabled = config.dynamicFilterEnabled;
  SCOPE_EXIT {
    config.dynamicFilterEnabled = enabled;
  };
  for (bool dynamicFilterEnabled : {true, false}) {
    SCOPED_TRACE(fmt::format("dynamicFilterEnabled: {}", dynamicFilterEnabled));
    config.dynamicFilterEnabled = dynamicFilterEnabled;
    auto task = AssertQueryBuilder(plan)
                    .maxDrivers(1)
                    .splits(scanId, makeCudfHiveConnectorSplits({file}))
                    .assertResults(expected);
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_EQ(
        stats.at(joinId).customStats.count("dynamicFiltersProduced"),
        dynamicFilterEnabled ? 1 : 0);
    EXPECT_EQ(
        stats.at(scanId).customStats.count("dynamicFiltersAccepted"),
        dynamicFilterEnabled ? 1 : 0);
    EXPECT_EQ(stats.at(scanId).outputRows, dynamicFilterEnabled ? 2 : 1'000);
  }
}

// A build side with many rows and few distinct keys sends the exact set of
// its keys, which replaces a semi join.
TEST_F(TableScanTest, dynamicFilterFromRepeatedBuildKeys) {
  constexpr vector_size_t kNumBuildRows{150'000};
  ASSERT_GT(kNumBuildRows, CudfConfig::getInstance().dynamicFilterMaxValues);
  auto probe =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(1'000, folly::identity)});
  // 100 distinct keys: 0, 10, ..., 990.
  auto build = makeRowVector(
      {"b"}, {makeFlatVector<int64_t>(kNumBuildRows, [](auto row) {
        return (row % 100) * 10;
      })});
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), probe);
  auto rowType = asRowType(probe->type());
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildPlan = PlanBuilder(ids, pool_.get()).values({build}).planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan =
      PlanBuilder(ids, pool_.get())
          .startTableScan()
          .connectorId(kCudfHiveConnectorId)
          .tableHandle(makeTableHandle("t", rowType))
          .outputType(rowType)
          .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
          .endTableScan()
          .capturePlanNodeId(scanId)
          .hashJoin(
              {"c0"},
              {"b"},
              buildPlan,
              "",
              {"c0"},
              core::JoinType::kLeftSemiFilter)
          .capturePlanNodeId(joinId)
          .planNode();
  auto expected = makeRowVector(
      {"c0"},
      {makeFlatVector<int64_t>(100, [](auto row) { return row * 10; })});
  auto task = AssertQueryBuilder(plan)
                  .maxDrivers(1)
                  .splits(scanId, makeCudfHiveConnectorSplits({file}))
                  .assertResults(expected);
  const auto stats = toPlanStats(task->taskStats());
  // The bounds of the keys, 0 to 990, would keep 991 rows.
  EXPECT_EQ(stats.at(scanId).outputRows, 100);
  EXPECT_EQ(
      stats.at(joinId).customStats.at("replacedWithDynamicFilterRows").sum,
      100);
}

TEST_F(TableScanTest, integerDynamicFilterPreservesKeyWidth) {
  auto check = [&]<typename T>() {
    const T lo = std::numeric_limits<T>::min();
    const T hi = std::numeric_limits<T>::max();
    auto probe = makeRowVector(
        {"c0"}, {makeNullableFlatVector<T>({lo, 0, hi, std::nullopt})});
    auto file = TempFilePath::create();
    writeToFile(file->getPath(), probe);
    // Unique extreme keys exercise sparse filter production and native width.
    for (int count :
         {3, 2 * ::facebook::velox::exec::VectorHasher::kMaxDistinct}) {
      std::vector<std::optional<T>> values;
      int64_t matches = 0;
      for (int i = 0; i < count; ++i) {
        values.push_back(
            i % 3 ? std::optional<T>{i % 2 ? lo : hi} : std::nullopt);
        matches += values.back().has_value();
      }
      auto build = makeRowVector({"b"}, {makeNullableFlatVector<T>(values)});
      auto ids = std::make_shared<core::PlanNodeIdGenerator>();
      auto buildPlan = PlanBuilder(ids, pool_.get()).values({build}).planNode();
      core::PlanNodeId scanId;
      auto plan =
          PlanBuilder(ids, pool_.get())
              .startTableScan()
              .connectorId(kCudfHiveConnectorId)
              .tableHandle(makeTableHandle("t", probe->rowType()))
              .outputType(probe->rowType())
              .assignments(
                  HiveConnectorTestBase::allRegularColumns(probe->rowType()))
              .endTableScan()
              .capturePlanNodeId(scanId)
              .hashJoin(
                  {"c0"}, {"b"}, buildPlan, "", {"c0"}, core::JoinType::kInner)
              .singleAggregation({}, {"count(1) AS n"})
              .planNode();
      auto expected =
          makeRowVector({"n"}, {makeFlatVector<int64_t>({matches})});
      auto task = AssertQueryBuilder(plan)
                      .maxDrivers(1)
                      .splits(scanId, makeCudfHiveConnectorSplits({file}))
                      .assertResults(expected);
      const auto stats = toPlanStats(task->taskStats());
      const auto& scanStats = stats.at(scanId);
      if (matches <= ::facebook::velox::exec::VectorHasher::kMaxDistinct) {
        EXPECT_EQ(scanStats.customStats.at("dynamicFiltersAccepted").sum, 1);
        EXPECT_EQ(scanStats.outputRows, 2);
      } else {
        // Many build rows of two distinct keys: the set of the two.
        EXPECT_EQ(scanStats.customStats.at("dynamicFiltersAccepted").sum, 1);
        EXPECT_EQ(scanStats.outputRows, 2);
      }
    }
  };
  check.template operator()<int8_t>();
  check.template operator()<int16_t>();
  check.template operator()<int32_t>();
  check.template operator()<int64_t>();
}

TEST_F(TableScanTest, integerMembershipMask) {
  using facebook::velox::cudf_velox::connector::hive::applyIntegerBitmapToMask;
  using facebook::velox::cudf_velox::connector::hive::applyIntegerRangeToMask;
  auto stream = cudfGlobalStreamPool().get_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  auto copyColumn = [&]<typename T>(const std::vector<T>& values) {
    auto column = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_to_id<T>()},
        values.size(),
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        column->mutable_view().template data<T>(),
        values.data(),
        values.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        stream.get()));
    stream.sync();
    return column;
  };
  auto check = [&]<typename T>() {
    const T lo = std::numeric_limits<T>::min();
    const T hi = std::numeric_limits<T>::max();
    auto input = copyColumn(
        std::vector<T>{
            lo, static_cast<T>(lo + 1), static_cast<T>(lo + 2), -1, 0, hi, 0});
    auto nulls =
        cudf::create_null_mask(7, cudf::mask_state::ALL_VALID, stream, mr);
    const cudf::bitmask_type validity = ~(cudf::bitmask_type{1} << 6);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        nulls.data(),
        &validity,
        sizeof(validity),
        cudaMemcpyHostToDevice,
        stream.get()));
    stream.sync();
    input->set_null_mask(std::move(nulls), 1);
    auto bitmap = copyColumn(std::vector<uint32_t>{5});
    auto values = copyColumn(std::vector<T>{static_cast<T>(lo + 2), hi});
    cudf_velox::connector::hive::CudfIntegerHashSet hash(
        values->view(), lo, stream, mr);

    // Each representation initializes the mask once. Their intersection keeps
    // only lo + 2; NULL passes both sets but fails the range.
    for (int first = 0; first < 3; ++first) {
      std::unique_ptr<cudf::column> mask;
      for (int i = 0; i < 3; ++i) {
        switch ((first + i) % 3) {
          case 0:
            applyIntegerRangeToMask(
                input->view(), lo, 0, false, mask, stream, mr);
            break;
          case 1:
            applyIntegerBitmapToMask(
                input->view(), bitmap->view(), lo, true, mask, stream, mr);
            break;
          case 2:
            hash.apply(input->view(), true, mask, stream, mr);
            break;
        }
      }
      std::array<bool, 7> actual;
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          actual.data(),
          mask->view().data<bool>(),
          sizeof(actual),
          cudaMemcpyDeviceToHost,
          stream.get()));
      stream.sync();
      EXPECT_EQ(
          actual,
          (std::array<bool, 7>{
              false, false, true, false, false, false, false}));
    }
  };
  check.template operator()<int8_t>();
  check.template operator()<int16_t>();
  check.template operator()<int32_t>();
  check.template operator()<int64_t>();
}

TEST_F(TableScanTest, cachedIntegerHashMembership) {
  using cudf_velox::connector::hive::applyIntegerRangeToMask;
  using cudf_velox::connector::hive::CudfIntegerHashSet;
  auto buildStream = cudfGlobalStreamPool().get_stream();
  auto probeStream = cudfGlobalStreamPool().get_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  auto check = [&]<typename T>() {
    const T lo = std::numeric_limits<T>::min();
    const T hi = std::numeric_limits<T>::max();
    auto values = makeNullableFlatVector<T>(
        {hi,
         lo,
         static_cast<T>(lo + 1),
         static_cast<T>(lo + 2),
         0,
         hi,
         std::nullopt,
         lo,
         0});
    // Initialize the complete validity word directly, including padding bits.
    auto copyColumn = [&](const std::vector<T>& host) {
      auto column = cudf::make_fixed_width_column(
          cudf::data_type{cudf::type_to_id<T>()},
          host.size(),
          cudf::mask_state::ALL_VALID,
          buildStream,
          mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          column->mutable_view().template data<T>(),
          host.data(),
          host.size() * sizeof(T),
          cudaMemcpyHostToDevice,
          buildStream.get()));
      buildStream.sync();
      return column;
    };
    auto input = copyColumn(
        {hi,
         lo,
         static_cast<T>(lo + 1),
         static_cast<T>(lo + 2),
         0,
         hi,
         0,
         lo,
         0});
    const cudf::bitmask_type validity = ~(cudf::bitmask_type{1} << 6);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        input->mutable_view().null_mask(),
        &validity,
        sizeof(validity),
        cudaMemcpyHostToDevice,
        buildStream.get()));
    input->set_null_count(1);
    auto keys =
        copyColumn({lo, static_cast<T>(lo + 2), hi, static_cast<T>(lo + 2)});
    CudfIntegerHashSet set(keys->view(), lo + 1, buildStream, mr);
    buildStream.sync();
    auto slices = cudf::slice(input->view(), {1, 8}, probeStream);
    const auto probe = slices.front();
    const std::unordered_set<T> expectedKeys{lo, static_cast<T>(lo + 2), hi};
    for (bool nullAllowed : {false, true}) {
      for (bool precedingRange : {false, true}) {
        std::unique_ptr<cudf::column> mask;
        if (precedingRange) {
          applyIntegerRangeToMask(probe, lo, 0, false, mask, probeStream, mr);
        }
        set.apply(probe, nullAllowed, mask, probeStream, mr);
        // Reuse both the table and an existing mask; false rows stay false.
        set.apply(probe, nullAllowed, mask, probeStream, mr);
        std::vector<uint8_t> actual(probe.size());
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            actual.data(),
            mask->view().data<bool>(),
            actual.size(),
            cudaMemcpyDeviceToHost,
            probeStream.get()));
        probeStream.sync();
        for (int i = 0; i < probe.size(); ++i) {
          const bool isNull = values->isNullAt(i + 1);
          const bool expected = isNull
              ? nullAllowed && !precedingRange
              : expectedKeys.contains(values->valueAt(i + 1)) &&
                  (!precedingRange || values->valueAt(i + 1) <= 0);
          EXPECT_EQ(actual[i] != 0, expected) << "row " << i;
        }
      }
    }
    auto emptyKeys = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_to_id<T>()},
        0,
        cudf::mask_state::UNALLOCATED,
        buildStream,
        mr);
    CudfIntegerHashSet empty(emptyKeys->view(), lo + 1, buildStream, mr);
    buildStream.sync();
    for (bool nullAllowed : {false, true}) {
      std::unique_ptr<cudf::column> mask;
      empty.apply(probe, nullAllowed, mask, probeStream, mr);
      std::vector<uint8_t> actual(probe.size());
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          actual.data(),
          mask->view().data<bool>(),
          actual.size(),
          cudaMemcpyDeviceToHost,
          probeStream.get()));
      probeStream.sync();
      for (int i = 0; i < probe.size(); ++i) {
        EXPECT_EQ(actual[i] != 0, nullAllowed && values->isNullAt(i + 1));
      }
    }
  };
  check.template operator()<int8_t>();
  check.template operator()<int16_t>();
  check.template operator()<int32_t>();
  check.template operator()<int64_t>();
}

TEST_F(TableScanTest, exactDynamicFilterUsesInputType) {
  using cudf_velox::connector::hive::makeAdvisoryFilter;
  auto check = [&]<typename T>() {
    const T lo = std::numeric_limits<T>::min();
    const T hi = std::numeric_limits<T>::max();
    auto input = makeRowVector(
        {"c0", "c1", "c2", "c3"},
        {makeNullableFlatVector<T>({lo, 0, 1, hi, std::nullopt, lo, hi}),
         makeFlatVector<T>(
             {static_cast<T>(lo + 10),
              static_cast<T>(lo + 20),
              static_cast<T>(lo + 30),
              static_cast<T>(lo + 40),
              static_cast<T>(lo + 50),
              lo,
              static_cast<T>(lo + 51)}),
         makeNullableFlatVector<int64_t>(
             {std::nullopt, 1, 2, 3, std::nullopt, 1, 3}),
         makeFlatVector<int64_t>(7, folly::identity)});
    auto file = TempFilePath::create();
    writeToFile(file->getPath(), input);
    auto properties = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{});
    ::facebook::velox::connector::ConnectorQueryCtx ctx(
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
    auto connector = ::facebook::velox::connector::ConnectorRegistry::tryGet(
        kCudfHiveConnectorId);
    for (bool lateRange : {false, true}) {
      for (bool nullAllowed : {false, true}) {
        auto source = connector->createDataSource(
            asRowType(input->type()),
            makeTableHandle("t", asRowType(input->type())),
            HiveConnectorTestBase::allRegularColumns(asRowType(input->type())),
            &ctx);
        const std::vector<int64_t> values{
            std::numeric_limits<int64_t>::min(),
            lo,
            hi,
            std::numeric_limits<int64_t>::max()};
        auto sorted = values;
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        source->addDynamicFilter(
            0, common::createBigintValues(sorted, nullAllowed));
        source->addDynamicFilter(
            1,
            common::createBigintValues(
                {lo + 10, lo + 20, lo + 40, lo + 50}, false));
        if (lateRange) {
          std::vector<std::unique_ptr<common::BigintRange>> ranges;
          for (int64_t key : {1, 3}) {
            ranges.push_back(
                std::make_unique<common::BigintRange>(key, key, false));
          }
          source->addDynamicFilter(
              2,
              std::make_shared<common::BigintMultiRange>(
                  std::move(ranges), nullAllowed));
        } else {
          source->addDynamicFilter(
              2, common::createBigintValues({1, 3}, nullAllowed));
        }
        for (int split = 0; split < 2; ++split) {
          if (lateRange) {
            source->addDynamicFilter(
                0, std::make_shared<common::BigintRange>(lo, hi, true));
          }
          source->addDynamicFilter(
              3,
              makeAdvisoryFilter(
                  std::make_unique<common::BigintRange>(0, 2, false)));
          source->addSplit(makeCudfHiveConnectorSplit(file->getPath()));
          // An advisory filter that arrives after the split was prepared
          // filters rows like the others, also next to a filter that is
          // evaluated on the host: it drops the first row, which the other
          // filters keep when they allow nulls.
          source->addDynamicFilter(
              3,
              makeAdvisoryFilter(
                  std::make_unique<common::BigintRange>(1, 6, false)));
          if (lateRange) {
            // A prepared reader still owns the old, wider Range snapshot.
            source->addDynamicFilter(
                0, std::make_shared<common::BigintRange>(lo, 0, nullAllowed));
          }
          std::vector<RowVectorPtr> actual;
          ContinueFuture future;
          while (auto next = source->next(100, future)) {
            if (!*next) {
              break;
            }
            auto gpu = std::dynamic_pointer_cast<CudfVector>(*next);
            actual.push_back(
                with_arrow::toVeloxColumn(
                    gpu->getTableView(),
                    pool_.get(),
                    asRowType(input->type()),
                    gpu->stream(),
                    cudf::get_current_device_resource_ref()));
            gpu->stream().sync();
          }
          // Without the advisory filter the first row of the file, whose
          // c2 is null, would be returned too when nulls are allowed.
          const T second = lateRange ? 0 : hi;
          const T secondC1 = lo + (lateRange ? 20 : 40);
          const int64_t secondC2 = lateRange ? 1 : 3;
          const int64_t secondC3 = lateRange ? 1 : 3;
          auto expected = makeRowVector(
              {"c0", "c1", "c2", "c3"},
              {nullAllowed ? makeNullableFlatVector<T>({second, std::nullopt})
                           : makeNullableFlatVector<T>({second}),
               nullAllowed
                   ? makeFlatVector<T>({secondC1, static_cast<T>(lo + 50)})
                   : makeFlatVector<T>({secondC1}),
               nullAllowed
                   ? makeNullableFlatVector<int64_t>({secondC2, std::nullopt})
                   : makeNullableFlatVector<int64_t>({secondC2}),
               nullAllowed ? makeFlatVector<int64_t>({secondC3, 4})
                           : makeFlatVector<int64_t>({secondC3})});
          ::facebook::velox::exec::test::assertEqualResults({expected}, actual);
        }
      }
    }
  };
  check.template operator()<int8_t>();
  check.template operator()<int16_t>();
  check.template operator()<int32_t>();
  check.template operator()<int64_t>();
}

// A scan applies a dynamic filter of any kind to every row, as a join that
// was replaced by its filter and a filter that comes from another task need
// it to: the rows it returns are those a CPU reader's test of the filter
// keeps.
TEST_F(TableScanTest, dynamicFilterKinds) {
  constexpr vector_size_t kNumRows{1'000};
  auto rowType = ROW(
      {{"c0", BIGINT()},
       {"c1", DATE()},
       {"c2", SMALLINT()},
       {"c3", VARCHAR()},
       {"c4", DOUBLE()},
       {"c5", DECIMAL(10, 2)}});
  auto input = makeRowVector(
      rowType->names(),
      {makeFlatVector<int64_t>(
           kNumRows, [](auto row) { return row * 3; }, nullEvery(7)),
       makeFlatVector<int32_t>(
           kNumRows,
           [](auto row) { return 10'000 + row; },
           nullEvery(11),
           DATE()),
       makeFlatVector<int16_t>(
           kNumRows, [](auto row) { return row % 50 - 25; }, nullEvery(13)),
       makeFlatVector<std::string>(
           kNumRows,
           [](auto row) { return fmt::format("s{}", row % 40); },
           nullEvery(5)),
       makeFlatVector<double>(
           kNumRows, [](auto row) { return row * 0.5; }, nullEvery(17)),
       makeFlatVector<int64_t>(
           kNumRows,
           [](auto row) { return row * 7; },
           nullEvery(19),
           DECIMAL(10, 2))});
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), input);

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  ::facebook::velox::connector::ConnectorQueryCtx ctx(
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
  auto connector = ::facebook::velox::connector::ConnectorRegistry::tryGet(
      kCudfHiveConnectorId);

  // Reads the file with 'filter' on 'channel' and expects exactly the rows
  // the filter passes.
  const auto check = [&](column_index_t channel,
                         const common::FilterPtr& filter) {
    SCOPED_TRACE(fmt::format("channel {}: {}", channel, filter->toString()));
    auto source = connector->createDataSource(
        rowType,
        makeTableHandle("t", rowType),
        HiveConnectorTestBase::allRegularColumns(rowType),
        &ctx);
    source->addDynamicFilter(channel, filter);
    source->addSplit(makeCudfHiveConnectorSplit(file->getPath()));
    const auto actual = readSplit(*source, rowType, pool_.get());

    DecodedVector decoded(*input->childAt(channel));
    std::vector<vector_size_t> passing;
    for (vector_size_t row = 0; row < kNumRows; ++row) {
      bool passes;
      if (decoded.isNullAt(row)) {
        passes = filter->testNull();
      } else {
        switch (rowType->childAt(channel)->kind()) {
          case TypeKind::BIGINT:
            passes = filter->testInt64(decoded.valueAt<int64_t>(row));
            break;
          case TypeKind::INTEGER:
            passes = filter->testInt64(decoded.valueAt<int32_t>(row));
            break;
          case TypeKind::SMALLINT:
            passes = filter->testInt64(decoded.valueAt<int16_t>(row));
            break;
          case TypeKind::VARCHAR:
            passes = filter->testStringView(decoded.valueAt<StringView>(row));
            break;
          case TypeKind::DOUBLE:
            passes = filter->testDouble(decoded.valueAt<double>(row));
            break;
          default:
            VELOX_UNREACHABLE();
        }
      }
      if (passes) {
        passing.push_back(row);
      }
    }
    auto indices = makeIndices(passing);
    std::vector<VectorPtr> expectedColumns;
    for (const auto& column : input->children()) {
      expectedColumns.push_back(wrapInDictionary(indices, column));
    }
    auto expected = makeRowVector(rowType->names(), expectedColumns);
    ::facebook::velox::exec::test::assertEqualResults({expected}, actual);
    EXPECT_EQ(numRows(actual), passing.size());
  };

  const auto multiRange = [](bool nullAllowed) {
    std::vector<std::unique_ptr<common::BigintRange>> ranges;
    ranges.push_back(std::make_unique<common::BigintRange>(30, 60, false));
    ranges.push_back(
        std::make_unique<common::BigintRange>(2'700, 2'760, false));
    return std::make_shared<common::BigintMultiRange>(
        std::move(ranges), nullAllowed);
  };
  const auto bloomFilter = [](const std::vector<int64_t>& values) {
    auto filter =
        std::make_shared<common::BigintValuesUsingBloomFilter>(1'000, false);
    for (const auto value : values) {
      filter->insert(value);
    }
    return filter;
  };

  // BIGINT column.
  check(0, std::make_shared<common::BigintRange>(300, 600, false));
  check(0, std::make_shared<common::BigintRange>(300, 600, true));
  check(0, common::createBigintValues({3, 9, 12, 2'997}, false));
  check(0, common::createBigintValues({3, 1'000'000'002}, true));
  check(0, multiRange(false));
  check(0, multiRange(true));
  check(0, bloomFilter({3, 1'500, 2'997}));
  check(0, std::make_shared<common::NegatedBigintRange>(30, 2'900, false));
  check(0, std::make_shared<common::IsNull>());
  check(0, std::make_shared<common::IsNotNull>());
  check(0, std::make_shared<common::AlwaysFalse>());
  // DATE column: the filter holds days.
  check(1, std::make_shared<common::BigintRange>(10'100, 10'110, false));
  check(1, common::createBigintValues({10'001, 10'500, 10'998}, false));
  check(1, common::createBigintValues({10'001, 10'002, 10'004}, true));
  check(1, bloomFilter({10'001, 10'777}));
  check(1, std::make_shared<common::AlwaysFalse>());
  // SMALLINT column: a value outside of the type matches no row.
  check(2, std::make_shared<common::BigintRange>(-5, 100'000, false));
  check(2, common::createBigintValues({-25, 0, 24, 100'000}, false));
  check(2, bloomFilter({-25, 24}));
  // VARCHAR column.
  check(
      3,
      std::make_shared<common::BytesValues>(
          std::vector<std::string>{"s1", "s17", "absent"}, false));
  check(
      3,
      std::make_shared<common::BytesValues>(
          std::vector<std::string>{"s39"}, true));
  check(
      3,
      std::make_shared<common::BytesRange>(
          "s10", false, false, "s19", false, true, false));
  check(3, std::make_shared<common::IsNull>());
  check(3, std::make_shared<common::IsNotNull>());
  check(3, std::make_shared<common::AlwaysFalse>());
  // DOUBLE column.
  check(
      4,
      std::make_shared<common::DoubleRange>(
          10.0, false, true, 20.5, false, false, false));
  check(4, std::make_shared<common::IsNull>());
  // DECIMAL column read as 64-bit integers: the filter holds unscaled
  // values.
  check(5, std::make_shared<common::BigintRange>(700, 1'400, false));
  check(5, common::createBigintValues({7, 70, 6'993}, true));
  check(5, bloomFilter({14, 21}));
}

// A filter is read with the test of the column's type: a filter of a kind
// without that test, and a file that stores the column in another type than
// the table's, fail the query instead of returning wrong rows.
TEST_F(TableScanTest, dynamicFilterOnMismatchedType) {
  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  ::facebook::velox::connector::ConnectorQueryCtx ctx(
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
  auto connector = ::facebook::velox::connector::ConnectorRegistry::tryGet(
      kCudfHiveConnectorId);
  const auto makeSource = [&](const RowTypePtr& tableType) {
    return connector->createDataSource(
        tableType,
        makeTableHandle("t", tableType),
        HiveConnectorTestBase::allRegularColumns(tableType),
        &ctx);
  };

  {
    // The file stores REAL, the table says DOUBLE.
    auto file = TempFilePath::create();
    writeToFile(
        file->getPath(),
        makeRowVector({"c0"}, {makeFlatVector<float>({1.5, 12.5, 30.0})}));
    auto source = makeSource(ROW("c0", DOUBLE()));
    source->addDynamicFilter(
        0,
        std::make_shared<common::DoubleRange>(
            10.0, false, false, 20.5, false, false, false));
    source->addSplit(makeCudfHiveConnectorSplit(file->getPath()));
    VELOX_ASSERT_THROW(
        countSplitRows(*source),
        "cuDF Hive scan cannot apply a dynamic filter to the type the file stores column c0 in");
  }
  {
    // A filter on strings sent to an integer column.
    auto file = TempFilePath::create();
    writeToFile(
        file->getPath(),
        makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3})}));
    auto source = makeSource(ROW("c0", BIGINT()));
    source->addDynamicFilter(
        0,
        std::make_shared<common::BytesValues>(
            std::vector<std::string>{"1"}, false));
    source->addSplit(makeCudfHiveConnectorSplit(file->getPath()));
    VELOX_ASSERT_THROW(countSplitRows(*source), "testInt64() is not supported");
  }
  {
    // The files of one scan store a BIGINT column as INT64 and as INT32. The
    // values are too far apart for a bitmap, so the scan probes a hash set,
    // whose keys have the width of the file's column.
    auto wideFile = TempFilePath::create();
    writeToFile(
        wideFile->getPath(),
        makeRowVector(
            {"c0"}, {makeFlatVector<int64_t>({5, 7, 1'000'000'000})}));
    auto narrowFile = TempFilePath::create();
    writeToFile(
        narrowFile->getPath(),
        makeRowVector({"c0"}, {makeFlatVector<int32_t>({5, 6, 9})}));
    auto filter = common::createBigintValues({5, 9, 1'000'000'000}, false);
    ASSERT_EQ(filter->kind(), common::FilterKind::kBigintValuesUsingHashTable);
    auto source = makeSource(ROW("c0", BIGINT()));
    source->addDynamicFilter(0, std::move(filter));
    for (auto i = 0; i < 2; ++i) {
      source->addSplit(makeCudfHiveConnectorSplit(wideFile->getPath()));
      EXPECT_EQ(countSplitRows(*source), 2);
      source->addSplit(makeCudfHiveConnectorSplit(narrowFile->getPath()));
      EXPECT_EQ(countSplitRows(*source), 2);
    }
  }
}

// A filter that arrives between two batches of a split applies from the next
// batch on.
TEST_F(TableScanTest, dynamicFilterBetweenBatches) {
  constexpr vector_size_t kNumRows{50'000};
  auto rowType = ROW({"c0", "c1"}, BIGINT());
  std::vector<RowVectorPtr> vectors;
  for (auto i = 0; i < 5; ++i) {
    vectors.push_back(makeRowVector(
        {"c0", "c1"},
        {makeFlatVector<int64_t>(
             kNumRows / 5, [i](auto row) { return i * (kNumRows / 5) + row; }),
         makeFlatVector<int64_t>(kNumRows / 5, folly::identity)}));
  }
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), vectors);

  // Small read limits make the reader return the split in several batches.
  auto properties = std::make_shared<
      config::ConfigBase>(std::unordered_map<std::string, std::string>{
      {cudf_velox::connector::hive::CudfHiveConfig::kMaxChunkReadLimitSession,
       "8192"},
      {cudf_velox::connector::hive::CudfHiveConfig::kMaxPassReadLimitSession,
       "32768"}});
  ::facebook::velox::connector::ConnectorQueryCtx ctx(
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
  auto connector = ::facebook::velox::connector::ConnectorRegistry::tryGet(
      kCudfHiveConnectorId);
  auto source = connector->createDataSource(
      rowType,
      makeTableHandle("t", rowType),
      HiveConnectorTestBase::allRegularColumns(rowType),
      &ctx);
  source->addSplit(makeCudfHiveConnectorSplit(file->getPath()));

  ContinueFuture future;
  auto first = source->next(1'000, future);
  ASSERT_TRUE(first.has_value());
  ASSERT_NE(*first, nullptr);
  const auto numFirstRows = (*first)->size();
  ASSERT_GT(numFirstRows, 0);
  ASSERT_LT(numFirstRows, kNumRows);

  // Every 1000th value of c0, which the file holds in ascending order.
  std::vector<int64_t> values;
  for (int64_t value = 0; value < kNumRows; value += 1'000) {
    values.push_back(value);
  }
  source->addDynamicFilter(0, common::createBigintValues(values, false));
  const auto numValuesLeft =
      std::count_if(values.begin(), values.end(), [&](auto value) {
        return value >= numFirstRows;
      });
  EXPECT_EQ(countSplitRows(*source), numValuesLeft);
}

// A reader that filters before it decodes the payload asks the data source to
// narrow its own row mask.
TEST_F(TableScanTest, intersectDynamicFilters) {
  using cudf_velox::connector::hive::CudfHiveDataSource;
  auto rowType = ROW({"c0", "c1"}, BIGINT());
  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  ::facebook::velox::connector::ConnectorQueryCtx ctx(
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
  auto connector = ::facebook::velox::connector::ConnectorRegistry::tryGet(
      kCudfHiveConnectorId);
  auto source = connector->createDataSource(
      rowType,
      makeTableHandle("t", rowType),
      HiveConnectorTestBase::allRegularColumns(rowType),
      &ctx);
  auto* cudfSource = checkedPointerCast<CudfHiveDataSource>(source.get());

  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto data = makeRowVector(
      {"c0", "c1", "mask"},
      {makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6}),
       makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
       makeFlatVector<bool>({false, true, true, true, true, true})});
  auto table = with_arrow::toCudfTable(data, pool_.get(), stream, mr);
  const auto readColumn = [&](size_t index) {
    return table->get_column(index).view();
  };
  const auto toHost = [&](const cudf::column_view& mask) {
    auto rows = with_arrow::toVeloxColumn(
        cudf::table_view{{mask}},
        pool_.get(),
        ROW("mask", BOOLEAN()),
        stream,
        mr);
    stream.sync();
    std::vector<bool> values;
    auto* flat = rows->childAt(0)->asFlatVector<bool>();
    for (vector_size_t row = 0; row < flat->size(); ++row) {
      values.push_back(flat->valueAt(row));
    }
    return values;
  };

  // Without a dynamic filter the mask stays as it is.
  std::unique_ptr<cudf::column> mask;
  EXPECT_FALSE(
      cudfSource->intersectDynamicFilters(readColumn, 6, mask, stream));
  EXPECT_EQ(mask, nullptr);
  EXPECT_TRUE(cudfSource->dynamicFilterColumns().empty());
  const auto version = cudfSource->dynamicFilterVersion();

  // A range on c0, tested on the device, and a Bloom filter on c1, tested on
  // the host.
  source->addDynamicFilter(
      0, std::make_shared<common::BigintRange>(2, 5, false));
  auto bloomFilter =
      std::make_shared<common::BigintValuesUsingBloomFilter>(1'000, false);
  bloomFilter->insert(10);
  bloomFilter->insert(30);
  bloomFilter->insert(50);
  source->addDynamicFilter(1, bloomFilter);
  EXPECT_GT(cudfSource->dynamicFilterVersion(), version);
  EXPECT_THAT(cudfSource->dynamicFilterColumns(), testing::ElementsAre(0, 1));
  const auto bounds = cudfSource->dynamicFilterBounds();
  ASSERT_EQ(bounds.size(), 1);
  EXPECT_EQ(bounds[0].readColumn, 0);
  EXPECT_EQ(bounds[0].lower, 2);
  EXPECT_EQ(bounds[0].upper, 5);
  EXPECT_FALSE(bounds[0].nullAllowed);

  // The mask is created: rows 3 and 5 pass both filters.
  EXPECT_TRUE(cudfSource->intersectDynamicFilters(readColumn, 6, mask, stream));
  EXPECT_THAT(
      toHost(mask->view()),
      testing::ElementsAre(false, false, true, false, true, false));

  // A mask of the caller is narrowed: its first row stays rejected.
  mask = std::make_unique<cudf::column>(table->get_column(2), stream, mr);
  source->addDynamicFilter(
      0, std::make_shared<common::BigintRange>(1, 3, false));
  EXPECT_TRUE(cudfSource->intersectDynamicFilters(readColumn, 6, mask, stream));
  EXPECT_THAT(
      toHost(mask->view()),
      testing::ElementsAre(false, false, true, false, false, false));

  cudfSource->recordDynamicFilterRows(6, 1);
  EXPECT_EQ(source->getRuntimeStats().at("dynamicFilterDroppedRows").sum, 5);
}

// A scan drops rows for an advisory filter only while that removes enough of
// them, and never when the row filter is switched off.
TEST_F(TableScanTest, advisoryFilterRows) {
  using cudf_velox::connector::hive::makeAdvisoryFilter;
  constexpr vector_size_t kNumRows{150'000};
  auto rowType = ROW("c0", BIGINT());
  auto input = makeRowVector(
      {"c0"}, {makeFlatVector<int64_t>(kNumRows, folly::identity)});
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), input);

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  ::facebook::velox::connector::ConnectorQueryCtx ctx(
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
  auto connector = ::facebook::velox::connector::ConnectorRegistry::tryGet(
      kCudfHiveConnectorId);

  // Returns the rows of two splits of the file read after 'filter' arrived.
  const auto readTwoSplits = [&](const common::FilterPtr& filter) {
    auto source = connector->createDataSource(
        rowType,
        makeTableHandle("t", rowType),
        HiveConnectorTestBase::allRegularColumns(rowType),
        &ctx);
    source->addDynamicFilter(0, filter);
    std::vector<vector_size_t> rows;
    for (auto i = 0; i < 2; ++i) {
      source->addSplit(makeCudfHiveConnectorSplit(file->getPath()));
      rows.push_back(numRows(readSplit(*source, rowType, pool_.get())));
    }
    return rows;
  };
  const auto range = [&](int64_t upper, bool advisory) -> common::FilterPtr {
    auto filter = std::make_unique<common::BigintRange>(0, upper, false);
    if (advisory) {
      return makeAdvisoryFilter(std::move(filter));
    }
    return filter;
  };

  // A selective advisory filter keeps filtering.
  EXPECT_THAT(
      readTwoSplits(range(14'999, true)), testing::ElementsAre(15'000, 15'000));
  // One that keeps 90% of the first split stops: the join that sent it drops
  // the rest.
  EXPECT_THAT(
      readTwoSplits(range(134'999, true)),
      testing::ElementsAre(135'000, kNumRows));
  // The same filter without the mark is applied to every row.
  EXPECT_THAT(
      readTwoSplits(range(134'999, false)),
      testing::ElementsAre(135'000, 135'000));

  auto& config = CudfConfig::getInstance();
  const auto rowFilterEnabled = config.dynamicFilterRowFilterEnabled;
  config.dynamicFilterRowFilterEnabled = false;
  SCOPE_EXIT {
    config.dynamicFilterRowFilterEnabled = rowFilterEnabled;
  };
  EXPECT_THAT(
      readTwoSplits(range(14'999, true)),
      testing::ElementsAre(kNumRows, kNumRows));
  EXPECT_THAT(
      readTwoSplits(range(14'999, false)),
      testing::ElementsAre(15'000, 15'000));
}

// A dynamic filter on a partition key decides for whole splits: those to
// come and the one being read.
TEST_F(TableScanTest, dynamicFilterOnPartitionKey) {
  auto fileType = ROW("c0", BIGINT());
  auto input = makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3})});
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), input);
  auto outputType = ROW({{"pk", BIGINT()}, {"c0", BIGINT()}});
  const facebook::velox::connector::ColumnHandleMap assignments = {
      {"pk", HiveConnectorTestBase::partitionKey("pk", BIGINT())},
      {"c0", HiveConnectorTestBase::regularColumn("c0", BIGINT())}};

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  ::facebook::velox::connector::ConnectorQueryCtx ctx(
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
  auto connector = ::facebook::velox::connector::ConnectorRegistry::tryGet(
      kCudfHiveConnectorId);
  auto source = connector->createDataSource(
      outputType, makeTableHandle("t", fileType), assignments, &ctx);
  const auto readPartition = [&](const std::string& value) {
    source->addSplit(makePartitionedSplit(file->getPath(), {{"pk", value}}));
    return numRows(readSplit(*source, outputType, pool_.get()));
  };

  EXPECT_EQ(readPartition("7"), 3);
  source->addDynamicFilter(
      0, std::make_shared<common::BigintRange>(8, 9, false));
  EXPECT_EQ(readPartition("7"), 0);
  EXPECT_EQ(readPartition("8"), 3);

  // The filter arrives while a split it rejects is being read.
  source->addSplit(makePartitionedSplit(file->getPath(), {{"pk", "9"}}));
  source->addDynamicFilter(
      0, std::make_shared<common::BigintRange>(8, 8, false));
  EXPECT_EQ(numRows(readSplit(*source, outputType, pool_.get())), 0);
  EXPECT_EQ(readPartition("8"), 3);
  EXPECT_EQ(source->getRuntimeStats().at("skippedSplits").sum, 2);
}

TEST_F(TableScanTest, rejectsDynamicFiltersItCannotApply) {
  auto physicalProbe = makeRowVector(
      {"c0"}, {makeFlatVector<int64_t>({100, 200}, DECIMAL(18, 2))});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), physicalProbe);

  auto checkJoin = [&](const TypePtr& scanType,
                       bool useGpuJoin,
                       const char* expectedError) {
    auto rowType = ROW("c0", scanType);
    auto build =
        makeRowVector({"c0"}, {makeFlatVector<int64_t>({100}, scanType)});
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({build})
                         .project({"c0 AS u_key"})
                         .planNode();
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    auto plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .outputType(rowType)
            .dataColumns(rowType)
            .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
            .endTableScan()
            .capturePlanNodeId(scanId)
            .hashJoin(
                {"c0"},
                {"u_key"},
                buildSide,
                "",
                {"c0"},
                useGpuJoin ? core::JoinType::kInner
                           : core::JoinType::kCountingLeftSemiFilter,
                /*nullAware=*/false,
                /*nullAsValue=*/false)
            .capturePlanNodeId(joinId)
            .planNode();

    AssertQueryBuilder query(plan);
    query.config(CudfConfig::kCudfEnabled, "true")
        .splits(scanId, makeCudfHiveConnectorSplits({filePath}));
    if (expectedError) {
      VELOX_ASSERT_THROW(query.copyResults(pool_.get()), expectedError);
    } else {
      auto expected =
          makeRowVector({"c0"}, {makeFlatVector<int64_t>({100}, scanType)});
      auto task = query.assertResults(expected);
      const auto stats = toPlanStats(task->taskStats());
      EXPECT_TRUE(stats.at(joinId).operatorStats.count(
          useGpuJoin ? "CudfHashJoinProbe" : "HashProbe"));
      EXPECT_TRUE(stats.at(scanId).dynamicFilterStats.producerNodeIds.empty());
    }
  };

  {
    auto& config = CudfConfig::getInstance();
    const auto previousFallback = config.allowCpuFallback;
    unregisterCudf();
    config.allowCpuFallback = true;
    registerCudf();
    SCOPE_EXIT {
      unregisterCudf();
      config.allowCpuFallback = previousFallback;
      registerCudf();
    };
    checkJoin(DECIMAL(18, 2), false, nullptr);
  }
  checkJoin(DECIMAL(18, 2), true, nullptr);
  checkJoin(
      BIGINT(),
      true,
      "cuDF Hive scan cannot apply a dynamic filter to the type the file stores column c0 in");
}

TEST_F(TableScanTest, exactDynamicFilterCanReplaceInnerJoin) {
  for (int64_t keyScale : {1, 1'000'000}) {
    auto probe = makeRowVector(
        {"p_key", "p_value"},
        {makeNullableFlatVector<int64_t>(
             {keyScale,
              2 * keyScale,
              2 * keyScale,
              3 * keyScale,
              5 * keyScale,
              std::nullopt}),
         makeFlatVector<int64_t>({6, 5, 4, 3, 2, 1})});
    auto file = TempFilePath::create();
    writeToFile(file->getPath(), probe);
    createDuckDbTable("t", {probe});
    for (bool duplicate : {false, true}) {
      auto build = makeRowVector(
          {"b_key", "b_value"},
          {makeNullableFlatVector<int64_t>(
               {2 * keyScale, (duplicate ? 2 : 5) * keyScale, std::nullopt}),
           makeFlatVector<int64_t>({5, 1, 0})});
      createDuckDbTable("u", {build});
      for (int mode = 0; mode < 6; ++mode) {
        SCOPED_TRACE(
            fmt::format(
                "keyScale: {}, duplicate: {}, mode: {}",
                keyScale,
                duplicate,
                mode));
        auto ids = std::make_shared<core::PlanNodeIdGenerator>();
        auto buildBuilder = PlanBuilder(ids, pool_.get()).values({build});
        if (mode == 5) {
          buildBuilder.filter("b_key IS NOT NULL");
        }
        auto buildPlan = buildBuilder.planNode();
        core::PlanNodeId scanId;
        core::PlanNodeId joinId;
        const std::vector<std::vector<std::string>> layouts{
            {"p_value", "p_key"},
            {"p_key", "b_value"},
            {},
            {"p_key"},
            {"p_key", "p_key"},
            {"p_key"}};
        const std::vector<std::string> selections{
            "t.p_value, t.p_key",
            "t.p_key, u.b_value",
            "count(*)",
            "t.p_key",
            "t.p_key, t.p_key",
            "t.p_key"};
        auto builder = PlanBuilder(ids, pool_.get());
        builder.startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .tableHandle(makeTableHandle("t", probe->rowType()))
            .outputType(probe->rowType())
            .assignments(
                HiveConnectorTestBase::allRegularColumns(probe->rowType()))
            .endTableScan()
            .capturePlanNodeId(scanId);
        if (mode == 5) {
          builder.filter("p_key IS NOT NULL");
        }
        builder
            .hashJoin(
                {"p_key"},
                {"b_key"},
                buildPlan,
                mode == 3 ? "p_value < b_value" : "",
                layouts[mode],
                core::JoinType::kInner,
                false,
                mode == 5)
            .capturePlanNodeId(joinId);
        if (mode == 2) {
          builder.partialAggregation({}, {"count(1) AS n"})
              .localPartition({})
              .finalAggregation();
        }
        auto task =
            AssertQueryBuilder(builder.planNode(), duckDbQueryRunner_)
                .maxDrivers(4)
                .splits(scanId, makeCudfHiveConnectorSplits({file}))
                .assertResults(
                    fmt::format(
                        "SELECT {} FROM t JOIN u ON t.p_key=u.b_key {}",
                        selections[mode],
                        mode == 3 ? "WHERE t.p_value < u.b_value" : ""));
        const auto stats = toPlanStats(task->taskStats());
        EXPECT_EQ(
            stats.at(joinId).customStats.count(
                "replacedWithDynamicFilterRows") != 0,
            !duplicate && (mode == 0 || mode == 2));
        if (mode == 5) {
          EXPECT_EQ(
              stats.at(joinId).customStats.count("dynamicFiltersProduced"), 0);
        } else {
          EXPECT_GT(
              stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 0);
        }
      }
    }
  }
}

TEST_F(TableScanTest, advisoryFilterKeepsJoin) {
  auto probe = makeRowVector(
      {"p_key"}, {makeNullableFlatVector<int64_t>({1, 2, 3, std::nullopt})});
  auto build = makeRowVector(
      {"b_key", "b_value"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  auto probeFile = TempFilePath::create();
  auto buildFile = TempFilePath::create();
  writeToFile(probeFile->getPath(), probe);
  writeToFile(buildFile->getPath(), build);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    auto ids = std::make_shared<core::PlanNodeIdGenerator>();
    common::SubfieldFilters filters;
    if (mode == 1) {
      filters.emplace(
          common::Subfield("b_key"),
          std::make_shared<common::BigintRange>(1, 1, false));
    }
    core::PlanNodeId buildScanId;
    auto buildBuilder = PlanBuilder(ids, pool_.get());
    buildBuilder.startTableScan()
        .connectorId(kCudfHiveConnectorId)
        .tableHandle(makeTableHandle("u", build->rowType(), std::move(filters)))
        .outputType(build->rowType())
        .assignments(HiveConnectorTestBase::allRegularColumns(build->rowType()))
        .endTableScan()
        .capturePlanNodeId(buildScanId)
        .project({"b_key", "b_value"})
        .localPartition({});
    if (mode == 3) {
      buildBuilder.filter("b_key = 1");
    }
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    const bool probeOnly = mode == 2;
    auto plan =
        PlanBuilder(ids, pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .tableHandle(makeTableHandle("t", probe->rowType()))
            .outputType(probe->rowType())
            .assignments(
                HiveConnectorTestBase::allRegularColumns(probe->rowType()))
            .endTableScan()
            .capturePlanNodeId(scanId)
            .hashJoin(
                {"p_key"},
                {"b_key"},
                buildBuilder.planNode(),
                "",
                probeOnly ? std::vector<std::string>{"p_key"}
                          : std::vector<std::string>{"p_key", "b_value"},
                core::JoinType::kInner)
            .capturePlanNodeId(joinId)
            .planNode();
    auto task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .splits(scanId, makeCudfHiveConnectorSplits({probeFile}))
            .splits(buildScanId, makeCudfHiveConnectorSplits({buildFile}))
            .assertResults(
                fmt::format(
                    "SELECT t.p_key{} FROM t JOIN u ON t.p_key=u.b_key{}",
                    probeOnly ? "" : ", u.b_value",
                    mode == 1 || mode == 3 ? " WHERE u.b_key=1" : ""));
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_GT(stats.at(joinId).customStats.at("dynamicFiltersProduced").sum, 0);
    // The scan drops the rows without a build key also when the join remains
    // to return build columns.
    EXPECT_EQ(stats.at(scanId).outputRows, mode == 1 || mode == 3 ? 1 : 2);
    EXPECT_EQ(
        stats.at(joinId).customStats.count("replacedWithDynamicFilterRows"),
        probeOnly ? 1 : 0);
  }
}

// The scan's own filter and a join's filter on the same column both apply.
TEST_F(TableScanTest, dynamicFilterWithStaticFilterOnSameColumn) {
  auto rowType = ROW("c0", BIGINT());
  auto probe =
      makeRowVector({"c0"}, {makeFlatVector<int64_t>(1'000, folly::identity)});
  auto build = makeRowVector({"b"}, {makeFlatVector<int64_t>({100, 600, 700})});
  auto file = TempFilePath::create();
  writeToFile(file->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});

  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("c0"),
      std::make_shared<common::BigintRange>(
          500, std::numeric_limits<int64_t>::max(), false));
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildPlan = PlanBuilder(ids, pool_.get()).values({build}).planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan =
      PlanBuilder(ids, pool_.get())
          .startTableScan()
          .connectorId(kCudfHiveConnectorId)
          .tableHandle(makeTableHandle("t", rowType, std::move(filters)))
          .outputType(rowType)
          .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
          .endTableScan()
          .capturePlanNodeId(scanId)
          .hashJoin(
              {"c0"}, {"b"}, buildPlan, "", {"c0"}, core::JoinType::kInner)
          .capturePlanNodeId(joinId)
          .planNode();
  auto task = AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .maxDrivers(1)
                  .splits(scanId, makeCudfHiveConnectorSplits({file}))
                  .assertResults(
                      "SELECT t.c0 FROM t JOIN u ON t.c0 = u.b "
                      "WHERE t.c0 >= 500");
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_EQ(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 1);
  // 600 and 700: the build key 100 fails the scan's own filter.
  EXPECT_EQ(stats.at(scanId).outputRows, 2);
  EXPECT_EQ(
      stats.at(joinId).customStats.at("replacedWithDynamicFilterRows").sum, 2);
}

TEST_F(TableScanTest, advisoryFilterMergesWithExactFilter) {
  auto rowType = ROW("c0", BIGINT());
  auto probe = makeRowVector(
      {"c0"}, {makeNullableFlatVector<int64_t>({1, 2, 3, std::nullopt})});
  auto firstBuild = makeRowVector(
      {"c0", "payload"},
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {firstBuild});
  for (int64_t key : {2, 3}) {
    SCOPED_TRACE(key);
    auto secondBuild = makeRowVector({"c0"}, {makeFlatVector<int64_t>({key})});
    createDuckDbTable("v", {secondBuild});
    auto ids = std::make_shared<core::PlanNodeIdGenerator>();
    auto firstBuildSide = PlanBuilder(ids, pool_.get())
                              .values({firstBuild})
                              .project({"c0 AS u_key", "payload"})
                              .planNode();
    auto secondBuildSide = PlanBuilder(ids, pool_.get())
                               .values({secondBuild})
                               .project({"c0 AS v_key"})
                               .planNode();
    core::PlanNodeId scanId;
    core::PlanNodeId joinId;
    auto plan =
        PlanBuilder(ids, pool_.get())
            .startTableScan()
            .connectorId(kCudfHiveConnectorId)
            .outputType(rowType)
            .dataColumns(rowType)
            .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
            .endTableScan()
            .capturePlanNodeId(scanId)
            .hashJoin(
                {"c0"},
                {"u_key"},
                firstBuildSide,
                "",
                {"c0", "payload"},
                core::JoinType::kInner)
            .hashJoin(
                {"c0"},
                {"v_key"},
                secondBuildSide,
                "",
                {"c0", "payload"},
                core::JoinType::kInner)
            .capturePlanNodeId(joinId)
            .planNode();
    auto task =
        AssertQueryBuilder(plan, duckDbQueryRunner_)
            .splits(scanId, makeCudfHiveConnectorSplits({filePath}))
            .assertResults(
                "SELECT t.c0, u.payload FROM t JOIN u ON t.c0=u.c0 JOIN v ON t.c0=v.c0");
    const auto stats = toPlanStats(task->taskStats());
    EXPECT_GE(stats.at(scanId).customStats.at("dynamicFiltersAccepted").sum, 2);
    // The second join can be replaced only after the merged predicate is fully
    // enforced. Keeping the pruning-only marker would leak keys 1 and 3.
    EXPECT_EQ(stats.at(scanId).outputRows, key == 2 ? 1 : 0);
    if (key == 2) {
      EXPECT_GT(
          stats.at(joinId).customStats.at("replacedWithDynamicFilterRows").sum,
          0);
    }
  }
}

TEST_F(TableScanTest, dynamicFilterStopsAtCpuScanConversion) {
  const std::string cpuConnectorId{"cpu-hive-dynamic-filter"};
  facebook::velox::connector::hive::HiveConnectorFactory factory;
  auto cpuConnector = factory.newConnector(
      cpuConnectorId,
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{}),
      ioExecutor_.get());
  ConnectorRegistry::global().insert(cpuConnectorId, cpuConnector);
  SCOPE_EXIT {
    ConnectorRegistry::global().erase(cpuConnectorId);
  };

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), BIGINT()});
  auto probe = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({101, 102, 103, 104}),
       makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto build = makeRowVector({"c0"}, {makeFlatVector<int64_t>({2, 4})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), probe);
  createDuckDbTable("t", {probe});
  createDuckDbTable("u", {build});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .values({build})
                       .project({"c0 AS u_key"})
                       .planNode();
  core::PlanNodeId scanId;
  core::PlanNodeId joinId;
  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .startTableScan()
          .connectorId(cpuConnectorId)
          .outputType(rowType)
          .dataColumns(rowType)
          .assignments(HiveConnectorTestBase::allRegularColumns(rowType))
          .endTableScan()
          .capturePlanNodeId(scanId)
          .hashJoin(
              {"c1"},
              {"u_key"},
              buildSide,
              "",
              {"c0", "c1"},
              core::JoinType::kInner)
          .capturePlanNodeId(joinId)
          .planNode();
  std::vector<std::shared_ptr<ConnectorSplit>> splits{
      facebook::velox::connector::hive::HiveConnectorSplitBuilder(
          filePath->getPath())
          .connectorId(cpuConnectorId)
          .fileFormat(dwio::common::FileFormat::PARQUET)
          .build()};

  auto task =
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .maxDrivers(2)
          .splits(scanId, splits)
          .assertResults("SELECT t.c0, t.c1 FROM t JOIN u ON t.c1 = u.c0");
  const auto stats = toPlanStats(task->taskStats());
  EXPECT_TRUE(stats.at(joinId).operatorStats.count("CudfFromVelox"));
  EXPECT_EQ(stats.at(joinId).customStats.count("dynamicFiltersProduced"), 0);
  EXPECT_EQ(stats.at(scanId).customStats.count("dynamicFiltersAccepted"), 0);
  EXPECT_EQ(stats.at(scanId).outputRows, probe->size());
}

TEST_F(TableScanTest, decimalSubfieldFilter) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(5, 2), BIGINT()});
  auto vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, -500, -700, -500}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3, 4})});

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  common::SubfieldFilters subfieldFilters =
      common::test::SubfieldFiltersBuilder()
          .add(
              "c0",
              std::make_unique<common::BigintRange>(
                  int64_t{-500}, int64_t{-500}, /*nullAllowed*/ false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, std::move(subfieldFilters), nullptr);
  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT c0, c1 FROM tmp WHERE c0 = CAST('-5.00' AS DECIMAL(5, 2))");
}

TEST_F(TableScanTest, mixedCaseDecimalFilterUsesSplitPhysicalType) {
  auto fileVector = makeRowVector(
      {"Price", "Value"},
      {makeFlatVector<int64_t>({100, -500, -700}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3})});
  auto filePath = TempFilePath::create();
  writeCompactDecimalParquet(filePath, fileVector);
  createDuckDbTable({fileVector});

  auto rowType = ROW({{"price", DECIMAL(18, 2)}, {"value", BIGINT()}});
  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "price",
                         std::make_unique<common::BigintRange>(
                             -500, -500, /*nullAllowed=*/false))
                     .build();
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(makeTableHandle(
                      "parquet_table", rowType, std::move(filters), nullptr))
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .connectorSessionProperty(
          kCudfHiveConnectorId,
          facebook::velox::connector::hive::HiveConfig::
              kFileColumnNamesReadAsLowerCaseSession,
          "true")
      .splits(makeCudfHiveConnectorSplits({filePath}))
      .assertResults(
          "SELECT Price, Value FROM tmp "
          "WHERE Price = CAST('-5.00' AS DECIMAL(5, 2))");
}

TEST_F(TableScanTest, decimalFilterUsesSplitPhysicalType) {
  // Both file schemas fit the table's wider logical precision.
  auto rowType = ROW({{"c0", DECIMAL(18, 2)}, {"c1", BIGINT()}});
  auto decimal32Vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, -500, -700}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3})});
  auto decimal64Vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({200, -500, -800}, DECIMAL(18, 2)),
       makeFlatVector<int64_t>({4, 5, 6})});

  auto decimal32Path = TempFilePath::create();
  auto decimal64Path = TempFilePath::create();
  writeCompactDecimalParquet(decimal32Path, decimal32Vector);
  writeToFile(decimal64Path->getPath(), {decimal64Vector});
  createDuckDbTable({makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>(
           {100, -500, -700, 200, -500, -800}, DECIMAL(18, 2)),
       makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6})})});

  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "c0",
                         std::make_unique<common::BigintRange>(
                             int64_t{-500},
                             int64_t{-500},
                             /*nullAllowed*/ false))
                     .build();
  auto tableHandle =
      makeTableHandle("parquet_table", rowType, std::move(filters), nullptr);
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  const auto expected =
      "SELECT c0, c1 FROM tmp "
      "WHERE c0 = CAST('-5.00' AS DECIMAL(5, 2))";
  for (const auto& paths :
       {std::vector{decimal32Path, decimal64Path},
        std::vector{decimal64Path, decimal32Path}}) {
    // A missing file is skipped before its footer is read for the filter.
    auto splits = makeCudfHiveConnectorSplits(paths);
    splits.insert(
        splits.begin() + 1,
        makeCudfHiveConnectorSplit(paths.front()->getPath() + ".missing"));
    AssertQueryBuilder(plan, duckDbQueryRunner_)
        .maxDrivers(1)
        .connectorSessionProperty(
            kCudfHiveConnectorId,
            hive::HiveConfig::kIgnoreMissingFilesSession,
            "true")
        .splits(splits)
        .assertResults(expected);
  }
}

TEST_F(TableScanTest, decimalFilterUsesSplitScale) {
  auto rowType = ROW({{"c0", DECIMAL(12, 4)}, {"c1", BIGINT()}});
  auto fileVector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({30000000, 123}, DECIMAL(9, 2)),
       makeFlatVector<int64_t>({1, 2})});
  auto filePath = TempFilePath::create();
  writeCompactDecimalParquet(filePath, fileVector);
  createDuckDbTable({makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({3000000000, 12300}, DECIMAL(12, 4)),
       makeFlatVector<int64_t>({1, 2})})});

  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "c0",
                         std::make_unique<common::BigintRange>(
                             int64_t{3000000000},
                             int64_t{3000000000},
                             /*nullAllowed*/ false))
                     .build();
  auto tableHandle =
      makeTableHandle("parquet_table", rowType, std::move(filters), nullptr);
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT * FROM tmp WHERE c0 = CAST('300000.0000' AS DECIMAL(12, 4))");
}

TEST_F(TableScanTest, decimalFilterRejectsFileScaleAboveTableScale) {
  auto fileVector =
      makeRowVector({makeFlatVector<int64_t>({12399}, DECIMAL(7, 4))});
  auto filePath = TempFilePath::create();
  writeCompactDecimalParquet(filePath, fileVector);

  auto rowType = ROW({{"c0", DECIMAL(5, 2)}});
  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "c0",
                         std::make_unique<common::BigintRange>(
                             123, 123, /*nullAllowed=*/false))
                     .build();
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(makeTableHandle(
                      "parquet_table", rowType, std::move(filters), nullptr))
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  VELOX_ASSERT_THROW(
      assertQuery(plan, {filePath}, "SELECT CAST(1.23 AS DECIMAL(5, 2))"),
      "Parquet decimal scale of 'c0' exceeds the table scale");
}

TEST_F(TableScanTest, decimalFilterUsesRawIntegerStorage) {
  auto rowType = ROW({{"c0", DECIMAL(10, 0)}, {"c1", BIGINT()}});
  auto fileVector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int32_t>({123, 456}), makeFlatVector<int64_t>({1, 2})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), {fileVector});
  createDuckDbTable({makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({123, 456}, DECIMAL(10, 0)),
       makeFlatVector<int64_t>({1, 2})})});

  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "c0",
                         std::make_unique<common::BigintRange>(
                             123, 123, /*nullAllowed=*/false))
                     .build();
  auto tableHandle =
      makeTableHandle("parquet_table", rowType, std::move(filters), nullptr);
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT * FROM tmp WHERE c0 = CAST(123 AS DECIMAL(10, 0))");
}

TEST_F(TableScanTest, decimalFilterUsesAnnotatedIntegerStorage) {
  auto fileVector =
      makeRowVector({makeFlatVector<int8_t>({-100, 0, 100}, TINYINT())});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), {fileVector});
  createDuckDbTable({fileVector});

  auto rowType = ROW({{"c0", DECIMAL(10, 0)}});
  auto filters = common::test::SubfieldFiltersBuilder()
                     .add(
                         "c0",
                         std::make_unique<common::BigintRange>(
                             -100, -100, /*nullAllowed=*/false))
                     .build();
  auto plan = PlanBuilder()
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(makeTableHandle(
                      "parquet_table", rowType, std::move(filters), nullptr))
                  .assignments(
                      facebook::velox::exec::test::HiveConnectorTestBase::
                          allRegularColumns(rowType))
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT CAST(c0 AS DECIMAL(10, 0)) FROM tmp WHERE c0 = -100");
}

TEST_F(TableScanTest, nullableDecimalFilterUsesSplitPhysicalType) {
  // The file retains precision 5; the table schema widens it to precision 12.
  auto rowType = ROW({{"c0", DECIMAL(12, 2)}, {"c1", BIGINT()}});
  auto outputType = ROW("c1", BIGINT());
  auto mixedPath = TempFilePath::create();
  auto allNullPath = TempFilePath::create();
  auto nonNullPath = TempFilePath::create();
  writeCompactDecimalParquet(
      mixedPath,
      makeRowVector(
          {"c0", "c1"},
          {makeNullableFlatVector<int64_t>(
               {std::nullopt, -500, 200}, DECIMAL(5, 2)),
           makeFlatVector<int64_t>({1, 2, 3})}));
  writeCompactDecimalParquet(
      allNullPath,
      makeRowVector(
          {"c0", "c1"},
          {makeNullableFlatVector<int64_t>(
               {std::nullopt, std::nullopt, std::nullopt}, DECIMAL(5, 2)),
           makeFlatVector<int64_t>({4, 5, 6})}));
  writeCompactDecimalParquet(
      nonNullPath,
      makeRowVector(
          {"c0", "c1"},
          {makeFlatVector<int64_t>({100, -500, 200}, DECIMAL(5, 2)),
           makeFlatVector<int64_t>({7, 8, 9})}));

  for (const bool nullAllowed : {false, true}) {
    auto check = [&](std::unique_ptr<common::Filter> filter,
                     bool matchesNonNull) {
      SCOPED_TRACE(filter->toString());
      common::SubfieldFilters filters;
      filters.emplace(common::Subfield("c0"), std::move(filter));
      auto tableHandle = makeTableHandle(
          "parquet_table", rowType, std::move(filters), nullptr);
      auto plan = PlanBuilder()
                      .startTableScan()
                      .outputType(outputType)
                      .tableHandle(tableHandle)
                      .assignments(
                          facebook::velox::exec::test::HiveConnectorTestBase::
                              allRegularColumns(outputType))
                      .endTableScan()
                      .planNode();
      std::vector<int64_t> expectedIds;
      if (nullAllowed) {
        expectedIds = {1, 4, 5, 6};
      }
      if (matchesNonNull) {
        expectedIds.insert(expectedIds.end(), {2, 8});
      }
      auto expected =
          makeNamedRowVector("c1", makeFlatVector<int64_t>(expectedIds));
      AssertQueryBuilder(plan)
          .maxDrivers(1)
          .splits(makeCudfHiveConnectorSplits(
              {mixedPath, allNullPath, nonNullPath}))
          .assertResults(expected);
    };
    check(std::make_unique<common::BigintRange>(-500, -500, nullAllowed), true);
    // These bounds fit the logical decimal, but no DECIMAL32 value matches.
    check(
        std::make_unique<common::BigintRange>(
            3'000'000'000, 4'000'000'000, nullAllowed),
        false);
    check(common::createBigintValues({-500, 3'000'000'000}, nullAllowed), true);
    check(
        common::createBigintValues({3'000'000'000, 3'000'000'002}, nullAllowed),
        false);
  }
}

TEST_F(TableScanTest, decimalRemainingFilter) {
  auto rowType = ROW({"c0", "c1"}, {DECIMAL(5, 2), BIGINT()});
  auto vector = makeRowVector(
      {"c0", "c1"},
      {makeFlatVector<int64_t>({100, -500, -700, -500}, DECIMAL(5, 2)),
       makeFlatVector<int64_t>({1, 2, 3, 4})});

  std::vector<RowVectorPtr> vectors = {vector};
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors);
  createDuckDbTable(vectors);

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);

  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(assignments)
                  .remainingFilter("c0 = CAST('-5.00' AS DECIMAL(5, 2))")
                  .endTableScan()
                  .planNode();

  assertQuery(
      plan,
      {filePath},
      "SELECT c0, c1 FROM tmp WHERE c0 = CAST('-5.00' AS DECIMAL(5, 2))");
}

// The text of a DATE depends on no session setting, so a remaining filter that
// casts one to VARCHAR runs on the data source.
TEST_F(TableScanTest, remainingFilterCastDateToVarchar) {
  const auto day = DATE()->toDays("2026-01-02");
  const auto filePath = TempFilePath::create();
  auto plan = makeRemainingFilterPlan(
      makeNamedRowVector("c0", makeFlatVector<int32_t>({day, 0}, DATE())),
      "cast(c0 as varchar) like '2026-%'",
      filePath);
  AssertQueryBuilder(plan)
      .splits(makeCudfHiveConnectorSplits({filePath}))
      .assertResults(
          makeNamedRowVector("c0", makeFlatVector<int32_t>({day}, DATE())));
}

TEST_F(TableScanTest, unsupportedScanColumnProjectedOut) {
  auto rowType = ROW({"k", "m"}, {BIGINT(), MAP(BIGINT(), BIGINT())});
  auto filePath = writeUnsupportedMapInput();

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(assignments)
                  .endTableScan()
                  .project({"k"})
                  .planNode();

  std::shared_ptr<Task> task;
  auto result = AssertQueryBuilder(plan)
                    .config(cudf_velox::CudfConfig::kCudfAllowCpuFallback, true)
                    .split(makeCudfHiveConnectorSplit(filePath->getPath()))
                    .copyResults(pool(), task);
  auto expected = makeNamedRowVector("k", makeFlatVector<int64_t>({1, 2}));
  facebook::velox::test::assertEqualVectors(expected, result);

  const auto operatorStats = toOperatorStats(task->taskStats());
  EXPECT_EQ(operatorStats.count("CudfFilterProject"), 0);
  EXPECT_EQ(operatorStats.count("FilterProject"), 1);
}

TEST_F(TableScanTest, unsupportedScanOutputFallsBack) {
  auto rowType = ROW({"k", "m"}, {BIGINT(), MAP(BIGINT(), BIGINT())});
  auto filePath = writeUnsupportedMapInput();

  auto assignments =
      facebook::velox::exec::test::HiveConnectorTestBase::allRegularColumns(
          rowType);
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .connectorId(kCudfHiveConnectorId)
                  .outputType(rowType)
                  .dataColumns(rowType)
                  .assignments(assignments)
                  .endTableScan()
                  .planNode();

  auto expected = makeRowVector(
      {"k", "m"},
      {makeFlatVector<int64_t>({1, 2}),
       makeMapVector<int64_t, int64_t>({{{10, 100}}, {{20, 200}}})});
  AssertQueryBuilder(plan)
      .config(cudf_velox::CudfConfig::kCudfAllowCpuFallback, true)
      .split(makeCudfHiveConnectorSplit(filePath->getPath()))
      .assertResults(expected);
}

// Velox's parquet writer stores DECIMAL(7, 2) as INT32 when
// enableStoreDecimalAsInteger is true, and cuDF's reader maps INT32 decimals
// to DECIMAL32. Velox short decimals are always DECIMAL64, so the scan output
// must be cast from DECIMAL32 to DECIMAL64.
TEST_F(TableScanTest, lowPrecisionDecimalScan) {
  auto rowType = ROW({"d"}, {DECIMAL(7, 2)});
  auto vector = makeNamedRowVector(
      "d",
      makeNullableFlatVector<int64_t>(
          {12345, std::nullopt, -2500, 300}, DECIMAL(7, 2)));
  assertDecimalScanRoundTrip(vector, rowType);
}

TEST_F(TableScanTest, lowPrecisionDecimalScanNoCast) {
  auto rowType = ROW({"d"}, {DECIMAL(12, 4)});
  auto vector = makeNamedRowVector(
      "d",
      makeNullableFlatVector<int64_t>(
          {123456789, std::nullopt, -999999}, DECIMAL(12, 4)));
  assertDecimalScanRoundTrip(vector, rowType);
}

TEST_F(TableScanTest, nestedDecimalScan) {
  auto rowType = ROW({"s"}, {ROW({"x", "d"}, {INTEGER(), DECIMAL(7, 2)})});
  auto vector = makeNamedRowVector(
      "s",
      makeRowVector(
          {"x", "d"},
          {makeNullableFlatVector<int32_t>({1, 2, std::nullopt}),
           makeNullableFlatVector<int64_t>(
               {100, std::nullopt, -200}, DECIMAL(7, 2))}));
  assertDecimalScanRoundTrip(vector, rowType);
}

TEST_F(TableScanTest, arrayDecimalScan) {
  auto rowType = ROW({"a"}, {ARRAY(DECIMAL(7, 2))});
  auto elements = makeNullableFlatVector<int64_t>(
      {100, 200, std::nullopt, 300}, DECIMAL(7, 2));
  auto vector = makeNamedRowVector("a", makeArrayVector({0, 2}, elements));
  assertDecimalScanRoundTrip(vector, rowType);
}

// Exercises the recursive cast through struct -> struct -> list nesting so a
// decimal buried several levels deep is normalized. Schema:
// struct<int, decimal, struct<int, list<decimal>>>.
TEST_F(TableScanTest, multiLevelNestedDecimalScan) {
  auto rowType =
      ROW({"s"},
          {ROW(
              {"x", "d", "nested"},
              {INTEGER(),
               DECIMAL(7, 2),
               ROW({"y", "a"}, {INTEGER(), ARRAY(DECIMAL(7, 2))})})});
  auto listElements = makeNullableFlatVector<int64_t>(
      {100, 200, std::nullopt, 300, 400}, DECIMAL(7, 2));
  auto vector = makeNamedRowVector(
      "s",
      makeRowVector(
          {"x", "d", "nested"},
          {makeNullableFlatVector<int32_t>({1, 2, std::nullopt}),
           makeNullableFlatVector<int64_t>(
               {100, std::nullopt, -200}, DECIMAL(7, 2)),
           makeRowVector(
               {"y", "a"},
               {makeNullableFlatVector<int32_t>({10, std::nullopt, 30}),
                makeArrayVector({0, 2, 4}, listElements)})}));
  assertDecimalScanRoundTrip(vector, rowType);
}
