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
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/benchmarks/CudfBenchmarkDiscard.h"
#include "velox/experimental/cudf/benchmarks/CudfRawReadBenchmark.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchIoConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/file/FileSystems.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <experimental/cudf/connectors/hive/CudfHiveConnector.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>
#include <utility>

DECLARE_int64(max_coalesced_bytes);
DECLARE_string(max_coalesced_distance_bytes);
DECLARE_int32(parquet_prefetch_rowgroups);
DECLARE_string(data_format);
DECLARE_string(data_path);
DECLARE_int32(num_drivers);
DECLARE_int32(num_repeats);
DECLARE_int32(run_query_verbose);
DECLARE_int32(io_meter_column_pct);

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

DEFINE_uint64(
    cudf_chunk_read_limit,
    0,
    "Output table chunk read limit for cudf::parquet_chunked_reader.");

DEFINE_uint64(
    cudf_pass_read_limit,
    0,
    "Pass read limit for cudf::parquet_chunked_reader.");

DEFINE_int32(
    cudf_gpu_batch_size_rows,
    100000,
    "Preferred output batch size in rows for cudf operators.");

DEFINE_uint64(
    cudf_local_exchange_buffer_size,
    1UL << 30,
    "Maximum buffered bytes per local exchange before applying backpressure.");

DEFINE_bool(velox_cudf_table_scan, true, "Enable cuDF table scan");

DEFINE_string(
    cudf_properties,
    "",
    "Path to a properties file for CudfConfig. Each line should be key=value "
    "(e.g. cudf.memory_resource=async). See CudfConfig for available keys.");

// Phase 2: I/O benchmark mode flags.

DEFINE_string(
    cudf_io_mode,
    "",
    "I/O benchmark mode. One of: '' (disabled), 'decode_discard', "
    "'raw_parquet_ranges', 'raw_file'.");

DEFINE_string(
    cudf_io_table,
    "lineitem",
    "TPC-H table to scan in I/O benchmark mode.");

DEFINE_int64(
    cudf_io_read_size_bytes,
    128 << 20,
    "Target read chunk size in bytes. Only 'raw_file' reads the value, but "
    "every enabled I/O mode requires it to be positive.");

DEFINE_string(
    connector_properties,
    "",
    "Path to a connector properties file for S3 credentials/endpoint. "
    "Each line should be key=value. Keys must not contain '='.");

namespace {

// Asked of the file-system registry to find out whether the process already
// serves S3. Never opened.
constexpr std::string_view kS3RegistrationProbe{"s3://probe/object"};

// Value of an environment variable, or nullopt when it is unset.
std::optional<std::string_view> environmentValue(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return std::nullopt;
  }
  return std::string_view(value);
}

// Physical Parquet column names the scan reads, in scan-output order. The
// TPC-H builder names outputs after the canonical TPC-H schema, which the
// Parquet files need not use, so a raw run has to select the names the scan
// assignments carry rather than the output names.
std::vector<std::string> physicalColumnNames(const core::PlanNode& plan) {
  const auto* scanNode = dynamic_cast<const core::TableScanNode*>(&plan);
  VELOX_USER_CHECK_NOT_NULL(
      scanNode,
      "Exact Parquet-range mode needs a table scan at the root of the scan "
      "plan but found '{}'.",
      plan.name());

  const auto& assignments = scanNode->assignments();
  std::vector<std::string> columnNames;
  columnNames.reserve(scanNode->outputType()->size());
  for (const auto& outputName : scanNode->outputType()->names()) {
    const auto assignment = assignments.find(outputName);
    VELOX_USER_CHECK(
        assignment != assignments.end(),
        "Scan output column '{}' has no assignment, so its physical Parquet "
        "column name is unknown.",
        outputName);
    const auto handle =
        std::dynamic_pointer_cast<const connector::hive::HiveColumnHandle>(
            assignment->second);
    VELOX_USER_CHECK_NOT_NULL(
        handle,
        "Scan output column '{}' is not assigned a Hive column handle, so its "
        "physical Parquet column name is unknown.",
        outputName);
    VELOX_USER_CHECK(
        handle->columnType() ==
            connector::hive::HiveColumnHandle::ColumnType::kRegular,
        "Scan output column '{}' is not a regular file column, so it has no "
        "physical Parquet column to read.",
        outputName);
    columnNames.push_back(handle->name());
  }

  VELOX_USER_CHECK(
      !columnNames.empty(),
      "The table scan produced no physical Parquet column names to read.");
  return columnNames;
}

} // namespace

void CudfTpchBenchmark::initialize() {
  if (!FLAGS_cudf_properties.empty()) {
    cudf_velox::CudfConfig::getInstance().initialize(
        cudf_velox::loadPropertiesFile(FLAGS_cudf_properties));
  }

  const auto ioMode = cudf_velox::parseCudfTpchIoMode(FLAGS_cudf_io_mode);
  const bool ioEnabled = ioMode != cudf_velox::CudfTpchIoMode::kDisabled;

  if (ioEnabled) {
    // Ahead of initializeResources() because the base initialization it runs
    // reads the schema of the first object each manifest lists. Nothing has
    // been acquired yet either, so a rejected run needs no cleanup. The
    // connector properties built here are the ones the base will reuse.
    cudf_velox::validateCudfTpchIoPreflight(
        cudf_velox::CudfTpchIoSettings{
            .mode = ioMode,
            .dataFormat = FLAGS_data_format,
            .dataPath = FLAGS_data_path,
            .table = FLAGS_cudf_io_table,
            .runQueryVerbose = FLAGS_run_query_verbose,
            .ioMeterColumnPct = FLAGS_io_meter_column_pct,
            .numDrivers = FLAGS_num_drivers,
            .numRepeats = FLAGS_num_repeats,
            .readSizeBytes = FLAGS_cudf_io_read_size_bytes,
            .cudfTableScan = FLAGS_velox_cudf_table_scan,
        },
        *ensureConnectorProperties(),
        environmentValue("AWS_ENDPOINT_URL"));
  }

  // Everything below acquires process-global state, so a failure part way
  // through has to release what was already acquired before the original
  // error escapes.
  try {
    initializeResources(ioEnabled);
  } catch (...) {
    lifecycle_.releaseAfterFailure("benchmark initialization");
    throw;
  }
}

void CudfTpchBenchmark::ownConnector(
    std::shared_ptr<connector::Connector> installed) {
  using cudf_velox::CudfBenchmarkResource;

  benchmarkConnector_ = std::move(installed);
  lifecycle_.own(CudfBenchmarkResource::kConnector, [this] {
    const auto& connectorId = facebook::velox::exec::test::kHiveConnectorId;
    // Anything else registered under this id since is not this benchmark's to
    // remove, whether it replaced the entry or the entry is already gone.
    if (connector::ConnectorRegistry::tryGet(connectorId) ==
        benchmarkConnector_) {
      connector::ConnectorRegistry::global().erase(connectorId);
    }
    benchmarkConnector_.reset();
  });
}

void CudfTpchBenchmark::initializeResources(bool ioEnabled) {
  using cudf_velox::CudfBenchmarkResource;
  const auto& connectorId = facebook::velox::exec::test::kHiveConnectorId;

  if (ioEnabled) {
#ifndef VELOX_ENABLE_S3
    VELOX_USER_FAIL(
        "S3 support is not available in this build. "
        "Rebuild with -DVELOX_ENABLE_S3=ON to use I/O benchmark modes.");
#endif
    // Finalizing is process-global and one-shot, so only the benchmark that
    // introduced the S3 file system may take it away; registering over a host
    // process's own registration is a no-op that leaves it in charge.
    // Recorded before the call because a registration that fails part way
    // still has state to undo.
    if (!filesystems::isPathSupportedByRegisteredFileSystems(
            kS3RegistrationProbe)) {
      lifecycle_.own(CudfBenchmarkResource::kS3FileSystem, [] {
        filesystems::finalizeS3FileSystem();
      });
    }
    filesystems::registerS3FileSystem();
  }

  // Recorded before the call that creates them, because a failure inside it can
  // leave either behind. Each release below does nothing when its resource was
  // never created.
  lifecycle_.own(CudfBenchmarkResource::kBaseBenchmark, [this] {
    TpchBenchmark::shutdown();
  });
  lifecycle_.own(
      CudfBenchmarkResource::kIoExecutor, [this] { ioExecutor_.reset(); });

  // A connector the host process registered is not this benchmark's to erase,
  // and the base's insert would fail rather than displace it, so what the
  // registry holds now is what tells the two apart afterwards. The base
  // registers its connector before it reads any schema, so a failure after
  // that point still leaves one to claim.
  const auto preexistingConnector =
      connector::ConnectorRegistry::tryGet(connectorId);
  const auto claimConnector = [&] {
    auto current = connector::ConnectorRegistry::tryGet(connectorId);
    if (current != nullptr && current != preexistingConnector) {
      ownConnector(std::move(current));
    }
  };
  try {
    TpchBenchmark::initialize();
  } catch (...) {
    claimConnector();
    throw;
  }
  claimConnector();

  if (FLAGS_velox_cudf_table_scan) {
    // The entry is the base's own connector: an externally owned one would
    // have failed the insert above rather than been replaced here.
    connector::ConnectorRegistry::global().erase(connectorId);

    // When an I/O mode is active, start from the full connector properties
    // (which includes S3 settings from --connector_properties) so the
    // CudfHive connector can reach cloud storage.  In ordinary mode, start
    // from an empty map to preserve the pre-Phase-2 behavior of cuDF-only
    // overrides.
    std::unordered_map<std::string, std::string> cudfHiveConfigurationValues;
    if (ioEnabled) {
      cudfHiveConfigurationValues = connectorProperties_->rawConfigsCopy();
    }

    // CuDF-specific overrides always win.
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxChunkReadLimit] =
            std::to_string(FLAGS_cudf_chunk_read_limit);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxPassReadLimit] =
            std::to_string(FLAGS_cudf_pass_read_limit);
    cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                    kAllowMismatchedCudfHiveSchemas] =
        std::to_string(true);
    auto cudfHiveProperties = std::make_shared<const config::ConfigBase>(
        std::move(cudfHiveConfigurationValues));

    cudf_velox::connector::hive::CudfHiveConnectorFactory cudfHiveFactory;
    auto cudfHiveConnector = cudfHiveFactory.newConnector(
        connectorId, cudfHiveProperties, ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        cudfHiveConnector->connectorId(), cudfHiveConnector);
    ownConnector(std::move(cudfHiveConnector));
  }

  // Unregistering a cuDF the host process registered would take its operators
  // away, so only a benchmark that found cuDF unregistered may do it.
  // Recorded before the call for the same reason as the two above.
  if (!cudf_velox::cudfIsRegistered()) {
    lifecycle_.own(
        CudfBenchmarkResource::kCudf, [] { cudf_velox::unregisterCudf(); });
  }
  cudf_velox::registerCudf();
  // Register the BenchmarkDiscard adapter after registerCudf() because
  // registerCudf() rebuilds the adapter registry from scratch.
  cudf_velox::registerCudfBenchmarkDiscard();
  cudf_velox::registerPrestoFunctions(
      cudf_velox::CudfConfig::getInstance().functionNamePrefix);

  queryConfigs_[facebook::velox::cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
      std::to_string(FLAGS_cudf_gpu_batch_size_rows);
  queryConfigs_[core::QueryConfig::kMaxLocalExchangeBufferSize] =
      std::to_string(FLAGS_cudf_local_exchange_buffer_size);
}

std::shared_ptr<config::ConfigBase>
CudfTpchBenchmark::makeConnectorProperties() {
  auto cfg = TpchBenchmark::makeConnectorProperties();
  using CudfHiveCfg = cudf_velox::connector::hive::CudfHiveConfig;

  // CuDF-specific properties.
  cfg->set(
      CudfHiveCfg::kMaxChunkReadLimit,
      std::to_string(FLAGS_cudf_chunk_read_limit));
  cfg->set(
      CudfHiveCfg::kMaxPassReadLimit,
      std::to_string(FLAGS_cudf_pass_read_limit));
  cfg->set(CudfHiveCfg::kAllowMismatchedCudfHiveSchemas, "true");

  const auto ioMode = cudf_velox::parseCudfTpchIoMode(FLAGS_cudf_io_mode);
  const bool ioEnabled = ioMode != cudf_velox::CudfTpchIoMode::kDisabled;

  if (ioEnabled) {
    // Apply any connector_properties file on top of the base config.
    if (!FLAGS_connector_properties.empty()) {
      for (auto& [k, v] :
           cudf_velox::loadPropertiesFile(FLAGS_connector_properties)) {
        cfg->set(k, v);
      }
    }

    // Benchmark-owned settings win over property-file collisions.
    cfg->set(CudfHiveCfg::kUseBufferedInput, "false");
    if (ioMode == cudf_velox::CudfTpchIoMode::kDecodeDiscard) {
      cfg->set(CudfHiveCfg::kUseExperimentalCudfReader, "true");
    }
  }

  return cfg;
}

std::vector<std::shared_ptr<connector::ConnectorSplit>>
CudfTpchBenchmark::listSplits(
    const std::string& path,
    int32_t numSplitsPerFile,
    const exec::test::TpchPlan& plan) {
  // TODO (dm): Figure out a way to enforce 1 split per file in
  // CudfHiveDataSource outside of this benchmark
  if (FLAGS_velox_cudf_table_scan) {
    // One split covering the whole object, built without asking storage how
    // large it is: an unbounded length reads every row group, and the test
    // helper that would size the split instead opens the file through a file
    // system configured with no properties, which cannot reach cloud storage.
    return {connector::hive::HiveConnectorSplitBuilder(path)
                .connectorId(facebook::velox::exec::test::kHiveConnectorId)
                .fileFormat(plan.dataFileFormat)
                .build()};
  }

  return TpchBenchmark::listSplits(path, numSplitsPerFile, plan);
}

void CudfTpchBenchmark::shutdown() {
  lifecycle_.release();
}

void CudfTpchBenchmark::runMain(std::ostream& out, RunStats& runStats) {
  using namespace cudf_velox;
  namespace cudf_hive = cudf_velox::connector::hive;
  using Clock = std::chrono::steady_clock;

  const auto ioMode = parseCudfTpchIoMode(FLAGS_cudf_io_mode);

  // Disabled: delegate to the standard TPC-H benchmark.
  if (ioMode == CudfTpchIoMode::kDisabled) {
    TpchBenchmark::runMain(out, runStats);
    return;
  }

  // Both raw modes bypass the task and read through the raw runner below;
  // decode-discard falls through to the plan-based path after them.
  std::optional<CudfRawReadMode> rawMode;
  switch (ioMode) {
    case CudfTpchIoMode::kDisabled:
      VELOX_UNREACHABLE("Disabled mode is delegated above.");
    case CudfTpchIoMode::kDecodeDiscard:
      break;
    case CudfTpchIoMode::kRawFile:
      rawMode = CudfRawReadMode::kFile;
      break;
    case CudfTpchIoMode::kRawParquetRanges:
      rawMode = CudfRawReadMode::kParquetRanges;
      break;
  }

  // Everything that does not need the plan was checked by the preflight in
  // initialize(), before the base initialization could read a schema. What
  // follows is the shape of the plan that check could not see, kept here as
  // defense in depth.

  // Build the unfiltered table-scan plan and wrap it with BenchmarkDiscard.
  const TpchPlan scanPlan =
      queryBuilder()->getTableScanPlan(FLAGS_cudf_io_table);

  VELOX_USER_CHECK(
      !scanPlan.dataFiles.empty(),
      "No data files found for table '{}'. "
      "Check --data_path and --cudf_io_table.",
      FLAGS_cudf_io_table);

  // Verify every node binding is non-empty and every path is an S3 URI.
  for (const auto& [nodeId, paths] : scanPlan.dataFiles) {
    VELOX_USER_CHECK(
        !paths.empty(),
        "Empty file list for scan node '{}' in table '{}'. "
        "Check --data_path and --cudf_io_table.",
        nodeId,
        FLAGS_cudf_io_table);
    for (const auto& path : paths) {
      VELOX_USER_CHECK(
          path.rfind("s3:", 0) == 0 || path.rfind("s3a:", 0) == 0,
          "Expected an S3 path (s3:// or s3a://) but got '{}'. "
          "Set --data_path to an S3 prefix.",
          path);
    }
  }

  // Payload counters are published per scan node, so a single scan node is
  // required to attribute them unambiguously.
  VELOX_USER_CHECK_EQ(
      scanPlan.dataFiles.size(),
      1,
      "I/O benchmark modes require exactly one scan node but table '{}' "
      "produced {}.",
      FLAGS_cudf_io_table,
      scanPlan.dataFiles.size());

  // Raw modes read the manifest the scan plan discovered without building a
  // task, so the single binding can be flattened now that every path has been
  // validated.
  if (rawMode.has_value()) {
    const std::vector<std::string>& paths = scanPlan.dataFiles.begin()->second;

    out << "note: read_wall_s is summed over all workers and can overlap in "
           "wall time, so it is a diagnostic rather than a throughput "
           "denominator."
        << std::endl;

    // Exact ranges are selected by physical Parquet column name; whole-object
    // reads never look at the schema. Enumerated rather than tested against
    // one mode, so a mode added later has to say what it selects here.
    std::optional<std::vector<std::string>> columnNames;
    switch (*rawMode) {
      case CudfRawReadMode::kParquetRanges:
        columnNames = physicalColumnNames(*scanPlan.plan);
        break;
      case CudfRawReadMode::kFile:
        columnNames.emplace();
        break;
    }
    VELOX_CHECK(
        columnNames.has_value(),
        "Unhandled raw read mode {}.",
        static_cast<int>(*rawMode));

    for (int32_t rep = 0; rep < FLAGS_num_repeats; ++rep) {
      const auto rawStats = runCudfRawRead(
          paths,
          CudfRawReadOptions{
              .mode = *rawMode,
              .numWorkers = FLAGS_num_drivers,
              .readSizeBytes =
                  static_cast<uint64_t>(FLAGS_cudf_io_read_size_bytes),
              .columnNames = *columnNames,
          },
          get_temp_mr());

      const double setupSecs = static_cast<double>(rawStats.setupNanos) / 1e9;
      const double elapsedSecs =
          static_cast<double>(rawStats.elapsedNanos) / 1e9;
      const double readWallSecs =
          static_cast<double>(rawStats.readWallNanos) / 1e9;
      // Bytes that actually arrived are the throughput numerator. The runner
      // fails a short read rather than reporting fewer, so this equals the
      // requested count on every run that returns.
      const double payloadBytesPerSec = elapsedSecs > 0
          ? static_cast<double>(rawStats.completedBytes) / elapsedSecs
          : 0.0;

      out << "mode=" << FLAGS_cudf_io_mode << " table=" << FLAGS_cudf_io_table
          << " repeat=" << rep << " files=" << rawStats.numFiles
          << " requested_workers=" << FLAGS_num_drivers
          << " effective_workers=" << rawStats.effectiveWorkers;

      switch (*rawMode) {
        case CudfRawReadMode::kParquetRanges:
          out << " selected_row_groups=" << rawStats.selectedRowGroups
              << " column_chunk_logical_ranges=" << rawStats.logicalRanges
              << " column_chunk_physical_requests="
              << rawStats.physicalRequests;
          break;
        case CudfRawReadMode::kFile:
          out << " read_size_bytes=" << FLAGS_cudf_io_read_size_bytes
              << " read_chunks=" << rawStats.logicalRanges
              << " physical_requests=" << rawStats.physicalRequests;
          break;
      }

      out << " requested_bytes=" << rawStats.requestedBytes
          << " completed_bytes=" << rawStats.completedBytes
          << " setup_s=" << setupSecs << " elapsed_s=" << elapsedSecs
          << " read_wall_s=" << readWallSecs
          << " completed_bytes_per_s=" << payloadBytesPerSec << std::endl;
    }
    return;
  }

  const core::PlanNodeId scanNodeId = scanPlan.dataFiles.begin()->first;

  // Count total files for effective-driver calculation.
  size_t totalFiles = 0;
  for (const auto& [nodeId, paths] : scanPlan.dataFiles) {
    totalFiles += paths.size();
  }
  VELOX_USER_CHECK_GT(
      totalFiles,
      0,
      "Total file count is zero for table '{}'. "
      "Check --data_path and --cudf_io_table.",
      FLAGS_cudf_io_table);
  const int32_t effectiveDrivers =
      std::min<int32_t>(FLAGS_num_drivers, static_cast<int32_t>(totalFiles));

  // Wrap the scan with the discard sink.
  TpchPlan discardPlan = scanPlan;
  discardPlan.plan = addBenchmarkDiscard(scanPlan.plan);

  out << "note: column_chunk_read_wall_s and parquet_decode_gpu_s are summed "
         "over all drivers and can overlap in wall time, so they are "
         "diagnostics rather than throughput denominators."
      << std::endl;

  // Run the benchmark loop.
  for (int32_t rep = 0; rep < FLAGS_num_repeats; ++rep) {
    const auto t0 = Clock::now();
    auto [cursor, results] = runOnce(discardPlan, queryConfigs_);
    const double elapsedSecs =
        std::chrono::duration<double>(Clock::now() - t0).count();

    VELOX_CHECK(
        results.empty(),
        "BenchmarkDiscard should produce no output but got {} vectors.",
        results.size());

    // Collect runtime counters from the task.
    const auto planStats = toPlanStats(cursor->task()->taskStats());

    // Read only the discard root node; require it and all three counters.
    const core::PlanNodeId discardNodeId = discardPlan.plan->id();
    VELOX_CHECK(
        planStats.count(discardNodeId),
        "BenchmarkDiscard node '{}' not found in task plan stats.",
        discardNodeId);
    const auto& discardStats = planStats.at(discardNodeId).customStats;
    VELOX_CHECK(
        discardStats.count(std::string(kDiscardedRows)),
        "Runtime counter '{}' missing from BenchmarkDiscard node '{}'.",
        kDiscardedRows,
        discardNodeId);
    VELOX_CHECK(
        discardStats.count(std::string(kDiscardedBytes)),
        "Runtime counter '{}' missing from BenchmarkDiscard node '{}'.",
        kDiscardedBytes,
        discardNodeId);
    VELOX_CHECK(
        discardStats.count(std::string(kDiscardedBatches)),
        "Runtime counter '{}' missing from BenchmarkDiscard node '{}'.",
        kDiscardedBatches,
        discardNodeId);

    const int64_t totalRows = discardStats.at(std::string(kDiscardedRows)).sum;
    const int64_t totalBytes =
        discardStats.at(std::string(kDiscardedBytes)).sum;
    const int64_t totalBatches =
        discardStats.at(std::string(kDiscardedBatches)).sum;

    // Read the scan node's payload counters, published by the cuDF split
    // reader through the data source's runtime stats.
    VELOX_CHECK(
        planStats.count(scanNodeId),
        "Scan node '{}' not found in task plan stats.",
        scanNodeId);
    const auto& scanStats = planStats.at(scanNodeId).customStats;
    for (const auto counterName : {
             cudf_hive::kColumnChunkRequestedBytes,
             cudf_hive::kColumnChunkCompletedBytes,
             cudf_hive::kColumnChunkLogicalRanges,
             cudf_hive::kColumnChunkPhysicalRequests,
             cudf_hive::kColumnChunkReadWallNanos,
             cudf_hive::kParquetDecodeGpuNanos,
         }) {
      VELOX_CHECK(
          scanStats.count(std::string(counterName)),
          "Runtime counter '{}' missing from scan node '{}'.",
          counterName,
          scanNodeId);
    }

    const int64_t compressedRequestedBytes =
        scanStats.at(std::string(cudf_hive::kColumnChunkRequestedBytes)).sum;
    const int64_t compressedCompletedBytes =
        scanStats.at(std::string(cudf_hive::kColumnChunkCompletedBytes)).sum;
    VELOX_CHECK_EQ(
        compressedRequestedBytes,
        compressedCompletedBytes,
        "Scan node '{}' requested {} column chunk bytes but completed {}.",
        scanNodeId,
        compressedRequestedBytes,
        compressedCompletedBytes);
    VELOX_CHECK(
        totalRows == 0 || compressedRequestedBytes > 0,
        "Scan node '{}' decoded {} rows but reported zero requested column "
        "chunk bytes.",
        scanNodeId,
        totalRows);

    const int64_t logicalRanges =
        scanStats.at(std::string(cudf_hive::kColumnChunkLogicalRanges)).sum;
    const int64_t physicalRequests =
        scanStats.at(std::string(cudf_hive::kColumnChunkPhysicalRequests)).sum;
    const double readWallSecs =
        static_cast<double>(
            scanStats.at(std::string(cudf_hive::kColumnChunkReadWallNanos))
                .sum) /
        1e9;
    const double decodeGpuSecs =
        static_cast<double>(
            scanStats.at(std::string(cudf_hive::kParquetDecodeGpuNanos)).sum) /
        1e9;

    const double rowsPerSec =
        elapsedSecs > 0 ? static_cast<double>(totalRows) / elapsedSecs : 0.0;
    const double bytesPerSec =
        elapsedSecs > 0 ? static_cast<double>(totalBytes) / elapsedSecs : 0.0;
    // Bytes that actually arrived are the throughput numerator. A short read
    // fails the run rather than lowering the completed count, so this equals
    // the requested count on every run that reaches this line.
    const double compressedBytesPerSec = elapsedSecs > 0
        ? static_cast<double>(compressedCompletedBytes) / elapsedSecs
        : 0.0;

    out << "mode=" << FLAGS_cudf_io_mode << " table=" << FLAGS_cudf_io_table
        << " repeat=" << rep << " files=" << totalFiles
        << " requested_drivers=" << FLAGS_num_drivers
        << " effective_drivers=" << effectiveDrivers
        << " decoded_rows=" << totalRows << " decoded_bytes=" << totalBytes
        << " decoded_batches=" << totalBatches << " elapsed_s=" << elapsedSecs
        << " rows_per_s=" << rowsPerSec
        << " decoded_bytes_per_s=" << bytesPerSec
        << " compressed_requested_bytes=" << compressedRequestedBytes
        << " compressed_completed_bytes=" << compressedCompletedBytes
        << " compressed_completed_bytes_per_s=" << compressedBytesPerSec
        << " column_chunk_logical_ranges=" << logicalRanges
        << " column_chunk_physical_requests=" << physicalRequests
        << " column_chunk_read_wall_s=" << readWallSecs
        << " parquet_decode_gpu_s=" << decodeGpuSecs << std::endl;
  }
}
