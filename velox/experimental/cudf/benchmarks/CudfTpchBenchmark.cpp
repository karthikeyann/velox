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
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <cudf/io/parquet.hpp>

#include <experimental/cudf/connectors/hive/CudfHiveConnector.h>

#include <iostream>

DECLARE_int64(max_coalesced_bytes);
DECLARE_string(max_coalesced_distance_bytes);
DECLARE_int32(parquet_prefetch_rowgroups);

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
DEFINE_bool(
    cudf_hive_use_buffered_input,
    true,
    "Use BufferedInputDataSource instead of KvikIO for cuDF Hive reads");

DEFINE_bool(
    cudf_parquet_use_tile_decode,
    false,
    "Decode pages of flat fixed-width columns with libcudf's tile decode "
    "kernels");

DEFINE_bool(
    cudf_parquet_use_jit_decode_filter,
    false,
    "Evaluate single-column conjuncts of pushed-down filters with kernels "
    "that libcudf links at runtime for the scan");

DEFINE_bool(
    cudf_hive_preload_column_chunks,
    false,
    "Load column chunks while preloading cuDF Hive splits");

DEFINE_bool(
    cudf_hive_serialize_io_requests,
    false,
    "Serialize I/O submissions to prevent interleaving between concurrent splits");

DEFINE_bool(
    cudf_hive_pinned_staging_enabled,
    true,
    "Stage column chunks in page-locked host buffers on their way to the device");

DEFINE_uint64(
    cudf_hive_pinned_staging_pool_size,
    256ULL << 20,
    "Upper bound in bytes on the page-locked staging memory; 0 disables it");

DEFINE_uint64(
    cudf_hive_pinned_staging_buffer_size,
    4ULL << 20,
    "Size in bytes of one page-locked staging buffer");

DEFINE_bool(
    cudf_hive_fetch_scheduler_enabled,
    true,
    "Move staged column chunks to the device with the connector's fetch "
    "scheduler instead of on the IO executor");

DEFINE_uint32(
    cudf_hive_fetch_scheduler_max_readers,
    0,
    "Upper bound on the reader threads of the fetch scheduler; 0 takes half "
    "of the hardware threads, between 4 and 64");

DEFINE_bool(
    cudf_hive_fetch_ahead_enabled,
    true,
    "Fetch the column chunks of splits prepared ahead of the drivers within "
    "the fetch-ahead budget");

DEFINE_uint64(
    cudf_hive_fetch_ahead_bytes,
    0,
    "Capacity of the fetch-ahead budget in bytes; 0 takes one sixteenth of "
    "the device's memory");

DEFINE_string(
    cudf_properties,
    "",
    "Path to a properties file for CudfConfig. Each line should be key=value "
    "(e.g. cudf.memory_resource=async). See CudfConfig for available keys.");

void CudfTpchBenchmark::initialize() {
  if (!FLAGS_cudf_properties.empty()) {
    cudf_velox::CudfConfig::getInstance().initialize(
        cudf_velox::loadPropertiesFile(FLAGS_cudf_properties));
  }

  TpchBenchmark::initialize();

  if (FLAGS_velox_cudf_table_scan) {
    connector::ConnectorRegistry::global().erase(
        facebook::velox::exec::test::kHiveConnectorId);

    auto cudfHiveConfigurationValues =
        std::unordered_map<std::string, std::string>();
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxChunkReadLimit] =
            std::to_string(FLAGS_cudf_chunk_read_limit);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxPassReadLimit] =
            std::to_string(FLAGS_cudf_pass_read_limit);
    cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                    kAllowMismatchedCudfHiveSchemas] =
        std::to_string(true);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kUseBufferedInput] =
            std::to_string(FLAGS_cudf_hive_use_buffered_input);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kUseTileDecode] =
            std::to_string(FLAGS_cudf_parquet_use_tile_decode);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kUseJitDecodeFilter] =
            std::to_string(FLAGS_cudf_parquet_use_jit_decode_filter);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kPreloadColumnChunks] =
            std::to_string(FLAGS_cudf_hive_preload_column_chunks);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kSerializeIoRequests] =
            std::to_string(FLAGS_cudf_hive_serialize_io_requests);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kPinnedStagingEnabled] =
            std::to_string(FLAGS_cudf_hive_pinned_staging_enabled);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kPinnedStagingPoolSize] =
            std::to_string(FLAGS_cudf_hive_pinned_staging_pool_size);
    cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                    kPinnedStagingBufferSize] =
        std::to_string(FLAGS_cudf_hive_pinned_staging_buffer_size);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kFetchSchedulerEnabled] =
            std::to_string(FLAGS_cudf_hive_fetch_scheduler_enabled);
    cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                    kFetchSchedulerMaxReaders] =
        std::to_string(FLAGS_cudf_hive_fetch_scheduler_max_readers);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kFetchAheadEnabled] =
            std::to_string(FLAGS_cudf_hive_fetch_ahead_enabled);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kFetchAheadBytes] =
            std::to_string(FLAGS_cudf_hive_fetch_ahead_bytes);
    auto cudfHiveProperties = std::make_shared<const config::ConfigBase>(
        std::move(cudfHiveConfigurationValues));

    cudf_velox::connector::hive::CudfHiveConnectorFactory cudfHiveFactory;
    auto cudfHiveConnector = cudfHiveFactory.newConnector(
        facebook::velox::exec::test::kHiveConnectorId,
        cudfHiveProperties,
        ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        cudfHiveConnector->connectorId(), cudfHiveConnector);
  }

  cudf_velox::registerCudf();
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
  cfg->set(
      CudfHiveCfg::kUseBufferedInput,
      std::to_string(FLAGS_cudf_hive_use_buffered_input));
  cfg->set(
      CudfHiveCfg::kUseTileDecode,
      std::to_string(FLAGS_cudf_parquet_use_tile_decode));
  cfg->set(
      CudfHiveCfg::kUseJitDecodeFilter,
      std::to_string(FLAGS_cudf_parquet_use_jit_decode_filter));
  cfg->set(
      CudfHiveCfg::kPreloadColumnChunks,
      std::to_string(FLAGS_cudf_hive_preload_column_chunks));
  cfg->set(
      CudfHiveCfg::kSerializeIoRequests,
      std::to_string(FLAGS_cudf_hive_serialize_io_requests));
  cfg->set(
      CudfHiveCfg::kPinnedStagingEnabled,
      std::to_string(FLAGS_cudf_hive_pinned_staging_enabled));
  cfg->set(
      CudfHiveCfg::kPinnedStagingPoolSize,
      std::to_string(FLAGS_cudf_hive_pinned_staging_pool_size));
  cfg->set(
      CudfHiveCfg::kPinnedStagingBufferSize,
      std::to_string(FLAGS_cudf_hive_pinned_staging_buffer_size));
  cfg->set(
      CudfHiveCfg::kFetchSchedulerEnabled,
      std::to_string(FLAGS_cudf_hive_fetch_scheduler_enabled));
  cfg->set(
      CudfHiveCfg::kFetchSchedulerMaxReaders,
      std::to_string(FLAGS_cudf_hive_fetch_scheduler_max_readers));
  cfg->set(
      CudfHiveCfg::kFetchAheadEnabled,
      std::to_string(FLAGS_cudf_hive_fetch_ahead_enabled));
  cfg->set(
      CudfHiveCfg::kFetchAheadBytes,
      std::to_string(FLAGS_cudf_hive_fetch_ahead_bytes));
  cfg->set(CudfHiveCfg::kAllowMismatchedCudfHiveSchemas, "true");

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
    // TODO (dm): Instead of this, we can maybe use
    // makeHiveConnectorSplits(vector<shared_ptr<TempFilePath>>&
    // filePaths)
    std::vector<std::shared_ptr<connector::ConnectorSplit>> result;
    auto temp = HiveConnectorTestBase::makeHiveConnectorSplits(
        path, 1, plan.dataFileFormat);
    for (auto& i : temp) {
      result.push_back(i);
    }
    return result;
  }

  return TpchBenchmark::listSplits(path, numSplitsPerFile, plan);
}

void CudfTpchBenchmark::shutdown() {
  if (FLAGS_cudf_parquet_use_jit_decode_filter) {
    // Shows whether the pushed-down filters were evaluated by the linked
    // kernels at all, and where.
    const auto stats = cudf::io::get_parquet_jit_filter_stats();
    std::cout << "cuDF Parquet JIT decode filter: " << stats.decoded
              << " predicates answered while decoding, " << stats.evaluated
              << " evaluated after decoding, " << stats.fallback
              << " filters evaluated as expressions" << std::endl;
  }
  cudf_velox::unregisterCudf();
  TpchBenchmark::shutdown();
}

int main(int argc, char** argv) {
  std::string kUsage(
      "This program benchmarks TPC-H queries. Run 'velox_cudf_tpch_benchmark -helpon=TpchBenchmark' for available options.\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  benchmark = std::make_unique<CudfTpchBenchmark>();
  tpchBenchmarkMain();
}
