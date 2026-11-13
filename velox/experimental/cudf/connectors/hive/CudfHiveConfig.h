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

#include "velox/common/config/Config.h"

#include <cudf/types.hpp>

#include <optional>

namespace facebook::velox::config {
class ConfigBase;
}

namespace facebook::velox::cudf_velox::connector::hive {

class CudfHiveConfig {
 public:
  // Reader config options

  // This isn't a typo; parquet connector and session config names are different
  // ('-' vs '_').
  static constexpr const char* kMaxChunkReadLimit =
      "parquet.reader.chunk-read-limit";
  static constexpr const char* kMaxChunkReadLimitSession =
      "parquet.reader.chunk_read_limit";

  static constexpr const char* kMaxPassReadLimit =
      "parquet.reader.pass-read-limit";
  static constexpr const char* kMaxPassReadLimitSession =
      "parquet.reader.pass_read_limit";

  // Whether to store string data as categorical type
  static constexpr const char* kConvertStringsToCategories =
      "parquet.reader.convert-strings-to-categories";
  static constexpr const char* kConvertStringsToCategoriesSession =
      "parquet.reader.convert_strings_to_categories";

  // Whether to use PANDAS metadata to load columns
  static constexpr const char* kUsePandasMetadata =
      "parquet.reader.use-pandas-metadata";
  static constexpr const char* kUsePandasMetadataSession =
      "parquet.reader.use_pandas_metadata";

  // Whether to read and use ARROW schema
  static constexpr const char* kUseArrowSchema =
      "parquet.reader.use-arrow-schema";
  static constexpr const char* kUseArrowSchemaSession =
      "parquet.reader.use_arrow_schema";

  // Whether to allow reading matching select columns from mismatched CudfHive
  // files.
  static constexpr const char* kAllowMismatchedCudfHiveSchemas =
      "parquet.reader.allow-mismatched-parquet-schemas";
  static constexpr const char* kAllowMismatchedCudfHiveSchemasSession =
      "parquet.reader.allow_mismatched_parquet_schemas";

  // Cast timestamp columns to a specific type
  static constexpr const char* kTimestampType = "parquet.reader.timestamp-type";
  static constexpr const char* kTimestampTypeSession =
      "parquet.reader.timestamp_type";

  // Whether pages of flat fixed-width columns are decoded by libcudf's tile
  // decode kernels. Results do not depend on it.
  static constexpr const char* kUseTileDecode =
      "parquet.reader.use-tile-decode";
  static constexpr const char* kUseTileDecodeSession =
      "parquet.reader.use_tile_decode";

  // Whether single-column conjuncts of the pushed-down filter are evaluated by
  // kernels that libcudf links at runtime for the scan, while the column is
  // decoded if the tile decode kernels are also on. Results do not depend on
  // it.
  static constexpr const char* kUseJitDecodeFilter =
      "parquet.reader.use-jit-decode-filter";
  static constexpr const char* kUseJitDecodeFilterSession =
      "parquet.reader.use_jit_decode_filter";

  // Whether to use the BufferedInput source for CudfHiveDataSource. The
  // buffered source takes advantage of the `AsyncDataCache` if available. The
  // `AsyncDataCache` is controlled (default: enabled) using the
  // HiveConfig::kEnableFileHandleCache config option
  static constexpr const char* kUseBufferedInput =
      "cudf.hive.use-buffered-input";
  static constexpr const char* kUseBufferedInputSession =
      "cudf.hive.use_buffered_input";

  // Whether background split preparation offers the first column-chunk pass
  // to the fetch-ahead budget, as kFetchAheadEnabled does. For a scan that
  // does not read through the fetch scheduler the preparation starts reading
  // it, unbounded.
  static constexpr const char* kPreloadColumnChunks =
      "cudf.hive.preload-column-chunks";
  static constexpr const char* kPreloadColumnChunksSession =
      "cudf.hive.preload_column_chunks";

  // Whether to serialize concurrent I/O request submission batches.
  static constexpr const char* kSerializeIoRequests =
      "cudf.hive.serialize-io-requests";
  static constexpr const char* kSerializeIoRequestsSession =
      "cudf.hive.serialize_io_requests";

  // Whether column chunks read through the BufferedInput source are staged in
  // page-locked host buffers on their way to the device. The buffers are
  // reused across reads, and a file that no cache sits in front of is read
  // straight into them. Disable to stage in freshly allocated pageable
  // memory.
  static constexpr const char* kPinnedStagingEnabled =
      "cudf.hive.pinned-staging-enabled";
  static constexpr const char* kPinnedStagingEnabledSession =
      "cudf.hive.pinned_staging_enabled";

  // Upper bound, in bytes, on the page-locked host memory a connector holds
  // for staging column chunks. Each connector has its own pool. The memory is
  // locked on demand, one buffer per concurrent reader, and kept for the
  // connector's lifetime; reads wait for a free buffer at the bound. One
  // buffer is always allowed. 0 disables the staging.
  static constexpr const char* kPinnedStagingPoolSize =
      "cudf.hive.pinned-staging-pool-size";

  // Size, in bytes, of one staging buffer, at least
  // kMinPinnedStagingBufferSize. Also the size of the reads issued to a local
  // file that no cache sits in front of: a longer column chunk range is split
  // into reads of this size that run in parallel. Other files are read
  // through BufferedInput, one read per range.
  static constexpr const char* kPinnedStagingBufferSize =
      "cudf.hive.pinned-staging-buffer-size";

  // Whether the column chunks staged in page-locked buffers are moved to the
  // device by the connector's fetch scheduler, which every scan of the
  // connector shares. It reads with its own threads, as many at a time as
  // the storage and the copies to the device can use. Disable to read on the
  // IO executor. Has no effect when pinned staging is off.
  static constexpr const char* kFetchSchedulerEnabled =
      "cudf.hive.fetch-scheduler-enabled";
  static constexpr const char* kFetchSchedulerEnabledSession =
      "cudf.hive.fetch_scheduler_enabled";

  // Upper bound on the reader threads of a connector's fetch scheduler, at
  // most kMaxFetchSchedulerReaders. Zero stands for half of the CPUs the
  // process may run on, divided by the number of devices it sees, between 4
  // and 64. The scheduler chooses how many of them read at a time.
  static constexpr const char* kFetchSchedulerMaxReaders =
      "cudf.hive.fetch-scheduler-max-readers";

  // Fewest readers a connector's fetch scheduler keeps at work while column
  // chunks wait. Equal to the upper bound it pins the reader count, which is
  // meant for measurements.
  static constexpr const char* kFetchSchedulerMinReaders =
      "cudf.hive.fetch-scheduler-min-readers";

  // Largest accepted bound on the reader threads of a fetch scheduler.
  static constexpr uint32_t kMaxFetchSchedulerReaders = 256;

  // How a connector's fetch scheduler waits for the copy of a filled staging
  // buffer: "auto" chooses by a measurement of the machine, "wait" makes the
  // reader wait before it fills another buffer, "queue" lets it go on.
  // Anything but "auto" is meant for measurements.
  static constexpr const char* kFetchSchedulerCopyMode =
      "cudf.hive.fetch-scheduler-copy-mode";

  // Whether the column chunks of splits prepared ahead of the drivers are
  // fetched ahead too, while the device memory held by fetched and not yet
  // decoded column chunks stays within the fetch-ahead budget. Applies to
  // scans that read through the fetch scheduler. kPreloadColumnChunks asks
  // for the same, whatever this says.
  static constexpr const char* kFetchAheadEnabled =
      "cudf.hive.fetch-ahead-enabled";
  static constexpr const char* kFetchAheadEnabledSession =
      "cudf.hive.fetch_ahead_enabled";

  // Capacity in bytes of a fetch-ahead budget of the connector's own. Zero
  // makes the connector use the budget all connectors of the process share:
  // one sixteenth of the device's memory.
  static constexpr const char* kFetchAheadBytes = "cudf.hive.fetch-ahead-bytes";

  // Smallest accepted staging buffer size. A range takes one queued read per
  // buffer, so a tiny buffer would flood the IO executor.
  static constexpr uint64_t kMinPinnedStagingBufferSize = 64 << 10;

  // Writer config options

  /// Whether new data can be inserted into a CudfHive file
  /// Cudf-Velox currently does not support appending data to existing files.
  static constexpr const char* kImmutableFiles = "parquet.immutable-files";

  /// Sort Writer will exit finish() method after this many milliseconds even if
  /// it has not completed its work yet. Zero means no time limit.
  static constexpr const char* kSortWriterFinishTimeSliceLimitMs =
      "sort-writer-finish-time-slice-limit-ms";
  static constexpr const char* kSortWriterFinishTimeSliceLimitMsSession =
      "sort_writer_finish_time_slice_limit_ms";

  static constexpr const char* kWriteTimestampsAsUTC =
      "parquet.writer.write-timestamps-as-utc";
  static constexpr const char* kWriteTimestampsAsUTCSession =
      "parquet.writer.write_timestamps_as_utc";

  static constexpr const char* kWriteArrowSchema =
      "parquet.writer.write-arrow-schema";
  static constexpr const char* kWriteArrowSchemaSession =
      "parquet.writer.write_arrow_schema";

  static constexpr const char* kWritev2PageHeaders =
      "parquet.writer.write-v2-page-headers";
  static constexpr const char* kWritev2PageHeadersSession =
      "parquet.writer.write_v2_page_headers";

  CudfHiveConfig(std::shared_ptr<const config::ConfigBase> config) {
    VELOX_CHECK_NOT_NULL(
        config, "Config is null for CudfHiveConfig initialization");
    config_ = std::move(config);
  }

  const std::shared_ptr<const config::ConfigBase>& config() const {
    return config_;
  }

  uint64_t sortWriterFinishTimeSliceLimitMs(
      const config::ConfigBase* session) const;

  std::size_t maxChunkReadLimit() const;
  std::size_t maxChunkReadLimitSession(const config::ConfigBase* session) const;

  std::size_t maxPassReadLimit() const;
  std::size_t maxPassReadLimitSession(const config::ConfigBase* session) const;

  bool isConvertStringsToCategories() const;
  bool isConvertStringsToCategoriesSession(
      const config::ConfigBase* session) const;

  bool isUsePandasMetadata() const;
  bool isUsePandasMetadataSession(const config::ConfigBase* session) const;

  bool isUseArrowSchema() const;
  bool isUseArrowSchemaSession(const config::ConfigBase* session) const;

  bool isAllowMismatchedCudfHiveSchemas() const;
  bool isAllowMismatchedCudfHiveSchemasSession(
      const config::ConfigBase* session) const;

  cudf::data_type timestampType() const;
  cudf::data_type timestampTypeSession(const config::ConfigBase* session) const;

  bool isUseTileDecode() const;
  bool isUseTileDecodeSession(const config::ConfigBase* session) const;

  bool isUseJitDecodeFilter() const;
  bool isUseJitDecodeFilterSession(const config::ConfigBase* session) const;

  bool useBufferedInput() const;
  bool useBufferedInputSession(const config::ConfigBase* session) const;

  bool preloadColumnChunks() const;
  bool preloadColumnChunksSession(const config::ConfigBase* session) const;

  bool serializeIoRequests() const;
  bool serializeIoRequestsSession(const config::ConfigBase* session) const;

  bool pinnedStagingEnabled() const;
  bool pinnedStagingEnabledSession(const config::ConfigBase* session) const;

  uint64_t pinnedStagingPoolSize() const;

  uint64_t pinnedStagingBufferSize() const;

  /// Returns whether scans move staged column chunks to the device through
  /// the connector's fetch scheduler.
  bool fetchSchedulerEnabled() const;

  /// Returns fetchSchedulerEnabled() unless 'session' overrides it.
  bool fetchSchedulerEnabledSession(const config::ConfigBase* session) const;

  /// Returns the upper bound on the reader threads of the connector's fetch
  /// scheduler.
  uint32_t fetchSchedulerMaxReaders() const;

  /// Returns the fewest readers the connector's fetch scheduler keeps at
  /// work.
  uint32_t fetchSchedulerMinReaders() const;

  /// Returns how the connector's fetch scheduler waits for copies: "auto",
  /// "wait" or "queue".
  std::string fetchSchedulerCopyMode() const;

  /// Returns whether the column chunks of splits prepared ahead are fetched
  /// ahead within the fetch-ahead budget.
  bool fetchAheadEnabled() const;

  /// Returns fetchAheadEnabled() unless 'session' overrides it.
  bool fetchAheadEnabledSession(const config::ConfigBase* session) const;

  /// Returns the capacity of the fetch-ahead budget in bytes, zero when it
  /// is to be taken from the device's memory.
  uint64_t fetchAheadBytes() const;

  bool immutableFiles() const;

  bool writeTimestampsAsUTC() const;
  bool writeTimestampsAsUTCSession(const config::ConfigBase* session) const;

  bool writeArrowSchema() const;
  bool writeArrowSchemaSession(const config::ConfigBase* session) const;

  bool writev2PageHeaders() const;
  bool writev2PageHeadersSession(const config::ConfigBase* session) const;

 private:
  std::shared_ptr<const config::ConfigBase> config_;
};
} // namespace facebook::velox::cudf_velox::connector::hive
