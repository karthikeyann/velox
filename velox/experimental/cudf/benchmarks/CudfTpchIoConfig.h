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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox {

/// I/O mode for the cuDF TPC-H S3 benchmark.
enum class CudfTpchIoMode {
  /// No I/O benchmark; delegates to the normal TPC-H query path.
  kDisabled,
  /// TableScan (all columns, no filter) followed by BenchmarkDiscard.
  /// Reports decoded rows/bytes alongside the compressed column-chunk bytes
  /// the scan fetched.
  kDecodeDiscard,
  /// Fetches, without decoding, the compressed Parquet column chunks a scan of
  /// every column would read.
  kRawParquetRanges,
  /// Fetches every byte of every object in bounded pieces, reading no Parquet
  /// metadata.
  kRawFile,
};

/// Parses an I/O mode string into a CudfTpchIoMode value.
///
/// Recognized strings: "" → kDisabled, "decode_discard" → kDecodeDiscard,
/// "raw_parquet_ranges" → kRawParquetRanges, "raw_file" → kRawFile.
/// All other values produce a VELOX_USER_FAIL with the supplied string
/// embedded in the message.
CudfTpchIoMode parseCudfTpchIoMode(std::string_view mode);

/// Returns a canonical endpoint URL.
///
/// The function:
/// - trims surrounding ASCII whitespace;
/// - rejects an empty endpoint after trimming;
/// - detects an explicit `http://` or `https://` prefix (case-insensitive);
/// - when no scheme is present, prepends `https://` if sslEnabled is true,
///   otherwise `http://`;
/// - lowercases the URI scheme and host while preserving path and port;
/// - strips all trailing `/` characters.
std::string normalizeS3Endpoint(std::string_view endpoint, bool sslEnabled);

/// Validates that the Velox connector endpoint and the KvikIO endpoint agree.
///
/// Reads `hive.s3.endpoint` and `hive.s3.ssl.enabled` from
/// connectorProperties.  The check is a no-op when neither side specifies
/// an endpoint.  The function fails when exactly one side specifies an
/// endpoint, or when both sides specify endpoints whose normalized forms
/// differ.  Error messages include both normalized endpoints but no
/// credentials.
void validateS3EndpointConsistency(
    const config::ConfigBase& connectorProperties,
    std::optional<std::string_view> kvikioEndpoint);

/// Everything an enabled I/O benchmark mode is configured with that can be
/// judged without a decoded plan. One field per flag the mode reads, in the
/// raw form the flag carries.
struct CudfTpchIoSettings {
  /// Parsed `--cudf_io_mode`. Must name an enabled mode.
  CudfTpchIoMode mode;

  /// `--data_format`.
  std::string_view dataFormat;

  /// `--data_path`.
  std::string_view dataPath;

  /// `--cudf_io_table`.
  std::string_view table;

  /// `--run_query_verbose`.
  int32_t runQueryVerbose;

  /// `--io_meter_column_pct`.
  int32_t ioMeterColumnPct;

  /// `--num_drivers`.
  int32_t numDrivers;

  /// `--num_repeats`.
  int32_t numRepeats;

  /// `--cudf_io_read_size_bytes`.
  int64_t readSizeBytes;

  /// `--velox_cudf_table_scan`.
  bool cudfTableScan;
};

/// Rejects a misconfigured I/O benchmark run before it can reach storage.
///
/// The benchmark contacts storage as soon as it initializes, because the TPC-H
/// query builder reads the schema of the first object each table lists. Every
/// check that does not need a decoded plan therefore belongs here, ahead of
/// that: a run that cannot produce a meaningful measurement must fail without
/// having opened a connector, registered a file system or issued a request.
///
/// Checks, in order: the format is Parquet; neither of the query-only flags is
/// set; drivers, repeats and read size are positive; the table is a canonical
/// TPC-H table; the cuDF table scan is enabled, without which the payload
/// counters the modes report would never be published; the connector and
/// KvikIO endpoints agree; and `dataPath/table` is a local manifest file, not a
/// directory, holding at least one line, every line of which is a non-empty
/// `s3:` or `s3a:` URI.
///
/// Only the manifest is read, and only from the local file system.
///
/// @param settings Flag values of the run.
/// @param connectorProperties Connector configuration the run would use,
/// already including anything `--connector_properties` supplied.
/// @param kvikioEndpoint Value of `AWS_ENDPOINT_URL`, or nullopt when it is
/// unset.
///
/// @throws VeloxUserError naming the setting at fault. Endpoint messages carry
/// normalized endpoints but no credentials.
void validateCudfTpchIoPreflight(
    const CudfTpchIoSettings& settings,
    const config::ConfigBase& connectorProperties,
    std::optional<std::string_view> kvikioEndpoint);

} // namespace facebook::velox::cudf_velox
