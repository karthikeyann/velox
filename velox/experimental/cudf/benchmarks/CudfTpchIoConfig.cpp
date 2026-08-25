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

#include "velox/experimental/cudf/benchmarks/CudfTpchIoConfig.h"

#include "velox/common/base/Exceptions.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"

#include <folly/String.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox {
namespace {

// URI schemes the benchmark accepts in a manifest. KvikIO speaks only "s3:",
// but the Velox connector also accepts the Hadoop-style "s3a:", which the raw
// runner and the data source both rewrite before opening.
constexpr std::string_view kS3Scheme{"s3:"};
constexpr std::string_view kS3aScheme{"s3a:"};

// Validates the manifest 'dataPath/table' without opening anything remote.
void validateManifest(std::string_view dataPath, std::string_view table) {
  const std::filesystem::path manifestPath =
      std::filesystem::path(std::string(dataPath)) / std::string(table);

  std::error_code error;
  const auto status = std::filesystem::status(manifestPath, error);
  VELOX_USER_CHECK(
      !error,
      "Cannot read the manifest '{}' for table '{}': {}. Set --data_path to a "
      "directory holding one manifest file per table.",
      manifestPath.string(),
      table,
      error.message());
  VELOX_USER_CHECK(
      !std::filesystem::is_directory(status),
      "The manifest '{}' for table '{}' is a directory. An I/O benchmark mode "
      "reads objects listed in a file, not files found in a directory.",
      manifestPath.string(),
      table);
  VELOX_USER_CHECK(
      std::filesystem::is_regular_file(status),
      "The manifest '{}' for table '{}' is not a regular file.",
      manifestPath.string(),
      table);

  std::ifstream manifest(manifestPath);
  VELOX_USER_CHECK(
      manifest.is_open(),
      "Cannot open the manifest '{}' for table '{}'.",
      manifestPath.string(),
      table);

  // The query builder takes every line as a URI, skipping neither blanks nor
  // comments, so every line is checked the same way.
  size_t numLines = 0;
  std::string line;
  while (std::getline(manifest, line)) {
    ++numLines;
    VELOX_USER_CHECK(
        !line.empty(),
        "Line {} of the manifest '{}' is empty. Every line names an object.",
        numLines,
        manifestPath.string());
    VELOX_USER_CHECK(
        line.compare(0, kS3Scheme.size(), kS3Scheme) == 0 ||
            line.compare(0, kS3aScheme.size(), kS3aScheme) == 0,
        "Line {} of the manifest '{}' is '{}', which is not an S3 URI. An I/O "
        "benchmark mode measures object storage, so every line must begin with "
        "'{}' or '{}'.",
        numLines,
        manifestPath.string(),
        line,
        kS3Scheme,
        kS3aScheme);
  }
  VELOX_USER_CHECK_GT(
      numLines,
      0,
      "The manifest '{}' for table '{}' lists no objects.",
      manifestPath.string(),
      table);
}

} // namespace

CudfTpchIoMode parseCudfTpchIoMode(std::string_view mode) {
  // Exhaustive switch over the four recognized string values.
  if (mode.empty()) {
    return CudfTpchIoMode::kDisabled;
  }
  if (mode == "decode_discard") {
    return CudfTpchIoMode::kDecodeDiscard;
  }
  if (mode == "raw_parquet_ranges") {
    return CudfTpchIoMode::kRawParquetRanges;
  }
  if (mode == "raw_file") {
    return CudfTpchIoMode::kRawFile;
  }
  VELOX_USER_FAIL(
      "Unknown cudf_io_mode '{}'. Valid values: '', 'decode_discard', "
      "'raw_parquet_ranges', 'raw_file'.",
      mode);
}

std::string normalizeS3Endpoint(std::string_view endpoint, bool sslEnabled) {
  // Trim ASCII whitespace from both ends.
  const auto ltrim = endpoint.find_first_not_of(" \t\r\n\f\v");
  VELOX_USER_CHECK_NE(
      ltrim,
      std::string_view::npos,
      "S3 endpoint must not be empty after trimming whitespace.");
  const auto rtrim = endpoint.find_last_not_of(" \t\r\n\f\v");
  std::string s(endpoint.substr(ltrim, rtrim - ltrim + 1));

  // Detect an explicit scheme (case-insensitive search for "://").
  std::string lower = s;
  std::transform(
      lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });

  const std::string_view http{"http://"};
  const std::string_view https{"https://"};

  bool hasScheme = false;
  std::string_view detectedScheme;
  if (lower.substr(0, https.size()) == https) {
    hasScheme = true;
    detectedScheme = https;
  } else if (lower.substr(0, http.size()) == http) {
    hasScheme = true;
    detectedScheme = http;
  }

  std::string result;
  if (hasScheme) {
    // Lowercase the scheme and copy the rest.
    result.reserve(s.size());
    result = std::string(detectedScheme); // already lowercase
    result += s.substr(detectedScheme.size());
  } else {
    const std::string_view scheme = sslEnabled ? https : http;
    result = std::string(scheme) + s;
  }

  // Lowercase only the authority (host + optional port), which is everything
  // between the scheme and the first '/' that follows it (or end of string).
  const auto schemeEnd = result.find("://");
  const auto authorityStart =
      (schemeEnd != std::string::npos) ? schemeEnd + 3 : 0;
  const auto pathStart = result.find('/', authorityStart);
  const auto authorityEnd =
      (pathStart != std::string::npos) ? pathStart : result.size();
  std::transform(
      result.begin() + static_cast<std::ptrdiff_t>(authorityStart),
      result.begin() + static_cast<std::ptrdiff_t>(authorityEnd),
      result.begin() + static_cast<std::ptrdiff_t>(authorityStart),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  // Strip trailing slashes.
  while (!result.empty() && result.back() == '/') {
    result.pop_back();
  }

  VELOX_USER_CHECK(
      !result.empty(),
      "S3 endpoint reduced to empty string after normalization.");
  return result;
}

void validateS3EndpointConsistency(
    const config::ConfigBase& connectorProperties,
    std::optional<std::string_view> kvikioEndpoint) {
  // Read connector endpoint and SSL flag.
  const auto connEndpointOpt =
      connectorProperties.get<std::string>("hive.s3.endpoint");
  const bool sslEnabled = connectorProperties.get<std::string>(
                              "hive.s3.ssl.enabled", "true") == "true";

  const bool hasConnector = connEndpointOpt.has_value();
  const bool hasKvikio = kvikioEndpoint.has_value();

  // No-op when neither side specifies a custom endpoint.
  if (!hasConnector && !hasKvikio) {
    return;
  }

  // Fail when exactly one side specifies an endpoint.
  VELOX_USER_CHECK(
      hasConnector && hasKvikio,
      "S3 endpoint mismatch: {} side specifies an endpoint but {} does not. "
      "Set both or neither.",
      hasConnector ? "connector" : "KvikIO",
      hasConnector ? "KvikIO" : "connector");

  const std::string connNorm =
      normalizeS3Endpoint(*connEndpointOpt, sslEnabled);
  const std::string kvikioNorm =
      normalizeS3Endpoint(*kvikioEndpoint, sslEnabled);

  VELOX_USER_CHECK_EQ(
      connNorm,
      kvikioNorm,
      "S3 endpoint mismatch: connector='{}' kvikio='{}'. "
      "Set both to the same endpoint.",
      connNorm,
      kvikioNorm);
}

void validateCudfTpchIoPreflight(
    const CudfTpchIoSettings& settings,
    const config::ConfigBase& connectorProperties,
    std::optional<std::string_view> kvikioEndpoint) {
  VELOX_CHECK(
      settings.mode != CudfTpchIoMode::kDisabled,
      "Only an enabled I/O benchmark mode needs a preflight check.");

  VELOX_USER_CHECK_EQ(
      settings.dataFormat,
      "parquet",
      "I/O benchmark modes require --data_format=parquet.");
  VELOX_USER_CHECK_EQ(
      settings.runQueryVerbose,
      -1,
      "I/O benchmark modes do not support --run_query_verbose.");
  VELOX_USER_CHECK_EQ(
      settings.ioMeterColumnPct,
      0,
      "I/O benchmark modes do not support --io_meter_column_pct.");
  VELOX_USER_CHECK_GT(
      settings.numDrivers, 0, "--num_drivers must be positive.");
  VELOX_USER_CHECK_GT(
      settings.numRepeats, 0, "--num_repeats must be positive.");
  VELOX_USER_CHECK_GT(
      settings.readSizeBytes, 0, "--cudf_io_read_size_bytes must be positive.");

  const auto& tableNames = exec::test::TpchQueryBuilder::getTableNames();
  VELOX_USER_CHECK(
      std::find(tableNames.begin(), tableNames.end(), settings.table) !=
          tableNames.end(),
      "--cudf_io_table='{}' is not a TPC-H table. Valid tables: {}.",
      settings.table,
      folly::join(", ", tableNames));

  // Without the cuDF table scan the decode mode would silently run on the CPU
  // Hive connector, which publishes none of the payload counters the mode
  // reports, and the raw modes would measure a transport the run is not
  // configured for.
  VELOX_USER_CHECK(
      settings.cudfTableScan,
      "I/O benchmark modes require --velox_cudf_table_scan=true, because they "
      "measure the direct cuDF/KvikIO read path.");

  validateS3EndpointConsistency(connectorProperties, kvikioEndpoint);

  validateManifest(settings.dataPath, settings.table);
}

} // namespace facebook::velox::cudf_velox
