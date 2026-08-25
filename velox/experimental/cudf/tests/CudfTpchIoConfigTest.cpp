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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TempDirectoryPath.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace facebook::velox::cudf_velox;

// ---- parseCudfTpchIoMode tests ----

TEST(CudfTpchIoConfigTest, parseModeDisabled) {
  EXPECT_EQ(parseCudfTpchIoMode(""), CudfTpchIoMode::kDisabled);
}

TEST(CudfTpchIoConfigTest, parseModeDecodeDiscard) {
  EXPECT_EQ(
      parseCudfTpchIoMode("decode_discard"), CudfTpchIoMode::kDecodeDiscard);
}

TEST(CudfTpchIoConfigTest, parseModeRawParquetRanges) {
  EXPECT_EQ(
      parseCudfTpchIoMode("raw_parquet_ranges"),
      CudfTpchIoMode::kRawParquetRanges);
}

TEST(CudfTpchIoConfigTest, parseModeRawFile) {
  EXPECT_EQ(parseCudfTpchIoMode("raw_file"), CudfTpchIoMode::kRawFile);
}

TEST(CudfTpchIoConfigTest, parseModeInvalidContainsInput) {
  EXPECT_THROW(
      {
        try {
          parseCudfTpchIoMode("bogus_mode");
        } catch (const std::exception& e) {
          EXPECT_NE(std::string(e.what()).find("bogus_mode"), std::string::npos)
              << "Error message should contain the bad mode value";
          throw;
        }
      },
      std::exception);
}

// ---- normalizeS3Endpoint tests ----

TEST(CudfTpchIoConfigTest, normalizeHttpEndpointNoTrailingSlash) {
  // Plain http MinIO endpoint, no trailing slash.
  EXPECT_EQ(
      normalizeS3Endpoint("http://minio:9000", /*sslEnabled=*/false),
      "http://minio:9000");
}

TEST(CudfTpchIoConfigTest, normalizeHttpEndpointWithTrailingSlash) {
  // Trailing slash must be stripped.
  EXPECT_EQ(
      normalizeS3Endpoint("http://minio:9000/", /*sslEnabled=*/false),
      "http://minio:9000");
}

TEST(CudfTpchIoConfigTest, normalizeHttpsEndpointNoScheme) {
  // When SSL is true and no scheme is present, https:// must be prepended.
  EXPECT_EQ(
      normalizeS3Endpoint("s3.amazonaws.com", /*sslEnabled=*/true),
      "https://s3.amazonaws.com");
}

TEST(CudfTpchIoConfigTest, normalizeNoSchemeNoSsl) {
  // When SSL is false and no scheme is present, http:// must be prepended.
  EXPECT_EQ(
      normalizeS3Endpoint("minio:9000", /*sslEnabled=*/false),
      "http://minio:9000");
}

TEST(CudfTpchIoConfigTest, normalizeLowercasesSchemeAndHost) {
  // Scheme and host must be lowercased; path must be preserved.
  EXPECT_EQ(
      normalizeS3Endpoint("HTTP://MINIO:9000/MyBucket", /*sslEnabled=*/false),
      "http://minio:9000/MyBucket");
}

TEST(CudfTpchIoConfigTest, normalizeRejectsEmptyEndpoint) {
  EXPECT_THROW(normalizeS3Endpoint("", /*sslEnabled=*/false), std::exception);
  EXPECT_THROW(
      normalizeS3Endpoint("   ", /*sslEnabled=*/false), std::exception);
}

// ---- validateS3EndpointConsistency tests ----

TEST(CudfTpchIoConfigTest, validateBothAbsentIsNoOp) {
  // When neither connector nor KvikIO specifies an endpoint, no error.
  facebook::velox::config::ConfigBase props(
      std::unordered_map<std::string, std::string>{});
  EXPECT_NO_THROW(validateS3EndpointConsistency(props, std::nullopt));
}

TEST(CudfTpchIoConfigTest, validateOnlyConnectorEndpointRejects) {
  // Connector has endpoint but KvikIO does not: error.
  facebook::velox::config::ConfigBase props(
      std::unordered_map<std::string, std::string>{
          {"hive.s3.endpoint", "http://minio:9000"},
      });
  EXPECT_THROW(
      validateS3EndpointConsistency(props, std::nullopt), std::exception);
}

TEST(CudfTpchIoConfigTest, validateOnlyKvikioEndpointRejects) {
  // KvikIO has endpoint but connector does not: error.
  facebook::velox::config::ConfigBase props(
      std::unordered_map<std::string, std::string>{});
  EXPECT_THROW(
      validateS3EndpointConsistency(
          props, std::string_view{"http://minio:9000"}),
      std::exception);
}

TEST(CudfTpchIoConfigTest, validateUnequalEndpointsReject) {
  // Connector and KvikIO have different endpoints: error.
  facebook::velox::config::ConfigBase props(
      std::unordered_map<std::string, std::string>{
          {"hive.s3.endpoint", "http://minio:9000"},
          {"hive.s3.ssl.enabled", "false"},
      });
  EXPECT_THROW(
      validateS3EndpointConsistency(
          props, std::string_view{"http://other:9000"}),
      std::exception);
}

TEST(CudfTpchIoConfigTest, validateEqualNormalizedEndpointsAccepted) {
  // Connector and KvikIO specify the same endpoint (possibly with trailing
  // slash on one): no error.
  facebook::velox::config::ConfigBase props(
      std::unordered_map<std::string, std::string>{
          {"hive.s3.endpoint", "http://minio:9000/"},
          {"hive.s3.ssl.enabled", "false"},
      });
  EXPECT_NO_THROW(validateS3EndpointConsistency(
      props, std::string_view{"http://minio:9000"}));
}

// ---- validateCudfTpchIoPreflight tests ----

namespace {

using ::facebook::velox::common::testutil::TempDirectoryPath;

class CudfTpchIoPreflightTest : public testing::Test {
 protected:
  // The table every test configures, chosen because it is canonical.
  static constexpr const char* kTable = "lineitem";

  void SetUp() override {
    dataPath_ = TempDirectoryPath::create();
    writeManifest({"s3://bucket/tpch/lineitem/part-00000.parquet"});
  }

  // Replaces the manifest with one holding 'lines', each on its own line.
  void writeManifest(const std::vector<std::string>& lines) {
    std::ofstream manifest(manifestPath());
    for (const auto& line : lines) {
      manifest << line << "\n";
    }
    manifest.close();
    ASSERT_TRUE(std::filesystem::exists(manifestPath()));
  }

  std::string manifestPath() const {
    return dataPath_->getPath() + "/" + kTable;
  }

  // Settings a valid decode-discard run would carry. Each test makes exactly
  // one of them wrong, so a failure names the setting the test is about.
  CudfTpchIoSettings validSettings() const {
    return CudfTpchIoSettings{
        .mode = CudfTpchIoMode::kDecodeDiscard,
        .dataFormat = "parquet",
        .dataPath = dataPath_->getPath(),
        .table = kTable,
        .runQueryVerbose = -1,
        .ioMeterColumnPct = 0,
        .numDrivers = 4,
        .numRepeats = 1,
        .readSizeBytes = 128 << 20,
        .cudfTableScan = true,
    };
  }

  // Connector properties that name no endpoint, matching an unset
  // AWS_ENDPOINT_URL.
  static const facebook::velox::config::ConfigBase& noEndpointProperties() {
    static const facebook::velox::config::ConfigBase properties(
        std::unordered_map<std::string, std::string>{});
    return properties;
  }

  // Runs the preflight over 'settings' with matching empty endpoints.
  static void preflight(const CudfTpchIoSettings& settings) {
    validateCudfTpchIoPreflight(settings, noEndpointProperties(), std::nullopt);
  }

  std::shared_ptr<TempDirectoryPath> dataPath_;
};

} // namespace

TEST_F(CudfTpchIoPreflightTest, acceptsAValidConfiguration) {
  EXPECT_NO_THROW(preflight(validSettings()));
}

// Every enabled mode is checked the same way, so none of them can slip past
// with a configuration another mode would be rejected for.
TEST_F(CudfTpchIoPreflightTest, checksEveryEnabledMode) {
  for (const auto mode :
       {CudfTpchIoMode::kDecodeDiscard,
        CudfTpchIoMode::kRawParquetRanges,
        CudfTpchIoMode::kRawFile}) {
    auto settings = validSettings();
    settings.mode = mode;
    EXPECT_NO_THROW(preflight(settings));

    settings.cudfTableScan = false;
    VELOX_ASSERT_THROW(preflight(settings), "--velox_cudf_table_scan=true");
  }
}

TEST_F(CudfTpchIoPreflightTest, rejectsANonParquetFormat) {
  auto settings = validSettings();
  settings.dataFormat = "dwrf";

  VELOX_ASSERT_THROW(preflight(settings), "--data_format=parquet");
}

TEST_F(CudfTpchIoPreflightTest, rejectsRunQueryVerbose) {
  auto settings = validSettings();
  settings.runQueryVerbose = 6;

  VELOX_ASSERT_THROW(preflight(settings), "--run_query_verbose");
}

TEST_F(CudfTpchIoPreflightTest, rejectsIoMeterColumnPct) {
  auto settings = validSettings();
  settings.ioMeterColumnPct = 50;

  VELOX_ASSERT_THROW(preflight(settings), "--io_meter_column_pct");
}

TEST_F(CudfTpchIoPreflightTest, rejectsNonPositiveDriversRepeatsAndReadSize) {
  auto drivers = validSettings();
  drivers.numDrivers = 0;
  VELOX_ASSERT_THROW(preflight(drivers), "--num_drivers must be positive.");

  auto repeats = validSettings();
  repeats.numRepeats = 0;
  VELOX_ASSERT_THROW(preflight(repeats), "--num_repeats must be positive.");

  auto readSize = validSettings();
  readSize.readSizeBytes = 0;
  VELOX_ASSERT_THROW(
      preflight(readSize), "--cudf_io_read_size_bytes must be positive.");
}

TEST_F(CudfTpchIoPreflightTest, rejectsATableThatIsNotATpchTable) {
  auto settings = validSettings();
  settings.table = "not_a_tpch_table";

  VELOX_ASSERT_THROW(preflight(settings), "is not a TPC-H table");
}

// The decode mode would otherwise run on the CPU Hive connector and fail only
// later, when the KvikIO payload counters it reports turned out to be absent.
TEST_F(CudfTpchIoPreflightTest, rejectsTheCpuTableScan) {
  auto settings = validSettings();
  settings.cudfTableScan = false;

  VELOX_ASSERT_THROW(
      preflight(settings), "require --velox_cudf_table_scan=true");
}

TEST_F(CudfTpchIoPreflightTest, rejectsMismatchedEndpoints) {
  const facebook::velox::config::ConfigBase properties(
      std::unordered_map<std::string, std::string>{
          {"hive.s3.endpoint", "minio:9000"},
          {"hive.s3.ssl.enabled", "false"},
      });

  VELOX_ASSERT_THROW(
      validateCudfTpchIoPreflight(
          validSettings(), properties, std::string_view{"http://other:9000"}),
      "S3 endpoint mismatch");
}

TEST_F(CudfTpchIoPreflightTest, acceptsMatchingEndpoints) {
  const facebook::velox::config::ConfigBase properties(
      std::unordered_map<std::string, std::string>{
          {"hive.s3.endpoint", "minio:9000"},
          {"hive.s3.ssl.enabled", "false"},
      });

  EXPECT_NO_THROW(validateCudfTpchIoPreflight(
      validSettings(), properties, std::string_view{"http://minio:9000"}));
}

// A directory is the layout the ordinary TPC-H benchmark reads, and the one an
// I/O mode cannot use: it would list local files rather than object URIs.
TEST_F(CudfTpchIoPreflightTest, rejectsADirectoryInsteadOfAManifest) {
  ASSERT_TRUE(std::filesystem::remove(manifestPath()));
  ASSERT_TRUE(std::filesystem::create_directory(manifestPath()));

  VELOX_ASSERT_THROW(preflight(validSettings()), "is a directory");
}

TEST_F(CudfTpchIoPreflightTest, rejectsAMissingManifest) {
  ASSERT_TRUE(std::filesystem::remove(manifestPath()));

  VELOX_ASSERT_THROW(preflight(validSettings()), "Cannot read the manifest");
}

TEST_F(CudfTpchIoPreflightTest, rejectsAnEmptyManifest) {
  writeManifest({});

  VELOX_ASSERT_THROW(preflight(validSettings()), "lists no objects");
}

// The query builder takes every line as a URI, so a blank line would become an
// object named "" rather than being skipped.
TEST_F(CudfTpchIoPreflightTest, rejectsABlankManifestLine) {
  writeManifest({"s3://bucket/a.parquet", "", "s3://bucket/b.parquet"});

  VELOX_ASSERT_THROW(preflight(validSettings()), "Line 2 of the manifest");
}

TEST_F(CudfTpchIoPreflightTest, rejectsAWhitespaceOnlyManifestLine) {
  writeManifest({"s3://bucket/a.parquet", "   "});

  VELOX_ASSERT_THROW(preflight(validSettings()), "is not an S3 URI");
}

TEST_F(CudfTpchIoPreflightTest, rejectsALocalPathInTheManifest) {
  writeManifest({"/tmp/tpch/lineitem/part-00000.parquet"});

  VELOX_ASSERT_THROW(preflight(validSettings()), "is not an S3 URI");
}

TEST_F(CudfTpchIoPreflightTest, acceptsBothS3Schemes) {
  writeManifest({"s3://bucket/a.parquet", "s3a://bucket/b.parquet"});

  EXPECT_NO_THROW(preflight(validSettings()));
}

// Nothing here opens an object: the URIs name a bucket no file system in this
// process serves, and the preflight still succeeds. That is what lets it run
// before any file system is registered.
TEST_F(CudfTpchIoPreflightTest, readsOnlyTheManifestAndNotTheObjects) {
  writeManifest({"s3://no-such-bucket/no-such-object.parquet"});

  EXPECT_NO_THROW(preflight(validSettings()));
}
