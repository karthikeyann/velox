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

// Drives the whole cuDF TPC-H I/O benchmark against a MinIO server through the
// entry point the executable uses, so what these tests cover is the public
// orchestration rather than the components underneath it.

// S3Test.h has to come first, which is why it is kept out of the sorted list
// below. It reaches duckdb's headers while MinioServer.h has already opened
// facebook::velox at global scope. Once facebook::velox::duckdb also exists,
// which the benchmark header brings in through PlanBuilder, every "duckdb::"
// name inside duckdb's own headers becomes ambiguous, so duckdb has to be
// parsed before that namespace is declared.
// clang-format off
#include "velox/connectors/hive/storage_adapters/s3fs/tests/S3Test.h"
// clang-format on

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Config.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/tpch/gen/TpchGen.h"
#include "velox/vector/tests/utils/VectorMaker.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <folly/Conv.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

DECLARE_string(data_path);
DECLARE_string(data_format);
DECLARE_string(test_flags_file);
DECLARE_int32(num_drivers);
DECLARE_int32(num_repeats);
DECLARE_int32(run_query_verbose);
DECLARE_int32(io_meter_column_pct);
DECLARE_string(cudf_io_mode);
DECLARE_string(cudf_io_table);
DECLARE_int64(cudf_io_read_size_bytes);
DECLARE_string(connector_properties);
DECLARE_bool(velox_cudf_table_scan);

using namespace facebook::velox;

namespace {

namespace fs = std::filesystem;

using ::facebook::velox::common::testutil::TempDirectoryPath;
using ::facebook::velox::exec::test::kHiveConnectorId;

// Rows written into the region object. Small enough to keep every transfer
// trivial, large enough for the writer to emit real column chunks.
constexpr int32_t kNumRegionRows = 2'000;

// Bound on each whole-object read. Small enough that the object needs several
// of them, so the reported chunk count is worth asserting.
constexpr int64_t kReadSizeBytes = 4096;

// Saves the environment variables KvikIO reads and puts them back when it goes
// out of scope. KvikIO resolves the endpoint, region and credentials from the
// environment on every open and never sees the connector properties, so a test
// that points it at its own server has to take the previous values back once
// that server is gone.
class ScopedEnvironment {
 public:
  ScopedEnvironment() = default;
  ScopedEnvironment(const ScopedEnvironment&) = delete;
  ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

  ~ScopedEnvironment() {
    // Reverse order, so a name that was set more than once regains the value
    // it held before the first set().
    for (auto entry = saved_.rbegin(); entry != saved_.rend(); ++entry) {
      if (entry->second.has_value()) {
        ::setenv(entry->first.c_str(), entry->second->c_str(), 1);
      } else {
        ::unsetenv(entry->first.c_str());
      }
    }
  }

  /// Sets 'name' to 'value', remembering whatever it held before. The value is
  /// never reported, because some of these names carry credentials.
  void set(const std::string& name, const std::string& value) {
    const char* previous = ::getenv(name.c_str());
    saved_.emplace_back(
        name,
        previous == nullptr ? std::optional<std::string>{}
                            : std::optional<std::string>{previous});
    EXPECT_EQ(::setenv(name.c_str(), value.c_str(), 1), 0)
        << "Failed to set the environment variable " << name;
  }

 private:
  std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

// Collects what a run writes to std::cout, which is where the benchmark's
// entry point reports its results.
class ScopedCoutCapture {
 public:
  ScopedCoutCapture() : previous_(std::cout.rdbuf(captured_.rdbuf())) {}

  ScopedCoutCapture(const ScopedCoutCapture&) = delete;
  ScopedCoutCapture& operator=(const ScopedCoutCapture&) = delete;

  ~ScopedCoutCapture() {
    std::cout.rdbuf(previous_);
  }

  std::string text() const {
    return captured_.str();
  }

 private:
  std::ostringstream captured_;
  std::streambuf* const previous_;
};

// One result line, split into its "name=value" fields.
using ResultFields = std::unordered_map<std::string, std::string>;

// Fields of the last result line in 'output'. The note line each mode prints
// first, and anything glog interleaves, do not start with "mode=".
ResultFields resultFields(const std::string& output) {
  ResultFields fields;
  std::istringstream lines(output);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.rfind("mode=", 0) != 0) {
      continue;
    }
    fields.clear();
    std::istringstream tokens(line);
    std::string token;
    while (tokens >> token) {
      const auto separator = token.find('=');
      if (separator != std::string::npos) {
        fields.emplace(token.substr(0, separator), token.substr(separator + 1));
      }
    }
  }
  return fields;
}

// Numeric value of the 'name' field. A field the run failed to report is the
// failure these tests guard against, so it is looked up rather than assumed.
uint64_t numericField(const ResultFields& fields, const std::string& name) {
  const auto field = fields.find(name);
  EXPECT_NE(field, fields.end()) << "the run did not report " << name;
  if (field == fields.end()) {
    return 0;
  }
  return folly::to<uint64_t>(field->second);
}

class CudfTpchBenchmarkS3Test : public S3Test {
 protected:
  // The canonical TPC-H table the object holds. Region is the narrowest one,
  // so its schema is the cheapest valid one to store.
  static constexpr const char* kTable = "region";

  // Bucket and key the object is served under.
  static constexpr const char* kBucketName = "tpch";
  static constexpr const char* kObjectKey = "region/part-00000.parquet";

  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    filesystems::registerLocalFileSystem();
    // Registered for the whole suite rather than by each benchmark, so every
    // test also stands as evidence that a benchmark leaves an S3 file system
    // it did not register alone: finalizing is one-shot, and doing it here
    // would break every later test in this process.
    filesystems::registerS3FileSystem();

    sourceDirectory_ = TempDirectoryPath::create();
    writeRegionObject(sourceObjectPath());
  }

  static void TearDownTestSuite() {
    sourceDirectory_.reset();
    filesystems::finalizeS3FileSystem();
  }

  void SetUp() override {
    S3Test::SetUp();

    uploadRegionObject();
    exportKvikioEnvironment();

    manifestDirectory_ = TempDirectoryPath::create();
    writeManifest({objectUri()});
    writeConnectorProperties();
    setValidFlags();
  }

  void TearDown() override {
    // The benchmark is process-global and owns what it was given, so no test
    // may leave one behind for the next.
    benchmark.reset();

    EXPECT_EQ(connector::ConnectorRegistry::tryGet(kHiveConnectorId), nullptr)
        << "a test left a connector registered";
    EXPECT_FALSE(cudf_velox::cudfIsRegistered())
        << "a test left cuDF registered";

    manifestDirectory_.reset();
    S3Test::TearDown();
    // Every connector, task and raw runner that could still hold a KvikIO
    // handle is gone, so the endpoint this test published can be taken back.
    kvikioEnvironment_.reset();
    flagSaver_.reset();
  }

  // S3 URI the uploaded object is served under.
  static std::string objectUri() {
    return filesystems::s3URI(kBucketName, kObjectKey);
  }

  // Size of the object, measured on the local copy so that no reader under
  // test is the source of the expectation.
  static uint64_t objectSize() {
    return fs::file_size(sourceObjectPath());
  }

  // Replaces the manifest with one listing 'uris', one per line.
  void writeManifest(const std::vector<std::string>& uris) {
    std::ofstream manifest(manifestPath());
    for (const auto& uri : uris) {
      manifest << uri << "\n";
    }
    manifest.close();
    EXPECT_TRUE(fs::exists(manifestPath()));
  }

  std::string manifestPath() const {
    return manifestDirectory_->getPath() + "/" + kTable;
  }

  // Runs the benchmark the way the executable's main does: initialize, run,
  // shut down, with the failure handling that entry point provides. Returns
  // what the run printed.
  std::string runBenchmark() {
    ScopedCoutCapture captured;
    benchmark = std::make_unique<CudfTpchBenchmark>();
    tpchBenchmarkMain();
    benchmark.reset();
    return captured.text();
  }

  // Runs the benchmark expecting it to fail, and returns the message. The
  // benchmark is dropped either way, so a failing assertion cannot strand one.
  std::string runBenchmarkExpectingFailure() {
    std::string message;
    {
      ScopedCoutCapture captured;
      benchmark = std::make_unique<CudfTpchBenchmark>();
      try {
        tpchBenchmarkMain();
        ADD_FAILURE() << "the run should have failed";
      } catch (const VeloxException& e) {
        message = e.message();
      } catch (const std::exception& e) {
        message = e.what();
      }
    }
    benchmark.reset();
    return message;
  }

  // Connector configuration of this test's MinIO server.
  std::shared_ptr<const config::ConfigBase> hiveConfig() const {
    return minioServer_->hiveConfig();
  }

  // Points every flag the benchmark reads at this test's server, manifest and
  // object, in the configuration a decode-discard run uses.
  void setValidFlags() {
    flagSaver_.emplace();
    FLAGS_test_flags_file = "";
    FLAGS_data_path = manifestDirectory_->getPath();
    FLAGS_data_format = "parquet";
    FLAGS_cudf_io_mode = "decode_discard";
    FLAGS_cudf_io_table = kTable;
    FLAGS_cudf_io_read_size_bytes = kReadSizeBytes;
    FLAGS_connector_properties = connectorPropertiesPath();
    FLAGS_velox_cudf_table_scan = true;
    FLAGS_num_drivers = 1;
    FLAGS_num_repeats = 1;
    FLAGS_run_query_verbose = -1;
    FLAGS_io_meter_column_pct = 0;
  }

  std::string connectorPropertiesPath() const {
    return manifestDirectory_->getPath() + "/connector.properties";
  }

  // A Hive connector registered by something other than the benchmark, so a
  // test can show what the benchmark does and does not do to one.
  std::shared_ptr<connector::Connector> registerForeignConnector() {
    connector::hive::HiveConnectorFactory factory;
    auto foreign = factory.newConnector(kHiveConnectorId, hiveConfig());
    connector::ConnectorRegistry::global().insert(
        foreign->connectorId(), foreign);
    return foreign;
  }

  std::optional<ScopedEnvironment> kvikioEnvironment_;

 private:
  // Writes the local copy of the object, once for the whole suite. Every
  // memory pool it needs is released before it returns, because a benchmark
  // replaces the process memory manager when it initializes and a pool
  // outliving that swap would fail the manager's leak check.
  static void writeRegionObject(const std::string& path) {
    const auto rowType = tpch::getTableSchema(tpch::Table::TBL_REGION);
    auto rootPool = memory::memoryManager()->addRootPool("regionObjectWriter");
    auto leafPool = rootPool->addLeafChild("leaf");
    facebook::velox::test::VectorMaker vectorMaker(leafPool.get());

    auto data = vectorMaker.rowVector(
        rowType->names(),
        {vectorMaker.flatVector<int64_t>(
             kNumRegionRows,
             [](vector_size_t row) { return static_cast<int64_t>(row); }),
         vectorMaker.flatVector<std::string>(
             kNumRegionRows,
             [](vector_size_t row) {
               return fmt::format("REGION{}", row % 5);
             }),
         vectorMaker.flatVector<std::string>(
             kNumRegionRows, [](vector_size_t row) {
               return fmt::format("comment for region row {}", row);
             })});

    auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
    auto table = cudf_velox::with_arrow::toCudfTable(
        data, leafPool.get(), stream, cudf_velox::get_temp_mr());
    stream.synchronize();

    cudf::io::table_input_metadata metadata(table->view());
    ASSERT_EQ(metadata.column_metadata.size(), rowType->size());
    for (size_t column = 0; column < rowType->size(); ++column) {
      metadata.column_metadata[column].set_name(rowType->nameOf(column));
    }
    cudf::io::write_parquet(
        cudf::io::parquet_writer_options::builder(
            cudf::io::sink_info(path), table->view())
            .metadata(metadata)
            .build(),
        stream);
    stream.synchronize();

    ASSERT_GT(fs::file_size(path), 0);
  }

  static std::string sourceObjectPath() {
    return sourceDirectory_->getPath() + "/region.parquet";
  }

  // Copies the object into this test's bucket and waits until the server
  // serves it.
  void uploadRegionObject() {
    minioServer_->addBucket(kBucketName);
    const std::string destination =
        S3Test::localPath(kBucketName) + "/" + kObjectKey;
    fs::create_directories(fs::path(destination).parent_path());
    fs::copy_file(
        sourceObjectPath(), destination, fs::copy_options::overwrite_existing);
    ASSERT_EQ(fs::file_size(destination), objectSize());
    waitUntilServed(objectUri());
  }

  // Blocks until this test's server answers a read of 's3Uri'.
  //
  // MinioServer::start() returns as soon as the child process is spawned, so
  // the server is usually still binding its port. KvikIO does not retry a
  // refused connection: it treats the object as unreachable and silently falls
  // back to an unauthenticated public-S3 endpoint, whose s3:// URL libcurl
  // then rejects as an unsupported scheme. Every KvikIO read here therefore
  // has to find the server already serving. The S3 client used to wait is
  // deliberately not the one under test; it only has to answer whether the
  // object can be read yet.
  void waitUntilServed(const std::string& s3Uri) {
    constexpr auto kTimeout = std::chrono::seconds(60);
    constexpr auto kPollInterval = std::chrono::milliseconds(50);
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    do {
      try {
        // Held only for this attempt, so that finalizing S3 at the end of the
        // suite does not find the cached file system still in use.
        auto fileSystem = filesystems::getFileSystem(s3Uri, hiveConfig());
        if (fileSystem->openFileForRead(s3Uri)->size() > 0) {
          return;
        }
      } catch (const std::exception&) {
        // The server is not up yet, or the object has not appeared yet. The
        // reason is deliberately not reported: S3 client errors can echo
        // request credentials.
      }
      std::this_thread::sleep_for(kPollInterval);
    } while (std::chrono::steady_clock::now() < deadline);
    FAIL() << "The S3 server did not serve " << s3Uri << " within "
           << kTimeout.count() << "s";
  }

  // Publishes this test's endpoint and credentials under the names KvikIO
  // reads.
  void exportKvikioEnvironment() {
    const filesystems::S3Config s3(kBucketName, hiveConfig());
    ASSERT_TRUE(s3.endpoint().has_value());
    ASSERT_TRUE(s3.accessKey().has_value());
    ASSERT_TRUE(s3.secretKey().has_value());

    kvikioEnvironment_.emplace();
    // KvikIO needs an explicit scheme, while the connector accepts a bare
    // host:port. Deriving one from the other is what makes hive.s3.endpoint
    // and AWS_ENDPOINT_URL resolve to the same endpoint.
    kvikioEnvironment_->set(
        "AWS_ENDPOINT_URL",
        (s3.useSSL() ? "https://" : "http://") + s3.endpoint().value());
    kvikioEnvironment_->set("AWS_ACCESS_KEY_ID", s3.accessKey().value());
    kvikioEnvironment_->set("AWS_SECRET_ACCESS_KEY", s3.secretKey().value());
    // MinIO ignores the region but KvikIO refuses to sign a request without
    // one.
    kvikioEnvironment_->set("AWS_DEFAULT_REGION", "us-east-1");
    kvikioEnvironment_->set("AWS_EC2_METADATA_DISABLED", "true");
    // An exported endpoint already implies path-style access on the KvikIO
    // side; the connector properties written below ask for the same.
    ASSERT_FALSE(s3.useVirtualAddressing());
  }

  void writeConnectorProperties() {
    // Held for the loop: the raw configs are a reference into the config, so
    // iterating over a temporary one would outlive what it reads.
    const auto config = hiveConfig();
    std::ofstream properties(connectorPropertiesPath());
    for (const auto& [key, value] : config->rawConfigs()) {
      properties << key << "=" << value << "\n";
    }
    properties.close();
    EXPECT_GT(fs::file_size(connectorPropertiesPath()), 0);
  }

  static std::shared_ptr<TempDirectoryPath> sourceDirectory_;

  std::shared_ptr<TempDirectoryPath> manifestDirectory_;
  std::optional<gflags::FlagSaver> flagSaver_;
};

std::shared_ptr<TempDirectoryPath> CudfTpchBenchmarkS3Test::sourceDirectory_ =
    nullptr;

// ---------------- The three modes over the public entry point -------------

TEST_F(CudfTpchBenchmarkS3Test, decodeDiscardReportsExplicitByteFields) {
  const auto fields = resultFields(runBenchmark());

  ASSERT_FALSE(fields.empty()) << "the run reported no result line";
  EXPECT_EQ(fields.at("mode"), "decode_discard");
  EXPECT_EQ(fields.at("table"), kTable);
  EXPECT_EQ(numericField(fields, "files"), 1);
  EXPECT_EQ(numericField(fields, "decoded_rows"), kNumRegionRows);
  EXPECT_GT(numericField(fields, "decoded_bytes"), 0);

  // Requested and completed are reported apart, so a short read would be
  // visible rather than folded into one number.
  const auto requested = numericField(fields, "compressed_requested_bytes");
  const auto completed = numericField(fields, "compressed_completed_bytes");
  EXPECT_GT(requested, 0);
  EXPECT_EQ(requested, completed);
  EXPECT_EQ(fields.count("compressed_completed_bytes_per_s"), 1);
  // The compressed payload is a subset of the object, which also holds the
  // footer, page index and file header.
  EXPECT_LT(completed, objectSize());
}

TEST_F(CudfTpchBenchmarkS3Test, rawParquetRangesMatchTheDecodedPayload) {
  const auto decoded = resultFields(runBenchmark());
  const auto decodedRequested =
      numericField(decoded, "compressed_requested_bytes");

  FLAGS_cudf_io_mode = "raw_parquet_ranges";
  const auto raw = resultFields(runBenchmark());

  ASSERT_FALSE(raw.empty()) << "the run reported no result line";
  EXPECT_EQ(raw.at("mode"), "raw_parquet_ranges");
  EXPECT_GT(numericField(raw, "selected_row_groups"), 0);
  EXPECT_EQ(raw.count("column_chunk_logical_ranges"), 1);
  EXPECT_EQ(raw.count("column_chunk_physical_requests"), 1);
  EXPECT_EQ(raw.count("completed_bytes_per_s"), 1);

  const auto rawRequested = numericField(raw, "requested_bytes");
  EXPECT_EQ(rawRequested, numericField(raw, "completed_bytes"));
  // Both modes select the same column chunks of the same object, so any
  // difference means the raw options and the scan options diverged.
  EXPECT_EQ(rawRequested, decodedRequested);
}

TEST_F(CudfTpchBenchmarkS3Test, rawFileReadsEveryByteOfTheObject) {
  FLAGS_cudf_io_mode = "raw_file";
  const auto fields = resultFields(runBenchmark());

  ASSERT_FALSE(fields.empty()) << "the run reported no result line";
  EXPECT_EQ(fields.at("mode"), "raw_file");
  EXPECT_EQ(numericField(fields, "read_size_bytes"), kReadSizeBytes);
  EXPECT_EQ(numericField(fields, "requested_bytes"), objectSize());
  EXPECT_EQ(numericField(fields, "completed_bytes"), objectSize());
  EXPECT_EQ(
      numericField(fields, "read_chunks"),
      (objectSize() + kReadSizeBytes - 1) / kReadSizeBytes);
  EXPECT_EQ(fields.count("physical_requests"), 1);
  EXPECT_EQ(fields.count("completed_bytes_per_s"), 1);
}

// ---------------- Preflight over the public entry point ----------------

// The preflight has to reject the run before the base initialization reads the
// first object's schema, so nothing is registered and no request is made.
TEST_F(CudfTpchBenchmarkS3Test, endpointMismatchFailsBeforeAnythingIsTaken) {
  kvikioEnvironment_->set("AWS_ENDPOINT_URL", "http://not-this-server:1");

  const auto message = runBenchmarkExpectingFailure();

  EXPECT_NE(message.find("S3 endpoint mismatch"), std::string::npos) << message;
  EXPECT_EQ(connector::ConnectorRegistry::tryGet(kHiveConnectorId), nullptr);
  EXPECT_FALSE(cudf_velox::cudfIsRegistered());
}

TEST_F(CudfTpchBenchmarkS3Test, cpuTableScanFailsBeforeAnythingIsTaken) {
  FLAGS_velox_cudf_table_scan = false;

  const auto message = runBenchmarkExpectingFailure();

  EXPECT_NE(message.find("--velox_cudf_table_scan=true"), std::string::npos)
      << message;
  EXPECT_EQ(connector::ConnectorRegistry::tryGet(kHiveConnectorId), nullptr);
  EXPECT_FALSE(cudf_velox::cudfIsRegistered());
}

TEST_F(CudfTpchBenchmarkS3Test, aNonParquetFormatFailsBeforeAnythingIsTaken) {
  FLAGS_data_format = "dwrf";

  const auto message = runBenchmarkExpectingFailure();

  EXPECT_NE(message.find("--data_format=parquet"), std::string::npos)
      << message;
  EXPECT_EQ(connector::ConnectorRegistry::tryGet(kHiveConnectorId), nullptr);
}

TEST_F(CudfTpchBenchmarkS3Test, anUnknownTableFailsBeforeAnythingIsTaken) {
  FLAGS_cudf_io_table = "not_a_tpch_table";

  const auto message = runBenchmarkExpectingFailure();

  EXPECT_NE(message.find("is not a TPC-H table"), std::string::npos) << message;
  EXPECT_EQ(connector::ConnectorRegistry::tryGet(kHiveConnectorId), nullptr);
}

TEST_F(CudfTpchBenchmarkS3Test, aQueryOnlyFlagFailsBeforeAnythingIsTaken) {
  FLAGS_run_query_verbose = 6;

  const auto message = runBenchmarkExpectingFailure();

  EXPECT_NE(message.find("--run_query_verbose"), std::string::npos) << message;
  EXPECT_EQ(connector::ConnectorRegistry::tryGet(kHiveConnectorId), nullptr);
}

// A directory under --data_path is the layout the ordinary TPC-H benchmark
// reads. An I/O mode has to reject it rather than measure whatever local files
// it finds there.
TEST_F(CudfTpchBenchmarkS3Test, aDirectoryLayoutFailsBeforeAnythingIsTaken) {
  ASSERT_TRUE(fs::remove(manifestPath()));
  ASSERT_TRUE(fs::create_directory(manifestPath()));

  const auto message = runBenchmarkExpectingFailure();

  EXPECT_NE(message.find("is a directory"), std::string::npos) << message;
  EXPECT_EQ(connector::ConnectorRegistry::tryGet(kHiveConnectorId), nullptr);
}

TEST_F(CudfTpchBenchmarkS3Test, aLocalUriInTheManifestFails) {
  writeManifest({"/tmp/region/part-00000.parquet"});

  const auto message = runBenchmarkExpectingFailure();

  EXPECT_NE(message.find("is not an S3 URI"), std::string::npos) << message;
}

TEST_F(CudfTpchBenchmarkS3Test, anEmptyManifestFails) {
  writeManifest({});

  const auto message = runBenchmarkExpectingFailure();

  EXPECT_NE(message.find("lists no objects"), std::string::npos) << message;
}

TEST_F(CudfTpchBenchmarkS3Test, aBlankManifestLineFails) {
  writeManifest({objectUri(), ""});

  const auto message = runBenchmarkExpectingFailure();

  EXPECT_NE(message.find("Line 2 of the manifest"), std::string::npos)
      << message;
}

// ---------------- Remote failures name the URI the caller gave ----------

// KvikIO reports its own URL, and its unauthenticated fallback reports a
// generic public-S3 one, so neither can be relied on to point back at the
// manifest line that is wrong.
TEST_F(CudfTpchBenchmarkS3Test, wrongCredentialsFailNamingTheOriginalUri) {
  kvikioEnvironment_->set("AWS_ACCESS_KEY_ID", "not-the-access-key");
  kvikioEnvironment_->set("AWS_SECRET_ACCESS_KEY", "not-the-secret-key");
  FLAGS_cudf_io_mode = "raw_file";

  const auto message = runBenchmarkExpectingFailure();

  EXPECT_NE(message.find(objectUri()), std::string::npos) << message;
}

TEST_F(CudfTpchBenchmarkS3Test, aMissingObjectFailsNamingItsUri) {
  // The first object exists so that the schema read succeeds and the failure
  // comes from the payload pass rather than from initialization.
  const auto missingUri =
      filesystems::s3URI(kBucketName, "region/missing.parquet");
  writeManifest({objectUri(), missingUri});
  FLAGS_cudf_io_mode = "raw_file";

  const auto message = runBenchmarkExpectingFailure();

  EXPECT_NE(message.find(missingUri), std::string::npos) << message;
}

// ---------------- Ownership of process-global state ----------------

// Finalizing the S3 file system is process-global and one-shot, so a benchmark
// that found one already registered must leave it registered. The read after
// shutdown is what shows it still works.
TEST_F(CudfTpchBenchmarkS3Test, preregisteredS3IsNotFinalized) {
  runBenchmark();

  ASSERT_TRUE(filesystems::isPathSupportedByRegisteredFileSystems(objectUri()));
  auto fileSystem = filesystems::getFileSystem(objectUri(), hiveConfig());
  EXPECT_EQ(fileSystem->openFileForRead(objectUri())->size(), objectSize());
}

// Unregistering a cuDF the host process registered would take its operators
// away from whatever else is using them.
TEST_F(CudfTpchBenchmarkS3Test, preregisteredCudfIsNotUnregistered) {
  cudf_velox::registerCudf();
  ASSERT_TRUE(cudf_velox::cudfIsRegistered());

  runBenchmark();

  EXPECT_TRUE(cudf_velox::cudfIsRegistered());
  // Left as this test found it, so the fixture's own check still holds.
  cudf_velox::unregisterCudf();
}

TEST_F(CudfTpchBenchmarkS3Test, benchmarkOwnedCudfAndConnectorAreReleased) {
  ASSERT_FALSE(cudf_velox::cudfIsRegistered());

  runBenchmark();

  EXPECT_FALSE(cudf_velox::cudfIsRegistered());
  EXPECT_EQ(connector::ConnectorRegistry::tryGet(kHiveConnectorId), nullptr);
}

// A second shutdown must not release anything twice: unregistering cuDF or
// erasing a connector again would take state the benchmark no longer owns.
TEST_F(CudfTpchBenchmarkS3Test, repeatedShutdownReleasesNothingTwice) {
  auto owned = std::make_unique<CudfTpchBenchmark>();
  owned->initialize();
  RunStats stats;
  std::ostringstream out;
  owned->runMain(out, stats);
  owned->shutdown();

  auto foreign = registerForeignConnector();
  cudf_velox::registerCudf();

  owned->shutdown();
  owned->shutdown();

  EXPECT_EQ(connector::ConnectorRegistry::tryGet(kHiveConnectorId), foreign);
  EXPECT_TRUE(cudf_velox::cudfIsRegistered());

  connector::ConnectorRegistry::global().erase(kHiveConnectorId);
  cudf_velox::unregisterCudf();
}

// Initialization fails on the duplicate connector id. The connector the host
// process registered is not the benchmark's, so cleanup must leave it.
TEST_F(CudfTpchBenchmarkS3Test, aForeignConnectorSurvivesAFailedInitialize) {
  auto foreign = registerForeignConnector();

  auto owned = std::make_unique<CudfTpchBenchmark>();
  VELOX_ASSERT_THROW(owned->initialize(), "Key already registered");

  EXPECT_EQ(connector::ConnectorRegistry::tryGet(kHiveConnectorId), foreign);
  // Shutting the failed benchmark down must not change that either.
  owned->shutdown();
  EXPECT_EQ(connector::ConnectorRegistry::tryGet(kHiveConnectorId), foreign);

  connector::ConnectorRegistry::global().erase(kHiveConnectorId);
}

// Something replaced the entry after the benchmark installed its own. Erasing
// it at shutdown would take a connector the benchmark never registered.
TEST_F(CudfTpchBenchmarkS3Test, aReplacedConnectorIsNotErasedByShutdown) {
  auto owned = std::make_unique<CudfTpchBenchmark>();
  owned->initialize();
  const auto installed = connector::ConnectorRegistry::tryGet(kHiveConnectorId);
  ASSERT_NE(installed, nullptr);

  connector::ConnectorRegistry::global().erase(kHiveConnectorId);
  auto replacement = registerForeignConnector();
  ASSERT_NE(replacement, installed);

  owned->shutdown();

  EXPECT_EQ(
      connector::ConnectorRegistry::tryGet(kHiveConnectorId), replacement);

  connector::ConnectorRegistry::global().erase(kHiveConnectorId);
}

} // namespace

// A dedicated main because the benchmark library rejects an empty --data_path
// through a gflags validator that folly::Init runs against the default. Every
// test points the flag at its own manifest directory; this only has to make it
// non-empty before any flag is parsed.
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  FLAGS_data_path = "/";
  folly::Init init(&argc, &argv, false);
  return RUN_ALL_TESTS();
}
