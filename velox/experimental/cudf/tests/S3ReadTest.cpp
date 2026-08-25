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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/benchmarks/CudfBenchmarkDiscard.h"
#include "velox/experimental/cudf/benchmarks/CudfRawReadBenchmark.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/storage_adapters/s3fs/RegisterS3FileSystem.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Config.h"
#include "velox/connectors/hive/storage_adapters/s3fs/tests/S3Test.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox::exec::test;
namespace {

namespace cudf_hive = facebook::velox::cudf_velox::connector::hive;

using facebook::velox::cudf_velox::CudfRawReadMode;
using facebook::velox::cudf_velox::CudfRawReadOptions;
using facebook::velox::cudf_velox::CudfRawReadStats;

// Resolves int.parquet, falling back to a working-directory lookup when the
// build-time path baked into the binary is unreachable (velox-cudf CI builds
// and runs the binaries on different hosts).
std::string resolveIntParquetPath() {
  const std::string relativePath =
      "../../../dwio/parquet/tests/examples/int.parquet";
  auto path =
      test::getDataFilePath("velox/experimental/cudf/tests", relativePath);
  if (std::filesystem::exists(path)) {
    return path;
  }
  return (std::filesystem::current_path() / relativePath)
      .lexically_normal()
      .string();
}

// Saves the environment variables KvikIO reads and puts them back when it goes
// out of scope. KvikIO resolves the endpoint, region and credentials from the
// environment on every open and never sees the connector properties, so a
// fixture that points it at a per-test server has to take the previous values
// back once that server is gone.
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
    ASSERT_EQ(::setenv(name.c_str(), value.c_str(), 1), 0)
        << "Failed to set the environment variable " << name;
  }

 private:
  std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

class S3ReadTest : public S3Test, public ::test::VectorTestBase {
 protected:
  // The single bucket every test reads from, so all three I/O modes compare
  // one S3 URI against itself.
  static constexpr const char* kBucketName = "data";

  // Object key of the two-column Parquet file the tests read.
  static constexpr const char* kObjectKey = "int.parquet";

  // Rows in int.parquet.
  static constexpr int64_t kExpectedRows = 10;

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    filesystems::registerS3FileSystem();
  }

  // Finalizing S3 shuts the AWS SDK down for the whole process, and that also
  // tears down the global libcurl state the KvikIO reads below go through.
  // Neither can be brought back, so both belong to the suite rather than to a
  // single test.
  static void TearDownTestCase() {
    filesystems::finalizeS3FileSystem();
  }

  void SetUp() override {
    S3Test::SetUp();
    exportKvikioEnvironment();

    // Register cudf to enable the CudfDatasource creation from
    // CudfHiveConnector, then the discard sink, whose adapter registerCudf()
    // would otherwise rebuild away.
    facebook::velox::cudf_velox::registerCudf();
    facebook::velox::cudf_velox::registerCudfBenchmarkDiscard();

    // Register Hive connector
    facebook::velox::cudf_velox::connector::hive::CudfHiveConnectorFactory
        factory;
    auto hiveConnector = factory.newConnector(
        kCudfHiveConnectorId, cudfHiveConfig(), ioExecutor_.get());
    facebook::velox::connector::ConnectorRegistry::global().insert(
        hiveConnector->connectorId(), hiveConnector);
  }

  void TearDown() override {
    facebook::velox::connector::ConnectorRegistry::global().erase(
        kCudfHiveConnectorId);
    facebook::velox::cudf_velox::unregisterCudf();
    S3Test::TearDown();
    // Every connector, task, datasource and raw runner that could still open a
    // KvikIO handle is gone, so the endpoint this fixture published can be
    // taken back.
    kvikioEnvironment_.reset();
  }

  // Connector properties of the MinIO fixture, plus the settings that force
  // the direct-KvikIO experimental read path the I/O benchmark modes measure.
  std::shared_ptr<const config::ConfigBase> cudfHiveConfig() const {
    using CudfHiveConfig = cudf_hive::CudfHiveConfig;
    return minioServer_->hiveConfig({
        {CudfHiveConfig::kUseBufferedInput, "false"},
        {CudfHiveConfig::kUseExperimentalCudfReader, "true"},
        {CudfHiveConfig::kAllowMismatchedCudfHiveSchemas, "true"},
    });
  }

  // Local path of the object inside the MinIO data directory.
  std::string localObjectPath() {
    return S3Test::localPath(kBucketName) + "/" + kObjectKey;
  }

  // An object uploaded to the fixture's bucket.
  struct UploadedObject {
    // S3 URI the object is served under.
    std::string s3Uri;
    // Size of the object in bytes, taken from the local copy.
    uint64_t size;
  };

  // Copies int.parquet into the fixture's bucket and returns it, once the
  // server actually serves it.
  UploadedObject uploadIntParquet() {
    const auto sourceFile = resolveIntParquetPath();
    minioServer_->addBucket(kBucketName);
    const auto destinationFile = localObjectPath();
    std::ifstream source(sourceFile, std::ios::binary);
    std::ofstream destination(destinationFile, std::ios::binary);
    destination << source.rdbuf();
    EXPECT_GT(destination.tellp(), 0)
        << "Unable to copy from source " << sourceFile;
    destination.close();

    // Measured on the local copy before the readiness probe below opens the
    // object remotely, so the size never depends on what a reader reports.
    const uint64_t size = std::filesystem::file_size(destinationFile);

    const auto s3Uri = filesystems::s3URI(kBucketName, kObjectKey);
    waitUntilServed(s3Uri);
    return UploadedObject{.s3Uri = s3Uri, .size = size};
  }

  // Sum recorded for the 'name' counter on 'stats'. Fails the test when the
  // counter is absent, which is what proves the node published it at all.
  static int64_t customStatSum(
      const facebook::velox::exec::PlanNodeStats& stats,
      std::string_view name) {
    const auto counter = stats.customStats.find(std::string(name));
    EXPECT_NE(counter, stats.customStats.end())
        << "Runtime counter " << name << " was not published";
    if (counter == stats.customStats.end()) {
      return -1;
    }
    return counter->second.sum;
  }

 private:
  // Blocks until the fixture's server answers a read of 's3Uri'.
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
        auto fileSystem =
            filesystems::getFileSystem(s3Uri, minioServer_->hiveConfig());
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

  // Publishes the fixture's endpoint and credentials under the names KvikIO
  // reads, keeping the previous values for teardown.
  void exportKvikioEnvironment() {
    const filesystems::S3Config s3Config(
        kBucketName, minioServer_->hiveConfig());
    ASSERT_TRUE(s3Config.endpoint().has_value());
    ASSERT_TRUE(s3Config.accessKey().has_value());
    ASSERT_TRUE(s3Config.secretKey().has_value());

    kvikioEnvironment_.emplace();
    // KvikIO needs an explicit scheme, while the connector accepts a bare
    // host:port. Deriving one from the other is what makes hive.s3.endpoint
    // and AWS_ENDPOINT_URL resolve to the same endpoint.
    kvikioEnvironment_->set(
        "AWS_ENDPOINT_URL",
        (s3Config.useSSL() ? "https://" : "http://") +
            s3Config.endpoint().value());
    kvikioEnvironment_->set("AWS_ACCESS_KEY_ID", s3Config.accessKey().value());
    kvikioEnvironment_->set(
        "AWS_SECRET_ACCESS_KEY", s3Config.secretKey().value());
    // MinIO ignores the region but KvikIO refuses to sign a request without
    // one.
    kvikioEnvironment_->set("AWS_DEFAULT_REGION", "us-east-1");
    kvikioEnvironment_->set("AWS_EC2_METADATA_DISABLED", "true");
    // An exported endpoint already implies path-style access: KvikIO addresses
    // the object as "<AWS_ENDPOINT_URL>/<bucket>/<key>" and only builds a
    // virtual-host URL when no endpoint is exported. The connector is
    // configured the same way, so both sides agree without a further setting.
    EXPECT_FALSE(s3Config.useVirtualAddressing());
  }

  std::optional<ScopedEnvironment> kvikioEnvironment_;
};
} // namespace

TEST_F(S3ReadTest, s3ReadTest) {
  const auto s3Uri = uploadIntParquet().s3Uri;

  // Read the parquet file via the S3 bucket.
  auto rowType = ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
  auto tableHandle =
      std::make_shared<facebook::velox::connector::hive::HiveTableHandle>(
          kCudfHiveConnectorId,
          "int_table",
          common::SubfieldFilters{},
          nullptr);
  auto plan = PlanBuilder(pool())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();
  auto split =
      facebook::velox::connector::hive::HiveConnectorSplitBuilder(s3Uri)
          .connectorId(kCudfHiveConnectorId)
          .fileFormat(dwio::common::FileFormat::PARQUET)
          .build();

  auto copy = AssertQueryBuilder(plan).split(split).copyResults(pool());

  // expectedResults is the data in int.parquet file.
  auto expectedResults = makeRowVector(
      {makeFlatVector<int32_t>(
           kExpectedRows, [](auto row) { return row + 100; }),
       makeFlatVector<int64_t>(
           kExpectedRows, [](auto row) { return row + 1000; })});
  assertEqualResults({expectedResults}, {copy});
}

// Exercises all three I/O benchmark modes against one object on one MinIO
// server, so the payload each mode moves is comparable byte for byte and every
// mode resolves the same KvikIO endpoint. Splitting the modes across fixtures
// would give each one its own server on its own port and lose that comparison.
TEST_F(S3ReadTest, ioModesReadTheSameS3Object) {
  const auto uploaded = uploadIntParquet();
  const auto& s3Uri = uploaded.s3Uri;
  // Recorded from the local copy before the object was opened remotely, so the
  // whole-object expectation is independent of what the readers report.
  const uint64_t objectSize = uploaded.size;
  ASSERT_GT(objectSize, 0);

  // Physical Parquet column names of int.parquet, in file order. Exact-range
  // selection and the scan projection have to name the same columns for their
  // byte counts to be comparable.
  const std::vector<std::string> columnNames{"int", "bigint"};

  // ---------------- Decode-discard ----------------

  auto scan = PlanBuilder(pool())
                  .startTableScan()
                  .tableHandle(
                      std::make_shared<
                          facebook::velox::connector::hive::HiveTableHandle>(
                          kCudfHiveConnectorId,
                          "int_table",
                          common::SubfieldFilters{},
                          nullptr))
                  .outputType(ROW({"int", "bigint"}, {INTEGER(), BIGINT()}))
                  .endTableScan()
                  .planNode();
  const auto scanNodeId = scan->id();
  auto plan = facebook::velox::cudf_velox::addBenchmarkDiscard(scan);

  auto split =
      facebook::velox::connector::hive::HiveConnectorSplitBuilder(s3Uri)
          .connectorId(kCudfHiveConnectorId)
          .fileFormat(dwio::common::FileFormat::PARQUET)
          .build();

  std::shared_ptr<facebook::velox::exec::Task> task;
  const auto results =
      AssertQueryBuilder(plan).split(split).copyResults(pool(), task);
  EXPECT_TRUE(results == nullptr || results->size() == 0);

  const auto planStats = facebook::velox::exec::toPlanStats(task->taskStats());

  ASSERT_EQ(planStats.count(plan->id()), 1);
  const auto& discardStats = planStats.at(plan->id());
  EXPECT_EQ(
      customStatSum(discardStats, facebook::velox::cudf_velox::kDiscardedRows),
      kExpectedRows);
  EXPECT_GT(
      customStatSum(discardStats, facebook::velox::cudf_velox::kDiscardedBytes),
      0);
  EXPECT_GT(
      customStatSum(
          discardStats, facebook::velox::cudf_velox::kDiscardedBatches),
      0);

  ASSERT_EQ(planStats.count(scanNodeId), 1);
  const auto& scanStats = planStats.at(scanNodeId);
  const int64_t decodeRequestedBytes =
      customStatSum(scanStats, cudf_hive::kColumnChunkRequestedBytes);
  EXPECT_GT(decodeRequestedBytes, 0);
  EXPECT_EQ(
      decodeRequestedBytes,
      customStatSum(scanStats, cudf_hive::kColumnChunkCompletedBytes));
  EXPECT_GT(customStatSum(scanStats, cudf_hive::kColumnChunkLogicalRanges), 0);
  EXPECT_GT(
      customStatSum(scanStats, cudf_hive::kColumnChunkPhysicalRequests), 0);
  // The timings only have to be published: an object this small can spend
  // less than the counter's resolution in either phase.
  EXPECT_EQ(
      scanStats.customStats.count(
          std::string(cudf_hive::kColumnChunkReadWallNanos)),
      1);
  EXPECT_EQ(
      scanStats.customStats.count(
          std::string(cudf_hive::kParquetDecodeGpuNanos)),
      1);

  // ---------------- Exact Parquet ranges ----------------

  const auto exact = runCudfRawRead(
      {s3Uri},
      CudfRawReadOptions{
          .mode = CudfRawReadMode::kParquetRanges,
          .numWorkers = 1,
          // Exact-range mode reads whole column chunks and ignores this bound;
          // a value below the object size would show up if it did not.
          .readSizeBytes = 512,
          .columnNames = columnNames},
      facebook::velox::cudf_velox::get_temp_mr());
  // Neither raw mode materializes anything: the runner's only result is a
  // counter struct, so there is no table or vector for it to return.
  static_assert(
      std::is_same_v<std::decay_t<decltype(exact)>, CudfRawReadStats>);

  EXPECT_GT(exact.selectedRowGroups, 0);
  EXPECT_GT(exact.requestedBytes, 0);
  EXPECT_EQ(exact.requestedBytes, exact.completedBytes);
  EXPECT_GT(exact.logicalRanges, 0);
  EXPECT_GT(exact.physicalRequests, 0);
  EXPECT_GT(exact.setupNanos, 0);
  EXPECT_GT(exact.elapsedNanos, 0);
  EXPECT_GT(exact.readWallNanos, 0);
  // Both paths run the same selector over the same physical projection against
  // the same object, so any difference means the raw options and the scan
  // options diverged.
  EXPECT_EQ(exact.requestedBytes, static_cast<uint64_t>(decodeRequestedBytes));

  // ---------------- Whole object ----------------

  constexpr uint64_t kReadSizeBytes = 512;
  ASSERT_LE(kReadSizeBytes, objectSize);
  const auto whole = runCudfRawRead(
      {s3Uri},
      CudfRawReadOptions{
          .mode = CudfRawReadMode::kFile,
          .numWorkers = 1,
          .readSizeBytes = kReadSizeBytes},
      facebook::velox::cudf_velox::get_temp_mr());
  static_assert(
      std::is_same_v<std::decay_t<decltype(whole)>, CudfRawReadStats>);

  EXPECT_EQ(whole.requestedBytes, objectSize);
  EXPECT_EQ(whole.completedBytes, objectSize);
  EXPECT_EQ(
      whole.logicalRanges, (objectSize + kReadSizeBytes - 1) / kReadSizeBytes);
  // The footer, page index and file header sit outside every column chunk.
  EXPECT_LE(exact.requestedBytes, whole.requestedBytes);

  // ---------------- Missing object ----------------

  // The failure has to name the URI the caller passed, not the HTTP URL
  // KvikIO builds from it, or a manifest typo cannot be traced back.
  const auto missingUri = filesystems::s3URI(kBucketName, "missing.parquet");
  VELOX_ASSERT_THROW(
      runCudfRawRead(
          {missingUri},
          CudfRawReadOptions{
              .mode = CudfRawReadMode::kFile,
              .numWorkers = 1,
              .readSizeBytes = kReadSizeBytes},
          facebook::velox::cudf_velox::get_temp_mr()),
      missingUri);
}
