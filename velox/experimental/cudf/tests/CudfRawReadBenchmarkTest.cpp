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
#include "velox/experimental/cudf/benchmarks/CudfRawReadBenchmark.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <cudf/io/text/byte_range_info.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

using ::facebook::velox::common::testutil::TempDirectoryPath;
using ::facebook::velox::common::testutil::TempFilePath;
using ::facebook::velox::cudf_velox::connector::hive::CudfHiveConfig;
using ::facebook::velox::cudf_velox::connector::hive::
    kColumnChunkRequestedBytes;
using ::facebook::velox::cudf_velox::exec::test::CudfHiveConnectorTestBase;
using ::facebook::velox::exec::toPlanStats;
using ::facebook::velox::exec::test::AssertQueryBuilder;
using ::facebook::velox::exec::test::PlanBuilder;

// Rows written by makeParquetFile(). Large enough for the writer to emit
// column chunks whose sizes differ noticeably between columns.
constexpr int32_t kParquetRowCount = 20'000;

// Physical column names of the file makeParquetFile() writes, in file order.
const std::vector<std::string> kAllParquetColumns{"c0", "c1", "c2"};

class CudfRawReadBenchmarkTest : public CudfHiveConnectorTestBase {
 protected:
  // Largest value cuDF's byte_range_info can represent, since it stores signed
  // offsets and sizes.
  static constexpr uint64_t kMaxCudfRange =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

  // Creates a temporary file holding exactly 'numBytes' bytes. Content is
  // deterministic but never inspected: the fetch helper's copy correctness is
  // covered by the Phase 3 tests.
  static std::shared_ptr<TempFilePath> makeBinaryFile(uint64_t numBytes) {
    auto file = TempFilePath::create();
    std::string contents(numBytes, '\0');
    for (uint64_t i = 0; i < numBytes; ++i) {
      contents[i] = static_cast<char>(i % 251);
    }
    file->append(contents);
    EXPECT_EQ(file->fileSize(), static_cast<int64_t>(numBytes));
    return file;
  }

  // Type of the file makeParquetFile() writes.
  static RowTypePtr parquetRowType() {
    return ROW({"c0", "c1", "c2"}, {BIGINT(), DOUBLE(), INTEGER()});
  }

  // Writes a Parquet file whose three columns differ in width, so that
  // selecting one of them is observably cheaper than selecting all three.
  std::shared_ptr<TempFilePath> makeParquetFile() {
    auto file = TempFilePath::create();
    writeToFile(
        file->getPath(),
        makeRowVector(
            {"c0", "c1", "c2"},
            {makeFlatVector<int64_t>(
                 kParquetRowCount, [](auto row) { return row; }),
             makeFlatVector<double>(
                 kParquetRowCount, [](auto row) { return 0.25 * row; }),
             makeFlatVector<int32_t>(kParquetRowCount, [](auto row) {
               return static_cast<int32_t>(row % 1000);
             })}));
    return file;
  }

  // Reads every byte of 'paths' in whole-file mode.
  static CudfRawReadStats readWholeFiles(
      const std::vector<std::string>& paths,
      int32_t numWorkers,
      uint64_t readSizeBytes) {
    return runCudfRawRead(
        paths,
        CudfRawReadOptions{
            .mode = CudfRawReadMode::kFile,
            .numWorkers = numWorkers,
            .readSizeBytes = readSizeBytes,
        },
        get_temp_mr());
  }

  // Reads only the compressed column chunks that 'columnNames' occupies in
  // 'paths'. The read-size bound is deliberately smaller than any of these
  // files so that a run which wrongly applied it would be visible.
  static CudfRawReadStats readParquetRanges(
      const std::vector<std::string>& paths,
      const std::vector<std::string>& columnNames,
      int32_t numWorkers) {
    return runCudfRawRead(
        paths,
        CudfRawReadOptions{
            .mode = CudfRawReadMode::kParquetRanges,
            .numWorkers = numWorkers,
            .readSizeBytes = 4096,
            .columnNames = columnNames,
        },
        get_temp_mr());
  }

  // Scans 'file' with the experimental reader through the cuDF Hive connector
  // and returns the compressed column-chunk bytes that scan asked storage for.
  // The connector is configured the way the cuDF TPC-H decode path configures
  // it, so the two paths differ only in what they do with the bytes.
  uint64_t decodeScanRequestedBytes(const std::shared_ptr<TempFilePath>& file) {
    const auto rowType = parquetRowType();
    resetCudfHiveConnector(
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{
                {CudfHiveConfig::kUseExperimentalCudfReader, "true"},
                {CudfHiveConfig::kUseBufferedInput, "false"},
                {CudfHiveConfig::kAllowMismatchedCudfHiveSchemas, "true"}}));

    auto plan = PlanBuilder(pool_.get())
                    .startTableScan()
                    .tableHandle(
                        CudfHiveConnectorTestBase::makeTableHandle(
                            "parquet_table", rowType))
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();

    std::shared_ptr<::facebook::velox::exec::Task> task;
    auto results = AssertQueryBuilder(plan)
                       .splits(makeCudfHiveConnectorSplits({file}))
                       .copyResults(pool_.get(), task);
    VELOX_CHECK_NOT_NULL(results);
    VELOX_CHECK_EQ(results->size(), kParquetRowCount);

    const auto planStats = toPlanStats(task->taskStats());
    const auto scanStats = planStats.find(plan->id());
    VELOX_CHECK(scanStats != planStats.end());
    const auto& customStats = scanStats->second.customStats;
    const auto requested =
        customStats.find(std::string(kColumnChunkRequestedBytes));
    VELOX_CHECK(customStats.end() != requested);
    VELOX_CHECK_GT(requested->second.sum, 0);
    return static_cast<uint64_t>(requested->second.sum);
  }

  // Path of an object that does not exist. The enclosing directory is unique to
  // this test instance and nothing ever creates the entry, so no stale or
  // concurrently created file can satisfy the open.
  std::string missingObjectPath() const {
    const std::string path = directory_->getPath() + "/missing-object.bin";
    EXPECT_FALSE(std::filesystem::exists(path));
    return path;
  }

 private:
  const std::shared_ptr<TempDirectoryPath> directory_{
      TempDirectoryPath::create()};
};

// ---------------- Whole-file range planning ----------------

TEST_F(CudfRawReadBenchmarkTest, rejectsZeroFileSize) {
  VELOX_ASSERT_THROW(makeWholeFileRanges(0, 1024), "file size");
}

TEST_F(CudfRawReadBenchmarkTest, rejectsZeroReadSize) {
  VELOX_ASSERT_THROW(makeWholeFileRanges(1024, 0), "read size");
}

TEST_F(CudfRawReadBenchmarkTest, fileSmallerThanReadSizeYieldsOneExactRange) {
  const auto ranges = makeWholeFileRanges(1000, 4096);

  ASSERT_EQ(ranges.size(), 1);
  EXPECT_EQ(ranges[0].offset(), 0);
  EXPECT_EQ(ranges[0].size(), 1000);
}

TEST_F(CudfRawReadBenchmarkTest, exactMultipleYieldsNoEmptyTrailingRange) {
  const auto ranges = makeWholeFileRanges(4096, 1024);

  ASSERT_EQ(ranges.size(), 4);
  for (const auto& range : ranges) {
    EXPECT_EQ(range.size(), 1024);
  }
  EXPECT_EQ(ranges.back().offset(), 3072);
}

TEST_F(CudfRawReadBenchmarkTest, nonMultipleYieldsOneShortFinalRange) {
  const auto ranges = makeWholeFileRanges(5000, 1024);

  ASSERT_EQ(ranges.size(), 5);
  for (size_t i = 0; i + 1 < ranges.size(); ++i) {
    EXPECT_EQ(ranges[i].size(), 1024) << "at range " << i;
  }
  EXPECT_EQ(ranges.back().offset(), 4096);
  EXPECT_EQ(ranges.back().size(), 904);
}

TEST_F(CudfRawReadBenchmarkTest, rangesCoverEveryByteExactlyOnce) {
  constexpr int64_t kFileSize = 10'000;
  constexpr int64_t kReadSize = 3'000;
  const auto ranges = makeWholeFileRanges(kFileSize, kReadSize);

  ASSERT_EQ(ranges.size(), 4);
  int64_t nextOffset = 0;
  for (const auto& range : ranges) {
    // Starting exactly where the previous range ended leaves no gap and no
    // overlap.
    EXPECT_EQ(range.offset(), nextOffset);
    EXPECT_GT(range.size(), 0);
    EXPECT_LE(range.size(), kReadSize);
    nextOffset += range.size();
  }
  EXPECT_EQ(nextOffset, kFileSize);
}

TEST_F(CudfRawReadBenchmarkTest, rejectsFileSizeBeyondCudfRangeRepresentation) {
  VELOX_ASSERT_THROW(
      makeWholeFileRanges(kMaxCudfRange + 1, 1 << 20), "file size");
}

TEST_F(CudfRawReadBenchmarkTest, rejectsReadSizeBeyondCudfRangeRepresentation) {
  VELOX_ASSERT_THROW(
      makeWholeFileRanges(1 << 20, kMaxCudfRange + 1), "read size");
}

// ---------------- Whole-file runner over local files ----------------

TEST_F(CudfRawReadBenchmarkTest, singleChunkFileCompletesEveryByte) {
  constexpr uint64_t kFileSize = 4096;
  auto file = makeBinaryFile(kFileSize);

  const auto stats = readWholeFiles({file->getPath()}, 4, 1 << 20);

  EXPECT_EQ(stats.numFiles, 1);
  EXPECT_EQ(stats.effectiveWorkers, 1);
  EXPECT_EQ(stats.requestedBytes, kFileSize);
  EXPECT_EQ(stats.completedBytes, kFileSize);
  EXPECT_EQ(stats.logicalRanges, 1);
  EXPECT_EQ(stats.physicalRequests, 1);
  EXPECT_GT(stats.setupNanos, 0);
  EXPECT_GT(stats.elapsedNanos, 0);
  EXPECT_GT(stats.readWallNanos, 0);
}

TEST_F(CudfRawReadBenchmarkTest, multipleChunksReadEveryByteOnce) {
  constexpr uint64_t kFileSize = 5000;
  auto file = makeBinaryFile(kFileSize);

  const auto stats = readWholeFiles({file->getPath()}, 1, 1024);

  EXPECT_EQ(stats.numFiles, 1);
  EXPECT_EQ(stats.effectiveWorkers, 1);
  EXPECT_EQ(stats.requestedBytes, kFileSize);
  EXPECT_EQ(stats.completedBytes, kFileSize);
  EXPECT_EQ(stats.logicalRanges, 5);
  // Each helper call carries a single range, so nothing can coalesce.
  EXPECT_EQ(stats.physicalRequests, 5);
}

TEST_F(CudfRawReadBenchmarkTest, moreWorkersThanFilesCapsWorkersAtFileCount) {
  constexpr uint64_t kFileSize = 2048;
  std::vector<std::shared_ptr<TempFilePath>> files;
  std::vector<std::string> paths;
  for (int i = 0; i < 3; ++i) {
    files.push_back(makeBinaryFile(kFileSize));
    paths.push_back(files.back()->getPath());
  }

  const auto stats = readWholeFiles(paths, 8, 1 << 20);

  EXPECT_EQ(stats.numFiles, 3);
  EXPECT_EQ(stats.effectiveWorkers, 3);
  EXPECT_EQ(stats.requestedBytes, 3 * kFileSize);
  EXPECT_EQ(stats.completedBytes, 3 * kFileSize);
  EXPECT_EQ(stats.logicalRanges, 3);
}

TEST_F(
    CudfRawReadBenchmarkTest,
    fewerWorkersThanFilesCapsWorkersAtWorkerCount) {
  constexpr uint64_t kFileSize = 1024;
  std::vector<std::shared_ptr<TempFilePath>> files;
  std::vector<std::string> paths;
  for (int i = 0; i < 4; ++i) {
    files.push_back(makeBinaryFile(kFileSize));
    paths.push_back(files.back()->getPath());
  }

  const auto stats = readWholeFiles(paths, 2, 1 << 20);

  EXPECT_EQ(stats.numFiles, 4);
  EXPECT_EQ(stats.effectiveWorkers, 2);
  EXPECT_EQ(stats.requestedBytes, 4 * kFileSize);
  EXPECT_EQ(stats.completedBytes, 4 * kFileSize);
  EXPECT_EQ(stats.logicalRanges, 4);
}

TEST_F(CudfRawReadBenchmarkTest, rejectsEmptyPathList) {
  VELOX_ASSERT_THROW(readWholeFiles({}, 4, 1 << 20), "no paths");
}

TEST_F(CudfRawReadBenchmarkTest, rejectsNonPositiveWorkerCount) {
  auto file = makeBinaryFile(1024);

  VELOX_ASSERT_THROW(readWholeFiles({file->getPath()}, 0, 1 << 20), "workers");
}

TEST_F(CudfRawReadBenchmarkTest, rejectsEmptyObjectNamingItsPath) {
  auto file = TempFilePath::create();
  ASSERT_EQ(file->fileSize(), 0);

  VELOX_ASSERT_THROW(
      readWholeFiles({file->getPath()}, 1, 1 << 20), file->getPath());
}

TEST_F(CudfRawReadBenchmarkTest, missingObjectFailsNamingItsPath) {
  const std::string missingPath = missingObjectPath();

  VELOX_ASSERT_THROW(readWholeFiles({missingPath}, 1, 1 << 20), missingPath);
}

// ---------------- Exact Parquet-range runner over local files ----------------

TEST_F(CudfRawReadBenchmarkTest, parquetRangesModeIsImplemented) {
  auto file = makeParquetFile();

  const auto stats = readParquetRanges({file->getPath()}, {"c0"}, 1);

  EXPECT_GT(stats.requestedBytes, 0);
}

TEST_F(CudfRawReadBenchmarkTest, exactModeRejectsEmptyColumnNames) {
  // A path that cannot be opened proves the option is rejected before any
  // datasource is created.
  VELOX_ASSERT_THROW(
      readParquetRanges({missingObjectPath()}, {}, 1), "column names");
}

TEST_F(CudfRawReadBenchmarkTest, exactModeReportsSelectionAndPayloadCounters) {
  auto file = makeParquetFile();

  const auto stats =
      readParquetRanges({file->getPath()}, kAllParquetColumns, 4);

  EXPECT_EQ(stats.numFiles, 1);
  EXPECT_EQ(stats.effectiveWorkers, 1);
  EXPECT_GT(stats.selectedRowGroups, 0);
  EXPECT_GT(stats.requestedBytes, 0);
  EXPECT_EQ(stats.completedBytes, stats.requestedBytes);
  EXPECT_GT(stats.logicalRanges, 0);
  EXPECT_GT(stats.physicalRequests, 0);
  EXPECT_GT(stats.setupNanos, 0);
  EXPECT_GT(stats.elapsedNanos, 0);
  EXPECT_GT(stats.readWallNanos, 0);
}

TEST_F(CudfRawReadBenchmarkTest, exactBytesStayBelowWholeObjectBytes) {
  auto file = makeParquetFile();

  const auto exact =
      readParquetRanges({file->getPath()}, kAllParquetColumns, 1);
  const auto whole = readWholeFiles({file->getPath()}, 1, 1 << 20);

  EXPECT_GT(exact.requestedBytes, 0);
  EXPECT_GT(exact.completedBytes, 0);
  // The footer, page index and file header sit outside every column chunk, so
  // reading exact ranges must always ask for strictly fewer bytes.
  EXPECT_LT(exact.requestedBytes, whole.requestedBytes);
  EXPECT_LT(exact.completedBytes, whole.completedBytes);
}

TEST_F(CudfRawReadBenchmarkTest, oneColumnRequestsNoMoreThanAllColumns) {
  auto file = makeParquetFile();

  const auto all = readParquetRanges({file->getPath()}, kAllParquetColumns, 1);
  const auto one = readParquetRanges({file->getPath()}, {"c0"}, 1);

  EXPECT_GT(one.requestedBytes, 0);
  EXPECT_LE(one.requestedBytes, all.requestedBytes);
  EXPECT_LE(one.logicalRanges, all.logicalRanges);
  // Narrowing the projection must not change which row groups are selected.
  EXPECT_EQ(one.selectedRowGroups, all.selectedRowGroups);
}

TEST_F(CudfRawReadBenchmarkTest, multipleFilesSumRowGroupsAndBytes) {
  std::vector<std::shared_ptr<TempFilePath>> files;
  std::vector<std::string> paths;
  for (int i = 0; i < 3; ++i) {
    files.push_back(makeParquetFile());
    paths.push_back(files.back()->getPath());
  }

  // The GPU compressor does not produce byte-identical output for identical
  // input, so each file is measured on its own rather than assumed equal.
  CudfRawReadStats expected;
  for (const auto& path : paths) {
    const auto single = readParquetRanges({path}, kAllParquetColumns, 1);
    expected.selectedRowGroups += single.selectedRowGroups;
    expected.requestedBytes += single.requestedBytes;
    expected.completedBytes += single.completedBytes;
    expected.logicalRanges += single.logicalRanges;
  }

  const auto all = readParquetRanges(paths, kAllParquetColumns, 2);

  EXPECT_EQ(all.numFiles, 3);
  EXPECT_EQ(all.effectiveWorkers, 2);
  EXPECT_EQ(all.selectedRowGroups, expected.selectedRowGroups);
  EXPECT_EQ(all.requestedBytes, expected.requestedBytes);
  EXPECT_EQ(all.completedBytes, expected.completedBytes);
  EXPECT_EQ(all.logicalRanges, expected.logicalRanges);
}

TEST_F(CudfRawReadBenchmarkTest, exactModeMissingObjectFailsNamingItsPath) {
  const std::string missingPath = missingObjectPath();

  VELOX_ASSERT_THROW(
      readParquetRanges({missingPath}, kAllParquetColumns, 1), missingPath);
}

TEST_F(CudfRawReadBenchmarkTest, exactModeNonParquetObjectFailsNamingItsPath) {
  auto file = makeBinaryFile(4096);

  VELOX_ASSERT_THROW(
      readParquetRanges({file->getPath()}, kAllParquetColumns, 1),
      file->getPath());
}

TEST_F(CudfRawReadBenchmarkTest, exactBytesMatchExperimentalDecodeScan) {
  auto file = makeParquetFile();

  const auto raw = readParquetRanges({file->getPath()}, kAllParquetColumns, 1);
  const uint64_t decoded = decodeScanRequestedBytes(file);

  // Both paths run the same selector over the same physical projection, so any
  // difference means the raw setup options or column names diverged.
  EXPECT_EQ(raw.requestedBytes, decoded);
}

} // namespace
} // namespace facebook::velox::cudf_velox
