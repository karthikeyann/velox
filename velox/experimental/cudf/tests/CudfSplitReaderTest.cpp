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

#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/caching/FileHandle.h"
#include "velox/common/config/Config.h"
#include "velox/common/file/File.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

using ::facebook::velox::exec::toPlanStats;
using ::facebook::velox::exec::test::AssertQueryBuilder;
using ::facebook::velox::exec::test::PlanBuilder;

// A cudf datasource that serves reads from an in-memory deterministic byte
// pattern and records every read it is asked to perform. It is deliberately
// not a BufferedInputDataSource so that fetchByteRangesAsync takes its
// coalescing device-read branch.
class RecordingDataSource : public cudf::io::datasource {
 public:
  // One read as submitted to the datasource, after any coalescing the caller
  // performed.
  struct SubmittedRead {
    size_t offset;
    size_t size;
  };

  explicit RecordingDataSource(size_t fileSize) : contents_(fileSize) {
    for (size_t i = 0; i < fileSize; ++i) {
      contents_[i] = static_cast<uint8_t>(i % 251);
    }
  }

  // Makes the read starting at 'offset' report only 'returnedSize' bytes.
  void injectShortRead(size_t offset, size_t returnedSize) {
    shortReads_[offset] = returnedSize;
  }

  // Makes the read starting at 'offset' complete with an exception.
  void injectFailure(size_t offset) {
    failedOffsets_.insert(offset);
  }

  // Makes submission of the read starting at 'offset' throw synchronously,
  // before device_read_async hands back a future.
  void injectSubmissionFailure(size_t offset) {
    submissionFailureOffsets_.insert(offset);
  }

  const std::vector<SubmittedRead>& submittedReads() const {
    return submittedReads_;
  }

  // Returns the byte this source serves at 'offset'.
  uint8_t byteAt(size_t offset) const {
    return contents_[offset];
  }

  size_t size() const override {
    return contents_.size();
  }

  bool supports_device_read() const override {
    return true;
  }

  bool is_device_read_preferred(size_t /*size*/) const override {
    return true;
  }

  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      rmm::cuda_stream_view stream) override {
    if (submissionFailureOffsets_.count(offset) != 0) {
      throw std::runtime_error("injected submission failure");
    }
    submittedReads_.push_back({offset, size});
    std::promise<size_t> promise;
    auto future = promise.get_future();
    if (failedOffsets_.count(offset) != 0) {
      promise.set_exception(
          std::make_exception_ptr(
              std::runtime_error("injected device read failure")));
      return future;
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        dst,
        contents_.data() + offset,
        size,
        cudaMemcpyDefault,
        stream.value()));
    const auto shortRead = shortReads_.find(offset);
    promise.set_value(
        shortRead == shortReads_.end() ? size : shortRead->second);
    return future;
  }

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override {
    return datasource::buffer::create(
        std::vector<uint8_t>(
            contents_.begin() + offset, contents_.begin() + offset + size));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override {
    std::memcpy(dst, contents_.data() + offset, size);
    return size;
  }

 private:
  std::vector<uint8_t> contents_;
  std::vector<SubmittedRead> submittedReads_;
  std::unordered_map<size_t, size_t> shortReads_;
  std::unordered_set<size_t> failedOffsets_;
  std::unordered_set<size_t> submissionFailureOffsets_;
};

class MetadataOnlySplitReader final : public CudfSplitReader {
 public:
  using CudfSplitReader::CudfSplitReader;

  cudf::ast::expression const* logicalFilter() const {
    return subfieldFilter();
  }

  cudf::ast::expression const* splitFilter() const {
    return pushdownFilter();
  }

  bool hasSplitFilter() const {
    return hasSplitSpecificPushdownFilter();
  }

 protected:
  void prepareSplitInternal(
      dwio::common::RuntimeStats& /*runtimeStats*/) override {
    fileMetaDatas();
    // Metadata caching must not rebuild the filter during one preparation.
    fileMetaDatas();
  }
};

// Number of rows each chunk written by writeRowGroupFile() holds.
constexpr int32_t kRowsPerRowGroup = 20'000;

// Owns the datasource, footer and options that a hybrid scan reader borrows,
// so that a test can select ranges from a local Parquet file the same way the
// split reader does. Declared members are destroyed in reverse order, which
// keeps the reader alive no longer than what it refers to.
class LocalParquetReader {
 public:
  // 'configure' adjusts the reader options after they are built over the
  // datasource and before the reader is created. Column selection, byte window
  // and filter are all resolved at reader construction, so that is the only
  // point at which they can still take effect.
  explicit LocalParquetReader(
      const std::string& path,
      const std::function<void(cudf::io::parquet_reader_options&)>& configure =
          nullptr) {
    dataSource_ = std::move(
        cudf::io::make_datasources(cudf::io::source_info{path}).front());
    fileSize_ = dataSource_->size();

    std::vector<std::unique_ptr<cudf::io::datasource>> sources;
    sources.push_back(cudf::io::datasource::create(dataSource_.get()));
    auto footers = cudf::io::read_parquet_footers(sources);
    VELOX_CHECK_EQ(footers.size(), 1);
    footer_ = std::move(footers.front());

    options_ = cudf::io::parquet_reader_options::builder(
                   cudf::io::source_info{dataSource_.get()})
                   .build();
    if (configure) {
      configure(options_);
    }
    reader_ =
        std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(
            footer_, options_);
  }

  cudf::io::parquet::experimental::hybrid_scan_reader& reader() {
    return *reader_;
  }

  const cudf::io::parquet_reader_options& options() const {
    return options_;
  }

  size_t fileSize() const {
    return fileSize_;
  }

 private:
  std::unique_ptr<cudf::io::datasource> dataSource_;
  size_t fileSize_{0};
  cudf::io::parquet::FileMetaData footer_;
  cudf::io::parquet_reader_options options_;
  std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_reader> reader_;
};

int64_t totalRangeBytes(
    const std::vector<cudf::io::text::byte_range_info>& ranges) {
  return std::accumulate(
      ranges.begin(), ranges.end(), int64_t{0}, [](int64_t sum, auto& range) {
        return sum + range.size();
      });
}

// Builds a range from its field layout rather than its constructor, which
// rejects a negative offset or size. The fetch's own checks cover values that
// reach it without having passed through that constructor, so testing them
// needs a range the constructor would refuse to make.
cudf::io::text::byte_range_info rawRange(int64_t offset, int64_t size) {
  using Range = cudf::io::text::byte_range_info;
  static_assert(std::is_trivially_copyable_v<Range>);
  static_assert(sizeof(Range) == 2 * sizeof(int64_t));
  const int64_t fields[2] = {offset, size};
  Range range;
  std::memcpy(&range, fields, sizeof(range));
  // A layout change would otherwise leave these tests silently validating
  // something other than what they name.
  EXPECT_EQ(range.offset(), offset);
  EXPECT_EQ(range.size(), size);
  return range;
}

class CudfSplitReaderTest : public ::facebook::velox::cudf_velox::exec::test::
                                CudfHiveConnectorTestBase {
 protected:
  // Writes a Parquet file holding 'numRowGroups' chunks of kRowsPerRowGroup
  // rows each. The chunked writer emits at least one row group per chunk, and
  // c0 ranges do not overlap between chunks so that a statistics filter can
  // prune whole row groups.
  std::shared_ptr<common::testutil::TempFilePath> writeRowGroupFile(
      int32_t numRowGroups) {
    std::vector<RowVectorPtr> chunks;
    chunks.reserve(numRowGroups);
    for (int32_t group = 0; group < numRowGroups; ++group) {
      const int64_t base = int64_t{group} * kRowsPerRowGroup;
      chunks.push_back(makeRowVector(
          {"c0", "c1"},
          {makeFlatVector<int64_t>(
               kRowsPerRowGroup, [base](auto row) { return base + row; }),
           makeFlatVector<double>(
               kRowsPerRowGroup, [](auto row) { return 0.5 * row; })}));
    }
    auto file = common::testutil::TempFilePath::create();
    writeToFile(file->getPath(), chunks);
    return file;
  }

  // Fetches 'byteRanges' from 'source' and joins the outer future so that all
  // reads have completed when this returns.
  void fetchAndJoin(
      const std::shared_ptr<cudf::io::datasource>& source,
      const std::vector<cudf::io::text::byte_range_info>& byteRanges,
      IoStats* ioStats) {
    auto ioData = fetchByteRangesAsync(
        source,
        byteRanges,
        stream_.view(),
        cudf::get_current_device_resource_ref(),
        ioStats);
    std::get<2>(ioData).get();
  }

  // Validates 'ranges' against an object of 'dataSourceSize' bytes. Taking a
  // vector lets a caller write the ranges as a braced list.
  static void validate(
      const std::vector<cudf::io::text::byte_range_info>& ranges,
      size_t dataSourceSize) {
    validateByteRanges(ranges, dataSourceSize);
  }

  // Returns the counter named 'name', failing the test when it is absent.
  static RuntimeMetric counter(const IoStats& ioStats, std::string_view name) {
    const auto stats = ioStats.stats();
    const auto it = stats.find(std::string(name));
    EXPECT_NE(it, stats.end()) << "missing counter " << name;
    return it == stats.end() ? RuntimeMetric{} : it->second;
  }

  rmm::cuda_stream stream_;
};

// ---------------- Byte-range validation ----------------

TEST_F(CudfSplitReaderTest, acceptsAdjacentAndDisjointRangesWithinTheObject) {
  // The last range ends exactly at the object's end, which is in bounds.
  EXPECT_NO_THROW(validate({{0, 128}, {128, 256}, {4032, 64}}, 4096));
}

TEST_F(CudfSplitReaderTest, acceptsNoRanges) {
  EXPECT_NO_THROW(validate({}, 4096));
}

TEST_F(CudfSplitReaderTest, rejectsNegativeRangeOffset) {
  VELOX_ASSERT_THROW(
      validate({rawRange(-1, 64)}, 4096),
      "Byte range 0 of 1 at offset -1 of size 64 in a 4096-byte object starts "
      "before the object.");
}

TEST_F(CudfSplitReaderTest, rejectsZeroSizedRange) {
  VELOX_ASSERT_THROW(
      validate({{0, 128}, {}}, 4096),
      "Byte range 1 of 2 at offset 0 of size 0 in a 4096-byte object is "
      "empty.");
}

TEST_F(CudfSplitReaderTest, rejectsNegativeRangeSize) {
  VELOX_ASSERT_THROW(
      validate({rawRange(64, -1)}, 4096),
      "Byte range 0 of 1 at offset 64 of size -1 in a 4096-byte object is "
      "empty.");
}

TEST_F(CudfSplitReaderTest, rejectsRangeEndingPastTheObject) {
  VELOX_ASSERT_THROW(
      validate({{0, 128}, {4032, 65}}, 4096),
      "Byte range 1 of 2 at offset 4032 of size 65 in a 4096-byte object ends "
      "past the object.");
}

TEST_F(CudfSplitReaderTest, rejectsRangeWhoseEndOverflows) {
  constexpr int64_t kMaxOffset = std::numeric_limits<int64_t>::max();

  VELOX_ASSERT_THROW(
      validate({{kMaxOffset - 1, 2}}, std::numeric_limits<size_t>::max()),
      "ends past the largest representable offset.");
}

TEST_F(CudfSplitReaderTest, rejectsUnorderedRanges) {
  VELOX_ASSERT_THROW(
      validate({{512, 100}, {0, 100}}, 4096),
      "Byte range 1 of 2 at offset 0 of size 100 in a 4096-byte object starts "
      "before range 0, which starts at offset 512.");
}

TEST_F(CudfSplitReaderTest, rejectsOverlappingRanges) {
  VELOX_ASSERT_THROW(
      validate({{0, 200}, {100, 100}}, 4096),
      "Byte range 1 of 2 at offset 100 of size 100 in a 4096-byte object "
      "overlaps range 0, which ends at offset 200.");
}

// Two ranges of the largest representable size sum past what a size_t can hold
// once the fetch's buffer padding is added. Only an object of the largest
// representable size could hold them, so that is the size they are checked
// against.
TEST_F(CudfSplitReaderTest, rejectsTotalSizeOverflow) {
  constexpr int64_t kMaxSize = std::numeric_limits<int64_t>::max();

  VELOX_ASSERT_THROW(
      validate(
          {{0, kMaxSize}, {0, kMaxSize}}, std::numeric_limits<size_t>::max()),
      "takes the fetch past the");
}

// The point of validating is to stop a nonsensical range before it sizes a
// device allocation or reaches storage, so the fetch must refuse one without
// submitting anything.
TEST_F(CudfSplitReaderTest, fetchRejectsBadRangesBeforeSubmittingAnyRead) {
  auto source = std::make_shared<RecordingDataSource>(4096);
  IoStats ioStats;

  VELOX_ASSERT_THROW(
      fetchByteRangesAsync(
          source,
          std::vector<cudf::io::text::byte_range_info>{{0, 100}, {4000, 200}},
          stream_.view(),
          cudf::get_current_device_resource_ref(),
          &ioStats),
      "ends past the object.");

  EXPECT_TRUE(source->submittedReads().empty());
  // Not even the counters recorded before submission were published, so
  // nothing was requested of storage.
  EXPECT_TRUE(ioStats.stats().empty());
}

TEST_F(CudfSplitReaderTest, adjacentByteRangesBecomeOnePhysicalRequest) {
  auto source = std::make_shared<RecordingDataSource>(4096);
  IoStats ioStats;

  // Two adjacent ranges must coalesce; the disjoint third range must not.
  fetchAndJoin(source, {{0, 128}, {128, 256}, {1024, 64}}, &ioStats);

  EXPECT_EQ(counter(ioStats, kColumnChunkLogicalRanges).sum, 3);
  EXPECT_EQ(counter(ioStats, kColumnChunkPhysicalRequests).sum, 2);

  ASSERT_EQ(source->submittedReads().size(), 2);
  EXPECT_EQ(source->submittedReads()[0].offset, 0);
  EXPECT_EQ(source->submittedReads()[0].size, 384);
  EXPECT_EQ(source->submittedReads()[1].offset, 1024);
  EXPECT_EQ(source->submittedReads()[1].size, 64);
}

TEST_F(CudfSplitReaderTest, recordsRequestedAndCompletedPayloadBytes) {
  auto source = std::make_shared<RecordingDataSource>(4096);
  IoStats ioStats;

  fetchAndJoin(source, {{0, 100}, {512, 200}}, &ioStats);

  const auto requested = counter(ioStats, kColumnChunkRequestedBytes);
  const auto completed = counter(ioStats, kColumnChunkCompletedBytes);
  EXPECT_EQ(requested.sum, 300);
  EXPECT_EQ(completed.sum, 300);
  EXPECT_EQ(requested.unit, RuntimeCounter::Unit::kBytes);
  EXPECT_EQ(completed.unit, RuntimeCounter::Unit::kBytes);
  EXPECT_EQ(counter(ioStats, kColumnChunkLogicalRanges).sum, 2);
  EXPECT_EQ(counter(ioStats, kColumnChunkPhysicalRequests).sum, 2);
}

TEST_F(CudfSplitReaderTest, recordsReadWallTimeOnlyAfterOuterFutureCompletes) {
  auto source = std::make_shared<RecordingDataSource>(4096);
  IoStats ioStats;

  auto ioData = fetchByteRangesAsync(
      source,
      std::vector<cudf::io::text::byte_range_info>{{0, 512}},
      stream_.view(),
      cudf::get_current_device_resource_ref(),
      &ioStats);
  EXPECT_EQ(ioStats.stats().count(std::string(kColumnChunkReadWallNanos)), 0);

  std::get<2>(ioData).get();

  const auto readWall = counter(ioStats, kColumnChunkReadWallNanos);
  EXPECT_GE(readWall.sum, 0);
  EXPECT_EQ(readWall.unit, RuntimeCounter::Unit::kNanos);
}

TEST_F(CudfSplitReaderTest, shortReadFailsWithOffsetAndSizes) {
  auto source = std::make_shared<RecordingDataSource>(4096);
  source->injectShortRead(512, 150);
  IoStats ioStats;

  auto ioData = fetchByteRangesAsync(
      source,
      std::vector<cudf::io::text::byte_range_info>{{0, 100}, {512, 200}},
      stream_.view(),
      cudf::get_current_device_resource_ref(),
      &ioStats);

  try {
    std::get<2>(ioData).get();
    FAIL() << "Expected the short read to fail the fetch";
  } catch (const VeloxException& e) {
    const std::string message = e.message();
    EXPECT_NE(message.find("offset 512"), std::string::npos) << message;
    EXPECT_NE(message.find("expecting 200 bytes"), std::string::npos)
        << message;
    EXPECT_NE(message.find("150 bytes"), std::string::npos) << message;
  }

  EXPECT_EQ(counter(ioStats, kColumnChunkRequestedBytes).sum, 300);
  EXPECT_EQ(counter(ioStats, kColumnChunkCompletedBytes).sum, 100);
}

TEST_F(CudfSplitReaderTest, underlyingReadExceptionPropagatesFromOuterFuture) {
  auto source = std::make_shared<RecordingDataSource>(4096);
  source->injectFailure(0);
  IoStats ioStats;

  auto ioData = fetchByteRangesAsync(
      source,
      std::vector<cudf::io::text::byte_range_info>{{0, 100}},
      stream_.view(),
      cudf::get_current_device_resource_ref(),
      &ioStats);

  try {
    std::get<2>(ioData).get();
    FAIL() << "Expected the failed read to fail the fetch";
  } catch (const VeloxException& e) {
    // The failure must name the request it belongs to without losing the text
    // the data source reported.
    const std::string message = e.message();
    EXPECT_NE(message.find("offset 0"), std::string::npos) << message;
    EXPECT_NE(message.find("expecting 100 bytes"), std::string::npos)
        << message;
    EXPECT_NE(message.find("injected device read failure"), std::string::npos)
        << message;
  }
}

TEST_F(CudfSplitReaderTest, failedReadDrainsLaterReadsAndReportsFirstFailure) {
  auto source = std::make_shared<RecordingDataSource>(4096);
  source->injectFailure(0);
  IoStats ioStats;

  // The failing read is submitted first so that the later, successful read is
  // only accounted for if every pending read is drained.
  auto ioData = fetchByteRangesAsync(
      source,
      std::vector<cudf::io::text::byte_range_info>{{0, 100}, {512, 200}},
      stream_.view(),
      cudf::get_current_device_resource_ref(),
      &ioStats);

  try {
    std::get<2>(ioData).get();
    FAIL() << "Expected the failed read to fail the fetch";
  } catch (const VeloxException& e) {
    const std::string message = e.message();
    EXPECT_NE(message.find("offset 0"), std::string::npos) << message;
    EXPECT_NE(message.find("expecting 100 bytes"), std::string::npos)
        << message;
    EXPECT_NE(message.find("injected device read failure"), std::string::npos)
        << message;
  }

  EXPECT_EQ(counter(ioStats, kColumnChunkRequestedBytes).sum, 300);
  EXPECT_EQ(counter(ioStats, kColumnChunkCompletedBytes).sum, 200);
  EXPECT_GE(counter(ioStats, kColumnChunkReadWallNanos).sum, 0);
}

TEST_F(CudfSplitReaderTest, shortReadDrainsLaterReadsAndReportsFirstFailure) {
  auto source = std::make_shared<RecordingDataSource>(4096);
  source->injectShortRead(0, 40);
  IoStats ioStats;

  auto ioData = fetchByteRangesAsync(
      source,
      std::vector<cudf::io::text::byte_range_info>{{0, 100}, {512, 200}},
      stream_.view(),
      cudf::get_current_device_resource_ref(),
      &ioStats);

  try {
    std::get<2>(ioData).get();
    FAIL() << "Expected the short read to fail the fetch";
  } catch (const VeloxException& e) {
    const std::string message = e.message();
    EXPECT_NE(message.find("offset 0"), std::string::npos) << message;
    EXPECT_NE(message.find("expecting 100 bytes"), std::string::npos)
        << message;
    EXPECT_NE(message.find("40 bytes"), std::string::npos) << message;
  }

  // The short read contributes nothing, but the later read must still be
  // drained and counted.
  EXPECT_EQ(counter(ioStats, kColumnChunkRequestedBytes).sum, 300);
  EXPECT_EQ(counter(ioStats, kColumnChunkCompletedBytes).sum, 200);
  EXPECT_GE(counter(ioStats, kColumnChunkReadWallNanos).sum, 0);
}

TEST_F(
    CudfSplitReaderTest,
    synchronousSubmissionFailureDrainsLaterReadsAndReportsFirstFailure) {
  auto source = std::make_shared<RecordingDataSource>(4096);
  source->injectSubmissionFailure(0);
  IoStats ioStats;

  // The throwing submission comes first, so the later range is only submitted
  // at all if the throw does not unwind out of the submission loop.
  auto ioData = fetchByteRangesAsync(
      source,
      std::vector<cudf::io::text::byte_range_info>{{0, 100}, {512, 200}},
      stream_.view(),
      cudf::get_current_device_resource_ref(),
      &ioStats);

  try {
    std::get<2>(ioData).get();
    FAIL() << "Expected the failed submission to fail the fetch";
  } catch (const VeloxException& e) {
    const std::string message = e.message();
    EXPECT_NE(message.find("offset 0"), std::string::npos) << message;
    EXPECT_NE(message.find("expecting 100 bytes"), std::string::npos)
        << message;
    EXPECT_NE(message.find("injected submission failure"), std::string::npos)
        << message;
  }

  ASSERT_EQ(source->submittedReads().size(), 1);
  EXPECT_EQ(source->submittedReads()[0].offset, 512);
  EXPECT_EQ(counter(ioStats, kColumnChunkRequestedBytes).sum, 300);
  EXPECT_EQ(counter(ioStats, kColumnChunkCompletedBytes).sum, 200);
  // The range whose submission threw must not be counted twice.
  EXPECT_EQ(counter(ioStats, kColumnChunkPhysicalRequests).sum, 2);
  EXPECT_GE(counter(ioStats, kColumnChunkReadWallNanos).sum, 0);
}

TEST_F(CudfSplitReaderTest, nullIoStatsStillFetchesRangeContents) {
  auto source = std::make_shared<RecordingDataSource>(4096);

  auto ioData = fetchByteRangesAsync(
      source,
      std::vector<cudf::io::text::byte_range_info>{{256, 64}},
      stream_.view(),
      cudf::get_current_device_resource_ref(),
      nullptr);
  std::get<2>(ioData).get();

  const auto& span = std::get<1>(ioData).front();
  std::vector<uint8_t> fetched(span.size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      fetched.data(),
      span.data(),
      span.size(),
      cudaMemcpyDefault,
      stream_.value()));
  stream_.synchronize();

  for (size_t i = 0; i < fetched.size(); ++i) {
    ASSERT_EQ(fetched[i], source->byteAt(256 + i)) << "at byte " << i;
  }
}

TEST_F(CudfSplitReaderTest, experimentalReaderPublishesPayloadAndDecodeStats) {
  constexpr int32_t kNumRows = 20'000;
  const auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeRowVector(
          {"c0", "c1"},
          {makeFlatVector<int64_t>(kNumRows, [](auto row) { return row; }),
           makeFlatVector<double>(
               kNumRows, [](auto row) { return 0.5 * row; })}));

  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{
              {CudfHiveConfig::kUseExperimentalCudfReader, "true"},
              {CudfHiveConfig::kUseBufferedInput, "false"}}));

  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .tableHandle(
                      ::facebook::velox::cudf_velox::exec::test::
                          CudfHiveConnectorTestBase::makeTableHandle(
                              "parquet_table", rowType))
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  std::shared_ptr<::facebook::velox::exec::Task> task;
  auto results = AssertQueryBuilder(plan)
                     .splits(makeCudfHiveConnectorSplits({dataFile}))
                     .copyResults(pool_.get(), task);
  ASSERT_NE(results, nullptr);
  EXPECT_EQ(results->size(), kNumRows);

  const auto planStats = toPlanStats(task->taskStats());
  const auto scanStats = planStats.find(plan->id());
  ASSERT_NE(scanStats, planStats.end());
  const auto& customStats = scanStats->second.customStats;

  ASSERT_EQ(customStats.count(std::string(kColumnChunkRequestedBytes)), 1);
  ASSERT_EQ(customStats.count(std::string(kColumnChunkCompletedBytes)), 1);
  ASSERT_EQ(customStats.count(std::string(kColumnChunkLogicalRanges)), 1);
  ASSERT_EQ(customStats.count(std::string(kColumnChunkPhysicalRequests)), 1);
  ASSERT_EQ(customStats.count(std::string(kColumnChunkReadWallNanos)), 1);
  ASSERT_EQ(customStats.count(std::string(kParquetDecodeGpuNanos)), 1);

  const auto requestedBytes =
      customStats.at(std::string(kColumnChunkRequestedBytes)).sum;
  EXPECT_GT(requestedBytes, 0);
  EXPECT_EQ(
      customStats.at(std::string(kColumnChunkCompletedBytes)).sum,
      requestedBytes);
  EXPECT_GT(customStats.at(std::string(kColumnChunkLogicalRanges)).sum, 0);
  EXPECT_GT(customStats.at(std::string(kColumnChunkPhysicalRequests)).sum, 0);

  // CUDA event resolution can round a small decode down to zero, so only the
  // presence of a recorded interval is asserted.
  EXPECT_GE(customStats.at(std::string(kParquetDecodeGpuNanos)).count, 1);
}

// ---------------- Shared column-chunk range selection ----------------

TEST_F(CudfSplitReaderTest, selectsPositiveRowGroupsAndRangesForAllColumns) {
  auto dataFile = writeRowGroupFile(3);
  LocalParquetReader source(dataFile->getPath());

  const auto selected = selectParquetColumnChunkRanges(
      source.reader(), source.options(), stream_.view());

  EXPECT_GT(selected.numRowGroups(), 0);
  EXPECT_FALSE(selected.ranges.empty());
  EXPECT_GT(totalRangeBytes(selected.ranges), 0);
}

TEST_F(CudfSplitReaderTest, oneColumnSelectsNoMoreThanAllColumns) {
  auto dataFile = writeRowGroupFile(3);
  LocalParquetReader allColumns(dataFile->getPath());
  LocalParquetReader oneColumn(dataFile->getPath(), [](auto& options) {
    options.set_column_names({"c0"});
  });

  const auto all = selectParquetColumnChunkRanges(
      allColumns.reader(), allColumns.options(), stream_.view());
  const auto one = selectParquetColumnChunkRanges(
      oneColumn.reader(), oneColumn.options(), stream_.view());

  EXPECT_LE(one.ranges.size(), all.ranges.size());
  EXPECT_LE(totalRangeBytes(one.ranges), totalRangeBytes(all.ranges));
  EXPECT_GT(totalRangeBytes(one.ranges), 0);
  for (const auto& range : one.ranges) {
    EXPECT_GE(range.offset(), 0);
    EXPECT_GT(range.size(), 0);
    EXPECT_LE(
        range.offset() + range.size(),
        static_cast<int64_t>(oneColumn.fileSize()));
  }
}

TEST_F(CudfSplitReaderTest, statisticsFilterSelectsFewerRowGroups) {
  auto dataFile = writeRowGroupFile(3);
  LocalParquetReader unfiltered(dataFile->getPath());
  const auto all = selectParquetColumnChunkRanges(
      unfiltered.reader(), unfiltered.options(), stream_.view());
  ASSERT_GT(all.numRowGroups(), 1)
      << "the writer must produce more than one row group for pruning to be "
         "observable";

  // Only the first chunk holds c0 values below kRowsPerRowGroup, so every
  // later row group is excluded by its own min/max statistics.
  cudf::numeric_scalar<int64_t> bound(kRowsPerRowGroup);
  cudf::ast::literal boundLiteral(bound);
  cudf::ast::column_name_reference columnRef("c0");
  cudf::ast::operation filter(
      cudf::ast::ast_operator::LESS, columnRef, boundLiteral);

  LocalParquetReader filtered(
      dataFile->getPath(), [&](auto& options) { options.set_filter(filter); });
  const auto pruned = selectParquetColumnChunkRanges(
      filtered.reader(), filtered.options(), stream_.view());

  EXPECT_GT(pruned.numRowGroups(), 0);
  EXPECT_LT(pruned.numRowGroups(), all.numRowGroups());
  EXPECT_LT(pruned.ranges.size(), all.ranges.size());
}

TEST_F(CudfSplitReaderTest, byteWindowSelectsTheSameRowGroupsAsTheReader) {
  auto dataFile = writeRowGroupFile(3);
  LocalParquetReader unwindowed(dataFile->getPath());
  const auto all = selectParquetColumnChunkRanges(
      unwindowed.reader(), unwindowed.options(), stream_.view());
  ASSERT_GT(all.numRowGroups(), 1)
      << "the writer must produce more than one row group for a byte window "
         "to exclude anything";

  const auto halfFile = static_cast<size_t>(unwindowed.fileSize() / 2);
  LocalParquetReader windowed(dataFile->getPath(), [halfFile](auto& options) {
    options.set_num_bytes(halfFile);
  });
  const auto selected = selectParquetColumnChunkRanges(
      windowed.reader(), windowed.options(), stream_.view());

  // The selector must apply the reader's own byte-range filter, unchanged.
  const auto expected = windowed.reader().filter_row_groups_with_byte_range(
      windowed.reader().all_row_groups(windowed.options()), windowed.options());
  EXPECT_EQ(selected.rowGroupIndices, expected);
  EXPECT_LT(selected.numRowGroups(), all.numRowGroups());
}

TEST_F(CudfSplitReaderTest, experimentalDecodeReturnsEveryRowAfterSelection) {
  constexpr int32_t kNumRowGroups = 3;
  const auto rowType = ROW({"c0", "c1"}, {BIGINT(), DOUBLE()});
  auto dataFile = writeRowGroupFile(kNumRowGroups);

  resetCudfHiveConnector(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{
              {CudfHiveConfig::kUseExperimentalCudfReader, "true"},
              {CudfHiveConfig::kUseBufferedInput, "false"}}));

  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .tableHandle(
                      ::facebook::velox::cudf_velox::exec::test::
                          CudfHiveConnectorTestBase::makeTableHandle(
                              "parquet_table", rowType))
                  .outputType(rowType)
                  .endTableScan()
                  .planNode();

  auto results = AssertQueryBuilder(plan)
                     .splits(makeCudfHiveConnectorSplits({dataFile}))
                     .copyResults(pool_.get());
  ASSERT_NE(results, nullptr);
  EXPECT_EQ(results->size(), kNumRowGroups * kRowsPerRowGroup);
}

TEST_F(CudfSplitReaderTest, buildsPushdownFilterForEachSplitPreparation) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeRowVector({"c0"}, {makeFlatVector<int64_t>({1, 2, 3})}));

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  ::facebook::velox::connector::ConnectorQueryCtx connectorQueryCtx(
      pool_.get(),
      pool_.get(),
      properties.get(),
      nullptr,
      common::PrefixSortConfig{},
      nullptr,
      nullptr,
      "query.CudfSplitReaderTest",
      "task.CudfSplitReaderTest",
      "plan.CudfSplitReaderTest",
      0,
      "");
  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto split =
      CudfHiveConnectorSplitBuilder(dataFile->getPath())
          .connectorId(
              ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
          .build();

  cudf::ast::column_reference logicalFilter{0};
  cudf::ast::column_reference firstSplitFilter{0};
  cudf::ast::column_reference secondSplitFilter{0};
  MetadataOnlySplitReader reader(
      std::move(split),
      ::facebook::velox::cudf_velox::exec::test::CudfHiveConnectorTestBase::
          makeTableHandle("parquet_table", rowType),
      rowType,
      {"c0"},
      &fileHandleFactory,
      ioExecutor_.get(),
      &connectorQueryCtx,
      std::make_shared<CudfHiveConfig>(properties),
      std::make_shared<io::IoStatistics>(),
      std::make_shared<IoStats>(),
      false,
      &logicalFilter);

  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &logicalFilter);
  EXPECT_FALSE(reader.hasSplitFilter());

  size_t builderCalls = 0;
  std::vector<size_t> schemaSizes;
  reader.setPushdownFilterBuilder(
      [&](const cudf::io::parquet::FileMetaData& metadata) {
        schemaSizes.push_back(metadata.schema.size());
        return builderCalls++ == 0
            ? static_cast<cudf::ast::expression const*>(&firstSplitFilter)
            : static_cast<cudf::ast::expression const*>(&secondSplitFilter);
      });

  // Installing a builder does not change the filter until split metadata is
  // available.
  EXPECT_EQ(reader.splitFilter(), &logicalFilter);
  EXPECT_FALSE(reader.hasSplitFilter());

  dwio::common::RuntimeStats runtimeStats;
  reader.prepareSplit(runtimeStats);
  EXPECT_EQ(builderCalls, 1);
  ASSERT_EQ(schemaSizes.size(), 1);
  EXPECT_GT(schemaSizes.front(), 1);
  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &firstSplitFilter);
  EXPECT_TRUE(reader.hasSplitFilter());

  // Preparing again resets the previous split filter and rebuilds it from the
  // footer without replacing the logical filter.
  reader.prepareSplit(runtimeStats);
  EXPECT_EQ(builderCalls, 2);
  ASSERT_EQ(schemaSizes.size(), 2);
  EXPECT_GT(schemaSizes.back(), 1);
  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &secondSplitFilter);
  EXPECT_TRUE(reader.hasSplitFilter());
  EXPECT_EQ(runtimeStats.processedSplits, 2);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
