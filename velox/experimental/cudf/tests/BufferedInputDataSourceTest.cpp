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
#include "velox/experimental/cudf/connectors/hive/BufferedInputDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderByteFetch.h"
#include "velox/experimental/cudf/connectors/hive/PinnedBufferPool.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/File.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <folly/Executor.h>
#include <folly/executors/IOThreadPoolExecutor.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// An in-memory file that counts its reads, calls 'readHook' at the start of
// each and fails those that reach 'failOffset'.
class FailingReadFile : public InMemoryReadFile {
 public:
  FailingReadFile(std::string content, uint64_t failOffset)
      : InMemoryReadFile(std::move(content)), failOffset_(failOffset) {}

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const FileIoContext& context = {}) const override {
    ++numReads_;
    if (readHook_ != nullptr) {
      readHook_();
    }
    if (offset + length > failOffset_) {
      VELOX_FAIL("Injected read failure at offset: {}", offset);
    }
    return InMemoryReadFile::pread(offset, length, buf, context);
  }

  int32_t numReads() const {
    return numReads_;
  }

  // Reads do not synchronize with this, so it must be set before the first.
  void setReadHook(std::function<void()> readHook) {
    readHook_ = std::move(readHook);
  }

 private:
  const uint64_t failOffset_;
  mutable std::atomic_int32_t numReads_{0};
  std::function<void()> readHook_;
};

// An executor that holds what it is given until drain() runs it, and fails
// submissions past 'maxAdds'.
class HoldingExecutor : public folly::Executor {
 public:
  explicit HoldingExecutor(int32_t maxAdds) : maxAdds_(maxAdds) {}

  void add(folly::Func func) override {
    if (held_.size() >= maxAdds_) {
      VELOX_FAIL("Injected submission failure");
    }
    held_.push_back(std::move(func));
  }

  size_t numHeld() const {
    return held_.size();
  }

  void drain() {
    for (auto& func : held_) {
      func();
    }
    held_.clear();
  }

 private:
  const size_t maxAdds_;
  std::vector<folly::Func> held_;
};

// A buffered input that has an IO executor, reports whether a cache sits in
// front of the file as the test asks, and counts into 'stats'.
class TestBufferedInput : public dwio::common::BufferedInput {
 public:
  TestBufferedInput(
      std::shared_ptr<ReadFile> file,
      memory::MemoryPool& pool,
      io::IoStatistics* stats,
      folly::Executor* executor,
      bool hasCache)
      : BufferedInput(
            std::move(file),
            pool,
            dwio::common::MetricsLog::voidLog(),
            stats),
        executor_(executor),
        hasCache_(hasCache) {}

  folly::Executor* executor() const override {
    return executor_;
  }

  bool hasCache() const override {
    return hasCache_;
  }

 private:
  folly::Executor* const executor_;
  const bool hasCache_;
};

// How a data source stages the bytes of a device read.
struct Staging {
  // Whether the bytes are staged in page-locked buffers.
  bool pinned;
  // Whether the buffered input reports a cache.
  bool hasCache{false};
  // Whether the file asks for coalesced reads, as remote files do.
  bool coalesce{false};
  // Whether a fetch scheduler moves the bytes to the device. Otherwise the
  // reads run on the IO executor.
  bool scheduled{false};

  // Whether the file is read straight into the staging buffers, one read per
  // buffer. Otherwise it is read through the buffered input.
  bool direct() const {
    return pinned && !hasCache && !coalesce;
  }

  std::string toString() const {
    return fmt::format(
        "pinned {} hasCache {} coalesce {} scheduled {}",
        pinned,
        hasCache,
        coalesce,
        scheduled);
  }
};

constexpr Staging kStagings[] = {
    {.pinned = false},
    {.pinned = true},
    {.pinned = true, .hasCache = true},
    {.pinned = true, .coalesce = true},
    {.pinned = true, .scheduled = true},
    {.pinned = true, .hasCache = true, .scheduled = true},
    {.pinned = true, .coalesce = true, .scheduled = true},
};

constexpr size_t kBufferSize{4'096};
constexpr uint64_t kNoFailure{std::numeric_limits<uint64_t>::max()};

class BufferedInputDataSourceTest : public ::facebook::velox::cudf_velox::exec::
                                        test::CudfHiveConnectorTestBase {
 protected:
  // Options of makeDataSource().
  struct Options {
    // File offset from which reads fail.
    uint64_t failOffset{kNoFailure};
    // Runs the reads. The IO executor when null.
    folly::Executor* executor{nullptr};
    // Size of a staging buffer.
    size_t bufferSize{kBufferSize};
    // Buffers in the pool, far fewer than a multi-buffer read needs.
    size_t numBuffers{2};
  };

  static std::string makeContent(size_t size) {
    std::string content(size, '\0');
    for (size_t i = 0; i < size; ++i) {
      content[i] = static_cast<char>((i * 131 + i / 4'096) % 251);
    }
    return content;
  }

  std::unique_ptr<BufferedInputDataSource> makeDataSource(
      const std::string& content,
      Staging staging,
      const Options& options) {
    stagingPool_ = staging.pinned
        ? std::make_shared<PinnedBufferPool>(
              options.bufferSize, options.numBuffers * options.bufferSize)
        : nullptr;
    fetchScheduler_ = staging.scheduled
        ? std::make_shared<FetchScheduler>(
              stagingPool_, FetchScheduler::Options{})
        : nullptr;
    fetchStats_ = std::make_shared<IoStats>();
    file_ = std::make_shared<FailingReadFile>(content, options.failOffset);
    file_->setShouldCoalesce(staging.coalesce);
    stats_ = std::make_unique<io::IoStatistics>();
    return std::make_unique<BufferedInputDataSource>(
        std::make_shared<TestBufferedInput>(
            file_,
            *pool_,
            stats_.get(),
            options.executor != nullptr ? options.executor : ioExecutor_.get(),
            staging.hasCache),
        stagingPool_,
        fetchScheduler_,
        fetchStats_);
  }

  std::unique_ptr<BufferedInputDataSource> makeDataSource(
      const std::string& content,
      Staging staging) {
    return makeDataSource(content, staging, Options{});
  }

  // Reads 'size' bytes at 'offset' to the device and returns what arrived
  // there, which is shorter than 'size' when the read crosses the end of the
  // file.
  static std::string
  deviceRead(BufferedInputDataSource& dataSource, size_t offset, size_t size) {
    auto stream = cudfGlobalStreamPool().get_stream();
    rmm::device_buffer device(size, stream, get_temp_mr());
    const auto numRead =
        dataSource
            .device_read_async(
                offset, size, static_cast<uint8_t*>(device.data()), stream)
            .get();
    std::string result(numRead, '\0');
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        result.data(),
        device.data(),
        numRead,
        cudaMemcpyDeviceToHost,
        stream.get()));
    stream.sync();
    return result;
  }

  // The pool of the data source made last. Null when it stages in pageable
  // memory.
  std::shared_ptr<PinnedBufferPool> stagingPool_;

  // The scheduler of the data source made last. Null when its reads run on
  // the IO executor.
  std::shared_ptr<FetchScheduler> fetchScheduler_;

  // What the scheduled reads of the data source made last cost.
  std::shared_ptr<IoStats> fetchStats_;

  // The file of the data source made last.
  std::shared_ptr<FailingReadFile> file_;

  // The I/O statistics of the data source made last.
  std::unique_ptr<io::IoStatistics> stats_;
};

TEST_F(BufferedInputDataSourceTest, deviceRead) {
  const auto content = makeContent((1 << 20) + 123);
  const auto numBuffersInFile =
      (content.size() + kBufferSize - 1) / kBufferSize;
  for (const auto staging : kStagings) {
    SCOPED_TRACE(staging.toString());
    auto dataSource = makeDataSource(content, staging);

    // The whole file takes many more staging buffers than the pool has.
    EXPECT_EQ(deviceRead(*dataSource, 0, content.size()), content);
    // Every byte is counted once.
    EXPECT_EQ(stats_->rawBytesRead(), content.size());
    const auto fetchStats = fetchStats_->stats();
    if (staging.scheduled) {
      const auto& fetchBytes =
          fetchStats.at(BufferedInputDataSource::kFetchBytes);
      EXPECT_EQ(fetchBytes.sum, content.size());
      EXPECT_EQ(fetchBytes.count, 1);
      EXPECT_GT(
          fetchStats.at(BufferedInputDataSource::kFetchReadWallNanos).sum, 0);
      EXPECT_GE(fetchStats.at(BufferedInputDataSource::kFetchReaders).min, 1);
    } else {
      EXPECT_TRUE(fetchStats.empty());
    }
    // Only a local file that nothing caches is read one staging buffer at a
    // time; otherwise the buffered input chooses the read size.
    if (staging.direct()) {
      EXPECT_EQ(file_->numReads(), numBuffersInFile);
    } else {
      EXPECT_NE(file_->numReads(), numBuffersInFile);
    }

    // Within one staging buffer.
    EXPECT_EQ(deviceRead(*dataSource, 5, 1'000), content.substr(5, 1'000));
    // An exact multiple of the staging buffer size.
    EXPECT_EQ(
        deviceRead(*dataSource, kBufferSize, 3 * kBufferSize),
        content.substr(kBufferSize, 3 * kBufferSize));
    // Across staging buffer boundaries, unaligned at both ends.
    EXPECT_EQ(
        deviceRead(*dataSource, 4'090, 10'000), content.substr(4'090, 10'000));
    // Across the end of the file.
    EXPECT_EQ(
        deviceRead(*dataSource, content.size() - 10, 100),
        content.substr(content.size() - 10));
    // Past the end of the file.
    EXPECT_EQ(deviceRead(*dataSource, content.size(), 100), "");

    if (staging.pinned) {
      EXPECT_GE(stagingPool_->numAllocated(), 1);
      EXPECT_LE(stagingPool_->numAllocated(), 2);
    }
  }
}

// One buffer serves reads of any length from any number of threads.
TEST_F(BufferedInputDataSourceTest, concurrentDeviceReadsWithOneBuffer) {
  const auto content = makeContent(200'000);
  for (const auto staging : kStagings) {
    if (!staging.pinned) {
      continue;
    }
    SCOPED_TRACE(staging.toString());
    auto dataSource = makeDataSource(content, staging, {.numBuffers = 1});
    constexpr size_t kNumThreads{8};
    const auto size = content.size() / kNumThreads;
    std::vector<std::string> results(kNumThreads);
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (size_t i = 0; i < kNumThreads; ++i) {
      threads.emplace_back(
          [&, i]() { results[i] = deviceRead(*dataSource, i * size, size); });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    for (size_t i = 0; i < kNumThreads; ++i) {
      EXPECT_EQ(results[i], content.substr(i * size, size)) << i;
    }
    EXPECT_EQ(stagingPool_->numAllocated(), 1);
  }
}

// A pool that cannot page-lock any buffer does not fail the read: the read
// stages in pageable memory.
TEST_F(BufferedInputDataSourceTest, deviceReadWithoutPageLockedMemory) {
  const auto content = makeContent(200'000);
  for (const auto staging : kStagings) {
    if (!staging.pinned) {
      continue;
    }
    SCOPED_TRACE(staging.toString());
    auto dataSource =
        makeDataSource(content, staging, {.bufferSize = 1ULL << 46});
    EXPECT_EQ(deviceRead(*dataSource, 0, content.size()), content);
    EXPECT_EQ(deviceRead(*dataSource, 77, 5'000), content.substr(77, 5'000));
    // Short reads are on the device when the read completes, like long ones.
    for (size_t offset = 0; offset < 100; ++offset) {
      ASSERT_EQ(
          deviceRead(*dataSource, offset, 3'000), content.substr(offset, 3'000))
          << offset;
    }
    EXPECT_EQ(stagingPool_->numAllocated(), 0);
  }
}

// An IO thread pool runs what one of its own threads submits on that same
// thread. A device read submitted from a pool thread, as when a split is
// prepared in the background, must still be read by the other threads: here
// the submitting thread stays busy until some reads have run.
TEST_F(BufferedInputDataSourceTest, deviceReadSubmittedFromIoThread) {
  const auto content = makeContent(1 << 20);
  folly::IOThreadPoolExecutor executor(4);
  auto dataSource =
      makeDataSource(content, {.pinned = true}, {.executor = &executor});
  auto stream = cudfGlobalStreamPool().get_stream();
  rmm::device_buffer device(content.size(), stream, get_temp_mr());

  std::promise<std::future<size_t>> submitted;
  std::promise<void> release;
  executor.add([&]() {
    submitted.set_value(dataSource->device_read_async(
        0, content.size(), static_cast<uint8_t*>(device.data()), stream));
    release.get_future().wait();
  });
  auto pending = submitted.get_future().get();

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (file_->numReads() == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_GT(file_->numReads(), 0);
  release.set_value();

  EXPECT_EQ(pending.get(), content.size());
  std::string result(content.size(), '\0');
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      result.data(),
      device.data(),
      result.size(),
      cudaMemcpyDeviceToHost,
      stream.get()));
  stream.sync();
  EXPECT_EQ(result, content);
}

// A device read whose result is dropped must not leave reads behind that
// write to the destination later: reads that have not started are skipped.
TEST_F(BufferedInputDataSourceTest, abandonedDeviceRead) {
  const auto content = makeContent(1 << 20);
  for (const auto staging : kStagings) {
    if (staging.scheduled) {
      continue;
    }
    SCOPED_TRACE(staging.toString());
    HoldingExecutor executor(std::numeric_limits<int32_t>::max());
    auto dataSource = makeDataSource(content, staging, {.executor = &executor});
    auto stream = cudfGlobalStreamPool().get_stream();
    {
      rmm::device_buffer device(content.size(), stream, get_temp_mr());
      auto pending = dataSource->device_read_async(
          0, content.size(), static_cast<uint8_t*>(device.data()), stream);
    }
    EXPECT_GT(executor.numHeld(), 0);
    executor.drain();
    EXPECT_EQ(file_->numReads(), 0);
  }
}

// A device read whose result is dropped while one of its reads is running
// must wait for that read, which writes to the destination until it is done.
TEST_F(BufferedInputDataSourceTest, abandonedDeviceReadInProgress) {
  const auto content = makeContent(1 << 20);
  // With a cache in front the device read is a single read through the
  // buffered input, and its copy has landed when the read returns. With a
  // scheduler and no cache several readers are inside the file at once.
  for (const Staging staging :
       {Staging{.pinned = true, .hasCache = true},
        Staging{.pinned = true, .hasCache = true, .scheduled = true},
        Staging{.pinned = true, .scheduled = true}}) {
    SCOPED_TRACE(staging.toString());
    auto dataSource = makeDataSource(content, staging);
    std::promise<void> started;
    std::promise<void> resume;
    auto resumed = resume.get_future().share();
    std::once_flag startedOnce;
    file_->setReadHook([&]() {
      std::call_once(startedOnce, [&]() { started.set_value(); });
      resumed.wait();
    });
    auto stream = cudfGlobalStreamPool().get_stream();
    rmm::device_buffer device(content.size(), stream, get_temp_mr());
    auto pending = dataSource->device_read_async(
        0, content.size(), static_cast<uint8_t*>(device.data()), stream);
    started.get_future().wait();

    auto dropped = std::async(
        std::launch::async, [pending = std::move(pending)]() mutable {
          pending = std::future<size_t>();
        });
    EXPECT_EQ(
        dropped.wait_for(std::chrono::milliseconds(100)),
        std::future_status::timeout);
    resume.set_value();
    dropped.get();
    // Nothing reads once the result is gone.
    const auto numReads = file_->numReads();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(file_->numReads(), numReads);
  }
}

// A fetch of several ranges that is given up waits for the reads in progress.
// With a data source whose futures drop the reads that have not started it
// reads nothing more; otherwise it reads everything first.
TEST_F(BufferedInputDataSourceTest, abandonedFetch) {
  const auto content = makeContent(1 << 20);
  const auto numPieces = content.size() / kBufferSize;
  for (const bool dropsUnstartedReads : {true, false}) {
    SCOPED_TRACE(dropsUnstartedReads);
    std::shared_ptr<cudf::io::datasource> dataSource =
        makeDataSource(content, {.pinned = true, .scheduled = true});
    std::promise<void> started;
    std::promise<void> resume;
    auto resumed = resume.get_future().share();
    std::once_flag startedOnce;
    file_->setReadHook([&]() {
      std::call_once(startedOnce, [&]() { started.set_value(); });
      resumed.wait();
    });
    auto stream = cudfGlobalStreamPool().get_stream();
    const std::vector<cudf::io::text::byte_range_info> byteRanges{
        {0, static_cast<int64_t>(content.size() / 2)},
        {static_cast<int64_t>(content.size() / 2),
         static_cast<int64_t>(content.size() / 2)},
    };
    auto fetch = fetchByteRangesAsync(
        dataSource, byteRanges, false, stream, {get_temp_mr(), get_temp_mr()});
    started.get_future().wait();

    auto abandoned = std::async(
        std::launch::async, [&]() { fetch.abandon(dropsUnstartedReads); });
    EXPECT_EQ(
        abandoned.wait_for(std::chrono::milliseconds(100)),
        std::future_status::timeout);
    resume.set_value();
    abandoned.get();
    EXPECT_TRUE(fetch.buffers.empty());
    EXPECT_FALSE(fetch.pending.valid());
    if (dropsUnstartedReads) {
      // Only the reads that stood at the hook ran: one per staging buffer.
      EXPECT_LE(file_->numReads(), 2);
    } else {
      EXPECT_EQ(file_->numReads(), numPieces);
    }
  }
}

// A scheduled device read whose result is dropped is cancelled: when the
// future is gone, no read is running or still to run.
TEST_F(BufferedInputDataSourceTest, abandonedScheduledDeviceRead) {
  const auto content = makeContent(4 << 20);
  for (const auto staging : kStagings) {
    if (!staging.scheduled) {
      continue;
    }
    SCOPED_TRACE(staging.toString());
    auto dataSource = makeDataSource(content, staging);
    auto stream = cudfGlobalStreamPool().get_stream();
    {
      rmm::device_buffer device(content.size(), stream, get_temp_mr());
      auto pending = dataSource->device_read_async(
          0, content.size(), static_cast<uint8_t*>(device.data()), stream);
    }
    const auto numReads = file_->numReads();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(file_->numReads(), numReads);

    // The scheduler serves the next read.
    EXPECT_EQ(deviceRead(*dataSource, 0, 100'000), content.substr(0, 100'000));
  }
}

// A submission that fails part way through a device read must not leave the
// reads submitted before it behind.
TEST_F(BufferedInputDataSourceTest, deviceReadSubmissionFailure) {
  const auto content = makeContent(1 << 20);
  HoldingExecutor executor(3);
  auto dataSource =
      makeDataSource(content, {.pinned = true}, {.executor = &executor});
  auto stream = cudfGlobalStreamPool().get_stream();
  {
    rmm::device_buffer device(content.size(), stream, get_temp_mr());
    VELOX_ASSERT_THROW(
        dataSource->device_read_async(
            0, content.size(), static_cast<uint8_t*>(device.data()), stream),
        "Injected submission failure");
  }
  EXPECT_EQ(executor.numHeld(), 3);
  executor.drain();
  EXPECT_EQ(file_->numReads(), 0);
}

TEST_F(BufferedInputDataSourceTest, deviceReadFailure) {
  const auto content = makeContent(1 << 20);
  const uint64_t failOffset{64 << 10};
  for (const auto staging : kStagings) {
    SCOPED_TRACE(staging.toString());
    auto dataSource =
        makeDataSource(content, staging, {.failOffset = failOffset});

    VELOX_ASSERT_THROW(
        deviceRead(*dataSource, 0, content.size()),
        "Injected read failure at offset");

    // The failed read gave its staging buffers back, so a later read does
    // not wait for them.
    EXPECT_EQ(
        deviceRead(*dataSource, 100, failOffset - 100),
        content.substr(100, failOffset - 100));
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
