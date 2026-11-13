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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"
#include "velox/experimental/cudf/tests/utils/RecordingPinnedResource.h"

#include "velox/common/file/File.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/aligned.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <fmt/format.h>
#include <folly/Executor.h>
#include <folly/ScopeGuard.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// An in-memory file that counts its reads.
class CountingReadFile : public InMemoryReadFile {
 public:
  explicit CountingReadFile(std::string content)
      : InMemoryReadFile(std::move(content)) {}

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const FileIoContext& context = {}) const override {
    ++numReads_;
    if (onRead_ != nullptr) {
      onRead_();
    }
    return InMemoryReadFile::pread(offset, length, buf, context);
  }

  int32_t numReads() const {
    return numReads_;
  }

  // Sets a function that every read calls before it reads.
  void setOnRead(std::function<void()> onRead) {
    onRead_ = std::move(onRead);
  }

 private:
  mutable std::atomic_int32_t numReads_{0};
  std::function<void()> onRead_;
};

// An executor that holds what it is given and never runs it, so that a read
// runs on the thread that waits for it.
class HoldingExecutor : public folly::Executor {
 public:
  void add(folly::Func func) override {
    held_.push_back(std::move(func));
  }

 private:
  std::vector<folly::Func> held_;
};

// A buffered input with an IO executor.
class TestBufferedInput : public dwio::common::BufferedInput {
 public:
  TestBufferedInput(
      std::shared_ptr<ReadFile> file,
      memory::MemoryPool& pool,
      folly::Executor* executor)
      : BufferedInput(std::move(file), pool), executor_(executor) {}

  folly::Executor* executor() const override {
    return executor_;
  }

 private:
  folly::Executor* const executor_;
};

class BufferedInputDataSourceTest : public ::facebook::velox::cudf_velox::exec::
                                        test::CudfHiveConnectorTestBase {
 protected:
  static std::string makeContent(size_t size) {
    std::string content(size, '\0');
    for (size_t i = 0; i < size; ++i) {
      content[i] = static_cast<char>((i * 131 + i / 4'096) % 251);
    }
    return content;
  }

  std::unique_ptr<BufferedInputDataSource> makeDataSource(
      const std::string& content,
      uint64_t pinnedPieceBytes) {
    file_ = std::make_shared<CountingReadFile>(content);
    return std::make_unique<BufferedInputDataSource>(
        std::make_shared<TestBufferedInput>(file_, *pool_, &executor_),
        pinnedPieceBytes);
  }

  // Returns a device buffer of 'size' bytes that all have 'value'.
  static rmm::device_buffer makeDeviceBuffer(size_t size, int value) {
    auto stream = cudfGlobalStreamPool().get_stream();
    rmm::device_buffer device(size, stream, get_temp_mr());
    CUDF_CUDA_TRY(cudaMemsetAsync(device.data(), value, size, stream.get()));
    stream.sync();
    return device;
  }

  static std::string toHost(const rmm::device_buffer& device) {
    auto stream = cudfGlobalStreamPool().get_stream();
    std::string result(device.size(), '\0');
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        result.data(),
        device.data(),
        device.size(),
        cudaMemcpyDeviceToHost,
        stream.get()));
    stream.sync();
    return result;
  }

  HoldingExecutor executor_;
  std::shared_ptr<CountingReadFile> file_;
};

// A device read through page-locked pieces delivers the bytes of the range
// with at most two pieces of the configured size allocated, however long the
// range, and reads the file once for every piece; with a piece size of zero
// it allocates no page-locked memory.
TEST_F(BufferedInputDataSourceTest, pinnedPieces) {
  constexpr size_t kOffset{100};
  constexpr size_t kPieceBytes{4'096};
  const auto content = makeContent(100'000);
  auto stream = cudfGlobalStreamPool().get_stream();

  struct TestCase {
    uint64_t pinnedPieceBytes;
    size_t size;
    size_t expectedSize;
    size_t expectedNumPieces;
    size_t expectedMaxPinnedBytes;
  };
  const std::vector<TestCase> testCases = {
      {0, 50'000, 50'000, 0, 0},
      {kPieceBytes, 1'000, 1'000, 1, 1'000},
      {kPieceBytes, kPieceBytes, kPieceBytes, 1, kPieceBytes},
      {kPieceBytes, 6'000, 6'000, 2, 2 * kPieceBytes},
      {kPieceBytes, 50'000, 50'000, 2, 2 * kPieceBytes},
      // Past the end of the file.
      {kPieceBytes, 200'000, 99'900, 2, 2 * kPieceBytes},
  };
  for (const auto& testCase : testCases) {
    SCOPED_TRACE(
        fmt::format(
            "piece {} size {}", testCase.pinnedPieceBytes, testCase.size));
    test::RecordingPinnedResource pinned{cudf::get_pinned_memory_resource()};
    const auto previous = cudf::set_pinned_memory_resource(pinned);
    SCOPE_EXIT {
      cudf::set_pinned_memory_resource(previous);
    };
    auto dataSource = makeDataSource(content, testCase.pinnedPieceBytes);
    auto device = makeDeviceBuffer(testCase.expectedSize, 7);
    auto read = dataSource->device_read_async(
        kOffset, testCase.size, static_cast<uint8_t*>(device.data()), stream);
    EXPECT_EQ(read.get(), testCase.expectedSize);
    stream.sync();
    EXPECT_EQ(toHost(device), content.substr(kOffset, testCase.expectedSize));
    EXPECT_EQ(pinned.numAllocations(), testCase.expectedNumPieces);
    EXPECT_EQ(pinned.maxBytesInUse(), testCase.expectedMaxPinnedBytes);
    EXPECT_EQ(pinned.bytesInUse(), 0);
    if (testCase.pinnedPieceBytes > 0) {
      // The file is read into the pieces: one read for each.
      EXPECT_EQ(
          file_->numReads(),
          (testCase.expectedSize + kPieceBytes - 1) / kPieceBytes);
    }
  }
}

// A read through pieces that is given up while it reads its first piece
// copies that piece and stops: it reads the file once.
TEST_F(BufferedInputDataSourceTest, abandonedBetweenPieces) {
  constexpr size_t kPieceBytes{4'096};
  constexpr size_t kSize{50'000};
  const auto content = makeContent(100'000);
  auto stream = cudfGlobalStreamPool().get_stream();
  auto dataSource = makeDataSource(content, kPieceBytes);
  file_->setOnRead([&]() { dataSource->abandonDeviceReads(); });
  auto device = makeDeviceBuffer(kSize, 7);
  auto read = dataSource->device_read_async(
      0, kSize, static_cast<uint8_t*>(device.data()), stream);
  EXPECT_EQ(read.get(), kPieceBytes);
  stream.sync();
  EXPECT_EQ(file_->numReads(), 1);
  EXPECT_EQ(
      toHost(device),
      content.substr(0, kPieceBytes) + std::string(kSize - kPieceBytes, 7));
}

// A read whose second piece cannot be allocated fails where the consumer
// waits for it, after the copy of the first piece, and leaves no page-locked
// memory in use.
TEST_F(BufferedInputDataSourceTest, pinnedAllocationFailure) {
  constexpr size_t kPieceBytes{4'096};
  constexpr size_t kSize{50'000};
  const auto content = makeContent(100'000);
  auto stream = cudfGlobalStreamPool().get_stream();
  test::RecordingPinnedResource pinned{
      cudf::get_pinned_memory_resource(), /*failingAllocation=*/2};
  const auto previous = cudf::set_pinned_memory_resource(pinned);
  SCOPE_EXIT {
    cudf::set_pinned_memory_resource(previous);
  };
  auto dataSource = makeDataSource(content, kPieceBytes);
  auto device = makeDeviceBuffer(kSize, 7);
  auto read = dataSource->device_read_async(
      0, kSize, static_cast<uint8_t*>(device.data()), stream);
  EXPECT_THROW(read.get(), std::bad_alloc);
  stream.sync();
  EXPECT_EQ(pinned.numAllocations(), 1);
  EXPECT_EQ(pinned.bytesInUse(), 0);
  EXPECT_EQ(
      toHost(device).substr(0, kPieceBytes), content.substr(0, kPieceBytes));
}

// Device reads that have not begun when the consumer gives them up do not
// read the file or write to the device, whether they were asked for before or
// after; without that, waiting for them performs them.
TEST_F(BufferedInputDataSourceTest, abandonedDeviceReads) {
  const auto content = makeContent(100'000);
  constexpr size_t kSize{10'000};
  auto stream = cudfGlobalStreamPool().get_stream();
  {
    SCOPED_TRACE("waited for");
    auto dataSource = makeDataSource(content, 0);
    auto device = makeDeviceBuffer(kSize, 7);
    auto read = dataSource->device_read_async(
        100, kSize, static_cast<uint8_t*>(device.data()), stream);
    EXPECT_EQ(file_->numReads(), 0);
    EXPECT_EQ(read.get(), kSize);
    stream.sync();
    EXPECT_GT(file_->numReads(), 0);
    EXPECT_EQ(toHost(device), content.substr(100, kSize));
  }
  {
    SCOPED_TRACE("abandoned");
    auto dataSource = makeDataSource(content, 0);
    auto device = makeDeviceBuffer(kSize, 7);
    auto before = dataSource->device_read_async(
        100, kSize, static_cast<uint8_t*>(device.data()), stream);
    dataSource->abandonDeviceReads();
    auto after = dataSource->device_read_async(
        200, kSize, static_cast<uint8_t*>(device.data()), stream);
    EXPECT_EQ(before.get(), 0);
    EXPECT_EQ(after.get(), 0);
    stream.sync();
    EXPECT_EQ(file_->numReads(), 0);
    EXPECT_EQ(toHost(device), std::string(kSize, 7));
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
