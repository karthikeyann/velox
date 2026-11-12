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
#include "velox/experimental/cudf/connectors/hive/PinnedBufferPool.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/File.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <limits>
#include <memory>
#include <string>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// An in-memory file whose reads fail when they reach 'failOffset'.
class FailingReadFile : public InMemoryReadFile {
 public:
  FailingReadFile(std::string content, uint64_t failOffset)
      : InMemoryReadFile(std::move(content)), failOffset_(failOffset) {}

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buf,
      const FileIoContext& context = {}) const override {
    if (offset + length > failOffset_) {
      VELOX_FAIL("Injected read failure at offset: {}", offset);
    }
    return InMemoryReadFile::pread(offset, length, buf, context);
  }

 private:
  const uint64_t failOffset_;
};

// A buffered input that has an IO executor and reports whether a cache sits
// in front of the file as the test asks.
class TestBufferedInput : public dwio::common::BufferedInput {
 public:
  TestBufferedInput(
      std::shared_ptr<ReadFile> file,
      memory::MemoryPool& pool,
      folly::Executor* executor,
      bool hasCache)
      : BufferedInput(std::move(file), pool),
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
  // Whether the buffered input reports a cache, which routes reads through
  // it instead of straight to the file.
  bool hasCache;
};

constexpr Staging kStagings[] = {
    {.pinned = false, .hasCache = false},
    {.pinned = true, .hasCache = false},
    {.pinned = true, .hasCache = true},
};

constexpr size_t kBufferSize{4'096};

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

  // Creates a data source over 'content' whose reads fail at 'failOffset'.
  // The pool has two buffers, far fewer than a multi-buffer read needs.
  std::unique_ptr<BufferedInputDataSource> makeDataSource(
      const std::string& content,
      Staging staging,
      uint64_t failOffset = std::numeric_limits<uint64_t>::max()) {
    stagingPool_ = staging.pinned
        ? std::make_shared<PinnedBufferPool>(kBufferSize, 2 * kBufferSize)
        : nullptr;
    return std::make_unique<BufferedInputDataSource>(
        std::make_shared<TestBufferedInput>(
            std::make_shared<FailingReadFile>(content, failOffset),
            *pool_,
            ioExecutor_.get(),
            staging.hasCache),
        stagingPool_);
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
};

TEST_F(BufferedInputDataSourceTest, deviceRead) {
  const auto content = makeContent((1 << 20) + 123);
  for (const auto staging : kStagings) {
    SCOPED_TRACE(
        fmt::format("pinned {} hasCache {}", staging.pinned, staging.hasCache));
    auto dataSource = makeDataSource(content, staging);

    // The whole file takes many more staging buffers than the pool has.
    EXPECT_EQ(deviceRead(*dataSource, 0, content.size()), content);
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

TEST_F(BufferedInputDataSourceTest, deviceReadFailure) {
  const auto content = makeContent(1 << 20);
  const uint64_t failOffset{64 << 10};
  for (const auto staging : kStagings) {
    SCOPED_TRACE(
        fmt::format("pinned {} hasCache {}", staging.pinned, staging.hasCache));
    auto dataSource = makeDataSource(content, staging, failOffset);

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
