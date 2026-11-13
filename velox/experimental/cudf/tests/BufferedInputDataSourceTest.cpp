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

#include "velox/common/file/File.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <folly/Executor.h>

#include <atomic>
#include <future>
#include <memory>
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
    return InMemoryReadFile::pread(offset, length, buf, context);
  }

  int32_t numReads() const {
    return numReads_;
  }

 private:
  mutable std::atomic_int32_t numReads_{0};
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
      const std::string& content) {
    file_ = std::make_shared<CountingReadFile>(content);
    return std::make_unique<BufferedInputDataSource>(
        std::make_shared<TestBufferedInput>(file_, *pool_, &executor_));
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

// Device reads that have not begun when the consumer gives them up do not
// read the file or write to the device, whether they were asked for before or
// after; without that, waiting for them performs them.
TEST_F(BufferedInputDataSourceTest, abandonedDeviceReads) {
  const auto content = makeContent(100'000);
  constexpr size_t kSize{10'000};
  auto stream = cudfGlobalStreamPool().get_stream();
  {
    SCOPED_TRACE("waited for");
    auto dataSource = makeDataSource(content);
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
    auto dataSource = makeDataSource(content);
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
