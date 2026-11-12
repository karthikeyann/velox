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

#include "velox/common/base/Exceptions.h"
#include "velox/common/time/Timer.h"

#include <folly/Executor.h>

#include <algorithm>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// A read that runs on the IO executor or on the thread that joins it,
// whichever gets to it first.
class ReadTask {
 public:
  ReadTask(folly::Executor* executor, std::function<void()> read)
      : state_(std::make_shared<State>(std::move(read))) {
    executor->add([state = state_]() { state->run(); });
  }

  // Runs the read unless it has started, waits for it and rethrows its
  // failure. Called at most once.
  void join() {
    state_->run();
    state_->result.get();
  }

 private:
  struct State {
    explicit State(std::function<void()> read)
        : task(std::move(read)), result(task.get_future()) {}

    void run() {
      std::call_once(once, [this]() { task(); });
    }

    std::once_flag once;
    std::packaged_task<void()> task;
    std::future<void> result;
  };

  std::shared_ptr<State> state_;
};

// Joins every task, then rethrows the first failure. The reads write through
// pointers their submitter owns, so none may outlive the join.
void joinAll(std::vector<ReadTask>& tasks) {
  std::exception_ptr failure;
  for (auto& task : tasks) {
    try {
      task.join();
    } catch (...) {
      if (failure == nullptr) {
        failure = std::current_exception();
      }
    }
  }
  if (failure != nullptr) {
    std::rethrow_exception(failure);
  }
}

// Copies the first 'size' bytes of 'buffer' to 'dst' and waits for the copy,
// after which the buffer can be reused.
void copyToDevice(
    const PinnedBufferPool::StagingBuffer& buffer,
    size_t size,
    uint8_t* dst) {
  ensureCudaContextForThread();
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      dst, buffer.data, size, cudaMemcpyHostToDevice, buffer.stream));
  CUDF_CUDA_TRY(cudaEventRecord(buffer.event, buffer.stream));
  CUDF_CUDA_TRY(cudaEventSynchronize(buffer.event));
}

} // namespace

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
    std::shared_ptr<PinnedBufferPool> stagingPool)
    : input_(std::move(input)),
      fileSize_(input_->getReadFile()->size()),
      stagingPool_(std::move(stagingPool)) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readContiguous(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    cuda::stream_ref stream) {
  auto* executor = input_->executor();
  VELOX_CHECK_NOT_NULL(executor, "IO executor is not initialized");
  const size_t readSize =
      offset >= fileSize_ ? 0 : std::min(size, fileSize_ - offset);

  std::vector<ReadTask> tasks;
  if (readSize == 0) {
    // Nothing to read.
  } else if (stagingPool_ == nullptr) {
    tasks.emplace_back(executor, [this, offset, readSize, dst, stream]() {
      auto hostBuffer = host_read(offset, readSize);
      ensureCudaContextForThread();
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          dst,
          hostBuffer->data(),
          hostBuffer->size(),
          cudaMemcpyDefault,
          stream.get()));
    });
  } else if (input_->hasCache()) {
    // One read per requested range keeps the cache entries the same as for
    // any other reader of the range.
    tasks.emplace_back(executor, [this, offset, readSize, dst]() {
      stageFromInput(offset, readSize, dst);
    });
  } else {
    // Nothing caches these bytes, so the file is read straight into the
    // staging buffers, one independent read per buffer.
    const auto bufferSize = stagingPool_->bufferSize();
    tasks.reserve((readSize + bufferSize - 1) / bufferSize);
    for (size_t done = 0; done < readSize; done += bufferSize) {
      const auto partSize = std::min(bufferSize, readSize - done);
      tasks.emplace_back(executor, [this, offset, done, partSize, dst]() {
        stageFromFile(offset + done, partSize, dst + done);
      });
    }
  }

  return std::async(
      std::launch::deferred, [tasks = std::move(tasks), readSize]() mutable {
        joinAll(tasks);
        return readSize;
      });
}

void BufferedInputDataSource::stageFromFile(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  auto buffer = stagingPool_->acquire();
  const auto& file = input_->getInputStream();
  uint64_t readUs{0};
  {
    MicrosecondWallTimer timer(&readUs);
    file->read(buffer->data, size, offset, dwio::common::LogType::FILE);
  }
  if (auto* stats = file->getStats()) {
    stats->read().increment(size);
    stats->queryThreadIoLatencyUs().increment(readUs);
    stats->storageReadLatencyUs().increment(readUs);
  }
  copyToDevice(*buffer, size, dst);
}

void BufferedInputDataSource::stageFromInput(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  auto input = input_->read(offset, size, dwio::common::LogType::FILE);
  VELOX_CHECK_NOT_NULL(input, "Buffered input returned no stream");
  auto buffer = stagingPool_->acquire();
  const auto bufferSize = stagingPool_->bufferSize();
  for (size_t done = 0; done < size; done += bufferSize) {
    const auto partSize = std::min(bufferSize, size - done);
    input->readFully(reinterpret_cast<char*>(buffer->data), partSize);
    copyToDevice(*buffer, partSize, dst + done);
  }
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

} // namespace facebook::velox::cudf_velox::connector::hive
