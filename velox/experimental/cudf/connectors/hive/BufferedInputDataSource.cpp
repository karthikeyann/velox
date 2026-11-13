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

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/aligned.hpp>
#include <rmm/resource_ref.hpp>

#include <folly/Executor.h>

#include <algorithm>
#include <array>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// Launches the `read` task on the IO executor or waiting thread, whichever
// starts first.
std::future<size_t> submitDeviceRead(
    folly::Executor* executor,
    std::function<size_t()> read) {
  auto task = std::make_shared<std::packaged_task<size_t()>>(std::move(read));
  auto result = task->get_future();
  auto once = std::make_shared<std::once_flag>();
  auto run = [once, task]() { std::call_once(*once, [task]() { (*task)(); }); };
  executor->add(run);
  return std::async(
      std::launch::deferred,
      [run = std::move(run), result = std::move(result)]() mutable {
        run();
        return result.get();
      });
}

// A page-locked buffer from cuDF's pinned memory resource that one copy to
// the device at a time reads from.
class PinnedPiece {
 public:
  explicit PinnedPiece(size_t size)
      : resource_{cudf::get_pinned_memory_resource()}, size_{size} {
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&copied_, cudaEventDisableTiming));
    try {
      data_ = static_cast<uint8_t*>(
          resource_.allocate_sync(size_, rmm::CUDA_ALLOCATION_ALIGNMENT));
    } catch (...) {
      cudaEventDestroy(copied_);
      throw;
    }
  }

  PinnedPiece(const PinnedPiece&) = delete;
  PinnedPiece& operator=(const PinnedPiece&) = delete;

  // Waits for the copy in flight: the device reads the buffer until then.
  ~PinnedPiece() {
    if (isCopying_) {
      cudaEventSynchronize(copied_);
    }
    cudaEventDestroy(copied_);
    resource_.deallocate_sync(data_, size_, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

  uint8_t* data() {
    return data_;
  }

  // Starts copying the first 'size' bytes to 'dst' on 'stream'.
  void copyToDevice(uint8_t* dst, size_t size, cuda::stream_ref stream) {
    CUDF_CUDA_TRY(
        cudaMemcpyAsync(dst, data_, size, cudaMemcpyDefault, stream.get()));
    isCopying_ = true;
    CUDF_CUDA_TRY(cudaEventRecord(copied_, stream.get()));
  }

  // Waits for the copy started last, after which the buffer can be reused.
  void waitForCopy() {
    if (isCopying_) {
      CUDF_CUDA_TRY(cudaEventSynchronize(copied_));
      isCopying_ = false;
    }
  }

 private:
  rmm::host_device_async_resource_ref resource_;
  const size_t size_;
  uint8_t* data_{nullptr};
  // Recorded after each copy.
  cudaEvent_t copied_{nullptr};
  bool isCopying_{false};
};

} // namespace

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
    uint64_t pinnedPieceBytes)
    : input_(std::move(input)),
      fileSize_(input_->getReadFile()->size()),
      pinnedPieceBytes_(pinnedPieceBytes) {}

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
  VELOX_CHECK(input_->executor() != nullptr, "IO executor is not initialized");
  return submitDeviceRead(
      input_->executor(), [this, offset, size, dst, stream]() {
        if (deviceReadsAbandoned_) {
          return size_t{0};
        }
        ensureCudaContextForThread();
        if (pinnedPieceBytes_ > 0) {
          return readToDeviceInPieces(offset, size, dst, stream);
        }
        auto hostBuffer = host_read(offset, size);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dst,
            hostBuffer->data(),
            hostBuffer->size(),
            cudaMemcpyDefault,
            stream.get()));
        return hostBuffer->size();
      });
}

size_t BufferedInputDataSource::readToDeviceInPieces(
    size_t offset,
    size_t size,
    uint8_t* dst,
    cuda::stream_ref stream) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  auto input = input_->read(
      offset, readSize, facebook::velox::dwio::common::LogType::FILE);
  VELOX_CHECK_NOT_NULL(input, "read() returned null stream");

  const size_t pieceSize = std::min<size_t>(readSize, pinnedPieceBytes_);
  std::array<std::optional<PinnedPiece>, 2> pieces;
  size_t numCopied{0};
  for (size_t i = 0; numCopied < readSize && !deviceReadsAbandoned_; ++i) {
    auto& piece = pieces[i % pieces.size()];
    if (piece.has_value()) {
      piece->waitForCopy();
    } else {
      piece.emplace(pieceSize);
    }
    const size_t numBytes = std::min(pieceSize, readSize - numCopied);
    input->readFully(reinterpret_cast<char*>(piece->data()), numBytes);
    piece->copyToDevice(dst + numCopied, numBytes, stream);
    numCopied += numBytes;
  }
  for (auto& piece : pieces) {
    if (piece.has_value()) {
      piece->waitForCopy();
    }
  }
  return numCopied;
}

void BufferedInputDataSource::abandonDeviceReads() {
  deviceReadsAbandoned_ = true;
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
