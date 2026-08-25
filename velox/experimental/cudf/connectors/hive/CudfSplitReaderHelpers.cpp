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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"

#include "velox/common/Casts.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/file/File.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/types.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <fmt/format.h>
#include <folly/futures/Future.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using facebook::velox::IoStats;
using facebook::velox::RuntimeCounter;
using facebook::velox::saturateCast;
using facebook::velox::cudf_velox::connector::hive::kColumnChunkCompletedBytes;
using facebook::velox::cudf_velox::connector::hive::kColumnChunkLogicalRanges;
using facebook::velox::cudf_velox::connector::hive::
    kColumnChunkPhysicalRequests;
using facebook::velox::cudf_velox::connector::hive::kColumnChunkReadWallNanos;
using facebook::velox::cudf_velox::connector::hive::kColumnChunkRequestedBytes;

using SteadyClock = std::chrono::steady_clock;

// Pad buffer sizes to be a multiple of 8 bytes. Required by
// `decode_page_data_kernel` in cuDF Parquet reader.
constexpr size_t kBufferPaddingMultiple = 8;

// Largest total the payload of one fetch may reach. Rounding that total up to
// kBufferPaddingMultiple has to stay representable, because the destination
// buffer is allocated from the padded value.
constexpr size_t kMaxTotalRangeBytes =
    std::numeric_limits<size_t>::max() - (kBufferPaddingMultiple - 1);

/**
 * @brief Static mutex to serialize batches of IO operations across drivers
 *
 * Mutex to ensure no interleaving of IO operations across drivers to ensure
 * drivers can move ahead without waiting for other drivers to finish their IO.
 */
std::mutex& ioBatchMutex() {
  static std::mutex mutex;
  return mutex;
}

// A submitted read paired with the request it must satisfy, so a failed or
// short read can be reported with its exact offset and sizes.
struct PendingRead {
  size_t offset;
  size_t size;
  std::future<size_t> bytesRead;
};

// Publishes the counters that are known before any read is submitted. Ranges
// whose reads later fail still count as requested.
void recordSubmittedRanges(
    IoStats* ioStats,
    size_t requestedBytes,
    size_t logicalRanges,
    size_t physicalRequests) {
  if (ioStats == nullptr) {
    return;
  }
  ioStats->addCounter(
      std::string(kColumnChunkRequestedBytes),
      RuntimeCounter(
          saturateCast(requestedBytes), RuntimeCounter::Unit::kBytes));
  ioStats->addCounter(
      std::string(kColumnChunkLogicalRanges),
      RuntimeCounter(saturateCast(logicalRanges)));
  ioStats->addCounter(
      std::string(kColumnChunkPhysicalRequests),
      RuntimeCounter(saturateCast(physicalRequests)));
}

// Publishes the counters that are only known once the reads have been joined.
// Called on the failure path too, so 'completedBytes' reflects the transfers
// that did succeed.
void recordJoinedReads(
    IoStats* ioStats,
    size_t completedBytes,
    SteadyClock::time_point readStart) {
  if (ioStats == nullptr) {
    return;
  }
  const auto readWallNanos =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          SteadyClock::now() - readStart)
          .count();
  ioStats->addCounter(
      std::string(kColumnChunkCompletedBytes),
      RuntimeCounter(
          saturateCast(completedBytes), RuntimeCounter::Unit::kBytes));
  ioStats->addCounter(
      std::string(kColumnChunkReadWallNanos),
      RuntimeCounter(readWallNanos, RuntimeCounter::Unit::kNanos));
}

// Packages 'error' as an already-satisfied future, so a read that never made
// it past submission is drained and reported through the same path as one that
// failed after it was accepted.
std::future<size_t> failedFuture(std::exception_ptr error) {
  std::promise<size_t> promise;
  promise.set_exception(std::move(error));
  return promise.get_future();
}

// Attributes 'cause' to the request 'read' was meant to satisfy, so a caller
// that only sees the joined future can tell which transfer went wrong.
std::exception_ptr readFailure(
    const PendingRead& read,
    std::string_view cause) {
  try {
    VELOX_FAIL(
        "Read failed at offset {} expecting {} bytes: {}",
        read.offset,
        read.size,
        cause);
  } catch (...) {
    return std::current_exception();
  }
}

// Waits for every read in 'reads', adding each fully satisfied transfer to
// 'completedBytes' and returning the first failure rather than throwing it, so
// the caller can finish draining and publish accurate counters first. Draining
// the whole vector is what guarantees that no read is still writing into the
// destination buffer once this returns, so the buffer stays valid without
// relying on when a future happens to be destroyed.
std::exception_ptr drainPendingReads(
    std::vector<PendingRead>& reads,
    size_t& completedBytes) {
  std::exception_ptr firstFailure;
  const auto retainFailure = [&](const PendingRead& read,
                                 std::string_view cause) {
    if (firstFailure == nullptr) {
      firstFailure = readFailure(read, cause);
    }
  };

  for (auto& read : reads) {
    size_t bytesRead = 0;
    try {
      bytesRead = read.bytesRead.get();
    } catch (const std::exception& e) {
      retainFailure(read, e.what());
      continue;
    } catch (...) {
      retainFailure(read, "unknown exception");
      continue;
    }
    if (bytesRead != read.size) {
      retainFailure(
          read, fmt::format("short read returned {} bytes", bytesRead));
      continue;
    }
    completedBytes += bytesRead;
  }
  return firstFailure;
}

template <typename T>
std::future<T> toStdFuture(folly::Future<T> follyFuture) {
  auto promise = std::make_shared<std::promise<T>>();
  auto stdFuture = promise->get_future();

  std::move(follyFuture).thenTry([promise](folly::Try<T>&& result) mutable {
    if (result.hasValue()) {
      promise->set_value(std::move(result.value()));
    } else {
      promise->set_exception(result.exception().to_exception_ptr());
    }
  });

  return stdFuture;
}
} // namespace

namespace facebook::velox::cudf_velox::connector::hive {

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input)
    : input_(std::move(input)), fileSize_(input_->getReadFile()->size()) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

void BufferedInputDataSource::enqueueForDevice(
    uint64_t offset,
    uint64_t size,
    uint8_t* dst) {
  auto inputStream = input_->enqueue({offset, size});
  std::shared_ptr sharedStream(std::move(inputStream));
  pendingDeviceLoads_.push_back(
      [dst, size, sharedStream](rmm::cuda_stream_view stream) {
        std::vector<uint8_t> buffer(size);
        sharedStream->readFully(reinterpret_cast<char*>(buffer.data()), size);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dst, buffer.data(), size, cudaMemcpyDefault, stream.value()));
      });
}

void BufferedInputDataSource::load(rmm::cuda_stream_view stream) {
  input_->load(velox::dwio::common::LogType::FILE);
  std::lock_guard<std::mutex> lock(ioBatchMutex());
  for (auto& deviceLoad : pendingDeviceLoads_) {
    deviceLoad(stream);
  }
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
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(input_->executor() != nullptr, "IO executor is not initialized");
  auto future = folly::via(input_->executor())
                    .thenValue([this, offset, size, dst, stream](auto&&) {
                      auto hostBuffer = this->host_read(offset, size);
                      CUDF_CUDA_TRY(cudaMemcpyAsync(
                          dst,
                          hostBuffer->data(),
                          hostBuffer->size(),
                          cudaMemcpyDefault,
                          stream.value()));
                      return hostBuffer->size();
                    });
  return toStdFuture(std::move(future));
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

ParquetColumnChunkRanges selectParquetColumnChunkRanges(
    cudf::io::parquet::experimental::hybrid_scan_reader& reader,
    const cudf::io::parquet_reader_options& options,
    rmm::cuda_stream_view stream) {
  auto rowGroupIndices = reader.all_row_groups(options);

  // Filter row groups using row group byte ranges
  if (options.get_skip_bytes() > 0 or options.get_num_bytes().has_value()) {
    rowGroupIndices =
        reader.filter_row_groups_with_byte_range(rowGroupIndices, options);
  }

  // Filter row groups using column chunk statistics
  if (options.get_filter().has_value()) {
    rowGroupIndices =
        reader.filter_row_groups_with_stats(rowGroupIndices, options, stream);
  }

  auto ranges = reader.all_column_chunks_byte_ranges(rowGroupIndices, options);
  return ParquetColumnChunkRanges{
      .ranges = std::move(ranges),
      .rowGroupIndices = std::move(rowGroupIndices)};
}

void validateByteRanges(
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    size_t dataSourceSize) {
  size_t totalSize = 0;
  int64_t previousOffset = 0;
  int64_t previousEnd = 0;
  for (size_t index = 0; index < byteRanges.size(); ++index) {
    const int64_t offset = byteRanges[index].offset();
    const int64_t size = byteRanges[index].size();
    // Repeated by every message so that a failure identifies the range without
    // naming the object, whose URI can carry a credential.
    const auto where = [&] {
      return fmt::format(
          "Byte range {} of {} at offset {} of size {} in a {}-byte object",
          index,
          byteRanges.size(),
          offset,
          size,
          dataSourceSize);
    };

    VELOX_USER_CHECK_GE(offset, 0, "{} starts before the object.", where());
    VELOX_USER_CHECK_GT(size, 0, "{} is empty.", where());
    // Checked as each size is added, so the sum itself cannot overflow while
    // it is being validated.
    VELOX_USER_CHECK_LE(
        static_cast<size_t>(size),
        kMaxTotalRangeBytes - totalSize,
        "{} takes the fetch past the {} bytes it can allocate for.",
        where(),
        kMaxTotalRangeBytes);
    totalSize += static_cast<size_t>(size);

    VELOX_USER_CHECK_LE(
        size,
        std::numeric_limits<int64_t>::max() - offset,
        "{} ends past the largest representable offset.",
        where());
    const int64_t end = offset + size;
    VELOX_USER_CHECK_LE(
        static_cast<size_t>(end),
        dataSourceSize,
        "{} ends past the object.",
        where());

    if (index > 0) {
      VELOX_USER_CHECK_GE(
          offset,
          previousOffset,
          "{} starts before range {}, which starts at offset {}.",
          where(),
          index - 1,
          previousOffset);
      VELOX_USER_CHECK_GE(
          offset,
          previousEnd,
          "{} overlaps range {}, which ends at offset {}.",
          where(),
          index - 1,
          previousEnd);
    }
    previousOffset = offset;
    previousEnd = end;
  }
}

std::tuple<
    std::vector<rmm::device_buffer>,
    std::vector<cudf::device_span<const uint8_t>>,
    std::future<void>>
fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref memoryResource,
    IoStats* ioStats) {
  VELOX_CHECK_NOT_NULL(dataSource, "A range fetch needs a data source.");
  validateByteRanges(byteRanges, dataSource->size());

  const auto readStart = SteadyClock::now();

  // Allocate device spans for each column chunk
  std::vector<cudf::device_span<const uint8_t>> columnChunkData{};
  columnChunkData.reserve(byteRanges.size());

  // Total IO size across all byte ranges
  auto totalSize = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) { return acc + byteRange.size(); });

  // Allocate single device buffer for all column chunks
  std::vector<rmm::device_buffer> columnChunkBuffers{};
  columnChunkBuffers.emplace_back(
      cudf::util::round_up_safe<size_t>(totalSize, kBufferPaddingMultiple),
      stream,
      memoryResource);

  // Compute device spans for each column chunk
  auto bufferData = static_cast<uint8_t*>(columnChunkBuffers.back().data());
  std::ignore = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) {
        columnChunkData.emplace_back(
            bufferData + acc, static_cast<size_t>(byteRange.size()));
        return acc + byteRange.size();
      });

  // For BufferedInputDataSource, enqueue reads into the buffer and launch the
  // actual load asynchronously.
  if (auto bufferedInput =
          dynamic_cast<BufferedInputDataSource*>(dataSource.get())) {
    // BufferedInput may coalesce the enqueued ranges further downstream, so
    // one physical request per enqueued range is the finest granularity
    // observable here.
    recordSubmittedRanges(
        ioStats, totalSize, byteRanges.size(), byteRanges.size());

    auto iter =
        cuda::make_zip_iterator(byteRanges.begin(), columnChunkData.begin());
    std::for_each(
        iter, iter + byteRanges.size(), [bufferedInput](const auto& tuple) {
          const auto& byteRange = cuda::std::get<0>(tuple);
          const auto& destination = cuda::std::get<1>(tuple);
          bufferedInput->enqueueForDevice(
              static_cast<uint64_t>(byteRange.offset()),
              static_cast<uint64_t>(byteRange.size()),
              const_cast<uint8_t*>(destination.data()));
        });

    // load buffered input data source. A successful load() satisfies every
    // enqueued range, so it accounts for all requested bytes at once.
    auto syncFunction = [ioStats, readStart, totalSize](
                            std::shared_ptr<cudf::io::datasource> dataSource,
                            rmm::cuda_stream_view stream) {
      size_t completedBytes = 0;
      try {
        auto buffer =
            checkedPointerCast<BufferedInputDataSource>(dataSource.get());
        buffer->load(stream);
        completedBytes = totalSize;
      } catch (...) {
        recordJoinedReads(ioStats, completedBytes, readStart);
        throw;
      }
      recordJoinedReads(ioStats, completedBytes, readStart);
    };

    return {
        std::move(columnChunkBuffers),
        std::move(columnChunkData),
        std::async(std::launch::deferred, syncFunction, dataSource, stream)};
  }

  // KvikIO dataSource: Impl borrowed from `fetch_byte_ranges_to_device_async()`
  // in `parquet_io_utils.cpp` in cuDF.
  std::vector<size_t> ioOffsets;
  std::vector<size_t> ioSizes;
  std::vector<uint8_t*> destinations;

  for (size_t chunk = 0; chunk < byteRanges.size();) {
    const auto ioOffset = static_cast<size_t>(byteRanges[chunk].offset());
    auto ioSize = static_cast<size_t>(byteRanges[chunk].size());
    size_t nextChunk = chunk + 1;
    while (nextChunk < byteRanges.size()) {
      const size_t nextOffset = byteRanges[nextChunk].offset();
      if (nextOffset != ioOffset + ioSize) {
        break;
      }
      ioSize += byteRanges[nextChunk].size();
      nextChunk++;
    }
    if (ioSize != 0) {
      ioOffsets.push_back(ioOffset);
      ioSizes.push_back(ioSize);
      destinations.push_back(
          const_cast<uint8_t*>(columnChunkData[chunk].data()));
    }
    chunk = nextChunk;
  }
  VELOX_CHECK_EQ(
      ioOffsets.size(),
      ioSizes.size(),
      "Number of IO offsets and sizes must be equal");
  VELOX_CHECK_EQ(
      ioSizes.size(),
      destinations.size(),
      "Number of IO sizes and destinations must be equal");

  recordSubmittedRanges(
      ioStats, totalSize, byteRanges.size(), ioOffsets.size());

  auto iter = cuda::make_zip_iterator(
      ioOffsets.begin(), ioSizes.begin(), destinations.begin());

  std::vector<PendingRead> deviceReadTasks;
  std::vector<PendingRead> hostReadTasks;
  deviceReadTasks.reserve(ioOffsets.size());
  hostReadTasks.reserve(ioOffsets.size());

  // device_read_async is not guaranteed to follow stream-ordering (see
  // datasource API docs)
  stream.synchronize();

  {
    std::lock_guard<std::mutex> lock(ioBatchMutex());

    std::for_each(iter, iter + ioOffsets.size(), [&](const auto& tuple) {
      const auto ioOffset = cuda::std::get<0>(tuple);
      const auto ioSize = cuda::std::get<1>(tuple);
      const auto dest = cuda::std::get<2>(tuple);

      // Submission that throws instead of returning a future is stored as an
      // already-failed future rather than unwinding out of this loop. That
      // keeps every earlier read joinable by the deferred drain before the
      // destination buffer is released, lets later ranges still be submitted,
      // and reports the failure with the same offset and size context as a
      // read that failed after it was accepted.
      bool preferDevice = false;
      std::future<size_t> submitted;
      try {
        preferDevice = dataSource->supports_device_read() and
            dataSource->is_device_read_preferred(ioSize);
        submitted = preferDevice
            ? dataSource->device_read_async(ioOffset, ioSize, dest, stream)
            // TODO(mh): We can't yet guarantee (without a safe thread pool)
            // that all `cudaMemcpyAsync`s will be launched by the time we
            // release the mutex. That said, this is a rare usecase as
            // host-buffer data should prefer using a `BufferedInputDataSource`
            // datasource.
            : std::async(
                  std::launch::async,
                  [dataSource, ioOffset, ioSize, dest, stream]() {
                    auto hostBuffer = dataSource->host_read(ioOffset, ioSize);
                    CUDF_CUDA_TRY(cudaMemcpyAsync(
                        dest,
                        hostBuffer->data(),
                        hostBuffer->size(),
                        cudaMemcpyDefault,
                        stream.value()));
                    return hostBuffer->size();
                  });
      } catch (...) {
        submitted = failedFuture(std::current_exception());
      }

      // Exactly one pending read per range, so a failed submission does not
      // add a second physical request.
      auto& tasks = preferDevice ? deviceReadTasks : hostReadTasks;
      tasks.emplace_back(PendingRead{ioOffset, ioSize, std::move(submitted)});
    });
  }

  auto syncFunction = [ioStats, readStart](
                          decltype(hostReadTasks)&& hostReadTasks,
                          decltype(deviceReadTasks)&& deviceReadTasks) {
    size_t completedBytes = 0;
    // Both vectors are drained before anything is reported or rethrown, so the
    // counters cover every transfer that did succeed and no read outlives this
    // call.
    const auto hostFailure = drainPendingReads(hostReadTasks, completedBytes);
    const auto deviceFailure =
        drainPendingReads(deviceReadTasks, completedBytes);
    recordJoinedReads(ioStats, completedBytes, readStart);
    if (const auto failure =
            hostFailure != nullptr ? hostFailure : deviceFailure) {
      std::rethrow_exception(failure);
    }
  };

  return {
      std::move(columnChunkBuffers),
      std::move(columnChunkData),
      std::async(
          std::launch::deferred,
          std::move(syncFunction),
          std::move(hostReadTasks),
          std::move(deviceReadTasks))};
}

} // namespace facebook::velox::cudf_velox::connector::hive
