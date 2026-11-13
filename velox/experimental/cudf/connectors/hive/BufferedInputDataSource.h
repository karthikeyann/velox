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

#pragma once

#include "velox/experimental/cudf/connectors/hive/FetchScheduler.h"
#include "velox/experimental/cudf/connectors/hive/PinnedBufferPool.h"

#include "velox/common/file/File.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/io/datasource.hpp>

#include <cuda/stream>

#include <future>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

// A cudf::io::datasource that serves bytes via Velox BufferedInput so that
// reads benefit from AsyncDataCache / SSD cache and are always returned as
// contiguous buffers.
class BufferedInputDataSource : public cudf::io::datasource {
 public:
  /// Runtime statistics of the device reads that go through a fetch
  /// scheduler, added to the 'fetchStats' of the constructor once per read.
  /// Bytes copied to the device.
  static constexpr const char* kFetchBytes = "cudfFetchBytes";
  /// Time the scheduler's readers spent reading from the file or the
  /// buffered input.
  static constexpr const char* kFetchReadWallNanos = "cudfFetchReadWallNanos";
  /// Time a read waited to be admitted: for its first reader, and for the
  /// staging buffers of its later pieces when one reader fills them in turn.
  static constexpr const char* kFetchQueuedWallNanos =
      "cudfFetchQueuedWallNanos";
  /// Time the scheduler's readers waited for the copies of a read's pieces
  /// to the device, summed over pieces.
  static constexpr const char* kFetchCopyWallNanos = "cudfFetchCopyWallNanos";
  /// Share of a read's time in the scheduler during which a copy to the
  /// device was in flight, in percent. The mean over reads tells how busy
  /// the link to the device was.
  static constexpr const char* kFetchCopyBusyPercent =
      "cudfFetchCopyBusyPercent";
  /// Share of a read's time in the scheduler during which reads waited for
  /// a reader, in percent.
  static constexpr const char* kFetchReaderLimitedPercent =
      "cudfFetchReaderLimitedPercent";
  /// Share of a read's time in the scheduler during which reads waited for
  /// a staging buffer, in percent. A high share means the copies to the
  /// device are the limit.
  static constexpr const char* kFetchBufferLimitedPercent =
      "cudfFetchBufferLimitedPercent";
  /// Readers the scheduler allowed when a read completed. Minimum, maximum
  /// and mean over reads show how the reader count moved.
  static constexpr const char* kFetchReaders = "cudfFetchReaders";
  /// Share of a read's pieces whose reader went on while the copy to the
  /// device ran, in percent. The mean over reads shows which way of waiting
  /// for copies the scheduler used.
  static constexpr const char* kFetchCopyQueuedPercent =
      "cudfFetchCopyQueuedPercent";
  /// Time a staging buffer takes to copy to the device over the time it
  /// takes to fill from memory, in percent, as the scheduler measured it
  /// when it started. Below 60 its readers go on while their copies run.
  static constexpr const char* kFetchCopyToFillPercent =
      "cudfFetchCopyToFillPercent";

  /// 'stagingPool' supplies the page-locked buffers that device reads stage
  /// file bytes in. Device reads stage in pageable memory when it is null,
  /// and for a read that finds the pool unable to page-lock any buffer.
  /// With a 'fetchScheduler', device reads are handed to it and staged in
  /// its buffers instead, and what they cost is added to 'fetchStats' when
  /// that is not null.
  BufferedInputDataSource(
      std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
      std::shared_ptr<PinnedBufferPool> stagingPool,
      std::shared_ptr<FetchScheduler> fetchScheduler,
      std::shared_ptr<IoStats> fetchStats);

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(
      size_t offset,
      size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst);

  [[nodiscard]] bool supports_device_read() const override;

  /// Reads through the fetch scheduler when there is one, and otherwise on
  /// the IO executor, or on the thread waiting for the returned future when
  /// that gets to a read first. A local file that nothing caches is read in
  /// pieces of one staging buffer; any other file is read through the
  /// buffered input, one read per call. With staging buffers the bytes are
  /// on the device when the future completes; without them the copy is queued
  /// on 'stream'. Destroying the future without waiting skips the reads that
  /// have not started and waits for the rest, so 'dst' can be released right
  /// after.
  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      cuda::stream_ref stream) override;

  /// Drops, at once, the parts that no reader has started of every device
  /// read handed to the fetch scheduler and not yet collected. For a
  /// consumer that gives all of them up; the futures must still be waited
  /// for or destroyed. Does nothing without a fetch scheduler.
  void dropQueuedDeviceReads();

 private:
  void readContiguous(size_t offset, size_t size, uint8_t* dst);

  // Reads 'size' bytes at 'offset' straight from the file into 'buffer' and
  // counts them in the I/O statistics.
  void readFromFile(size_t offset, size_t size, uint8_t* buffer);

  // Hands the read of 'size' bytes at 'offset' to the fetch scheduler.
  std::future<size_t>
  scheduleDeviceRead(size_t offset, size_t size, uint8_t* dst);

  // Reads 'size' bytes at 'offset' into pageable memory and queues their copy
  // to 'dst' on 'stream'.
  void stagePageable(
      size_t offset,
      size_t size,
      uint8_t* dst,
      cuda::stream_ref stream);

  // Reads 'size' bytes at 'offset' straight from the file into one staging
  // buffer and copies them to 'dst'. 'size' must fit in a staging buffer.
  // Stages in pageable memory, with the copy on 'stream', when the pool has
  // no buffer to give.
  void stageFromFile(
      size_t offset,
      size_t size,
      uint8_t* dst,
      cuda::stream_ref stream);

  // Reads 'size' bytes at 'offset' through the buffered input, one staging
  // buffer at a time, and copies them to 'dst'. Stages in pageable memory,
  // with the copy on 'stream', when the pool has no buffer to give.
  void stageFromInput(
      size_t offset,
      size_t size,
      uint8_t* dst,
      cuda::stream_ref stream);

  std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input_;
  const size_t fileSize_;
  // Null when device reads stage in pageable memory.
  const std::shared_ptr<PinnedBufferPool> stagingPool_;
  // Null when device reads run on the IO executor.
  const std::shared_ptr<FetchScheduler> fetchScheduler_;
  // The device reads of this data source, which the scheduler serves
  // together. Null without a scheduler.
  const std::shared_ptr<FetchScheduler::Group> fetchGroup_;
  // Receives what the scheduled device reads cost. Can be null.
  const std::shared_ptr<IoStats> fetchStats_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
