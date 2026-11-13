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

#include "velox/dwio/common/BufferedInput.h"

#include <cudf/io/datasource.hpp>

#include <cuda/stream>

#include <atomic>
#include <future>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

// A cudf::io::datasource that serves bytes via Velox BufferedInput so that
// reads benefit from AsyncDataCache / SSD cache and are always returned as
// contiguous buffers.
class BufferedInputDataSource : public cudf::io::datasource {
 public:
  /// @param pinnedPieceBytes Size of the page-locked pieces a device read
  /// copies through, taken from cuDF's pinned memory resource: at most two per
  /// read in progress. An input without a data cache is then read from the
  /// file into the pieces. Zero reads each byte range into one pageable
  /// buffer.
  BufferedInputDataSource(
      std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
      uint64_t pinnedPieceBytes);

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(
      size_t offset,
      size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst);

  [[nodiscard]] bool supports_device_read() const override;

  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      cuda::stream_ref stream) override;

  /// Makes the device reads that have not begun return zero without reading
  /// or copying anything, including those submitted later, and a read that
  /// copies through page-locked pieces stop at its next piece. For a consumer
  /// that gives up what it asked for: it still waits for the futures, and the
  /// wait is then only for the reads in progress.
  void abandonDeviceReads();

 private:
  void readContiguous(size_t offset, size_t size, uint8_t* dst);

  // Reads up to 'size' bytes at 'offset' and copies them to 'dst' on 'stream'
  // through page-locked pieces, reading a piece while the one before it is
  // copied. The file is read into the pieces unless the input has the bytes
  // in memory or in a data cache. Returns the number of bytes copied, after
  // the last copy completed.
  size_t readToDeviceInPieces(
      size_t offset,
      size_t size,
      uint8_t* dst,
      cuda::stream_ref stream);

  std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input_;
  const size_t fileSize_;
  const uint64_t pinnedPieceBytes_;
  // Whether 'input_' serves byte ranges from a data cache.
  const bool inputHasCache_;
  // Set by abandonDeviceReads().
  std::atomic_bool deviceReadsAbandoned_{false};
};

} // namespace facebook::velox::cudf_velox::connector::hive
