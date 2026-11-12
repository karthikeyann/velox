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

#include "velox/experimental/cudf/connectors/hive/PinnedBufferPool.h"

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
  /// 'stagingPool' supplies the page-locked buffers that device reads stage
  /// file bytes in. Device reads stage in pageable memory when it is null,
  /// and for a read that finds the pool unable to page-lock any buffer.
  BufferedInputDataSource(
      std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
      std::shared_ptr<PinnedBufferPool> stagingPool);

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(
      size_t offset,
      size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst);

  [[nodiscard]] bool supports_device_read() const override;

  /// Reads on the IO executor, or on the thread waiting for the returned
  /// future when that gets to a read first. A local file that nothing caches
  /// is read in pieces of one staging buffer; any other file is read through
  /// the buffered input, one read per call. With staging buffers the bytes are
  /// on the device when the future completes; without them the copy is queued
  /// on 'stream'. Destroying the future without waiting skips the reads that
  /// have not started and waits for the rest, so 'dst' can be released right
  /// after.
  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      cuda::stream_ref stream) override;

 private:
  void readContiguous(size_t offset, size_t size, uint8_t* dst);

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
};

} // namespace facebook::velox::cudf_velox::connector::hive
