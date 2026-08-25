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

#include "velox/common/file/File.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

// ---------------- Column-chunk payload and decode metrics ----------------
// These names are storage-neutral: the range-fetch helper serves both the
// BufferedInput datasource and a direct cuDF/KvikIO datasource, and reports
// the same counters for either.

/// Runtime-stat name for the compressed column-chunk bytes asked of storage.
/// Recorded before any read is submitted, so it also covers ranges whose reads
/// later fail.
inline constexpr std::string_view kColumnChunkRequestedBytes{
    "columnChunkRequestedBytes"};

/// Runtime-stat name for the compressed column-chunk bytes that transferred
/// successfully. A read returning fewer bytes than requested fails the fetch
/// rather than lowering this count.
inline constexpr std::string_view kColumnChunkCompletedBytes{
    "columnChunkCompletedBytes"};

/// Runtime-stat name for the number of column-chunk byte ranges the reader
/// asked for, before any coalescing.
inline constexpr std::string_view kColumnChunkLogicalRanges{
    "columnChunkLogicalRanges"};

/// Runtime-stat name for the number of reads submitted to the datasource. For
/// a direct cuDF/KvikIO datasource this counts submissions after adjacent
/// logical ranges are merged. For BufferedInput it counts the ranges enqueued
/// here, which BufferedInput may coalesce further downstream.
inline constexpr std::string_view kColumnChunkPhysicalRequests{
    "columnChunkPhysicalRequests"};

/// Runtime-stat name for the wall time spent fetching column-chunk ranges,
/// measured from range-fetch entry until every submitted read has completed.
/// Cumulative across drivers and may overlap, so it is a diagnostic rather
/// than a throughput denominator.
inline constexpr std::string_view kColumnChunkReadWallNanos{
    "columnChunkReadWallNanos"};

/// Runtime-stat name for the GPU time the experimental reader spends
/// materializing Parquet chunks and casting decimal columns. Cumulative across
/// drivers and may overlap, so it is a diagnostic rather than a throughput
/// denominator.
inline constexpr std::string_view kParquetDecodeGpuNanos{
    "parquetDecodeGpuNanos"};

// ---------------- Internal helper ----------------
// A cudf::io::datasource that serves bytes via Velox BufferedInput so that
// reads benefit from AsyncDataCache / SSD cache and are always returned as
// contiguous buffers.
class BufferedInputDataSource : public cudf::io::datasource {
 public:
  explicit BufferedInputDataSource(
      std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input);

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
      rmm::cuda_stream_view stream) override;

  // Use the enqueue API from dwio::common::BufferedInput.
  // Pass a device buffer to copy to after load.
  void enqueueForDevice(uint64_t offset, uint64_t size, uint8_t* dst);

  // loads and copies to device.
  void load(rmm::cuda_stream_view stream);

 private:
  void readContiguous(size_t offset, size_t size, uint8_t* dst);

  std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input_;
  const size_t fileSize_;
  std::vector<std::function<void(rmm::cuda_stream_view stream)>>
      pendingDeviceLoads_;
};

/**
 * @brief Hybrid scan reader state
 *
 * This struct is used to store the column chunk data for the hybrid scan reader
 * and a once flag to ensure the setup is only done once.
 */
struct HybridScanState {
  HybridScanState() : isHybridScanSetup_(std::make_unique<std::once_flag>()) {}

  std::vector<rmm::device_buffer> columnChunkBuffers_;
  std::vector<cudf::device_span<const uint8_t>> columnChunkData_;
  std::unique_ptr<std::once_flag> isHybridScanSetup_;
};

/// Compressed Parquet column-chunk ranges selected from one file, together
/// with the row groups they were selected from. Owns its vectors, so it stays
/// valid after the reader, footer and options it was derived from are gone.
struct ParquetColumnChunkRanges {
  /// Selected compressed column-chunk byte ranges in file-offset order.
  std::vector<cudf::io::text::byte_range_info> ranges;

  /// Indices of the row groups 'ranges' was selected from, in the order the
  /// reader returned them. Chunk setup and materialization need the indices
  /// themselves; callers that only fetch bytes need no more than their count.
  std::vector<cudf::size_type> rowGroupIndices;

  /// Number of row groups represented by 'ranges'.
  [[nodiscard]] uint64_t numRowGroups() const {
    return rowGroupIndices.size();
  }
};

/// Selects the compressed column-chunk ranges a Parquet scan of 'reader' would
/// read, applying the byte window and the statistics filter that 'options'
/// carries. Reads no column-chunk data and decodes nothing.
///
/// @param reader Hybrid scan reader over the file's footer.
/// @param options Reader options the reader was constructed with. Column
/// selection, byte window and filter are all read from here.
/// @param stream CUDA stream used by statistics-based row-group filtering.
///
/// @return The selected ranges in file-offset order and the row groups they
/// came from, both empty when the options exclude everything.
ParquetColumnChunkRanges selectParquetColumnChunkRanges(
    cudf::io::parquet::experimental::hybrid_scan_reader& reader,
    const cudf::io::parquet_reader_options& options,
    rmm::cuda_stream_view stream);

/// Validates the byte ranges a fetch is about to submit against the object
/// they are read from.
///
/// Ranges are derived from Parquet metadata or from a caller's read plan, so a
/// malformed file or a planning mistake can otherwise size a device allocation
/// and a read request from a nonsensical range. Every range must start at a
/// non-negative offset, be non-empty, end within `dataSourceSize` without
/// overflowing, and start at or after the end of the range before it. Their
/// sizes must also sum, together with the padding a fetch adds to its
/// destination buffer, to a representable `size_t`; that bound is what also
/// keeps a run of coalesced adjacent ranges from overflowing, since such a run
/// is never larger than the whole set.
///
/// @param byteRanges Ranges in the order they will be fetched. An empty span
/// is valid and reads nothing.
/// @param dataSourceSize Size in bytes of the object the ranges are read from.
///
/// @throws VeloxUserError naming the offending range's index, offset and size
/// together with the object size. The message carries no URI, so it cannot
/// echo a credential a URI might embed.
void validateByteRanges(
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    size_t dataSourceSize);

/// Fetches a list of byte ranges from a host buffer into device buffers.
///
/// The ranges are validated with `validateByteRanges` against the data
/// source's size before anything is allocated or submitted.
///
/// Requested bytes, logical range count and physical request count are
/// recorded on `ioStats` before any read is submitted. Completed bytes and read
/// wall time are recorded when the returned future is resolved, and account for
/// every read that succeeded even when another read failed.
///
/// Callers must call `get()` on the returned future: it joins every read and
/// rethrows the first failure, whereas `wait()` would silently drop it. The
/// returned device buffers must stay alive until that call returns.
///
/// @param dataSource Input datasource
/// @param byteRanges Byte ranges to fetch
/// @param stream CUDA stream
/// @param memoryResource Device memory resource
/// @param ioStats Accumulator for the column-chunk payload counters, or nullptr
/// to fetch without recording anything
///
/// @return A tuple containing the device buffers, the device spans of the
/// fetched data, and a future that joins and validates the read tasks
///
/// @throws VeloxUserError, before allocating or submitting anything, when the
/// ranges do not satisfy `validateByteRanges`.
/// @throws VeloxRuntimeError from the returned future if a read fails or
/// returns fewer bytes than requested. The error names the offset and expected
/// size of that read and keeps the underlying error text as its cause.
std::tuple<
    std::vector<rmm::device_buffer>,
    std::vector<cudf::device_span<const uint8_t>>,
    std::future<void>>
fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref memoryResource,
    IoStats* ioStats);

} // namespace facebook::velox::cudf_velox::connector::hive
