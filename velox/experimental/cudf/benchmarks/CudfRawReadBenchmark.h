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

#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Selects the raw range source.
enum class CudfRawReadMode {
  /// Reads only the compressed column-chunk ranges a Parquet scan of
  /// `columnNames` would touch, reading Parquet metadata during setup but
  /// decoding nothing.
  kParquetRanges,

  /// Reads every byte of every object, without reading Parquet metadata or
  /// decoding anything.
  kFile,
};

/// Configures one raw-read run.
struct CudfRawReadOptions {
  /// Selects the raw range source.
  CudfRawReadMode mode;

  /// Bounds the number of files read concurrently.
  int32_t numWorkers;

  /// Bounds each whole-file device allocation and read request. Ignored in
  /// exact-range mode, which reads whole column chunks.
  uint64_t readSizeBytes;

  /// Physical Parquet column names to fetch in exact-range mode.
  std::vector<std::string> columnNames;
};

/// Payload counters and timings of one raw-read run.
struct CudfRawReadStats {
  /// Requested payload bytes.
  uint64_t requestedBytes{0};

  /// Successfully completed payload bytes.
  uint64_t completedBytes{0};

  /// Logical ranges passed to the fetch helper.
  uint64_t logicalRanges{0};

  /// Post-coalescing datasource submissions.
  uint64_t physicalRequests{0};

  /// Number of non-empty files in the run.
  uint64_t numFiles{0};

  /// Number of workers that can be occupied by the file set.
  uint64_t effectiveWorkers{0};

  /// Selected row groups summed across files. Zero in whole-file mode.
  uint64_t selectedRowGroups{0};

  /// Datasource creation, size discovery and, in exact-range mode, Parquet
  /// footer reading and range selection time.
  uint64_t setupNanos{0};

  /// Payload-pass wall time across all workers.
  uint64_t elapsedNanos{0};

  /// Cumulative range-fetch wall time from IoStats.
  uint64_t readWallNanos{0};
};

/// Splits `[0, fileSize)` into adjacent, non-overlapping ranges of at most
/// `readSizeBytes` bytes each, truncating only the final range. Performs no
/// I/O.
///
/// @param fileSize Object size in bytes. Must be positive.
/// @param readSizeBytes Upper bound on each range. Must be positive.
///
/// @return Ranges covering every byte of the object exactly once, in
/// ascending offset order, none of them empty.
///
/// @throws VeloxUserError when either argument is zero or exceeds what cuDF's
/// signed range offset and size can represent.
std::vector<cudf::io::text::byte_range_info> makeWholeFileRanges(
    uint64_t fileSize,
    uint64_t readSizeBytes);

/// Reads raw bytes of every object in `paths` through a cuDF datasource,
/// discarding them as soon as each read completes.
///
/// Files are the unit of scheduling: at most
/// `min(options.numWorkers, paths.size())` workers run, each claiming whole
/// files. In whole-file mode a worker reads one bounded range at a time, so
/// device memory is bounded by one `options.readSizeBytes` buffer per worker.
/// In exact-range mode a worker fetches all of a file's selected column chunks
/// at once, which allocates one compressed buffer per file exactly as the
/// experimental decode path does; `options.readSizeBytes` does not apply.
///
/// Everything a run needs before it can transfer payload happens up front and
/// is timed separately: datasource creation and size discovery in both modes,
/// plus Parquet footer reading and range selection in exact-range mode. Footer
/// bytes are therefore excluded from the payload counters.
///
/// The runner accepts any URI a cuDF datasource can open, including local
/// paths; callers that require remote storage validate the scheme themselves.
/// A leading `s3a:` is rewritten to `s3:` because KvikIO does not accept the
/// Hadoop-style scheme.
///
/// @param paths Object URIs to read. Must be non-empty.
/// @param options Range source, worker bound, read-size bound and, in
/// exact-range mode, the physical column names to select.
/// @param memoryResource Device memory resource for the read destinations.
///
/// @return Payload counters and timings of the run.
///
/// @throws VeloxUserError when `paths` is empty, when `options.numWorkers` is
/// not positive, when exact-range mode is asked for without column names, when
/// an object is empty or cannot be opened, or when an object selects no row
/// groups or no ranges.
/// @throws VeloxRuntimeError naming the URI and the ranges when a read fails.
/// Every worker is joined before the failure is rethrown, and no statistics
/// are returned for a failed run.
CudfRawReadStats runCudfRawRead(
    const std::vector<std::string>& paths,
    const CudfRawReadOptions& options,
    rmm::device_async_resource_ref memoryResource);

} // namespace facebook::velox::cudf_velox
