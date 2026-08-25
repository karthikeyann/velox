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
#include "velox/experimental/cudf/benchmarks/CudfRawReadBenchmark.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/file/File.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <folly/container/F14Map.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

using ::facebook::velox::cudf_velox::connector::hive::fetchByteRangesAsync;
using ::facebook::velox::cudf_velox::connector::hive::
    kColumnChunkCompletedBytes;
using ::facebook::velox::cudf_velox::connector::hive::kColumnChunkLogicalRanges;
using ::facebook::velox::cudf_velox::connector::hive::
    kColumnChunkPhysicalRequests;
using ::facebook::velox::cudf_velox::connector::hive::kColumnChunkReadWallNanos;
using ::facebook::velox::cudf_velox::connector::hive::
    kColumnChunkRequestedBytes;
using ::facebook::velox::cudf_velox::connector::hive::ParquetColumnChunkRanges;
using ::facebook::velox::cudf_velox::connector::hive::
    selectParquetColumnChunkRanges;

using SteadyClock = std::chrono::steady_clock;

// Largest offset or size cuDF's byte_range_info can hold, which is narrower
// than the byte counts this API accepts.
constexpr uint64_t kMaxCudfRange =
    static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

// One object of the run, prepared before the payload pass starts.
struct RawReadFile {
  // URI as the caller supplied it, so failures name what the caller asked for
  // rather than the normalized form.
  std::string uri;

  // Datasource opened for 'uri'. Stays alive for the whole run.
  std::shared_ptr<cudf::io::datasource> dataSource;

  // Ranges to read, grouped into the batches the payload pass submits. Each
  // batch becomes one fetch, so grouping decides how many compressed buffers a
  // worker holds at a time.
  std::vector<std::vector<cudf::io::text::byte_range_info>> batches;

  // Row groups the ranges were selected from. Zero in whole-file mode, which
  // reads no Parquet metadata.
  uint64_t numRowGroups{0};
};

// Joins every worker on destruction, so a failure while starting workers
// cannot leave running threads referring to state that is about to die.
class WorkerJoiner {
 public:
  explicit WorkerJoiner(std::vector<std::thread>& workers)
      : workers_(workers) {}

  WorkerJoiner(const WorkerJoiner&) = delete;
  WorkerJoiner& operator=(const WorkerJoiner&) = delete;

  ~WorkerJoiner() {
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

 private:
  std::vector<std::thread>& workers_;
};

uint64_t nanosSince(SteadyClock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          SteadyClock::now() - start)
          .count());
}

// KvikIO does not accept the Hadoop-style "s3a:" scheme, so rewrite it the
// same way CudfHiveDataSource does before handing the URI to cuDF.
std::string normalizeDatasourceUri(const std::string& uri) {
  constexpr std::string_view kS3aPrefix{"s3a:"};
  if (uri.compare(0, kS3aPrefix.size(), kS3aPrefix) != 0) {
    return uri;
  }
  std::string normalized = uri;
  normalized.erase(kS3aPrefix.size() - 2, 1);
  return normalized;
}

// Opens 'uri' as a cuDF datasource and reports its size. Failures name the URI
// as the caller supplied it.
std::shared_ptr<cudf::io::datasource> openDataSource(
    const std::string& uri,
    uint64_t& fileSize) {
  std::shared_ptr<cudf::io::datasource> dataSource;
  try {
    auto dataSources = cudf::io::make_datasources(
        cudf::io::source_info{normalizeDatasourceUri(uri)});
    VELOX_USER_CHECK_EQ(
        dataSources.size(),
        1,
        "Expected exactly one datasource for '{}' but cuDF made {}.",
        uri,
        dataSources.size());
    VELOX_USER_CHECK(
        dataSources.front() != nullptr,
        "cuDF made a null datasource for '{}'.",
        uri);
    dataSource = std::move(dataSources.front());
    fileSize = dataSource->size();
  } catch (const VeloxException&) {
    throw;
  } catch (const std::exception& e) {
    VELOX_USER_FAIL("Failed to open '{}': {}", uri, e.what());
  }

  VELOX_USER_CHECK_GT(
      fileSize,
      uint64_t{0},
      "Object '{}' is empty, so there is nothing to read.",
      uri);
  return dataSource;
}

// Opens 'uri' and plans its whole-file ranges, one bounded range per batch so
// that a worker holds a single read destination at a time. This is all the
// work whole-file mode does before the payload pass, and it is what setupNanos
// measures.
RawReadFile openWholeFile(const std::string& uri, uint64_t readSizeBytes) {
  uint64_t fileSize{0};
  auto dataSource = openDataSource(uri, fileSize);

  std::vector<std::vector<cudf::io::text::byte_range_info>> batches;
  for (const auto& range : makeWholeFileRanges(fileSize, readSizeBytes)) {
    batches.push_back({range});
  }
  return RawReadFile{
      .uri = uri,
      .dataSource = std::move(dataSource),
      .batches = std::move(batches)};
}

// Opens 'uri', reads its Parquet footer and selects the compressed column
// chunks that a scan of 'columnNames' would read, without decoding anything.
// All of it happens before the payload pass, so footer bytes never reach the
// payload counters. The reader, footer and options are temporary: only the
// selected ranges and their row-group count outlive this call.
RawReadFile openParquetRangeFile(
    const std::string& uri,
    const std::vector<std::string>& columnNames) {
  uint64_t fileSize{0};
  auto dataSource = openDataSource(uri, fileSize);

  ParquetColumnChunkRanges selected;
  try {
    // Wrap the datasource the run keeps without transferring ownership.
    std::vector<std::unique_ptr<cudf::io::datasource>> footerSources;
    footerSources.push_back(cudf::io::datasource::create(dataSource.get()));
    auto footers = cudf::io::read_parquet_footers(footerSources);
    VELOX_USER_CHECK_EQ(
        footers.size(),
        1,
        "Expected exactly one Parquet footer in '{}' but read {}.",
        uri,
        footers.size());

    // Range selection depends only on the physical names, the row groups, the
    // split window and the filter, so the options carry the projection and the
    // schema tolerance the cuDF TPC-H decode path uses and nothing else.
    auto options = cudf::io::parquet_reader_options::builder(
                       cudf::io::source_info{dataSource.get()})
                       .allow_mismatched_pq_schemas(true)
                       .build();
    options.set_column_names(columnNames);

    cudf::io::parquet::experimental::hybrid_scan_reader reader(
        footers.front(), options);
    selected = selectParquetColumnChunkRanges(
        reader, options, cudfGlobalStreamPool().get_stream());
  } catch (const VeloxException&) {
    throw;
  } catch (const std::exception& e) {
    VELOX_USER_FAIL(
        "Failed to select Parquet ranges in '{}': {}", uri, e.what());
  }

  VELOX_USER_CHECK_GT(
      selected.numRowGroups(),
      uint64_t{0},
      "Object '{}' selected no row groups for the requested columns.",
      uri);
  VELOX_USER_CHECK(
      !selected.ranges.empty(),
      "Object '{}' selected no column-chunk ranges for the requested columns.",
      uri);

  // One batch holding every selected chunk, so the fetch helper allocates one
  // compressed buffer and coalesces adjacent chunks exactly as decode does.
  const uint64_t numRowGroups = selected.numRowGroups();
  std::vector<std::vector<cudf::io::text::byte_range_info>> batches;
  batches.push_back(std::move(selected.ranges));
  return RawReadFile{
      .uri = uri,
      .dataSource = std::move(dataSource),
      .batches = std::move(batches),
      .numRowGroups = numRowGroups};
}

// Attributes 'cause' to the object it came from. The fetch helper names the
// offset and size of a failed read but not the object, and an allocation
// failure names neither.
std::exception_ptr batchFailure(
    const std::string& uri,
    const std::vector<cudf::io::text::byte_range_info>& batch,
    std::string_view cause) {
  const int64_t offset = batch.empty() ? 0 : batch.front().offset();
  const int64_t size = std::accumulate(
      batch.begin(), batch.end(), int64_t{0}, [](int64_t sum, auto& range) {
        return sum + range.size();
      });
  try {
    VELOX_FAIL(
        "Raw read of '{}' failed at offset {} over {} range(s) totalling {} "
        "bytes: {}",
        uri,
        offset,
        batch.size(),
        size,
        cause);
  } catch (...) {
    return std::current_exception();
  }
}

// Returns the sum recorded for 'name'. The payload pass always publishes
// every counter it is asked about, so a missing one means the run did not
// measure what it reports.
uint64_t metricSum(
    const folly::F14FastMap<std::string, RuntimeMetric>& stats,
    std::string_view name) {
  const auto it = stats.find(std::string(name));
  VELOX_CHECK(
      it != stats.end(), "Raw read did not record the '{}' counter.", name);
  VELOX_CHECK_GE(
      it->second.sum, 0, "Raw read recorded a negative '{}' counter.", name);
  return static_cast<uint64_t>(it->second.sum);
}

} // namespace

std::vector<cudf::io::text::byte_range_info> makeWholeFileRanges(
    uint64_t fileSize,
    uint64_t readSizeBytes) {
  VELOX_USER_CHECK_GT(
      fileSize, uint64_t{0}, "Whole-file read needs a positive file size.");
  VELOX_USER_CHECK_GT(
      readSizeBytes,
      uint64_t{0},
      "Whole-file read needs a positive read size.");
  VELOX_USER_CHECK_LE(
      fileSize,
      kMaxCudfRange,
      "Whole-file read supports a file size of at most {} bytes.",
      kMaxCudfRange);
  VELOX_USER_CHECK_LE(
      readSizeBytes,
      kMaxCudfRange,
      "Whole-file read supports a read size of at most {} bytes.",
      kMaxCudfRange);

  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.reserve((fileSize + readSizeBytes - 1) / readSizeBytes);
  for (uint64_t offset = 0; offset < fileSize; offset += readSizeBytes) {
    const uint64_t size = std::min(readSizeBytes, fileSize - offset);
    ranges.emplace_back(
        static_cast<int64_t>(offset), static_cast<int64_t>(size));
  }
  return ranges;
}

CudfRawReadStats runCudfRawRead(
    const std::vector<std::string>& paths,
    const CudfRawReadOptions& options,
    rmm::device_async_resource_ref memoryResource) {
  VELOX_USER_CHECK(
      !paths.empty(), "Raw read needs at least one object but got no paths.");
  VELOX_USER_CHECK_GT(
      options.numWorkers,
      0,
      "Raw read needs a positive number of workers but got {}.",
      options.numWorkers);

  // One exhaustive dispatch over the mode decides both what each mode needs
  // and how it opens an object, so a mode added later has to be handled here
  // rather than silently reading whole objects.
  std::function<RawReadFile(const std::string&)> openFile;
  switch (options.mode) {
    case CudfRawReadMode::kParquetRanges:
      VELOX_USER_CHECK(
          !options.columnNames.empty(),
          "Raw read of exact Parquet ranges needs the physical column names to "
          "select but got none.");
      openFile = [&](const std::string& path) {
        return openParquetRangeFile(path, options.columnNames);
      };
      break;
    case CudfRawReadMode::kFile:
      openFile = [&](const std::string& path) {
        return openWholeFile(path, options.readSizeBytes);
      };
      break;
  }
  VELOX_CHECK(
      openFile != nullptr,
      "Unhandled raw read mode {}.",
      static_cast<int>(options.mode));

  const auto setupStart = SteadyClock::now();
  std::vector<RawReadFile> files;
  files.reserve(paths.size());
  for (const auto& path : paths) {
    files.push_back(openFile(path));
  }
  const auto setupNanos = nanosSince(setupStart);

  const uint64_t numFiles = files.size();
  const uint64_t effectiveWorkers =
      std::min<uint64_t>(static_cast<uint64_t>(options.numWorkers), numFiles);
  const uint64_t selectedRowGroups = std::accumulate(
      files.begin(), files.end(), uint64_t{0}, [](uint64_t sum, auto& file) {
        return sum + file.numRowGroups;
      });

  IoStats ioStats;
  std::atomic<uint64_t> nextFile{0};
  std::atomic<bool> stopped{false};
  std::mutex failureMutex;
  std::exception_ptr firstFailure;

  const auto retainFailure = [&](std::exception_ptr failure) {
    {
      std::lock_guard<std::mutex> lock(failureMutex);
      if (firstFailure == nullptr) {
        firstFailure = std::move(failure);
      }
    }
    stopped.store(true, std::memory_order_relaxed);
  };

  // Files are the unit of scheduling: one worker reads one file at a time so
  // that effectiveWorkers describes the achievable concurrency and so that a
  // worker holds at most one read destination.
  const auto readFiles = [&]() {
    // An exception leaving a worker would terminate the process, so anything
    // raised outside a range fetch is retained the same way a failed fetch is.
    try {
      const auto stream = cudfGlobalStreamPool().get_stream();
      while (!stopped.load(std::memory_order_relaxed)) {
        const auto fileIndex = nextFile.fetch_add(1, std::memory_order_relaxed);
        if (fileIndex >= numFiles) {
          return;
        }
        const auto& file = files[fileIndex];
        for (const auto& batch : file.batches) {
          if (stopped.load(std::memory_order_relaxed)) {
            return;
          }
          try {
            auto ioData = fetchByteRangesAsync(
                file.dataSource, batch, stream, memoryResource, &ioStats);
            // get() rather than wait() so a failed or short read surfaces
            // here, and so no read is still writing when the buffers are
            // released at the end of this iteration.
            std::get<2>(ioData).get();
          } catch (const std::exception& e) {
            retainFailure(batchFailure(file.uri, batch, e.what()));
            return;
          } catch (...) {
            retainFailure(batchFailure(file.uri, batch, "unknown exception"));
            return;
          }
        }
      }
    } catch (...) {
      retainFailure(std::current_exception());
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(effectiveWorkers);

  const auto payloadStart = SteadyClock::now();
  {
    WorkerJoiner joiner(workers);
    try {
      for (uint64_t worker = 0; worker < effectiveWorkers; ++worker) {
        workers.emplace_back(readFiles);
      }
    } catch (...) {
      // Started workers would otherwise hold the join until they have read
      // every remaining file.
      stopped.store(true, std::memory_order_relaxed);
      throw;
    }
  }
  const auto elapsedNanos = nanosSince(payloadStart);

  // Every worker has been joined, so the failure and the counters are stable.
  if (firstFailure != nullptr) {
    std::rethrow_exception(firstFailure);
  }

  const auto snapshot = ioStats.stats();
  CudfRawReadStats stats{
      .requestedBytes = metricSum(snapshot, kColumnChunkRequestedBytes),
      .completedBytes = metricSum(snapshot, kColumnChunkCompletedBytes),
      .logicalRanges = metricSum(snapshot, kColumnChunkLogicalRanges),
      .physicalRequests = metricSum(snapshot, kColumnChunkPhysicalRequests),
      .numFiles = numFiles,
      .effectiveWorkers = effectiveWorkers,
      .selectedRowGroups = selectedRowGroups,
      .setupNanos = setupNanos,
      .elapsedNanos = elapsedNanos,
      .readWallNanos = metricSum(snapshot, kColumnChunkReadWallNanos)};

  VELOX_CHECK_EQ(
      stats.requestedBytes,
      stats.completedBytes,
      "Raw read requested {} bytes but completed {}.",
      stats.requestedBytes,
      stats.completedBytes);
  return stats;
}

} // namespace facebook::velox::cudf_velox
