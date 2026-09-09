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
#include "velox/experimental/cudf/connectors/hive/CudfDecodedColumnCache.h"
#include "velox/experimental/ucx-exchange/UcxColumnCodec.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/memory_pool>
#include <cuda_runtime_api.h>

#include <folly/ScopeGuard.h>
#include <gflags/gflags.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <tuple>
#include <unordered_map>

DEFINE_bool(
    cudf_decoded_cache_skip_full_range_slice,
    false,
    "Reuse full cached column views without recomputing slice null counts");
DEFINE_bool(
    cudf_gpu_cache_store_packed,
    false,
    "Store encoded host-cache payloads in the GPU tier when available");
DEFINE_uint64(
    cudf_gpu_cache_raw_prefix_bytes,
    0,
    "Keep an initial GPU-cache byte prefix uncompressed before packed admission");
DEFINE_string(
    cudf_gpu_cache_packed_types,
    "all",
    "Types eligible for packed GPU storage: all, nonfloating, integral, smallstrings, or lowcardstrings");
DEFINE_double(
    cudf_gpu_cache_packed_max_fraction,
    1.0,
    "Maximum encoded/original packed byte ratio for packed GPU admission");
DEFINE_double(
    cudf_gpu_cache_min_host_fraction,
    0.0,
    "Skip GPU admission for exact host-cache chunks whose compressed/raw byte ratio is below this value; zero disables");
DEFINE_bool(
    cudf_gpu_cache_scaled_float,
    false,
    "Store DOUBLE columns as small scaled integers only after full bit-exact reconstruction validation");
DEFINE_bool(
    cudf_decoded_cache_single_stream_restore,
    false,
    "Diagnostic: place host-cache transfer and restore on the consumer stream");
DEFINE_bool(
    cudf_cache_restore_packed_views,
    false,
    "Retain restored packed buffers in composite cache leases without a final column copy");
DEFINE_bool(
    cudf_cache_stream_ordered_release,
    false,
    "Order raw GPU cache storage release with CUDA events instead of host synchronization");

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

constexpr size_t kPackStagingBytes = 16ULL << 20;

struct ScaledFloatColumn {
  std::unique_ptr<cudf::column> column;
  double scale{0};
};

bool isLowCardinalityStringColumn(
    cudf::column_view column,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (column.type().id() != cudf::type_id::STRING || column.size() == 0) {
    return false;
  }
  if (column.null_count() == column.size()) {
    return true;
  }
  const auto sample = column.size() <= 1024
      ? column
      : cudf::slice(column, {0, 1024}, stream).front();
  auto dictionary = cudf::distinct(
      cudf::table_view{{sample}},
      {0},
      cudf::duplicate_keep_option::KEEP_ANY,
      cudf::null_equality::EQUAL,
      cudf::nan_equality::ALL_EQUAL,
      stream,
      mr);
  if (dictionary->num_rows() > 64) {
    return false;
  }
  // Sampling only proposes a small dictionary. Validate every non-null value
  // before classifying the complete range; unseen rare labels fall back raw.
  auto membership =
      cudf::contains(dictionary->view().column(0), column, stream, mr);
  auto all = cudf::reduce(
      membership->view(),
      *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  return all->is_valid(stream) &&
      static_cast<cudf::numeric_scalar<bool>&>(*all).value(stream);
}

std::unique_ptr<cudf::column> decodeScaledFloat(
    cudf::column_view input,
    double scale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const cudf::data_type outputType{cudf::type_id::FLOAT64};
  if (scale == 1) {
    return cudf::cast(input, outputType, stream, mr);
  }
  cudf::numeric_scalar<double> divisor(scale, true, stream, mr);
  return cudf::binary_operation(
      input, divisor, cudf::binary_operator::DIV, outputType, stream, mr);
}

ScaledFloatColumn tryEncodeScaledFloat(
    cudf::column_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (input.type().id() != cudf::type_id::FLOAT64 || input.size() == 0 ||
      input.null_count() == input.size()) {
    return {};
  }
  const auto bitView = [](cudf::column_view view) {
    return cudf::column_view(
        cudf::data_type{cudf::type_id::INT64},
        view.size(),
        view.head(),
        view.null_mask(),
        view.null_count(),
        view.offset());
  };
  for (double scale : {1.0, 10.0, 100.0, 1000.0, 1000000.0}) {
    cudf::numeric_scalar<double> multiplier(scale, true, stream, mr);
    auto scaled = cudf::binary_operation(
        input,
        multiplier,
        cudf::binary_operator::MUL,
        input.type(),
        stream,
        mr);
    auto rounded =
        cudf::unary_operation(*scaled, cudf::unary_operator::RINT, stream, mr);
    scaled.reset();
    auto [minimum, maximum] = cudf::minmax(*rounded, stream, mr);
    if (!minimum->is_valid(stream) || !maximum->is_valid(stream)) {
      return {};
    }
    const auto minValue =
        static_cast<cudf::numeric_scalar<double>&>(*minimum).value(stream);
    const auto maxValue =
        static_cast<cudf::numeric_scalar<double>&>(*maximum).value(stream);
    if (!std::isfinite(minValue) || !std::isfinite(maxValue) ||
        minValue < std::numeric_limits<int32_t>::min() ||
        maxValue > std::numeric_limits<int32_t>::max()) {
      continue;
    }
    auto storageType = cudf::type_id::INT32;
    if (minValue >= std::numeric_limits<int8_t>::min() &&
        maxValue <= std::numeric_limits<int8_t>::max()) {
      storageType = cudf::type_id::INT8;
    } else if (
        minValue >= std::numeric_limits<int16_t>::min() &&
        maxValue <= std::numeric_limits<int16_t>::max()) {
      storageType = cudf::type_id::INT16;
    }
    auto encoded =
        cudf::cast(*rounded, cudf::data_type{storageType}, stream, mr);
    rounded.reset();
    auto reconstructed = decodeScaledFloat(*encoded, scale, stream, mr);
    auto identical = cudf::binary_operation(
        bitView(input),
        bitView(*reconstructed),
        cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto allEqual = cudf::reduce(
        *identical,
        *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    if (allEqual->is_valid(stream) &&
        static_cast<cudf::numeric_scalar<bool>&>(*allEqual).value(stream)) {
      return {std::move(encoded), scale};
    }
  }
  return {};
}

cudf::column_view cacheSlice(
    cudf::column_view column,
    int64_t first,
    int64_t last,
    rmm::cuda_stream_view stream) {
  // The cached view already has the correct null count and child offsets.
  // A full-range slice needlessly launches segmented null-count kernels and
  // synchronizes their results back to the host, even for all-valid masks.
  if (FLAGS_cudf_decoded_cache_skip_full_range_slice && first == 0 &&
      last == column.size()) {
    return column;
  }
  return cudf::slice(
             column,
             {static_cast<cudf::size_type>(first),
              static_cast<cudf::size_type>(last)},
             stream)
      .front();
}

template <typename T>
void hashCombine(size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct FileKeyHash {
  size_t operator()(const CudfDecodedColumnCache::FileKey& key) const {
    size_t seed = 0;
    hashCombine(seed, key.connectorId);
    hashCombine(seed, key.filePath);
    return seed;
  }
};

struct RowGroupSelectionKeyHash {
  size_t operator()(
      const CudfDecodedColumnCache::RowGroupSelectionKey& key) const {
    size_t seed = FileKeyHash{}(key.file);
    hashCombine(seed, key.splitStart);
    hashCombine(seed, key.splitSize);
    hashCombine(seed, key.filterKey);
    hashCombine(seed, static_cast<int>(key.timestampType));
    hashCombine(seed, key.usePandasMetadata);
    hashCombine(seed, key.useArrowSchema);
    hashCombine(seed, key.allowMismatchedSchemas);
    return seed;
  }
};

struct ColumnKeyHash {
  size_t operator()(const CudfDecodedColumnCache::ColumnKey& key) const {
    size_t seed = FileKeyHash{}(key.file);
    hashCombine(seed, key.deviceId);
    hashCombine(seed, key.columnName);
    hashCombine(seed, key.veloxType);
    hashCombine(seed, static_cast<int>(key.timestampType));
    hashCombine(seed, key.usePandasMetadata);
    hashCombine(seed, key.useArrowSchema);
    hashCombine(seed, key.allowMismatchedSchemas);
    return seed;
  }
};

cuda::memory_pool_properties pinnedPoolProperties(uint64_t maxPinnedBytes) {
  cuda::memory_pool_properties properties;
  properties.release_threshold = maxPinnedBytes;
  properties.max_pool_size = maxPinnedBytes;
  return properties;
}

struct CacheConfiguration {
  std::mutex mutex;
  uint64_t maxPinnedBytes{CudfDecodedColumnCache::kMaxPinnedBytes};
  uint64_t maxGpuBytes{CudfDecodedColumnCache::kMaxGpuBytes};
  bool initialized{false};
};

CacheConfiguration& cacheConfiguration() {
  static auto* configuration = new CacheConfiguration();
  return *configuration;
}

uint64_t takeConfiguredMaxPinnedBytes() {
  auto& configuration = cacheConfiguration();
  std::lock_guard<std::mutex> lock(configuration.mutex);
  configuration.initialized = true;
  return configuration.maxPinnedBytes;
}

uint64_t takeConfiguredMaxGpuBytes() {
  auto& configuration = cacheConfiguration();
  std::lock_guard<std::mutex> lock(configuration.mutex);
  return configuration.maxGpuBytes;
}

int currentNumaNode() {
  unsigned cpu = 0;
  unsigned node = 0;
  VELOX_CHECK_EQ(
      ::syscall(SYS_getcpu, &cpu, &node, nullptr),
      0,
      "Failed to determine the NUMA node for the decoded column cache");
  return static_cast<int>(node);
}

class CachePipelineEvent {
 public:
  CachePipelineEvent() {
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  ~CachePipelineEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  CachePipelineEvent(const CachePipelineEvent&) = delete;
  CachePipelineEvent& operator=(const CachePipelineEvent&) = delete;

  void record(rmm::cuda_stream_view stream) const {
    CUDF_CUDA_TRY(cudaEventRecord(event_, stream.value()));
  }

  void wait(rmm::cuda_stream_view stream) const {
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.value(), event_, 0));
  }

 private:
  cudaEvent_t event_{nullptr};
};

} // namespace

class PackedColumnCompression {
 public:
  PackedColumnCompression(
      std::vector<ucx_exchange::EncodedRegion> regions,
      size_t uncompressedBytes)
      : regions_(std::move(regions)), uncompressedBytes_(uncompressedBytes) {}

 private:
  friend class CudfDecodedColumnCache;
  friend class PinnedColumnChunk;

  std::vector<ucx_exchange::EncodedRegion> regions_;
  size_t uncompressedBytes_;
};

struct CudfDecodedColumnCache::GpuColumnChunk {
  int64_t firstRow;
  int64_t lastRow;
  uint64_t bytes;
  std::unique_ptr<cudf::column> column;
  // Exactly one representation is used. Packed payload metadata and codec
  // regions remain immutable through the shared host chunk's lifetime.
  ColumnRangePtr packedMetadata;
  rmm::device_buffer packedData;
  double scaledFloatScale{0};
  rmm::cuda_stream_view allocationStream;
};

class PinnedHostAllocation {
 public:
  PinnedHostAllocation(
      cuda::pinned_memory_pool* pool,
      std::atomic<uint64_t>* allocatedBytes,
      void* data,
      size_t size)
      : pool_(pool),
        allocatedBytes_(allocatedBytes),
        data_(data),
        size_(size) {}

  ~PinnedHostAllocation() {
    if (data_ != nullptr) {
      pool_->deallocate_sync(data_, size_);
      allocatedBytes_->fetch_sub(size_, std::memory_order_relaxed);
    }
  }

  const void* data() const {
    return data_;
  }

  size_t size() const {
    return size_;
  }

 private:
  cuda::pinned_memory_pool* pool_;
  std::atomic<uint64_t>* allocatedBytes_;
  void* data_;
  size_t size_;
};

struct CudfDecodedColumnCache::Impl {
  Impl(uint64_t maxPinnedBytes, uint64_t maxGpuBytes)
      : maxPinnedBytes(maxPinnedBytes),
        maxGpuBytes(maxGpuBytes),
        pinnedPool(currentNumaNode(), pinnedPoolProperties(maxPinnedBytes)) {}

  std::shared_ptr<const PinnedHostAllocation> allocate(size_t size) {
    if (size == 0) {
      return std::make_shared<const PinnedHostAllocation>(
          &pinnedPool, &allocatedBytes, nullptr, 0);
    }

    auto current = allocatedBytes.load(std::memory_order_relaxed);
    do {
      if (size > maxPinnedBytes - current) {
        hostAdmissionRejectedAllocations.fetch_add(
            1, std::memory_order_relaxed);
        hostAdmissionRejectedBytes.fetch_add(size, std::memory_order_relaxed);
        return nullptr;
      }
    } while (not allocatedBytes.compare_exchange_weak(
        current,
        current + size,
        std::memory_order_relaxed,
        std::memory_order_relaxed));

    try {
      auto* data = pinnedPool.allocate_sync(size);
      return std::make_shared<const PinnedHostAllocation>(
          &pinnedPool, &allocatedBytes, data, size);
    } catch (const std::exception& error) {
      allocatedBytes.fetch_sub(size, std::memory_order_relaxed);
      LOG(WARNING) << "Skipping decoded column cache admission after pinned "
                      "allocation failed: "
                   << error.what();
      return nullptr;
    }
  }

  std::optional<std::vector<CoveredColumnRange>> findColumnRangesLocked(
      const ColumnKey& key,
      int64_t firstRow,
      int64_t lastRow) const {
    const auto it = columns.find(key);
    if (it == columns.end()) {
      return std::nullopt;
    }

    std::vector<CoveredColumnRange> result;
    auto cursor = firstRow;
    while (cursor < lastRow) {
      ColumnRangePtr best;
      for (const auto& chunk : it->second) {
        if (chunk->firstRow() > cursor) {
          break;
        }
        if (chunk->lastRow() > cursor and
            (not best or chunk->lastRow() > best->lastRow())) {
          best = chunk;
        }
      }
      if (not best) {
        return std::nullopt;
      }

      const auto coveredUntil = std::min(lastRow, best->lastRow());
      result.push_back({best, cursor, coveredUntil});
      cursor = coveredUntil;
    }
    return result;
  }

  struct CoveredGpuColumnRange {
    std::shared_ptr<const GpuColumnChunk> chunk;
    int64_t firstRow;
    int64_t lastRow;
  };

  std::optional<std::vector<CoveredGpuColumnRange>> findGpuColumnRangesLocked(
      const ColumnKey& key,
      int64_t firstRow,
      int64_t lastRow) const {
    const auto it = gpuColumns.find(key);
    if (it == gpuColumns.end()) {
      return std::nullopt;
    }

    std::vector<CoveredGpuColumnRange> result;
    auto cursor = firstRow;
    while (cursor < lastRow) {
      std::shared_ptr<const GpuColumnChunk> best;
      for (const auto& chunk : it->second) {
        if (chunk->firstRow > cursor) {
          break;
        }
        if (chunk->lastRow > cursor and
            (not best or chunk->lastRow > best->lastRow)) {
          best = chunk;
        }
      }
      if (not best) {
        return std::nullopt;
      }

      const auto coveredUntil = std::min(lastRow, best->lastRow);
      result.push_back({best, cursor, coveredUntil});
      cursor = coveredUntil;
    }
    return result;
  }

  bool reserveGpuBytes(uint64_t size) {
    if (maxGpuBytes == 0 or size > maxGpuBytes) {
      return false;
    }
    auto current = gpuBytes.load(std::memory_order_relaxed);
    do {
      if (size > maxGpuBytes - current) {
        return false;
      }
    } while (not gpuBytes.compare_exchange_weak(
        current,
        current + size,
        std::memory_order_relaxed,
        std::memory_order_relaxed));
    return true;
  }

  mutable std::mutex mutex;
  const uint64_t maxPinnedBytes;
  const uint64_t maxGpuBytes;
  cuda::pinned_memory_pool pinnedPool;
  std::atomic<uint64_t> allocatedBytes{0};
  std::atomic<uint64_t> hostAdmissionRejectedAllocations{0};
  std::atomic<uint64_t> hostAdmissionRejectedBytes{0};
  std::atomic<uint64_t> insertedUncompressedBytes{0};
  std::atomic<uint64_t> insertedStoredBytes{0};
  std::atomic<uint64_t> insertedCompressedRanges{0};
  std::atomic<uint64_t> insertedRawRanges{0};
  std::atomic<uint64_t> compressionAttempts{0};
  std::atomic<uint64_t> compressionEncodeNanos{0};
  std::atomic<uint64_t> restoreCalls{0};
  std::atomic<uint64_t> restoredStoredBytes{0};
  std::atomic<uint64_t> restoredUncompressedBytes{0};
  std::atomic<uint64_t> decompressionNanos{0};
  std::atomic<uint64_t> pipelinedRestoreBatches{0};
  std::atomic<uint64_t> gpuBytes{0};
  std::atomic<uint64_t> gpuInsertedBytes{0};
  std::atomic<uint64_t> gpuInsertedRanges{0};
  std::atomic<uint64_t> gpuAdmissionRejectedRanges{0};
  std::atomic<uint64_t> gpuAdmissionPolicySkippedRanges{0};
  std::atomic<uint64_t> gpuRestoreCalls{0};
  std::atomic<uint64_t> gpuRestoredBytes{0};
  std::atomic<uint64_t> gpuRestoreBatches{0};
  std::atomic<uint64_t> gpuPackedInsertedBytes{0};
  std::atomic<uint64_t> gpuPackedRestoreCalls{0};
  std::atomic<uint64_t> gpuPackedRestoredStoredBytes{0};
  std::atomic<uint64_t> gpuPackedDecompressionNanos{0};
  std::atomic<uint64_t> gpuScaledInsertedBytes{0};
  std::atomic<uint64_t> gpuScaledInsertedRanges{0};
  std::atomic<uint64_t> gpuScaledRestoreCalls{0};
  std::unordered_map<FileKey, MetadataPtr, FileKeyHash> metadata;
  std::unordered_map<
      RowGroupSelectionKey,
      RowGroupSelectionPtr,
      RowGroupSelectionKeyHash>
      rowGroupSelections;
  std::unordered_map<ColumnKey, std::vector<ColumnRangePtr>, ColumnKeyHash>
      columns;
  std::unordered_map<
      ColumnKey,
      std::vector<std::shared_ptr<const GpuColumnChunk>>,
      ColumnKeyHash>
      gpuColumns;
};

size_t PinnedColumnChunk::packedSize() const {
  return data_->size();
}

size_t PinnedColumnChunk::uncompressedPackedSize() const {
  return compression_ ? compression_->uncompressedBytes_ : packedSize();
}

bool PinnedColumnChunk::compressed() const {
  return compression_ != nullptr;
}

const void* PinnedColumnChunk::pinnedData() const {
  return data_->data();
}

CudfDecodedColumnCache::CudfDecodedColumnCache()
    : impl_(
          std::make_unique<Impl>(
              takeConfiguredMaxPinnedBytes(),
              takeConfiguredMaxGpuBytes())) {}

CudfDecodedColumnCache::~CudfDecodedColumnCache() = default;

CudfDecodedColumnCache& CudfDecodedColumnCache::instance() {
  // Intentionally process-lifetime: avoids CUDA pool destruction during static
  // teardown and implements the prototype's non-evicting lifetime.
  static auto* cache = new CudfDecodedColumnCache();
  return *cache;
}

void CudfDecodedColumnCache::configureMaxPinnedBytes(uint64_t maxPinnedBytes) {
  VELOX_USER_CHECK_GT(
      maxPinnedBytes, 0, "Decoded column cache limit must be positive");
  auto& configuration = cacheConfiguration();
  std::lock_guard<std::mutex> lock(configuration.mutex);
  VELOX_USER_CHECK(
      not configuration.initialized,
      "Decoded column cache limit must be configured before first use");
  configuration.maxPinnedBytes = maxPinnedBytes;
}

void CudfDecodedColumnCache::configureMaxGpuBytes(uint64_t maxGpuBytes) {
  auto& configuration = cacheConfiguration();
  std::lock_guard<std::mutex> lock(configuration.mutex);
  VELOX_USER_CHECK(
      not configuration.initialized,
      "Decoded column GPU cache limit must be configured before first use");
  configuration.maxGpuBytes = maxGpuBytes;
}

CudfDecodedColumnCache::CompressionMode
CudfDecodedColumnCache::compressionModeFromString(std::string_view value) {
  if (value == "none") {
    return CompressionMode::kNone;
  }
  if (value == "column") {
    return CompressionMode::kColumn;
  }
  VELOX_USER_CHECK_EQ(
      value,
      "column-advanced",
      "Unsupported decoded column cache compression '{}'. Expected none, column, or column-advanced",
      value);
  return CompressionMode::kColumnAdvanced;
}

CudfDecodedColumnCache::MetadataPtr CudfDecodedColumnCache::findMetadata(
    const FileKey& key) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto it = impl_->metadata.find(key);
  return it == impl_->metadata.end() ? nullptr : it->second;
}

CudfDecodedColumnCache::MetadataPtr
CudfDecodedColumnCache::insertMetadataIfAbsent(
    FileKey key,
    ParquetMetadataPtr metadata) {
  VELOX_CHECK_NOT_NULL(metadata);
  auto candidate = std::make_shared<CachedParquetFileMetadata>();
  candidate->parquetMetadata = std::move(metadata);
  const auto numRowGroups = candidate->parquetMetadata->row_groups.size();
  candidate->rowOffsets.reserve(numRowGroups + 1);
  candidate->rowOffsets.push_back(0);
  for (const auto& rowGroup : candidate->parquetMetadata->row_groups) {
    VELOX_CHECK_GE(rowGroup.num_rows, 0);
    candidate->rowOffsets.push_back(
        candidate->rowOffsets.back() + rowGroup.num_rows);
  }
  candidate->allRowGroups.resize(numRowGroups);
  std::iota(candidate->allRowGroups.begin(), candidate->allRowGroups.end(), 0);

  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->metadata.try_emplace(std::move(key), std::move(candidate))
      .first->second;
}

CudfDecodedColumnCache::RowGroupSelectionPtr
CudfDecodedColumnCache::findRowGroupSelection(
    const RowGroupSelectionKey& key) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto it = impl_->rowGroupSelections.find(key);
  return it == impl_->rowGroupSelections.end() ? nullptr : it->second;
}

CudfDecodedColumnCache::RowGroupSelectionPtr
CudfDecodedColumnCache::insertRowGroupSelectionIfAbsent(
    RowGroupSelectionKey key,
    std::vector<cudf::size_type> rowGroups) {
  auto candidate = std::make_shared<const std::vector<cudf::size_type>>(
      std::move(rowGroups));
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->rowGroupSelections
      .try_emplace(std::move(key), std::move(candidate))
      .first->second;
}

std::optional<std::vector<CoveredColumnRange>>
CudfDecodedColumnCache::findColumnRanges(
    const ColumnKey& key,
    int64_t firstRow,
    int64_t lastRow) const {
  VELOX_CHECK_LT(firstRow, lastRow, "Decoded cache range must be non-empty");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->findColumnRangesLocked(key, firstRow, lastRow);
}

bool CudfDecodedColumnCache::containsColumnRange(
    const ColumnKey& key,
    int64_t firstRow,
    int64_t lastRow) const {
  return findColumnRanges(key, firstRow, lastRow).has_value();
}

bool CudfDecodedColumnCache::insertColumnRangeIfAbsent(
    ColumnKey key,
    int64_t firstRow,
    int64_t lastRow,
    cudf::column_view column,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref tempMr,
    CompressionMode compressionMode,
    std::optional<rmm::device_async_resource_ref> gpuCacheMr) {
  VELOX_CHECK_LT(firstRow, lastRow, "Decoded cache range must be non-empty");
  VELOX_CHECK_EQ(
      lastRow - firstRow,
      column.size(),
      "Decoded cache range must match column size");
  if (containsColumnRange(key, firstRow, lastRow)) {
    return false;
  }

  const std::vector<cudf::column_view> columns{column};
  const auto table = cudf::table_view{columns};
  std::vector<uint8_t> metadata;
  std::shared_ptr<const PinnedHostAllocation> pinnedData;
  std::shared_ptr<const PackedColumnCompression> compression;
  size_t uncompressedPackedSize{0};
  uint64_t compressionEncodeNanos{0};
  bool compressionAttempted{false};

  const auto packRawChunked = [&]() -> bool {
    auto packer =
        cudf::chunked_pack::create(table, kPackStagingBytes, stream, tempMr);
    uncompressedPackedSize = packer->get_total_contiguous_size();
    pinnedData = impl_->allocate(uncompressedPackedSize);
    if (not pinnedData) {
      return false;
    }

    rmm::device_buffer staging(kPackStagingBytes, stream, tempMr);
    auto* destination =
        const_cast<uint8_t*>(static_cast<const uint8_t*>(pinnedData->data()));
    size_t offset = 0;
    while (packer->has_next()) {
      const auto bytes = packer->next(
          cudf::device_span<uint8_t>{
              static_cast<uint8_t*>(staging.data()), staging.size()});
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          destination + offset,
          staging.data(),
          bytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
      offset += bytes;
    }
    VELOX_CHECK_EQ(offset, uncompressedPackedSize);
    metadata = std::move(*packer->build_metadata());
    stream.synchronize();
    return true;
  };

  if (compressionMode == CompressionMode::kNone) {
    if (not packRawChunked()) {
      return false;
    }
  } else {
    try {
      auto packed = cudf::pack(table, stream, tempMr);
      uncompressedPackedSize = packed.gpu_data->size();
      compressionAttempted = uncompressedPackedSize > 0;

      ucx_exchange::PackedCompressResult compressed;
      if (compressionAttempted) {
        const auto start = std::chrono::steady_clock::now();
        compressed = ucx_exchange::compressPacked(
            packed.metadata->data(),
            packed.gpu_data->data(),
            packed.gpu_data->size(),
            stream,
            0.02,
            compressionMode == CompressionMode::kColumnAdvanced,
            0);
        compressionEncodeNanos =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
      }

      const auto* storedData =
          compressed.used ? compressed.data.data() : packed.gpu_data->data();
      const auto storedSize =
          compressed.used ? compressed.data.size() : packed.gpu_data->size();
      pinnedData = impl_->allocate(storedSize);
      if (not pinnedData) {
        return false;
      }
      if (storedSize > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            const_cast<void*>(pinnedData->data()),
            storedData,
            storedSize,
            cudaMemcpyDeviceToHost,
            stream.value()));
      }
      metadata = std::move(*packed.metadata);
      if (compressed.used) {
        compression = std::make_shared<const PackedColumnCompression>(
            std::move(compressed.regions), uncompressedPackedSize);
      }
      stream.synchronize();
    } catch (const std::exception& error) {
      LOG(WARNING)
          << "Decoded column cache compression failed; storing the range raw: "
          << error.what();
      compressionAttempted = true;
      compression.reset();
      pinnedData.reset();
      metadata.clear();
      uncompressedPackedSize = 0;
      if (not packRawChunked()) {
        return false;
      }
    }
  }

  auto candidate = std::make_shared<PinnedColumnChunk>();
  candidate->firstRow_ = firstRow;
  candidate->lastRow_ = lastRow;
  candidate->metadata_ = std::move(metadata);
  candidate->data_ = std::move(pinnedData);
  candidate->compression_ = std::move(compression);

  auto gpuKey = key;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->findColumnRangesLocked(key, firstRow, lastRow).has_value()) {
      return false;
    }
    const auto storedSize = candidate->packedSize();
    const auto isCompressed = candidate->compressed();
    auto& chunks = impl_->columns[std::move(key)];
    chunks.push_back(std::move(candidate));
    std::sort(
        chunks.begin(), chunks.end(), [](const auto& left, const auto& right) {
          return std::tie(left->firstRow_, left->lastRow_) <
              std::tie(right->firstRow_, right->lastRow_);
        });
    impl_->insertedUncompressedBytes.fetch_add(
        uncompressedPackedSize, std::memory_order_relaxed);
    impl_->insertedStoredBytes.fetch_add(storedSize, std::memory_order_relaxed);
    (isCompressed ? impl_->insertedCompressedRanges : impl_->insertedRawRanges)
        .fetch_add(1, std::memory_order_relaxed);
    if (compressionAttempted) {
      impl_->compressionAttempts.fetch_add(1, std::memory_order_relaxed);
      impl_->compressionEncodeNanos.fetch_add(
          compressionEncodeNanos, std::memory_order_relaxed);
    }
  }
  if (gpuCacheMr.has_value()) {
    insertGpuColumnRangeIfAbsent(
        std::move(gpuKey),
        firstRow,
        lastRow,
        column,
        uncompressedPackedSize,
        stream,
        gpuCacheMr.value());
  }
  return true;
}

bool CudfDecodedColumnCache::insertGpuColumnRangeIfAbsent(
    ColumnKey key,
    int64_t firstRow,
    int64_t lastRow,
    cudf::column_view column,
    uint64_t estimatedBytes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref cacheMr) {
  VELOX_CHECK_LT(
      firstRow, lastRow, "Decoded GPU cache range must be non-empty");
  VELOX_CHECK_EQ(
      lastRow - firstRow,
      column.size(),
      "Decoded GPU cache range must match column size");
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->findGpuColumnRangesLocked(key, firstRow, lastRow).has_value()) {
      return false;
    }
  }

  VELOX_USER_CHECK(
      std::isfinite(FLAGS_cudf_gpu_cache_min_host_fraction) &&
          FLAGS_cudf_gpu_cache_min_host_fraction >= 0 &&
          FLAGS_cudf_gpu_cache_min_host_fraction <= 1,
      "GPU cache minimum host fraction must be in [0, 1]");
  if (FLAGS_cudf_gpu_cache_min_host_fraction > 0) {
    const auto coverage = findColumnRanges(key, firstRow, lastRow);
    // Never skip unless the identical range already has a complete host
    // representation. No data is discarded and normal host restoration is
    // unchanged. Standalone GPU insertion and raw host chunks retain admission.
    if (coverage && coverage->size() == 1) {
      const auto& chunk = coverage->front().chunk;
      if (chunk->firstRow() == firstRow && chunk->lastRow() == lastRow &&
          chunk->compressed() &&
          chunk->packedSize() <
              static_cast<long double>(chunk->uncompressedPackedSize()) *
                  FLAGS_cudf_gpu_cache_min_host_fraction) {
        impl_->gpuAdmissionPolicySkippedRanges.fetch_add(
            1, std::memory_order_relaxed);
        return false;
      }
    }
  }

  ScaledFloatColumn scaledFloat;
  if (FLAGS_cudf_gpu_cache_scaled_float) {
    try {
      scaledFloat = tryEncodeScaledFloat(column, stream, cacheMr);
    } catch (const std::exception& error) {
      LOG(WARNING) << "Scaled-float GPU cache encoding fell back: "
                   << error.what();
    }
  }
  if (scaledFloat.column) {
    estimatedBytes = scaledFloat.column->alloc_size();
  }
  ColumnRangePtr packedMetadata;
  bool packType = false;
  if (FLAGS_cudf_gpu_cache_store_packed) {
    const auto& types = FLAGS_cudf_gpu_cache_packed_types;
    VELOX_USER_CHECK(
        types == "all" || types == "nonfloating" || types == "integral" ||
            types == "smallstrings" || types == "lowcardstrings",
        "Packed GPU cache types must be all, nonfloating, integral, smallstrings, or lowcardstrings");
    VELOX_USER_CHECK(
        std::isfinite(FLAGS_cudf_gpu_cache_packed_max_fraction) &&
            FLAGS_cudf_gpu_cache_packed_max_fraction > 0 &&
            FLAGS_cudf_gpu_cache_packed_max_fraction <= 1,
        "Packed GPU cache maximum fraction must be in (0, 1]");
    const bool floating = column.type().id() == cudf::type_id::FLOAT32 ||
        column.type().id() == cudf::type_id::FLOAT64;
    const auto id = column.type().id();
    const bool integral = id == cudf::type_id::INT8 ||
        id == cudf::type_id::INT16 || id == cudf::type_id::INT32 ||
        id == cudf::type_id::INT64 || id == cudf::type_id::UINT8 ||
        id == cudf::type_id::UINT16 || id == cudf::type_id::UINT32 ||
        id == cudf::type_id::UINT64;
    packType = types == "all" || (types == "nonfloating" && !floating) ||
        (types == "integral" && integral);
    if (types == "smallstrings" && id == cudf::type_id::STRING &&
        column.size() > 0) {
      // A generic footprint heuristic, not a schema-specific column list.
      // Keep long free-form text raw; tiny labels have large relative offset
      // overhead and often compress cheaply enough to free meaningful HBM.
      packType = cudf::strings_column_view(column).chars_size(stream) <=
          int64_t{column.size()} * 3;
    }
    if (types == "lowcardstrings") {
      packType = isLowCardinalityStringColumn(column, stream, cacheMr);
    }
  }
  if (!scaledFloat.column && packType &&
      impl_->gpuBytes.load(std::memory_order_relaxed) >=
          FLAGS_cudf_gpu_cache_raw_prefix_bytes) {
    auto hostCoverage = findColumnRanges(key, firstRow, lastRow);
    // Reuse only exact metadata/payload pairs. Standalone GPU insertion or
    // partial host coverage retains the existing owned-column representation.
    if (hostCoverage && hostCoverage->size() == 1 &&
        hostCoverage->front().chunk->firstRow() == firstRow &&
        hostCoverage->front().chunk->lastRow() == lastRow &&
        hostCoverage->front().chunk->packedSize() <=
            static_cast<long double>(estimatedBytes) *
                FLAGS_cudf_gpu_cache_packed_max_fraction) {
      packedMetadata = hostCoverage->front().chunk;
      estimatedBytes = packedMetadata->packedSize();
    }
  }
  estimatedBytes = std::max<uint64_t>(estimatedBytes, 1);
  if (not impl_->reserveGpuBytes(estimatedBytes)) {
    impl_->gpuAdmissionRejectedRanges.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  std::unique_ptr<cudf::column> deviceColumn = std::move(scaledFloat.column);
  rmm::device_buffer packedData;
  uint64_t reservedBytes = estimatedBytes;
  try {
    if (packedMetadata) {
      packedData =
          rmm::device_buffer(packedMetadata->packedSize(), stream, cacheMr);
      if (packedData.size() > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            packedData.data(),
            packedMetadata->pinnedData(),
            packedData.size(),
            cudaMemcpyHostToDevice,
            stream.value()));
      }
    } else if (!deviceColumn) {
      deviceColumn = std::make_unique<cudf::column>(column, stream, cacheMr);
    }
    const auto actualBytes =
        packedMetadata ? packedData.size() : deviceColumn->alloc_size();
    if (actualBytes > reservedBytes) {
      const auto additionalBytes = actualBytes - reservedBytes;
      if (not impl_->reserveGpuBytes(additionalBytes)) {
        impl_->gpuBytes.fetch_sub(reservedBytes, std::memory_order_relaxed);
        impl_->gpuAdmissionRejectedRanges.fetch_add(
            1, std::memory_order_relaxed);
        return false;
      }
    } else if (actualBytes < reservedBytes) {
      impl_->gpuBytes.fetch_sub(
          reservedBytes - actualBytes, std::memory_order_relaxed);
    }
    reservedBytes = actualBytes;
    stream.synchronize();
  } catch (const std::exception& error) {
    impl_->gpuBytes.fetch_sub(reservedBytes, std::memory_order_relaxed);
    impl_->gpuAdmissionRejectedRanges.fetch_add(1, std::memory_order_relaxed);
    LOG(WARNING) << "Skipping decoded column GPU cache admission after device "
                    "allocation failed: "
                 << error.what();
    return false;
  }

  auto candidate = std::make_shared<GpuColumnChunk>(GpuColumnChunk{
      .firstRow = firstRow,
      .lastRow = lastRow,
      .bytes = reservedBytes,
      .column = std::move(deviceColumn),
      .packedMetadata = packedMetadata,
      .packedData = std::move(packedData),
      .scaledFloatScale = scaledFloat.scale,
      .allocationStream = stream,
  });

  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->findGpuColumnRangesLocked(key, firstRow, lastRow).has_value()) {
    impl_->gpuBytes.fetch_sub(reservedBytes, std::memory_order_relaxed);
    return false;
  }
  auto& chunks = impl_->gpuColumns[std::move(key)];
  chunks.push_back(std::move(candidate));
  std::sort(
      chunks.begin(), chunks.end(), [](const auto& left, const auto& right) {
        return std::tie(left->firstRow, left->lastRow) <
            std::tie(right->firstRow, right->lastRow);
      });
  impl_->gpuInsertedBytes.fetch_add(reservedBytes, std::memory_order_relaxed);
  if (packedMetadata) {
    impl_->gpuPackedInsertedBytes.fetch_add(
        reservedBytes, std::memory_order_relaxed);
  }
  if (scaledFloat.scale != 0) {
    impl_->gpuScaledInsertedBytes.fetch_add(
        reservedBytes, std::memory_order_relaxed);
    impl_->gpuScaledInsertedRanges.fetch_add(1, std::memory_order_relaxed);
  }
  impl_->gpuInsertedRanges.fetch_add(1, std::memory_order_relaxed);
  return true;
}

std::unique_ptr<cudf::column> CudfDecodedColumnCache::materializeColumnRange(
    const ColumnKey& key,
    int64_t firstRow,
    int64_t lastRow,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref outputMr,
    rmm::device_async_resource_ref tempMr) const {
  auto coverage = findColumnRanges(key, firstRow, lastRow);
  if (not coverage) {
    return nullptr;
  }

  std::vector<std::unique_ptr<cudf::column>> pieces;
  pieces.reserve(coverage->size());
  for (const auto& range : *coverage) {
    const auto& chunk = range.chunk;
    rmm::device_buffer storedData(chunk->packedSize(), stream, tempMr);
    if (chunk->packedSize() > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          storedData.data(),
          chunk->pinnedData(),
          chunk->packedSize(),
          cudaMemcpyHostToDevice,
          stream.value()));
    }
    rmm::device_buffer decompressedData;
    const uint8_t* packedData = static_cast<const uint8_t*>(storedData.data());
    if (chunk->compressed()) {
      const auto start = std::chrono::steady_clock::now();
      decompressedData = ucx_exchange::decompressPacked(
          storedData.data(),
          chunk->compression_->regions_,
          chunk->compression_->uncompressedBytes_,
          stream);
      impl_->decompressionNanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count(),
          std::memory_order_relaxed);
      packedData = static_cast<const uint8_t*>(decompressedData.data());
    }
    impl_->restoreCalls.fetch_add(1, std::memory_order_relaxed);
    impl_->restoredStoredBytes.fetch_add(
        chunk->packedSize(), std::memory_order_relaxed);
    impl_->restoredUncompressedBytes.fetch_add(
        chunk->uncompressedPackedSize(), std::memory_order_relaxed);
    const auto unpacked = cudf::unpack(chunk->metadata_.data(), packedData);
    VELOX_CHECK_EQ(unpacked.num_columns(), 1);

    const auto relativeFirst = range.firstRow - chunk->firstRow();
    const auto relativeLast = range.lastRow - chunk->firstRow();
    VELOX_CHECK_LE(
        relativeLast,
        static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()));
    const auto slice =
        cacheSlice(unpacked.column(0), relativeFirst, relativeLast, stream);

    if (coverage->size() == 1) {
      return std::make_unique<cudf::column>(slice, stream, outputMr);
    }
    pieces.push_back(std::make_unique<cudf::column>(slice, stream, tempMr));
  }

  std::vector<cudf::column_view> pieceViews;
  pieceViews.reserve(pieces.size());
  for (const auto& piece : pieces) {
    pieceViews.push_back(piece->view());
  }
  return cudf::concatenate(pieceViews, stream, outputMr);
}

std::optional<std::vector<std::unique_ptr<cudf::column>>>
CudfDecodedColumnCache::materializeColumnRanges(
    const std::vector<ColumnRangeRequest>& requests,
    rmm::cuda_stream_view stream,
    rmm::cuda_stream_view transferStream,
    rmm::device_async_resource_ref outputMr,
    rmm::device_async_resource_ref tempMr) const {
  if (FLAGS_cudf_decoded_cache_single_stream_restore) {
    transferStream = stream;
  }
  struct WorkItem {
    size_t requestIndex;
    CoveredColumnRange range;
  };

  std::vector<WorkItem> work;
  std::vector<size_t> pieceCounts(requests.size(), 0);
  for (size_t requestIndex = 0; requestIndex < requests.size();
       ++requestIndex) {
    for (const auto& [firstRow, lastRow] : requests[requestIndex].ranges) {
      auto coverage =
          findColumnRanges(requests[requestIndex].key, firstRow, lastRow);
      if (not coverage) {
        return std::nullopt;
      }
      pieceCounts[requestIndex] += coverage->size();
      for (auto& range : *coverage) {
        work.push_back({requestIndex, std::move(range)});
      }
    }
  }

  std::vector<std::vector<std::unique_ptr<cudf::column>>> pieces(
      requests.size());
  for (size_t requestIndex = 0; requestIndex < requests.size();
       ++requestIndex) {
    pieces[requestIndex].reserve(pieceCounts[requestIndex]);
  }
  if (work.empty()) {
    return std::vector<std::unique_ptr<cudf::column>>{};
  }

  struct TransferSlot {
    rmm::device_buffer storedData;
    CachePipelineEvent ready;
    CachePipelineEvent consumed;
    bool hasPendingConsumer{false};
  };
  std::array<TransferSlot, 2> slots;

  const auto stage = [&](size_t workIndex) {
    auto& slot = slots[workIndex % slots.size()];
    if (slot.hasPendingConsumer) {
      slot.consumed.wait(transferStream);
    }
    const auto& chunk = work[workIndex].range.chunk;
    slot.storedData =
        rmm::device_buffer(chunk->packedSize(), transferStream, tempMr);
    if (chunk->packedSize() > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          slot.storedData.data(),
          chunk->pinnedData(),
          chunk->packedSize(),
          cudaMemcpyHostToDevice,
          transferStream.value()));
    }
    slot.ready.record(transferStream);
    slot.hasPendingConsumer = false;
  };

  stage(0);
  for (size_t workIndex = 0; workIndex < work.size(); ++workIndex) {
    if (workIndex + 1 < work.size()) {
      stage(workIndex + 1);
    }

    auto& slot = slots[workIndex % slots.size()];
    const auto& item = work[workIndex];
    const auto& chunk = item.range.chunk;
    slot.ready.wait(stream);

    rmm::device_buffer decompressedData;
    const uint8_t* packedData =
        static_cast<const uint8_t*>(slot.storedData.data());
    if (chunk->compressed()) {
      const auto start = std::chrono::steady_clock::now();
      decompressedData = ucx_exchange::decompressPacked(
          slot.storedData.data(),
          chunk->compression_->regions_,
          chunk->compression_->uncompressedBytes_,
          stream);
      impl_->decompressionNanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count(),
          std::memory_order_relaxed);
      packedData = static_cast<const uint8_t*>(decompressedData.data());
    }

    impl_->restoreCalls.fetch_add(1, std::memory_order_relaxed);
    impl_->restoredStoredBytes.fetch_add(
        chunk->packedSize(), std::memory_order_relaxed);
    impl_->restoredUncompressedBytes.fetch_add(
        chunk->uncompressedPackedSize(), std::memory_order_relaxed);

    const auto unpacked = cudf::unpack(chunk->metadata_.data(), packedData);
    VELOX_CHECK_EQ(unpacked.num_columns(), 1);
    const auto relativeFirst = item.range.firstRow - chunk->firstRow();
    const auto relativeLast = item.range.lastRow - chunk->firstRow();
    VELOX_CHECK_LE(
        relativeLast,
        static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()));
    const auto slice =
        cacheSlice(unpacked.column(0), relativeFirst, relativeLast, stream);
    auto piece = std::make_unique<cudf::column>(
        slice, stream, pieceCounts[item.requestIndex] == 1 ? outputMr : tempMr);
    pieces[item.requestIndex].push_back(std::move(piece));
    slot.consumed.record(stream);
    slot.hasPendingConsumer = true;
  }

  // Order the staging-buffer releases after their final consumers without a
  // device-wide or host-side synchronization.
  for (auto& slot : slots) {
    if (slot.hasPendingConsumer) {
      slot.consumed.wait(transferStream);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> outputs;
  outputs.reserve(requests.size());
  for (auto& requestPieces : pieces) {
    VELOX_CHECK(not requestPieces.empty());
    if (requestPieces.size() == 1) {
      outputs.push_back(std::move(requestPieces.front()));
      continue;
    }
    std::vector<cudf::column_view> pieceViews;
    pieceViews.reserve(requestPieces.size());
    for (const auto& piece : requestPieces) {
      pieceViews.push_back(piece->view());
    }
    outputs.push_back(cudf::concatenate(pieceViews, stream, outputMr));
  }
  impl_->pipelinedRestoreBatches.fetch_add(1, std::memory_order_relaxed);
  return outputs;
}

std::unique_ptr<CudfDecodedColumnCache::BorrowedGpuColumns>
CudfDecodedColumnCache::restoreColumnRangeViews(
    const std::vector<ColumnRangeRequest>& requests,
    rmm::cuda_stream_view stream,
    rmm::cuda_stream_view transferStream,
    rmm::device_async_resource_ref outputMr,
    rmm::device_async_resource_ref tempMr) const {
  if (requests.empty()) {
    return nullptr;
  }
  if (FLAGS_cudf_decoded_cache_single_stream_restore) {
    transferStream = stream;
  }
  struct WorkItem {
    size_t requestIndex;
    CoveredColumnRange range;
  };
  std::vector<WorkItem> work;
  std::vector<size_t> pieceCounts(requests.size(), 0);
  for (size_t index = 0; index < requests.size(); ++index) {
    for (const auto& [first, last] : requests[index].ranges) {
      auto coverage = findColumnRanges(requests[index].key, first, last);
      if (!coverage) {
        return nullptr;
      }
      pieceCounts[index] += coverage->size();
      for (auto& range : *coverage) {
        work.push_back({index, std::move(range)});
      }
    }
    if (pieceCounts[index] == 0) {
      return nullptr;
    }
  }
  auto result = std::make_unique<BorrowedGpuColumns>(stream);
  result->views.resize(requests.size());
  std::vector<std::vector<std::unique_ptr<cudf::column>>> pieces(
      requests.size());
  struct TransferSlot {
    rmm::device_buffer storedData;
    CachePipelineEvent ready;
    CachePipelineEvent consumed;
    bool hasPendingConsumer{false};
  };
  std::array<TransferSlot, 2> slots;
  // On an exceptional exit, drain both streams before staging storage and
  // pinned source owners disappear. The successful path uses events only.
  SCOPE_FAIL {
    cudaStreamSynchronize(transferStream.value());
    cudaStreamSynchronize(stream.value());
  };
  const auto stage = [&](size_t index) {
    auto& slot = slots[index % slots.size()];
    if (slot.hasPendingConsumer) {
      slot.consumed.wait(transferStream);
    }
    const auto& chunk = work[index].range.chunk;
    slot.storedData =
        rmm::device_buffer(chunk->packedSize(), transferStream, tempMr);
    if (chunk->packedSize() > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          slot.storedData.data(),
          chunk->pinnedData(),
          chunk->packedSize(),
          cudaMemcpyHostToDevice,
          transferStream.value()));
    }
    slot.ready.record(transferStream);
    slot.hasPendingConsumer = false;
  };
  stage(0);
  for (size_t index = 0; index < work.size(); ++index) {
    if (index + 1 < work.size()) {
      stage(index + 1);
    }
    auto& slot = slots[index % slots.size()];
    const auto& item = work[index];
    const auto& chunk = item.range.chunk;
    slot.ready.wait(stream);
    rmm::device_buffer decoded;
    const auto* data = static_cast<const uint8_t*>(slot.storedData.data());
    if (chunk->compressed()) {
      const auto start = std::chrono::steady_clock::now();
      decoded = ucx_exchange::decompressPacked(
          data,
          chunk->compression_->regions_,
          chunk->compression_->uncompressedBytes_,
          stream);
      impl_->decompressionNanos.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - start)
              .count(),
          std::memory_order_relaxed);
      data = static_cast<const uint8_t*>(decoded.data());
    }
    impl_->restoreCalls.fetch_add(1, std::memory_order_relaxed);
    impl_->restoredStoredBytes.fetch_add(
        chunk->packedSize(), std::memory_order_relaxed);
    impl_->restoredUncompressedBytes.fetch_add(
        chunk->uncompressedPackedSize(), std::memory_order_relaxed);
    const auto unpacked = cudf::unpack(chunk->metadata_.data(), data);
    VELOX_CHECK_EQ(unpacked.num_columns(), 1);
    const auto first = item.range.firstRow - chunk->firstRow();
    const auto last = item.range.lastRow - chunk->firstRow();
    VELOX_CHECK_LE(
        last,
        static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()));
    const auto view = cacheSlice(unpacked.column(0), first, last, stream);
    // Keep every pinned H2D source alive, including fragmented requests whose
    // output owns copied columns but whose asynchronous transfers may still
    // be in flight when the process cache is cleared.
    result->owners.push_back(chunk);
    if (pieceCounts[item.requestIndex] == 1) {
      result->views[item.requestIndex] = view;
      auto& retained = chunk->compressed() ? decoded : slot.storedData;
      result->retainedBytes += retained.size();
      result->decodedBuffers.push_back(std::move(retained));
    } else {
      pieces[item.requestIndex].push_back(
          std::make_unique<cudf::column>(view, stream, tempMr));
    }
    slot.consumed.record(stream);
    slot.hasPendingConsumer = true;
  }
  for (auto& slot : slots) {
    if (slot.hasPendingConsumer) {
      slot.consumed.wait(transferStream);
    }
  }
  for (size_t index = 0; index < requests.size(); ++index) {
    if (pieceCounts[index] == 1) {
      continue;
    }
    std::vector<cudf::column_view> views;
    for (const auto& piece : pieces[index]) {
      views.push_back(piece->view());
    }
    auto column = cudf::concatenate(views, stream, outputMr);
    result->retainedBytes += column->alloc_size();
    result->views[index] = column->view();
    result->decodedColumns.push_back(std::move(column));
  }
  impl_->pipelinedRestoreBatches.fetch_add(1, std::memory_order_relaxed);
  return result;
}

std::vector<std::unique_ptr<cudf::column>>
CudfDecodedColumnCache::materializeGpuColumnRanges(
    const std::vector<ColumnRangeRequest>& requests,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref outputMr) const {
  std::vector<std::unique_ptr<cudf::column>> outputs(requests.size());
  bool restoredAny = false;
  for (size_t requestIndex = 0; requestIndex < requests.size();
       ++requestIndex) {
    std::vector<Impl::CoveredGpuColumnRange> coverage;
    bool fullyCovered = true;
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      for (const auto& [firstRow, lastRow] : requests[requestIndex].ranges) {
        auto rangeCoverage = impl_->findGpuColumnRangesLocked(
            requests[requestIndex].key, firstRow, lastRow);
        if (not rangeCoverage) {
          fullyCovered = false;
          break;
        }
        coverage.insert(
            coverage.end(),
            std::make_move_iterator(rangeCoverage->begin()),
            std::make_move_iterator(rangeCoverage->end()));
      }
    }
    if (not fullyCovered or coverage.empty()) {
      continue;
    }

    std::vector<cudf::column_view> pieceViews;
    pieceViews.reserve(coverage.size());
    // Decoded buffers own all temporary views until the output copy/concatenate
    // is enqueued. Their frees are ordered on the same stream as those copies.
    std::vector<rmm::device_buffer> decodedBuffers;
    decodedBuffers.reserve(coverage.size());
    std::vector<std::unique_ptr<cudf::column>> decodedFloatColumns;
    for (const auto& range : coverage) {
      const auto relativeFirst = range.firstRow - range.chunk->firstRow;
      const auto relativeLast = range.lastRow - range.chunk->firstRow;
      VELOX_CHECK_LE(
          relativeLast,
          static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()));
      if (range.chunk->scaledFloatScale != 0) {
        decodedFloatColumns.push_back(decodeScaledFloat(
            cacheSlice(
                range.chunk->column->view(),
                relativeFirst,
                relativeLast,
                stream),
            range.chunk->scaledFloatScale,
            stream,
            outputMr));
        pieceViews.push_back(decodedFloatColumns.back()->view());
        impl_->gpuScaledRestoreCalls.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      auto view = [&]() -> cudf::column_view {
        if (!range.chunk->packedMetadata) {
          return range.chunk->column->view();
        }
        const auto& metadata = range.chunk->packedMetadata;
        auto* data =
            static_cast<const uint8_t*>(range.chunk->packedData.data());
        if (metadata->compressed()) {
          const auto start = std::chrono::steady_clock::now();
          decodedBuffers.push_back(
              ucx_exchange::decompressPacked(
                  data,
                  metadata->compression_->regions_,
                  metadata->compression_->uncompressedBytes_,
                  stream));
          data = static_cast<const uint8_t*>(decodedBuffers.back().data());
          impl_->gpuPackedDecompressionNanos.fetch_add(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - start)
                  .count(),
              std::memory_order_relaxed);
        }
        impl_->gpuPackedRestoreCalls.fetch_add(1, std::memory_order_relaxed);
        impl_->gpuPackedRestoredStoredBytes.fetch_add(
            metadata->packedSize(), std::memory_order_relaxed);
        auto unpacked = cudf::unpack(metadata->metadata_.data(), data);
        VELOX_CHECK_EQ(unpacked.num_columns(), 1);
        return unpacked.column(0);
      }();
      pieceViews.push_back(
          cacheSlice(view, relativeFirst, relativeLast, stream));
    }

    auto output = pieceViews.size() == 1 && decodedFloatColumns.size() == 1
        ? std::move(decodedFloatColumns.front())
        : pieceViews.size() == 1
        ? std::make_unique<cudf::column>(pieceViews.front(), stream, outputMr)
        : cudf::concatenate(pieceViews, stream, outputMr);
    impl_->gpuRestoreCalls.fetch_add(
        coverage.size(), std::memory_order_relaxed);
    impl_->gpuRestoredBytes.fetch_add(
        output->alloc_size(), std::memory_order_relaxed);
    outputs[requestIndex] = std::move(output);
    restoredAny = true;
  }
  if (restoredAny) {
    impl_->gpuRestoreBatches.fetch_add(1, std::memory_order_relaxed);
  }
  return outputs;
}

CudfDecodedColumnCache::BorrowedGpuColumns::~BorrowedGpuColumns() {
  // A cache clear may have dropped the cache's reference while this lease was
  // still being consumed. Its original allocation stream need not match ours.
  if (!owners.empty() || !decodedColumns.empty() || !decodedBuffers.empty()) {
    if (orderRelease) {
      try {
        orderRelease(stream);
        return;
      } catch (const std::exception& error) {
        LOG(ERROR) << "GPU cache release event fell back to synchronization: "
                   << error.what();
      }
    }
    const auto status = cudaStreamSynchronize(stream.value());
    if (status != cudaSuccess) {
      LOG(ERROR) << "GPU cache lease synchronization failed: "
                 << cudaGetErrorString(status);
    }
  }
}

std::unique_ptr<CudfDecodedColumnCache::BorrowedGpuColumns>
CudfDecodedColumnCache::borrowGpuColumnRanges(
    const std::vector<ColumnRangeRequest>& requests,
    rmm::cuda_stream_view stream,
    std::optional<rmm::device_async_resource_ref> decodeMr) const {
  if (requests.empty()) {
    return nullptr;
  }
  std::vector<Impl::CoveredGpuColumnRange> ranges;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (const auto& request : requests) {
      if (request.ranges.size() != 1) {
        return nullptr;
      }
      const auto [first, last] = request.ranges.front();
      auto coverage =
          impl_->findGpuColumnRangesLocked(request.key, first, last);
      if (!coverage || coverage->size() != 1 ||
          coverage->front().chunk->packedMetadata ||
          (coverage->front().chunk->scaledFloatScale != 0 && !decodeMr)) {
        return nullptr;
      }
      ranges.push_back(coverage->front());
    }
  }
  auto result = std::make_unique<BorrowedGpuColumns>(stream);
  result->gpuColumns = ranges.size();
  result->owners.reserve(ranges.size());
  result->views.reserve(ranges.size());
  for (const auto& range : ranges) {
    result->owners.push_back(range.chunk);
    result->retainedBytes += range.chunk->bytes;
    const auto view = cacheSlice(
        range.chunk->column->view(),
        range.firstRow - range.chunk->firstRow,
        range.lastRow - range.chunk->firstRow,
        stream);
    if (range.chunk->scaledFloatScale != 0) {
      auto decoded = decodeScaledFloat(
          view, range.chunk->scaledFloatScale, stream, *decodeMr);
      result->retainedBytes += decoded->alloc_size();
      result->views.push_back(decoded->view());
      result->decodedColumns.push_back(std::move(decoded));
      impl_->gpuScaledRestoreCalls.fetch_add(1, std::memory_order_relaxed);
    } else {
      result->views.push_back(view);
    }
  }
  if (FLAGS_cudf_cache_stream_ordered_release &&
      result->decodedColumns.empty()) {
    // Raw entries were copied wholly on allocationStream. Every data/null/
    // child buffer frees on that stream, so an event wait orders the eventual
    // release even if a concurrent clear drops the cache's own reference.
    // Capture chunks (not only streams) through event submission.
    result->orderRelease =
        [ranges = std::move(ranges)](rmm::cuda_stream_view consumer) {
          std::vector<rmm::cuda_stream_view> allocationStreams;
          for (const auto& range : ranges) {
            const auto allocation = range.chunk->allocationStream;
            if (allocation.value() != consumer.value() &&
                std::none_of(
                    allocationStreams.begin(),
                    allocationStreams.end(),
                    [&](const auto other) {
                      return other.value() == allocation.value();
                    })) {
              allocationStreams.push_back(allocation);
            }
          }
          if (allocationStreams.empty()) {
            return;
          }
          CachePipelineEvent consumed;
          consumed.record(consumer);
          for (const auto allocation : allocationStreams) {
            consumed.wait(allocation);
          }
        };
  }
  return result;
}

std::unique_ptr<CudfDecodedColumnCache::BorrowedGpuColumns>
CudfDecodedColumnCache::borrowOrRestoreColumnRanges(
    const std::vector<ColumnRangeRequest>& requests,
    rmm::cuda_stream_view stream,
    rmm::cuda_stream_view transferStream,
    rmm::device_async_resource_ref outputMr,
    rmm::device_async_resource_ref tempMr) const {
  if (requests.empty()) {
    return nullptr;
  }
  auto result = std::make_unique<BorrowedGpuColumns>(stream);
  result->views.resize(requests.size());
  std::vector<ColumnRangeRequest> pending;
  std::vector<size_t> indices;
  for (size_t index = 0; index < requests.size(); ++index) {
    auto raw = borrowGpuColumnRanges({requests[index]}, stream);
    if (raw) {
      result->views[index] = raw->views.front();
      result->retainedBytes += raw->retainedBytes;
      result->gpuColumns += raw->gpuColumns;
      for (auto& owner : raw->owners) {
        result->owners.push_back(std::move(owner));
      }
      raw->owners.clear();
      continue;
    }
    // Check complete coverage before scheduling restoration. In particular,
    // a cold file should fall back without making speculative full-column
    // copies.
    bool gpuCovered = !requests[index].ranges.empty();
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      for (const auto& [first, last] : requests[index].ranges) {
        gpuCovered &=
            impl_->findGpuColumnRangesLocked(requests[index].key, first, last)
                .has_value();
      }
    }
    if (!gpuCovered) {
      if (requests[index].ranges.empty()) {
        return nullptr;
      }
      for (const auto& [first, last] : requests[index].ranges) {
        if (!containsColumnRange(requests[index].key, first, last)) {
          return nullptr;
        }
      }
    }
    pending.push_back(requests[index]);
    indices.push_back(index);
  }
  auto materialized = materializeGpuColumnRanges(pending, stream, outputMr);
  std::vector<ColumnRangeRequest> hostRequests;
  std::vector<size_t> hostIndices;
  auto retainColumn = [&](size_t index, std::unique_ptr<cudf::column> column) {
    result->retainedBytes += column->alloc_size();
    result->views[index] = column->view();
    result->decodedColumns.push_back(std::move(column));
  };
  for (size_t index = 0; index < pending.size(); ++index) {
    if (materialized[index]) {
      retainColumn(indices[index], std::move(materialized[index]));
      ++result->gpuColumns;
    } else {
      hostRequests.push_back(pending[index]);
      hostIndices.push_back(indices[index]);
    }
  }
  if (!hostRequests.empty() && FLAGS_cudf_cache_restore_packed_views) {
    auto restored = restoreColumnRangeViews(
        hostRequests, stream, transferStream, outputMr, tempMr);
    if (!restored) {
      return nullptr;
    }
    for (size_t index = 0; index < hostRequests.size(); ++index) {
      result->views[hostIndices[index]] = restored->views[index];
    }
    result->retainedBytes += restored->retainedBytes;
    for (auto& owner : restored->owners) {
      result->owners.push_back(std::move(owner));
    }
    for (auto& column : restored->decodedColumns) {
      result->decodedColumns.push_back(std::move(column));
    }
    for (auto& buffer : restored->decodedBuffers) {
      result->decodedBuffers.push_back(std::move(buffer));
    }
    restored->owners.clear();
    restored->decodedColumns.clear();
    restored->decodedBuffers.clear();
  } else if (!hostRequests.empty()) {
    auto restored = materializeColumnRanges(
        hostRequests, stream, transferStream, outputMr, tempMr);
    if (!restored) {
      // A test-only concurrent cache clear can invalidate the coverage check.
      return nullptr;
    }
    for (size_t index = 0; index < hostRequests.size(); ++index) {
      retainColumn(hostIndices[index], std::move(restored->at(index)));
    }
  }
  return result;
}

uint64_t CudfDecodedColumnCache::pinnedBytes() const {
  return impl_->allocatedBytes.load(std::memory_order_relaxed);
}

uint64_t CudfDecodedColumnCache::maxPinnedBytes() const {
  return impl_->maxPinnedBytes;
}

uint64_t CudfDecodedColumnCache::gpuBytes() const {
  return impl_->gpuBytes.load(std::memory_order_relaxed);
}

uint64_t CudfDecodedColumnCache::maxGpuBytes() const {
  return impl_->maxGpuBytes;
}

CudfDecodedColumnCache::Stats CudfDecodedColumnCache::stats() const {
  return {
      .maxPinnedBytes = impl_->maxPinnedBytes,
      .pinnedBytes = impl_->allocatedBytes.load(std::memory_order_relaxed),
      .hostAdmissionRejectedAllocations =
          impl_->hostAdmissionRejectedAllocations.load(
              std::memory_order_relaxed),
      .hostAdmissionRejectedBytes =
          impl_->hostAdmissionRejectedBytes.load(std::memory_order_relaxed),
      .insertedUncompressedBytes =
          impl_->insertedUncompressedBytes.load(std::memory_order_relaxed),
      .insertedStoredBytes =
          impl_->insertedStoredBytes.load(std::memory_order_relaxed),
      .insertedCompressedRanges =
          impl_->insertedCompressedRanges.load(std::memory_order_relaxed),
      .insertedRawRanges =
          impl_->insertedRawRanges.load(std::memory_order_relaxed),
      .compressionAttempts =
          impl_->compressionAttempts.load(std::memory_order_relaxed),
      .compressionEncodeNanos =
          impl_->compressionEncodeNanos.load(std::memory_order_relaxed),
      .restoreCalls = impl_->restoreCalls.load(std::memory_order_relaxed),
      .restoredStoredBytes =
          impl_->restoredStoredBytes.load(std::memory_order_relaxed),
      .restoredUncompressedBytes =
          impl_->restoredUncompressedBytes.load(std::memory_order_relaxed),
      .decompressionNanos =
          impl_->decompressionNanos.load(std::memory_order_relaxed),
      .pipelinedRestoreBatches =
          impl_->pipelinedRestoreBatches.load(std::memory_order_relaxed),
      .maxGpuBytes = impl_->maxGpuBytes,
      .gpuBytes = impl_->gpuBytes.load(std::memory_order_relaxed),
      .gpuInsertedBytes =
          impl_->gpuInsertedBytes.load(std::memory_order_relaxed),
      .gpuInsertedRanges =
          impl_->gpuInsertedRanges.load(std::memory_order_relaxed),
      .gpuAdmissionRejectedRanges =
          impl_->gpuAdmissionRejectedRanges.load(std::memory_order_relaxed),
      .gpuAdmissionPolicySkippedRanges =
          impl_->gpuAdmissionPolicySkippedRanges.load(
              std::memory_order_relaxed),
      .gpuRestoreCalls = impl_->gpuRestoreCalls.load(std::memory_order_relaxed),
      .gpuRestoredBytes =
          impl_->gpuRestoredBytes.load(std::memory_order_relaxed),
      .gpuRestoreBatches =
          impl_->gpuRestoreBatches.load(std::memory_order_relaxed),
      .gpuPackedInsertedBytes =
          impl_->gpuPackedInsertedBytes.load(std::memory_order_relaxed),
      .gpuPackedRestoreCalls =
          impl_->gpuPackedRestoreCalls.load(std::memory_order_relaxed),
      .gpuPackedRestoredStoredBytes =
          impl_->gpuPackedRestoredStoredBytes.load(std::memory_order_relaxed),
      .gpuPackedDecompressionNanos =
          impl_->gpuPackedDecompressionNanos.load(std::memory_order_relaxed),
      .gpuScaledInsertedBytes =
          impl_->gpuScaledInsertedBytes.load(std::memory_order_relaxed),
      .gpuScaledInsertedRanges =
          impl_->gpuScaledInsertedRanges.load(std::memory_order_relaxed),
      .gpuScaledRestoreCalls =
          impl_->gpuScaledRestoreCalls.load(std::memory_order_relaxed),
  };
}

void CudfDecodedColumnCache::clearForTesting() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->columns.clear();
  impl_->gpuColumns.clear();
  impl_->metadata.clear();
  impl_->rowGroupSelections.clear();
  impl_->hostAdmissionRejectedAllocations.store(0, std::memory_order_relaxed);
  impl_->hostAdmissionRejectedBytes.store(0, std::memory_order_relaxed);
  impl_->insertedUncompressedBytes.store(0, std::memory_order_relaxed);
  impl_->insertedStoredBytes.store(0, std::memory_order_relaxed);
  impl_->insertedCompressedRanges.store(0, std::memory_order_relaxed);
  impl_->insertedRawRanges.store(0, std::memory_order_relaxed);
  impl_->compressionAttempts.store(0, std::memory_order_relaxed);
  impl_->compressionEncodeNanos.store(0, std::memory_order_relaxed);
  impl_->restoreCalls.store(0, std::memory_order_relaxed);
  impl_->restoredStoredBytes.store(0, std::memory_order_relaxed);
  impl_->restoredUncompressedBytes.store(0, std::memory_order_relaxed);
  impl_->decompressionNanos.store(0, std::memory_order_relaxed);
  impl_->pipelinedRestoreBatches.store(0, std::memory_order_relaxed);
  impl_->gpuBytes.store(0, std::memory_order_relaxed);
  impl_->gpuInsertedBytes.store(0, std::memory_order_relaxed);
  impl_->gpuInsertedRanges.store(0, std::memory_order_relaxed);
  impl_->gpuAdmissionRejectedRanges.store(0, std::memory_order_relaxed);
  impl_->gpuAdmissionPolicySkippedRanges.store(0, std::memory_order_relaxed);
  impl_->gpuRestoreCalls.store(0, std::memory_order_relaxed);
  impl_->gpuRestoredBytes.store(0, std::memory_order_relaxed);
  impl_->gpuRestoreBatches.store(0, std::memory_order_relaxed);
  impl_->gpuPackedInsertedBytes.store(0, std::memory_order_relaxed);
  impl_->gpuPackedRestoreCalls.store(0, std::memory_order_relaxed);
  impl_->gpuPackedRestoredStoredBytes.store(0, std::memory_order_relaxed);
  impl_->gpuPackedDecompressionNanos.store(0, std::memory_order_relaxed);
  impl_->gpuScaledInsertedBytes.store(0, std::memory_order_relaxed);
  impl_->gpuScaledInsertedRanges.store(0, std::memory_order_relaxed);
  impl_->gpuScaledRestoreCalls.store(0, std::memory_order_relaxed);
}

} // namespace facebook::velox::cudf_velox::connector::hive
