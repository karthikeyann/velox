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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/exec/GpuCapabilities.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/caching/CacheTTLController.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"

#include <rmm/error.hpp>

#include <atomic>
#ifdef VELOX_ENABLE_ABFS
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"
#endif

#include <cudf/column/column.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <memory>
#include <ranges>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

namespace {

// Checks whether the `path` uses an ABFS scheme
bool isAbfsPath([[maybe_unused]] const std::string_view path) {
#ifdef VELOX_ENABLE_ABFS
  return ::facebook::velox::filesystems::isAbfsFile(path);
#else
  return false;
#endif
}

// Rebuilds a struct/list column in-place after possibly transforming (e.g.,
// decimal-casting) its children.
template <typename TransformChildrenFn>
std::unique_ptr<cudf::column> rebuildWithTransformedChildren(
    std::unique_ptr<cudf::column> col,
    TransformChildrenFn&& transformFn) {
  auto const type = col->type();
  auto const size = col->size();
  auto const nullCount = col->null_count();
  auto contents = col->release();
  transformFn(contents.children);
  return std::make_unique<cudf::column>(
      type,
      size,
      std::move(*contents.data),
      std::move(*contents.null_mask),
      nullCount,
      std::move(contents.children));
}

// Recursively casts columns to the expected Velox type iff the column is:
//  - Decimal type but not the expected Velox type.
//  - Struct type: with any of its children being decimal type but not the
//  expected Velox type. Rebuilt in place with the casted children.
//  - List type: with its `child` being decimal type but not the expected Velox
//  type. Rebuilt in place with the casted children.
std::unique_ptr<cudf::column> castDecimalColumns(
    std::unique_ptr<cudf::column> col,
    const TypePtr& veloxType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  // Decimal type (base case)
  if (veloxType->isDecimal()) {
    auto const targetType = veloxToCudfDataType(veloxType);
    if (col->type() != targetType) {
      return cudf::cast(col->view(), targetType, stream, mr);
    }
    return col;
  }

  // Struct type
  if (veloxType->kind() == TypeKind::ROW) {
    auto const& rowType = veloxType->asRow();
    auto const numChildren = static_cast<size_t>(col->num_children());
    VELOX_CHECK_EQ(
        numChildren,
        rowType.size(),
        "Scanned STRUCT column has {} fields but the expected schema has {}.",
        numChildren,
        rowType.size());
    return rebuildWithTransformedChildren(std::move(col), [&](auto& children) {
      for (size_t i = 0; i < numChildren; ++i) {
        children[i] = castDecimalColumns(
            std::move(children[i]), rowType.childAt(i), stream, mr);
      }
    });
  }

  // List type
  if (veloxType->kind() == TypeKind::ARRAY) {
    // A LIST column stores [offsets, child]; only the child may hold decimal
    // data.
    VELOX_CHECK_EQ(
        col->num_children(),
        2,
        "LIST column must have exactly 2 children: [offsets, child]");
    return rebuildWithTransformedChildren(std::move(col), [&](auto& children) {
      auto const childIdx = cudf::lists_column_view::child_column_index;
      children[childIdx] = castDecimalColumns(
          std::move(children[childIdx]), veloxType->childAt(0), stream, mr);
    });
  }

  return col;
}

std::unique_ptr<cudf::table> castDecimalColumnsToVeloxTypes(
    std::unique_ptr<cudf::table>&& table,
    const RowTypePtr& rowType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto numColumns =
      std::min<size_t>(table->view().num_columns(), rowType->size());
  auto columns = table->release();
  for (size_t i = 0; i < numColumns; ++i) {
    columns[i] = castDecimalColumns(
        std::move(columns[i]), rowType->childAt(i), stream, mr);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

} // namespace

CudfSplitReader::CudfSplitReader(
    std::shared_ptr<CudfHiveConnectorSplit> split,
    std::shared_ptr<const HiveTableHandle> tableHandle,
    const RowTypePtr& outputType,
    const std::vector<std::string>& readColumnNames,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    bool useExperimentalCudfReader,
    const cudf::ast::expression* subfieldFilterAst,
    std::shared_ptr<std::atomic<std::size_t>> degradedChunkReadLimit)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241},
          std::nullopt,
          fmt::format("[split:{}]", split ? split->filePath : "unknown")),
      split_(std::move(split)),
      tableHandle_(std::move(tableHandle)),
      outputType_(outputType),
      readColumnNames_(readColumnNames),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      ioStatistics_(ioStatistics),
      ioStats_(ioStats),
      cudfHiveConfig_(cudfHiveConfig),
      pool_(connectorQueryCtx->memoryPool()),
      useExperimentalCudfReader_(useExperimentalCudfReader),
      baseReaderOpts_(pool_),
      degradedChunkReadLimit_(
          degradedChunkReadLimit != nullptr
              ? std::move(degradedChunkReadLimit)
              : std::make_shared<std::atomic<std::size_t>>(0)),
      subfieldFilterAst_(subfieldFilterAst),
      pushdownFilterExpr_(subfieldFilterAst) {
  baseReaderOpts_.setDataIoStats(ioStatistics_);
  baseReaderOpts_.setMetadataIoStats(ioStatistics_);
}

void CudfSplitReader::setupReader() {
  if (useExperimentalCudfReader_) {
    createExperimentalReader();
  } else {
    createCudfReader();
  }
}

void CudfSplitReader::prepareSplitInternal(
    dwio::common::RuntimeStats& /*runtimeStats*/) {
  setupReader();
}

void CudfSplitReader::prepareSplit(dwio::common::RuntimeStats& runtimeStats) {
  // Reset existing split and split readers, if any
  resetSplit();

  // Acquire a stream from the global stream pool
  stream_ = cudfGlobalStreamPool().get_stream();

  // Perform split-specific setup.
  prepareSplitInternal(runtimeStats);

  // Update runtime stats.
  if (isSplitSkipped()) {
    runtimeStats.skippedSplits++;
    // An unbounded length means the whole file, whose size the split does not
    // carry, so it contributes no byte count.
    if (split_->length != std::numeric_limits<uint64_t>::max()) {
      runtimeStats.skippedSplitBytes += static_cast<int64_t>(split_->length);
    }
  } else {
    runtimeStats.processedSplits++;
  }
}

std::optional<std::unique_ptr<cudf::table>> CudfSplitReader::next(
    uint64_t /*size*/) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  // Record start time before reading chunk
  auto startTimeUs = getCurrentTimeMicro();

  auto chunkOpt = readNextChunk();
  if (!chunkOpt.has_value()) {
    return std::nullopt;
  }

  TotalScanTimeCallbackData* callbackData =
      new TotalScanTimeCallbackData{startTimeUs, ioStatistics_};

  // Launch host callback to calculate timing when scan completes
  cudaLaunchHostFunc(
      stream_.get(), &CudfSplitReader::totalScanTimeCalculator, callbackData);

  return std::move(chunkOpt.value());
}

std::optional<std::unique_ptr<cudf::table>> CudfSplitReader::readNextChunk() {
  auto output_mr = determineCudfMemoryResource();

  if (!useExperimentalCudfReader_) {
    // Read table using the regular cudf parquet reader
    VELOX_CHECK_NOT_NULL(splitReader_, "cudf parquet reader not present");

    if (!splitReader_->has_next()) {
      return std::nullopt;
    }

    // Reading a chunk is the largest transient allocation a scan makes, and on
    // a GPU that also holds operator state it is the allocation most likely to
    // fail. Rather than bounding every scan up front - which costs scan-heavy
    // queries that are never under pressure - degrade only the scans that
    // actually fail: halve this split's chunk read limit and read it again.
    while (true) {
      try {
        auto tableWithMetadata = splitReader_->read_chunk();
        return castDecimalColumnsToVeloxTypes(
            std::move(tableWithMetadata.tbl), outputType_, stream_, output_mr);
      } catch (const rmm::out_of_memory&) {
        if (!halveChunkReadLimitAndRebuild()) {
          throw;
        }
      }
    }
  }

  // Read table using the experimental parquet reader
  VELOX_CHECK_NOT_NULL(exptSplitReader_, "cuDF hybrid scan reader not present");
  VELOX_CHECK_NOT_NULL(hybridScanState_, "hybrid scan state not present");

  std::call_once(*hybridScanState_->isHybridScanSetup_, [&]() {
    auto rowGroupIndices = exptSplitReader_->all_row_groups(readerOptions_);

    // Filter row groups using row group byte ranges
    if (readerOptions_.get_skip_bytes() > 0 or
        readerOptions_.get_num_bytes().has_value()) {
      rowGroupIndices = exptSplitReader_->filter_row_groups_with_byte_range(
          rowGroupIndices, readerOptions_);
    }

    // Filter row groups using column chunk statistics
    if (readerOptions_.get_filter().has_value()) {
      rowGroupIndices = exptSplitReader_->filter_row_groups_with_stats(
          rowGroupIndices, readerOptions_, stream_);
    }

    // Get column chunk byte ranges to fetch
    const auto columnChunkByteRanges =
        exptSplitReader_->all_column_chunks_byte_ranges(
            rowGroupIndices, readerOptions_);

    // Fetch column chunk byte ranges
    nvtxRangePush("fetchByteRanges");

    // Tuple containing a vector of device buffers, a vector of device spans
    // for each input byte range, and a future to wait for all reads to
    // complete
    auto ioData = fetchByteRangesAsync(
        dataSource_, columnChunkByteRanges, stream_, get_temp_mr());

    // Wait for all pending reads to complete
    std::get<2>(ioData).wait();
    nvtxRangePop();

    // Save state for hybrid scan reader for future calls to `next()`
    hybridScanState_->columnChunkBuffers_ = std::move(std::get<0>(ioData));
    hybridScanState_->columnChunkData_ = std::move(std::get<1>(ioData));

    exptSplitReader_->setup_chunking_for_all_columns(
        cudfHiveConfig_->maxChunkReadLimitSession(
            connectorQueryCtx_->sessionProperties()),
        cudfHiveConfig_->maxPassReadLimitSession(
            connectorQueryCtx_->sessionProperties()),
        rowGroupIndices,
        hybridScanState_->columnChunkData_,
        readerOptions_,
        stream_,
        output_mr);
  });

  if (!exptSplitReader_->has_next_table_chunk()) {
    return std::nullopt;
  }

  auto tableWithMetadata = exptSplitReader_->materialize_all_columns_chunk();
  return castDecimalColumnsToVeloxTypes(
      std::move(tableWithMetadata.tbl), outputType_, stream_, output_mr);
}

void CudfSplitReader::resetSplit() {
  splitReader_.reset();
  exptSplitReader_.reset();
  hybridScanState_.reset();
  dataSource_.reset();
  fileMetaData_.clear();
  pushdownFilterExpr_ = subfieldFilterAst_;
  hasSplitSpecificPushdownFilter_ = false;
}

cudf::ast::expression const* CudfSplitReader::pushdownFilter() const {
  return pushdownFilterExpr_;
}

const cudf::ast::expression* CudfSplitReader::subfieldFilterAst() const {
  return subfieldFilterAst_;
}

bool CudfSplitReader::isSplitSkipped() const {
  return false;
}

bool CudfSplitReader::hasSplitSpecificPushdownFilter() const {
  return hasSplitSpecificPushdownFilter_;
}

void CudfSplitReader::setupCudfDataSource() {
  if (dataSource_) {
    return;
  }

  const auto useBufferedInput = cudfHiveConfig_->useBufferedInputSession(
      connectorQueryCtx_->sessionProperties());

  VELOX_CHECK(
      not isAbfsPath(split_->filePath) or useBufferedInput,
      "ABFS blobs require buffered input data source. "
      "Set the session property '{}' (or connector property '{}') to 'true'. "
      "Blob Path: {}.",
      CudfHiveConfig::kUseBufferedInputSession,
      CudfHiveConfig::kUseBufferedInput,
      split_->filePath);

  // Use KvikIO data source if we don't want to use the BufferedInput source
  if (not useBufferedInput) {
    VLOG(1) << fmt::format(
        "Using KvikIO data source for file: {}", split_->filePath);
    dataSource_ = std::move(
        cudf::io::make_datasources(cudf::io::source_info{split_->filePath})
            .front());
    return;
  }

  auto fileHandleCachePtr = FileHandleCachedPtr{};
  try {
    const auto fileHandleKey = FileHandleKey{
        .filename = split_->filePath,
        .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
    auto fileProperties = FileProperties{};
    fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
    VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());
  } catch (const VeloxRuntimeError& e) {
    // ABFS blobs can not fall back to KvikIO. Throw the original error.
    if (isAbfsPath(split_->filePath)) {
      VELOX_USER_FAIL(
          "Failed to generate file handle cache for ABFS blob. Ensure "
          "registerAbfsFileSystem() and registerAzureClientProvider() have "
          "been called and the connector config provides Azure credentials. "
          "Blob path: {}. Error: {}.",
          split_->filePath,
          e.what());
    }

    LOG(WARNING) << fmt::format(
        "Failed to generate file handle cache for file. Falling back to KvikIO. Path: {}",
        split_->filePath);
    dataSource_ = std::move(
        cudf::io::make_datasources(cudf::io::source_info{split_->filePath})
            .front());
    return;
  }

  // Here we keep adding new entries to CacheTTLController when new
  // fileHandles are generated, if CacheTTLController was created. Creator of
  // CacheTTLController needs to make sure a size control strategy was
  // available such as removing aged out entries.
  if (auto* cacheTTLController = cache::CacheTTLController::getInstance()) {
    cacheTTLController->addOpenFileInfo(fileHandleCachePtr->uuid.id());
  }

  auto bufferedInput =
      velox::connector::hive::BufferedInputBuilder::getInstance()->create(
          *fileHandleCachePtr,
          baseReaderOpts_,
          connectorQueryCtx_,
          ioStatistics_,
          ioStats_,
          executor_);
  if (not bufferedInput) {
    // ABFS blobs can not fall back to KvikIO
    if (isAbfsPath(split_->filePath)) {
      VELOX_USER_FAIL(
          "Failed to create buffered input data source for the ABFS blob. Ensure that the registered "
          "BufferedInputBuilder is ABFS-aware. Blob path: {}.",
          split_->filePath);
    }

    LOG(WARNING) << fmt::format(
        "Failed to create buffered input data source for file. Falling back to the KvikIO. Path: {}",
        split_->filePath);
    dataSource_ = std::move(
        cudf::io::make_datasources(cudf::io::source_info{split_->filePath})
            .front());
    return;
  }
  dataSource_ =
      std::make_unique<BufferedInputDataSource>(std::move(bufferedInput));
}

void CudfSplitReader::setupReaderOptions() {
  VELOX_CHECK_NOT_NULL(
      dataSource_,
      "CudfSplitReader does not have a datasource. Call setupCudfDataSource() first");
  auto sourceInfo = cudf::io::source_info{dataSource_.get()};

  // Reader options
  readerOptions_ =
      cudf::io::parquet_reader_options::builder(std::move(sourceInfo))
          .use_pandas_metadata(cudfHiveConfig_->isUsePandasMetadata())
          .use_arrow_schema(cudfHiveConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemas())
          .timestamp_type(cudfHiveConfig_->timestampType())
          .build();

  // Set skip_bytes and num_bytes if available
  if (split_->start != 0) {
    readerOptions_.set_skip_bytes(split_->start);
  }
  if (split_->size() != std::numeric_limits<uint64_t>::max()) {
    readerOptions_.set_num_bytes(split_->size());
  }

  if (auto* filter = pushdownFilter(); filter != nullptr) {
    readerOptions_.set_filter(*filter);
  }

  // Set column projection if needed
  if (readColumnNames_.size()) {
    readerOptions_.set_column_names(readColumnNames_);
  }

  if (prependRowIndex_) {
    readerOptions_.enable_prepend_row_index_column(true);
  }
}

rmm::device_async_resource_ref CudfSplitReader::determineCudfMemoryResource()
    const {
  return get_output_mr();
}

void CudfSplitReader::fileMetaDatas() {
  if (not fileMetaData_.empty()) {
    return;
  }

  // Setup the datasource
  setupCudfDataSource();

  // Check that the datasource is set up
  VELOX_CHECK_NOT_NULL(
      dataSource_,
      "CudfSplitReader does not have a datasource. Call setupCudfDataSource() first");

  // Wrap the existing datasource without transferring ownership.
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(cudf::io::datasource::create(dataSource_.get()));
  fileMetaData_ = cudf::io::read_parquet_footers(sources);
  VELOX_CHECK_GE(
      fileMetaData_.size(),
      1,
      "CudfSplitReader failed to read any parquet metadatas");

  if (pushdownFilterBuilder_) {
    VELOX_CHECK_EQ(
        fileMetaData_.size(),
        1,
        "Split-specific pushdown filters require exactly one Parquet metadata");
    pushdownFilterExpr_ = pushdownFilterBuilder_(fileMetaData_.front());
    VELOX_CHECK_NOT_NULL(
        pushdownFilterExpr_,
        "Split-specific pushdown filter builder must return an expression");
    hasSplitSpecificPushdownFilter_ = true;
  }
}

void CudfSplitReader::createCudfReader() {
  // Read file metadatas
  fileMetaDatas();

  // Setup reader options
  setupReaderOptions();

  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(cudf::io::datasource::create(dataSource_.get()));

  // Create a parquet reader
  splitReader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
      currentChunkReadLimit(),
      currentPassReadLimit(),
      std::move(sources),
      std::move(fileMetaData_),
      readerOptions_,
      stream_,
      determineCudfMemoryResource());

  // Metadata ingested
  fileMetaData_.clear();
}

uint64_t CudfSplitReader::projectedDecodeBytes() const {
  if (fileMetaData_.empty()) {
    return 0;
  }
  // total_uncompressed_size per column chunk, summed over the row groups of
  // this split and over only the columns actually read. That last part is what
  // makes this worth computing: two splits of the same file size decode wildly
  // different amounts depending on the projection, and it is the decoded size
  // that has to fit.
  const std::unordered_set<std::string> wanted(
      readColumnNames_.begin(), readColumnNames_.end());
  uint64_t bytes = 0;
  for (const auto& metadata : fileMetaData_) {
    for (const auto& rowGroup : metadata.row_groups) {
      for (const auto& column : rowGroup.columns) {
        // A leaf's path is its ancestry; the first element is the top-level
        // column, which is what the scan names.
        if (column.meta_data.path_in_schema.empty() ||
            wanted.count(column.meta_data.path_in_schema.front()) == 0) {
          continue;
        }
        bytes += static_cast<uint64_t>(
            std::max<int64_t>(column.meta_data.total_uncompressed_size, 0));
      }
    }
  }
  return bytes;
}

std::size_t CudfSplitReader::currentPassReadLimit() const {
  const auto configured = cudfHiveConfig_->maxPassReadLimitSession(
      connectorQueryCtx_->sessionProperties());
  if (configured != 0) {
    return configured;
  }

  // A pass is the set of row groups the reader decodes together, and it is a
  // different quantity from the chunk it hands back. Bounding the chunk alone
  // was measured to leave the decode footprint untouched: a wide projection
  // still materialised most of a file, which at TPC-H SF1000 is 12 GB of
  // customer per scan driver and is why Q10 could not run. Bounding the pass
  // everywhere is not the answer either - it was measured to cost other
  // queries their own headroom and lose them instead.
  //
  // So bound it exactly where it is needed, which the split can work out for
  // itself: sum the uncompressed size of the columns this scan actually reads.
  // A split that would decode more than its share of the device gets a bound;
  // one that would not is left alone and pays nothing. The same file gives
  // different answers for different queries, which is the point - Q9 reads two
  // narrow columns of orders and needs no bound, Q10 reads nearly all of
  // customer and does.
  const auto budget = gpu_defaults::parquetPassReadBytes(
      0, CudfConfig::getInstance().maxDriversPerTaskHint);
  if (budget == 0 || projectedDecodeBytes() <= budget) {
    return 0;
  }
  return static_cast<std::size_t>(budget);
}

std::size_t CudfSplitReader::currentChunkReadLimit() const {
  const auto learned = degradedChunkReadLimit_->load(std::memory_order_relaxed);
  if (learned != 0) {
    return learned;
  }
  return cudfHiveConfig_->maxChunkReadLimitSession(
      connectorQueryCtx_->sessionProperties());
}

bool CudfSplitReader::halveChunkReadLimitAndRebuild() {
  // Floor chosen so a split still makes forward progress; below this the chunk
  // is small enough that the failure is not the scan's to solve.
  constexpr std::size_t kMinChunkReadLimit = 32UL << 20;
  const auto current = currentChunkReadLimit();
  // An unlimited chunk has no size to halve, so start from a bound that is
  // large enough to stay efficient but small enough to relieve the pressure.
  const auto next = current == 0 ? (1UL << 30) : current / 2;
  if (next < kMinChunkReadLimit) {
    return false;
  }
  degradedChunkReadLimit_->store(next, std::memory_order_relaxed);

  LOG(WARNING) << "cuDF scan hit an allocation failure; retrying this split "
               << "with a " << next << " byte chunk read limit and a "
               << currentPassReadLimit() << " byte pass read limit";
  // read_chunk() has no partial-consumption contract, so the split is restarted
  // from the beginning with the reduced limit.
  splitReader_.reset();
  createCudfReader();
  return splitReader_ != nullptr;
}

void CudfSplitReader::createExperimentalReader() {
  // Read file metadatas
  fileMetaDatas();

  // Setup reader options
  setupReaderOptions();

  VELOX_CHECK_EQ(
      fileMetaData_.size(),
      1,
      "cuDF experimental reader requires exactly one parquet metadata");

  // Create a hybrid scan reader
  nvtxRangePush("hybridScanReader");
  auto reader = std::make_unique<CudfHybridScanReader>(
      std::move(fileMetaData_.front()), readerOptions_);
  nvtxRangePop();

  exptSplitReader_ = std::move(reader);
  hybridScanState_ = std::make_unique<HybridScanState>();

  // Metadata ingested
  fileMetaData_.clear();
}

void CudfSplitReader::totalScanTimeCalculator(void* userData) {
  TotalScanTimeCallbackData* data =
      static_cast<TotalScanTimeCallbackData*>(userData);

  // Record end time in callback
  auto endTimeUs = getCurrentTimeMicro();

  // Calculate elapsed time in microseconds and convert to nanoseconds
  auto elapsedUs = endTimeUs - data->startTimeUs;
  auto elapsedNs = elapsedUs * 1000; // Convert microseconds to nanoseconds

  // Update totalScanTime
  data->ioStatistics->incTotalScanTimeNs(elapsedNs);

  delete data;
}

} // namespace facebook::velox::cudf_velox::connector::hive
