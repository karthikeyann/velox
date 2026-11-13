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
#include "velox/experimental/cudf/connectors/hive/CudfDynamicFilter.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/caching/CacheTTLController.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/common/BufferedInputBuilder.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/Filter.h"
#ifdef VELOX_ENABLE_ABFS
#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"
#endif

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <folly/ScopeGuard.h>
#include <folly/lang/Bits.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <system_error>
#include <tuple>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// Returns the total size of 'byteRanges'.
uint64_t numBytesOf(
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges) {
  uint64_t numBytes{0};
  for (const auto& byteRange : byteRanges) {
    numBytes += byteRange.size();
  }
  return numBytes;
}

} // namespace

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

} // namespace

std::unique_ptr<cudf::table> castDecimalColumnsToVeloxTypes(
    std::unique_ptr<cudf::table>&& table,
    std::span<const TypePtr> columnTypes,
    size_t numPrependedColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_EQ(
      numPrependedColumns + columnTypes.size(),
      table->view().num_columns(),
      "Read column types must match the cuDF table width");
  auto columns = table->release();
  for (size_t i = 0; i < columnTypes.size(); ++i) {
    const auto columnIndex = numPrependedColumns + i;
    columns[columnIndex] = castDecimalColumns(
        std::move(columns[columnIndex]), columnTypes[i], stream, mr);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

std::optional<cudf::type_id> constantTimestampType(
    const CudfHiveConfig& config,
    const config::ConfigBase* session) {
  const auto timestampType = config.timestampTypeSession(session).id();
  if (timestampType == cudf::type_id::TIMESTAMP_DAYS) {
    return std::nullopt;
  }
  return timestampType;
}

CudfSplitReader::CudfSplitReader(
    std::shared_ptr<CudfHiveConnectorSplit> split,
    std::shared_ptr<const HiveTableHandle> tableHandle,
    const RowTypePtr& outputType,
    const std::vector<std::string>& readColumnNames,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<FetchAheadContext>& fetchAheadContext,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    const cudf::ast::expression* subfieldFilterAst,
    const common::SubfieldFilters* dynamicFilters)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241},
          std::nullopt,
          fmt::format("[split:{}]", split ? split->filePath : "unknown")),
      split_(std::move(split)),
      tableHandle_(std::move(tableHandle)),
      outputType_(outputType),
      readColumnNames_(readColumnNames),
      readColumnTypes_(outputType->children()),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      ioStatistics_(ioStatistics),
      ioStats_(ioStats),
      cudfHiveConfig_(cudfHiveConfig),
      pool_(connectorQueryCtx->memoryPool()),
      fetchAheadContext_(fetchAheadContext),
      fetchAheadBudget_(
          fetchAheadContext_ != nullptr ? &fetchAheadContext_->budget()
                                        : nullptr),
      baseReaderOpts_(pool_),
      subfieldFilterAst_(subfieldFilterAst),
      pushdownFilterExpr_(subfieldFilterAst) {
  VELOX_CHECK_GE(
      readColumnNames_.size(),
      readColumnTypes_.size(),
      "Read columns must include all output columns");
  if (readColumnNames_.size() > readColumnTypes_.size()) {
    const auto& dataColumns = tableHandle_->dataColumns();
    VELOX_CHECK_NOT_NULL(
        dataColumns,
        "Table schema is required to resolve filter-only column types");
    for (size_t i = readColumnTypes_.size(); i < readColumnNames_.size(); ++i) {
      readColumnTypes_.push_back(dataColumns->findChild(readColumnNames_[i]));
    }
  }
  VELOX_DCHECK_EQ(readColumnNames_.size(), readColumnTypes_.size());
  if (fetchAheadBudget_ != nullptr) {
    fetchAheadScan_ = std::hash<std::string>{}(fmt::format(
        "{}/{}/{}",
        connectorQueryCtx_->queryId(),
        connectorQueryCtx_->taskId(),
        connectorQueryCtx_->planNodeId()));
    fetchAheadBudget_->attach(fetchAheadScan_);
  }
  if (dynamicFilters) {
    const auto filterRowType = ROW(readColumnNames_, readColumnTypes_);
    for (const auto& [field, filter] : *dynamicFilters) {
      auto name = field.toString();
      const auto type = filterRowType->findChild(name);
      VELOX_CHECK_NOT_NULL(type);
      readerDynamicFilters_.emplace_back(
          std::move(name), veloxToCudfDataType(type), filter);
    }
  }
  baseReaderOpts_.setDataIoStats(ioStatistics_);
  baseReaderOpts_.setMetadataIoStats(ioStatistics_);

  // The cuDF path doesn't use connector::hive::configureReaderOptions() to
  // apply configurations so load-quantum, max-coalesced-bytes and
  // max-coalesced-distance-bytes are applied here so a tuned load-quantum
  // reaches CachedBufferedInput.
  const ::facebook::velox::connector::hive::HiveConfig hiveConfig(
      cudfHiveConfig_->config());
  const auto* sessionProperties = connectorQueryCtx_->sessionProperties();
  baseReaderOpts_.setLoadQuantum(hiveConfig.loadQuantum(sessionProperties));
  baseReaderOpts_.setMaxCoalesceBytes(
      hiveConfig.maxCoalescedBytes(sessionProperties));
  baseReaderOpts_.setMaxCoalesceDistance(
      hiveConfig.maxCoalescedDistanceBytes(sessionProperties));
  caseInsensitiveColumnNames_ =
      hiveConfig.isFileColumnNamesReadAsLowerCase(sessionProperties);
}

CudfSplitReader::~CudfSplitReader() {
  SCOPE_EXIT {
    if (fetchAheadBudget_ != nullptr) {
      fetchAheadBudget_->detach(fetchAheadScan_);
    }
  };
  // A split abandoned before it is read, e.g. when the task is cancelled while
  // the preloader prepares it, can still have reads in flight or queued.
  try {
    withdrawFetchOffer();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to take back a fetch offered ahead: " << e.what();
  }
  if (passState_ == nullptr) {
    return;
  }
  try {
    releaseCurrentPassData();
  } catch (const std::exception& e) {
    // The data of a failed read is being dropped anyway, so the failure must
    // not propagate out of the destructor.
    LOG(ERROR) << fmt::format(
        "Failed to discard the reads of an abandoned split. Path: {}. Error: {}.",
        split_ != nullptr ? split_->filePath : "unknown",
        e.what());
  }
}

cuda::stream_ref CudfSplitReader::fetchAheadStream() {
  if (!fetchAheadStream_.has_value()) {
    fetchAheadStream_ = fetchAheadContext_->nextStream();
  }
  return *fetchAheadStream_;
}

void CudfSplitReader::prepareSplitInternal(
    dwio::common::RuntimeStats& /*runtimeStats*/) {
  // Read file metadata and cache schema information
  cacheSchemaFromMetadata();

  // No columns to read (e.g. `count(*)`): take the row count from the footer.
  noColumnsToRead_ = readColumnNames_.empty();
  if (noColumnsToRead_) {
    VELOX_CHECK_NULL(
        pushdownFilter(),
        "A footer-derived row count cannot honor a pushed-down filter");
    return;
  }

  createCudfReader();
}

void CudfSplitReader::cacheSchemaFromMetadata() {
  fileMetaDatas();

  VELOX_CHECK_EQ(
      fileMetaDatas_.size(),
      1,
      "Expected a single parquet footer for the split's data file");
  const auto& meta = fileMetaDatas_.front();
  VELOX_CHECK(not meta.schema.empty(), "Parquet footer schema is empty");
  VELOX_CHECK_GE(meta.num_rows, 0, "Parquet footer reports negative row count");
  std::tie(baseReadOffset_, splitRowCount_) = computeSplitRowRange();

  const auto& root = meta.schema.front();
  fileColumnNames_.clear();
  fileColumnNames_.reserve(root.children_idx.size());
  for (const auto childIdx : root.children_idx) {
    VELOX_CHECK_LT(
        childIdx,
        meta.schema.size(),
        "Parquet schema child index out of range");
    const auto& name = meta.schema[childIdx].name;
    fileColumnNames_.insert(
        caseInsensitiveColumnNames_
            ? ::facebook::velox::functions::stringImpl::utf8StrToLowerCopy(name)
            : name);
  }
}

std::pair<std::size_t, std::size_t> CudfSplitReader::computeSplitRowRange()
    const {
  // Note: This function implements the same logic as cuDF's hybrid scan
  // reader's `filter_row_groups_with_byte_range()` API
  const auto rowGroupOffset = [](const auto& rowGroup) {
    if (rowGroup.file_offset.has_value()) {
      return rowGroup.file_offset.value();
    }
    if (rowGroup.columns.front().file_offset != 0) {
      return rowGroup.columns.front().file_offset;
    }
    const auto& column = rowGroup.columns.front().meta_data;
    return column.dictionary_page_offset != 0
        ? std::min(column.dictionary_page_offset, column.data_page_offset)
        : column.data_page_offset;
  };

  std::size_t startRow{0};
  std::size_t numRows{0};
  for (const auto& rowGroup : fileMetaDatas_.front().row_groups) {
    VELOX_CHECK(
        not rowGroup.columns.empty(),
        "Parquet footer reports a row group with no column chunks");
    const auto offset = rowGroupOffset(rowGroup);
    if (std::cmp_less(offset, split_->start)) {
      startRow += rowGroup.num_rows;
    } else if (offset - split_->start < split_->size()) {
      numRows += rowGroup.num_rows;
    }
  }
  return {startRow, numRows};
}

void CudfSplitReader::setPushdownFilter(const cudf::ast::expression* filter) {
  VELOX_CHECK_NOT_NULL(filter);
  pushdownFilterExpr_ = filter;
  hasSplitSpecificPushdownFilter_ = true;
}

void CudfSplitReader::setFilterFirstFilter(
    const cudf::ast::expression* filter,
    std::vector<std::string> filterColumns) {
  VELOX_CHECK_NOT_NULL(filter);
  VELOX_CHECK(!filterColumns.empty(), "A filter must reference a column");
  VELOX_CHECK_NULL(
      splitReader_, "The filter of a prepared split cannot be replaced");
  filterFirstFilter_ = filter;
  filterFirstColumns_ = std::move(filterColumns);
}

void CudfSplitReader::prepareSplit(dwio::common::RuntimeStats& runtimeStats) {
  stream_ = cudfGlobalStreamPool().get_stream();
  fileMetaDatas();
  if (emptySplit_) {
    return;
  }
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
  runtimeStats.skippedStrides += numDynamicFilterSkippedRowGroups_;
  runtimeStats.processedStrides += numSelectedRowGroups_;
}

std::optional<CudfSplitReader::TableChunk> CudfSplitReader::next(
    uint64_t /*size*/,
    const RowFilter& rowFilter) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  rowFilter_ = rowFilter ? &rowFilter : nullptr;
  SCOPE_EXIT {
    rowFilter_ = nullptr;
  };

  // Record start time before reading chunk
  auto startTimeUs = SteadyTime::now<std::chrono::microseconds>();

  auto chunkOpt = readNextChunk();
  if (!chunkOpt.has_value()) {
    return std::nullopt;
  }

  TotalScanTimeCallbackData* callbackData =
      new TotalScanTimeCallbackData{startTimeUs, ioStatistics_};

  // Launch host callback to calculate timing when scan completes
  cudaLaunchHostFunc(
      stream_.get(), &CudfSplitReader::totalScanTimeCalculator, callbackData);

  return chunkOpt;
}

void CudfSplitReader::setConnectorQueryCtx(
    const ConnectorQueryCtx* connectorQueryCtx) {
  VELOX_CHECK_NOT_NULL(connectorQueryCtx);
  // A fetch of the reader may be starting ahead on another thread.
  withdrawFetchOffer();
  // A preloaded split can be adopted by a driver with a different memory pool.
  // Only the context pointer changes; 'pool_' and its buffers stay with the
  // pool that allocated them.
  connectorQueryCtx_ = connectorQueryCtx;
}

std::optional<CudfSplitReader::TableChunk> CudfSplitReader::readNextChunk() {
  if (noColumnsToRead_) {
    // A Velox vector holds at most `vector_size_t` rows, so the footer row
    // count of a larger split is returned over several tables.
    const auto numRows = std::min<std::size_t>(
        splitRowCount_,
        static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()));
    if (numRows == 0) {
      return std::nullopt;
    }
    splitRowCount_ -= numRows;
    return TableChunk{
        std::make_unique<cudf::table>(
            std::vector<std::unique_ptr<cudf::column>>{}),
        static_cast<vector_size_t>(numRows)};
  }

  VELOX_CHECK_NOT_NULL(splitReader_, "cuDF parquet reader not present");
  VELOX_CHECK_NOT_NULL(passState_, "Row group pass state not present");

  auto outputMr = determineCudfMemoryResource();

  while (not passState_->isChunkingSetup) {
    if (passState_->currentPass >= passState_->passes.size()) {
      return std::nullopt;
    }
    if (not readsFilterColumnsFirst_) {
      setupChunkingForCurrentPass(outputMr);
      ++readStats_.numOneStepPasses;
    } else if (not setupFilterFirstPass(outputMr)) {
      // The filters keep no row of this pass.
      finishCurrentPass();
      continue;
    }
    VELOX_CHECK(
        splitReader_->has_next_table_chunk(),
        "cuDF row group pass did not produce a table chunk");
  }

  auto chunk = readsFilterColumnsFirst_
      ? readFilterFirstChunk(outputMr)
      : std::move(splitReader_->materialize_all_columns_chunk().tbl);
  const bool rowFilterApplied = passState_->rowFilterApplied;

  // This was the last chunk of the pass. Drop its fetch buffers and begin
  // I/O for the next pass, unless it was fetched ahead, while the caller
  // consumes this table.
  if (not splitReader_->has_next_table_chunk()) {
    finishCurrentPass();
  }

  auto table = castDecimalColumnsToVeloxTypes(
      std::move(chunk),
      readColumnTypes_,
      prependRowIndex_ ? 1 : 0,
      stream_,
      outputMr);
  const auto numRows = table->num_rows();
  return TableChunk{std::move(table), numRows, rowFilterApplied};
}

void CudfSplitReader::startColumnChunkFetch() {
  // From here on the fetch-ahead budget leaves the fetch to this thread.
  withdrawFetchOffer();
  if (passState_ != nullptr and
      passState_->current.stage == FetchAheadBudget::Stage::kAhead) {
    passState_->current.stage = FetchAheadBudget::Stage::kAwaited;
    fetchAheadBudget_->claim(fetchAheadScan_);
  }
  // Return if there's no passes or if the fetch is already started or if
  // the chunking is already setup.
  if (passState_ == nullptr or
      passState_->currentPass >= passState_->passes.size() or
      passState_->current.fetch.pending.valid() or
      passState_->isChunkingSetup) {
    return;
  }

  const auto& byteRanges = passState_->byteRanges[passState_->currentPass];
  startFetch(
      passState_->current,
      byteRanges,
      numBytesOf(byteRanges.filter) + numBytesOf(byteRanges.other),
      /*isHeld=*/false,
      stream_);
}

void CudfSplitReader::startFetch(
    PassFetch& pass,
    const RowGroupPassState::ColumnChunkByteRanges& byteRanges,
    uint64_t numBytes,
    bool isHeld,
    cuda::stream_ref stream) {
  nvtxRangePush("startColumnChunkFetch");
  SCOPE_EXIT {
    nvtxRangePop();
  };
  const auto fetch =
      [&](const std::vector<cudf::io::text::byte_range_info>& columnChunks) {
        return fetchByteRangesAsync(
            dataSource_,
            columnChunks,
            serializeIoRequests_,
            stream,
            {get_temp_mr(), get_temp_mr()});
      };
  if (readsFilterColumnsFirst_) {
    // The filter column chunks are requested first, so the other column
    // chunks keep arriving while the filter columns are decoded.
    pass.filterFetch = fetch(byteRanges.filter);
  }
  pass.fetch = fetch(byteRanges.other);
  // Counted once the fetch exists: a fetch that fails to start holds nothing.
  if (fetchAheadBudget_ != nullptr) {
    if (!isHeld) {
      fetchAheadBudget_->hold(numBytes, fetchAheadScan_);
    }
    pass.heldBytes = numBytes;
    pass.stage = isHeld ? FetchAheadBudget::Stage::kAhead
                        : FetchAheadBudget::Stage::kAwaited;
  }
  // Only fetches the budget counted already were started ahead.
  pass.isOnFetchAheadStream = isHeld;
}

bool CudfSplitReader::offerColumnChunkFetch() {
  if (fetchAheadBudget_ == nullptr or executor_ == nullptr) {
    return false;
  }
  if (passState_ == nullptr or fetchAhead_ != nullptr or
      passState_->currentPass >= passState_->passes.size() or
      passState_->current.fetch.pending.valid() or
      passState_->isChunkingSetup) {
    return true;
  }
  offerPassFetch(
      passState_->currentPass,
      passState_->current,
      fetchAheadStream(),
      FetchAheadBudget::Distance::kLater);
  return true;
}

void CudfSplitReader::offerPassFetch(
    size_t pass,
    PassFetch& target,
    cuda::stream_ref stream,
    FetchAheadBudget::Distance distance) {
  // A copy for the thread that starts the fetch, which does not use the
  // reader.
  auto byteRanges = passState_->byteRanges[pass];
  const auto numBytes =
      numBytesOf(byteRanges.filter) + numBytesOf(byteRanges.other);
  // The start allocates the device memory of the pass and submits its reads.
  // That is not work for the thread that gives the offer room, which only
  // released memory.
  fetchAhead_ = std::make_shared<PostedFetchStart>(
      fetchAheadBudget_,
      numBytes,
      fetchAheadScan_,
      [context = fetchAheadContext_](std::function<void()> task) {
        context->post(std::move(task));
      },
      [this, &target, stream, byteRanges = std::move(byteRanges), numBytes]() {
        ensureCudaContextForThread();
        startFetch(target, byteRanges, numBytes, /*isHeld=*/true, stream);
      });
  fetchAheadBudget_->offer(fetchAhead_, numBytes, distance, fetchAheadScan_);
}

void CudfSplitReader::withdrawFetchOffer() {
  if (fetchAhead_ != nullptr) {
    fetchAheadBudget_->withdraw(fetchAhead_.get());
    // Waits for a start that is running, and takes back one that is only
    // queued: the caller then starts the fetch itself.
    fetchAhead_->withdraw();
    fetchAhead_.reset();
  }
}

void CudfSplitReader::setColumnChunkByteRanges() {
  // Byte ranges are flattened across sources; the source index map is only
  // needed once a reader spans multiple data sources.
  const auto& passes = passState_->passes;
  auto& byteRanges = passState_->byteRanges;
  byteRanges.resize(passes.size());
  if (not readsFilterColumnsFirst_) {
    for (size_t i = 0; i < passes.size(); ++i) {
      byteRanges[i].other =
          splitReader_->all_column_chunks_byte_ranges(passes[i], readerOptions_)
              .first;
    }
    return;
  }
  for (size_t i = 0; i < passes.size(); ++i) {
    byteRanges[i].filter =
        splitReader_
            ->filter_column_chunks_byte_ranges(passes[i], filterFirstOptions_)
            .first;
  }
  for (size_t i = 0; i < passes.size(); ++i) {
    byteRanges[i].other =
        splitReader_
            ->payload_column_chunks_byte_ranges(passes[i], filterFirstOptions_)
            .first;
  }
}

void CudfSplitReader::awaitCurrentPassFetch(ByteRangeFetch& fetch) {
  // No-op when the fetch was already started while preparing the split.
  startColumnChunkFetch();

  // Wait for the reads to complete.
  uint64_t waitNanos{0};
  {
    // The budget starts nothing ahead while this driver waits.
    if (fetchAheadBudget_ != nullptr) {
      fetchAheadBudget_->waitBegin();
    }
    SCOPE_EXIT {
      if (fetchAheadBudget_ != nullptr) {
        fetchAheadBudget_->waitEnd();
      }
    };
    NanosecondWallTimer timer(&waitNanos);
    fetch.wait();
  }
  // Bytes of this pass are on the device and its decode begins: the
  // time to fetch the next pass, within the fetch-ahead budget. Offered
  // before the budget learns that the bytes arrived, which starts what is on
  // offer: the next pass goes before the splits prepared ahead.
  if (fetchAheadBudget_ != nullptr and executor_ != nullptr and
      passState_->currentPass + 1 < passState_->passes.size()) {
    offerPassFetch(
        passState_->currentPass + 1,
        passState_->next,
        fetchAheadStream(),
        FetchAheadBudget::Distance::kNext);
  }
  if (passState_->current.stage == FetchAheadBudget::Stage::kAwaited) {
    passState_->current.stage = FetchAheadBudget::Stage::kReceived;
    fetchAheadBudget_->receive(fetchAheadScan_);
  }
  if (ioStats_ != nullptr) {
    ioStats_->addCounter(
        kFetchWaitWallNanos,
        RuntimeCounter(
            static_cast<int64_t>(waitNanos), RuntimeCounter::Unit::kNanos));
  }
  // A data source may queue its copies on the stream of the fetch. Decode
  // runs on the reader's stream, so those of another stream must have run.
  if (passState_->current.isOnFetchAheadStream) {
    CUDF_CUDA_TRY(cudaStreamSynchronize(fetchAheadStream().get()));
  }
}

void CudfSplitReader::waitForFetch(ByteRangeFetch& fetch) {
  {
    // The budget starts nothing ahead while this driver waits.
    if (fetchAheadBudget_ != nullptr) {
      fetchAheadBudget_->waitBegin();
    }
    SCOPE_EXIT {
      if (fetchAheadBudget_ != nullptr) {
        fetchAheadBudget_->waitEnd();
      }
    };
    fetch.wait();
  }
  // As in awaitCurrentPassFetch(): the copies a data source queued on the
  // stream of the fetch must have run before the reader's stream decodes.
  if (passState_->current.isOnFetchAheadStream) {
    CUDF_CUDA_TRY(cudaStreamSynchronize(fetchAheadStream().get()));
  }
}

void CudfSplitReader::setupChunkingForCurrentPass(
    rmm::device_async_resource_ref mr) {
  awaitCurrentPassFetch(passState_->current.fetch);

  splitReader_->setup_chunking_for_all_columns(
      chunkReadLimit_,
      passReadLimit_,
      passState_->passes[passState_->currentPass],
      passState_->current.fetch.data,
      readerOptions_,
      stream_,
      mr);

  passState_->isChunkingSetup = true;
}

namespace {

// Returns the number of true rows of the BOOL8 column 'mask', which has no
// nulls.
int64_t countTrueRows(const cudf::column_view& mask, cuda::stream_ref stream) {
  const auto sum = cudf::reduce(
      mask,
      *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::INT64},
      stream,
      get_temp_mr());
  return static_cast<const cudf::numeric_scalar<int64_t>&>(*sum).value(stream);
}

} // namespace

bool CudfSplitReader::setupFilterFirstPass(rmm::device_async_resource_ref mr) {
  namespace hybrid = cudf::io::parquet::experimental;
  // The filter column chunks are enough to begin with.
  awaitCurrentPassFetch(passState_->current.filterFetch);

  const auto& rowGroupIndices = passState_->passes[passState_->currentPass];
  const cudf::memory_resources temporary{get_temp_mr(), get_temp_mr()};
  auto rowMask = splitReader_->build_all_true_row_mask(
      rowGroupIndices, stream_, temporary);
  ++readStats_.numFilterFirstPasses;
  readStats_.numFilterFirstRows += rowMask->size();

  auto rowMaskView = rowMask->mutable_view();
  // Every row is a candidate, so no page of a filter column can be skipped.
  auto filterTable = splitReader_->materialize_filter_columns(
      rowGroupIndices,
      passState_->current.filterFetch.data,
      rowMaskView,
      hybrid::use_data_page_mask::NO,
      filterFirstOptions_,
      stream_,
      mr);
  // The column chunks of the filter columns stay with the pass until it is
  // released: the fetch-ahead budget counts the pass as one.

  // The reader returns the columns its filter references, in its own order.
  const auto& schema = filterTable.metadata.schema_info;
  VELOX_CHECK_EQ(schema.size(), filterTable.tbl->num_columns());
  const auto foldCase = [&](const std::string& name) {
    return caseInsensitiveColumnNames_
        ? ::facebook::velox::functions::stringImpl::utf8StrToLowerCopy(name)
        : name;
  };
  auto& filterReadColumns = passState_->filterReadColumns;
  filterReadColumns.clear();
  for (const auto& column : schema) {
    const auto it = std::find(
        readColumnNames_.begin(),
        readColumnNames_.end(),
        foldCase(column.name));
    VELOX_CHECK(
        it != readColumnNames_.end(),
        "Filter column is not a read column: {}",
        column.name);
    filterReadColumns.push_back(std::distance(readColumnNames_.begin(), it));
  }

  auto table = std::move(filterTable.tbl);
  passState_->rowFilterApplied = false;
  if (rowFilter_ != nullptr and table->num_rows() > 0) {
    const auto keep = (*rowFilter_)(table->view(), filterReadColumns, stream_);
    if (keep != nullptr) {
      VELOX_CHECK_EQ(keep->size(), table->num_rows());
      passState_->rowFilterApplied = true;
      const auto numKeptRows = countTrueRows(keep->view(), stream_);
      readStats_.numRowFilterRows += table->num_rows();
      readStats_.numRowFilterKeptRows += numKeptRows;
      // A filter that keeps every row leaves the mask and the table as they
      // are.
      if (numKeptRows < table->num_rows()) {
        // The rows of 'table' are the true entries of the row mask, in order.
        auto narrowed = cudf::boolean_mask_scatter(
                            cudf::table_view{{keep->view()}},
                            cudf::table_view{{rowMask->view()}},
                            rowMask->view(),
                            stream_,
                            get_temp_mr())
                            ->release();
        rowMask = std::move(narrowed.front());
        table = cudf::apply_retention_mask(
            table->view(), keep->view(), stream_, mr);
      }
    }
  }
  readStats_.numFilterFirstKeptRows += table->num_rows();
  if (table->num_rows() == 0) {
    return false;
  }

  waitForFetch(passState_->current.fetch);
  splitReader_->setup_chunking_for_payload_columns(
      chunkReadLimit_,
      passReadLimit_,
      rowGroupIndices,
      rowMask->view(),
      hybrid::use_data_page_mask::YES,
      passState_->current.fetch.data,
      filterFirstOptions_,
      stream_,
      mr);

  passState_->rowMask = std::move(rowMask);
  passState_->filterTable = std::move(table);
  passState_->numFilterRowsReturned = 0;
  passState_->isChunkingSetup = true;
  return true;
}

std::unique_ptr<cudf::table> CudfSplitReader::readFilterFirstChunk(
    rmm::device_async_resource_ref mr) {
  // The mask has to be the one the chunked read was set up with, unchanged:
  // the reader chose the pages to decode from it.
  auto payloadColumns =
      splitReader_
          ->materialize_payload_columns_chunk(passState_->rowMask->view())
          .tbl->release();
  const auto& filterReadColumns = passState_->filterReadColumns;
  VELOX_CHECK_EQ(
      payloadColumns.size() + filterReadColumns.size(),
      readColumnNames_.size(),
      "Filter-first read returned an unexpected number of columns");
  VELOX_CHECK(
      !payloadColumns.empty(),
      "Filter-first read returned no column besides the filter columns");

  // A chunk holds the next kept rows of the pass, as do the filter columns
  // from the first row not returned yet.
  const auto numRows = payloadColumns.front()->size();
  const auto begin = passState_->numFilterRowsReturned;
  auto& filterTable = passState_->filterTable;
  VELOX_CHECK_LE(begin + numRows, filterTable->num_rows());
  std::vector<std::unique_ptr<cudf::column>> filterColumns;
  if (begin == 0 and not splitReader_->has_next_table_chunk()) {
    // The only chunk of the pass takes the filter columns as they are.
    VELOX_CHECK_EQ(numRows, filterTable->num_rows());
    filterColumns = filterTable->release();
  } else {
    const auto rows =
        cudf::slice(filterTable->view(), {begin, begin + numRows}, stream_)
            .front();
    filterColumns = cudf::table(rows, stream_, mr).release();
  }
  passState_->numFilterRowsReturned += numRows;

  std::vector<std::unique_ptr<cudf::column>> columns(readColumnNames_.size());
  for (size_t i = 0; i < filterReadColumns.size(); ++i) {
    columns[filterReadColumns[i]] = std::move(filterColumns[i]);
  }
  // The other columns come in read column order.
  auto payloadColumn = payloadColumns.begin();
  for (auto& column : columns) {
    if (column == nullptr) {
      column = std::move(*payloadColumn++);
    }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

void CudfSplitReader::releasePassData(PassFetch& pass, bool isDecoded) {
  std::exception_ptr failure;
  try {
    // Queued or active reads can still write into the buffers being released.
    // Both fetches are waited for before either failure is rethrown.
    try {
      pass.filterFetch.wait();
    } catch (...) {
      failure = std::current_exception();
    }
    pass.fetch.wait();
    // Buffers of a fetch-ahead stream are freed on that stream, where the
    // next fetch reuses them. A data source may have queued copies there.
    // Decode read the buffers on the reader's stream, which has drained when
    // decode returned but not after it failed.
    if (pass.isOnFetchAheadStream) {
      CUDF_CUDA_TRY(cudaStreamSynchronize(fetchAheadStream().get()));
      if (isDecoded) {
        CUDF_CUDA_TRY(cudaStreamSynchronize(stream_.get()));
      }
    }
  } catch (...) {
    failure = failure ? failure : std::current_exception();
  }
  pass.filterFetch = {};
  pass.fetch = {};
  pass.isOnFetchAheadStream = false;
  if (pass.stage.has_value()) {
    fetchAheadBudget_->release(
        std::exchange(pass.heldBytes, 0),
        fetchAheadScan_,
        *std::exchange(pass.stage, std::nullopt));
  }
  if (failure != nullptr) {
    std::rethrow_exception(failure);
  }
}

void CudfSplitReader::releaseCurrentPassData() {
  // The split is done with or given up: reads that have not begun need not
  // run. The KvikIO data source has no such switch.
  if (auto* bufferedInput =
          dynamic_cast<BufferedInputDataSource*>(dataSource_.get())) {
    bufferedInput->abandonDeviceReads();
  }
  // The fetch of the pass after it goes with it. Both are released even when
  // the first fails.
  SCOPE_EXIT {
    try {
      releasePassData(passState_->next, /*isDecoded=*/false);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to discard the reads of a pass fetched ahead: "
                 << e.what();
    }
  };
  passState_->rowMask.reset();
  passState_->filterTable.reset();
  passState_->rowFilterApplied = false;
  // Decode may have started on it and failed.
  releasePassData(passState_->current, /*isDecoded=*/true);
}

void CudfSplitReader::advancePass() {
  // The offer of the next pass, if it is still out, ends here: its start has
  // run, or is taken back and the caller starts the fetch itself.
  withdrawFetchOffer();
  passState_->rowMask.reset();
  passState_->filterTable.reset();
  passState_->rowFilterApplied = false;
  releasePassData(passState_->current, /*isDecoded=*/true);
  passState_->isChunkingSetup = false;
  ++passState_->currentPass;
  passState_->current = std::move(passState_->next);
  passState_->next = {};
}

void CudfSplitReader::finishCurrentPass() {
  advancePass();
  if (passState_->currentPass < passState_->passes.size()) {
    startColumnChunkFetch();
  } else {
    passState_->passes.clear();
  }
}

void CudfSplitReader::resetSplit() {
  withdrawFetchOffer();
  if (passState_ != nullptr) {
    releaseCurrentPassData();
  }
  splitReader_.reset();
  passState_.reset();
  dataSource_.reset();
  fileMetaDatas_.clear();
  pushdownFilterExpr_ = subfieldFilterAst_;
  dynamicRowGroups_.clear();
  dynamicFiltersSatisfied_ = false;
  numDynamicFilterSkippedRowGroups_ = 0;
  numSelectedRowGroups_ = 0;
  hasSplitSpecificPushdownFilter_ = false;
  filterFirstFilter_ = nullptr;
  filterFirstColumns_.clear();
  readsFilterColumnsFirst_ = false;
  emptySplit_ = false;
  fileColumnNames_.clear();
  baseReadOffset_ = 0;
  splitRowCount_ = 0;
  readAllFileColumns_ = false;
  noColumnsToRead_ = false;
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

namespace {

// Names the split's file for cuDF, with its size when the split carries one,
// so that a remote open skips the request for the size.
cudf::io::source_info makeSourceInfo(const CudfHiveConnectorSplit& split) {
  cudf::io::filepath_source source{.path = split.filePath};
  if (const auto fileSize = split.fileSize()) {
    source.size = static_cast<std::size_t>(*fileSize);
  }
  return cudf::io::source_info{
      std::vector<cudf::io::filepath_source>{std::move(source)}};
}

} // namespace

bool CudfSplitReader::ignoreMissingFiles() const {
  const HiveConfig hiveConfig(cudfHiveConfig_->config());
  return hiveConfig.ignoreMissingFiles(connectorQueryCtx_->sessionProperties());
}

void CudfSplitReader::markSplitEmpty() {
  emptySplit_ = true;
  dataSource_.reset();
  fileMetaDatas_.clear();
}

void CudfSplitReader::setupCudfDataSource() {
  if (dataSource_ or emptySplit_) {
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

  const bool ignoreMissingFiles = this->ignoreMissingFiles();

  // Helper to create a KvikIO data source.
  auto makeKvikioDataSource = [&]() {
    try {
      dataSource_ = std::move(
          cudf::io::make_datasources(makeSourceInfo(*split_)).front());
    } catch (const std::system_error& e) {
      // KvikIO preserves errno for local opens only.
      // TODO: Ignore missing remote files once KvikIO reports them
      // distinctly.
      if (ignoreMissingFiles and
          e.code() == std::errc::no_such_file_or_directory) {
        markSplitEmpty();
        return;
      }
      throw;
    }
  };

  // Use KvikIO data source if we don't want to use the BufferedInput source
  if (not useBufferedInput) {
    VLOG(1) << fmt::format(
        "Using KvikIO data source for file: {}", split_->filePath);
    makeKvikioDataSource();
    return;
  }

  auto fileHandleCachePtr = FileHandleCachedPtr{};
  try {
    const auto fileHandleKey = FileHandleKey{
        .filename = split_->filePath,
        .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
    auto fileProperties = split_->properties.value_or(FileProperties{});
    // The size can also come from the $file_size info column.
    fileProperties.fileSize = split_->fileSize();
    fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
    VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());
  } catch (const VeloxRuntimeError& e) {
    if (ignoreMissingFiles) {
      if (e.errorCode() == error_code::kFileNotFound) {
        markSplitEmpty();
        return;
      }
      // Rethrow actual filesystem errors only. Unsupported paths may fall back
      // to KvikIO.
      if (filesystems::isPathSupportedByRegisteredFileSystems(
              split_->filePath)) {
        throw;
      }
    }
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
    makeKvikioDataSource();
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
      velox::connector::BufferedInputBuilder::getInstance()->create(
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
    makeKvikioDataSource();
    return;
  }
  dataSource_ = std::make_unique<BufferedInputDataSource>(
      std::move(bufferedInput),
      cudfHiveConfig_->bufferedInputPinnedPieceBytes());
}

void CudfSplitReader::setupReaderOptions() {
  VELOX_CHECK_NOT_NULL(
      dataSource_,
      "CudfSplitReader does not have a datasource. Call setupCudfDataSource() first");

  const auto* sessionProperties = connectorQueryCtx_->sessionProperties();

  // Reader options
  readerOptions_ =
      cudf::io::parquet_reader_options::builder()
          .use_pandas_metadata(
              cudfHiveConfig_->isUsePandasMetadataSession(sessionProperties))
          .use_arrow_schema(
              cudfHiveConfig_->isUseArrowSchemaSession(sessionProperties))
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemasSession(
                  sessionProperties))
          .timestamp_type(
              cudfHiveConfig_->timestampTypeSession(sessionProperties))
          .use_tile_decode(
              cudfHiveConfig_->isUseTileDecodeSession(sessionProperties))
          // cuDF currently only folds ASCII letters, unlike the CPU reader's
          // UTF-8 folding, so non-ASCII names (rare case) that differ in case
          // do not match.
          .case_sensitive_names(not caseInsensitiveColumnNames_)
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

  // Skip column selection when reading every file column in file order
  if (not readAllFileColumns_) {
    readerOptions_.set_column_names(readColumnNames_);
  }

  if (prependRowIndex_) {
    readerOptions_.enable_prepend_row_index_column(true);
  }

  // A filter-first read needs a column to read after the filter columns, and
  // filter columns that the file has and the reader returns as read columns.
  const auto isFilterColumn = [&](const std::string& name) {
    return std::ranges::find(filterFirstColumns_, name) !=
        filterFirstColumns_.end();
  };
  readsFilterColumnsFirst_ = filterFirstFilter_ != nullptr and
      supportsFilterFirst() and not prependRowIndex_ and
      std::ranges::all_of(filterFirstColumns_,
                          [&](const auto& name) {
                            return fileColumnNames_.contains(name) and
                                std::ranges::find(readColumnNames_, name) !=
                                readColumnNames_.end();
                          }) and
      not std::ranges::all_of(readColumnNames_, isFilterColumn);
  if (readsFilterColumnsFirst_) {
    filterFirstOptions_ = readerOptions_;
    filterFirstOptions_.set_filter(*filterFirstFilter_);
  }
}

rmm::device_async_resource_ref CudfSplitReader::determineCudfMemoryResource()
    const {
  return get_output_mr();
}

const std::vector<cudf::io::parquet::FileMetaData>&
CudfSplitReader::fileMetaDatas() {
  if (not fileMetaDatas_.empty()) {
    return fileMetaDatas_;
  }

  // Setup the datasource
  setupCudfDataSource();
  if (emptySplit_) {
    return fileMetaDatas_;
  }

  // Check that the datasource is set up
  VELOX_CHECK_NOT_NULL(
      dataSource_,
      "CudfSplitReader does not have a datasource. Call setupCudfDataSource() first");

  // Wrap the existing datasource without transferring ownership.
  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(cudf::io::datasource::create(dataSource_.get()));
  try {
    fileMetaDatas_ = cudf::io::read_parquet_footers(sources);
  } catch (const VeloxRuntimeError& e) {
    // An open that knows the file size sends no request to the storage, so a
    // missing file first shows at this read.
    if (ignoreMissingFiles() and e.errorCode() == error_code::kFileNotFound) {
      markSplitEmpty();
      return fileMetaDatas_;
    }
    throw;
  }
  VELOX_CHECK_GE(
      fileMetaDatas_.size(),
      1,
      "CudfSplitReader failed to read any parquet metadatas");

  filterDynamicRowGroups();
  return fileMetaDatas_;
}

namespace {

// Whether Filter::testInt64Range() of a filter of 'kind' decides if a range of
// integers can hold a passing value.
bool testsIntegerRange(common::FilterKind kind) {
  switch (kind) {
    case common::FilterKind::kAlwaysFalse:
    case common::FilterKind::kAlwaysTrue:
    case common::FilterKind::kIsNull:
    case common::FilterKind::kIsNotNull:
    case common::FilterKind::kBigintRange:
    case common::FilterKind::kNegatedBigintRange:
    case common::FilterKind::kBigintValuesUsingHashTable:
    case common::FilterKind::kBigintValuesUsingBitmask:
    case common::FilterKind::kNegatedBigintValuesUsingHashTable:
    case common::FilterKind::kNegatedBigintValuesUsingBitmask:
    case common::FilterKind::kBigintMultiRange:
    case common::FilterKind::kBigintValuesUsingBloomFilter:
      return true;
    default:
      return false;
  }
}

} // namespace

std::optional<size_t> CudfSplitReader::findIntegerColumn(
    const std::string& name,
    cudf::data_type type) const {
  using ParquetType = cudf::io::parquet::Type;
  const auto& schema = fileMetaDatas_.front().schema;
  const auto& columns = schema.front().children_idx;
  const auto fileName = [&](const std::string& name) {
    return caseInsensitiveColumnNames_
        ? ::facebook::velox::functions::stringImpl::utf8StrToLowerCopy(name)
        : name;
  };
  // The column the reader decodes for 'name': the first one of that name.
  const auto requestedName = fileName(name);
  const auto column =
      std::find_if(columns.begin(), columns.end(), [&](auto index) {
        return fileName(schema[index].name) == requestedName;
      });
  if (column == columns.end()) {
    return std::nullopt;
  }
  const auto& field = schema[*column];
  if (field.max_repetition_level != 0 || !field.children_idx.empty()) {
    return std::nullopt;
  }
  const auto stored = [&] {
    if (type.id() == cudf::type_id::TIMESTAMP_DAYS) {
      return field.type == ParquetType::INT32 &&
          ((field.logical_type.has_value() &&
            field.logical_type->type == cudf::io::parquet::LogicalType::DATE) ||
           field.converted_type == cudf::io::parquet::ConvertedType::DATE);
    }
    if (!cudf::is_integral(type) || !cudf::is_signed(type)) {
      return false;
    }
    const auto width = cudf::size_of(type) * 8;
    if (field.logical_type.has_value()) {
      return field.logical_type->type ==
          cudf::io::parquet::LogicalType::INTEGER &&
          field.logical_type->is_signed() &&
          field.logical_type->bit_width() == width;
    }
    if (field.converted_type.has_value()) {
      return false;
    }
    return (field.type == ParquetType::INT32 && width == 32) ||
        (field.type == ParquetType::INT64 && width == 64);
  }();
  return stored ? std::optional<size_t>(*column) : std::nullopt;
}

bool CudfSplitReader::storesIntegersOfType(
    const std::string& name,
    cudf::data_type type) {
  return !fileMetaDatas().empty() && findIntegerColumn(name, type).has_value();
}

void CudfSplitReader::filterDynamicRowGroups() {
  if (readerDynamicFilters_.empty()) {
    return;
  }
  using ParquetType = cudf::io::parquet::Type;
  const auto& metadata = fileMetaDatas_.front();
  const auto& schema = metadata.schema;
  dynamicRowGroups_.assign(metadata.row_groups.size(), true);
  std::vector<bool> allRowsMatch(metadata.row_groups.size(), true);
  for (const auto& [name, type, filter] : readerDynamicFilters_) {
    if (filter->kind() == common::FilterKind::kAlwaysFalse) {
      dynamicRowGroups_.assign(dynamicRowGroups_.size(), false);
      continue;
    }
    // The statistics are comparable with the filter's 64-bit integers only if
    // the file stores the column as the signed integer of the table's type.
    const auto column = findIntegerColumn(name, type);
    const bool comparable =
        column.has_value() && testsIntegerRange(filter->kind());
    if (!comparable) {
      // Nothing is known about this filter: keep every row group and leave
      // the filter to the rows.
      allRowsMatch.assign(allRowsMatch.size(), false);
      continue;
    }
    const auto& columnName = schema[*column].name;
    for (size_t groupIndex = 0; groupIndex < metadata.row_groups.size();
         ++groupIndex) {
      if (!dynamicRowGroups_[groupIndex]) {
        continue;
      }
      const auto& group = metadata.row_groups[groupIndex];
      if (filter->kind() == common::FilterKind::kAlwaysFalse) {
        dynamicRowGroups_[groupIndex] = false;
        continue;
      }
      const auto chunk = std::find_if(
          group.columns.begin(), group.columns.end(), [&](const auto& c) {
            return c.meta_data.path_in_schema.size() == 1 &&
                c.meta_data.path_in_schema.front() == columnName;
          });
      if (chunk == group.columns.end()) {
        allRowsMatch[groupIndex] = false;
        continue;
      }
      const auto& stats = chunk->meta_data.statistics;
      if (stats.null_count.has_value() && *stats.null_count == group.num_rows) {
        dynamicRowGroups_[groupIndex] = filter->testNull();
        continue;
      }
      const auto& lower =
          stats.min_value.has_value() ? stats.min_value : stats.min;
      const auto& upper =
          stats.max_value.has_value() ? stats.max_value : stats.max;
      const auto width = chunk->meta_data.type == ParquetType::INT32 ? 4
          : chunk->meta_data.type == ParquetType::INT64              ? 8
                                                                     : 0;
      if (width == 0 || !lower || !upper || lower->size() != width ||
          upper->size() != width) {
        allRowsMatch[groupIndex] = false;
        continue;
      }
      const auto decode = [width](const auto& bytes) -> int64_t {
        return width == 4
            ? folly::Endian::little(folly::loadUnaligned<int32_t>(bytes.data()))
            : folly::Endian::little(
                  folly::loadUnaligned<int64_t>(bytes.data()));
      };
      const auto minimum = decode(*lower);
      const auto maximum = decode(*upper);
      if (minimum > maximum) {
        allRowsMatch[groupIndex] = false;
        continue;
      }
      const bool hasNulls = stats.null_count.value_or(1) != 0;
      dynamicRowGroups_[groupIndex] =
          filter->testInt64Range(minimum, maximum, hasNulls);
      if (filter->kind() == common::FilterKind::kBigintRange &&
          advisoryFilterBitmap(filter) == nullptr) {
        const auto& range = static_cast<const common::BigintRange&>(*filter);
        allRowsMatch[groupIndex] = allRowsMatch[groupIndex] &&
            range.lower() <= minimum && range.upper() >= maximum &&
            (!hasNulls || range.testNull());
      } else {
        allRowsMatch[groupIndex] = false;
      }
    }
  }
  dynamicFiltersSatisfied_ = true;
  for (size_t i = 0; i < dynamicRowGroups_.size(); ++i) {
    if (dynamicRowGroups_[i] && !allRowsMatch[i]) {
      dynamicFiltersSatisfied_ = false;
      break;
    }
  }
}

void CudfSplitReader::createCudfReader() {
  fileMetaDatas();

  // Check if we are reading all columns in order
  const auto& schema = fileMetaDatas_.front().schema;
  readAllFileColumns_ = std::ranges::equal(
      schema.front().children_idx, readColumnNames_, {}, [&](auto childIdx) {
        const auto& name = schema[childIdx].name;
        return caseInsensitiveColumnNames_
            ? ::facebook::velox::functions::stringImpl::utf8StrToLowerCopy(name)
            : name;
      });

  // Setup reader options
  setupReaderOptions();

  // A reader spanning multiple sources also needs the byte range source index
  // map to fetch column chunks from the matching data source.
  VELOX_CHECK_EQ(
      fileMetaDatas_.size(),
      1,
      "cuDF parquet reader supports a single parquet metadata per split");

  const auto* sessionProperties = connectorQueryCtx_->sessionProperties();
  chunkReadLimit_ =
      cudfHiveConfig_->maxChunkReadLimitSession(sessionProperties);
  passReadLimit_ = cudfHiveConfig_->maxPassReadLimitSession(sessionProperties);
  serializeIoRequests_ =
      cudfHiveConfig_->serializeIoRequestsSession(sessionProperties);

  // Create a hybrid scan reader over all sources of the split
  splitReader_ = std::make_unique<CudfParquetReader>(
      std::move(fileMetaDatas_), readerOptions_);

  setupPageIndexes();

  passState_ = std::make_unique<RowGroupPassState>();
  passState_->passes = selectRowGroupPasses();

  // The rows of a pass read filter columns first are the rows of one column.
  // The row groups the filter of such a read pruned above stay pruned: every
  // row of them fails a filter the caller applies to the rows otherwise.
  readsFilterColumnsFirst_ =
      readsFilterColumnsFirst_ and
      std::ranges::all_of(passState_->passes, [&](const auto& pass) {
        return splitReader_->total_rows_in_row_groups(pass) <=
            static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
      });
  setColumnChunkByteRanges();
}

void CudfSplitReader::setupPageIndexes() {
  const auto pageIndexByteRanges = splitReader_->page_index_byte_ranges();

  // No page index bytes
  if (std::ranges::all_of(pageIndexByteRanges, [](const auto& byteRange) {
        return byteRange.is_empty();
      })) {
    return;
  }

  [[maybe_unused]] auto [pageIndexBuffers, pageIndexData] =
      fetchPageIndexes(dataSource_, pageIndexByteRanges);
  splitReader_->setup_page_indexes(pageIndexData);
}

CudfSplitReader::RowGroupPasses CudfSplitReader::selectRowGroupPasses() {
  auto rowGroupIndices = splitReader_->all_row_groups(readerOptions_);

  // Filter row groups using row group byte ranges
  if (readerOptions_.get_skip_bytes() > 0 or
      readerOptions_.get_num_bytes().has_value()) {
    rowGroupIndices = splitReader_->filter_row_groups_with_byte_range(
        rowGroupIndices, readerOptions_);
  }

  // Filter row groups using column chunk statistics. The filter of a
  // filter-first read rejects every row the pushdown filter rejects.
  const auto& filterOptions =
      readsFilterColumnsFirst_ ? filterFirstOptions_ : readerOptions_;
  if (filterOptions.get_filter().has_value()) {
    rowGroupIndices = splitReader_->filter_row_groups_with_stats(
        rowGroupIndices, filterOptions, stream_);
  }

  if (!dynamicRowGroups_.empty()) {
    for (auto& groups : rowGroupIndices) {
      const auto numGroups = groups.size();
      std::erase_if(
          groups, [&](auto group) { return !dynamicRowGroups_[group]; });
      numDynamicFilterSkippedRowGroups_ += numGroups - groups.size();
    }
  }

  const auto numRowGroups = std::accumulate(
      rowGroupIndices.begin(),
      rowGroupIndices.end(),
      std::size_t{0},
      [](auto sum, const auto& sourceRowGroups) {
        return sum + sourceRowGroups.size();
      });
  numSelectedRowGroups_ += numRowGroups;
  auto selectedRowGroups = numRowGroups;
  common::testutil::TestValue::adjust(
      "facebook::velox::cudf_velox::connector::hive::CudfSplitReader::selectedRowGroups",
      &selectedRowGroups);

  // No row groups to read.
  if (numRowGroups == 0) {
    return {};
  }

  // Construct row group passes using all (or selected) columns for pass memory
  // estimation.
  return splitReader_->construct_row_group_passes(
      cudf::io::parquet::experimental::read_columns_mode::ALL_COLUMNS,
      rowGroupIndices,
      passReadLimit_,
      readerOptions_);
}

void CudfSplitReader::totalScanTimeCalculator(void* userData) {
  TotalScanTimeCallbackData* data =
      static_cast<TotalScanTimeCallbackData*>(userData);

  // Record end time in callback
  auto endTimeUs = SteadyTime::now<std::chrono::microseconds>();

  // Calculate elapsed time in microseconds and convert to nanoseconds
  auto elapsedUs = endTimeUs - data->startTimeUs;
  auto elapsedNs = elapsedUs * 1000; // Convert microseconds to nanoseconds

  // Update totalScanTime
  data->ioStatistics->incTotalScanTimeNs(elapsedNs);

  delete data;
}

} // namespace facebook::velox::cudf_velox::connector::hive
