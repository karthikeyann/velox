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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/ExprOptimizer.h"

#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>

#include <gflags/gflags.h>

#include <algorithm>
#include <numeric>

DEFINE_bool(
    cudf_scan_prune_before_filter,
    false,
    "Do not gather filter-only columns after evaluating the final scan predicate");
DEFINE_bool(
    cudf_scan_async_output,
    false,
    "Return stream-owning GPU scan vectors without a redundant final host synchronization");
DEFINE_bool(
    cudf_scan_borrow_gpu_cache,
    false,
    "Read raw GPU cache views directly when a scan filter materializes owned output");
DEFINE_bool(
    cudf_scan_borrow_gpu_cache_unfiltered,
    false,
    "Extend borrowed GPU cache views to unfiltered scan outputs with shared read-only ownership");
DEFINE_bool(
    cudf_scan_jit_subfield_filters,
    false,
    "Use cuDF JIT for post-read subfield and dynamic predicates");
DEFINE_uint64(
    cudf_scan_jit_subfield_min_rows,
    4096,
    "Minimum input rows for JIT post-read scan predicates");

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

namespace {

std::string rowGroupSelectionFilterKey(const common::SubfieldFilters& filters) {
  std::vector<std::string> entries;
  entries.reserve(filters.size());
  for (const auto& [field, filter] : filters) {
    VELOX_CHECK_NOT_NULL(filter);
    entries.push_back(
        fmt::format("{}={}", field.toString(), filter->toString()));
  }
  std::sort(entries.begin(), entries.end());
  std::string result;
  for (const auto& entry : entries) {
    result += fmt::format("{}:{}", entry.size(), entry);
  }
  return result;
}

} // namespace

CudfHiveDataSource::CudfHiveDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    facebook::velox::FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241}, // CudfHive blue,
          std::nullopt,
          fmt::format(
              "[{}:{}]",
              tableHandle->name(),
              connectorQueryCtx->planNodeId())),
      cudfHiveConfig_(cudfHiveConfig),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      outputType_(outputType),
      pool_(connectorQueryCtx->memoryPool()),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  // Set up column projection if needed
  auto readColumnTypes = outputType_->children();
  for (const auto& outputName : outputType_->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const hive::HiveColumnHandle*>(it->second.get());
    readColumnSet_.emplace(handle->name());
    readColumnNames_.emplace_back(handle->name());
  }

  tableHandle_ =
      std::dynamic_pointer_cast<const hive::HiveTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of HiveTableHandle");

  // Copy subfield filters.
  for (const auto& [k, v] : tableHandle_->subfieldFilters()) {
    subfieldFilters_.emplace(k.clone(), v->clone());
  }

  // Extract additional simple filters from remainingFilter (same as CPU path).
  // This extracts single-column filters like "col = 'X'" or "col <> 'Y'" from
  // complex expressions and adds them to subfieldFilters_ for pushdown.
  double sampleRate = tableHandle_->sampleRate();
  auto remainingFilter =
      facebook::velox::connector::hive::extractFiltersFromRemainingFilter(
          tableHandle_->remainingFilter(),
          expressionEvaluator_,
          subfieldFilters_,
          sampleRate);

  // Add fields in the filter to the columns to read if not there
  for (const auto& [field, _] : subfieldFilters_) {
    if (readColumnSet_.count(field.toString()) == 0) {
      readColumnSet_.emplace(field.toString());
      readColumnNames_.emplace_back(field.toString());
    }
  }
  // Optimize (rewrites + constant folding) the remaining filter before
  // evaluator selection so CudfFunctions never see scalar-only operand sets.
  // TODO: ConnectorQueryCtx does not expose the session QueryCtx, only an
  // ExpressionEvaluator, so constant folding here runs against a transient
  // QueryCtx with default query config rather than the session's. Passing the
  // real session QueryCtx (e.g. by exposing it on ConnectorQueryCtx) should be
  // figured out later. A local QueryCtx is required because
  // expression::optimize constant-folds through exec::ExprSet, whose
  // constructor dereferences the QueryCtx unconditionally; a null QueryCtx
  // would crash.
  auto optimizeQueryCtx = core::QueryCtx::create();
  optimizedRemainingFilter_ = remainingFilter
      ? expression::optimize(remainingFilter, optimizeQueryCtx.get(), pool_)
      : nullptr;
  if (optimizedRemainingFilter_) {
    // Add fields referenced by the filter to the columns to read. Collect from
    // the optimized expression since folding may drop branches and the columns
    // they reference. Read-column order does not affect results: the data
    // source projects its output to the requested output type.
    for (const auto& name : referencedInputFields(optimizedRemainingFilter_)) {
      if (readColumnSet_.count(name) == 0) {
        readColumnSet_.emplace(name);
        readColumnNames_.emplace_back(name);
      }
    }

    // TODO: Prune struct columns to the subfields referenced by the remaining
    // filter; currently the whole column is read even if only one field is
    // used.

    // The filter is already optimized and constant folded above, so compile it
    // directly.
    auto const remainingFilterType = getTableRowType();
    cudfRemainingFilterExpression_ = createCudfExpression(
        optimizedRemainingFilter_, remainingFilterType, pool_);
  }

  // Build a combined AST for all subfield filters once. This is query-constant
  // and doesn't depend on split-specific state.
  if (!subfieldFilters_.empty()) {
    auto const readerFilterType = getTableRowType();
    subfieldFilterExpr_ = &createAstFromSubfieldFilters(
        subfieldFilters_, subfieldTree_, subfieldScalars_, readerFilterType);
  }
  rowGroupSelectionFilterKey_ = rowGroupSelectionFilterKey(subfieldFilters_);

  VELOX_CHECK_NOT_NULL(fileHandleFactory_, "No FileHandleFactory present");

  // Create empty IOStats and FsStats for later use
  ioStatistics_ = std::make_shared<io::IoStatistics>();
  ioStats_ = std::make_shared<facebook::velox::IoStats>();

  // Whether to use the experimental cuDF reader
  useExperimentalCudfReader_ =
      cudfHiveConfig_->useExperimentalCudfReaderSession(
          connectorQueryCtx_->sessionProperties());
}

std::unique_ptr<CudfSplitReader> CudfHiveDataSource::createCudfSplitReader() {
  return std::make_unique<CudfSplitReader>(
      split_,
      tableHandle_,
      outputType_,
      readColumnNames_,
      fileHandleFactory_,
      executor_,
      connectorQueryCtx_,
      cudfHiveConfig_,
      ioStatistics_,
      ioStats_,
      useExperimentalCudfReader_,
      subfieldFilterExpr_,
      rowGroupSelectionFilterKey_);
}

void CudfHiveDataSource::convertSplit(std::shared_ptr<ConnectorSplit> split) {
  // Dynamic cast split to `CudfHiveConnectorSplit`
  if (std::dynamic_pointer_cast<CudfHiveConnectorSplit>(split)) {
    split_ = std::dynamic_pointer_cast<CudfHiveConnectorSplit>(split);
    return;
  }

  // Convert `HiveConnectorSplit` to `CudfHiveConnectorSplit`
  auto hiveSplit = checkedPointerCast<hive::HiveConnectorSplit>(split);

  VELOX_CHECK_EQ(
      hiveSplit->fileFormat,
      dwio::common::FileFormat::PARQUET,
      "Unsupported file format for conversion from HiveConnectorSplit to CudfHiveConnectorSplit");

  // Remove "file:" prefix from the file path if present
  std::string cleanedPath = hiveSplit->filePath;
  constexpr std::string_view kFilePrefix = "file:";
  constexpr std::string_view kS3APrefix = "s3a:";
  if (cleanedPath.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
    cleanedPath = cleanedPath.substr(kFilePrefix.size());
  } else if (cleanedPath.compare(0, kS3APrefix.size(), kS3APrefix) == 0) {
    // KvikIO does not support "s3a:" prefix. We need to translate it to "s3:".
    cleanedPath.erase(kS3APrefix.size() - 2, 1);
  }

  auto cudfHiveSplitBuilder = CudfHiveConnectorSplitBuilder(cleanedPath)
                                  .start(hiveSplit->start)
                                  .length(hiveSplit->length)
                                  .connectorId(hiveSplit->connectorId)
                                  .splitWeight(hiveSplit->splitWeight);
  for (auto const& infoColumn : hiveSplit->infoColumns) {
    cudfHiveSplitBuilder.infoColumn(infoColumn.first, infoColumn.second);
  }
  split_ = cudfHiveSplitBuilder.build();

  VLOG(1) << "Adding split " << split_->toString();
}

void CudfHiveDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  // Virtual method for class-specific conversion of the split
  convertSplit(split);

  if (cudfSplitReader_) {
    decodedColumnCacheHits_ += cudfSplitReader_->decodedColumnCacheHits();
    decodedColumnCacheMisses_ += cudfSplitReader_->decodedColumnCacheMisses();
    decodedColumnCacheDecodeCalls_ +=
        cudfSplitReader_->decodedColumnCacheDecodeCalls();
    decodedColumnGpuCacheHits_ += cudfSplitReader_->decodedColumnGpuCacheHits();
  }
  cudfSplitReader_ = createCudfSplitReader();
  cudfSplitReader_->prepareSplit(runtimeStats_);

  // A complete decoded-column cache hit needs neither the Parquet data nor its
  // footer. Avoid reopening the file solely for this approximate completed-byte
  // statistic.
  if (cudfSplitReader_->isFullyDecodedColumnCacheHit()) {
    return;
  }

  // TODO: `completedBytes_` should be updated in `next()` as we read more and
  // more table bytes
  try {
    const auto fileHandleKey = FileHandleKey{
        .filename = split_->filePath,
        .tokenProvider = connectorQueryCtx_->fsTokenProvider()};
    auto fileProperties = FileProperties{};
    auto const fileHandleCachePtr = fileHandleFactory_->generate(
        fileHandleKey, &fileProperties, ioStats_ ? ioStats_.get() : nullptr);
    if (fileHandleCachePtr.get() and fileHandleCachePtr.get()->file) {
      completedBytes_ += fileHandleCachePtr->file->size();
    }
  } catch (const std::exception& e) {
    // Unable to get the file size, log a warning and continue
    LOG(WARNING) << "Failed to get file size for " << split_->filePath << ": "
                 << e.what();
  }
}

void CudfHiveDataSource::addDynamicFilter(
    column_index_t outputChannel,
    const std::shared_ptr<common::Filter>& filter) {
  VELOX_CHECK_LT(outputChannel, outputType_->size());
  VELOX_CHECK_NOT_NULL(filter);
  const common::Subfield field(readColumnNames_.at(outputChannel));
  auto found = dynamicFilters_.find(field);
  if (found == dynamicFilters_.end()) {
    dynamicFilters_.emplace(field.clone(), filter->clone());
  } else {
    found->second = found->second->mergeWith(filter.get());
  }
  dynamicFilterExpr_ = &createAstFromSubfieldFilters(
      dynamicFilters_,
      dynamicFilterTree_,
      dynamicFilterScalars_,
      getTableRowType());
}

std::optional<RowVectorPtr> CudfHiveDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /* future */) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  VELOX_CHECK_NOT_NULL(split_, "No split present. Call addSplit() first.");
  VELOX_CHECK_NOT_NULL(cudfSplitReader_, "No split to process.");
  auto stream = cudfSplitReader_->stream();
  const bool hasFilter =
      cudfSplitReader_->shouldApplySubfieldFilterAfterRead() ||
      optimizedRemainingFilter_ != nullptr || dynamicFilterExpr_ != nullptr;
  auto borrowed = FLAGS_cudf_scan_borrow_gpu_cache &&
          (hasFilter ||
           (FLAGS_cudf_scan_borrow_gpu_cache_unfiltered &&
            outputType_->size() > 0))
      ? cudfSplitReader_->tryNextBorrowedGpuColumns()
      : nullptr;
  std::unique_ptr<cudf::table> cudfTable;
  if (!borrowed) {
    auto chunkOpt = cudfSplitReader_->next(size);
    if (!chunkOpt.has_value()) {
      return nullptr;
    }
    cudfTable = std::move(chunkOpt.value());
  } else {
    borrowedGpuCacheBatches_.fetch_add(1, std::memory_order_relaxed);
  }
  auto inputView =
      borrowed ? cudf::table_view(borrowed->views) : cudfTable->view();

  if (borrowed && !hasFilter) {
    std::vector<cudf::size_type> channels(outputType_->size());
    std::iota(channels.begin(), channels.end(), 0);
    auto view = inputView.select(channels);
    const auto rows = view.num_rows();
    const bool gpuOutput = cudfIsRegistered();
    RowVectorPtr output;
    if (gpuOutput) {
      const auto bytes = borrowed->retainedBytes;
      auto orderRelease = borrowed->orderRelease;
      std::shared_ptr<const void> owner(std::move(borrowed));
      output = std::make_shared<CudfVector>(
          pool_,
          outputType_,
          rows,
          view,
          std::move(owner),
          bytes,
          stream,
          get_output_mr(),
          std::move(orderRelease));
    } else {
      output = with_arrow::toVeloxColumn(
          view, pool_, outputType_, stream, get_temp_mr());
    }
    if (!FLAGS_cudf_scan_async_output || !gpuOutput) {
      stream.synchronize();
    } else {
      asynchronousScanOutputs_.fetch_add(1, std::memory_order_relaxed);
    }
    borrowedUnfilteredGpuCacheBatches_.fetch_add(1, std::memory_order_relaxed);
    completedRows_ += output->size();
    return output;
  }

  auto applyFilter = [&](cudf::table_view table,
                         cudf::column_view predicate,
                         bool finalFilter) {
    if (FLAGS_cudf_scan_prune_before_filter && finalFilter &&
        outputType_->size() > 0 && outputType_->size() < table.num_columns()) {
      // Read columns are ordered as output columns followed by filter-only
      // columns. Evaluate the predicate on all columns before narrowing the
      // gather; earlier filters must retain inputs needed by later filters.
      std::vector<cudf::size_type> channels(outputType_->size());
      std::iota(channels.begin(), channels.end(), 0);
      prunedFilterColumns_.fetch_add(
          table.num_columns() - channels.size(), std::memory_order_relaxed);
      return cudf::apply_boolean_mask(
          table.select(channels), predicate, stream, get_output_mr());
    }
    return cudf::apply_boolean_mask(table, predicate, stream, get_output_mr());
  };

  auto evaluateScanPredicate = [&](const cudf::ast::expression& expression) {
    if (FLAGS_cudf_scan_jit_subfield_filters &&
        inputView.num_rows() >= FLAGS_cudf_scan_jit_subfield_min_rows) {
      jitSubfieldFilterBatches_.fetch_add(1, std::memory_order_relaxed);
      return cudf::compute_column_jit(
          inputView, expression, stream, get_temp_mr());
    }
    return cudf::compute_column(inputView, expression, stream, get_temp_mr());
  };

  if (cudfSplitReader_->shouldApplySubfieldFilterAfterRead()) {
    VELOX_CHECK_NOT_NULL(subfieldFilterExpr_);
    auto predicate = evaluateScanPredicate(*subfieldFilterExpr_);
    cudfTable = applyFilter(
        inputView,
        predicate->view(),
        !optimizedRemainingFilter_ && !dynamicFilterExpr_);
    inputView = cudfTable->view();
  }

  uint64_t filterTimeUs{0};
  if (optimizedRemainingFilter_) {
    MicrosecondWallTimer filterTimer(&filterTimeUs);
    std::vector<cudf::column_view> inputViews(
        inputView.begin(), inputView.end());
    auto filterResult =
        cudfRemainingFilterExpression_->eval(inputViews, stream, get_temp_mr());
    cudfTable =
        applyFilter(inputView, asView(filterResult), !dynamicFilterExpr_);
    inputView = cudfTable->view();
  }
  totalRemainingFilterTime_.fetch_add(
      filterTimeUs * 1000, std::memory_order_relaxed);

  if (dynamicFilterExpr_) {
    auto predicate = evaluateScanPredicate(*dynamicFilterExpr_);
    cudfTable = applyFilter(inputView, predicate->view(), true);
  }

  VELOX_CHECK_NOT_NULL(
      cudfTable, "Borrowed cache input requires an owning filter output");
  const auto nRows = cudfTable->num_rows();

  if (outputType_->size() < cudfTable->num_columns()) {
    auto cudfTableColumns = cudfTable->release();
    std::vector<std::unique_ptr<cudf::column>> outputColumns;
    outputColumns.reserve(outputType_->size());
    std::move(
        cudfTableColumns.begin(),
        cudfTableColumns.begin() + outputType_->size(),
        std::back_inserter(outputColumns));
    cudfTable = std::make_unique<cudf::table>(std::move(outputColumns));
  }

  // TODO (dm): Should we only enable table scan if cudf is registered?
  // Earlier we could enable cudf table scans without using other cudf operators
  // We still can, but I'm wondering if this is the right thing to do
  const bool gpuOutput = cudfIsRegistered();
  auto output = gpuOutput
      ? std::make_shared<CudfVector>(
            pool_, outputType_, nRows, std::move(cudfTable), stream)
      : with_arrow::toVeloxColumn(
            cudfTable->view(), pool_, outputType_, stream, get_temp_mr());
  if (!FLAGS_cudf_scan_async_output || !gpuOutput) {
    stream.synchronize();
  } else {
    // CudfVector retains the producing stream; downstream GPU operators join
    // streams before consuming it. CPU/Arrow conversion keeps the old fence.
    asynchronousScanOutputs_.fetch_add(1, std::memory_order_relaxed);
  }

  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");

  completedRows_ += output->size();

  // TODO: Update `completedBytes_` here instead of in `addSplit()`

  return output;
}

std::unordered_map<std::string, RuntimeMetric>
CudfHiveDataSource::getRuntimeStats() {
  auto result = runtimeStats_.toRuntimeMetricMap();
  const auto decodedColumnCacheHits = decodedColumnCacheHits_ +
      (cudfSplitReader_ ? cudfSplitReader_->decodedColumnCacheHits() : 0);
  const auto decodedColumnCacheMisses = decodedColumnCacheMisses_ +
      (cudfSplitReader_ ? cudfSplitReader_->decodedColumnCacheMisses() : 0);
  const auto decodedColumnCacheDecodeCalls = decodedColumnCacheDecodeCalls_ +
      (cudfSplitReader_ ? cudfSplitReader_->decodedColumnCacheDecodeCalls()
                        : 0);
  const auto decodedColumnGpuCacheHits = decodedColumnGpuCacheHits_ +
      (cudfSplitReader_ ? cudfSplitReader_->decodedColumnGpuCacheHits() : 0);
  result.insert({
      {"prunedFilterColumns",
       RuntimeMetric(prunedFilterColumns_.load(std::memory_order_relaxed))},
      {"asynchronousScanOutputs",
       RuntimeMetric(asynchronousScanOutputs_.load(std::memory_order_relaxed))},
      {"borrowedGpuCacheBatches",
       RuntimeMetric(borrowedGpuCacheBatches_.load(std::memory_order_relaxed))},
      {"jitSubfieldFilterBatches",
       RuntimeMetric(
           jitSubfieldFilterBatches_.load(std::memory_order_relaxed))},
      {"borrowedUnfilteredGpuCacheBatches",
       RuntimeMetric(
           borrowedUnfilteredGpuCacheBatches_.load(std::memory_order_relaxed))},
      {std::string(connector::hive::HiveDataSource::kTotalScanTime),
       RuntimeMetric(
           ioStatistics_->totalScanTimeNs(), RuntimeCounter::Unit::kNanos)},
      {std::string(Connector::kTotalRemainingFilterTime),
       RuntimeMetric(
           totalRemainingFilterTime_.load(std::memory_order_relaxed),
           RuntimeCounter::Unit::kNanos)},
  });
  if (decodedColumnCacheHits > 0 or decodedColumnCacheMisses > 0) {
    result.emplace(
        std::string(kDecodedColumnCacheHits),
        RuntimeMetric(decodedColumnCacheHits));
    result.emplace(
        std::string(kDecodedColumnCacheMisses),
        RuntimeMetric(decodedColumnCacheMisses));
    result.emplace(
        std::string(kDecodedColumnCacheDecodeCalls),
        RuntimeMetric(decodedColumnCacheDecodeCalls));
    result.emplace(
        std::string(kDecodedColumnGpuCacheHits),
        RuntimeMetric(decodedColumnGpuCacheHits));
  }
  const auto& ioStats = ioStats_->stats();
  for (const auto& storageStats : ioStats) {
    result.emplace(storageStats.first, storageStats.second);
  }
  return result;
}

const RowTypePtr CudfHiveDataSource::getTableRowType() {
  if (cachedTableRowType_) {
    return cachedTableRowType_;
  }
  if (tableHandle_->dataColumns()) {
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    for (const auto& name : readColumnNames_) {
      auto parsedType = tableHandle_->dataColumns()->findChild(name);
      names.emplace_back(std::move(name));
      types.push_back(parsedType);
    }
    cachedTableRowType_ = ROW(std::move(names), std::move(types));
    return cachedTableRowType_;
  }
  cachedTableRowType_ = outputType_;
  return cachedTableRowType_;
}

} // namespace facebook::velox::cudf_velox::connector::hive
