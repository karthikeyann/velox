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
#include "velox/experimental/cudf/exec/CudfTopN.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/TopKSortKeys.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/merge.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>

#include <gflags/gflags.h>

DEFINE_bool(
    cudf_topn_select_candidates,
    false,
    "Select numeric TopN candidates before the final stable sort");
DEFINE_int64(
    cudf_topn_select_min_rows,
    100'000,
    "Minimum input rows for TopN candidate selection");

namespace facebook::velox::cudf_velox {
CudfTopN::CudfTopN(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNNode>& topNNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          topNNode->outputType(),
          topNNode->id(),
          "CudfTopN",
          nvtx3::rgb{175, 238, 238}, // Pale Turquoise
          NvtxMethodFlag::kAll,
          std::nullopt,
          topNNode),
      count_(topNNode->count()),
      topNNode_(topNNode),
      cudaEvent_(std::make_unique<CudaEvent>(cudaEventDisableTiming)) {
  kBatchSize_ = driverCtx->queryConfig().get<int32_t>(
      CudfConfig::kCudfTopNBatchSize, kBatchSize_);
  const auto numColumns{outputType_->children().size()};
  const auto numSortingKeys{topNNode->sortingKeys().size()};
  std::vector<bool> isSortingKey(numColumns);
  sortKeys_.reserve(numSortingKeys);
  columnOrder_.reserve(numSortingKeys);
  nullOrder_.reserve(numSortingKeys);

  for (int i = 0; i < numSortingKeys; ++i) {
    const auto channel =
        exec::exprToChannel(topNNode->sortingKeys()[i].get(), outputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "TopN doesn't allow constant sorting keys");
    sortKeys_.push_back(channel);
    isSortingKey[channel] = true;
    auto const& sortingOrder = topNNode->sortingOrders()[i];
    columnOrder_.push_back(
        sortingOrder.isAscending() ? cudf::order::ASCENDING
                                   : cudf::order::DESCENDING);
    nullOrder_.push_back(
        (sortingOrder.isNullsFirst() ^ !sortingOrder.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

CudfVectorPtr CudfTopN::mergeTopK(
    std::vector<CudfVectorPtr> topNBatches,
    int32_t k,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<cudf::table_view> tableViews;
  std::vector<rmm::cuda_stream_view> inputStreams;
  tableViews.reserve(topNBatches.size());
  inputStreams.reserve(topNBatches.size());
  for (const auto& batch : topNBatches) {
    if (!batch) {
      continue;
    }
    tableViews.push_back(batch->getTableView());
    inputStreams.push_back(batch->stream());
  }
  cudf::detail::join_streams(inputStreams, stream);
  auto mergedTable =
      cudf::merge(tableViews, sortKeys_, columnOrder_, nullOrder_, stream, mr);
  // Ensure input-stream deallocations don't race with merge stream.
  streamsWaitForStream(*cudaEvent_, inputStreams, stream);
  // slice it
  auto topk =
      cudf::split(
          mergedTable->view(), {std::min(k, mergedTable->num_rows())}, stream)
          .front();
  auto const size = topk.num_rows();
  return std::make_shared<CudfVector>(
      topNBatches[0]->pool(),
      outputType_,
      size,
      std::make_unique<cudf::table>(topk, stream, mr),
      stream);
}

std::unique_ptr<cudf::table> CudfTopN::getTopK(
    cudf::table_view const& values,
    int32_t k,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto keys = values.select(sortKeys_);
  auto primary = keys.column(0);
  if (FLAGS_cudf_topn_select_candidates &&
      values.num_rows() >= FLAGS_cudf_topn_select_min_rows &&
      int64_t{k} * 32 < values.num_rows() && primary.null_count() == 0 &&
      (primary.type().id() == cudf::type_id::INT32 ||
       primary.type().id() == cudf::type_id::INT64 ||
       primary.type().id() == cudf::type_id::FLOAT64)) {
    std::unique_ptr<cudf::column> encoded;
    if (primary.type().id() == cudf::type_id::FLOAT64) {
      encoded = makeDoubleTopKSortKeys(primary, stream, get_temp_mr());
      primary = encoded->view();
    }
    auto selected =
        cudf::top_k_order(primary, k, columnOrder_[0], stream, get_temp_mr());
    auto selectedKeys = cudf::gather(
        cudf::table_view({primary}),
        selected->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        get_temp_mr());
    auto [low, high] =
        cudf::minmax(selectedKeys->view().column(0), stream, get_temp_mr());
    const bool descending = columnOrder_[0] == cudf::order::DESCENDING;
    auto mask = cudf::binary_operation(
        primary,
        descending ? *low : *high,
        descending ? cudf::binary_operator::GREATER_EQUAL
                   : cudf::binary_operator::LESS_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_temp_mr());
    auto rowIds = cudf::sequence(
        values.num_rows(),
        cudf::numeric_scalar<int32_t>(0, true, stream, get_temp_mr()),
        cudf::numeric_scalar<int32_t>(1, true, stream, get_temp_mr()),
        stream,
        get_temp_mr());
    auto candidates = cudf::apply_boolean_mask(
        cudf::table_view({rowIds->view()}),
        mask->view(),
        stream,
        get_temp_mr());
    {
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat("topKSelectionBatches", RuntimeCounter(1));
      lockedStats->addRuntimeStat(
          "topKInputRows", RuntimeCounter(values.num_rows()));
      lockedStats->addRuntimeStat(
          "topKCandidateRows", RuntimeCounter(candidates->num_rows()));
    }
    VELOX_CHECK_GE(candidates->num_rows(), k);
    if (candidates->num_rows() < values.num_rows() / 2) {
      // Keep ALL ties on the leading key. Only then apply secondary keys and
      // stable ordering; an arbitrary top-k boundary subset would be wrong.
      auto candidateKeys = cudf::gather(
          keys,
          candidates->view().column(0),
          cudf::out_of_bounds_policy::DONT_CHECK,
          cudf::negative_index_policy::NOT_ALLOWED,
          stream,
          get_temp_mr());
      auto order = cudf::stable_sorted_order(
          candidateKeys->view(),
          columnOrder_,
          nullOrder_,
          stream,
          get_temp_mr());
      auto first = cudf::split(order->view(), {k}, stream).front();
      auto indices = cudf::gather(
          candidates->view(),
          first,
          cudf::out_of_bounds_policy::DONT_CHECK,
          cudf::negative_index_policy::NOT_ALLOWED,
          stream,
          get_temp_mr());
      return cudf::gather(
          values,
          indices->view().column(0),
          cudf::out_of_bounds_policy::DONT_CHECK,
          cudf::negative_index_policy::NOT_ALLOWED,
          stream,
          mr);
    }
  }
  auto const indices =
      cudf::stable_sorted_order(keys, columnOrder_, nullOrder_, stream, mr);
  auto const kIndices =
      cudf::split(indices->view(), {std::min(k, indices->size())}, stream)
          .front();
  return cudf::gather(
      values,
      kIndices,
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);
}

// helper to get topk of a table
CudfVectorPtr CudfTopN::getTopKBatch(CudfVectorPtr cudfInput, int32_t k) {
  if (k == 0 || cudfInput->size() == 0) {
    return nullptr;
  }
  auto stream = cudfInput->stream();
  auto mr = get_output_mr();
  auto values = cudfInput->getTableView();
  auto result = getTopK(values, k, stream, mr);
  auto const size = result->num_rows();
  return std::make_shared<CudfVector>(
      cudfInput->pool(), cudfInput->type(), size, std::move(result), stream);
}

void CudfTopN::doAddInput(RowVectorPtr input) {
  if (count_ == 0 || input->size() == 0) {
    return;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  // Take topk of each input, add to batch.
  // If got kBatchSize_ batches, concat batches and topk once.
  // During getOutput, concat batches and topk once.
  topNBatches_.push_back(getTopKBatch(cudfInput, count_));
  // sum of sizes of topNBatches_ >= count_, then concat and topk once.
  auto totalSize = std::accumulate(
      topNBatches_.begin(),
      topNBatches_.end(),
      0,
      [](int32_t sum, const auto& batch) {
        return sum + (batch ? batch->size() : 0);
      });
  if (topNBatches_.size() >= kBatchSize_ and totalSize >= count_) {
    auto stream = cudfGlobalStreamPool().get_stream();
    auto mr = get_output_mr();

    auto result = mergeTopK(topNBatches_, count_, stream, mr);
    topNBatches_.clear();
    topNBatches_.push_back(std::move(result));
  }
}

RowVectorPtr CudfTopN::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  if (topNBatches_.empty()) {
    finished_ = noMoreInput_;
    return nullptr;
  }

  auto stream = topNBatches_[0]->stream();
  auto mr = get_output_mr();
  auto result = mergeTopK(topNBatches_, count_, stream, mr);
  topNBatches_.clear();
  finished_ = noMoreInput_ && topNBatches_.empty();
  return result;
}

void CudfTopN::doNoMoreInput() {
  Operator::noMoreInput();
  if (topNBatches_.empty()) {
    finished_ = true;
    return;
  }
}

bool CudfTopN::isFinished() {
  return finished_;
}
} // namespace facebook::velox::cudf_velox
