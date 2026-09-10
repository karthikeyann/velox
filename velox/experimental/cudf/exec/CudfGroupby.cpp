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
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ShortStringGroupKeys.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/reduction/unique_count.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <cuda/std/numeric>

#include <gflags/gflags.h>

#include <cmath>
#include <limits>
#include <map>
#include <mutex>

DEFINE_bool(
    cudf_groupby_detect_sorted_keys,
    false,
    "Check integer grouping-key order and use sorted cuDF groupby when verified");
DEFINE_int64(
    cudf_groupby_detect_sorted_min_rows,
    1'000'000,
    "Minimum input rows for runtime sorted-key detection");
DEFINE_int64(
    cudf_partial_groupby_merge_min_rows,
    0,
    "Buffer small partial groupby results before merging; zero preserves eager merging");
DEFINE_bool(
    cudf_groupby_share_nonnull_counts,
    false,
    "Share raw unmasked count requests after verifying input columns have no nulls");
DEFINE_bool(
    cudf_groupby_complete_batches,
    false,
    "Honor single-step noGroupsSpanBatches with task-wide non-null integer range validation");
DEFINE_bool(
    cudf_groupby_stream_raw_single,
    false,
    "Use persistent streaming groupby for single-step raw SUM/MIN/MAX fields");
DEFINE_bool(
    cudf_final_groupby_unique_batches,
    false,
    "Bypass final integer MIN/MAX aggregation after proving all input keys unique");
DEFINE_uint64(
    cudf_final_groupby_unique_max_bytes,
    uint64_t{16} << 30,
    "Maximum retained input bytes per driver for checked unique final aggregation");
DEFINE_int64(
    cudf_final_groupby_unique_min_rows,
    1'000'000,
    "Minimum first-batch rows for checked unique final aggregation");
DEFINE_uint64(
    cudf_dense_integer_sum_max_range,
    0,
    "Enable direct-address single INT64 SUM over one integer key up to this range; zero disables");
DEFINE_int64(
    cudf_dense_integer_sum_min_rows,
    1'000'000,
    "Minimum first-batch rows for bounded integer SUM");
DEFINE_bool(
    cudf_dense_integer_count_rows,
    false,
    "Also use bounded integer-key direct aggregation for unmasked COUNT(*) or non-null constants");
DEFINE_uint64(
    cudf_dense_integer_count_32_max_rows,
    0,
    "Use UINT32 dense COUNT state up to this verified total input-row bound (at most UINT32_MAX); zero disables");
DEFINE_bool(
    cudf_groupby_pack_short_string_keys,
    false,
    "Losslessly pack one/two non-null grouping strings after verifying each is at most three bytes");
DEFINE_int64(
    cudf_groupby_pack_short_string_min_rows,
    100'000,
    "Minimum batch rows for lossless short-string grouping key packing");

namespace facebook::velox::cudf_velox {

// The registry retains validation state until the task dies, even if an early
// driver closes before another driver initializes. It does not retain Tasks.
class DisjointGroupbyRanges {
 public:
  static std::shared_ptr<DisjointGroupbyRanges> get(
      const std::shared_ptr<exec::Task>& task,
      const core::PlanNodeId& node) {
    struct Entry {
      std::weak_ptr<exec::Task> task;
      std::shared_ptr<DisjointGroupbyRanges> ranges;
    };
    static std::mutex registryMutex;
    static std::map<std::pair<exec::Task*, core::PlanNodeId>, Entry> registry;
    std::lock_guard lock(registryMutex);
    for (auto it = registry.begin(); it != registry.end();) {
      if (it->second.task.expired()) {
        it = registry.erase(it);
      } else {
        ++it;
      }
    }
    auto [it, inserted] = registry.try_emplace({task.get(), node});
    if (inserted) {
      it->second = {task, std::make_shared<DisjointGroupbyRanges>()};
    }
    return it->second.ranges;
  }

  void add(int64_t low, int64_t high) {
    std::lock_guard lock(mutex_);
    auto next = ranges_.lower_bound(low);
    VELOX_USER_CHECK(
        (next == ranges_.end() || high < next->first) &&
            (next == ranges_.begin() || std::prev(next)->second < low),
        "noGroupsSpanBatches contract violated: overlapping integer key ranges [{}, {}]",
        low,
        high);
    ranges_.emplace(low, high);
  }

 private:
  std::mutex mutex_;
  std::map<int64_t, int64_t> ranges_;
};

} // namespace facebook::velox::cudf_velox

namespace {

using namespace facebook::velox;
using cudf_velox::castDecimal64InputToDecimal128;
using cudf_velox::CountInputKind;
using cudf_velox::finalizeDecimalAverage;
using cudf_velox::get_output_mr;
using cudf_velox::get_temp_mr;
using cudf_velox::GroupbyAggregator;
using cudf_velox::ResolvedAggregateInfo;
using cudf_velox::serializeDecimalPartialOrIntermediateState;
using cudf_velox::StreamingGroupbyAggregator;
using cudf_velox::validateIntermediateColumnType;

size_t streamingGroupbySafeCapacity() {
  // libcudf encodes an in-flight row as max_distinct_keys + row_index in a
  // signed cudf::size_type. Keeping both operands at or below half of the
  // maximum also bounds the physical cuco table used by streaming_groupby.
  return static_cast<size_t>(std::numeric_limits<cudf::size_type>::max()) / 2;
}

size_t scaleStreamingGroupbyCapacity(
    size_t capacity,
    double multiplier,
    size_t safeCapacity) {
  const auto scaledCapacity = static_cast<long double>(capacity) * multiplier;
  if (scaledCapacity >= safeCapacity) {
    return safeCapacity;
  }
  return std::max<size_t>(static_cast<size_t>(std::ceil(scaledCapacity)), 1);
}

std::unique_ptr<cudf::column> castStreamingOutput(
    std::unique_ptr<cudf::column> column,
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto outputType = cudf_velox::veloxToCudfDataType(type);
  if (column->type() != outputType) {
    column = cudf::cast(*column, outputType, stream, mr);
  }
  return column;
}

template <auto MakeAggregation>
struct SimpleStreamingGroupbyAggregator final : StreamingGroupbyAggregator {
  SimpleStreamingGroupbyAggregator(
      column_index_t inputIndex,
      TypePtr resultType)
      : StreamingGroupbyAggregator(inputIndex, std::move(resultType)) {}

  void prepareInput(
      cudf::table_view input,
      std::vector<cudf::column_view>& preparedColumns,
      rmm::cuda_stream_view stream) override {
    preparedInputIndex_ = prepareColumn(input, preparedColumns, stream);
  }

  void addStreamingRequest(
      std::vector<cudf::groupby::streaming_aggregation_request>& requests)
      override {
    VELOX_CHECK(preparedInputIndex_.has_value());
    resultIndex_ = requests.size();
    requests.push_back(
        cudf::groupby::streaming_aggregation_request{
            static_cast<cudf::size_type>(*preparedInputIndex_),
            MakeAggregation()});
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    return castStreamingOutput(
        std::move(results[resultIndex_].results[0]), resultType, stream, mr);
  }

 private:
  std::optional<column_index_t> preparedInputIndex_;
  size_t resultIndex_{0};
};

using StreamingGroupbySumAggregator = SimpleStreamingGroupbyAggregator<
    &cudf::make_sum_aggregation<cudf::groupby_aggregation>>;
using StreamingGroupbyMinAggregator = SimpleStreamingGroupbyAggregator<
    &cudf::make_min_aggregation<cudf::groupby_aggregation>>;
using StreamingGroupbyMaxAggregator = SimpleStreamingGroupbyAggregator<
    &cudf::make_max_aggregation<cudf::groupby_aggregation>>;
// A final count consumes partial counts, so its streaming request is SUM.
using StreamingGroupbyCountAggregator = SimpleStreamingGroupbyAggregator<
    &cudf::make_sum_aggregation<cudf::groupby_aggregation>>;

struct StreamingGroupbyAverageAggregator final : StreamingGroupbyAggregator {
  StreamingGroupbyAverageAggregator(
      column_index_t inputIndex,
      TypePtr resultType)
      : StreamingGroupbyAggregator(inputIndex, std::move(resultType)) {}

  void prepareInput(
      cudf::table_view input,
      std::vector<cudf::column_view>& preparedColumns,
      rmm::cuda_stream_view stream) override {
    sumInputIndex_ = prepareColumn(input, preparedColumns, stream, 0);
    countInputIndex_ = prepareColumn(input, preparedColumns, stream, 1);
  }

  void addStreamingRequest(
      std::vector<cudf::groupby::streaming_aggregation_request>& requests)
      override {
    VELOX_CHECK(sumInputIndex_.has_value());
    VELOX_CHECK(countInputIndex_.has_value());
    sumResultIndex_ = requests.size();
    requests.push_back(
        cudf::groupby::streaming_aggregation_request{
            static_cast<cudf::size_type>(*sumInputIndex_),
            cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
        });
    countResultIndex_ = requests.size();
    requests.push_back(
        cudf::groupby::streaming_aggregation_request{
            static_cast<cudf::size_type>(*countInputIndex_),
            cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
        });
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto sum = std::move(results[sumResultIndex_].results[0]);
    auto count = std::move(results[countResultIndex_].results[0]);
    auto average = cudf::binary_operation(
        *sum,
        *count,
        cudf::binary_operator::DIV,
        cudf_velox::veloxToCudfDataType(resultType),
        stream,
        mr);

    // Preserve valid NaN inputs while making AVG of an all-null group NULL.
    cudf::numeric_scalar<int64_t> zero(0, true, stream, get_temp_mr());
    auto hasValues = cudf::binary_operation(
        *count,
        zero,
        cudf::binary_operator::GREATER,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_temp_mr());
    auto [validity, nullCount] =
        cudf::bools_to_mask(*hasValues, stream, get_temp_mr());
    average->set_null_mask(std::move(*validity), nullCount);
    return average;
  }

 private:
  std::optional<column_index_t> sumInputIndex_;
  std::optional<column_index_t> countInputIndex_;
  size_t sumResultIndex_{0};
  size_t countResultIndex_{0};
};

template <auto MakeAggregation>
struct SimpleGroupbyAggregator final : GroupbyAggregator {
  SimpleGroupbyAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType,
      std::optional<uint32_t> maskIndex)
      : GroupbyAggregator(
            step,
            inputIndex,
            std::move(constant),
            resultType,
            maskIndex) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    VELOX_CHECK(
        constant == nullptr,
        "Simple groupby aggregator does not yet support constant input");
    auto& request = requests.emplace_back();
    outputIndex_ = requests.size() - 1;
    request.values = materializeMaskedInput(tbl, inputIndex, stream, mr);
    request.aggregations.push_back(MakeAggregation());
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto column = std::move(results[outputIndex_].results[0]);
    const auto cudfType = cudf_velox::veloxToCudfDataType(resultType);
    if (column->type() != cudfType) {
      column = cudf::cast(*column, cudfType, stream, mr);
    }
    return column;
  }

 private:
  uint32_t outputIndex_{0};
};

using GroupbySumAggregator = SimpleGroupbyAggregator<
    &cudf::make_sum_aggregation<cudf::groupby_aggregation>>;
using GroupbyMinAggregator = SimpleGroupbyAggregator<
    &cudf::make_min_aggregation<cudf::groupby_aggregation>>;
using GroupbyMaxAggregator = SimpleGroupbyAggregator<
    &cudf::make_max_aggregation<cudf::groupby_aggregation>>;

// Decimal SUM and AVG aggregators are separate implementations, as they need to
// handle the VARBINARY encoded intermediate state for streaming aggregation.
// Due to the packing and unpacking of that intermediate state, and the special
// handling required for the decimal divide, we cannot just use the existing
// cudf::make_mean_aggregation class. Also, unlike other aggregators, these
// classes hold state (the decoded intermediate sum and count columns and
// associated indices) in order to guarantee a lifetime constraint between
// aggregation steps.

void addDecimalSumCountRequestsAfterDecode(
    cudf::column_view encodedColumn,
    int32_t scale,
    std::vector<cudf::groupby::aggregation_request>& requests,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    uint32_t& countIdx,
    std::unique_ptr<cudf::column>& decodedSum,
    std::unique_ptr<cudf::column>& decodedCount) {
  auto sumAndCount =
      cudf_velox::deserializeDecimalSumState(encodedColumn, scale, stream);
  decodedSum.swap(sumAndCount.sum);
  decodedCount.swap(sumAndCount.count);

  sumIdx = requests.size();
  auto& sumRequest = requests.emplace_back();
  sumRequest.values = decodedSum->view();
  sumRequest.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  countIdx = requests.size();
  auto& countRequest = requests.emplace_back();
  countRequest.values = decodedCount->view();
  countRequest.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

// Decodes serialized state and adds sum + count groupby requests, used by the
// intermediate step (both SUM and AVG) and the final AVG step. resultType is
// DECIMAL for final AVG and DECIMAL or VARBINARY for intermediate; VARBINARY
// carries no scale, so decode at scale 0.
void addDecimalDecodedSumCountRequests(
    cudf::table_view const& tbl,
    uint32_t inputIndex,
    const TypePtr& resultType,
    std::vector<cudf::groupby::aggregation_request>& requests,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    uint32_t& countIdx,
    std::unique_ptr<cudf::column>& decodedSum,
    std::unique_ptr<cudf::column>& decodedCount) {
  validateIntermediateColumnType(tbl.column(inputIndex));
  auto scale = resultType->isDecimal()
      ? getDecimalPrecisionScale(*resultType).second
      : 0;
  addDecimalSumCountRequestsAfterDecode(
      tbl.column(inputIndex),
      scale,
      requests,
      stream,
      sumIdx,
      countIdx,
      decodedSum,
      decodedCount);
}

void addDecimalFinalSumOnlyRequest(
    cudf::table_view const& tbl,
    uint32_t inputIndex,
    const TypePtr& resultType,
    std::vector<cudf::groupby::aggregation_request>& requests,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    std::unique_ptr<cudf::column>& decodedSum) {
  validateIntermediateColumnType(tbl.column(inputIndex));
  auto scale = getDecimalPrecisionScale(*resultType).second;
  auto& request = requests.emplace_back();
  sumIdx = requests.size() - 1;
  auto sumAndCount = cudf_velox::deserializeDecimalSumState(
      tbl.column(inputIndex), scale, stream);
  decodedSum.swap(sumAndCount.sum);
  request.values = decodedSum->view();
  request.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

void addDecimalRawPartialSingleSumRequest(
    cudf::column_view input,
    std::vector<cudf::groupby::aggregation_request>& requests,
    bool includeCountAggregation,
    rmm::cuda_stream_view stream,
    uint32_t& sumIdx,
    std::unique_ptr<cudf::column>& castedInput) {
  auto inputView = castDecimal64InputToDecimal128(input, castedInput, stream);
  auto& request = requests.emplace_back();
  sumIdx = requests.size() - 1;
  request.values = inputView;
  request.aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  if (includeCountAggregation) {
    request.aggregations.push_back(
        cudf::make_count_aggregation<cudf::groupby_aggregation>(
            cudf::null_policy::EXCLUDE));
  }
}

struct GroupbyDecimalSumAggregator : GroupbyAggregator {
  GroupbyDecimalSumAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType,
      std::optional<uint32_t> maskIndex)
      : GroupbyAggregator(step, inputIndex, constant, resultType, maskIndex) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    if (step == core::AggregationNode::Step::kIntermediate) {
      addDecimalDecodedSumCountRequests(
          tbl,
          inputIndex,
          resultType,
          requests,
          stream,
          sumIdx_,
          countIdx_,
          decodedSum_,
          decodedCount_);
    } else if (step == core::AggregationNode::Step::kFinal) {
      addDecimalFinalSumOnlyRequest(
          tbl, inputIndex, resultType, requests, stream, sumIdx_, decodedSum_);
    } else {
      // Raw input (kPartial/kSingle): null-inject masked rows so cuDF's
      // null-excluding sum (and the partial count) honor the mask.
      // materializeMaskedInput returns the plain column when this aggregate has
      // no mask.
      addDecimalRawPartialSingleSumRequest(
          materializeMaskedInput(tbl, inputIndex, stream, mr),
          requests,
          step == core::AggregationNode::Step::kPartial,
          stream,
          sumIdx_,
          castedInput_);
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto col = std::move(results[sumIdx_].results[0]);
    if (step == core::AggregationNode::Step::kPartial) {
      auto count = std::move(results[sumIdx_].results[1]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    if (step == core::AggregationNode::Step::kIntermediate) {
      auto count = std::move(results[countIdx_].results[0]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    auto const cudfResType = cudf_velox::veloxToCudfDataType(resultType);
    if (col->type() != cudfResType) {
      col = cudf::cast(*col, cudfResType, stream, mr);
    }
    return col;
  }

 private:
  uint32_t sumIdx_{0};
  uint32_t countIdx_{0};
  std::unique_ptr<cudf::column> decodedSum_;
  std::unique_ptr<cudf::column> decodedCount_;
  // Holds the DECIMAL64->DECIMAL128 cast of raw input (kPartial/kSingle), kept
  // alive while the groupby request references its view.
  std::unique_ptr<cudf::column> castedInput_;
};

struct GroupbyDecimalAvgAggregator : GroupbyAggregator {
  GroupbyDecimalAvgAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : GroupbyAggregator(
            step,
            inputIndex,
            constant,
            resultType,
            std::nullopt) {}

  // Decimal avg uses a dedicated path that does not honor masks; masked avg
  // already falls back to CPU (see canGroupbyBeEvaluatedByCudf).
  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref /*mr*/) override {
    VELOX_CHECK(!maskIndex.has_value(), "decimal avg does not support masks");
    if (step == core::AggregationNode::Step::kIntermediate ||
        step == core::AggregationNode::Step::kFinal) {
      addDecimalDecodedSumCountRequests(
          tbl,
          inputIndex,
          resultType,
          requests,
          stream,
          sumIdx_,
          countIdx_,
          decodedSum_,
          decodedCount_);
    } else {
      addDecimalRawPartialSingleSumRequest(
          tbl.column(inputIndex),
          requests,
          step == core::AggregationNode::Step::kPartial ||
              step == core::AggregationNode::Step::kSingle,
          stream,
          sumIdx_,
          castedInput_);
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto col = std::move(results[sumIdx_].results[0]);
    if (step == core::AggregationNode::Step::kSingle) {
      auto count = std::move(results[sumIdx_].results[1]);
      return finalizeDecimalAverage(
          std::move(col), std::move(count), resultType, stream, mr);
    }
    if (step == core::AggregationNode::Step::kPartial) {
      auto count = std::move(results[sumIdx_].results[1]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    if (step == core::AggregationNode::Step::kIntermediate) {
      auto count = std::move(results[countIdx_].results[0]);
      return serializeDecimalPartialOrIntermediateState(
          std::move(col), std::move(count), stream, mr);
    }
    if (step == core::AggregationNode::Step::kFinal) {
      auto count = std::move(results[countIdx_].results[0]);
      return finalizeDecimalAverage(
          std::move(col), std::move(count), resultType, stream, mr);
    }
    // All four aggregation steps are handled above.
    VELOX_UNREACHABLE();
  }

 private:
  uint32_t sumIdx_{0};
  uint32_t countIdx_{0};
  std::unique_ptr<cudf::column> decodedSum_;
  std::unique_ptr<cudf::column> decodedCount_;
  // Holds the DECIMAL64->DECIMAL128 cast of raw input (kPartial/kSingle), kept
  // alive while the groupby request references its view.
  std::unique_ptr<cudf::column> castedInput_;
};

struct GroupbyCountAggregator : GroupbyAggregator {
  GroupbyCountAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      CountInputKind inputKind,
      const TypePtr& resultType,
      std::optional<uint32_t> maskIndex)
      : GroupbyAggregator(step, inputIndex, nullptr, resultType, maskIndex),
        inputKind_(inputKind) {}

  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    // kCountAll and kNullConstant both submit a count-all-rows request;
    // kNullConstant overrides the result with zeros in makeOutputColumn.
    const bool countAll = (inputKind_ != CountInputKind::kColumn);
    sharedCount_ = FLAGS_cudf_groupby_share_nonnull_counts &&
        exec::isRawInput(step) && !maskIndex.has_value() &&
        (countAll || tbl.column(inputIndex).null_count() == 0);
    if (sharedCount_) {
      // All canonical requests use column zero and INCLUDE, even when the
      // grouping key in column zero contains nulls. Only GroupbyCountAggregator
      // emits these single-COUNT_ALL requests, and every such producer keeps
      // the result alive until all outputs have been materialized.
      const auto canonical = tbl.column(0);
      for (size_t i = 0; i < requests.size(); ++i) {
        const auto& existing = requests[i];
        if (existing.aggregations.size() == 1 &&
            existing.aggregations[0]->kind == cudf::aggregation::COUNT_ALL &&
            existing.values.head<void>() == canonical.head<void>() &&
            existing.values.type() == canonical.type() &&
            existing.values.offset() == canonical.offset() &&
            existing.values.size() == canonical.size()) {
          outputIndex_ = i;
          return;
        }
      }
    }
    auto& request = requests.emplace_back();
    outputIndex_ = requests.size() - 1;
    if (exec::isRawInput(step) && maskIndex.has_value()) {
      if (countAll) {
        // count(*)/count(const) FILTER(WHERE m): count mask-true rows via a
        // validity-only column + COUNT_VALID.
        maskedCount_ = cudf_velox::maskToValidityColumn(
            tbl.column(*maskIndex), stream, mr);
        request.values = maskedCount_->view();
      } else {
        // count(col) FILTER(WHERE m): null-inject col so validity = m &&
        // valid(col).
        request.values = materializeMaskedInput(tbl, inputIndex, stream, mr);
      }
      request.aggregations.push_back(
          cudf::make_count_aggregation<cudf::groupby_aggregation>(
              cudf::null_policy::EXCLUDE));
    } else if (exec::isRawInput(step)) {
      // For raw input, count(*) can use any column (column 0) since we just
      // need a row count.
      request.values =
          (countAll || sharedCount_) ? tbl.column(0) : tbl.column(inputIndex);
      request.aggregations.push_back(
          cudf::make_count_aggregation<cudf::groupby_aggregation>(
              (countAll || sharedCount_) ? cudf::null_policy::INCLUDE
                                         : cudf::null_policy::EXCLUDE));
    } else {
      // For non-raw input (intermediate/final in streaming), the input is
      // partial results; sum the partial counts.
      request.values = tbl.column(inputIndex);
      request.aggregations.push_back(
          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    auto col = sharedCount_
        ? std::make_unique<cudf::column>(
              results[outputIndex_].results[0]->view(), stream, mr)
        : std::move(results[outputIndex_].results[0]);
    if (inputKind_ == CountInputKind::kNullConstant) {
      auto zero = cudf::numeric_scalar<int64_t>(0, true, stream, get_temp_mr());
      col = cudf::make_column_from_scalar(zero, col->size(), stream, mr);
    }
    // cudf produces int32 for count but velox expects int64.
    const auto cudfOutputType = cudf_velox::veloxToCudfDataType(resultType);
    if (col->type() != cudfOutputType) {
      col = cudf::cast(*col, cudfOutputType, stream, mr);
    }
    return col;
  }

 private:
  CountInputKind inputKind_;
  bool sharedCount_{false};
  uint32_t outputIndex_;
  // Transient validity column for masked count(*)/count(const), valid until the
  // next addGroupbyRequest on this aggregator.
  std::unique_ptr<cudf::column> maskedCount_;
};

struct GroupbyMeanAggregator : GroupbyAggregator {
  GroupbyMeanAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType,
      std::optional<uint32_t> maskIndex)
      : GroupbyAggregator(step, inputIndex, constant, resultType, maskIndex) {}

  // Masked avg falls back to CPU; never masked here.
  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref /*mr*/) override {
    VELOX_CHECK(!maskIndex.has_value(), "avg does not support masks");
    switch (step) {
      case core::AggregationNode::Step::kSingle: {
        auto& request = requests.emplace_back();
        meanIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex);
        request.aggregations.push_back(
            cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case core::AggregationNode::Step::kPartial: {
        auto& request = requests.emplace_back();
        sumIdx_ = requests.size() - 1;
        request.values = tbl.column(inputIndex);
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        request.aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE));
        break;
      }
      case core::AggregationNode::Step::kIntermediate:
      case core::AggregationNode::Step::kFinal: {
        // In intermediate and final aggregation, the previously computed sum
        // and count are in the child columns of the input column. A borrowed
        // partition may slice the struct without slicing its children.
        const cudf::structs_column_view state(tbl.column(inputIndex));
        auto& request = requests.emplace_back();
        sumIdx_ = requests.size() - 1;
        request.values = state.get_sliced_child(0, stream);
        request.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());

        auto& request2 = requests.emplace_back();
        countIdx_ = requests.size() - 1;
        request2.values = state.get_sliced_child(1, stream);
        // The counts are already computed in partial aggregation, so we just
        // need to sum them up again.
        request2.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        break;
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    const auto& outputType = asRowType(resultType);
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return std::move(results[meanIdx_].results[0]);
      case core::AggregationNode::Step::kPartial: {
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[sumIdx_].results[1]);

        auto const size = sum->size();
        auto const cudfSumType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        if (sum->type() != cudf::data_type(cudfSumType)) {
          sum = cudf::cast(*sum, cudf::data_type(cudfSumType), stream, mr);
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count =
              cudf::cast(*count, cudf::data_type(cudfCountType), stream, mr);
        }

        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum));
        children.push_back(std::move(count));

        // TODO: Handle nulls. This can happen if all values are null in a
        // group.
        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            size,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kIntermediate: {
        // The difference between intermediate and partial is in where the
        // sum and count are coming from. In partial, since the input column is
        // the same, the sum and count are in the same agg result. In
        // intermediate, the input columns are different (it's the child
        // columns of the input column) and so the sum and count are in
        // different agg results.
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[countIdx_].results[0]);

        auto size = sum->size();
        auto const cudfSumType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        if (sum->type() != cudf::data_type(cudfSumType)) {
          sum = cudf::cast(*sum, cudf::data_type(cudfSumType), stream, mr);
        }
        if (count->type() != cudf::data_type(cudfCountType)) {
          count =
              cudf::cast(*count, cudf::data_type(cudfCountType), stream, mr);
        }

        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sum));
        children.push_back(std::move(count));

        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            size,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kFinal: {
        auto sum = std::move(results[sumIdx_].results[0]);
        auto count = std::move(results[countIdx_].results[0]);
        auto avg = cudf::binary_operation(
            *sum,
            *count,
            cudf::binary_operator::DIV,
            cudf_velox::veloxToCudfDataType(resultType),
            stream,
            mr);
        // Null out groups where count == 0 (empty groups).
        // SQL semantics require avg of an empty group to be NULL, but
        // cudf's 0/0 division produces NaN.  We mask on count rather
        // than using column_nans_to_nulls so that legitimate NaN
        // results (from NaN inputs) are preserved.
        cudf::numeric_scalar<int64_t> zero(0, true, stream, get_temp_mr());
        auto validMask = cudf::binary_operation(
            *count,
            zero,
            cudf::binary_operator::GREATER,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            get_temp_mr());
        auto [mask, nullCount] =
            cudf::bools_to_mask(*validMask, stream, get_temp_mr());
        avg->set_null_mask(std::move(*mask), nullCount);
        return avg;
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }

 private:
  // These indices are used to track where the desired result columns
  // (mean/<sum, count>) are in the output of cudf::groupby::aggregate().
  uint32_t meanIdx_;
  uint32_t sumIdx_;
  uint32_t countIdx_;
};

struct GroupbyStddevSampAggregator : GroupbyAggregator {
  GroupbyStddevSampAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType,
      std::optional<uint32_t> maskIndex)
      : GroupbyAggregator(step, inputIndex, constant, resultType, maskIndex) {}

  // Masked stddev falls back to CPU; never masked here.
  void addGroupbyRequest(
      cudf::table_view const& tbl,
      std::vector<cudf::groupby::aggregation_request>& requests,
      rmm::cuda_stream_view /*stream*/,
      rmm::device_async_resource_ref /*mr*/) override {
    VELOX_CHECK(!maskIndex.has_value(), "stddev does not support masks");
    auto& request = requests.emplace_back();
    outputIdx_ = requests.size() - 1;
    request.values = tbl.column(inputIndex);

    switch (step) {
      case core::AggregationNode::Step::kSingle:
        // Use cuDF's built-in std aggregation with ddof=1 (sample stddev)
        request.aggregations.push_back(
            cudf::make_std_aggregation<cudf::groupby_aggregation>(1));
        break;
      case core::AggregationNode::Step::kPartial:
        // Compute count, mean, m2 from raw values
        request.aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(
                cudf::null_policy::EXCLUDE));
        request.aggregations.push_back(
            cudf::make_mean_aggregation<cudf::groupby_aggregation>());
        request.aggregations.push_back(
            cudf::make_m2_aggregation<cudf::groupby_aggregation>());
        break;
      case core::AggregationNode::Step::kIntermediate:
      case core::AggregationNode::Step::kFinal:
        // Input is struct(count, mean, m2) - use MERGE_M2 to merge
        request.aggregations.push_back(
            cudf::make_merge_m2_aggregation<cudf::groupby_aggregation>());
        break;
      default:
        VELOX_NYI("Unsupported aggregation step for stddev_samp");
    }
  }

  std::unique_ptr<cudf::column> makeOutputColumn(
      std::vector<cudf::groupby::aggregation_result>& results,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return std::move(results[outputIdx_].results[0]);
      case core::AggregationNode::Step::kPartial: {
        auto count = std::move(results[outputIdx_].results[0]);
        auto mean = std::move(results[outputIdx_].results[1]);
        auto m2 = std::move(results[outputIdx_].results[2]);
        return makeM2StructColumn(
            std::move(count), std::move(mean), std::move(m2), stream, mr);
      }
      case core::AggregationNode::Step::kIntermediate: {
        auto merged = std::move(results[outputIdx_].results[0]);

        // Check if types already match expected output - avoid copies if so
        const auto& outputType = asRowType(resultType);
        auto const cudfCountType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(0));
        auto const cudfMeanType =
            cudf_velox::veloxToCudfDataType(outputType->childAt(1));
        auto const cudfM2Type =
            cudf_velox::veloxToCudfDataType(outputType->childAt(2));

        auto mergedView = merged->view();
        bool typesMatch = mergedView.child(0).type() == cudfCountType &&
            mergedView.child(1).type() == cudfMeanType &&
            mergedView.child(2).type() == cudfM2Type;

        if (typesMatch) {
          // Types match - return merged directly to avoid device copies
          return merged;
        }

        // Types don't match - need to copy and cast (use output_mr since
        // these become part of the output)
        auto count =
            std::make_unique<cudf::column>(mergedView.child(0), stream, mr);
        auto mean =
            std::make_unique<cudf::column>(mergedView.child(1), stream, mr);
        auto m2 =
            std::make_unique<cudf::column>(mergedView.child(2), stream, mr);
        return makeM2StructColumn(
            std::move(count), std::move(mean), std::move(m2), stream, mr);
      }
      case core::AggregationNode::Step::kFinal: {
        // MERGE_M2 returns struct(count, mean, m2)
        // Compute sqrt(m2 / (count - 1)) with NULL where count < 2
        auto merged = std::move(results[outputIdx_].results[0]);
        auto mergedView = merged->view();
        auto countView = mergedView.child(0);
        auto m2View = mergedView.child(2);

        // count - 1 (binary_operation handles type promotion)
        cudf::numeric_scalar<double> one(1.0, true, stream, get_temp_mr());
        auto countMinus1 = cudf::binary_operation(
            countView,
            one,
            cudf::binary_operator::SUB,
            cudf::data_type{cudf::type_id::FLOAT64},
            stream,
            get_temp_mr());

        // m2 / (count - 1)
        auto variance = cudf::binary_operation(
            m2View,
            *countMinus1,
            cudf::binary_operator::DIV,
            cudf::data_type{cudf::type_id::FLOAT64},
            stream,
            get_temp_mr());

        // sqrt(variance)
        auto stddev = cudf::unary_operation(
            *variance, cudf::unary_operator::SQRT, stream, get_temp_mr());

        // count >= 2
        cudf::numeric_scalar<int64_t> two(2, true, stream, get_temp_mr());
        auto validMask = cudf::binary_operation(
            countView,
            two,
            cudf::binary_operator::GREATER_EQUAL,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            get_temp_mr());

        // Apply mask: where count < 2, result is NULL
        cudf::numeric_scalar<double> nullDouble(
            0.0, false, stream, get_temp_mr());
        return cudf::copy_if_else(*stddev, nullDouble, *validMask, stream, mr);
      }
      default:
        VELOX_NYI("Unsupported aggregation step for stddev_samp");
    }
  }

 private:
  // Build a struct column with (count, mean, m2), casting to expected types.
  std::unique_ptr<cudf::column> makeM2StructColumn(
      std::unique_ptr<cudf::column> count,
      std::unique_ptr<cudf::column> mean,
      std::unique_ptr<cudf::column> m2,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    const auto& outputType = asRowType(resultType);
    auto const cudfCountType =
        cudf_velox::veloxToCudfDataType(outputType->childAt(0));
    auto const cudfMeanType =
        cudf_velox::veloxToCudfDataType(outputType->childAt(1));
    auto const cudfM2Type =
        cudf_velox::veloxToCudfDataType(outputType->childAt(2));
    if (count->type() != cudfCountType) {
      count = cudf::cast(*count, cudfCountType, stream, mr);
    }
    if (mean->type() != cudfMeanType) {
      mean = cudf::cast(*mean, cudfMeanType, stream, mr);
    }
    if (m2->type() != cudfM2Type) {
      m2 = cudf::cast(*m2, cudfM2Type, stream, mr);
    }

    auto const size = count->size();
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(std::move(count));
    children.push_back(std::move(mean));
    children.push_back(std::move(m2));

    return std::make_unique<cudf::column>(
        cudf::data_type(cudf::type_id::STRUCT),
        size,
        rmm::device_buffer{},
        rmm::device_buffer{},
        0,
        std::move(children));
  }

  uint32_t outputIdx_;
};

std::unique_ptr<GroupbyAggregator> createGroupbyAggregator(
    const ResolvedAggregateInfo& p) {
  auto const& kind = p.kind;
  auto prefix = cudf_velox::CudfConfig::getInstance().functionNamePrefix;
  if (kind.rfind(prefix + "sum", 0) == 0) {
    if (p.isDecimalAggregate) {
      return std::make_unique<GroupbyDecimalSumAggregator>(
          p.companionStep, p.inputIndex, p.constant, p.resultType, p.maskIndex);
    }
    return std::make_unique<GroupbySumAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType, p.maskIndex);
  } else if (kind.rfind(prefix + "count", 0) == 0) {
    VELOX_CHECK(p.countInputKind.has_value());
    return std::make_unique<GroupbyCountAggregator>(
        p.companionStep,
        p.inputIndex,
        *p.countInputKind,
        p.resultType,
        p.maskIndex);
  } else if (kind.rfind(prefix + "min", 0) == 0) {
    return std::make_unique<GroupbyMinAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType, p.maskIndex);
  } else if (kind.rfind(prefix + "max", 0) == 0) {
    return std::make_unique<GroupbyMaxAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType, p.maskIndex);
  } else if (kind.rfind(prefix + "avg", 0) == 0) {
    if (p.isDecimalAggregate) {
      return std::make_unique<GroupbyDecimalAvgAggregator>(
          p.companionStep, p.inputIndex, p.constant, p.resultType);
    }
    return std::make_unique<GroupbyMeanAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType, p.maskIndex);
  } else if (kind.rfind(prefix + "stddev_samp", 0) == 0) {
    return std::make_unique<GroupbyStddevSampAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType, p.maskIndex);
  } else if (kind.rfind(prefix + "stddev", 0) == 0) {
    // stddev is an alias for stddev_samp
    return std::make_unique<GroupbyStddevSampAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType, p.maskIndex);
  } else {
    VELOX_NYI("Aggregation not yet supported, kind: {}", kind);
  }
}

std::unique_ptr<StreamingGroupbyAggregator> createStreamingGroupbyAggregator(
    const ResolvedAggregateInfo& aggregate,
    column_index_t inputIndex,
    const TypePtr& inputType,
    const TypePtr& resultType) {
  if (aggregate.isDecimalAggregate || aggregate.constant != nullptr ||
      aggregate.maskIndex.has_value()) {
    return nullptr;
  }

  const auto prefix = cudf_velox::CudfConfig::getInstance().functionNamePrefix;
  if (aggregate.kind == prefix + "sum") {
    if (!cudf::groupby::is_streaming_groupby_supported(
            cudf_velox::veloxToCudfDataType(inputType),
            cudf::aggregation::SUM)) {
      return nullptr;
    }
    return std::make_unique<StreamingGroupbySumAggregator>(
        inputIndex, resultType);
  }
  if (aggregate.kind == prefix + "min") {
    if (!cudf::groupby::is_streaming_groupby_supported(
            cudf_velox::veloxToCudfDataType(inputType),
            cudf::aggregation::MIN)) {
      return nullptr;
    }
    return std::make_unique<StreamingGroupbyMinAggregator>(
        inputIndex, resultType);
  }
  if (aggregate.kind == prefix + "max") {
    if (!cudf::groupby::is_streaming_groupby_supported(
            cudf_velox::veloxToCudfDataType(inputType),
            cudf::aggregation::MAX)) {
      return nullptr;
    }
    return std::make_unique<StreamingGroupbyMaxAggregator>(
        inputIndex, resultType);
  }
  if (aggregate.kind == prefix + "count") {
    if (!cudf::groupby::is_streaming_groupby_supported(
            cudf_velox::veloxToCudfDataType(inputType),
            cudf::aggregation::SUM)) {
      return nullptr;
    }
    return std::make_unique<StreamingGroupbyCountAggregator>(
        inputIndex, resultType);
  }
  if (aggregate.kind == prefix + "avg") {
    if (inputType->kind() != TypeKind::ROW) {
      return nullptr;
    }
    const auto inputRowType = asRowType(inputType);
    if (inputRowType->size() != 2 ||
        !cudf::groupby::is_streaming_groupby_supported(
            cudf_velox::veloxToCudfDataType(inputRowType->childAt(0)),
            cudf::aggregation::SUM) ||
        !cudf::groupby::is_streaming_groupby_supported(
            cudf_velox::veloxToCudfDataType(inputRowType->childAt(1)),
            cudf::aggregation::SUM)) {
      return nullptr;
    }
    return std::make_unique<StreamingGroupbyAverageAggregator>(
        inputIndex, resultType);
  }
  return nullptr;
}

} // namespace

namespace facebook::velox::cudf_velox {

column_index_t StreamingGroupbyAggregator::prepareColumn(
    cudf::table_view input,
    std::vector<cudf::column_view>& preparedColumns,
    rmm::cuda_stream_view stream,
    std::optional<column_index_t> childIndex) const {
  VELOX_CHECK_LT(inputIndex, input.num_columns());
  auto column = input.column(inputIndex);
  if (childIndex.has_value()) {
    VELOX_CHECK_LT(*childIndex, column.num_children());
    column =
        cudf::structs_column_view(column).get_sliced_child(*childIndex, stream);
  }
  VELOX_CHECK_EQ(column.size(), input.num_rows());
  preparedColumns.push_back(column);
  return static_cast<column_index_t>(preparedColumns.size() - 1);
}

cudf::column_view GroupbyAggregator::materializeMaskedInput(
    cudf::table_view const& tbl,
    uint32_t valueIdx,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (!maskIndex.has_value()) {
    return tbl.column(valueIdx);
  }
  VELOX_CHECK(exec::isRawInput(step), "mask only valid at raw-input steps");
  maskedValues_ = materializeMaskedColumn(tbl, valueIdx, maskIndex, stream, mr);
  return maskedValues_->view();
}

std::vector<std::unique_ptr<GroupbyAggregator>> toGroupbyAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants,
    std::vector<std::optional<uint32_t>> const& maskChannels) {
  auto params = resolveAggregateInfos(
      aggregationNode, step, outputType, constants, maskChannels);

  std::vector<std::unique_ptr<GroupbyAggregator>> aggregators;
  aggregators.reserve(params.size());
  for (const auto& p : params) {
    aggregators.push_back(createGroupbyAggregator(p));
  }
  return aggregators;
}

std::optional<std::vector<std::unique_ptr<StreamingGroupbyAggregator>>>
toStreamingGroupbyAggregators(
    const core::AggregationNode& aggregationNode,
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& aggregationInputChannels,
    const TypePtr& outputType,
    const std::vector<VectorPtr>& constants,
    const std::vector<std::optional<uint32_t>>& maskChannels) {
  const auto params = resolveAggregateInfos(
      aggregationNode,
      aggregationNode.step(),
      outputType,
      constants,
      maskChannels);
  const auto numKeys = aggregationNode.groupingKeys().size();

  std::vector<std::unique_ptr<StreamingGroupbyAggregator>> aggregators;
  aggregators.reserve(params.size());
  for (size_t i = 0; i < params.size(); ++i) {
    if (aggregationNode.step() == core::AggregationNode::Step::kSingle) {
      const auto prefix = CudfConfig::getInstance().functionNamePrefix;
      // COUNT uses an INT32 accumulator in cuDF and AVG expects a partial
      // (sum,count) input in the final-only adapter below. Neither mapping is
      // valid for unbounded raw input. Constants also have no input channel.
      if (params[i].constant || params[i].maskIndex.has_value() ||
          params[i].isDecimalAggregate ||
          (params[i].kind != prefix + "sum" &&
           params[i].kind != prefix + "min" &&
           params[i].kind != prefix + "max")) {
        return std::nullopt;
      }
    }
    const auto inputIndex = aggregationInputChannels.at(params[i].inputIndex);
    auto aggregator = createStreamingGroupbyAggregator(
        params[i],
        inputIndex,
        inputType->childAt(inputIndex),
        outputType->childAt(numKeys + i));
    if (!aggregator) {
      return std::nullopt;
    }
    aggregators.push_back(std::move(aggregator));
  }
  return aggregators;
}

bool canGroupbyAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx) {
  return canAggregationBeEvaluatedByRegistry(
      getGroupbyAggregationRegistry(), call, step, rawInputTypes, queryCtx);
}

bool canGroupbyBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  const core::PlanNode* sourceNode = aggregationNode.sources().empty()
      ? nullptr
      : aggregationNode.sources()[0].get();

  // Get the aggregation step from the node
  auto step = aggregationNode.step();

  // Check supported aggregation functions using step-aware aggregation registry
  for (const auto& aggregate : aggregationNode.aggregates()) {
    // Use step-aware validation that handles partial/final/intermediate steps
    if (!canGroupbyAggregationBeEvaluatedByCudf(
            *aggregate.call, step, aggregate.rawInputTypes, queryCtx)) {
      return false;
    }

    // `distinct` aggregations are not supported, in testing fails with "De-dup
    // before aggregation is not yet supported"
    if (aggregate.distinct) {
      return false;
    }

    if (!maskSupportedByCudf(aggregate, step)) {
      return false;
    }

    if (isCountFunctionName(aggregate.call->name())) {
      continue;
    }

    // Check input expressions can be evaluated by cuDF, expand the input first.
    for (const auto& input : aggregate.call->inputs()) {
      auto expandedInput = expandFieldReference(input, sourceNode);
      if (!canExprRunOnGpu(expandedInput, queryCtx, pool)) {
        return false;
      }
    }
  }

  // Check grouping key expressions
  if (!canGroupingKeysBeEvaluatedByCudf(
          aggregationNode.groupingKeys(), sourceNode, queryCtx, pool)) {
    return false;
  }

  return true;
}

CudfGroupby::CudfGroupby(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<core::AggregationNode const> const& aggregationNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          aggregationNode->outputType(),
          aggregationNode->id(),
          std::string{"CudfGroupby"} +
              std::string{
                  core::AggregationNode::toName(aggregationNode->step())},
          nvtx3::rgb{34, 139, 34}, // Forest Green
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          std::nullopt,
          aggregationNode),
      aggregationNode_(aggregationNode),
      isPartialOutput_(
          exec::isPartialOutput(aggregationNode->step()) &&
          !hasFinalAggs(aggregationNode->aggregates())),
      isSingleStep_(
          aggregationNode->step() == core::AggregationNode::Step::kSingle),
      maxPartialAggregationMemoryUsage_(
          driverCtx->queryConfig().maxPartialAggregationMemoryUsage()) {}

bool CudfGroupby::initializeStreamingGroupby(
    const RowTypePtr& inputRowSchema,
    const std::vector<VectorPtr>& constants,
    const std::vector<std::optional<uint32_t>>& maskChannels) {
  const auto& config = CudfConfig::getInstance();
  if (!config.streamingGroupbyEnabled || !incrementalAggregationEnabled_ ||
      (aggregationNode_->step() != core::AggregationNode::Step::kFinal &&
       !(isSingleStep_ && FLAGS_cudf_groupby_stream_raw_single)) ||
      aggregationNode_->groupingKeys().empty() ||
      aggregationNode_->aggregates().empty()) {
    return false;
  }

  // libcudf's streaming_groupby does not accept an MR for persistent state and
  // allocates it from the current device resource. Preserve the existing
  // groupby contract when output allocations use a distinct resource by
  // falling back until libcudf exposes a persistent-state MR parameter.
  if (!config.outputMemoryResource.empty() &&
      config.outputMemoryResource != config.memoryResource) {
    return false;
  }

  auto aggregators = toStreamingGroupbyAggregators(
      *aggregationNode_,
      inputRowSchema,
      aggregationInputChannels_,
      outputType_,
      constants,
      maskChannels);
  if (!aggregators.has_value()) {
    return false;
  }

  streamingGroupbyAggregators_ = std::move(*aggregators);
  return true;
}

cudf::table_view CudfGroupby::makeStreamingGroupbyInputView(
    cudf::table_view input,
    rmm::cuda_stream_view stream) {
  std::vector<cudf::column_view> columns;
  columns.reserve(
      groupingKeyOutputChannels_.size() +
      streamingGroupbyAggregators_.size() * 2);

  // Keys are deliberately the first columns in the prepared table. Therefore
  // streaming_groupby's key indices are the identity range regardless of the
  // output-channel permutation used by the regular groupby path.
  for (const auto inputIndex : groupingKeyInputChannels_) {
    VELOX_CHECK_LT(inputIndex, input.num_columns());
    columns.push_back(input.column(inputIndex));
  }
  for (auto& aggregator : streamingGroupbyAggregators_) {
    aggregator->prepareInput(input, columns, stream);
  }
  return cudf::table_view{columns};
}

std::unique_ptr<cudf::groupby::streaming_groupby>
CudfGroupby::createStreamingGroupby(size_t capacity) {
  VELOX_CHECK_GT(capacity, 0);
  VELOX_CHECK_LE(capacity, streamingGroupbySafeCapacity());

  std::vector<cudf::size_type> keyIndices;
  keyIndices.reserve(groupingKeyInputChannels_.size());
  for (cudf::size_type i = 0; i < groupingKeyInputChannels_.size(); ++i) {
    keyIndices.push_back(i);
  }

  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.reserve(streamingGroupbyAggregators_.size() * 2);
  for (const auto& aggregator : streamingGroupbyAggregators_) {
    aggregator->addStreamingRequest(requests);
  }

  return std::make_unique<cudf::groupby::streaming_groupby>(
      keyIndices,
      requests,
      static_cast<cudf::size_type>(capacity),
      ignoreNullKeys_ ? cudf::null_policy::EXCLUDE
                      : cudf::null_policy::INCLUDE);
}

void CudfGroupby::computeFinalGroupbyStreaming(CudfVectorPtr input) {
  const auto inputRows = static_cast<size_t>(input->size());
  const auto safeCapacity = streamingGroupbySafeCapacity();
  const auto capacityMultiplier =
      CudfConfig::getInstance().streamingGroupbyCapacityMultiplier;
  VELOX_USER_CHECK(
      std::isfinite(capacityMultiplier) && capacityMultiplier > 1.0,
      "streaming_groupby capacity multiplier must be finite and greater than "
      "one. Multiplier: {}",
      capacityMultiplier);

  // If the first batch cannot satisfy libcudf's signed size_type encoding
  // bound, use the existing groupby path before creating any persistent state.
  if (!streamingGroupby_ && inputRows > safeCapacity) {
    streamingGroupbyEnabled_ = false;
    computeFinalGroupbyIncrementally(std::move(input));
    return;
  }

  const auto inputStream = input->stream();
  if (!streamingGroupbyStream_.has_value()) {
    streamingGroupbyStream_ = inputStream;
  }
  const auto stateStream = *streamingGroupbyStream_;
  const bool needsStreamJoin = stateStream.value() != inputStream.value();
  if (needsStreamJoin) {
    if (!streamingGroupbyEvent_) {
      streamingGroupbyEvent_ =
          std::make_unique<CudaEvent>(cudaEventDisableTiming);
    }
    streamingGroupbyEvent_->recordFrom(inputStream).waitOn(stateStream);
  }

  auto orderInputDeallocation = [&]() {
    if (needsStreamJoin) {
      streamingGroupbyEvent_->recordFrom(stateStream).waitOn(inputStream);
    }
  };

  auto preparedInput =
      makeStreamingGroupbyInputView(input->getTableView(), stateStream);
  try {
    if (!streamingGroupby_) {
      // max_distinct_keys is a logical capacity. libcudf's 0.5 cuco load
      // factor allocates roughly two physical hash slots per logical key. The
      // configured multiplier controls both this initial logical headroom and
      // subsequent geometric growth.
      streamingGroupbyCapacity_ = scaleStreamingGroupbyCapacity(
          inputRows, capacityMultiplier, safeCapacity);
      streamingGroupby_ = createStreamingGroupby(streamingGroupbyCapacity_);
      streamingGroupby_->aggregate(preparedInput, stateStream);
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat(
          std::string{kStreamingGroupbyUsedStat}, RuntimeCounter(1));
    } else {
      const auto distinctKeys =
          static_cast<size_t>(streamingGroupby_->distinct_keys());
      const auto requiredCapacity =
          cuda::std::saturating_add(distinctKeys, inputRows);
      VELOX_USER_CHECK_LE(
          requiredCapacity,
          safeCapacity,
          "streaming_groupby cannot safely hold the existing keys plus the "
          "input batch. Reduce the upstream GPU batch size. Keys: {}, batch "
          "rows: {}",
          distinctKeys,
          inputRows);

      if (requiredCapacity > streamingGroupbyCapacity_) {
        const auto geometricCapacity = scaleStreamingGroupbyCapacity(
            streamingGroupbyCapacity_, capacityMultiplier, safeCapacity);
        const auto newCapacity = std::min(
            std::max(requiredCapacity, geometricCapacity), safeCapacity);

        // Overflow invalidates streaming_groupby. Grow transactionally before
        // insertion, aggregate the new batch into the replacement, then merge
        // the old valid state. The conservative distinct+rows bound guarantees
        // that neither operation can overflow the replacement.
        auto replacement = createStreamingGroupby(newCapacity);
        replacement->aggregate(preparedInput, stateStream);
        replacement->merge(*streamingGroupby_, stateStream);
        // streaming_groupby's destructor has no stream parameter. Ensure the
        // merge has finished reading the old persistent state before dropping
        // it.
        stateStream.synchronize();
        streamingGroupby_ = std::move(replacement);
        streamingGroupbyCapacity_ = newCapacity;

        auto lockedStats = stats_.wlock();
        lockedStats->addRuntimeStat(
            std::string{kStreamingGroupbyRebuildsStat}, RuntimeCounter(1));
      } else {
        streamingGroupby_->aggregate(preparedInput, stateStream);
      }
    }
  } catch (...) {
    orderInputDeallocation();
    throw;
  }
  orderInputDeallocation();
}

CudfVectorPtr CudfGroupby::finalizeStreamingGroupby() {
  if (!streamingGroupby_) {
    return nullptr;
  }

  VELOX_CHECK(streamingGroupbyStream_.has_value());
  const auto stream = *streamingGroupbyStream_;
  auto [groupKeys, results] =
      streamingGroupby_->finalize(stream, get_output_mr());

  std::vector<std::unique_ptr<cudf::column>> outputColumns;
  auto keyColumns = groupKeys->release();
  outputColumns.reserve(
      keyColumns.size() + streamingGroupbyAggregators_.size());
  outputColumns.insert(
      outputColumns.end(),
      std::make_move_iterator(keyColumns.begin()),
      std::make_move_iterator(keyColumns.end()));

  for (const auto& aggregator : streamingGroupbyAggregators_) {
    outputColumns.push_back(
        aggregator->makeOutputColumn(results, stream, get_output_mr()));
  }

  auto resultTable = std::make_unique<cudf::table>(std::move(outputColumns));
  const auto numRows = resultTable->num_rows();
  auto result = numRows == 0
      ? nullptr
      : std::make_shared<cudf_velox::CudfVector>(
            pool(), outputType_, numRows, std::move(resultTable), stream);

  // libcudf finalization reads persistent state asynchronously. Its destructor
  // has no stream parameter, so wait before releasing that state.
  stream.synchronize();
  streamingGroupby_.reset();
  streamingGroupbyStream_.reset();
  streamingGroupbyEvent_.reset();
  streamingGroupbyCapacity_ = 0;
  return result;
}

void CudfGroupby::initialize() {
  Operator::initialize();

  inputType_ = aggregationNode_->sources()[0]->outputType();
  ignoreNullKeys_ = aggregationNode_->ignoreNullKeys();
  setupGroupingKeyChannelProjections(
      *aggregationNode_, groupingKeyInputChannels_, groupingKeyOutputChannels_);

  // Velox CPU does optimizations related to pre-grouped keys. This can be
  // done in cudf by passing sort information to cudf::groupby() constructor.
  // We're postponing this for now.

  numAggregates_ = aggregationNode_->aggregates().size();
  const auto inputRowSchema = asRowType(inputType_);
  auto aggregationInput = buildAggregationInputChannels(
      *aggregationNode_,
      *operatorCtx_,
      inputRowSchema,
      groupingKeyInputChannels_);
  aggregationInputChannels_ = std::move(aggregationInput.channels);
  aggregators_ = toGroupbyAggregators(
      *aggregationNode_,
      aggregationNode_->step(),
      outputType_,
      aggregationInput.constants,
      aggregationInput.maskChannels);
  incrementalAggregationEnabled_ =
      !hasCompanionAggregates(aggregationNode_->aggregates());

  if (FLAGS_cudf_groupby_complete_batches && isSingleStep_ &&
      aggregationNode_->noGroupsSpanBatches()) {
    VELOX_USER_CHECK_EQ(
        groupingKeyInputChannels_.size(),
        1,
        "Complete-batch GPU groupby requires one integer key");
    const auto keyType = inputRowSchema->childAt(groupingKeyInputChannels_[0]);
    VELOX_USER_CHECK(
        keyType->isInteger() || keyType->isBigint(),
        "Complete-batch GPU groupby requires INT32 or INT64 keys");
    VELOX_USER_CHECK(
        incrementalAggregationEnabled_,
        "Complete-batch GPU groupby does not support companion aggregates");
    disjointGroupRanges_ = DisjointGroupbyRanges::get(
        operatorCtx_->task(), aggregationNode_->id());
  }

  // Make aggregators for intermediate step when streaming is enabled.
  if (incrementalAggregationEnabled_) {
    const bool isFinalOrSingle =
        aggregationNode_->step() == core::AggregationNode::Step::kFinal ||
        aggregationNode_->step() == core::AggregationNode::Step::kSingle;
    bufferedResultType_ = isFinalOrSingle
        ? getBufferedResultType(*aggregationNode_)
        : outputType_;

    std::vector<VectorPtr> nullConstants(numAggregates_);
    // Non-raw steps carry no masks; pass an empty maskChannels so maskIndex
    // stays nullopt there.
    intermediateAggregators_ = toGroupbyAggregators(
        *aggregationNode_,
        core::AggregationNode::Step::kIntermediate,
        bufferedResultType_,
        nullConstants,
        {});

    if (isSingleStep_) {
      // The kSingle streaming partial path runs for a kSingle masked query, so
      // it must carry the raw-input mask channels.
      partialAggregators_ = toGroupbyAggregators(
          *aggregationNode_,
          core::AggregationNode::Step::kPartial,
          bufferedResultType_,
          aggregationInput.constants,
          aggregationInput.maskChannels);
      finalAggregators_ = toGroupbyAggregators(
          *aggregationNode_,
          core::AggregationNode::Step::kFinal,
          outputType_,
          nullConstants,
          {});
    }
  }

  streamingGroupbyEnabled_ = !disjointGroupRanges_ &&
      initializeStreamingGroupby(
          inputRowSchema,
          aggregationInput.constants,
          aggregationInput.maskChannels);

  if (FLAGS_cudf_final_groupby_unique_batches &&
      aggregationNode_->step() == core::AggregationNode::Step::kFinal &&
      incrementalAggregationEnabled_ && groupingKeyInputChannels_.size() == 1 &&
      numAggregates_ > 0 &&
      aggregationInputChannels_.size() == numAggregates_ + 1) {
    const auto keyType = inputRowSchema->childAt(groupingKeyInputChannels_[0]);
    const auto prefix = CudfConfig::getInstance().functionNamePrefix;
    // Match logical types, not just their storage kind: DATE and short
    // DECIMAL also use integer storage but have different cuDF representations.
    uniqueFinalBatchesEligible_ =
        INTEGER()->equivalent(*keyType) || BIGINT()->equivalent(*keyType);
    for (size_t i = 0; i < numAggregates_; ++i) {
      const auto& aggregate = aggregationNode_->aggregates()[i];
      const auto& call = aggregate.call;
      // A single nullable integer MIN/MAX state is already its final value.
      // Exclude other functions, companions, masks, constants and type changes.
      uniqueFinalBatchesEligible_ = uniqueFinalBatchesEligible_ &&
          (call->name() == prefix + "min" || call->name() == prefix + "max") &&
          BIGINT()->equivalent(*call->type()) && call->inputs().size() == 1 &&
          BIGINT()->equivalent(
              *inputRowSchema->childAt(aggregationInputChannels_[i + 1])) &&
          !aggregationInput.constants[i] && !aggregate.mask &&
          !aggregate.distinct && aggregate.sortingKeys.empty();
    }
  }

  if (FLAGS_cudf_dense_integer_sum_max_range > 0 && isSingleStep_ &&
      incrementalAggregationEnabled_ && !disjointGroupRanges_ &&
      groupingKeyInputChannels_.size() == 1 && numAggregates_ == 1 &&
      aggregationInputChannels_.size() == 2 &&
      groupingKeyOutputChannels_[0] == 0 &&
      aggregationInputChannels_[0] == groupingKeyInputChannels_[0] &&
      outputType_->childAt(1)->isBigint() &&
      bufferedResultType_->childAt(1)->isBigint() &&
      !aggregationInput.maskChannels[0].has_value()) {
    const auto keyType = inputRowSchema->childAt(groupingKeyInputChannels_[0]);
    const auto valueType =
        inputRowSchema->childAt(aggregationInputChannels_[1]);
    const auto prefix = CudfConfig::getInstance().functionNamePrefix;
    const auto& aggregate = aggregationNode_->aggregates()[0];
    denseIntegerCountRows_ = FLAGS_cudf_dense_integer_count_rows &&
        aggregate.call->name() == prefix + "count" &&
        getCountInputKind(aggregate, aggregationInput.constants[0]) ==
            CountInputKind::kCountAll;
    const bool sum = !aggregationInput.constants[0] && valueType->isBigint() &&
        aggregate.call->name() == prefix + "sum";
    denseIntegerSumEligible_ = (keyType->isInteger() || keyType->isBigint()) &&
        (sum || denseIntegerCountRows_);
    denseIntegerSumValueChannel_ = aggregationInputChannels_[1];
  }

  // Check that aggregate result type match the output type.
  // TODO: This is output schema validation. In velox CPU, it's done using
  // output types reported by aggregation functions. We can't do that in cudf
  // groupby.

  // TODO: Set identity projections used by HashProbe to pushdown dynamic
  // filters to table scan.

  // TODO: Add support for grouping sets and group ids.

  aggregationNode_.reset();
}

void CudfGroupby::computePartialGroupbyIncrementally(CudfVectorPtr tbl) {
  // For every input, we'll do a groupby and compact results with the existing
  // intermediate groupby results.

  auto inputTableStream = tbl->stream();
  // Use getTableView() to avoid expensive materialization for packed_table.
  // tbl stays alive during this function call, keeping the view valid.
  auto permutedInputView = tbl->getTableView().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  auto groupbyOnInput = doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      aggregators_,
      bufferedResultType_,
      inputTableStream,
      get_output_mr());

  if (FLAGS_cudf_partial_groupby_merge_min_rows > 0) {
    if (!groupbyOnInput) {
      return;
    }
    pendingPartialRows_ += groupbyOnInput->size();
    pendingPartialBytes_ += groupbyOnInput->estimateFlatSize();
    pendingPartialResults_.push_back(std::move(groupbyOnInput));
    {
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat("deferredPartialBatches", RuntimeCounter(1));
    }
    const auto bufferedBytes =
        bufferedResult_ ? bufferedResult_->estimateFlatSize() : 0;
    // Bound per-batch allocation/ownership overhead even for one-row partials.
    if (pendingPartialResults_.size() >= 64 ||
        pendingPartialRows_ >= FLAGS_cudf_partial_groupby_merge_min_rows ||
        pendingPartialBytes_ + bufferedBytes >=
            maxPartialAggregationMemoryUsage_) {
      flushPendingPartialResults();
    }
    return;
  }

  // If we already have partial output, concatenate the new results with it.
  if (bufferedResult_) {
    auto partialOutputStream = bufferedResult_->stream();
    std::vector<CudfVectorPtr> tablesToConcat;
    tablesToConcat.push_back(bufferedResult_);
    tablesToConcat.push_back(groupbyOnInput);
    auto concatenatedTable = getConcatenatedTable(
        std::move(tablesToConcat),
        bufferedResultType_,
        partialOutputStream,
        get_temp_mr());

    // Now we have to groupby again but this time with intermediate aggregators.
    // Keep concatenatedTable alive while we use its view.
    auto compactedOutput = doGroupByAggregation(
        concatenatedTable->view(),
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        partialOutputStream,
        get_output_mr());
    bufferedResult_ = compactedOutput;
  } else {
    // First time processing, just store the result of the input batch's groupby
    // This means we're storing the stream from the first batch.
    bufferedResult_ = groupbyOnInput;
  }
}

void CudfGroupby::flushPendingPartialResults() {
  if (pendingPartialResults_.empty()) {
    return;
  }
  if (bufferedResult_) {
    pendingPartialResults_.push_back(std::move(bufferedResult_));
  }
  const auto batches = pendingPartialResults_.size();
  if (batches == 1) {
    bufferedResult_ = std::move(pendingPartialResults_.front());
    pendingPartialResults_.clear();
  } else {
    const auto stream = pendingPartialResults_.front()->stream();
    auto concatenated = getConcatenatedTable(
        std::exchange(pendingPartialResults_, {}),
        bufferedResultType_,
        stream,
        get_temp_mr());
    bufferedResult_ = doGroupByAggregation(
        concatenated->view(),
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        stream,
        get_output_mr());
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat("partialMergeFlushes", RuntimeCounter(1));
    lockedStats->addRuntimeStat("partialMergeBatches", RuntimeCounter(batches));
    lockedStats->addRuntimeStat(
        "partialMergeRows", RuntimeCounter(concatenated->num_rows()));
  }
  pendingPartialRows_ = 0;
  pendingPartialBytes_ = 0;
}

void CudfGroupby::computeFinalGroupbyIncrementally(CudfVectorPtr tbl) {
  auto inputTableStream = tbl->stream();
  auto permutedInputView = tbl->getTableView().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());

  if (!bufferedResult_) {
    auto groupbyOnInput = doGroupByAggregation(
        permutedInputView,
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        inputTableStream,
        get_output_mr());
    if (!groupbyOnInput) {
      return;
    }
    bufferedResult_ = groupbyOnInput;
    return;
  }

  std::vector<cudf::table_view> tablesToConcat;
  tablesToConcat.push_back(bufferedResult_->getTableView());
  tablesToConcat.push_back(permutedInputView);

  auto finalStream = bufferedResult_->stream();
  cudf::detail::join_streams(
      std::vector<rmm::cuda_stream_view>{inputTableStream}, finalStream);

  auto concatenatedTable =
      cudf::concatenate(tablesToConcat, finalStream, get_temp_mr());
  cudf::detail::join_streams(
      std::vector<rmm::cuda_stream_view>{finalStream}, inputTableStream);
  auto compactedOutput = doGroupByAggregation(
      concatenatedTable->view(),
      groupingKeyOutputChannels_,
      intermediateAggregators_,
      bufferedResultType_,
      finalStream,
      get_output_mr());
  bufferedResult_ = compactedOutput;
}

void CudfGroupby::computeSingleGroupbyIncrementally(CudfVectorPtr tbl) {
  auto inputTableStream = tbl->stream();
  auto permutedInputView = tbl->getTableView().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  auto groupbyOnInput = doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      partialAggregators_,
      bufferedResultType_,
      inputTableStream,
      get_output_mr());

  if (bufferedResult_) {
    auto partialOutputStream = bufferedResult_->stream();
    std::vector<CudfVectorPtr> tablesToConcat;
    tablesToConcat.push_back(bufferedResult_);
    tablesToConcat.push_back(groupbyOnInput);
    auto concatenatedTable = getConcatenatedTable(
        std::move(tablesToConcat),
        bufferedResultType_,
        partialOutputStream,
        get_temp_mr());

    auto compactedOutput = doGroupByAggregation(
        concatenatedTable->view(),
        groupingKeyOutputChannels_,
        intermediateAggregators_,
        bufferedResultType_,
        partialOutputStream,
        get_output_mr());
    bufferedResult_ = compactedOutput;
  } else {
    bufferedResult_ = groupbyOnInput;
  }
}

bool CudfGroupby::tryAddDenseIntegerSum(CudfVectorPtr input) {
  if (!denseIntegerSumEligible_) {
    return false;
  }
  if (!denseIntegerSum_ &&
      input->size() < FLAGS_cudf_dense_integer_sum_min_rows) {
    denseIntegerSumEligible_ = false;
    return false;
  }
  const auto inputStream = input->stream();
  const auto table = input->getTableView();
  if (!denseIntegerSum_) {
    denseIntegerSum_ = std::make_unique<DenseIntegerSum>(
        table.column(groupingKeyInputChannels_[0]).type(),
        FLAGS_cudf_dense_integer_sum_max_range,
        ignoreNullKeys_,
        inputStream,
        get_output_mr(),
        denseIntegerCountRows_,
        FLAGS_cudf_dense_integer_count_32_max_rows);
  }
  const auto stream = denseIntegerSum_->stream();
  const bool joinStreams = stream.value() != inputStream.value();
  if (joinStreams) {
    cudf::detail::join_streams(
        std::vector<rmm::cuda_stream_view>{inputStream}, stream);
  }
  bool accepted;
  try {
    accepted = denseIntegerSum_->add(
        table.column(groupingKeyInputChannels_[0]),
        table.column(denseIntegerSumValueChannel_));
  } catch (...) {
    if (joinStreams) {
      cudf::detail::join_streams(
          std::vector<rmm::cuda_stream_view>{stream}, inputStream);
    }
    throw;
  }
  if (joinStreams) {
    cudf::detail::join_streams(
        std::vector<rmm::cuda_stream_view>{stream}, inputStream);
  }
  if (accepted) {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat("denseIntegerSumBatches", RuntimeCounter(1));
    if (denseIntegerCountRows_) {
      lockedStats->addRuntimeStat(
          "denseIntegerCountBatches", RuntimeCounter(1));
      if (FLAGS_cudf_dense_integer_count_32_max_rows != 0) {
        lockedStats->addRuntimeStat(
            "denseIntegerCount32Batches", RuntimeCounter(1));
      }
    }
    lockedStats->addRuntimeStat(
        "denseIntegerSumRows", RuntimeCounter(input->size()));
    lockedStats->addRuntimeStat(
        "denseIntegerSumStateBytes", RuntimeCounter(denseIntegerSum_->bytes()));
    return true;
  }
  // Existing sums are sufficient partial states. Materialize them once, then
  // merge this and later raw batches through the ordinary single-step path.
  // No original input needs to be retained or replayed.
  auto partial = denseIntegerSum_->finalize();
  if (partial->num_rows() > 0) {
    const auto rows = partial->num_rows();
    bufferedResult_ = std::make_shared<CudfVector>(
        pool(), bufferedResultType_, rows, std::move(partial), stream);
  }
  denseIntegerSum_.reset();
  denseIntegerSumEligible_ = false;
  streamingGroupbyEnabled_ = false;
  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat("denseIntegerSumFallbacks", RuntimeCounter(1));
  return false;
}

bool CudfGroupby::tryBufferUniqueFinalBatch(const CudfVectorPtr& input) {
  const auto bytes = input->estimateFlatSize();
  const auto limit = FLAGS_cudf_final_groupby_unique_max_bytes;
  if ((uniqueFinalBatches_.empty() &&
       input->size() < FLAGS_cudf_final_groupby_unique_min_rows) ||
      uniqueFinalBytes_ > limit || bytes > limit - uniqueFinalBytes_) {
    return false;
  }
  const auto stream = input->stream();
  const auto key = input->getTableView().column(groupingKeyInputChannels_[0]);
  // Null keys and any uncertain range overlap take the ordinary path.
  if (key.null_count() != 0) {
    return false;
  }
  auto [minimum, maximum] = cudf::minmax(key, stream, get_temp_mr());
  const auto value =
      [stream](const std::unique_ptr<cudf::scalar>& scalar) -> int64_t {
    if (scalar->type().id() == cudf::type_id::INT32) {
      return static_cast<const cudf::numeric_scalar<int32_t>*>(scalar.get())
          ->value(stream);
    }
    return static_cast<const cudf::numeric_scalar<int64_t>*>(scalar.get())
        ->value(stream);
  };
  const auto low = value(minimum);
  const auto high = value(maximum);
  const auto next = uniqueFinalRanges_.lower_bound(low);
  if ((next != uniqueFinalRanges_.end() && high >= next->first) ||
      (next != uniqueFinalRanges_.begin() && std::prev(next)->second >= low)) {
    return false;
  }
  // Hash exchanges need not preserve order. Count adjacent groups only after
  // verifying ordering; otherwise prove exact distinctness with a temporary
  // per-batch set instead of retaining a hash state for the entire operator.
  const bool sorted = cudf::is_sorted(cudf::table_view{{key}}, {}, {}, stream);
  const auto distinct = sorted ? cudf::unique_count(
                                     key,
                                     cudf::null_policy::INCLUDE,
                                     cudf::nan_policy::NAN_IS_VALID,
                                     stream)
                               : cudf::distinct_count(
                                     key,
                                     cudf::null_policy::INCLUDE,
                                     cudf::nan_policy::NAN_IS_VALID,
                                     stream);
  if (distinct != key.size()) {
    return false;
  }
  uniqueFinalRanges_.emplace(low, high);
  uniqueFinalBatches_.push_back(input);
  uniqueFinalBytes_ += bytes;
  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      "uniqueFinalGroupbyBufferedBatches", RuntimeCounter(1));
  lockedStats->addRuntimeStat(
      "uniqueFinalGroupbyBufferedBytes",
      RuntimeCounter(bytes, RuntimeCounter::Unit::kBytes));
  lockedStats->addRuntimeStat(
      "uniqueFinalGroupbyUnsortedBatches", RuntimeCounter(sorted ? 0 : 1));
  return true;
}

void CudfGroupby::abandonUniqueFinalBatches() {
  uniqueFinalBatchesEligible_ = false;
  uniqueFinalRanges_.clear();
  uniqueFinalBytes_ = 0;
  while (!uniqueFinalBatches_.empty()) {
    auto input = std::move(uniqueFinalBatches_.front());
    uniqueFinalBatches_.pop_front();
    if (streamingGroupbyEnabled_) {
      computeFinalGroupbyStreaming(std::move(input));
    } else {
      computeFinalGroupbyIncrementally(std::move(input));
    }
  }
  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat("uniqueFinalGroupbyFallbacks", RuntimeCounter(1));
}

void CudfGroupby::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  numInputRows_ += input->size();

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);

  if (uniqueFinalBatchesEligible_) {
    if (tryBufferUniqueFinalBatch(cudfInput)) {
      return;
    }
    abandonUniqueFinalBatches();
  }

  if (tryAddDenseIntegerSum(cudfInput)) {
    return;
  }

  if (disjointGroupRanges_) {
    VELOX_CHECK_NULL(disjointBatchOutput_);
    const auto stream = cudfInput->stream();
    const auto view = cudfInput->getTableView();
    const auto key = view.column(groupingKeyInputChannels_[0]);
    VELOX_USER_CHECK_EQ(
        key.null_count(),
        0,
        "Complete-batch GPU groupby requires non-null keys");
    VELOX_USER_CHECK(
        cudf::is_sorted(cudf::table_view{{key}}, {}, {}, stream),
        "Complete-batch GPU groupby requires sorted integer keys within each batch");
    auto [minimum, maximum] = cudf::minmax(key, stream, get_temp_mr());
    auto value =
        [stream](const std::unique_ptr<cudf::scalar>& scalar) -> int64_t {
      if (scalar->type().id() == cudf::type_id::INT32) {
        return static_cast<const cudf::numeric_scalar<int32_t>*>(scalar.get())
            ->value(stream);
      }
      return static_cast<const cudf::numeric_scalar<int64_t>*>(scalar.get())
          ->value(stream);
    };
    disjointGroupRanges_->add(value(minimum), value(maximum));
    disjointBatchOutput_ = doGroupByAggregation(
        view.select(
            aggregationInputChannels_.begin(), aggregationInputChannels_.end()),
        groupingKeyOutputChannels_,
        aggregators_,
        outputType_,
        stream,
        get_output_mr());
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat("completeGroupBatches", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "completeGroupInputRows", RuntimeCounter(input->size()));
    return;
  }

  if (streamingGroupbyEnabled_) {
    computeFinalGroupbyStreaming(std::move(cudfInput));
    return;
  }

  if (incrementalAggregationEnabled_) {
    if (isPartialOutput_) {
      computePartialGroupbyIncrementally(cudfInput);
      return;
    } else if (isSingleStep_) {
      computeSingleGroupbyIncrementally(cudfInput);
      return;
    } else {
      computeFinalGroupbyIncrementally(cudfInput);
      return;
    }
  }

  // Handle non-streaming cases.
  inputs_.push_back(std::move(cudfInput));
}

CudfVectorPtr CudfGroupby::doGroupByAggregation(
    cudf::table_view tableView,
    std::vector<column_index_t> const& groupByKeys,
    std::vector<std::unique_ptr<GroupbyAggregator>>& aggregators,
    TypePtr const& outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto groupbyKeyView =
      tableView.select(groupByKeys.begin(), groupByKeys.end());
  std::unique_ptr<cudf::column> packedStringKeys;
  if (FLAGS_cudf_groupby_pack_short_string_keys &&
      tableView.num_rows() >= FLAGS_cudf_groupby_pack_short_string_min_rows) {
    packedStringKeys =
        tryPackShortStringKeys(groupbyKeyView, stream, get_temp_mr());
    if (packedStringKeys) {
      groupbyKeyView = cudf::table_view{{packedStringKeys->view()}};
      auto lockedStats = stats_.wlock();
      lockedStats->addRuntimeStat(
          "packedStringGroupBatches", RuntimeCounter(1));
      lockedStats->addRuntimeStat(
          "packedStringGroupRows", RuntimeCounter(tableView.num_rows()));
    }
  }

  bool sortedKeys = disjointGroupRanges_ != nullptr;
  if (!sortedKeys && FLAGS_cudf_groupby_detect_sorted_keys &&
      tableView.num_rows() >= FLAGS_cudf_groupby_detect_sorted_min_rows &&
      groupbyKeyView.num_columns() == 1 &&
      (groupbyKeyView.column(0).type().id() == cudf::type_id::INT32 ||
       groupbyKeyView.column(0).type().id() == cudf::type_id::INT64)) {
    // Verify every batch, including intermediate compaction: independently
    // sorted input splits do not imply their concatenation is sorted.
    sortedKeys = cudf::is_sorted(groupbyKeyView, {}, {}, stream);
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat("sortedGroupbyChecks", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "sortedGroupbyBatches", RuntimeCounter(sortedKeys ? 1 : 0));
    lockedStats->addRuntimeStat(
        "sortedGroupbyRows",
        RuntimeCounter(sortedKeys ? tableView.num_rows() : 0));
  }
  cudf::groupby::groupby groupByOwner(
      groupbyKeyView,
      ignoreNullKeys_ ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE,
      sortedKeys ? cudf::sorted::YES : cudf::sorted::NO);

  std::vector<cudf::groupby::aggregation_request> requests;
  for (auto& aggregator : aggregators) {
    aggregator->addGroupbyRequest(tableView, requests, stream, get_temp_mr());
  }

  if (FLAGS_cudf_groupby_share_nonnull_counts) {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "groupbyRequestedAggregates", RuntimeCounter(aggregators.size()));
    lockedStats->addRuntimeStat(
        "groupbySubmittedRequests", RuntimeCounter(requests.size()));
  }

  auto [groupKeys, results] = groupByOwner.aggregate(requests, stream, mr);
  // flatten the results
  std::vector<std::unique_ptr<cudf::column>> resultColumns;

  // first fill the grouping keys
  auto groupKeysColumns = groupKeys->release();
  if (packedStringKeys) {
    auto decodedKeys = unpackShortStringKeys(
        groupKeysColumns[0]->view(), groupByKeys.size(), stream, mr);
    groupKeysColumns = std::move(decodedKeys);
  }
  resultColumns.insert(
      resultColumns.begin(),
      std::make_move_iterator(groupKeysColumns.begin()),
      std::make_move_iterator(groupKeysColumns.end()));

  // then fill the aggregation results
  for (auto& aggregator : aggregators) {
    resultColumns.push_back(aggregator->makeOutputColumn(results, stream, mr));
  }

  // make a cudf table out of columns
  auto resultTable = std::make_unique<cudf::table>(std::move(resultColumns));

  auto numRows = resultTable->num_rows();

  // velox expects nullptr instead of a table with 0 rows
  if (numRows == 0) {
    return nullptr;
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(), outputType, numRows, std::move(resultTable), stream);
}

CudfVectorPtr CudfGroupby::releaseAndResetBufferedResult() {
  auto numOutputRows = bufferedResult_->size();
  const double aggregationPct =
      numOutputRows == 0 ? 0 : (numOutputRows * 1.0) / numInputRows_ * 100;
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kFlushRowCount),
        RuntimeCounter(numOutputRows));
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kFlushTimes), RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        std::string(exec::HashAggregation::kPartialAggregationPct),
        RuntimeCounter(aggregationPct));
  }

  numInputRows_ = 0;
  // We're moving bufferedResult_ to the caller because we want it to be null
  // after this call.
  return std::move(bufferedResult_);
}

RowVectorPtr CudfGroupby::doGetOutput() {
  if (disjointGroupRanges_) {
    if (noMoreInput_) {
      finished_ = true;
    }
    return std::exchange(disjointBatchOutput_, nullptr);
  }
  // Handle partial streaming groupby.
  if (isPartialOutput_ && incrementalAggregationEnabled_) {
    if (noMoreInput_) {
      flushPendingPartialResults();
    }
    if (bufferedResult_ &&
        bufferedResult_->estimateFlatSize() >
            maxPartialAggregationMemoryUsage_) {
      return releaseAndResetBufferedResult();
    }
    if (not noMoreInput_) {
      // Don't produce output if the partial output hasn't reached memory limit
      // and there's more batches to come.
      return nullptr;
    }
    if (!bufferedResult_ && finished_) {
      return nullptr;
    }
    return releaseAndResetBufferedResult();
  }

  if (finished_) {
    return nullptr;
  }

  if (!isPartialOutput_ && !noMoreInput_) {
    // Final aggregation has to wait for all batches to arrive so we cannot
    // return any results here.
    return nullptr;
  }

  if (denseIntegerSum_) {
    finished_ = true;
    const auto stream = denseIntegerSum_->stream();
    auto table = denseIntegerSum_->finalize();
    denseIntegerSum_.reset();
    const auto rows = table->num_rows();
    return rows == 0 ? nullptr
                     : std::make_shared<CudfVector>(
                           pool(), outputType_, rows, std::move(table), stream);
  }
  if (uniqueFinalBatchesEligible_) {
    if (uniqueFinalBatches_.empty()) {
      finished_ = true;
      return nullptr;
    }
    auto input = std::move(uniqueFinalBatches_.front());
    uniqueFinalBatches_.pop_front();
    finished_ = uniqueFinalBatches_.empty();
    const auto view = input->getTableView().select(
        aggregationInputChannels_.begin(), aggregationInputChannels_.end());
    auto result = std::make_shared<CudfVector>(
        pool(),
        outputType_,
        input->size(),
        view,
        input,
        input->estimateFlatSize(),
        input->stream(),
        get_output_mr());
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat("uniqueFinalGroupbyBatches", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "uniqueFinalGroupbyRows", RuntimeCounter(input->size()));
    return result;
  }
  if (streamingGroupbyEnabled_) {
    finished_ = true;
    return finalizeStreamingGroupby();
  }

  // Streaming finalization: single step uses finalAggregators_ to convert
  // intermediate results to final output; final step uses aggregators_.
  // At this point isPartialOutput_ is false (handled above) and noMoreInput_
  // is true (guarded by the check above).
  if (incrementalAggregationEnabled_) {
    finished_ = true;
    if (!bufferedResult_) {
      return nullptr;
    }
    auto& aggs = isSingleStep_ ? finalAggregators_ : aggregators_;
    auto stream = bufferedResult_->stream();
    auto result = doGroupByAggregation(
        bufferedResult_->getTableView(),
        groupingKeyOutputChannels_,
        aggs,
        outputType_,
        stream,
        get_output_mr());
    stream.synchronize();
    bufferedResult_.reset();
    return result;
  }

  if (inputs_.empty() && !noMoreInput_) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();

  auto tbl = getConcatenatedTable(
      std::exchange(inputs_, {}), inputType_, stream, get_temp_mr());

  // Release input data after synchronizing.
  stream.synchronize();
  inputs_.clear();

  if (noMoreInput_) {
    finished_ = true;
  }

  VELOX_CHECK_NOT_NULL(tbl);

  auto permutedInputView = tbl->view().select(
      aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  return doGroupByAggregation(
      permutedInputView,
      groupingKeyOutputChannels_,
      aggregators_,
      outputType_,
      stream,
      get_output_mr());
}

void CudfGroupby::doNoMoreInput() {
  Operator::noMoreInput();
  if (isPartialOutput_ && inputs_.empty()) {
    finished_ = true;
  }
}

void CudfGroupby::doClose() {
  uniqueFinalBatches_.clear();
  uniqueFinalRanges_.clear();
  uniqueFinalBytes_ = 0;
  uniqueFinalBatchesEligible_ = false;
  denseIntegerSum_.reset();
  denseIntegerSumEligible_ = false;
  if (streamingGroupby_ && streamingGroupbyStream_.has_value()) {
    // Match rebuild and finalization: wait before dropping persistent state
    // that an asynchronous aggregate or merge may still reference.
    streamingGroupbyStream_->synchronize();
  }
  streamingGroupby_.reset();
  streamingGroupbyEvent_.reset();
  streamingGroupbyStream_.reset();
  streamingGroupbyCapacity_ = 0;
  streamingGroupbyAggregators_.clear();
  disjointBatchOutput_.reset();
  disjointGroupRanges_.reset();
  inputs_.clear();
  bufferedResult_.reset();
  pendingPartialResults_.clear();
  pendingPartialRows_ = 0;
  pendingPartialBytes_ = 0;
  aggregators_.clear();
  intermediateAggregators_.clear();
  partialAggregators_.clear();
  finalAggregators_.clear();
  Operator::close();
}

bool CudfGroupby::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
