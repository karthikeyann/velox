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
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Validation.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/expression/ExprOptimizer.h"

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <gflags/gflags.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_map>

DEFINE_bool(
    cudf_fuse_double_projections,
    false,
    "Fuse non-null DOUBLE plus/minus/multiply projections into one GPU kernel");
DEFINE_int64(
    cudf_fuse_double_projections_min_rows,
    100'000,
    "Minimum rows for fused DOUBLE projection");
DEFINE_bool(
    cudf_project_borrowed_views,
    false,
    "Forward immutable borrowed input views through projections without copying unrelated columns");

namespace facebook::velox::cudf_velox {

namespace {

std::string makeFusedDoubleProjection(
    const std::vector<core::TypedExprPtr>& expressions,
    const RowTypePtr& inputType,
    std::vector<column_index_t>& inputs) {
  if (expressions.size() < 2) {
    return {};
  }
  // Deliberately narrow eligibility: no integer overflow, division errors,
  // casts, functions, or null propagation are reimplemented here.
  std::function<std::optional<std::string>(const core::TypedExprPtr&)> emit;
  emit = [&](const core::TypedExprPtr& expr) -> std::optional<std::string> {
    if (!expr->type()->isDouble()) {
      return std::nullopt;
    }
    if (auto field = core::TypedExprs::asFieldAccess(expr)) {
      if (!field->inputs().empty() &&
          (field->inputs().size() != 1 ||
           !dynamic_cast<const core::InputTypedExpr*>(
               field->inputs()[0].get()))) {
        return std::nullopt;
      }
      const auto channel = inputType->getChildIdx(field->name());
      auto found = std::find(inputs.begin(), inputs.end(), channel);
      if (found == inputs.end()) {
        inputs.push_back(channel);
        return fmt::format("in{}", inputs.size() - 1);
      }
      return fmt::format("in{}", found - inputs.begin());
    }
    if (auto constant = core::TypedExprs::asConstant(expr)) {
      if (constant->isNull()) {
        return std::nullopt;
      }
      const auto value = constant->hasValueVector()
          ? constant->valueVector()->as<SimpleVector<double>>()->valueAt(0)
          : constant->value().value<TypeKind::DOUBLE>();
      if (!std::isfinite(value)) {
        return std::nullopt;
      }
      // Scientific notation guarantees a DOUBLE literal, including -0.0.
      return fmt::format("({:.17e})", value);
    }
    auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expr);
    if (!call || call->inputs().size() != 2) {
      return std::nullopt;
    }
    std::string op;
    if (call->name() == "plus" || call->name() == "add") {
      op = "+";
    } else if (call->name() == "minus" || call->name() == "subtract") {
      op = "-";
    } else if (call->name() == "multiply") {
      op = "*";
    } else {
      return std::nullopt;
    }
    auto left = emit(call->inputs()[0]);
    auto right = emit(call->inputs()[1]);
    if (!left || !right) {
      return std::nullopt;
    }
    return fmt::format("({} {} {})", *left, op, *right);
  };
  std::vector<std::string> outputs;
  for (const auto& expr : expressions) {
    auto output = emit(expr);
    if (!output) {
      inputs.clear();
      return {};
    }
    outputs.push_back(std::move(*output));
  }
  if (inputs.empty()) {
    return {};
  }
  std::string udf = "__device__ void fused_projection(";
  for (size_t i = 0; i < outputs.size(); ++i) {
    udf += fmt::format("{}double* out{}", i == 0 ? "" : ", ", i);
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    udf += fmt::format(", double in{}", i);
  }
  udf += ") {\n";
  for (size_t i = 0; i < outputs.size(); ++i) {
    udf += fmt::format("  *out{} = {};\n", i, outputs[i]);
  }
  return udf + "}\n";
}

void debugPrintTree(
    const core::TypedExprPtr& expr,
    int indent = 0,
    std::ostream& os = std::cout) {
  if (indent == 0)
    os << "=== Expression Tree ===" << std::endl;
  os << std::string(indent, ' ') << core::ExprKindName::toName(expr->kind())
     << "(" << expr->type()->toString() << ")" << std::endl;
  for (auto& input : expr->inputs()) {
    debugPrintTree(input, indent + 2, os);
  }
}

bool checkAddIdentityProjection(
    const core::TypedExprPtr& projection,
    const RowTypePtr& inputType,
    column_index_t outputChannel,
    std::vector<exec::IdentityProjection>& identityProjections) {
  if (auto field = core::TypedExprs::asFieldAccess(projection)) {
    const auto& inputs = field->inputs();
    if (inputs.empty() ||
        (inputs.size() == 1 &&
         dynamic_cast<const core::InputTypedExpr*>(inputs[0].get()))) {
      const auto inputChannel = inputType->getChildIdx(field->name());
      identityProjections.emplace_back(inputChannel, outputChannel);
      return true;
    }
  }

  return false;
}

// Split stats to attrbitute cardinality reduction to the Filter node.
std::vector<exec::OperatorStats> splitStats(
    const exec::OperatorStats& combinedStats,
    const core::PlanNodeId& filterNodeId) {
  exec::OperatorStats filterStats;

  filterStats.operatorId = combinedStats.operatorId;
  filterStats.pipelineId = combinedStats.pipelineId;
  filterStats.planNodeId = filterNodeId;
  filterStats.operatorType = combinedStats.operatorType;
  filterStats.numDrivers = combinedStats.numDrivers;

  filterStats.inputBytes = combinedStats.inputBytes;
  filterStats.inputPositions = combinedStats.inputPositions;
  filterStats.inputVectors = combinedStats.inputVectors;

  // Estimate Filter's output bytes based on cardinality change.
  const double filterRate = combinedStats.inputPositions > 0
      ? (combinedStats.outputPositions * 1.0 / combinedStats.inputPositions)
      : 1.0;

  filterStats.outputBytes = (uint64_t)(filterStats.inputBytes * filterRate);
  filterStats.outputPositions = combinedStats.outputPositions;
  filterStats.outputVectors = combinedStats.outputVectors;

  auto projectStats = combinedStats;
  projectStats.inputBytes = filterStats.outputBytes;
  projectStats.inputPositions = filterStats.outputPositions;
  projectStats.inputVectors = filterStats.outputVectors;

  return {std::move(projectStats), std::move(filterStats)};
}

} // namespace

CudfFilterProject::CudfFilterProject(
    int32_t operatorId,
    velox::exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::FilterNode>& filter,
    const std::shared_ptr<const core::ProjectNode>& project)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          project ? project->outputType() : filter->outputType(),
          project ? project->id() : filter->id(),
          "CudfFilterProject",
          nvtx3::rgb{220, 20, 60}, // Crimson
          NvtxMethodFlag::kAll,
          std::nullopt,
          project ? std::static_pointer_cast<const core::PlanNode>(project)
                  : std::static_pointer_cast<const core::PlanNode>(filter)),
      hasFilter_(filter != nullptr),
      project_(project),
      filter_(filter) {
  if (filter_ != nullptr && project_ != nullptr) {
    folly::Synchronized<exec::OperatorStats>& opStats = Operator::stats();
    opStats.withWLock([&](auto& stats) {
      stats.setStatSplitter(
          [filterId = filter_->id()](const auto& combinedStats) {
            return splitStats(combinedStats, filterId);
          });
    });
  }
}

void CudfFilterProject::initialize() {
  Operator::initialize();

  std::vector<core::TypedExprPtr> allExprs;
  if (hasFilter_) {
    VELOX_CHECK_NOT_NULL(filter_);
    allExprs.push_back(filter_->filter());
  }

  if (project_) {
    const auto& inputType = project_->sources()[0]->outputType();

    for (column_index_t i = 0; i < project_->projections().size(); i++) {
      auto& projection = project_->projections()[i];
      bool identityProjection = checkAddIdentityProjection(
          projection, inputType, i, identityProjections_);
      if (!identityProjection) {
        allExprs.push_back(projection);
        resultProjections_.emplace_back(allExprs.size() - 1, i);
      }
    }
  } else {
    for (column_index_t i = 0; i < outputType_->size(); ++i) {
      identityProjections_.emplace_back(i, i);
    }
    isIdentityProjection_ = true;
  }

  auto lazyDereference =
      (dynamic_cast<const core::LazyDereferenceNode*>(project_.get()) !=
       nullptr);
  VELOX_CHECK(!(lazyDereference && filter_));

  const auto inputType = project_ ? project_->sources()[0]->outputType()
                                  : filter_->sources()[0]->outputType();

  // convert to AST
  if (CudfConfig::getInstance().debugEnabled) {
    int i = 0;
    for (const auto& expr : allExprs) {
      LOG(INFO) << "expr[" << i++ << "] " << expr->toString();
      debugPrintTree(expr, 0, LOG(INFO));
    }
  }
  // Optimize (rewrites + constant folding) each expression before evaluator
  // selection so CudfFunctions never see scalar-only operand sets, then
  // compile. The operator pool owns the folded constants for the evaluator's
  // lifetime.
  auto* const queryCtx = operatorCtx_->execCtx()->queryCtx();
  auto* const pool = operatorCtx_->pool();
  const auto optimizeAndCompile =
      [inputType, queryCtx, pool](const core::TypedExprPtr& expr) {
        return createCudfExpression(
            expression::optimize(expr, queryCtx, pool), inputType, pool);
      };
  if (hasFilter_) {
    // First expr is Filter, rest are Project.
    filterEvaluator_ = optimizeAndCompile(allExprs.front());
    std::transform(
        allExprs.begin() + 1,
        allExprs.end(),
        std::back_inserter(projectEvaluators_),
        optimizeAndCompile);
  } else {
    std::transform(
        allExprs.begin(),
        allExprs.end(),
        std::back_inserter(projectEvaluators_),
        optimizeAndCompile);
  }

  if (FLAGS_cudf_fuse_double_projections) {
    std::vector<core::TypedExprPtr> projections;
    for (size_t i = hasFilter_ ? 1 : 0; i < allExprs.size(); ++i) {
      projections.push_back(expression::optimize(allExprs[i], queryCtx, pool));
    }
    fusedDoubleProjection_ =
        makeFusedDoubleProjection(projections, inputType, fusedDoubleInputs_);
  }

  filter_.reset();
  project_.reset();
}

void CudfFilterProject::doAddInput(RowVectorPtr input) {
  input_ = std::move(input);
}

RowVectorPtr CudfFilterProject::doGetOutput() {
  if (allInputProcessed()) {
    return nullptr;
  }
  if (input_->size() == 0) {
    input_.reset();
    return nullptr;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto stream = cudfInput->stream();
  if (FLAGS_cudf_project_borrowed_views && !hasFilter_ &&
      cudfInput->hasBorrowedStorage() && outputType_->size() > 0) {
    struct ProjectionStorage {
      RowVectorPtr input;
      std::vector<ColumnOrView> computed;
    };
    const auto table = cudfInput->getTableView();
    std::vector<cudf::column_view> inputs(table.begin(), table.end());
    auto owner = std::make_shared<ProjectionStorage>();
    owner->input = std::move(input_);
    owner->computed = evaluateProjections(inputs, stream);
    std::vector<cudf::column_view> views(outputType_->size());
    auto retainedBytes = cudfInput->estimateFlatSize();
    for (size_t i = 0; i < resultProjections_.size(); ++i) {
      auto& value = owner->computed[i];
      views[resultProjections_[i].outputChannel] = asView(value);
      if (std::holds_alternative<std::unique_ptr<cudf::column>>(value)) {
        retainedBytes +=
            std::get<std::unique_ptr<cudf::column>>(value)->alloc_size();
      }
    }
    for (const auto& identity : identityProjections_) {
      views[identity.outputChannel] = inputs[identity.inputChannel];
    }
    auto output = std::make_shared<CudfVector>(
        cudfInput->pool(),
        outputType_,
        cudfInput->size(),
        cudf::table_view(views),
        std::move(owner),
        retainedBytes,
        stream,
        get_output_mr());
    addRuntimeStat("borrowedProjectionBatches", RuntimeCounter(1));
    return output;
  }
  auto inputTableColumns = cudfInput->release()->release();
  auto outputSize = input_->size();

  if (hasFilter_) {
    filter(inputTableColumns, stream);
  }
  if (!inputTableColumns.empty()) {
    outputSize = inputTableColumns.front()->size();
  }
  auto outputColumns = project(inputTableColumns, stream);

  auto outputTable = std::make_unique<cudf::table>(std::move(outputColumns));
  auto const numColumns = outputTable->num_columns();
  auto const size = numColumns > 0 ? outputTable->num_rows() : outputSize;
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "cudfProject Output: " << size << " rows, " << numColumns
            << " columns";
  }
  if (size == 0) {
    input_.reset();
    return nullptr;
  }
  auto cudfOutput = std::make_shared<CudfVector>(
      input_->pool(), outputType_, size, std::move(outputTable), stream);
  input_.reset();
  return cudfOutput;
}

void CudfFilterProject::filter(
    std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
    rmm::cuda_stream_view stream) {
  // Evaluate the Filter
  std::vector<cudf::column_view> inputViews;
  inputViews.reserve(inputTableColumns.size());
  for (auto& col : inputTableColumns) {
    inputViews.push_back(col->view());
  }
  auto filterColumn =
      filterEvaluator_->eval(inputViews, stream, get_temp_mr(), true);
  auto filterColumnView = asView(filterColumn);
  bool shouldApplyFilter = [&]() {
    if (filterColumnView.has_nulls()) {
      return true;
    }
    // check if all values in filterColumnView are true
    auto isAllTrue = cudf::reduce(
        filterColumnView,
        *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
        cudf::data_type(cudf::type_id::BOOL8),
        stream,
        get_temp_mr());
    using ScalarType = cudf::scalar_type_t<bool>;
    auto result = static_cast<ScalarType*>(isAllTrue.get());
    // If filter is not all true, apply the filter
    return !(result->is_valid(stream) && result->value(stream));
  }();
  if (shouldApplyFilter) {
    auto filterTable =
        std::make_unique<cudf::table>(std::move(inputTableColumns));
    auto filteredTable = cudf::apply_boolean_mask(
        *filterTable, filterColumnView, stream, get_output_mr());
    inputTableColumns = filteredTable->release();
  }
}

std::vector<ColumnOrView> CudfFilterProject::evaluateProjections(
    const std::vector<cudf::column_view>& inputViews,
    rmm::cuda_stream_view stream) {
  std::vector<ColumnOrView> columns;
  bool fuse = !fusedDoubleProjection_.empty() && !inputViews.empty() &&
      inputViews[0].size() >= FLAGS_cudf_fuse_double_projections_min_rows &&
      std::all_of(
          fusedDoubleInputs_.begin(),
          fusedDoubleInputs_.end(),
          [&](auto index) { return inputViews[index].null_count() == 0; });
  if (fuse) {
    std::vector<cudf::transform_input> inputs;
    for (auto index : fusedDoubleInputs_) {
      inputs.emplace_back(inputViews[index]);
    }
    std::vector<cudf::transform_output> outputs(
        projectEvaluators_.size(),
        {cudf::data_type{cudf::type_id::FLOAT64},
         cudf::output_nullability::ALL_VALID});
    auto result = cudf::multi_transform(
        fusedDoubleProjection_,
        cudf::udf_source_type::CUDA,
        cudf::null_aware::NO,
        std::nullopt,
        inputs,
        outputs,
        {},
        inputViews[0].size(),
        stream,
        get_output_mr());
    for (auto& column : result->release()) {
      columns.emplace_back(std::move(column));
    }
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "fusedDoubleProjectionBatches", RuntimeCounter(1));
    lockedStats->addRuntimeStat(
        "fusedDoubleProjectionRows", RuntimeCounter(inputViews[0].size()));
    lockedStats->addRuntimeStat(
        "fusedDoubleProjectionOutputs", RuntimeCounter(outputs.size()));
  } else {
    for (auto& projectEvaluator : projectEvaluators_) {
      columns.push_back(
          projectEvaluator->eval(inputViews, stream, get_output_mr(), true));
    }
  }

  return columns;
}

std::vector<std::unique_ptr<cudf::column>> CudfFilterProject::project(
    std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
    rmm::cuda_stream_view stream) {
  std::vector<cudf::column_view> inputViews;
  inputViews.reserve(inputTableColumns.size());
  for (const auto& column : inputTableColumns) {
    inputViews.push_back(column->view());
  }
  auto columns = evaluateProjections(inputViews, stream);

  // Rearrange columns to match outputType_
  std::vector<std::unique_ptr<cudf::column>> outputColumns(outputType_->size());
  // computed resultProjections
  for (int i = 0; i < resultProjections_.size(); i++) {
    auto& columnOrView = columns[i];
    if (std::holds_alternative<std::unique_ptr<cudf::column>>(columnOrView)) {
      // Move the owned column
      outputColumns[resultProjections_[i].outputChannel] =
          std::move(std::get<std::unique_ptr<cudf::column>>(columnOrView));
    } else {
      // Materialize the column_view into an owned column
      auto view = std::get<cudf::column_view>(columnOrView);
      outputColumns[resultProjections_[i].outputChannel] =
          std::make_unique<cudf::column>(view, stream, get_output_mr());
    }
  }

  // Count occurrences of each inputChannel, and move columns if they occur only
  // once
  std::unordered_map<column_index_t, int> inputChannelCount;
  for (const auto& identity : identityProjections_) {
    inputChannelCount[identity.inputChannel]++;
  }

  // identityProjections (input to output copy)
  for (auto const& identity : identityProjections_) {
    VELOX_CHECK_NOT_NULL(inputTableColumns[identity.inputChannel]);
    if (inputChannelCount[identity.inputChannel] == 1) {
      // Move the column if it occurs only once
      outputColumns[identity.outputChannel] =
          std::move(inputTableColumns[identity.inputChannel]);
    } else {
      // Otherwise, copy the column and decrement the count
      outputColumns[identity.outputChannel] = std::make_unique<cudf::column>(
          *inputTableColumns[identity.inputChannel], stream, get_output_mr());
    }
    VELOX_CHECK_GT(inputChannelCount[identity.inputChannel], 0);
    inputChannelCount[identity.inputChannel]--;
  }

  return outputColumns;
}

bool CudfFilterProject::allInputProcessed() {
  return !input_;
}

bool CudfFilterProject::isFinished() {
  return noMoreInput_ && allInputProcessed();
}

} // namespace facebook::velox::cudf_velox
