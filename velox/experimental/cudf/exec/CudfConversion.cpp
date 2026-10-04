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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/QueryConfig.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace facebook::velox::cudf_velox {

namespace {
// Concatenate multiple RowVectors into a single RowVector.
// Copied from AggregationFuzzer.cpp.
RowVectorPtr mergeRowVectors(
    const std::vector<RowVectorPtr>& results,
    velox::memory::MemoryPool* pool) {
  VELOX_NVTX_FUNC_RANGE();
  if (results.size() == 1) {
    return results[0];
  }
  vector_size_t totalCount = 0;
  for (const auto& result : results) {
    totalCount += result->size();
  }
  auto copy =
      BaseVector::create<RowVector>(results[0]->type(), totalCount, pool);
  auto copyCount = 0;
  for (const auto& result : results) {
    copy->copy(result.get(), copyCount, 0, result->size());
    copyCount += result->size();
  }
  return copy;
}

cudf::size_type preferredGpuBatchSizeRows(
    const facebook::velox::core::QueryConfig& queryConfig) {
  constexpr cudf::size_type kDefaultGpuBatchSizeRows = 100000;
  const auto batchSize = queryConfig.get<int32_t>(
      CudfFromVelox::kGpuBatchSizeRows, kDefaultGpuBatchSizeRows);
  VELOX_CHECK_GT(batchSize, 0, "velox.cudf.gpu_batch_size_rows must be > 0");
  VELOX_CHECK_LE(
      batchSize,
      std::numeric_limits<vector_size_t>::max(),
      "velox.cudf.gpu_batch_size_rows must be <= max(vector_size_t)");
  return batchSize;
}
// True if `type` is or contains a VARBINARY anywhere in its tree.
bool containsVarbinary(const TypePtr& type) {
  if (type->kind() == TypeKind::VARBINARY) {
    return true;
  }
  for (uint32_t i = 0; i < type->size(); ++i) {
    if (containsVarbinary(type->childAt(i))) {
      return true;
    }
  }
  return false;
}

// Walks a nested (ROW/ARRAY/MAP) Velox type alongside its cuDF column and
// fails if a VARBINARY below the top level is physically a cuDF STRUCT, i.e. a
// self-describing decimal aggregate state. Aggregation intermediates are
// always top-level columns, so nested states are never produced; this guards
// against silently exporting a STRUCT where Arrow expects binary data.
void checkNoNestedDecimalState(
    const TypePtr& type,
    const cudf::column_view& column,
    const std::string& path) {
  switch (type->kind()) {
    case TypeKind::VARBINARY:
      VELOX_CHECK(
          column.type().id() != cudf::type_id::STRUCT,
          "CudfToVelox: VARBINARY field '{}' nested inside a complex type is "
          "physically a cuDF STRUCT (unmaterialized decimal aggregate state); "
          "nested decimal aggregate states are not supported",
          path);
      return;
    case TypeKind::ROW: {
      if (column.type().id() != cudf::type_id::STRUCT ||
          column.num_children() != static_cast<cudf::size_type>(type->size())) {
        return;
      }
      const auto& rowType = type->asRow();
      for (uint32_t i = 0; i < rowType.size(); ++i) {
        if (containsVarbinary(rowType.childAt(i))) {
          checkNoNestedDecimalState(
              rowType.childAt(i),
              column.child(i),
              path + "." + rowType.nameOf(i));
        }
      }
      return;
    }
    case TypeKind::ARRAY:
      if (column.type().id() == cudf::type_id::LIST &&
          column.num_children() > cudf::lists_column_view::child_column_index) {
        checkNoNestedDecimalState(
            type->childAt(0),
            column.child(cudf::lists_column_view::child_column_index),
            path + "[]");
      }
      return;
    case TypeKind::MAP: {
      // cuDF represents MAP as LIST<STRUCT<key, value>>.
      if (column.type().id() != cudf::type_id::LIST ||
          column.num_children() <=
              cudf::lists_column_view::child_column_index) {
        return;
      }
      auto entries = column.child(cudf::lists_column_view::child_column_index);
      if (entries.type().id() != cudf::type_id::STRUCT ||
          entries.num_children() != 2) {
        return;
      }
      checkNoNestedDecimalState(
          type->childAt(0), entries.child(0), path + ".key");
      checkNoNestedDecimalState(
          type->childAt(1), entries.child(1), path + ".value");
      return;
    }
    default:
      return;
  }
}

// Prepares a GPU table for Arrow export. Any top-level column whose Velox type
// is VARBINARY but whose cuDF column is a self-describing decimal aggregate
// state STRUCT is packed into the 32-byte STRING blob that CPU Velox expects,
// so Arrow export sees a STRING column. Returns a view over the original
// columns with the packed ones substituted; the packed columns are appended to
// `packed`, which the caller must keep alive until the export has completed.
// Returns `table` unchanged when nothing needs packing.
cudf::table_view packDecimalStatesForExport(
    const cudf::table_view& table,
    const RowTypePtr& outputType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    std::vector<std::unique_ptr<cudf::column>>& packed) {
  VELOX_CHECK_EQ(
      table.num_columns(),
      static_cast<cudf::size_type>(outputType->size()),
      "CudfToVelox: GPU table column count does not match output type {}",
      outputType->toString());
  std::vector<cudf::column_view> columns;
  bool replaced = false;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    const auto& type = outputType->childAt(i);
    auto column = table.column(i);
    if (type->kind() == TypeKind::VARBINARY &&
        column.type().id() == cudf::type_id::STRUCT) {
      VELOX_CHECK(
          isDecimalStateStruct(column),
          "CudfToVelox: VARBINARY column '{}' is a cuDF STRUCT that is not a "
          "recognized decimal aggregate state shape",
          outputType->nameOf(i));
      packed.push_back(packDecimalState(column, stream, mr));
      column = packed.back()->view();
      replaced = true;
    } else if (type->size() > 0 && containsVarbinary(type)) {
      checkNoNestedDecimalState(type, column, outputType->nameOf(i));
    }
    columns.push_back(column);
  }
  return replaced ? cudf::table_view(columns) : table;
}

// Exports a GPU table to a Velox RowVector of `outputType`, packing any
// self-describing decimal aggregate state columns to their VARBINARY blob form
// first. Every GPU -> CPU conversion in CudfToVelox goes through here.
RowVectorPtr exportToVelox(
    const cudf::table_view& table,
    const RowTypePtr& outputType,
    memory::MemoryPool* pool,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<std::unique_ptr<cudf::column>> packed;
  auto exportView =
      packDecimalStatesForExport(table, outputType, stream, mr, packed);
  auto output =
      with_arrow::toVeloxColumn(exportView, pool, outputType, "", stream, mr);
  // Synchronize before `packed` is released so the export's device reads of
  // the packed columns have completed.
  stream.sync();
  output->setType(outputType);
  return output;
}
} // namespace

CudfFromVelox::CudfFromVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          outputType,
          planNodeId,
          "CudfFromVelox",
          nvtx3::rgb{255, 140, 0}, // Orange
          NvtxMethodFlag::kAll,
          std::nullopt,
          std::nullopt),
      timestampTimeZone_(driverCtx->queryConfig().get<std::string>(
          facebook::velox::core::QueryConfig::kSessionTimezone)) {
  auto parentId = planNodeId.substr(0, planNodeId.find("-from-velox"));
  stats_.withWLock([&](auto& stats) {
    stats.setStatSplitter(
        [parentId = std::move(parentId)](const auto& combinedStats) {
          auto result = combinedStats;
          result.planNodeId = parentId;
          return std::vector<exec::OperatorStats>{std::move(result)};
        });
  });
}

void CudfFromVelox::doAddInput(RowVectorPtr input) {
  // compile() places CudfFromVelox only after operators classified as
  // producing CPU output, so device-resident input means the upstream operator
  // is misclassified.
  VELOX_CHECK_NULL(
      dynamic_cast<const CudfVector*>(input.get()),
      "CudfFromVelox received a device-resident CudfVector. The upstream "
      "operator produces GPU output but is classified as a CPU operator. "
      "Make it emit host vectors, or fix its operator adapter so that no "
      "CudfFromVelox is placed after it.");

  if (input->size() > 0) {
    // Materialize lazy vectors
    for (auto& child : input->children()) {
      child->loadedVector();
    }
    input->loadedVector();

    // Accumulate inputs
    inputs_.push_back(input);
    currentOutputSize_ += input->size();
  }
}

RowVectorPtr CudfFromVelox::doGetOutput() {
  const auto targetOutputSize =
      preferredGpuBatchSizeRows(operatorCtx_->driverCtx()->queryConfig());

  finished_ = noMoreInput_ && inputs_.empty();

  if (finished_ or
      (currentOutputSize_ < targetOutputSize and not noMoreInput_) or
      inputs_.empty()) {
    return nullptr;
  }

  // Select inputs that don't exceed the max vector size limit
  std::vector<RowVectorPtr> selectedInputs;
  vector_size_t totalSize = 0;
  auto const maxVectorSize = std::numeric_limits<vector_size_t>::max();

  for (const auto& input : inputs_) {
    if (totalSize + input->size() <= maxVectorSize) {
      selectedInputs.push_back(input);
      totalSize += input->size();
    } else {
      break;
    }
  }

  // Combine selected RowVectors into a single RowVector
  auto input = mergeRowVectors(selectedInputs, inputs_[0]->pool());

  // Remove processed inputs
  inputs_.erase(inputs_.begin(), inputs_.begin() + selectedInputs.size());
  currentOutputSize_ -= totalSize;

  // Early return if no input
  if (input->size() == 0) {
    return nullptr;
  }

  // Get a stream from the global stream pool
  auto stream = cudfGlobalStreamPool().get_stream();

  // cuDF tables with zero columns cannot represent a row count, so we
  // create a CudfVector directly with an empty table, preserving the
  // logical row count. This mirrors the zero-column handling in
  // CudfToVelox::doGetOutput().
  if (input->childrenSize() == 0) {
    auto emptyTable = std::make_unique<cudf::table>();
    return std::make_shared<CudfVector>(
        input->pool(),
        outputType_,
        input->size(),
        std::move(emptyTable),
        stream);
  }

  // Convert RowVector to cudf table.  toCudfTable synchronizes the stream
  // internally before releasing Arrow host buffers, so no additional sync
  // is needed here.
  auto tbl = with_arrow::toCudfTable(
      input, input->pool(), stream, get_output_mr(), timestampTimeZone_);

  VELOX_CHECK_NOT_NULL(tbl);

  // Return a CudfVector that owns the cudf table
  const auto size = tbl->num_rows();

  return std::make_shared<CudfVector>(
      input->pool(), outputType_, size, std::move(tbl), stream);
}

void CudfFromVelox::doClose() {
  // TODO(kn): Remove default stream after redesign of CudfFromVelox
  cudf::get_default_stream(cudf::allow_default_stream).sync();
  Operator::close();
  inputs_.clear();
}

CudfToVelox::CudfToVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          outputType,
          planNodeId,
          "CudfToVelox",
          nvtx3::rgb{148, 0, 211}, // Purple
          NvtxMethodFlag::kAll,
          std::nullopt,
          std::nullopt) {
  auto parentId = planNodeId.substr(0, planNodeId.find("-to-velox"));
  stats_.withWLock([&](auto& stats) {
    stats.setStatSplitter(
        [parentId = std::move(parentId)](const auto& combinedStats) {
          auto result = combinedStats;
          result.planNodeId = parentId;
          return std::vector<exec::OperatorStats>{std::move(result)};
        });
  });
}

bool CudfToVelox::isPassthroughMode() const {
  return operatorCtx_->driverCtx()->queryConfig().get<bool>(
      kPassthroughMode, true);
}

void CudfToVelox::doAddInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    inputs_.push_back(std::move(cudfInput));
  }
}

std::optional<uint64_t> CudfToVelox::averageRowSize() {
  if (!averageRowSize_) {
    averageRowSize_ =
        inputs_.front()->estimateFlatSize() / inputs_.front()->size();
  }
  return averageRowSize_;
}

// Pop inputs_.front(), convert its GPU table to a Velox RowVector via a
// single to_arrow_host + synchronize, and return it.  The caller is
// responsible for any further slicing.
RowVectorPtr CudfToVelox::convertFrontToVelox() {
  auto cudfVector = std::move(inputs_.front());
  inputs_.pop_front();
  auto stream = cudfVector->stream();
  auto tableView = cudfVector->getTableView();
  return exportToVelox(tableView, outputType_, pool(), stream, get_temp_mr());
}

// Output batching strategy
// ========================
// The key constraint is minimising D->H (device-to-host) transfers.
// Each call to toVeloxColumn / to_arrow_host triggers one D->H copy per
// column, so calling it once per output batch (rather than once per row
// or once per input batch) is critical for performance.
//
// Two cases arise depending on the size of the front GPU input relative
// to targetBatchSize:
//
//  (A) Front input >= targetBatchSize  (e.g. CudfOrderBy: one large sorted
//      table).  We convert the whole input to Velox in one shot and then
//      slice it purely on the CPU using BaseVector::slice().  Subsequent
//      getOutput() calls return successive CPU slices with no additional
//      D->H work until veloxBuffer_ is exhausted.
//
//  (B) Front input < targetBatchSize  (e.g. CudfFilterProject with high
//      selectivity: many small GPU batches).  We concatenate inputs on device
//      until we accumulate targetBatchSize rows, then convert the concat
//      result to Velox in one shot.  This preserves the GPU-side merge
//      that avoids emitting many undersized Velox batches downstream.
//
// In both cases exactly one toVeloxColumn + stream.sync() is issued
// per output batch, regardless of how many GPU inputs were consumed.
RowVectorPtr CudfToVelox::doGetOutput() {
  if (finished_) {
    return nullptr;
  }

  if (outputType_->size() == 0) {
    // cuDF zero-column tables do not have a row count, so we sum the sizes
    // of all CudfVectors in the inputs_, to maintain the logical count.
    // This is necessary to ensure correct behavior for e.g. `count` operators.
    vector_size_t totalSize = 0;
    while (!inputs_.empty()) {
      totalSize += inputs_.front()->size();
      inputs_.pop_front();
    }
    finished_ = noMoreInput_ && inputs_.empty();
    if (totalSize == 0) {
      return nullptr;
    }
    return BaseVector::create<RowVector>(outputType_, totalSize, pool());
  }

  // Drain veloxBuffer_ (populated on a previous call) before consuming
  // more GPU inputs.
  if (!veloxBuffer_) {
    if (inputs_.empty()) {
      finished_ = noMoreInput_;
      return nullptr;
    }

    // Passthrough mode: emit each GPU input as a single Velox batch with no
    // re-batching.  Used when the caller knows the batch size is already
    // correct (e.g. default pipeline without explicit batch-size overrides).
    if (isPassthroughMode()) {
      auto output = convertFrontToVelox();
      finished_ = noMoreInput_ && inputs_.empty();
      if (output->size() == 0) {
        return nullptr;
      }
      return output;
    }

    const auto targetBatchSize = outputBatchRows(averageRowSize());

    if (static_cast<vector_size_t>(inputs_.front()->size()) >=
        targetBatchSize) {
      // Case A: large input.  Convert once; subsequent calls slice CPU-side.
      veloxBuffer_ = convertFrontToVelox();
      veloxOffset_ = 0;
      averageRowSize_ = std::nullopt; // recompute from next input
    } else {
      // Case B: small inputs.  GPU-concat until we reach targetBatchSize,
      // then convert the merged table in one D->H transfer.
      auto stream = inputs_.front()->stream();
      std::vector<CudfVectorPtr> toConcat;
      vector_size_t accumulated = 0;
      while (!inputs_.empty() && accumulated < targetBatchSize) {
        accumulated += static_cast<vector_size_t>(inputs_.front()->size());
        toConcat.push_back(std::move(inputs_.front()));
        inputs_.pop_front();
      }
      VELOX_CHECK_LE(
          accumulated,
          std::numeric_limits<cudf::size_type>::max(),
          "Accumulated row count exceeds cudf int32 limit");
      auto concatTable = getConcatenatedTable(
          std::move(toConcat), outputType_, stream, get_temp_mr());
      veloxBuffer_ = exportToVelox(
          concatTable->view(), outputType_, pool(), stream, get_temp_mr());
      veloxOffset_ = 0;
      averageRowSize_ = std::nullopt;
    }
  }

  // Slice veloxBuffer_ on the CPU to produce the next output batch.
  const auto totalRows = static_cast<vector_size_t>(veloxBuffer_->size());
  if (veloxOffset_ >= totalRows) {
    veloxBuffer_.reset();
    finished_ = noMoreInput_ && inputs_.empty();
    return nullptr;
  }

  const auto targetBatchSize = outputBatchRows(
      veloxBuffer_->estimateFlatSize() /
      static_cast<uint64_t>(std::max<vector_size_t>(totalRows, 1)));
  const auto take = std::min(targetBatchSize, totalRows - veloxOffset_);

  auto slice = std::dynamic_pointer_cast<RowVector>(
      veloxBuffer_->slice(veloxOffset_, take));
  VELOX_CHECK_NOT_NULL(slice);
  veloxOffset_ += take;

  if (veloxOffset_ >= totalRows) {
    veloxBuffer_.reset();
    finished_ = noMoreInput_ && inputs_.empty();
  }

  return slice;
}

void CudfToVelox::doClose() {
  Operator::close();
  inputs_.clear();
  veloxBuffer_.reset();
}

} // namespace facebook::velox::cudf_velox
