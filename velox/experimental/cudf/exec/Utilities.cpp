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
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/testutil/TestValue.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

int getNumCudaDevices() {
  int numDevices{};
  CUDF_CUDA_TRY(cudaGetDeviceCount(&numDevices));
  return numDevices;
}

int getCurrentCudaDevice() {
  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  return device;
}

CudaEvent& eventForThread() {
  // Intentionally leak per-thread, per-device events to avoid CUDA calls from
  // thread-local destructors after CUDA context teardown.
  thread_local static std::vector<CudaEvent*> events(getNumCudaDevices());
  auto const device = getCurrentCudaDevice();
  VELOX_CHECK_GE(device, 0);
  auto const deviceIndex = static_cast<size_t>(device);
  VELOX_CHECK_LT(deviceIndex, events.size());

  if (events[deviceIndex] == nullptr) {
    events[deviceIndex] = new CudaEvent(cudaEventDisableTiming);
  }
  return *events[deviceIndex];
}

size_t maxBatchRows() {
  const auto& cudfConfig = CudfConfig::getInstance();
  if (cudfConfig.batchSizeMaxThreshold) {
    VELOX_CHECK_GT(
        cudfConfig.batchSizeMaxThreshold.value(),
        0,
        "cuDF max batch size must be positive");
    return static_cast<size_t>(cudfConfig.batchSizeMaxThreshold.value());
  }
  return static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
}

vector_size_t checkedVectorSize(size_t rowCount) {
  VELOX_CHECK_LE(
      rowCount,
      static_cast<size_t>(std::numeric_limits<vector_size_t>::max()),
      "cuDF vector row count exceeds Velox vector size limit");
  return static_cast<vector_size_t>(rowCount);
}

// True if two decimal-state candidate columns have the same physical form,
// i.e. cudf::concatenate would accept them together. STRING columns match
// regardless of offset width (cudf::concatenate handles mixed INT32/INT64
// offsets). STRUCT columns match when their direct children have identical
// data types (which includes the decimal scale). Host-side metadata only.
bool samePhysicalForm(
    const cudf::column_view& left,
    const cudf::column_view& right) {
  if (left.type() != right.type()) {
    return false;
  }
  if (left.type().id() != cudf::type_id::STRUCT) {
    return true;
  }
  if (left.num_children() != right.num_children()) {
    return false;
  }
  for (cudf::size_type i = 0; i < left.num_children(); ++i) {
    if (left.child(i).type() != right.child(i).type()) {
      return false;
    }
  }
  return true;
}

// True if column `columnIndex` does not have one physical form across
// `views`.
bool hasMixedPhysicalForm(
    const std::vector<cudf::table_view>& views,
    cudf::size_type columnIndex) {
  const auto first = views.front().column(columnIndex);
  for (size_t i = 1; i < views.size(); ++i) {
    if (!samePhysicalForm(first, views[i].column(columnIndex))) {
      return true;
    }
  }
  return false;
}

// libcudf's string concatenation copies the characters with a fused kernel
// when the inputs are many and small (fewer than about 393 KB of characters
// per input, 1.5 MB when nullable) and with one memcpy per input otherwise.
// The fused kernel takes the output character count as cudf::size_type, so a
// total of 2^31 characters or more is truncated and nothing is copied. With at
// most this many inputs per call the fused kernel is only chosen for totals
// below 1.5 GB, which fit; larger totals take the memcpy path, which is
// correct for any size.
constexpr size_t kMaxStringConcatenateFanIn = 1'024;

bool containsStrings(const cudf::column_view& column) {
  if (column.type().id() == cudf::type_id::STRING) {
    return true;
  }
  for (cudf::size_type i = 0; i < column.num_children(); ++i) {
    if (containsStrings(column.child(i))) {
      return true;
    }
  }
  return false;
}

// Concatenates one column's views with at most kMaxStringConcatenateFanIn
// inputs per libcudf call: groups of that many views are concatenated into
// temporaries first, then the temporaries are concatenated (recursively, so
// any fan-in is handled). Costs one extra copy of the column only when the
// bound is exceeded.
std::unique_ptr<cudf::column> concatenateColumnViewsBoundedFanIn(
    const std::vector<cudf::column_view>& columnViews,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (columnViews.size() <= kMaxStringConcatenateFanIn) {
    return cudf::concatenate(columnViews, stream, mr);
  }
  std::vector<std::unique_ptr<cudf::column>> groups;
  std::vector<cudf::column_view> groupViews;
  for (size_t begin = 0; begin < columnViews.size();
       begin += kMaxStringConcatenateFanIn) {
    const auto count =
        std::min(kMaxStringConcatenateFanIn, columnViews.size() - begin);
    groups.push_back(
        cudf::concatenate(
            cudf::host_span<cudf::column_view const>(
                columnViews.data() + begin, count),
            stream,
            get_temp_mr()));
    groupViews.push_back(groups.back()->view());
  }
  return concatenateColumnViewsBoundedFanIn(groupViews, stream, mr);
}

} // namespace

std::unique_ptr<cudf::table> concatenateTableViews(
    const std::vector<cudf::table_view>& tableViews,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_GT(tableViews.size(), 0);
  const auto& first = tableViews.front();
  const bool boundFanIn = tableViews.size() > kMaxStringConcatenateFanIn &&
      std::any_of(first.begin(), first.end(), containsStrings);
  if (!boundFanIn) {
    return cudf::concatenate(tableViews, stream, mr);
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(first.num_columns());
  std::vector<cudf::column_view> columnViews(tableViews.size());
  for (cudf::size_type columnIndex = 0; columnIndex < first.num_columns();
       ++columnIndex) {
    for (size_t i = 0; i < tableViews.size(); ++i) {
      columnViews[i] = tableViews[i].column(columnIndex);
    }
    columns.push_back(
        containsStrings(first.column(columnIndex))
            ? concatenateColumnViewsBoundedFanIn(columnViews, stream, mr)
            : cudf::concatenate(columnViews, stream, mr));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> concatenateTables(
    std::vector<std::unique_ptr<cudf::table>> tables,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  // Check for empty vector
  VELOX_CHECK_GT(tables.size(), 0);

  if (tables.size() == 1) {
    return std::move(tables[0]);
  }
  std::vector<cudf::table_view> tableViews;
  tableViews.reserve(tables.size());
  std::transform(
      tables.begin(),
      tables.end(),
      std::back_inserter(tableViews),
      [&](const auto& tbl) { return tbl->view(); });
  return concatenateTableViews(tableViews, stream, mr);
}

namespace {

std::unique_ptr<cudf::table> makeEmptyTable(
    TypePtr const& inputType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<std::unique_ptr<cudf::column>> emptyColumns;
  for (size_t i = 0; i < inputType->size(); ++i) {
    if (auto const& childType = inputType->childAt(i);
        childType->kind() == TypeKind::ROW) {
      auto tbl = makeEmptyTable(childType, stream, mr);
      auto structColumn = std::make_unique<cudf::column>(
          cudf::data_type(cudf::type_id::STRUCT),
          0,
          rmm::device_buffer(),
          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
          0,
          tbl->release());
      emptyColumns.push_back(std::move(structColumn));
    } else {
      // VARBINARY becomes STRING, so an empty table never carries a decimal
      // state struct; the concat funnel drops or re-types it when it meets
      // struct batches.
      auto emptyColumn = cudf::make_empty_column(
          cudf_velox::veloxToCudfDataType(inputType->childAt(i)));
      emptyColumns.push_back(std::move(emptyColumn));
    }
  }
  return std::make_unique<cudf::table>(std::move(emptyColumns));
}
} // namespace

std::vector<std::unique_ptr<cudf::column>> normalizeDecimalStateTableViews(
    std::vector<cudf::table_view>& tableViews,
    const TypePtr& tableType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<std::unique_ptr<cudf::column>> replacements;
  if (tableViews.size() <= 1 || tableType == nullptr || !tableType->isRow()) {
    return replacements;
  }

  // Candidate columns: logical VARBINARY. Every view of a CudfVector has as
  // many columns as its row type; tolerate narrower views (e.g. projections)
  // by only considering indices present in all of them.
  cudf::size_type numColumns = std::numeric_limits<cudf::size_type>::max();
  for (const auto& view : tableViews) {
    numColumns = std::min(numColumns, view.num_columns());
  }
  numColumns =
      std::min(numColumns, static_cast<cudf::size_type>(tableType->size()));

  std::vector<cudf::size_type> mixedColumns;
  for (cudf::size_type columnIndex = 0; columnIndex < numColumns;
       ++columnIndex) {
    if (tableType->childAt(columnIndex)->kind() == TypeKind::VARBINARY &&
        hasMixedPhysicalForm(tableViews, columnIndex)) {
      mixedColumns.push_back(columnIndex);
    }
  }
  if (mixedColumns.empty()) {
    return replacements;
  }

  // Zero-row views contribute nothing to the result. Drop them while at least
  // one non-empty view remains, so an empty STRING table from makeEmptyTable
  // never has to be re-typed. If every view is empty they are all kept and
  // normalizeDecimalStateBatches re-types them to one form.
  const bool anyNonEmpty =
      std::any_of(tableViews.begin(), tableViews.end(), [](const auto& view) {
        return view.num_rows() > 0;
      });
  if (anyNonEmpty) {
    std::erase_if(
        tableViews, [](const auto& view) { return view.num_rows() == 0; });
    if (tableViews.size() <= 1) {
      return replacements;
    }
    std::erase_if(mixedColumns, [&](cudf::size_type columnIndex) {
      return !hasMixedPhysicalForm(tableViews, columnIndex);
    });
    if (mixedColumns.empty()) {
      return replacements;
    }
  }

  // Materialize per-batch column lists so individual columns can be rebound.
  std::vector<std::vector<cudf::column_view>> batchColumns;
  batchColumns.reserve(tableViews.size());
  for (const auto& view : tableViews) {
    batchColumns.emplace_back(view.begin(), view.end());
  }

  std::vector<cudf::column_view> columnViews(tableViews.size());
  for (const auto columnIndex : mixedColumns) {
    for (size_t batch = 0; batch < tableViews.size(); ++batch) {
      columnViews[batch] = batchColumns[batch][columnIndex];
      VELOX_CHECK(
          isDecimalStateColumn(columnViews[batch]),
          "Cannot concatenate a VARBINARY column whose physical cuDF type differs across batches and is not a decimal aggregate state: column {}, batch {}, type {}",
          columnIndex,
          batch,
          cudf::type_to_name(columnViews[batch].type()));
    }
    auto columnReplacements =
        normalizeDecimalStateBatches(columnViews, stream, mr);
    for (size_t batch = 0; batch < tableViews.size(); ++batch) {
      batchColumns[batch][columnIndex] = columnViews[batch];
    }
    std::move(
        columnReplacements.begin(),
        columnReplacements.end(),
        std::back_inserter(replacements));
  }

  for (size_t batch = 0; batch < tableViews.size(); ++batch) {
    tableViews[batch] = cudf::table_view(batchColumns[batch]);
  }
  return replacements;
}

std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>&& tables,
    const TypePtr& tableType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  // Check for empty vector
  if (tables.size() == 0) {
    return makeEmptyTable(tableType, stream, mr);
  }

  auto inputStreams = std::vector<cuda::stream_ref>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  cudf::detail::join_streams(inputStreams, stream);

  // Bring decimal aggregate state columns to one physical form. The returned
  // replacement columns back the rebound views until concatenate is enqueued.
  const auto decimalStateReplacements = normalizeDecimalStateTableViews(
      tableViews, tableType, stream, get_temp_mr());

  // Even for a single input table we must concatenate (copy) rather than
  // release in-place: the output is owned by `stream` but the input buffer was
  // allocated on a different stream, so releasing it would bind deallocation to
  // the wrong stream.
  auto output = concatenateTableViews(tableViews, stream, mr);

  orderCudfVectorDeallocationsAfterStream(tables, inputStreams, stream);
  // Input tables are deallocated here when 'tables' goes out of scope.
  return output;
}

std::vector<std::unique_ptr<cudf::table>> getConcatenatedTableBatched(
    std::vector<CudfVectorPtr>&& tables,
    const TypePtr& tableType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<std::unique_ptr<cudf::table>> concatTables;
  // Check for empty vector
  if (tables.size() == 0) {
    concatTables.push_back(makeEmptyTable(tableType, stream, mr));
    return concatTables;
  }

  std::vector<std::unique_ptr<cudf::table>> outputTables;
  auto const maxRows = maxBatchRows();
  size_t start = 0;
  while (start < tables.size()) {
    size_t end = start;
    size_t runningRows = 0;
    std::vector<cudf::table_view> tableViews;
    tableViews.reserve(tables.size() - start);
    while (end < tables.size()) {
      VELOX_CHECK_NOT_NULL(tables[end]);
      auto const tableView = tables[end]->getTableView();
      auto const numRows = static_cast<size_t>(tableView.num_rows());
      if (runningRows > 0 && runningRows + numRows > maxRows) {
        break;
      }
      runningRows += numRows;
      tableViews.push_back(tableView);
      ++end;
    }

    std::vector<CudfVectorPtr> batch;
    std::vector<cuda::stream_ref> inputStreams;
    batch.reserve(end - start);
    inputStreams.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
      batch.push_back(std::move(tables[i]));
      inputStreams.push_back(batch.back()->stream());
    }

    cudf::detail::join_streams(inputStreams, stream);
    // Normalization is per output batch: each output is uniform, but two
    // outputs may carry a decimal state column in different physical forms.
    // Consumers accept either form per batch, and normalizing per batch keeps
    // peak memory bounded by one output batch.
    {
      const auto decimalStateReplacements = normalizeDecimalStateTableViews(
          tableViews, tableType, stream, get_temp_mr());
      outputTables.push_back(concatenateTableViews(tableViews, stream, mr));
    }

    // Rebind deallocation to the output stream where possible, then release
    // this group's inputs. Each completed output replaces its source inputs,
    // so peak memory is approximately the original input footprint plus the
    // largest output batch instead of the full input and output footprints.
    orderCudfVectorDeallocationsAfterStream(batch, inputStreams, stream);
    batch.clear();

    size_t retainedInputBatches =
        std::count_if(tables.begin(), tables.end(), [](const auto& table) {
          return table != nullptr;
        });
    common::testutil::TestValue::adjust(
        "facebook::velox::cudf_velox::getConcatenatedTableBatched::retainedInputBatchesAfterBatchRelease",
        &retainedInputBatches);
    start = end;
  }
  return outputTables;
}

std::vector<CudfVectorPtr> getConcatenatedCudfVectorsBatched(
    memory::MemoryPool* pool,
    std::vector<CudfVectorPtr>&& vectors,
    const TypePtr& tableType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_NOT_NULL(pool);

  std::vector<CudfVectorPtr> outputVectors;
  if (tableType->size() > 0) {
    auto tables =
        getConcatenatedTableBatched(std::move(vectors), tableType, stream, mr);
    outputVectors.reserve(tables.size());
    for (auto& table : tables) {
      VELOX_CHECK_NOT_NULL(table);
      const auto rowCount =
          checkedVectorSize(static_cast<size_t>(table->num_rows()));
      outputVectors.push_back(
          std::make_shared<CudfVector>(
              pool, tableType, rowCount, std::move(table), stream));
    }
    return outputVectors;
  }

  size_t remainingRows = 0;
  for (const auto& vector : vectors) {
    VELOX_CHECK_NOT_NULL(vector);
    VELOX_CHECK_EQ(vector->getTableView().num_columns(), 0);
    const auto rowCount = static_cast<size_t>(vector->size());
    VELOX_CHECK_LE(
        rowCount,
        std::numeric_limits<size_t>::max() - remainingRows,
        "zero-column cuDF vector row count overflow");
    remainingRows += rowCount;
  }

  const auto maxRows = maxBatchRows();
  do {
    const auto chunkRows = std::min(remainingRows, maxRows);
    outputVectors.push_back(
        std::make_shared<CudfVector>(
            pool,
            tableType,
            checkedVectorSize(chunkRows),
            makeEmptyTable(tableType, stream, mr),
            stream));
    remainingRows -= chunkRows;
  } while (remainingRows > 0);

  return outputVectors;
}

void streamsWaitForStream(
    CudaEvent& event,
    std::span<const cuda::stream_ref> streams,
    cuda::stream_ref stream) {
  event.recordFrom(stream);
  for (const auto& strm : streams) {
    event.waitOn(strm);
  }
}

CudaEvent::CudaEvent(unsigned int flags) {
  cudaEvent_t ev{};
  CUDF_CUDA_TRY(cudaEventCreateWithFlags(&ev, flags));
  event_ = ev;
}

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) {
    cudaEventDestroy(event_);
    event_ = nullptr;
  }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
  other.event_ = nullptr;
}

const CudaEvent& CudaEvent::recordFrom(cuda::stream_ref stream) const {
  CUDF_CUDA_TRY(cudaEventRecord(event_, stream.get()));
  return *this;
}

const CudaEvent& CudaEvent::waitOn(cuda::stream_ref stream) const {
  CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.get(), event_, 0));
  return *this;
}

std::string getBaseFunctionName(const std::string& fullName) {
  auto pos = fullName.rfind('.');
  return pos == std::string::npos ? fullName : fullName.substr(pos + 1);
}

std::string stripFunctionPrefix(
    const std::string& name,
    const std::string& prefix) {
  auto base = getBaseFunctionName(name);
  if (!prefix.empty() && base.find(prefix) == 0) {
    return base.substr(prefix.size());
  }
  return base;
}

void orderCudfVectorDeallocationsAfterStream(
    std::span<const CudfVectorPtr> vectors,
    std::span<const cuda::stream_ref> inputStreams,
    cuda::stream_ref stream) {
  bool allRebound = true;
  for (const auto& vector : vectors) {
    VELOX_CHECK_NOT_NULL(vector);
    allRebound &= vector->rebindStream(stream);
  }

  if (!allRebound) {
    streamsWaitForStream(eventForThread(), inputStreams, stream);
  }
}

} // namespace facebook::velox::cudf_velox
