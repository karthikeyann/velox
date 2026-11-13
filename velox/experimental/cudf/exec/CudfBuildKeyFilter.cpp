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
#include "velox/experimental/cudf/connectors/hive/CudfIntegerMembership.h"
#include "velox/experimental/cudf/exec/CudfBuildKeyFilter.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda_runtime.h>

#include <glog/logging.h>

#include <algorithm>
#include <optional>
#include <type_traits>

namespace facebook::velox::cudf_velox {
namespace {

// Whether a join key of 'type' can carry a dynamic filter: the integer types
// and DATE, whose values a scan compares as 64-bit integers.
bool isDynamicFilterKeyType(const TypePtr& type) {
  return type == TINYINT() || type == SMALLINT() || type == INTEGER() ||
      type == BIGINT() || type->isDate();
}

// Views a DATE column as the INT32 days it stores.
cudf::column_view integerStorageView(const cudf::column_view& column) {
  if (column.type().id() != cudf::type_id::TIMESTAMP_DAYS) {
    return column;
  }
  return cudf::column_view{
      cudf::data_type{cudf::type_id::INT32},
      column.size(),
      column.head(),
      column.null_mask(),
      column.null_count(),
      column.offset()};
}

// Returns the smallest and largest non-null value of a signed integer
// 'column', or std::nullopt if it has none.
std::optional<std::pair<int64_t, int64_t>> integerBounds(
    const cudf::column_view& column,
    cuda::stream_ref stream) {
  if (column.size() == column.null_count()) {
    return std::nullopt;
  }
  const auto [lower, upper] = cudf::minmax(column, stream, get_temp_mr());
  if (!lower->is_valid(stream) || !upper->is_valid(stream)) {
    return std::nullopt;
  }
  return cudf::type_dispatcher(
      column.type(),
      [&]<typename T>() -> std::optional<std::pair<int64_t, int64_t>> {
        if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
          return std::make_pair<int64_t, int64_t>(
              static_cast<const cudf::numeric_scalar<T>&>(*lower).value(stream),
              static_cast<const cudf::numeric_scalar<T>&>(*upper).value(
                  stream));
        } else {
          VELOX_UNSUPPORTED(
              "Dynamic join filter requires a signed integer key");
        }
      });
}

// A join with at least this many build-side key rows per integer between its
// smallest and largest key keeps most probe rows, so it builds no bitmap.
constexpr double kMaxDynamicFilterBitmapDensity{0.8};

// Copies the non-null values of the signed integer 'columns' to the host.
std::vector<int64_t> copyIntegersToHost(
    const std::vector<cudf::column_view>& columns,
    uint64_t numValues,
    cuda::stream_ref stream) {
  return cudf::type_dispatcher(
      columns.front().type(), [&]<typename T>() -> std::vector<int64_t> {
        if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
          std::vector<T> hostValues;
          std::vector<std::unique_ptr<cudf::table>> nonNullTables;
          hostValues.reserve(numValues);
          nonNullTables.reserve(columns.size());
          for (auto input : columns) {
            if (input.null_count() != 0) {
              nonNullTables.push_back(
                  cudf::drop_nulls(
                      cudf::table_view{{input}},
                      std::vector<cudf::size_type>{0},
                      stream,
                      get_temp_mr()));
              input = nonNullTables.back()->view().column(0);
            }
            if (input.size() == 0) {
              continue;
            }
            const auto offset = hostValues.size();
            hostValues.resize(offset + input.size());
            CUDF_CUDA_TRY(cudaMemcpyAsync(
                hostValues.data() + offset,
                input.template data<T>(),
                input.size() * sizeof(T),
                cudaMemcpyDeviceToHost,
                stream.get()));
          }
          stream.sync();
          if constexpr (std::is_same_v<T, int64_t>) {
            return hostValues;
          } else {
            return std::vector<int64_t>(hostValues.begin(), hostValues.end());
          }
        } else {
          VELOX_UNSUPPORTED(
              "Dynamic join filter requires a signed integer key");
        }
      });
}

// A build side with up to this many key rows per value of
// CudfConfig::dynamicFilterMaxValues is checked for distinct keys: a join of a
// fact table with a filtered dimension has many rows and few keys.
constexpr uint64_t kMaxRowsPerDistinctValue{64};

// Returns the distinct non-null values of the signed integer 'columns', or
// std::nullopt if there are more than 'maxValues'. Works table by table, so
// that the device holds one table's distinct keys at a time.
std::optional<std::vector<int64_t>> copyDistinctIntegersToHost(
    const std::vector<cudf::column_view>& columns,
    uint64_t maxValues,
    cuda::stream_ref stream) {
  std::vector<int64_t> values;
  for (const auto& column : columns) {
    const auto distinctKeys = cudf::distinct(
        cudf::table_view{{column}},
        std::vector<cudf::size_type>{0},
        cudf::duplicate_keep_option::KEEP_ANY,
        cudf::null_equality::EQUAL,
        cudf::nan_equality::ALL_EQUAL,
        stream,
        get_temp_mr());
    const auto distinct = distinctKeys->view().column(0);
    const uint64_t numDistinct = distinct.size() - distinct.null_count();
    if (numDistinct > maxValues) {
      return std::nullopt;
    }
    const auto tableValues =
        copyIntegersToHost({distinct}, numDistinct, stream);
    values.insert(values.end(), tableValues.begin(), tableValues.end());
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  if (values.size() > maxValues) {
    return std::nullopt;
  }
  return values;
}

BuildKeyFilter makeBuildKeyFilterOrThrow(
    const std::vector<cudf::column_view>& keyColumns,
    const TypePtr& type,
    cuda::stream_ref stream) {
  BuildKeyFilter result;
  if (!isDynamicFilterKeyType(type) || keyColumns.empty()) {
    return result;
  }
  std::vector<cudf::column_view> keys;
  keys.reserve(keyColumns.size());
  uint64_t numValues{0};
  uint64_t numRows{0};
  for (const auto& column : keyColumns) {
    keys.push_back(integerStorageView(column));
    // The build side has to hold the key as the signed integer of its type.
    if (!cudf::is_integral(keys.back().type()) ||
        !cudf::is_signed(keys.back().type()) ||
        keys.back().type() != keys.front().type()) {
      return result;
    }
    numValues += keys.back().size() - keys.back().null_count();
    numRows += keys.back().size();
  }
  if (numValues == 0) {
    // No probe row has a matching build key.
    result.filter = std::make_unique<common::AlwaysFalse>();
    result.exact = true;
    result.uniqueKeys = true;
    return result;
  }

  const auto maxValues = static_cast<uint64_t>(
      std::max<int64_t>(0, CudfConfig::getInstance().dynamicFilterMaxValues));
  std::optional<std::vector<int64_t>> values;
  if (numValues <= maxValues) {
    values = copyIntegersToHost(keys, numValues, stream);
  } else if (numRows <= kMaxRowsPerDistinctValue * maxValues) {
    values = copyDistinctIntegersToHost(keys, maxValues, stream);
  }
  const auto maxBitmapBits = 8 *
      static_cast<uint64_t>(std::max<int64_t>(
          0, CudfConfig::getInstance().dynamicFilterMaxBitmapBytes));
  if (values.has_value()) {
    std::sort(values->begin(), values->end());
    values->erase(std::unique(values->begin(), values->end()), values->end());
    result.filter = common::createBigintValues(*values, false);
    result.exact = true;
    result.uniqueKeys = values->size() == numValues;
    // A cuDF scan would have to list the values of a set on the host again,
    // which for a set kept as a bitmask means walking its whole range. The
    // keys are on the device: give the scan their bitmap.
    const auto span = static_cast<uint64_t>(values->back()) -
        static_cast<uint64_t>(values->front());
    if (result.filter->kind() != common::FilterKind::kBigintRange &&
        span < maxBitmapBits) {
      try {
        result.deviceBitmap = connector::hive::makeIntegerBitmap(
            keys, values->front(), values->back(), stream, get_output_mr());
        // Scans read the bitmap on their own streams.
        stream.sync();
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to build the bitmap of a dynamic filter: "
                     << e.what();
        result.deviceBitmap = nullptr;
      }
    }
    return result;
  }

  // Too many keys to send. Their bounds still prune row groups and rows
  // outside of them.
  std::optional<std::pair<int64_t, int64_t>> bounds;
  for (const auto& key : keys) {
    const auto keyBounds = integerBounds(key, stream);
    if (!keyBounds.has_value()) {
      continue;
    }
    bounds = bounds.has_value()
        ? std::make_pair(
              std::min(bounds->first, keyBounds->first),
              std::max(bounds->second, keyBounds->second))
        : keyBounds;
  }
  if (!bounds.has_value()) {
    return result;
  }
  // A bitmap of the keys lets a cuDF scan drop rows inside of the bounds. It
  // pays if it fits and the keys leave a good share of the bounds out.
  const auto span = static_cast<uint64_t>(bounds->second) -
      static_cast<uint64_t>(bounds->first);
  if (span < maxBitmapBits &&
      numValues < kMaxDynamicFilterBitmapDensity * span) {
    try {
      result.deviceBitmap = connector::hive::makeIntegerBitmap(
          keys, bounds->first, bounds->second, stream, get_output_mr());
      // Scans read the bitmap on their own streams.
      stream.sync();
    } catch (const std::exception& e) {
      // The bounds alone remain a valid filter.
      LOG(WARNING) << "Failed to build the bitmap of a dynamic filter: "
                   << e.what();
      result.deviceBitmap = nullptr;
    }
  }
  result.filter = std::make_unique<common::BigintRange>(
      bounds->first, bounds->second, false);
  return result;
}

} // namespace

BuildKeyFilter makeBuildKeyFilter(
    const std::vector<cudf::column_view>& keyColumns,
    const TypePtr& type,
    cuda::stream_ref stream) {
  try {
    return makeBuildKeyFilterOrThrow(keyColumns, type, stream);
  } catch (const std::exception& e) {
    // A dynamic filter is an optimization: a join whose build side cannot be
    // described, for example for lack of device memory, runs without one.
    LOG(WARNING) << "Failed to build a dynamic filter from join keys: "
                 << e.what();
    return {};
  }
}

} // namespace facebook::velox::cudf_velox
