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
#include "velox/experimental/cudf/connectors/hive/CudfDynamicFilter.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/connectors/hive/CudfIntegerMembership.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/Casts.h"
#include "velox/common/io/IoStatisticsRuntimeStats.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/common/FileColumnHandle.h"
#include "velox/connectors/common/FileConnectorUtil.h"
#include "velox/connectors/common/FileSplitReader.h"
#include "velox/connectors/common/PartitionValue.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/ExprOptimizer.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/type/Filter.h"
#include "velox/vector/DecodedVector.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

namespace {

std::optional<cudf::type_id> parquetIntegerType(
    const cudf::io::parquet::SchemaElement& schema) {
  using LogicalType = cudf::io::parquet::LogicalType;
  using ParquetType = cudf::io::parquet::Type;

  if (schema.logical_type.has_value()) {
    if (schema.logical_type->type != LogicalType::INTEGER) {
      return std::nullopt;
    }
    const auto isSigned = schema.logical_type->is_signed();
    switch (schema.logical_type->bit_width()) {
      case 8:
        return isSigned ? cudf::type_id::INT8 : cudf::type_id::UINT8;
      case 16:
        return isSigned ? cudf::type_id::INT16 : cudf::type_id::UINT16;
      case 32:
        return isSigned ? cudf::type_id::INT32 : cudf::type_id::UINT32;
      case 64:
        return isSigned ? cudf::type_id::INT64 : cudf::type_id::UINT64;
      default:
        VELOX_FAIL(
            "Invalid Parquet integer bit width: {}",
            schema.logical_type->bit_width());
    }
  }
  switch (schema.type) {
    case ParquetType::INT32:
      return cudf::type_id::INT32;
    case ParquetType::INT64:
      return cudf::type_id::INT64;
    default:
      return std::nullopt;
  }
}

std::optional<SubfieldFilterDecimalType> parquetDecimalType(
    const cudf::io::parquet::SchemaElement& schema) {
  using ParquetType = cudf::io::parquet::Type;

  if (!schema.logical_type.has_value() ||
      schema.logical_type->type != cudf::io::parquet::LogicalType::DECIMAL) {
    if (const auto integerType = parquetIntegerType(schema)) {
      return SubfieldFilterDecimalType{*integerType, 0};
    }
    return std::nullopt;
  }

  const auto scale = schema.decimal_scale;
  VELOX_CHECK_EQ(
      schema.logical_type->scale(),
      scale,
      "Parquet decimal logical type and schema element scales differ");
  auto decimalType = [&]() -> cudf::type_id {
    switch (schema.type) {
      case ParquetType::INT32:
        return cudf::type_id::DECIMAL32;
      case ParquetType::INT64:
        return cudf::type_id::DECIMAL64;
      case ParquetType::FIXED_LEN_BYTE_ARRAY:
        VELOX_CHECK_GT(
            schema.type_length,
            0,
            "Invalid fixed-length Parquet decimal width");
        if (std::cmp_less_equal(schema.type_length, sizeof(int32_t))) {
          return cudf::type_id::DECIMAL32;
        }
        if (std::cmp_less_equal(schema.type_length, sizeof(int64_t))) {
          return cudf::type_id::DECIMAL64;
        }
        if (std::cmp_less_equal(schema.type_length, sizeof(int128_t))) {
          return cudf::type_id::DECIMAL128;
        }
        VELOX_FAIL(
            "Unsupported fixed-length Parquet decimal width: {}",
            schema.type_length);
      case ParquetType::BYTE_ARRAY: {
        const auto precision = schema.logical_type->precision();
        VELOX_CHECK_GT(
            precision, 0, "Parquet decimal is missing a valid precision");
        VELOX_CHECK_LE(
            precision,
            LongDecimalType::kMaxPrecision,
            "Parquet decimal precision exceeds the maximum supported precision");
        if (precision <= std::numeric_limits<int32_t>::digits10) {
          return cudf::type_id::DECIMAL32;
        }
        if (precision <= std::numeric_limits<int64_t>::digits10) {
          return cudf::type_id::DECIMAL64;
        }
        return cudf::type_id::DECIMAL128;
      }
      default:
        VELOX_FAIL(
            "Unsupported physical Parquet type for decimal column: {}",
            static_cast<int32_t>(schema.type));
    }
  }();
  return SubfieldFilterDecimalType{decimalType, scale};
}

SubfieldFilterDecimalTypes parquetDecimalTypes(
    std::span<const cudf::io::parquet::SchemaElement> metadataSchema,
    const RowTypePtr& readerSchema,
    bool fileColumnNamesReadAsLowerCase) {
  VELOX_CHECK(
      !metadataSchema.empty(), "Cannot build a filter from an empty schema");

  SubfieldFilterDecimalTypes decimalTypes;
  for (const auto childIndex : metadataSchema.front().children_idx) {
    VELOX_CHECK_LT(
        childIndex,
        metadataSchema.size(),
        "Parquet schema child index out of range");
    const auto& child = metadataSchema[childIndex];
    const auto name = fileColumnNamesReadAsLowerCase
        ? functions::stringImpl::utf8StrToLowerCopy(child.name)
        : child.name;
    if (!readerSchema->containsChild(name)) {
      continue;
    }
    const auto& logicalType = readerSchema->findChild(name);
    if (logicalType->isDecimal()) {
      if (auto decimalType = parquetDecimalType(child)) {
        if (cudf::is_fixed_point(cudf::data_type{decimalType->type})) {
          const auto [_, tableScale] = getDecimalPrecisionScale(*logicalType);
          VELOX_CHECK_LE(
              decimalType->scale,
              tableScale,
              "Parquet decimal scale of '{}' exceeds the table scale",
              name);
        }
        decimalTypes.emplace(name, *decimalType);
      }
    }
  }
  return decimalTypes;
}

// Returns whether a constant-folded filter keeps every row. A null constant
// keeps none, matching SQL three-valued logic.
bool isTrueConstant(
    const core::ConstantTypedExpr& constant,
    memory::MemoryPool* pool) {
  VELOX_USER_CHECK_EQ(
      constant.type()->kind(),
      TypeKind::BOOLEAN,
      "Remaining filter must be a boolean expression: {}",
      constant.toString());
  return !constant.isNull() &&
      constant.toConstantVector(pool)->as<ConstantVector<bool>>()->valueAt(0);
}

// Returns whether a pushed-down filter accepts a partition key value, which
// is NULL when absent.
bool testPartitionKeyFilter(
    const common::Filter& filter,
    const TypePtr& type,
    const std::optional<std::string>& value,
    bool dateValueDaysSinceEpoch,
    bool timestampValueAsLocalTime) {
  if (!value.has_value()) {
    return filter.testNull();
  }
  return common::applyFilter(
      filter,
      ::facebook::velox::connector::PartitionValue::fromString(
          *value,
          *type,
          timestampValueAsLocalTime
              ? ::facebook::velox::connector::PartitionValue::TimestampMode::
                    kLocalTime
              : ::facebook::velox::connector::PartitionValue::TimestampMode::
                    kUtc,
          dateValueDaysSinceEpoch
              ? ::facebook::velox::connector::PartitionValue::DateMode::
                    kDaysSinceEpoch
              : ::facebook::velox::connector::PartitionValue::DateMode::
                    kIsoString));
}

// A connector sees the session time zone settings but not the session
// QueryConfig. As a workaround, the remaining filter is folded and compiled
// against a local QueryCtx with only those settings, and a filter that depends
// on another setting runs on the CPU data source.
// TODO: Remove the workaround once ConnectorQueryCtx exposes the session
// QueryConfig (facebookincubator/velox#19476) and expression::optimize can fold
// through the session's evaluator (facebookincubator/velox#17898).
// NOTE: A setting left at its default can still change a fold, such as how
// TIMESTAMP WITH TIME ZONE renders as VARCHAR.
std::shared_ptr<core::QueryCtx> makeRemainingFilterQueryCtx(
    const ConnectorQueryCtx& connectorQueryCtx) {
  std::unordered_map<std::string, std::string> sessionSettings{
      {core::QueryConfig::kAdjustTimestampToTimezone,
       connectorQueryCtx.adjustTimestampToTimezone() ? "true" : "false"}};
  // An empty time zone is not a valid setting; leaving it unset means UTC.
  if (!connectorQueryCtx.sessionTimezone().empty()) {
    sessionSettings.emplace(
        core::QueryConfig::kSessionTimezone,
        connectorQueryCtx.sessionTimezone());
  }
  return core::QueryCtx::create(
      nullptr, core::QueryConfig{std::move(sessionSettings)});
}

// Returns whether the remaining filter depends on a session setting the local
// QueryCtx lacks. A cast from VARCHAR follows legacy_cast and
// cast_string_to_floating_point_lenient_inf_nan. A cast from TIMESTAMP to
// VARCHAR follows legacy_cast and the session time zone, which the data source
// cannot apply to a column's text.
// NOTE: A cast from TIMESTAMP to VARCHAR is looked for before folding, which
// would fold a constant one with the default settings. A cast from VARCHAR is
// looked for after folding, so that a cast of a VARCHAR constant, the form a
// plan gives a DECIMAL constant, still folds away.
bool needsSessionQueryConfig(
    const core::TypedExprPtr& unfoldedFilter,
    const core::TypedExprPtr& foldedFilter) {
  return containsTimestampToVarcharCast(unfoldedFilter) ||
      containsCastFromVarchar(foldedFilter, /*floatingPointOnly=*/false);
}

// Filters of a table handle in the form the data source compiles them.
struct ScanFilters {
  // Subfield filters, including those extracted from the remaining filter.
  // They compile into a cuDF AST.
  common::SubfieldFilters subfieldFilters;

  // Rest of the remaining filter, folded, or null if nothing is left. It
  // compiles into a cuDF expression.
  core::TypedExprPtr remainingFilter;

  // Whether the remaining filter must run on the CPU data source, because it
  // depends on a session setting the local QueryCtx lacks.
  bool needsSessionQueryConfig{false};

  // Fraction of rows to keep, from the table handle and the remaining filter.
  double sampleRate{1};
};

ScanFilters splitAndFoldFilters(
    const HiveTableHandle& tableHandle,
    core::ExpressionEvaluator* expressionEvaluator,
    core::QueryCtx* remainingFilterQueryCtx,
    memory::MemoryPool* pool) {
  ScanFilters filters;
  for (const auto& [subfield, filter] : tableHandle.subfieldFilters()) {
    filters.subfieldFilters.emplace(subfield.clone(), filter->clone());
  }

  // Extract single-column filters like "col = 'X'" or "col <> 'Y'" from the
  // remaining filter into subfield filters, as the CPU data source does.
  filters.sampleRate = tableHandle.sampleRate();
  const auto remainingFilter =
      facebook::velox::connector::extractFiltersFromRemainingFilter(
          tableHandle.remainingFilter(),
          expressionEvaluator,
          filters.subfieldFilters,
          filters.sampleRate);

  // Optimize (rewrites + constant folding) the remaining filter before
  // evaluator selection so CudfFunctions never see scalar-only operand sets.
  if (remainingFilter) {
    filters.remainingFilter =
        expression::optimize(remainingFilter, remainingFilterQueryCtx, pool);
    filters.needsSessionQueryConfig =
        needsSessionQueryConfig(remainingFilter, filters.remainingFilter);
  }
  return filters;
}

// Returns why the data source cannot scan with 'filters' on the GPU, or
// std::nullopt if it can. isSupported() and the constructor share it, so that
// the connectors pick this data source only for a scan the constructor
// accepts.
std::optional<std::string> unsupportedFiltersReason(
    const ScanFilters& filters,
    core::QueryCtx* remainingFilterQueryCtx) {
  // The cuDF data source does not sample rows.
  if (filters.sampleRate < 1) {
    return fmt::format("Sampling is not supported: {}", filters.sampleRate);
  }
  for (const auto& [subfield, filter] : filters.subfieldFilters) {
    if (!canCreateAstFromSubfieldFilter(subfield, *filter)) {
      return fmt::format(
          "Subfield filter is not supported by the cuDF AST: {} {}",
          subfield.toString(),
          filter->toString());
    }
  }
  if (filters.needsSessionQueryConfig) {
    return fmt::format(
        "Remaining filter that depends on a session setting is not supported: {}",
        filters.remainingFilter->toString());
  }
  if (!filters.remainingFilter) {
    return std::nullopt;
  }
  // The filter is evaluated over the columns read for it, so a filter that
  // reads none has no rows to be evaluated over. A filter that folds to a
  // constant needs no evaluation.
  if (!filters.remainingFilter->isConstantKind() &&
      referencedInputFields(filters.remainingFilter).empty()) {
    return fmt::format(
        "Remaining filter that references no column is not supported: {}",
        filters.remainingFilter->toString());
  }
  // The filter is already folded, so pass no pool to skip folding again.
  if (!canExprRunOnGpu(
          filters.remainingFilter, remainingFilterQueryCtx, /*pool=*/nullptr)) {
    return fmt::format(
        "Remaining filter is not supported by cuDF: {}",
        filters.remainingFilter->toString());
  }
  return std::nullopt;
}

// Carries the advisory mark in the owner of the filter, so that it follows
// shared copies and is absent from a clone or a merge.
struct AdvisoryFilterDeleter {
  // See makeAdvisoryFilter().
  std::shared_ptr<const cudf::column> deviceBitmap;

  void operator()(common::Filter* filter) const {
    delete filter;
  }
};

// Whether the scan reads a column of 'type' as signed integers a dynamic
// filter tests as 64-bit values: the integer types and DATE.
bool hasIntegerStorage(const TypePtr& type) {
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

// Clears the entries of 'pass' for the rows of the signed integer 'input'
// that 'filter' rejects. Evaluates on the host, for the filters without a
// device path.
void intersectOnHost(
    const cudf::column_view& input,
    const common::Filter& filter,
    std::vector<uint8_t>& pass,
    cuda::stream_ref stream) {
  std::vector<uint8_t> valid;
  if (input.has_nulls()) {
    const auto validColumn = cudf::is_valid(input, stream, get_temp_mr());
    valid.resize(input.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        valid.data(),
        validColumn->view().data<uint8_t>(),
        valid.size(),
        cudaMemcpyDeviceToHost,
        stream.get()));
    stream.sync();
  }
  cudf::type_dispatcher(input.type(), [&]<typename T>() {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      std::vector<T> values(input.size());
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          values.data(),
          input.template data<T>(),
          values.size() * sizeof(T),
          cudaMemcpyDeviceToHost,
          stream.get()));
      stream.sync();
      const bool nullAllowed = filter.testNull();
      for (size_t i = 0; i < values.size(); ++i) {
        if (pass[i] == 0) {
          continue;
        }
        pass[i] = (!valid.empty() && valid[i] == 0)
            ? nullAllowed
            : filter.testInt64(values[i]);
      }
    } else {
      VELOX_UNSUPPORTED("Dynamic filter requires a signed integer column");
    }
  });
}

// Tests the value of 'decoded' at 'row' with 'filter' the way a CPU reader
// tests a value of that type.
template <TypeKind kind>
bool testValue(
    const common::Filter& filter,
    const DecodedVector& decoded,
    vector_size_t row) {
  return common::applyFilter(
      filter, decoded.valueAt<typename TypeTraits<kind>::NativeType>(row));
}

// Clears the entries of 'pass' for the rows of 'column', of Velox type 'type',
// that 'filter' rejects. Evaluates on the host over a copy of the column, for
// the column types without a device path.
void intersectOnHost(
    const cudf::column_view& column,
    const std::string& name,
    const TypePtr& type,
    const common::Filter& filter,
    std::vector<uint8_t>& pass,
    memory::MemoryPool* pool,
    cuda::stream_ref stream) {
  // The values are read below as the table's type, so the file has to store
  // that type. A timestamp may be stored in any unit.
  const auto tableType = veloxToCudfDataType(type);
  VELOX_USER_CHECK(
      column.type().id() == tableType.id() ||
          (type->isTimestamp() && cudf::is_timestamp(column.type())),
      "cuDF Hive scan cannot apply a dynamic filter to the type the file stores column {} in",
      name);
  const auto rows = with_arrow::toVeloxColumn(
      cudf::table_view{{column}}, pool, ROW("c0", type), stream, get_temp_mr());
  VELOX_CHECK(
      rows->childAt(0)->type()->kindEquals(type),
      "Unexpected type of the host copy of column {}: {}",
      name,
      rows->childAt(0)->type()->toString());
  DecodedVector decoded(*rows->childAt(0));
  const bool nullAllowed = filter.testNull();
  for (vector_size_t row = 0; row < decoded.size(); ++row) {
    if (pass[row] == 0) {
      continue;
    }
    if (decoded.isNullAt(row)) {
      pass[row] = nullAllowed;
    } else if (type->isPrimitiveType()) {
      pass[row] = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          testValue, type->kind(), filter, decoded, row);
    } else {
      // Only the null tests are defined on a value that is not a scalar.
      VELOX_USER_CHECK(
          filter.kind() == common::FilterKind::kIsNull ||
              filter.kind() == common::FilterKind::kIsNotNull ||
              filter.kind() == common::FilterKind::kAlwaysTrue,
          "Dynamic filter cannot be applied to a column of type {}: {}",
          type->toString(),
          filter.toString());
      pass[row] = filter.kind() != common::FilterKind::kIsNull;
    }
  }
}

// Number of true rows of the non-nullable BOOL8 'mask'.
int64_t countRetainedRows(
    const cudf::column_view& mask,
    cuda::stream_ref stream) {
  const auto sum = cudf::reduce(
      mask,
      *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::INT64},
      stream,
      get_temp_mr());
  return static_cast<const cudf::numeric_scalar<int64_t>&>(*sum).value(stream);
}

// Returns the values of a value-set filter in ascending order.
std::vector<int64_t> sortedFilterValues(const common::Filter& filter) {
  auto values = filter.kind() == common::FilterKind::kBigintValuesUsingHashTable
      ? static_cast<const common::BigintValuesUsingHashTable&>(filter).values()
      : static_cast<const common::BigintValuesUsingBitmask&>(filter).values();
  std::sort(values.begin(), values.end());
  return values;
}

// Returns the device bitmap of the value-set 'filter', whose values lie in
// ['lower', 'upper']: UINT32 words with bit 'value - lower' set for each
// value. The data sources of a scan, one per driver and per split prepared
// ahead, receive the same filter object and share one bitmap, so that its
// values are listed and copied to the device once. Sets 'built' if this call
// made the bitmap.
std::shared_ptr<const cudf::column> sharedValueBitmap(
    const common::FilterPtr& filter,
    int64_t lower,
    int64_t upper,
    cuda::stream_ref stream,
    bool& built) {
  built = false;
  struct Entry {
    std::weak_ptr<common::Filter> filter;
    std::weak_ptr<const cudf::column> bitmap;
  };
  static std::mutex mutex;
  static std::unordered_map<const common::Filter*, Entry> entries;

  std::lock_guard<std::mutex> lock(mutex);
  std::erase_if(
      entries, [](const auto& entry) { return entry.second.bitmap.expired(); });
  if (const auto it = entries.find(filter.get()); it != entries.end()) {
    // The address of a freed filter can be taken by a new one.
    if (auto bitmap = it->second.bitmap.lock();
        bitmap != nullptr && it->second.filter.lock() == filter) {
      return bitmap;
    }
  }

  const auto span = static_cast<uint64_t>(upper) - static_cast<uint64_t>(lower);
  std::vector<uint32_t> words(span / 32 + 1, 0);
  for (const auto value : sortedFilterValues(*filter)) {
    const auto index =
        static_cast<uint64_t>(value) - static_cast<uint64_t>(lower);
    words[index / 32] |= uint32_t{1} << (index % 32);
  }
  auto column = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::UINT32},
      words.size(),
      cudf::mask_state::UNALLOCATED,
      stream,
      get_output_mr());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      column->mutable_view().data<uint32_t>(),
      words.data(),
      words.size() * sizeof(uint32_t),
      cudaMemcpyHostToDevice,
      stream.get()));
  // 'words' has to outlive the copy, and the other data sources read the
  // bitmap on their own streams.
  stream.sync();
  std::shared_ptr<const cudf::column> bitmap = std::move(column);
  entries.insert_or_assign(filter.get(), Entry{filter, bitmap});
  built = true;
  return bitmap;
}

// Advisory filters stop filtering rows once they have kept more than this
// share of at least 'kAdvisoryFilterMinRows' rows.
constexpr double kAdvisoryFilterMaxKeptRatio{0.8};
constexpr uint64_t kAdvisoryFilterMinRows{100'000};

// Widest range of values, in 32-bit words, a value set is tested with a
// bitmap for. A wider set is tested with a hash set.
constexpr uint64_t kMaxBitmapWords{8'388'608};
} // namespace

common::FilterPtr makeAdvisoryFilter(
    std::unique_ptr<common::Filter> filter,
    std::shared_ptr<const cudf::column> deviceBitmap) {
  VELOX_CHECK(
      deviceBitmap == nullptr ||
          filter->kind() == common::FilterKind::kBigintRange,
      "Only a range carries a device bitmap");
  return {filter.release(), AdvisoryFilterDeleter{std::move(deviceBitmap)}};
}

bool isAdvisoryFilter(const common::FilterPtr& filter) {
  return std::get_deleter<AdvisoryFilterDeleter>(filter) != nullptr;
}

const cudf::column* advisoryFilterBitmap(const common::FilterPtr& filter) {
  const auto* deleter = std::get_deleter<AdvisoryFilterDeleter>(filter);
  return deleter != nullptr ? deleter->deviceBitmap.get() : nullptr;
}

bool CudfHiveDataSource::isSupported(
    const ConnectorTableHandlePtr& tableHandle,
    const ConnectorQueryCtx& connectorQueryCtx) {
  return !unsupportedReason(tableHandle, connectorQueryCtx).has_value();
}

std::optional<std::string> CudfHiveDataSource::unsupportedReason(
    const ConnectorTableHandlePtr& tableHandle,
    const ConnectorQueryCtx& connectorQueryCtx) {
  auto hiveTableHandle =
      std::dynamic_pointer_cast<const HiveTableHandle>(tableHandle);
  if (!hiveTableHandle) {
    return fmt::format(
        "Table handle that is not a Hive table handle is not supported: {}",
        tableHandle->toString());
  }

  const auto queryCtx = makeRemainingFilterQueryCtx(connectorQueryCtx);
  const auto filters = splitAndFoldFilters(
      *hiveTableHandle,
      connectorQueryCtx.expressionEvaluator(),
      queryCtx.get(),
      connectorQueryCtx.memoryPool());
  return unsupportedFiltersReason(filters, queryCtx.get());
}

CudfHiveDataSource::CudfHiveDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    facebook::velox::FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<FetchAheadContext>& fetchAheadContext)
    : CudfHiveDataSource(
          outputType,
          tableHandle,
          columnHandles,
          fileHandleFactory,
          executor,
          connectorQueryCtx,
          cudfHiveConfig,
          fetchAheadContext,
          PartitionKeySource::kDataSource) {}

CudfHiveDataSource::CudfHiveDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    facebook::velox::FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<FetchAheadContext>& fetchAheadContext,
    PartitionKeySource partitionKeySource)
    : NvtxHelper(
          nvtx3::rgb{80, 171, 241}, // CudfHive blue,
          std::nullopt,
          fmt::format("[{}]", tableHandle->name())),
      cudfHiveConfig_(cudfHiveConfig),
      fetchAheadContext_(fetchAheadContext),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      outputType_(outputType),
      pool_(connectorQueryCtx->memoryPool()),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()),
      partitionKeySource_(partitionKeySource),
      filterFirst_(cudfHiveConfig->isFilterFirstSession(
          connectorQueryCtx->sessionProperties())) {
  tableHandle_ =
      std::dynamic_pointer_cast<const hive::HiveTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of HiveTableHandle");

  // Column handles by table column name, including those of the columns only
  // the filters read.
  std::unordered_map<
      std::string,
      const ::facebook::velox::connector::FileColumnHandle*>
      columnHandlesByName;
  for (const auto& [_, handle] : columnHandles) {
    const auto* fileColumnHandle =
        checkedPointerCast<
            const ::facebook::velox::connector::FileColumnHandle>(handle)
            .get();
    columnHandlesByName.emplace(fileColumnHandle->name(), fileColumnHandle);
  }
  for (const auto& handle : tableHandle_->hiveFilterColumnHandles()) {
    columnHandlesByName.emplace(handle->name(), handle.get());
  }

  // Returns the handle of a partition key the data source resolves, or null.
  const auto partitionKeyHandle = [&](const std::string& name)
      -> const ::facebook::velox::connector::FileColumnHandle* {
    if (partitionKeySource_ != PartitionKeySource::kDataSource) {
      return nullptr;
    }
    const auto it = columnHandlesByName.find(name);
    return it != columnHandlesByName.end() && it->second->isPartitionKey()
        ? it->second
        : nullptr;
  };
  const auto addPartitionKey =
      [&](const std::string& name,
          const TypePtr& type,
          const ::facebook::velox::connector::FileColumnHandle& handle) {
        if (partitionKeys_
                .emplace(
                    name,
                    PartitionKey{
                        type, handle.isPartitionDateValueDaysSinceEpoch()})
                .second) {
          columnTypes_.emplace(name, type);
        }
      };
  const auto materializePartitionKey = [&](const std::string& name) {
    if (std::find(partitionKeyNames_.begin(), partitionKeyNames_.end(), name) ==
        partitionKeyNames_.end()) {
      partitionKeyNames_.push_back(name);
    }
  };

  // Set up column projection if needed. A partition key is not in the data
  // file, so it is materialized per split instead of being read.
  for (uint32_t i = 0; i < outputType_->size(); ++i) {
    const auto& outputName = outputType_->nameOf(i);
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto* handle = static_cast<const hive::HiveColumnHandle*>(it->second.get());
    const auto& name = handle->name();
    outputSourceNames_.push_back(name);
    if (const auto* partitionKey = partitionKeyHandle(name)) {
      addPartitionKey(name, outputType_->childAt(i), *partitionKey);
      materializePartitionKey(name);
      continue;
    }
    columnTypes_.emplace(name, outputType_->childAt(i));
    readColumnSet_.emplace(name);
    readColumnNames_.emplace_back(name);
  }

  optimizeQueryCtx_ = makeRemainingFilterQueryCtx(*connectorQueryCtx);
  auto filters = splitAndFoldFilters(
      *tableHandle_, expressionEvaluator_, optimizeQueryCtx_.get(), pool_);
  const auto reason =
      unsupportedFiltersReason(filters, optimizeQueryCtx_.get());
  VELOX_CHECK(
      !reason.has_value(),
      "Table scan must run on the CPU data source: {}",
      *reason);
  subfieldFilters_ = std::move(filters.subfieldFilters);

  // A filter on a partition key decides for the whole split, so it is tested
  // per split instead of being compiled into the reader's filter.
  for (auto it = subfieldFilters_.begin(); it != subfieldFilters_.end();) {
    const auto& subfield = it->first;
    const auto* partitionKey = subfield.path().size() == 1
        ? partitionKeyHandle(subfield.baseName())
        : nullptr;
    if (partitionKey == nullptr) {
      ++it;
      continue;
    }
    addPartitionKey(
        subfield.baseName(), partitionKey->dataType(), *partitionKey);
    partitionKeyFilters_.emplace(subfield.clone(), std::move(it->second));
    it = subfieldFilters_.erase(it);
  }

  // Add fields in the filter to the columns to read if not there
  for (const auto& [field, _] : subfieldFilters_) {
    if (readColumnSet_.count(field.toString()) == 0) {
      readColumnSet_.emplace(field.toString());
      readColumnNames_.emplace_back(field.toString());
    }
  }
  optimizedRemainingFilter_ = std::move(filters.remainingFilter);
  if (const auto constantFilter =
          std::dynamic_pointer_cast<const core::ConstantTypedExpr>(
              optimizedRemainingFilter_)) {
    // A filter that folds to a constant keeps every row or none, so it needs
    // no columns and no evaluation.
    remainingFilterRejectsAllRows_ = !isTrueConstant(*constantFilter, pool_);
    optimizedRemainingFilter_ = nullptr;
  }
  if (optimizedRemainingFilter_) {
    // Add fields referenced by the filter to the columns to read. Collect from
    // the optimized expression since folding may drop branches and the columns
    // they reference. Read-column order does not affect results: the data
    // source projects its output to the requested output type.
    const auto filterFields = referencedInputFields(optimizedRemainingFilter_);
    for (const auto& name : filterFields) {
      // A partition key is materialized before the filter runs.
      if (const auto* partitionKey = partitionKeyHandle(name)) {
        addPartitionKey(name, partitionKey->dataType(), *partitionKey);
        materializePartitionKey(name);
        continue;
      }
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
        optimizedRemainingFilter_,
        remainingFilterType,
        pool_,
        optimizeQueryCtx_->queryConfig());
  }

  // Keep a logical-width AST for filters that must run after the Parquet read.
  // A second AST with split-specific physical widths is built after reading
  // each Parquet footer.
  if (!subfieldFilters_.empty()) {
    auto const readerFilterType = getTableRowType();
    hasDecimalSubfieldFilter_ =
        hasDecimalSubfieldFilter(subfieldFilters_, readerFilterType);
    subfieldFilterAst_ = &createAstFromSubfieldFilters(
        subfieldFilters_, subfieldTree_, subfieldScalars_, readerFilterType);
  }

  {
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    names.reserve(readColumnNames_.size());
    types.reserve(readColumnNames_.size());
    for (const auto& name : readColumnNames_) {
      names.push_back(name);
      types.push_back(columnType(name));
    }
    readerOutputType_ = ROW(std::move(names), std::move(types));
  }

  VELOX_CHECK_NOT_NULL(fileHandleFactory_, "No FileHandleFactory present");

  // Create empty IOStats and FsStats for later use
  ioStatistics_ = std::make_shared<io::IoStatistics>();
  ioStats_ = std::make_shared<facebook::velox::IoStats>();
}

std::unique_ptr<CudfSplitReader> CudfHiveDataSource::createCudfSplitReader() {
  readerFiltersCurrent_ = true;
  // The reader prunes row groups with the filters on integer columns as they
  // are now.
  common::SubfieldFilters readerFilters;
  for (const auto& [index, dynamicFilter] : dynamicFilters_) {
    if (hasIntegerStorage(columnType(readColumnNames_[index]))) {
      readerFilters.emplace(
          common::Subfield::create(readColumnNames_[index])->clone(),
          dynamicFilter.filter);
    }
  }
  return std::make_unique<CudfSplitReader>(
      split_,
      tableHandle_,
      readerOutputType_,
      readColumnNames_,
      fileHandleFactory_,
      executor_,
      connectorQueryCtx_,
      cudfHiveConfig_,
      fetchAheadContext_,
      ioStatistics_,
      ioStats_,
      subfieldFilterAst_,
      readerFilters.empty() ? nullptr : &readerFilters);
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
  for (const auto& [name, value] : hiveSplit->partitionKeys) {
    cudfHiveSplitBuilder.partitionKey(name, value);
  }
  if (hiveSplit->properties.has_value()) {
    cudfHiveSplitBuilder.fileProperties(hiveSplit->properties.value());
  }
  split_ = cudfHiveSplitBuilder.build();

  VLOG(1) << "Adding split " << split_->toString();
}

void CudfHiveDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  ensureCudaContextForThread();
  // Virtual method for class-specific conversion of the split
  convertSplit(split);
  partitionKeyScalars_.clear();
  splitExhausted_ = false;

  // A partition key filter the split fails rejects every row of it, so open
  // nothing.
  splitSkipped_ = !partitionKeyFiltersAcceptSplit();
  if (splitSkipped_) {
    cudfSplitReader_.reset();
    recordSkippedSplit();
    return;
  }

  cudfSplitReader_ = createCudfSplitReader();

  // No row of the split can pass the filter, so open nothing.
  if (remainingFilterRejectsAllRows_) {
    recordSkippedSplit();
    return;
  }

  // An empty split has no footer to build the filter from.
  const auto* pushdownFilter = subfieldFilterAst_;
  if (hasDecimalSubfieldFilter_ and
      not cudfSplitReader_->fileMetaDatas().empty()) {
    const auto& fileMetadatas = cudfSplitReader_->fileMetaDatas();
    const auto readerFilterType = getTableRowType();
    const HiveConfig hiveConfig(cudfHiveConfig_->config());
    const auto fileColumnNamesReadAsLowerCase =
        hiveConfig.isFileColumnNamesReadAsLowerCase(
            connectorQueryCtx_->sessionProperties());
    pushdownFilterTree_ = cudf::ast::tree{};
    pushdownFilterScalars_.clear();
    const auto decimalTypes = parquetDecimalTypes(
        fileMetadatas.front().schema,
        readerFilterType,
        fileColumnNamesReadAsLowerCase);
    const auto& filter = createAstFromSubfieldFilters(
        subfieldFilters_,
        pushdownFilterTree_,
        pushdownFilterScalars_,
        readerFilterType,
        &decimalTypes);
    cudfSplitReader_->setPushdownFilter(&filter);
    pushdownFilter = &filter;
  }
  if (filterFirst_ and pushdownFilter != nullptr) {
    std::vector<std::string> filterColumns;
    filterColumns.reserve(subfieldFilters_.size());
    for (const auto& [field, _] : subfieldFilters_) {
      filterColumns.push_back(field.toString());
    }
    cudfSplitReader_->setFilterFirstFilter(
        pushdownFilter, std::move(filterColumns));
  }
  cudfSplitReader_->prepareSplit(runtimeStats_);

  // An ignored missing file leaves nothing to read.
  if (cudfSplitReader_->emptySplit()) {
    return;
  }

  // Parse the split's partition values once, on the stream next() expands
  // them into columns on.
  partitionKeyScalars_ = makePartitionKeyScalars(cudfSplitReader_->stream());

  // A preloaded split offers the first pass of its column chunks to the
  // fetch-ahead budget, or, without fetch-ahead, starts fetching them when
  // asked to. The offer is the last use of the reader on this thread: from
  // here on the budget may start the fetch on another one.
  const bool isPreloadedSplit = split->dataSource != nullptr;
  if (isPreloadedSplit && !cudfSplitReader_->offerColumnChunkFetch() &&
      cudfHiveConfig_->preloadColumnChunksSession(
          connectorQueryCtx_->sessionProperties())) {
    cudfSplitReader_->startColumnChunkFetch();
  }

  // TODO: `completedBytes_` should be updated in `next()` as we read more and
  // more table bytes
  if (const auto fileSize = split_->fileSize()) {
    completedBytes_ += static_cast<size_t>(*fileSize);
    return;
  }
  // The split does not carry the size, so open the file for it. On S3 the
  // open costs a HEAD request.
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

void CudfHiveDataSource::setFromDataSource(std::unique_ptr<DataSource> source) {
  ensureCudaContextForThread();
  auto* preparedSource = checkedPointerCast<CudfHiveDataSource>(source.get());

  split_ = std::move(preparedSource->split_);
  splitSkipped_ = preparedSource->splitSkipped_;
  splitExhausted_ = false;
  runtimeStats_.skippedSplits += preparedSource->runtimeStats_.skippedSplits;
  runtimeStats_.processedSplits +=
      preparedSource->runtimeStats_.processedSplits;
  runtimeStats_.skippedSplitBytes +=
      preparedSource->runtimeStats_.skippedSplitBytes;
  runtimeStats_.skippedStrides += preparedSource->runtimeStats_.skippedStrides;
  runtimeStats_.processedStrides +=
      preparedSource->runtimeStats_.processedStrides;
  completedBytes_ += preparedSource->completedBytes_;
  completedRows_ += preparedSource->completedRows_;

  // Drop the reader of the previous split before replacing the AST storage it
  // references below.
  cudfSplitReader_.reset();

  // The reader's Parquet filter references the AST expressions and literal
  // scalars owned by 'source', which is freed right after this call. Adopt that
  // storage so the reader keeps pointing at live expressions; the nodes are
  // heap-allocated, so moving the owners does not move the expressions.
  subfieldScalars_ = std::move(preparedSource->subfieldScalars_);
  subfieldTree_ = std::move(preparedSource->subfieldTree_);
  subfieldFilterAst_ = preparedSource->subfieldFilterAst_;
  pushdownFilterScalars_ = std::move(preparedSource->pushdownFilterScalars_);
  pushdownFilterTree_ = std::move(preparedSource->pushdownFilterTree_);

  cudfSplitReader_ = std::move(preparedSource->cudfSplitReader_);
  partitionKeyScalars_ = std::move(preparedSource->partitionKeyScalars_);
  // The split was prepared with the dynamic filters of that time.
  readerFiltersCurrent_ = false;
  if (!splitSkipped_ && !partitionKeyFiltersAcceptSplit()) {
    skipCurrentSplit();
  }

  // A split a partition key filter rejected has no reader.
  if (!splitSkipped_) {
    VELOX_CHECK_NOT_NULL(cudfSplitReader_);

    // 'source' owns the query context the reader was prepared with and is
    // freed right after this call.
    cudfSplitReader_->setConnectorQueryCtx(connectorQueryCtx_);

    // Start column chunk fetch if it is not already started
    cudfSplitReader_->startColumnChunkFetch();
  }

  // The adopted reader keeps writing I/O statistics to the objects of
  // 'source', so carry the balance accumulated here over to those.
  preparedSource->ioStatistics_->merge(*ioStatistics_);
  ioStatistics_ = std::move(preparedSource->ioStatistics_);
  preparedSource->ioStats_->merge(*ioStats_);
  ioStats_ = std::move(preparedSource->ioStats_);
}

void CudfHiveDataSource::addDynamicFilter(
    column_index_t outputChannel,
    const std::shared_ptr<common::Filter>& filter) {
  VELOX_CHECK_NOT_NULL(filter);
  VELOX_CHECK_LT(outputChannel, outputSourceNames_.size());
  const auto& name = outputSourceNames_[outputChannel];

  // The filter replaces the one the column had: the sender merges every
  // filter of the column before sending.
  if (partitionKeys_.contains(name)) {
    dynamicPartitionKeyFilters_.insert_or_assign(
        common::Subfield::create(name)->clone(), filter);
    if (split_ != nullptr && !splitSkipped_ && !splitExhausted_ &&
        !partitionKeyFiltersAcceptSplit()) {
      skipCurrentSplit();
    }
    return;
  }

  const auto readIndex = static_cast<size_t>(std::distance(
      readColumnNames_.begin(),
      std::find(readColumnNames_.begin(), readColumnNames_.end(), name)));
  VELOX_CHECK_LT(
      readIndex,
      readColumnNames_.size(),
      "Dynamic filter is on a column the scan does not read: {}",
      name);
  // The sender repeats the filters that did not change when another column
  // gets one.
  if (const auto it = dynamicFilters_.find(readIndex);
      it != dynamicFilters_.end() && it->second.filter == filter) {
    return;
  }
  dynamicFilters_.insert_or_assign(readIndex, makeDynamicFilter(name, filter));
  ++dynamicFilterVersion_;
  readerFiltersCurrent_ = false;
  // What the advisory filters kept so far says nothing about the new set.
  advisoryFilterInputRows_ = 0;
  advisoryFilterOutputRows_ = 0;
  advisoryRowFilterDisabled_ = false;
}

CudfHiveDataSource::DynamicFilter CudfHiveDataSource::makeDynamicFilter(
    const std::string& name,
    const common::FilterPtr& filter) const {
  DynamicFilter dynamicFilter;
  dynamicFilter.filter = filter;
  dynamicFilter.advisory = isAdvisoryFilter(filter);
  if (filter->kind() == common::FilterKind::kAlwaysFalse) {
    dynamicFilter.evaluation = DynamicFilterEvaluation::kAlwaysFalse;
    return dynamicFilter;
  }
  if (!hasIntegerStorage(columnType(name))) {
    dynamicFilter.evaluation = DynamicFilterEvaluation::kHost;
    return dynamicFilter;
  }
  switch (filter->kind()) {
    case common::FilterKind::kBigintRange:
      dynamicFilter.evaluation = advisoryFilterBitmap(filter) != nullptr
          ? DynamicFilterEvaluation::kDeviceBitmap
          : DynamicFilterEvaluation::kRange;
      break;
    case common::FilterKind::kBigintValuesUsingHashTable: {
      const auto& values =
          static_cast<const common::BigintValuesUsingHashTable&>(*filter);
      dynamicFilter.evaluation = DynamicFilterEvaluation::kValues;
      dynamicFilter.lower = values.min();
      dynamicFilter.upper = values.max();
      break;
    }
    case common::FilterKind::kBigintValuesUsingBitmask: {
      const auto& values =
          static_cast<const common::BigintValuesUsingBitmask&>(*filter);
      dynamicFilter.evaluation = DynamicFilterEvaluation::kValues;
      dynamicFilter.lower = values.min();
      dynamicFilter.upper = values.max();
      break;
    }
    default:
      dynamicFilter.evaluation = DynamicFilterEvaluation::kHost;
      break;
  }
  return dynamicFilter;
}

void CudfHiveDataSource::skipCurrentSplit() {
  splitSkipped_ = true;
  cudfSplitReader_.reset();
  recordSkippedSplit();
}

void CudfHiveDataSource::uploadDynamicFilterValues(
    DynamicFilter& dynamicFilter,
    const cudf::column_view& input,
    cuda::stream_ref stream) {
  dynamicFilter.hashSet.reset();
  dynamicFilter.deviceValues.reset();
  dynamicFilter.bitmap.reset();
  const auto span = static_cast<uint64_t>(dynamicFilter.upper) -
      static_cast<uint64_t>(dynamicFilter.lower);
  if (span / 32 < kMaxBitmapWords) {
    bool built;
    dynamicFilter.bitmap = sharedValueBitmap(
        dynamicFilter.filter,
        dynamicFilter.lower,
        dynamicFilter.upper,
        stream,
        built);
    numDynamicFilterValueUploads_ += built;
    return;
  }
  ++numDynamicFilterValueUploads_;

  const auto values = sortedFilterValues(*dynamicFilter.filter);
  dynamicFilter.deviceValues = cudf::type_dispatcher(
      input.type(), [&]<typename T>() -> std::unique_ptr<cudf::column> {
        if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
          // A value outside of the column's type matches no row.
          std::vector<T> typedValues;
          typedValues.reserve(values.size());
          for (const auto value : values) {
            if (value >= std::numeric_limits<T>::min() &&
                value <= std::numeric_limits<T>::max()) {
              typedValues.push_back(static_cast<T>(value));
            }
          }
          auto column = cudf::make_fixed_width_column(
              input.type(),
              typedValues.size(),
              cudf::mask_state::UNALLOCATED,
              stream,
              get_temp_mr());
          if (!typedValues.empty()) {
            CUDF_CUDA_TRY(cudaMemcpyAsync(
                column->mutable_view().template data<T>(),
                typedValues.data(),
                typedValues.size() * sizeof(T),
                cudaMemcpyHostToDevice,
                stream.get()));
            // 'typedValues' has to outlive the copy.
            stream.sync();
          }
          using Key =
              std::conditional_t<(sizeof(T) < sizeof(int32_t)), int32_t, T>;
          // The hash set needs a key that is not in it, also when the
          // smallest key of the type is a value.
          int64_t emptyKey = std::numeric_limits<Key>::min();
          for (const auto value : typedValues) {
            if (value < emptyKey) {
              continue;
            }
            if (value != emptyKey) {
              break;
            }
            ++emptyKey;
          }
          dynamicFilter.hashSet = std::make_unique<CudfIntegerHashSet>(
              column->view(), emptyKey, stream, get_temp_mr());
          dynamicFilter.hashSetType = input.type().id();
          return column;
        } else {
          VELOX_UNSUPPORTED("Dynamic filter requires a signed integer column");
        }
      });
}

std::vector<CudfHiveDataSource::DynamicFilterBounds>
CudfHiveDataSource::dynamicFilterBounds() const {
  std::vector<DynamicFilterBounds> bounds;
  for (const auto& [index, dynamicFilter] : dynamicFilters_) {
    switch (dynamicFilter.evaluation) {
      case DynamicFilterEvaluation::kRange:
      case DynamicFilterEvaluation::kDeviceBitmap: {
        const auto& range =
            static_cast<const common::BigintRange&>(*dynamicFilter.filter);
        bounds.push_back(
            {index, range.lower(), range.upper(), range.testNull()});
        break;
      }
      case DynamicFilterEvaluation::kValues:
        bounds.push_back(
            {index,
             dynamicFilter.lower,
             dynamicFilter.upper,
             dynamicFilter.filter->testNull()});
        break;
      default:
        break;
    }
  }
  return bounds;
}

bool CudfHiveDataSource::appliesDynamicFilter(
    const DynamicFilter& dynamicFilter) const {
  return !dynamicFilter.advisory ||
      (!advisoryRowFilterDisabled_ &&
       CudfConfig::getInstance().dynamicFilterRowFilterEnabled);
}

std::vector<size_t> CudfHiveDataSource::dynamicFilterColumns() const {
  std::vector<size_t> columns;
  for (const auto& [index, dynamicFilter] : dynamicFilters_) {
    if (appliesDynamicFilter(dynamicFilter)) {
      columns.push_back(index);
    }
  }
  return columns;
}

bool CudfHiveDataSource::intersectDynamicFilters(
    const std::function<cudf::column_view(size_t)>& readColumn,
    cudf::size_type numRows,
    std::unique_ptr<cudf::column>& mask,
    cuda::stream_ref stream) {
  if (std::ranges::none_of(dynamicFilters_, [&](const auto& entry) {
        return appliesDynamicFilter(entry.second);
      })) {
    return false;
  }
  const auto integerColumn = [&](size_t index) {
    const auto column = integerStorageView(readColumn(index));
    VELOX_USER_CHECK(
        cudf::is_integral(column.type()) && cudf::is_signed(column.type()),
        "cuDF Hive scan cannot apply a dynamic filter to the type the file stores column {} in",
        readColumnNames_[index]);
    return column;
  };

  if (std::ranges::any_of(dynamicFilters_, [&](const auto& entry) {
        return entry.second.evaluation ==
            DynamicFilterEvaluation::kAlwaysFalse &&
            appliesDynamicFilter(entry.second);
      })) {
    const cudf::numeric_scalar<bool> none(false, true, stream, get_temp_mr());
    mask = cudf::make_column_from_scalar(none, numRows, stream, get_temp_mr());
    return true;
  }

  // Host-evaluated filters first: a mask made of them is what the device
  // filters narrow.
  std::vector<uint8_t> hostPass;
  for (const auto& [index, dynamicFilter] : dynamicFilters_) {
    if (dynamicFilter.evaluation != DynamicFilterEvaluation::kHost ||
        !appliesDynamicFilter(dynamicFilter)) {
      continue;
    }
    if (hostPass.empty()) {
      hostPass.assign(numRows, 1);
    }
    if (const auto& type = columnType(readColumnNames_[index]);
        hasIntegerStorage(type)) {
      intersectOnHost(
          integerColumn(index), *dynamicFilter.filter, hostPass, stream);
    } else {
      intersectOnHost(
          readColumn(index),
          readColumnNames_[index],
          type,
          *dynamicFilter.filter,
          hostPass,
          pool_,
          stream);
    }
  }
  if (!hostPass.empty()) {
    auto hostMask = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::BOOL8},
        numRows,
        cudf::mask_state::UNALLOCATED,
        stream,
        get_temp_mr());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostMask->mutable_view().data<uint8_t>(),
        hostPass.data(),
        hostPass.size(),
        cudaMemcpyHostToDevice,
        stream.get()));
    // 'hostPass' has to outlive the copy.
    stream.sync();
    mask = mask == nullptr ? std::move(hostMask)
                           : cudf::binary_operation(
                                 mask->view(),
                                 hostMask->view(),
                                 cudf::binary_operator::LOGICAL_AND,
                                 cudf::data_type{cudf::type_id::BOOL8},
                                 stream,
                                 get_temp_mr());
  }

  for (auto& [index, dynamicFilter] : dynamicFilters_) {
    if (!appliesDynamicFilter(dynamicFilter)) {
      continue;
    }
    if (dynamicFilter.evaluation == DynamicFilterEvaluation::kRange) {
      const auto& range =
          static_cast<const common::BigintRange&>(*dynamicFilter.filter);
      applyIntegerRangeToMask(
          integerColumn(index),
          range.lower(),
          range.upper(),
          range.testNull(),
          mask,
          stream,
          get_temp_mr());
    } else if (
        dynamicFilter.evaluation == DynamicFilterEvaluation::kDeviceBitmap) {
      const auto& range =
          static_cast<const common::BigintRange&>(*dynamicFilter.filter);
      applyIntegerBitmapToMask(
          integerColumn(index),
          advisoryFilterBitmap(dynamicFilter.filter)->view(),
          range.lower(),
          range.testNull(),
          mask,
          stream,
          get_temp_mr());
    } else if (dynamicFilter.evaluation == DynamicFilterEvaluation::kValues) {
      const auto input = integerColumn(index);
      // A hash set holds keys of one integer width; another file of the
      // scan may store the column in another.
      if ((dynamicFilter.bitmap == nullptr &&
           dynamicFilter.hashSet == nullptr) ||
          (dynamicFilter.hashSet != nullptr &&
           dynamicFilter.hashSetType != input.type().id())) {
        uploadDynamicFilterValues(dynamicFilter, input, stream);
      }
      if (dynamicFilter.hashSet != nullptr) {
        dynamicFilter.hashSet->apply(
            input,
            dynamicFilter.filter->testNull(),
            mask,
            stream,
            get_temp_mr());
      } else {
        applyIntegerBitmapToMask(
            input,
            dynamicFilter.bitmap->view(),
            dynamicFilter.lower,
            dynamicFilter.filter->testNull(),
            mask,
            stream,
            get_temp_mr());
      }
    }
  }
  return true;
}

void CudfHiveDataSource::recordDynamicFilterRows(
    uint64_t numInputRows,
    uint64_t numRetainedRows) {
  dynamicFilterDroppedRows_ += numInputRows - numRetainedRows;
  const bool hasRequired = std::ranges::any_of(
      dynamicFilters_,
      [](const auto& entry) { return !entry.second.advisory; });
  if (hasRequired) {
    return;
  }
  advisoryFilterInputRows_ += numInputRows;
  advisoryFilterOutputRows_ += numRetainedRows;
  if (advisoryFilterInputRows_ >= kAdvisoryFilterMinRows &&
      advisoryFilterOutputRows_ >
          kAdvisoryFilterMaxKeptRatio * advisoryFilterInputRows_) {
    advisoryRowFilterDisabled_ = true;
  }
}

void CudfHiveDataSource::applyDynamicFilters(
    std::unique_ptr<cudf::table>& table,
    cuda::stream_ref stream) {
  // The footer statistics of the row groups the reader kept already prove
  // that every row passes. They say nothing about a column that is not read
  // as integers.
  const bool allIntegerColumns =
      std::ranges::all_of(dynamicFilters_, [&](const auto& entry) {
        return hasIntegerStorage(columnType(readColumnNames_[entry.first]));
      });
  if (readerFiltersCurrent_ && allIntegerColumns &&
      cudfSplitReader_->dynamicFiltersSatisfied()) {
    return;
  }

  const auto numInputRows = table->num_rows();
  std::unique_ptr<cudf::column> retentionMask;
  if (!intersectDynamicFilters(
          [&](size_t index) { return table->get_column(index).view(); },
          numInputRows,
          retentionMask,
          stream)) {
    return;
  }
  const auto numRetained = countRetainedRows(retentionMask->view(), stream);
  if (numRetained == 0) {
    table = cudf::empty_like(table->view());
  } else if (numRetained < numInputRows) {
    table = cudf::apply_retention_mask(
        *table, retentionMask->view(), stream, get_output_mr());
  }
  recordDynamicFilterRows(numInputRows, numRetained);
}

std::optional<RowVectorPtr> CudfHiveDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /* future */) {
  VELOX_CHECK_NOT_NULL(split_, "No split present. Call addSplit() first.");
  if (splitSkipped_) {
    return nullptr;
  }
  VELOX_CHECK_NOT_NULL(cudfSplitReader_, "No split to process.");
  if (remainingFilterRejectsAllRows_ or cudfSplitReader_->emptySplit()) {
    cudfSplitReader_->resetSplit();
    return nullptr;
  }
  ensureCudaContextForThread();
  auto chunkOpt = cudfSplitReader_->next(size, {});
  const auto readStats = cudfSplitReader_->takeReadStats();
  readStats_.numFilterFirstPasses += readStats.numFilterFirstPasses;
  readStats_.numOneStepPasses += readStats.numOneStepPasses;
  readStats_.numFilterFirstRows += readStats.numFilterFirstRows;
  readStats_.numFilterFirstKeptRows += readStats.numFilterFirstKeptRows;
  if (!chunkOpt.has_value()) {
    cudfSplitReader_->resetSplit();
    splitExhausted_ = true;
    return nullptr;
  }
  auto nRows = chunkOpt.value().numRows;
  auto cudfTable = std::move(chunkOpt.value().table);
  auto stream = cudfSplitReader_->stream();
  if (!dynamicFilters_.empty() && cudfTable->num_rows() > 0) {
    applyDynamicFilters(cudfTable, stream);
    nRows = cudfTable->num_rows();
  }

  if (!partitionKeyNames_.empty()) {
    // The partition key columns follow the read columns, the layout the
    // remaining filter was compiled against. They are temporaries when that
    // filter copies the rows it keeps.
    auto columns = cudfTable->release();
    VELOX_CHECK_EQ(
        columns.size(),
        readColumnNames_.size(),
        "cuDF table scan read an unexpected number of columns");
    auto partitionKeyColumns = makePartitionKeyColumns(
        nRows,
        stream,
        optimizedRemainingFilter_ ? get_temp_mr() : get_output_mr());
    std::move(
        partitionKeyColumns.begin(),
        partitionKeyColumns.end(),
        std::back_inserter(columns));
    cudfTable = std::make_unique<cudf::table>(std::move(columns));
  }

  uint64_t filterTimeUs{0};
  if (optimizedRemainingFilter_) {
    MicrosecondWallTimer filterTimer(&filterTimeUs);
    auto cudfTableColumns = cudfTable->release();
    std::vector<cudf::column_view> inputViews;
    inputViews.reserve(cudfTableColumns.size());
    for (auto& col : cudfTableColumns) {
      inputViews.push_back(col->view());
    }
    auto filterResult =
        cudfRemainingFilterExpression_->eval(inputViews, stream, get_temp_mr());
    auto originalTable =
        std::make_unique<cudf::table>(std::move(cudfTableColumns));
    cudfTable = cudf::apply_retention_mask(
        *originalTable, asView(filterResult), stream, get_output_mr());
    nRows = cudfTable->num_rows();
  }
  totalRemainingFilterTime_.fetch_add(
      filterTimeUs * 1000, std::memory_order_relaxed);

  if (!partitionKeyNames_.empty()) {
    // Project to the output columns by name: the table holds the read columns
    // and then the partition keys, while the output interleaves them. A
    // partition key may be projected under several names.
    auto columns = cudfTable->release();
    std::unordered_map<std::string, size_t> columnIndices;
    for (size_t i = 0; i < readColumnNames_.size(); ++i) {
      columnIndices.emplace(readColumnNames_[i], i);
    }
    for (size_t i = 0; i < partitionKeyNames_.size(); ++i) {
      columnIndices.emplace(partitionKeyNames_[i], readColumnNames_.size() + i);
    }
    std::vector<size_t> outputIndices;
    outputIndices.reserve(outputSourceNames_.size());
    std::vector<size_t> remainingUses(columns.size(), 0);
    for (const auto& name : outputSourceNames_) {
      const auto it = columnIndices.find(name);
      VELOX_CHECK(
          it != columnIndices.end(),
          "Output column {} was neither read nor materialized",
          name);
      outputIndices.push_back(it->second);
      ++remainingUses[it->second];
    }
    std::vector<std::unique_ptr<cudf::column>> outputColumns;
    outputColumns.reserve(outputIndices.size());
    for (const auto index : outputIndices) {
      auto& column = columns[index];
      outputColumns.push_back(
          --remainingUses[index] == 0 ? std::move(column)
                                      : std::make_unique<cudf::column>(
                                            *column, stream, get_output_mr()));
    }
    cudfTable = std::make_unique<cudf::table>(std::move(outputColumns));
  } else if (outputType_->size() < cudfTable->num_columns()) {
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
  RowVectorPtr output;
  if (cudfIsRegistered()) {
    output = std::make_shared<CudfVector>(
        pool_, outputType_, nRows, std::move(cudfTable), stream);
  } else if (cudfTable->num_columns() == 0) {
    // There is no device data to convert, only the row count.
    output = std::make_shared<RowVector>(
        pool_, outputType_, nullptr, nRows, std::vector<VectorPtr>{});
  } else {
    output = with_arrow::toVeloxColumn(
        cudfTable->view(), pool_, outputType_, stream, get_temp_mr());
  }
  stream.sync();

  VELOX_CHECK_NOT_NULL(output, "Cudf to Velox conversion yielded a nullptr");

  completedRows_ += output->size();

  // TODO: Update `completedBytes_` here instead of in `addSplit()`

  return output;
}

std::unordered_map<std::string, RuntimeMetric>
CudfHiveDataSource::getRuntimeStats() {
  auto result = runtimeStats_.toRuntimeMetricMap();
  io::addIoStatsToRuntimeStats(*ioStatistics_, "", result);
  if (const auto it = result.find(std::string(io::kStorageReadBytes));
      it != result.end()) {
    // Preserve the DWIO value before a ReadFile-layer value overrides it.
    // Overread bytes are defined relative to this counter.
    result.emplace(kDwioStorageReadBytes, it->second);
  }
  // Preserve a zero-valued totalScanTime before scan timing is recorded.
  result.insert({
      {std::string(io::kTotalScanTime),
       RuntimeMetric(
           ioStatistics_->totalScanTimeNs(), RuntimeCounter::Unit::kNanos)},
      {std::string(Connector::kTotalRemainingFilterTime),
       RuntimeMetric(
           totalRemainingFilterTime_.load(std::memory_order_relaxed),
           RuntimeCounter::Unit::kNanos)},
  });
  if (dynamicFilterDroppedRows_ > 0) {
    result.emplace(
        kDynamicFilterDroppedRows, RuntimeMetric(dynamicFilterDroppedRows_));
  }
  if (numDynamicFilterValueUploads_ > 0) {
    result.emplace(
        kDynamicFilterValueUploads,
        RuntimeMetric(numDynamicFilterValueUploads_));
  }
  if (filterFirst_) {
    // Reported also when zero, so that a scan that never read filter columns
    // first shows it.
    result.insert({
        {std::string(kFilterFirstPasses),
         RuntimeMetric(readStats_.numFilterFirstPasses)},
        {std::string(kFilterFirstOneStepPasses),
         RuntimeMetric(readStats_.numOneStepPasses)},
        {std::string(kFilterFirstRows),
         RuntimeMetric(readStats_.numFilterFirstRows)},
        {std::string(kFilterFirstKeptRows),
         RuntimeMetric(readStats_.numFilterFirstKeptRows)},
    });
  }
  const auto& ioStats = ioStats_->stats();
  for (const auto& [key, value] : ioStats) {
    // Keep the ReadFile-layer value under the established key.
    if (key == io::kStorageReadBytes) {
      result[std::string(key)] = value;
    } else {
      result.emplace(key, value);
    }
  }
  return result;
}

const TypePtr& CudfHiveDataSource::columnType(const std::string& name) const {
  if (const auto it = columnTypes_.find(name); it != columnTypes_.end()) {
    return it->second;
  }
  const auto& dataColumns = tableHandle_->dataColumns();
  VELOX_USER_CHECK_NOT_NULL(
      dataColumns,
      "Table schema is required to resolve the type of filter-only column {}",
      name);
  return dataColumns->findChild(name);
}

std::optional<std::string> CudfHiveDataSource::partitionKeyValue(
    const std::string& name) const {
  const auto it = split_->partitionKeys.find(name);
  return it == split_->partitionKeys.end() ? std::nullopt : it->second;
}

bool CudfHiveDataSource::readTimestampPartitionValueAsLocalTime() const {
  const HiveConfig hiveConfig(cudfHiveConfig_->config());
  return hiveConfig.readTimestampPartitionValueAsLocalTime(
      connectorQueryCtx_->sessionProperties());
}

bool CudfHiveDataSource::partitionKeyFiltersAcceptSplit() const {
  if (partitionKeyFilters_.empty() && dynamicPartitionKeyFilters_.empty()) {
    return true;
  }
  const bool timestampValueAsLocalTime =
      readTimestampPartitionValueAsLocalTime();
  for (const auto* filters :
       {&partitionKeyFilters_, &dynamicPartitionKeyFilters_}) {
    for (const auto& [subfield, filter] : *filters) {
      const auto& name = subfield.baseName();
      const auto& partitionKey = partitionKeys_.at(name);
      if (!testPartitionKeyFilter(
              *filter,
              partitionKey.type,
              partitionKeyValue(name),
              partitionKey.dateValueDaysSinceEpoch,
              timestampValueAsLocalTime)) {
        VLOG(1) << "Skipping split " << split_->toString()
                << " because the filter on partition key " << name
                << " rejects its value";
        return false;
      }
    }
  }
  return true;
}

std::vector<std::unique_ptr<cudf::scalar>>
CudfHiveDataSource::makePartitionKeyScalars(cuda::stream_ref stream) const {
  const bool timestampValueAsLocalTime =
      readTimestampPartitionValueAsLocalTime();
  // A TIMESTAMP key takes the type the split reader gives file timestamps
  // rather than the unit cuDF expressions default to, so that an expression
  // can combine the two.
  const auto timestampType = constantTimestampType(
      *cudfHiveConfig_, connectorQueryCtx_->sessionProperties());
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  scalars.reserve(partitionKeyNames_.size());
  for (const auto& name : partitionKeyNames_) {
    const auto& partitionKey = partitionKeys_.at(name);
    const auto constant = ::facebook::velox::connector::newConstantFromString(
        partitionKey.type,
        partitionKeyValue(name),
        pool_,
        timestampValueAsLocalTime,
        partitionKey.dateValueDaysSinceEpoch);
    scalars.push_back(makeScalarFromConstantExpr(
        std::make_shared<core::ConstantTypedExpr>(constant),
        pool_,
        partitionKey.type->isTimestamp() ? timestampType : std::nullopt,
        stream));
  }
  return scalars;
}

std::vector<std::unique_ptr<cudf::column>>
CudfHiveDataSource::makePartitionKeyColumns(
    cudf::size_type numRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK_EQ(
      partitionKeyScalars_.size(),
      partitionKeyNames_.size(),
      "The partition key scalars of the split are missing");
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(partitionKeyScalars_.size());
  for (const auto& scalar : partitionKeyScalars_) {
    columns.push_back(
        cudf::make_column_from_scalar(*scalar, numRows, stream, mr));
  }
  return columns;
}

void CudfHiveDataSource::recordSkippedSplit() {
  runtimeStats_.skippedSplits++;
  // An unbounded length means the whole file, whose size the split does not
  // carry, so it contributes no byte count.
  if (split_->length != std::numeric_limits<uint64_t>::max()) {
    runtimeStats_.skippedSplitBytes += static_cast<int64_t>(split_->length);
  }
}

const RowTypePtr CudfHiveDataSource::getTableRowType() {
  if (cachedTableRowType_) {
    return cachedTableRowType_;
  }
  // The read columns and then the materialized partition keys: the layout of
  // the table the remaining filter is evaluated on.
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(readColumnNames_.size() + partitionKeyNames_.size());
  types.reserve(readColumnNames_.size() + partitionKeyNames_.size());
  for (const auto& name : readColumnNames_) {
    names.push_back(name);
    types.push_back(columnType(name));
  }
  for (const auto& name : partitionKeyNames_) {
    names.push_back(name);
    types.push_back(partitionKeys_.at(name).type);
  }
  cachedTableRowType_ = ROW(std::move(names), std::move(types));
  return cachedTableRowType_;
}

} // namespace facebook::velox::cudf_velox::connector::hive
