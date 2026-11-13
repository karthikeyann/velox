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

#pragma once

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfIntegerMembership.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/base/RandomUtil.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/type/Type.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>

#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;

class CudfHiveDataSource : public DataSource, public NvtxHelper {
 public:
  /// DWIO bytes read, preserved separately from ReadFile bytes.
  static constexpr std::string_view kDwioStorageReadBytes{
      "dwio.storageReadBytes"};

  /// Runtime statistic: rows the scan dropped because a dynamic filter
  /// rejected them.
  static constexpr std::string_view kDynamicFilterDroppedRows{
      "dynamicFilterDroppedRows"};

  /// Runtime statistic: times the scan listed the values of a value-set
  /// dynamic filter and copied them to the device. The data sources of a
  /// scan share the copy, so this counts once per filter, not per split.
  static constexpr std::string_view kDynamicFilterValueUploads{
      "dynamicFilterValueUploads"};

  /// Runtime statistics of a scan with 'parquet.reader.filter-first' on: row
  /// group passes read filter columns first, and passes read with all columns
  /// in one step because the split had no filter or could not be read that
  /// way.
  static constexpr std::string_view kFilterFirstPasses{"filterFirstPasses"};
  static constexpr std::string_view kFilterFirstOneStepPasses{
      "filterFirstOneStepPasses"};

  /// Runtime statistics of a scan with 'parquet.reader.filter-first' on: rows
  /// of the passes read filter columns first, and the rows of those the
  /// filters kept, whose other columns were then read.
  static constexpr std::string_view kFilterFirstRows{"filterFirstRows"};
  static constexpr std::string_view kFilterFirstKeptRows{"filterFirstKeptRows"};

  /// Returns whether this data source can scan 'tableHandle' on the GPU:
  ///  - 'tableHandle' is a Hive table handle,
  ///  - no rows are sampled,
  ///  - every subfield filter is on a top-level column the cuDF AST converts,
  ///  - the remaining filter folds to a constant or references a column, and
  ///  - cuDF evaluates the remaining filter.
  /// The filters are split and folded with 'connectorQueryCtx', and the
  /// constructor fails for a scan this rejects.
  /// NOTE: A remaining filter that casts from VARCHAR, or from TIMESTAMP to
  /// VARCHAR, is not supported, since such a cast depends on session settings
  /// a connector does not see.
  /// NOTE: Partition key filters are checked too, although the Hive data source
  /// tests them per split instead of compiling them.
  static bool isSupported(
      const ConnectorTableHandlePtr& tableHandle,
      const ConnectorQueryCtx& connectorQueryCtx);

  /// Returns why isSupported() rejects 'tableHandle', or std::nullopt if it
  /// accepts it.
  static std::optional<std::string> unsupportedReason(
      const ConnectorTableHandlePtr& tableHandle,
      const ConnectorQueryCtx& connectorQueryCtx);

  CudfHiveDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      facebook::velox::FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& CudfHiveConfig,
      const std::shared_ptr<FetchAheadContext>& fetchAheadContext);

  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  void setFromDataSource(std::unique_ptr<DataSource> source) override;

  /// Applies 'filter' to the rows of 'outputChannel' from the next batch on,
  /// and to the row groups of the splits opened from now on. Every row the
  /// filter rejects is dropped, unless the filter is marked advisory (see
  /// CudfDynamicFilter.h). May be called before, between and during splits.
  ///
  /// The filter has to be of a kind that tests the column's type, e.g. a
  /// Bigint* filter on an integer, DATE or short DECIMAL column and a Bytes*
  /// filter on a VARCHAR column. Any other filter, and a file that stores the
  /// column in another type than the table's, fails the query at the next
  /// batch.
  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<facebook::velox::common::Filter>& filter) override;

  /// The values a dynamic filter on an integer or DATE read column lets
  /// through lie in ['lower', 'upper'].
  struct DynamicFilterBounds {
    /// Index of the column among the read columns.
    size_t readColumn;
    int64_t lower;
    int64_t upper;
    /// Whether the filter passes null.
    bool nullAllowed;
  };

  /// Changes whenever the dynamic filters on the read columns change, for a
  /// caller that remembers what it did with them.
  uint64_t dynamicFilterVersion() const {
    return dynamicFilterVersion_;
  }

  /// Returns the bounds of the dynamic filters that have them, for a reader
  /// that prunes with a filter expression of its own.
  std::vector<DynamicFilterBounds> dynamicFilterBounds() const;

  /// Returns the read columns intersectDynamicFilters() would test now, in
  /// ascending order.
  std::vector<size_t> dynamicFilterColumns() const;

  /// Clears in 'mask' the rows that fail a dynamic filter. 'readColumn'
  /// returns the decoded read column of an index from dynamicFilterColumns(),
  /// holding 'numRows' rows. 'mask' is a BOOL8 column without nulls of
  /// 'numRows' rows; if null, it is created. Returns false, leaving 'mask'
  /// as it is, if no dynamic filter is to be tested on rows now. Runs on
  /// 'stream'.
  bool intersectDynamicFilters(
      const std::function<cudf::column_view(size_t)>& readColumn,
      cudf::size_type numRows,
      std::unique_ptr<cudf::column>& mask,
      cuda::stream_ref stream);

  /// Tells the data source that a mask from intersectDynamicFilters() kept
  /// 'numRetainedRows' of 'numInputRows' rows. Feeds the statistics and the
  /// decision to stop testing advisory filters on rows.
  void recordDynamicFilterRows(uint64_t numInputRows, uint64_t numRetainedRows);

  std::optional<RowVectorPtr> next(
      uint64_t size,
      velox::ContinueFuture& /* future */) override;

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  const common::SubfieldFilters* getFilters() const override {
    return &subfieldFilters_;
  }

  uint64_t getCompletedBytes() override {
    return completedBytes_;
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

 protected:
  // Indicates the component that turns the partition keys of a split into
  // columns.
  enum class PartitionKeySource {
    // Partition key columns are left out of the split reader's projection:
    // pushed-down filters on them are tested once per split against the
    // split's partition value, and the columns are materialized as constants
    // after the read.
    kDataSource,
    // Partition key columns stay in the split reader's projection, for table
    // formats whose readers resolve them per data file.
    kSplitReader,
  };

  // Same as the public constructor, with 'partitionKeySource' choosing which
  // component resolves partition keys.
  CudfHiveDataSource(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      facebook::velox::FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
      const std::shared_ptr<FetchAheadContext>& fetchAheadContext,
      PartitionKeySource partitionKeySource);

  // Virtual method to create a `CudfSplitReader` or subclass for the data
  // source.
  virtual std::unique_ptr<CudfSplitReader> createCudfSplitReader();

  // Virtual method to convert the input `ConnectorSplit` to appropriate
  // subclass(es).
  virtual void convertSplit(std::shared_ptr<ConnectorSplit> split);

  std::shared_ptr<CudfHiveConnectorSplit> split_;
  std::shared_ptr<const ::facebook::velox::connector::hive::HiveTableHandle>
      tableHandle_;

  const std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  // The connector's fetch-ahead state, or null.
  const std::shared_ptr<FetchAheadContext> fetchAheadContext_;
  facebook::velox::FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const executor_;
  const ConnectorQueryCtx* const connectorQueryCtx_;

  // Carries the session settings the remaining filter is folded and compiled
  // against. Kept for the data source's lifetime: createCudfExpression borrows
  // its config, and an evaluator may hold on to it.
  std::shared_ptr<core::QueryCtx> optimizeQueryCtx_;

  // Columns to read.
  std::vector<std::string> readColumnNames_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<velox::IoStats> ioStats_;

  // The row type for the data source output, not including filter-only columns.
  const RowTypePtr outputType_;

  // Cached combined AST filter expression compiled from 'subfieldFilters_',
  // owned by 'subfieldTree_'.
  const cudf::ast::expression* subfieldFilterAst_{nullptr};
  // Whether any subfield filter targets a decimal column. If so, addSplit()
  // rebuilds the pushdown AST against each split's Parquet decimal types.
  bool hasDecimalSubfieldFilter_{false};

 private:
  // A partition key the data source resolves from the split.
  struct PartitionKey {
    TypePtr type;
    // Whether a DATE value is a count of days since the epoch rather than an
    // ISO date.
    bool dateValueDaysSinceEpoch;
  };

  // Construct and cache a RowTypePtr for the table column names and types.
  const RowTypePtr getTableRowType();

  // Returns the type of a column read from the file or materialized by the
  // data source, by table column name.
  const TypePtr& columnType(const std::string& name) const;

  // Returns the current split's value of a partition key. A key the split
  // does not carry reads as NULL.
  std::optional<std::string> partitionKeyValue(const std::string& name) const;

  // Returns whether every pushed-down partition key filter accepts the current
  // split.
  bool partitionKeyFiltersAcceptSplit() const;

  // Returns the current split's partition key values as scalars, in
  // 'partitionKeyNames_' order, each parsed per its key's type. A TIMESTAMP
  // value takes the type the split reader gives file timestamps.
  std::vector<std::unique_ptr<cudf::scalar>> makePartitionKeyScalars(
      cuda::stream_ref stream) const;

  // Returns the materialized partition key columns of the current split, each
  // a constant column of 'numRows' rows built from 'partitionKeyScalars_'.
  std::vector<std::unique_ptr<cudf::column>> makePartitionKeyColumns(
      cudf::size_type numRows,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

  // Returns whether TIMESTAMP partition values are read as local time.
  bool readTimestampPartitionValueAsLocalTime() const;

  // Counts the current split as skipped without opening it.
  void recordSkippedSplit();

  RowTypePtr cachedTableRowType_{};

  memory::MemoryPool* const pool_;

  size_t completedRows_{0};
  size_t completedBytes_{0};

  dwio::common::RuntimeStats runtimeStats_;

  std::unique_ptr<CudfSplitReader> cudfSplitReader_;

  // Optimized remaining-filter expression, or null when there is no remaining
  // filter or it folded to a constant. Gates remaining-filter evaluation in
  // next().
  core::TypedExprPtr optimizedRemainingFilter_;

  // Whether the remaining filter folded to a constant that no row satisfies,
  // in which case splits are skipped without opening them.
  bool remainingFilterRejectsAllRows_{false};

  // Compiled cuDF evaluator for the remaining filter, applied post-read in
  // next(). Null when there is no remaining filter.
  std::shared_ptr<velox::cudf_velox::CudfExpression>
      cudfRemainingFilterExpression_;

  std::atomic<uint64_t> totalRemainingFilterTime_{0};

  std::unordered_set<std::string> readColumnSet_;

  // Expression evaluator for remaining filter.
  core::ExpressionEvaluator* const expressionEvaluator_;

  // Logical and split-specific physical AST storage for subfield filters.
  std::vector<std::unique_ptr<cudf::scalar>> subfieldScalars_;
  cudf::ast::tree subfieldTree_;
  std::vector<std::unique_ptr<cudf::scalar>> pushdownFilterScalars_;
  cudf::ast::tree pushdownFilterTree_;

  // The table handle's subfield filters, merged with the ones extracted from
  // its remaining filter, except the ones on partition keys the data source
  // resolves, which are kept in 'partitionKeyFilters_'.
  common::SubfieldFilters subfieldFilters_;

  const PartitionKeySource partitionKeySource_;

  // Whether splits with a filter are read filter columns first.
  const bool filterFirst_;

  // What the split readers of this data source read, by reading mode.
  CudfSplitReader::ReadStats readStats_;

  // The dynamic filters, by read column, that the reader tested on the rows
  // of the last pass it read filter columns first. Empty if it tested none.
  // Holding the filters keeps a later filter from being taken for one of
  // them.
  std::vector<std::pair<size_t, common::FilterPtr>> readerTestedFilters_;

  // Table column name of each output column, in 'outputType_' order.
  std::vector<std::string> outputSourceNames_;

  // Types of the output and materialized columns, by table column name.
  std::unordered_map<std::string, TypePtr> columnTypes_;

  // Row type of what the split reader returns: 'readColumnNames_' with their
  // types.
  RowTypePtr readerOutputType_;

  // Partition keys the data source resolves from the split, by table column
  // name.
  std::unordered_map<std::string, PartitionKey> partitionKeys_;

  // The partition keys materialized as columns, those of the output and of
  // the remaining filter, in the order they follow the read columns in the
  // table the remaining filter sees.
  std::vector<std::string> partitionKeyNames_;

  // Pushed-down filters on partition keys the data source resolves. Each
  // decides for the whole split, so none is compiled into the reader's
  // filter.
  common::SubfieldFilters partitionKeyFilters_;

  // The current split's partition key values as scalars, in
  // 'partitionKeyNames_' order. Built once per split in addSplit(); next()
  // expands each into a constant column per batch.
  std::vector<std::unique_ptr<cudf::scalar>> partitionKeyScalars_;

  // Whether a partition key filter rejected the current split, which then has
  // no split reader.
  bool splitSkipped_{false};
  // How next() evaluates a dynamic filter.
  enum class DynamicFilterEvaluation {
    // Rejects every row.
    kAlwaysFalse,
    // An inclusive range of an integer column, tested on the device.
    kRange,
    // A set of values of an integer column, tested on the device.
    kValues,
    // A range of an integer column narrowed by the device bitmap its
    // producer attached.
    kDeviceBitmap,
    // Any other filter, and any filter on a column that is not read as
    // integers, tested on the host over a copy of the column.
    kHost,
  };

  // A dynamic filter on one read column and what next() needs to apply it.
  struct DynamicFilter {
    // The filter as the operator upstream sent it.
    common::FilterPtr filter;

    // Whether the scan may keep rows 'filter' rejects.
    bool advisory{false};

    DynamicFilterEvaluation evaluation{DynamicFilterEvaluation::kHost};

    // The smallest and largest value of a kValues filter.
    int64_t lower{0};
    int64_t upper{0};

    // The values of a kValues filter as a bitmap on the device, made at the
    // first batch and shared by the data sources of the scan: UINT32 words
    // with bit 'value - lower' set. Null if the values span too wide a range.
    std::shared_ptr<const cudf::column> bitmap;

    // The values in the column's type, for 'hashSet'.
    std::unique_ptr<cudf::column> deviceValues;

    // Set when the values span too wide a range for a bitmap.
    std::unique_ptr<CudfIntegerHashSet> hashSet;

    // The integer type 'hashSet' holds its keys in: that of the column it
    // was built for.
    cudf::type_id hashSetType{cudf::type_id::EMPTY};
  };

  // Makes the DynamicFilter for 'filter' on the read column 'name'.
  DynamicFilter makeDynamicFilter(
      const std::string& name,
      const common::FilterPtr& filter) const;

  // Whether 'dynamicFilter' is tested on rows now: always, unless it is
  // advisory and advisory filters have stopped paying or are switched off.
  bool appliesDynamicFilter(const DynamicFilter& dynamicFilter) const;

  // Drops the rows of 'table', which holds the read columns, that fail the
  // dynamic filters.
  void applyDynamicFilters(
      std::unique_ptr<cudf::table>& table,
      cuda::stream_ref stream);

  // Copies the values of a kValues filter to the device in the form that fits
  // the type of 'input'.
  void uploadDynamicFilterValues(
      DynamicFilter& dynamicFilter,
      const cudf::column_view& input,
      cuda::stream_ref stream);

  // Lets the reader about to be prepared read its split filter columns
  // first, with 'pushdownFilter', which may be null, narrowed by the bounds
  // of the dynamic filters.
  void setFilterFirstFilter(const cudf::ast::expression* pushdownFilter);

  // Tests the dynamic filters on the filter columns the reader decoded for a
  // row group pass. A CudfSplitReader::RowFilter.
  std::unique_ptr<cudf::column> testDynamicFiltersInReader(
      const cudf::table_view& filterColumns,
      std::span<const size_t> readColumns,
      cuda::stream_ref stream);

  // Whether the filters the reader last tested are the dynamic filters as
  // they are now.
  bool readerTestedDynamicFilters() const;

  // Ends the current split without reading the rest of it.
  void skipCurrentSplit();

  // Counts the changes of 'dynamicFilters_'.
  uint64_t dynamicFilterVersion_{0};

  // Whether next() has returned the last rows of the current split.
  bool splitExhausted_{false};

  // Dynamic filters by index in 'readColumnNames_'.
  std::map<size_t, DynamicFilter> dynamicFilters_;

  // Dynamic filters on partition keys. Each decides for a whole split.
  common::SubfieldFilters dynamicPartitionKeyFilters_;

  // No dynamic filter has changed since this data source created its reader.
  bool readerFiltersCurrent_{false};

  // Rows the advisory filters have seen and kept while no other dynamic
  // filter was in place.
  uint64_t advisoryFilterInputRows_{0};
  uint64_t advisoryFilterOutputRows_{0};

  // Set once the advisory filters have proven to keep almost every row, so
  // that masking and compacting each batch costs more than it saves.
  bool advisoryRowFilterDisabled_{false};

  // Rows the dynamic filters dropped.
  uint64_t dynamicFilterDroppedRows_{0};

  // Times this data source listed the values of a value-set filter and
  // copied them to the device.
  uint64_t numDynamicFilterValueUploads_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive
