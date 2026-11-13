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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderByteFetch.h"
#include "velox/experimental/cudf/connectors/hive/FetchAheadBudget.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/type/Filter.h"
#include "velox/type/Type.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;

using CudfParquetReader =
    cudf::io::parquet::experimental::hybrid_scan_multifile;
using CudfParquetReaderPtr = std::unique_ptr<CudfParquetReader>;

/// Normalizes decimal columns, recursively, to their logical Velox types.
/// columnTypes must describe every column after numPrependedColumns, which
/// are left unchanged. Casts and buffer releases use the supplied stream.
std::unique_ptr<cudf::table> castDecimalColumnsToVeloxTypes(
    std::unique_ptr<cudf::table>&& table,
    std::span<const TypePtr> columnTypes,
    size_t numPrependedColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Returns the type a TIMESTAMP constant takes to match the split reader's
/// timestamp columns: the session's timestamp type setting, or the connector's
/// when the session sets none. Returns nullopt when that setting is
/// TIMESTAMP_DAYS, a unit no TIMESTAMP scalar can take.
std::optional<cudf::type_id> constantTimestampType(
    const CudfHiveConfig& config,
    const config::ConfigBase* session);

class CudfSplitReader : public NvtxHelper {
 public:
  /// Runtime statistic of the time a scan waited for the column chunks of a
  /// row group pass to be on the device, one sample per pass.
  static constexpr const char* kFetchWaitWallNanos = "cudfFetchWaitWallNanos";

  /// 'dynamicFilters' are the dynamic filters on integer columns at this
  /// time, by read column. The reader skips the row groups whose statistics
  /// they exclude; the caller filters the rows.
  CudfSplitReader(
      std::shared_ptr<CudfHiveConnectorSplit> split,
      std::shared_ptr<const ::facebook::velox::connector::hive::HiveTableHandle>
          tableHandle,
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
      const common::SubfieldFilters* dynamicFilters);

  virtual ~CudfSplitReader();

  /// Reads and returns the Parquet file metadatas, cached until prepareSplit()
  /// hands them to the cuDF reader. Returns no metadata when the split is
  /// empty.
  const std::vector<cudf::io::parquet::FileMetaData>& fileMetaDatas();

  /// Returns whether the file stores the top-level column 'name' as the
  /// signed integers of 'type', or as dates when 'type' is TIMESTAMP_DAYS, so
  /// that the reader decodes it in 'type'. False for a column the file does
  /// not have and for a split without a file. Reads the footer if needed;
  /// must be called before prepareSplit().
  bool storesIntegersOfType(const std::string& name, cudf::data_type type);

  /// Sets the split-specific pushdown filter. The expression must remain alive
  /// while the split is being read.
  void setPushdownFilter(const cudf::ast::expression* filter);

  /// Allows the split to be read filter columns first: the read columns
  /// named by 'filterColumns', then the other columns for the rows 'filter'
  /// keeps. 'filter' references exactly 'filterColumns', keeps no row the
  /// pushdown filter rejects and takes its place in such a read. Must be
  /// called before prepareSplit(). The expression must remain alive while the
  /// split is being read.
  void setFilterFirstFilter(
      const cudf::ast::expression* filter,
      std::vector<std::string> filterColumns);

  /// Returns whether the prepared split is read filter columns first. It is
  /// not if no such filter was set, if nothing is left to read after the
  /// filter columns, if a filter column is not a column the file has and the
  /// reader returns, if a row group pass has more rows than a column holds,
  /// or if the reader has to change the rows or the columns it reads, as for
  /// a row index or Iceberg deletes.
  bool readsFilterColumnsFirst() const {
    return readsFilterColumnsFirst_;
  }

  /// Narrows the rows a filter-first read keeps. Receives the filter columns
  /// of the rows of a row group pass that the reader's filter kept, and the
  /// index of each in the read columns. Returns a BOOL8 column without nulls
  /// that is true for the rows to keep, or null if it tested no row.
  using RowFilter = std::function<std::unique_ptr<cudf::column>(
      const cudf::table_view& filterColumns,
      std::span<const size_t> readColumns,
      cuda::stream_ref stream)>;

  /// What the reader has read of the split by reading mode.
  struct ReadStats {
    /// Row group passes read filter columns first.
    int64_t numFilterFirstPasses{0};
    /// Row group passes read with all columns in one step.
    int64_t numOneStepPasses{0};
    /// Rows of the passes read filter columns first.
    int64_t numFilterFirstRows{0};
    /// Rows of those the filters kept, whose other columns were then read.
    int64_t numFilterFirstKeptRows{0};
    /// Rows the row filter tested, and the rows of those it kept.
    int64_t numRowFilterRows{0};
    int64_t numRowFilterKeptRows{0};
  };

  /// Returns the statistics of what was read since the last call.
  ReadStats takeReadStats() {
    return std::exchange(readStats_, {});
  }

  /// Opens the cuDF reader and sets up its options. Returns early without
  /// updating runtime statistics when the split is empty.
  /// @param runtimeStats Reference to the DataSource's runtime statistics
  void prepareSplit(dwio::common::RuntimeStats& runtimeStats);

  /// Returns whether the split is empty because its missing data file is
  /// ignored.
  bool emptySplit() const {
    return emptySplit_;
  }

  /// A table chunk and its row count
  struct TableChunk {
    std::unique_ptr<cudf::table> table;
    vector_size_t numRows{0};
    /// Whether the row filter given to next() tested the rows of this chunk
    /// before their other columns were read: in the call that started the
    /// chunk's row group pass, which is this call or an earlier one.
    bool rowFilterApplied{false};
  };

  /// Reads the next raw cudf table chunk. Returns nullopt when done.
  /// 'rowFilter', if not empty, is called while a row group pass is read
  /// filter columns first, during this call only.
  virtual std::optional<TableChunk> next(
      uint64_t size,
      const RowFilter& rowFilter);

  /// Rebinds the query context of a reader prepared in the background to the
  /// context owned by the driver that reads it. Must be called before reading
  /// from a reader that outlives the context it was prepared with.
  void setConnectorQueryCtx(const ConnectorQueryCtx* connectorQueryCtx);

  /// Start the reads of the column chunks of the current pass without waiting
  /// for them. Does nothing when they are already in flight or complete.
  void startColumnChunkFetch();

  /// Offers the reads of the column chunks of the current pass to the
  /// fetch-ahead budget, which starts them on the connector's start thread
  /// when the device memory held by fetched and not yet decoded column
  /// chunks leaves room. The reader must not be used by another thread until
  /// startColumnChunkFetch() or resetSplit() is called or the reader is
  /// destroyed. Returns false, having done nothing, when the connector has
  /// no fetch-ahead budget or the reader has no executor. With a budget, the
  /// reader makes the same offer for each later pass when the pass before it
  /// starts decoding.
  bool offerColumnChunkFetch();

  /// Get the stream.
  cuda::stream_ref stream() const {
    return stream_;
  }

  /// Whether footer statistics prove all retained rows satisfy the dynamic
  /// snapshot.
  bool dynamicFiltersSatisfied() const {
    return dynamicFiltersSatisfied_;
  }

  /// Releases the reader, data source, parquet metadata and any pending
  /// column chunk fetches.
  virtual void resetSplit();

 protected:
  // Returns the connector config the reader was built with.
  const CudfHiveConfig& cudfHiveConfig() const {
    return *cudfHiveConfig_;
  }

  // Returns the cuDF reader options of the current split. Set while the split
  // is prepared.
  const cudf::io::parquet_reader_options& readerOptions() const {
    return readerOptions_;
  }

  // Performs split-specific setup after the Parquet footer is loaded.
  virtual void prepareSplitInternal(dwio::common::RuntimeStats& runtimeStats);

  // Returns whether the split is skipped.
  virtual bool isSplitSkipped() const;

  // Return the split-specific filter to push down to the cuDF reader.
  virtual cudf::ast::expression const* pushdownFilter() const;

  // Returns whether the split can be read filter columns first with the
  // filter set for that. A reader that pushes down another filter than the
  // one it was given, or reads other columns from the file, returns false.
  virtual bool supportsFilterFirst() const {
    return true;
  }

  // Determine the output memory resource for the cuDF reader.
  virtual rmm::device_async_resource_ref determineCudfMemoryResource() const;

  // Read the next table chunk from the parquet reader. Returns nullopt when no
  // more data. All read decimals, including nested,filter-only and
  // equality-delete key columns, have their logical Velox scale and storage
  // width (DECIMAL64 for short decimals, DECIMAL128 for long decimals) before
  // deferred filters or equality deletes consume the table. A prepended
  // row-index column is not part of the logical read schema.
  virtual std::optional<TableChunk> readNextChunk();

  // Setup the cuDF data source. Marks the split empty when its missing data
  // file is ignored.
  void setupCudfDataSource();

  // Returns whether the session ignores missing files.
  bool ignoreMissingFiles() const;

  // Marks the split empty after an ignored missing data file, dropping its
  // data source and footer metadata.
  void markSplitEmpty();

  // Create the parquet reader and select the row group passes to read.
  void createCudfReader();

  // Read file metadatas and cache `fileColumnNames_`, `baseReadOffset_` and
  // `splitRowCount_` from the Parquet footer.
  void cacheSchemaFromMetadata();

  // Returns the {start row, row count} covered by the split.
  std::pair<std::size_t, std::size_t> computeSplitRowRange() const;

  // Return the logical subfield filter AST used after reading.
  const cudf::ast::expression* subfieldFilterAst() const;

  // Return whether the pushdown filter was built for the current split.
  bool hasSplitSpecificPushdownFilter() const;

  std::shared_ptr<CudfHiveConnectorSplit> split_;
  std::shared_ptr<const ::facebook::velox::connector::hive::HiveTableHandle>
      tableHandle_;
  const RowTypePtr outputType_;
  std::vector<std::string> readColumnNames_;
  // Logical types aligned with readColumnNames_, including hidden read columns.
  std::vector<TypePtr> readColumnTypes_;

  FileHandleFactory* fileHandleFactory_;
  folly::Executor* executor_;
  const ConnectorQueryCtx* connectorQueryCtx_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<IoStats> ioStats_;

  cuda::stream_ref stream_{cudaStream_t{cudaStreamDefault}};

  // Parquet metadata(s) for the current split(s).
  std::vector<cudf::io::parquet::FileMetaData> fileMetaDatas_;

  // Whether to prepend a row index column to the output.
  bool prependRowIndex_{false};

  // Whether Parquet column names are matched case-insensitively.
  bool caseInsensitiveColumnNames_{false};

  // Top-level column names from the file metadata.
  std::unordered_set<std::string> fileColumnNames_;

  // Whether `readColumnNames_` is just file columns in the same order.
  bool readAllFileColumns_{false};

  // Tracks the absolute row range covered by the split.
  std::size_t baseReadOffset_{0};
  std::size_t splitRowCount_{0};

  // Whether the split reads no columns, so no cuDF reader exists and the row
  // count comes from the Parquet footer.
  bool noColumnsToRead_{false};

 private:
  // Stores row group indices for one Parquet source.
  using RowGroupIndices = std::vector<cudf::size_type>;

  // Stores row group indices by split.
  using RowGroupIndicesBySplit = std::vector<RowGroupIndices>;

  // Stores per-split row group indices by read pass.
  using RowGroupPasses = std::vector<RowGroupIndicesBySplit>;

  // The fetch of one row group pass.
  struct PassFetch {
    // Owns the device data of the pass: all column chunks, or the ones of the
    // columns that are not filter columns when the split is read filter
    // columns first.
    ByteRangeFetch fetch;
    // Owns the device data of the filter column chunks of the pass when the
    // split is read filter columns first.
    ByteRangeFetch filterFetch;
    // Bytes of the fetch counted in the fetch-ahead budget.
    uint64_t heldBytes{0};
    // Where the fetch stands for the budget, while the budget counts it.
    std::optional<FetchAheadBudget::Stage> stage;
    // Whether the fetch was started on the fetch-ahead stream. The reader's
    // stream, which decodes the pass, is not ordered against that stream:
    // the reader synchronises the fetch-ahead stream before the pass is
    // decoded, and its own stream before the pass's buffers are freed.
    bool isOnFetchAheadStream{false};
  };

  // Tracks how far the row group passes of the current split have been read.
  struct RowGroupPassState {
    // Row groups to read, indexed as passes[pass][split][rowGroup], in read
    // order.
    RowGroupPasses passes;

    // The pass being materialized.
    size_t currentPass{0};

    // Whether the chunked read of the current pass has been set up.
    bool isChunkingSetup{false};

    // The pass being materialized.
    PassFetch current;
    // The pass after it, when its fetch was started ahead.
    PassFetch next;

    // Byte ranges of the column chunks of one pass.
    struct ColumnChunkByteRanges {
      // Filter column chunks. Empty unless the split is read filter columns
      // first.
      std::vector<cudf::io::text::byte_range_info> filter;
      // The other column chunks: all of them unless the split is read filter
      // columns first.
      std::vector<cudf::io::text::byte_range_info> other;
    };

    // Byte ranges of each pass, known before any pass is read so that
    // starting or offering a fetch does not disturb a read in progress.
    std::vector<ColumnChunkByteRanges> byteRanges;

    // One BOOL8 entry per row of the current pass, true for the rows its
    // filters keep.
    std::unique_ptr<cudf::column> rowMask;

    // Filter columns of the kept rows of the current pass.
    std::unique_ptr<cudf::table> filterTable;

    // Index in the read columns of each column of 'filterTable'.
    std::vector<size_t> filterReadColumns;

    // Rows of 'filterTable' returned so far.
    cudf::size_type numFilterRowsReturned{0};

    // Whether the row filter tested the rows of the current pass.
    bool rowFilterApplied{false};
  };

  // Setup the cuDF reader options
  void setupReaderOptions();

  // Returns the index in the footer's schema of the top-level column 'name'
  // if the file stores it as the signed integers of 'type', or as dates for
  // TIMESTAMP_DAYS.
  std::optional<size_t> findIntegerColumn(
      const std::string& name,
      cudf::data_type type) const;

  // Decides from the footer's integer statistics which row groups can hold a
  // row that passes the captured dynamic filters, and whether every row of
  // those does.
  void filterDynamicRowGroups();

  // Setup Parquet column and offset indexes for the cudf split reader.
  void setupPageIndexes();

  // Return the row groups to read, grouped into passes bounded by the pass
  // read limit. Empty when the split has no row groups left after pruning.
  RowGroupPasses selectRowGroupPasses();

  // Wait for the column chunks of the current pass, fetching them first if
  // that has not started yet, and set up its chunked read.
  void setupChunkingForCurrentPass(rmm::device_async_resource_ref mr);

  // Wait for any reads of the current pass, then release its column chunk data.
  void releaseCurrentPassData();

  // Wait for any reads of 'pass', then release its column chunk data and give
  // its bytes back to the fetch-ahead budget. 'isDecoded' says whether the
  // pass was given to decode.
  void releasePassData(PassFetch& pass, bool isDecoded);

  // Start the reads of 'byteRanges', the column chunks of a pass, into
  // 'pass': the filter column chunks, if the split is read filter columns
  // first, and then the others. 'isHeld' says whether the fetch-ahead budget
  // already counts 'numBytes' for them.
  void startFetch(
      PassFetch& pass,
      const RowGroupPassState::ColumnChunkByteRanges& byteRanges,
      uint64_t numBytes,
      bool isHeld,
      cuda::stream_ref stream);

  // Waits for 'fetch', which the first step of decoding the current pass
  // needs, starting the fetches of the pass if they are not yet. Then offers
  // the fetch of the next pass to the fetch-ahead budget.
  void awaitCurrentPassFetch(ByteRangeFetch& fetch);

  // Waits for 'fetch', a later fetch of the current pass, while the
  // fetch-ahead budget starts nothing ahead.
  void waitForFetch(ByteRangeFetch& fetch);

  // Offers the fetch of 'pass', which is the current pass or the one after
  // it, to the fetch-ahead budget.
  void offerPassFetch(
      size_t pass,
      PassFetch& target,
      cuda::stream_ref stream,
      FetchAheadBudget::Distance distance);

  // Takes back the reader's offer to the fetch-ahead budget. Returns once the
  // budget is not starting the fetch.
  void withdrawFetchOffer();

  // Makes the pass after the one just read the current pass, with the fetch
  // that was started ahead for it, if any.
  void advancePass();

  // Returns the stream that the reader's fetches are started ahead on.
  cuda::stream_ref fetchAheadStream();

  // Makes the next pass the current one and starts its fetch unless it was
  // started ahead.
  void finishCurrentPass();

  // Finds the byte ranges of the column chunks of every pass.
  void setColumnChunkByteRanges();

  // Reads the filter columns of the current pass, narrows the row mask of the
  // pass with the reader's filter and the row filter, and sets up the chunked
  // read of the other columns. Returns false if no row of the pass is kept.
  bool setupFilterFirstPass(rmm::device_async_resource_ref mr);

  // Reads the next chunk of the columns that are not filter columns and joins
  // it with the filter columns of the same rows, in read column order.
  std::unique_ptr<cudf::table> readFilterFirstChunk(
      rmm::device_async_resource_ref mr);

  std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  memory::MemoryPool* pool_;
  // The dynamic filters supplied when this reader was created: read column
  // name, the column's type and the filter.
  std::vector<std::tuple<std::string, cudf::data_type, common::FilterPtr>>
      readerDynamicFilters_;
  // One entry per row group of the file, false if the dynamic filters exclude
  // it; empty without dynamic filters.
  std::vector<bool> dynamicRowGroups_;
  // Every row in a retained group satisfies the captured dynamic filters.
  bool dynamicFiltersSatisfied_{false};
  // Row groups of the split the captured dynamic filters skipped.
  int64_t numDynamicFilterSkippedRowGroups_{0};
  // Row groups of the split left to read after every kind of pruning.
  int64_t numSelectedRowGroups_{0};

  // The connector's fetch-ahead state. Null when the connector has no
  // fetch-ahead budget. Declared before the pass state, whose buffers can be
  // allocated on its streams.
  const std::shared_ptr<FetchAheadContext> fetchAheadContext_;
  // The budget of 'fetchAheadContext_', or null.
  FetchAheadBudget* const fetchAheadBudget_;

  // cuDF split reader stuff.
  std::shared_ptr<cudf::io::datasource> dataSource_;
  cudf::io::parquet_reader_options readerOptions_;
  // 'readerOptions_' with the filter of a filter-first read.
  cudf::io::parquet_reader_options filterFirstOptions_;
  CudfParquetReaderPtr splitReader_;

  // Row group passes and byte fetch states.
  std::unique_ptr<RowGroupPassState> passState_;
  // Names the scan to the fetch-ahead budget: the plan node of the task.
  uint64_t fetchAheadScan_{0};
  // Set while the fetch of a pass is on offer to the fetch-ahead budget.
  std::shared_ptr<PostedFetchStart> fetchAhead_;
  // One of the connector's fetch-ahead streams, taken on first use. Not the
  // reader's stream, which decodes.
  std::optional<cuda::stream_ref> fetchAheadStream_;
  bool serializeIoRequests_{false};

  // Chunk and pass read limits resolved from the session when the reader is
  // created.
  std::size_t chunkReadLimit_{0};
  std::size_t passReadLimit_{0};

  dwio::common::ReaderOptions baseReaderOpts_;
  const cudf::ast::expression* subfieldFilterAst_;
  cudf::ast::expression const* pushdownFilterExpr_;
  bool hasSplitSpecificPushdownFilter_{false};
  // Set when an ignored missing data file leaves nothing to read.
  bool emptySplit_{false};

  // Filter of a filter-first read and the read columns it references. Null
  // when the split is always read in one step.
  const cudf::ast::expression* filterFirstFilter_{nullptr};
  std::vector<std::string> filterFirstColumns_;
  // Whether the prepared split is read filter columns first.
  bool readsFilterColumnsFirst_{false};
  // The row filter of the next() call in progress. Not owned.
  const RowFilter* rowFilter_{nullptr};
  // What was read since the last takeReadStats().
  ReadStats readStats_;

  struct TotalScanTimeCallbackData {
    // SteadyTime reading in microseconds when the chunk read started.
    uint64_t startTimeUs;
    std::shared_ptr<io::IoStatistics> ioStatistics;
  };

  static void totalScanTimeCalculator(void* userData);
};

} // namespace facebook::velox::cudf_velox::connector::hive
