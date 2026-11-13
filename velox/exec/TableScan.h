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

#include <string_view>

#include <folly/stop_watch.h>

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/exec/ScaledScanController.h"
#include "velox/exec/TaskDynamicFilters.h"

namespace facebook::velox::exec {

class TableScan : public SourceOperator {
 public:
  TableScan(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::TableScanNode>& tableScanNode);

  void initialize() override;

  RowVectorPtr getOutput() override;

  BlockingReason isBlocked(ContinueFuture* future) override {
    if (blockingFuture_.valid()) {
      *future = std::move(blockingFuture_);
      return blockingReason_;
    }
    return BlockingReason::kNotBlocked;
  }

  bool startDrain() override {
    return false;
  }

  bool isFinished() override;

  void close() override;

  bool canAddDynamicFilter() const override {
    return connector_->canAddDynamicFilter();
  }

  void addDynamicFilterLocked(
      const core::PlanNodeId& producer,
      const PushdownFilters& filters) override;

  /// The name of runtime stats specific to table scan.
  /// The number of running table scan drivers.
  ///
  /// NOTE: we only report the number of running scan drivers at the point that
  /// all the splits have been dispatched.
  static constexpr std::string_view kNumRunningScaleThreads =
      "numRunningScaleThreads";

  /// Time spent reading from the data source.
  static constexpr std::string_view kDataSourceReadWallNanos =
      "dataSourceReadWallNanos";

  /// Number of splits that started background preload.
  static constexpr std::string_view kPreloadedSplits = "preloadedSplits";

  /// Number of preloaded splits that finished before being read.
  static constexpr std::string_view kReadyPreloadedSplits =
      "readyPreloadedSplits";

  /// Size of the connector split.
  static constexpr std::string_view kConnectorSplitSize = "connectorSplitSize";

  /// Time waiting for a preloaded split to become available.
  static constexpr std::string_view kWaitForPreloadSplitNanos =
      "waitForPreloadSplitNanos";

  /// Time for preload split preparation.
  static constexpr std::string_view kPreloadSplitPrepareTimeNanos =
      "preloadSplitPrepareTimeNanos";

  /// Time spent adding a split to the data source.
  static constexpr std::string_view kDataSourceAddSplitWallNanos =
      "dataSourceAddSplitWallNanos";

  /// Time the scan waited before reading for the dynamic filters that joins
  /// outside of its pipeline make.
  static constexpr std::string_view kDynamicFilterWaitWallNanos =
      "dynamicFilterWaitWallNanos";

  /// Number of dynamic filters from joins outside of the pipeline that had not
  /// arrived when the scan stopped waiting and started to read.
  static constexpr std::string_view kDynamicFiltersTimedOut =
      "dynamicFiltersTimedOut";

  std::shared_ptr<ScaledScanController> testingScaledController() const {
    return scaledController_;
  }

  /// Returns the current read batch size. Used for testing.
  vector_size_t testingReadBatchSize() const {
    return readBatchSize_;
  }

 private:
  // Checks if this table scan operator needs to yield before processing the
  // next split.
  bool shouldYield(
      StopReason taskStopReason,
      const folly::stop_watch<std::chrono::milliseconds>& getOutputWatch) const;

  // Checks if this table scan operator needs to stop because the task has been
  // terminated.
  bool shouldStop(StopReason taskStopReason) const;

  // Returns true if a new split is fetched from the task otherwise false.
  bool getSplit();

  // Sets 'maxPreloadSplits' and 'splitPreloader' if prefetching splits is
  // appropriate. The preloader will be applied to the 'first 'maxPreloadSplits'
  // of the Task's split queue for 'this' when getting splits.
  void checkPreload();

  // Sets 'split->dataSource' to be an AsyncSource that makes a DataSource to
  // read 'split'. This source will be prepared in the background on the
  // executor of the connector. If the DataSource is needed before prepare is
  // done, it will be made when needed.
  void preload(const std::shared_ptr<connector::ConnectorSplit>& split);

  // Returns true if the scan has to wait, before it reads, for dynamic filters
  // that joins outside of its pipeline make. Sets 'blockingFuture_' and
  // 'blockingReason_' in that case. Applies the filters that have arrived.
  bool waitForDynamicFilters();

  // Applies the dynamic filters from outside of the pipeline that arrived
  // since the last call. Returns the number of filters still to wait for and
  // sets 'future' to complete when the next one is resolved.
  int32_t applyTaskDynamicFilters(ContinueFuture* future);

  // Invoked by scan operator to check if it needs to stop to wait for scale up.
  bool shouldWaitForScaleUp();

  // Invoked after scan operator finishes processing a non-empty split to update
  // the scan driver memory usage and check to see if we need to scale up scan
  // processing or not.
  void tryScaleUp();

  // Calculates the batch size to read based on available row size information.
  // Returns the number of rows to read in the next batch.
  int32_t calculateBatchSize(int64_t currentEstimatedRowSize);

  const connector::ConnectorTableHandlePtr tableHandle_;
  const connector::ColumnHandleMap columnHandles_;
  DriverCtx* const driverCtx_;
  const int32_t maxSplitPreloadPerDriver_{0};
  const vector_size_t maxReadBatchSize_;
  memory::MemoryPool* const connectorPool_;
  const std::shared_ptr<connector::Connector> connector_;
  // Exits getOutput() method after this many milliseconds. Zero means 'no
  // limit'.
  const size_t getOutputTimeLimitMs_{0};

  const uint32_t outputBatchRowsOverride_;

  // True if the plan fragment declares dynamic filters from outside of the
  // pipeline for this scan and the connector can apply them.
  const bool hasTaskDynamicFilters_;

  // Longest time to wait for those filters before reading.
  const std::chrono::milliseconds dynamicFilterMaxWait_;

  // If set, used for scan scale processing. It is shared by all the scan
  // operators instantiated from the same table scan node.
  const std::shared_ptr<ScaledScanController> scaledController_;

  vector_size_t readBatchSize_;

  ContinueFuture blockingFuture_{ContinueFuture::makeEmpty()};
  BlockingReason blockingReason_{BlockingReason::kNotBlocked};
  bool needNewSplit_ = true;
  std::shared_ptr<connector::ConnectorQueryCtx> connectorQueryCtx_;
  std::unique_ptr<connector::DataSource> dataSource_;
  bool noMoreSplits_ = false;

  int32_t maxPreloadedSplits_{0};

  // Callback passed to getSplitOrFuture() for triggering async preload. The
  // callback's lifetime is the lifetime of 'this'. This callback can schedule
  // preloads on an executor. These preloads may outlive the Task and therefore
  // need to capture a shared_ptr to it.
  std::function<void(const std::shared_ptr<connector::ConnectorSplit>&)>
      splitPreloader_{nullptr};

  // Count of splits that started background preload.
  int32_t numPreloadedSplits_{0};

  // Count of splits that finished preloading before being read.
  int32_t numReadyPreloadedSplits_{0};

  double maxFilteringRatio_{0};

  // Per-split batch size hint. Set from ConnectorSplit::batchSizeHint when a
  // new split is loaded. Reset to 0 when the split has no hint.
  int32_t splitBatchSizeHint_{0};

  // Row size estimate from the file reader. It is set to the last known
  // estimated row size from the current split reader or the previous ones.
  int64_t fileEstimatedRowSize_{connector::DataSource::kUnknownRowSize};

  // String shown in ExceptionContext inside DataSource and LazyVector loading.
  std::string debugString_;

  // False while the scan may still wait for dynamic filters before reading.
  bool dynamicFilterWaitDone_{false};

  // Measures the wait for dynamic filters. Set when the wait starts.
  std::optional<folly::stop_watch<std::chrono::milliseconds>>
      dynamicFilterWaitWatch_;

  // The dynamic filters from outside of the pipeline that have arrived. Each
  // data source of this scan gets them combined with the pipeline's filters.
  std::vector<TaskDynamicFilters::ScanFilter> taskFilters_;

  // Size of 'taskFilters_' as the cursor the task advances.
  uint32_t numTaskDynamicFilters_{0};

  // The total number of raw input rows read up till the last finished split.
  // This is used to detect if a finished split is empty or not.
  uint64_t rawInputRowsSinceLastSplit_{0};
};
} // namespace facebook::velox::exec
