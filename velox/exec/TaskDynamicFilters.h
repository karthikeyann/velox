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

#include <atomic>
#include <mutex>
#include <unordered_map>

#include <folly/futures/Future.h>

#include "velox/common/future/VeloxPromise.h"
#include "velox/core/PlanFragment.h"
#include "velox/type/Filter.h"

namespace facebook::velox::exec {

/// Holds the dynamic filters of one task that travel outside of a pipeline:
/// the filters its hash joins make for table scans elsewhere
/// (core::DynamicFilterSource) and the filters its table scans wait for
/// (core::DynamicFilterTarget).
///
/// Each filter is resolved once, either with a filter or with nullptr, which
/// means that the filter is disabled and the scans read without it. A filter
/// with a source and targets in the same task is resolved for the scans when
/// the join publishes it. All methods are thread-safe.
class TaskDynamicFilters {
 public:
  /// A filter that arrived for a table scan.
  struct ScanFilter {
    /// Id of the filter.
    std::string filterId;

    /// The join that made the filter if it runs in the same task. Empty
    /// otherwise.
    core::PlanNodeId producerNodeId;

    /// The output column of the scan the filter applies to.
    column_index_t outputChannel;

    /// The filter. Never nullptr.
    common::FilterPtr filter;
  };

  /// Checks the sources and targets of 'fragment' against its plan. A target
  /// that could deadlock the task does not make its scan wait: this is a scan
  /// that feeds the build side of the join that makes its filter, directly or
  /// through other scans that wait for filters.
  explicit TaskDynamicFilters(const core::PlanFragment& fragment);

  ~TaskDynamicFilters() {
    close();
  }

  /// Returns true if a column of 'type' can be tested with a filter of the
  /// kind of 'filter', and 'filter' can be merged with every other filter of
  /// such a column. A scan gets no other kind of filter.
  static bool fitsType(const common::Filter& filter, const Type& type);

  /// Returns true if 'join' gives the same result when the probe-side rows
  /// whose key is null or not among the build-side keys are removed before
  /// the join. Only such a join can make a dynamic filter for its probe side.
  static bool canFilterProbeSide(const core::HashJoinNode& join);

  /// Returns the filters the hash join 'joinNodeId' has to publish.
  const std::vector<core::DynamicFilterSource>& sources(
      const core::PlanNodeId& joinNodeId) const;

  /// Returns true if the fragment declares filters for the table scan
  /// 'scanNodeId'.
  bool hasTargets(const core::PlanNodeId& scanNodeId) const;

  /// Resolves a filter this task is the source of. Called by the operator of
  /// the join when its build side is complete. 'filter' must pass every
  /// non-null build-side value of the key; nullptr disables the filter. A
  /// filter of a kind that does not fit the type of a column it applies to
  /// disables the filter as well. Only the first call for a filter has an
  /// effect.
  void publish(const std::string& filterId, common::FilterPtr filter);

  /// Resolves a filter this task has targets for and no source. 'filter' is
  /// the combined filter of all tasks that make it; nullptr disables the
  /// filter. Only the first call for a filter has an effect. Throws if the
  /// kind of 'filter' does not fit the type of a column it applies to.
  void add(const std::string& filterId, common::FilterPtr filter);

  /// Returns a future that completes when the filter is resolved, with the
  /// filter or nullptr if it is disabled. Completes with nullptr if the task
  /// ends first.
  folly::SemiFuture<common::FilterPtr> future(const std::string& filterId);

  /// Returns the number of filters that arrived for the table scan
  /// 'scanNodeId' so far. A scan compares it with the count it has applied to
  /// find out cheaply whether poll() has something new.
  uint32_t numArrived(const core::PlanNodeId& scanNodeId) const;

  /// Appends to 'filters' the filters that arrived for the table scan
  /// 'scanNodeId' after the first 'numSeen' ones and advances 'numSeen'.
  /// Returns the number of filters the scan should still wait for. If that is
  /// not zero and 'future' is not nullptr, sets 'future' to complete when the
  /// next filter of the scan is resolved or the task ends.
  int32_t poll(
      const core::PlanNodeId& scanNodeId,
      uint32_t& numSeen,
      std::vector<ScanFilter>& filters,
      ContinueFuture* future);

  /// Makes the drivers of the table scan 'scanNodeId' stop waiting for the
  /// filters that are not resolved yet, now and when they start later. The
  /// scan still applies those filters when they arrive. Called when the first
  /// driver of the scan has waited for as long as it may.
  void stopWaiting(const core::PlanNodeId& scanNodeId);

  /// Disables the unresolved filters, releases the waiting scans and drops the
  /// filters of the scans. Called when the task ends.
  void close();

 private:
  // A target of a table scan.
  struct Target {
    // Id of the filter.
    std::string filterId;

    // The output column of the scan the filter applies to.
    column_index_t outputChannel;

    // Type of that column.
    TypePtr type;

    // True if the scan waits for the filter before it reads.
    bool wait{true};

    // True once the filter is resolved.
    bool resolved{false};
  };

  // The targets of one table scan node.
  struct Scan {
    // The targets the fragment declares.
    std::vector<Target> targets;

    // The filters resolved with a filter, in arrival order.
    std::vector<ScanFilter> arrived;

    // Size of 'arrived', readable without the mutex.
    std::atomic_uint32_t numArrived{0};

    // Promises of the drivers that wait for a target to be resolved.
    std::vector<ContinuePromise> waiters;
  };

  // One filter id.
  struct Entry {
    // The join that publishes the filter. Empty if the filter comes from other
    // tasks.
    core::PlanNodeId producerNodeId;

    // The scans and the positions in Scan::targets that receive the filter.
    std::vector<std::pair<Scan*, size_t>> targets;

    // True once publish(), add() or close() resolved the filter.
    bool resolved{false};

    // The filter, or nullptr if disabled or not resolved.
    common::FilterPtr filter;

    // Promises of the futures handed out before the filter was resolved.
    std::vector<folly::Promise<common::FilterPtr>> readers;
  };

  // Makes the targets that could deadlock the task not wait. Visits 'targets'
  // in order, so the outcome does not depend on hashing. A planner that places
  // a filter on the probe side of its join never makes such a cycle; this
  // guards declarations made by hand.
  void breakWaitCycles(
      const std::unordered_map<core::PlanNodeId, const core::PlanNode*>& nodes,
      const std::vector<core::DynamicFilterTarget>& targets);

  // Resolves 'entry' unless it is resolved or the task ended.
  void resolve(Entry& entry, common::FilterPtr filter);

  // Filters by join node. Not modified after construction.
  std::unordered_map<core::PlanNodeId, std::vector<core::DynamicFilterSource>>
      sources_;

  // Targets by table scan node. The map is not modified after construction.
  std::unordered_map<core::PlanNodeId, std::unique_ptr<Scan>> scans_;

  // Guards 'entries_', 'closed_' and the non-atomic members of every Scan.
  mutable std::mutex mutex_;

  // State by filter id. The map is not modified after construction.
  std::unordered_map<std::string, Entry> entries_;

  // True once close() ran.
  bool closed_{false};
};

} // namespace facebook::velox::exec
