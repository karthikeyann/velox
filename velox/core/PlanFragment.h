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
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>
#include "velox/core/PlanNode.h"

namespace facebook::velox::core {

class QueryConfig;

/// Gives hints on how to execute the fragment of a plan.
enum class ExecutionStrategy {
  /// Process splits as they come in any available driver.
  kUngrouped,
  /// Process splits from each split group only in one driver.
  /// It is used when split groups represent separate partitions of the data on
  /// the grouping keys or join keys. In that case it is sufficient to keep only
  /// the keys from a single split group in a hash table used by group-by or
  /// join.
  kGrouped,
};

std::string executionStrategyToString(ExecutionStrategy strategy);

/// Declares that a hash join of a plan fragment makes a dynamic filter from the
/// build-side values of one of its join keys, to be applied by table scans
/// outside of the join's pipeline: in another pipeline of the same task or in
/// another task. See DynamicFilterTarget for the receiving side.
///
/// Table scans wait for the filter, so the operator that implements the build
/// side of the join must resolve it with Task::publishDynamicFilter() on every
/// way out of the build, with nullptr if it cannot make the filter.
struct DynamicFilterSource {
  /// Identifies the filter within the query. The targets of the filter carry
  /// the same id.
  std::string filterId;

  /// The HashJoinNode that makes the filter.
  PlanNodeId joinNodeId;

  /// Position of the key in the join's list of keys.
  column_index_t keyIndex{0};
};

/// Declares that a table scan of a plan fragment applies a dynamic filter that
/// a hash join outside of the scan's pipeline makes. The scan waits for the
/// filter before it reads. The filter is advisory: the join that makes it
/// still filters the rows, so a scan may apply it late, partly or not at all.
struct DynamicFilterTarget {
  /// Identifies the filter within the query.
  std::string filterId;

  /// The TableScanNode that applies the filter.
  PlanNodeId scanNodeId;

  /// The output column of the scan the filter applies to.
  column_index_t outputChannel{0};
};

/// Contains some information on how to execute the fragment of a plan.
/// Used to construct Task.
struct PlanFragment {
  /// Top level (root) Plan Node.
  std::shared_ptr<const core::PlanNode> planNode;
  ExecutionStrategy executionStrategy{ExecutionStrategy::kUngrouped};
  int numSplitGroups{0};

  /// Contains leaf plan nodes that need to be executed in the grouped mode.
  std::unordered_set<PlanNodeId> groupedExecutionLeafNodeIds;

  /// Identifies this task among all tasks running the same query stage.
  /// Supplied by the consumer at task creation time. AssignUniqueId packs it
  /// into the high 24 bits of each generated id and the per-row counter into
  /// the low 40 bits, so it must fit in 24 bits (0 <= taskUniqueId < 2^24).
  /// When unset, AssignUniqueId falls back to the value carried on its plan
  /// node (0 in the canonical construction path).
  std::optional<int32_t> taskUniqueId{};

  /// Dynamic filters the hash joins of this fragment make for table scans
  /// outside of their pipelines. Filter ids are unique in this list. The
  /// operator of the join publishes each filter with
  /// Task::publishDynamicFilter() when the build side is complete.
  std::vector<DynamicFilterSource> dynamicFilterSources;

  /// Dynamic filters the table scans of this fragment wait for and apply. A
  /// filter listed in 'dynamicFilterSources' as well stays inside the task:
  /// the task hands it to its scans when the join publishes it. Any other
  /// filter comes from other tasks and whoever runs the tasks delivers it with
  /// Task::addDynamicFilter(). One filter can have targets on several scans,
  /// one per scan.
  std::vector<DynamicFilterTarget> dynamicFilterTargets;

  /// Returns true if the fragment uses grouped execution strategy meaning that
  /// at least one pipeline has a leaf node that should run grouped execution.
  /// Note that it does not mean that all pipelines run grouped execution -
  /// some leaf nodes might still run ungrouped execution.
  inline bool isGroupedExecution() const {
    return executionStrategy == ExecutionStrategy::kGrouped;
  }

  /// Returns true for leaf nodes that use grouped execution, false otherwise.
  inline bool leafNodeRunsGroupedExecution(const PlanNodeId& planNodeId) const {
    return groupedExecutionLeafNodeIds.find(planNodeId) !=
        groupedExecutionLeafNodeIds.end();
  }

  /// Returns first node that does not support barrier.
  /// Returns nullptr if all nodes support barrier.
  const PlanNode* firstNodeNotSupportingBarrier() const;

  /// Returns true if the spilling is enabled and there is at least one node in
  /// the plan, whose operator can spill. Returns false otherwise.
  bool canSpill(const QueryConfig& queryConfig) const;

  PlanFragment() = default;

  explicit PlanFragment(std::shared_ptr<const core::PlanNode> topNode)
      : planNode(std::move(topNode)) {}

  PlanFragment(
      std::shared_ptr<const core::PlanNode> topNode,
      ExecutionStrategy strategy,
      int numberOfSplitGroups,
      const std::unordered_set<PlanNodeId>& groupedExecLeafNodeIds)
      : planNode(std::move(topNode)),
        executionStrategy(strategy),
        numSplitGroups(numberOfSplitGroups),
        groupedExecutionLeafNodeIds(groupedExecLeafNodeIds) {}
};

} // namespace facebook::velox::core

template <>
struct fmt::formatter<facebook::velox::core::ExecutionStrategy>
    : formatter<int> {
  auto format(
      const facebook::velox::core::ExecutionStrategy& s,
      format_context& ctx) const {
    return formatter<int>::format(static_cast<int>(s), ctx);
  }
};
