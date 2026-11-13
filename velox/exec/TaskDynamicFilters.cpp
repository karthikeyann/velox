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
#include "velox/exec/TaskDynamicFilters.h"

#include <algorithm>
#include <unordered_set>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec {
namespace {

using PlanNodeMap = std::unordered_map<core::PlanNodeId, const core::PlanNode*>;

void collectNodes(const core::PlanNode& node, PlanNodeMap& nodes) {
  nodes.emplace(node.id(), &node);
  for (const auto& source : node.sources()) {
    collectNodes(*source, nodes);
  }
}

// Appends the ids of the table scans below and including 'node'.
void collectScans(
    const core::PlanNode& node,
    std::vector<core::PlanNodeId>& scans) {
  if (dynamic_cast<const core::TableScanNode*>(&node) != nullptr) {
    scans.push_back(node.id());
  }
  for (const auto& source : node.sources()) {
    collectScans(*source, scans);
  }
}

const core::HashJoinNode& joinNode(
    const PlanNodeMap& nodes,
    const core::DynamicFilterSource& source) {
  const auto it = nodes.find(source.joinNodeId);
  VELOX_USER_CHECK(
      it != nodes.end(),
      "Plan node of a dynamic filter source is not in the plan fragment: {}",
      source.joinNodeId);
  const auto* join = dynamic_cast<const core::HashJoinNode*>(it->second);
  VELOX_USER_CHECK_NOT_NULL(
      join,
      "Plan node of a dynamic filter source must be a hash join: {}",
      source.joinNodeId);
  return *join;
}

} // namespace

TaskDynamicFilters::TaskDynamicFilters(const core::PlanFragment& fragment) {
  if (fragment.dynamicFilterSources.empty() &&
      fragment.dynamicFilterTargets.empty()) {
    return;
  }
  VELOX_CHECK_NOT_NULL(fragment.planNode);
  PlanNodeMap nodes;
  collectNodes(*fragment.planNode, nodes);

  for (const auto& source : fragment.dynamicFilterSources) {
    const auto& join = joinNode(nodes, source);
    VELOX_USER_CHECK_LT(
        source.keyIndex,
        join.rightKeys().size(),
        "Key of a dynamic filter source is not a key of the join: {}",
        source.filterId);
    const bool isNew = entries_.try_emplace(source.filterId).second;
    VELOX_USER_CHECK(
        isNew,
        "Dynamic filter must have one source in a plan fragment: {}",
        source.filterId);
    entries_[source.filterId].producerNodeId = source.joinNodeId;
    sources_[source.joinNodeId].push_back(source);
  }

  for (const auto& target : fragment.dynamicFilterTargets) {
    const auto it = nodes.find(target.scanNodeId);
    VELOX_USER_CHECK(
        it != nodes.end(),
        "Plan node of a dynamic filter target is not in the plan fragment: {}",
        target.scanNodeId);
    const auto* scanNode = dynamic_cast<const core::TableScanNode*>(it->second);
    VELOX_USER_CHECK_NOT_NULL(
        scanNode,
        "Plan node of a dynamic filter target must be a table scan: {}",
        target.scanNodeId);
    VELOX_USER_CHECK_LT(
        target.outputChannel,
        scanNode->outputType()->size(),
        "Column of a dynamic filter target is not an output of the scan: {}",
        target.filterId);

    auto& entry = entries_[target.filterId];
    if (!entry.producerNodeId.empty()) {
      const auto& joinSources = sources_.at(entry.producerNodeId);
      const auto sourceIt = std::find_if(
          joinSources.begin(), joinSources.end(), [&](const auto& source) {
            return source.filterId == target.filterId;
          });
      const auto& join = joinNode(nodes, *sourceIt);
      VELOX_USER_CHECK_EQ(
          join.rightKeys()[sourceIt->keyIndex]->type()->kind(),
          scanNode->outputType()->childAt(target.outputChannel)->kind(),
          "Dynamic filter must have a source and a target of the same type: {}",
          target.filterId);
    }

    auto& scan = scans_[target.scanNodeId];
    if (scan == nullptr) {
      scan = std::make_unique<Scan>();
    }
    for (const auto& other : scan->targets) {
      VELOX_USER_CHECK_NE(
          other.filterId,
          target.filterId,
          "Dynamic filter must have one target on a table scan: {}",
          target.scanNodeId);
    }
    entry.targets.emplace_back(scan.get(), scan->targets.size());
    scan->targets.push_back(
        Target{
            .filterId = target.filterId,
            .outputChannel = target.outputChannel,
            .type = scanNode->outputType()->childAt(target.outputChannel)});
  }

  breakWaitCycles(nodes, fragment.dynamicFilterTargets);
}

void TaskDynamicFilters::breakWaitCycles(
    const PlanNodeMap& nodes,
    const std::vector<core::DynamicFilterTarget>& targets) {
  // A scan that waits for a filter made in this task depends on the scans
  // under the build side of the filter's join: the filter exists once they are
  // done.
  std::unordered_map<std::string, std::vector<core::PlanNodeId>> buildScans;
  for (const auto& [joinNodeId, sources] : sources_) {
    std::vector<core::PlanNodeId> scans;
    collectScans(*nodes.at(joinNodeId)->sources()[1], scans);
    for (const auto& source : sources) {
      buildScans[source.filterId] = scans;
    }
  }

  // Position of the next target of each scan in Scan::targets, which has the
  // targets of the scan in declaration order.
  std::unordered_map<core::PlanNodeId, size_t> numVisited;
  for (const auto& declared : targets) {
    auto& target = scans_.at(declared.scanNodeId)
                       ->targets[numVisited[declared.scanNodeId]++];
    const auto it = buildScans.find(target.filterId);
    if (it == buildScans.end()) {
      continue;
    }
    std::unordered_set<core::PlanNodeId> visited;
    auto pending = it->second;
    while (!pending.empty()) {
      const auto current = std::move(pending.back());
      pending.pop_back();
      if (current == declared.scanNodeId) {
        target.wait = false;
        break;
      }
      if (!visited.insert(current).second) {
        continue;
      }
      const auto scanIt = scans_.find(current);
      if (scanIt == scans_.end()) {
        continue;
      }
      for (const auto& next : scanIt->second->targets) {
        const auto nextIt = buildScans.find(next.filterId);
        if (next.wait && nextIt != buildScans.end()) {
          pending.insert(
              pending.end(), nextIt->second.begin(), nextIt->second.end());
        }
      }
    }
  }
}

// static
bool TaskDynamicFilters::fitsType(
    const common::Filter& filter,
    const Type& type) {
  switch (filter.kind()) {
    case common::FilterKind::kAlwaysFalse:
    case common::FilterKind::kAlwaysTrue:
    case common::FilterKind::kIsNull:
    case common::FilterKind::kIsNotNull:
      return true;
    case common::FilterKind::kBoolValue:
      return type.kind() == TypeKind::BOOLEAN;
    case common::FilterKind::kBigintRange:
    case common::FilterKind::kNegatedBigintRange:
    case common::FilterKind::kBigintValuesUsingHashTable:
    case common::FilterKind::kBigintValuesUsingBitmask:
    case common::FilterKind::kBigintValuesUsingBloomFilter:
    case common::FilterKind::kNegatedBigintValuesUsingHashTable:
    case common::FilterKind::kNegatedBigintValuesUsingBitmask:
    case common::FilterKind::kBigintMultiRange:
      return type.kind() == TypeKind::TINYINT ||
          type.kind() == TypeKind::SMALLINT ||
          type.kind() == TypeKind::INTEGER || type.kind() == TypeKind::BIGINT;
    case common::FilterKind::kHugeintRange:
    case common::FilterKind::kHugeintValuesUsingHashTable:
      return type.kind() == TypeKind::HUGEINT;
    case common::FilterKind::kDoubleRange:
      return type.kind() == TypeKind::DOUBLE;
    case common::FilterKind::kFloatRange:
      return type.kind() == TypeKind::REAL;
    case common::FilterKind::kBytesRange:
    case common::FilterKind::kNegatedBytesRange:
    case common::FilterKind::kBytesValues:
    case common::FilterKind::kNegatedBytesValues:
      return type.kind() == TypeKind::VARCHAR ||
          type.kind() == TypeKind::VARBINARY;
    case common::FilterKind::kTimestampRange:
      return type.kind() == TypeKind::TIMESTAMP;
    default:
      // Includes kMultiRange, whose members would have to fit one by one.
      return false;
  }
}

// static
bool TaskDynamicFilters::canFilterProbeSide(const core::HashJoinNode& join) {
  return (join.isInnerJoin() || join.isLeftSemiFilterJoin() ||
          join.isRightSemiFilterJoin() || join.isRightJoin()) &&
      !join.isNullAware() && !join.isNullAsValue();
}

const std::vector<core::DynamicFilterSource>& TaskDynamicFilters::sources(
    const core::PlanNodeId& joinNodeId) const {
  static const std::vector<core::DynamicFilterSource> kNone;
  const auto it = sources_.find(joinNodeId);
  return it == sources_.end() ? kNone : it->second;
}

bool TaskDynamicFilters::hasTargets(const core::PlanNodeId& scanNodeId) const {
  return scans_.contains(scanNodeId);
}

void TaskDynamicFilters::publish(
    const std::string& filterId,
    common::FilterPtr filter) {
  const auto it = entries_.find(filterId);
  VELOX_CHECK(
      it != entries_.end(),
      "Task is not a source of the dynamic filter: {}",
      filterId);
  auto& entry = it->second;
  VELOX_CHECK(
      !entry.producerNodeId.empty(),
      "Task is not a source of the dynamic filter: {}",
      filterId);
  if (filter != nullptr) {
    // Not modified after construction, so readable without the mutex.
    for (const auto& [scan, index] : entry.targets) {
      if (!fitsType(*filter, *scan->targets[index].type)) {
        filter = nullptr;
        break;
      }
    }
  }
  resolve(entry, std::move(filter));
}

void TaskDynamicFilters::add(
    const std::string& filterId,
    common::FilterPtr filter) {
  const auto it = entries_.find(filterId);
  VELOX_USER_CHECK(
      it != entries_.end(),
      "Task has no target for the dynamic filter: {}",
      filterId);
  VELOX_USER_CHECK(
      !it->second.targets.empty(),
      "Task has no target for the dynamic filter: {}",
      filterId);
  VELOX_USER_CHECK(
      it->second.producerNodeId.empty(),
      "Dynamic filter is made by a join of the task itself: {}",
      filterId);
  if (filter != nullptr) {
    // Not modified after construction, so readable without the mutex.
    for (const auto& [scan, index] : it->second.targets) {
      const auto& type = scan->targets[index].type;
      VELOX_USER_CHECK(
          fitsType(*filter, *type),
          "Dynamic filter does not fit the type of the scan column: {}, {} for {}",
          filterId,
          filter->toString(),
          type->toString());
    }
  }
  resolve(it->second, std::move(filter));
}

void TaskDynamicFilters::resolve(Entry& entry, common::FilterPtr filter) {
  std::vector<folly::Promise<common::FilterPtr>> readers;
  std::vector<ContinuePromise> waiters;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (closed_ || entry.resolved) {
      return;
    }
    entry.resolved = true;
    entry.filter = filter;
    readers.swap(entry.readers);
    for (auto& [scan, index] : entry.targets) {
      auto& target = scan->targets[index];
      target.resolved = true;
      if (filter != nullptr) {
        scan->arrived.push_back(
            ScanFilter{
                .filterId = target.filterId,
                .producerNodeId = entry.producerNodeId,
                .outputChannel = target.outputChannel,
                .filter = filter});
        scan->numArrived = scan->arrived.size();
      }
      std::move(
          scan->waiters.begin(),
          scan->waiters.end(),
          std::back_inserter(waiters));
      scan->waiters.clear();
    }
  }
  for (auto& reader : readers) {
    reader.setValue(filter);
  }
  for (auto& waiter : waiters) {
    waiter.setValue();
  }
}

folly::SemiFuture<common::FilterPtr> TaskDynamicFilters::future(
    const std::string& filterId) {
  const auto it = entries_.find(filterId);
  VELOX_CHECK(
      it != entries_.end(),
      "Task has no source or target for the dynamic filter: {}",
      filterId);
  auto& entry = it->second;
  std::lock_guard<std::mutex> l(mutex_);
  if (entry.resolved) {
    return folly::makeSemiFuture(entry.filter);
  }
  return entry.readers.emplace_back().getSemiFuture();
}

uint32_t TaskDynamicFilters::numArrived(
    const core::PlanNodeId& scanNodeId) const {
  const auto it = scans_.find(scanNodeId);
  return it == scans_.end() ? 0 : it->second->numArrived.load();
}

int32_t TaskDynamicFilters::poll(
    const core::PlanNodeId& scanNodeId,
    uint32_t& numSeen,
    std::vector<ScanFilter>& filters,
    ContinueFuture* future) {
  const auto it = scans_.find(scanNodeId);
  if (it == scans_.end()) {
    return 0;
  }
  auto& scan = *it->second;
  std::lock_guard<std::mutex> l(mutex_);
  if (closed_) {
    return 0;
  }
  for (auto i = numSeen; i < scan.arrived.size(); ++i) {
    filters.push_back(scan.arrived[i]);
  }
  numSeen = scan.arrived.size();
  int32_t numPending{0};
  for (const auto& target : scan.targets) {
    numPending += target.wait && !target.resolved;
  }
  if (numPending > 0 && future != nullptr) {
    auto [promise, waitFuture] =
        makeVeloxContinuePromiseContract("TaskDynamicFilters::poll");
    scan.waiters.push_back(std::move(promise));
    *future = std::move(waitFuture);
  }
  return numPending;
}

void TaskDynamicFilters::stopWaiting(const core::PlanNodeId& scanNodeId) {
  const auto it = scans_.find(scanNodeId);
  if (it == scans_.end()) {
    return;
  }
  std::vector<ContinuePromise> waiters;
  {
    std::lock_guard<std::mutex> l(mutex_);
    for (auto& target : it->second->targets) {
      if (!target.resolved) {
        target.wait = false;
      }
    }
    waiters.swap(it->second->waiters);
  }
  for (auto& waiter : waiters) {
    waiter.setValue();
  }
}

void TaskDynamicFilters::close() {
  std::vector<folly::Promise<common::FilterPtr>> readers;
  std::vector<ContinuePromise> waiters;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (closed_) {
      return;
    }
    closed_ = true;
    for (auto& [_, entry] : entries_) {
      entry.resolved = true;
      std::move(
          entry.readers.begin(),
          entry.readers.end(),
          std::back_inserter(readers));
      entry.readers.clear();
      // The scans are done. A filter without targets stays for whoever runs
      // the tasks to collect.
      if (!entry.targets.empty()) {
        entry.filter.reset();
      }
    }
    for (auto& [_, scan] : scans_) {
      scan->arrived.clear();
      std::move(
          scan->waiters.begin(),
          scan->waiters.end(),
          std::back_inserter(waiters));
      scan->waiters.clear();
    }
  }
  for (auto& reader : readers) {
    reader.setValue(nullptr);
  }
  for (auto& waiter : waiters) {
    waiter.setValue();
  }
}

} // namespace facebook::velox::exec
