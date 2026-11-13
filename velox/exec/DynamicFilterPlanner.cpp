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
#include "velox/exec/DynamicFilterPlanner.h"

#include "velox/exec/TaskDynamicFilters.h"

namespace facebook::velox::exec {
namespace {

// A table scan column a join key comes from.
struct ScanColumn {
  core::PlanNodeId scanNodeId;
  column_index_t outputChannel;
};

// Integers and dates. A logical type stored as an integer, such as a decimal
// or an interval, is left out: a scan may hold it in another form.
bool isSupportedKeyType(const TypePtr& type) {
  return type->isDate() || *type == *TINYINT() || *type == *SMALLINT() ||
      *type == *INTEGER() || *type == *BIGINT();
}

// Returns true if the plan under 'node' may return a part of the rows of its
// tables: it has a filter, or a node that keeps some of its input.
bool mayRemoveRows(const core::PlanNode& node) {
  if (const auto* scan = dynamic_cast<const core::TableScanNode*>(&node)) {
    return scan->tableHandle()->hasFilters();
  }
  if (dynamic_cast<const core::ProjectNode*>(&node) != nullptr ||
      dynamic_cast<const core::LocalPartitionNode*>(&node) != nullptr ||
      dynamic_cast<const core::AggregationNode*>(&node) != nullptr) {
    for (const auto& source : node.sources()) {
      if (mayRemoveRows(*source)) {
        return true;
      }
    }
    return false;
  }
  if (const auto* join = dynamic_cast<const core::HashJoinNode*>(&node)) {
    // An inner join with a table that has all the keys keeps every row.
    if (join->isInnerJoin()) {
      return mayRemoveRows(*node.sources()[0]) ||
          mayRemoveRows(*node.sources()[1]);
    }
  }
  // A filter, a limit, a semi join. Also what is not known: values, an
  // exchange.
  return true;
}

// Appends to 'scans' the table scan columns that output column 'channel' of
// 'node' comes from unchanged, such that a filter on the column commutes with
// every node on the way. 'inPipeline' is true while 'node' runs in the
// pipeline of the join the search started from. 'filteredInPipeline' is true
// if a join above 'node' in the pipeline of 'node' filters its probe side.
void findScanColumns(
    const core::PlanNode& node,
    column_index_t channel,
    bool inPipeline,
    bool filteredInPipeline,
    const DynamicFilterPlanner::Options& options,
    std::vector<ScanColumn>& scans) {
  if (dynamic_cast<const core::TableScanNode*>(&node) != nullptr) {
    if ((!inPipeline || options.includeProbePipeline) &&
        !(filteredInPipeline && options.skipScansFilteredInPipeline)) {
      scans.push_back({node.id(), channel});
    }
    return;
  }

  if (dynamic_cast<const core::FilterNode*>(&node) != nullptr) {
    findScanColumns(
        *node.sources()[0],
        channel,
        inPipeline,
        filteredInPipeline,
        options,
        scans);
    return;
  }

  if (const auto* project = dynamic_cast<const core::ProjectNode*>(&node)) {
    const auto field =
        std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
            project->projections()[channel]);
    if (field == nullptr || !field->isInputColumn()) {
      return;
    }
    const auto& source = *node.sources()[0];
    findScanColumns(
        source,
        source.outputType()->getChildIdx(field->name()),
        inPipeline,
        filteredInPipeline,
        options,
        scans);
    return;
  }

  if (dynamic_cast<const core::LocalPartitionNode*>(&node) != nullptr) {
    for (const auto& source : node.sources()) {
      findScanColumns(*source, channel, false, false, options, scans);
    }
    return;
  }

  if (const auto* aggregation =
          dynamic_cast<const core::AggregationNode*>(&node)) {
    // A filter on a grouping key removes whole groups. With global grouping
    // sets the keys of some output rows are not input values.
    if (channel >= aggregation->groupingKeys().size() ||
        !aggregation->globalGroupingSets().empty()) {
      return;
    }
    const auto& source = *node.sources()[0];
    findScanColumns(
        source,
        source.outputType()->getChildIdx(
            aggregation->groupingKeys()[channel]->name()),
        inPipeline,
        filteredInPipeline,
        options,
        scans);
    return;
  }

  if (const auto* join = dynamic_cast<const core::HashJoinNode*>(&node)) {
    const auto& name = node.outputType()->nameOf(channel);
    const auto& probe = *node.sources()[0];
    const auto& build = *node.sources()[1];
    if (const auto probeChannel =
            probe.outputType()->getChildIdxIfExists(name)) {
      // Every output row has the probe columns of one probe row, so removing
      // probe rows first is the same as removing the output rows.
      if (join->isInnerJoin() || join->isLeftJoin() ||
          join->isLeftSemiFilterJoin() || join->isLeftSemiProjectJoin() ||
          join->isAntiJoin()) {
        // This join pushes its own filter down its pipeline if its build side
        // is filtered.
        const bool filtersProbe =
            TaskDynamicFilters::canFilterProbeSide(*join) &&
            mayRemoveRows(build);
        findScanColumns(
            probe,
            probeChannel.value(),
            inPipeline,
            filteredInPipeline || filtersProbe,
            options,
            scans);
      }
      return;
    }
    if (const auto buildChannel =
            build.outputType()->getChildIdxIfExists(name)) {
      // Likewise for the build columns of one build row.
      if (join->isInnerJoin() || join->isRightJoin()) {
        findScanColumns(
            build, buildChannel.value(), false, false, options, scans);
      }
    }
    return;
  }

  // A filter below any other node could change what the node returns for the
  // rows that pass, e.g. the first row of a limit or the rank of a row.
}

void planNode(
    const core::PlanNode& node,
    const DynamicFilterPlanner::Options& options,
    DynamicFilterPlanner::Result& result) {
  for (const auto& source : node.sources()) {
    planNode(*source, options, result);
  }
  const auto* join = dynamic_cast<const core::HashJoinNode*>(&node);
  if (join == nullptr || !TaskDynamicFilters::canFilterProbeSide(*join)) {
    return;
  }
  if (options.selectiveBuildOnly && !mayRemoveRows(*node.sources()[1])) {
    return;
  }
  const auto& probe = *node.sources()[0];
  for (column_index_t i = 0; i < join->leftKeys().size(); ++i) {
    const auto& key = join->leftKeys()[i];
    if (!key->isInputColumn() || !isSupportedKeyType(key->type()) ||
        !join->rightKeys()[i]->type()->equivalent(*key->type())) {
      continue;
    }
    std::vector<ScanColumn> scans;
    findScanColumns(
        probe,
        probe.outputType()->getChildIdx(key->name()),
        true,
        false,
        options,
        scans);
    if (scans.empty()) {
      continue;
    }
    auto filterId = fmt::format("{}:{}", node.id(), i);
    for (auto& scan : scans) {
      result.targets.push_back(
          {.filterId = filterId,
           .scanNodeId = std::move(scan.scanNodeId),
           .outputChannel = scan.outputChannel});
    }
    result.sources.push_back(
        {.filterId = std::move(filterId),
         .joinNodeId = node.id(),
         .keyIndex = i});
  }
}

} // namespace

// static
DynamicFilterPlanner::Result DynamicFilterPlanner::plan(
    const core::PlanNode& root,
    const Options& options) {
  Result result;
  planNode(root, options, result);
  return result;
}

} // namespace facebook::velox::exec
