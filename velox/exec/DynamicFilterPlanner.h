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

#include "velox/core/PlanFragment.h"

namespace facebook::velox::exec {

/// Derives the dynamic filters that hash joins of a plan can make for table
/// scans of the same plan that are outside of the join's pipeline, for plans
/// that come without them. A query optimizer that sees the whole query and
/// its statistics places filters itself, including across plan fragments, and
/// does not need this.
class DynamicFilterPlanner {
 public:
  /// What filters to place. The defaults fit a plan without statistics.
  struct Options {
    /// If true, also places a filter when the scan is in the pipeline of the
    /// join, where the join pushes its filter down by itself.
    bool includeProbePipeline{false};

    /// If true, places no filter for a join whose build side returns its
    /// tables whole: no filter on a scan, no filter node, no join or other
    /// node that keeps a part of its input. The keys of such a build side are
    /// usually all the keys there are, and a scan that waits for them loses
    /// time for nothing. This stands in for the cost estimate of a query
    /// optimizer.
    bool selectiveBuildOnly{true};

    /// If true, places no filter on a scan that a join of the scan's own
    /// pipeline with a filtered build side already filters. Such a scan
    /// returns a part of its rows, and what a second filter removes on top
    /// rarely makes up for the time the scan waits for it. Like
    /// 'selectiveBuildOnly', this stands in for a cost estimate.
    bool skipScansFilteredInPipeline{true};
  };

  /// The filters derived for a plan.
  struct Result {
    /// One per join key that reaches at least one scan.
    std::vector<core::DynamicFilterSource> sources;

    /// The scans each filter applies to.
    std::vector<core::DynamicFilterTarget> targets;
  };

  /// Returns the filters for the plan under 'root'.
  ///
  /// A filter is made for a key of a hash join if removing the probe-side
  /// rows without a match before the join keeps the result (inner, left semi
  /// filter, right semi filter and right joins that are not null-aware), the
  /// key is an integer or a date, and the probe-side key is a column of a
  /// table scan carried unchanged up to the join. The column is followed
  /// down from the join through these nodes only: projections that pass it as
  /// is, filters, local exchanges, aggregations without global grouping sets
  /// that group by it, the probe side of joins that return a probe row as it
  /// came or not at all (inner, left, left semi, anti) and the build side of
  /// inner and right joins. Any other node ends the search.
  static Result plan(const core::PlanNode& root, const Options& options);
};

} // namespace facebook::velox::exec
