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

#include "velox/type/Filter.h"

namespace facebook::velox::exec {

/// Combines the dynamic filters that the tasks running one hash join publish
/// for one filter id (Task::dynamicFilterFuture()) into the filter to deliver
/// to the tasks that apply it (Task::addDynamicFilter()). Used by whoever runs
/// the tasks. Not thread-safe.
class DynamicFilterCollector {
 public:
  /// How the build side of the join is distributed over the tasks.
  enum class BuildDistribution {
    /// Each task builds from a part of the build side. The result is the
    /// union of the filters of all tasks.
    kPartitioned,
    /// Each task builds from the whole build side. The result is the filter
    /// of the first task.
    kBroadcast,
  };

  /// Limits of the combined filter.
  struct Options {
    /// The largest number of values of a union of sets of values. A larger
    /// union becomes the range of its values.
    int32_t maxValues{100'000};
  };

  /// @param numTasks Number of tasks that publish the filter.
  DynamicFilterCollector(
      BuildDistribution distribution,
      int32_t numTasks,
      const Options& options);

  /// Adds the filter of one task; nullptr if the task opted out. Returns
  /// done().
  bool add(const common::FilterPtr& filter);

  /// Returns true once the result is final: every task has added its filter,
  /// the first one has for a broadcast build, or a task opted out.
  bool done() const {
    return done_;
  }

  /// Returns the combined filter, or nullptr if the filter is disabled: a
  /// task opted out or the filters of the tasks cannot be combined. Requires
  /// done().
  const common::FilterPtr& result() const;

  /// Returns a filter that passes every value 'left' or 'right' passes, and
  /// possibly more, or nullptr if the two kinds of filter have no union here.
  /// Unions exist for combinations of AlwaysFalse, BigintRange and sets of
  /// integer values, and for sets of string values.
  static common::FilterPtr unionOf(
      const common::FilterPtr& left,
      const common::FilterPtr& right,
      const Options& options);

 private:
  const BuildDistribution distribution_;
  const int32_t numTasks_;
  const Options options_;

  // Number of filters added.
  int32_t numAdded_{0};

  // True once 'result_' is final.
  bool done_{false};

  // The union of the filters added so far, or nullptr if disabled.
  common::FilterPtr result_;
};

} // namespace facebook::velox::exec
