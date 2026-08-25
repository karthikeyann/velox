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

#include "velox/core/PlanNode.h"

#include <memory>
#include <sstream>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Runtime-stat name under which the total decoded row count is reported.
inline constexpr std::string_view kDiscardedRows{"discardedRows"};

/// Runtime-stat name under which the total decoded byte count is reported.
inline constexpr std::string_view kDiscardedBytes{"discardedBytes"};

/// Runtime-stat name under which the total batch count is reported.
inline constexpr std::string_view kDiscardedBatches{"discardedBatches"};

/// Sink plan node that drops every input CudfVector immediately.
///
/// Its output type mirrors the source output type, but the operator produces
/// no output batches.  The operator accumulates row, byte, and batch counts
/// and exposes them as runtime stats after no-more-input.
class BenchmarkDiscardNode final : public core::PlanNode {
 public:
  /// Constructs the node.  The plan-node ID is derived deterministically
  /// from the source ID by appending "-discard".
  BenchmarkDiscardNode(const core::PlanNodeId& id, core::PlanNodePtr source);

  /// Returns the source output type (the node itself produces no batches).
  const RowTypePtr& outputType() const override;

  /// Returns the single upstream source.
  const std::vector<core::PlanNodePtr>& sources() const override;

  /// Returns "BenchmarkDiscard".
  std::string_view name() const override;

 private:
  // Emits nothing; the node has no configuration details to surface.
  void addDetails(std::stringstream& stream) const override;

  // The single upstream node.
  const std::vector<core::PlanNodePtr> sources_;
};

/// Wraps a plan subtree with a BenchmarkDiscardNode.
///
/// The node ID is source->id() + "-discard".
core::PlanNodePtr addBenchmarkDiscard(core::PlanNodePtr source);

/// Registers the BenchmarkDiscard plan-node translator (once, globally) and
/// registers a fresh OperatorAdapter into the current adapter registry.
///
/// Call immediately after registerCudf() because registerCudf() rebuilds the
/// adapter registry from scratch.
void registerCudfBenchmarkDiscard();

} // namespace facebook::velox::cudf_velox
