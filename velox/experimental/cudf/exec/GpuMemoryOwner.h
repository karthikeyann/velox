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

#include <cstdint>
#include <string>

namespace facebook::velox::cudf_velox {

/// Identifies one concrete operator instance that originated an allocation.
struct GpuMemoryOwner {
  /// Universally unique Velox task identifier.
  std::string taskUuid;
  /// User-visible Velox task identifier.
  std::string taskId;
  /// Velox query identifier.
  std::string queryId;
  /// Velox plan node identifier.
  std::string planNodeId;
  /// Display-only Velox PlanNode type. Does not affect attribution identity.
  std::string planNodeType;
  /// Pipeline containing the operator instance.
  int32_t pipelineId{-1};
  /// Driver executing the operator instance.
  int32_t driverId{-1};
  /// Operator position within the pipeline.
  int32_t operatorId{-1};
  /// User-visible operator implementation name.
  std::string operatorType;

  /// Compares stable allocation identity and ignores display-only metadata.
  bool operator==(const GpuMemoryOwner& other) const {
    return taskUuid == other.taskUuid && taskId == other.taskId &&
        queryId == other.queryId && planNodeId == other.planNodeId &&
        pipelineId == other.pipelineId && driverId == other.driverId &&
        operatorId == other.operatorId && operatorType == other.operatorType;
  }
};

/// Describes one fully ordered logical-memory transition.
struct GpuMemoryTraceUpdate {
  /// Strictly increasing timestamp shared by all counters for this transition.
  uint64_t timestampNs{0};
  /// Strictly increasing event number used for diagnostics.
  uint64_t sequence{0};
  /// Registered allocation-origin owner.
  uint64_t ownerId{0};
  /// Registered task-local PlanNode aggregate.
  uint64_t planNodeId{0};
  /// Process-wide live logical bytes after this transition.
  uint64_t globalCurrentBytes{0};
  /// Process-wide high-water mark after this transition.
  uint64_t globalPeakBytes{0};
  /// Query live logical bytes after this transition.
  uint64_t queryCurrentBytes{0};
  /// Task, or PlanFragment instance, live logical bytes after this transition.
  uint64_t taskCurrentBytes{0};
  /// PlanNode live logical bytes after this transition.
  uint64_t planNodeCurrentBytes{0};
  /// Operator-instance live logical bytes after this transition.
  uint64_t ownerCurrentBytes{0};
  /// Signed allocation delta applied by this transition.
  int64_t deltaBytes{0};
};

} // namespace facebook::velox::cudf_velox
