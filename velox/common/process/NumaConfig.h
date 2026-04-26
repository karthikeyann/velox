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
#include <vector>

namespace facebook::velox::process {

struct NumaNodeInfo {
  int nodeId;
  std::vector<int> cpuIds;
  uint64_t totalMemoryBytes{0};
};

/// Lightweight, lazily-initialized NUMA topology descriptor.
///
/// Queries /sys/devices/system/node/ once and caches the result.
/// On non-NUMA or single-node systems, isAvailable() returns false and
/// all assignment helpers return 0 (the single node).
///
/// An alternate sysfs root can be injected for testing.
class NumaConfig {
 public:
  /// Returns the process-wide singleton, populated from the real sysfs.
  static const NumaConfig& instance();

  /// Build from an explicit sysfs path (for unit tests).
  static NumaConfig fromSysFs(const std::string& sysfsRoot);

  bool isAvailable() const {
    return nodes_.size() > 1;
  }

  int numNodes() const {
    return static_cast<int>(nodes_.size());
  }

  const NumaNodeInfo& node(int id) const {
    return nodes_.at(id);
  }

  const std::vector<NumaNodeInfo>& nodes() const {
    return nodes_;
  }

  /// Assign a NUMA node for a driver identified by (splitGroupId, pipelineId).
  /// Round-robins across available nodes. Returns 0 when NUMA is unavailable.
  int assignNode(uint32_t splitGroupId, int pipelineId) const;

 private:
  NumaConfig() = default;

  void discoverFromSysFs(const std::string& sysfsRoot);

  /// Parse a CPU list string like "0-3,8-11" into individual CPU ids.
  static std::vector<int> parseCpuList(const std::string& cpuListStr);

  std::vector<NumaNodeInfo> nodes_;
};

} // namespace facebook::velox::process
