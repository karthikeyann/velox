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

#include <folly/Executor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "velox/common/process/NumaConfig.h"

namespace facebook::velox {

/// Thread factory that pins each created thread to a specific set of CPUs
/// belonging to a NUMA node. This ensures Linux first-touch memory policy
/// allocates pages on the local node.
class NumaThreadFactory : public folly::NamedThreadFactory {
 public:
  NumaThreadFactory(
      const std::string& prefix,
      int numaNode,
      std::vector<int> cpuIds);

  std::thread newThread(folly::Func&& func) override;

 private:
  int numaNode_;
  std::vector<int> cpuIds_;
};

struct NumaAwareExecutorOptions {
  /// Threads per NUMA node. 0 = one thread per CPU on the node.
  size_t threadsPerNode{0};
  std::string threadPrefix{"VeloxNuma"};
  /// When false, pools are created per-node but threads are not pinned.
  /// Useful for functional testing on single-socket machines.
  bool pinThreads{true};
};

/// A folly::Executor that maintains one CPUThreadPoolExecutor per NUMA node,
/// each with threads pinned to that node's CPUs.
///
/// - `add(Func)` round-robins across nodes (backward-compatible fallback).
/// - `addOnNode(Func, node)` dispatches to a specific node's pool.
///
/// When only one NUMA node exists, behaves identically to a single
/// CPUThreadPoolExecutor.
class NumaAwareExecutor : public folly::Executor {
 public:
  using Options = NumaAwareExecutorOptions;

  explicit NumaAwareExecutor(
      Options opts = {},
      const process::NumaConfig& config = process::NumaConfig::instance());
  ~NumaAwareExecutor() override;

  NumaAwareExecutor(const NumaAwareExecutor&) = delete;
  NumaAwareExecutor& operator=(const NumaAwareExecutor&) = delete;

  void add(folly::Func func) override;

  /// Dispatch work to the pool bound to the given NUMA node.
  /// Falls back to round-robin if numaNode is out of range.
  void addOnNode(folly::Func func, int numaNode);

  int numNodes() const {
    return static_cast<int>(pools_.size());
  }

 private:
  std::vector<std::unique_ptr<folly::CPUThreadPoolExecutor>> pools_;
  std::atomic<uint64_t> roundRobinCounter_{0};
};

} // namespace facebook::velox
