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

#include "velox/common/base/NumaAwareExecutor.h"

#include <fmt/format.h>
#include <glog/logging.h>

#ifdef __linux__
#include <sched.h>
#endif

namespace facebook::velox {

NumaThreadFactory::NumaThreadFactory(
    const std::string& prefix,
    int numaNode,
    std::vector<int> cpuIds)
    : folly::NamedThreadFactory(prefix),
      numaNode_(numaNode),
      cpuIds_(std::move(cpuIds)) {}

std::thread NumaThreadFactory::newThread(folly::Func&& func) {
#ifdef __linux__
  auto pinnedCpuIds = cpuIds_;
  auto node = numaNode_;
  return folly::NamedThreadFactory::newThread(
      [f = std::move(func), pinnedCpuIds, node]() mutable {
        if (!pinnedCpuIds.empty()) {
          cpu_set_t cpuSet;
          CPU_ZERO(&cpuSet);
          for (int cpu : pinnedCpuIds) {
            CPU_SET(cpu, &cpuSet);
          }
          if (sched_setaffinity(0, sizeof(cpuSet), &cpuSet) != 0) {
            LOG(WARNING) << "Failed to set CPU affinity for NUMA node " << node
                         << ": errno=" << errno;
          }
        }
        f();
      });
#else
  return folly::NamedThreadFactory::newThread(std::move(func));
#endif
}

NumaAwareExecutor::NumaAwareExecutor(
    Options opts,
    const process::NumaConfig& config) {
  const int numNodes = config.numNodes();
  pools_.reserve(numNodes);

  for (int i = 0; i < numNodes; ++i) {
    const auto& nodeInfo = config.node(i);
    const size_t numThreads = opts.threadsPerNode > 0
        ? opts.threadsPerNode
        : std::max<size_t>(1, nodeInfo.cpuIds.size());

    auto prefix = fmt::format("{}-n{}", opts.threadPrefix, nodeInfo.nodeId);

    std::shared_ptr<folly::ThreadFactory> factory;
    if (opts.pinThreads && !nodeInfo.cpuIds.empty()) {
      factory = std::make_shared<NumaThreadFactory>(
          prefix, nodeInfo.nodeId, nodeInfo.cpuIds);
    } else {
      factory = std::make_shared<folly::NamedThreadFactory>(prefix);
    }

    pools_.push_back(
        std::make_unique<folly::CPUThreadPoolExecutor>(numThreads, factory));
  }

  LOG(INFO) << "NumaAwareExecutor created with " << pools_.size()
            << " NUMA node pool(s)";
}

NumaAwareExecutor::~NumaAwareExecutor() {
  for (auto& pool : pools_) {
    pool->join();
  }
}

void NumaAwareExecutor::add(folly::Func func) {
  const auto idx = roundRobinCounter_.fetch_add(1, std::memory_order_relaxed) %
      pools_.size();
  pools_[idx]->add(std::move(func));
}

void NumaAwareExecutor::addOnNode(folly::Func func, int numaNode) {
  if (numaNode < 0 || numaNode >= static_cast<int>(pools_.size())) {
    add(std::move(func));
    return;
  }
  pools_[numaNode]->add(std::move(func));
}

} // namespace facebook::velox
