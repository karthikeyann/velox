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

#include "velox/common/process/NumaConfig.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <glog/logging.h>

namespace facebook::velox::process {

namespace {

const std::string kDefaultSysfsRoot = "/sys/devices/system/node";

uint64_t readMemInfoTotal(const std::string& nodeDir) {
  std::ifstream meminfo(nodeDir + "/meminfo");
  if (!meminfo.is_open()) {
    return 0;
  }
  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.find("MemTotal") == std::string::npos) {
      continue;
    }
    // Line format: "Node 0 MemTotal:    16000000 kB"
    // Extract the numeric value preceding "kB".
    auto colonPos = line.find(':');
    if (colonPos == std::string::npos) {
      return 0;
    }
    std::istringstream iss(line.substr(colonPos + 1));
    uint64_t value = 0;
    if (iss >> value) {
      return value * 1024; // meminfo reports in kB.
    }
    return 0;
  }
  return 0;
}

} // namespace

/*static*/ const NumaConfig& NumaConfig::instance() {
  static NumaConfig config = NumaConfig::fromSysFs(kDefaultSysfsRoot);
  return config;
}

/*static*/ NumaConfig NumaConfig::fromSysFs(const std::string& sysfsRoot) {
  NumaConfig config;
  config.discoverFromSysFs(sysfsRoot);
  return config;
}

void NumaConfig::discoverFromSysFs(const std::string& sysfsRoot) {
  nodes_.clear();
  namespace fs = std::filesystem;

  if (!fs::exists(sysfsRoot)) {
    // Single node fallback.
    NumaNodeInfo fallback;
    fallback.nodeId = 0;
    nodes_.push_back(std::move(fallback));
    return;
  }

  std::vector<int> nodeIds;
  for (const auto& entry : fs::directory_iterator(sysfsRoot)) {
    const auto name = entry.path().filename().string();
    if (name.size() > 4 && name.substr(0, 4) == "node") {
      try {
        nodeIds.push_back(std::stoi(name.substr(4)));
      } catch (...) {
        continue;
      }
    }
  }

  if (nodeIds.empty()) {
    NumaNodeInfo fallback;
    fallback.nodeId = 0;
    nodes_.push_back(std::move(fallback));
    return;
  }

  std::sort(nodeIds.begin(), nodeIds.end());

  for (int id : nodeIds) {
    NumaNodeInfo info;
    info.nodeId = id;

    const auto nodeDir = sysfsRoot + "/node" + std::to_string(id);

    // Read CPU list.
    std::ifstream cpuListFile(nodeDir + "/cpulist");
    if (cpuListFile.is_open()) {
      std::string cpuListStr;
      std::getline(cpuListFile, cpuListStr);
      info.cpuIds = parseCpuList(cpuListStr);
    }

    info.totalMemoryBytes = readMemInfoTotal(nodeDir);
    nodes_.push_back(std::move(info));
  }
}

/*static*/ std::vector<int> NumaConfig::parseCpuList(
    const std::string& cpuListStr) {
  std::vector<int> cpus;
  std::istringstream stream(cpuListStr);
  std::string token;
  while (std::getline(stream, token, ',')) {
    auto dash = token.find('-');
    if (dash != std::string::npos) {
      int start = std::stoi(token.substr(0, dash));
      int end = std::stoi(token.substr(dash + 1));
      for (int i = start; i <= end; ++i) {
        cpus.push_back(i);
      }
    } else {
      if (!token.empty()) {
        cpus.push_back(std::stoi(token));
      }
    }
  }
  return cpus;
}

int NumaConfig::assignNode(uint32_t splitGroupId, int pipelineId) const {
  if (nodes_.size() <= 1) {
    return 0;
  }
  const auto n = static_cast<int>(nodes_.size());
  return static_cast<int>((splitGroupId + pipelineId) % n);
}

} // namespace facebook::velox::process
