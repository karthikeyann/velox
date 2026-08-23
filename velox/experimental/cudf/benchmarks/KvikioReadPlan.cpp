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

#include "velox/experimental/cudf/benchmarks/KvikioReadPlan.h"

#include "velox/common/base/Exceptions.h"

#include <algorithm>
#include <random>

namespace facebook::velox::cudf_velox {

namespace {

std::string trim(const std::string& line) {
  const auto begin = line.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = line.find_last_not_of(" \t\r\n");
  return line.substr(begin, end - begin + 1);
}

std::vector<ReadTask> makeColdPlan(
    const std::vector<TargetInfo>& targets,
    const ReadPlanOptions& options,
    uint64_t availableBytes) {
  VELOX_USER_CHECK_LE(
      options.measurementBytes,
      availableBytes,
      "Cold mode cannot read more bytes than the manifest holds. Reading the "
      "excess would repeat bytes already read and report a warm result. "
      "Requested: {}, available: {}",
      options.measurementBytes,
      availableBytes);

  std::vector<ReadTask> plan;
  uint64_t emitted{0};
  for (size_t index = 0;
       index < targets.size() && emitted < options.measurementBytes;
       ++index) {
    const uint64_t targetSize = targets[index].size;
    for (uint64_t offset = 0;
         offset < targetSize && emitted < options.measurementBytes;
         offset += options.requestBytes) {
      const uint64_t size = std::min(
          {options.requestBytes,
           targetSize - offset,
           options.measurementBytes - emitted});
      plan.push_back(ReadTask{index, offset, size});
      emitted += size;
    }
  }
  return plan;
}

std::vector<ReadTask> makeWarmPlan(
    const std::vector<TargetInfo>& targets,
    const ReadPlanOptions& options) {
  std::mt19937_64 rng(options.seed);
  std::uniform_int_distribution<size_t> targetDistribution(
      0, targets.size() - 1);

  std::vector<ReadTask> plan;
  uint64_t emitted{0};
  while (emitted < options.measurementBytes) {
    size_t index = targetDistribution(rng);
    // Skip zero-length targets rather than looping forever on them.
    if (targets[index].size == 0) {
      continue;
    }
    const uint64_t targetSize = targets[index].size;
    const uint64_t span = std::min(options.requestBytes, targetSize);
    const uint64_t maxOffset = targetSize - span;
    const uint64_t offset = maxOffset == 0
        ? 0
        : std::uniform_int_distribution<uint64_t>(0, maxOffset)(rng);
    const uint64_t size = std::min(span, options.measurementBytes - emitted);
    plan.push_back(ReadTask{index, offset, size});
    emitted += size;
  }
  return plan;
}

} // namespace

std::vector<std::string> parseManifest(std::istream& in) {
  std::vector<std::string> uris;
  std::string line;
  while (std::getline(in, line)) {
    const auto trimmed = trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }
    uris.push_back(trimmed);
  }
  return uris;
}

uint64_t totalTargetBytes(const std::vector<TargetInfo>& targets) {
  uint64_t total{0};
  for (const auto& target : targets) {
    total += target.size;
  }
  return total;
}

std::vector<ReadTask> makeReadPlan(
    const std::vector<TargetInfo>& targets,
    const ReadPlanOptions& options) {
  VELOX_USER_CHECK_GT(
      options.requestBytes, 0, "Request size must be greater than zero");
  VELOX_USER_CHECK_GT(
      options.measurementBytes,
      0,
      "Measurement size must be greater than zero");

  const uint64_t availableBytes = totalTargetBytes(targets);
  VELOX_USER_CHECK_GT(
      availableBytes,
      0,
      "Manifest contains no readable bytes. Targets: {}",
      targets.size());

  if (options.mode == ReadMode::kCold) {
    return makeColdPlan(targets, options, availableBytes);
  }
  return makeWarmPlan(targets, options);
}

} // namespace facebook::velox::cudf_velox
