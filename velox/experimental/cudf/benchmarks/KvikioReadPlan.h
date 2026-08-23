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

#include <cstddef>
#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Selects how a read plan distributes requests over the available bytes.
enum class ReadMode {
  /// Covers every byte at most once, so nothing in the run is served from a
  /// cache that the same run populated.
  kCold,
  /// Draws offsets at random, allowing overlap and repetition.
  kWarm,
};

/// Names one object from the manifest together with its size.
struct TargetInfo {
  /// URI exactly as it appeared in the manifest.
  std::string uri;

  /// Object size in bytes as reported by the remote endpoint.
  uint64_t size;
};

/// Describes a single range request against one target.
struct ReadTask {
  /// Index into the target vector the plan was built from.
  size_t targetIndex;

  /// Byte offset within that target.
  uint64_t offset;

  /// Number of bytes to read.
  uint64_t size;
};

/// Carries the inputs that fully determine a read plan.
struct ReadPlanOptions {
  /// Cold or warm request distribution.
  ReadMode mode{ReadMode::kCold};

  /// Size of each range request. In cold mode the last request covering a
  /// target is shorter when the target size is not a multiple of this.
  uint64_t requestBytes{8ULL << 20};

  /// Total bytes the plan covers across all targets.
  uint64_t measurementBytes{1ULL << 30};

  /// Seeds warm-mode offset selection. Ignored in cold mode.
  uint64_t seed{0};
};

/// Reads a manifest holding one URI per line. Skips blank lines and lines
/// whose first non-whitespace character is '#', and trims surrounding
/// whitespace from the rest.
std::vector<std::string> parseManifest(std::istream& in);

/// Returns the combined size of every target in 'targets'.
uint64_t totalTargetBytes(const std::vector<TargetInfo>& targets);

/// Builds the sequence of range requests to issue against 'targets'.
///
/// Cold mode emits pairwise disjoint tasks in target order, so no byte is
/// read twice. It rejects a 'measurementBytes' larger than the combined size
/// of 'targets', because wrapping around to satisfy the request would serve
/// the excess from cache and silently report a warm number as a cold one.
///
/// Warm mode draws offsets at random and may repeat them, so it accepts any
/// 'measurementBytes'.
///
/// Throws if 'targets' is empty or holds no bytes.
std::vector<ReadTask> makeReadPlan(
    const std::vector<TargetInfo>& targets,
    const ReadPlanOptions& options);

} // namespace facebook::velox::cudf_velox
