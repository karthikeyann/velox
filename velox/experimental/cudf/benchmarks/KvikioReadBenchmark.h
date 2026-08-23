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

#include "velox/experimental/cudf/benchmarks/KvikioReadPlan.h"

#include <kvikio/remote_handle.hpp>

#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Opens every URI in a manifest and keeps the handles alive for the run.
///
/// A single handle is shared by all reader threads. That is safe because
/// `kvikio::RemoteHandle::pread` already dispatches its own work to KvikIO's
/// thread pool, with every task calling `read()` on the same handle.
class RemoteTargets {
 public:
  /// Opens each URI in order. Opening also resolves DNS and completes the TLS
  /// handshake, keeping that cost out of the measured window.
  ///
  /// Throws naming the offending URI if any target fails to open, rather than
  /// dropping it, because a shorter manifest changes what a run measures
  /// without changing what it reports.
  explicit RemoteTargets(const std::vector<std::string>& uris);

  /// Returns the URI and size of every target, in manifest order.
  const std::vector<TargetInfo>& infos() const {
    return infos_;
  }

  /// Returns the handle for the target at 'index'.
  kvikio::RemoteHandle& handleAt(size_t index) {
    return handles_[index];
  }

  /// Returns the combined size of every target.
  uint64_t totalBytes() const;

 private:
  // URI and size per target, parallel to handles_.
  std::vector<TargetInfo> infos_;

  // Open handle per target, parallel to infos_.
  std::vector<kvikio::RemoteHandle> handles_;
};

/// Reports what one timed pass over a read plan moved.
struct RunResult {
  /// Holds the total payload bytes transferred.
  uint64_t bytesRead{0};

  /// Counts the range requests issued by the benchmark; KvikIO may split each
  /// further when a non-zero task size is in effect.
  uint64_t numRequests{0};

  /// Records the wall time of the pass, excluding target opening.
  uint64_t elapsedMicros{0};
};

/// Issues every task in 'plan' and returns what the pass moved.
///
/// 'numThreads' reader threads each own one destination buffer of
/// 'requestBytes' and pull tasks until the plan is exhausted. A
/// 'kvikioTaskSize' of zero issues each task as a single range GET; any other
/// value hands the task to KvikIO's thread pool split at that granularity.
/// 'deviceMemory' selects a device rather than host destination, which for a
/// remote source routes through KvikIO's bounce buffer.
RunResult runPlan(
    RemoteTargets& targets,
    const std::vector<ReadTask>& plan,
    int32_t numThreads,
    uint64_t requestBytes,
    uint64_t kvikioTaskSize,
    bool deviceMemory);

/// Reads a manifest from 'path'. Throws if the file is missing, unreadable,
/// or yields no URIs.
std::vector<std::string> readManifestFile(const std::string& path);

} // namespace facebook::velox::cudf_velox
