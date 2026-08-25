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

#include <array>
#include <cstddef>
#include <functional>
#include <string_view>

namespace facebook::velox::cudf_velox {

/// A resource a cuDF benchmark takes ownership of while initializing.
///
/// The enumerator order is the order CudfBenchmarkLifecycle releases them in.
/// A resource that another one borrows or references comes first, so that by
/// the time a resource is released nothing is left that can reach it.
enum class CudfBenchmarkResource {
  /// The connector registered under the benchmark's connector id. It borrows
  /// the I/O executor and owns the data sources and file handles that hold a
  /// reference to a cached file system.
  kConnector,
  /// The base benchmark's own state, such as the query builder and the data
  /// cache, which can hold readers opened against that same file system.
  kBaseBenchmark,
  /// The I/O executor the connector borrowed.
  kIoExecutor,
  /// The cuDF registration, including its device memory resource, which has to
  /// outlive every device buffer the resources above hold.
  kCudf,
  /// The S3 file system registration. Finalizing it is process-global and
  /// one-shot, and requires every cached S3 file system to be unreferenced,
  /// which is why it is released last.
  kS3FileSystem,
};

/// Number of CudfBenchmarkResource enumerators. Derived from the last one,
/// which the release-order contract above pins in place.
constexpr size_t kNumCudfBenchmarkResources =
    static_cast<size_t>(CudfBenchmarkResource::kS3FileSystem) + 1;

/// Tracks which resources a benchmark owns and releases each exactly once.
///
/// A resource is forgotten only once its release callback has returned
/// normally, so a release that throws leaves the resource owned and a later
/// release() retries it. release() attempts every owned resource before
/// reporting anything, so one resource that cannot be released does not strand
/// the rest, and then rethrows the first failure it saw. Once every resource
/// has been released, release() does nothing.
class CudfBenchmarkLifecycle {
 public:
  CudfBenchmarkLifecycle() = default;

  /// Neither copyable nor movable. Releasing each resource exactly once means
  /// there is exactly one owner: a copy would release the same resources a
  /// second time, and no caller needs to hand ownership on, so a move is
  /// deleted rather than defined.
  CudfBenchmarkLifecycle(const CudfBenchmarkLifecycle&) = delete;
  CudfBenchmarkLifecycle& operator=(const CudfBenchmarkLifecycle&) = delete;
  CudfBenchmarkLifecycle(CudfBenchmarkLifecycle&&) = delete;
  CudfBenchmarkLifecycle& operator=(CudfBenchmarkLifecycle&&) = delete;

  /// Records that the benchmark owns 'resource' and that 'release' releases it,
  /// replacing any callback previously recorded for 'resource'. 'release' must
  /// be callable again after it throws.
  void own(CudfBenchmarkResource resource, std::function<void()> release);

  /// Whether 'resource' is still owned.
  bool owns(CudfBenchmarkResource resource) const;

  /// Whether any resource is still owned.
  bool ownsAny() const;

  /// Releases every owned resource in CudfBenchmarkResource order, then
  /// rethrows the first failure if there was one.
  void release();

  /// Releases every owned resource while another failure is being reported.
  ///
  /// Cleaning up must not replace the failure that caused it, so a release that
  /// throws is logged, with 'context' naming the operation that failed, instead
  /// of propagating. Returns whether everything was released.
  bool releaseAfterFailure(std::string_view context);

 private:
  // Release callback per resource, indexed by CudfBenchmarkResource. An empty
  // entry means the resource is not owned.
  std::array<std::function<void()>, kNumCudfBenchmarkResources> release_;
};

} // namespace facebook::velox::cudf_velox
