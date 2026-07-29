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

#include "velox/exec/DriverListener.h"

#include <memory>
#include <string>

namespace facebook::velox::cudf_velox {

/// Attributes GPU allocations and records execution spans for observed Tasks.
///
/// Replaces the per-operator wrapping that only cuDF operators could carry, so
/// attribution and spans now cover every operator a driver runs, including
/// TableScan and the cuDF conversion operators.
class GpuMemoryCaptureDriverListenerFactory
    : public exec::DriverListenerFactory {
 public:
  /// Returns a listener while GPU-memory tracking is enabled, and nullptr
  /// otherwise so untracked Tasks keep their callback-free driver path.
  std::shared_ptr<exec::DriverListener> create(
      const std::string& taskId,
      const std::string& taskUuid,
      const core::QueryConfig& config) override;
};

} // namespace facebook::velox::cudf_velox
