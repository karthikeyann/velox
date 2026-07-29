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

#include "velox/experimental/cudf/exec/GpuMemoryOwner.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox {

namespace gpu_memory_detail {

/// Registers the query, task, PlanNode, driver, and operator counter
/// hierarchy for one allocation owner.
void registerGpuMemoryNvtxOwner(
    uint64_t ownerId,
    uint64_t planNodeId,
    const GpuMemoryOwner& owner) noexcept;

/// Samples every counter affected by one logical-memory transition.
void emitGpuMemoryNvtxUpdate(const GpuMemoryTraceUpdate& update) noexcept;

/// Zeroes and releases the counter hierarchy.
void resetGpuMemoryNvtxCounters() noexcept;

} // namespace gpu_memory_detail

} // namespace facebook::velox::cudf_velox
