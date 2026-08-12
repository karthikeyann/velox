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

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <optional>
#include <string_view>

namespace facebook::velox::cudf_velox {

extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
    output_mr_;

/// Returns the temporary resource for the operator currently running on this
/// thread. Falls back to RMM's current resource when attribution is disabled.
rmm::device_async_resource_ref get_temp_mr();

/// Returns the memory resource designated for output vector allocations.
rmm::device_async_resource_ref get_output_mr();

/// Temporarily pins resources for connector work that can execute outside a
/// Driver thread, such as TableScan split preloading.
class ScopedCudfMemoryResources {
 public:
  ScopedCudfMemoryResources(
      rmm::device_async_resource_ref temp,
      rmm::device_async_resource_ref output);
  ~ScopedCudfMemoryResources();

  ScopedCudfMemoryResources(const ScopedCudfMemoryResources&) = delete;
  ScopedCudfMemoryResources& operator=(const ScopedCudfMemoryResources&) =
      delete;

 private:
  std::optional<rmm::device_async_resource_ref> previousTemp_;
  std::optional<rmm::device_async_resource_ref> previousOutput_;
};

/**
 * @brief Creates a memory resource based on the given mode.
 *
 * @param mode rmm::mr::pool_memory_resource mode.
 * @param percent The initial percent of GPU memory to allocate for memory
 * resource.
 */
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
createMemoryResource(std::string_view mode, int percent);

/**
 * @brief Returns the global CUDA stream pool used by cudf.
 */
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

} // namespace facebook::velox::cudf_velox
