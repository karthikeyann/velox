/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/resource_ref.hpp>

namespace facebook::velox::cudf_velox {

// Lossless encoding of one/two non-null strings of at most three bytes each.
// Every batch is validated. Unsupported types/nulls/long strings return null.
std::unique_ptr<cudf::column> tryPackShortStringKeys(
    cudf::table_view keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

std::vector<std::unique_ptr<cudf::column>> unpackShortStringKeys(
    cudf::column_view packed,
    int numKeys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
