/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cudf/column/column_view.hpp>

#include <rmm/device_uvector.hpp>

#include <memory>
#include <utility>

namespace facebook::velox::cudf_velox {

// Immutable direct-address map. Construction requires verified unique,
// non-null build keys; callers must not infer uniqueness from a sample.
struct DenseJoinIndex {
  int64_t minimum;
  rmm::device_uvector<cudf::size_type> rows;

  DenseJoinIndex(
      int64_t minimum,
      size_t range,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr)
      : minimum(minimum), rows(range, stream, mr) {}
};

std::unique_ptr<DenseJoinIndex> tryMakeDenseJoinIndex(
    cudf::column_view keys,
    uint64_t maxRange,
    double minDensity,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

std::pair<
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
probeDenseJoinIndex(
    const DenseJoinIndex& index,
    cudf::column_view keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
