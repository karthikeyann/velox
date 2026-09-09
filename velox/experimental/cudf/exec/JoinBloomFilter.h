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

#include <cudf/column/column_view.hpp>

#include <rmm/device_uvector.hpp>

namespace facebook::velox::cudf_velox {

// Blocked, two-bit Bloom filter on one integer join key. A positive response
// is only a candidate: the full original tuple must still pass the hash join.
struct JoinBloomFilter {
  rmm::device_uvector<uint64_t> words;
  // A nonzero range selects an exact integer bitmap instead of hashed bits.
  int64_t minimum{0};
  uint64_t range{0};
  JoinBloomFilter(
      size_t size,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr)
      : words(size, stream, mr) {}
};

std::unique_ptr<JoinBloomFilter> makeJoinBloomFilter(
    cudf::column_view keys,
    uint64_t bitsPerRow,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    uint64_t maxExactRange = 0,
    uint64_t maxExactToBloomRatio = 4);

std::unique_ptr<rmm::device_uvector<cudf::size_type>> filterJoinProbeKeys(
    const JoinBloomFilter& filter,
    cudf::column_view keys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

void remapJoinProbeIndices(
    rmm::device_uvector<cudf::size_type>& joinedIndices,
    const rmm::device_uvector<cudf::size_type>& candidateIndices,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::cudf_velox
