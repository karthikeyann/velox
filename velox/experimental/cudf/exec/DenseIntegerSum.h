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
#include <cudf/table/table.hpp>

#include <rmm/resource_ref.hpp>

namespace facebook::velox::cudf_velox {

// One INT32/INT64 key and one nullable INT64 SUM (or COUNT of all rows).
// Values are accumulated only
// while conservative positive/negative bounds rule out signed overflow.
// add() returns false without changing state when a batch requires fallback.
class DenseIntegerSum {
 public:
  DenseIntegerSum(
      cudf::data_type keyType,
      uint64_t maxRange,
      bool ignoreNullKeys,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool countRows = false,
      uint64_t count32MaxRows = 0);
  ~DenseIntegerSum();
  DenseIntegerSum(const DenseIntegerSum&) = delete;
  DenseIntegerSum& operator=(const DenseIntegerSum&) = delete;

  bool add(cudf::column_view keys, cudf::column_view values);
  std::unique_ptr<cudf::table> finalize();
  uint64_t range() const;
  uint64_t bytes() const;
  rmm::cuda_stream_view stream() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace facebook::velox::cudf_velox
