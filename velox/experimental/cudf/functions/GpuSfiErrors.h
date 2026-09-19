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

#include "velox/common/base/Exceptions.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <optional>

namespace facebook::velox::cudf_velox::gpu_sfi {

/// What class of error a declined row hit, host-side mirror of the device
/// enum. Kept in its own header so host code -- operators, the evaluator --
/// does not have to include a .cuh.
enum class ErrorClass : uint8_t {
  kNone = 0,
  /// A TRY above the expression may turn this row into a null.
  kUserError = 1,
  /// A TRY must not swallow this; the query has to fail.
  kRuntimeError = 2,
};

/// Collects the rows a GPU simple-function launch declined, for one
/// evaluation of one expression tree.
///
/// Opt-in, and that is the point. A caller passes one of these only if it can
/// do something about a declined row -- which today means it still holds the
/// input and the expression, so it can re-evaluate on the CPU and let Velox
/// raise the real error. A caller that cannot recover passes nothing and the
/// launch behaves as it always has, because a mechanism that reports an error
/// nobody can act on is worse than one that does not: the row would be nulled,
/// turning an error into a different answer.
///
/// Why the caller and not the node: a `GpuSfiExpression` is one node in a
/// tree, and the tree may sit under a conditional that discards exactly the
/// rows that failed. Only the owner of the whole expression knows that, so
/// only the owner can decide. See GPU_SFI_ERROR_DESIGN.md, constraint C4.
class GpuSfiErrors {
 public:
  GpuSfiErrors(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
      : stream_(stream), mr_(mr) {}

  /// The buffer a launch records into: one byte per row, holding an
  /// ErrorClass. Allocated once and shared by every launch under this
  /// evaluation, which is what makes the result a union over the whole tree.
  ///
  /// A launch writes a byte only when it declines a row, never zero, so a
  /// later launch cannot erase an earlier one's mark. Two launches declining
  /// the same row leave whichever wrote last, which the reduction then treats
  /// as that row's class -- immaterial while the decision is per batch, and
  /// worth revisiting if it ever becomes per row.
  uint8_t* declinedRows(cudf::size_type numRows) {
    if (!buffer_.has_value()) {
      buffer_.emplace(numRows, stream_, mr_);
      // rmm::device_uvector does not zero, and an uninitialized byte here
      // declines a row that did nothing wrong.
      CUDF_CUDA_TRY(
          cudaMemsetAsync(buffer_->data(), 0, buffer_->size(), stream_.get()));
    }
    VELOX_CHECK_LE(
        static_cast<std::size_t>(numRows),
        buffer_->size(),
        "A launch under one evaluation has more rows than the first");
    return buffer_->data();
  }

  /// Reduces what the launches recorded to the worst class in the batch.
  ///
  /// Deliberately not done per launch. Reading a device scalar synchronises
  /// the stream, and doing that inside each eval() would cost a sync per
  /// expression node per batch on the happy path -- for a query where nothing
  /// ever fails. Here the owner pays it once, after everything it cares about
  /// has been queued, and only if it asks.
  ///
  /// Max rather than any(): the classes are ordered, so one pass answers both
  /// "was anything declined" and "may a TRY swallow it".
  ErrorClass resolve() {
    if (!buffer_.has_value()) {
      return ErrorClass::kNone;
    }
    auto worst = cudf::reduce(
        cudf::column_view{
            cudf::data_type{cudf::type_id::UINT8},
            static_cast<cudf::size_type>(buffer_->size()),
            buffer_->data(),
            nullptr,
            0},
        *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
        cudf::data_type{cudf::type_id::UINT8},
        stream_,
        mr_);
    auto const* scalar =
        static_cast<cudf::numeric_scalar<uint8_t>*>(worst.get());
    if (!scalar->is_valid(stream_)) {
      return ErrorClass::kNone;
    }
    return static_cast<ErrorClass>(scalar->value(stream_));
  }

 private:
  cuda::stream_ref stream_;
  rmm::device_async_resource_ref mr_;
  std::optional<rmm::device_uvector<uint8_t>> buffer_;
};

} // namespace facebook::velox::cudf_velox::gpu_sfi
