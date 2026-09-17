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

// GPU shadow for velox/common/base/Status.h.
//
// This exists to let a header *parse*, not to make Status work on GPU. Several
// function headers define a checked variant next to an unchecked one -- Spark's
// checked_add beside add, for instance -- and a device translation unit has to
// get through the whole header even when only the unchecked function is
// registered. The real Status is built on folly::Expected and carries a
// heap-allocated message, neither of which exists in device code.
//
// So the shadow is deliberately just enough to parse: a one-byte value with an
// ok flag. GpuUDFHolder still refuses to register a Status-returning call(),
// which is what keeps this from being mistaken for support.
//
// TODO(gpu-sfi-checks): a non-OK Status means "this row is a user error". The
// nearest GPU equivalent would be declining the row, which reports a null
// rather than an error -- a different answer, not a suppressed diagnostic. That
// distinction is why Status-returning functions wait for per-row error
// reporting rather than being mapped onto the bool convention.
//
// The macros below do raise into the error sink, even though nothing is
// registered that can reach them. A macro whose polarity is wrong is a trap
// that springs on whoever first lifts the static_assert, and this one was
// wrong: it returned OK() on the failure condition.
#pragma once

#include "velox/experimental/cudf/functions/GpuErrorSink.cuh"
#include "velox/experimental/cudf/types/GpuProxyTypes.cuh"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox {

class Status {
 public:
  GPU_HOST_DEVICE Status() = default;

  GPU_HOST_DEVICE static Status OK() {
    return Status{};
  }

  /// The real one carries a message built with fmt. Here the message is what
  /// the host recovers by re-evaluating the row, so a failed Status is just
  /// the bit that says so.
  GPU_HOST_DEVICE static Status UserError() {
    return Status{false};
  }

  GPU_HOST_DEVICE bool ok() const {
    return ok_;
  }

 private:
  GPU_HOST_DEVICE explicit Status(bool ok) : ok_(ok) {}

  bool ok_{true};
};

} // namespace facebook::velox

// Returns a user error when the condition holds -- the same polarity as the
// real macro, which is worth stating because this shadow had it backwards and
// returned OK() on exactly the failures it exists to report.
//
// The raise is what lets the host find out: a Status travelling up through
// GpuUDFHolder would only say "this row has no value", which is what a null
// says too. See GpuErrorSink.cuh.
#define VELOX_USER_RETURN(expr, ...)                                      \
  do {                                                                    \
    if (static_cast<bool>(expr)) {                                        \
      ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__);         \
      ::facebook::velox::cudf_velox::gpu_sfi::gpuRaise(                   \
          ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kFailed); \
      return ::facebook::velox::Status::UserError();                      \
    }                                                                     \
  } while (0)

// The comparison forms. Spark's decimal and arithmetic headers use these, and
// the shadow had none of them -- latent only because those headers are not yet
// compiled for the device.
#define VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, op, ...) \
  VELOX_USER_RETURN(!((e1)op(e2))__VA_OPT__(, ) __VA_ARGS__)

#define VELOX_USER_RETURN_EQ(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, == __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_RETURN_NE(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, != __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_RETURN_LT(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, < __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_RETURN_LE(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, <= __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_RETURN_GT(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, > __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_RETURN_GE(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, >= __VA_OPT__(, ) __VA_ARGS__)
