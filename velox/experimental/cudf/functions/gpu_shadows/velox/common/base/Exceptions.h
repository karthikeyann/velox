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

// GPU shadow for velox/common/base/Exceptions.h
//
// Real Velox throws VeloxException on a failed check. A kernel cannot throw
// and cannot unwind, so the device form of a check is: evaluate the condition,
// and on failure record that this row was declined and keep going.
//
// What that means for a function body, and why:
//
//   * The condition is evaluated. It used to be discarded, which is what made
//     every check on this path silent. This is the cost of error reporting and
//     it is paid on the happy path, by every row.
//
//   * A failed check does not return. There is no way to leave a nested device
//     frame early, so the body runs on with data a check has just rejected and
//     produces a value from it. That value is garbage and the host discards
//     the row -- see GpuErrorSink.cuh. It is the same garbage the body
//     produced before this mechanism existed; the difference is that it is no
//     longer silent.
//
//   * The message is dropped. The host does not render the recorded kind as
//     user-visible text: it re-evaluates the declined row through Velox's own
//     CPU evaluator, which produces the real message, the real error code, and
//     the real TRY behaviour with nothing duplicated here. Formatting on the
//     device would mean a second copy of every format string, kept in step by
//     hand.
//
//   * VELOX_DCHECK* stay no-ops, matching a release CPU build.
//
// See GPU_SFI_ERROR_DESIGN.md for the design and the options rejected.
#pragma once

#include "velox/experimental/cudf/functions/GpuErrorSink.cuh"

namespace facebook::velox::gpu_shadow_detail {

// Variadic no-op consumer for the message arguments of a check. They are not
// formatted on the device, but they have to be *used*, or a local computed
// only to feed a check draws an unused-variable warning, and a parameter used
// only by a check in a FOLLY_ALWAYS_INLINE helper draws nvcc warning #550-D.
//
// Pass-by-const-ref, and called only on the failure path: the condition itself
// already uses the operands, so nothing needs to be evaluated twice.
template <typename... Ts>
__host__ __device__ constexpr void useArgs(const Ts&...) {}

} // namespace facebook::velox::gpu_shadow_detail

// A check whose condition failed.
#define VELOX_GPU_SHADOW_CHECK(cond, ...)                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__);              \
      ::facebook::velox::cudf_velox::gpu_sfi::gpuRaise(                        \
          ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kCheckFailed); \
    }                                                                          \
  } while (0)

// The binary-comparison forms. The operands are used by the condition, so only
// the message arguments go to useArgs.
#define VELOX_GPU_SHADOW_CHECK_OP(a, b, op, ...) \
  VELOX_GPU_SHADOW_CHECK((a)op(b) __VA_OPT__(, ) __VA_ARGS__)

// An unconditional failure: VELOX_FAIL and friends.
#define VELOX_GPU_SHADOW_FAIL(kind, ...)                        \
  do {                                                          \
    ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__); \
    ::facebook::velox::cudf_velox::gpu_sfi::gpuRaise(kind);     \
  } while (0)

// Retained for the debug-only macros below, which stay silent.
#define VELOX_GPU_SHADOW_NOOP_CHECK(...) \
  ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__)

#ifndef VELOX_CHECK
#define VELOX_CHECK(...) VELOX_GPU_SHADOW_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_CHECK_EQ
#define VELOX_CHECK_EQ(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, == __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_CHECK_NE
#define VELOX_CHECK_NE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, != __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_CHECK_LT
#define VELOX_CHECK_LT(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, < __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_CHECK_LE
#define VELOX_CHECK_LE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, <= __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_CHECK_GT
#define VELOX_CHECK_GT(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, > __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_CHECK_GE
#define VELOX_CHECK_GE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, >= __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_CHECK_NOT_NULL
#define VELOX_CHECK_NOT_NULL(p, ...) \
  VELOX_GPU_SHADOW_CHECK((p) != nullptr __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_CHECK_NULL
#define VELOX_CHECK_NULL(p, ...) \
  VELOX_GPU_SHADOW_CHECK((p) == nullptr __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_FAIL
#define VELOX_FAIL(...)                                             \
  VELOX_GPU_SHADOW_FAIL(                                            \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kFailed \
          __VA_OPT__(, ) __VA_ARGS__)
#endif

// Deliberately not __builtin_unreachable(). A body that reaches this has been
// wrong about something, and telling the optimizer it cannot happen turns that
// into undefined behaviour for the whole launch rather than one declined row.
#ifndef VELOX_UNREACHABLE
#define VELOX_UNREACHABLE(...)                                      \
  VELOX_GPU_SHADOW_FAIL(                                            \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kFailed \
          __VA_OPT__(, ) __VA_ARGS__)
#endif

#ifndef VELOX_USER_CHECK
#define VELOX_USER_CHECK(...) VELOX_GPU_SHADOW_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_EQ
#define VELOX_USER_CHECK_EQ(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, == __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_NE
#define VELOX_USER_CHECK_NE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, != __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_LT
#define VELOX_USER_CHECK_LT(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, < __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_LE
#define VELOX_USER_CHECK_LE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, <= __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_GT
#define VELOX_USER_CHECK_GT(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, > __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_GE
#define VELOX_USER_CHECK_GE(a, b, ...) \
  VELOX_GPU_SHADOW_CHECK_OP(a, b, >= __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_NOT_NULL
#define VELOX_USER_CHECK_NOT_NULL(p, ...) \
  VELOX_GPU_SHADOW_CHECK((p) != nullptr __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_FAIL
#define VELOX_USER_FAIL(...)                                        \
  VELOX_GPU_SHADOW_FAIL(                                            \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kFailed \
          __VA_OPT__(, ) __VA_ARGS__)
#endif

// Debug-only on the CPU, and compiled out of a release build there. Left
// silent here for the same reason.
#ifndef VELOX_DCHECK
#define VELOX_DCHECK(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_EQ
#define VELOX_DCHECK_EQ(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_NE
#define VELOX_DCHECK_NE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_LT
#define VELOX_DCHECK_LT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_LE
#define VELOX_DCHECK_LE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_GT
#define VELOX_DCHECK_GT(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_GE
#define VELOX_DCHECK_GE(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_NOT_NULL
#define VELOX_DCHECK_NOT_NULL(...) VELOX_GPU_SHADOW_NOOP_CHECK(__VA_ARGS__)
#endif

#ifndef VELOX_NYI
#define VELOX_NYI(...)                                                   \
  VELOX_GPU_SHADOW_FAIL(                                                 \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kUnsupported \
          __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_UNSUPPORTED
#define VELOX_UNSUPPORTED(...)                                           \
  VELOX_GPU_SHADOW_FAIL(                                                 \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kUnsupported \
          __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_ARITHMETIC_ERROR
#define VELOX_ARITHMETIC_ERROR(...)                                 \
  VELOX_GPU_SHADOW_FAIL(                                            \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kFailed \
          __VA_OPT__(, ) __VA_ARGS__)
#endif
