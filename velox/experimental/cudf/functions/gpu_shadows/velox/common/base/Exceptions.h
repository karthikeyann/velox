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
// Real Velox throws VeloxException on a failed check. A kernel cannot throw and
// cannot unwind, so the device form of a check is: evaluate the condition, and
// on failure record that this row was declined and keep going.
//
// What that means for a function body, and why:
//
//   * The condition is evaluated. It used to be discarded, which is what made
//     every check on this path silent. This is the cost of error reporting and
//     it is paid on the happy path, by every row. Measured at +0.0% to +0.4%
//     at the check densities real bodies have; see GPU_SFI_ERROR_DESIGN.md.
//
//   * A failed check does not return. There is no way to leave a nested device
//     frame early, so the body runs on with data a check has just rejected and
//     produces a value from it. That value is garbage and the row is declined
//     -- see GpuErrorSink.cuh. It is the same garbage the body produced before
//     this mechanism existed; the difference is that it is no longer silent.
//
//   * The message is dropped, and the *class* of error is kept. The host does
//     not render a recorded kind as user-visible text: it re-evaluates the
//     declined row through Velox's own CPU evaluator, which produces the real
//     message and the real error code with nothing duplicated here. What it
//     cannot recover afterwards is whether a TRY is allowed to swallow the
//     row, so that much travels with it -- user error versus runtime error,
//     the distinction EvalCtx::setStatus turns on.
//
// EVERY macro the real header defines is defined here, not only the ones
// today's registered functions happen to use. A body that reaches an
// undefined one fails to compile, which is survivable; a body that reaches a
// *wrongly* defined one returns a wrong answer, which is not. The two headers
// are held to the same list by scripts/checks/check-gpu-shadow-macros.py.
//
// The DCHECK families follow NDEBUG exactly as the real header does, rather
// than being unconditionally silent: in a release build they expand to
// nothing, in a debug build they are real checks. Note the real release form
// is `VELOX_CHECK(true)`, which does not evaluate the condition *or* the
// message arguments -- so neither does this, and VELOX_DEBUG_ONLY is what a
// body uses to mark a local that only a debug check reads.
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

// A condition that must hold, in the two error classes.
#define VELOX_GPU_SHADOW_CHECK_KIND(kind, cond, ...)              \
  do {                                                            \
    if (!(cond)) {                                                \
      ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__); \
      ::facebook::velox::cudf_velox::gpu_sfi::gpuRaise(kind);     \
    }                                                             \
  } while (0)

#define VELOX_GPU_SHADOW_CHECK(cond, ...)                                  \
  VELOX_GPU_SHADOW_CHECK_KIND(                                             \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kRuntimeError, \
      cond __VA_OPT__(, ) __VA_ARGS__)

#define VELOX_GPU_SHADOW_USER_CHECK(cond, ...)                          \
  VELOX_GPU_SHADOW_CHECK_KIND(                                          \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kUserError, \
      cond __VA_OPT__(, ) __VA_ARGS__)

// The binary-comparison forms. The operands are used by the condition, so only
// the message arguments go to useArgs.
#define VELOX_GPU_SHADOW_CHECK_OP(a, b, op, ...) \
  VELOX_GPU_SHADOW_CHECK((a)op(b) __VA_OPT__(, ) __VA_ARGS__)

#define VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, op, ...) \
  VELOX_GPU_SHADOW_USER_CHECK((a)op(b) __VA_OPT__(, ) __VA_ARGS__)

// An unconditional failure: VELOX_FAIL and the error-specific forms.
#define VELOX_GPU_SHADOW_FAIL_KIND(kind, ...)                   \
  do {                                                          \
    ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__); \
    ::facebook::velox::cudf_velox::gpu_sfi::gpuRaise(kind);     \
  } while (0)

#define VELOX_GPU_SHADOW_FAIL(...)                                        \
  VELOX_GPU_SHADOW_FAIL_KIND(                                             \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kRuntimeError \
          __VA_OPT__(, ) __VA_ARGS__)

#define VELOX_GPU_SHADOW_USER_FAIL(...)                                \
  VELOX_GPU_SHADOW_FAIL_KIND(                                          \
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kUserError \
          __VA_OPT__(, ) __VA_ARGS__)

// ---------------------------------------------------------------------------
// Runtime errors: VeloxRuntimeError on the real path, which a TRY must not
// swallow.
// ---------------------------------------------------------------------------

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
#ifndef VELOX_CHECK_NULL
#define VELOX_CHECK_NULL(p, ...) \
  VELOX_GPU_SHADOW_CHECK((p) == nullptr __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_CHECK_NOT_NULL
#define VELOX_CHECK_NOT_NULL(p, ...) \
  VELOX_GPU_SHADOW_CHECK((p) != nullptr __VA_OPT__(, ) __VA_ARGS__)
#endif

// The real one renders Status::toString() into the message. The shadow Status
// has no message to render -- that is the host's job after the row is declined
// -- so this is the ok() bit and nothing else.
#ifndef VELOX_CHECK_OK
#define VELOX_CHECK_OK(expr) VELOX_GPU_SHADOW_CHECK((expr).ok(), #expr)
#endif

// "Uncatchable" describes what TRY may do with it, and a runtime error is
// already uncatchable, so it needs no kind of its own.
#ifndef VELOX_CHECK_UNSUPPORTED_INPUT_UNCATCHABLE
#define VELOX_CHECK_UNSUPPORTED_INPUT_UNCATCHABLE(...) \
  VELOX_GPU_SHADOW_CHECK(__VA_ARGS__)
#endif

#ifndef VELOX_FAIL
#define VELOX_FAIL(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
#endif
#ifndef VELOX_FAIL_UNSUPPORTED_INPUT_UNCATCHABLE
#define VELOX_FAIL_UNSUPPORTED_INPUT_UNCATCHABLE(...) \
  VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
#endif
#ifndef VELOX_NYI
#define VELOX_NYI(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
#endif
#ifndef VELOX_UNSUPPORTED
#define VELOX_UNSUPPORTED(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
#endif
#ifndef VELOX_FILE_NOT_FOUND_ERROR
#define VELOX_FILE_NOT_FOUND_ERROR(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
#endif
#ifndef VELOX_TRACE_LIMIT_EXCEEDED
#define VELOX_TRACE_LIMIT_EXCEEDED(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
#endif

// Deliberately not __builtin_unreachable(). A body that reaches this has been
// wrong about something, and telling the optimizer it cannot happen turns that
// into undefined behaviour for the whole launch rather than one declined row.
#ifndef VELOX_UNREACHABLE
#define VELOX_UNREACHABLE(...) VELOX_GPU_SHADOW_FAIL(__VA_ARGS__)
#endif

// ---------------------------------------------------------------------------
// User errors: VeloxUserError on the real path, which a TRY above the
// expression is allowed to turn into a null.
// ---------------------------------------------------------------------------

#ifndef VELOX_USER_CHECK
#define VELOX_USER_CHECK(...) VELOX_GPU_SHADOW_USER_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_EQ
#define VELOX_USER_CHECK_EQ(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, == __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_NE
#define VELOX_USER_CHECK_NE(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, != __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_LT
#define VELOX_USER_CHECK_LT(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, < __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_LE
#define VELOX_USER_CHECK_LE(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, <= __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_GT
#define VELOX_USER_CHECK_GT(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, > __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_GE
#define VELOX_USER_CHECK_GE(a, b, ...) \
  VELOX_GPU_SHADOW_USER_CHECK_OP(a, b, >= __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_NULL
#define VELOX_USER_CHECK_NULL(p, ...) \
  VELOX_GPU_SHADOW_USER_CHECK((p) == nullptr __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_CHECK_NOT_NULL
#define VELOX_USER_CHECK_NOT_NULL(p, ...) \
  VELOX_GPU_SHADOW_USER_CHECK((p) != nullptr __VA_OPT__(, ) __VA_ARGS__)
#endif

#ifndef VELOX_USER_FAIL
#define VELOX_USER_FAIL(...) VELOX_GPU_SHADOW_USER_FAIL(__VA_ARGS__)
#endif

// VeloxUserError on the real path, despite the company it keeps below -- and
// it is the class that matters most here, because decimal overflow is the
// check the registered functions actually reach.
#ifndef VELOX_ARITHMETIC_ERROR
#define VELOX_ARITHMETIC_ERROR(...) VELOX_GPU_SHADOW_USER_FAIL(__VA_ARGS__)
#endif
#ifndef VELOX_SCHEMA_MISMATCH_ERROR
#define VELOX_SCHEMA_MISMATCH_ERROR(...) VELOX_GPU_SHADOW_USER_FAIL(__VA_ARGS__)
#endif

// ---------------------------------------------------------------------------
// Debug-only, following NDEBUG as the real header does. In a release build the
// real forms are VELOX_CHECK(true), which evaluates neither the condition nor
// the message arguments; these expand to nothing for the same reason.
// ---------------------------------------------------------------------------

#ifndef NDEBUG

#ifndef VELOX_DCHECK
#define VELOX_DCHECK(...) VELOX_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_EQ
#define VELOX_DCHECK_EQ(...) VELOX_CHECK_EQ(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_NE
#define VELOX_DCHECK_NE(...) VELOX_CHECK_NE(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_LT
#define VELOX_DCHECK_LT(...) VELOX_CHECK_LT(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_LE
#define VELOX_DCHECK_LE(...) VELOX_CHECK_LE(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_GT
#define VELOX_DCHECK_GT(...) VELOX_CHECK_GT(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_GE
#define VELOX_DCHECK_GE(...) VELOX_CHECK_GE(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_NULL
#define VELOX_DCHECK_NULL(...) VELOX_CHECK_NULL(__VA_ARGS__)
#endif
#ifndef VELOX_DCHECK_NOT_NULL
#define VELOX_DCHECK_NOT_NULL(...) VELOX_CHECK_NOT_NULL(__VA_ARGS__)
#endif

#ifndef VELOX_USER_DCHECK
#define VELOX_USER_DCHECK(...) VELOX_USER_CHECK(__VA_ARGS__)
#endif
#ifndef VELOX_USER_DCHECK_EQ
#define VELOX_USER_DCHECK_EQ(...) VELOX_USER_CHECK_EQ(__VA_ARGS__)
#endif
#ifndef VELOX_USER_DCHECK_NE
#define VELOX_USER_DCHECK_NE(...) VELOX_USER_CHECK_NE(__VA_ARGS__)
#endif
#ifndef VELOX_USER_DCHECK_LT
#define VELOX_USER_DCHECK_LT(...) VELOX_USER_CHECK_LT(__VA_ARGS__)
#endif
#ifndef VELOX_USER_DCHECK_LE
#define VELOX_USER_DCHECK_LE(...) VELOX_USER_CHECK_LE(__VA_ARGS__)
#endif
#ifndef VELOX_USER_DCHECK_GT
#define VELOX_USER_DCHECK_GT(...) VELOX_USER_CHECK_GT(__VA_ARGS__)
#endif
#ifndef VELOX_USER_DCHECK_GE
#define VELOX_USER_DCHECK_GE(...) VELOX_USER_CHECK_GE(__VA_ARGS__)
#endif
#ifndef VELOX_USER_DCHECK_NULL
#define VELOX_USER_DCHECK_NULL(...) VELOX_USER_CHECK_NULL(__VA_ARGS__)
#endif
#ifndef VELOX_USER_DCHECK_NOT_NULL
#define VELOX_USER_DCHECK_NOT_NULL(...) VELOX_USER_CHECK_NOT_NULL(__VA_ARGS__)
#endif

#ifndef VELOX_DEBUG_ONLY
#define VELOX_DEBUG_ONLY
#endif

#else // NDEBUG

// Retained because a body may still want the arguments referenced; the debug
// macros below do not use it, matching the real release forms.
#define VELOX_GPU_SHADOW_NOOP_CHECK(...) \
  ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__)

#ifndef VELOX_DCHECK
#define VELOX_DCHECK(...)
#endif
#ifndef VELOX_DCHECK_EQ
#define VELOX_DCHECK_EQ(...)
#endif
#ifndef VELOX_DCHECK_NE
#define VELOX_DCHECK_NE(...)
#endif
#ifndef VELOX_DCHECK_LT
#define VELOX_DCHECK_LT(...)
#endif
#ifndef VELOX_DCHECK_LE
#define VELOX_DCHECK_LE(...)
#endif
#ifndef VELOX_DCHECK_GT
#define VELOX_DCHECK_GT(...)
#endif
#ifndef VELOX_DCHECK_GE
#define VELOX_DCHECK_GE(...)
#endif
#ifndef VELOX_DCHECK_NULL
#define VELOX_DCHECK_NULL(...)
#endif
#ifndef VELOX_DCHECK_NOT_NULL
#define VELOX_DCHECK_NOT_NULL(...)
#endif

#ifndef VELOX_USER_DCHECK
#define VELOX_USER_DCHECK(...)
#endif
#ifndef VELOX_USER_DCHECK_EQ
#define VELOX_USER_DCHECK_EQ(...)
#endif
#ifndef VELOX_USER_DCHECK_NE
#define VELOX_USER_DCHECK_NE(...)
#endif
#ifndef VELOX_USER_DCHECK_LT
#define VELOX_USER_DCHECK_LT(...)
#endif
#ifndef VELOX_USER_DCHECK_LE
#define VELOX_USER_DCHECK_LE(...)
#endif
#ifndef VELOX_USER_DCHECK_GT
#define VELOX_USER_DCHECK_GT(...)
#endif
#ifndef VELOX_USER_DCHECK_GE
#define VELOX_USER_DCHECK_GE(...)
#endif
#ifndef VELOX_USER_DCHECK_NULL
#define VELOX_USER_DCHECK_NULL(...)
#endif
#ifndef VELOX_USER_DCHECK_NOT_NULL
#define VELOX_USER_DCHECK_NOT_NULL(...)
#endif

#ifndef VELOX_DEBUG_ONLY
#define VELOX_DEBUG_ONLY [[maybe_unused]]
#endif

#endif // NDEBUG
