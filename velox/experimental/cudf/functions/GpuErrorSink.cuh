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

#include <cstdint>

/// Where a failed VELOX_CHECK inside a device-compiled Velox function body
/// records itself.
///
/// A check site sits several frames below the kernel and is handed nothing --
/// not the row, not the output column, not a context:
///
///     simpleFunctionKernel            knows the row
///       evaluateRow
///         GpuUDFHolder::invoke
///           DecimalDivideFunction::call
///             DecimalArithmetic::divideWithRoundUp
///               VELOX_USER_CHECK_NE(b, 0, "Division by zero")   knows nothing
///
/// So a check can only reach state it can find on its own. It has two such
/// things: its thread index, and whatever memory it can name without being
/// handed a pointer. That is the whole basis of what follows.
///
/// **Why the launch's dynamic shared memory rather than a device global.** A
/// `__device__` pointer set per launch is the obvious spelling and it is
/// wrong: without relocatable device code the symbol is per translation unit,
/// and every driver thread in the query launches these kernels out of that
/// same translation unit. Two concurrent launches on two streams would race to
/// set one pointer and one of them would write its rows into the other's
/// buffer. The dynamic shared-memory region has no such problem -- it is
/// per launch and per block by construction -- and it is faster to reach.
namespace facebook::velox::cudf_velox::gpu_sfi {

/// What class of error a declined row hit.
///
/// Deliberately not a catalogue of which macro fired. The host recovers the
/// message, the error code and the TRY behaviour by re-evaluating the row
/// through Velox's own evaluator, so the only thing this has to carry is what
/// *policy* needs, and policy needs exactly one distinction:
///
///     EvalCtx::setStatus records a user error against the row, where a TRY
///     may turn it into a null, and VELOX_FAILs anything else on the spot
///     (EvalCtx.cpp:206-217).
///
/// So a user error is suppressible and a runtime error is not, and that is the
/// whole enum. Splitting it by macro shape instead -- check versus fail versus
/// unsupported -- looks more informative and has no consumer.
enum class GpuErrorKind : uint8_t {
  kNone = 0,
  /// VELOX_USER_CHECK*, VELOX_USER_FAIL, VELOX_ARITHMETIC_ERROR,
  /// VELOX_SCHEMA_MISMATCH_ERROR: VeloxUserError on the real path, which a
  /// TRY above this expression is allowed to swallow.
  kUserError = 1,
  /// VELOX_CHECK*, VELOX_FAIL, VELOX_UNREACHABLE, VELOX_NYI,
  /// VELOX_UNSUPPORTED and the uncatchable pair: VeloxRuntimeError, which a
  /// TRY must not swallow. A row carrying this has to fail the query however
  /// it is wrapped.
  kRuntimeError = 2,
};

/// One byte per thread of the block, carved out of the launch's dynamic shared
/// memory. Every kernel that compiles a Velox body has to request at least
/// `blockDim.x` bytes of it and zero its own byte on entry; the adapter does
/// both, and `raisedError()` is the matching read.
///
/// Declared outside any `__CUDA_ARCH__` guard on purpose: nvcc parses the
/// whole translation unit in the host pass as well, including the bodies of
/// `__device__` functions that name this.
extern __shared__ uint8_t gpuErrorBytes[];

/// Records that the row this thread is evaluating failed a check.
///
/// Three properties the call sites depend on:
///
/// - It does not return or unwind. There is no way to leave a nested device
///   frame early, so the body runs on and computes a value from data a check
///   just rejected. That value is garbage by construction and the row is
///   declined, which is the same garbage the body produced before this
///   mechanism existed -- the difference is that it is no longer silent.
/// - First failure wins, so the kind describes the earliest rejected
///   precondition rather than whatever the damaged arithmetic hit afterwards.
/// - No atomics and no global memory: a byte belongs to exactly one thread.
__host__ __device__ inline void gpuRaise(GpuErrorKind kind) {
#ifdef __CUDA_ARCH__
  if (gpuErrorBytes[threadIdx.x] == static_cast<uint8_t>(GpuErrorKind::kNone)) {
    gpuErrorBytes[threadIdx.x] = static_cast<uint8_t>(kind);
  }
#else
  // The host pass of a translation unit compiled behind the shadow never runs
  // a function body; only registration code is host code there.
  (void)kind;
#endif
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
