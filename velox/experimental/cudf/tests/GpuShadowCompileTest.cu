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

// COMPILE-ONLY TEST.
//
// Each `verify*Function()` below exists solely to force a template
// instantiation of the corresponding Velox SFI struct's `call()` method
// with `TExec = GpuExec`. The runtime CALLS (e.g. `fn.call(r, 1.0, 2.0)`)
// and their return values are discarded -- this file's purpose is to
// catch compile-time errors when the shadow include path is wired up,
// not to verify functional correctness.
//
// RUNTIME correctness for the proxy types (`GpuStringView`,
// `GpuTimestamp`) and the shadow `bits::countBits` lives in
// GpuTypesTest.cpp. Per-function runtime correctness will be added in
// PR 4+ once GpuSimpleFunctionAdapter and friends exist to launch the
// SFI bodies from device kernels.
//
// The functions are tagged `[[maybe_unused]]` because nothing calls them
// from elsewhere; they're picked up by the compiler when this TU is
// compiled into the OBJECT library target `velox_cudf_gpu_shadow_compile_test`.

#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/common/base/BitUtil.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Bitwise.h"
#include "velox/functions/prestosql/Comparisons.h"
// sparksql/Arithmetic.h deferred -- requires ToHexUtil host-only dependency
// sparksql/Comparisons.h deferred -- mixes VectorFunction factories

using namespace facebook::velox::gpu;

namespace {

// --- Presto Arithmetic ---
[[maybe_unused]] void verifyPlusFunction() {
  facebook::velox::functions::PlusFunction<GpuExec> fn;
  double r = 0;
  fn.call(r, 1.0, 2.0);
  (void)r;
}

[[maybe_unused]] void verifyMinusFunction() {
  facebook::velox::functions::MinusFunction<GpuExec> fn;
  int64_t r = 0;
  fn.call(r, int64_t{5}, int64_t{3});
  (void)r;
}

[[maybe_unused]] void verifyMultiplyFunction() {
  facebook::velox::functions::MultiplyFunction<GpuExec> fn;
  float r = 0;
  fn.call(r, 2.0f, 3.0f);
  (void)r;
}

[[maybe_unused]] void verifyDivideFunction() {
  facebook::velox::functions::DivideFunction<GpuExec> fn;
  double r = 0;
  fn.call(r, 6.0, 2.0);
  (void)r;
}

[[maybe_unused]] void verifyCeilFunction() {
  facebook::velox::functions::CeilFunction<GpuExec> fn;
  double r = 0;
  fn.call(r, 1.5);
  (void)r;
}

[[maybe_unused]] void verifyFloorFunction() {
  facebook::velox::functions::FloorFunction<GpuExec> fn;
  int64_t r = 0;
  fn.call(r, int64_t{5});
  (void)r;
}

[[maybe_unused]] void verifyAbsFunction() {
  facebook::velox::functions::AbsFunction<GpuExec> fn;
  double r = 0;
  fn.call(r, -5.0);
  (void)r;
}

[[maybe_unused]] void verifyNegateFunction() {
  facebook::velox::functions::NegateFunction<GpuExec> fn;
  int32_t r = 0;
  fn.call(r, int32_t{5});
  (void)r;
}

[[maybe_unused]] void verifyModulusFunction() {
  facebook::velox::functions::ModulusFunction<GpuExec> fn;
  int64_t r = 0;
  fn.call(r, int64_t{10}, int64_t{3});
  (void)r;
}

// --- Presto Comparisons ---
[[maybe_unused]] void verifyLtFunction() {
  facebook::velox::functions::LtFunction<GpuExec> fn;
  bool r = false;
  fn.call(r, 1.0, 2.0);
  (void)r;
}

[[maybe_unused]] void verifyGtFunction() {
  facebook::velox::functions::GtFunction<GpuExec> fn;
  bool r = false;
  fn.call(r, 3.0, 1.0);
  (void)r;
}

[[maybe_unused]] void verifyEqFunction() {
  facebook::velox::functions::EqFunction<GpuExec> fn;
  bool r = false;
  fn.call(r, 42, 42);
  (void)r;
}

[[maybe_unused]] void verifyNeqFunction() {
  facebook::velox::functions::NeqFunction<GpuExec> fn;
  bool r = false;
  fn.call(r, 1, 2);
  (void)r;
}

// --- Presto Bitwise (uses BitUtil.h shadow via BitCountFunction) ---
[[maybe_unused]] void verifyBitwiseAndFunction() {
  facebook::velox::functions::BitwiseAndFunction<GpuExec> fn;
  int64_t r = 0;
  fn.call(r, int64_t{0xF}, int64_t{0x5});
  (void)r;
}

[[maybe_unused]] void verifyBitwiseOrFunction() {
  facebook::velox::functions::BitwiseOrFunction<GpuExec> fn;
  int64_t r = 0;
  fn.call(r, int64_t{0xF}, int64_t{0x5});
  (void)r;
}

[[maybe_unused]] void verifyBitCountFunction() {
  facebook::velox::functions::BitCountFunction<GpuExec> fn;
  int64_t r = 0;
  // Exercises the bits::countBits shadow implementation.
  fn.call(r, int64_t{0xF}, int32_t{8});
  (void)r;
}

// Exercises the kPowersOfTen inline definition in the FloatingPointUtil.h
// shadow via the `truncate()` helper that `TruncateFunction::call` invokes.
[[maybe_unused]] void verifyTruncateFunctionWithDecimals() {
  facebook::velox::functions::TruncateFunction<GpuExec> fn;
  double r = 0.0;
  fn.call(r, 3.14159, int32_t{2});
  (void)r;
}

// --- Direct exercise of the bits::countBits shadow ---
// Compile-time `static_assert` to validate countBits is `constexpr`-evaluable
// is not possible (`__builtin_popcountll` is not constexpr in all compilers),
// so we leave a __device__-callable wrapper as an additional check that the
// shadow is callable from kernel code.
[[maybe_unused]] __device__ int32_t
deviceCountBitsWrapper(const uint64_t* bits, int32_t b, int32_t e) {
  return facebook::velox::bits::countBits(bits, b, e);
}

} // namespace
