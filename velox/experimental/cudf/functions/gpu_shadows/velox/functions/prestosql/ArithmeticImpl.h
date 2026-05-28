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

// GPU shadow for velox/functions/prestosql/ArithmeticImpl.h.
//
// Mirrors the real header, including its doc comments, but adds
// `__host__ __device__` annotations to free-function helpers that the
// upstream version leaves unannotated. Functions already marked
// `FOLLY_ALWAYS_INLINE` get the annotation automatically through the
// `folly/CPortability.h` shadow.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include "folly/CPortability.h"
#include "velox/common/base/Exceptions.h"
#include "velox/experimental/cudf/types/GpuProxyTypes.cuh"
#include "velox/type/FloatingPointUtil.h"

namespace facebook::velox::functions {

/// Round function
/// When AlwaysRoundNegDec is false, presto semantics is followed which does not
/// round negative decimals for integrals and round it otherwise
/// Note that is is likely techinically impossible for this function to return
/// expected results in all cases as the loss of precision plagues it on both
/// paths: factor multiplication for large numbers and addition of truncated
/// number to the rounded fraction for small numbers.
/// We are trying to minimize the loss of precision by using the best path for
/// the number, but the journey is likely not over yet.
template <typename TNum, typename TDecimals, bool alwaysRoundNegDec = false>
FOLLY_ALWAYS_INLINE TNum
round(const TNum& number, const TDecimals& decimals = 0) {
  static_assert(!std::is_same_v<TNum, bool> && "round not supported for bool");

  if constexpr (std::is_integral_v<TNum>) {
    if constexpr (alwaysRoundNegDec) {
      if (decimals >= 0)
        return number;
    } else {
      return number;
    }
  }
  if (!std::isfinite(number)) {
    return number;
  }

  // If we just need to get rid of all the decimals.
  if (decimals == 0) {
    return std::round(number);
  }

  // For negative 'decimals', we aren't going to lose any precision - we divide
  // first (multiply by factor which is < 1.0).
  if (decimals < 0) {
    const double factor = std::pow(10, decimals);
    return std::round(number * factor) / factor;
  }

  // Get the fraction part and return number 'as is' if fraction part is 0.
  const TNum trancated = std::trunc(number);
  const TNum fraction = number - trancated;
  if (fraction == 0.0)
    return number;

  const double factor = std::pow(10, decimals);

  // Smaller numbers are less affected by precision loss being multiplied by the
  // factor, but more affected by precision loss by adding truncated number to
  // the rounded fraction in the end. Because of that, we use factor
  // multiplication path on the smaller numbers.
  // The threshold is a somewhat arbitrary/empirical number taking up 44 bits in
  // the integer form.
  if constexpr (!std::is_integral_v<TNum>) {
    if (fabs(number) < 17592186044415.F) {
      return std::round(number * factor) / factor;
    }
  }

  // We implement the algorithm below for positive 'decimals' nd the large
  // numbers because on the large numbers we would have precision loss when
  // multiplying number by the factor, which would lose or gain some [power of
  // 2] whole units.

  const TNum roundedFractions = std::round(fraction * factor) / factor;
  return trancated + roundedFractions;
}

// This is used by Velox for floating points plus.
//
// GPU shadow difference vs upstream: marked `__host__ __device__` so the
// helper is callable from CUDA kernels. The upstream version's
// `__attribute__((__no_sanitize__("signed-integer-overflow")))` is dropped
// because ASAN is host-only and nvcc rejects unknown function attributes.
template <typename T>
GPU_HOST_DEVICE T plus(const T a, const T b) {
  return a + b;
}

// This is used by Velox for floating points minus.
//
// GPU shadow difference vs upstream: see note on `plus()` above.
template <typename T>
GPU_HOST_DEVICE T minus(const T a, const T b) {
  return a - b;
}

// This is used by Velox for floating points multiply.
//
// GPU shadow difference vs upstream: see note on `plus()` above.
template <typename T>
GPU_HOST_DEVICE T multiply(const T a, const T b) {
  return a * b;
}

// This is used by Velox for floating points divide.
//
// GPU shadow difference vs upstream: see note on `plus()` above.
// Upstream additionally drops `__no_sanitize__("float-divide-by-zero")`.
template <typename T>
GPU_HOST_DEVICE T divide(const T& a, const T& b) {
  T result = a / b;
  return result;
}

// This is used by Velox for modulus, both floating point and integer.
//
// NOTE on integer T: real Velox's `ModulusFunction<int>::call` also
// routes through this helper, even though for integers
// `std::numeric_limits<int>::quiet_NaN()` is 0 (no NaN representation)
// and `std::fmod(int, int)` loses precision via an int->double->int
// round-trip. The behavior is preserved here to match upstream exactly;
// fixing it is an upstream Velox concern, not a shadow concern.
// See ArithmeticImpl audit comments for follow-up.
template <typename T>
GPU_HOST_DEVICE T modulus(const T a, const T b) {
  if (b == 0) {
    // Match Presto semantics.
    return std::numeric_limits<T>::quiet_NaN();
  }
  return std::fmod(a, b);
}

template <typename T>
GPU_HOST_DEVICE T negate(const T& arg) {
  return -arg;
}

template <typename T>
GPU_HOST_DEVICE T abs(const T& arg) {
  // GPU shadow difference vs upstream: upstream uses `std::negate<>` and
  // VELOX_USER_FAIL on integer overflow at INT_MIN. Our shadow makes
  // VELOX_USER_FAIL a no-op, so a naive `-arg` for arg == INT_MIN would
  // be undefined behavior (signed integer overflow). We use unsigned
  // wrap-around instead, which gives a defined (though mathematically
  // incorrect at INT_MIN) value. The TODO(gpu-sfi-checks) marker is on
  // the surrounding macro -- once the per-row error propagation lands in
  // PR 3+, the check fires properly and this fallback no longer matters.
  if constexpr (std::is_integral_v<T>) {
    if (arg < 0) {
      return static_cast<T>(-static_cast<std::make_unsigned_t<T>>(arg));
    }
    return arg;
  } else {
    return std::abs(arg);
  }
}

// floor/ceil for integer T: upstream Velox unconditionally calls
// std::floor / std::ceil. Under nvcc those overloads aren't always
// available for every integral type and the round-trip
// integer -> double -> integer can lose precision for values beyond
// the double mantissa. We short-circuit for integer T (floor/ceil of
// an integer is the integer itself) to produce the same observable
// result via a faster, host- and device-portable code path.
template <typename T>
GPU_HOST_DEVICE T floor(const T& arg) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::floor(arg);
  } else {
    return arg;
  }
}

template <typename T>
GPU_HOST_DEVICE T ceil(const T& arg) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::ceil(arg);
  } else {
    return arg;
  }
}

FOLLY_ALWAYS_INLINE double truncate(double number, int32_t decimals) {
  const bool decNegative = (decimals < 0);
  const auto log10Size = DoubleUtil::kPowersOfTen.size(); // 309
  if (decNegative && decimals <= -static_cast<int32_t>(log10Size)) {
    return 0.0;
  }

  const uint64_t absDec = std::abs(decimals);
  const double tmp = (absDec < log10Size) ? DoubleUtil::kPowersOfTen[absDec]
                                          : std::pow(10.0, (double)absDec);

  const double valueMulTmp = number * tmp;
  if (!decNegative && !std::isfinite(valueMulTmp)) {
    return number;
  }

  const double valueDivTmp = number / tmp;
  if (number >= 0.0) {
    return decimals < 0 ? std::floor(valueDivTmp) * tmp
                        : std::floor(valueMulTmp) / tmp;
  } else {
    return decimals < 0 ? std::ceil(valueDivTmp) * tmp
                        : std::ceil(valueMulTmp) / tmp;
  }
}

// helper function for calculating upper and lower limit of wilson interval
template <bool isUpper>
FOLLY_ALWAYS_INLINE double
wilsonInterval(int64_t successes, int64_t trials, double z) {
  VELOX_USER_CHECK_GE(successes, 0, "number of successes must not be negative");
  VELOX_USER_CHECK_GT(trials, 0, "number of trials must be positive");
  VELOX_USER_CHECK_LE(
      successes,
      trials,
      "number of successes must not be larger than number of trials");
  VELOX_USER_CHECK_GE(z, 0, "z-score must not be negative");

  double s{static_cast<double>(successes)};
  double n{static_cast<double>(trials)};
  double p{s / n};

  // Wilson interval limits are solutions of a quadratic equation.
  // Let the equation be {ax^2 + bx + c = 0}.
  // r will store the value (-b + sqrt(b*b - 4*a*c)).
  double a, c, r;

  // Compute the equations differently depending on whether z is large or small.
  // This helps to avoid computations like (INFINITY/INFINITY),
  // yielding accurate results in the limit as z approaches infinity.
  if (z < 1) {
    a = n + z * z;
    c = s * p;
    r = 2 * s + z * z + z * std::sqrt(z * z + 4 * s * (1 - p));
  } else {
    a = n / (z * z) + 1;
    c = s * p / (z * z);
    r = 2 * s / (z * z) + 1 + std::sqrt(1 + 4 * s * (1 - p) / (z * z));
  }

  // Since (s, n, z >= 0), r >= 0 is guaranteed, but r == 0 needs to be handled.
  if constexpr (isUpper) {
    return r / (2 * a);
  } else {
    return (r > 0) ? (2 * c) / r : 0;
  }
}

} // namespace facebook::velox::functions
