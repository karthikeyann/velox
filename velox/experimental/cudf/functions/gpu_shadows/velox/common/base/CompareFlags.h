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

// GPU shadow for velox/common/base/CompareFlags.h.
//
// Mirrors the real header's `CompareFlags` value type (including its
// NullHandlingMode doc comments) so that Velox SFI headers that reference
// `CompareFlags` for comparison configuration compile under nvcc.
// The host-only helpers `nullHandlingModeToStr()`, `reverseDirection()`,
// and `toString()` are dropped: they pull in `<fmt/core.h>` and are not
// callable from a simple function `call()` body.
#pragma once

#include <optional>

namespace facebook::velox {

constexpr auto kIndeterminate = std::nullopt;

// Describes value collation in comparison.
struct CompareFlags {
  bool nullsFirst = true;

  bool ascending = true;

  // When true, comparison should return non-0 early when sizes mismatch.
  bool equalsOnly = false;

  enum class NullHandlingMode {
    /// The default null handling mode where nulls are treated as values such
    /// that:
    ///    - null == null is true,
    ///    - null == value is false.
    ///    - when equalsOnly=false null ordering is determined using the
    ///    nullsFirst flag.
    kNullAsValue,

    /// Presto semantics for handling nulls.
    /// It matches the behavior of ==, >, < functions and many other Presto
    /// functions such as array_remove and array_contains.
    ///
    /// Under this mode, result of comparison can be indeterminate.
    /// Such result is represented as std::nullopt and means that the
    /// function can not decide on the result of the comparison due to some
    /// existing nulls. Not every null results in indeterminate result.
    ///
    /// See velox/common/base/CompareFlags.h for the full semantics
    /// (equalsOnly=true vs equalsOnly=false; primitive, array, row, and
    /// map handling). The doc block has been kept short here because the
    /// behavior itself lives in host-side comparison helpers that are not
    /// part of the GPU SFI call() path.
    kNullAsIndeterminate
  };

  NullHandlingMode nullHandlingMode = NullHandlingMode::kNullAsValue;

  bool nullAsValue() const {
    return nullHandlingMode == CompareFlags::NullHandlingMode::kNullAsValue;
  }

  // Helper method to construct compare flags with equalsOnly = true; in that
  // case nullsFirst and ascending are not needed.
  static constexpr CompareFlags equality(NullHandlingMode nullHandlingMode) {
    return CompareFlags{
        .equalsOnly = true, .nullHandlingMode = nullHandlingMode};
  }
};

} // namespace facebook::velox
