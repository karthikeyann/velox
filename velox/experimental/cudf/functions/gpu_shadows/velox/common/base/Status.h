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

// Adapted from Apache Arrow.
//
// GPU shadow for velox/common/base/Status.h.
//
// Mirrors the public surface of the real `Status` value type used by Velox
// SFI call() bodies: `OK()`, `UserError(...)`, and `ok()`. Drops the
// folly::Expected machinery, `StatusCode` enum, error-message storage,
// fmt formatting, and ExceptionHelper -- all host-only.
//
// The error-message arguments to `UserError(...)` are intentionally
// ignored on GPU because exceptions cannot be raised from device code.
// TODO(gpu-sfi-checks): surface error context through the per-row error
// propagation mechanism added in the PR 3+ adapter.
#pragma once

namespace facebook::velox {

/// The Status object is an object holding the outcome of an operation
/// (success or error).
///
/// In the real Velox implementation the outcome is represented as a
/// StatusCode, holding either a success (StatusCode::kOK) or an error
/// (any other of the StatusCode enumeration values). If an error occurred,
/// a specific error message is generally attached.
///
/// This GPU shadow keeps only a boolean OK/non-OK distinction so that
/// `call()` bodies can return `Status::OK()` or `Status::UserError(...)`
/// without dragging in folly::Expected. The error message and code are
/// discarded.
class Status {
 public:
  Status() = default;

  static Status OK() {
    return Status();
  }

  template <typename... Args>
  static Status UserError(Args&&...) {
    Status s;
    s.ok_ = false;
    return s;
  }

  bool ok() const {
    return ok_;
  }

 private:
  bool ok_ = true;
};

} // namespace facebook::velox
