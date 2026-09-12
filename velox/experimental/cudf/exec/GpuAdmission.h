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

namespace facebook::velox::cudf_velox {

/// Serialises the few operations large enough that two drivers doing them at
/// once is what fills the device.
///
/// Drivers do not coordinate with each other and do not need to. They consult
/// one counter before the one allocation that dominates their working set, and
/// it admits one at a time while the device is near full. Everything else runs
/// exactly as before, at full concurrency.
///
/// Why this is safe, and why it is confined to this shape: the caller holds
/// admission only while doing bounded work on data it already has - a merge, a
/// gather, a decode. It never waits on another driver while holding it, so it
/// always finishes and always gives it back. There is no way for two holders to
/// wait on each other. Parking a driver across an input boundary would not have
/// this property: a consumer that stops draining its exchange queue can starve
/// the very driver whose progress would release it, because a local exchange
/// shares one buffer budget across all of its partitions.
///
/// The device figure comes from cudfAllocatedBytes(), which counts what callers
/// hold rather than what a pool retains. Without that distinction a cached pool
/// reads as permanently full and nothing would ever be admitted. Where that
/// figure is unavailable the rule degrades rather than switching off: an
/// operation large enough to exceed the threshold on its own is still admitted
/// one at a time, which needs no accounting to decide.
class GpuAdmission {
 public:
  /// Admission for one operation, released when it goes out of scope. Holding
  /// none - because the device had room, or because accounting is unavailable -
  /// is the common case and costs nothing.
  class Guard {
   public:
    explicit Guard(bool held) : held_{held} {}
    ~Guard();
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    Guard(Guard&& other) noexcept : held_{other.held_} {
      other.held_ = false;
    }
    Guard& operator=(Guard&&) = delete;

    bool held() const {
      return held_;
    }

   private:
    bool held_;
  };

  /// Waits, if necessary, until this operation can proceed, and returns the
  /// admission to hold while it runs.
  ///
  /// `projectedBytes` is what the operation is about to allocate, as well as
  /// the caller can tell. It is advisory: an underestimate makes admission
  /// slightly too generous, never incorrect.
  ///
  /// Returns immediately without admission when the device is comfortable, so
  /// a query that was never in trouble never serialises anything.
  static Guard acquire(uint64_t projectedBytes);

  /// Device bytes above which operations start being admitted one at a time.
  /// Zero disables admission entirely, which is the default.
  static void setThresholdBytes(uint64_t bytes);
  static uint64_t thresholdBytes();

  /// Number of times an operation had to wait. For tests and for judging
  /// whether the mechanism is doing anything.
  static uint64_t waitCount();
};

} // namespace facebook::velox::cudf_velox
