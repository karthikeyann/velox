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

#include "velox/common/time/CpuWallTimer.h"

#include <algorithm>

namespace facebook::velox {

CpuWallTimer::CpuWallTimer(CpuWallTiming& timing)
    : wallTimeStart_(std::chrono::steady_clock::now()),
      cpuTimeStart_(process::threadCpuNanos()),
      timing_(timing) {
  ++timing_.count;
}

CpuWallTimer::~CpuWallTimer() {
  // NOTE: End the cpu-time timing first, and then end the wall-time timing,
  // so as to avoid the counter-intuitive phenomenon that the final calculated
  // cpu-time is slightly larger than the wall-time.
  timing_.cpuNanos += process::threadCpuNanos() - cpuTimeStart_;
  const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - wallTimeStart_);
  timing_.wallNanos += duration.count();
}

void CpuWallLapTimer::reset() {
  cpuTime_ = process::threadCpuNanos();
  wallTime_ = std::chrono::steady_clock::now();
}

CpuWallTiming CpuWallLapTimer::lap() {
  const auto cpuTime = process::threadCpuNanos();
  const auto wallTime = std::chrono::steady_clock::now();
  const uint64_t wallNanos =
      std::chrono::duration_cast<std::chrono::nanoseconds>(wallTime - wallTime_)
          .count();
  // One reading of each clock bounds two intervals, so the CPU interval can
  // extend past the wall interval by the time between the two readings.
  const CpuWallTiming timing{
      1, wallNanos, std::min<uint64_t>(cpuTime - cpuTime_, wallNanos)};
  cpuTime_ = cpuTime;
  wallTime_ = wallTime;
  return timing;
}

} // namespace facebook::velox
