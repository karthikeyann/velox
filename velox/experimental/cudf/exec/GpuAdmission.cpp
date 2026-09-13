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

#include "velox/experimental/cudf/exec/GpuAdmission.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace facebook::velox::cudf_velox {
namespace {

struct AdmissionState {
  std::mutex mutex;
  std::condition_variable available;
  bool occupied{false};
  std::atomic<uint64_t> threshold{0};
  std::atomic<uint64_t> waits{0};
};

AdmissionState& state() {
  static AdmissionState instance;
  return instance;
}

} // namespace

void GpuAdmission::setThresholdBytes(uint64_t bytes) {
  state().threshold.store(bytes, std::memory_order_relaxed);
}

uint64_t GpuAdmission::thresholdBytes() {
  return state().threshold.load(std::memory_order_relaxed);
}

uint64_t GpuAdmission::waitCount() {
  return state().waits.load(std::memory_order_relaxed);
}

GpuAdmission::Guard GpuAdmission::acquire(uint64_t projectedBytes) {
  auto& s = state();
  const auto threshold = s.threshold.load(std::memory_order_relaxed);
  if (threshold == 0) {
    return Guard{false};
  }

  // What callers hold right now, not what the pool has cached - a pool that
  // caches reads as permanently full, and nothing would ever be admitted.
  //
  // A negative reading means accounting is unavailable. Treating that as zero
  // rather than giving up keeps the useful half of the rule: an operation
  // large enough to exceed the threshold on its own is worth serialising
  // whatever else is running, and that judgement needs no accounting at all.
  const auto allocated = cudfAllocatedBytes();
  const uint64_t inUse =
      allocated < 0 ? uint64_t{0} : static_cast<uint64_t>(allocated);

  // Admit freely while the device has room for this on top of what is already
  // out. Only when it does not is there anything to serialise, and a query
  // that never reaches the threshold never touches the lock.
  if (inUse + projectedBytes <= threshold) {
    return Guard{false};
  }

  std::unique_lock<std::mutex> lock(s.mutex);
  if (s.occupied) {
    s.waits.fetch_add(1, std::memory_order_relaxed);
    s.available.wait(lock, [&s] { return !s.occupied; });
  }
  s.occupied = true;
  return Guard{true};
}

GpuAdmission::Guard::~Guard() {
  if (!held_) {
    return;
  }
  auto& s = state();
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    s.occupied = false;
  }
  s.available.notify_one();
}

} // namespace facebook::velox::cudf_velox
