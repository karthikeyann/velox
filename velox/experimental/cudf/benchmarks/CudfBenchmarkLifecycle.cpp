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

#include "velox/experimental/cudf/benchmarks/CudfBenchmarkLifecycle.h"

#include "velox/common/base/Exceptions.h"

#include <glog/logging.h>

#include <exception>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

size_t indexOf(CudfBenchmarkResource resource) {
  const auto index = static_cast<size_t>(resource);
  VELOX_CHECK_LT(
      index,
      kNumCudfBenchmarkResources,
      "Unknown benchmark resource {}.",
      index);
  return index;
}

} // namespace

void CudfBenchmarkLifecycle::own(
    CudfBenchmarkResource resource,
    std::function<void()> release) {
  VELOX_CHECK(
      release != nullptr, "A benchmark resource needs a release callback.");
  release_[indexOf(resource)] = std::move(release);
}

bool CudfBenchmarkLifecycle::owns(CudfBenchmarkResource resource) const {
  return release_[indexOf(resource)] != nullptr;
}

bool CudfBenchmarkLifecycle::ownsAny() const {
  for (const auto& callback : release_) {
    if (callback != nullptr) {
      return true;
    }
  }
  return false;
}

void CudfBenchmarkLifecycle::release() {
  std::exception_ptr firstFailure;
  for (auto& callback : release_) {
    if (callback == nullptr) {
      continue;
    }
    try {
      callback();
    } catch (...) {
      if (firstFailure == nullptr) {
        firstFailure = std::current_exception();
      }
      // Keeping the callback leaves the resource owned, so a caller that
      // removes the cause can release it on a later call.
      continue;
    }
    callback = nullptr;
  }
  if (firstFailure != nullptr) {
    std::rethrow_exception(firstFailure);
  }
}

bool CudfBenchmarkLifecycle::releaseAfterFailure(std::string_view context) {
  try {
    release();
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Cleanup after a failed " << context
               << " failed: " << e.what();
  } catch (...) {
    LOG(ERROR) << "Cleanup after a failed " << context
               << " failed with an unknown exception";
  }
  return false;
}

} // namespace facebook::velox::cudf_velox
