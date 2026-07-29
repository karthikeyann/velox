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

#include "velox/exec/DriverListener.h"

#include <folly/Synchronized.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec {
namespace {

folly::Synchronized<std::vector<std::shared_ptr<DriverListenerFactory>>>&
driverListenerFactories() {
  static folly::Synchronized<
      std::vector<std::shared_ptr<DriverListenerFactory>>>
      kFactories;
  return kFactories;
}

} // namespace

bool registerDriverListenerFactory(
    const std::shared_ptr<DriverListenerFactory>& factory) {
  VELOX_CHECK_NOT_NULL(factory);
  return driverListenerFactories().withWLock([&](auto& factories) {
    for (const auto& existingFactory : factories) {
      if (existingFactory == factory) {
        return false;
      }
    }
    factories.push_back(factory);
    return true;
  });
}

bool unregisterDriverListenerFactory(
    const std::shared_ptr<DriverListenerFactory>& factory) {
  return driverListenerFactories().withWLock([&](auto& factories) {
    for (auto it = factories.cbegin(); it != factories.cend(); ++it) {
      if ((*it) == factory) {
        factories.erase(it);
        return true;
      }
    }
    return false;
  });
}

namespace detail {

std::vector<std::shared_ptr<DriverListener>> createDriverListeners(
    const std::string& taskId,
    const std::string& taskUuid,
    const core::QueryConfig& config) {
  std::vector<std::shared_ptr<DriverListener>> listeners;
  driverListenerFactories().withRLock([&](const auto& factories) {
    for (const auto& factory : factories) {
      auto listener = factory->create(taskId, taskUuid, config);
      if (listener != nullptr) {
        listeners.push_back(std::move(listener));
      }
    }
  });
  return listeners;
}

} // namespace detail
} // namespace facebook::velox::exec
