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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "velox/exec/BlockingReason.h"

namespace facebook::velox::core {
class QueryConfig;
}

namespace facebook::velox::exec {

class Operator;

/// Observes the execution of a Task's Drivers.
///
/// One callback pair is delivered per Operator method invocation, which lets an
/// implementation record execution spans, attribute resource usage acquired
/// during a call, or emit events to an external tracing system. Profilers are
/// the intended consumers; Velox itself does not interpret the callbacks.
///
/// A single instance is shared by every Driver of the Task it was created for,
/// so implementations must tolerate concurrent calls from all driver threads.
/// Callbacks run on the driver hot path: they must not block, allocate
/// unboundedly, or throw.
///
/// Every method has an empty default, so an implementation overrides only the
/// events it cares about.
class DriverListener {
 public:
  virtual ~DriverListener() = default;

  /// Invoked immediately before 'op' runs the method named 'callName', on the
  /// thread that runs it. 'callName' is one of the kOpMethod* literals declared
  /// in Driver.h and remains valid for the process lifetime.
  virtual void onOperatorCallBegin(
      const Operator& /* op */,
      std::string_view /* callName */) noexcept {}

  /// Invoked after the matching onOperatorCallBegin, including when the call
  /// threw. Always paired one-to-one with onOperatorCallBegin.
  virtual void onOperatorCallEnd(
      const Operator& /* op */,
      std::string_view /* callName */) noexcept {}

  /// Invoked when the driver leaves its thread because 'op' reported 'reason'.
  virtual void onDriverBlocked(
      const Operator& /* op */,
      BlockingReason /* reason */) noexcept {}

  /// Invoked when a driver that previously reported onDriverBlocked resumes.
  virtual void onDriverUnblocked(const Operator& /* op */) noexcept {}
};

/// Creates DriverListeners for Tasks. Factories are registered process-wide;
/// each Task constructor calls 'create' once per registered factory and holds
/// the returned listeners for the Task's lifetime.
class DriverListenerFactory {
 public:
  virtual ~DriverListenerFactory() = default;

  /// Returns a listener for the identified Task, or nullptr to decline it.
  ///
  /// Declining is the supported way to restrict observation to Tasks of
  /// interest. A declined Task carries no per-call overhead because its Drivers
  /// never see a listener.
  virtual std::shared_ptr<DriverListener> create(
      const std::string& taskId,
      const std::string& taskUuid,
      const core::QueryConfig& config) = 0;
};

/// Registers a factory to be consulted by every Task created afterwards.
/// Returns false if the same factory is already registered.
bool registerDriverListenerFactory(
    const std::shared_ptr<DriverListenerFactory>& factory);

/// Unregisters a factory registered earlier. Returns false if it was not
/// registered. Tasks already holding listeners from it are unaffected.
bool unregisterDriverListenerFactory(
    const std::shared_ptr<DriverListenerFactory>& factory);

namespace detail {

/// Returns one listener for each registered factory that accepted the Task.
/// Called by the Task constructor.
std::vector<std::shared_ptr<DriverListener>> createDriverListeners(
    const std::string& taskId,
    const std::string& taskUuid,
    const core::QueryConfig& config);

} // namespace detail

} // namespace facebook::velox::exec
