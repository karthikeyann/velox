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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuMemoryCapture.h"
#include "velox/experimental/cudf/exec/GpuMemoryCaptureListener.h"
#include "velox/experimental/cudf/exec/GpuMemoryOwner.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/exec/Operator.h"

#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

/// Calls whose duration is worth retaining as a span.
///
/// The driver loop polls isBlocked, needsInput and isFinished on every
/// iteration, so recording those would exhaust a capture's bounded event budget
/// long before the query finished and would bury the calls that do work. They
/// still establish allocation attribution; only the span is skipped.
bool isSpanWorthy(std::string_view callName) {
  return callName == exec::kOpMethodAddInput ||
      callName == exec::kOpMethodGetOutput ||
      callName == exec::kOpMethodNoMoreInput ||
      callName == exec::kOpMethodInitialize || callName == exec::kOpMethodClose;
}

/// Per-call state, stacked because a listener is shared by every driver thread
/// of its Task and each thread must unwind its own calls.
struct ActiveCall {
  gpu_memory_detail::GpuMemoryActiveOwner previousOwner;
  GpuMemoryCaptureCallHandle handle;
};

std::vector<ActiveCall>& activeCalls() {
  thread_local std::vector<ActiveCall> calls;
  return calls;
}

class GpuMemoryCaptureDriverListener : public exec::DriverListener {
 public:
  void onOperatorCallBegin(const exec::Operator& op, std::string_view callName)
      override {
    // activateGpuMemoryOperator registers the owner, which is what makes an
    // allocation during this call attributable to this operator instance.
    auto& call = activeCalls().emplace_back();
    call.previousOwner = gpu_memory_detail::activateGpuMemoryOperator(
        const_cast<exec::Operator*>(&op));
    if (isSpanWorthy(callName)) {
      call.handle =
          gpu_memory_detail::beginActiveGpuMemoryCaptureCall(callName);
    }
  }

  void onOperatorCallEnd(const exec::Operator& op, std::string_view callName)
      override {
    auto& calls = activeCalls();
    if (calls.empty()) {
      return;
    }
    const auto call = calls.back();
    calls.pop_back();
    gpu_memory_detail::endGpuMemoryCaptureOperatorCall(call.handle);
    if (callName == exec::kOpMethodClose) {
      recordCounts(op);
    }
    gpu_memory_detail::restoreGpuMemoryOwner(call.previousOwner);
  }

  void onDriverBlocked(const exec::Operator& op, exec::BlockingReason reason)
      override {
    gpu_memory_detail::beginGpuMemoryCaptureBlockedSpanFor(
        const_cast<exec::Operator*>(&op),
        exec::BlockingReasonName::toName(reason));
  }

  void onDriverUnblocked(const exec::Operator& op) override {
    gpu_memory_detail::endGpuMemoryCaptureBlockedSpanFor(
        const_cast<exec::Operator*>(&op));
  }

 private:
  /// Reads what Velox already counted for this operator, once, as it closes.
  ///
  /// Taken at close because the counts are cumulative: sampling per call would
  /// lock the stats on the hot path to learn nothing a final reading does not
  /// already say. The owner is still active here, so the capture resolves the
  /// canonical metadata the same way it does for a call span.
  static void recordCounts(const exec::Operator& op) {
    GpuMemoryOperatorCounts counts;
    {
      auto locked = const_cast<exec::Operator&>(op).stats().rlock();
      counts.inputRows = locked->inputPositions;
      counts.inputBytes = locked->inputBytes;
      counts.inputBatches = locked->inputVectors;
      counts.outputRows = locked->outputPositions;
      counts.outputBytes = locked->outputBytes;
      counts.outputBatches = locked->outputVectors;
      counts.rawInputRows = locked->rawInputPositions;
      counts.rawInputBytes = locked->rawInputBytes;
      counts.blockedWallNanos = locked->blockedWallNanos;
      counts.cpuNanos = locked->addInputTiming.cpuNanos +
          locked->getOutputTiming.cpuNanos + locked->finishTiming.cpuNanos;
      counts.wallNanos = locked->addInputTiming.wallNanos +
          locked->getOutputTiming.wallNanos + locked->finishTiming.wallNanos;
    }
    gpu_memory_detail::recordActiveGpuMemoryCaptureOperatorCounts(counts);
  }
};

} // namespace

std::shared_ptr<exec::DriverListener>
GpuMemoryCaptureDriverListenerFactory::create(
    const std::string& /* taskId */,
    const std::string& /* taskUuid */,
    const core::QueryConfig& /* config */) {
  if (!CudfConfig::getInstance().gpuMemoryTrackingEnabled()) {
    return nullptr;
  }
  return std::make_shared<GpuMemoryCaptureDriverListener>();
}

} // namespace facebook::velox::cudf_velox
