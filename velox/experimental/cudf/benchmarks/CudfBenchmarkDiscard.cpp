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

#include "velox/experimental/cudf/benchmarks/CudfBenchmarkDiscard.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/future/VeloxPromise.h"
#include "velox/exec/Operator.h"

#include <mutex>

namespace facebook::velox::cudf_velox {

// ---- BenchmarkDiscardNode ---------------------------------------------------

BenchmarkDiscardNode::BenchmarkDiscardNode(
    const core::PlanNodeId& id,
    core::PlanNodePtr source)
    : PlanNode(id), sources_({std::move(source)}) {}

const RowTypePtr& BenchmarkDiscardNode::outputType() const {
  return sources_[0]->outputType();
}

const std::vector<core::PlanNodePtr>& BenchmarkDiscardNode::sources() const {
  return sources_;
}

std::string_view BenchmarkDiscardNode::name() const {
  return "BenchmarkDiscard";
}

void BenchmarkDiscardNode::addDetails(std::stringstream& /*stream*/) const {}

// ---- BenchmarkDiscard operator ----------------------------------------------

// Not exported from the header; only the adapter and translator need this type.
class BenchmarkDiscard : public exec::Operator {
 public:
  BenchmarkDiscard(
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const BenchmarkDiscardNode>& node,
      int32_t operatorId)
      : Operator(
            driverCtx,
            node->outputType(),
            operatorId,
            node->id(),
            "BenchmarkDiscard") {}

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override {
    // Dynamically verify the input is a GPU vector.
    VELOX_USER_CHECK(
        dynamic_cast<const CudfVector*>(input.get()) != nullptr,
        "BenchmarkDiscard expects a CudfVector input, got {}.",
        typeid(*input.get()).name());

    rows_ += input->size();
    bytes_ += static_cast<int64_t>(input->estimateFlatSize());
    ++batches_;
    // Drop the shared pointer immediately; no data is retained.
  }

  RowVectorPtr getOutput() override {
    return nullptr;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_;
  }

  void noMoreInput() override {
    addRuntimeStat(kDiscardedRows, RuntimeCounter(rows_));
    addRuntimeStat(
        kDiscardedBytes, RuntimeCounter(bytes_, RuntimeCounter::Unit::kBytes));
    addRuntimeStat(kDiscardedBatches, RuntimeCounter(batches_));
    Operator::noMoreInput();
  }

 private:
  int64_t rows_{0};
  int64_t bytes_{0};
  int64_t batches_{0};
};

// ---- Plan-node translator ---------------------------------------------------

class BenchmarkDiscardTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    auto discardNode =
        std::dynamic_pointer_cast<const BenchmarkDiscardNode>(node);
    if (!discardNode) {
      return nullptr;
    }
    return std::make_unique<BenchmarkDiscard>(ctx, discardNode, id);
  }
};

// ---- Adapter ----------------------------------------------------------------

class BenchmarkDiscardAdapter : public OperatorAdapter {
 public:
  BenchmarkDiscardAdapter() : OperatorAdapter("BenchmarkDiscard") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const BenchmarkDiscard*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const BenchmarkDiscardNode>(planNode) !=
        nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    // Keep the original operator; return no replacements.
    return {};
  }

  bool keepOperator() const override {
    return true;
  }
};

// ---- Registration -----------------------------------------------------------

core::PlanNodePtr addBenchmarkDiscard(core::PlanNodePtr source) {
  const core::PlanNodeId id = source->id() + "-discard";
  return std::make_shared<BenchmarkDiscardNode>(id, std::move(source));
}

void registerCudfBenchmarkDiscard() {
  // Register the plan-node translator exactly once for the lifetime of the
  // process.  Operator translators are global state that unregisterCudf does
  // not remove, so registering more than once would add duplicate translators.
  static std::once_flag translatorFlag;
  std::call_once(translatorFlag, []() {
    exec::Operator::registerOperator(
        std::make_unique<BenchmarkDiscardTranslator>());
  });

  // The adapter registry is rebuilt by registerCudf() on every call, so we
  // always register a fresh adapter here.
  OperatorAdapterRegistry::getInstance().registerAdapter(
      std::make_unique<BenchmarkDiscardAdapter>());
}

} // namespace facebook::velox::cudf_velox
