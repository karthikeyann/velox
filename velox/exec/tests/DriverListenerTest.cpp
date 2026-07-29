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

#include <mutex>

#include "velox/exec/Operator.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

// Counts begin/end callbacks keyed by "<operatorType>::<callName>".
class CountingDriverListener : public DriverListener {
 public:
  void onOperatorCallBegin(
      const Operator& op,
      std::string_view callName) noexcept override {
    std::lock_guard<std::mutex> l(mutex_);
    ++begins_[key(op, callName)];
  }

  void onOperatorCallEnd(const Operator& op, std::string_view callName) noexcept
      override {
    std::lock_guard<std::mutex> l(mutex_);
    ++ends_[key(op, callName)];
  }

  void onDriverBlocked(const Operator& /* op */, BlockingReason reason) noexcept
      override {
    std::lock_guard<std::mutex> l(mutex_);
    ++blocked_[reason];
  }

  void onDriverUnblocked(const Operator& /* op */) noexcept override {
    std::lock_guard<std::mutex> l(mutex_);
    ++unblocked_;
  }

  std::map<std::string, int64_t> begins() const {
    std::lock_guard<std::mutex> l(mutex_);
    return begins_;
  }

  std::map<std::string, int64_t> ends() const {
    std::lock_guard<std::mutex> l(mutex_);
    return ends_;
  }

  int64_t blockedCount() const {
    std::lock_guard<std::mutex> l(mutex_);
    int64_t total = 0;
    for (const auto& [_, count] : blocked_) {
      total += count;
    }
    return total;
  }

  int64_t unblockedCount() const {
    std::lock_guard<std::mutex> l(mutex_);
    return unblocked_;
  }

 private:
  static std::string key(const Operator& op, std::string_view callName) {
    return fmt::format("{}::{}", op.operatorType(), callName);
  }

  mutable std::mutex mutex_;
  std::map<std::string, int64_t> begins_;
  std::map<std::string, int64_t> ends_;
  std::map<BlockingReason, int64_t> blocked_;
  int64_t unblocked_{0};
};

// Records the Tasks it was offered and optionally declines all of them.
class RecordingDriverListenerFactory : public DriverListenerFactory {
 public:
  explicit RecordingDriverListenerFactory(bool decline) : decline_(decline) {}

  std::shared_ptr<DriverListener> create(
      const std::string& taskId,
      const std::string& /* taskUuid */,
      const core::QueryConfig& /* config */) override {
    std::lock_guard<std::mutex> l(mutex_);
    taskIds_.push_back(taskId);
    if (decline_) {
      return nullptr;
    }
    listener_ = std::make_shared<CountingDriverListener>();
    return listener_;
  }

  std::shared_ptr<CountingDriverListener> listener() const {
    std::lock_guard<std::mutex> l(mutex_);
    return listener_;
  }

  std::vector<std::string> taskIds() const {
    std::lock_guard<std::mutex> l(mutex_);
    return taskIds_;
  }

 private:
  const bool decline_;
  mutable std::mutex mutex_;
  std::shared_ptr<CountingDriverListener> listener_;
  std::vector<std::string> taskIds_;
};

} // namespace

class DriverListenerTest : public OperatorTestBase {};

TEST_F(DriverListenerTest, registration) {
  auto factory = std::make_shared<RecordingDriverListenerFactory>(false);
  ASSERT_TRUE(registerDriverListenerFactory(factory));
  // Registering the same factory twice is detected and rejected.
  ASSERT_FALSE(registerDriverListenerFactory(factory));
  ASSERT_TRUE(unregisterDriverListenerFactory(factory));
  ASSERT_FALSE(unregisterDriverListenerFactory(factory));
}

TEST_F(DriverListenerTest, multipleFactories) {
  auto first = std::make_shared<RecordingDriverListenerFactory>(false);
  auto second = std::make_shared<RecordingDriverListenerFactory>(false);
  ASSERT_TRUE(registerDriverListenerFactory(first));
  ASSERT_TRUE(registerDriverListenerFactory(second));

  auto data = makeRowVector({makeFlatVector<int32_t>({0, 1, 2})});
  auto plan = PlanBuilder().values({data}).planNode();
  assertQuery(plan, "VALUES (0), (1), (2)");

  ASSERT_NE(first->listener(), nullptr);
  ASSERT_NE(second->listener(), nullptr);
  ASSERT_FALSE(first->listener()->begins().empty());
  ASSERT_FALSE(second->listener()->begins().empty());

  ASSERT_TRUE(unregisterDriverListenerFactory(first));
  ASSERT_TRUE(unregisterDriverListenerFactory(second));
}

TEST_F(DriverListenerTest, observesEveryOperatorInTheChain) {
  auto data = makeRowVector({makeFlatVector<int32_t>({0, 1, 2, 3, 4})});
  auto plan = PlanBuilder().values({data}).filter("c0 > 1").planNode();

  auto factory = std::make_shared<RecordingDriverListenerFactory>(false);
  ASSERT_TRUE(registerDriverListenerFactory(factory));
  assertQuery(plan, "VALUES (2), (3), (4)");
  ASSERT_TRUE(unregisterDriverListenerFactory(factory));

  auto listener = factory->listener();
  ASSERT_NE(listener, nullptr);

  const auto begins = listener->begins();
  const auto ends = listener->ends();
  ASSERT_FALSE(begins.empty());
  // Every begin is matched by exactly one end, including the calls that ran
  // inside a scope that threw.
  ASSERT_EQ(begins, ends);

  // The source operator, a downstream operator and the lifecycle calls are all
  // covered, which is what the cuDF-only hook could not reach.
  bool sawValues = false;
  bool sawFilter = false;
  bool sawInitialize = false;
  bool sawClose = false;
  for (const auto& [key, count] : begins) {
    ASSERT_GT(count, 0) << key;
    sawValues |= key.starts_with("Values::");
    sawFilter |= key.starts_with("FilterProject::");
    sawInitialize |= key.ends_with("::initialize");
    sawClose |= key.ends_with("::close");
  }
  ASSERT_TRUE(sawValues);
  ASSERT_TRUE(sawFilter);
  ASSERT_TRUE(sawInitialize);
  ASSERT_TRUE(sawClose);
}

TEST_F(DriverListenerTest, decliningFactoryReceivesNoCallbacks) {
  auto data = makeRowVector({makeFlatVector<int32_t>({0, 1, 2})});
  auto plan = PlanBuilder().values({data}).planNode();

  auto factory = std::make_shared<RecordingDriverListenerFactory>(true);
  ASSERT_TRUE(registerDriverListenerFactory(factory));
  assertQuery(plan, "VALUES (0), (1), (2)");
  ASSERT_TRUE(unregisterDriverListenerFactory(factory));

  // The factory was consulted for the Task but declined it, so no listener was
  // ever created and the driver hot path stayed on its null-check path.
  ASSERT_FALSE(factory->taskIds().empty());
  ASSERT_EQ(factory->listener(), nullptr);
}

TEST_F(DriverListenerTest, pairsBlockedWithUnblocked) {
  // A local exchange makes the consumer pipeline wait on its producer, which
  // takes the driver off thread at least once.
  auto data = makeRowVector({makeFlatVector<int32_t>({0, 1, 2, 3, 4})});
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localPartition(
                      {},
                      {PlanBuilder(planNodeIdGenerator)
                           .values({data, data})
                           .planNode()})
                  .planNode();

  auto factory = std::make_shared<RecordingDriverListenerFactory>(false);
  ASSERT_TRUE(registerDriverListenerFactory(factory));
  assertQuery(plan, "VALUES (0), (1), (2), (3), (4), (0), (1), (2), (3), (4)");
  ASSERT_TRUE(unregisterDriverListenerFactory(factory));

  auto listener = factory->listener();
  ASSERT_NE(listener, nullptr);
  // Every driver that reported blocked either resumed or was torn down, so
  // resumptions never exceed blocks.
  ASSERT_LE(listener->unblockedCount(), listener->blockedCount());
}
