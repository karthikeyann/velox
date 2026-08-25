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

#include "velox/benchmarks/tpch/TpchBenchmark.h"

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

using facebook::velox::RunStats;

namespace {

// The exception the run fails with. A dedicated type so a test can tell it
// apart from any exception the cleanup throws.
struct RunFailure : public std::runtime_error {
  RunFailure() : std::runtime_error("The benchmark run failed.") {}
};

// A benchmark that fails the way the handler under test has to cope with:
// initialize() succeeds, so the handler is reached; runMain() throws the
// failure that must reach the caller; and shutdown() throws whatever the test
// asks for, standing in for a release callback that cannot do its job.
//
// Nothing here touches a dataset, a connector or a memory manager, so the
// handler can be driven without the environment a real run needs.
class FailingBenchmark : public TpchBenchmark {
 public:
  explicit FailingBenchmark(std::function<void()> failShutdown)
      : failShutdown_{std::move(failShutdown)} {}

  void initialize() override {}

  void runMain(std::ostream& /* out */, RunStats& /* runStats */) override {
    ++numRuns_;
    throw RunFailure();
  }

  void shutdown() override {
    ++numShutdowns_;
    failShutdown_();
  }

  int numRuns() const {
    return numRuns_;
  }

  int numShutdowns() const {
    return numShutdowns_;
  }

 private:
  const std::function<void()> failShutdown_;
  int numRuns_{0};
  int numShutdowns_{0};
};

class TpchBenchmarkMainTest : public testing::Test {
 protected:
  void SetUp() override {
    // Empty selects runMain() over runAllCombinations(). It is a process-global
    // flag, so it is restored rather than assumed.
    savedTestFlagsFile_ = FLAGS_test_flags_file;
    FLAGS_test_flags_file = "";
  }

  void TearDown() override {
    FLAGS_test_flags_file = savedTestFlagsFile_;
    // The benchmark is process-global and owns whatever it was given, so no
    // test may leave one behind for the next.
    benchmark.reset();
  }

  // Runs tpchBenchmarkMain() with a benchmark whose shutdown fails as
  // 'failShutdown' says, and returns it so the caller can count the attempts.
  // Fails the test unless a RunFailure, rather than the cleanup failure,
  // escaped.
  FailingBenchmark* runExpectingRunFailure(std::function<void()> failShutdown) {
    auto owned = std::make_unique<FailingBenchmark>(std::move(failShutdown));
    auto* benchmarkPtr = owned.get();
    benchmark = std::move(owned);

    try {
      tpchBenchmarkMain();
      ADD_FAILURE() << "tpchBenchmarkMain() should have rethrown the run "
                       "failure.";
    } catch (const RunFailure& e) {
      EXPECT_STREQ(e.what(), "The benchmark run failed.");
    } catch (const std::exception& e) {
      ADD_FAILURE() << "The cleanup failure replaced the run failure: "
                    << e.what();
    } catch (...) {
      ADD_FAILURE() << "A non-standard cleanup failure replaced the run "
                       "failure.";
    }
    return benchmarkPtr;
  }

  std::string savedTestFlagsFile_;
};

// A cleanup failure that derives from std::exception is logged with its
// message, and the run failure is still what the caller sees.
TEST_F(
    TpchBenchmarkMainTest,
    aStandardCleanupFailureDoesNotReplaceTheRunFailure) {
  auto* failing = runExpectingRunFailure(
      [] { throw std::runtime_error("Shutdown failed."); });

  EXPECT_EQ(failing->numRuns(), 1);
  // Exactly once: the handler must not skip cleanup, and the success-path
  // shutdown at the end of tpchBenchmarkMain() must not also run.
  EXPECT_EQ(failing->numShutdowns(), 1);
}

// Cleanup runs arbitrary release callbacks, which need not throw a
// std::exception. Without a catch-all the cleanup failure would replace the run
// failure it was reporting, so this is the regression test for that branch.
TEST_F(
    TpchBenchmarkMainTest,
    aNonStandardCleanupFailureDoesNotReplaceTheRunFailure) {
  auto* failing = runExpectingRunFailure([] { throw 7; });

  EXPECT_EQ(failing->numRuns(), 1);
  EXPECT_EQ(failing->numShutdowns(), 1);
}

} // namespace

// A dedicated main rather than gtest_main or folly::Init: the benchmark library
// declares a validator that rejects an empty --data_path, which folly::Init
// would run against the empty default. These tests never read a dataset, so
// they parse no flags instead of relaxing the production validator.
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
