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
#include "velox/common/base/tests/GTestUtils.h"

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

// A duplicate or a moved-from husk would break the exactly-once guarantee, so
// the type has to reject both at compile time.
static_assert(!std::is_copy_constructible_v<CudfBenchmarkLifecycle>);
static_assert(!std::is_copy_assignable_v<CudfBenchmarkLifecycle>);
static_assert(!std::is_move_constructible_v<CudfBenchmarkLifecycle>);
static_assert(!std::is_move_assignable_v<CudfBenchmarkLifecycle>);

// Order the benchmark takes its resources in, which is not the order they are
// released in: the S3 file system is registered first and finalized last.
constexpr CudfBenchmarkResource kAcquisitionOrder[] = {
    CudfBenchmarkResource::kS3FileSystem,
    CudfBenchmarkResource::kConnector,
    CudfBenchmarkResource::kBaseBenchmark,
    CudfBenchmarkResource::kIoExecutor,
    CudfBenchmarkResource::kCudf,
};

std::string_view name(CudfBenchmarkResource resource) {
  switch (resource) {
    case CudfBenchmarkResource::kConnector:
      return "connector";
    case CudfBenchmarkResource::kBaseBenchmark:
      return "base";
    case CudfBenchmarkResource::kIoExecutor:
      return "ioExecutor";
    case CudfBenchmarkResource::kCudf:
      return "cudf";
    case CudfBenchmarkResource::kS3FileSystem:
      return "s3";
  }
  VELOX_UNREACHABLE("Unknown benchmark resource.");
}

// Stands in for the benchmark's real resources. Records the order releases
// happen in so a test can assert the ordering contract, and can be told to
// fail a named resource so a test can assert what happens when a release
// cannot complete.
class FakeResources {
 public:
  // Records ownership of 'resource' with a release that appends to the release
  // log, or throws when 'resource' is failing.
  void own(CudfBenchmarkLifecycle& lifecycle, CudfBenchmarkResource resource) {
    lifecycle.own(resource, [this, resource] {
      released_.push_back(std::string(name(resource)));
      if (failing_.count(std::string(name(resource))) > 0) {
        VELOX_FAIL("Cannot release {}.", name(resource));
      }
    });
  }

  // Records ownership of every resource the benchmark takes for an enabled I/O
  // mode.
  void ownAll(CudfBenchmarkLifecycle& lifecycle) {
    for (const auto resource : kAcquisitionOrder) {
      own(lifecycle, resource);
    }
  }

  // Makes 'resource' fail every release until stopFailing() is called.
  void failOn(CudfBenchmarkResource resource) {
    failing_.insert(std::string(name(resource)));
  }

  void stopFailing() {
    failing_.clear();
  }

  // Names of the resources whose release was attempted, in order, including
  // attempts that failed.
  const std::vector<std::string>& released() const {
    return released_;
  }

  void clearLog() {
    released_.clear();
  }

 private:
  std::vector<std::string> released_;
  std::set<std::string> failing_;
};

class CudfBenchmarkLifecycleTest : public testing::Test {
 protected:
  CudfBenchmarkLifecycle lifecycle_;
  FakeResources resources_;
};

// A benchmark that ran to completion releases every resource once, in the
// order the enum declares rather than the order it took them in.
TEST_F(CudfBenchmarkLifecycleTest, normalShutdownReleasesEverythingInOrder) {
  resources_.ownAll(lifecycle_);
  ASSERT_TRUE(lifecycle_.ownsAny());

  lifecycle_.release();

  EXPECT_EQ(
      resources_.released(),
      (std::vector<std::string>{
          "connector", "base", "ioExecutor", "cudf", "s3"}));
  EXPECT_FALSE(lifecycle_.ownsAny());
}

// Disabled mode never registers the S3 file system, so shutdown must not
// finalize it. Finalizing is process-global and one-shot, so doing it here
// would poison S3 for anything else in the process.
TEST_F(CudfBenchmarkLifecycleTest, disabledModeNeverReleasesTheFileSystem) {
  for (const auto resource : kAcquisitionOrder) {
    if (resource != CudfBenchmarkResource::kS3FileSystem) {
      resources_.own(lifecycle_, resource);
    }
  }
  ASSERT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kS3FileSystem));

  lifecycle_.release();

  EXPECT_EQ(
      resources_.released(),
      (std::vector<std::string>{"connector", "base", "ioExecutor", "cudf"}));
  EXPECT_FALSE(lifecycle_.ownsAny());
}

// An enabled run that finds cuDF and the S3 file system already registered
// uses both but owns neither, so shutdown must leave the host process's
// registrations in place.
TEST_F(CudfBenchmarkLifecycleTest, preexistingGlobalStateIsNeverReleased) {
  resources_.own(lifecycle_, CudfBenchmarkResource::kConnector);
  resources_.own(lifecycle_, CudfBenchmarkResource::kBaseBenchmark);
  resources_.own(lifecycle_, CudfBenchmarkResource::kIoExecutor);
  ASSERT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kCudf));
  ASSERT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kS3FileSystem));

  lifecycle_.release();

  EXPECT_EQ(
      resources_.released(),
      (std::vector<std::string>{"connector", "base", "ioExecutor"}));
  EXPECT_FALSE(lifecycle_.ownsAny());
}

// Initialization that fails part way through releases what it took and nothing
// else. Here it failed before registering cuDF.
TEST_F(CudfBenchmarkLifecycleTest, partialInitializeReleasesOnlyWhatItTook) {
  resources_.own(lifecycle_, CudfBenchmarkResource::kS3FileSystem);
  resources_.own(lifecycle_, CudfBenchmarkResource::kConnector);
  resources_.own(lifecycle_, CudfBenchmarkResource::kBaseBenchmark);
  resources_.own(lifecycle_, CudfBenchmarkResource::kIoExecutor);

  EXPECT_TRUE(lifecycle_.releaseAfterFailure("benchmark initialization"));

  EXPECT_EQ(
      resources_.released(),
      (std::vector<std::string>{"connector", "base", "ioExecutor", "s3"}));
  EXPECT_FALSE(lifecycle_.ownsAny());
}

// A run that throws still has to release everything, and the cleanup must not
// replace the failure that explains the run.
TEST_F(CudfBenchmarkLifecycleTest, runExceptionReleasesWithoutMaskingTheCause) {
  resources_.ownAll(lifecycle_);

  std::string reported;
  try {
    VELOX_USER_FAIL("The run failed.");
  } catch (const VeloxException& e) {
    lifecycle_.releaseAfterFailure("benchmark run");
    reported = e.message();
  }

  EXPECT_EQ(reported, "The run failed.");
  EXPECT_EQ(
      resources_.released(),
      (std::vector<std::string>{
          "connector", "base", "ioExecutor", "cudf", "s3"}));
  EXPECT_FALSE(lifecycle_.ownsAny());
}

// A release that fails must not strand the resources after it, and the caller
// has to hear about it.
TEST_F(
    CudfBenchmarkLifecycleTest,
    releaseAttemptsEverythingAndReportsTheFirstFailure) {
  resources_.ownAll(lifecycle_);
  resources_.failOn(CudfBenchmarkResource::kBaseBenchmark);
  resources_.failOn(CudfBenchmarkResource::kCudf);

  VELOX_ASSERT_THROW(lifecycle_.release(), "Cannot release base.");

  EXPECT_EQ(
      resources_.released(),
      (std::vector<std::string>{
          "connector", "base", "ioExecutor", "cudf", "s3"}));
  // The two that failed stay owned; everything else is gone.
  EXPECT_TRUE(lifecycle_.owns(CudfBenchmarkResource::kBaseBenchmark));
  EXPECT_TRUE(lifecycle_.owns(CudfBenchmarkResource::kCudf));
  EXPECT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kConnector));
  EXPECT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kIoExecutor));
  EXPECT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kS3FileSystem));
}

// Finalizing the S3 file system is the release most likely to fail, because it
// refuses while any cached file system is still referenced. A caller that
// drops the reference can retry, and only the file system is retried.
TEST_F(CudfBenchmarkLifecycleTest, retryReleasesOnlyWhatIsStillOwned) {
  resources_.ownAll(lifecycle_);
  resources_.failOn(CudfBenchmarkResource::kS3FileSystem);

  VELOX_ASSERT_THROW(lifecycle_.release(), "Cannot release s3.");
  ASSERT_TRUE(lifecycle_.owns(CudfBenchmarkResource::kS3FileSystem));

  resources_.stopFailing();
  resources_.clearLog();
  lifecycle_.release();

  EXPECT_EQ(resources_.released(), (std::vector<std::string>{"s3"}));
  EXPECT_FALSE(lifecycle_.ownsAny());
}

// Shutting down twice must not release anything twice. Finalizing the S3 file
// system a second time would tear down state the process no longer owns.
TEST_F(CudfBenchmarkLifecycleTest, repeatedShutdownReleasesNothingTwice) {
  resources_.ownAll(lifecycle_);

  lifecycle_.release();
  const auto afterFirst = resources_.released();
  lifecycle_.release();
  lifecycle_.release();

  EXPECT_EQ(resources_.released(), afterFirst);
  EXPECT_FALSE(lifecycle_.ownsAny());
}

// Shutting down a benchmark that never initialized is a no-op rather than a
// failure, because the caller cannot always tell how far initialization got.
TEST_F(CudfBenchmarkLifecycleTest, releasingNothingIsAllowed) {
  EXPECT_FALSE(lifecycle_.ownsAny());
  lifecycle_.release();
  EXPECT_TRUE(resources_.released().empty());
}

// A resource taken twice, as happens when the connector is replaced, must
// still be released once.
TEST_F(CudfBenchmarkLifecycleTest, retakingAResourceReplacesItsRelease) {
  int firstReleases = 0;
  int secondReleases = 0;
  lifecycle_.own(CudfBenchmarkResource::kConnector, [&] { ++firstReleases; });
  lifecycle_.own(CudfBenchmarkResource::kConnector, [&] { ++secondReleases; });

  lifecycle_.release();

  EXPECT_EQ(firstReleases, 0);
  EXPECT_EQ(secondReleases, 1);
}

TEST_F(CudfBenchmarkLifecycleTest, aResourceNeedsARelease) {
  VELOX_ASSERT_THROW(
      lifecycle_.own(CudfBenchmarkResource::kCudf, nullptr),
      "A benchmark resource needs a release callback.");
}

// Duplicating or transferring the owner would let the same resource be
// released twice, so the type rejects both. The static_asserts above are the
// real check; this repeats them at runtime so the guarantee is a named test.
TEST_F(CudfBenchmarkLifecycleTest, isNeitherCopyableNorMovable) {
  EXPECT_FALSE(std::is_copy_constructible_v<CudfBenchmarkLifecycle>);
  EXPECT_FALSE(std::is_copy_assignable_v<CudfBenchmarkLifecycle>);
  EXPECT_FALSE(std::is_move_constructible_v<CudfBenchmarkLifecycle>);
  EXPECT_FALSE(std::is_move_assignable_v<CudfBenchmarkLifecycle>);
}

// A release is arbitrary caller code, so it can throw something that does not
// derive from std::exception. release() has to carry it out intact rather than
// let it escape the loop and strand the resources behind it.
TEST_F(CudfBenchmarkLifecycleTest, releaseCarriesOutANonStandardFailure) {
  resources_.own(lifecycle_, CudfBenchmarkResource::kConnector);
  lifecycle_.own(CudfBenchmarkResource::kBaseBenchmark, [] { throw 7; });
  resources_.own(lifecycle_, CudfBenchmarkResource::kCudf);

  int caught = 0;
  try {
    lifecycle_.release();
    FAIL() << "release() should have rethrown the non-standard failure";
  } catch (int thrown) {
    caught = thrown;
  }

  EXPECT_EQ(caught, 7);
  // The resources on either side were still released, and only the failing one
  // stays owned.
  EXPECT_EQ(
      resources_.released(), (std::vector<std::string>{"connector", "cudf"}));
  EXPECT_TRUE(lifecycle_.owns(CudfBenchmarkResource::kBaseBenchmark));
  EXPECT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kConnector));
  EXPECT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kCudf));
}

// Cleaning up after a failure must not replace the failure being reported, and
// that holds for a cleanup failure of any type. releaseAfterFailure() reports
// the non-standard one without a message it cannot obtain, and swallows it.
TEST_F(
    CudfBenchmarkLifecycleTest,
    releaseAfterFailureSuppressesANonStandardFailure) {
  resources_.ownAll(lifecycle_);
  lifecycle_.own(CudfBenchmarkResource::kS3FileSystem, [] { throw 7; });

  std::string reported;
  try {
    VELOX_USER_FAIL("The run failed.");
  } catch (const VeloxException& e) {
    EXPECT_FALSE(lifecycle_.releaseAfterFailure("benchmark run"));
    reported = e.message();
  }

  EXPECT_EQ(reported, "The run failed.");
  // The file system stays owned for a retry; everything else is released.
  EXPECT_TRUE(lifecycle_.owns(CudfBenchmarkResource::kS3FileSystem));
  EXPECT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kConnector));
  EXPECT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kBaseBenchmark));
  EXPECT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kIoExecutor));
  EXPECT_FALSE(lifecycle_.owns(CudfBenchmarkResource::kCudf));
}

} // namespace
} // namespace facebook::velox::cudf_velox
