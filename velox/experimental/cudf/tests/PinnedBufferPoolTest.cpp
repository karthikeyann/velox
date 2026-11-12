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
#include "velox/experimental/cudf/connectors/hive/PinnedBufferPool.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/common/base/tests/GTestUtils.h"

#include <chrono>
#include <cstring>
#include <future>
#include <thread>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// The base registers cuDF, which the pool needs to page-lock memory.
class PinnedBufferPoolTest : public ::facebook::velox::cudf_velox::exec::test::
                                 CudfHiveConnectorTestBase {};

TEST_F(PinnedBufferPoolTest, capacity) {
  {
    PinnedBufferPool pool(4'096, 10'000);
    EXPECT_EQ(pool.bufferSize(), 4'096);
    EXPECT_EQ(pool.maxBuffers(), 2);
    EXPECT_EQ(pool.numAllocated(), 0);
  }
  {
    // A capacity below one buffer still allows one buffer.
    PinnedBufferPool pool(4'096, 100);
    EXPECT_EQ(pool.maxBuffers(), 1);
  }
  VELOX_ASSERT_THROW(
      PinnedBufferPool(0, 4'096), "Pinned buffer size must be positive");
}

TEST_F(PinnedBufferPoolTest, reusesBuffers) {
  PinnedBufferPool pool(4'096, 8'192);
  auto buffer = pool.acquire();
  ASSERT_NE(buffer.get(), nullptr);
  ASSERT_NE(buffer->stream, nullptr);
  ASSERT_NE(buffer->event, nullptr);
  std::memset(buffer->data, 0xAB, pool.bufferSize());
  const auto* first = buffer.get();
  buffer.reset();

  buffer = pool.acquire();
  EXPECT_EQ(buffer.get(), first);
  EXPECT_EQ(pool.numAllocated(), 1);

  auto other = pool.acquire();
  EXPECT_NE(other.get(), first);
  EXPECT_EQ(pool.numAllocated(), 2);
}

// A pool that cannot page-lock any buffer hands out empty handles, keeps its
// accounting and shuts down without waiting.
TEST_F(PinnedBufferPoolTest, allocationFailure) {
  const size_t tooLarge{1ULL << 46};
  PinnedBufferPool pool(tooLarge, 2 * tooLarge);
  for (auto i = 0; i < 2; ++i) {
    auto buffer = pool.acquire();
    EXPECT_EQ(buffer.get(), nullptr);
    EXPECT_EQ(pool.numAllocated(), 0);
  }
}

TEST_F(PinnedBufferPoolTest, waitsAtCapacity) {
  PinnedBufferPool pool(4'096, 8'192);
  auto first = pool.acquire();
  auto second = pool.acquire();
  const auto* released = first.get();

  std::promise<PinnedBufferPool::Buffer> promise;
  auto acquired = promise.get_future();
  std::thread waiter([&]() { promise.set_value(pool.acquire()); });
  EXPECT_EQ(
      acquired.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout);

  first.reset();
  auto third = acquired.get();
  waiter.join();
  EXPECT_EQ(third.get(), released);
  EXPECT_EQ(pool.numAllocated(), 2);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
