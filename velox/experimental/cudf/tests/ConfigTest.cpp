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
#include "velox/experimental/cudf/connectors/hive/CudfDecodedColumnCache.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#if defined(VELOX_CUDF_HAS_UCX)
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"

#include "velox/exec/OutputTransportRegistry.h"
#endif

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox::test {

TEST(ConfigTest, cudfConfig) {
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfEnabled, "false"},
      {CudfConfig::kCudfDebugEnabled, "true"},
      {CudfConfig::kCudfMemoryResource, "arena"},
      {CudfConfig::kCudfMemoryPercent, "25"},
      {CudfConfig::kCudfFunctionNamePrefix, "presto"},
      {CudfConfig::kCudfStreamingGroupbyEnabled, "true"},
      {CudfConfig::kCudfStreamingGroupbyCapacityMultiplier, "3.5"},
      {CudfConfig::kCudfAllowCpuFallback, "false"},
      {CudfConfig::kUcxExchange, "true"},
      {CudfConfig::kUcxxErrorHandling, "false"},
      {CudfConfig::kUcxIntraNodeExchange, "true"},
      {CudfConfig::kUcxxBlockingPolling, "false"},
      {CudfConfig::kUcxExchangeLogLevel, "2"},
      {CudfConfig::kUcxPartitionedOutputBatchRows, "100000"},
      {CudfConfig::kUcxExchangeCompression, "column-adaptive-freq-pfor-min128"},
      {CudfConfig::kUcxExchangeCompressionPipeline, "true"},
      {CudfConfig::kUcxExchangeCompressionPipelineThreads, "2"},
      {CudfConfig::kUcxExchangeCompressionMinBytes, "268435456"},
      {CudfConfig::kUcxExchangeCompressionSafetyMargin, "1.5"}};

  CudfConfig config;
  ASSERT_FALSE(config.streamingGroupbyEnabled);
  ASSERT_EQ(config.streamingGroupbyCapacityMultiplier, 2.0);
  config.initialize(std::move(options));
  ASSERT_EQ(config.enabled, false);
  ASSERT_EQ(config.debugEnabled, true);
  ASSERT_EQ(config.memoryResource, "arena");
  ASSERT_EQ(config.memoryPercent, 25);
  ASSERT_EQ(config.functionNamePrefix, "presto");
  ASSERT_EQ(config.streamingGroupbyEnabled, true);
  ASSERT_EQ(config.streamingGroupbyCapacityMultiplier, 3.5);
  ASSERT_EQ(config.allowCpuFallback, false);
  ASSERT_TRUE(config.exchange);
  ASSERT_FALSE(config.ucxxErrorHandling);
  ASSERT_TRUE(config.intraNodeExchange);
  ASSERT_FALSE(config.ucxxBlockingPolling);
  ASSERT_EQ(config.exchangeLogLevel, 2);
  ASSERT_EQ(config.partitionedOutputBatchRows, 100000);
  ASSERT_EQ(config.exchangeCompression, "column-adaptive-freq-pfor-min128");
  ASSERT_TRUE(config.exchangeCompressionPipeline);
  ASSERT_EQ(config.exchangeCompressionPipelineThreads, 2);
  ASSERT_EQ(config.exchangeCompressionMinBytes, 268435456);
  ASSERT_DOUBLE_EQ(config.exchangeCompressionSafetyMargin, 1.5);
}

TEST(ConfigTest, decodedColumnCacheBudgets) {
  CudfConfig config;
  EXPECT_FALSE(config.decodedColumnCacheMaxPinnedBytes.has_value());
  EXPECT_FALSE(config.decodedColumnCacheMaxGpuBytes.has_value());

  config.initialize({
      {CudfConfig::kCudfDecodedColumnCacheMaxPinnedBytes, "137438953472"},
      {CudfConfig::kCudfDecodedColumnCacheMaxGpuBytes, "223338299392"},
  });
  EXPECT_EQ(config.decodedColumnCacheMaxPinnedBytes, uint64_t{128} << 30);
  EXPECT_EQ(config.decodedColumnCacheMaxGpuBytes, uint64_t{208} << 30);

  // Reinitializing other options must not reset explicit application budgets.
  config.initialize({});
  EXPECT_EQ(config.decodedColumnCacheMaxPinnedBytes, uint64_t{128} << 30);
  EXPECT_EQ(config.decodedColumnCacheMaxGpuBytes, uint64_t{208} << 30);
  config.initialize({{CudfConfig::kCudfDecodedColumnCacheMaxGpuBytes, "0"}});
  EXPECT_EQ(config.decodedColumnCacheMaxGpuBytes, 0);

  EXPECT_ANY_THROW(config.initialize(
      {{CudfConfig::kCudfDecodedColumnCacheMaxPinnedBytes, "0"}}));
  for (const auto* key :
       {CudfConfig::kCudfDecodedColumnCacheMaxPinnedBytes,
        CudfConfig::kCudfDecodedColumnCacheMaxGpuBytes}) {
    for (const auto* value : {"-1", "1GiB", "", "18446744073709551616"}) {
      SCOPED_TRACE(std::string(key) + "=" + value);
      EXPECT_ANY_THROW(config.initialize({{key, value}}));
    }
  }
}

TEST(ConfigTest, decodedColumnCacheStartupBudgets) {
  auto& config = CudfConfig::getInstance();
  const auto saved = config;
  SCOPE_EXIT {
    unregisterCudf();
    config = saved;
  };
  config.initialize({
      {CudfConfig::kCudfDecodedColumnCacheMaxPinnedBytes, "134217728"},
      {CudfConfig::kCudfDecodedColumnCacheMaxGpuBytes, "67108864"},
  });
  registerCudf();
  auto& cache = connector::hive::CudfDecodedColumnCache::instance();
  EXPECT_EQ(cache.maxPinnedBytes(), uint64_t{128} << 20);
  EXPECT_EQ(cache.maxGpuBytes(), uint64_t{64} << 20);
  // Registration is idempotent even after the cache has been constructed.
  EXPECT_NO_THROW(registerCudf());
}

#if defined(VELOX_CUDF_HAS_UCX)
TEST(ConfigTest, ucxTransportRegistration) {
  auto& config = CudfConfig::getInstance();
  std::unordered_map<std::string, std::string> options = {
      {CudfConfig::kCudfEnabled, "true"}, {CudfConfig::kUcxExchange, "true"}};
  config.initialize(std::move(options));

  exec::OutputTransportRegistry::unregisterAll();
  registerAllOperatorAdapters();

  auto entry = exec::OutputTransportRegistry::tryGet(
      std::string{core::TransportKind::kUcx});
  ASSERT_NE(entry, nullptr);
  EXPECT_NE(
      std::dynamic_pointer_cast<ucx_exchange::UcxOutputQueueManager>(
          entry->manager),
      nullptr);
  exec::OutputTransportRegistry::unregisterAll();
}
#endif
} // namespace facebook::velox::cudf_velox::test
