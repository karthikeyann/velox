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
#include "velox/serializers/ArrowIpcSerializer.h"

#include <gtest/gtest.h>
#include <sstream>

#include "velox/common/memory/ByteStream.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::serializer {
namespace {

class ArrowIpcSerializerTest : public ::testing::Test,
                               public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
    serde_ = std::make_unique<ArrowIpcVectorSerde>();
  }

  RowVectorPtr fuzz(
      const RowTypePtr& rowType,
      vector_size_t numRows,
      double nullRatio,
      uint32_t seed) {
    VectorFuzzer::Options options;
    options.vectorSize = numRows;
    options.nullRatio = nullRatio;
    options.containerVariableLength = false;
    options.containerLength = 5;
    options.stringVariableLength = true;
    options.stringLength = 16;
    VectorFuzzer fuzzer(options, pool_.get(), seed);
    return fuzzer.fuzzInputFlatRow(rowType);
  }

  // Iterative-path round trip: appends `input` once, flushes, deserializes,
  // and checks that the result matches `input` row-for-row.
  void testIterativeRoundTrip(
      const RowVectorPtr& input,
      const VectorSerde::Options* options = nullptr) {
    auto rowType = asRowType(input->type());
    const auto numRows = input->size();
    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto serializer = serde_->createIterativeSerializer(
        rowType, numRows, arena.get(), options);

    Scratch scratch;
    if (numRows > 0) {
      IndexRange range{0, numRows};
      serializer->append(
          input, folly::Range<const IndexRange*>(&range, 1), scratch);
    }

    std::ostringstream sink;
    OStreamOutputStream out(&sink);
    serializer->flush(&out);

    auto inputStream = toByteStream(sink.str());
    RowVectorPtr result;
    serde_->deserialize(
        inputStream.get(), pool_.get(), rowType, &result, options);
    test::assertEqualVectors(input, result);
  }

  // Batch-path round trip: serializes the full input as a single batch and
  // checks deserialization.
  void testBatchRoundTrip(
      const RowVectorPtr& input,
      const VectorSerde::Options* options = nullptr) {
    auto rowType = asRowType(input->type());
    auto serializer = serde_->createBatchSerializer(pool_.get(), options);

    std::ostringstream sink;
    OStreamOutputStream out(&sink);
    serializer->serialize(input, &out);

    auto inputStream = toByteStream(sink.str());
    RowVectorPtr result;
    serde_->deserialize(
        inputStream.get(), pool_.get(), rowType, &result, options);
    test::assertEqualVectors(input, result);
  }

  void testRoundTrip(
      const RowVectorPtr& input,
      const VectorSerde::Options* options = nullptr) {
    SCOPED_TRACE("iterative path");
    testIterativeRoundTrip(input, options);
    SCOPED_TRACE("batch path");
    testBatchRoundTrip(input, options);
  }

  std::unique_ptr<ByteInputStream> toByteStream(const std::string& input) {
    backing_ = input;
    ByteRange byteRange{
        reinterpret_cast<uint8_t*>(backing_.data()),
        static_cast<int32_t>(backing_.size()),
        0};
    return std::make_unique<BufferInputStream>(
        std::vector<ByteRange>{byteRange});
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<ArrowIpcVectorSerde> serde_;
  std::string backing_;
};

TEST_F(ArrowIpcSerializerTest, roundtripFlat) {
  const auto rowType =
      ROW({BIGINT(), DOUBLE(), BOOLEAN(), TINYINT(), REAL(), INTEGER()});
  testRoundTrip(fuzz(rowType, 1024, /*nullRatio=*/0.0, /*seed=*/1));
}

TEST_F(ArrowIpcSerializerTest, roundtripStrings) {
  const auto rowType = ROW({BIGINT(), VARCHAR()});
  testRoundTrip(fuzz(rowType, 256, /*nullRatio=*/0.0, /*seed=*/1));
}

TEST_F(ArrowIpcSerializerTest, roundtripArrays) {
  const auto rowType = ROW({BIGINT(), ARRAY(BIGINT())});
  testRoundTrip(fuzz(rowType, 128, /*nullRatio=*/0.0, /*seed=*/1));
}

TEST_F(ArrowIpcSerializerTest, roundtripMaps) {
  const auto rowType = ROW({BIGINT(), MAP(BIGINT(), REAL())});
  testRoundTrip(fuzz(rowType, 128, /*nullRatio=*/0.0, /*seed=*/1));
}

TEST_F(ArrowIpcSerializerTest, roundtripStructs) {
  const auto rowType = ROW({BIGINT(), ROW({BIGINT(), DOUBLE(), BOOLEAN()})});
  testRoundTrip(fuzz(rowType, 256, /*nullRatio=*/0.0, /*seed=*/1));
}

TEST_F(ArrowIpcSerializerTest, roundtripWithNulls) {
  const auto rowType = ROW({BIGINT(), DOUBLE(), VARCHAR(), ARRAY(BIGINT())});
  testRoundTrip(fuzz(rowType, 512, /*nullRatio=*/0.25, /*seed=*/2));
}

TEST_F(ArrowIpcSerializerTest, roundtripEmpty) {
  const auto rowType = ROW({BIGINT(), VARCHAR()});
  auto empty = BaseVector::create<RowVector>(rowType, 0, pool_.get());
  testRoundTrip(empty);
}

TEST_F(ArrowIpcSerializerTest, roundtripLz4) {
  const auto rowType = ROW({BIGINT(), DOUBLE(), VARCHAR()});
  auto data = fuzz(rowType, 4096, /*nullRatio=*/0.1, /*seed=*/3);
  ArrowIpcVectorSerde::ArrowIpcOptions options;
  options.compressionKind = common::CompressionKind::CompressionKind_LZ4;
  testRoundTrip(data, &options);
}

TEST_F(ArrowIpcSerializerTest, roundtripZstd) {
  const auto rowType = ROW({BIGINT(), DOUBLE(), VARCHAR()});
  auto data = fuzz(rowType, 4096, /*nullRatio=*/0.1, /*seed=*/3);
  ArrowIpcVectorSerde::ArrowIpcOptions options;
  options.compressionKind = common::CompressionKind::CompressionKind_ZSTD;
  testRoundTrip(data, &options);
}

TEST_F(ArrowIpcSerializerTest, unsupportedCompressionThrows) {
  const auto rowType = ROW({BIGINT()});
  auto data = fuzz(rowType, 16, /*nullRatio=*/0.0, /*seed=*/1);
  ArrowIpcVectorSerde::ArrowIpcOptions options;
  options.compressionKind = common::CompressionKind::CompressionKind_GZIP;
  EXPECT_THROW(testBatchRoundTrip(data, &options), VeloxUserError);
}

TEST_F(ArrowIpcSerializerTest, registration) {
  // tryRegisterNamedVectorSerde should be idempotent and never throw if
  // called twice.
  ArrowIpcVectorSerde::tryRegisterNamedVectorSerde();
  ArrowIpcVectorSerde::tryRegisterNamedVectorSerde();
  EXPECT_TRUE(isRegisteredNamedVectorSerde(ArrowIpcVectorSerde::name()));
  auto* registered = getNamedVectorSerde(ArrowIpcVectorSerde::name());
  ASSERT_NE(registered, nullptr);
  EXPECT_EQ(registered->kind(), ArrowIpcVectorSerde::name());
  deregisterNamedVectorSerde(ArrowIpcVectorSerde::name());
}

} // namespace
} // namespace facebook::velox::serializer
