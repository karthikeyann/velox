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
#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include <iomanip>
#include <iostream>

#include "velox/common/memory/ByteStream.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

#ifdef VELOX_ENABLE_ARROW_IPC_BENCHMARK
#include "velox/serializers/ArrowIpcSerializer.h"
#endif

namespace facebook::velox::test {
namespace {

/// Bench parameters that vary per benchmark instantiation. Threading them
/// through one struct keeps the benchmark methods to two arguments and
/// avoids exploding the macro expansions.
struct BenchParams {
  vector_size_t numRows;
  double nullRatio{0.0};
  common::CompressionKind compression{
      common::CompressionKind::CompressionKind_NONE};
};

/// Microbenchmark for PrestoVectorSerde covering serialize, deserialize,
/// and round-trip paths for both the iterative and batch APIs. Closes the
/// gap originally requested by issue facebookincubator/velox#1732 and
/// extends coverage with compression, null ratios, and wide rows so the
/// numbers can drive future serialization decisions (e.g. exchange over
/// UCX) on more than one workload.
class PrestoSerializerBenchmark {
 public:
  PrestoSerializerBenchmark()
      : serde_(std::make_unique<serializer::presto::PrestoVectorSerde>()) {
    if (!isRegisteredVectorSerde()) {
      serializer::presto::PrestoVectorSerde::registerVectorSerde();
    }
  }

  /// Times append() + flush() through the iterative serializer, the path
  /// used by the Velox PartitionedOutput operator. Suspends the benchmark
  /// timer while building the input vector and the StreamArena so only the
  /// serialization cost is measured.
  void iterativeSerialize(const RowTypePtr& rowType, BenchParams params) {
    folly::BenchmarkSuspender suspender;
    auto data = makeData(rowType, params);
    IndexRange range{0, params.numRows};
    folly::Range<const IndexRange*> ranges(&range, 1);
    auto opts = optionsFor(params);
    suspender.dismiss();

    Scratch scratch;
    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto serializer = serde_->createIterativeSerializer(
        rowType, params.numRows, arena.get(), &opts);
    serializer->append(data, ranges, scratch);

    IOBufOutputStream out(*pool_, nullptr, serializer->maxSerializedSize());
    serializer->flush(&out);
    folly::doNotOptimizeAway(out.tellp());
  }

  /// Times BatchVectorSerializer::serialize(), the single-shot path used
  /// when an entire RowVector goes to one destination. This is the path
  /// the upstream "Optimize PrestoBatchVectorSerializer [N/7]" series
  /// targeted before being reverted in Feb 2025.
  void batchSerialize(const RowTypePtr& rowType, BenchParams params) {
    folly::BenchmarkSuspender suspender;
    auto data = makeData(rowType, params);
    IndexRange range{0, params.numRows};
    folly::Range<const IndexRange*> ranges(&range, 1);
    auto opts = optionsFor(params);
    auto serializer = serde_->createBatchSerializer(pool_.get(), &opts);
    suspender.dismiss();

    Scratch scratch;
    IOBufOutputStream out(*pool_, nullptr);
    serializer->serialize(data, ranges, scratch, &out);
    folly::doNotOptimizeAway(out.tellp());
  }

  /// Times deserialize() only. The serialized payload is built outside the
  /// timed region so the measurement reflects pure deserialization cost.
  void deserialize(const RowTypePtr& rowType, BenchParams params) {
    folly::BenchmarkSuspender suspender;
    auto serialized = serializeOnce(rowType, params);
    auto byteRanges = byteRangesFromIOBuf(serialized.get());
    auto opts = optionsFor(params);
    suspender.dismiss();

    BufferInputStream input(byteRanges);
    RowVectorPtr result;
    serde_->deserialize(&input, pool_.get(), rowType, &result, &opts);
    folly::doNotOptimizeAway(result);
  }

  /// Times serialize + deserialize in one go. Useful as a sanity check that
  /// the ratio of serialize to deserialize is reasonable.
  void roundtrip(const RowTypePtr& rowType, BenchParams params) {
    folly::BenchmarkSuspender suspender;
    auto data = makeData(rowType, params);
    IndexRange range{0, params.numRows};
    folly::Range<const IndexRange*> ranges(&range, 1);
    auto opts = optionsFor(params);
    suspender.dismiss();

    Scratch scratch;
    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto serializer = serde_->createIterativeSerializer(
        rowType, params.numRows, arena.get(), &opts);
    serializer->append(data, ranges, scratch);

    IOBufOutputStream out(*pool_, nullptr, serializer->maxSerializedSize());
    serializer->flush(&out);
    auto iobuf = out.getIOBuf();

    auto byteRanges = byteRangesFromIOBuf(iobuf.get());
    BufferInputStream input(byteRanges);
    RowVectorPtr result;
    serde_->deserialize(&input, pool_.get(), rowType, &result, &opts);
    folly::doNotOptimizeAway(result);
  }

  /// Returns the size in bytes of one serialized payload for the given
  /// scenario. Used by main() to print a bytes/op companion table after
  /// folly::runBenchmarks() so MB/s can be derived from the timing output.
  int64_t serializedBytes(const RowTypePtr& rowType, BenchParams params) {
    auto iobuf = serializeOnce(rowType, params);
    return iobuf->computeChainDataLength();
  }

 private:
  static serializer::presto::PrestoVectorSerde::PrestoOptions optionsFor(
      const BenchParams& params) {
    serializer::presto::PrestoVectorSerde::PrestoOptions opts;
    opts.compressionKind = params.compression;
    return opts;
  }

  std::unique_ptr<folly::IOBuf> serializeOnce(
      const RowTypePtr& rowType,
      BenchParams params) {
    auto data = makeData(rowType, params);
    IndexRange range{0, params.numRows};
    folly::Range<const IndexRange*> ranges(&range, 1);
    auto opts = optionsFor(params);

    Scratch scratch;
    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto serializer = serde_->createIterativeSerializer(
        rowType, params.numRows, arena.get(), &opts);
    serializer->append(data, ranges, scratch);

    IOBufOutputStream out(*pool_, nullptr, serializer->maxSerializedSize());
    serializer->flush(&out);
    return out.getIOBuf();
  }

  RowVectorPtr makeData(const RowTypePtr& rowType, const BenchParams& params) {
    VectorFuzzer::Options options;
    options.vectorSize = params.numRows;
    options.nullRatio = params.nullRatio;
    constexpr uint32_t kSeed = 1;
    VectorFuzzer fuzzer(options, pool_.get(), kSeed);
    return fuzzer.fuzzInputFlatRow(rowType);
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  std::unique_ptr<serializer::presto::PrestoVectorSerde> serde_;
};

#ifdef VELOX_ENABLE_ARROW_IPC_BENCHMARK
/// Parallel benchmark for ArrowIpcVectorSerde that uses the same
/// VectorFuzzer-driven inputs and the same BenchParams as
/// PrestoSerializerBenchmark, so folly's relative-time column lines up
/// apples-to-apples between the two.
class ArrowIpcSerializerBenchmark {
 public:
  ArrowIpcSerializerBenchmark()
      : serde_(std::make_unique<serializer::ArrowIpcVectorSerde>()) {}

  void iterativeSerialize(const RowTypePtr& rowType, BenchParams params) {
    folly::BenchmarkSuspender suspender;
    auto data = makeData(rowType, params);
    IndexRange range{0, params.numRows};
    folly::Range<const IndexRange*> ranges(&range, 1);
    auto opts = optionsFor(params);
    suspender.dismiss();

    Scratch scratch;
    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto serializer = serde_->createIterativeSerializer(
        rowType, params.numRows, arena.get(), &opts);
    serializer->append(data, ranges, scratch);

    IOBufOutputStream out(*pool_, nullptr);
    serializer->flush(&out);
    folly::doNotOptimizeAway(out.tellp());
  }

  void batchSerialize(const RowTypePtr& rowType, BenchParams params) {
    folly::BenchmarkSuspender suspender;
    auto data = makeData(rowType, params);
    IndexRange range{0, params.numRows};
    folly::Range<const IndexRange*> ranges(&range, 1);
    auto opts = optionsFor(params);
    auto serializer = serde_->createBatchSerializer(pool_.get(), &opts);
    suspender.dismiss();

    Scratch scratch;
    IOBufOutputStream out(*pool_, nullptr);
    serializer->serialize(data, ranges, scratch, &out);
    folly::doNotOptimizeAway(out.tellp());
  }

  void deserialize(const RowTypePtr& rowType, BenchParams params) {
    folly::BenchmarkSuspender suspender;
    auto serialized = serializeOnce(rowType, params);
    auto byteRanges = byteRangesFromIOBuf(serialized.get());
    suspender.dismiss();

    BufferInputStream input(byteRanges);
    RowVectorPtr result;
    serde_->deserialize(&input, pool_.get(), rowType, &result, nullptr);
    folly::doNotOptimizeAway(result);
  }

  void roundtrip(const RowTypePtr& rowType, BenchParams params) {
    folly::BenchmarkSuspender suspender;
    auto data = makeData(rowType, params);
    IndexRange range{0, params.numRows};
    folly::Range<const IndexRange*> ranges(&range, 1);
    auto opts = optionsFor(params);
    auto serializer = serde_->createBatchSerializer(pool_.get(), &opts);
    suspender.dismiss();

    Scratch scratch;
    IOBufOutputStream out(*pool_, nullptr);
    serializer->serialize(data, ranges, scratch, &out);
    auto iobuf = out.getIOBuf();

    auto byteRanges = byteRangesFromIOBuf(iobuf.get());
    BufferInputStream input(byteRanges);
    RowVectorPtr result;
    serde_->deserialize(&input, pool_.get(), rowType, &result, nullptr);
    folly::doNotOptimizeAway(result);
  }

  int64_t serializedBytes(const RowTypePtr& rowType, BenchParams params) {
    auto iobuf = serializeOnce(rowType, params);
    return iobuf->computeChainDataLength();
  }

 private:
  static serializer::ArrowIpcVectorSerde::ArrowIpcOptions optionsFor(
      const BenchParams& params) {
    serializer::ArrowIpcVectorSerde::ArrowIpcOptions opts;
    opts.compressionKind = params.compression;
    return opts;
  }

  std::unique_ptr<folly::IOBuf> serializeOnce(
      const RowTypePtr& rowType,
      BenchParams params) {
    auto data = makeData(rowType, params);
    IndexRange range{0, params.numRows};
    folly::Range<const IndexRange*> ranges(&range, 1);
    auto opts = optionsFor(params);
    auto serializer = serde_->createBatchSerializer(pool_.get(), &opts);

    Scratch scratch;
    IOBufOutputStream out(*pool_, nullptr);
    serializer->serialize(data, ranges, scratch, &out);
    return out.getIOBuf();
  }

  RowVectorPtr makeData(const RowTypePtr& rowType, const BenchParams& params) {
    VectorFuzzer::Options options;
    options.vectorSize = params.numRows;
    options.nullRatio = params.nullRatio;
    constexpr uint32_t kSeed = 1;
    VectorFuzzer fuzzer(options, pool_.get(), kSeed);
    return fuzzer.fuzzInputFlatRow(rowType);
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  std::unique_ptr<serializer::ArrowIpcVectorSerde> serde_;
};

#define ARROW_IPC_ITER_SERIALIZE(name, rowType, suffix, params)    \
  BENCHMARK_RELATIVE(arrow_ipc_iter_serialize_##name##_##suffix) { \
    ArrowIpcSerializerBenchmark benchmark;                         \
    benchmark.iterativeSerialize(rowType, params);                 \
  }
#define ARROW_IPC_BATCH_SERIALIZE(name, rowType, suffix, params)    \
  BENCHMARK_RELATIVE(arrow_ipc_batch_serialize_##name##_##suffix) { \
    ArrowIpcSerializerBenchmark benchmark;                          \
    benchmark.batchSerialize(rowType, params);                      \
  }
#define ARROW_IPC_DESERIALIZE(name, rowType, suffix, params)    \
  BENCHMARK_RELATIVE(arrow_ipc_deserialize_##name##_##suffix) { \
    ArrowIpcSerializerBenchmark benchmark;                      \
    benchmark.deserialize(rowType, params);                     \
  }
#define ARROW_IPC_ROUNDTRIP(name, rowType, suffix, params)    \
  BENCHMARK_RELATIVE(arrow_ipc_roundtrip_##name##_##suffix) { \
    ArrowIpcSerializerBenchmark benchmark;                    \
    benchmark.roundtrip(rowType, params);                     \
  }
#else
#define ARROW_IPC_ITER_SERIALIZE(name, rowType, suffix, params)
#define ARROW_IPC_BATCH_SERIALIZE(name, rowType, suffix, params)
#define ARROW_IPC_DESERIALIZE(name, rowType, suffix, params)
#define ARROW_IPC_ROUNDTRIP(name, rowType, suffix, params)
#endif // VELOX_ENABLE_ARROW_IPC_BENCHMARK

// Token-pasting can't include digit separators or punctuation, so the
// macro takes a short suffix (e.g. 1k, 100k_lz4, 1m_nulls) for naming and
// the BenchParams struct separately. Each Presto benchmark is followed
// immediately by the matching Arrow IPC benchmark (when enabled) so
// folly's relative-time column produces a direct apples-to-apples
// comparison.
#define PRESTO_SERIALIZER_BENCHMARKS(name, rowType, suffix, params) \
  BENCHMARK(presto_iter_serialize_##name##_##suffix) {              \
    PrestoSerializerBenchmark benchmark;                            \
    benchmark.iterativeSerialize(rowType, params);                  \
  }                                                                 \
  ARROW_IPC_ITER_SERIALIZE(name, rowType, suffix, params)           \
  BENCHMARK(presto_batch_serialize_##name##_##suffix) {             \
    PrestoSerializerBenchmark benchmark;                            \
    benchmark.batchSerialize(rowType, params);                      \
  }                                                                 \
  ARROW_IPC_BATCH_SERIALIZE(name, rowType, suffix, params)          \
  BENCHMARK(presto_deserialize_##name##_##suffix) {                 \
    PrestoSerializerBenchmark benchmark;                            \
    benchmark.deserialize(rowType, params);                         \
  }                                                                 \
  ARROW_IPC_DESERIALIZE(name, rowType, suffix, params)              \
  BENCHMARK(presto_roundtrip_##name##_##suffix) {                   \
    PrestoSerializerBenchmark benchmark;                            \
    benchmark.roundtrip(rowType, params);                           \
  }                                                                 \
  ARROW_IPC_ROUNDTRIP(name, rowType, suffix, params)                \
  BENCHMARK_DRAW_LINE();

// Default sweep across row counts at nullRatio=0, no compression. Mirrors
// the original baseline that landed alongside ArrowIpcVectorSerde.
#define PRESTO_SERIALIZER_ALL_SIZES(name, rowType)                          \
  PRESTO_SERIALIZER_BENCHMARKS(name, rowType, 1k, (BenchParams{1'000}))     \
  PRESTO_SERIALIZER_BENCHMARKS(name, rowType, 10k, (BenchParams{10'000}))   \
  PRESTO_SERIALIZER_BENCHMARKS(name, rowType, 100k, (BenchParams{100'000})) \
  PRESTO_SERIALIZER_BENCHMARKS(name, rowType, 1m, (BenchParams{1'000'000}))

// Exact shape from facebookincubator/velox#1732 (1M rows of int + float).
PRESTO_SERIALIZER_ALL_SIZES(intReal, ROW({INTEGER(), REAL()}))

PRESTO_SERIALIZER_ALL_SIZES(
    fixedWidth5,
    ROW({BIGINT(), DOUBLE(), BOOLEAN(), TINYINT(), REAL()}))

PRESTO_SERIALIZER_ALL_SIZES(strings, ROW({BIGINT(), VARCHAR()}))

PRESTO_SERIALIZER_ALL_SIZES(arrays, ROW({BIGINT(), ARRAY(BIGINT())}))

PRESTO_SERIALIZER_ALL_SIZES(maps, ROW({BIGINT(), MAP(BIGINT(), REAL())}))

PRESTO_SERIALIZER_ALL_SIZES(
    structs,
    ROW({BIGINT(), ROW({BIGINT(), DOUBLE(), BOOLEAN()})}))

// Wide row scenario: 30 columns, mix of fixed-width and strings, modeled
// on the kind of intermediate exchange row a TPC-DS-class query produces.
// Run only at 1M rows since the per-cell cost dominates anyway and adding
// other sizes adds little signal.
PRESTO_SERIALIZER_BENCHMARKS(
    wideRow30,
    ROW({BIGINT(),  INTEGER(), SMALLINT(), TINYINT(), REAL(),    DOUBLE(),
         BOOLEAN(), BIGINT(),  INTEGER(),  DOUBLE(),  VARCHAR(), VARCHAR(),
         VARCHAR(), BIGINT(),  INTEGER(),  REAL(),    DOUBLE(),  BIGINT(),
         INTEGER(), DOUBLE(),  VARCHAR(),  BIGINT(),  INTEGER(), REAL(),
         DOUBLE(),  BIGINT(),  INTEGER(),  DOUBLE(),  VARCHAR(), BIGINT()}),
    100k,
    (BenchParams{100'000}))

// Null-ratio sweep at 1M rows. Same six scenarios, but with VectorFuzzer
// generating ~25% nulls so we see the per-row null bookkeeping cost in
// both serdes.
PRESTO_SERIALIZER_BENCHMARKS(
    intReal,
    ROW({INTEGER(), REAL()}),
    1m_nulls,
    (BenchParams{1'000'000, 0.25}))
PRESTO_SERIALIZER_BENCHMARKS(
    fixedWidth5,
    ROW({BIGINT(), DOUBLE(), BOOLEAN(), TINYINT(), REAL()}),
    1m_nulls,
    (BenchParams{1'000'000, 0.25}))
PRESTO_SERIALIZER_BENCHMARKS(
    strings,
    ROW({BIGINT(), VARCHAR()}),
    1m_nulls,
    (BenchParams{1'000'000, 0.25}))
PRESTO_SERIALIZER_BENCHMARKS(
    structs,
    ROW({BIGINT(), ROW({BIGINT(), DOUBLE(), BOOLEAN()})}),
    1m_nulls,
    (BenchParams{1'000'000, 0.25}))

// Compression sweep at 1M rows. Run on a representative subset rather
// than all six scenarios so wall-clock stays bounded; intReal exercises
// fixed-width compression, strings exercises variable-length compression,
// and structs exercises nested compression.
PRESTO_SERIALIZER_BENCHMARKS(
    intReal,
    ROW({INTEGER(), REAL()}),
    1m_lz4,
    (BenchParams{1'000'000, 0.0, common::CompressionKind::CompressionKind_LZ4}))
PRESTO_SERIALIZER_BENCHMARKS(
    intReal,
    ROW({INTEGER(), REAL()}),
    1m_zstd,
    (BenchParams{
        1'000'000,
        0.0,
        common::CompressionKind::CompressionKind_ZSTD}))
PRESTO_SERIALIZER_BENCHMARKS(
    strings,
    ROW({BIGINT(), VARCHAR()}),
    1m_lz4,
    (BenchParams{1'000'000, 0.0, common::CompressionKind::CompressionKind_LZ4}))
PRESTO_SERIALIZER_BENCHMARKS(
    strings,
    ROW({BIGINT(), VARCHAR()}),
    1m_zstd,
    (BenchParams{
        1'000'000,
        0.0,
        common::CompressionKind::CompressionKind_ZSTD}))
PRESTO_SERIALIZER_BENCHMARKS(
    structs,
    ROW({BIGINT(), ROW({BIGINT(), DOUBLE(), BOOLEAN()})}),
    1m_lz4,
    (BenchParams{1'000'000, 0.0, common::CompressionKind::CompressionKind_LZ4}))
PRESTO_SERIALIZER_BENCHMARKS(
    structs,
    ROW({BIGINT(), ROW({BIGINT(), DOUBLE(), BOOLEAN()})}),
    1m_zstd,
    (BenchParams{
        1'000'000,
        0.0,
        common::CompressionKind::CompressionKind_ZSTD}))

struct Scenario {
  std::string name;
  RowTypePtr type;
};

void printSerializedBytesTable() {
  const std::vector<Scenario> scenarios{
      {"intReal", ROW({INTEGER(), REAL()})},
      {"fixedWidth5", ROW({BIGINT(), DOUBLE(), BOOLEAN(), TINYINT(), REAL()})},
      {"strings", ROW({BIGINT(), VARCHAR()})},
      {"arrays", ROW({BIGINT(), ARRAY(BIGINT())})},
      {"maps", ROW({BIGINT(), MAP(BIGINT(), REAL())})},
      {"structs", ROW({BIGINT(), ROW({BIGINT(), DOUBLE(), BOOLEAN()})})},
  };
  const std::vector<vector_size_t> sizes{1'000, 10'000, 100'000, 1'000'000};

  std::cout << "\nSerialized payload size (uncompressed)\n";
  std::cout << "--------------------------------------\n";
  std::cout << std::setw(14) << "scenario" << std::setw(10) << "rows"
            << std::setw(14) << "presto B" << std::setw(14) << "presto B/row";
#ifdef VELOX_ENABLE_ARROW_IPC_BENCHMARK
  std::cout << std::setw(14) << "arrow B" << std::setw(14) << "arrow B/row"
            << std::setw(10) << "ratio";
#endif
  std::cout << "\n";

  PrestoSerializerBenchmark prestoBench;
#ifdef VELOX_ENABLE_ARROW_IPC_BENCHMARK
  ArrowIpcSerializerBenchmark arrowBench;
#endif
  for (const auto& scenario : scenarios) {
    for (auto numRows : sizes) {
      const auto prestoBytes =
          prestoBench.serializedBytes(scenario.type, BenchParams{numRows});
      std::cout << std::setw(14) << scenario.name << std::setw(10) << numRows
                << std::setw(14) << prestoBytes << std::setw(14) << std::fixed
                << std::setprecision(2)
                << static_cast<double>(prestoBytes) / numRows;
#ifdef VELOX_ENABLE_ARROW_IPC_BENCHMARK
      const auto arrowBytes =
          arrowBench.serializedBytes(scenario.type, BenchParams{numRows});
      std::cout << std::setw(14) << arrowBytes << std::setw(14) << std::fixed
                << std::setprecision(2)
                << static_cast<double>(arrowBytes) / numRows << std::setw(10)
                << std::fixed << std::setprecision(2)
                << (static_cast<double>(arrowBytes) / prestoBytes);
#endif
      std::cout << "\n";
    }
  }
  std::cout << std::endl;

  // A compression-aware companion table at 1M rows so the benchmark
  // output carries enough information to compute compressed throughput.
  const std::vector<std::pair<std::string, common::CompressionKind>>
      compressions{
          {"none", common::CompressionKind::CompressionKind_NONE},
          {"lz4", common::CompressionKind::CompressionKind_LZ4},
          {"zstd", common::CompressionKind::CompressionKind_ZSTD},
      };
  const std::vector<Scenario> compressionScenarios{
      {"intReal", ROW({INTEGER(), REAL()})},
      {"strings", ROW({BIGINT(), VARCHAR()})},
      {"structs", ROW({BIGINT(), ROW({BIGINT(), DOUBLE(), BOOLEAN()})})},
  };
  std::cout << "\nSerialized payload size at 1M rows by compression\n";
  std::cout << "-------------------------------------------------\n";
  std::cout << std::setw(14) << "scenario" << std::setw(8) << "codec"
            << std::setw(14) << "presto B"
#ifdef VELOX_ENABLE_ARROW_IPC_BENCHMARK
            << std::setw(14) << "arrow B" << std::setw(10) << "ratio"
#endif
            << "\n";
  for (const auto& scenario : compressionScenarios) {
    for (const auto& [codecName, codec] : compressions) {
      BenchParams params{1'000'000, 0.0, codec};
      const auto prestoBytes =
          prestoBench.serializedBytes(scenario.type, params);
      std::cout << std::setw(14) << scenario.name << std::setw(8) << codecName
                << std::setw(14) << prestoBytes;
#ifdef VELOX_ENABLE_ARROW_IPC_BENCHMARK
      const auto arrowBytes = arrowBench.serializedBytes(scenario.type, params);
      std::cout << std::setw(14) << arrowBytes << std::setw(10) << std::fixed
                << std::setprecision(2)
                << (static_cast<double>(arrowBytes) / prestoBytes);
#endif
      std::cout << "\n";
    }
  }
  std::cout << std::endl;
}

} // namespace
} // namespace facebook::velox::test

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  folly::runBenchmarks();
  // Setting SKIP_BYTES_TABLE=1 turns the post-run companion table off so
  // CPU profilers only sample the actual benchmark, not the size summary.
  if (std::getenv("SKIP_BYTES_TABLE") == nullptr) {
    facebook::velox::test::printSerializedBytesTable();
  }
  return 0;
}
