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
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <gperftools/profiler.h>

#include "velox/common/memory/ByteStream.h"
#include "velox/common/memory/Memory.h"
#include "velox/serializers/ArrowIpcSerializer.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

// Standalone profiler for the same workloads exposed by
// PrestoSerializerBenchmark. Avoids folly's BenchmarkSuspender / clock-read
// instrumentation noise so gperftools sees the actual hot path. The
// program runs ONE workload N times inside ProfilerStart/ProfilerStop, so
// each invocation produces a clean cpu profile dominated by the work.

namespace facebook::velox {
namespace {

using common::CompressionKind;

struct WorkloadConfig {
  std::string name;
  RowTypePtr rowType;
  vector_size_t numRows;
  double nullRatio;
  CompressionKind compression;
  std::string serde; // "presto" or "arrow"
  std::string op; // "iter_ser", "batch_ser", "deserialize"
};

const std::vector<WorkloadConfig>& workloads() {
  static const std::vector<WorkloadConfig> kWorkloads = {
      // Workloads that this profiler can run. Names match what we used in
      // the brainstorm so the resulting profiles map back 1:1.
      {"arrow_iter_serialize_strings_1m_nulls",
       ROW({BIGINT(), VARCHAR()}),
       1'000'000,
       0.25,
       CompressionKind::CompressionKind_NONE,
       "arrow",
       "iter_ser"},
      {"arrow_batch_serialize_strings_1m",
       ROW({BIGINT(), VARCHAR()}),
       1'000'000,
       0.0,
       CompressionKind::CompressionKind_NONE,
       "arrow",
       "batch_ser"},
      {"arrow_batch_serialize_strings_1m_zstd",
       ROW({BIGINT(), VARCHAR()}),
       1'000'000,
       0.0,
       CompressionKind::CompressionKind_ZSTD,
       "arrow",
       "batch_ser"},
      {"arrow_batch_serialize_wideRow30_100k",
       ROW({BIGINT(),  INTEGER(), SMALLINT(), TINYINT(), REAL(),    DOUBLE(),
            BOOLEAN(), BIGINT(),  INTEGER(),  DOUBLE(),  VARCHAR(), VARCHAR(),
            VARCHAR(), BIGINT(),  INTEGER(),  REAL(),    DOUBLE(),  BIGINT(),
            INTEGER(), DOUBLE(),  VARCHAR(),  BIGINT(),  INTEGER(), REAL(),
            DOUBLE(),  BIGINT(),  INTEGER(),  DOUBLE(),  VARCHAR(), BIGINT()}),
       100'000,
       0.0,
       CompressionKind::CompressionKind_NONE,
       "arrow",
       "batch_ser"},
      {"arrow_deserialize_wideRow30_100k",
       ROW({BIGINT(),  INTEGER(), SMALLINT(), TINYINT(), REAL(),    DOUBLE(),
            BOOLEAN(), BIGINT(),  INTEGER(),  DOUBLE(),  VARCHAR(), VARCHAR(),
            VARCHAR(), BIGINT(),  INTEGER(),  REAL(),    DOUBLE(),  BIGINT(),
            INTEGER(), DOUBLE(),  VARCHAR(),  BIGINT(),  INTEGER(), REAL(),
            DOUBLE(),  BIGINT(),  INTEGER(),  DOUBLE(),  VARCHAR(), BIGINT()}),
       100'000,
       0.0,
       CompressionKind::CompressionKind_NONE,
       "arrow",
       "deserialize"},
      {"arrow_deserialize_intReal_1m_zstd",
       ROW({INTEGER(), REAL()}),
       1'000'000,
       0.0,
       CompressionKind::CompressionKind_ZSTD,
       "arrow",
       "deserialize"},
  };
  return kWorkloads;
}

class WorkloadRunner {
 public:
  explicit WorkloadRunner(const WorkloadConfig& cfg)
      : cfg_(cfg),
        pool_(memory::memoryManager()->addLeafPool()),
        prestoSerde_(std::make_unique<serializer::presto::PrestoVectorSerde>()),
        arrowSerde_(std::make_unique<serializer::ArrowIpcVectorSerde>()) {
    if (!isRegisteredVectorSerde()) {
      serializer::presto::PrestoVectorSerde::registerVectorSerde();
    }
    data_ = makeData();
    if (cfg_.op == "deserialize") {
      payload_ = serializeOnce();
    }
  }

  // One round of the workload. Must be called inside the timed region.
  void runOnce() {
    if (cfg_.op == "iter_ser") {
      doIterativeSerialize();
    } else if (cfg_.op == "batch_ser") {
      doBatchSerialize();
    } else if (cfg_.op == "deserialize") {
      doDeserialize();
    } else {
      VELOX_FAIL("Unknown op: {}", cfg_.op);
    }
  }

 private:
  RowVectorPtr makeData() {
    VectorFuzzer::Options options;
    options.vectorSize = cfg_.numRows;
    options.nullRatio = cfg_.nullRatio;
    constexpr uint32_t kSeed = 1;
    VectorFuzzer fuzzer(options, pool_.get(), kSeed);
    return fuzzer.fuzzInputFlatRow(cfg_.rowType);
  }

  std::unique_ptr<folly::IOBuf> serializeOnce() {
    IndexRange range{0, cfg_.numRows};
    folly::Range<const IndexRange*> ranges(&range, 1);
    auto opts = arrowOpts();
    auto serializer = arrowSerde_->createBatchSerializer(pool_.get(), &opts);
    Scratch scratch;
    IOBufOutputStream out(*pool_, nullptr);
    serializer->serialize(data_, ranges, scratch, &out);
    return out.getIOBuf();
  }

  serializer::presto::PrestoVectorSerde::PrestoOptions prestoOpts() const {
    serializer::presto::PrestoVectorSerde::PrestoOptions opts;
    opts.compressionKind = cfg_.compression;
    return opts;
  }

  serializer::ArrowIpcVectorSerde::ArrowIpcOptions arrowOpts() const {
    serializer::ArrowIpcVectorSerde::ArrowIpcOptions opts;
    opts.compressionKind = cfg_.compression;
    return opts;
  }

  VectorSerde* serde() const {
    return cfg_.serde == "arrow"
        ? static_cast<VectorSerde*>(arrowSerde_.get())
        : static_cast<VectorSerde*>(prestoSerde_.get());
  }

  VectorSerde::Options optsHandle() const {
    static thread_local serializer::ArrowIpcVectorSerde::ArrowIpcOptions
        arrowHandle;
    static thread_local serializer::presto::PrestoVectorSerde::PrestoOptions
        prestoHandle;
    if (cfg_.serde == "arrow") {
      arrowHandle = arrowOpts();
      return arrowHandle;
    }
    prestoHandle = prestoOpts();
    return prestoHandle;
  }

  void doIterativeSerialize() {
    IndexRange range{0, cfg_.numRows};
    folly::Range<const IndexRange*> ranges(&range, 1);
    Scratch scratch;
    auto arena = std::make_unique<StreamArena>(pool_.get());
    auto opts = optsHandle();
    auto serializer = serde()->createIterativeSerializer(
        cfg_.rowType, cfg_.numRows, arena.get(), &opts);
    serializer->append(data_, ranges, scratch);
    IOBufOutputStream out(*pool_, nullptr, serializer->maxSerializedSize());
    serializer->flush(&out);
    auto sink = out.tellp();
    if (sink == 0) {
      VELOX_FAIL("flush wrote 0 bytes");
    }
  }

  void doBatchSerialize() {
    IndexRange range{0, cfg_.numRows};
    folly::Range<const IndexRange*> ranges(&range, 1);
    Scratch scratch;
    auto opts = optsHandle();
    auto serializer = serde()->createBatchSerializer(pool_.get(), &opts);
    IOBufOutputStream out(*pool_, nullptr);
    serializer->serialize(data_, ranges, scratch, &out);
    auto sink = out.tellp();
    if (sink == 0) {
      VELOX_FAIL("serialize wrote 0 bytes");
    }
  }

  void doDeserialize() {
    auto byteRanges = byteRangesFromIOBuf(payload_.get());
    BufferInputStream input(byteRanges);
    RowVectorPtr result;
    auto opts = optsHandle();
    serde()->deserialize(&input, pool_.get(), cfg_.rowType, &result, &opts);
    if (result == nullptr || result->size() != cfg_.numRows) {
      VELOX_FAIL("deserialize did not reconstruct the expected vector");
    }
  }

  const WorkloadConfig cfg_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<serializer::presto::PrestoVectorSerde> prestoSerde_;
  std::unique_ptr<serializer::ArrowIpcVectorSerde> arrowSerde_;
  RowVectorPtr data_;
  std::unique_ptr<folly::IOBuf> payload_;
};

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --workload <name> [--iters N] [--profile-path /path/to/out.prof]\n\n"
      << "Available workloads:\n";
  for (const auto& cfg : workloads()) {
    std::cerr << "  " << cfg.name << "\n";
  }
}

int run(int argc, char** argv) {
  std::string name;
  int iters = 50;
  std::string profilePath;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--workload" && i + 1 < argc) {
      name = argv[++i];
    } else if (arg == "--iters" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (arg == "--profile-path" && i + 1 < argc) {
      profilePath = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      usage(argv[0]);
      return 1;
    }
  }
  if (name.empty()) {
    usage(argv[0]);
    return 1;
  }

  const WorkloadConfig* match = nullptr;
  for (const auto& cfg : workloads()) {
    if (cfg.name == name) {
      match = &cfg;
      break;
    }
  }
  if (match == nullptr) {
    std::cerr << "Unknown workload: " << name << "\n";
    usage(argv[0]);
    return 1;
  }

  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  WorkloadRunner runner(*match);

  // Warm up so first-call costs (JIT-style codec setup, lazy schema build,
  // memory pool growth) don't pollute the timed region.
  for (int i = 0; i < 3; ++i) {
    runner.runOnce();
  }

  if (!profilePath.empty()) {
    if (!ProfilerStart(profilePath.c_str())) {
      std::cerr << "ProfilerStart failed for path " << profilePath << "\n";
      return 1;
    }
  }
  for (int i = 0; i < iters; ++i) {
    runner.runOnce();
  }
  if (!profilePath.empty()) {
    ProfilerStop();
    std::cerr << "Wrote profile to " << profilePath << "\n";
  }
  return 0;
}

} // namespace
} // namespace facebook::velox

int main(int argc, char** argv) {
  return facebook::velox::run(argc, argv);
}
