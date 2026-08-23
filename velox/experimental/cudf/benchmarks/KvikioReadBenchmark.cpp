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

#include "velox/experimental/cudf/benchmarks/KvikioReadBenchmark.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/time/Timer.h"

#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

DEFINE_string(
    paths,
    "",
    "Path to a manifest file holding one object URI per line. Blank lines "
    "and lines starting with '#' are ignored. Credentials come from the "
    "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION and "
    "AWS_SESSION_TOKEN environment variables; set AWS_ENDPOINT_URL to target "
    "a non-AWS S3 server.");

DEFINE_bool(
    list_targets,
    false,
    "Open every target, print its size, and exit without reading payload.");

DEFINE_string(
    mode,
    "cold",
    "'cold' reads each byte at most once so nothing is served from a cache "
    "this run populated. 'warm' re-reads at random offsets.");

DEFINE_uint64(request_bytes, 8ULL << 20, "Size of each range request.");

DEFINE_uint64(
    measurement_bytes,
    1ULL << 30,
    "Total payload bytes to move in one run, summed across reader threads.");

DEFINE_uint64(seed, 0, "Seed for warm-mode offset selection.");

namespace facebook::velox::cudf_velox {

namespace {

// Owns one destination buffer, allocated on the host or the device.
class ReadBuffer {
 public:
  ReadBuffer(uint64_t bytes, bool device) : device_{device} {
    if (device_) {
      CUDF_CUDA_TRY(cudaMalloc(&data_, bytes));
    } else {
      data_ = std::malloc(bytes);
      VELOX_CHECK_NOT_NULL(
          data_, "Failed to allocate host read buffer of {} bytes", bytes);
    }
  }

  ~ReadBuffer() {
    if (data_ == nullptr) {
      return;
    }
    if (device_) {
      static_cast<void>(cudaFree(data_));
    } else {
      std::free(data_);
    }
  }

  ReadBuffer(const ReadBuffer&) = delete;
  ReadBuffer& operator=(const ReadBuffer&) = delete;

  void* data() const {
    return data_;
  }

 private:
  void* data_{nullptr};
  bool device_{false};
};

} // namespace

RemoteTargets::RemoteTargets(const std::vector<std::string>& uris) {
  infos_.reserve(uris.size());
  handles_.reserve(uris.size());
  for (const auto& uri : uris) {
    try {
      auto handle = kvikio::RemoteHandle::open(uri);
      infos_.push_back(TargetInfo{uri, handle.nbytes()});
      handles_.push_back(std::move(handle));
    } catch (const std::exception& e) {
      VELOX_USER_FAIL(
          "Failed to open remote target. URI: {}, error: {}", uri, e.what());
    }
  }
}

uint64_t RemoteTargets::totalBytes() const {
  uint64_t total{0};
  for (const auto& info : infos_) {
    total += info.size;
  }
  return total;
}

RunResult runPlan(
    RemoteTargets& targets,
    const std::vector<ReadTask>& plan,
    int32_t numThreads,
    uint64_t requestBytes,
    uint64_t kvikioTaskSize,
    bool deviceMemory) {
  VELOX_USER_CHECK_GT(numThreads, 0, "Reader thread count must be positive");

  std::atomic<size_t> nextTask{0};
  std::atomic<uint64_t> bytesRead{0};
  std::mutex errorMutex;
  std::exception_ptr firstError;

  RunResult result;
  {
    MicrosecondTimer timer(&result.elapsedMicros);
    std::vector<std::thread> readers;
    readers.reserve(numThreads);
    for (int32_t i = 0; i < numThreads; ++i) {
      readers.emplace_back([&]() {
        try {
          ReadBuffer buffer{requestBytes, deviceMemory};
          for (size_t index = nextTask.fetch_add(1); index < plan.size();
               index = nextTask.fetch_add(1)) {
            const auto& task = plan[index];
            VELOX_CHECK_LE(task.size, requestBytes);
            auto& handle = targets.handleAt(task.targetIndex);
            const size_t got = kvikioTaskSize == 0
                ? handle.read(buffer.data(), task.size, task.offset)
                : handle
                      .pread(
                          buffer.data(), task.size, task.offset, kvikioTaskSize)
                      .get();
            VELOX_CHECK_EQ(
                got,
                task.size,
                "Short read from {} at offset {}",
                targets.infos()[task.targetIndex].uri,
                task.offset);
            bytesRead.fetch_add(got);
          }
        } catch (...) {
          std::lock_guard<std::mutex> lock(errorMutex);
          if (!firstError) {
            firstError = std::current_exception();
          }
        }
      });
    }
    for (auto& reader : readers) {
      reader.join();
    }
  }
  if (firstError) {
    std::rethrow_exception(firstError);
  }

  result.bytesRead = bytesRead.load();
  result.numRequests = plan.size();
  return result;
}

std::vector<std::string> readManifestFile(const std::string& path) {
  std::ifstream in(path);
  VELOX_USER_CHECK(in.is_open(), "Cannot open manifest file: {}", path);
  auto uris = parseManifest(in);
  VELOX_USER_CHECK(!uris.empty(), "Manifest file holds no URIs: {}", path);
  return uris;
}

} // namespace facebook::velox::cudf_velox

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Measures read throughput from S3 through KvikIO's remote path. "
      "Run with --helpon=KvikioReadBenchmark for the full flag list.");
  folly::Init init{&argc, &argv, false};

  using namespace facebook::velox::cudf_velox;

  VELOX_USER_CHECK(!FLAGS_paths.empty(), "--paths is required");
  RemoteTargets targets{readManifestFile(FLAGS_paths)};

  if (FLAGS_list_targets) {
    for (const auto& info : targets.infos()) {
      std::cout << info.size << '\t' << info.uri << std::endl;
    }
    std::cout << "total_bytes=" << targets.totalBytes() << std::endl;
    return 0;
  }

  VELOX_USER_CHECK(
      FLAGS_mode == "cold" || FLAGS_mode == "warm",
      "--mode must be 'cold' or 'warm', got: {}",
      FLAGS_mode);
  const auto mode = FLAGS_mode == "cold" ? ReadMode::kCold : ReadMode::kWarm;

  const ReadPlanOptions options{
      .mode = mode,
      .requestBytes = FLAGS_request_bytes,
      .measurementBytes = FLAGS_measurement_bytes,
      .seed = FLAGS_seed,
  };
  const auto plan = makeReadPlan(targets.infos(), options);

  const auto result = runPlan(
      targets,
      plan,
      /*numThreads=*/1,
      FLAGS_request_bytes,
      /*kvikioTaskSize=*/0,
      /*deviceMemory=*/false);

  // Bytes per microsecond equals megabytes per second, matching the units
  // velox_read_benchmark prints. Guard against zero elapsed time so a
  // degenerate or empty-plan run does not print "inf MB/s".
  const double throughputMBs = result.elapsedMicros == 0
      ? 0.0
      : static_cast<double>(result.bytesRead) /
          static_cast<double>(result.elapsedMicros);
  std::cout << fmt::format(
                   "{:.1f} MB/s mode={} request={} threads={} "
                   "kvikio_task_size={} kvikio_nthreads={} device={} "
                   "bytes={} requests={} elapsed_s={:.3f}",
                   throughputMBs,
                   FLAGS_mode,
                   FLAGS_request_bytes,
                   1,
                   0,
                   0,
                   false,
                   result.bytesRead,
                   result.numRequests,
                   static_cast<double>(result.elapsedMicros) / 1'000'000.0)
            << std::endl;
  return 0;
}
