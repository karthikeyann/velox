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

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/utils.hpp>

#include <cuda_runtime.h>

#include <fmt/format.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <latch>
#include <memory>
#include <mutex>
#include <thread>

DEFINE_string(
    paths,
    "",
    "Path to a manifest file holding one object URI per line. Blank lines "
    "and lines starting with '#' are ignored. Credentials come from the "
    "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION and "
    "AWS_SESSION_TOKEN environment variables and from nowhere else, not from "
    "the EC2 instance metadata service; set AWS_ENDPOINT_URL to target a "
    "non-AWS S3 server.");

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

DEFINE_int32(
    reader_threads,
    8,
    "Number of reader threads, each issuing its own range requests. This is "
    "concurrency external to KvikIO and is independent of "
    "--kvikio_task_size.");

DEFINE_uint64(
    kvikio_task_size,
    0,
    "Zero issues each request as one range GET. Any other value hands the "
    "request to KvikIO's thread pool, split at this granularity.");

DEFINE_int32(
    kvikio_nthreads,
    0,
    "Width of KvikIO's internal thread pool, which serves --kvikio_task_size "
    "splits. Zero leaves the KvikIO default in place, and that default is one "
    "thread unless KVIKIO_NTHREADS says otherwise.");

DEFINE_uint64(
    kvikio_bounce_buffer_bytes,
    0,
    "Size of the bounce buffer KvikIO uses for device destinations. Zero "
    "leaves the KvikIO default in place.");

DEFINE_bool(
    device_memory,
    false,
    "Read into device rather than host memory. A remote source has no GDS "
    "path, so this routes through KvikIO's bounce buffer and a "
    "host-to-device copy.");

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

// Warns when KvikIO's thread pool is narrower than the reader concurrency it
// has to serve. The combination is legal but almost never intended: every
// reader queues behind the same worker, so the run measures one serialized
// stream while still reporting the requested thread count.
void warnIfKvikioPoolSerializesReaders(
    int32_t numThreads,
    uint64_t kvikioTaskSize) {
  if (kvikioTaskSize == 0 || numThreads <= 1 ||
      kvikio::defaults::thread_pool_nthreads() > 1) {
    return;
  }
  std::cerr << "Warning: --kvikio_task_size is non-zero and KvikIO's thread "
               "pool is one thread wide, so all "
            << numThreads
            << " reader threads serialize through a single worker and this "
               "run measures a single stream. Pass --kvikio_nthreads to widen "
               "the pool."
            << std::endl;
}

// Warns when the plan holds too few requests to keep every reader busy past
// the ramp, in which case the run reports request latency rather than the
// sustained throughput its thread count implies.
void warnIfPlanTooShallow(size_t planSize, int32_t numThreads) {
  if (planSize >= 4 * static_cast<size_t>(numThreads)) {
    return;
  }
  std::cerr << "Warning: the plan holds " << planSize << " requests for "
            << numThreads
            << " reader threads, fewer than four each, so this run reports "
               "request latency and ramp rather than sustained throughput. "
               "Raise --measurement_bytes or lower --request_bytes."
            << std::endl;
}

} // namespace

RemoteTargets::RemoteTargets(const std::vector<std::string>& uris) {
  infos_.reserve(uris.size());
  handles_.reserve(uris.size());
  std::cerr << "Opening " << uris.size() << " remote targets..." << std::endl;
  for (const auto& uri : uris) {
    try {
      auto handle = kvikio::RemoteHandle::open(uri);
      infos_.push_back(TargetInfo{uri, handle.nbytes()});
      handles_.push_back(std::move(handle));
    } catch (const std::exception& e) {
      VELOX_USER_FAIL(
          "Failed to open remote target. KvikIO reads credentials only from "
          "the AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and "
          "AWS_DEFAULT_REGION environment variables, plus AWS_SESSION_TOKEN "
          "for temporary keys; it never queries the EC2 instance metadata "
          "service, so an instance profile alone leaves it unauthenticated. "
          "Missing credentials surface here as an unsupported-protocol or "
          "HEAD failure rather than as an authentication error, because "
          "KvikIO falls back to its public-bucket endpoint. URI: {}, error: "
          "{}",
          uri,
          e.what());
    }
  }
  std::cerr << "Opened " << infos_.size() << " remote targets holding "
            << totalBytes() << " bytes." << std::endl;
}

RunResult runPlan(
    RemoteTargets& targets,
    const std::vector<ReadTask>& plan,
    int32_t numThreads,
    uint64_t requestBytes,
    uint64_t kvikioTaskSize,
    bool deviceMemory) {
  VELOX_USER_CHECK_GT(numThreads, 0, "Reader thread count must be positive");
  warnIfKvikioPoolSerializesReaders(numThreads, kvikioTaskSize);
  warnIfPlanTooShallow(plan.size(), numThreads);

  // Allocate and first-touch every destination up front. In device mode the
  // first cudaMalloc creates the CUDA primary context, and a host buffer takes
  // a page fault per page on first write; both are startup costs rather than
  // transfer costs, and the former lands only on device mode, biasing the very
  // comparison --device_memory exists to make.
  std::vector<std::unique_ptr<ReadBuffer>> buffers;
  buffers.reserve(numThreads);
  for (int32_t i = 0; i < numThreads; ++i) {
    buffers.push_back(std::make_unique<ReadBuffer>(requestBytes, deviceMemory));
    if (!deviceMemory) {
      std::memset(buffers.back()->data(), 0, requestBytes);
    }
  }

  // Force KvikIO's one-time dlopen of libcuda now. Every read begins by asking
  // whether its destination is host memory, and that query is what loads the
  // driver, so leaving it to the readers charges the load to the measured
  // window.
  static_cast<void>(kvikio::is_host_memory(buffers.front()->data()));

  // Pre-warm KvikIO's pinned bounce-buffer pool. A remote source has no GDS
  // path, so every device transfer stages through a pinned host buffer that
  // the pool allocates lazily, holding one global mutex across the allocation,
  // before the transfer starts. Left to the readers that cost lands inside the
  // measured window and only on device mode, and releasing every reader at
  // once maximizes the contention. Acquiring the buffers here and letting them
  // destruct returns them to the pool's free stack, where the readers pop them
  // for free.
  if (deviceMemory) {
    // With a non-zero task size the concurrent transfers are KvikIO's pool
    // workers rather than the reader threads, so warm enough for either.
    const size_t warmCount =
        std::max<size_t>(numThreads, kvikio::defaults::thread_pool_nthreads());
    std::vector<kvikio::CudaPinnedBounceBufferPool::Buffer> warmup;
    warmup.reserve(warmCount);
    for (size_t i = 0; i < warmCount; ++i) {
      warmup.push_back(kvikio::CudaPinnedBounceBufferPool::instance().get());
    }
  }

  std::atomic<size_t> nextTask{0};
  std::atomic<uint64_t> bytesRead{0};
  std::mutex errorMutex;
  std::exception_ptr firstError;

  // Readers park on 'startGate' until the clock is running, so thread creation
  // stays outside the measured window and every reader starts together rather
  // than in creation order. 'finishGate' ends the window as the last reader
  // stops, leaving thread teardown outside it too. 'aborted' drains the plan
  // without work, both when a later thread fails to start and when a reader
  // throws.
  std::latch startGate{1};
  std::latch finishGate{numThreads};
  std::atomic<bool> aborted{false};

  std::vector<std::thread> readers;
  readers.reserve(numThreads);
  try {
    for (int32_t i = 0; i < numThreads; ++i) {
      readers.emplace_back([&, buffer = buffers[i]->data()]() {
        try {
          startGate.wait();
          while (!aborted.load()) {
            const size_t index = nextTask.fetch_add(1);
            if (index >= plan.size()) {
              break;
            }
            const auto& task = plan[index];
            VELOX_CHECK_LE(task.size, requestBytes);
            auto& handle = targets.handleAt(task.targetIndex);
            const size_t got = kvikioTaskSize == 0
                ? handle.read(buffer, task.size, task.offset)
                : handle.pread(buffer, task.size, task.offset, kvikioTaskSize)
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
          // Stop the other readers as well. Against an endpoint that times out
          // rather than refuses, each task still in the plan would otherwise
          // burn the full KVIKIO_HTTP_TIMEOUT budget once per attempt, so a
          // deep plan takes hours to surface the failure already in hand.
          aborted.store(true);
          std::lock_guard<std::mutex> lock(errorMutex);
          if (!firstError) {
            firstError = std::current_exception();
          }
        }
        // Counted down on every path, including failure, so the caller's wait
        // always completes.
        finishGate.count_down();
      });
    }
  } catch (const std::exception& e) {
    // Release every thread that did start, without work since the run is
    // already abandoned, so none is destructed while joinable. Nothing waits
    // on 'finishGate' here, so the readers that were never created cannot
    // stall the join.
    aborted.store(true);
    startGate.count_down();
    for (auto& reader : readers) {
      reader.join();
    }
    VELOX_FAIL("Failed to create reader thread: {}", e.what());
  }

  RunResult result;
  {
    MicrosecondTimer timer(&result.elapsedMicros);
    startGate.count_down();
    finishGate.wait();
  }
  for (auto& reader : readers) {
    reader.join();
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

  try {
    using namespace facebook::velox::cudf_velox;

    if (FLAGS_kvikio_nthreads > 0) {
      kvikio::defaults::set_thread_pool_nthreads(
          static_cast<unsigned int>(FLAGS_kvikio_nthreads));
    }
    if (FLAGS_kvikio_bounce_buffer_bytes > 0) {
      kvikio::defaults::set_bounce_buffer_size(
          FLAGS_kvikio_bounce_buffer_bytes);
    }

    VELOX_USER_CHECK(!FLAGS_paths.empty(), "--paths is required");
    // Reject a bad mode before opening anything, so a typo costs no HEAD
    // requests and --list_targets cannot accept one silently.
    VELOX_USER_CHECK(
        FLAGS_mode == "cold" || FLAGS_mode == "warm",
        "--mode must be 'cold' or 'warm', got: {}",
        FLAGS_mode);

    RemoteTargets targets{readManifestFile(FLAGS_paths)};

    if (FLAGS_list_targets) {
      for (const auto& info : targets.infos()) {
        std::cout << info.size << '\t' << info.uri << std::endl;
      }
      std::cout << "total_bytes=" << targets.totalBytes() << std::endl;
      return 0;
    }

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
        FLAGS_reader_threads,
        FLAGS_request_bytes,
        FLAGS_kvikio_task_size,
        FLAGS_device_memory);

    // Reporting a throughput for an unmeasurably short run would be a
    // confident wrong answer, so refuse rather than divide by zero.
    VELOX_CHECK_GT(
        result.elapsedMicros,
        0,
        "Run finished in under a microsecond, so it has no throughput to "
        "report. Raise --measurement_bytes. Bytes read: {}",
        result.bytesRead);

    // Bytes per microsecond equals megabytes per second, matching the units
    // velox_read_benchmark prints.
    std::cout << fmt::format(
                     "{:.1f} MB/s mode={} request={} threads={} "
                     "kvikio_task_size={} kvikio_nthreads={} device={} "
                     "bytes={} requests={} elapsed_s={:.3f}",
                     static_cast<double>(result.bytesRead) /
                         static_cast<double>(result.elapsedMicros),
                     FLAGS_mode,
                     FLAGS_request_bytes,
                     FLAGS_reader_threads,
                     FLAGS_kvikio_task_size,
                     kvikio::defaults::thread_pool_nthreads(),
                     FLAGS_device_memory,
                     result.bytesRead,
                     result.numRequests,
                     static_cast<double>(result.elapsedMicros) / 1'000'000.0)
              << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "velox_cudf_kvikio_read_benchmark: " << e.what() << "\n";
    return 1;
  }
}
