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
#include "velox/experimental/cudf/exec/GpuMemoryTracker.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Task.h"

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCounters.h>
#include <nvtx3/nvToolsExtPayload.h>
#include <nvtx3/nvToolsExtSemanticsCounters.h>
#include <nvtx3/nvToolsExtSemanticsTime.h>

#include <fmt/format.h>
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

constexpr int64_t kFallbackTrackingCapacity{1LL << 40};
constexpr std::string_view kQueryRegistryKey{"cudfGpuMemoryTracking"};

int64_t calculateTrackingCapacity(int32_t percent) noexcept {
  size_t freeBytes{0};
  size_t totalBytes{0};
  if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
    return kFallbackTrackingCapacity;
  }
  percent = std::clamp(percent, int32_t{1}, int32_t{100});
  const auto capacity =
      freeBytes * static_cast<uint64_t>(percent) / uint64_t{100};
  return static_cast<int64_t>(std::max<uint64_t>(capacity, 1));
}

std::atomic<int64_t> configuredTrackingCapacity{0};

int64_t trackingCapacity() noexcept {
  const auto configured =
      configuredTrackingCapacity.load(std::memory_order_acquire);
  return configured > 0
      ? configured
      : calculateTrackingCapacity(CudfConfig::getInstance().memoryPercent);
}

uint64_t asUnsigned(int64_t bytes) {
  return bytes <= 0 ? 0 : static_cast<uint64_t>(bytes);
}

std::string queryKey(const GpuMemoryOwner& owner) {
  return fmt::format("query=[{}] taskUuid=[{}]", owner.queryId, owner.taskUuid);
}

std::string ownerKey(const GpuMemoryOwner& owner) {
  return fmt::format(
      "{}\x1f{}\x1f{}\x1f{}\x1f{}\x1f{}\x1f{}\x1f{}",
      owner.queryId,
      owner.taskId,
      owner.taskUuid,
      owner.planNodeId,
      owner.pipelineId,
      owner.driverId,
      owner.operatorId,
      owner.operatorType);
}

std::string displayField(std::string_view value) {
  if (value.empty()) {
    return "<none>";
  }
  std::string result;
  result.reserve(std::min<size_t>(value.size(), 96));
  for (const char character : value) {
    if (result.size() == 96) {
      result += "...";
      break;
    }
    result.push_back(
        character == '\n' || character == '\r' || character == '\t'
            ? ' '
            : character);
  }
  return result;
}

std::string ownerIdentity(const GpuMemoryOwner& owner) {
  return fmt::format(
      "query=[{}] task=[{}] taskUuid=[{}] plan=[{}] pipeline={} driver={} "
      "operator={} operatorType=[{}]",
      displayField(owner.queryId),
      displayField(owner.taskId),
      displayField(owner.taskUuid),
      displayField(owner.planNodeId),
      owner.pipelineId,
      owner.driverId,
      owner.operatorId,
      displayField(owner.operatorType));
}

// NVTX counter metadata is intentionally flat: one root scope, one global
// counter, one counter per query and one per operator. Stable identity is in
// each counter name, and parquet export can recover its scope through
// NVTX_COUNTER_GROUPS. This avoids the six-counters-per-operator hierarchy and
// registration-marker replay used by the original PR.

int64_t nowNanos() {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC_RAW, &now);
  return now.tv_sec * 1'000'000'000LL + now.tv_nsec;
}

int64_t nextSampleTimestamp() {
  static std::atomic<int64_t> lastTimestamp{0};
  auto timestamp = nowNanos();
  auto observed = lastTimestamp.load(std::memory_order_relaxed);
  do {
    timestamp = std::max(timestamp, observed + 1);
  } while (!lastTimestamp.compare_exchange_weak(
      observed,
      timestamp,
      std::memory_order_relaxed,
      std::memory_order_relaxed));
  return timestamp;
}

const nvtxSemanticsTime_t& timeSemantics() {
  static const auto semantics = [] {
    nvtxSemanticsTime_t result{};
    result.header.structSize = sizeof(nvtxSemanticsTime_t);
    result.header.semanticId = NVTX_SEMANTIC_ID_TIME_V1;
    result.header.version = NVTX_TIME_SEMANTIC_VERSION;
    result.header.next = nullptr;
    result.timeDomainId = NVTX_TIMESTAMP_TYPE_CPU_CLOCK_GETTIME_MONOTONIC_RAW;
    return result;
  }();
  return semantics;
}

const nvtxSemanticsCounter_t& byteCounterSemantics() {
  static const auto semantics = [] {
    nvtxSemanticsCounter_t result{};
    result.header.structSize = sizeof(nvtxSemanticsCounter_t);
    result.header.semanticId = NVTX_SEMANTIC_ID_COUNTERS_V1;
    result.header.version = NVTX_COUNTER_SEMANTIC_VERSION;
    result.header.next = &timeSemantics().header;
    result.flags = NVTX_COUNTER_FLAG_LIMIT_MIN |
        NVTX_COUNTER_FLAG_VALUETYPE_ABSOLUTE |
        NVTX_COUNTER_FLAG_INTERPOLATION_UNTIL_NEXT;
    result.unit = "bytes";
    result.unitScaleNumerator = 1;
    result.unitScaleDenominator = 1;
    result.limitType = NVTX_COUNTER_LIMIT_I64;
    result.min.i64 = 0;
    return result;
  }();
  return semantics;
}

uint64_t registerScope(nvtxDomainHandle_t domain, std::string_view name) {
  const std::string ownedName{name};
  nvtxScopeAttr_t attributes{};
  attributes.structSize = sizeof(nvtxScopeAttr_t);
  attributes.path = ownedName.c_str();
  attributes.parentScope = NVTX_SCOPE_ROOT;
  attributes.scopeId = NVTX_SCOPE_NONE;
  return nvtxScopeRegister(domain, &attributes);
}

uint64_t registerByteCounter(
    nvtxDomainHandle_t domain,
    std::string_view name,
    uint64_t scopeId) {
  const std::string ownedName{name};
  nvtxCounterAttr_t attributes{};
  attributes.structSize = sizeof(nvtxCounterAttr_t);
  attributes.schemaId = NVTX_PAYLOAD_ENTRY_TYPE_INT64;
  attributes.name = ownedName.c_str();
  attributes.description =
      "Logical requested GPU bytes attributed by Velox-cuDF.";
  attributes.scopeId = scopeId;
  attributes.semantics = &byteCounterSemantics().header;
  attributes.counterId = NVTX_COUNTER_ID_NONE;
  return nvtxCounterRegister(domain, &attributes);
}

int64_t clampToInt64(uint64_t bytes) {
  return static_cast<int64_t>(std::min(
      bytes, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
}

struct BufferedSample {
  nvtxDomainHandle_t domain;
  uint64_t counterId;
  int64_t value;
  int64_t timestamp;
};

struct CounterSeries {
  std::vector<int64_t> values;
  std::vector<int64_t> timestamps;
};

struct SubmissionState {
  std::mutex mutex;
  std::unordered_map<nvtxDomainHandle_t, std::unordered_map<uint64_t, int64_t>>
      lastTimestamps;
};

SubmissionState& submissionState() {
  static auto* state = new SubmissionState;
  return *state;
}

void submitSamples(std::vector<BufferedSample> samples) {
  std::sort(
      samples.begin(), samples.end(), [](const auto& left, const auto& right) {
        return std::tie(left.timestamp, left.counterId) <
            std::tie(right.timestamp, right.counterId);
      });
  std::unordered_map<
      nvtxDomainHandle_t,
      std::unordered_map<uint64_t, CounterSeries>>
      grouped;
  for (const auto& sample : samples) {
    auto& series = grouped[sample.domain][sample.counterId];
    auto& lastTimestamp =
        submissionState().lastTimestamps[sample.domain][sample.counterId];
    const auto timestamp = lastTimestamp == 0
        ? sample.timestamp
        : std::max(sample.timestamp, lastTimestamp + 1);
    series.values.push_back(sample.value);
    series.timestamps.push_back(timestamp);
    lastTimestamp = timestamp;
  }
  for (const auto& [domain, counters] : grouped) {
    for (const auto& [counterId, series] : counters) {
      nvtxCounterBatch_t batch{};
      batch.counterId = counterId;
      batch.counters = series.values.data();
      batch.countersSize = series.values.size() * sizeof(int64_t);
      batch.flags = NVTX_BATCH_FLAG_TIME_SORTED;
      batch.timestamps = series.timestamps.data();
      batch.timestampsSize = series.timestamps.size() * sizeof(int64_t);
      nvtxCounterBatchSubmit(domain, &batch);
    }
  }
}

class ThreadSampleBuffer;

// Serialize accounting transitions with profile submission. Otherwise a
// flusher can drain a later transition from one thread while an earlier
// transition remains in another thread's buffer.
std::mutex trackingUpdateMutex;

struct BufferRegistry {
  std::mutex mutex;
  std::unordered_set<ThreadSampleBuffer*> buffers;
  std::vector<BufferedSample> retiredSamples;
};

BufferRegistry& bufferRegistry() {
  static auto* registry = new BufferRegistry;
  return *registry;
}

void flushAllThreadBuffers();

void startSampleFlusher() {
  static const bool started = [] {
    std::thread([] {
      while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        flushAllThreadBuffers();
      }
    }).detach();
    return true;
  }();
  static_cast<void>(started);
}

class ThreadSampleBuffer {
 public:
  ThreadSampleBuffer() {
    auto& registry = bufferRegistry();
    {
      std::lock_guard<std::mutex> lock(registry.mutex);
      registry.buffers.insert(this);
    }
    startSampleFlusher();
  }

  ~ThreadSampleBuffer() {
    auto& registry = bufferRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.buffers.erase(this);
    auto samples = drain();
    registry.retiredSamples.insert(
        registry.retiredSamples.end(), samples.begin(), samples.end());
  }

  void append(std::initializer_list<BufferedSample> samples) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& sample : samples) {
      if (sample.counterId != NVTX_COUNTER_ID_NONE) {
        samples_.push_back(sample);
      }
    }
  }

  std::vector<BufferedSample> drain() {
    std::vector<BufferedSample> samples;
    std::lock_guard<std::mutex> lock(mutex_);
    samples.swap(samples_);
    return samples;
  }

 private:
  std::mutex mutex_;
  std::vector<BufferedSample> samples_;
};

ThreadSampleBuffer& threadSampleBuffer() {
  thread_local ThreadSampleBuffer buffer;
  return buffer;
}

void flushAllThreadBuffersLocked() {
  auto& registry = bufferRegistry();
  std::vector<BufferedSample> samples;
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    samples.swap(registry.retiredSamples);
    for (auto* buffer : registry.buffers) {
      auto buffered = buffer->drain();
      samples.insert(samples.end(), buffered.begin(), buffered.end());
    }
  }
  submitSamples(std::move(samples));
}

void flushAllThreadBuffers() {
  std::lock_guard<std::mutex> updateLock(trackingUpdateMutex);
  auto& submission = submissionState();
  std::lock_guard<std::mutex> submissionLock(submission.mutex);
  flushAllThreadBuffersLocked();
}

void sampleCounter(
    nvtxDomainHandle_t domain,
    uint64_t counterId,
    uint64_t bytes,
    int64_t timestamp) {
  if (counterId == NVTX_COUNTER_ID_NONE) {
    return;
  }
  threadSampleBuffer().append(
      {{domain, counterId, clampToInt64(bytes), timestamp}});
}

void emitMark(nvtxDomainHandle_t domain, const std::string& message) {
  nvtxEventAttributes_t attributes{};
  attributes.version = NVTX_VERSION;
  attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attributes.message.ascii = message.c_str();
  nvtxDomainMarkEx(domain, &attributes);
}

struct NvtxCounters {
  nvtxDomainHandle_t domain{nullptr};
  uint64_t epoch{0};
  uint64_t global{NVTX_COUNTER_ID_NONE};
  uint64_t query{NVTX_COUNTER_ID_NONE};
  uint64_t owner{NVTX_COUNTER_ID_NONE};
};

struct NvtxState {
  std::mutex mutex;
  bool initialized{false};
  nvtxDomainHandle_t domain{nullptr};
  uint64_t rootScope{NVTX_SCOPE_NONE};
  uint64_t globalCounter{NVTX_COUNTER_ID_NONE};
  std::unordered_map<std::string, uint64_t> queryCounters;
  std::unordered_map<std::string, uint64_t> ownerCounters;
};

NvtxState& nvtxState() {
  static auto* state = new NvtxState;
  return *state;
}

std::atomic<uint64_t> nvtxEpoch{1};
std::atomic<uint64_t> nvtxSessionSequence{0};

void initializeNvtxLocked(NvtxState& state) {
  if (state.initialized) {
    return;
  }
  const auto session =
      nvtxSessionSequence.fetch_add(1, std::memory_order_relaxed);
  state.domain = nvtxDomainCreateA(
      fmt::format("velox-cudf-gpu-memory-attribution/session-{}", session)
          .c_str());
  state.rootScope = registerScope(state.domain, "Velox GPU memory");
  state.globalCounter = registerByteCounter(
      state.domain, "overall logical live bytes", state.rootScope);
  state.initialized = true;
  sampleCounter(state.domain, state.globalCounter, 0, nextSampleTimestamp());
}

NvtxCounters registerNvtxOwner(
    const GpuMemoryOwner& owner,
    std::string_view logicalQueryId,
    std::string_view queryInstance) noexcept {
  try {
    std::lock_guard<std::mutex> updateLock(trackingUpdateMutex);
    auto& state = nvtxState();
    std::lock_guard<std::mutex> lock(state.mutex);
    initializeNvtxLocked(state);

    const auto query = fmt::format("{}\x1f{}", logicalQueryId, queryInstance);
    auto [queryIt, queryInserted] = state.queryCounters.try_emplace(query, 0);
    if (queryInserted) {
      queryIt->second = registerByteCounter(
          state.domain,
          fmt::format(
              "query [{}] instance={} logical live bytes",
              displayField(logicalQueryId),
              queryInstance),
          state.rootScope);
      sampleCounter(state.domain, queryIt->second, 0, nextSampleTimestamp());
    }

    const auto key = query + '\x1f' + ownerKey(owner);
    auto [ownerIt, ownerInserted] = state.ownerCounters.try_emplace(key, 0);
    if (ownerInserted) {
      ownerIt->second = registerByteCounter(
          state.domain,
          fmt::format(
              "operator logical live bytes queryInstance={} {}",
              queryInstance,
              ownerIdentity(owner)),
          state.rootScope);
      sampleCounter(state.domain, ownerIt->second, 0, nextSampleTimestamp());
    }

    return {
        state.domain,
        nvtxEpoch.load(std::memory_order_relaxed),
        state.globalCounter,
        queryIt->second,
        ownerIt->second};
  } catch (...) {
    return {};
  }
}

void sampleNvtx(
    const NvtxCounters& counters,
    uint64_t globalBytes,
    uint64_t queryBytes,
    uint64_t ownerBytes) noexcept {
  if (counters.epoch != nvtxEpoch.load(std::memory_order_acquire)) {
    return;
  }
  try {
    const auto timestamp = nextSampleTimestamp();
    threadSampleBuffer().append({
        {counters.domain,
         counters.global,
         clampToInt64(globalBytes),
         timestamp},
        {counters.domain, counters.query, clampToInt64(queryBytes), timestamp},
        {counters.domain, counters.owner, clampToInt64(ownerBytes), timestamp},
    });
  } catch (...) {
  }
}

void markAllocationFailure(
    nvtxDomainHandle_t domain,
    const GpuMemoryOwner& owner,
    size_t requestedBytes,
    uint64_t globalBytes,
    uint64_t queryBytes,
    uint64_t ownerBytes) noexcept {
  try {
    size_t freeBytes{0};
    size_t totalBytes{0};
    const auto status = cudaMemGetInfo(&freeBytes, &totalBytes);
    const std::string_view statusName{cudaGetErrorName(status)};
    const auto message = fmt::format(
        "velox-gpu-memory-allocation-failure requestedBytes={} "
        "globalCurrentBytes={} queryCurrentBytes={} ownerCurrentBytes={} "
        "cudaFreeBytes={} cudaTotalBytes={} cudaStatus=[{}] {}",
        requestedBytes,
        globalBytes,
        queryBytes,
        ownerBytes,
        freeBytes,
        totalBytes,
        statusName,
        ownerIdentity(owner));
    emitMark(domain, message);
    LOG(ERROR) << message;
  } catch (...) {
  }
}

bool resetNvtx() noexcept {
  try {
    auto& state = nvtxState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.initialized) {
      return true;
    }
    nvtxEpoch.fetch_add(1, std::memory_order_release);
    auto& submission = submissionState();
    std::lock_guard<std::mutex> submissionLock(submission.mutex);
    flushAllThreadBuffersLocked();
    const auto timestamp = nextSampleTimestamp();
    for (const auto& [_, counter] : state.ownerCounters) {
      sampleCounter(state.domain, counter, 0, timestamp);
    }
    for (const auto& [_, counter] : state.queryCounters) {
      sampleCounter(state.domain, counter, 0, timestamp);
    }
    sampleCounter(state.domain, state.globalCounter, 0, timestamp);
    flushAllThreadBuffersLocked();
    submission.lastTimestamps.erase(state.domain);
    if (state.domain != nullptr) {
      nvtxDomainDestroy(state.domain);
    }
    state.ownerCounters.clear();
    state.queryCounters.clear();
    state.domain = nullptr;
    state.rootScope = NVTX_SCOPE_NONE;
    state.globalCounter = NVTX_COUNTER_ID_NONE;
    state.initialized = false;
    return true;
  } catch (...) {
    return false;
  }
}

class TrackingSession;

// Profiling is opt-in and bounded. Serialize logical accounting transitions so
// every emitted global/query/owner sample describes the same state even when
// multiple driver threads allocate concurrently.
std::shared_mutex trackingLifecycleMutex;

/// Per-owner shared resource adapted from the first commit on Devavret's
/// memorypool-cudf-tracking branch. Unlike that commit, selection happens from
/// Velox's active operator scope, so connector allocations made by TableScan
/// are covered without importing GPU arbitration or changing every operator.
class TrackingResourceImpl {
 public:
  TrackingResourceImpl(
      GpuMemoryResource upstream,
      std::shared_ptr<void> sessionLease,
      std::shared_ptr<memory::MemoryPool> rootPool,
      std::shared_ptr<memory::MemoryPool> queryPool,
      std::shared_ptr<memory::MemoryPool> ownerPool,
      std::shared_ptr<memory::CustomMemoryResource> resourceOwner,
      GpuMemoryOwner owner,
      NvtxCounters counters)
      : sessionLease_(std::move(sessionLease)),
        resourceOwner_(std::move(resourceOwner)),
        rootPool_(std::move(rootPool)),
        queryPool_(std::move(queryPool)),
        ownerPool_(std::move(ownerPool)),
        upstream_(std::move(upstream)),
        owner_(std::move(owner)),
        counters_(counters) {}

  void* allocate_sync(size_t bytes, size_t alignment) {
    return allocateImpl(bytes, [this, bytes, alignment] {
      return upstream_.allocate_sync(bytes, alignment);
    });
  }

  void deallocate_sync(void* pointer, size_t bytes, size_t alignment) noexcept {
    upstream_.deallocate_sync(pointer, bytes, alignment);
    recordFree(bytes);
  }

  void* allocate(cuda::stream_ref stream, size_t bytes, size_t alignment) {
    return allocateImpl(bytes, [this, stream, bytes, alignment] {
      return upstream_.allocate(stream, bytes, alignment);
    });
  }

  void deallocate(
      cuda::stream_ref stream,
      void* pointer,
      size_t bytes,
      size_t alignment) noexcept {
    upstream_.deallocate(stream, pointer, bytes, alignment);
    recordFree(bytes);
  }

  bool operator==(const TrackingResourceImpl& other) const noexcept {
    return this == std::addressof(other);
  }

  bool operator!=(const TrackingResourceImpl& other) const noexcept {
    return !(*this == other);
  }

  friend void get_property(
      const TrackingResourceImpl&,
      cuda::mr::device_accessible) noexcept {}

 private:
  template <typename Allocate>
  void* allocateImpl(size_t bytes, Allocate&& allocate) {
    if (bytes == 0) {
      return allocate();
    }
    try {
      std::lock_guard<std::mutex> lock(trackingUpdateMutex);
      VELOX_CHECK_LE(
          bytes,
          static_cast<size_t>(std::numeric_limits<int64_t>::max()),
          "GPU allocation size exceeds Velox accounting range");
      ownerPool_->reportExternalAllocation(static_cast<int64_t>(bytes));
      sample();
    } catch (...) {
      markAllocationFailure(
          counters_.domain,
          owner_,
          bytes,
          asUnsigned(rootPool_->usedBytes()),
          asUnsigned(queryPool_->usedBytes()),
          asUnsigned(ownerPool_->usedBytes()));
      throw;
    }
    try {
      return allocate();
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(trackingUpdateMutex);
        ownerPool_->reportExternalFree(static_cast<int64_t>(bytes));
        sample();
      }
      markAllocationFailure(
          counters_.domain,
          owner_,
          bytes,
          asUnsigned(rootPool_->usedBytes()),
          asUnsigned(queryPool_->usedBytes()),
          asUnsigned(ownerPool_->usedBytes()));
      throw;
    }
  }

  void recordFree(size_t bytes) noexcept {
    if (bytes != 0) {
      std::lock_guard<std::mutex> lock(trackingUpdateMutex);
      ownerPool_->reportExternalFree(static_cast<int64_t>(bytes));
      sample();
    }
  }

  void sample() noexcept {
    sampleNvtx(
        counters_,
        asUnsigned(rootPool_->usedBytes()),
        asUnsigned(queryPool_->usedBytes()),
        asUnsigned(ownerPool_->usedBytes()));
  }

  // The pools borrow allocator and arbitrator pointers from resourceOwner_.
  // Declare the session lease first so it is released after every other
  // wrapper member has been destroyed.
  std::shared_ptr<void> sessionLease_;
  std::shared_ptr<memory::CustomMemoryResource> resourceOwner_;
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> queryPool_;
  std::shared_ptr<memory::MemoryPool> ownerPool_;
  GpuMemoryResource upstream_;
  GpuMemoryOwner owner_;
  NvtxCounters counters_;
};

class TrackingResource
    : public cuda::mr::shared_resource<TrackingResourceImpl> {
  using SharedBase = cuda::mr::shared_resource<TrackingResourceImpl>;

 public:
  template <typename... Args>
  explicit TrackingResource(Args&&... args)
      : SharedBase(
            cuda::mr::make_shared_resource<TrackingResourceImpl>(
                std::forward<Args>(args)...)) {}

  friend void get_property(
      const TrackingResource&,
      cuda::mr::device_accessible) noexcept {}
};

static_assert(
    cuda::mr::resource_with<TrackingResource, cuda::mr::device_accessible>);

struct OwnerRecord {
  GpuMemoryOwner owner;
  std::shared_ptr<memory::MemoryPool> pool;
};

class QueryResourceRegistry;

class TrackingSession : public std::enable_shared_from_this<TrackingSession> {
 public:
  TrackingSession() {
    capacity_ = trackingCapacity();
    memory::MemoryAllocator::Options allocatorOptions;
    allocatorOptions.capacity = capacity_;
    auto allocator =
        std::make_shared<memory::MallocAllocator>(allocatorOptions);
    memory::MemoryArbitrator::Config arbitratorConfig;
    arbitratorConfig.capacity = capacity_;
    auto arbitrator = memory::MemoryArbitrator::create(arbitratorConfig);
    resourceOwner_ = std::make_shared<memory::CustomMemoryResource>(
        "cudf-gpu-tracking",
        std::move(allocator),
        std::move(arbitrator),
        [] { return std::unique_ptr<memory::MemoryReclaimer>{}; },
        capacity_);
    static std::atomic<uint64_t> sequence{0};
    rootPool_ = memory::memoryManager()->addCustomRootPool(
        fmt::format(
            "cudf-gpu-tracking.{}",
            sequence.fetch_add(1, std::memory_order_relaxed)),
        resourceOwner_);
  }

  const std::shared_ptr<memory::MemoryPool>& rootPool() const {
    return rootPool_;
  }

  const std::shared_ptr<memory::CustomMemoryResource>& resourceOwner() const {
    return resourceOwner_;
  }

  std::shared_ptr<void> acquireLease();

  void registerOwner(
      const GpuMemoryOwner& owner,
      std::string_view queryInstance,
      const std::shared_ptr<memory::MemoryPool>& pool) {
    std::lock_guard<std::mutex> lock(mutex_);
    owners_.try_emplace(
        fmt::format("{}\x1f{}", queryInstance, ownerKey(owner)),
        OwnerRecord{owner, pool});
  }

  GpuMemorySnapshot snapshot() const {
    std::lock_guard<std::mutex> accountingLock(trackingUpdateMutex);
    GpuMemorySnapshot result;
    result.capacityBytes = capacity_;
    result.currentBytes = asUnsigned(rootPool_->usedBytes());
    result.peakBytes = asUnsigned(rootPool_->peakBytes());
    std::lock_guard<std::mutex> lock(mutex_);
    result.owners.reserve(owners_.size());
    for (const auto& [_, record] : owners_) {
      const auto stats = record.pool->stats();
      result.cumulativeRequestedBytes += stats.cumulativeExternalBytes;
      result.owners.push_back(
          {record.owner,
           stats.usedBytes,
           stats.peakBytes,
           stats.cumulativeExternalBytes,
           stats.numExternalAllocs,
           stats.numExternalFrees});
    }
    std::sort(
        result.owners.begin(),
        result.owners.end(),
        [](const auto& left, const auto& right) {
          return std::tie(left.currentBytes, left.peakBytes) >
              std::tie(right.currentBytes, right.peakBytes);
        });
    return result;
  }

  std::shared_ptr<QueryResourceRegistry> createRegistry(
      const GpuMemoryOwner& owner);

  std::shared_ptr<QueryResourceRegistry> testingRegistry(
      const GpuMemoryOwner& owner);

  bool canRetire();

  void markRetired() noexcept;

  void bindOperatorPool(
      memory::MemoryPool* operatorPool,
      const std::shared_ptr<QueryResourceRegistry>& registry,
      const GpuMemoryOwner& owner);

  std::optional<GpuMemoryResourceRefs> resourcesForOperatorPool(
      memory::MemoryPool* operatorPool,
      GpuMemoryResource& tempUpstream,
      GpuMemoryResource& outputUpstream);

 private:
  int64_t capacity_{0};
  std::shared_ptr<memory::CustomMemoryResource> resourceOwner_;
  std::shared_ptr<memory::MemoryPool> rootPool_;
  mutable std::mutex mutex_;
  bool retired_{false};
  std::atomic<uint64_t> activeLeases_{0};
  std::unordered_map<std::string, OwnerRecord> owners_;
  std::vector<std::weak_ptr<QueryResourceRegistry>> registries_;
  std::unordered_map<std::string, std::weak_ptr<QueryResourceRegistry>>
      testingRegistries_;
  struct BoundOperatorPool {
    std::weak_ptr<QueryResourceRegistry> registry;
    GpuMemoryOwner owner;
  };
  std::unordered_map<memory::MemoryPool*, BoundOperatorPool>
      boundOperatorPools_;
};

struct Resources {
  Resources(
      GpuMemoryResource tempUpstream,
      GpuMemoryResource outputUpstream,
      std::shared_ptr<void> sessionLease,
      const std::shared_ptr<memory::MemoryPool>& rootPool,
      const std::shared_ptr<memory::MemoryPool>& queryPool,
      const std::shared_ptr<memory::MemoryPool>& ownerPool,
      const std::shared_ptr<memory::CustomMemoryResource>& resourceOwner,
      const GpuMemoryOwner& owner,
      NvtxCounters counters)
      : tempUpstream(std::move(tempUpstream)),
        outputUpstream(std::move(outputUpstream)),
        temp(
            this->tempUpstream,
            sessionLease,
            rootPool,
            queryPool,
            ownerPool,
            resourceOwner,
            owner,
            counters) {
    if (this->tempUpstream != this->outputUpstream) {
      output.emplace(
          this->outputUpstream,
          std::move(sessionLease),
          rootPool,
          queryPool,
          ownerPool,
          resourceOwner,
          owner,
          counters);
    }
  }

  GpuMemoryResourceRefs refs() {
    auto tempRef = rmm::device_async_resource_ref{temp};
    return {
        tempRef, output ? rmm::device_async_resource_ref{*output} : tempRef};
  }

  GpuMemoryResource tempUpstream;
  GpuMemoryResource outputUpstream;
  TrackingResource temp;
  std::optional<TrackingResource> output;
};

class QueryResourceRegistry {
 public:
  QueryResourceRegistry(
      std::shared_ptr<TrackingSession> session,
      const GpuMemoryOwner& owner)
      : session_(session),
        rootPool_(session->rootPool()),
        resourceOwner_(session->resourceOwner()) {
    static std::atomic<uint64_t> sequence{0};
    const auto instance = sequence.fetch_add(1, std::memory_order_relaxed);
    logicalQueryId_ = owner.queryId;
    queryInstance_ = std::to_string(instance);
    queryPool_ = rootPool_->addAggregateChild(
        fmt::format("query.{}.{}", displayField(owner.queryId), instance));
  }

  bool belongsTo(const std::shared_ptr<TrackingSession>& session) const {
    return session_.lock() == session;
  }

  GpuMemoryResourceRefs resourcesFor(
      const GpuMemoryOwner& owner,
      GpuMemoryResource& tempUpstream,
      GpuMemoryResource& outputUpstream) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto key = ownerKey(owner);
    if (auto it = resources_.find(key); it != resources_.end()) {
      VELOX_CHECK(
          it->second->tempUpstream == tempUpstream &&
              it->second->outputUpstream == outputUpstream,
          "GPU tracking resource reused with different upstream resources");
      return it->second->refs();
    }

    auto task =
        childPool(taskPools_, owner.taskUuid, queryPool_, "task", owner.taskId);
    const auto nodeKey = owner.taskUuid + '\x1f' + owner.planNodeId;
    auto node = childPool(nodePools_, nodeKey, task, "node", owner.planNodeId);
    const auto pipelineKey =
        nodeKey + '\x1f' + std::to_string(owner.pipelineId);
    auto pipeline = childPool(
        pipelinePools_,
        pipelineKey,
        node,
        "pipeline",
        std::to_string(owner.pipelineId));
    auto ownerPool = pipeline->addLeafChild(
        fmt::format(
            "operator.{}.{}.{}",
            owner.operatorId,
            owner.driverId,
            displayField(owner.operatorType)));

    auto session = session_.lock();
    VELOX_CHECK_NOT_NULL(session, "GPU memory tracking session expired");
    auto resources = std::make_unique<Resources>(
        tempUpstream,
        outputUpstream,
        session->acquireLease(),
        rootPool_,
        queryPool_,
        ownerPool,
        resourceOwner_,
        owner,
        registerNvtxOwner(owner, logicalQueryId_, queryInstance_));
    auto refs = resources->refs();
    auto [_, inserted] = resources_.emplace(key, std::move(resources));
    VELOX_CHECK(inserted, "Duplicate GPU tracking owner");
    session->registerOwner(owner, queryInstance_, ownerPool);
    return refs;
  }

 private:
  static std::shared_ptr<memory::MemoryPool> childPool(
      std::unordered_map<std::string, std::shared_ptr<memory::MemoryPool>>& map,
      const std::string& key,
      const std::shared_ptr<memory::MemoryPool>& parent,
      std::string_view level,
      std::string_view label) {
    if (auto it = map.find(key); it != map.end()) {
      return it->second;
    }
    auto pool = parent->addAggregateChild(
        fmt::format("{}.{}", level, displayField(label)));
    map.emplace(key, pool);
    return pool;
  }

  std::weak_ptr<TrackingSession> session_;
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::CustomMemoryResource> resourceOwner_;
  std::shared_ptr<memory::MemoryPool> queryPool_;
  std::string logicalQueryId_;
  std::string queryInstance_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<memory::MemoryPool>>
      taskPools_;
  std::unordered_map<std::string, std::shared_ptr<memory::MemoryPool>>
      nodePools_;
  std::unordered_map<std::string, std::shared_ptr<memory::MemoryPool>>
      pipelinePools_;
  std::unordered_map<std::string, std::unique_ptr<Resources>> resources_;
};

std::shared_ptr<QueryResourceRegistry> TrackingSession::createRegistry(
    const GpuMemoryOwner& owner) {
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK(!retired_, "GPU memory tracking session has retired");
  auto registry =
      std::make_shared<QueryResourceRegistry>(shared_from_this(), owner);
  registries_.push_back(registry);
  return registry;
}

std::shared_ptr<void> TrackingSession::acquireLease() {
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK(!retired_, "GPU memory tracking session has retired");
  activeLeases_.fetch_add(1, std::memory_order_relaxed);
  return std::shared_ptr<void>(this, [session = shared_from_this()](void*) {
    session->activeLeases_.fetch_sub(1, std::memory_order_release);
  });
}

std::shared_ptr<QueryResourceRegistry> TrackingSession::testingRegistry(
    const GpuMemoryOwner& owner) {
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK(!retired_, "GPU memory tracking session has retired");
  const auto key = queryKey(owner);
  if (auto registry = testingRegistries_[key].lock()) {
    return registry;
  }
  auto registry =
      std::make_shared<QueryResourceRegistry>(shared_from_this(), owner);
  testingRegistries_[key] = registry;
  registries_.push_back(registry);
  return registry;
}

bool TrackingSession::canRetire() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::erase_if(
      registries_, [](const auto& registry) { return registry.expired(); });
  std::erase_if(testingRegistries_, [](const auto& entry) {
    return entry.second.expired();
  });
  if (!registries_.empty() ||
      activeLeases_.load(std::memory_order_acquire) != 0 ||
      rootPool_->usedBytes() != 0) {
    return false;
  }
  return true;
}

void TrackingSession::markRetired() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  retired_ = true;
}

void TrackingSession::bindOperatorPool(
    memory::MemoryPool* operatorPool,
    const std::shared_ptr<QueryResourceRegistry>& registry,
    const GpuMemoryOwner& owner) {
  std::lock_guard<std::mutex> lock(mutex_);
  boundOperatorPools_.insert_or_assign(
      operatorPool, BoundOperatorPool{registry, owner});
}

std::optional<GpuMemoryResourceRefs> TrackingSession::resourcesForOperatorPool(
    memory::MemoryPool* operatorPool,
    GpuMemoryResource& tempUpstream,
    GpuMemoryResource& outputUpstream) {
  std::shared_ptr<QueryResourceRegistry> registry;
  GpuMemoryOwner owner;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = boundOperatorPools_.find(operatorPool);
    if (it == boundOperatorPools_.end()) {
      return std::nullopt;
    }
    registry = it->second.registry.lock();
    if (!registry) {
      boundOperatorPools_.erase(it);
      return std::nullopt;
    }
    owner = it->second.owner;
  }
  return registry->resourcesFor(owner, tempUpstream, outputUpstream);
}

std::mutex sessionMutex;
std::shared_ptr<TrackingSession> installedSession;

std::shared_ptr<TrackingSession> trackingSession() {
  std::lock_guard<std::mutex> lock(sessionMutex);
  if (!installedSession) {
    installedSession = std::make_shared<TrackingSession>();
  }
  return installedSession;
}

std::shared_ptr<QueryResourceRegistry> queryRegistryFor(
    core::QueryCtx& queryCtx,
    const GpuMemoryOwner& owner,
    const std::shared_ptr<TrackingSession>& session) {
  if (auto registry =
          queryCtx.registry<QueryResourceRegistry>(kQueryRegistryKey)) {
    if (registry->belongsTo(session)) {
      return registry;
    }
  }
  static std::mutex registryCreationMutex;
  std::lock_guard<std::mutex> lock(registryCreationMutex);
  if (auto registry =
          queryCtx.registry<QueryResourceRegistry>(kQueryRegistryKey)) {
    if (registry->belongsTo(session)) {
      return registry;
    }
  }
  auto registry = session->createRegistry(owner);
  queryCtx.setRegistry(kQueryRegistryKey, registry, true);
  return registry;
}

exec::Operator* runtimeOperator() {
  return dynamic_cast<exec::Operator*>(getThreadLocalRunTimeStatWriter());
}

GpuMemoryOwner ownerForOperator(exec::Operator* op) {
  auto* driverCtx = op->operatorCtx()->driverCtx();
  const auto& task = op->operatorCtx()->task();
  return {
      task->queryCtx()->queryId(),
      task->taskId(),
      task->uuid(),
      op->planNodeId(),
      driverCtx->pipelineId,
      driverCtx->driverId,
      op->operatorId(),
      op->operatorType()};
}

GpuMemoryOwner unattributedOwner() {
  return {
      "<unattributed>",
      "<unattributed>",
      "<unattributed>",
      "<unattributed>",
      -1,
      -1,
      -1,
      "unattributed"};
}

std::atomic<uint64_t> resolutionGeneration{1};

struct CachedResources {
  const exec::Operator* op{nullptr};
  uint64_t generation{0};
  std::optional<GpuMemoryResourceRefs> refs;
  std::shared_ptr<QueryResourceRegistry> keepAlive;
};

thread_local CachedResources cachedResources;

GpuMemoryResourceRefs untrackedRefs(
    GpuMemoryResource& tempUpstream,
    GpuMemoryResource& outputUpstream) {
  return {
      rmm::device_async_resource_ref{tempUpstream},
      rmm::device_async_resource_ref{outputUpstream}};
}

} // namespace

void configureGpuMemoryTrackingCapacity(int32_t memoryPercent) noexcept {
  configuredTrackingCapacity.store(
      calculateTrackingCapacity(memoryPercent), std::memory_order_release);
}

GpuMemoryResourceRefs gpuMemoryResourcesForCurrentOperator(
    GpuMemoryResource& tempUpstream,
    GpuMemoryResource& outputUpstream) {
  std::shared_lock<std::shared_mutex> lifecycleLock(trackingLifecycleMutex);
  const auto generation = resolutionGeneration.load(std::memory_order_acquire);
  auto* op = runtimeOperator();
  if (cachedResources.refs && cachedResources.op == op &&
      cachedResources.generation == generation) {
    return *cachedResources.refs;
  }

  try {
    auto session = trackingSession();
    std::shared_ptr<QueryResourceRegistry> registry;
    GpuMemoryOwner owner;
    if (op != nullptr) {
      owner = ownerForOperator(op);
      registry = queryRegistryFor(
          *op->operatorCtx()->task()->queryCtx(), owner, session);
    } else {
      owner = unattributedOwner();
      registry = session->testingRegistry(owner);
    }
    auto refs = registry->resourcesFor(owner, tempUpstream, outputUpstream);
    cachedResources = {
        op, generation, refs, op == nullptr ? std::move(registry) : nullptr};
    return refs;
  } catch (const std::exception& error) {
    LOG(ERROR) << "GPU memory attribution disabled for this call: "
               << error.what();
  } catch (...) {
    LOG(ERROR) << "GPU memory attribution disabled for this call";
  }
  return untrackedRefs(tempUpstream, outputUpstream);
}

bool registerGpuMemoryOperator(exec::Operator* op) noexcept {
  if (op == nullptr || !mr_ || !output_mr_) {
    return false;
  }
  try {
    std::shared_lock<std::shared_mutex> lifecycleLock(trackingLifecycleMutex);
    const auto owner = ownerForOperator(op);
    auto session = trackingSession();
    auto registry = queryRegistryFor(
        *op->operatorCtx()->task()->queryCtx(), owner, session);
    static_cast<void>(registry->resourcesFor(owner, *mr_, *output_mr_));
    session->bindOperatorPool(op->pool(), registry, owner);
    return true;
  } catch (const std::exception& error) {
    LOG(ERROR) << "Failed to register GPU memory owner for "
               << op->operatorType() << ": " << error.what();
  } catch (...) {
    LOG(ERROR) << "Failed to register GPU memory owner for "
               << op->operatorType();
  }
  return false;
}

std::optional<GpuMemoryResourceRefs> gpuMemoryResourcesForOperatorPool(
    memory::MemoryPool* operatorPool,
    GpuMemoryResource& tempUpstream,
    GpuMemoryResource& outputUpstream) {
  std::shared_lock<std::shared_mutex> lifecycleLock(trackingLifecycleMutex);
  std::shared_ptr<TrackingSession> session;
  {
    std::lock_guard<std::mutex> lock(sessionMutex);
    session = installedSession;
  }
  if (!session || operatorPool == nullptr) {
    return std::nullopt;
  }
  return session->resourcesForOperatorPool(
      operatorPool, tempUpstream, outputUpstream);
}

void discardGpuMemoryOwnerCache() noexcept {
  resolutionGeneration.fetch_add(1, std::memory_order_release);
}

GpuMemoryResourcePair createGpuMemoryTrackingResources(
    GpuMemoryResource tempUpstream,
    GpuMemoryResource outputUpstream,
    const GpuMemoryOwner& owner) {
  std::shared_lock<std::shared_mutex> lifecycleLock(trackingLifecycleMutex);
  auto registry = trackingSession()->testingRegistry(owner);
  auto refs = registry->resourcesFor(owner, tempUpstream, outputUpstream);
  return {
      GpuMemoryResource{refs.temp},
      GpuMemoryResource{refs.output},
      std::move(registry)};
}

GpuMemorySnapshot getGpuMemorySnapshot() {
  std::shared_lock<std::shared_mutex> lifecycleLock(trackingLifecycleMutex);
  std::shared_ptr<TrackingSession> session;
  {
    std::lock_guard<std::mutex> lock(sessionMutex);
    session = installedSession;
  }
  return session ? session->snapshot() : GpuMemorySnapshot{};
}

bool resetGpuMemoryTracking() noexcept {
  try {
    std::unique_lock<std::shared_mutex> lifecycleLock(trackingLifecycleMutex);
    std::lock_guard<std::mutex> accountingLock(trackingUpdateMutex);
    std::shared_ptr<TrackingSession> session;
    {
      std::lock_guard<std::mutex> lock(sessionMutex);
      session = installedSession;
      if (session && !session->canRetire()) {
        return false;
      }
      if (installedSession != session) {
        return false;
      }
    }
    if (!resetNvtx()) {
      return false;
    }
    if (session) {
      session->markRetired();
    }
    {
      std::lock_guard<std::mutex> lock(sessionMutex);
      installedSession.reset();
    }
    resolutionGeneration.fetch_add(1, std::memory_order_release);
    return true;
  } catch (...) {
    return false;
  }
}

} // namespace facebook::velox::cudf_velox
