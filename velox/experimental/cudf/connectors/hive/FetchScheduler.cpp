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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/FetchScheduler.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <folly/system/ThreadName.h>
#include <glog/logging.h>
#include <sched.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <unordered_map>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

uint64_t nanosBetween(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
  if (end <= start) {
    return 0;
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
      .count();
}

std::exception_ptr makeShutdownFailure() {
  try {
    VELOX_FAIL("Fetch scheduler stopped with the range still queued");
  } catch (...) {
    return std::current_exception();
  }
}

// Starts the copy of the first 'size' bytes of 'buffer' to 'dst' on the
// buffer's stream and marks its end with the buffer's event.
void startCopy(
    const PinnedBufferPool::StagingBuffer& buffer,
    size_t size,
    uint8_t* dst) {
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      dst, buffer.data, size, cudaMemcpyHostToDevice, buffer.stream));
  const auto status = cudaEventRecord(buffer.event, buffer.stream);
  if (status != cudaSuccess) {
    // The copy is queued and reads the buffer, so the buffer cannot go back
    // to its pool before the stream has drained.
    cudaStreamSynchronize(buffer.stream);
    CUDF_CUDA_TRY(status);
  }
}

// Waits for the copy started from 'buffer', after which the buffer can be
// reused.
void waitForCopy(const PinnedBufferPool::StagingBuffer& buffer) {
  // A copy of one buffer over a fast link is done sooner than a sleeping
  // thread is woken, so the thread looks for about as long as such a copy
  // takes before it sleeps.
  constexpr std::chrono::microseconds kMaxLook{200};
  const auto deadline = std::chrono::steady_clock::now() + kMaxLook;
  auto status = cudaEventQuery(buffer.event);
  while (status == cudaErrorNotReady &&
         std::chrono::steady_clock::now() < deadline) {
    status = cudaEventQuery(buffer.event);
  }
  if (status == cudaErrorNotReady) {
    status = cudaEventSynchronize(buffer.event);
  }
  if (status != cudaSuccess) {
    cudaStreamSynchronize(buffer.stream);
    CUDF_CUDA_TRY(status);
  }
}

// Copies 'size' bytes of pageable memory at 'src' to 'dst' and waits until
// they are on the device. A plain synchronous copy can return while a short
// piece is still on its way, so the copy gets a stream that is drained.
void copyPageableToDevice(const uint8_t* src, size_t size, uint8_t* dst) {
  cudaStream_t stream{nullptr};
  CUDF_CUDA_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const auto copyStatus =
      cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
  const auto syncStatus = cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  CUDF_CUDA_TRY(copyStatus);
  CUDF_CUDA_TRY(syncStatus);
}

// Returns the time a staging buffer of 'pool' takes to copy to the device
// over the time it takes to fill from memory, each the best of a few, or
// zero when that cannot be measured. The device memory comes from cuDF's
// memory resource, like that of the scans.
double measureCopyToFillRatio(PinnedBufferPool& pool) {
  using Clock = std::chrono::steady_clock;
  constexpr size_t kNumSamples{4};
  try {
    ensureCudaContextForThread();
    auto buffer = pool.acquire();
    if (buffer == nullptr) {
      return 0;
    }
    const auto size = pool.bufferSize();
    // Two buffers, filled from in turn. The processor may still hold them in
    // its cache, so this is the fastest fill the machine does, and the ratio
    // reads higher than one taken from files would.
    const std::vector<uint8_t> source(2 * size, 1);
    uint64_t fillNanos{std::numeric_limits<uint64_t>::max()};
    for (size_t i = 0; i < kNumSamples; ++i) {
      const auto start = Clock::now();
      std::memcpy(buffer->data, source.data() + (i % 2) * size, size);
      fillNanos = std::min(fillNanos, nanosBetween(start, Clock::now()));
    }

    auto stream = cudfGlobalStreamPool().get_stream();
    rmm::device_buffer device(size, stream, get_temp_mr());
    stream.sync();
    uint64_t copyNanos{std::numeric_limits<uint64_t>::max()};
    for (size_t i = 0; i < kNumSamples; ++i) {
      const auto start = Clock::now();
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          device.data(),
          buffer->data,
          size,
          cudaMemcpyHostToDevice,
          buffer->stream));
      CUDF_CUDA_TRY(cudaStreamSynchronize(buffer->stream));
      copyNanos = std::min(copyNanos, nanosBetween(start, Clock::now()));
    }
    return fillNanos == 0 ? 0 : static_cast<double>(copyNanos) / fillNanos;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to measure the copy of a staging buffer: "
                 << e.what();
    return 0;
  }
}

// Returns the fetch-ahead budget that the schedulers of the process share
// for the device cuDF is registered on, creating it on first use.
std::shared_ptr<FetchAheadBudget> sharedFetchAheadBudget() {
  static std::mutex mutex;
  static std::unordered_map<int, std::shared_ptr<FetchAheadBudget>> budgets;
  ensureCudaContextForThread();
  int device{0};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(mutex);
  auto& budget = budgets[device];
  if (budget == nullptr) {
    // From the device's size and not from what is free now: free memory
    // depends on when the first scan runs and on how cuDF's memory resource
    // keeps what it has released.
    size_t freeBytes{0};
    size_t totalBytes{0};
    CUDF_CUDA_TRY(cudaMemGetInfo(&freeBytes, &totalBytes));
    const auto capacity = totalBytes / 16;
    budget = std::make_shared<FetchAheadBudget>(capacity);
  }
  return budget;
}

// Returns the bound on readers for a scheduler that was given none: half of
// the CPUs the process may run on, shared between the devices it sees.
size_t defaultMaxReaders() {
  size_t numCpus = std::thread::hardware_concurrency();
  cpu_set_t cpus;
  CPU_ZERO(&cpus);
  if (sched_getaffinity(0, sizeof(cpus), &cpus) == 0) {
    numCpus = CPU_COUNT(&cpus);
  }
  int numDevices{1};
  if (cudaGetDeviceCount(&numDevices) != cudaSuccess || numDevices < 1) {
    numDevices = 1;
    // Clear the error so that an unrelated CUDA call does not report it.
    cudaGetLastError();
  }
  return std::clamp<size_t>(numCpus / 2 / numDevices, 4, 64);
}

} // namespace

struct FetchScheduler::Group {
  // Arrival number among the queued groups. Set when the group queues.
  uint64_t sequence{0};
  // Consumers blocked in wait() on a range of the group.
  size_t numWaiters{0};
  // Bytes of the group's ranges that no reader has been given.
  uint64_t queuedBytes{0};
  // Bytes of groups that asked later served since this one, while a consumer
  // waited for it. Reset when the group queues.
  uint64_t overtakenBytes{0};
  // Ranges with pieces no reader has started, in submission order.
  std::deque<std::shared_ptr<Read>> pending;
};

class FetchScheduler::Read {
 public:
  Read(
      std::shared_ptr<Group> _group,
      size_t _size,
      uint8_t* _dst,
      FillOrder _order,
      Fill _fill,
      Clock::time_point _submittedAt)
      : group(std::move(_group)),
        size(_size),
        dst(_dst),
        order(_order),
        submittedAt(_submittedAt),
        fill(std::move(_fill)) {}

  // Whether no piece is left to start and none is in flight.
  bool done() const {
    return numOutstanding == 0 && nextOffset >= size;
  }

  const std::shared_ptr<Group> group;
  const size_t size;
  uint8_t* const dst;
  const FillOrder order;
  const Clock::time_point submittedAt;

  // Readers call it without the scheduler's mutex while a piece of the range
  // is outstanding. Taken out by the consumer once the range is complete.
  Fill fill;

  // The fields below are guarded by the scheduler's mutex.

  // The scheduler's copy-busy, reader-limited and buffer-limited times when
  // the range was submitted.
  uint64_t copyBusyAtSubmit{0};
  uint64_t readerLimitedAtSubmit{0};
  uint64_t bufferLimitedAtSubmit{0};
  // First byte no reader has been given.
  size_t nextOffset{0};
  // Pieces given to a reader whose bytes are not on the device yet.
  size_t numOutstanding{0};
  // Pieces on the device, and those of them whose reader did not wait for
  // the copy.
  size_t numPieces{0};
  size_t numQueuedCopies{0};
  // Set once done() holds and the waiters have been told.
  bool completed{false};
  // First failure of a fill or a copy.
  std::exception_ptr failure;
  ReadStats stats;
};

// static
bool FetchScheduler::prefersQueuedCopies(double copyToFillRatio) {
  return copyToFillRatio > 0 && copyToFillRatio < kMaxQueuedCopyToFillRatio;
}

// static
size_t FetchScheduler::initialReadersFor(double copyToFillRatio) {
  if (copyToFillRatio <= 0) {
    return 16;
  }
  return static_cast<size_t>(std::clamp(10 / copyToFillRatio, 4.0, 24.0));
}

// static
FetchScheduler::ControlState FetchScheduler::nextControlState(
    const Options& options,
    const ControlState& state,
    const ControlWindow& window) {
  VELOX_CHECK_GT(state.numReaders, 0);
  auto next = state;
  if (window.saturatedNanos == 0) {
    return next;
  }
  const double rate = window.bytes * 1e9 / window.saturatedNanos;
  // Readers that wait for copies half of the time have a queue of copies in
  // front of them.
  const bool isCopyLimited = window.readerBusyNanos > 0 &&
      window.copyWaitNanos * 2 >= window.readerBusyNanos;
  // Readers kept from their next piece for lack of a staging buffer have
  // more copies queued than the device takes: more readers cannot help, and
  // the ones there do no harm.
  const bool isBufferLimited =
      window.bufferLimitedNanos * 2 >= window.saturatedNanos;
  using Phase = ControlState::Phase;
  const auto hold = [&](size_t numReaders) {
    next.phase = Phase::kHold;
    next.numReaders = numReaders;
    next.numHeldReaders = numReaders;
    // Trials of more readers that were not kept one after the other say
    // that the number is enough: the hold gets twice as long with each, up
    // to eight times.
    next.numHoldsLeft = options.numHoldPeriods
        << std::min<size_t>(next.numFailedTrials, 3);
  };
  // Starts a trial of more readers than 'numHeld', or holds at the bound.
  const auto tryMore = [&](size_t numHeld) {
    const auto numReaders = std::min(
        options.maxReaders, numHeld + std::max<size_t>(1, numHeld / 4));
    if (numReaders == numHeld) {
      hold(numHeld);
      return;
    }
    next.phase = Phase::kTryMore;
    next.numHeldReaders = numHeld;
    next.numReaders = numReaders;
  };
  // Starts a trial of fewer readers than 'numHeld', or holds at the bound.
  const auto tryFewer = [&](size_t numHeld) {
    const auto step = std::max<size_t>(1, numHeld / 8);
    const auto numReaders = numHeld > options.minReaders + step
        ? numHeld - step
        : options.minReaders;
    if (numReaders == numHeld) {
      hold(numHeld);
      return;
    }
    next.phase = Phase::kTryFewer;
    next.numHeldReaders = numHeld;
    next.numReaders = numReaders;
  };
  switch (state.phase) {
    case Phase::kHold: {
      next.numHeldReaders = state.numReaders;
      next.baseRate =
          state.baseRate == 0 ? rate : (3 * state.baseRate + rate) / 4;
      if (state.numHoldsLeft > 0) {
        --next.numHoldsLeft;
        break;
      }
      // Before anything was delivered there is no rate to compare a trial
      // with.
      if (next.baseRate <= 0) {
        break;
      }
      if (isCopyLimited) {
        if (state.numReaders > options.minReaders) {
          tryFewer(state.numReaders);
        }
      } else if (!isBufferLimited && state.numReaders < options.maxReaders) {
        // Against the latest rate when that is the higher one: a scan that
        // is still getting faster must not pass for the trial's gain.
        next.baseRate = std::max(next.baseRate, rate);
        tryMore(state.numReaders);
      }
      break;
    }
    case Phase::kTryMore: {
      // The share of the rate that the added readers would bring if every
      // reader brought as much as the held ones.
      const double share =
          static_cast<double>(state.numReaders - state.numHeldReaders) /
          state.numHeldReaders;
      // Nothing delivered says nothing in favour of more readers either.
      if (rate > 0 && state.baseRate > 0 &&
          rate >= state.baseRate * (1 + share / 2) && !isCopyLimited) {
        next.baseRate = rate;
        next.numFailedTrials = 0;
        tryMore(state.numReaders);
      } else {
        ++next.numFailedTrials;
        hold(state.numHeldReaders);
      }
      break;
    }
    case Phase::kTryFewer: {
      const double share =
          static_cast<double>(state.numHeldReaders - state.numReaders) /
          state.numHeldReaders;
      // Nothing delivered says nothing in favour of fewer readers.
      if (rate > 0 && rate >= state.baseRate * (1 - share / 4)) {
        next.baseRate = rate;
        if (isCopyLimited) {
          tryFewer(state.numReaders);
        } else {
          hold(state.numReaders);
        }
      } else {
        hold(state.numHeldReaders);
      }
      break;
    }
  }
  return next;
}

FetchScheduler::FetchScheduler(
    std::shared_ptr<PinnedBufferPool> pool,
    const Options& options)
    : pool_(std::move(pool)), options_(options), lastAdvance_(Clock::now()) {
  VELOX_CHECK_NOT_NULL(pool_);
  VELOX_CHECK_GT(options_.minReaders, 0);
  if (options_.maxReaders == 0) {
    options_.maxReaders = std::max(defaultMaxReaders(), options_.minReaders);
  }
  VELOX_CHECK_LE(options_.minReaders, options_.maxReaders);
  control_.numReaders = std::clamp(
      options_.initialReaders == 0 ? initialReadersFor(0)
                                   : options_.initialReaders,
      options_.minReaders,
      options_.maxReaders);
  control_.numHeldReaders = control_.numReaders;
  isCopyQueued_ = options_.copyMode == CopyMode::kQueue;
}

FetchScheduler::~FetchScheduler() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    advance(Clock::now());
    stopping_ = true;
    while (!queuedGroups_.empty()) {
      const auto read = queuedGroups_.back()->pending.front();
      if (read->failure == nullptr) {
        read->failure = makeShutdownFailure();
      }
      dropRemaining(*read);
      completeIfDone(*read, lastAdvance_);
    }
    workAvailable_.notify_all();
    bufferFreed_.notify_all();
  }
  for (auto& reader : readers_) {
    reader.join();
  }
  if (copyWaiter_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      readersStopped_ = true;
    }
    copyQueued_.notify_all();
    copyWaiter_.join();
  }
}

void FetchScheduler::dropQueued(Group& group) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto now = Clock::now();
  advance(now);
  while (!group.pending.empty()) {
    const auto read = group.pending.front();
    dropRemaining(*read);
    completeIfDone(*read, now);
  }
  bufferFreed_.notify_all();
}

FetchAheadBudget& FetchScheduler::fetchAhead() {
  std::call_once(fetchAheadOnce_, [&]() {
    fetchAhead_ = options_.fetchAheadBytes > 0
        ? std::make_shared<FetchAheadBudget>(options_.fetchAheadBytes)
        : sharedFetchAheadBudget();
  });
  return *fetchAhead_;
}

std::shared_ptr<FetchScheduler::Group> FetchScheduler::makeGroup() {
  return std::make_shared<Group>();
}

void FetchScheduler::ensureStarted() {
  if (started_) {
    return;
  }
  // A failure before the first thread exists leaves nothing behind, so the
  // next submission starts over.
  readers_.reserve(options_.maxReaders);
  while (readers_.size() < options_.maxReaders) {
    try {
      readers_.emplace_back([this]() { readerLoop(); });
    } catch (const std::exception& e) {
      if (readers_.empty()) {
        throw;
      }
      // The scheduler works with the readers it has.
      LOG(WARNING) << "Fetch scheduler started " << readers_.size() << " of "
                   << options_.maxReaders << " reader threads: " << e.what();
      break;
    }
  }
  try {
    copyWaiter_ = std::thread([this]() { copyWaiterLoop(); });
  } catch (const std::exception& e) {
    LOG(WARNING) << "Fetch scheduler readers wait for their own copies: "
                 << e.what();
    options_.copyMode = CopyMode::kWait;
    isCopyQueued_ = false;
  }
  options_.maxReaders = readers_.size();
  options_.minReaders = std::min(options_.minReaders, options_.maxReaders);
  if (options_.copyMode == CopyMode::kAuto || options_.initialReaders == 0) {
    stats_.copyToFillRatio = measureCopyToFillRatio(*pool_);
    if (options_.copyMode == CopyMode::kAuto) {
      isCopyQueued_ = prefersQueuedCopies(stats_.copyToFillRatio);
    }
    if (options_.initialReaders == 0) {
      control_.numReaders = std::clamp(
          initialReadersFor(stats_.copyToFillRatio),
          options_.minReaders,
          options_.maxReaders);
      control_.numHeldReaders = control_.numReaders;
    }
    LOG(INFO) << "Fetch scheduler: a staging buffer takes "
              << stats_.copyToFillRatio
              << " times as long to copy to the device as to fill: "
              << control_.numReaders << " readers that "
              << (isCopyQueued_ ? "go on while their copies run"
                                : "wait for their copies");
  }
  control_.numReaders = std::min(control_.numReaders, options_.maxReaders);
  control_.numHeldReaders =
      std::min(control_.numHeldReaders, options_.maxReaders);
  started_ = true;
}

std::shared_ptr<FetchScheduler::Read> FetchScheduler::submit(
    const std::shared_ptr<Group>& group,
    size_t size,
    uint8_t* dst,
    FillOrder order,
    Fill fill) {
  VELOX_CHECK_NOT_NULL(group);
  const auto now = Clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK(!stopping_, "Fetch scheduler is stopping");
  ensureStarted();
  advance(now);
  auto read =
      std::make_shared<Read>(group, size, dst, order, std::move(fill), now);
  read->copyBusyAtSubmit = stats_.copyBusyNanos;
  read->readerLimitedAtSubmit = stats_.readerLimitedNanos;
  read->bufferLimitedAtSubmit = stats_.bufferLimitedNanos;
  if (size == 0) {
    completeIfDone(*read, now);
    return read;
  }
  if (group->pending.empty()) {
    group->sequence = nextSequence_++;
    group->overtakenBytes = 0;
    queuedGroups_.push_back(group.get());
  }
  group->pending.push_back(read);
  group->queuedBytes += size;
  workAvailable_.notify_one();
  return read;
}

FetchScheduler::ReadStats FetchScheduler::wait(Read& read) {
  Fill fill;
  std::exception_ptr failure;
  ReadStats stats;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    auto& group = *read.group;
    ++group.numWaiters;
    ++numWaiters_;
    completed_.wait(lock, [&]() { return read.completed; });
    --numWaiters_;
    --group.numWaiters;
    // No reader uses the fill of a complete range. It is destroyed below,
    // outside the lock, since it can own a stream of the file.
    fill = std::move(read.fill);
    read.fill = nullptr;
    failure = read.failure;
    stats = read.stats;
  }
  fill = nullptr;
  if (failure != nullptr) {
    std::rethrow_exception(failure);
  }
  return stats;
}

void FetchScheduler::cancel(Read& read) noexcept {
  Fill fill;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!read.completed) {
      const auto now = Clock::now();
      advance(now);
      dropRemaining(read);
      completeIfDone(read, now);
      // A reader between two pieces of the range waits for a staging buffer.
      bufferFreed_.notify_all();
      completed_.wait(lock, [&]() { return read.completed; });
    }
    fill = std::move(read.fill);
    read.fill = nullptr;
  }
}

void FetchScheduler::post(std::function<void()> task) {
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK(!stopping_, "Fetch scheduler is stopping");
  ensureStarted();
  tasks_.push_back(std::move(task));
  workAvailable_.notify_one();
}

FetchScheduler::Stats FetchScheduler::stats() {
  std::lock_guard<std::mutex> lock(mutex_);
  advance(Clock::now());
  auto stats = stats_;
  stats.numReaders = control_.numReaders;
  stats.isCopyQueued = isCopyQueued_;
  stats.numWaiters = numWaiters_;
  return stats;
}

size_t FetchScheduler::depthLimit() const {
  // A reader that goes on while its copy runs has one buffer being copied
  // and one being filled.
  const auto numBuffersPerReader = isCopyQueued_ ? 2 : 1;
  return std::max<size_t>(
      1,
      std::min(pool_->maxBuffers(), control_.numReaders * numBuffersPerReader));
}

// static
bool FetchScheduler::isMuchSmaller(const Group& group, const Group& other) {
  return group.queuedBytes * 4 <= other.queuedBytes;
}

bool FetchScheduler::isAged(const Group& group) const {
  return group.overtakenBytes >= options_.maxOvertakenBytes;
}

bool FetchScheduler::canDispatch() const {
  return !queuedGroups_.empty() && numRunning_ < control_.numReaders &&
      numBuffersInUse_ < depthLimit();
}

void FetchScheduler::advance(Clock::time_point now) {
  const auto elapsed = nanosBetween(lastAdvance_, now);
  lastAdvance_ = std::max(lastAdvance_, now);
  const bool hasQueued = !queuedGroups_.empty();
  if (!hasQueued && numRunning_ == 0 && numCopies_ == 0) {
    return;
  }
  stats_.activeNanos += elapsed;
  stats_.readerNanos += elapsed * control_.numReaders;
  stats_.readerBusyNanos += elapsed * numRunning_;
  stats_.copyWaitNanos += elapsed * numCopyWaiters_;
  if (numCopies_ > 0) {
    stats_.copyBusyNanos += elapsed;
  }
  const bool isBufferLimited = numBufferWaiters_ > 0 ||
      (hasQueued && numRunning_ < control_.numReaders &&
       numBuffersInUse_ >= depthLimit());
  if (isBufferLimited) {
    stats_.bufferLimitedNanos += elapsed;
  } else if (hasQueued && numRunning_ >= control_.numReaders) {
    stats_.readerLimitedNanos += elapsed;
  }
  if (!hasQueued) {
    return;
  }
  // The rate only tells what the readers can deliver while there is more
  // for them to do.
  stats_.saturatedNanos += elapsed;
  window_.saturatedNanos += elapsed;
  window_.readerBusyNanos += elapsed * numRunning_;
  window_.copyWaitNanos += elapsed * numCopyWaiters_;
  if (isBufferLimited) {
    window_.bufferLimitedNanos += elapsed;
  }
  uint64_t periodNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             options_.controlPeriod)
                             .count();
  // The rate of one period depends on what is being read. A trial of a
  // number of readers is measured for twice as long, since a wrong verdict
  // stays for a hold. A trial of the other way of waiting for copies is
  // measured twice instead, so that one that loses ends after one period.
  if (control_.phase == ControlState::Phase::kTryMore ||
      control_.phase == ControlState::Phase::kTryFewer) {
    periodNanos *= 2;
  }
  if (window_.saturatedNanos < periodNanos) {
    return;
  }
  const auto numReaders = control_.numReaders;
  control_ = nextControlState(options_, control_, window_);
  window_ = {};
  if (control_.numReaders > numReaders) {
    ++stats_.numRaises;
    workAvailable_.notify_one();
    bufferFreed_.notify_all();
  } else if (control_.numReaders < numReaders) {
    ++stats_.numLowers;
  }
}

FetchScheduler::Job FetchScheduler::takeJob(Clock::time_point now) {
  advance(now);
  size_t best{0};
  for (size_t i = 1; i < queuedGroups_.size(); ++i) {
    const auto* group = queuedGroups_[i];
    const auto* bestGroup = queuedGroups_[best];
    const bool waited = group->numWaiters > 0;
    const bool bestWaited = bestGroup->numWaiters > 0;
    bool better;
    if (waited != bestWaited) {
      better = waited;
    } else if (waited && isAged(*group) != isAged(*bestGroup)) {
      better = isAged(*group);
    } else if (waited && !isAged(*group) && isMuchSmaller(*group, *bestGroup)) {
      better = true;
    } else if (waited && !isAged(*group) && isMuchSmaller(*bestGroup, *group)) {
      better = false;
    } else {
      better = group->sequence < bestGroup->sequence;
    }
    if (better) {
      best = i;
    }
  }
  auto* group = queuedGroups_[best];
  Job job;
  job.read = group->pending.front();
  auto& read = *job.read;
  job.offset = read.nextOffset;
  job.size = std::min(pool_->bufferSize(), read.size - read.nextOffset);
  read.nextOffset += job.size;
  // A sequential range stays with the reader that starts it.
  group->queuedBytes -=
      read.order == FillOrder::kSequential ? read.size : job.size;
  if (read.order == FillOrder::kSequential || read.nextOffset >= read.size) {
    group->pending.pop_front();
    if (group->pending.empty()) {
      queuedGroups_.erase(queuedGroups_.begin() + best);
    }
  }
  ++read.numOutstanding;
  // What the groups that asked earlier, and are waited for, were made to
  // wait for.
  for (auto* other : queuedGroups_) {
    if (other->numWaiters > 0 && other->sequence < group->sequence) {
      other->overtakenBytes += job.size;
    }
  }
  if (job.offset == 0) {
    const auto queuedNanos = nanosBetween(read.submittedAt, now);
    read.stats.queuedNanos += queuedNanos;
    stats_.queuedNanos += queuedNanos;
  }
  ++numRunning_;
  ++numBuffersInUse_;
  stats_.maxRunningReaders = std::max(stats_.maxRunningReaders, numRunning_);
  stats_.maxBuffersInUse = std::max(stats_.maxBuffersInUse, numBuffersInUse_);
  return job;
}

void FetchScheduler::dropRemaining(Read& read) {
  if (read.nextOffset >= read.size) {
    return;
  }
  auto& pending = read.group->pending;
  const auto it =
      std::find_if(pending.begin(), pending.end(), [&](const auto& queued) {
        return queued.get() == &read;
      });
  if (it != pending.end()) {
    read.group->queuedBytes -= read.size - read.nextOffset;
    pending.erase(it);
    if (pending.empty()) {
      queuedGroups_.erase(
          std::find(
              queuedGroups_.begin(), queuedGroups_.end(), read.group.get()));
    }
  }
  read.nextOffset = read.size;
}

void FetchScheduler::completeIfDone(Read& read, Clock::time_point now) {
  if (read.completed || !read.done()) {
    return;
  }
  read.completed = true;
  const auto elapsed = nanosBetween(read.submittedAt, now);
  if (elapsed > 0) {
    const auto percent = [&](uint64_t nanos) {
      return std::min<uint64_t>(100, nanos * 100 / elapsed);
    };
    read.stats.copyBusyPercent =
        percent(stats_.copyBusyNanos - read.copyBusyAtSubmit);
    read.stats.readerLimitedPercent =
        percent(stats_.readerLimitedNanos - read.readerLimitedAtSubmit);
    read.stats.bufferLimitedPercent =
        percent(stats_.bufferLimitedNanos - read.bufferLimitedAtSubmit);
  }
  if (read.numPieces > 0) {
    read.stats.copyQueuedPercent = read.numQueuedCopies * 100 / read.numPieces;
  }
  read.stats.numReaders = control_.numReaders;
  read.stats.copyToFillPercent =
      static_cast<uint64_t>(stats_.copyToFillRatio * 100);
  completed_.notify_all();
}

void FetchScheduler::recordPiece(
    Read& read,
    size_t size,
    uint64_t readNanos,
    uint64_t copyNanos,
    bool isCopyQueued) {
  ++read.numPieces;
  if (isCopyQueued) {
    ++read.numQueuedCopies;
    ++stats_.numQueuedCopies;
  }
  read.stats.bytes += size;
  read.stats.readNanos += readNanos;
  read.stats.copyNanos += copyNanos;
  stats_.bytes += size;
  ++stats_.numPieces;
  stats_.readNanos += readNanos;
  if (!queuedGroups_.empty()) {
    window_.bytes += size;
    stats_.saturatedBytes += size;
  }
}

void FetchScheduler::finishPiece(
    Read& read,
    Clock::time_point now,
    std::exception_ptr failure) {
  --numBuffersInUse_;
  --read.numOutstanding;
  if (failure != nullptr) {
    if (read.failure == nullptr) {
      read.failure = std::move(failure);
    }
    dropRemaining(read);
  }
  completeIfDone(read, now);
  bufferFreed_.notify_all();
  workAvailable_.notify_one();
}

void FetchScheduler::readerLoop() {
  folly::setThreadName("CudfFetchRead");
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    workAvailable_.wait(
        lock, [&]() { return stopping_ || !tasks_.empty() || canDispatch(); });
    if (!tasks_.empty()) {
      auto task = std::move(tasks_.front());
      tasks_.pop_front();
      lock.unlock();
      task();
      task = nullptr;
      lock.lock();
      continue;
    }
    if (stopping_) {
      return;
    }
    auto job = takeJob(Clock::now());
    if (canDispatch()) {
      workAvailable_.notify_one();
    }
    lock.unlock();
    if (job.read->order == FillOrder::kSequential) {
      runSequential(std::move(job));
    } else {
      fillAndCopy(job);
    }
    job = {};
    lock.lock();
    advance(Clock::now());
    --numRunning_;
  }
}

bool FetchScheduler::fillAndCopy(const Job& job) {
  auto& read = *job.read;
  auto* dst = read.dst + job.offset;
  std::exception_ptr failure;
  uint64_t readNanos{0};
  uint64_t copyNanos{0};
  bool isCopying{false};
  bool isWaiting{false};
  try {
    ensureCudaContextForThread();
    auto buffer = pool_->acquire();
    const auto start = Clock::now();
    // The pool has no page-locked memory to give when the buffer is null:
    // the piece is then staged in pageable memory.
    std::unique_ptr<uint8_t[]> pageable;
    if (buffer == nullptr) {
      pageable.reset(new uint8_t[job.size]);
    }
    read.fill(
        buffer != nullptr ? buffer->data : pageable.get(),
        job.offset,
        job.size);
    const auto copyStart = Clock::now();
    readNanos = nanosBetween(start, copyStart);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      advance(copyStart);
      ++numCopies_;
      isCopying = true;
      // The reader of a sequential range starts its next piece itself, so it
      // gains nothing from going on.
      isWaiting = !isCopyQueued_ || buffer == nullptr ||
          read.order == FillOrder::kSequential;
      if (isWaiting) {
        ++numCopyWaiters_;
      }
    }
    if (buffer == nullptr) {
      copyPageableToDevice(pageable.get(), job.size, dst);
    } else {
      startCopy(*buffer, job.size, dst);
      if (!isWaiting) {
        std::lock_guard<std::mutex> lock(mutex_);
        queuedCopies_.push_back(
            QueuedCopy{job, std::move(buffer), copyStart, readNanos});
        copyQueued_.notify_one();
        return true;
      }
      waitForCopy(*buffer);
    }
    copyNanos = nanosBetween(copyStart, Clock::now());
  } catch (...) {
    failure = std::current_exception();
  }
  // The buffer is back in the pool here, before its reservation is given
  // up, so that the reader that takes the reservation finds a buffer.
  const auto now = Clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  advance(now);
  if (isCopying) {
    --numCopies_;
  }
  if (isWaiting) {
    --numCopyWaiters_;
  }
  if (failure == nullptr) {
    recordPiece(read, job.size, readNanos, copyNanos, false);
  }
  const bool succeeded = failure == nullptr;
  finishPiece(read, now, std::move(failure));
  return succeeded;
}

void FetchScheduler::copyWaiterLoop() {
  folly::setThreadName("CudfFetchCopy");
  QueuedCopy copy;
  std::unique_lock<std::mutex> lock(mutex_);

  while (true) {
    if (queuedCopies_.empty() && copy.job.read != nullptr) {
      // The range is let go without the lock, as the readers do.
      lock.unlock();
      copy = {};
      lock.lock();
    }
    copyQueued_.wait(
        lock, [&]() { return !queuedCopies_.empty() || readersStopped_; });
    if (queuedCopies_.empty()) {
      return;
    }
    auto next = std::move(queuedCopies_.front());
    queuedCopies_.pop_front();
    lock.unlock();
    copy = std::move(next);
    std::exception_ptr failure;
    try {
      ensureCudaContextForThread();
      waitForCopy(*copy.buffer);
    } catch (...) {
      failure = std::current_exception();
    }
    const auto now = Clock::now();
    // Back in the pool before its reservation is given up.
    copy.buffer.reset();
    lock.lock();
    advance(now);
    --numCopies_;
    auto& read = *copy.job.read;
    if (failure == nullptr) {
      recordPiece(
          read,
          copy.job.size,
          copy.readNanos,
          nanosBetween(copy.start, now),
          true);
    }
    finishPiece(read, now, std::move(failure));
  }
}

void FetchScheduler::runSequential(Job job) {
  auto& read = *job.read;
  while (true) {
    const bool succeeded = fillAndCopy(job);
    std::unique_lock<std::mutex> lock(mutex_);
    // A failure and a cancellation both drop the pieces that are left.
    if (!succeeded || read.nextOffset >= read.size) {
      return;
    }
    const auto waitStart = Clock::now();
    advance(waitStart);
    ++numBufferWaiters_;
    bufferFreed_.wait(lock, [&]() {
      return stopping_ || read.nextOffset >= read.size ||
          numBuffersInUse_ < depthLimit();
    });
    const auto now = Clock::now();
    advance(now);
    --numBufferWaiters_;
    if (read.nextOffset >= read.size) {
      return;
    }
    if (stopping_) {
      if (read.failure == nullptr) {
        read.failure = makeShutdownFailure();
      }
      dropRemaining(read);
      completeIfDone(read, now);
      return;
    }
    job.offset = read.nextOffset;
    job.size = std::min(pool_->bufferSize(), read.size - read.nextOffset);
    read.nextOffset += job.size;
    ++read.numOutstanding;
    ++numBuffersInUse_;
    stats_.maxBuffersInUse = std::max(stats_.maxBuffersInUse, numBuffersInUse_);
    const auto queuedNanos = nanosBetween(waitStart, now);
    read.stats.queuedNanos += queuedNanos;
    stats_.queuedNanos += queuedNanos;
  }
}

} // namespace facebook::velox::cudf_velox::connector::hive
