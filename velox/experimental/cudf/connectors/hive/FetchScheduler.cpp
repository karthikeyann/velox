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
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/FetchScheduler.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/utilities/error.hpp>

#include <cuda_runtime.h>

#include <folly/system/ThreadName.h>
#include <glog/logging.h>
#include <sched.h>

#include <algorithm>
#include <deque>
#include <exception>
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

// Copies the first 'size' bytes of 'buffer' to 'dst' and waits for the copy,
// after which the buffer can be reused. The calling thread sleeps meanwhile:
// a copy whose submitter keeps running was measured to take longer.
void copyToDevice(
    const PinnedBufferPool::StagingBuffer& buffer,
    size_t size,
    uint8_t* dst) {
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      dst, buffer.data, size, cudaMemcpyHostToDevice, buffer.stream));
  auto status = cudaEventRecord(buffer.event, buffer.stream);
  if (status == cudaSuccess) {
    status = cudaEventSynchronize(buffer.event);
  }
  if (status != cudaSuccess) {
    // The copy is queued and reads the buffer, so the buffer cannot go back
    // to its pool before the stream has drained.
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
    size_t freeBytes{0};
    size_t totalBytes{0};
    CUDF_CUDA_TRY(cudaMemGetInfo(&freeBytes, &totalBytes));
    auto capacity = totalBytes / 16;
    // With a pool or an arena the device's free memory says how much the
    // process has not reserved, which is unrelated to what it can use.
    const auto& memoryResource = CudfConfig::getInstance().memoryResource;
    const bool isPooled = memoryResource.find("pool") != std::string::npos ||
        memoryResource.find("arena") != std::string::npos;
    if (!isPooled) {
      capacity = std::min(capacity, freeBytes / 4);
    }
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
  // Set once done() holds and the waiters have been told.
  bool completed{false};
  // First failure of a fill or a copy.
  std::exception_ptr failure;
  ReadStats stats;
};

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
  using Phase = ControlState::Phase;
  const auto hold = [&](size_t numReaders) {
    next.phase = Phase::kHold;
    next.numReaders = numReaders;
    next.numHeldReaders = numReaders;
    next.numHoldsLeft = options.numHoldPeriods;
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
      const bool canTryMore = state.numReaders < options.maxReaders;
      const bool canTryFewer = state.numReaders > options.minReaders;
      if (canTryMore && !isCopyLimited && (state.tryMoreNext || !canTryFewer)) {
        tryMore(state.numReaders);
      } else if (canTryFewer) {
        tryFewer(state.numReaders);
      }
      break;
    }
    case Phase::kTryMore: {
      // The share of the rate that the added readers would bring if every
      // reader brought as much as the held ones.
      const double share =
          static_cast<double>(state.numReaders - state.numHeldReaders) /
          state.numHeldReaders;
      if (rate >= state.baseRate * (1 + share / 2) && !isCopyLimited) {
        next.baseRate = rate;
        tryMore(state.numReaders);
      } else {
        next.tryMoreNext = false;
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
        tryFewer(state.numReaders);
      } else {
        next.tryMoreNext = true;
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
      options_.initialReaders, options_.minReaders, options_.maxReaders);
  control_.numHeldReaders = control_.numReaders;
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
  options_.maxReaders = readers_.size();
  options_.minReaders = std::min(options_.minReaders, options_.maxReaders);
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
    queuedGroups_.push_back(group.get());
  }
  group->pending.push_back(read);
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

FetchScheduler::Stats FetchScheduler::stats() {
  std::lock_guard<std::mutex> lock(mutex_);
  advance(Clock::now());
  auto stats = stats_;
  stats.numReaders = control_.numReaders;
  stats.numWaiters = numWaiters_;
  return stats;
}

size_t FetchScheduler::depthLimit() const {
  return std::max<size_t>(
      1, std::min(pool_->maxBuffers(), control_.numReaders));
}

bool FetchScheduler::canDispatch() const {
  return !queuedGroups_.empty() && numRunning_ < control_.numReaders &&
      numBuffersInUse_ < depthLimit();
}

void FetchScheduler::advance(Clock::time_point now) {
  const auto elapsed = nanosBetween(lastAdvance_, now);
  lastAdvance_ = std::max(lastAdvance_, now);
  const bool hasQueued = !queuedGroups_.empty();
  if (!hasQueued && numRunning_ == 0) {
    return;
  }
  stats_.activeNanos += elapsed;
  stats_.readerNanos += elapsed * control_.numReaders;
  stats_.readerBusyNanos += elapsed * numRunning_;
  stats_.copyWaitNanos += elapsed * numCopies_;
  if (numCopies_ > 0) {
    stats_.copyBusyNanos += elapsed;
  }
  if (numBufferWaiters_ > 0 ||
      (hasQueued && numRunning_ < control_.numReaders &&
       numBuffersInUse_ >= depthLimit())) {
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
  window_.copyWaitNanos += elapsed * numCopies_;
  uint64_t periodNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             options_.controlPeriod)
                             .count();
  // The rate of one period depends on what is being read. A trial is
  // measured for twice as long, since a wrong verdict stays for a hold.
  if (control_.phase != ControlState::Phase::kHold) {
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
    if (waited != bestWaited ? waited : group->sequence < bestGroup->sequence) {
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
  if (read.order == FillOrder::kSequential || read.nextOffset >= read.size) {
    group->pending.pop_front();
    if (group->pending.empty()) {
      queuedGroups_.erase(queuedGroups_.begin() + best);
    }
  }
  ++read.numOutstanding;
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
  read.stats.numReaders = control_.numReaders;
  completed_.notify_all();
}

void FetchScheduler::recordPiece(
    Read& read,
    size_t size,
    uint64_t readNanos,
    uint64_t copyNanos) {
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
    workAvailable_.wait(lock, [&]() { return stopping_ || canDispatch(); });
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
    }
    if (buffer != nullptr) {
      copyToDevice(*buffer, job.size, dst);
    } else {
      copyPageableToDevice(pageable.get(), job.size, dst);
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
  if (failure == nullptr) {
    recordPiece(read, job.size, readNanos, copyNanos);
  }
  const bool succeeded = failure == nullptr;
  finishPiece(read, now, std::move(failure));
  return succeeded;
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
