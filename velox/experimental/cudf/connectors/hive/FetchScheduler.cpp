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

#include <folly/ScopeGuard.h>
#include <folly/system/ThreadName.h>
#include <glog/logging.h>

#include <algorithm>
#include <exception>
#include <optional>
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
size_t FetchScheduler::nextNumReaders(
    const Options& options,
    size_t numReaders,
    const ControlWindow& window) {
  VELOX_CHECK_GT(numReaders, 0);
  if (window.activeNanos == 0) {
    return numReaders;
  }
  if (window.bufferLimitedNanos * 4 >= window.activeNanos) {
    return std::max(options.minReaders, numReaders - 1);
  }
  const auto copyIdleNanos = window.activeNanos - window.copyBusyNanos;
  if (window.readerLimitedNanos * 4 >= window.activeNanos &&
      copyIdleNanos * 10 >= window.activeNanos) {
    return std::min(options.maxReaders, numReaders + 1);
  }
  return numReaders;
}

FetchScheduler::FetchScheduler(
    std::shared_ptr<PinnedBufferPool> pool,
    const Options& options)
    : pool_(std::move(pool)), options_(options), lastAdvance_(Clock::now()) {
  VELOX_CHECK_NOT_NULL(pool_);
  VELOX_CHECK_GT(options_.minReaders, 0);
  VELOX_CHECK_LE(options_.minReaders, options_.maxReaders);
  VELOX_CHECK_GT(options_.numCopyStreams, 0);
  numReaders_ = std::clamp(
      options_.initialReaders, options_.minReaders, options_.maxReaders);
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
  // The completion threads finish the copies already queued, then destroy
  // their streams.
  stopCopyStreams(copyStreams_);
}

// static
void FetchScheduler::stopCopyStreams(
    std::vector<std::unique_ptr<CopyStream>>& copyStreams) {
  for (auto& copyStream : copyStreams) {
    if (!copyStream->completer.joinable()) {
      // Its completion thread never started, so nothing else destroys the
      // stream.
      cudaStreamDestroy(copyStream->stream);
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(copyStream->mutex);
      copyStream->stopping = true;
    }
    copyStream->queued.notify_all();
    copyStream->completer.join();
  }
  copyStreams.clear();
}

std::shared_ptr<FetchScheduler::Group> FetchScheduler::makeGroup() {
  return std::make_shared<Group>();
}

void FetchScheduler::ensureStarted() {
  if (started_) {
    return;
  }
  // All or nothing up to the first reader: a failure leaves no thread and no
  // stream behind, so that the next submission starts over.
  std::vector<std::unique_ptr<CopyStream>> copyStreams;
  try {
    for (size_t i = 0; i < options_.numCopyStreams; ++i) {
      auto copyStream = std::make_unique<CopyStream>();
      CUDF_CUDA_TRY(cudaStreamCreateWithFlags(
          &copyStream->stream, cudaStreamNonBlocking));
      copyStreams.push_back(std::move(copyStream));
    }
    for (auto& copyStream : copyStreams) {
      copyStream->completer = std::thread(
          [this, &stream = *copyStream]() { completerLoop(stream); });
    }
    readers_.reserve(options_.maxReaders);
    readers_.emplace_back([this]() { readerLoop(); });
  } catch (...) {
    stopCopyStreams(copyStreams);
    throw;
  }
  copyStreams_ = std::move(copyStreams);
  while (readers_.size() < options_.maxReaders) {
    try {
      readers_.emplace_back([this]() { readerLoop(); });
    } catch (const std::exception& e) {
      // The scheduler works with the readers it has.
      LOG(WARNING) << "Fetch scheduler started " << readers_.size() << " of "
                   << options_.maxReaders << " reader threads: " << e.what();
      break;
    }
  }
  options_.maxReaders = readers_.size();
  options_.minReaders = std::min(options_.minReaders, options_.maxReaders);
  numReaders_ = std::min(numReaders_, options_.maxReaders);
  started_ = true;
}

std::shared_ptr<FetchScheduler::Read> FetchScheduler::submit(
    const std::shared_ptr<Group>& group,
    size_t size,
    uint8_t* dst,
    FillOrder order,
    Fill fill) {
  VELOX_CHECK_NOT_NULL(group);
  ensureCudaContextForThread();
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
  // Destroyed after the lock is released.
  Fill fill;
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

FetchScheduler::Stats FetchScheduler::stats() {
  std::lock_guard<std::mutex> lock(mutex_);
  advance(Clock::now());
  auto stats = stats_;
  stats.numReaders = numReaders_;
  stats.numWaiters = numWaiters_;
  return stats;
}

size_t FetchScheduler::depthLimit() const {
  return std::max<size_t>(
      1,
      std::min(pool_->maxBuffers(), 2 * numReaders_ + options_.numCopyStreams));
}

bool FetchScheduler::canDispatch() const {
  return !queuedGroups_.empty() && numRunning_ < numReaders_ &&
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
  window_.activeNanos += elapsed;
  stats_.readerNanos += elapsed * numReaders_;
  if (numCopies_ > 0) {
    stats_.copyBusyNanos += elapsed;
    window_.copyBusyNanos += elapsed;
  }
  if (numBufferWaiters_ > 0 ||
      (hasQueued && numRunning_ < numReaders_ &&
       numBuffersInUse_ >= depthLimit())) {
    stats_.bufferLimitedNanos += elapsed;
    window_.bufferLimitedNanos += elapsed;
  } else if (hasQueued && numRunning_ >= numReaders_) {
    stats_.readerLimitedNanos += elapsed;
    window_.readerLimitedNanos += elapsed;
  }

  const uint64_t periodNanos =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          options_.controlPeriod)
          .count();
  if (window_.activeNanos < periodNanos) {
    return;
  }
  const auto numReaders = nextNumReaders(options_, numReaders_, window_);
  window_ = {};
  if (numReaders > numReaders_) {
    ++stats_.numRaises;
    numReaders_ = numReaders;
    workAvailable_.notify_one();
    bufferFreed_.notify_all();
  } else if (numReaders < numReaders_) {
    ++stats_.numLowers;
    numReaders_ = numReaders;
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
  read.stats.numReaders = numReaders_;
  completed_.notify_all();
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
  std::exception_ptr failure;
  try {
    ensureCudaContextForThread();
    auto buffer = pool_->acquire();
    const auto start = Clock::now();
    if (buffer != nullptr) {
      read.fill(buffer->data, job.offset, job.size);
      queueCopy(job, std::move(buffer), nanosBetween(start, Clock::now()));
      return true;
    }
    // The pool has no page-locked memory to give: stage the piece in
    // pageable memory and copy it before that memory goes away.
    const std::unique_ptr<uint8_t[]> pageable(new uint8_t[job.size]);
    read.fill(pageable.get(), job.offset, job.size);
    const auto copyStart = Clock::now();
    // A copy from pageable memory can return while a short piece is still on
    // its way to the device, so the stream is drained as well.
    const auto stream = copyStreams_.front()->stream;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        read.dst + job.offset,
        pageable.get(),
        job.size,
        cudaMemcpyHostToDevice,
        stream));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream));
    const auto now = Clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    advance(now);
    recordPiece(
        read,
        job.size,
        nanosBetween(start, copyStart),
        nanosBetween(copyStart, now));
    finishPiece(read, now, nullptr);
    return true;
  } catch (...) {
    failure = std::current_exception();
  }
  const auto now = Clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  advance(now);
  finishPiece(read, now, std::move(failure));
  return false;
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

void FetchScheduler::queueCopy(
    const Job& job,
    PinnedBufferPool::Buffer buffer,
    uint64_t readNanos) {
  auto& read = *job.read;
  const auto queuedAt = Clock::now();
  CopyStream* copyStream{nullptr};
  {
    std::lock_guard<std::mutex> lock(mutex_);
    advance(queuedAt);
    ++numCopies_;
    read.stats.readNanos += readNanos;
    stats_.readNanos += readNanos;
    copyStream = copyStreams_[nextCopyStream_++ % copyStreams_.size()].get();
  }
  cudaError_t status{cudaSuccess};
  {
    // Held across the two CUDA calls so that the queue below is in stream
    // order.
    std::lock_guard<std::mutex> lock(copyStream->mutex);
    status = cudaMemcpyAsync(
        read.dst + job.offset,
        buffer->data,
        job.size,
        cudaMemcpyHostToDevice,
        copyStream->stream);
    if (status == cudaSuccess) {
      status = cudaEventRecord(buffer->event, copyStream->stream);
      if (status != cudaSuccess) {
        // The copy is queued and reads the buffer, so the buffer cannot go
        // back to its pool before the stream has drained.
        cudaStreamSynchronize(copyStream->stream);
      }
    }
    if (status == cudaSuccess) {
      copyStream->pending.push_back(
          PendingCopy{std::move(buffer), job.read, job.size, queuedAt});
    }
  }
  if (status != cudaSuccess) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      advance(Clock::now());
      --numCopies_;
    }
    CUDF_CUDA_TRY(status);
  }
  copyStream->queued.notify_one();
}

void FetchScheduler::completerLoop(CopyStream& copyStream) {
  folly::setThreadName("CudfFetchCopy");
  try {
    ensureCudaContextForThread();
  } catch (const std::exception&) {
    // Reported by the first copy this thread completes.
  }
  // The stream is destroyed here, on a thread bound to its device, because
  // the thread destroying the scheduler may have no CUDA context.
  SCOPE_EXIT {
    cudaStreamDestroy(copyStream.stream);
    // Clear any error so that an unrelated CUDA call does not report it. At
    // process exit the CUDA runtime can be gone already, and it has released
    // the stream itself.
    cudaGetLastError();
  };
  while (true) {
    std::optional<PendingCopy> next;
    {
      std::unique_lock<std::mutex> lock(copyStream.mutex);
      copyStream.queued.wait(lock, [&]() {
        return copyStream.stopping || !copyStream.pending.empty();
      });
      if (copyStream.pending.empty()) {
        return;
      }
      next.emplace(std::move(copyStream.pending.front()));
      copyStream.pending.pop_front();
    }
    auto& copy = *next;
    std::exception_ptr failure;
    try {
      ensureCudaContextForThread();
      // Copies on a fast link are done by the time they are looked at, and
      // asking is cheaper than blocking.
      auto status = cudaEventQuery(copy.buffer->event);
      if (status == cudaErrorNotReady) {
        status = cudaEventSynchronize(copy.buffer->event);
      }
      if (status != cudaSuccess) {
        // The buffer cannot go back to its pool while the copy may still
        // read it.
        cudaStreamSynchronize(copyStream.stream);
        CUDF_CUDA_TRY(status);
      }
    } catch (...) {
      failure = std::current_exception();
    }
    const auto now = Clock::now();
    // Back in the pool before its reservation is given up, so that the
    // reader that takes the reservation finds a buffer.
    copy.buffer.reset();
    std::lock_guard<std::mutex> lock(mutex_);
    advance(now);
    --numCopies_;
    if (failure == nullptr) {
      recordPiece(*copy.read, copy.size, 0, nanosBetween(copy.queuedAt, now));
    }
    finishPiece(*copy.read, now, std::move(failure));
  }
}

} // namespace facebook::velox::cudf_velox::connector::hive
