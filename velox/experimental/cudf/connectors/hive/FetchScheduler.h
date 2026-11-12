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

#pragma once

#include "velox/experimental/cudf/connectors/hive/PinnedBufferPool.h"

#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Moves file bytes to the device for every scan that shares it. Ranges are
/// cut into pieces of one staging buffer. Reader threads owned by the
/// scheduler fill page-locked buffers with the pieces and queue their copies
/// to the device on a few streams; a completion thread per stream returns
/// each buffer once its copy has run, so a reader never waits for a copy.
///
/// The number of readers at work is not configured. A controller raises it
/// while ranges wait for a reader and the copy streams sit idle, and lowers
/// it while ranges wait for a staging buffer, which means the copies cannot
/// keep up. The buffers in circulation follow the readers: one being filled
/// and one being copied per reader. Neither depends on the thread that
/// submits a range.
///
/// Ranges belong to groups. The ranges of a group that a consumer is waiting
/// for are served before those nobody waits for yet, and within each class
/// the group that asked first is served first, so the ranges of one consumer
/// arrive together.
///
/// Thread-safe.
class FetchScheduler {
 public:
  /// Bounds the scheduler works within.
  struct Options {
    /// Fewest readers the controller settles on. At least one.
    size_t minReaders{1};
    /// Most readers the controller settles on, which is also the number of
    /// reader threads.
    size_t maxReaders{16};
    /// Readers at work before the first measurement.
    size_t initialReaders{4};
    /// Streams the copies are queued on.
    size_t numCopyStreams{2};
    /// Time with work in flight between two decisions of the controller.
    std::chrono::microseconds controlPeriod{5'000};
  };

  /// What the controller measured between two decisions.
  struct ControlWindow {
    /// Time with a range queued, being read or being copied.
    uint64_t activeNanos{0};
    /// Part of 'activeNanos' with a copy queued or running.
    uint64_t copyBusyNanos{0};
    /// Part of 'activeNanos' with a range waiting while every allowed reader
    /// was reading.
    uint64_t readerLimitedNanos{0};
    /// Part of 'activeNanos' with a range waiting for a staging buffer while
    /// a reader was free.
    uint64_t bufferLimitedNanos{0};
  };

  /// Counters since the scheduler was created.
  struct Stats {
    /// Bytes copied to the device.
    uint64_t bytes{0};
    /// Pieces copied to the device.
    uint64_t numPieces{0};
    /// Time readers spent filling staging buffers, summed over readers.
    uint64_t readNanos{0};
    /// Time ranges waited for their first reader, and their readers for the
    /// staging buffers of later pieces, summed over ranges.
    uint64_t queuedNanos{0};
    /// Time with a range queued, being read or being copied.
    uint64_t activeNanos{0};
    /// Part of 'activeNanos' with a copy queued or running.
    uint64_t copyBusyNanos{0};
    /// Part of 'activeNanos' with a range waiting while every allowed reader
    /// was reading.
    uint64_t readerLimitedNanos{0};
    /// Part of 'activeNanos' with a range waiting for a staging buffer while
    /// a reader was free.
    uint64_t bufferLimitedNanos{0};
    /// Allowed readers integrated over 'activeNanos'. Divided by
    /// 'activeNanos' it is the mean number of allowed readers.
    uint64_t readerNanos{0};
    /// Times the controller raised the number of readers.
    uint64_t numRaises{0};
    /// Times the controller lowered the number of readers.
    uint64_t numLowers{0};
    /// Readers allowed now.
    size_t numReaders{0};
    /// Consumers blocked in wait() now.
    size_t numWaiters{0};
    /// Most readers that filled buffers at the same time.
    size_t maxRunningReaders{0};
    /// Most staging buffers in use at the same time.
    size_t maxBuffersInUse{0};
  };

  /// What one range cost.
  struct ReadStats {
    /// Bytes copied to the device.
    uint64_t bytes{0};
    /// Time readers spent filling staging buffers with the range.
    uint64_t readNanos{0};
    /// Time the range waited for its first reader, and that reader for the
    /// staging buffers of later pieces when one reader fills them in turn.
    uint64_t queuedNanos{0};
    /// Time from queueing a piece's copy to its completion, summed over
    /// pieces.
    uint64_t copyNanos{0};
    /// Share of the time between submission and completion during which a
    /// copy of any range was queued or running, in percent.
    uint64_t copyBusyPercent{0};
    /// Share of that time during which a range waited while every allowed
    /// reader was reading, in percent.
    uint64_t readerLimitedPercent{0};
    /// Share of that time during which a range waited for a staging buffer
    /// while a reader was free, in percent.
    uint64_t bufferLimitedPercent{0};
    /// Readers allowed when the range completed.
    uint64_t numReaders{0};
  };

  /// Reads 'size' bytes that start 'offset' bytes into a range into
  /// 'buffer'. Runs on a reader thread.
  using Fill = std::function<void(uint8_t* buffer, size_t offset, size_t size)>;

  /// How the pieces of a range are filled.
  enum class FillOrder {
    /// Pieces are independent: several readers fill them at once, in any
    /// order. The fill must be thread-safe.
    kAny,
    /// One reader fills the pieces one after the other in increasing offset
    /// order, as reading from a stream needs.
    kSequential,
  };

  /// The ranges of one consumer that are needed together, such as the column
  /// chunks of a split.
  struct Group;

  /// A submitted range.
  class Read;

  /// Returns the number of readers to allow next, given what the controller
  /// measured with 'numReaders' allowed, which must be positive. Lowers by one
  /// when ranges waited for a staging buffer for a quarter of the window, since
  /// the copies then cannot keep up with the readers. Otherwise raises by one
  /// when ranges waited for a reader for a quarter of the window and the copy
  /// streams were idle for a tenth of it.
  static size_t nextNumReaders(
      const Options& options,
      size_t numReaders,
      const ControlWindow& window);

  /// 'pool' supplies the staging buffers and bounds how many are in use.
  FetchScheduler(
      std::shared_ptr<PinnedBufferPool> pool,
      const Options& options);

  /// Fails the ranges still queued, waits for the pieces in flight and stops
  /// the threads. Every range must have been waited for or cancelled.
  ~FetchScheduler();

  FetchScheduler(const FetchScheduler&) = delete;
  FetchScheduler& operator=(const FetchScheduler&) = delete;

  /// Returns the pool of staging buffers.
  const std::shared_ptr<PinnedBufferPool>& pool() const {
    return pool_;
  }

  /// Creates a group to submit ranges under.
  std::shared_ptr<Group> makeGroup();

  /// Queues the copy of a range of 'size' bytes to 'dst' on the device.
  /// 'fill' is called once per piece of at most one staging buffer. 'dst'
  /// and everything 'fill' refers to must stay valid until wait() or
  /// cancel() returns for the range; 'fill' itself is destroyed by that
  /// call, on its thread. Requires cuDF to be registered. Throws when the
  /// threads cannot be started, after which a later call tries again.
  std::shared_ptr<Read> submit(
      const std::shared_ptr<Group>& group,
      size_t size,
      uint8_t* dst,
      FillOrder order,
      Fill fill);

  /// Waits until the bytes of 'read' are on the device and returns what the
  /// range cost. Rethrows the first failure of a fill or a copy after every
  /// piece in flight has finished. Moves the group of 'read' ahead of the
  /// groups nobody waits for.
  ReadStats wait(Read& read);

  /// Drops the pieces of 'read' that no reader has started and waits for the
  /// rest, after which nothing refers to the range's destination or fill.
  void cancel(Read& read) noexcept;

  /// Returns the counters, brought up to now.
  Stats stats();

 private:
  using Clock = std::chrono::steady_clock;

  // A piece of a range handed to a reader.
  struct Job {
    std::shared_ptr<Read> read;
    size_t offset{0};
    size_t size{0};
  };

  // A copy queued on a stream.
  struct PendingCopy {
    PinnedBufferPool::Buffer buffer;
    std::shared_ptr<Read> read;
    size_t size{0};
    Clock::time_point queuedAt;
  };

  // A stream that copies staging buffers to the device, and the copies
  // queued on it in stream order.
  struct CopyStream {
    cudaStream_t stream{nullptr};
    std::mutex mutex;
    std::condition_variable queued;
    std::deque<PendingCopy> pending;
    bool stopping{false};
    std::thread completer;
  };

  // Starts the threads and creates the copy streams on first use. Requires
  // 'mutex_' and a CUDA context. Throws, with nothing left running, when the
  // streams, a completion thread or the first reader cannot be created.
  void ensureStarted();

  // Stops and joins the completion threads of 'copyStreams' and destroys
  // their streams.
  static void stopCopyStreams(
      std::vector<std::unique_ptr<CopyStream>>& copyStreams);

  // Body of a reader thread.
  void readerLoop();

  // Body of the completion thread of 'copyStream'.
  void completerLoop(CopyStream& copyStream);

  // Charges the time since the previous call to the state the scheduler was
  // in, and lets the controller decide when a window is full. Called with
  // 'mutex_' held before every change of that state.
  void advance(Clock::time_point now);

  // Staging buffers that may be in use at once with the current number of
  // readers.
  size_t depthLimit() const;

  // Whether a reader may take a piece now.
  bool canDispatch() const;

  // Takes the next piece of the group to serve first, reserving a reader and
  // a staging buffer for it. Requires canDispatch().
  Job takeJob(Clock::time_point now);

  // Fills a staging buffer with the piece 'job' and queues its copy. The
  // buffer and piece reserved for the job are given up by the copy's
  // completion, or here when the piece fails. Returns false when it failed.
  bool fillAndCopy(const Job& job);

  // Adds a piece that reached the device to the counters. Requires 'mutex_'.
  void
  recordPiece(Read& read, size_t size, uint64_t readNanos, uint64_t copyNanos);

  // Fills the pieces of a sequential range one after the other, starting
  // with 'job'.
  void runSequential(Job job);

  // Queues the copy of the first 'job.size' bytes of 'buffer', which took
  // 'readNanos' to fill.
  void queueCopy(
      const Job& job,
      PinnedBufferPool::Buffer buffer,
      uint64_t readNanos);

  // Accounts for a piece that is on the device or has failed: gives up its
  // staging buffer reservation and wakes whoever waits for that or for the
  // range. Requires 'mutex_'.
  void
  finishPiece(Read& read, Clock::time_point now, std::exception_ptr failure);

  // Forgets the pieces of 'read' no reader has started. Requires 'mutex_'.
  void dropRemaining(Read& read);

  // Marks 'read' complete and wakes its waiters once no piece is left to
  // start and none is in flight. Requires 'mutex_'.
  void completeIfDone(Read& read, Clock::time_point now);

  const std::shared_ptr<PinnedBufferPool> pool_;
  // 'maxReaders' is lowered to the number of reader threads that could be
  // started.
  Options options_;

  std::mutex mutex_;
  // Signalled when a reader may be able to take a piece.
  std::condition_variable workAvailable_;
  // Signalled when a staging buffer reservation is given up.
  std::condition_variable bufferFreed_;
  // Signalled when a range completes.
  std::condition_variable completed_;

  bool started_{false};
  bool stopping_{false};
  std::vector<std::thread> readers_;
  std::vector<std::unique_ptr<CopyStream>> copyStreams_;
  // Stream the next copy goes to.
  size_t nextCopyStream_{0};

  // Groups with ranges that have pieces no reader has started.
  std::vector<Group*> queuedGroups_;
  // Arrival number given to the next group that queues.
  uint64_t nextSequence_{0};
  // Readers allowed to fill buffers at once.
  size_t numReaders_{1};
  // Readers filling a buffer or holding a sequential range.
  size_t numRunning_{0};
  // Staging buffers reserved: being filled or being copied.
  size_t numBuffersInUse_{0};
  // Copies queued or running.
  size_t numCopies_{0};
  // Readers of sequential ranges waiting for a staging buffer.
  size_t numBufferWaiters_{0};
  // Consumers blocked in wait().
  size_t numWaiters_{0};

  Clock::time_point lastAdvance_;
  ControlWindow window_;
  Stats stats_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
