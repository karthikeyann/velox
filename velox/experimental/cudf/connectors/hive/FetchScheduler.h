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

#include "velox/experimental/cudf/connectors/hive/FetchAheadBudget.h"
#include "velox/experimental/cudf/connectors/hive/PinnedBufferPool.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Moves file bytes to the device for every scan that shares it. Ranges are
/// cut into pieces of one staging buffer. A reader thread owned by the
/// scheduler fills a page-locked buffer with a piece, copies it to the device
/// on the buffer's stream and sleeps until that copy has run.
///
/// The number of readers at work is not configured. A controller measures
/// the bytes per second that reach the device while ranges wait for a
/// reader, and tries one reader more or one fewer from time to time: it
/// keeps an added reader that raises the rate and gives up one whose absence
/// does not lower it. The result does not depend on the thread that submits
/// a range.
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
    /// reader threads. Zero stands for half of the CPUs the process may run
    /// on, divided by the number of devices it sees, between 4 and 64.
    size_t maxReaders{0};
    /// Readers at work before the first measurement.
    size_t initialReaders{8};
    /// Time with ranges waiting for a reader that one measurement of the
    /// controller covers while it holds a number of readers. The measurement
    /// of a trial covers twice as much.
    std::chrono::microseconds controlPeriod{10'000};
    /// Measurements the controller stays with a number of readers before it
    /// tries another.
    size_t numHoldPeriods{8};
    /// Capacity in bytes of a fetch-ahead budget of the scheduler's own.
    /// Zero makes the scheduler use the budget that all schedulers of the
    /// process share for the device: one sixteenth of the device's memory,
    /// and no more than a quarter of what is free when the budget is
    /// created.
    uint64_t fetchAheadBytes{0};
  };

  /// What the controller measured in one control period.
  struct ControlWindow {
    /// Time with a range waiting for a reader.
    uint64_t saturatedNanos{0};
    /// Bytes that reached the device in that time.
    uint64_t bytes{0};
  };

  /// Where the controller stands between two measurements.
  struct ControlState {
    /// What the readers allowed now are.
    enum class Phase {
      /// The number the controller stays with.
      kHold,
      /// One more than that number, on trial.
      kTryMore,
      /// One fewer than that number, on trial.
      kTryFewer,
    };

    /// Readers allowed now.
    size_t numReaders{1};
    Phase phase{Phase::kHold};
    /// Bytes per second measured with the number the controller stays
    /// with. Zero before the first measurement.
    double baseRate{0};
    /// Measurements left before the next trial.
    size_t numHoldsLeft{0};
    /// Whether the next trial adds a reader. Trials alternate.
    bool tryMoreNext{true};
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
    /// was at work.
    uint64_t readerLimitedNanos{0};
    /// Part of 'activeNanos' with a range waiting for a staging buffer while
    /// a reader was free.
    uint64_t bufferLimitedNanos{0};
    /// Time readers were at work, summed over readers.
    uint64_t readerBusyNanos{0};
    /// Part of 'readerBusyNanos' spent waiting for copies.
    uint64_t copyWaitNanos{0};
    /// Time with a range waiting for a reader, which is the time the
    /// controller measures over.
    uint64_t saturatedNanos{0};
    /// Bytes that reached the device in that time.
    uint64_t saturatedBytes{0};
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
    /// Most readers at work at the same time.
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
    /// Time readers waited for the copies of the range's pieces.
    uint64_t copyNanos{0};
    /// Share of the time between submission and completion during which a
    /// copy of any range was queued or running, in percent.
    uint64_t copyBusyPercent{0};
    /// Share of that time during which a range waited while every allowed
    /// reader was at work, in percent.
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

  /// Returns where the controller stands after measuring 'window' in
  /// 'state'. While it holds a number of readers it averages the rate, and
  /// after the hold periods it tries one more reader or one fewer, in turn.
  /// One more is kept when the rate rises by at least half of that reader's
  /// share, 1 / (2 x readers), and then the next one is tried right away.
  /// One fewer is kept when the rate falls by less than a quarter of that
  /// share. A trial that is not kept goes back to the held number.
  static ControlState nextControlState(
      const Options& options,
      const ControlState& state,
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

  /// Returns the budget of device memory for column chunks that scans have
  /// fetched and not yet decoded: the scheduler's own, or the one shared by
  /// the process. Requires cuDF to be registered when the budget is shared.
  FetchAheadBudget& fetchAhead();

  /// Creates a group to submit ranges under.
  std::shared_ptr<Group> makeGroup();

  /// Queues the copy of a range of 'size' bytes to 'dst' on the device.
  /// 'fill' is called once per piece of at most one staging buffer. 'dst'
  /// and everything 'fill' refers to must stay valid until wait() or
  /// cancel() returns for the range; 'fill' itself is destroyed by that
  /// call, on its thread. Requires cuDF to be registered. Throws when no
  /// reader thread can be started, after which a later call tries again.
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

  // Starts the reader threads on first use. Requires 'mutex_'. Throws, with
  // nothing started, when not even one thread can be started.
  void ensureStarted();

  // Body of a reader thread.
  void readerLoop();

  // Charges the time since the previous call to the state the scheduler was
  // in, and lets the controller decide when a window is full. Called with
  // 'mutex_' held before every change of that state.
  void advance(Clock::time_point now);

  // Staging buffers that may be in use at once.
  size_t depthLimit() const;

  // Whether a reader may take a piece now.
  bool canDispatch() const;

  // Takes the next piece of the group to serve first, reserving a reader and
  // a staging buffer for it. Requires canDispatch().
  Job takeJob(Clock::time_point now);

  // Fills a staging buffer with the piece 'job', copies it to the device and
  // waits for the copy, then gives up the buffer reserved for the job.
  // Returns false when the piece failed.
  bool fillAndCopy(const Job& job);

  // Fills the pieces of a sequential range one after the other, starting
  // with 'job'.
  void runSequential(Job job);

  // Adds a piece that reached the device to the counters. Requires 'mutex_'.
  void
  recordPiece(Read& read, size_t size, uint64_t readNanos, uint64_t copyNanos);

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

  // Created by the first call of fetchAhead().
  std::once_flag fetchAheadOnce_;
  std::shared_ptr<FetchAheadBudget> fetchAhead_;

  const std::shared_ptr<PinnedBufferPool> pool_;
  // 'maxReaders' is lowered to the number of threads that could be started.
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

  // Groups with ranges that have pieces no reader has started.
  std::vector<Group*> queuedGroups_;
  // Arrival number given to the next group that queues.
  uint64_t nextSequence_{0};
  // The controller's state. Its 'numReaders' is the number of readers
  // allowed to work at once.
  ControlState control_;
  // Readers filling a buffer, waiting for its copy or holding a sequential
  // range.
  size_t numRunning_{0};
  // Staging buffers reserved: being filled or being copied.
  size_t numBuffersInUse_{0};
  // Readers waiting for the copy of their buffer.
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
