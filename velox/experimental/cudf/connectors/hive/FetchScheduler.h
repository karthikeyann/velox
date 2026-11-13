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
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Moves file bytes to the device for every scan that shares it. Ranges are
/// cut into pieces of one staging buffer. A reader thread owned by the
/// scheduler fills a page-locked buffer with a piece and copies it to the
/// device on the buffer's stream. Either the reader then waits until that
/// copy has run, or it goes on to its next piece and one thread of the
/// scheduler waits for the copies in the order they were started.
///
/// Neither the number of readers at work nor the way copies are waited for
/// is configured. Which way of waiting delivers more depends on the machine:
/// where filling buffers slows the copies that run meanwhile, a reader that
/// waits delivers more; where it does not, a reader that waits is idle for
/// nothing. The scheduler measures a copy against a fill when its readers
/// start and chooses by that. A controller then measures the bytes per
/// second that reach the device while ranges wait for a reader, and from
/// time to time tries more readers, fewer readers or the other way of
/// waiting, and keeps what raises the rate. The result does not depend on
/// the thread that submits a range.
///
/// Ranges belong to groups. The ranges of a group that a consumer is waiting
/// for are served before those nobody waits for yet. Among the groups waited
/// for, one with at most a quarter of the bytes of another left to read goes
/// ahead of it, so that a small scan, such as the build side of a join, is
/// not held up behind a large one. A group that later ones have been served
/// ahead of for 'maxOvertakenBytes' goes first in turn, so a large group is
/// delayed and not starved. Otherwise groups are served one after the other
/// in the order they asked, so the ranges of one consumer arrive together.
///
/// Thread-safe.
class FetchScheduler {
 public:
  /// How the copy of a filled staging buffer is waited for.
  enum class CopyMode {
    /// Chosen by prefersQueuedCopies() from a measurement made when the
    /// readers start. From time to time the controller measures the other
    /// way against it and changes when that delivers clearly more.
    kAuto,
    /// The reader that filled the buffer waits for its copy before it fills
    /// another.
    kWait,
    /// The reader goes on to its next piece, and a thread of the scheduler
    /// waits for the copies.
    kQueue,
  };

  /// Bounds the scheduler works within.
  struct Options {
    /// Fewest readers the controller settles on. At least one.
    size_t minReaders{1};
    /// Most readers the controller settles on, which is also the number of
    /// reader threads. Zero stands for half of the CPUs the process may run
    /// on, divided by the number of devices it sees, between 4 and 64.
    size_t maxReaders{0};
    /// Readers at work before the first measurement.
    size_t initialReaders{16};
    /// Time with ranges waiting for a reader that one measurement of the
    /// controller covers. The measurement of a trial of a number of readers
    /// covers twice as much.
    std::chrono::microseconds controlPeriod{10'000};
    /// Measurements the controller stays with a number of readers before it
    /// tries another.
    size_t numHoldPeriods{3};
    /// How copies are waited for. Anything but kAuto pins it, which is
    /// meant for measurements.
    CopyMode copyMode{CopyMode::kAuto};
    /// Bytes of groups that asked later that are served ahead of a group a
    /// consumer waits for, before that group goes first.
    uint64_t maxOvertakenBytes{256ULL << 20};
    /// Capacity in bytes of a fetch-ahead budget of the scheduler's own.
    /// Zero makes the scheduler use the budget that all schedulers of the
    /// process share for the device: one sixteenth of the device's memory.
    uint64_t fetchAheadBytes{0};
  };

  /// What the controller measured in one control period.
  struct ControlWindow {
    /// Time with a range waiting for a reader.
    uint64_t saturatedNanos{0};
    /// Bytes that reached the device in that time.
    uint64_t bytes{0};
    /// Time readers were at work in that time, filling a buffer or waiting
    /// for its copy, summed over readers.
    uint64_t readerBusyNanos{0};
    /// Part of 'readerBusyNanos' spent waiting for copies.
    uint64_t copyWaitNanos{0};
    /// Part of 'saturatedNanos' with a reader kept from its next piece
    /// because every staging buffer it may use was being filled or copied.
    uint64_t bufferLimitedNanos{0};
  };

  /// Where the controller stands between two measurements.
  struct ControlState {
    /// What the readers allowed now are.
    enum class Phase {
      /// The number the controller stays with.
      kHold,
      /// More than that number, on trial.
      kTryMore,
      /// Fewer than that number, on trial.
      kTryFewer,
      /// That number with the other way of waiting for copies, on trial.
      kTryOtherCopy,
      /// Back to the way of waiting for copies it came from, to measure it
      /// again after a trial of the other way that delivered more.
      kCheckCopyTrial,
      /// The other way once more, after the way it came from delivered less
      /// again.
      kConfirmOtherCopy,
    };

    /// Readers allowed now.
    size_t numReaders{1};
    /// The number the controller stays with, to which it returns from a
    /// trial that is not kept. Equal to 'numReaders' while it holds.
    size_t numHeldReaders{1};
    Phase phase{Phase::kHold};
    /// Bytes per second measured with the number the controller stays
    /// with. Zero before the first measurement.
    double baseRate{0};
    /// Bytes per second of the previous measurement during a trial of the
    /// other way of waiting for copies, which the next one is compared
    /// with.
    double copyTrialRate{0};
    /// Measurements left before the next trial.
    size_t numHoldsLeft{0};
    /// Trials of more readers that were not kept since the last one that
    /// was. The hold doubles with each, up to eight times.
    size_t numFailedTrials{0};
    /// Whether readers go on while their copies run, instead of waiting for
    /// them.
    bool isCopyQueued{false};
    /// Trials of a number of readers left before the next trial of the
    /// other way of waiting for copies.
    size_t numTrialsToCopyTrial{0};
    /// Trials of a number of readers between two trials of the other way of
    /// waiting for copies. Grows fourfold, up to kMaxCopyTrialInterval, with
    /// every such trial.
    size_t copyTrialInterval{1};
  };

  /// Most trials of a number of readers between two trials of the other way
  /// of waiting for copies.
  static constexpr size_t kMaxCopyTrialInterval{64};

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
    /// Pieces whose reader went on while the copy ran.
    uint64_t numQueuedCopies{0};
    /// Times the controller changed how copies are waited for, trials
    /// included.
    uint64_t numCopyModeChanges{0};
    /// Whether readers go on while their copies run now.
    bool isCopyQueued{false};
    /// Time a staging buffer takes to copy to the device over the time it
    /// takes to fill from memory, measured when the readers started with
    /// CopyMode::kAuto. Zero when not measured.
    double copyToFillRatio{0};
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
    /// Time from the start of the copies of the range's pieces until each
    /// was seen to have run, summed over pieces.
    uint64_t copyNanos{0};
    /// Share of the range's pieces whose reader went on while the copy ran,
    /// in percent.
    uint64_t copyQueuedPercent{0};
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
    /// The scheduler's Stats::copyToFillRatio, in percent.
    uint64_t copyToFillPercent{0};
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

  /// Below this ratio of the time a staging buffer takes to copy to the
  /// device to the time it takes to fill from memory, readers go on while
  /// their copies run.
  static constexpr double kMaxQueuedCopyToFillRatio{0.5};

  /// Returns whether readers should go on while their copies run, given
  /// 'copyToFillRatio': the time a staging buffer takes to copy to the
  /// device over the time it takes to fill from memory. A link that is slow
  /// for the machine's memory, a ratio near one as measured on a PCIe 4
  /// workstation, is fed by two or three readers, and what the others fill
  /// meanwhile takes memory bandwidth from the copies: there readers wait.
  /// A fast link, a low ratio, needs many readers and leaves bandwidth to
  /// spare, so a reader that waits is idle for nothing. Zero, for a ratio
  /// that could not be measured, returns false.
  static bool prefersQueuedCopies(double copyToFillRatio);

  /// Returns where the controller stands after measuring 'window' in
  /// 'state'. While it holds a number of readers it averages the rate, and
  /// after the hold periods it tries another number. What it tries depends
  /// on what the readers wait for.
  ///
  /// While they spend less than half of their time waiting for copies, the
  /// source is the limit, which fewer readers cannot help: it tries a
  /// quarter more, at least one more. They are kept when the rate rises by
  /// at least half of what the added readers would bring if each brought as
  /// much as the others, and then more are tried right away. Readers are
  /// never given up in this state: how many concurrent reads a drive needs
  /// does not show in one measurement.
  ///
  /// While they wait for copies half of their time or more, the copies are
  /// the limit: it tries an eighth fewer, at least one fewer. They are kept
  /// when the rate falls by less than a quarter of the share of the readers
  /// left out, and fewer still are tried while the readers keep waiting.
  /// More readers are neither tried nor kept.
  ///
  /// While, half of the time or more, a reader is kept from its next piece
  /// because the staging buffers it may use are all being filled or copied,
  /// more readers are not tried. A trial that is not kept goes back to the
  /// held number. After a trial of more readers that is not kept, the hold
  /// is twice as long with each such trial in a row, up to eight times.
  ///
  /// With CopyMode::kAuto, every 'copyTrialInterval' trials of a number of
  /// readers, the first time before any, the trial is of the other way of
  /// waiting for copies with the held number of readers. The rate of a scan
  /// drifts with what is being read, so the two ways are measured in turn,
  /// four measurements starting with the last one of the hold: the other
  /// way is kept when each of its two delivered at least 5% more than the
  /// measurement before it, and the way it came from 5% less in between.
  /// The trial ends at the first measurement that does not fit, so one that
  /// loses costs one period. A measurement in which nothing was delivered
  /// starts no trial and decides none. The interval becomes four times as
  /// long with every trial, so that a machine on which one way is clearly
  /// better soon stops paying for trials of the other, and one on which
  /// both deliver the same does not go back and forth.
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

  /// Drops the pieces no reader has started of every range of 'group', at
  /// once. The ranges still have to be cancelled or waited for; a wait
  /// returns as for a cancelled range.
  void dropQueued(Group& group) noexcept;

  /// Runs 'task' on one of the scheduler's reader threads, ahead of the
  /// ranges that wait. For short work that belongs to the fetch and should
  /// not be done by the thread that noticed it was due, such as starting a
  /// fetch ahead. The destructor runs the tasks still waiting.
  void post(std::function<void()> task);

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

  // Whether groups that asked later have been served ahead of 'group' for
  // as long as they may.
  bool isAged(const Group& group) const;

  // Whether 'group' has so much less left to read than 'other' that it goes
  // first: at most a quarter. Groups of about one size keep their order.
  static bool isMuchSmaller(const Group& group, const Group& other);

  // Whether a reader may take a piece now.
  bool canDispatch() const;

  // Takes the next piece of the group to serve first, reserving a reader and
  // a staging buffer for it. Requires canDispatch().
  Job takeJob(Clock::time_point now);

  // A copy that its reader did not wait for.
  struct QueuedCopy {
    Job job;
    PinnedBufferPool::Buffer buffer{
        nullptr,
        PinnedBufferPool::Releaser(nullptr)};
    // When the copy was started.
    Clock::time_point start;
    // Time the reader took to fill the buffer.
    uint64_t readNanos{0};
  };

  // Body of the thread that waits for the copies readers did not wait for.
  void copyWaiterLoop();

  // Fills a staging buffer with the piece 'job' and copies it to the device.
  // Either waits for the copy and gives up the buffer reserved for the job,
  // or leaves both to the thread that waits for queued copies. Returns false
  // when the piece failed.
  bool fillAndCopy(const Job& job);

  // Fills the pieces of a sequential range one after the other, starting
  // with 'job'.
  void runSequential(Job job);

  // Adds a piece that reached the device to the counters. Requires 'mutex_'.
  void recordPiece(
      Read& read,
      size_t size,
      uint64_t readNanos,
      uint64_t copyNanos,
      bool isCopyQueued);

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
  // Signalled when a copy is queued and when the readers have stopped.
  std::condition_variable copyQueued_;

  bool started_{false};
  bool stopping_{false};
  std::vector<std::thread> readers_;
  // Waits for the copies in 'queuedCopies_'. Not joinable when it could not
  // be started, in which case every reader waits for its own copy.
  std::thread copyWaiter_;
  // Set by the destructor once no reader can queue another copy.
  bool readersStopped_{false};
  // Copies their readers did not wait for, in the order started.
  std::deque<QueuedCopy> queuedCopies_;

  // Tasks handed to post() that no reader thread has taken.
  std::deque<std::function<void()>> tasks_;
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
  // Copies started and not yet seen to have run.
  size_t numCopies_{0};
  // Readers waiting for the copy of their buffer.
  size_t numCopyWaiters_{0};
  // Readers of sequential ranges waiting for a staging buffer.
  size_t numBufferWaiters_{0};
  // Consumers blocked in wait().
  size_t numWaiters_{0};

  Clock::time_point lastAdvance_;
  ControlWindow window_;
  Stats stats_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
