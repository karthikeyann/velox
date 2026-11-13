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

#include <rmm/cuda_stream.hpp>

#include <cuda/stream>

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace folly {
class CPUThreadPoolExecutor;
}

namespace facebook::velox::cudf_velox::connector::hive {

/// Bounds the device memory that scans hold in column chunks they have
/// fetched and not yet decoded, and decides which fetches start before their
/// consumer asks for them. A fetch a consumer is waiting for always starts.
/// A fetch offered ahead of its consumer starts while the bytes held stay
/// within the capacity, so that how far ahead the scans read is a number of
/// bytes and not a number of splits or drivers. Every scan may have one
/// fetch ahead of its consumers, whatever its size and whatever is held, so
/// that a scan whose units are larger than the capacity still reads ahead.
///
/// A fetch starts ahead only while the consumers of its scan have received
/// every fetch they asked for, the first one included, and no consumer of
/// any scan is blocked waiting for bytes. The readers below have no notion
/// of priority: they serve fetches in the order they are started, so a fetch
/// started ahead while somebody needs bytes, or before the scan's consumers
/// have their first bytes, would be served before the fetch that is needed
/// first. Fetches ahead therefore start when consumers are busy with what
/// they have, which is when the storage and the link would otherwise be
/// idle.
///
/// The bytes counted are those of the fetches: compressed column chunks. The
/// budget does not bound the device memory of a scan. Consumers' own fetches
/// are counted and never held back, every scan may add one fetch beyond the
/// capacity, and what decoding allocates is not counted.
///
/// Thread-safe.
class FetchAheadBudget {
 public:
  /// A fetch that can start before its consumer asks for it.
  class Candidate {
   public:
    virtual ~Candidate() = default;

    /// Starts the fetch. Returns false when there is nothing to start: the
    /// consumer has started it or no longer wants it. Called on the thread
    /// that makes room, with no lock of the budget held.
    virtual bool start() = 0;
  };

  /// Counters since the budget was created.
  struct Stats {
    /// Bytes held now.
    uint64_t heldBytes{0};
    /// Most bytes held at once.
    uint64_t maxHeldBytes{0};
    /// Fetches offered ahead of their consumer.
    uint64_t numOffered{0};
    /// Offered fetches the budget started.
    uint64_t numStartedAhead{0};
    /// Offered fetches waiting for room now.
    uint64_t numWaiting{0};
    /// Fetches started ahead that their consumers have not asked for yet.
    uint64_t numAhead{0};
    /// Fetches started ahead that their consumers asked for afterwards.
    uint64_t numClaimed{0};
    /// Consumers blocked waiting for bytes now.
    uint64_t numWaitingConsumers{0};
  };

  /// Where a fetch whose bytes are held stands with its consumer.
  enum class Stage {
    /// Started ahead: its consumer has not asked for it.
    kAhead,
    /// Its consumer asked for it, by starting or claiming it, and does not
    /// have its bytes yet.
    kAwaited,
    /// Its consumer has its bytes.
    kReceived,
  };

  /// 'capacity' is the bytes that fetches started ahead may bring the total
  /// held to.
  explicit FetchAheadBudget(uint64_t capacity);

  /// Returns the capacity in bytes.
  uint64_t capacity() const {
    return capacity_;
  }

  /// Tells the budget that a reader of 'scan' exists, until the matching
  /// detach(). While a scan has readers the budget remembers that its
  /// consumers have received bytes.
  void attach(uint64_t scan);

  /// Ends an attach().
  void detach(uint64_t scan) noexcept;

  /// Counts 'bytes' as held by a fetch of 'scan' that its consumer started:
  /// an awaited fetch. Always succeeds, also beyond the capacity.
  void hold(uint64_t bytes, uint64_t scan);

  /// Tells the budget that the consumer of a fetch of 'scan' that was
  /// started ahead asks for it now: the fetch is awaited.
  void claim(uint64_t scan);

  /// Tells the budget that the consumer of an awaited fetch of 'scan' has
  /// its bytes. Starts the offered fetches that can start now.
  void receive(uint64_t scan);

  /// Gives back the 'bytes' of a fetch of 'scan' that is at 'stage'. Starts
  /// the offered fetches that can start now.
  void release(uint64_t bytes, uint64_t scan, Stage stage);

  /// Tells the budget that a consumer blocks until the bytes of a fetch
  /// have arrived. Until the matching waitEnd() no offered fetch is started.
  void waitBegin();

  /// Ends a waitBegin(), and starts the offered fetches that can start now.
  void waitEnd() noexcept;

  /// How soon the consumer of an offered fetch gets to it.
  enum class Distance {
    /// Next: the consumer is working on what comes right before it, such as
    /// the row group pass before it in the same split.
    kNext,
    /// After whatever the consumer is working on now has been read, such as
    /// a split prepared ahead.
    kLater,
  };

  /// Offers a fetch of 'bytes' for 'scan', a number that names the scan the
  /// fetch belongs to, to start ahead of its consumer. Starts it before
  /// returning when it can start, and otherwise when that changes: after the
  /// fetches offered before it at the same distance, and the next ones
  /// before the later ones. A fetch can start when the consumers of its
  /// scan have received a fetch and await none, no consumer is blocked
  /// waiting for bytes, and it has room: the bytes held stay within the
  /// capacity with it. A fetch of a scan that has none ahead needs no room,
  /// and does not wait for the offers before it. The bytes of a fetch
  /// started this way are held, ahead, until claim(). The budget forgets
  /// that a scan received bytes when the scan has no reader attached, holds
  /// nothing and has no offer waiting.
  void offer(
      std::shared_ptr<Candidate> candidate,
      uint64_t bytes,
      Distance distance,
      uint64_t scan);

  /// Takes back an offer that has not started, and starts the offers behind
  /// it that now have room. Does nothing when the fetch has started or is
  /// being started.
  void withdraw(const Candidate* candidate);

  /// Returns the counters.
  Stats stats() const;

 private:
  struct Offer {
    std::shared_ptr<Candidate> candidate;
    uint64_t bytes{0};
    Distance distance{Distance::kLater};
    uint64_t scan{0};
  };

  // Starts offered fetches in order while the next one can start.
  void startWaiting();

  const uint64_t capacity_;

  mutable std::mutex mutex_;
  // Fetches offered and not started: the next ones before the later ones,
  // and the oldest first among each.
  std::deque<Offer> waiting_;
  // Whether a thread is in startWaiting(). Others leave the starting to it,
  // which keeps the order of the offers.
  bool starting_{false};
  // What the budget knows of a scan.
  struct Scan {
    // Fetches at each stage.
    uint32_t numAhead{0};
    uint32_t numAwaited{0};
    uint32_t numReceived{0};
    // Offers of the scan that have not started.
    uint32_t numOffersWaiting{0};
    // Readers attached.
    uint32_t numReaders{0};
    // Whether a consumer has received a fetch since the entry was made.
    bool hasReceived{false};

    // Returns the number of fetches at 'stage'.
    uint32_t& numAt(Stage stage);
  };

  // Takes one fetch of 'scan' out of 'stage'. Throws when the scan has none
  // there.
  void leaveStage(uint64_t scan, Stage stage);

  // Removes the entry of 'scan' when it holds no fetch and has no offer
  // waiting and no reader.
  void eraseIfIdle(uint64_t scan);

  // Scans that hold fetches or have offers waiting or readers.
  std::unordered_map<uint64_t, Scan> scans_;
  Stats stats_;
};

/// What a connector with a fetch-ahead budget owns and hands to its readers:
/// the budget, the thread that starts fetches ahead, and the CUDA streams
/// those fetches run on. The scans of one connector are bounded together; a
/// second connector has a budget, a thread and streams of its own.
class FetchAheadContext {
 public:
  /// Number of streams the fetches ahead of a connector share.
  static constexpr size_t kNumStreams{8};

  /// 'capacity' is the capacity of the budget in bytes and must be positive.
  explicit FetchAheadContext(uint64_t capacity);

  /// Runs the starts that were posted before returning.
  ~FetchAheadContext();

  /// Returns the budget.
  FetchAheadBudget& budget() {
    return budget_;
  }

  /// Runs 'task' on the thread that starts fetches ahead, after the tasks
  /// posted before it. The thread belongs to no IO executor: a start submits
  /// the reads of a row group pass, and an IO executor that runs what one of
  /// its own threads adds on that same thread, as folly::IOThreadPoolExecutor
  /// does, would run all of them on the thread that started the fetch.
  void post(std::function<void()> task);

  /// Returns a stream for the fetches ahead of one reader, taking the
  /// connector's streams in turn, so that readers can share a stream. The
  /// streams are created on first use and live as long as the context. They
  /// are not streams of cuDF's pool, which decode kernels of any reader run
  /// on: a data source synchronizes the stream a fetch is given, and would
  /// wait for those.
  cuda::stream_ref nextStream();

 private:
  FetchAheadBudget budget_;
  // Guards the streams.
  std::mutex mutex_;
  std::vector<rmm::cuda_stream> streams_;
  // Index of the stream nextStream() returns.
  size_t nextStream_{0};
  // One thread. Declared last: its tasks use the members above.
  std::unique_ptr<folly::CPUThreadPoolExecutor> startExecutor_;
};

/// A fetch on offer to a FetchAheadBudget that another thread starts. When
/// the budget has room the start is posted to that thread. Until the posted
/// start begins, the consumer can take the offer back and start the fetch
/// itself, so a fetch somebody needs never waits behind work that is only
/// queued.
///
/// Thread-safe.
class PostedFetchStart final
    : public FetchAheadBudget::Candidate,
      public std::enable_shared_from_this<PostedFetchStart> {
 public:
  /// Runs a task on another thread, later. Throws when it cannot.
  using Post = std::function<void(std::function<void()>)>;

  /// 'startFetch' starts a fetch of 'numBytes' for 'scan' and runs at most
  /// once, on the thread that 'post' hands its task to. A start that throws
  /// gives the bytes back to 'budget', which must outlive the offer.
  PostedFetchStart(
      FetchAheadBudget* budget,
      uint64_t numBytes,
      uint64_t scan,
      Post post,
      std::function<void()> startFetch);

  /// Posts the start. Returns false when the offer was taken back. Throws
  /// what the post throws, and can then be called again.
  bool start() override;

  /// Takes the offer back: a start that has not begun never runs, and its
  /// bytes go back to the budget. Waits for a start that is running.
  void withdraw();

 private:
  // Body of the posted task.
  void run();

  FetchAheadBudget* const budget_;
  const uint64_t numBytes_;
  const uint64_t scan_;
  const Post post_;
  const std::function<void()> startFetch_;

  // Held while the fetch is being started, so that withdraw() waits for it.
  std::mutex mutex_;
  // Whether the start is posted and has not begun.
  bool isPosted_{false};
  // Whether the start has begun or the offer was taken back.
  bool isClosed_{false};
};

} // namespace facebook::velox::cudf_velox::connector::hive
