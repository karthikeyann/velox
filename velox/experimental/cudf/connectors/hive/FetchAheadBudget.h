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

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>

namespace facebook::velox::cudf_velox::connector::hive {

/// Bounds the device memory that scans hold in column chunks they have
/// fetched and not yet decoded, and decides which fetches start before their
/// consumer asks for them. A fetch a consumer is waiting for always starts.
/// A fetch offered ahead of its consumer starts while the bytes held stay
/// within the capacity, in the order of the offers, so that how far ahead the
/// scans read is a number of bytes and not a number of splits or drivers.
/// One fetch is always allowed to be ahead of its consumer, whatever its
/// size and whatever the consumers hold.
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
  };

  /// 'capacity' is the bytes that fetches started ahead may bring the total
  /// held to.
  explicit FetchAheadBudget(uint64_t capacity);

  /// Returns the capacity in bytes.
  uint64_t capacity() const {
    return capacity_;
  }

  /// Counts 'bytes' as held by a fetch that its consumer started. Always
  /// succeeds, also beyond the capacity.
  void hold(uint64_t bytes);

  /// Tells the budget that the consumer of a fetch started ahead asks for it
  /// now, so that it is no longer ahead. Called once for each fetch started
  /// ahead, before its bytes are released. Starts the offered fetches that
  /// now have room.
  void claim();

  /// Gives back 'bytes' held, by a fetch its consumer started or by one
  /// started ahead, and starts the offered fetches that now have room.
  void release(uint64_t bytes);

  /// Offers a fetch of 'bytes' to start ahead of its consumer. Starts it
  /// before returning when it has room, and otherwise when room appears,
  /// after the fetches offered before it. A fetch has room when the bytes
  /// held stay within the capacity with it, or when no fetch is ahead of its
  /// consumer. The bytes of a fetch started this way are held until
  /// release().
  void offer(std::shared_ptr<Candidate> candidate, uint64_t bytes);

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
  };

  // Starts offered fetches in order while the next one has room.
  void startWaiting();

  const uint64_t capacity_;

  mutable std::mutex mutex_;
  // Fetches offered and not started, oldest first.
  std::deque<Offer> waiting_;
  // Whether a thread is in startWaiting(). Others leave the starting to it,
  // which keeps the order of the offers.
  bool starting_{false};
  Stats stats_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
