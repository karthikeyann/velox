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

#include "velox/experimental/cudf/connectors/hive/FetchAheadBudget.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"

#include <folly/ScopeGuard.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/thread_factory/NamedThreadFactory.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <string_view>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {

FetchAheadBudget::FetchAheadBudget(uint64_t capacity) : capacity_(capacity) {}

uint32_t& FetchAheadBudget::Scan::numAt(Stage stage) {
  switch (stage) {
    case Stage::kAhead:
      return numAhead;
    case Stage::kAwaited:
      return numAwaited;
    case Stage::kReceived:
      return numReceived;
  }
  VELOX_UNREACHABLE();
}

void FetchAheadBudget::attach(uint64_t scan) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++scans_[scan].numReaders;
}

void FetchAheadBudget::detach(uint64_t scan) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = scans_.find(scan);
  if (it == scans_.end() || it->second.numReaders == 0) {
    return;
  }
  --it->second.numReaders;
  eraseIfIdle(scan);
}

void FetchAheadBudget::hold(uint64_t bytes, uint64_t scan) {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_.heldBytes += bytes;
  stats_.maxHeldBytes = std::max(stats_.maxHeldBytes, stats_.heldBytes);
  ++scans_[scan].numAwaited;
}

void FetchAheadBudget::leaveStage(uint64_t scan, Stage stage) {
  static constexpr std::array<std::string_view, 3> kStageNames = {
      "ahead", "awaited", "received"};
  const auto it = scans_.find(scan);
  const auto numAtStage = it == scans_.end() ? 0 : it->second.numAt(stage);
  VELOX_CHECK_GT(
      numAtStage,
      0,
      "No fetch of the scan is {}: {}",
      kStageNames[static_cast<size_t>(stage)],
      scan);
  --it->second.numAt(stage);
  if (stage == Stage::kAhead) {
    --stats_.numAhead;
  }
}

void FetchAheadBudget::eraseIfIdle(uint64_t scan) {
  const auto it = scans_.find(scan);
  if (it != scans_.end() && it->second.numAhead == 0 &&
      it->second.numAwaited == 0 && it->second.numReceived == 0 &&
      it->second.numOffersWaiting == 0 && it->second.numReaders == 0) {
    scans_.erase(it);
  }
}

void FetchAheadBudget::claim(uint64_t scan) {
  std::lock_guard<std::mutex> lock(mutex_);
  leaveStage(scan, Stage::kAhead);
  ++scans_[scan].numAwaited;
  ++stats_.numClaimed;
}

void FetchAheadBudget::receive(uint64_t scan) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    leaveStage(scan, Stage::kAwaited);
    auto& entry = scans_[scan];
    ++entry.numReceived;
    entry.hasReceived = true;
  }
  startWaiting();
}

void FetchAheadBudget::withdraw(const Candidate* candidate) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it =
        std::find_if(waiting_.begin(), waiting_.end(), [&](const Offer& offer) {
          return offer.candidate.get() == candidate;
        });
    if (it == waiting_.end()) {
      return;
    }
    const auto scan = it->scan;
    waiting_.erase(it);
    stats_.numWaiting = waiting_.size();
    --scans_[scan].numOffersWaiting;
    eraseIfIdle(scan);
  }
  // The offers behind it may fit where it did not.
  startWaiting();
}

void FetchAheadBudget::waitBegin() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++stats_.numWaitingConsumers;
}

void FetchAheadBudget::waitEnd() noexcept {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_DCHECK_GT(stats_.numWaitingConsumers, 0);
    if (stats_.numWaitingConsumers > 0) {
      --stats_.numWaitingConsumers;
    }
  }
  try {
    startWaiting();
  } catch (const std::exception& e) {
    // The offers stay waiting and start at the next change.
    LOG(WARNING) << "Failed to start the fetches ahead: " << e.what();
  }
}

void FetchAheadBudget::release(uint64_t bytes, uint64_t scan, Stage stage) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK_GE(stats_.heldBytes, bytes);
    leaveStage(scan, stage);
    stats_.heldBytes -= bytes;
    eraseIfIdle(scan);
  }
  startWaiting();
}

void FetchAheadBudget::offer(
    std::shared_ptr<Candidate> candidate,
    uint64_t bytes,
    Distance distance,
    uint64_t scan) {
  VELOX_CHECK_NOT_NULL(candidate);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.numOffered;
    // Behind the offers of its own distance, ahead of the later ones.
    const auto position = distance == Distance::kLater
        ? waiting_.end()
        : std::find_if(
              waiting_.begin(), waiting_.end(), [](const Offer& offer) {
                return offer.distance == Distance::kLater;
              });
    waiting_.insert(
        position, Offer{std::move(candidate), bytes, distance, scan});
    stats_.numWaiting = waiting_.size();
    ++scans_[scan].numOffersWaiting;
  }
  startWaiting();
}

void FetchAheadBudget::startWaiting() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (starting_) {
    return;
  }
  starting_ = true;
  SCOPE_EXIT {
    if (!lock.owns_lock()) {
      lock.lock();
    }
    starting_ = false;
  };
  // Not while a consumer waits for bytes: what it waits for goes first.
  while (stats_.numWaitingConsumers == 0) {
    // The first offer in order that has room, or, ahead of its turn, the
    // first of a scan that has nothing ahead. Offers behind one that waits
    // for room wait with it. Only offers of scans whose consumers have
    // received bytes and await none: what they await was asked for first,
    // and the others have not received their first bytes yet.
    auto next = waiting_.end();
    bool isBlocked{false};
    for (auto it = waiting_.begin(); it != waiting_.end(); ++it) {
      const auto scan = scans_.find(it->scan);
      VELOX_CHECK(scan != scans_.end());
      if (!scan->second.hasReceived || scan->second.numAwaited > 0) {
        continue;
      }
      const bool hasRoom =
          !isBlocked && stats_.heldBytes + it->bytes <= capacity_;
      if (hasRoom || scan->second.numAhead == 0) {
        next = it;
        break;
      }
      isBlocked = true;
    }
    if (next == waiting_.end()) {
      break;
    }
    const auto bytes = next->bytes;
    const auto scan = next->scan;
    auto candidate = std::move(next->candidate);
    waiting_.erase(next);
    stats_.numWaiting = waiting_.size();
    // Counted before the fetch starts, so that an offer made meanwhile sees
    // it.
    stats_.heldBytes += bytes;
    stats_.maxHeldBytes = std::max(stats_.maxHeldBytes, stats_.heldBytes);
    ++stats_.numAhead;
    auto& entry = scans_[scan];
    ++entry.numAhead;
    --entry.numOffersWaiting;
    lock.unlock();
    bool started{false};
    try {
      started = candidate->start();
    } catch (...) {
      // The consumer meets the failure when it starts the fetch itself.
    }
    candidate.reset();
    lock.lock();
    if (started) {
      ++stats_.numStartedAhead;
    } else {
      stats_.heldBytes -= bytes;
      leaveStage(scan, Stage::kAhead);
      eraseIfIdle(scan);
    }
  }
}

FetchAheadBudget::Stats FetchAheadBudget::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

FetchAheadContext::FetchAheadContext(uint64_t capacity)
    : budget_(capacity),
      startExecutor_(
          std::make_unique<folly::CPUThreadPoolExecutor>(
              1,
              std::make_shared<folly::NamedThreadFactory>("CudfFetchAhead"))) {
  VELOX_CHECK_GT(capacity, 0);
}

FetchAheadContext::~FetchAheadContext() {
  startExecutor_->join();
}

void FetchAheadContext::post(std::function<void()> task) {
  startExecutor_->add(std::move(task));
}

cuda::stream_ref FetchAheadContext::nextStream() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (streams_.empty()) {
    ensureCudaContextForThread();
    streams_.reserve(kNumStreams);
    for (size_t i = 0; i < kNumStreams; ++i) {
      streams_.emplace_back(rmm::cuda_stream::flags::non_blocking);
    }
  }
  return streams_[nextStream_++ % kNumStreams];
}

PostedFetchStart::PostedFetchStart(
    FetchAheadBudget* budget,
    uint64_t numBytes,
    uint64_t scan,
    Post post,
    std::function<void()> startFetch)
    : budget_(budget),
      numBytes_(numBytes),
      scan_(scan),
      post_(std::move(post)),
      startFetch_(std::move(startFetch)) {
  VELOX_CHECK_NOT_NULL(budget_);
}

bool PostedFetchStart::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (isClosed_ || isPosted_) {
    return false;
  }
  post_([self = shared_from_this()]() { self->run(); });
  // Only now: a post that throws leaves the offer as it was.
  isPosted_ = true;
  return true;
}

void PostedFetchStart::withdraw() {
  bool tookBack{false};
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tookBack = isPosted_;
    isPosted_ = false;
    isClosed_ = true;
  }
  if (tookBack) {
    // The budget counted a fetch that will not start.
    budget_->release(numBytes_, scan_, FetchAheadBudget::Stage::kAhead);
  }
}

void PostedFetchStart::run() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!isPosted_) {
    return;
  }
  isPosted_ = false;
  isClosed_ = true;
  try {
    startFetch_();
    return;
  } catch (const std::exception& e) {
    LOG_EVERY_N(WARNING, 100) << "Failed to start a fetch ahead: " << e.what();
  } catch (...) {
    LOG_EVERY_N(WARNING, 100) << "Failed to start a fetch ahead";
  }
  // The budget counted a fetch that does not exist. The consumer starts the
  // fetch again when it gets to it, and meets the failure then.
  budget_->release(numBytes_, scan_, FetchAheadBudget::Stage::kAhead);
}

} // namespace facebook::velox::cudf_velox::connector::hive
