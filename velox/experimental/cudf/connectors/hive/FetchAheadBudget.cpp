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

#include "velox/common/base/Exceptions.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {

FetchAheadBudget::FetchAheadBudget(uint64_t capacity) : capacity_(capacity) {}

void FetchAheadBudget::hold(uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_.heldBytes += bytes;
  stats_.maxHeldBytes = std::max(stats_.maxHeldBytes, stats_.heldBytes);
}

void FetchAheadBudget::claim() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK_GT(stats_.numAhead, 0);
    --stats_.numAhead;
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
    waiting_.erase(it);
    stats_.numWaiting = waiting_.size();
  }
  // The offers behind it may fit where it did not.
  startWaiting();
}

void FetchAheadBudget::release(uint64_t bytes) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK_GE(stats_.heldBytes, bytes);
    stats_.heldBytes -= bytes;
  }
  startWaiting();
}

void FetchAheadBudget::offer(
    std::shared_ptr<Candidate> candidate,
    uint64_t bytes) {
  VELOX_CHECK_NOT_NULL(candidate);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.numOffered;
    waiting_.push_back(Offer{std::move(candidate), bytes});
    stats_.numWaiting = waiting_.size();
  }
  startWaiting();
}

void FetchAheadBudget::startWaiting() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (starting_) {
    return;
  }
  starting_ = true;
  while (!waiting_.empty()) {
    const auto bytes = waiting_.front().bytes;
    if (stats_.numAhead > 0 && stats_.heldBytes + bytes > capacity_) {
      break;
    }
    auto candidate = std::move(waiting_.front().candidate);
    waiting_.pop_front();
    stats_.numWaiting = waiting_.size();
    // Counted before the fetch starts, so that an offer made meanwhile sees
    // it.
    stats_.heldBytes += bytes;
    stats_.maxHeldBytes = std::max(stats_.maxHeldBytes, stats_.heldBytes);
    ++stats_.numAhead;
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
      --stats_.numAhead;
    }
  }
  starting_ = false;
}

FetchAheadBudget::Stats FetchAheadBudget::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

PostedFetchStart::PostedFetchStart(
    FetchAheadBudget* budget,
    uint64_t numBytes,
    Post post,
    std::function<void()> startFetch)
    : budget_(budget),
      numBytes_(numBytes),
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
    budget_->claim();
    budget_->release(numBytes_);
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
    LOG(WARNING) << "Failed to start a fetch ahead: " << e.what();
  } catch (...) {
    LOG(WARNING) << "Failed to start a fetch ahead";
  }
  // The budget counted a fetch that does not exist. The consumer starts the
  // fetch again when it gets to it, and meets the failure then.
  budget_->claim();
  budget_->release(numBytes_);
}

} // namespace facebook::velox::cudf_velox::connector::hive
