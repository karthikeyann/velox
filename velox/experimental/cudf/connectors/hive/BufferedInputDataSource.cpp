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
#include "velox/experimental/cudf/connectors/hive/BufferedInputDataSource.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/time/Timer.h"

#include <folly/Executor.h>
#include <folly/executors/IOThreadPoolExecutor.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/EventBaseManager.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// A read that runs on the IO executor or on the thread that joins it,
// whichever gets to it first.
class ReadTask {
 public:
  explicit ReadTask(std::function<void()> read)
      : state_(std::make_shared<State>(std::move(read))) {}

  // Returns what an executor thread calls to run the read.
  folly::Func runner() const {
    return [state = state_]() { state->run(); };
  }

  // Runs the read unless it has started, waits for it and rethrows its
  // failure. Called at most once.
  void join() {
    state_->run();
    state_->result.get();
  }

  // Skips the read if it has not started, and waits for it otherwise. For a
  // read whose result nobody is going to collect.
  void abandon() noexcept {
    state_->abandoned = true;
    state_->run();
    if (state_->result.valid()) {
      state_->result.wait();
    }
  }

 private:
  struct State {
    explicit State(std::function<void()> read)
        : task([this, read = std::move(read)]() {
            if (!abandoned) {
              read();
            }
          }),
          result(task.get_future()) {}

    void run() {
      std::call_once(once, [this]() { task(); });
    }

    std::atomic_bool abandoned{false};
    std::once_flag once;
    std::packaged_task<void()> task;
    std::future<void> result;
  };

  std::shared_ptr<State> state_;
};

// The reads of one device read. The reads write through pointers that the
// submitter owns, so none may outlive this object: destroying it without
// joining, as when a submission fails or the result is dropped, skips the
// reads that have not started and waits for the rest.
class ReadTasks {
 public:
  // The reads are submitted to 'executor'.
  explicit ReadTasks(folly::Executor* executor) : executor_(executor) {
    // An IO thread pool runs what one of its own threads submits on that
    // same thread. Reads submitted from such a thread, as when a split is
    // prepared in the background, would run one after the other, so they are
    // handed to the pool's threads in turn instead.
    auto* ioPool = dynamic_cast<folly::IOThreadPoolExecutorBase*>(executor);
    if (ioPool == nullptr) {
      return;
    }
    const auto* current =
        folly::EventBaseManager::get()->getExistingEventBase();
    if (current == nullptr) {
      return;
    }
    auto eventBases = ioPool->getAllEventBases();
    const bool onPoolThread = std::any_of(
        eventBases.begin(), eventBases.end(), [&](const auto& eventBase) {
          return eventBase.get() == current;
        });
    if (onPoolThread) {
      eventBases_ = std::move(eventBases);
    }
  }

  ReadTasks(ReadTasks&&) = default;

  ReadTasks(const ReadTasks&) = delete;
  ReadTasks& operator=(const ReadTasks&) = delete;
  ReadTasks& operator=(ReadTasks&&) = delete;

  ~ReadTasks() {
    for (auto& task : tasks_) {
      task.abandon();
    }
  }

  void reserve(size_t numTasks) {
    tasks_.reserve(numTasks);
  }

  // Submits 'read' to the executor.
  void add(std::function<void()> read) {
    auto runner = tasks_.emplace_back(std::move(read)).runner();
    if (eventBases_.empty()) {
      executor_->add(std::move(runner));
      return;
    }
    static std::atomic_size_t nextThread{0};
    const auto index = nextThread.fetch_add(1, std::memory_order_relaxed);
    eventBases_[index % eventBases_.size()]->runInEventBaseThread(
        std::move(runner));
  }

  // Joins every read, then rethrows the first failure.
  void join() {
    std::exception_ptr failure;
    for (auto& task : tasks_) {
      try {
        task.join();
      } catch (...) {
        if (failure == nullptr) {
          failure = std::current_exception();
        }
      }
    }
    tasks_.clear();
    if (failure != nullptr) {
      std::rethrow_exception(failure);
    }
  }

 private:
  folly::Executor* const executor_;
  // The threads of the executor, when the reads are handed to them in turn.
  // Empty when the executor chooses the thread.
  std::vector<folly::Executor::KeepAlive<folly::EventBase>> eventBases_;
  std::vector<ReadTask> tasks_;
};

// Copies the first 'size' bytes of 'buffer' to 'dst' and waits for the copy,
// after which the buffer can be reused.
void copyToDevice(
    const PinnedBufferPool::StagingBuffer& buffer,
    size_t size,
    uint8_t* dst) {
  ensureCudaContextForThread();
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

// A range handed to a fetch scheduler. The scheduler's readers write through
// pointers that the submitter owns, so the range may not outlive this object:
// destroying it without waiting drops the pieces that have not started and
// waits for the rest.
class ScheduledRead {
 public:
  ScheduledRead(
      std::shared_ptr<FetchScheduler> scheduler,
      std::shared_ptr<FetchScheduler::Read> read,
      std::shared_ptr<IoStats> stats)
      : scheduler_(std::move(scheduler)),
        read_(std::move(read)),
        stats_(std::move(stats)) {}

  ScheduledRead(ScheduledRead&&) = default;

  ScheduledRead(const ScheduledRead&) = delete;
  ScheduledRead& operator=(const ScheduledRead&) = delete;
  ScheduledRead& operator=(ScheduledRead&&) = delete;

  ~ScheduledRead() {
    if (read_ != nullptr) {
      scheduler_->cancel(*read_);
    }
  }

  // Waits for the bytes to be on the device, rethrows a failure and reports
  // what the range cost. Called at most once.
  void wait() {
    // The range is complete when the wait returns or throws.
    const auto read = std::move(read_);
    const auto cost = scheduler_->wait(*read);
    if (stats_ == nullptr) {
      return;
    }
    using Unit = RuntimeCounter::Unit;
    const auto add = [&](const char* name, uint64_t value, Unit unit) {
      stats_->addCounter(
          name, RuntimeCounter(static_cast<int64_t>(value), unit));
    };
    add(BufferedInputDataSource::kFetchBytes, cost.bytes, Unit::kBytes);
    add(BufferedInputDataSource::kFetchReadWallNanos,
        cost.readNanos,
        Unit::kNanos);
    add(BufferedInputDataSource::kFetchQueuedWallNanos,
        cost.queuedNanos,
        Unit::kNanos);
    add(BufferedInputDataSource::kFetchCopyWallNanos,
        cost.copyNanos,
        Unit::kNanos);
    add(BufferedInputDataSource::kFetchCopyBusyPercent,
        cost.copyBusyPercent,
        Unit::kNone);
    add(BufferedInputDataSource::kFetchReaderLimitedPercent,
        cost.readerLimitedPercent,
        Unit::kNone);
    add(BufferedInputDataSource::kFetchBufferLimitedPercent,
        cost.bufferLimitedPercent,
        Unit::kNone);
    add(BufferedInputDataSource::kFetchReaders, cost.numReaders, Unit::kNone);
  }

 private:
  std::shared_ptr<FetchScheduler> scheduler_;
  // Null once waited for.
  std::shared_ptr<FetchScheduler::Read> read_;
  std::shared_ptr<IoStats> stats_;
};

} // namespace

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
    std::shared_ptr<PinnedBufferPool> stagingPool,
    std::shared_ptr<FetchScheduler> fetchScheduler,
    std::shared_ptr<IoStats> fetchStats)
    : input_(std::move(input)),
      fileSize_(input_->getReadFile()->size()),
      stagingPool_(std::move(stagingPool)),
      fetchScheduler_(std::move(fetchScheduler)),
      fetchGroup_(
          fetchScheduler_ != nullptr ? fetchScheduler_->makeGroup() : nullptr),
      fetchStats_(std::move(fetchStats)) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readContiguous(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    cuda::stream_ref stream) {
  const size_t readSize =
      offset >= fileSize_ ? 0 : std::min(size, fileSize_ - offset);
  if (fetchScheduler_ != nullptr) {
    if (readSize == 0) {
      return std::async(std::launch::deferred, []() { return size_t{0}; });
    }
    return scheduleDeviceRead(offset, readSize, dst);
  }
  auto* executor = input_->executor();
  VELOX_CHECK_NOT_NULL(executor, "IO executor is not initialized");

  ReadTasks tasks(executor);
  if (readSize == 0) {
    // Nothing to read.
  } else if (stagingPool_ == nullptr) {
    tasks.add([this, offset, readSize, dst, stream]() {
      stagePageable(offset, readSize, dst, stream);
    });
  } else if (input_->hasCache() || input_->getReadFile()->shouldCoalesce()) {
    // One read of the whole range through the buffered input. That keeps the
    // cache entries the same as for any other reader of the range, and
    // leaves the request sizes to the buffered input for a file that asks
    // for coalesced reads, as files in remote storage do.
    tasks.add([this, offset, readSize, dst, stream]() {
      stageFromInput(offset, readSize, dst, stream);
    });
  } else {
    // A local file that nothing caches is read straight into the staging
    // buffers, one independent read per buffer.
    const auto bufferSize = stagingPool_->bufferSize();
    tasks.reserve((readSize + bufferSize - 1) / bufferSize);
    for (size_t done = 0; done < readSize; done += bufferSize) {
      const auto partSize = std::min(bufferSize, readSize - done);
      tasks.add([this, offset, done, partSize, dst, stream]() {
        stageFromFile(offset + done, partSize, dst + done, stream);
      });
    }
  }

  return std::async(
      std::launch::deferred, [tasks = std::move(tasks), readSize]() mutable {
        tasks.join();
        return readSize;
      });
}

std::future<size_t> BufferedInputDataSource::scheduleDeviceRead(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  std::shared_ptr<FetchScheduler::Read> read;
  if (input_->hasCache() || input_->getReadFile()->shouldCoalesce()) {
    // One read of the whole range through the buffered input, opened by the
    // reader that gets to it. That keeps the cache entries the same as for
    // any other reader of the range, and leaves the request sizes to the
    // buffered input for a file that asks for coalesced reads, as files in
    // remote storage do.
    read = fetchScheduler_->submit(
        fetchGroup_,
        size,
        dst,
        FetchScheduler::FillOrder::kSequential,
        [this,
         offset,
         size,
         stream = std::shared_ptr<dwio::common::SeekableInputStream>()](
            uint8_t* buffer, size_t /*pieceOffset*/, size_t pieceSize) mutable {
          if (stream == nullptr) {
            stream = input_->read(offset, size, dwio::common::LogType::FILE);
            VELOX_CHECK_NOT_NULL(stream, "Buffered input returned no stream");
          }
          stream->readFully(reinterpret_cast<char*>(buffer), pieceSize);
        });
  } else {
    // A local file that nothing caches is read straight into the staging
    // buffers, one independent read per buffer.
    read = fetchScheduler_->submit(
        fetchGroup_,
        size,
        dst,
        FetchScheduler::FillOrder::kAny,
        [this, offset](uint8_t* buffer, size_t pieceOffset, size_t pieceSize) {
          readFromFile(offset + pieceOffset, pieceSize, buffer);
        });
  }
  return std::async(
      std::launch::deferred,
      [scheduled = ScheduledRead(fetchScheduler_, std::move(read), fetchStats_),
       size]() mutable {
        scheduled.wait();
        return size;
      });
}

void BufferedInputDataSource::readFromFile(
    size_t offset,
    size_t size,
    uint8_t* buffer) {
  const auto& file = input_->getInputStream();
  uint64_t readUs{0};
  {
    MicrosecondWallTimer timer(&readUs);
    file->read(buffer, size, offset, dwio::common::LogType::FILE);
  }
  if (auto* stats = file->getStats()) {
    stats->read().increment(size);
    stats->queryThreadIoLatencyUs().increment(readUs);
    stats->storageReadLatencyUs().increment(readUs);
  }
}

void BufferedInputDataSource::stagePageable(
    size_t offset,
    size_t size,
    uint8_t* dst,
    cuda::stream_ref stream) {
  auto hostBuffer = host_read(offset, size);
  ensureCudaContextForThread();
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      dst,
      hostBuffer->data(),
      hostBuffer->size(),
      cudaMemcpyDefault,
      stream.get()));
}

void BufferedInputDataSource::stageFromFile(
    size_t offset,
    size_t size,
    uint8_t* dst,
    cuda::stream_ref stream) {
  auto buffer = stagingPool_->acquire();
  if (buffer == nullptr) {
    stagePageable(offset, size, dst, stream);
    return;
  }
  readFromFile(offset, size, buffer->data);
  copyToDevice(*buffer, size, dst);
}

void BufferedInputDataSource::stageFromInput(
    size_t offset,
    size_t size,
    uint8_t* dst,
    cuda::stream_ref stream) {
  auto buffer = stagingPool_->acquire();
  if (buffer == nullptr) {
    stagePageable(offset, size, dst, stream);
    return;
  }
  auto input = input_->read(offset, size, dwio::common::LogType::FILE);
  VELOX_CHECK_NOT_NULL(input, "Buffered input returned no stream");
  const auto bufferSize = stagingPool_->bufferSize();
  for (size_t done = 0; done < size; done += bufferSize) {
    const auto partSize = std::min(bufferSize, size - done);
    input->readFully(reinterpret_cast<char*>(buffer->data), partSize);
    copyToDevice(*buffer, partSize, dst + done);
  }
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

} // namespace facebook::velox::cudf_velox::connector::hive
