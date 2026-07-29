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
#include "velox/serializers/ArrowIpcSerializer.h"

#include <arrow/buffer.h>
#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/ipc/api.h>
#include <arrow/record_batch.h>
#include <arrow/util/compression.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::serializer {
namespace {

// Alias Velox stream types under different names so they don't collide
// with arrow::io::OutputStream / arrow::io::InputStream once we derive
// from those Arrow base classes (the injected base-class name would
// otherwise shadow the Velox type at the point of use).
using VeloxSink = ::facebook::velox::OutputStream;
using VeloxSource = ::facebook::velox::ByteInputStream;

// Translates a non-OK arrow::Status into a Velox check failure with the
// Arrow message preserved. Used so callers see the actual IPC / bridge
// error rather than a generic Velox failure.
void checkArrowStatus(const arrow::Status& status) {
  VELOX_CHECK(status.ok(), "Arrow IPC error: {}", status.ToString());
}

// Same as checkArrowStatus but extracts the value out of an arrow::Result.
// Mirrors the ARROW_ASSIGN_OR_RAISE pattern but throws a Velox exception
// on failure instead of returning Status.
template <typename T>
T arrowResultOrThrow(arrow::Result<T> result) {
  VELOX_CHECK(result.ok(), "Arrow IPC error: {}", result.status().ToString());
  return std::move(result).ValueUnsafe();
}

// arrow::io::OutputStream adapter that forwards writes directly to a
// Velox OutputStream. Lets arrow::ipc::MakeStreamWriter write into the
// caller's sink without staging the IPC payload through an intermediate
// arrow::io::BufferOutputStream + Finish() + memcpy. Profiling showed
// that intermediate-buffer chain accounted for 42-66% of every serialize
// path, so collapsing it is the single highest-impact change here.
class VeloxArrowOutputStream : public arrow::io::OutputStream {
 public:
  explicit VeloxArrowOutputStream(VeloxSink* sink) : sink_(sink) {}

  arrow::Status Close() override {
    closed_ = true;
    return arrow::Status::OK();
  }

  bool closed() const override {
    return closed_;
  }

  arrow::Result<int64_t> Tell() const override {
    return position_;
  }

  arrow::Status Write(const void* data, int64_t nbytes) override {
    if (closed_) {
      return arrow::Status::IOError("VeloxArrowOutputStream is closed");
    }
    if (nbytes < 0) {
      return arrow::Status::Invalid("Negative write size: ", nbytes);
    }
    if (nbytes == 0) {
      return arrow::Status::OK();
    }
    sink_->write(
        reinterpret_cast<const char*>(data),
        static_cast<std::streamsize>(nbytes));
    position_ += nbytes;
    return arrow::Status::OK();
  }

 private:
  VeloxSink* const sink_;
  int64_t position_{0};
  bool closed_{false};
};

// arrow::io::InputStream adapter that forwards reads to a Velox
// ByteInputStream. Replaces the previous "drain everything into one
// arrow::Buffer up front" path which profiling showed accounted for
// ~88% of deserialize time. Arrow's IPC stream reader is happy with a
// pure InputStream (no random access required for stream format), so
// this wrapper is enough.
class VeloxArrowInputStream : public arrow::io::InputStream {
 public:
  explicit VeloxArrowInputStream(VeloxSource* source)
      : source_(source),
        startOffset_(source_->tellp()),
        totalSize_(source_->size() - source_->tellp()) {}

  arrow::Status Close() override {
    closed_ = true;
    return arrow::Status::OK();
  }

  bool closed() const override {
    return closed_;
  }

  arrow::Result<int64_t> Tell() const override {
    return source_->tellp() - startOffset_;
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
    if (closed_) {
      return arrow::Status::IOError("VeloxArrowInputStream is closed");
    }
    if (nbytes < 0) {
      return arrow::Status::Invalid("Negative read size: ", nbytes);
    }
    const int64_t available = totalSize_ - (source_->tellp() - startOffset_);
    const int64_t toRead = std::min(nbytes, available);
    if (toRead <= 0) {
      return 0;
    }
    source_->readBytes(
        static_cast<uint8_t*>(out), static_cast<int32_t>(toRead));
    return toRead;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, arrow::AllocateResizableBuffer(nbytes));
    ARROW_ASSIGN_OR_RAISE(
        const int64_t bytesRead, Read(nbytes, buffer->mutable_data()));
    if (bytesRead < nbytes) {
      ARROW_RETURN_NOT_OK(buffer->Resize(bytesRead, /*shrink_to_fit=*/false));
    }
    return std::shared_ptr<arrow::Buffer>(std::move(buffer));
  }

 private:
  VeloxSource* const source_;
  const int64_t startOffset_;
  const int64_t totalSize_;
  bool closed_{false};
};

// Converts a Velox VectorSerde::Options into
// ArrowIpcVectorSerde::ArrowIpcOptions. Falls back to defaults if the caller
// did not pass an ArrowIpcOptions.
ArrowIpcVectorSerde::ArrowIpcOptions resolveOptions(
    const VectorSerde::Options* options) {
  if (options == nullptr) {
    return ArrowIpcVectorSerde::ArrowIpcOptions{};
  }
  if (auto* arrowOptions =
          dynamic_cast<const ArrowIpcVectorSerde::ArrowIpcOptions*>(options)) {
    return *arrowOptions;
  }
  ArrowIpcVectorSerde::ArrowIpcOptions resolved;
  resolved.compressionKind = options->compressionKind;
  resolved.minCompressionRatio = options->minCompressionRatio;
  resolved.minCompressionPageSizeBytes = options->minCompressionPageSizeBytes;
  return resolved;
}

// Maps Velox CompressionKind onto the matching arrow::Compression::type
// supported by Arrow IPC stream format. Only NONE / LZ4 / ZSTD are wired.
// Note: Arrow IPC uses LZ4_FRAME for the stream-format LZ4 codec
// (raw LZ4 is rejected by IpcWriteOptions::CheckCompressionSupported).
std::shared_ptr<arrow::util::Codec> arrowCodecFor(
    common::CompressionKind kind) {
  switch (kind) {
    case common::CompressionKind::CompressionKind_NONE:
      return nullptr;
    case common::CompressionKind::CompressionKind_LZ4:
      return std::shared_ptr<arrow::util::Codec>(arrowResultOrThrow(
          arrow::util::Codec::Create(arrow::Compression::LZ4_FRAME)));
    case common::CompressionKind::CompressionKind_ZSTD:
      return std::shared_ptr<arrow::util::Codec>(arrowResultOrThrow(
          arrow::util::Codec::Create(arrow::Compression::ZSTD)));
    default:
      VELOX_UNSUPPORTED(
          "ArrowIpcVectorSerde does not support compression kind {}",
          common::compressionKindToString(kind));
  }
}

// Builds an arrow::Schema from a Velox RowTypePtr by exporting a 0-length
// probe vector through the C-data bridge. Using a probe instead of a real
// vector means the schema is independent of the encodings of the vectors
// that will be serialized later.
std::shared_ptr<arrow::Schema> schemaFromRowType(
    const RowTypePtr& type,
    memory::MemoryPool* pool,
    const ArrowIpcVectorSerde::ArrowIpcOptions& options) {
  auto probe = BaseVector::create(type, 0, pool);
  ArrowSchema arrowSchema;
  exportToArrow(probe, arrowSchema, options.arrowOptions);
  return arrowResultOrThrow(arrow::ImportSchema(&arrowSchema));
}

// Materializes a slice [offset, offset + length) of `source` as a new
// flat RowVector owned by `pool`. Used to bring a subrange of a Velox
// vector into a contiguous shape so it can be exported through the
// Arrow C-data bridge as a single RecordBatch.
RowVectorPtr materializeSlice(
    const RowVectorPtr& source,
    vector_size_t offset,
    vector_size_t length,
    memory::MemoryPool* pool) {
  auto target = BaseVector::create<RowVector>(source->type(), length, pool);
  target->copy(source.get(), 0, offset, length);
  return target;
}

// Concatenates a list of (RowVector, IndexRange) entries into a single
// flat RowVector in `pool`. Returns an empty vector if there are no rows
// to materialize.
RowVectorPtr materializeBuffered(
    const RowTypePtr& type,
    const std::vector<std::pair<RowVectorPtr, std::vector<IndexRange>>>&
        buffered,
    memory::MemoryPool* pool) {
  vector_size_t total = 0;
  for (const auto& [vector, ranges] : buffered) {
    for (const auto& range : ranges) {
      total += range.size;
    }
  }
  auto result = BaseVector::create<RowVector>(type, total, pool);
  vector_size_t targetOffset = 0;
  for (const auto& [vector, ranges] : buffered) {
    for (const auto& range : ranges) {
      result->copy(vector.get(), targetOffset, range.begin, range.size);
      targetOffset += range.size;
    }
  }
  return result;
}

// Writes one RowVector as a single-batch Arrow IPC stream onto `out`.
// `arrowSchema` carries the schema used by the writer; it must be
// compatible with the schema implied by `vector`'s type.
void writeRecordBatch(
    const RowVectorPtr& vector,
    const std::shared_ptr<arrow::Schema>& arrowSchema,
    memory::MemoryPool* pool,
    const ArrowIpcVectorSerde::ArrowIpcOptions& options,
    OutputStream* out) {
  ArrowArray arrowArray;
  exportToArrow(vector, arrowArray, pool, options.arrowOptions);
  auto batch =
      arrowResultOrThrow(arrow::ImportRecordBatch(&arrowArray, arrowSchema));

  VeloxArrowOutputStream sink(out);
  auto ipcOptions = arrow::ipc::IpcWriteOptions::Defaults();
  if (auto codec = arrowCodecFor(options.compressionKind)) {
    ipcOptions.codec = std::move(codec);
    // Honor PrestoSerializer-style minCompressionRatio: tell Arrow IPC
    // to fall back to uncompressed buffers when the codec doesn't save
    // at least (1 - minCompressionRatio) of space. Matches the behavior
    // of PrestoBatchVectorSerializer so callers get the same "skip
    // compression on incompressible data" semantics regardless of which
    // serde they pick.
    ipcOptions.min_space_savings = 1.0 - options.minCompressionRatio;
  }
  auto writer = arrowResultOrThrow(
      arrow::ipc::MakeStreamWriter(&sink, arrowSchema, ipcOptions));
  checkArrowStatus(writer->WriteRecordBatch(*batch));
  checkArrowStatus(writer->Close());
}

class ArrowIpcIterativeSerializer : public IterativeVectorSerializer {
 public:
  ArrowIpcIterativeSerializer(
      RowTypePtr type,
      memory::MemoryPool* pool,
      ArrowIpcVectorSerde::ArrowIpcOptions options)
      : type_(std::move(type)), pool_(pool), options_(std::move(options)) {}

  void append(
      const RowVectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      Scratch& /*scratch*/) override {
    VELOX_CHECK_NOT_NULL(vector, "Cannot append null vector");
    if (ranges.empty()) {
      return;
    }
    std::vector<IndexRange> rangesCopy(ranges.begin(), ranges.end());
    vector_size_t added = 0;
    for (const auto& range : rangesCopy) {
      added += range.size;
    }
    numRows_ += added;
    buffered_.emplace_back(vector, std::move(rangesCopy));
  }

  size_t maxSerializedSize() const override {
    // Without serializing we cannot know the exact size; return a rough
    // upper bound that scales with row count and approximate row width.
    constexpr size_t kPerRowGuess = 64;
    constexpr size_t kIpcOverhead = 1024;
    return numRows_ * kPerRowGuess + kIpcOverhead;
  }

  void flush(OutputStream* out) override {
    auto arrowSchema = schemaFromRowType(type_, pool_, options_);
    auto materialized = materializeBuffered(type_, buffered_, pool_);
    writeRecordBatch(materialized, arrowSchema, pool_, options_, out);
  }

  void clear() override {
    buffered_.clear();
    numRows_ = 0;
  }

 private:
  const RowTypePtr type_;
  memory::MemoryPool* const pool_;
  const ArrowIpcVectorSerde::ArrowIpcOptions options_;

  std::vector<std::pair<RowVectorPtr, std::vector<IndexRange>>> buffered_;
  vector_size_t numRows_{0};
};

class ArrowIpcBatchSerializer : public BatchVectorSerializer {
 public:
  ArrowIpcBatchSerializer(
      memory::MemoryPool* pool,
      ArrowIpcVectorSerde::ArrowIpcOptions options)
      : pool_(pool), options_(std::move(options)) {}

  void serialize(
      const RowVectorPtr& vector,
      const folly::Range<const IndexRange*>& ranges,
      Scratch& /*scratch*/,
      OutputStream* stream) override {
    VELOX_CHECK_NOT_NULL(vector, "Vector to serialize is null.");
    VELOX_CHECK_NOT_NULL(stream, "Stream to serialize out to is null.");

    auto rowType = asRowType(vector->type());
    auto arrowSchema = schemaFromRowType(rowType, pool_, options_);

    RowVectorPtr toSerialize;
    if (ranges.size() == 1 && ranges[0].begin == 0 &&
        ranges[0].size == vector->size()) {
      toSerialize = vector;
    } else if (ranges.size() == 1) {
      toSerialize =
          materializeSlice(vector, ranges[0].begin, ranges[0].size, pool_);
    } else {
      std::vector<std::pair<RowVectorPtr, std::vector<IndexRange>>> buffered;
      buffered.emplace_back(
          vector, std::vector<IndexRange>(ranges.begin(), ranges.end()));
      toSerialize = materializeBuffered(rowType, buffered, pool_);
    }

    writeRecordBatch(toSerialize, arrowSchema, pool_, options_, stream);
  }

  void estimateSerializedSize(
      VectorPtr /*vector*/,
      const folly::Range<const IndexRange*>& ranges,
      vector_size_t** sizes,
      Scratch& /*scratch*/) override {
    constexpr vector_size_t kPerRowGuess = 64;
    for (const auto& range : ranges) {
      for (vector_size_t i = 0; i < range.size; ++i) {
        *sizes[range.begin + i] += kPerRowGuess;
      }
    }
  }

 private:
  memory::MemoryPool* const pool_;
  const ArrowIpcVectorSerde::ArrowIpcOptions options_;
};

} // namespace

std::unique_ptr<IterativeVectorSerializer>
ArrowIpcVectorSerde::createIterativeSerializer(
    RowTypePtr type,
    int32_t /*numRows*/,
    StreamArena* streamArena,
    const Options* options) {
  return std::make_unique<ArrowIpcIterativeSerializer>(
      std::move(type), streamArena->pool(), resolveOptions(options));
}

std::unique_ptr<BatchVectorSerializer>
ArrowIpcVectorSerde::createBatchSerializer(
    memory::MemoryPool* pool,
    const Options* options) {
  return std::make_unique<ArrowIpcBatchSerializer>(
      pool, resolveOptions(options));
}

void ArrowIpcVectorSerde::deserialize(
    ByteInputStream* source,
    velox::memory::MemoryPool* pool,
    RowTypePtr type,
    RowVectorPtr* result,
    const Options* /*options*/) {
  // Stream-format IPC only needs a forward-only InputStream, so wrap the
  // Velox ByteInputStream directly and let Arrow pull bytes on demand.
  // Avoids the previous "memcpy the entire payload into one arrow::Buffer
  // before parsing" path, which profiling showed was the dominant cost
  // (~88% of deserialize time).
  auto inputStream = std::make_shared<VeloxArrowInputStream>(source);
  auto reader = arrowResultOrThrow(
      arrow::ipc::RecordBatchStreamReader::Open(
          inputStream, arrow::ipc::IpcReadOptions::Defaults()));

  // Walk every batch in the stream. Each batch is exported via the
  // C-data interface and imported as a Velox RowVector. If there are
  // multiple batches, append them into a single contiguous result.
  RowVectorPtr accumulated;
  vector_size_t totalRows = 0;
  while (true) {
    auto batch = arrowResultOrThrow(reader->Next());
    if (batch == nullptr) {
      break;
    }
    ArrowSchema arrowSchema;
    ArrowArray arrowArray;
    checkArrowStatus(arrow::ExportSchema(*reader->schema(), &arrowSchema));
    checkArrowStatus(arrow::ExportRecordBatch(*batch, &arrowArray));
    auto vector = importFromArrowAsOwner(arrowSchema, arrowArray, pool);
    auto rowVector = std::dynamic_pointer_cast<RowVector>(vector);
    VELOX_CHECK_NOT_NULL(
        rowVector, "Arrow IPC stream did not deserialize to a RowVector");

    if (accumulated == nullptr) {
      accumulated = rowVector;
      totalRows = rowVector->size();
      continue;
    }
    auto next = BaseVector::create<RowVector>(
        type, totalRows + rowVector->size(), pool);
    next->copy(accumulated.get(), 0, 0, totalRows);
    next->copy(rowVector.get(), totalRows, 0, rowVector->size());
    accumulated = next;
    totalRows = next->size();
  }

  if (accumulated == nullptr) {
    *result = BaseVector::create<RowVector>(type, 0, pool);
  } else {
    *result = accumulated;
  }
}

// static
void ArrowIpcVectorSerde::registerVectorSerde() {
  velox::registerVectorSerde(std::make_unique<ArrowIpcVectorSerde>());
}

// static
void ArrowIpcVectorSerde::registerNamedVectorSerde() {
  velox::registerNamedVectorSerde(
      kSerdeKind, std::make_unique<ArrowIpcVectorSerde>());
}

// static
void ArrowIpcVectorSerde::tryRegisterNamedVectorSerde() {
  if (!velox::isRegisteredNamedVectorSerde(kSerdeKind)) {
    velox::registerNamedVectorSerde(
        kSerdeKind, std::make_unique<ArrowIpcVectorSerde>());
  }
}

} // namespace facebook::velox::serializer
