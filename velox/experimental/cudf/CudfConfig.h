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

#include <cudf/types.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

struct CudfConfig {
  /// Keys used by the initialize() method.
  static constexpr const char* kCudfEnabled{"cudf.enabled"};
  static constexpr const char* kCudfDebugEnabled{"cudf.debug_enabled"};
  static constexpr const char* kCudfMemoryResource{"cudf.memory_resource"};
  static constexpr const char* kCudfMemoryPercent{"cudf.memory_percent"};
  static constexpr const char* kCudfFunctionNamePrefix{
      "cudf.function_name_prefix"};
  static constexpr const char* kCudfAstExpressionEnabled{
      "cudf.ast_expression_enabled"};
  static constexpr const char* kCudfAstExpressionPriority{
      "cudf.ast_expression_priority"};
  static constexpr const char* kCudfJitExpressionEnabled{
      "cudf.jit_expression_enabled"};
  static constexpr const char* kCudfJitExpressionPriority{
      "cudf.jit_expression_priority"};
  static constexpr const char* kCudfOutputMr{"cudf.output_mr"};
  static constexpr const char* kCudfAllowCpuFallback{"cudf.allow_cpu_fallback"};
  static constexpr const char* kCudfLogFallback{"cudf.log_fallback"};
  static constexpr const char* kCudfBatchSizeMinThreshold{
      "cudf.batch_size_min_threshold"};
  /// Minimum buffered GPU byte target for CudfBatchConcat.
  static constexpr const char* kCudfBatchSizeMinBytes{
      "cudf.batch_size_min_bytes"};
  static constexpr const char* kCudfBatchSizeMaxThreshold{
      "cudf.batch_size_max_threshold"};
  /// Hash table occupancy for cudf::hash_join build tables.
  static constexpr const char* kCudfHashJoinLoadFactor{
      "cudf.hash_join_load_factor"};
  /// Build row count at or above which hashJoinLoadFactor is applied.
  static constexpr const char* kCudfHashJoinDenseLoadFactorMinRows{
      "cudf.hash_join_dense_load_factor_min_rows"};
  /// Live device bytes above which large operations are admitted one at a time.
  static constexpr const char* kCudfAdmissionThresholdBytes{
      "cudf.admission_threshold_bytes"};
  /// Hint for how many drivers share the device, used to size per-driver
  /// device-derived budgets.
  /// Row bound for a single join probe output batch.
  static constexpr const char* kCudfJoinOutputBatchRows{
      "cudf.join_output_batch_rows"};
  static constexpr const char* kCudfMaxDriversPerTaskHint{
      "cudf.max_drivers_per_task_hint"};
  static constexpr const char* kCudfConcatOptimizationEnabled{
      "cudf.concat_optimization_enabled"};
  /// Group count at or above which a final aggregation holds its state as
  /// device partitions. Zero disables partitioning entirely.
  static constexpr const char* kCudfPartitionedGroupbyMinGroups{
      "cudf.partitioned_groupby_min_groups"};
  static constexpr const char* kCudfStreamingGroupbyEnabled{
      "cudf.streaming_groupby_enabled"};
  static constexpr const char* kCudfStreamingGroupbyCapacityMultiplier{
      "cudf.streaming_groupby_capacity_multiplier"};
  static constexpr const char* kCudfTimestampUnit{"cudf.timestamp_unit"};
  static constexpr const char* kUcxExchange{"cudf.exchange"};
  static constexpr const char* kUcxxErrorHandling{"ucxx.error_handling"};
  static constexpr const char* kUcxIntraNodeExchange{
      "cudf.intra_node_exchange"};
  static constexpr const char* kUcxxBlockingProgress{"ucxx.blocking_progress"};
  static constexpr const char* kUcxExchangeLogLevel{"cudf.exchange_log_level"};
  static constexpr const char* kUcxPartitionedOutputBatchRows{
      "cudf.partitioned_output_batch_rows"};
  /// Query session configs for the cuDF Operators.
  static constexpr const char* kCudfTopNBatchSize{"cudf.topk_batch_size"};

  /// Singleton CudfConfig instance.
  /// Clients must set the configs below before invoking registerCudf().
  static CudfConfig& getInstance();

  /// Initialize from a map with the above keys.
  void initialize(std::unordered_map<std::string, std::string>&&);

  /// Enable cudf by default.
  /// Clients can disable here and enable it via the QueryConfig as well.
  bool enabled{true};

  /// Enable debug printing.
  bool debugEnabled{false};

  /// Allow fallback to CPU operators if GPU operator replacement fails.
  bool allowCpuFallback{true};

  /// Enable GPU exchange operators (UcxExchange / UcxPartitionedOutput). This
  /// is a capability, not a request: it is read once at registerCudf() to
  /// decide whether the UCX transports are registered in this process at all.
  /// Which transport a given edge uses is named per node in the plan, so
  /// different edges of one plan may differ. Naming a transport nothing
  /// registered -- kUcx while this is false, or on a worker built without the
  /// UCX exchange -- is a user error from exec::Task, never a silent fallback.
  bool exchange{false};

  /// Whether to enable error handling in UCXX endpoints.
  bool ucxxErrorHandling{true};

  /// Whether intra-node exchange optimization is enabled.
  bool intraNodeExchange{false};

  /// Whether the UCX worker is set up for UCXX blocking progress mode: the
  /// wakeup feature on the context plus an epoll file descriptor on the worker,
  /// which is also what lets enqueueing work signal it. False creates a
  /// tag/active-message-only worker that the progress loop polls.
  ///
  /// Requesting a feature the fabric cannot provide removes that transport from
  /// UCX's selection instead of failing, so setting this on a fabric without
  /// wakeup support silently costs RDMA. Tested with this and
  /// kUcxxErrorHandling both true on InfiniBand / A100, and both false on AWS
  /// SRD, which supports neither and otherwise stops using RDMA.
  bool ucxxBlockingProgress{true};

  /// VLOG level for ucx-exchange source files.
  int32_t exchangeLogLevel{0};

  /// Minimum number of rows to accumulate in UCX partitioned output before
  /// flushing. Small inputs are buffered and concatenated when this threshold
  /// is reached, avoiding pathologically small exchange chunks. Set to 0 to
  /// disable accumulation.
  int64_t partitionedOutputBatchRows{10'000};

  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  std::string memoryResource{"async"};

  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources.
  int32_t memoryPercent{50};

  /// Memory resource for output vectors. When set to a value different from
  /// memoryResource, a separate MR is created for output allocations.
  /// When empty, the main memoryResource is used.
  std::string outputMemoryResource;

  /// Register all the functions with the functionNamePrefix.
  std::string functionNamePrefix;

  /// Enable AST in expression evaluation.
  bool astExpressionEnabled{true};

  /// Enable JIT in expression evaluation.
  bool jitExpressionEnabled{true};

  /// Priority of AST expression. Expression with higher priority is chosen for
  /// a given root expression.
  /// Example:
  /// Priority of expression that uses individual cuDF functions is 50.
  /// If AST priority is 100 then for a velox expression node that is supported
  /// by both, AST will be chosen as replacement for cudf execution, if AST
  /// priority is 25 then standalone cudf function is chosen.
  int astExpressionPriority{100};

  /// Priority of JIT expression.
  int jitExpressionPriority{101};

  /// Whether to log a reason for falling back to Velox CPU execution.
  bool logFallback{true};

  /// Whether to insert CudfBatchConcat operators before supported Cudf
  /// operators.
  /// This can improve performance by reducing the number of cuda kernel
  /// launches on addInput of certain operators. Inputs are collected until
  /// batchSizeMinBytes is reached, or batchSizeMinThreshold otherwise.
  /// batchSizeMaxThreshold limits the rows in a concatenated batch.
  bool concatOptimizationEnabled{false};

  /// Use libcudf's persistent streaming_groupby for eligible final grouped
  /// aggregations. This is opt-in while it supports only a subset of the
  /// aggregation combinations supported by the regular cuDF groupby path.
  bool streamingGroupbyEnabled{false};

  /// Multiplier used to derive streaming_groupby's initial logical capacity
  /// from the first batch and to grow capacity when it is exhausted.
  double streamingGroupbyCapacityMultiplier{2.0};

  /// Live device bytes above which the few operations large enough to fill the
  /// device on their own are admitted one at a time rather than run
  /// concurrently.
  ///
  /// Drivers do not coordinate with each other; they consult one counter
  /// before their single dominant allocation. Below the threshold nothing is
  /// serialised and the counter is not even read, so a query that is not under
  /// pressure is untouched.
  ///
  /// Zero disables it, which is the default: this changes when work runs, and
  /// that is worth opting into rather than inheriting.
  uint64_t admissionThresholdBytes{0};

  /// Minimum rows to accumulate before GPU-side concatenation when
  /// batchSizeMinBytes is not configured. This is also the fallback target for
  /// zero-column vectors, which have no GPU buffers to count (default 100k).
  int32_t batchSizeMinThreshold{100000};

  /// Optional minimum GPU byte target for concatenation, measured by
  /// CudfVector::estimateFlatSize(). When unset, batchSizeMinThreshold applies.
  std::optional<uint64_t> batchSizeMinBytes;

  /// Maximum rows allowed in a concatenated batch (user configurable).
  /// When not set, cuDF's own `size_type::max()` is used.
  std::optional<int32_t> batchSizeMaxThreshold;

  /// Desired occupancy of the cudf::hash_join hash table in (0, 1]. The table
  /// stores one 8-byte (hash, row index) slot per capacity entry, so the build
  /// side of a join with N rows costs N / loadFactor * 8 bytes on top of the
  /// build table itself. libcudf's default is 0.5 (fastest probes); 0.8 halves
  /// the table for large builds at a small probe cost, which is what lets a
  /// 1.5 B-row build fit next to its own data on a 48 GiB GPU.
  double hashJoinLoadFactor{0.5};

  /// Builds with fewer rows than this keep libcudf's default occupancy, so a
  /// query whose joins comfortably fit is never slowed down by a denser table.
  /// The default is high enough that only multi-hundred-million-row builds -
  /// the ones whose hash table is a material fraction of the device - opt in.
  /// Zero means "derive from the device" (see gpu_defaults); a non-zero value
  /// is taken as configured and used as-is.
  uint64_t hashJoinDenseLoadFactorMinRows{0};

  /// Group count above which the incremental final aggregation splits its
  /// accumulated state into hash partitions that stay on the device, and from
  /// then on merges each input batch one partition at a time.
  ///
  /// The merge that grows the state concatenates it with the new batch and
  /// re-aggregates, so the old state, the concatenated copy, the hash table and
  /// the result are all resident at the peak - roughly twice the state.
  /// Partitioning does not make the state smaller; it makes that peak a
  /// function of one partition, which is sound because equal keys hash equally
  /// and a partition can be merged without consulting any other. Nothing leaves
  /// the device.
  ///
  /// Aggregations below the threshold keep exactly the path they used before,
  /// so the extra partitioning work only applies where the state was large
  /// enough to be a problem. Zero means "derive from the device" (see
  /// gpu_defaults); a non-zero value is taken as configured and used as-is.
  uint64_t partitionedGroupbyMinGroups{0};

  /// Maximum rows in one join probe output batch. A probe that matches many
  /// build rows produces an output far larger than either input, and that
  /// gather is the largest allocation the probe makes.
  ///
  /// Deliberately separate from batchSizeMaxThreshold, which is its fallback:
  /// that value also bounds the concatenated build table, and the two want
  /// opposite things. Splitting probe output finely is cheap, while splitting
  /// the build into many tables makes the probe loop over every one of them, so
  /// a single knob cannot serve both. Unset falls back to
  /// batchSizeMaxThreshold. Unset derives a value from the device (see
  /// gpu_defaults), which is what makes a large join fit without hand-tuning:
  /// inheriting the build cap gave batches sized for a table rather than for a
  /// gather, and a single one of those could be most of the device.
  std::optional<int32_t> joinOutputBatchRows;

  /// Drivers expected to share the device per task, used only to divide a
  /// device-derived per-driver budget. This is a hint: the real count is a
  /// query property (task.max-drivers-per-task) not known when the process
  /// starts, and getting it wrong only makes a budget slightly generous or
  /// slightly tight, never incorrect.
  int32_t maxDriversPerTaskHint{2};
  // Query config key for the TopN batch size in the cuDF TopN operator.
  int32_t topNBatchSize{5};

  /// Timestamp unit for cuDF timestamp types.
  /// Can be configured via kCudfTimestampUnit with string values:
  /// "s" (seconds), "ms" (milliseconds), "us" (microseconds), "ns"
  /// (nanoseconds).
  cudf::type_id timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS;
};

} // namespace facebook::velox::cudf_velox
