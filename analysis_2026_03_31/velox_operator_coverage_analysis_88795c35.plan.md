---
name: Velox Operator Coverage Analysis
overview: A detailed analysis of Velox core operators vs. cudf GPU operators, mapping correspondence, identifying gaps, and presenting a verified coverage matrix.
todos: []
isProject: false
---

# Velox Operator Coverage Analysis: Core vs. cudf GPU

## Summary Counts (Verified)

- **Velox Core Operators** (concrete, production): **44**
- **cudf GPU Operators** (exec::Operator subclasses): **12**
- **cudf operators with core correspondence**: **9**
- **cudf operators that are new / GPU-specific**: **3**
- **cudf Operator Adapters** (registered): **13** (these intercept core operators for GPU routing)
- **Core operators with some form of cudf coverage** (via operator + adapter): **13** (29.5%)
- **Core operators with NO cudf coverage**: **31** (70.5%)

---

## 1. All 44 Velox Core Operators

Source: [velox/exec/*.h](velox/exec/) -- classes inheriting from `Operator` or `SourceOperator`, excluding abstract bases (`Operator`, `SourceOperator`, `Merge`) and the `BlockedOperator` utility.

### Source Operators (9)


| #   | Operator          | Header                           | Base                   |
| --- | ----------------- | -------------------------------- | ---------------------- |
| 1   | ArrowStream       | `velox/exec/ArrowStream.h`       | SourceOperator         |
| 2   | Exchange          | `velox/exec/Exchange.h`          | SourceOperator         |
| 3   | LocalExchange     | `velox/exec/LocalPartition.h`    | SourceOperator         |
| 4   | LocalMerge        | `velox/exec/Merge.h`             | Merge > SourceOperator |
| 5   | MergeExchange     | `velox/exec/Merge.h`             | Merge > SourceOperator |
| 6   | MixedUnion        | `velox/exec/MixedUnion.h`        | SourceOperator         |
| 7   | OperatorTraceScan | `velox/exec/OperatorTraceScan.h` | SourceOperator         |
| 8   | TableScan         | `velox/exec/TableScan.h`         | SourceOperator         |
| 9   | Values            | `velox/exec/Values.h`            | SourceOperator         |


### Pipeline Operators (33)


| #   | Operator                 | Header                                  |
| --- | ------------------------ | --------------------------------------- |
| 10  | AssignUniqueId           | `velox/exec/AssignUniqueId.h`           |
| 11  | CallbackSink             | `velox/exec/CallbackSink.h`             |
| 12  | EnforceDistinct          | `velox/exec/EnforceDistinct.h`          |
| 13  | EnforceSingleRow         | `velox/exec/EnforceSingleRow.h`         |
| 14  | Expand                   | `velox/exec/Expand.h`                   |
| 15  | FilterProject            | `velox/exec/FilterProject.h`            |
| 16  | GroupId                  | `velox/exec/GroupId.h`                  |
| 17  | HashAggregation          | `velox/exec/HashAggregation.h`          |
| 18  | HashBuild                | `velox/exec/HashBuild.h`                |
| 19  | HashProbe                | `velox/exec/HashProbe.h`                |
| 20  | IndexLookupJoin          | `velox/exec/IndexLookupJoin.h`          |
| 21  | Limit                    | `velox/exec/Limit.h`                    |
| 22  | LocalPartition           | `velox/exec/LocalPartition.h`           |
| 23  | MarkDistinct             | `velox/exec/MarkDistinct.h`             |
| 24  | MarkSorted               | `velox/exec/MarkSorted.h`               |
| 25  | MergeJoin                | `velox/exec/MergeJoin.h`                |
| 26  | NestedLoopJoinBuild      | `velox/exec/NestedLoopJoinBuild.h`      |
| 27  | NestedLoopJoinProbe      | `velox/exec/NestedLoopJoinProbe.h`      |
| 28  | OrderBy                  | `velox/exec/OrderBy.h`                  |
| 29  | ParallelProject          | `velox/exec/ParallelProject.h`          |
| 30  | PartitionedOutput        | `velox/exec/PartitionedOutput.h`        |
| 31  | RowNumber                | `velox/exec/RowNumber.h`                |
| 32  | SpatialJoinBuild         | `velox/exec/SpatialJoinBuild.h`         |
| 33  | SpatialJoinProbe         | `velox/exec/SpatialJoinProbe.h`         |
| 34  | StreamingAggregation     | `velox/exec/StreamingAggregation.h`     |
| 35  | StreamingEnforceDistinct | `velox/exec/StreamingEnforceDistinct.h` |
| 36  | TableWriter              | `velox/exec/TableWriter.h`              |
| 37  | TableWriteMerge          | `velox/exec/TableWriteMerge.h`          |
| 38  | TopN                     | `velox/exec/TopN.h`                     |
| 39  | TopNRowNumber            | `velox/exec/TopNRowNumber.h`            |
| 40  | Unnest                   | `velox/exec/Unnest.h`                   |
| 41  | Window                   | `velox/exec/Window.h`                   |


### Specialized Variants and Extensions (3)


| #   | Operator                              | Header                                   | Base           |
| --- | ------------------------------------- | ---------------------------------------- | -------------- |
| 42  | ScaleWriterLocalPartition             | `velox/exec/ScaleWriterLocalPartition.h` | LocalPartition |
| 43  | ScaleWriterPartitioningLocalPartition | `velox/exec/ScaleWriterLocalPartition.h` | LocalPartition |
| 44  | RPCOperator                           | `velox/exec/rpc/RPCOperator.h`           | Operator       |


---

## 2. All 12 cudf GPU Operators

Source: [velox/experimental/cudf/exec/*.h](velox/experimental/cudf/exec/) -- classes inheriting from `exec::Operator`.


| #   | cudf Operator       | Header                            | Additional Base |
| --- | ------------------- | --------------------------------- | --------------- |
| 1   | CudfAssignUniqueId  | `cudf/exec/CudfAssignUniqueId.h`  | NvtxHelper      |
| 2   | CudfBatchConcat     | `cudf/exec/CudfBatchConcat.h`     | CudfOperator    |
| 3   | CudfFilterProject   | `cudf/exec/CudfFilterProject.h`   | NvtxHelper      |
| 4   | CudfFromVelox       | `cudf/exec/CudfConversion.h`      | NvtxHelper      |
| 5   | CudfHashAggregation | `cudf/exec/CudfHashAggregation.h` | NvtxHelper      |
| 6   | CudfHashJoinBuild   | `cudf/exec/CudfHashJoin.h`        | NvtxHelper      |
| 7   | CudfHashJoinProbe   | `cudf/exec/CudfHashJoin.h`        | NvtxHelper      |
| 8   | CudfLimit           | `cudf/exec/CudfLimit.h`           | NvtxHelper      |
| 9   | CudfLocalPartition  | `cudf/exec/CudfLocalPartition.h`  | NvtxHelper      |
| 10  | CudfOrderBy         | `cudf/exec/CudfOrderBy.h`         | NvtxHelper      |
| 11  | CudfToVelox         | `cudf/exec/CudfConversion.h`      | NvtxHelper      |
| 12  | CudfTopN            | `cudf/exec/CudfTopN.h`            | NvtxHelper      |


---

## 3. Correspondence Mapping: cudf Operators to Core Operators

### 9 cudf Operators WITH Core Correspondence


| cudf Operator       | Core Velox Operator | Relationship                             |
| ------------------- | ------------------- | ---------------------------------------- |
| CudfAssignUniqueId  | AssignUniqueId      | GPU replacement                          |
| CudfFilterProject   | FilterProject       | GPU replacement (filter + project fused) |
| CudfHashAggregation | HashAggregation     | GPU replacement (partial/final/single)   |
| CudfHashJoinBuild   | HashBuild           | GPU replacement (build side)             |
| CudfHashJoinProbe   | HashProbe           | GPU replacement (probe side)             |
| CudfLimit           | Limit               | GPU replacement                          |
| CudfLocalPartition  | LocalPartition      | GPU replacement                          |
| CudfOrderBy         | OrderBy             | GPU replacement                          |
| CudfTopN            | TopN                | GPU replacement                          |


### 3 cudf Operators WITHOUT Core Correspondence (GPU-specific / New)


| cudf Operator   | Purpose                                                                                   |
| --------------- | ----------------------------------------------------------------------------------------- |
| CudfBatchConcat | Concatenates multiple small GPU batches into larger ones for GPU efficiency               |
| CudfFromVelox   | Converts Velox RowVectors (CPU) to cudf tables (GPU) -- CPU-to-GPU transfer boundary      |
| CudfToVelox     | Converts cudf tables (GPU) back to Velox RowVectors (CPU) -- GPU-to-CPU transfer boundary |


These 3 are GPU pipeline infrastructure operators that handle the CPU/GPU boundary and GPU memory management -- they have no CPU counterparts because they only exist to bridge the two execution domains.

---

## 4. Operator Adapter Coverage

Beyond the 12 `exec::Operator` classes, the cudf module also registers 13 `OperatorAdapter` instances ([OperatorAdapters.cpp:768-780](velox/experimental/cudf/exec/OperatorAdapters.cpp)) that intercept core operators and route them to GPU execution or provide GPU-compatible passthrough:


| Adapter               | Core Operator Intercepted | Action                                                |
| --------------------- | ------------------------- | ----------------------------------------------------- |
| TableScanAdapter      | TableScan                 | Routes to CudfHiveDataSource for GPU-accelerated scan |
| FilterProjectAdapter  | FilterProject             | Replaces with CudfFilterProject                       |
| AggregationAdapter    | HashAggregation           | Replaces with CudfHashAggregation                     |
| HashJoinBuildAdapter  | HashBuild                 | Replaces with CudfHashJoinBuild                       |
| HashJoinProbeAdapter  | HashProbe                 | Replaces with CudfHashJoinProbe                       |
| OrderByAdapter        | OrderBy                   | Replaces with CudfOrderBy                             |
| TopNAdapter           | TopN                      | Replaces with CudfTopN                                |
| LimitAdapter          | Limit                     | Replaces with CudfLimit                               |
| LocalPartitionAdapter | LocalPartition            | Replaces with CudfLocalPartition                      |
| LocalExchangeAdapter  | LocalExchange             | GPU-compatible passthrough                            |
| AssignUniqueIdAdapter | AssignUniqueId            | Replaces with CudfAssignUniqueId                      |
| ValuesAdapter         | Values                    | GPU-compatible passthrough                            |
| CallbackSinkAdapter   | CallbackSink              | Handles GPU-to-CPU conversion at sink boundary        |


---

## 5. Full Coverage Matrix


| #   | Core Velox Operator       | Has cudf Operator   | Has cudf Adapter      | GPU Status   |
| --- | ------------------------- | ------------------- | --------------------- | ------------ |
| 1   | ArrowStream               | No                  | No                    | NOT COVERED  |
| 2   | AssignUniqueId            | CudfAssignUniqueId  | AssignUniqueIdAdapter | COVERED      |
| 3   | CallbackSink              | No                  | CallbackSinkAdapter   | ADAPTER ONLY |
| 4   | EnforceDistinct           | No                  | No                    | NOT COVERED  |
| 5   | EnforceSingleRow          | No                  | No                    | NOT COVERED  |
| 6   | Exchange                  | No                  | No                    | NOT COVERED  |
| 7   | Expand                    | No                  | No                    | NOT COVERED  |
| 8   | FilterProject             | CudfFilterProject   | FilterProjectAdapter  | COVERED      |
| 9   | GroupId                   | No                  | No                    | NOT COVERED  |
| 10  | HashAggregation           | CudfHashAggregation | AggregationAdapter    | COVERED      |
| 11  | HashBuild                 | CudfHashJoinBuild   | HashJoinBuildAdapter  | COVERED      |
| 12  | HashProbe                 | CudfHashJoinProbe   | HashJoinProbeAdapter  | COVERED      |
| 13  | IndexLookupJoin           | No                  | No                    | NOT COVERED  |
| 14  | Limit                     | CudfLimit           | LimitAdapter          | COVERED      |
| 15  | LocalExchange             | No                  | LocalExchangeAdapter  | ADAPTER ONLY |
| 16  | LocalMerge                | No                  | No                    | NOT COVERED  |
| 17  | LocalPartition            | CudfLocalPartition  | LocalPartitionAdapter | COVERED      |
| 18  | MarkDistinct              | No                  | No                    | NOT COVERED  |
| 19  | MarkSorted                | No                  | No                    | NOT COVERED  |
| 20  | MergeExchange             | No                  | No                    | NOT COVERED  |
| 21  | MergeJoin                 | No                  | No                    | NOT COVERED  |
| 22  | MixedUnion                | No                  | No                    | NOT COVERED  |
| 23  | NestedLoopJoinBuild       | No                  | No                    | NOT COVERED  |
| 24  | NestedLoopJoinProbe       | No                  | No                    | NOT COVERED  |
| 25  | OperatorTraceScan         | No                  | No                    | NOT COVERED  |
| 26  | OrderBy                   | CudfOrderBy         | OrderByAdapter        | COVERED      |
| 27  | ParallelProject           | No                  | No                    | NOT COVERED  |
| 28  | PartitionedOutput         | No                  | No                    | NOT COVERED  |
| 29  | RowNumber                 | No                  | No                    | NOT COVERED  |
| 30  | RPCOperator               | No                  | No                    | NOT COVERED  |
| 31  | ScaleWriterLocalPartition | No                  | No                    | NOT COVERED  |
| 32  | ScaleWriterPartitioningLP | No                  | No                    | NOT COVERED  |
| 33  | SpatialJoinBuild          | No                  | No                    | NOT COVERED  |
| 34  | SpatialJoinProbe          | No                  | No                    | NOT COVERED  |
| 35  | StreamingAggregation      | No                  | No                    | NOT COVERED  |
| 36  | StreamingEnforceDistinct  | No                  | No                    | NOT COVERED  |
| 37  | TableScan                 | No                  | TableScanAdapter      | ADAPTER ONLY |
| 38  | TableWriter               | No                  | No                    | NOT COVERED  |
| 39  | TableWriteMerge           | No                  | No                    | NOT COVERED  |
| 40  | TopN                      | CudfTopN            | TopNAdapter           | COVERED      |
| 41  | TopNRowNumber             | No                  | No                    | NOT COVERED  |
| 42  | Unnest                    | No                  | No                    | NOT COVERED  |
| 43  | Values                    | No                  | ValuesAdapter         | ADAPTER ONLY |
| 44  | Window                    | No                  | No                    | NOT COVERED  |


### GPU-only Operators (no core equivalent)


| cudf Operator   | Purpose                                |
| --------------- | -------------------------------------- |
| CudfBatchConcat | GPU batch concatenation for efficiency |
| CudfFromVelox   | CPU -> GPU data transfer boundary      |
| CudfToVelox     | GPU -> CPU data transfer boundary      |


---

## 6. Coverage Summary

- **COVERED** (cudf operator + adapter): 9 operators (20.5%)
- **ADAPTER ONLY** (adapter intercept, no dedicated GPU operator class): 4 operators (9.1%)
- **NOT COVERED**: 31 operators (70.5%)
- **Total with any GPU support**: 13 / 44 = 29.5%

### Methodology Notes

- Counts are derived from `class X : public Operator/SourceOperator` declarations in header files under `velox/exec/` (including subdirectories and indirect inheritance)
- Abstract bases (`Operator`, `SourceOperator`, `Merge`) and the `BlockedOperator` utility are excluded from the core count
- cudf operator classes are those under `velox/experimental/cudf/exec/` inheriting from `exec::Operator`
- Adapter registrations are verified from [OperatorAdapters.cpp lines 768-780](velox/experimental/cudf/exec/OperatorAdapters.cpp)

