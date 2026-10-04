# Cudf Operator Input Queuing Report

Audit of `input_` / `inputs_` storage and `queuedInputBytes_` tracking across all cudf operators.

## Summary Table


| Operator            | Input Storage                                                             | Queues Input?           | `queuedInputBytes_` | Increment Point                                                | Decrement Point                                                        |
| ------------------- | ------------------------------------------------------------------------- | ----------------------- | ------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------- |
| CudfHashAggregation | `inputs_` (vectorCudfVectorPtr)                                           | Yes (final/global only) | Yes                 | `addInput()` — per input pushed                                | `getOutput()` — set to 0 on `inputs_.clear()`                          |
| CudfOrderBy         | `inputs_` (vectorCudfVectorPtr)                                           | Yes                     | Yes                 | `addInput()` — per input pushed                                | `noMoreInput()` — set to 0 on `inputs_.clear()`                        |
| CudfTopN            | `topNBatches_` (vectorCudfVectorPtr)                                      | Yes                     | Yes                 | `addInput()` — per batch pushed; reset to merged size on merge | `getOutput()` — set to 0 on `topNBatches_.clear()`                     |
| CudfHashJoinBuild   | `inputs_` (vectorCudfVectorPtr)                                           | Yes                     | Yes                 | `addInput()` — per input; `noMoreInput()` — per peer input     | `noMoreInput()` — set to 0 on `inputs_.clear()`                        |
| CudfHashJoinProbe   | `input_` (single RowVectorPtr) + `inputs_` (vector, RightSemiFilter only) | Yes                     | Yes                 | `addInput()` — per input set/pushed                            | `getOutput()` — set to 0 on `input_.reset()`                           |
| CudfFromVelox       | `inputs_` (vectorRowVectorPtr)                                            | Yes                     | Yes                 | `addInput()` — per input pushed                                | `getOutput()` — decremented per processed input via `inputs_.erase()`  |
| CudfToVelox         | `inputs_` (dequeCudfVectorPtr)                                            | Yes                     | Yes                 | `addInput()` — per input pushed                                | `getOutput()` — decremented per consumed/split input via `pop_front()` |
| CudfFilterProject   | `input_` (single RowVectorPtr)                                            | Yes (single)            | Yes                 | `addInput()` — set to input size                               | `getOutput()` — set to 0 on `input_.reset()`                           |
| CudfLimit           | `input_` (single RowVectorPtr)                                            | Yes (single)            | Yes                 | `addInput()` — set to input size                               | `getOutput()` — set to 0 on `input_.reset()` / `input_ = nullptr`      |
| CudfAssignUniqueId  | `input_` (single RowVectorPtr)                                            | Yes (single)            | Yes                 | `addInput()` — set to input size                               | `getOutput()` — set to 0 on `input_ = nullptr`                         |
| CudfLocalPartition  | None                                                                      | No (immediate)          | No                  | N/A                                                            | N/A                                                                    |


## Detailed Operator Analysis

### CudfHashAggregation

- **Storage:** `inputs_` — `std::vector<CudfVectorPtr>`
- **Queuing behavior:** Only queues for final aggregation or global aggregation. Partial (non-global) aggregation processes input immediately in `addInput()` and returns without queuing.
- **Increment:** `addInput()` adds `input->estimateFlatSize()` to `queuedInputBytes_` when pushing to `inputs_`.
- **Decrement:** `getOutput()` sets `queuedInputBytes_ = 0` and calls `inputs_.clear()` after processing.

### CudfOrderBy

- **Storage:** `inputs_` — `std::vector<CudfVectorPtr>`
- **Queuing behavior:** Accumulates all inputs until `noMoreInput()`, then sorts.
- **Increment:** `addInput()` adds `input->estimateFlatSize()` per pushed input.
- **Decrement:** `noMoreInput()` sets `queuedInputBytes_ = 0` and calls `inputs_.clear()` after concatenating and sorting.

### CudfTopN

- **Storage:** `topNBatches_` — `std::vector<CudfVectorPtr>`
- **Queuing behavior:** Accumulates top-K batches. Periodically merges when total size exceeds threshold. Final merge in `getOutput()`.
- **Increment:** `addInput()` adds `topKBatch->estimateFlatSize()` per batch. On mid-stream merge, resets to `result->estimateFlatSize()`.
- **Decrement:** `getOutput()` sets `queuedInputBytes_ = 0` and calls `topNBatches_.clear()`.

### CudfHashJoinBuild

- **Storage:** `inputs_` — `std::vector<CudfVectorPtr>`
- **Queuing behavior:** Accumulates build-side inputs. In `noMoreInput()`, collects inputs from peer drivers, builds hash tables, then clears.
- **Increment:** `addInput()` adds per input. `noMoreInput()` adds per peer input collected.
- **Decrement:** `noMoreInput()` sets `queuedInputBytes_ = 0` after `inputs_.clear()`.

### CudfHashJoinProbe

- **Storage:** `input_` (single `RowVectorPtr`) for most join types; `inputs_` (`std::vector<CudfVectorPtr>`) for `RightSemiFilterJoin`.
- **Queuing behavior:**
  - **Most joins:** Holds one input at a time in `input_`, consumed per `getOutput()` call.
  - **RightSemiFilterJoin:** Accumulates all inputs in `inputs_`, concatenates in `noMoreInput()` into `input_`, then processes.
- **Increment:** `addInput()` — sets to input size for single-input path; accumulates for RightSemiFilter path.
- **Decrement:** `getOutput()` sets `queuedInputBytes_ = 0` on `input_.reset()`.

### CudfFromVelox

- **Storage:** `inputs_` — `std::vector<RowVectorPtr>`
- **Queuing behavior:** Accumulates Velox RowVectors, converts batch-by-batch in `getOutput()`.
- **Increment:** `addInput()` adds `input->estimateFlatSize()` per input.
- **Decrement:** `getOutput()` decrements per processed input (`queuedInputBytes_ -= selectedInputs[i]->estimateFlatSize()`).

### CudfToVelox

- **Storage:** `inputs_` — `std::deque<CudfVectorPtr>`
- **Queuing behavior:** Accumulates CudfVectors, converts one at a time to Velox format in `getOutput()`. Supports splitting large inputs.
- **Increment:** `addInput()` adds per input. Split handling re-adds partial input size.
- **Decrement:** `getOutput()` decrements per consumed input via `pop_front()`. Split handling subtracts full size and adds back partial size.

### CudfFilterProject

- **Storage:** `input_` — single `RowVectorPtr` (inherited from Operator base)
- **Queuing behavior:** Holds exactly one input, processes it in `getOutput()`.
- **Increment:** `addInput()` sets `queuedInputBytes_` to input size.
- **Decrement:** `getOutput()` sets `queuedInputBytes_ = 0` on `input_.reset()`.

### CudfLimit

- **Storage:** `input_` — single `RowVectorPtr` (inherited from Operator base)
- **Queuing behavior:** Holds exactly one input. May skip, slice, or pass through depending on offset/limit state.
- **Increment:** `addInput()` sets `queuedInputBytes_` to `input_->estimateFlatSize()`.
- **Decrement:** `getOutput()` sets `queuedInputBytes_ = 0` at each return path (`input_.reset()` or `input_ = nullptr`).

### CudfAssignUniqueId

- **Storage:** `input_` — single `RowVectorPtr` (inherited from Operator base)
- **Queuing behavior:** Holds exactly one input, appends a unique ID column in `getOutput()`.
- **Increment:** `addInput()` sets `queuedInputBytes_` to `input_->estimateFlatSize()`.
- **Decrement:** `getOutput()` sets `queuedInputBytes_ = 0` on `input_ = nullptr`.

### CudfLocalPartition

- **Storage:** None — processes input immediately in `addInput()`.
- **Queuing behavior:** Does not queue. Input is partitioned and enqueued to `LocalExchangeQueue` immediately.
- **Tracking:** No `queuedInputBytes_` needed.

## Operator Categories

### Multi-input accumulators (queue all inputs, process at once)

- CudfHashAggregation (final/global path)
- CudfOrderBy
- CudfTopN
- CudfHashJoinBuild
- CudfHashJoinProbe (RightSemiFilterJoin path only)
- CudfFromVelox
- CudfToVelox

### Single-input holders (hold one input, process per getOutput call)

- CudfFilterProject
- CudfLimit
- CudfAssignUniqueId
- CudfHashJoinProbe (most join types)

### Immediate processors (no queuing)

- CudfLocalPartition
- CudfHashAggregation (partial non-global path)
