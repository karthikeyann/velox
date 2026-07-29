# cuDF GPU memory profiling

Records GPU memory attribution and operator execution spans from a Velox-cuDF
worker into a bounded JSON document that an external tool replays offline.

## Why the attribution lives in the allocator

cuDF allocates through RMM, not through `memory::MemoryPool`, so the Velox pool
tree carries no GPU bytes. The only place that sees every GPU allocation is a
memory-resource adaptor over the upstream RMM resource, which is what
`GpuResources.cpp` installs when tracking is enabled.

The metric is instrumented RMM logical live bytes: the sum of successful
allocation sizes not yet freed through the wrapped resource. It is not physical
VRAM residency, allocator pool reservation, or managed-memory location. An
allocation is debited to the owner active when it was made, even when it is
freed later on another thread.

## Attribution and spans come from the driver, not from cuDF operators

Attribution was previously established by `CudfOperatorBase` wrapping its own
`addInput`, `getOutput`, `noMoreInput` and `close`. That reached only cuDF
operators, so everything else in a GPU driver's chain was invisible: a real
SF100 capture showed both `TableScan` instances holding 832 memory updates each
and no spans at all, despite being the largest memory holders in the query.

A `DriverListener` registered through `exec::registerDriverListenerFactory`
now observes every operator call the driver makes. `onOperatorCallBegin` sets
the thread-local allocation owner and opens a span; `onOperatorCallEnd` closes
both. Coverage therefore includes `TableScan`, `LocalExchange`, sinks and the
cuDF conversion operators.

Spans are recorded only for `initialize`, `addInput`, `getOutput`,
`noMoreInput` and `close`. The driver loop polls `isBlocked`, `needsInput` and
`isFinished` on every iteration; recording those would exhaust a capture's
bounded event budget and bury the calls that do work. They still establish
attribution, because an allocation made during any call must be attributable.

## Owner identity

An owner is one operator instance:
`(taskUuid, taskId, queryId, planNodeId, pipelineId, driverId, operatorId, operatorType)`.
That is the finest granularity Velox offers, because an operator instance
belongs to exactly one driver.

Two Velox irregularities are normalized. `FilterProject` fuses a `FilterNode`
and a `ProjectNode` and reports the project's id. The cuDF conversion operators
are constructed with synthetic `{planNodeId}-from-velox` and `-to-velox` ids;
the suffix is stripped so memory lands on the real source PlanNode while the
operator type still names the conversion.

## Capture bounds

One task per worker process is recorded, selected by `cudf.quent_query_filter`
matching a substring of the query id, task id or task uuid. The document is
written atomically at task completion.

`cudf.quent_max_events` bounds retained memory updates and operator calls
together. OOM and data-loss records have their own small reserve so a saturated
timeline cannot hide an allocation failure. When the budget is exhausted the
capture stays usable and reports it: `integrity.exact_memory_timeline` and
`operator_calls_complete` are independent, and `exact_timeline` requires both.

A capture-local peak is tracked in constant space, so the peak value and its
timestamp can remain exact even when the transition that produced it was
dropped. `summary.capture_local_peak_exact` distinguishes the two cases.

## Configuration

| Key | Effect |
| --- | --- |
| `cudf.quent_memory_profile_path` | Where to write the capture; also enables tracking. `%p`, `%q`, `%t` and `%u` expand to process id, query id, task id and task uuid |
| `cudf.quent_query_filter` | Substring selecting the task to record |
| `cudf.quent_max_events` | Retained memory updates plus operator calls |

## Non-goals

- Physical VRAM residency, pool reservation, and allocation stack traces.
- Host memory, which the Velox pool tree already reports.
- More than one task per worker process, and merging captures across workers.
- Live streaming; the document is written once, at task completion.
