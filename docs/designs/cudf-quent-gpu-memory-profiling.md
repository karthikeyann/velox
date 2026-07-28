# Velox-cuDF GPU Memory Profiling with Quent

## Status

This document proposes the first direct Velox-cuDF integration with Quent. The
MVP records one selected Velox Task on one worker process and converts the
completed capture into a native Quent profile. It does not use Perfetto as an
intermediate format.

The source metric is **instrumented RMM logical live bytes**: the sum of
requested sizes for successful allocations that have not yet been observed by
the wrapped resource's deallocation path. It is not physical GPU residency,
managed-memory location, RMM pool reservation, CUDA-driver allocation, or total
device usage.

## Goals

- Show exact, capture-local logical GPU-memory usage over time.
- Attribute allocation transitions to Velox query, Task, PlanNode, pipeline,
  driver, and operator identities.
- Show the selected Task's typed PlanNode graph and individual cuDF operator
  calls.
- Preserve process-wide memory accounting so concurrent holders are visible at
  a peak or threshold crossing.
- Compute an exact peak and threshold crossings after capture without
  capture-time thresholds.
- Keep the allocation path independent of Quent, Rust, serialization, file I/O,
  and CUDA synchronization.
- Make all telemetry failures diagnostic-only.

## Non-goals

- Physical VRAM or managed-memory residency measurement.
- Distributed Presto stage topology or cross-worker clock synchronization.
- Live streaming to a browser.
- Allocation stack traces.
- Temporary, output, or RMM-pool classifications.
- A general-purpose Quent C++ SDK.
- Complete call tracing for CPU operators.

## Architecture

```text
cuDF operator scopes              wrapped RMM resources
          |                                |
          v                                v
 operator-call facts       exact pointer and ownership ledger
          |                                |
          +---------------+----------------+
                          |
                          v
                 bounded raw recorder
                          |
                 selected Task completes
                          |
                          v
              optional Quent adapter/replay
                          |
                          v
           Velox-specific Quent model and UI
```

Velox owns the source facts, bounded recorder, and capture-integrity status.
The Quent integration owns model-specific event construction, analysis, and UI
behavior. Perfetto and NVTX remain optional independent validation sinks.

The Quent adapter must not appear in public Velox execution headers. A build
that does not enable the adapter has no Quent link dependency and requires no
Rust toolchain. An enabled production build consumes a pinned, prebuilt adapter
library. Building that adapter may require Rust because current Quent model
generation and analysis are implemented in Rust; running the default Velox
worker does not.

## Attribution model

Each successful allocation stores the allocation-time owner with its pointer:

```text
pointer -> requested bytes, owner ID, PlanNode ID
```

The owner identifies:

- Query ID.
- Universally unique Task ID and user-visible Task ID.
- PlanNode ID and display type.
- Pipeline ID.
- Driver ID.
- Operator ID and implementation type.

A deallocation always debits the stored owner, even when it occurs on another
thread or outside the operator call that created it. Unscoped allocations use
an explicit unattributed owner.

PlanNode identity is Task-local. Display type is metadata and does not alter
identity. The selected Task's plan graph is copied from its `PlanFragment`.
Synthetic conversion operators map to their source PlanNode while retaining
their physical operator identity.

## Selected Task and process-wide accounting

The MVP presents one selected Task in full detail, but the underlying ledger is
process-wide. During the selected interval the recorder retains every observed
allocation transition, including transitions owned by other Tasks.

One worker process selects at most one matching Task. Repeated benchmark cases
restart the worker, which keeps task selection deterministic and prevents a
later stage or query from overwriting the first artifact.

This distinction is required for OOM analysis:

- The selected Task receives a complete typed plan and operator-call view.
- Other Tasks are represented as concurrent holder groups unless their metadata
  has already been registered.
- The global series includes selected, concurrent, persistent, and
  unattributed holders.
- The sum of the latest owner values must equal the global value at each memory
  transition.

Silently filtering other Tasks would make the displayed overall peak false.

## Capture v1 contract

Capture v1 is a versioned JSON document written atomically by Velox. The
top-level `format` is `velox-cudf-gpu-memory-capture` and `version` is `1`.
Schema changes require a new version; the Quent adapter rejects unknown
formats and versions.

### Capture header

The header contains:

- Contract version.
- Process ID and worker identity when available.
- Source clock name and unit.
- Selected Query ID, Task UUID, and Task ID.
- Capture start timestamp.
- Beginning memory-sequence watermark.
- Configured recorder capacity.
- Metric name and its logical-memory disclaimer.

### Metadata tables

Metadata is interned outside the allocation path and referenced by numeric IDs:

- Tasks and queries.
- Typed PlanNodes and directed plan edges.
- Physical operator owners.
- Operator method names carried by bounded call-span records.
- Custom marker names carried by marker records.

An owner registration is idempotent. Metadata must be available for every owner
referenced by a retained event or snapshot.

### Beginning snapshot

The beginning snapshot is a consistent ledger snapshot containing:

- Global current logical bytes.
- Current bytes for every registered owner.
- Owner metadata required by non-zero values.
- Live allocation count and source data-loss count.
- The beginning source sequence.

The beginning snapshot is the value at the left edge of the chart. It also
preserves memory that was allocated before the selected Task began.

### Raw events

Every event has a source monotonic timestamp and a capture-local ordinal. Memory
events also carry the source ledger sequence.

Capture v1 supports:

1. **Memory transition**
   - Source sequence.
   - Owner and PlanNode IDs.
   - Signed requested-byte delta.
   - Global, query, Task, PlanNode, and owner current bytes after the delta.
2. **Operator call span**
   - Stable call ID.
   - Owner ID.
   - Bounded method name.
   - Source thread ID.
   - Begin and end source timestamps.
   - A truncation flag when Task termination clipped an in-flight call.
3. **Allocation failure**
   - Source timestamp and ledger sequence of the reported logical state.
   - Owner ID and requested bytes.
   - Current logical state.
   - Best-effort CUDA free/total bytes and CUDA status.
4. **Custom marker**
   - Bounded marker name and active owner ID.
5. **Data-loss fact**
   - Source sequence and bounded reason text.

Strings, Quent objects, JSON, stacks, and CUDA queries are prohibited in the
successful allocation and deallocation path. The allocation-failure path may
collect best-effort CUDA state because it is already exceptional.

### Ending snapshot

The ending snapshot contains:

- Selected Task terminal state and error status.
- Capture end timestamp.
- Ending source-sequence watermark.
- Global and per-owner current bytes.
- Source and recorder data-loss counts.
- Recorder overflow status and number of rejected events.

Capture completion does not require global memory to reach zero. Concurrent
Tasks and persistent allocations can legitimately remain live.

## Task watermarks

The process ledger can exist before and after one Task capture. Source sequence
watermarks define the exact memory interval:

```text
begin snapshot sequence = B
retained memory transitions = B < sequence <= E
end snapshot sequence = E
```

The beginning operation must make the snapshot and activation atomic with
respect to ledger sequencing, or it must discard a delayed publication whose
sequence is at or before `B`. This prevents a transition represented in the
snapshot from being counted twice.

The ending operation takes a consistent snapshot and establishes `E`. It must
wait until all already-issued transitions through `E` are published before
handing the immutable capture to the adapter. Events after `E` belong to later
process activity and are excluded.

The MVP uses a non-allocating append into Velox-owned, preallocated recorder
storage while holding the ledger mutex. It never calls file I/O, Quent, Rust,
Perfetto, NVTX, CUDA, or another external sink from that critical section.

## Task lifecycle

The cuDF driver adapter is the first useful no-Presto-change start seam. It has
the Task, PlanFragment, compiled operators, and knowledge that cuDF replacement
occurred. Capture starts after the first matching driver is adapted and before
operator initialization. Constructor-time allocations are represented by the
beginning snapshot.

A Velox `TaskListener` supplies the terminal state and PlanFragment without a
Presto source change. Velox can mark Task end and seal the ending watermark in
that callback, then enqueue conversion outside the callback.

Task terminal notification can precede all cancellation-path operator cleanup.
Therefore Task end means the selected execution interval ended; it does not
mean all process allocations were freed. The MVP reports the number of calls
still open at Task end and marks call coverage incomplete; completed calls
retain their exact spans. Normal Task completion is expected to contain
balanced call spans.

The Task listener must not perform Quent replay, blocking file writes, or Rust
FFI. It only seals and transfers ownership of an immutable capture.

## Bounded recorder

The recorder splits the configured high-volume event budget between memory
transitions and operator-call spans, then reserves both vectors before capture.
The combined reservation does not exceed the configured budget. Active calls
use a small fixed set of preallocated slots. Markers, allocation failures, and
data-loss facts use small, fixed out-of-band budgets so an exhausted ordinary
timeline cannot hide the reason a query failed. No storage grows on an
allocation callback.

When capacity is exhausted:

- Existing events are not overwritten.
- The source ledger continues exact accounting.
- The recorder increments an out-of-band rejected-event count.
- The affected timeline coverage is marked inexact.
- The ending snapshot is still retained.
- The producer continues updating a constant-space global peak tuple before
  rejecting an event.
- The Quent UI must display an incomplete-timeline warning. It may label the
  global peak exact only when the independent peak-integrity flag is true, but
  cannot claim exact holder context or threshold crossings across missing
  transitions.

Disabled recording is a fast no-op. Finishing a capture is idempotent. Recorder
or adapter exceptions are contained and cannot alter an allocation, free, OOM,
operator call, or Task state.

The first implementation may use one process recorder because memory
transitions are already serialized by the ledger. If operator-call contention
is material, per-thread fixed chunks can be introduced without changing the v1
contract.

## Capture-local peak and thresholds

Lifetime ledger peaks cannot be displayed as selected-capture peaks. An earlier
query can establish a larger lifetime value whose timestamp is outside this
capture.

For a complete timeline, the adapter reconstructs the capture-local global
series as a step function:

1. Start with the beginning snapshot's global current bytes.
2. Order memory transitions by source sequence.
3. Use each transition's global current bytes as the next value.
4. Verify the producer's constant-space peak tuple against the reconstructed
   maximum.
5. Associate the peak with capture start or the exact triggering transition.

The producer observes every source transition even after ordinary storage
overflows. It therefore retains the global peak bytes, timestamp, and source
sequence in constant space. If the triggering transition itself was dropped,
its capture-local event ordinal is zero. The peak can remain exact while the
timeline and concurrent-holder context are explicitly incomplete.

PlanNode and owner peaks follow the same rule using their beginning values and
subsequent current values. The lifetime peak may be retained as diagnostic
metadata but cannot be used as the profile headline.

Thresholds are analysis-time values. The UI applies a user-entered threshold to
exact source transitions, not down-sampled render bins. For each crossing it
shows:

- Crossing timestamp and global value.
- Triggering allocation owner.
- Ranked concurrent holders at that timestamp.
- Selected PlanNode and operator-call context.

Rendered bins use maximum values so a short spike remains visible when zoomed
out.

## Quent model and UI

The initial model composes Quent query-engine concepts where they fit and adds
Velox-specific facts:

- Worker, query, Task, Plan, PlanNode, and physical operator identities.
- Operator-call intervals.
- Logical-memory samples.
- OOM, marker, truncation, and data-loss events.

The initial UI provides:

- Typed PlanNode graph labels such as `HashJoinNode | 17`.
- Overall logical-live-memory timeline and exact capture-local peak.
- Selected PlanNode or operator series.
- Ranked holders at the peak and threshold crossings.
- Operator calls filtered to the visible time window.
- Plan-to-timeline cross-highlighting.
- Visible incomplete-capture and retained-memory status.

The Quent adapter sorts source facts by their source timestamp and ordinal. It
must not substitute replay time for execution time.

## Optional adapter and build boundary

The raw recorder is useful and testable without Quent. Quent support is enabled
with a separate build option and consumes an explicitly supplied adapter
prefix. The default is disabled.

The boundary must provide:

- A narrow C or generated CXX entry point for consuming an immutable capture.
- No Quent-generated types in Velox public headers.
- A pinned Quent revision and lockfile for reproducible adapter builds.
- Failure reporting that leaves the raw capture available for diagnosis.

The worker can either invoke the prebuilt adapter asynchronously after capture
or write the versioned raw artifact for a companion converter. Both preserve
the same v1 source contract; neither routes through Perfetto.

## Integrity and failure semantics

A capture is exact only when all of the following hold:

1. No source-ledger data loss occurred during its interval.
2. The recorder did not overflow or reject an event.
3. Memory source sequences are unique and contiguous from `B + 1` through `E`.
4. Memory timestamps are monotonically increasing in source order.
5. No reconstructed current value is negative.
6. Global current equals the sum of owner current values at every transition.
7. Each PlanNode current equals the sum of its physical owners.
8. The last reconstructed values equal the ending snapshot.
9. The reported capture-local peak equals the maximum reconstructed value.
10. Every referenced ID resolves to metadata.

Balanced call spans are required for a normally finished Task. Cancellation or
failure captures with open calls remain useful for memory analysis, but the UI
must label their operator-call coverage incomplete.

If these checks fail, the artifact remains inspectable, but the UI must state
which guarantees were lost. It must not silently interpolate gaps or present a
misleading exact peak.

Telemetry follows a fail-open policy:

- Allocation and deallocation always reach their upstream resource.
- A failed allocation is rethrown unchanged.
- A recorder, adapter, or output-path failure never changes query results.
- A temporary output is atomically renamed only after successful conversion.
- Partial output retains an explicit incomplete status or is removed.

## Validation

### Unit tests

- Beginning and ending watermark inclusion.
- Delayed publication at the beginning and ending boundaries.
- Bounded capacity and visible overflow.
- Disabled and repeated-finish behavior.
- Baseline memory and capture-local peak reconstruction.
- Process-wide concurrent holders during one selected Task.
- Owner metadata deduplication.
- Cross-thread free attribution.
- Balanced and truncated operator calls.
- Adapter failure isolation.

### Integration tests

- Capture one TPC-H Q1 Task at SF1, SF10, and SF100.
- Require one selected Task and a typed PlanNode graph.
- Compare source and reconstructed current values at every transition.
- Compare Quent's global peak and timestamp with the source ledger and a
  Perfetto validation capture.
- Verify a threshold can change without recapture.
- Verify disabled-path correctness and enabled-path wall time and artifact size.
- Use Nsight Systems only as an occasional physical-memory shape and timing
  cross-check.

The MVP target is zero data loss and no more than five percent median wall-time
overhead for representative queries. Results are reported even when the target
is missed.

## Extension path

The v1 identities and source timestamps allow later expansion without changing
the allocation ledger:

1. Multiple complete Tasks on one worker.
2. Live bounded streaming.
3. Presto-provided stage and fragment topology.
4. Cross-worker artifact merging and clock alignment.
5. Optional physical-residency or allocation-classification sources.

Each extension must preserve allocation-time ownership, process-wide accounting,
capture-local peak semantics, and explicit integrity status.
