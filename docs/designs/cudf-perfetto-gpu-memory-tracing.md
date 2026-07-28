# Velox-cuDF GPU Memory Tracing with Perfetto

## Decision

Build a small, opt-in Velox-cuDF telemetry layer that writes a process-wide
Perfetto trace. Intercept RMM allocations and frees once, attribute each
allocation to the active Velox operator instance, and emit:

- Overall live and peak logical GPU bytes at the top of the trace.
- Live logical bytes per PlanNode and operator instance.
- Host-side operator call slices.
- User markers and factual allocation-failure/OOM markers.

Threshold analysis belongs in packaged PerfettoSQL, not in the execution path.
An analyst can change one value after capture and identify the crossing,
allocation that triggered it, and operators that were holding memory then.

This is the fastest useful MVP. Quent integration, special temporary/output
classifications, and allocator-pool analysis can be added later without
changing the trace contract.

## Scope

The MVP answers these questions for concurrent drivers in one worker process:

1. When did overall logical GPU memory reach its peak?
2. Which operator allocation triggered a chosen threshold crossing?
3. Which operator and PlanNode instances held memory at that instant?
4. Which operator calls overlapped the growth and peak?
5. Did a real allocation failure occur, and what was live at that time?

The trace is enabled by a file-path property such as:

```properties
cudf.perfetto_memory_trace_path=/opt/presto-server/logs/velox-cudf-memory-%p.pftrace
```

`%p` expands to the worker PID. No path means tracing is disabled. A single
process-wide session is intentional: tasks and drivers overlap, and separate
query traces would hide the contention that causes process-level OOMs.

### Non-goals

- Measuring physical VRAM residency, managed-memory migration, allocator
  reservation, fragmentation, or CUDA allocations outside the wrapped RMM
  resource.
- Separating input, output, temporary, or pool memory in the MVP.
- Replacing Nsight Systems for CUDA API, kernel, or physical-memory analysis.
- Detecting an analyst-selected threshold in production code.
- Deliberately exhausting a device in routine tests.
- Quent ingestion or a new visualization frontend.
- Aggregating multiple worker processes into one trace.

## Memory semantics

The reported value is **logical requested live bytes**:

```text
sum(requested allocation sizes that succeeded and have not been freed)
```

This definition is exact for calls observed by the wrapper and works for async
and managed RMM resources. It is not expected to equal `nvidia-smi`, pool
reservation, or physical residency. The distinction must appear in track names,
SQL output, and reports.

An allocation is recorded only after the upstream allocation succeeds. A free
removes the allocation from the ledger immediately before calling upstream
deallocation, avoiding an address-reuse race. The ledger stores
`pointer -> {requestedBytes, allocationOwnerId}`. A free is always charged to
the allocation-time owner, even when it occurs on another driver thread or
under another active operator.

All wrapped RMM resources contribute to one shared process ledger. Multiple
resource wrappers therefore cannot double-count process peaks or produce
separate peaks that an analyst might incorrectly add.

## Attribution

An RAII scope installs the active owner in thread-local state around operator
execution. The stable owner identity is:

```text
queryId
taskUuid
taskId
planNodeId
pipelineId
driverId
operatorId
operatorType
```

`taskUuid` prevents reuse of human-readable task IDs from merging unrelated
instances. Pipeline, driver, and operator IDs keep concurrent instances
separate. PlanNode totals group owners by `{queryId, taskUuid, planNodeId}`.
The resolved concrete PlanNode type is display-only metadata: Perfetto labels
the group as, for example, `TableScanNode | 0`, while `planNodeId` remains the
stable identity used for attribution and reconciliation.

The primary scopes cover `addInput`, `getOutput`, `noMoreInput`, and `close`.
Construction or initialization is scoped where the driver has an operator
context available. Allocations outside any valid scope are assigned to an
explicit `Unattributed` owner; they are never silently omitted or guessed.

Attribution describes where an allocation originated. If an allocation remains
live after its call slice ends, its bytes remain with that owner until free.
This makes the owner snapshot at a later process peak meaningful.

## Trace model

The Perfetto hierarchy is:

```text
Velox-cuDF GPU memory
├── Overall RMM logical live bytes
├── Overall RMM logical peak bytes
├── Markers and allocation failures
└── Query
    └── Task
        └── ConcretePlanNodeType | planNodeId
            ├── PlanNode logical live bytes
            └── Operator instance (type, pipeline, driver, operator)
                ├── Operator logical live bytes
                └── Calls
```

Call slices carry the call name and owner metadata. They show host execution
intervals, not allocation lifetimes. `markGpuMemoryTrace(name)` emits a
timestamped marker through a tiny API for query phases or experiments.

Each successful allocation or observed free is assigned one strictly
increasing timestamp and an internal sequence number while updating the ledger.
The global, PlanNode, and owner counters for that delta use that **same
timestamp**. Events may be written after releasing the ledger lock; the unique
timestamp preserves their causal order in the trace. This exact join key is
what lets SQL name the allocation owner that triggered a global crossing
without time-window heuristics.

Owner metadata is emitted as structured arguments and human-readable track
names. The trace streams to a file through the vendored Perfetto SDK rather
than retaining the entire query history in memory.

The lifecycle contract is deliberately simple: start tracing once before
creating the tracked resources or registering owners, and stop only after
operator execution has quiesced. Starting the worker emits no synthetic owner
event, so the trace begins with real query activity rather than worker startup.
The `Unattributed` owner is registered lazily only when an actual fallback
allocation, allocation failure, or fallback call needs it. Restarting a trace
while retaining the same live tracker is unsupported in the MVP.
`registerCudf()`/`unregisterCudf()` enforce this ordering for the Presto worker.
Counter reconciliation is the capture acceptance criterion; the final
host-call slice tail remains best-effort during process shutdown.

Perfetto buffers TrackEvent packets per producer thread. After each complete
logical counter transition and host-call slice end, the emitter appends an empty
packet on that same thread. This lets the service safely scrape the preceding
event without an IPC flush on every allocation or operator call. Shutdown also
requests a blocking session flush before stopping. The empty packets are
Perfetto's thread-pool producer workaround and are not logical trace events.

### Allocation failure and OOM

An allocation-failure marker is emitted only when the upstream allocation
actually fails. It includes requested bytes, current and peak logical bytes,
the active owner, and best-effort `cudaMemGetInfo` values collected on this
cold path. Telemetry must preserve and rethrow the original exception.

This is distinct from an analyst threshold. Code does not emit speculative
threshold-crossing markers because the desired threshold is query-dependent
and should remain editable after capture.

## Editable threshold analysis

The repository ships a PerfettoSQL setup with a one-row editable view:

```sql
CREATE VIEW vgm_config AS
SELECT
  8.0 AS threshold_gib,
  8.0 * 1024.0 * 1024.0 * 1024.0 AS threshold_bytes;
```

Changing that value and rerunning the analysis requires no trace recapture.
The packaged queries provide:

1. **Crossings and peak**: every interval where the global live counter is
   strictly above the threshold, plus the first occurrence of the overall
   peak.
2. **Trigger**: the positive owner delta with the same timestamp as the
   crossing.
3. **Holders**: the last live value for every owner at the overall peak or at
   each threshold crossing, ranked by bytes, with independently reconciled
   PlanNode and operator views.
4. **Query peaks**: exact per-query live-byte peaks reconstructed from owner
   deltas, including overlapping queries.

A companion query lists operator and PlanNode instances whose own observed
peak exceeds the same value. Results include `ts`, duration where applicable,
track ID, owner identity, bytes, percentage, and rank so Perfetto can navigate
to the interval and display the selected rows as debug annotations. The trace
retains all samples; a later threshold cannot expose data that capture-time
sampling discarded.

The most useful interpretation at a crossing is:

```text
triggering allocator + concurrently live holders = explanation of the peak
```

The trigger is not automatically blamed for all held memory. The ranked holder
view makes indirect contributors visible.

## Concurrency, overhead, and failure policy

The hot path performs an owner lookup, one short ledger critical section, and
counter emission. It avoids per-event RuntimeStats, task listeners, JSON
snapshots, and stack capture. Owner strings and Perfetto track identities are
registered once and referenced by compact IDs thereafter.

Exact process peaks require observing every successful allocation and free.
The MVP does not sample. If measured overhead is unacceptable, a future mode
may coalesce display counters, but the global ledger and peak calculation must
remain exact.

Telemetry is diagnostic and must not fail a query:

- Failure to open or start a trace disables emission and logs one clear error.
- Source-ledger inconsistencies increment a data-loss count and emit a marker
  when possible. Perfetto import statistics detect transport-level drops.
- Allocation and deallocation still reach the upstream resource.
- Shutdown synchronously flushes the trace and closes the file descriptor once.
- Disabled mode adds no map updates or Perfetto events.

Trace files can be large for allocation-heavy workloads and are capped at
16 GiB per worker. The default remains off, artifacts are written to a
filesystem with adequate space, and validation reports event count, file size,
Perfetto dropped/error statistics, source-ledger data-loss markers, and
elapsed-time overhead.

## Validation plan

### Unit and concurrency tests

Verify:

- Failed allocations never increase live or peak bytes.
- Cross-thread frees debit the allocation-time owner.
- Concurrent allocation/free, pointer reuse, zero-byte calls, and alignment
  forwarding preserve ledger invariants.
- Pipeline and driver IDs keep otherwise identical operators separate.
- Unscoped allocations appear under `Unattributed`.
- Global, PlanNode, and owner samples for one delta share a unique timestamp.
- Global peak remains correct when multiple owners overlap.
- A real injected allocation failure emits an OOM marker and rethrows the
  original error.
- Disabled tracing and telemetry failures do not change allocator behavior.

### Presto integration

Build Velox and Presto with the same binaries used by `velox-testing`, then run:

```bash
./scripts/run_benchmark.sh \
  -b tpch -s sf100_v2_float -q 1 -i 1 -m \
  --scale-factor 100 --skip-analyze-check --skip-drop-cache
```

For a trace containing only one selected benchmark query, all three controls
are required: select one query with `-q`, run one iteration with `-i 1`, and
supply both `--scale-factor` and `--skip-analyze-check`. Without an explicit
scale factor, the harness discovers schema and scale metadata with preliminary
SQL such as `SHOW TABLES` and `SHOW CREATE TABLE`; those metadata queries also
appear in the process-wide trace. `--skip-analyze-check` prevents the separate
statistics check.

- SF1: TPC-H Q1-Q22 once, checking available reference results.
- SF10: Q1, Q8, Q9, Q18, and Q21.
- SF100: Q1, Q8, Q9, Q18, and Q21.

For every captured trace, run the packaged SQL and check:

- Global live equals the reconstructed sum of owner live bytes at each event.
- PlanNode totals equal the sum of their operator instances.
- No counter is negative and allocation/free timestamps are unique.
- The stored peak equals the maximum global live sample.
- Threshold trigger and holder rows resolve to complete owner identities.
- Actual OOM markers, when generated by a deterministic failing resource test,
  are factual and do not depend on the SQL threshold.

Record query correctness, wall time with tracing disabled and enabled, trace
size, event count, and data-loss count. A practical MVP target is no data loss
and no more than 5% median wall-time overhead on the representative query set;
results are reported even if the target is missed.

### Nsight cross-check

For the highest logical-memory query at each scale, restart the worker to clear
retained async-pool state and capture one Nsight Systems run with CUDA memory
usage enabled. Export the reduced SQLite tables and compare Perfetto logical
live bytes with Nsight `localMemoryPoolUtilizedSize` around the same peak.

The comparison validates timing, growth/decline shape, and attribution
plausibility. Numeric equality is not an acceptance criterion because the
tools measure different layers. Nsight remains the occasional physical-memory
cross-check; the Perfetto trace is the lightweight day-to-day debugging tool.

## Extension points

After the MVP proves useful, optional additions are:

- Quent ingestion of the trace or derived SQL tables.
- Temporary/output allocation tags when a low-cost semantic signal exists.
- Pool reserved bytes or managed-memory residency as clearly separate tracks.
- Cross-worker trace alignment and aggregation.
- Preset dashboards for common thresholds.

These extensions must preserve the allocation-time owner, shared process
ledger, and SQL-editable threshold contract.
