# Velox-cuDF GPU-memory PerfettoSQL

These queries apply analysis policy to factual counters in a captured Perfetto
trace. No threshold is compiled into Velox-cuDF.

## Capture contract

The analysis is exact if capture obeys these invariants:

1. Emit exactly one `Overall RMM logical live bytes...` counter per trace.
2. Emit operator and PlanNode counter names with the prefixes used by
   `00_setup.sql`.
3. Serialize allocation/free ledger transitions and assign each transition a
   strictly increasing timestamp, e.g. `max(traceClockNow, lastTimestamp + 1)`.
4. Emit the global, PlanNode, and allocation-origin operator samples for one
   transition with that same timestamp.
5. Attribute allocations without an active operator to explicit
   `unattributed` operator and PlanNode tracks so both attribution views sum to
   the global value.
6. Count only successful allocations. An allocation failure is a separate
   factual OOM instant event and does not change the counters.

Trace startup emits no synthetic owner event. Concrete owners are registered
when first used, and the `unattributed` owner is registered lazily only for an
actual fallback allocation, allocation failure, or fallback call. PlanNode
tracks use the resolved concrete type in their display label, for example
`TableScanNode | 0`; `planNodeId` remains the stable attribution identity.

## Capture one selected benchmark query

Use an explicit scale factor and skip the statistics check in addition to
selecting one query and one iteration:

```bash
./scripts/run_benchmark.sh \
  -b tpch -s sf100_v2_float -q 1 -i 1 -m \
  --scale-factor 100 --skip-analyze-check --skip-drop-cache
```

Without `--scale-factor`, `velox-testing` discovers schema and scale metadata
with preliminary SQL such as `SHOW TABLES` and `SHOW CREATE TABLE`. Those
queries precede the selected benchmark query and appear in the process-wide
trace. `--skip-analyze-check` suppresses the separate statistics query.

## Perfetto UI

1. Open the trace and run `00_setup.sql`.
2. Change only `threshold_gib` in that file and rerun it whenever a different
   analysis-time threshold is wanted. Rerun the selected analysis and replace
   its existing debug track; debug-track results are materialized when added.
3. Run one of the numbered analyses.
4. For files 01-03, 05, and 06, choose
   **Show timeline → Show debug track → Slice** and map `ts`, `dur`, and
   `name`. Select `pivot` as the pivot column.
5. Use 04 primarily as a ranked table. It is also debug-slice compatible; add
   `WHERE owner_kind = 'operator'` or `plan` when only one attribution view is
   wanted. File 06 gives the equivalent complete holder snapshot at every
   analysis-time threshold crossing.
6. Rerun `90_validate_capture.sql`; every row should be `PASS`. For a clean
   isolated capture, its terminal checks also require the final global,
   PlanNode, and operator totals to reconcile at zero. A non-zero terminal
   value usually means that the worker did not finish trace shutdown and the
   file is incomplete.

Threshold comparison is strictly greater-than. Change `>` to `>=` in files
02, 03, and 06 if equality should count.

The queries use the current official `counter`, `counter_track`,
`counters.intervals`, and debug-track conventions:

- https://perfetto.dev/docs/analysis/stdlib-docs#counters-intervals
- https://perfetto.dev/docs/analysis/debug-tracks
- https://perfetto.dev/docs/visualization/commands-automation-reference

## CLI validation

Run `trace_processor_shell` interactively so setup and analysis share the same
in-memory trace database:

```text
trace_processor_shell trace.pftrace
> .read 00_setup.sql
> .read 90_validate_capture.sql
> .read 01_global_peak.sql
> .read 02_global_threshold_violations.sql
> .read 03_owner_threshold_violations.sql
> .read 04_holders_at_global_peak.sql
> .read 05_query_peaks.sql
> .read 06_holders_at_threshold_crossings.sql
```

Validated with Perfetto trace processor
`v57.2-da1d152cf (da1d152cff27890903d158fe96751de3aab883cc)` using a synthetic
trace containing two owners, separated global violations, owner threshold
crossings, frees, and a later global maximum.
