# Velox-cuDF GPU Memory Profiling with Quent: MVP Validation

## Executive conclusion

The direct Velox-to-Quent MVP is implemented and usable for offline diagnosis
of one selected Velox Task on one GPU worker.

The chosen boundary is:

```text
Velox-cuDF C++ allocation ledger
        |
        v
bounded, task-scoped capture v1 JSON
        |
        v
Quent Rust validator and replay companion
        |
        v
native Quent model, analyzer, and GPU Memory UI
```

Perfetto is not in the production data path. It remains an optional independent
sink. Presto does not need source changes for this MVP, and the Presto worker
does not link Quent or Rust.

The UI now provides the capabilities needed for peak diagnosis:

- Exact capture-local, process-wide logical-live-memory peak.
- A timeline tightly bounded to selected Velox Task execution.
- Typed PlanNode graph and cross-highlighting.
- Per-PlanNode logical-memory series.
- Operator call spans with Linux thread IDs.
- Analysis-time, user-editable thresholds with exact violation windows.
- Concurrent holders at the peak and threshold crossings.
- Custom-marker, OOM, data-loss, and capture-integrity facts.

The metric is instrumented RMM **logical requested live bytes**. It is not
physical GPU residency, managed-memory location, RMM reservation, or total
device memory use.

## Revisions validated

| Component | Revision |
|---|---|
| Velox | `codex/velox-quent-gpu-memory-mvp` at `f81d74112930` |
| Quent | `codex/velox-quent-gpu-memory-mvp` at `6046be3d` |
| velox-testing | `codex/velox-quent-gpu-memory-mvp` at `d5a9ec94d034` |
| Presto | unchanged `master` at `e359411138d4` |
| Worker image | `sha256:bc764b60356882febd026c6bbd5d81c742fc59b22f5fd6517032662440cae410` |

The branches are local to `knataraj-ws` and were not pushed.

The Velox branch contains the earlier optional Perfetto/Nsight groundwork plus
the direct Quent capture commits from `f16cccbfd` through `f81d74112`. The
Quent integration is `09b4399b` through `6046be3d`.

## Production boundary

### Velox responsibilities

Velox owns facts that are only reliable at the allocation source:

- Pointer-to-requested-size and allocation-time-owner ledger.
- Process-wide global, query, Task, PlanNode, and operator-instance values.
- Cross-thread free attribution to the original owner.
- Typed PlanNode metadata copied from the selected `PlanFragment`.
- Task-bounded call spans for `addInput`, `getOutput`, `noMoreInput`, and
  `close`.
- Constant-space peak metadata and explicit loss/overflow status.
- Bounded memory, call, marker, OOM, and data-loss storage.
- Atomic final JSON publication after Task termination.

The allocation and deallocation paths do not perform JSON serialization, file
I/O, Quent calls, Rust FFI, CUDA synchronization, or threshold evaluation.
Telemetry failures are diagnostic-only.

### Quent responsibilities

Quent owns:

- Strict capture-v1 parsing and integrity validation.
- Deterministic replay into a native Velox Quent event model.
- Exact peak, threshold-window, and concurrent-holder analysis.
- Maximum-preserving timeline binning.
- Full-query and visible-window queries.
- Typed PlanNode graph, selected PlanNode series, call spans, OOMs, markers,
  integrity warnings, and analysis-time threshold controls.

Rust is required for this companion and Quent integration. It is not required
to build or run Velox or the Presto worker.

### Presto and velox-testing responsibilities

Presto is intentionally not part of the attribution source. A later multi-node
version can add stage and fragment metadata from Presto without moving memory
accounting out of Velox.

velox-testing supplies the worker configuration and single-query benchmark
mode:

```properties
cudf.memory_resource=async
cudf.memory_tracking_enabled=true
cudf.quent_memory_profile_path=/opt/presto-server/logs/velox-cudf-quent-%q-%t.json
cudf.quent_max_events=250000
```

Perfetto output is no longer enabled by default for a Quent capture.

## Final single-query matrix

TPC-H Q1 ran once per fresh worker against `/gds/datasets`, using
`--scale-factor`, `--skip-analyze-check`, and `--skip-drop-cache`. Each scale
produced exactly one coordinator query ID, one metrics file, and one Quent
capture.

| Scale | Presto Q1 | Task capture | Peak | Updates / calls | Owners / PlanNodes | JSON |
|---|---:|---:|---:|---:|---:|---:|
| SF1 | 9,984 ms | 387.2 ms | 0.747 GiB | 866 / 138 | 19 / 10 | 314,734 B |
| SF10 | 10,285 ms | 605.2 ms | 7.468 GiB | 866 / 138 | 19 / 10 | 318,003 B |
| SF100 | 12,094 ms | 2,447.8 ms | 33.029 GiB | 3,058 / 202 | 20 / 10 | 1,100,635 B |

All three captures have:

- `Finished`, complete capture, and complete cleanup status.
- Exact memory timeline, exact headline peak, and complete call coverage.
- Zero source loss, internal loss, dropped events, overflow, and open calls.
- Contiguous source sequences and a reconstructed final value of zero.
- Positive Linux thread IDs for all calls.
- Typed nodes including `TableScanNode`, `FilterNode`, `ProjectNode`,
  `AggregationNode`, `LocalPartitionNode`, `LocalMergeNode`, `OrderByNode`,
  and `PartitionedOutputNode`.
- Every owner PlanNode ID resolved in the graph.
- No `-to-velox` or `-from-velox` synthetic owner IDs.

The capture begins and ends around the useful Task interval:

| Scale | Empty leading capture | Empty trailing capture |
|---|---:|---:|
| SF1 | 0.52% | 2.92% |
| SF10 | 0.32% | 2.06% |
| SF100 | 0.13% | 0.39% |

The Quent view therefore covers 2.45 seconds for SF100 rather than showing the
roughly 12-second Presto query envelope with mostly empty scheduling time.

The benchmark harness did not receive an independent expected-results
directory, so its recorded status is `not-validated`. As a regression check,
the final SF100 Q1 result matches an earlier GPU baseline with the repository's
`rtol=1e-5`, `atol=1e-8` semantics. The maximum relative floating-point
difference is `1.81e-15`. This is a regression sanity check, not independent
TPC-H correctness proof.

## SF100 diagnosis produced by the MVP

At the exact 33.029 GiB global peak:

| PlanNode | Logical live bytes | Share |
|---|---:|---:|
| `AggregationNode | 537` | 20.570 GiB | 62.28% |
| `TableScanNode | 0` | 9.520 GiB | 28.82% |
| `ProjectNode | 2` | 2.939 GiB | 8.90% |

The two largest individual holders are `CudfBatchConcat` instances for drivers
1 and 0 of `AggregationNode | 537`, each retaining about 9.183 GiB.

Only two operator methods are executing at the exact global peak:

- Driver 0 `CudfBatchConcat::getOutput` for `AggregationNode | 537`.
- Driver 1 `CudfGroupbyPARTIAL::addInput` for `AggregationNode | 537`.

This demonstrates the useful distinction between the currently executing
operators and all concurrent memory holders.

With the UI threshold set to a strict `>30 GiB`, exact source transitions
produce two query-wide violation windows:

| Window | Capture-relative interval | Window peak | Trigger |
|---|---:|---:|---|
| 1 | 1,159.991–1,160.007 ms | 30.068 GiB | `TableScanNode | 0` |
| 2 | 1,194.418–1,207.730 ms | 33.029 GiB | `AggregationNode | 537`, `CudfBatchConcat` |

The threshold is not compiled into Velox and does not require recapture.

## Quent replay and visual validation

The final SF100 capture passed the actual companion commands:

```bash
quent-velox analyze <capture.json> --threshold-bytes 32212254720
quent-velox replay <capture.json> --output-dir <quent-events>
```

Replay produced one native Quent engine, query group, query, worker, typed plan,
operator set, memory stream, call stream, and diagnostic streams.

Headless Chromium validation then:

- Loaded the GPU Memory route.
- Changed the threshold to 30 GiB and found two violation windows.
- Selected `AggregationNode | 537`.
- Displayed its 24.2 GiB selected-window peak and overlaid series.
- Displayed exact global peak and concurrent-holder context.
- Found no synthetic PlanNode IDs.
- Observed no failed API responses or browser console errors.

The first browser iteration found that the always-mounted generic Query Plan
panel used a 501 response as a feature probe for data-flow timelines. Quent
commit `6046be3d` replaces that probe with an explicit analyzer capability and
adds regression tests. The repeated browser validation is clean.

Validated screenshots:

- `/gds/5214187/quent-velox-mvp-20260728-validated/final-v8-sf100/quent-gpu-memory-sf100.png`
- `/gds/5214187/quent-velox-mvp-20260728-validated/final-v8-sf100/quent-gpu-memory-selected-threshold.png`

The selected-and-threshold screenshot SHA-256 is
`5d52b95d5912917763e13707b374cd4053c19848f57dfa4be3c42631fc1f2410`.

## Managed-memory smoke

An additional SF1 run used `cudf.memory_resource=managed`:

- Q1 completed in 10,177 ms.
- Exact 0.747 GiB peak.
- 866 memory transitions and 142 call spans.
- Zero loss or overflow.
- Typed PlanNodes and positive Linux thread IDs.

The async setting was restored after the smoke. The full SF1/SF10/SF100 matrix
remains async-only.

## Same-image directional overhead

The comparison kept memory tracking enabled in both cases and changed only
`cudf.quent_memory_profile_path`.

| Mode | SF100 Q1 samples | Median |
|---|---|---:|
| Capture enabled | 12,094, 11,261, 11,232 ms | 11,261 ms |
| Capture disabled | 11,031, 11,359, 11,008 ms | 11,031 ms |

The measured median delta is **+2.09%**, below the 5% MVP target. This is a
directional three-sample check, not a full performance study.

## Nsight Systems verification

A fresh final-image SF1 run was captured with Nsight Systems and exported to
SQLite:

- JSON peak: `802,265,772` bytes.
- Nsight logical-counter peak: `802,265,772` bytes.
- JSON source peak sequence: 268.
- Nsight peak sample index: 268.
- 866 memory updates and 867 Nsight samples, including the initial value.
- All four capture Linux TIDs correlate with the TID component in
  `NVTX_EVENTS`, `GENERIC_EVENTS`, and `ThreadNames`.

Artifact sizes were:

| Artifact | Size |
|---|---:|
| Direct JSON | 314,445 B |
| `.nsys-rep` | 2,771,825 B |
| Exported Nsight SQLite | 215,511,040 B |

This validates transport and thread correlation for the instrumented logical
counter. It does not independently validate physical GPU residency.

## Test evidence

### Velox and Presto

- Focused Velox capture/resource tests: 25 passed.
- cuDF configuration tests: 4 passed.
- Final release Presto GPU worker build completed with warnings treated as
  errors.
- The final build explicitly recompiled `GpuMemoryCapture.cpp`,
  `GpuResources.cpp`, and dependent cuDF code.
- Final async SF1/SF10/SF100 and managed SF1 queries completed.

The first labeled post-fix image was rejected by the SF1 gate because BuildKit
had retained objects compiled from a source snapshot taken before two late
fixes. The capture exposed the old negative hashed TIDs and synthetic PlanNode
ID. The final image was rebuilt after forcing those translation units and
passes the gate. The rejected run is not included in any result above.

### Quent

- Capture parser/analysis: 15 tests passed.
- Velox analyzer: 5 tests passed.
- Replay/CLI: 5 tests passed.
- UI after the final capability fix: 621 tests passed.
- TypeScript typecheck and production UI build passed.
- Rust formatting and clippy with warnings denied passed.
- Fresh real capture analyze, replay, API, and Playwright validation passed.

## Current limitations

- The profile is written at Task completion; this MVP is not live streaming.
- One worker process records at most one selected Task. It does not yet merge
  multiple tasks, stages, workers, or nodes.
- Detailed PlanNode and call metadata belongs to the selected Task. The global
  ledger still retains concurrent holders from other work.
- The metric is logical requested bytes, not physical residency or allocator
  reservation.
- Call spans cover the main operator methods, not initialization or an
  explicit whole-operator-lifetime interval.
- The UI deliberately displays at most 32 call rows in one visible window.
- Custom marker and OOM paths are implemented and unit-tested, but the final Q1
  runs did not intentionally trigger an OOM or add a custom marker.
- Pool-resource behavior and temporary/output classifications remain bonuses
  and were not added.
- Multi-worker clock alignment and Presto stage topology are not implemented.

## Recommendation

Use this MVP now for single-worker, selected-Task, offline memory diagnosis.
The architecture is reusable: Velox owns a stable source contract, Quent owns
the model and UX, and Perfetto/Nsight remain independent tools.

The next smallest production steps are:

1. Publish or pin the Quent revision and replace the out-of-tree local viewer
   harness with the supported `quent-open` packaging path.
2. Add one velox-testing command that captures, replays, and opens a profile.
3. Add a user-visible Task selector before supporting multiple detailed Tasks
   per process.
4. Add Presto stage/fragment metadata and multi-worker merge only after the
   single-worker workflow is routinely useful.
5. Add an optional physical-residency overlay as a separate metric, without
   weakening allocation-time logical ownership.

## Artifact index

All validation artifacts are under:

`/gds/5214187/quent-velox-mvp-20260728-validated`

Important files:

- `final-v8-validation.json`: full structural/accounting validation.
- `final-v8-sf100/quent-analysis-30gib.json`: strict Quent analysis.
- `final-v8-sf100/quent-events/`: native Quent replay.
- `final-v8-sf100/quent-visual-validation.json`: browser validation.
- `final-v8-sf100/quent-visual-interaction-validation.json`: threshold and
  PlanNode-selection validation.
- `overhead/summary.json`: enabled/disabled samples.
- `managed-sf1-smoke/capture-validation.json`: managed smoke evidence.
- `final-v8-nsys-sf1/nsys-validation.json`: current-image Nsight correlation.
