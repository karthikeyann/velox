# Self-Describing Decimal Aggregate State for velox-cudf

Status: proposal, not started.
Written: 2026-10-02. File and line references are against branch
`perf/cudf-decimal64-global-reduction` at commit `8b8acc1ba9` (merge-base with
upstream main `af34c8f33a`). Re-verify line numbers before editing.

This document records the investigation of the branch, the reasons its
mechanism should not be merged as-is, and a contained replacement that keeps
its measured memory win.

## 1. Goal

Keep the Q18 memory reduction measured on the branch (peak allocation from
about 281 GB to about 90 GB with two drivers and managed memory) while:

- confining the change to `velox/experimental/cudf/`;
- adding no per-operator special cases, no column metadata side channel, and
  no plan-path eligibility analysis;
- adding no configuration to Presto, Prestissimo, or Velox;
- covering decimal `avg` and DECIMAL128 inputs, not only DECIMAL64 `sum`;
- being correct across local exchange, UCX exchange, HTTP exchange, and CPU
  fallback without any transport-specific code.

Non-goals: changing the planner-visible intermediate type (that is the
long-term fix, section 9), changing the CPU serialized format, and the
streaming final group-by or temporaries-release work on the branch, which are
orthogonal and stand on their own.

## 2. Background: where the VARBINARY intermediate comes from

Both sides declare it independently, and nothing enforces agreement.

- Presto Java derives an aggregate's intermediate type from its state
  serializer. `LongDecimalWithOverflowStateSerializer` and
  `LongDecimalWithOverflowAndLongStateSerializer` both return `VARBINARY`.
  `PushPartialAggregationThroughExchange` copies that type into the PARTIAL
  output symbol and the FINAL input, so it lands in the plan fragment.
- Velox registers Presto decimal `sum` and `avg` with
  `intermediateType("varbinary")` in
  `velox/functions/prestosql/aggregates/SumAggregate.cpp:47` and
  `AverageAggregate.cpp:69`. The state in
  `velox/functions/lib/aggregates/DecimalAggregate.h:32-69` is a fixed
  32-byte blob: `count int64, overflow int64, sum low uint64, sum high int64`.
- Prestissimo trusts the Java type. `PrestoToVeloxQueryPlan.cpp` converts
  the call as-is. Velox recomputes its own intermediate type from the
  registry (`velox/exec/AggregateInfo.cpp:86-87`) but uses it only for spill
  and companion-function signatures. No code compares the two.
- The byte layouts do not match Java anyway. Java SUM is 24 bytes with no
  count field; Java AVG is 32 bytes but stores the int128 in sign-magnitude.
  The Velox blob is compatible only with itself.
- velox-cudf registers the ordinary Presto functions under a name prefix and
  replaces operators at the `DriverAdapter` level (`exec/ToCudf.cpp`), after
  plan nodes and local-exchange queues already carry the Java types. It has
  no legitimate place to change a column's logical type.

Conclusion: the logical intermediate stays VARBINARY. The question is how the
GPU avoids 32-byte string columns underneath it.

## 3. What the branch does and why it sprawls

The branch stores only the DECIMAL128 sum under a logical VARBINARY column and
tags the column with `CudfPhysicalEncoding::kNativeDecimal64SumState`
(`vector/CudfVector.h:33-45`). Because count and overflow are dropped, a
valid 32-byte state can be rebuilt only if the column is known to come from a
DECIMAL64 `sum`. Carrying that provenance is what forces:

- per-column encoding metadata on `CudfVector` that must survive select,
  slice, split, concat, and partition;
- an `acceptsNativeDecimalSumState()` opt-in on `CudfOperatorBase`
  (`exec/CudfOperator.h:134-143`) with five operators opting in and fourteen
  silently materializing;
- `exec/NativeDecimalSumEligibility.cpp`, which re-derives from the plan what
  the data no longer carries;
- mixing rules in concat and group-by for batches that disagree;
- restriction to grouped DECIMAL64 `sum`; `avg`, DECIMAL128, and global
  reductions stay on the blob.

Every future operator must know about the tag or it quietly loses the
optimization. The branch also has a concrete gap: `UcxPartitionedOutput`
derives from `exec::Operator`, not `CudfOperatorBase`, and nothing under
`velox/experimental/ucx-exchange/` reads `physicalEncodings`. A tagged column
enters `cudf::pack` unmaterialized, and the receiver in `UcxExchange.cpp:196`
constructs a `CudfVector` with default encodings, which
`CudfVector::validatePhysicalEncodings` (`vector/CudfVector.cpp:178-182`)
rejects. The design doc lists remote exchange as a materialization boundary,
but no code enforces it for UCX.

## 4. Due diligence on the state fields

Verified against the CPU consumer in `DecimalAggregate.h`:

- `DecimalSumAggregate::computeFinalValue`
  (`velox/functions/lib/aggregates/SumAggregateBase.h:213-220`) reads
  `accumulator->sum` and `accumulator->overflow` only. Count is dead weight
  for SUM.
- Null-ness of a group comes from the aggregate's null flag, cleared by
  `clearNull(groups[i])` on the first non-null merge
  (`DecimalAggregate.h:176-199`), not from count.
- `DecimalAverageAggregateBase::computeFinalValue`
  (`AverageAggregateBase.h:403-409`) reads sum, count, and overflow.
- Nonzero overflow is not an error on the CPU. `addWithOverflow` keeps the
  low 127 bits of the running sum and counts carries of 2^127 in `overflow`.
  `DecimalSumAggregate::computeFinalValue` calls
  `DecimalUtil::adjustSumForOverflow`, which accepts `overflow == 1` with a
  negative sum and `overflow == -1` with a positive sum (adding
  `overflow * 2^127`, which wraps back to the exact total), raises
  "Decimal overflow" for any other nonzero value, and then range-checks
  against DECIMAL(38). `DecimalUtil::computeAverage` accepts any overflow and
  returns `(sum + overflow * 2^127) / count` with half-up rounding, computed
  without widening past 128 bits; the average is not range-checked. The GPU
  FINAL step reproduces both rules on device (`finalizeDecimalSum`,
  `finalizeDecimalAverage` in `exec/DecimalAggregationHostOps.h`) in a single
  pass with one host sync per output batch, and skips the overflow work
  entirely for shapes without an overflow child (DECIMAL64 raw input, the
  Q18 path).
- Overflow is provably zero for DECIMAL64 raw inputs: a DECIMAL64 value fits
  in 63 bits and the row count fits in 63 bits, so the sum fits in 126 bits.
- `mergeWith` (`DecimalAggregate.h:34-44`) only requires `count` to be
  summable and `overflow` to be summable. A fabricated `count = 1` for a
  non-null SUM state is semantically neutral.

## 5. Proposed representation

Inside velox-cudf, a decimal aggregate state column whose logical type is
VARBINARY may be physically either:

- the default cuDF STRING blob (what `CudfFromVelox` always produces), or
- a cuDF STRUCT whose shape is chosen by the producer from two facts it
  already knows locally: the aggregate function and its raw input type.

| Producer | Struct children (in order) | Bytes per row | Filled in when packing to the blob |
|---|---|---|---|
| SUM, DECIMAL64 input | sum DECIMAL128 | 16 | count 1, overflow 0 |
| SUM, DECIMAL128 input | overflow INT64, sum DECIMAL128 | 24 | count 1 |
| AVG, DECIMAL64 input | sum DECIMAL128, count INT64 | 24 | overflow 0 |
| AVG, DECIMAL128 input | sum DECIMAL128, count INT64, overflow INT64 | 32 | nothing |

Properties:

- Lossless. Every shape converts to a correct 32-byte blob from its own
  children. No downstream knowledge is needed.
- Self-describing. The position of the DECIMAL128 child distinguishes count
  from overflow, so the shape is recoverable from cuDF column types alone.
  The discriminator between blob and struct is the physical cuDF type under a
  logical VARBINARY, which is unambiguous because `CudfFromVelox` never
  produces a STRUCT for VARBINARY.
- Deterministic across the query. The shape is a pure function of the plan
  node's aggregate name and raw input type, so every producer of a given
  column on every worker emits the same shape. Struct-versus-struct mismatch
  can arise only from version skew between workers.
- Validity lives on the `sum` child only. The struct parent carries no null
  mask when a producer emits it, which avoids a redundant mask and avoids
  `make_structs_column` superimposing parent nulls onto children (which can
  copy child masks). A state row is null exactly when `sum` is null; cuDF
  group-by SUM yields null for an all-null group, which is the required SUM
  semantics. Consumers nevertheless tolerate a parent mask (a nullifying
  gather such as an outer-join probe adds one): `flattenDecimalState` and
  `packDecimalState` treat a row as null when the parent or the `sum` child
  is null, folding the parent mask into an owned copy of `sum` only in that
  case, at zero cost when the parent has no nulls.

Known limitation: GPU producers do not track int128 carries. cuDF's
DECIMAL128 SUM wraps modulo 2^128, so the `overflow` child emitted by a GPU
producer (kSum128 and kAvg128) is always 0 and `packDecimalState` writes that 0
into the blob. A DECIMAL(38) total that wraps back into range is therefore
reported silently wrong, and one that lands out of range is reported as
"Decimal overflow" even when the true average would be in range. This is the
pre-existing behaviour of the blob path and is unchanged by this design. The
overflow field is meaningful only when it came from a CPU-produced state; it
is honoured on merge (summed across partials) and at FINAL (folded per the CPU
rules above). Closing the gap needs a carry-tracking reduction for DECIMAL128
(a custom reduce over `(sum, overflow)` for the global path and a host UDF or
two-pass approach for `cudf::groupby`).

Q18 lands on the first row: 16 bytes plus one validity bit per group, the
same as the branch.

## 6. Where the bytes are interpreted: the only four code sites

### 6.1 `CudfGroupby` and `CudfReduce` (producer and consumer)

- PARTIAL: run the existing SUM and COUNT aggregations (and the fused
  DECIMAL64 reduce from commit `8b2c9edb13` for global paths), then wrap the
  resulting flat columns in a struct of the chosen shape by moving the
  children. No copy, no pack kernel.
- FINAL and INTERMEDIATE: flatten each incoming state column to flat child
  columns before any other work, accepting a STRING blob or any struct shape
  (child views plus synthesized zero or one columns for missing fields; see
  `flattenDecimalState`). The state buffered between incremental FINAL rounds
  is a struct of the plan shape with no parent mask, which is byte-identical
  to the flat children; the next round flattens it by view, so nothing is
  copied. Before the buffered struct is concatenated with a new batch, the
  concat funnel's `normalizeDecimalStateTableViews` brings both to one
  physical form (a CPU-produced blob is unpacked toward the struct). This
  replaces the raw `cudf::concatenate` at `exec/CudfGroupby.cpp:1951`
  operating on mixed forms.
- INTERMEDIATE output re-wraps in the plan-determined shape. FINAL output
  folds the merged overflow into the sum and runs the DECIMAL(38) range check
  (`finalizeDecimalSum`) or divides with the CPU's rounding
  (`finalizeDecimalAverage`), as described in section 4.

### 6.2 `CudfToVelox` (`exec/CudfConversion.cpp`)

After the concat in case B (`:343`) and before `with_arrow::toVeloxColumn`
(`:346`), pack any VARBINARY column that is physically a STRUCT into the
32-byte blob using the existing `serializeDecimalSumState` extended for all
four shapes. This single site covers HTTP exchange (`PartitionedOutputAdapter`
declares `acceptsGpuInput() == false`, so `ToCudf.cpp:209-212` inserts
`CudfToVelox` before it), CPU fallback, final output, and any CPU-side spill.

### 6.3 The concat funnel in `exec/Utilities.cpp`

`getConcatenatedTable`, `getConcatenatedTableBatched`, and
`getConcatenatedCudfVectorsBatched` are used by `CudfBatchConcat`,
`CudfToVelox`, `CudfGroupby`, `CudfReduce`, `CudfDistinct`, `CudfWindow`,
`CudfOrderBy`, `CudfHashJoin`, `CudfLocalMerge`, and `CudfNestedLoopJoin`.
Before `cudf::concatenate`, for each VARBINARY column whose physical form
differs across batches:

- unpack STRING batches to the struct shape present in the other batches
  (normalize toward the smaller form, not toward the blob);
- if struct shapes differ, widen all to the four-child-equivalent full form;
- treat zero-row batches as wildcards and drop or re-type them.
  `makeEmptyTable` (`exec/Utilities.cpp:208+`) builds a STRING column for
  VARBINARY, and a zero-row STRING concatenated with STRUCT batches is a cuDF
  type-mismatch error.

Raw `cudf::concatenate` calls outside the funnel: `CudfGroupby.cpp:1951`
(handled by 6.1), `CudfDistinct.cpp:83` and `CudfMarkDistinct.cpp:186`
(only reachable if a state column is a distinct key; guard with a check
rather than support), `UcxPartitionedOutput.cpp:200` (single-driver inputs,
always uniform).

### 6.4 Expression evaluator

If an expression reads a VARBINARY field whose physical column is a STRUCT,
pack it first or fail with a clear message. Presto plans never compute on an
aggregation intermediate, so this is a guard, not a hot path.

## 7. Exchange and merge boundaries, verified

| Boundary | Behavior | Change needed |
|---|---|---|
| Local exchange (`CudfLocalPartition`, `LocalExchangeAdapter`) | passes `CudfVector` by pointer; `hash_partition` handles struct payloads; state is never a key | none |
| UCX (`UcxPartitionedOutput`, `UcxExchange`) | `cudf::pack` and `contiguous_split` preserve struct types including children; receiver builds one `CudfVector` per message, no concat | none; 16 bytes per row on the wire for Q18 |
| HTTP (`PartitionedOutput`, `Exchange`, `PrestoSerializer`) | `CudfToVelox` packs before the serializer; receiver gets blobs via `CudfFromVelox`; FINAL unpacks | none beyond 6.2; pays the full blob because the plan type demands it |
| Ordered merges (`CudfLocalMerge`, `cudf::merge`) | struct payload columns supported; state is never a sort key | none |
| CPU fallback of FINAL | `CudfToVelox` packs; CPU `DecimalAggregate` merges | none beyond 6.2 |

When can representations mix? Fallback is decided per operator per driver in
`ToCudf.cpp:165-240` from the plan node and config, so all drivers of a
pipeline on a worker, and all workers with the same plan, binary, and
config, take the same path. `LocalExchangeAdapter` already relies on this.
STRING-versus-STRUCT mixing requires heterogeneous fallback across workers;
STRUCT-shape mixing requires version skew. Both are handled by 6.3. If
unhandled, `cudf::concatenate` throws a type-mismatch error rather than
producing wrong results.

## 8. Memory analysis: does any case get worse?

Compared against the baseline (STRING blob) and the branch (sum-only tag).
Baseline per-row cost is 32 bytes of chars plus 4- or 8-byte offsets (8 at
Q18 scale, where chars exceed 2^31), plus decode and encode temporaries on
every incremental final step.

| Case | Baseline | Branch | Proposed |
|---|---|---|---|
| Grouped SUM, DECIMAL64, GPU-to-GPU | 40 B + temporaries | 16 B | 16 B |
| Grouped AVG, DECIMAL64 | 40 B + temporaries | 40 B (ineligible) | 24 B |
| Grouped SUM, DECIMAL128 | 40 B + temporaries | 40 B (ineligible) | 24 B |
| Grouped AVG, DECIMAL128 | 40 B + temporaries | 40 B (ineligible) | 32 B |
| Global SUM or AVG | one row; kernel cost only | blob (ineligible) | struct; pack and unpack kernels removed |
| HTTP exchange, either side | blob | blob (materialized at `CudfToVelox`) | blob (packed at `CudfToVelox`) |
| Transient at `CudfToVelox` | encode temporary | encode temporary | pack temporary, same size |
| Mixed-representation normalization | n/a | packs struct to blob (grows) | unpacks blob to struct (shrinks) |
| Buffered state inside FINAL | decoded flat columns | native DECIMAL128 | flat columns, same bytes |
| Validity | one string null mask | one null mask | one null mask on `sum`; no parent mask |

No case uses more memory than the branch, and no case uses more than the
baseline. Two implementation details must hold for that to be true:

- construct the struct with `make_structs_column` passing no parent null
  mask, so children are moved rather than copied for null superimposition;
- flatten incoming structs via child `column_view`s, not copies.

The remaining gap to a compact HTTP wire format is inherent to the Java plan
type and is addressed only by section 9.

## 9. Relationship to the long-term fix

The proper fix is a typed ROW intermediate for Presto decimal `sum` and
`avg` at the Velox registration level, mirroring Spark's decimal aggregates
in `velox/functions/sparksql/aggregates/`, with a matching alternative
signature on the Presto side (the `approx_percentile` precedent, prestodb
PR 18386). That would also shrink the HTTP wire format, enable decimal
`sum_extract` companions, and give typed spill on CPU.

This proposal does not conflict with it. The struct the GPU produces here is
what the plan's logical type would become; when the registration changes,
the pack in 6.2 and the normalization in 6.3 delete themselves.

A one-line core change is worth making now regardless:
`DecimalAggregate::addIntermediateResults` (`DecimalAggregate.h:173-175`)
does an unchecked `dynamic_cast` to `FlatVector<StringView>`. Replace with
`VELOX_CHECK_NOT_NULL` so a mismatched intermediate is an error, not a null
dereference.

## 10. Plan

1. Rebase and keep the branch's orthogonal commits as-is: fused DECIMAL64
   reduce (`8b2c9edb13`), streaming final group-by, and temporaries release.
   Treat the cuDF fork pin in `CMake/resolve_dependency_modules/cudf.cmake`
   as a separate decision.
2. Add shape helpers to `exec/DecimalAggregationState.{h,cpp}`:
   `wrapDecimalState(flat columns, function, rawType) -> struct`,
   `flattenDecimalState(column_view) -> flat columns` for STRING or any
   struct shape, `packDecimalState(struct) -> STRING` for all shapes, and
   `normalizeDecimalStateBatches(views, rowType)`.
3. Rewrite the producer and consumer paths in `CudfGroupby` and `CudfReduce`
   against those helpers (6.1).
4. Add the pack step to `CudfToVelox` (6.2), the normalization to the concat
   funnel (6.3), and the evaluator guard (6.4).
5. Delete `CudfPhysicalEncoding`, `CudfColumnEncoding`, the tag plumbing in
   `CudfVector`, `acceptsNativeDecimalSumState`, `NativeDecimalSumEligibility`,
   the opt-ins in `CudfFilterProject`, `CudfLimit`, `CudfLocalPartition`,
   and `CudfBatchConcat`, and the branch's design notes.
6. Tests: all four shapes; positive, negative, zero, and cancelling sums;
   null and all-null groups; DECIMAL64 extremes; DECIMAL(38) range check;
   mixed STRING and STRUCT batches; zero-row batches; GPU partial to CPU
   final through `CudfToVelox`; CPU partial to GPU final through
   `CudfFromVelox`; UCX round trip; local-exchange round trip.
7. Re-measure Q18 under the two-driver managed-memory configuration used on
   the branch and confirm the peak holds at about 90 GB.
8. Separately, open the `VELOX_CHECK_NOT_NULL` change in Velox core and start
   the registration-level discussion in section 9.

## 11. Resuming this work on 4u8g-tur-0042

Facts gathered on 2026-10-02 from the target host:

- Hostname `4u8g-tur-0042`, user `knataraj`, `$HOME=/home/nfs/knataraj`
  (NFS, 97% full, about 879 GB free). Root filesystem is local NVMe with
  about 2.1 TB free. `/raid` is full. `/datasets` is an NFS share with
  about 32 TB free.
- 8 GPUs, NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each,
  driver 595.71.
- Docker works for this user. GitHub SSH works as `karthikeyann`.
- `~/.bash_aliases` already defines `docrun`, `docstop`, `compile`, and
  `tests` (the same file as on the workstation). `docrun` expects to be run
  from the Velox checkout and needs `~/docker-compose.override.yml`.
- None of `velox`, `velox-testing`, or `presto` is checked out there yet.

### 11.1 Workspace layout

`velox-testing/presto/scripts/start_presto_helper.sh` requires `presto` and
`velox` as sibling directories of `velox-testing`. Put all three under one
workspace root on the local NVMe disk rather than on NFS. Pick a writable
directory on `/` (check with `df -h /` and `touch`), for example:

```bash
export WS=/scratch/knataraj      # or another writable path on the local disk
mkdir -p "$WS" && cd "$WS"
```

If no local path is writable, use `$HOME` and expect slower builds.

### 11.2 Clone and check out

```bash
cd "$WS"

# Velox: upstream plus the fork that holds the branch under review.
git clone git@github.com:karthikeyann/velox.git velox
cd velox
git remote add upstream https://github.com/facebookincubator/velox.git
git remote add shrshi https://github.com/shrshi/velox.git
git remote add rapids git@github.com:rapidsai/velox.git
git fetch upstream main
git fetch shrshi perf/cudf-decimal64-global-reduction
git checkout -b perf/cudf-decimal64-global-reduction \
  shrshi/perf/cudf-decimal64-global-reduction
# Confirm the base this document was written against.
git merge-base upstream/main HEAD        # expect af34c8f33a
git log --oneline -1                     # expect 8b8acc1ba9
cd ..

# Presto: prestodb master is what the workstation uses.
git clone git@github.com:prestodb/presto.git presto
git -C presto remote add kn git@github.com:karthikeyann/presto.git

# velox-testing: rapidsai main.
git clone git@github.com:rapidsai/velox-testing.git velox-testing
git -C velox-testing remote add kn git@github.com:karthikeyann/velox-testing.git
```

Copy this document into the Velox checkout if it is not already there:

```bash
mkdir -p "$WS/velox/docs/designs"
cp ~/cudf-self-describing-decimal-aggregate-state.md "$WS/velox/docs/designs/"
```

### 11.3 Create the implementation branch

Do the implementation on a new branch that keeps the orthogonal commits and
replaces the two encoding commits (section 10, step 1):

```bash
cd "$WS/velox"
git checkout -b perf/cudf-decimal-state-self-describing \
  shrshi/perf/cudf-decimal64-global-reduction
```

Either revert the two encoding commits and re-add the producer and consumer
work on top, or cherry-pick the orthogonal commits onto `af34c8f33a` and
start clean. The orthogonal commits are `8b2c9edb13`, `cb8bf9270b`,
`d5f87999cb`, `2b4cf3c00c`, `a757c247fb`, `3e5c85b239`, `8b8acc1ba9`. The
encoding commits to replace are `ff261707e4` and `53bc297a6b`. Note that
`cb8bf9270b` is almost entirely the two markdown notes in `exec/`; drop
those files.

### 11.4 Build and unit test (inside Docker)

Velox compile and test commands only work inside the Docker environment.
From the Velox checkout:

```bash
cd "$WS/velox"
docrun                      # starts or attaches to velox-adapters-cuda-knataraj
# Inside the container the checkout is mounted at /velox.
cd /velox
compile                     # CUDA_ARCHITECTURES=native, cuDF, Arrow, Parquet, benchmarks
cd /velox/_build/release
ninja -j "$(nproc)"
```

The UCX exchange is built only if the container has UCX headers and
libraries. Check `grep UCX_LIBRARY CMakeCache.txt` in the build directory.
On the workstation it was not found, so the UCX operator tests did not
build there. The multi-GPU UCX test in section 11.6 goes through the
Prestissimo images built by velox-testing, which include UCX.

Targeted tests:

```bash
cd /velox/_build/release
ctest -R velox_cudf_decimal_aggregation_test --output-on-failure
ctest -R velox_cudf_aggregation_test --output-on-failure
ctest -R velox_cudf_vector_test --output-on-failure
ctest -R velox_cudf_batch_concat_test --output-on-failure
ctest -R ucx_exchange_test --output-on-failure      # only if UCX was found
ctest -R cudf --output-on-failure                    # full cuDF suite
```

Leave Docker with `docstop` when done.

### 11.5 SF1000 decimal data and single-GPU run

velox-testing generates TPC-H with native DECIMAL columns unless
`--convert-decimals-to-floats` is passed, so the default is the right one for
this work. Generation at SF1000 produces roughly 1 TB of Parquet; put
`PRESTO_DATA_DIR` on the local NVMe disk or on `/datasets`. Check first
whether an SF1000 dataset already exists under `/datasets`.

```bash
cd "$WS/velox-testing/presto/scripts"
export PRESTO_DATA_DIR=/datasets/knataraj/presto-data    # or a local path

# Build images from the sibling checkouts and start one GPU worker.
./start_native_gpu_presto.sh -b native-gpu --build-type release -w 1 -g 0

# Generate data and tables once. Decimals stay DECIMAL by default.
./setup_benchmark_data_and_tables.sh -b tpch -f 1000 -s tpch_sf1000 \
  -d tpch_sf1000 -j "$(( $(nproc) / 2 ))"

# Correctness: Q18 plus the decimal-heavy aggregation queries.
./run_benchmark.sh -b tpch -s tpch_sf1000 -q 1,6,18 -i 1 \
  -t self-describing-state -o ./benchmark_output
```

Q1 covers decimal `sum` and `avg` on both DECIMAL64 inputs and computed
expressions. Q6 covers a global decimal `sum`. Q18 is the memory case. For
result validation, run the same queries against the Java or native CPU
variant (`start_java_presto.sh` or `start_native_cpu_presto.sh`) into a
reference directory and pass it with `--reference-results-dir`.

Memory measurement for Q18 should reproduce the branch's configuration:
two drivers per task, `cudf.memory_resource=managed`,
`cudf.concat_optimization_enabled=true`,
`cudf.batch_size_min_threshold=100000000`,
`cudf.batch_size_max_threshold=500000000`. The worker template is
`presto/docker/config/template/overrides/gpu/etc_worker/config_native.properties`;
edit the generated copy under `presto/docker/config/generated/gpu/` or
regenerate with `--overwrite-config`. Profile with `run_benchmark.sh -p`,
which uses `profiler_functions.sh`, and compare peak allocation to the
branch's reported figure of about 90 GB.

### 11.6 Multi-GPU UCX run

`generate_presto_config.sh` flips `cudf.exchange=true` automatically when
more than one worker is requested, and `cudf.intra_node_exchange=true` is
already in the GPU worker template. So the multi-GPU UCX path is:

```bash
cd "$WS/velox-testing/presto/scripts"
./stop_presto.sh
./start_native_gpu_presto.sh -w 8 -g 0,1,2,3,4,5,6,7 --single-container
./run_benchmark.sh -b tpch -s tpch_sf1000 -q 1,6,18 -i 1 \
  -t self-describing-state-ucx -o ./benchmark_output
```

With several workers, the partial-to-final edge for Q18 goes through
`UcxPartitionedOutput` and `UcxExchange`, which is the path where the branch's
tag was lost (section 3) and where this design needs no code. Verify:

- results match the single-GPU run and the CPU reference;
- worker logs show no `validateIntermediateColumnType` or cuDF type-mismatch
  errors;
- peak GPU allocation per worker is consistent with 16 bytes per group for
  the Q18 state column.

Then run the same with `-w 2` to exercise a smaller exchange fan-out, and
once with a mixed CPU and GPU setup if practical, to hit the string-versus-
struct normalization in the concat funnel (section 6.3).

### 11.7 Implementation checklist

In this order, each step buildable and testable on its own:

1. Shape helpers in `exec/DecimalAggregationState.{h,cpp}` and kernel
   extensions in `exec/DecimalAggregationDevice.cu`. Unit-test pack and
   flatten round trips for all four shapes against the existing
   `serializeDecimalSumState` and `deserializeDecimalSumState` output.
2. Widen `validateIntermediateColumnType` in
   `exec/DecimalAggregationHostOps.cpp` to accept the struct shapes.
3. `CudfGroupby` and `CudfReduce` producer and consumer (section 6.1).
   Existing decimal aggregation tests must pass unchanged in results.
4. `CudfToVelox` pack (section 6.2). GPU partial to CPU final test.
5. Concat-funnel normalization (section 6.3), including zero-row batches.
6. Expression evaluator guard (section 6.4).
7. Delete the tag, opt-in, and eligibility code (section 10, step 5).
8. Audit every site that builds cuDF columns from a Velox `TypePtr`
   (`makeEmptyTable` and any empty outputs in joins or partitions) and
   confirm each flows through the funnel or never meets a struct batch.
9. SF1000 single-GPU, then multi-GPU UCX (sections 11.5 and 11.6).
