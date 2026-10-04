# Self-Describing Decimal Aggregate State for velox-cudf

Status: implemented on `perf/cudf-decimal-state-self-describing`.
Written: 2026-10-02; updated to the implementation. File references are to
`velox/experimental/cudf/` on that branch and name functions rather than
lines.

This document records why the aggregate state is carried as a self-describing
struct, the rules that keep it interchangeable with the CPU blob, and the
code sites that interpret it.

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

## 3. Why not tag the column

An earlier prototype stored only the DECIMAL128 sum under the VARBINARY column
and tagged the column with a per-column physical-encoding flag. Because count
and overflow were dropped, a valid blob could be rebuilt only if the column was
known to come from a DECIMAL64 `sum`, so the provenance had to travel with the
vector through every operator (select, slice, concat, partition, exchange) and
be re-derived from the plan where it was lost. Every operator that did not know
about the tag either materialized the blob or, on the UCX exchange path, failed
validation. The representation below removes the provenance problem instead of
plumbing it.

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
  low 127 bits of each same-sign add and counts the carry in `overflow`, so
  one unit of overflow is worth exactly 2^127 and the true total is
  `sum + overflow * 2^127` as an exact integer, with `|sum| < 2^127` after
  every add (the canonical form). `DecimalSumAggregate::computeFinalValue`
  calls `DecimalUtil::adjustSumForOverflow`, which accepts `overflow == 1`
  with a negative sum and `overflow == -1` with a positive sum (adding
  `overflow * 2^127` modulo 2^128, which is the exact total), raises
  "Decimal overflow" for any other nonzero value (the total does not fit in
  int128), and then range-checks against DECIMAL(38).
  `DecimalUtil::computeAverage` accepts any overflow and returns
  `(sum + overflow * 2^127) / count` with half-up rounding, computed as a
  split division without widening past 128 bits; the average is not
  range-checked.
- The GPU merge does not preserve the canonical form. cuDF's DECIMAL128 SUM
  adds the `sum` children modulo 2^128 while the `overflow` children are
  summed exactly, so after a merge that crosses +-2^127 the pair is a
  different representative of the same total modulo 2^128, and the CPU's
  sign predicate would reject a legal total. The GPU FINAL step therefore
  folds unconditionally: `finalizeDecimalSum` computes
  `sum + overflow * 2^127` modulo 2^128, which equals the true total whenever
  it fits in int128 (every total the CPU accepts), and then range-checks
  against DECIMAL(38). `finalizeDecimalAverage` uses the CPU's split division
  when the pair is canonical (bit-identical to `computeAverage`) and divides
  the folded total half-up otherwise; the two differ by at most one ulp in
  exact-half cases, and the CPU's own result is already order dependent by
  that ulp because a different accumulation order yields a different
  canonical pair. Both run on device (`exec/DecimalAggregationHostOps.h`) in a
  single pass with one host sync per output batch, and skip the overflow work
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
  semantics. `flattenDecimalState` and `packDecimalState` nevertheless
  tolerate a parent mask (a nullifying gather such as an outer-join probe
  adds one): a row is null when the parent or the child is null, and the
  parent mask is folded into an owned copy of every carried child only in
  that case, at zero cost when the parent has no nulls. The exception is
  `unwrapDecimalState`, which rejects a parent mask; it is reached only from
  the group-by's own buffered FINAL state, which never has one.

Known limitation: GPU producers do not track int128 carries. cuDF's
DECIMAL128 SUM wraps modulo 2^128, so the `overflow` child emitted by a GPU
producer (kSum128 and kAvg128) is always 0 and `packDecimalState` writes that 0
into the blob. A total beyond int128 is therefore indistinguishable from its
alias modulo 2^128: SUM raises "Decimal overflow" when the alias lands outside
DECIMAL(38) and is silently wrong when it wraps back into range, and AVG,
which is not range-checked, is silently wrong in both cases. This is the
pre-existing behaviour of the blob path and is unchanged by this design. The
overflow field is meaningful only when it came from a CPU-produced state. On
merge the overflow children are summed exactly but the sum children still
wrap, so the merged pair is congruent to the true total modulo 2^128 without
being canonical; FINAL folds it as described in section 4. The complete
divergence set from the CPU is therefore: (1) a total beyond int128, as above;
and (2) for AVG only, a total inside int128 whose merged pair is
non-canonical, where the average may differ from the CPU's by one unit in the
last place at an exact-half quotient. GPU SUM equals CPU SUM for every total
inside int128. Case (2) is inherent to not tracking carries: the CPU's own
result depends on its accumulation order in exact-half cases, so no carry-free
merge can match every CPU order (pinned by `cpuCarryExactHalfAvg` in
`DecimalAggregationTest.cpp`). Closing the gap needs a carry-tracking
reduction for DECIMAL128 (a custom reduce over `(sum, overflow)` for the
global path and a host UDF or two-pass approach for `cudf::groupby`).

Q18 lands on the first row: 16 bytes plus one validity bit per group, the
same as the branch.

## 6. Where the bytes are interpreted

### 6.1 `CudfGroupby` and `CudfReduce` (producer and consumer)

- PARTIAL: run the existing SUM and COUNT aggregations (and the fused
  DECIMAL64 reduce from commit `2b69b23170` for global paths), then wrap the
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
  replaces the raw `cudf::concatenate` of buffered and new state in
  `computeFinalGroupbyIncrementally` (`exec/CudfGroupby.cpp`) that would
  have failed on mixed forms.
- INTERMEDIATE output re-wraps in the plan-determined shape. FINAL output
  folds the merged overflow into the sum and runs the DECIMAL(38) range check
  (`finalizeDecimalSum`) or divides with the CPU's rounding
  (`finalizeDecimalAverage`), as described in section 4. Both read the
  overflow field through `flattenDecimalState`, so a CPU-produced blob and a
  GPU struct are treated alike.

### 6.2 `CudfToVelox` (`exec/CudfConversion.cpp`)

Both conversion paths (the per-batch output and the buffered concatenated
output) go through one `exportToVelox` helper, which calls
`packDecimalStatesForExport` before `with_arrow::toVeloxColumn`: every
top-level VARBINARY column that is physically a STRUCT is packed into the
32-byte blob with `packDecimalState`, which handles all four shapes. This
single site covers HTTP exchange (`PartitionedOutputAdapter` declares
`acceptsGpuInput() == false`, so `ToCudf.cpp` inserts `CudfToVelox` before
it), CPU fallback, final output, and any CPU-side spill.

### 6.3 The concat funnel in `exec/Utilities.cpp`

`getConcatenatedTable`, `getConcatenatedTableBatched`, and
`getConcatenatedCudfVectorsBatched` are used by `CudfBatchConcat`,
`CudfToVelox`, `CudfGroupby`, `CudfReduce`, `CudfDistinct`, `CudfWindow`,
`CudfOrderBy`, `CudfHashJoin`, `CudfLocalMerge`, and `CudfNestedLoopJoin`.
`normalizeDecimalStateTableViews` (`exec/Utilities.h`) runs before
`cudf::concatenate`, once per output batch in `getConcatenatedTableBatched`.
For each top-level VARBINARY column whose physical form differs across
batches it:

- unpacks STRING batches to the struct shape present in the other batches
  (normalizing toward the smaller form, not toward the blob);
- if struct shapes differ, widens all to the four-child-equivalent full form;
- treats zero-row batches as wildcards and drops or re-types them.
  `makeEmptyTable` (file-local in `exec/Utilities.cpp`) builds a STRING
  column for VARBINARY, and a zero-row STRING concatenated with STRUCT batches
  is a cuDF type-mismatch error.

Raw `cudf::concatenate` calls outside the funnel: the group-by's buffered
state concat (handled by 6.1); `CudfDistinct` and `CudfMarkDistinct`, where
a state column could only appear as a distinct key, so each now fails with a
clear error through `isDecimalStateUnderVarbinary` rather than supporting it;
and `UcxPartitionedOutput` (single-driver inputs, always uniform).

### 6.4 Expression evaluator

`CudfFilterProject` records at compile time which VARBINARY input channels
its filter or computed projections read, and checks per batch that none of
them is physically a STRUCT, failing with a clear message otherwise. Identity
projections pass the state through untouched, and nothing is ever packed
here. Presto plans never compute on an aggregation intermediate, so this is a
guard, not a hot path.

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
| Buffered state inside FINAL | decoded flat columns | native DECIMAL128 | struct of the plan shape over those flat columns, same bytes (SUM over DECIMAL128 also merges the 8-byte overflow child, no wider than the baseline's decoded count) |
| Validity | one string null mask | one null mask | one null mask on `sum`; no parent mask |

No case uses more memory than the branch, and no case uses more than the
baseline. Two implementation details must hold for that to be true:

- construct the struct with `make_structs_column` passing no parent null
  mask, so children are moved rather than copied for null superimposition
  (`wrapDecimalState`);
- flatten incoming structs via child `column_view`s, not copies
  (`flattenDecimalState`).

Two paths copy, both off the production hot path: a struct that arrives with
a parent null mask has every carried child copied with the mask folded in,
and the version-skew widen in `normalizeDecimalStateBatches` copies children
into the wider struct.

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

A one-line core change is worth making regardless, and is not part of this
series: `DecimalAggregate::addIntermediateResults` (`DecimalAggregate.h`)
does an unchecked `dynamic_cast` to `FlatVector<StringView>`. Replace with
`VELOX_CHECK_NOT_NULL` so a mismatched intermediate is an error, not a null
dereference.

## 10. Implementation map

1. Shape helpers in `exec/DecimalAggregationState.{h,cpp}`:
   `wrapDecimalState(DecimalStateColumns, DecimalStateShape, stream, mr)`,
   `unwrapDecimalState(struct)`,
   `flattenDecimalState(view, scale, needCount, needOverflow, stream, mr)`
   for STRING or any struct shape, `packDecimalState(struct, stream, mr)`
   for all shapes, `unpackDecimalState(blob, shape, scale, stream, mr)`, and
   `normalizeDecimalStateBatches(views, stream, mr)`; plus
   `normalizeDecimalStateTableViews(tableViews, rowType, stream, mr)` in
   `exec/Utilities.h`. The shape is resolved from the plan by
   `decimalStateInfoFor` and FINAL goes through `finalizeDecimalSum` and
   `finalizeDecimalAverage` (`exec/DecimalAggregationHostOps.h`).
2. Producer and consumer paths in `CudfGroupby` and `CudfReduce` against
   those helpers (6.1).
3. The pack step in `CudfToVelox` (6.2), the normalization in the concat
   funnel (6.3), and the guards in `CudfFilterProject`, `CudfDistinct` and
   `CudfMarkDistinct` (6.3, 6.4).
4. Tests: all four shapes; positive, negative, zero, and cancelling sums;
   null and all-null groups; DECIMAL64 extremes; DECIMAL(38) range check and
   CPU-blob overflow folding; mixed STRING and STRUCT batches; zero-row
   batches; GPU partial to CPU final through `CudfToVelox`; CPU partial to
   GPU final through `CudfFromVelox`; local-exchange round trip. The UCX
   round trip and the Q18 re-measurement under the two-driver managed-memory
   configuration remain to be run.
5. Separately, start the registration-level discussion in section 9.
