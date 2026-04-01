---
name: Velox CUDF Coverage Analysis
overview: Complete analysis of Velox function inventory vs. CUDF GPU-accelerated function coverage, with correspondence mapping and coverage percentages.
todos:
  - id: create-canvas
    content: Create an interactive HTML canvas/table visualization of the coverage matrix if the user wants it
    status: completed
isProject: false
---

# Velox CUDF Function Coverage Analysis

## 1. Velox Total Function Inventory

Source: canonical coverage data files under `velox/functions/{prestosql,sparksql}/coverage/data/`

### Presto Functions

- **Scalar functions**: 400 unique (`filter` is listed twice in the file, deduped to 400)
- **Aggregate functions**: 78
- **Window functions**: 11
- **Total unique Presto**: **489**

### Spark Functions

- **Scalar functions**: 342 (this list includes aggregate/window names that overlap)
- **Aggregate functions**: 50 (8 names not in scalar list: `array_agg`, `histogram_numeric`, `regr_avgx`, `regr_avgy`, `regr_count`, `regr_r2`, `try_avg`, `try_sum`)
- **Window functions**: 11 (all already in scalar list)
- **Total unique Spark**: **350** (342 + 8 agg-only names)

### Combined (Presto + Spark, each counted separately)

- **Presto**: 489 functions
- **Spark**: 350 functions

---

## 2. CUDF Functions Inventory (velox/experimental/cudf/)

All CUDF function registrations come from two files:

- Scalars: [ExpressionEvaluator.cpp](velox/experimental/cudf/expression/ExpressionEvaluator.cpp) -- `registerBuiltinFunctions()` + `registerSparkFunctions()`
- Aggregates: [CudfHashAggregation.cpp](velox/experimental/cudf/exec/CudfHashAggregation.cpp) -- `registerStepAwareBuiltinAggregationFunctions()`

### CUDF Scalar Functions: 17 unique implementations (21 registered names)


| #   | CUDF Function Name(s)          | C++ Class               | Category    | Engine     |
| --- | ------------------------------ | ----------------------- | ----------- | ---------- |
| 1   | `split`                        | SplitFunction           | String      | Both       |
| 2   | `cardinality`                  | CardinalityFunction     | Array       | Both       |
| 3   | `substr`, `substring`          | SubstrFunction          | String      | Both       |
| 4   | `coalesce` (no prefix)         | CoalesceFunction        | Conditional | Both       |
| 5   | `round`                        | RoundFunction           | Math        | Both       |
| 6   | `year`                         | YearFunction            | DateTime    | Both       |
| 7   | `length`                       | LengthFunction          | String      | Both       |
| 8   | `lower`                        | LowerFunction           | String      | Both       |
| 9   | `upper`                        | UpperFunction           | String      | Both       |
| 10  | `like`                         | LikeFunction            | String      | Both       |
| 11  | `concat`                       | ConcatFunction          | String      | Both       |
| 12  | `greaterthan`, `gt`            | BinaryFunction(GREATER) | Comparison  | Both       |
| 13  | `divide`                       | BinaryFunction(DIV)     | Math        | Both       |
| 14  | `switch`, `if` (no prefix)     | SwitchFunction          | Conditional | Both       |
| 15  | `cast`, `try_cast` (no prefix) | CastFunction            | Type Cast   | Both       |
| 16  | `hash_with_seed`               | HashFunction            | Hashing     | Spark only |
| 17  | `date_add`                     | DateAddFunction         | DateTime    | Spark only |


### CUDF Aggregate Functions: 6


| #   | CUDF Function Name | C++ Aggregator           | Presto Name     | Spark Name            |
| --- | ------------------ | ------------------------ | --------------- | --------------------- |
| 1   | `sum`              | SumAggregator            | sum             | sum                   |
| 2   | `count`            | CountAggregator          | count           | count                 |
| 3   | `min`              | MinAggregator            | min             | min                   |
| 4   | `max`              | MaxAggregator            | max             | max                   |
| 5   | `avg`              | MeanAggregator           | avg             | avg                   |
| 6   | `approx_distinct`  | ApproxDistinctAggregator | approx_distinct | approx_count_distinct |


### CUDF Total: **23 unique functions** (17 scalar + 6 aggregate)

---

## 3. AST Expression Path -- Additional GPU-Accelerated Operations

Beyond registered functions, the AST evaluator in [AstExpressionUtils.h](velox/experimental/cudf/expression/AstExpressionUtils.h) supports these operations natively on GPU via `cudf::compute_column`:

**Binary Operations (13 Presto + 12 Spark names, mapped to cudf AST operators):**


| Operation      | Presto Name | Spark Name           | cudf AST Op      |
| -------------- | ----------- | -------------------- | ---------------- |
| Addition       | `plus`      | `add`                | ADD              |
| Subtraction    | `minus`     | `subtract`           | SUB              |
| Multiplication | `multiply`  | `multiply`           | MUL              |
| Division       | `divide`    | `divide`             | DIV              |
| Equality       | `eq`        | `equalto`            | EQUAL            |
| Not Equal      | `neq`       | --                   | NOT_EQUAL        |
| Less Than      | `lt`        | `lessthan`           | LESS             |
| Greater Than   | `gt`        | `greaterthan`        | GREATER          |
| Less/Equal     | `lte`       | `lessthanorequal`    | LESS_EQUAL       |
| Greater/Equal  | `gte`       | `greaterthanorequal` | GREATER_EQUAL    |
| Logical AND    | `and`       | `and`                | NULL_LOGICAL_AND |
| Logical OR     | `or`        | `or`                 | NULL_LOGICAL_OR  |
| Modulo         | `mod`       | `mod`                | MOD              |


**Unary + Special Operations:**


| Operation                                | AST Op                           |
| ---------------------------------------- | -------------------------------- |
| `not`                                    | NOT                              |
| `is_null`                                | IS_NULL                          |
| `isnotnull`                              | IS_NULL + NOT                    |
| `between`                                | GREATER_EQUAL + LESS_EQUAL + AND |
| `in`                                     | chain of EQUAL + OR              |
| `cast`/`try_cast` (to int/bigint/double) | CAST_TO_INT64 / CAST_TO_FLOAT64  |


These are **16 additional unique operation types** handled on GPU, beyond the 23 registered functions.

---

## 4. Correspondence Analysis: CUDF vs Velox Functions

### Result: ALL 23 CUDF functions correspond to existing Velox functions. **0 new functions without correspondence.**

### Detailed Correspondence Table -- Scalar Functions


| #   | CUDF Function        | Presto Match                     | Spark Match                 | Notes                                      |
| --- | -------------------- | -------------------------------- | --------------------------- | ------------------------------------------ |
| 1   | `split`              | `split`                          | `split`                     | Exact match                                |
| 2   | `cardinality`        | `cardinality`                    | `cardinality`               | Exact match                                |
| 3   | `substr`/`substring` | `substr`                         | `substr`, `substring`       | Exact match                                |
| 4   | `coalesce`           | `coalesce` (special form)        | `coalesce`                  | Special form, not in Presto scalar list    |
| 5   | `round`              | `round`                          | `round`                     | Exact match                                |
| 6   | `year`               | `year`                           | `year`                      | Exact match                                |
| 7   | `length`             | `length`                         | `length`                    | Exact match                                |
| 8   | `lower`              | `lower`                          | `lower`                     | Exact match                                |
| 9   | `upper`              | `upper`                          | `upper`                     | Exact match                                |
| 10  | `like`               | `like` (special form)            | `like`                      | Pattern match function                     |
| 11  | `concat`             | `concat`                         | `concat`                    | Exact match                                |
| 12  | `greaterthan`/`gt`   | `gt` (internal op)               | `greaterthan` (internal op) | Binary operator, not in SQL function lists |
| 13  | `divide`             | `divide` (internal op)           | `divide` (internal op)      | Binary operator, not in SQL function lists |
| 14  | `switch`/`if`        | `switch`/`if` (special form)     | `if`/`when`                 | Control flow special form                  |
| 15  | `cast`/`try_cast`    | `cast`/`try_cast` (special form) | `cast`                      | Type conversion special form               |
| 16  | `hash_with_seed`     | N/A                              | `hash` (murmurhash3)        | Spark only                                 |
| 17  | `date_add`           | Different signature (3-arg)      | `date_add` (2-arg)          | Spark 2-arg version only                   |


### Detailed Correspondence Table -- Aggregate Functions


| #   | CUDF Function     | Presto Match      | Spark Match             | Notes                                                 |
| --- | ----------------- | ----------------- | ----------------------- | ----------------------------------------------------- |
| 1   | `sum`             | `sum`             | `sum`                   | Full step support (partial/final/intermediate/single) |
| 2   | `count`           | `count`           | `count`                 | Full step support                                     |
| 3   | `min`             | `min`             | `min`                   | Full step support                                     |
| 4   | `max`             | `max`             | `max`                   | Full step support                                     |
| 5   | `avg`             | `avg`             | `avg`                   | Full step support with sum+count intermediate         |
| 6   | `approx_distinct` | `approx_distinct` | `approx_count_distinct` | Global aggregation only (no group-by)                 |


### New Functions Without Correspondence: **NONE**

All 23 CUDF functions are GPU implementations of existing Velox/Presto/Spark functions.

---

## 5. Coverage Percentages

### Against Presto Catalog (489 total)


| Category                        | Presto Total | CUDF Covered                                                                                | Coverage        |
| ------------------------------- | ------------ | ------------------------------------------------------------------------------------------- | --------------- |
| Scalar (named in coverage file) | 400          | 9 (`split`, `cardinality`, `substr`, `round`, `year`, `length`, `lower`, `upper`, `concat`) | **2.3%**        |
| Scalar (incl. special forms)    | 400+         | 15 (above + `coalesce`, `like`, `cast`, `try_cast`, `if`, `divide`, `gt`)                   | ~3.8%           |
| Aggregate                       | 78           | 6 (`sum`, `count`, `min`, `max`, `avg`, `approx_distinct`)                                  | **7.7%**        |
| Window                          | 11           | 0                                                                                           | **0%**          |
| **Overall**                     | **489**      | **15-21**                                                                                   | **3.1% - 4.3%** |


### Against Spark Catalog (350 total)


| Category                        | Spark Total | CUDF Covered                                                                                                                                                    | Coverage        |
| ------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| Scalar (named in coverage file) | 342         | 16 (`split`, `cardinality`, `substr`, `substring`, `coalesce`, `round`, `year`, `length`, `lower`, `upper`, `like`, `concat`, `if`, `cast`, `date_add`, `hash`) | **4.7%**        |
| Scalar (incl. internal ops)     | 342         | 19 (above + `divide`, `greaterthan`, `switch`)                                                                                                                  | ~5.6%           |
| Aggregate                       | 50          | 5-6 (`sum`, `count`, `min`, `max`, `avg`, + `approx_count_distinct` mapping)                                                                                    | **10-12%**      |
| Window                          | 11          | 0                                                                                                                                                               | **0%**          |
| **Overall**                     | **350**     | **21-25**                                                                                                                                                       | **6.0% - 7.1%** |


---

## 6. Additional Notes

### Implemented but NOT Registered (Dead Code / Future Work)

Three function classes exist in `ExpressionEvaluator.cpp` but have NO `registerCudfFunction` call:

- `StartswithFunction` -- would map to Presto `starts_with`
- `EndswithFunction` -- would map to Presto `ends_with`
- `ContainsFunction` -- would map to Presto `contains`

### CUDF Operators (not functions, but GPU-accelerated plan nodes)

- `CudfFilterProject` -- filter + project
- `CudfHashAggregation` -- hash aggregation
- `CudfHashJoin` -- hash join
- `CudfOrderBy` -- order by / sort
- `CudfTopN` -- top N
- `CudfLimit` -- limit
- `CudfAssignUniqueId` -- unique ID assignment
- `CudfLocalPartition` -- local partitioning
- `CudfBatchConcat` -- batch concatenation

### Limitations

- `approx_distinct` supports global aggregation only (no group-by)
- `date_add` is Spark's 2-arg version only (not Presto's 3-arg version)
- `registerPrestoFunctions()` is an empty stub -- no Presto-specific scalar functions
- No window functions on GPU
- No decimal type support for `round`

