---
name: Velox cuDF Type Coverage
overview: Comprehensive analysis comparing all Velox core data types against velox/experimental/cudf/ type support, producing an accurate coverage matrix with counts, mappings, gaps, and representation details.
todos:
  - id: canvas
    content: Build interactive HTML canvas with coverage matrix table, summary cards, and comparison charts
    status: completed
isProject: false
---

# Velox vs cuDF Data Type Coverage Matrix Analysis

## Summary Counts

- **Velox TypeKind enum values:** 17 usable types + 1 sentinel (INVALID) = 18 total
- **Velox logical types (reusing a TypeKind):** 7 additional (DATE, SHORT DECIMAL, LONG DECIMAL, INTERVAL_DAY_TIME, INTERVAL_YEAR_MONTH, TIME, TIME_MICRO_UTC)
- **Presto extension types:** 18 (UUID, IPADDRESS, JSON, etc.)
- **Total distinct Velox types:** 43 (18 TypeKind + 7 logical + 18 Presto extension)
- **cudf type_id values:** 29 total (from libcudf)
- **Velox types mapped to cuDF:** 16 (13 TypeKind cases + 3 logical sub-type branches: DATE, SHORT DECIMAL, LONG DECIMAL)
- **Velox types NOT mapped to cuDF:** 27 (5 TypeKind + 4 logical + 18 Presto extension)
- **cuDF types with no Velox counterpart:** 15 (unsigned ints, extra timestamp precisions, durations, DICTIONARY32, DECIMAL32, EMPTY)

---

## Source Files (Authoritative)

- Velox TypeKind enum: [velox/type/Type.h](velox/type/Type.h) lines 71-93
- cuDF type mapping: [velox/experimental/cudf/exec/VeloxCudfInterop.cpp](velox/experimental/cudf/exec/VeloxCudfInterop.cpp) lines 48-117
- cuDF type_id mirror: [velox/experimental/cudf/vector/TableViewPrinter.h](velox/experimental/cudf/vector/TableViewPrinter.h) lines 41-103
- Expression-level type handling: [velox/experimental/cudf/expression/AstUtils.h](velox/experimental/cudf/expression/AstUtils.h)
- Logical types (DATE, DECIMAL, INTERVAL, TIME): [velox/type/Type.h](velox/type/Type.h) lines 893-1620
- Presto extension types: [velox/functions/prestosql/types/](velox/functions/prestosql/types/) (18 header files)

---

## Coverage Matrix: Velox TypeKind to cuDF Mapping

### Tier 1 -- Fully Mapped (13 TypeKind + 3 logical = 16 type cases)


| #   | Velox Type            | TypeKind            | C++ Backing Type           | cuDF type_id          | cuDF C++ Type            | Notes                                                                                |
| --- | --------------------- | ------------------- | -------------------------- | --------------------- | ------------------------ | ------------------------------------------------------------------------------------ |
| 1   | BOOLEAN               | BOOLEAN(0)          | bool                       | BOOL8                 | bool                     | Direct 1:1                                                                           |
| 2   | TINYINT               | TINYINT(1)          | int8_t                     | INT8                  | int8_t                   | Direct 1:1                                                                           |
| 3   | SMALLINT              | SMALLINT(2)         | int16_t                    | INT16                 | int16_t                  | Direct 1:1                                                                           |
| 4   | INTEGER               | INTEGER(3)          | int32_t                    | INT32                 | int32_t                  | Direct 1:1                                                                           |
| 5   | BIGINT                | BIGINT(4)           | int64_t                    | INT64                 | int64_t                  | Direct 1:1                                                                           |
| 6   | REAL                  | REAL(5)             | float                      | FLOAT32               | float                    | Direct 1:1                                                                           |
| 7   | DOUBLE                | DOUBLE(6)           | double                     | FLOAT64               | double                   | Direct 1:1                                                                           |
| 8   | VARCHAR               | VARCHAR(7)          | StringView                 | STRING                | string_view              | Direct 1:1                                                                           |
| 9   | VARBINARY             | VARBINARY(8)        | StringView                 | STRING                | string_view              | Lossy: both map to STRING; round-trip uses Arrow format "z" hack                     |
| 10  | TIMESTAMP             | TIMESTAMP(9)        | Timestamp (sec+nano)       | TIMESTAMP_NANOSECONDS | int64_t (ns since epoch) | Precision difference: Velox stores (seconds, nanos) pair; cuDF is single int64 nanos |
| 11  | DATE                  | INTEGER(3) logical  | int32_t (days since epoch) | TIMESTAMP_DAYS        | int32_t                  | Mapped via isDate() check                                                            |
| 12  | SHORT DECIMAL         | BIGINT(4) logical   | int64_t (unscaled)         | DECIMAL64 + scale     | int64_t                  | Scale preserved via cudf::numeric::scale_type                                        |
| 13  | LONG DECIMAL          | HUGEINT(10) logical | int128_t (unscaled)        | DECIMAL128 + scale    | __int128_t               | Scale preserved; HUGEINT MUST be decimal                                             |
| 14  | ARRAY                 | ARRAY(30)           | nested children            | LIST                  | nested children          | Recursive child type mapping                                                         |
| 15  | ROW                   | ROW(32)             | named fields               | STRUCT                | unnamed fields           | Field names lost in cuDF STRUCT                                                      |
| 16  | HUGEINT (non-decimal) | HUGEINT(10)         | int128_t                   | --                    | --                       | VELOX_CHECK fails: "HUGEINT should only be used for DECIMAL128"                      |


### Tier 2 -- Unsupported in cuDF Mapping (hit CUDF_FAIL or commented out)


| #   | Velox Type | TypeKind     | Status in cuDF | Notes                                                     |
| --- | ---------- | ------------ | -------------- | --------------------------------------------------------- |
| 17  | MAP        | MAP(31)      | Commented out  | cuDF has no native MAP; could decompose to LIST of STRUCT |
| 18  | UNKNOWN    | UNKNOWN(33)  | Commented out  | All-null sentinel type; no cuDF equivalent                |
| 19  | FUNCTION   | FUNCTION(34) | Commented out  | Lambda type; not a data column type                       |
| 20  | OPAQUE     | OPAQUE(35)   | Commented out  | Opaque shared_ptr wrapper; not serializable               |
| 21  | INVALID    | INVALID(36)  | Commented out  | Sentinel/error marker                                     |


### Tier 3 -- Logical Types Not Handled by cuDF Mapping Switch


| #   | Velox Logical Type  | Reuses TypeKind | cuDF Status                 | Expression Layer                                                  |
| --- | ------------------- | --------------- | --------------------------- | ----------------------------------------------------------------- |
| 22  | INTERVAL_DAY_TIME   | BIGINT(4)       | Not in veloxToCudfDataType  | Partial: AstUtils.h maps to cudf::duration_ms for scalar literals |
| 23  | INTERVAL_YEAR_MONTH | INTEGER(3)      | TODO in veloxToCudfDataType | Explicit VELOX_FAIL in AstUtils.h                                 |
| 24  | TIME (milli, local) | BIGINT(4)       | Not handled                 | No mapping exists                                                 |
| 25  | TIME_MICRO_UTC      | BIGINT(4)       | Not handled                 | No mapping exists                                                 |


### Tier 4 -- Presto Extension Types (none mapped to cuDF)


| #   | Extension Type           | Reuses TypeKind         | Native C++ Type                |
| --- | ------------------------ | ----------------------- | ------------------------------ |
| 26  | UUID                     | HUGEINT                 | int128_t                       |
| 27  | IPADDRESS                | HUGEINT                 | int128_t                       |
| 28  | IPPREFIX                 | ROW(IPADDRESS, TINYINT) | Row of int128_t + int8_t       |
| 29  | TIMESTAMP WITH TIME ZONE | BIGINT                  | packed int64_t (millis + zone) |
| 30  | TIME WITH TIME ZONE      | BIGINT                  | packed int64_t                 |
| 31  | BINGTILE                 | BIGINT                  | packed int64_t                 |
| 32  | BIGINT_ENUM              | BIGINT                  | int64_t + metadata             |
| 33  | VARCHAR_ENUM             | VARCHAR                 | StringView                     |
| 34  | JSON                     | VARCHAR                 | StringView (text JSON)         |
| 35  | HYPERLOGLOG              | VARBINARY               | bytes                          |
| 36  | P4HYPERLOGLOG            | VARBINARY               | bytes                          |
| 37  | KHYPERLOGLOG             | VARBINARY               | bytes                          |
| 38  | SETDIGEST                | VARBINARY               | bytes                          |
| 39  | QDIGEST                  | VARBINARY               | bytes                          |
| 40  | TDIGEST                  | VARBINARY               | bytes                          |
| 41  | SFMSKETCH                | VARBINARY               | bytes                          |
| 42  | GEOMETRY                 | VARBINARY               | bytes                          |
| 43  | SPHERICAL_GEOGRAPHY      | VARBINARY               | bytes                          |


### Tier 5 -- cuDF Types With No Velox Counterpart


| #   | cuDF type_id           | Notes                                                                                |
| --- | ---------------------- | ------------------------------------------------------------------------------------ |
| 1   | EMPTY                  | Sentinel/placeholder                                                                 |
| 2   | UINT8                  | Velox has no unsigned integer types                                                  |
| 3   | UINT16                 | Velox has no unsigned integer types                                                  |
| 4   | UINT32                 | Velox has no unsigned integer types                                                  |
| 5   | UINT64                 | Velox has no unsigned integer types                                                  |
| 6   | TIMESTAMP_SECONDS      | Velox only maps to TIMESTAMP_NANOSECONDS                                             |
| 7   | TIMESTAMP_MILLISECONDS | Velox only maps to TIMESTAMP_NANOSECONDS                                             |
| 8   | TIMESTAMP_MICROSECONDS | Velox only maps to TIMESTAMP_NANOSECONDS                                             |
| 9   | DURATION_DAYS          | Partial expression-layer support only                                                |
| 10  | DURATION_SECONDS       | No Velox mapping                                                                     |
| 11  | DURATION_MILLISECONDS  | INTERVAL_DAY_TIME uses this in expression layer only                                 |
| 12  | DURATION_MICROSECONDS  | No Velox mapping                                                                     |
| 13  | DURATION_NANOSECONDS   | No Velox mapping                                                                     |
| 14  | DICTIONARY32           | cuDF dictionary encoding; Velox handles dictionaries at vector level, not type level |
| 15  | DECIMAL32              | cuDF supports 32-bit decimal; Velox only has 64-bit and 128-bit                      |


---

## How Data Types Are Represented

### Velox Type Architecture

Velox uses a **two-tier type system**:

1. **Physical tier (TypeKind enum):** 18 discriminant values defining the storage layout and vector encoding. Each TypeKind maps to a C++ native type via `TypeTraits<KIND>::NativeType`.
2. **Logical tier (Type subclasses):** Multiple SQL-level types can share the same TypeKind. For example, DATE, INTERVAL_YEAR_MONTH, and plain INTEGER all have `kind() == INTEGER` but are distinguished by C++ class identity (`DateType`, `IntervalYearMonthType`, `IntegerType`). This is checked via methods like `isDate()`, `isDecimal()`, `isIntervalDayTime()`.

Key design consequences:

- Vectors are always typed by TypeKind for storage (a DATE column is physically a FlatVector of int32_t)
- Logical type information is carried in the Type metadata, not in the vector encoding
- Presto extension types add a `name()` string distinct from `kindName()` while keeping the same physical layout

### cuDF Type Architecture

cuDF uses a **flat enum** (`cudf::type_id`) with 29+ values. Every distinct type gets its own enum entry. Type metadata (like decimal scale) is carried in `cudf::data_type` which wraps `type_id` plus optional `scale_type`.

Key design consequences:

- Unsigned integers (UINT8-64) are first-class types; Velox has no equivalent
- Multiple timestamp/duration precisions are separate enum values; Velox uses a single TIMESTAMP with a (seconds, nanos) struct
- DECIMAL32/64/128 are all separate entries; Velox only uses 64 and 128
- No concept of "logical" type subclasses; everything is in the enum

### Key Similarities

- Both use fixed-width numeric types with identical C++ backing for INT8-64, FLOAT32/64, BOOL
- Both support nested types (LIST/ARRAY, STRUCT/ROW)
- Both support variable-width string data
- Both represent decimal with unscaled integer + scale metadata

### Key Differences

- **Unsigned integers:** cuDF has them (UINT8-64); Velox does not
- **Timestamp precision:** cuDF has 5 resolutions (days, seconds, ms, us, ns); Velox has one TIMESTAMP type (seconds + nanos pair), with DATE as a separate logical type
- **Duration/Interval:** cuDF has 5 DURATION types; Velox has INTERVAL_DAY_TIME (ms as int64) and INTERVAL_YEAR_MONTH (months as int32), both logical overlays on BIGINT/INTEGER
- **DECIMAL32:** cuDF supports it; Velox skips straight to DECIMAL64/128
- **Binary vs String:** cuDF has only STRING; Velox distinguishes VARCHAR and VARBINARY. The cuDF bridge loses VARBINARY identity (round-tripped via Arrow format marker "z")
- **MAP type:** Velox has a native MAP TypeKind; cuDF has no MAP -- typically represented as LIST of STRUCT with key/value children
- **Dictionary encoding:** cuDF has DICTIONARY32 as a type_id; Velox handles dictionary encoding at the vector level (DictionaryVector) independent of the type system
- **Extension types:** Velox has 18+ Presto extension types; cuDF has no extension type mechanism (though Arrow extensions could bridge some)

---

## Conversion Mechanism

The cuDF bridge uses two paths:

1. **Direct switch:** `veloxToCudfDataType()` in `VeloxCudfInterop.cpp` -- used for metadata, empty tables, AST expression building
2. **Arrow bridge:** `with_arrow::toCudfTable()` / `toVeloxColumn()` -- Velox exports to Arrow, cuDF imports from Arrow. This path can handle more types than the direct switch since Arrow has broader type coverage, but any path that calls `veloxToCudfDataType()` directly (empty tables, aggregation metadata, join keys) will reject unmapped types

There is **no reverse mapping** (`cudf::type_id` back to Velox `TypePtr`). The cuDF-to-Velox direction always goes through Arrow import.

---

## Deliverable

Create an interactive HTML canvas with:

- A filterable coverage matrix table showing all types with status indicators
- Summary statistics cards (total counts, coverage percentages)
- Visual breakdown by tier (pie/donut chart)
- Side-by-side type representation comparison
- Sortable/filterable columns for the full matrix

