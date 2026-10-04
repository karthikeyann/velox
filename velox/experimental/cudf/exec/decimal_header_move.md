# Move Decimal Function Structs from `.cpp` to Headers

## Goal

Move decimal simple-function struct definitions out of anonymous namespaces in `.cpp` files and into includable headers, so that GPU `.cu` compilation units can `#include` them directly. This is a prerequisite for the GPU SFI architecture to compile decimal `call()` bodies for device execution via `GpuSimpleFunctionAdapter`.

The change is **purely structural** -- no logic changes, no behavior changes, no API changes. All existing CPU tests must continue to pass unmodified.

## Motivation

Velox's GPU SFI architecture reuses existing simple-function `call()` bodies by compiling them with `nvcc` through shadow headers and a `GpuExec` type resolver. This requires the function struct templates to be visible at include time. Currently, all decimal function structs are defined inside `namespace { ... }` in `.cpp` files, making them invisible to any other translation unit.

**Affected files (production only):**

| `.cpp` file | Struct count | Dialect |
|---|:-:|---|
| `velox/functions/prestosql/DecimalFunctions.cpp` | 9 | Presto |
| `velox/functions/sparksql/DecimalArithmetic.cpp` | 11 (incl. 2 bases) | Spark |
| `velox/functions/sparksql/DecimalCeil.cpp` | 1 | Spark |

**Total: 21 struct definitions across 3 files.**

## Plan Overview

| Phase | What | Risk |
|:-----:|------|:----:|
| 1 | PrestoSQL decimal structs -> `DecimalMathFunctions.h` | Low |
| 2 | SparkSQL decimal arithmetic structs -> `DecimalArithmeticFunctions.h` | Low |
| 3 | SparkSQL decimal ceil struct -> `DecimalCeilFunction.h` | Low |
| 4 | Validation | None |

Each phase is an independent, reviewable PR. Phases 1-3 can be parallelized.

---

## Phase 1: PrestoSQL `DecimalFunctions.cpp`

**Source**: `velox/functions/prestosql/DecimalFunctions.cpp` (lines 27-360)

**Create**: `velox/functions/prestosql/DecimalMathFunctions.h`

**Structs to move** (9):

| Struct | Has `initialize()` |
|---|:-:|
| `DecimalPlusFunction` | Yes |
| `DecimalMinusFunction` | Yes |
| `DecimalMultiplyFunction` | No |
| `DecimalDivideFunction` | Yes |
| `DecimalModulusFunction` | Yes |
| `DecimalRoundFunction` | Yes (2 overloads) |
| `DecimalFloorFunction` | Yes |
| `DecimalCeilFunction` | Yes |
| `DecimalTruncateFunction` | Yes (2 overloads) |

**New header structure:**
```cpp
// DecimalMathFunctions.h
#pragma once

#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/ArithmeticImpl.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions {

template <typename TExec>
struct DecimalPlusFunction { ... };
// ... all 9 structs, verbatim from .cpp ...

} // namespace facebook::velox::functions
```

**What stays in `DecimalFunctions.cpp`:**
- `#include "velox/functions/prestosql/DecimalMathFunctions.h"`
- Registration helpers (`registerDecimalBinary`, `registerDecimalPlusMinus`)
- All `registerDecimal*()` function definitions
- These remain in the anonymous namespace inside the `.cpp` since they are registration-only helpers that use `Registerer.h`, `VectorFunction.h`, and `DecodedArgs.h` (host-only, not needed on GPU)

**Key detail**: The structs must be placed in `namespace facebook::velox::functions` (not an anonymous namespace) so the new header is includable. The `.cpp` registration code already references the structs by unqualified name from the same namespace, so no call-site changes are needed.

**Dependencies carried into the header:**
- `velox/functions/Macros.h` (for `VELOX_DEFINE_FUNCTION_TYPES`)
- `velox/functions/prestosql/ArithmeticImpl.h` (for `checkedPlus`, `checkedMinus`, etc.)
- `velox/type/DecimalUtil.h` (for `kPowersOfTen`, `valueInRange`)

**No new dependencies introduced.**

---

## Phase 2: SparkSQL `DecimalArithmetic.cpp`

**Source**: `velox/functions/sparksql/DecimalArithmetic.cpp` (lines 26-647)

**Create**: `velox/functions/sparksql/DecimalArithmeticFunctions.h`

**Structs to move** (11, including 2 helper bases):

| Struct | Has `initialize()` | Notes |
|---|:-:|---|
| `DecimalAddSubtractBase` | `initializeBase()` | Protected helper base |
| `DecimalAddFunction` | Yes | Inherits `DecimalAddSubtractBase` |
| `DecimalSubtractFunction` | Yes | Inherits `DecimalAddSubtractBase` |
| `CheckedDecimalAddFunction` | Yes | Inherits `DecimalAddSubtractBase` |
| `CheckedDecimalSubtractFunction` | Yes | Inherits `DecimalAddSubtractBase` |
| `DecimalMultiplyFunction` | Yes | Standalone |
| `CheckedDecimalMultiplyFunction` | Yes | Inherits `DecimalMultiplyFunction` |
| `DecimalDivideFunction` | Yes | Standalone |
| `DecimalIntegralDivideBase` | `initializeBase()` | Protected helper base |
| `DecimalIntegralDivideFunction` | Yes | Inherits `DecimalIntegralDivideBase` |
| `CheckedDecimalIntegralDivideFunction` | Yes | Inherits `DecimalIntegralDivideBase` |

**New header structure:**
```cpp
// DecimalArithmeticFunctions.h
#pragma once

#include "velox/functions/Macros.h"
#include "velox/functions/sparksql/DecimalUtil.h"

namespace facebook::velox::functions::sparksql {

struct DecimalAddSubtractBase { ... };
template <typename TExec, bool allowPrecisionLoss>
struct DecimalAddFunction : DecimalAddSubtractBase { ... };
// ... all 11 structs, verbatim ...

} // namespace facebook::velox::functions::sparksql
```

**What stays in `DecimalArithmetic.cpp`:**
- `#include "velox/functions/sparksql/DecimalArithmeticFunctions.h"`
- `kDenyPrecisionLoss` constant
- Template aliases (`AddFunctionAllowPrecisionLoss`, etc.)
- Registration helpers (`registerDecimalBinary`, `bounded`, `makeConstraints`, etc.)
- All `registerDecimal*()` function definitions

**Dependencies carried into the header:**
- `velox/functions/Macros.h`
- `velox/functions/sparksql/DecimalUtil.h`

---

## Phase 3: SparkSQL `DecimalCeil.cpp`

**Source**: `velox/functions/sparksql/DecimalCeil.cpp` (lines 25-46)

**Create**: `velox/functions/sparksql/DecimalCeilFunction.h`

**Structs to move** (1):

| Struct | Has `initialize()` |
|---|:-:|
| `DecimalCeilFunction` | Yes |

**New header structure:**
```cpp
// DecimalCeilFunction.h
#pragma once

#include "velox/functions/Macros.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::sparksql {

template <typename TExec>
struct DecimalCeilFunction { ... };

} // namespace facebook::velox::functions::sparksql
```

**What stays in `DecimalCeil.cpp`:**
- `#include "velox/functions/sparksql/DecimalCeilFunction.h"`
- `registerDecimalCeil()` with signature constraints

**Dependencies carried into the header:**
- `velox/functions/Macros.h`
- `velox/type/DecimalUtil.h` (for `kPowersOfTen`, `getDecimalPrecisionScale`)

---

## Phase 4: Validation

- [ ] All existing decimal unit tests pass without modification:
  - `velox/functions/prestosql/tests/DecimalArithmeticTest`
  - `velox/functions/sparksql/tests/DecimalArithmeticTest`
- [ ] No new compiler warnings introduced
- [ ] TPC-H decimal queries produce identical results
- [ ] New headers compile cleanly when included from a standalone `.cpp`
- [ ] (GPU follow-up) New headers compile with `nvcc` when routed through `gpu_shadows/`

## Not In Scope

- Moving non-decimal functions (e.g., `Size.cpp`, `RegexFunctions.cpp`, `In.cpp`) -- these are either host-only or depend on complex types deferred to v2.
- Moving `VectorFunction`-based decimal comparisons (`sparksql/DecimalCompare.cpp`) -- these use a different adapter path (`GpuVectorFunction`), not the simple-function adapter.
- Creating shadow headers or GPU compilation units -- that is tracked in the GPU SFI Architecture plan.
- Moving registration helpers or signature constraint logic -- these use host-only APIs (`fmt`, `exec::SignatureVariable`) and must stay in `.cpp` files.
