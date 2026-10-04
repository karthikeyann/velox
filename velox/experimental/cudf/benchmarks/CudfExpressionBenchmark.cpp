/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Compares GPU SFI, which compiles Velox's own simple-function structs to
// CUDA, with every other evaluator that serves the same Presto expressions:
// the cuDF JIT and AST evaluators, the function tier of hand-written cuDF
// calls, and Velox's CPU evaluation. The expressions are grouped in families
// over one input row type each: TIMESTAMP fields, date_trunc, date arithmetic,
// TIMESTAMP WITH TIME ZONE, BIGINT and DOUBLE arithmetic.
//
// Each expression is compiled through createCudfExpression() once per GPU
// path with the evaluator registry restricted to that one evaluator, so the
// whole tree runs there or the compile fails and the path is reported as
// unavailable; the evaluator is confirmed again from the compiled object's
// type. Restricting the registry also lifts the session-time-zone gate, so an
// evaluator that reads TIMESTAMP as UTC is timed under a session zone even
// though its answer is then wrong; the comparison with the CPU reports that.
// The CPU path runs exec::ExprSet over the same vectors single-threaded and
// is the oracle every GPU result is compared with, value by value.
//
// Only evaluation is timed: the inputs are generated on the host, converted
// to cuDF once, and cudaDeviceSynchronize() brackets every GPU iteration.
// Each timed GPU evaluation sits in an NVTX range named
// "<path>|<family>|<expression>" so that a profiler can count its kernels.
//
// Usage inside the Velox CUDA container:
//   CUDA_VISIBLE_DEVICES=0 ./velox_cudf_expression_benchmark \
//       --num_rows 50000000 --session_timezone America/Los_Angeles

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluatorRegistry.h"
#include "velox/experimental/cudf/expression/JitExpression.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"
#include "velox/experimental/cudf/functions/GpuSfiErrors.h"
#include "velox/experimental/cudf/functions/GpuSfiExpression.h"
#include "velox/experimental/cudf/functions/GpuTimeZone.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/ExprOptimizer.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Timestamp.h"
#include "velox/type/Type.h"
#include "velox/type/tz/TimeZoneMap.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/SelectivityVector.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include <fmt/format.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

DEFINE_int64(num_rows, 50'000'000, "Rows in each input column.");
DEFINE_int32(iterations, 10, "Timed evaluations per expression and GPU path.");
DEFINE_int32(
    cpu_iterations,
    -1,
    "Timed CPU evaluations per expression; -1 uses --iterations and 0 "
    "evaluates once, for verification only.");
DEFINE_int32(warmup, 2, "Untimed evaluations before the timed ones.");
DEFINE_double(null_ratio, 0.0, "Fraction of null rows in every input column.");
DEFINE_uint64(seed, 42, "Seed for the input generator.");
DEFINE_string(
    families,
    "",
    "Comma-separated subset of {timestamp_fields, date_trunc, "
    "date_arithmetic, tswtz, bigint, double}; empty runs all.");
DEFINE_string(
    expressions,
    "",
    "Comma-separated subset of the built-in expression list; empty runs all.");
DEFINE_string(
    paths,
    "cpu,gpu_sfi,jit,ast,function",
    "Comma-separated subset of {cpu, gpu_sfi, jit, ast, function}. The CPU "
    "is evaluated once regardless, as the oracle; listing it times it too.");
DEFINE_bool(
    collect_errors,
    true,
    "Pass a GpuSfiErrors to eval() and resolve it afterwards, as "
    "CudfFilterProject does for every projection.");
DEFINE_bool(
    nullable_inputs,
    false,
    "Keep the null masks the Arrow conversion attaches to input columns that "
    "hold no nulls, as a batch converted from a CPU operator carries them; "
    "false drops them, as a column read by cuDF with no nulls has none.");
DEFINE_string(session_timezone, "UTC", "Session time zone.");
DEFINE_bool(
    adjust_timestamp_to_session_timezone,
    true,
    "Read TIMESTAMP in the session time zone, as Velox does by default.");
DEFINE_bool(
    legacy_timestamp_with_timezone,
    true,
    "Render TIMESTAMP WITH TIME ZONE in each value's own zone (true) or in "
    "the session zone (false).");

namespace facebook::velox::cudf_velox {
namespace {

// A family of expressions over one input row type. The expressions are DuckDB
// SQL; round's digits are cast because an integer literal parses as BIGINT
// while round(double, n) takes INTEGER.
struct Family {
  std::string name;
  std::string inputSet;
  std::vector<std::string> expressions;
};

const std::vector<Family>& families() {
  static const std::vector<Family> kFamilies = {
      {"timestamp_fields",
       "timestamps",
       {"year(ts)", "hour(ts)", "day_of_week(ts)", "second(ts)"}},
      {"date_trunc",
       "timestamps",
       {"date_trunc('day', ts)", "date_trunc('hour', ts)"}},
      {"date_arithmetic",
       "timestamps",
       {"date_add('day', 1, ts)",
        "date_diff('day', ts, ts2)",
        "ts + INTERVAL '1' DAY"}},
      {"tswtz",
       "tswtz",
       {"year(tz)",
        "hour(tz)",
        "timezone_hour(tz)",
        "tz = tz2",
        "date_trunc('day', tz)",
        "date_add('day', 1, tz)"}},
      {"bigint", "bigint", {"a + b", "a * b", "a < b", "a + b * c"}},
      {"double", "double", {"a + b", "a / b", "round(a, cast(2 as integer))"}},
  };
  return kFamilies;
}

// The input sets, in the order they are generated.
const std::vector<std::string>& inputSets() {
  static const std::vector<std::string> kInputSets = {
      "timestamps", "tswtz", "bigint", "double"};
  return kInputSets;
}

enum class Path { kCpu, kGpuSfi, kJit, kAst, kFunction };

const std::vector<Path>& gpuPaths() {
  static const std::vector<Path> kPaths = {
      Path::kGpuSfi, Path::kJit, Path::kAst, Path::kFunction};
  return kPaths;
}

const char* pathName(Path path) {
  switch (path) {
    case Path::kCpu:
      return "cpu";
    case Path::kGpuSfi:
      return kGpuSfiEvaluatorName;
    case Path::kJit:
      return "jit";
    case Path::kAst:
      return "ast";
    case Path::kFunction:
      return "function";
  }
  return "?";
}

// The registry key of a GPU path's evaluator.
std::string registryName(Path path) {
  switch (path) {
    case Path::kJit:
      return kJitEvaluatorName;
    case Path::kAst:
      return kAstEvaluatorName;
    default:
      return pathName(path);
  }
}

void checkCuda(cudaError_t status) {
  VELOX_CHECK_EQ(
      static_cast<int>(status), 0, "CUDA: {}", cudaGetErrorString(status));
}

std::vector<std::string> splitList(const std::string& list) {
  std::vector<std::string> items;
  std::stringstream stream(list);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      items.push_back(item);
    }
  }
  return items;
}

bool listed(const std::string& list, const std::string& item) {
  if (list.empty()) {
    return true;
  }
  const auto items = splitList(list);
  return std::find(items.begin(), items.end(), item) != items.end();
}

// SplitMix64: enough randomness for a bandwidth benchmark and far cheaper than
// a Mersenne twister over hundreds of millions of values.
uint64_t nextRandom(uint64_t& state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Instants span 1990-01-01 to 2035-01-01, which covers every daylight saving
// rule the zones below use today and the years a warehouse holds.
constexpr int64_t kFromSeconds = 631'152'000;
constexpr int64_t kToSeconds = 2'051'222'400;

// Zones a TIMESTAMP WITH TIME ZONE column mixes: fixed offsets, zones with
// and without daylight saving time, a 30-minute daylight shift and a
// 45-minute offset. None moves its clocks at midnight: date_trunc('day') of
// a value on such a day names a local time that does not exist, which the
// CPU raises for and the GPU declines, and would leave the case unmeasured.
const std::vector<std::string>& timeZoneNames() {
  static const std::vector<std::string> kNames = {
      "UTC",
      "America/Los_Angeles",
      "America/New_York",
      "Europe/London",
      "Europe/Berlin",
      "Europe/Moscow",
      "Asia/Kolkata",
      "Asia/Tokyo",
      "Asia/Kathmandu",
      "Australia/Lord_Howe",
      "+05:30",
      "-08:00",
  };
  return kNames;
}

void applyNulls(BaseVector& vector, uint64_t& state) {
  if (FLAGS_null_ratio <= 0.0) {
    return;
  }
  const auto threshold =
      static_cast<uint64_t>(FLAGS_null_ratio * static_cast<double>(UINT64_MAX));
  for (vector_size_t row = 0; row < vector.size(); ++row) {
    if (nextRandom(state) < threshold) {
      vector.setNull(row, true);
    }
  }
}

VectorPtr makeTimestampColumn(
    vector_size_t numRows,
    uint64_t seed,
    memory::MemoryPool* pool) {
  auto vector =
      BaseVector::create<FlatVector<Timestamp>>(TIMESTAMP(), numRows, pool);
  auto* raw = vector->mutableRawValues();
  uint64_t state = seed;
  const auto span = static_cast<uint64_t>(kToSeconds - kFromSeconds);
  for (vector_size_t row = 0; row < numRows; ++row) {
    const auto seconds =
        kFromSeconds + static_cast<int64_t>(nextRandom(state) % span);
    raw[row] = Timestamp(seconds, nextRandom(state) % 1'000'000'000);
  }
  applyNulls(*vector, state);
  return vector;
}

// A TIMESTAMP WITH TIME ZONE column over the zones above. When `sameInstantAs`
// is given, a quarter of the rows repeat its instant under their own zone, so
// that an equality over the two columns has both answers.
VectorPtr makeTimestampWithTimeZoneColumn(
    vector_size_t numRows,
    uint64_t seed,
    memory::MemoryPool* pool,
    const VectorPtr& sameInstantAs) {
  std::vector<int16_t> zoneIds;
  for (const auto& name : timeZoneNames()) {
    zoneIds.push_back(tz::getTimeZoneID(name));
  }
  auto vector = BaseVector::create<FlatVector<int64_t>>(
      TIMESTAMP_WITH_TIME_ZONE(), numRows, pool);
  auto* raw = vector->mutableRawValues();
  const int64_t* other = sameInstantAs == nullptr
      ? nullptr
      : sameInstantAs->as<FlatVector<int64_t>>()->rawValues();
  uint64_t state = seed;
  const auto spanMillis =
      static_cast<uint64_t>(kToSeconds - kFromSeconds) * 1'000;
  for (vector_size_t row = 0; row < numRows; ++row) {
    int64_t millis = kFromSeconds * 1'000 +
        static_cast<int64_t>(nextRandom(state) % spanMillis);
    if (other != nullptr && nextRandom(state) % 4 == 0) {
      millis = unpackMillisUtc(other[row]);
    }
    const auto zoneId = zoneIds[nextRandom(state) % zoneIds.size()];
    raw[row] = pack(millis, zoneId);
  }
  applyNulls(*vector, state);
  return vector;
}

// Values in [0, 1'000'000), so that a * b and a + b * c stay far from
// overflow: GPU SFI's checked arithmetic would decline an overflowing row.
VectorPtr makeBigintColumn(
    vector_size_t numRows,
    uint64_t seed,
    memory::MemoryPool* pool) {
  auto vector =
      BaseVector::create<FlatVector<int64_t>>(BIGINT(), numRows, pool);
  auto* raw = vector->mutableRawValues();
  uint64_t state = seed;
  for (vector_size_t row = 0; row < numRows; ++row) {
    raw[row] = static_cast<int64_t>(nextRandom(state) % 1'000'000);
  }
  applyNulls(*vector, state);
  return vector;
}

// `a` spans [-1'000'000, 1'000'000) with three decimals; `b` and `c` are
// fractions in [0.01, 0.99], so that every divisor is nonzero.
VectorPtr makeDoubleColumn(
    vector_size_t numRows,
    uint64_t seed,
    memory::MemoryPool* pool,
    bool wide) {
  auto vector = BaseVector::create<FlatVector<double>>(DOUBLE(), numRows, pool);
  auto* raw = vector->mutableRawValues();
  uint64_t state = seed;
  for (vector_size_t row = 0; row < numRows; ++row) {
    raw[row] = wide
        ? static_cast<double>(nextRandom(state) % 2'000'000'000) / 1'000.0 -
            1'000'000.0
        : static_cast<double>(1 + nextRandom(state) % 99) / 100.0;
  }
  applyNulls(*vector, state);
  return vector;
}

RowVectorPtr makeInput(
    const std::string& inputSet,
    vector_size_t numRows,
    memory::MemoryPool* pool) {
  const auto seed = FLAGS_seed;
  std::vector<std::string> names;
  std::vector<VectorPtr> children;
  if (inputSet == "timestamps") {
    names = {"ts", "ts2"};
    children = {
        makeTimestampColumn(numRows, seed, pool),
        makeTimestampColumn(numRows, seed + 1, pool)};
  } else if (inputSet == "tswtz") {
    names = {"tz", "tz2"};
    children.push_back(
        makeTimestampWithTimeZoneColumn(numRows, seed, pool, nullptr));
    children.push_back(makeTimestampWithTimeZoneColumn(
        numRows, seed + 1, pool, children.front()));
  } else if (inputSet == "bigint") {
    names = {"a", "b", "c"};
    children = {
        makeBigintColumn(numRows, seed, pool),
        makeBigintColumn(numRows, seed + 1, pool),
        makeBigintColumn(numRows, seed + 2, pool)};
  } else if (inputSet == "double") {
    names = {"a", "b", "c"};
    children = {
        makeDoubleColumn(numRows, seed, pool, /*wide=*/true),
        makeDoubleColumn(numRows, seed + 1, pool, /*wide=*/false),
        makeDoubleColumn(numRows, seed + 2, pool, /*wide=*/false)};
  } else {
    VELOX_FAIL("Unknown input set: {}", inputSet);
  }
  std::vector<TypePtr> types;
  for (const auto& child : children) {
    types.push_back(child->type());
  }
  return std::make_shared<RowVector>(
      pool,
      ROW(std::move(names), std::move(types)),
      nullptr,
      numRows,
      children);
}

// The one spelling the DuckDB parser cannot deliver: it turns INTERVAL '1' DAY
// into a to_days(cast(...)) call, so the tree a planner produces is built by
// hand, as FilterProjectTest does.
const std::string kTimestampPlusDay = "ts + INTERVAL '1' DAY";

core::TypedExprPtr compileSql(
    const std::string& sql,
    const RowTypePtr& rowType,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  if (sql == kTimestampPlusDay) {
    constexpr int64_t kMillisInDay = 86'400'000;
    auto plus = std::make_shared<core::CallTypedExpr>(
        TIMESTAMP(),
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(TIMESTAMP(), "ts"),
            std::make_shared<core::ConstantTypedExpr>(
                INTERVAL_DAY_TIME(), Variant(kMillisInDay)),
        },
        "plus");
    return expression::optimize(plus, queryCtx, pool);
  }
  auto untyped = parse::DuckSqlExpressionsParser().parseExpr(sql);
  auto typed = core::Expressions::inferTypes(untyped, rowType, pool);
  return expression::optimize(typed, queryCtx, pool);
}

// The line of a Velox exception message that states the reason.
std::string reasonOf(const std::exception& e) {
  const std::string what = e.what();
  const auto reason = what.find("Reason: ");
  const auto start = reason == std::string::npos ? 0 : reason + 8;
  return what.substr(start, what.find('\n', start) - start);
}

// Restricts the evaluator registry to one evaluator for its lifetime, so that
// createCudfExpression() compiles the whole tree to it or throws. Also marks
// it as honoring the session time zone, so that the selection offers it a
// zone-sensitive call even when it would read TIMESTAMP as UTC: the point is
// to time the existing path, and the comparison with the CPU reports whether
// its answer holds.
class OnlyEvaluator {
 public:
  explicit OnlyEvaluator(const std::string& name)
      : registry_(
            (ensureBuiltinExpressionEvaluatorsRegistered(),
             getCudfExpressionEvaluatorRegistry())),
        saved_(registry_) {
    for (auto it = registry_.begin(); it != registry_.end();) {
      if (it->first == name) {
        it->second.honorsSessionTimeZone = true;
        ++it;
      } else {
        it = registry_.erase(it);
      }
    }
  }

  ~OnlyEvaluator() {
    registry_ = saved_;
  }

 private:
  std::unordered_map<std::string, CudfExpressionEvaluatorEntry>& registry_;
  const std::unordered_map<std::string, CudfExpressionEvaluatorEntry> saved_;
};

// Names the evaluator from the compiled object's concrete type.
std::string evaluatorName(const CudfExpression& expression) {
  if (dynamic_cast<const GpuSfiExpression*>(&expression) != nullptr) {
    return kGpuSfiEvaluatorName;
  }
  if (dynamic_cast<const JitExpression*>(&expression) != nullptr) {
    return kJitEvaluatorName;
  }
  if (dynamic_cast<const ASTExpression*>(&expression) != nullptr) {
    return kAstEvaluatorName;
  }
  if (dynamic_cast<const FunctionExpression*>(&expression) != nullptr) {
    return "function";
  }
  return "other";
}

// Lists, for every call node, which evaluators claim it and whether GPU SFI
// reports its result to depend on the session time zone.
void describeClaims(
    const core::TypedExprPtr& expr,
    std::vector<std::string>& lines) {
  if (expr->isCallKind()) {
    lines.push_back(
        fmt::format(
            "    {} -> gpu_sfi={} jit={} ast={} function={} zone_sensitive={}",
            expr->toString(),
            GpuSfiExpression::canEvaluate(expr),
            JitExpression::canEvaluate(expr),
            ASTExpression::canEvaluate(expr),
            FunctionExpression::canEvaluate(expr),
            GpuSfiExpression::dependsOnSessionTimeZone(expr)));
  }
  for (const auto& input : expr->inputs()) {
    describeClaims(input, lines);
  }
}

struct Timing {
  double medianMs{0};
  double p10Ms{0};
  double p90Ms{0};
};

Timing summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  auto at = [&](double quantile) {
    const auto index = static_cast<size_t>(quantile * (samples.size() - 1));
    return samples[index];
  };
  return Timing{at(0.5), at(0.1), at(0.9)};
}

double millisSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

struct CaseResult {
  bool ran{false};
  std::string note;
  std::string evaluator;
  // The first warm-up evaluation, which for JIT includes the kernel compile.
  double firstMs{0};
  Timing timing;
  std::unique_ptr<cudf::column> gpuOutput;
  VectorPtr cpuOutput;
  std::string comparison;
};

// Evaluates on the CPU: once for the oracle, then `iterations` timed runs.
CaseResult runCpuCase(
    const core::TypedExprPtr& typedExpr,
    const RowVectorPtr& input,
    core::ExecCtx& execCtx,
    int iterations) {
  CaseResult result;
  result.evaluator = "ExprSet";
  try {
    exec::ExprSet exprSet({typedExpr}, &execCtx);
    SelectivityVector rows(input->size());
    auto evaluateOnce = [&]() {
      exec::EvalCtx evalCtx(&execCtx, &exprSet, input.get());
      std::vector<VectorPtr> results(1);
      exprSet.eval(rows, evalCtx, results);
      return results[0];
    };
    const auto first = std::chrono::steady_clock::now();
    result.cpuOutput = evaluateOnce();
    result.firstMs = millisSince(first);
    if (iterations > 0) {
      std::vector<double> samples;
      for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        auto output = evaluateOnce();
        samples.push_back(millisSince(start));
      }
      result.timing = summarize(samples);
      result.ran = true;
    }
  } catch (const std::exception& e) {
    result.note = fmt::format("threw: {}", reasonOf(e));
  }
  return result;
}

CaseResult runGpuCase(
    Path path,
    const core::TypedExprPtr& typedExpr,
    const RowTypePtr& rowType,
    const std::vector<cudf::column_view>& inputViews,
    memory::MemoryPool* pool,
    const core::QueryConfig& config,
    const std::string& label) {
  CaseResult result;
  std::shared_ptr<CudfExpression> cudfExpr;
  {
    OnlyEvaluator only(registryName(path));
    // The root's claim is checked first so that an evaluator that declines the
    // call is reported as such without a failed VELOX_CHECK in the log; a
    // nested node it declines still surfaces as a compile failure.
    const auto& entry =
        getCudfExpressionEvaluatorRegistry().at(registryName(path));
    if (!entry.canEvaluate(typedExpr)) {
      result.note = "not claimed";
      return result;
    }
    try {
      cudfExpr = createCudfExpression(typedExpr, rowType, pool, config);
    } catch (const std::exception& e) {
      result.note = fmt::format("not available: {}", reasonOf(e));
      return result;
    }
  }
  result.evaluator = evaluatorName(*cudfExpr);
  if (result.evaluator != registryName(path)) {
    result.note = fmt::format("root compiled by {}", result.evaluator);
    cudfExpr->close();
    return result;
  }

  auto stream = getDefaultStreamForCurrentThread();
  auto outputMr = get_output_mr();
  auto tempMr = rmm::mr::get_current_device_resource_ref();

  auto evaluateOnce = [&]() -> std::unique_ptr<cudf::column> {
    gpu_sfi::GpuSfiErrors errors(stream, tempMr);
    auto evaluated = cudfExpr->eval(
        inputViews,
        stream,
        outputMr,
        /*finalize=*/true,
        FLAGS_collect_errors ? &errors : nullptr);
    if (FLAGS_collect_errors &&
        errors.resolve() != gpu_sfi::ErrorClass::kNone) {
      VELOX_FAIL("The GPU declined a row: {}", label);
    }
    if (std::holds_alternative<std::unique_ptr<cudf::column>>(evaluated)) {
      return std::move(std::get<std::unique_ptr<cudf::column>>(evaluated));
    }
    return std::make_unique<cudf::column>(
        std::get<cudf::column_view>(evaluated), stream, outputMr);
  };

  try {
    for (int i = 0; i < FLAGS_warmup; ++i) {
      checkCuda(cudaDeviceSynchronize());
      const auto start = std::chrono::steady_clock::now();
      evaluateOnce();
      checkCuda(cudaDeviceSynchronize());
      if (i == 0) {
        result.firstMs = millisSince(start);
      }
    }
    std::vector<double> samples;
    samples.reserve(FLAGS_iterations);
    for (int i = 0; i < FLAGS_iterations; ++i) {
      result.gpuOutput.reset();
      checkCuda(cudaDeviceSynchronize());
      const auto start = std::chrono::steady_clock::now();
      {
        nvtx3::scoped_range range{label.c_str()};
        result.gpuOutput = evaluateOnce();
        checkCuda(cudaDeviceSynchronize());
      }
      samples.push_back(millisSince(start));
    }
    result.timing = summarize(samples);
    result.ran = true;
  } catch (const std::exception& e) {
    result.note = fmt::format("threw: {}", reasonOf(e));
    result.gpuOutput.reset();
  }
  cudfExpr->close();
  return result;
}

// Distance in representable values between two floating-point numbers; zero
// for equal values and for two NaNs.
template <typename T>
int64_t ulpDistance(T left, T right) {
  if (left == right) {
    return 0;
  }
  if (std::isnan(left) || std::isnan(right)) {
    return std::isnan(left) && std::isnan(right)
        ? 0
        : std::numeric_limits<int64_t>::max();
  }
  using Bits = std::conditional_t<sizeof(T) == 8, int64_t, int32_t>;
  auto ordered = [](T value) {
    Bits bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits < 0 ? std::numeric_limits<Bits>::min() - bits : bits;
  };
  const __int128 distance =
      static_cast<__int128>(ordered(left)) - ordered(right);
  const __int128 magnitude = distance < 0 ? -distance : distance;
  return magnitude > std::numeric_limits<int64_t>::max()
      ? std::numeric_limits<int64_t>::max()
      : static_cast<int64_t>(magnitude);
}

struct Comparison {
  int64_t differingValues{0};
  int64_t differingNulls{0};
  int64_t maxUlp{0};
  vector_size_t firstRow{-1};
};

template <TypeKind Kind>
Comparison compareTyped(
    const DecodedVector& cpu,
    const DecodedVector& gpu,
    vector_size_t size) {
  using T = typename TypeTraits<Kind>::NativeType;
  Comparison comparison;
  auto noteRow = [&](vector_size_t row) {
    if (comparison.firstRow < 0) {
      comparison.firstRow = row;
    }
  };
  for (vector_size_t row = 0; row < size; ++row) {
    const bool cpuNull = cpu.isNullAt(row);
    const bool gpuNull = gpu.isNullAt(row);
    if (cpuNull || gpuNull) {
      if (cpuNull != gpuNull) {
        ++comparison.differingNulls;
        noteRow(row);
      }
      continue;
    }
    const T left = cpu.valueAt<T>(row);
    const T right = gpu.valueAt<T>(row);
    if constexpr (std::is_floating_point_v<T>) {
      const auto ulp = ulpDistance(left, right);
      if (ulp != 0) {
        ++comparison.differingValues;
        comparison.maxUlp = std::max(comparison.maxUlp, ulp);
        noteRow(row);
      }
    } else {
      if (!(left == right)) {
        ++comparison.differingValues;
        noteRow(row);
      }
    }
  }
  return comparison;
}

// Compares a GPU result with the CPU's, value by value, after bringing it back
// into a Velox vector of the expression's type. Integers, booleans and
// timestamps have to agree bit for bit; doubles are reported with the largest
// distance in representable values.
std::string compareWithCpu(
    const VectorPtr& cpuOutput,
    const cudf::column_view& gpuOutput,
    const TypePtr& type,
    memory::MemoryPool* pool,
    cuda::stream_ref stream) {
  if (cpuOutput == nullptr) {
    return "no CPU result";
  }
  auto outputType = ROW({"c0"}, {type});
  auto converted = with_arrow::toVeloxColumn(
      cudf::table_view({gpuOutput}),
      pool,
      outputType,
      "",
      stream,
      cudf::get_current_device_resource_ref());
  converted->setType(outputType);
  const auto gpuVector = converted->childAt(0);
  if (gpuVector->size() != cpuOutput->size()) {
    return fmt::format(
        "SIZE DIFFERS ({} vs {})", cpuOutput->size(), gpuVector->size());
  }
  const auto size = cpuOutput->size();
  DecodedVector cpu(*cpuOutput);
  DecodedVector gpu(*gpuVector);
  const auto comparison = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      compareTyped, type->kind(), cpu, gpu, size);
  if (comparison.differingNulls == 0 && comparison.differingValues == 0) {
    return "identical";
  }
  const bool floating =
      type->kind() == TypeKind::DOUBLE || type->kind() == TypeKind::REAL;
  if (floating && comparison.differingNulls == 0 && comparison.maxUlp <= 1) {
    return fmt::format("within 1 ulp ({} rows)", comparison.differingValues);
  }
  return fmt::format(
      "DIFFERS: {} values, {} null flags{}; first row {}: cpu={} gpu={}",
      comparison.differingValues,
      comparison.differingNulls,
      floating ? fmt::format(", max {} ulp", comparison.maxUlp) : "",
      comparison.firstRow,
      cpuOutput->toString(comparison.firstRow),
      gpuVector->toString(comparison.firstRow));
}

struct Row {
  std::string family;
  std::string expression;
  std::string resultType;
  std::map<Path, CaseResult> cases;
};

std::string configurationTag() {
  return fmt::format(
      "rows={} nulls={} masks={} collect={} tz={} adjust={} legacy={}",
      FLAGS_num_rows,
      FLAGS_null_ratio,
      FLAGS_nullable_inputs,
      FLAGS_collect_errors,
      FLAGS_session_timezone.empty() ? "none" : FLAGS_session_timezone,
      FLAGS_adjust_timestamp_to_session_timezone,
      FLAGS_legacy_timestamp_with_timezone);
}

void printSummary(const std::vector<Row>& rows, vector_size_t numRows) {
  std::cout << "\n== Summary (" << configurationTag()
            << "; median ms; 'vs gpu' = fastest of jit/ast/function over "
               "gpu_sfi, 'vs cpu' = cpu over gpu_sfi; >1 means GPU SFI is "
               "faster)\n";
  std::cout << fmt::format(
      "{:16} | {:30} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>14} | {:>8} | {:14} | {}\n",
      "family",
      "expression",
      "sfi ms",
      "jit ms",
      "ast ms",
      "fn ms",
      "cpu ms",
      "vs gpu",
      "vs cpu",
      "sfi==cpu",
      "gpu baseline==cpu");
  auto cell = [](const std::map<Path, CaseResult>& cases, Path path) {
    const auto it = cases.find(path);
    return it != cases.end() && it->second.ran
        ? fmt::format("{:.3f}", it->second.timing.medianMs)
        : std::string("-");
  };
  for (const auto& row : rows) {
    const auto sfi = row.cases.find(Path::kGpuSfi);
    const bool sfiRan = sfi != row.cases.end() && sfi->second.ran;
    const CaseResult* best = nullptr;
    Path bestPath = Path::kCpu;
    for (const auto path : {Path::kJit, Path::kAst, Path::kFunction}) {
      const auto it = row.cases.find(path);
      if (it != row.cases.end() && it->second.ran &&
          (best == nullptr ||
           it->second.timing.medianMs < best->timing.medianMs)) {
        best = &it->second;
        bestPath = path;
      }
    }
    const auto cpu = row.cases.find(Path::kCpu);
    const bool cpuRan = cpu != row.cases.end() && cpu->second.ran;
    const std::string vsGpu = sfiRan && best != nullptr
        ? fmt::format(
              "{:.2f}x ({})",
              best->timing.medianMs / sfi->second.timing.medianMs,
              pathName(bestPath))
        : (best == nullptr ? "no gpu path" : "-");
    const std::string vsCpu = sfiRan && cpuRan
        ? fmt::format(
              "{:.1f}x",
              cpu->second.timing.medianMs / sfi->second.timing.medianMs)
        : "-";
    std::cout << fmt::format(
        "{:16} | {:30} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>14} | {:>8} | {:14} | {}\n",
        row.family,
        row.expression,
        cell(row.cases, Path::kGpuSfi),
        cell(row.cases, Path::kJit),
        cell(row.cases, Path::kAst),
        cell(row.cases, Path::kFunction),
        cell(row.cases, Path::kCpu),
        vsGpu,
        vsCpu,
        sfiRan ? sfi->second.comparison
               : (sfi != row.cases.end() ? sfi->second.note : "-"),
        best != nullptr
            ? fmt::format("{}: {}", pathName(bestPath), best->comparison)
            : "-");
  }

  // One line per (expression, path) for a script to collect across runs.
  std::cout << "\n== CSV\n";
  for (const auto& row : rows) {
    for (const auto& [path, result] : row.cases) {
      std::cout << fmt::format(
          "CSV|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}\n",
          configurationTag(),
          row.family,
          row.expression,
          row.resultType,
          pathName(path),
          result.evaluator,
          result.ran ? fmt::format("{:.4f}", result.timing.medianMs) : "-",
          result.ran ? fmt::format("{:.4f}", result.timing.p10Ms) : "-",
          result.ran ? fmt::format("{:.4f}", result.timing.p90Ms) : "-",
          result.ran ? fmt::format(
                           "{:.1f}", numRows / result.timing.medianMs / 1'000.0)
                     : "-",
          result.comparison.empty() ? result.note : result.comparison,
          fmt::format("{:.3f}", result.firstMs));
    }
  }
}

int run() {
  // Keep every line visible even if a later case aborts the process.
  std::cout << std::unitbuf;
  int deviceCount = 0;
  checkCuda(cudaGetDeviceCount(&deviceCount));
  VELOX_CHECK_GT(deviceCount, 0, "No CUDA device visible");
  checkCuda(cudaSetDevice(0));
  checkCuda(cudaFree(nullptr));

  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  auto pool = memory::memoryManager()->addLeafPool("expression_benchmark");
  functions::prestosql::registerAllScalarFunctions();
  parse::registerTypeResolver();
  CudfConfig::getInstance().allowCpuFallback = false;
  registerCudf();
  // registerCudf() registers neither the function tier's Presto set, which
  // holds date_trunc, date_add and date_diff, nor the "function" evaluator
  // entry itself; the first comes from the application and the second from
  // the first compile. Both are needed before the registry is snapshotted.
  registerPrestoFunctions(CudfConfig::getInstance().functionNamePrefix);
  ensureBuiltinExpressionEvaluatorsRegistered();

  auto queryCtx = core::QueryCtx::create();
  queryCtx->testingOverrideConfigUnsafe({
      {core::QueryConfig::kSessionTimezone, FLAGS_session_timezone},
      {core::QueryConfig::kAdjustTimestampToTimezone,
       FLAGS_adjust_timestamp_to_session_timezone ? "true" : "false"},
      {core::QueryConfig::kLegacyTimestampWithTimezone,
       FLAGS_legacy_timestamp_with_timezone ? "true" : "false"},
  });
  core::ExecCtx execCtx(pool.get(), queryCtx.get());
  const auto& config = queryCtx->queryConfig();
  const int cpuIterations =
      FLAGS_cpu_iterations < 0 ? FLAGS_iterations : FLAGS_cpu_iterations;

  const auto& registry = getCudfExpressionEvaluatorRegistry();
  std::cout << "Evaluator priorities:";
  for (const auto& [name, entry] : registry) {
    std::cout << " " << name << "=" << entry.priority
              << (entry.honorsSessionTimeZone ? "(zone)" : "");
  }
  std::cout << "\n"
            << configurationTag() << ", iterations: " << FLAGS_iterations
            << ", cpu iterations: " << cpuIterations
            << ", session zone applies to TIMESTAMP: "
            << (FLAGS_adjust_timestamp_to_session_timezone &&
                !FLAGS_session_timezone.empty())
            << "\n";

  // The size of each zone's device table, which bounds the binary search a
  // conversion through that zone performs per row.
  std::cout << "Device zone tables (transitions):";
  std::vector<std::string> zoneNames = timeZoneNames();
  if (!FLAGS_session_timezone.empty()) {
    zoneNames.insert(zoneNames.begin(), FLAGS_session_timezone);
  }
  for (const auto& name : zoneNames) {
    std::cout << " " << name << "="
              << gpu_sfi::gpuTimeZone(*tz::locateZone(name)).numTransitions;
  }
  std::cout << "\n";

  const auto numRows = static_cast<vector_size_t>(FLAGS_num_rows);
  auto stream = getDefaultStreamForCurrentThread();
  auto tempMr = rmm::mr::get_current_device_resource_ref();
  std::vector<Row> rows;

  for (const auto& inputSet : inputSets()) {
    std::vector<const Family*> selected;
    for (const auto& family : families()) {
      if (family.inputSet == inputSet && listed(FLAGS_families, family.name)) {
        selected.push_back(&family);
      }
    }
    if (selected.empty()) {
      continue;
    }

    std::cout << "\n== input set " << inputSet << ": generating " << numRows
              << " rows\n";
    auto generating = std::chrono::steady_clock::now();
    auto input = makeInput(inputSet, numRows, pool.get());
    std::cout << "   generated in "
              << fmt::format("{:.0f}", millisSince(generating))
              << " ms: " << input->type()->toString() << "\n";
    auto converting = std::chrono::steady_clock::now();
    auto columns =
        with_arrow::toCudfTable(input, pool.get(), stream, tempMr)->release();
    stream.sync();
    std::cout << "   converted to cuDF in "
              << fmt::format("{:.0f}", millisSince(converting)) << " ms\n";
    // The conversion attaches a null mask to every column. A mask with no
    // null set still makes every evaluator take its nullable path, so it is
    // dropped unless asked for; a column read by cuDF with no nulls has none.
    for (std::size_t i = 0; i < columns.size(); ++i) {
      auto& column = columns[i];
      std::cout << fmt::format(
          "   {}: cudf type {} nullable={} nulls={}",
          input->rowType()->nameOf(i),
          static_cast<int>(column->type().id()),
          column->nullable(),
          column->null_count());
      if (!FLAGS_nullable_inputs && column->nullable() &&
          column->null_count() == 0) {
        column->set_null_mask(rmm::device_buffer{}, 0);
        std::cout << " -> mask dropped";
      }
      std::cout << "\n";
    }
    std::vector<cudf::column_view> inputViews;
    for (const auto& column : columns) {
      inputViews.push_back(column->view());
    }
    const auto rowType = asRowType(input->type());

    for (const auto* family : selected) {
      for (const auto& sql : family->expressions) {
        if (!listed(FLAGS_expressions, sql)) {
          continue;
        }
        auto typedExpr = compileSql(sql, rowType, queryCtx.get(), pool.get());
        Row row{family->name, sql, typedExpr->type()->toString(), {}};
        std::cout << "\n-- [" << family->name << "] " << sql << "  =>  "
                  << typedExpr->toString() << " : " << row.resultType << "\n";
        std::vector<std::string> claims;
        describeClaims(typedExpr, claims);
        for (const auto& line : claims) {
          std::cout << line << "\n";
        }

        auto cpuCase = runCpuCase(
            typedExpr,
            input,
            execCtx,
            listed(FLAGS_paths, "cpu") ? cpuIterations : 0);
        if (cpuCase.ran) {
          std::cout << fmt::format(
              "    {:8} evaluator={:8} median {:9.3f} ms  p10 {:9.3f}  p90 {:9.3f}  {:8.1f} Mrows/s\n",
              "cpu",
              cpuCase.evaluator,
              cpuCase.timing.medianMs,
              cpuCase.timing.p10Ms,
              cpuCase.timing.p90Ms,
              numRows / cpuCase.timing.medianMs / 1'000.0);
        } else if (!cpuCase.note.empty()) {
          std::cout << fmt::format(
              "    {:8} SKIPPED: {}\n", "cpu", cpuCase.note);
        } else {
          std::cout << fmt::format(
              "    {:8} evaluated once for verification ({:.1f} ms)\n",
              "cpu",
              cpuCase.firstMs);
        }

        for (const auto path : gpuPaths()) {
          if (!listed(FLAGS_paths, pathName(path))) {
            continue;
          }
          const auto label =
              fmt::format("{}|{}|{}", pathName(path), family->name, sql);
          auto caseResult = runGpuCase(
              path, typedExpr, rowType, inputViews, pool.get(), config, label);
          if (caseResult.ran) {
            caseResult.comparison = compareWithCpu(
                cpuCase.cpuOutput,
                caseResult.gpuOutput->view(),
                typedExpr->type(),
                pool.get(),
                stream);
            std::cout << fmt::format(
                "    {:8} evaluator={:8} median {:9.3f} ms  p10 {:9.3f}  p90 {:9.3f}  {:8.1f} Mrows/s  first {:8.3f} ms  vs cpu: {}\n",
                pathName(path),
                caseResult.evaluator,
                caseResult.timing.medianMs,
                caseResult.timing.p10Ms,
                caseResult.timing.p90Ms,
                numRows / caseResult.timing.medianMs / 1'000.0,
                caseResult.firstMs,
                caseResult.comparison);
          } else {
            std::cout << fmt::format(
                "    {:8} SKIPPED: {}\n", pathName(path), caseResult.note);
          }
          caseResult.gpuOutput.reset();
          row.cases.emplace(path, std::move(caseResult));
        }
        cpuCase.cpuOutput.reset();
        row.cases.emplace(Path::kCpu, std::move(cpuCase));
        rows.push_back(std::move(row));
      }
    }
    columns.clear();
    input.reset();
  }

  printSummary(rows, numRows);
  unregisterCudf();
  return 0;
}

} // namespace
} // namespace facebook::velox::cudf_velox

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv, false};
  return facebook::velox::cudf_velox::run();
}
