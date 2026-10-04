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

// Compares the two GPU evaluators that serve Presto decimal arithmetic:
// GpuSfiExpression, which compiles Velox's own simple-function structs to
// CUDA, and FunctionExpression, which calls the hand-written kernels in
// DecimalExpressionKernels. Each expression is compiled through
// createCudfExpression() twice, once with GPU SFI at its registered priority
// and once with it demoted below the function tier, so the same TypedExpr takes
// each path. The evaluator is confirmed from the compiled object's type, not
// from the priorities. Only expression evaluation is timed: the inputs are on
// the device before the clock starts and cudaDeviceSynchronize() brackets every
// iteration. Each timed evaluation sits in an NVTX range named
// "<path>|<type>|<expression>" so that a profiler can count its kernels.
//
// Usage inside the Velox CUDA container:
//   CUDA_VISIBLE_DEVICES=1 ./velox_cudf_decimal_expression_benchmark \
//       --num_rows 50000000 --iterations 10

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluatorRegistry.h"
#include "velox/experimental/cudf/functions/GpuSfiErrors.h"
#include "velox/experimental/cudf/functions/GpuSfiExpression.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/ExprOptimizer.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include <fmt/format.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

DEFINE_int64(num_rows, 50'000'000, "Rows in each input column.");
DEFINE_int32(iterations, 10, "Timed evaluations per expression and path.");
DEFINE_int32(warmup, 2, "Untimed evaluations before the timed ones.");
DEFINE_double(null_ratio, 0.0, "Fraction of null rows in every input column.");
DEFINE_uint64(seed, 42, "Seed for the input generator.");
DEFINE_string(
    expressions,
    "",
    "Comma-separated subset of the built-in expression list; empty runs all.");
DEFINE_string(types, "short,long", "Comma-separated subset of {short, long}.");
DEFINE_string(
    paths,
    "gpu_sfi,function",
    "Comma-separated subset of {gpu_sfi, function}.");
DEFINE_bool(
    collect_errors,
    true,
    "Pass a GpuSfiErrors to eval() and resolve it afterwards, as "
    "CudfFilterProject does for every projection.");

namespace facebook::velox::cudf_velox {
namespace {

// The expressions under test. `1.0` parses as DECIMAL(2,1), as in TPC-H Q1, and
// the round digits are cast because an integer literal parses as BIGINT while
// round(decimal, n) takes INTEGER.
const std::vector<std::string> kExpressions = {
    "a + b",
    "a - b",
    "a * b",
    "a / b",
    "a % b",
    "a * (1.0 - b)",
    "a * (1.0 - b) * (1.0 + c)",
    "round(a, cast(1 as integer))",
};

enum class Path { kGpuSfi, kFunction };

const char* pathName(Path path) {
  return path == Path::kGpuSfi ? kGpuSfiEvaluatorName : "function";
}

// One decimal width with the unscaled ranges that keep every expression above
// inside its result precision and every divisor nonzero. `a` spans
// [-aMagnitude, aMagnitude); `b` and `c` span [bMin, bMax], a fraction in
// (0, 1) so that the Q1 factors 1 - b and 1 + c stay positive.
struct DecimalWidth {
  std::string name;
  TypePtr veloxType;
  cudf::data_type cudfType;
  uint64_t aMagnitude;
  uint64_t bMin;
  uint64_t bMax;
};

std::vector<DecimalWidth> decimalWidths() {
  return {
      {"short",
       DECIMAL(10, 2),
       cudf::data_type{cudf::type_id::DECIMAL64, -2},
       100'000'000, // 1'000'000.00
       1, // 0.01
       99}, // 0.99
      {"long",
       DECIMAL(30, 10),
       cudf::data_type{cudf::type_id::DECIMAL128, -10},
       100'000'000'000'000, // 10'000.0000000000
       100'000'000, // 0.0100000000
       9'900'000'000}, // 0.9900000000
  };
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
// a Mersenne twister over 150M values.
uint64_t nextRandom(uint64_t& state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Builds one decimal column on the device from host-generated unscaled values
// uniform in [low, high).
template <typename Rep>
std::unique_ptr<cudf::column> makeDecimalColumn(
    cudf::data_type type,
    cudf::size_type numRows,
    int64_t low,
    int64_t high,
    uint64_t seed,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<Rep> values(numRows);
  uint64_t state = seed;
  const auto span = static_cast<uint64_t>(high - low);
  for (auto& value : values) {
    value =
        static_cast<Rep>(low + static_cast<int64_t>(nextRandom(state) % span));
  }
  rmm::device_buffer data(
      values.data(), values.size() * sizeof(Rep), stream, mr);

  rmm::device_buffer mask;
  cudf::size_type nullCount = 0;
  if (FLAGS_null_ratio > 0.0) {
    std::vector<cudf::bitmask_type> bits(
        cudf::bitmask_allocation_size_bytes(numRows) /
            sizeof(cudf::bitmask_type),
        ~cudf::bitmask_type{0});
    const auto threshold = static_cast<uint64_t>(
        FLAGS_null_ratio * static_cast<double>(UINT64_MAX));
    for (cudf::size_type row = 0; row < numRows; ++row) {
      if (nextRandom(state) < threshold) {
        cudf::clear_bit_unsafe(bits.data(), row);
        ++nullCount;
      }
    }
    mask = rmm::device_buffer(
        bits.data(), bits.size() * sizeof(cudf::bitmask_type), stream, mr);
  }
  stream.sync();
  return std::make_unique<cudf::column>(
      type, numRows, std::move(data), std::move(mask), nullCount);
}

std::vector<std::unique_ptr<cudf::column>> makeInputs(
    const DecimalWidth& width,
    cudf::size_type numRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const auto aMagnitude = static_cast<int64_t>(width.aMagnitude);
  const auto bMin = static_cast<int64_t>(width.bMin);
  const auto bMax = static_cast<int64_t>(width.bMax);
  std::vector<std::unique_ptr<cudf::column>> columns;
  if (width.cudfType.id() == cudf::type_id::DECIMAL64) {
    columns.push_back(
        makeDecimalColumn<int64_t>(
            width.cudfType,
            numRows,
            -aMagnitude,
            aMagnitude,
            FLAGS_seed,
            stream,
            mr));
    columns.push_back(
        makeDecimalColumn<int64_t>(
            width.cudfType,
            numRows,
            bMin,
            bMax + 1,
            FLAGS_seed + 1,
            stream,
            mr));
    columns.push_back(
        makeDecimalColumn<int64_t>(
            width.cudfType,
            numRows,
            bMin,
            bMax + 1,
            FLAGS_seed + 2,
            stream,
            mr));
  } else {
    columns.push_back(
        makeDecimalColumn<__int128_t>(
            width.cudfType,
            numRows,
            -aMagnitude,
            aMagnitude,
            FLAGS_seed,
            stream,
            mr));
    columns.push_back(
        makeDecimalColumn<__int128_t>(
            width.cudfType,
            numRows,
            bMin,
            bMax + 1,
            FLAGS_seed + 1,
            stream,
            mr));
    columns.push_back(
        makeDecimalColumn<__int128_t>(
            width.cudfType,
            numRows,
            bMin,
            bMax + 1,
            FLAGS_seed + 2,
            stream,
            mr));
  }
  return columns;
}

core::TypedExprPtr compileSql(
    const std::string& sql,
    const RowTypePtr& rowType,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  // The parser reads `1.0` as DOUBLE unless told otherwise; the TPC-H literal
  // is DECIMAL(2,1).
  parse::ParseOptions options;
  options.parseDecimalAsDouble = false;
  auto untyped = parse::DuckSqlExpressionsParser(options).parseExpr(sql);
  auto typed = core::Expressions::inferTypes(untyped, rowType, pool);
  return expression::optimize(typed, queryCtx, pool);
}

// Moves GPU SFI below the function tier for its lifetime, so the next
// createCudfExpression() compiles the same tree to the hand-written kernels.
class DemoteGpuSfi {
 public:
  explicit DemoteGpuSfi(bool demote)
      : registry_(getCudfExpressionEvaluatorRegistry()),
        saved_(registry_.at(kGpuSfiEvaluatorName)) {
    if (demote) {
      registry_.at(kGpuSfiEvaluatorName).priority = 0;
    }
  }

  ~DemoteGpuSfi() {
    registry_.at(kGpuSfiEvaluatorName) = saved_;
  }

 private:
  std::unordered_map<std::string, CudfExpressionEvaluatorEntry>& registry_;
  const CudfExpressionEvaluatorEntry saved_;
};

// Names the evaluator from the compiled object's concrete type.
std::string evaluatorName(const CudfExpression& expression) {
  if (dynamic_cast<const GpuSfiExpression*>(&expression) != nullptr) {
    return kGpuSfiEvaluatorName;
  }
  if (dynamic_cast<const FunctionExpression*>(&expression) != nullptr) {
    return "function";
  }
  return "other";
}

// Lists, for every call node, which of the two evaluators claims it. Nested
// nodes pick their own evaluator by the same priorities, so with AST declining
// decimals a node claimed by the expected tier runs there.
void describeClaims(
    const core::TypedExprPtr& expr,
    std::vector<std::string>& lines) {
  if (expr->isCallKind()) {
    lines.push_back(
        fmt::format(
            "    {} -> gpu_sfi={} function={}",
            expr->toString(),
            GpuSfiExpression::canEvaluate(expr),
            FunctionExpression::canEvaluate(expr)));
  }
  for (const auto& input : expr->inputs()) {
    describeClaims(input, lines);
  }
}

// True when every call node is claimable by the tier `path` names, which
// together with the root's type proves the whole tree ran there.
bool allNodesClaimedBy(const core::TypedExprPtr& expr, Path path) {
  if (expr->isCallKind()) {
    const bool claimed = path == Path::kGpuSfi
        ? GpuSfiExpression::canEvaluate(expr)
        : FunctionExpression::canEvaluate(expr);
    if (!claimed) {
      return false;
    }
  }
  for (const auto& input : expr->inputs()) {
    if (!allNodesClaimedBy(input, path)) {
      return false;
    }
  }
  return true;
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

struct CaseResult {
  bool ran{false};
  std::string note;
  Timing timing;
  std::unique_ptr<cudf::column> output;
};

CaseResult runCase(
    Path path,
    const core::TypedExprPtr& typedExpr,
    const RowTypePtr& rowType,
    const std::vector<cudf::column_view>& inputViews,
    memory::MemoryPool* pool,
    const core::QueryConfig& config,
    const std::string& label) {
  CaseResult result;
  DemoteGpuSfi demote(path == Path::kFunction);
  auto cudfExpr = createCudfExpression(typedExpr, rowType, pool, config);
  const auto rootEvaluator = evaluatorName(*cudfExpr);
  if (rootEvaluator != pathName(path)) {
    result.note = fmt::format("root compiled by {}", rootEvaluator);
    cudfExpr->close();
    return result;
  }
  if (!allNodesClaimedBy(typedExpr, path)) {
    result.note = "a nested node falls to another evaluator";
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
      evaluateOnce();
    }
    std::vector<double> samples;
    samples.reserve(FLAGS_iterations);
    for (int i = 0; i < FLAGS_iterations; ++i) {
      result.output.reset();
      checkCuda(cudaDeviceSynchronize());
      const auto start = std::chrono::steady_clock::now();
      {
        nvtx3::scoped_range range{label.c_str()};
        result.output = evaluateOnce();
        checkCuda(cudaDeviceSynchronize());
      }
      const auto end = std::chrono::steady_clock::now();
      samples.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());
    }
    result.timing = summarize(samples);
    result.ran = true;
  } catch (const std::exception& e) {
    result.note = fmt::format("threw: {}", e.what());
    result.output.reset();
  }
  cudfExpr->close();
  return result;
}

// Compares the two paths' outputs; the type must match exactly because a
// decimal column with a different scale holds different numbers even when the
// bits agree.
std::string compareOutputs(
    const cudf::column_view& left,
    const cudf::column_view& right,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  if (left.type() != right.type()) {
    return fmt::format(
        "TYPE DIFFERS (scale {} vs {})",
        left.type().scale(),
        right.type().scale());
  }
  if (left.size() != right.size()) {
    return "SIZE DIFFERS";
  }
  if (left.null_count() != right.null_count()) {
    return fmt::format(
        "NULL COUNT DIFFERS ({} vs {})", left.null_count(), right.null_count());
  }
  auto equal = cudf::binary_operation(
      left,
      right,
      cudf::binary_operator::NULL_EQUALS,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto allEqual = cudf::reduce(
      equal->view(),
      *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  if (static_cast<cudf::numeric_scalar<bool>*>(allEqual.get())->value(stream)) {
    return "identical";
  }
  try {
    auto numEqual = cudf::reduce(
        equal->view(),
        *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
        cudf::data_type{cudf::type_id::INT64},
        stream,
        mr);
    const auto count =
        static_cast<cudf::numeric_scalar<int64_t>*>(numEqual.get())
            ->value(stream);
    return fmt::format("VALUES DIFFER in {} rows", left.size() - count);
  } catch (const std::exception&) {
    return "VALUES DIFFER";
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
  auto pool = memory::memoryManager()->addLeafPool("decimal_benchmark");
  functions::prestosql::registerAllScalarFunctions();
  parse::registerTypeResolver();
  CudfConfig::getInstance().allowCpuFallback = false;
  registerCudf();

  auto queryCtx = core::QueryCtx::create();
  const auto& config = queryCtx->queryConfig();
  const auto& registry = getCudfExpressionEvaluatorRegistry();
  std::cout << "Evaluator priorities:";
  for (const auto& [name, entry] : registry) {
    std::cout << " " << name << "=" << entry.priority;
  }
  std::cout << "\nRows: " << FLAGS_num_rows
            << ", null ratio: " << FLAGS_null_ratio
            << ", iterations: " << FLAGS_iterations
            << ", collect_errors: " << FLAGS_collect_errors << "\n";

  const auto numRows = static_cast<cudf::size_type>(FLAGS_num_rows);
  auto stream = getDefaultStreamForCurrentThread();
  auto tempMr = rmm::mr::get_current_device_resource_ref();

  struct Row {
    std::string width;
    std::string expression;
    std::string resultType;
    CaseResult gpuSfi;
    CaseResult function;
    std::string comparison;
  };
  std::vector<Row> rows;

  for (const auto& width : decimalWidths()) {
    if (!listed(FLAGS_types, width.name)) {
      continue;
    }
    std::cout << "\n== " << width.name << " " << width.veloxType->toString()
              << ": generating " << numRows << " rows x 3 columns\n";
    auto inputs = makeInputs(width, numRows, stream, tempMr);
    std::vector<cudf::column_view> inputViews;
    for (const auto& column : inputs) {
      inputViews.push_back(column->view());
    }
    auto rowType = ROW({
        {"a", width.veloxType},
        {"b", width.veloxType},
        {"c", width.veloxType},
    });

    for (const auto& sql : kExpressions) {
      if (!listed(FLAGS_expressions, sql)) {
        continue;
      }
      auto typedExpr = compileSql(sql, rowType, queryCtx.get(), pool.get());
      Row row{width.name, sql, typedExpr->type()->toString(), {}, {}, ""};
      std::cout << "\n-- " << sql << "  =>  " << typedExpr->toString() << " : "
                << row.resultType << "\n";
      std::vector<std::string> claims;
      describeClaims(typedExpr, claims);
      for (const auto& line : claims) {
        std::cout << line << "\n";
      }

      for (auto path : {Path::kGpuSfi, Path::kFunction}) {
        if (!listed(FLAGS_paths, pathName(path))) {
          continue;
        }
        const auto label =
            fmt::format("{}|{}|{}", pathName(path), width.name, sql);
        auto caseResult = runCase(
            path, typedExpr, rowType, inputViews, pool.get(), config, label);
        if (caseResult.ran) {
          std::cout << fmt::format(
              "    {:8} evaluator={:8} median {:9.3f} ms  p10 {:9.3f}  p90 {:9.3f}  {:8.1f} Mrows/s\n",
              pathName(path),
              pathName(path),
              caseResult.timing.medianMs,
              caseResult.timing.p10Ms,
              caseResult.timing.p90Ms,
              numRows / caseResult.timing.medianMs / 1'000.0);
        } else {
          std::cout << fmt::format(
              "    {:8} SKIPPED: {}\n", pathName(path), caseResult.note);
        }
        (path == Path::kGpuSfi ? row.gpuSfi : row.function) =
            std::move(caseResult);
      }

      if (row.gpuSfi.ran && row.function.ran) {
        row.comparison = compareOutputs(
            row.gpuSfi.output->view(),
            row.function.output->view(),
            stream,
            tempMr);
        std::cout << "    outputs: " << row.comparison << "\n";
      }
      row.gpuSfi.output.reset();
      row.function.output.reset();
      rows.push_back(std::move(row));
    }
  }

  std::cout
      << "\n== Summary (median ms; speedup = function / gpu_sfi, >1 means GPU SFI is faster)\n";
  std::cout << fmt::format(
      "{:6} | {:28} | {:14} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {}\n",
      "width",
      "expression",
      "result type",
      "sfi ms",
      "sfi p90",
      "fn ms",
      "fn p90",
      "speedup",
      "outputs");
  for (const auto& row : rows) {
    auto cell = [](const CaseResult& result, double Timing::* member) {
      return result.ran ? fmt::format("{:.3f}", result.timing.*member)
                        : std::string("-");
    };
    const std::string speedup = row.gpuSfi.ran && row.function.ran
        ? fmt::format(
              "{:.2f}x",
              row.function.timing.medianMs / row.gpuSfi.timing.medianMs)
        : "-";
    std::cout << fmt::format(
        "{:6} | {:28} | {:14} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {}\n",
        row.width,
        row.expression,
        row.resultType,
        cell(row.gpuSfi, &Timing::medianMs),
        cell(row.gpuSfi, &Timing::p90Ms),
        cell(row.function, &Timing::medianMs),
        cell(row.function, &Timing::p90Ms),
        speedup,
        row.comparison.empty()
            ? (row.gpuSfi.ran ? row.function.note : row.gpuSfi.note)
            : row.comparison);
  }
  unregisterCudf();
  return 0;
}

} // namespace
} // namespace facebook::velox::cudf_velox

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv, false};
  return facebook::velox::cudf_velox::run();
}
