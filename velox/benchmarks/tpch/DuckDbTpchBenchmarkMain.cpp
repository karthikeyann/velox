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

#include <fstream>
#include <iostream>

#include <folly/String.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/benchmarks/QueryBenchmarkBase.h"
#include "velox/benchmarks/tpch/DuckDbTpchQueryBuilder.h"
#include "velox/exec/PlanNodeStats.h"

DEFINE_string(data_path, "", "Root of the TPC-H data: <data_path>/<table>/*.");
DEFINE_string(queries, "", "Comma-separated query numbers. Empty runs all.");
DEFINE_int32(num_runs, 3, "Timed runs of each query.");
DEFINE_string(
    results_dir,
    "",
    "If set, writes the result of each query to <results_dir>/q<N>.csv.");
DEFINE_bool(print_stats, false, "Prints the plan with runtime statistics.");

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace {

std::string csvField(const VectorPtr& vector, vector_size_t row) {
  if (vector->isNullAt(row)) {
    return "";
  }
  auto value = vector->toString(row);
  if (value.find_first_of(",\"\n") == std::string::npos) {
    return value;
  }
  std::string quoted = "\"";
  for (auto c : value) {
    quoted += c;
    if (c == '"') {
      quoted += c;
    }
  }
  return quoted + "\"";
}

void writeCsv(
    const std::vector<RowVectorPtr>& results,
    const std::string& path) {
  std::ofstream out(path);
  bool header = false;
  for (const auto& batch : results) {
    const auto& type = batch->rowType();
    if (!header) {
      out << folly::join(",", type->names()) << "\n";
      header = true;
    }
    for (vector_size_t row = 0; row < batch->size(); ++row) {
      for (auto column = 0; column < batch->childrenSize(); ++column) {
        out << (column > 0 ? "," : "") << csvField(batch->childAt(column), row);
      }
      out << "\n";
    }
  }
}

class DuckDbTpchBenchmark : public QueryBenchmarkBase {
 public:
  void runMain(std::ostream& out, RunStats& /*runStats*/) override {
    auto pool = memory::memoryManager()->addLeafPool();
    DuckDbTpchQueryBuilder builder(pool.get());
    builder.initialize(FLAGS_data_path);

    std::vector<int> queries;
    if (FLAGS_queries.empty()) {
      for (auto i = 1; i <= 22; ++i) {
        queries.push_back(i);
      }
    } else {
      folly::split(',', FLAGS_queries, queries);
    }

    for (auto query : queries) {
      const auto plan = builder.getQueryPlan(query);
      for (auto i = 0; i < FLAGS_num_runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        auto [cursor, results] = run(plan, {});
        const std::chrono::duration<double, std::milli> elapsed =
            std::chrono::steady_clock::now() - start;
        if (cursor == nullptr) {
          out << fmt::format("q={} run={} error", query, i) << std::endl;
          break;
        }
        out << fmt::format("q={} run={} ms={:.1f}", query, i, elapsed.count())
            << std::endl;
        if (i + 1 < FLAGS_num_runs) {
          continue;
        }
        if (!FLAGS_results_dir.empty()) {
          writeCsv(
              results, fmt::format("{}/q{}.csv", FLAGS_results_dir, query));
        }
        if (FLAGS_print_stats) {
          out << exec::printPlanWithStats(
                     *plan.plan, cursor->task()->taskStats(), true)
              << std::endl;
        }
      }
    }
  }
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv, false};
  DuckDbTpchBenchmark benchmark;
  benchmark.initialize();
  RunStats ignore;
  benchmark.runMain(std::cout, ignore);
  benchmark.shutdown();
}
