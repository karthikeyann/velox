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
#pragma once

#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"

namespace facebook::velox::exec::test {

/// Builds TPC-H query plans over Parquet data with DECIMAL money columns
/// (TPC-H spec types). Each plan mirrors the physical plan DuckDB chooses for
/// the query: the same join order, build sides, pushed-down filters and
/// projections. This lets the two engines be compared on execution alone,
/// independent of their optimizers. Aggregations and sorts are parallel:
/// partial aggregation, hash repartitioning on the grouping keys and final
/// aggregation; partial sort followed by a local merge.
class DuckDbTpchQueryBuilder {
 public:
  explicit DuckDbTpchQueryBuilder(memory::MemoryPool* pool) : pool_(pool) {}

  /// Reads the schema of each table from its first data file under
  /// 'dataPath'/<table>/.
  void initialize(const std::string& dataPath);

  /// Returns the plan for TPC-H query 'queryId' (1-22).
  TpchPlan getQueryPlan(int queryId) const;

 private:
  struct Table {
    RowTypePtr type;
    std::vector<std::string> dataFiles;
  };

  // Shared state of the plan under construction: the plan node id generator
  // and the data files of each scan node.
  struct Context {
    std::shared_ptr<core::PlanNodeIdGenerator> idGenerator{
        std::make_shared<core::PlanNodeIdGenerator>()};
    TpchPlan plan;
  };

  // Starts a plan fragment with a scan of 'table' producing 'columns'.
  // 'filters' are pushed into the reader; 'remainingFilter' is evaluated on
  // the rows that pass them.
  PlanBuilder scan(
      Context& context,
      const std::string& table,
      const std::vector<std::string>& columns,
      const std::vector<std::string>& filters = {},
      const std::string& remainingFilter = "") const;

  // Returns a builder for a fragment that does not start with a scan.
  PlanBuilder builder(Context& context) const;

  TpchPlan finish(Context& context, const core::PlanNodePtr& root) const;

  TpchPlan getQ1Plan() const;
  TpchPlan getQ2Plan() const;
  TpchPlan getQ3Plan() const;
  TpchPlan getQ4Plan() const;
  TpchPlan getQ5Plan() const;
  TpchPlan getQ6Plan() const;
  TpchPlan getQ7Plan() const;
  TpchPlan getQ8Plan() const;
  TpchPlan getQ9Plan() const;
  TpchPlan getQ10Plan() const;
  TpchPlan getQ11Plan() const;
  TpchPlan getQ12Plan() const;
  TpchPlan getQ13Plan() const;
  TpchPlan getQ14Plan() const;
  TpchPlan getQ15Plan() const;
  TpchPlan getQ16Plan() const;
  TpchPlan getQ17Plan() const;
  TpchPlan getQ18Plan() const;
  TpchPlan getQ19Plan() const;
  TpchPlan getQ20Plan() const;
  TpchPlan getQ21Plan() const;
  TpchPlan getQ22Plan() const;

  // Used to read the file schemas and to parse the plans' expressions.
  memory::MemoryPool* const pool_;
  // Schema and data files of each table, by table name.
  std::unordered_map<std::string, Table> tables_;
};

} // namespace facebook::velox::exec::test
