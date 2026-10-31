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

#include "velox/benchmarks/tpch/DuckDbTpchQueryBuilder.h"

#include <filesystem>

#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/ReaderFactory.h"

namespace facebook::velox::exec::test {

void DuckDbTpchQueryBuilder::initialize(const std::string& dataPath) {
  for (const auto* name :
       {"customer",
        "lineitem",
        "nation",
        "orders",
        "part",
        "partsupp",
        "region",
        "supplier"}) {
    auto& table = tables_[name];
    for (const auto& entry :
         std::filesystem::directory_iterator(dataPath + "/" + name)) {
      if (entry.is_regular_file() &&
          entry.path().filename().string()[0] != '.') {
        table.dataFiles.push_back(entry.path().string());
      }
    }
    VELOX_CHECK(!table.dataFiles.empty(), "No data files for table: {}", name);
    std::sort(table.dataFiles.begin(), table.dataFiles.end());

    dwio::common::ReaderOptions options(pool_);
    options.setFileFormat(dwio::common::FileFormat::PARQUET);
    std::shared_ptr<ReadFile> file =
        filesystems::getFileSystem(table.dataFiles[0], nullptr)
            ->openFileForRead(table.dataFiles[0]);
    auto reader =
        dwio::common::getReaderFactory(dwio::common::FileFormat::PARQUET)
            ->createReader(
                std::make_unique<dwio::common::BufferedInput>(file, *pool_),
                options);
    table.type = reader->rowType();
  }
}

PlanBuilder DuckDbTpchQueryBuilder::builder(Context& context) const {
  PlanBuilder builder(context.idGenerator, pool_);
  // Keeps literals such as 1.00 DECIMAL so they combine with DECIMAL columns.
  builder.setParseOptions({.parseDecimalAsDouble = false});
  return builder;
}

PlanBuilder DuckDbTpchQueryBuilder::scan(
    Context& context,
    const std::string& table,
    const std::vector<std::string>& columns,
    const std::vector<std::string>& filters,
    const std::string& remainingFilter) const {
  const auto& tableType = tables_.at(table).type;
  std::vector<TypePtr> types;
  types.reserve(columns.size());
  for (const auto& column : columns) {
    types.push_back(tableType->findChild(column));
  }
  core::PlanNodeId scanId;
  auto scanBuilder = builder(context);
  scanBuilder
      .tableScan(
          table,
          ROW(std::vector<std::string>(columns), std::move(types)),
          {},
          filters,
          remainingFilter,
          tableType)
      .capturePlanNodeId(scanId);
  context.plan.dataFiles[scanId] = tables_.at(table).dataFiles;
  return scanBuilder;
}

TpchPlan DuckDbTpchQueryBuilder::finish(
    Context& context,
    const core::PlanNodePtr& root) const {
  context.plan.plan = root;
  context.plan.dataFileFormat = dwio::common::FileFormat::PARQUET;
  return std::move(context.plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQueryPlan(int queryId) const {
  switch (queryId) {
    case 1:
      return getQ1Plan();
    case 2:
      return getQ2Plan();
    case 3:
      return getQ3Plan();
    case 4:
      return getQ4Plan();
    case 5:
      return getQ5Plan();
    case 6:
      return getQ6Plan();
    case 7:
      return getQ7Plan();
    case 8:
      return getQ8Plan();
    case 9:
      return getQ9Plan();
    case 10:
      return getQ10Plan();
    case 11:
      return getQ11Plan();
    case 12:
      return getQ12Plan();
    case 13:
      return getQ13Plan();
    case 14:
      return getQ14Plan();
    case 15:
      return getQ15Plan();
    case 16:
      return getQ16Plan();
    case 17:
      return getQ17Plan();
    case 18:
      return getQ18Plan();
    case 19:
      return getQ19Plan();
    case 20:
      return getQ20Plan();
    case 21:
      return getQ21Plan();
    case 22:
      return getQ22Plan();
    default:
      VELOX_USER_FAIL("TPC-H query does not exist: {}", queryId);
  }
}

TpchPlan DuckDbTpchQueryBuilder::getQ1Plan() const {
  Context context;
  const std::vector<std::string> keys{"l_returnflag", "l_linestatus"};
  auto plan = scan(
                  context,
                  "lineitem",
                  {"l_returnflag",
                   "l_linestatus",
                   "l_quantity",
                   "l_extendedprice",
                   "l_discount",
                   "l_tax"},
                  {"l_shipdate <= '1998-09-02'::DATE"})
                  .project(
                      {"l_returnflag",
                       "l_linestatus",
                       "l_quantity",
                       "l_extendedprice",
                       "l_discount",
                       "l_extendedprice * (1.00 - l_discount) AS disc_price",
                       "l_tax"})
                  .project(
                      {"l_returnflag",
                       "l_linestatus",
                       "l_quantity",
                       "l_extendedprice",
                       "l_discount",
                       "disc_price",
                       "disc_price * (1.00 + l_tax) AS charge"})
                  .partialAggregation(
                      keys,
                      {"sum(l_quantity) AS sum_qty",
                       "sum(l_extendedprice) AS sum_base_price",
                       "sum(disc_price) AS sum_disc_price",
                       "sum(charge) AS sum_charge",
                       "avg(l_quantity) AS avg_qty",
                       "avg(l_extendedprice) AS avg_price",
                       "avg(l_discount) AS avg_disc",
                       "count(1) AS count_order"})
                  .localPartition(keys)
                  .finalAggregation()
                  .orderBy(keys, false)
                  .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ2Plan() const {
  Context context;

  // DuckDB materializes the outer query block (partsupp, supplier, nation,
  // region and part) once as a CTE, probes it with the per-part minimum
  // supply cost, and also uses its distinct p_partkey values to semi-join the
  // subquery. Velox has no CTE materialization, so the semi-join build side
  // reads the filtered part keys directly instead of recomputing the CTE.
  // The final join on p_partkey restricts the result to the same rows.
  auto region = scan(context, "region", {"r_regionkey"}, {"r_name = 'EUROPE'"})
                    .planNode();
  auto nationJoinRegion =
      scan(context, "nation", {"n_nationkey", "n_name", "n_regionkey"})
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"n_nationkey", "n_name"})
          .planNode();
  auto supplierJoinNation = scan(
                                context,
                                "supplier",
                                {"s_suppkey",
                                 "s_name",
                                 "s_address",
                                 "s_nationkey",
                                 "s_phone",
                                 "s_acctbal",
                                 "s_comment"})
                                .hashJoin(
                                    {"s_nationkey"},
                                    {"n_nationkey"},
                                    nationJoinRegion,
                                    "",
                                    {"s_suppkey",
                                     "s_name",
                                     "s_address",
                                     "s_phone",
                                     "s_acctbal",
                                     "s_comment",
                                     "n_name"})
                                .planNode();
  auto part = scan(
                  context,
                  "part",
                  {"p_partkey", "p_mfgr"},
                  {"p_size = 15"},
                  "p_type LIKE '%BRASS'")
                  .planNode();

  auto subqueryRegion =
      scan(context, "region", {"r_regionkey"}, {"r_name = 'EUROPE'"})
          .planNode();
  auto subqueryNationJoinRegion =
      scan(context, "nation", {"n_nationkey", "n_regionkey"})
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              subqueryRegion,
              "",
              {"n_nationkey"})
          .planNode();
  auto subquerySupplierJoinNation =
      scan(context, "supplier", {"s_suppkey", "s_nationkey"})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              subqueryNationJoinRegion,
              "",
              {"s_suppkey"})
          .planNode();
  auto candidatePart = scan(
                           context,
                           "part",
                           {"p_partkey"},
                           {"p_size = 15"},
                           "p_type LIKE '%BRASS'")
                           .planNode();
  auto minSupplyCost =
      scan(context, "partsupp", {"ps_partkey", "ps_suppkey", "ps_supplycost"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              subquerySupplierJoinNation,
              "",
              {"ps_partkey", "ps_supplycost"})
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              candidatePart,
              "",
              {"ps_partkey", "ps_supplycost"},
              core::JoinType::kLeftSemiFilter)
          .partialAggregation(
              {"ps_partkey"}, {"min(ps_supplycost) AS min_supplycost"})
          .localPartition({"ps_partkey"})
          .finalAggregation()
          .planNode();

  const std::vector<std::string> orderKeys{
      "s_acctbal DESC", "n_name", "s_name", "p_partkey"};
  auto plan =
      scan(context, "partsupp", {"ps_partkey", "ps_suppkey", "ps_supplycost"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"ps_partkey",
               "ps_supplycost",
               "s_name",
               "s_address",
               "s_phone",
               "s_acctbal",
               "s_comment",
               "n_name"})
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              part,
              "",
              {"p_partkey",
               "p_mfgr",
               "ps_supplycost",
               "s_name",
               "s_address",
               "s_phone",
               "s_acctbal",
               "s_comment",
               "n_name"})
          .hashJoin(
              {"p_partkey", "ps_supplycost"},
              {"ps_partkey", "min_supplycost"},
              minSupplyCost,
              "",
              {"s_acctbal",
               "s_name",
               "n_name",
               "p_partkey",
               "p_mfgr",
               "s_address",
               "s_phone",
               "s_comment"})
          .topN(orderKeys, 100, true)
          .localMerge(orderKeys)
          .limit(0, 100, false)
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ3Plan() const {
  Context context;
  auto customer =
      scan(context, "customer", {"c_custkey"}, {"c_mktsegment = 'BUILDING'"})
          .planNode();
  auto ordersJoinCustomer =
      scan(
          context,
          "orders",
          {"o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"},
          {"o_orderdate < '1995-03-15'::DATE"})
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customer,
              "",
              {"o_orderkey", "o_orderdate", "o_shippriority"})
          .planNode();

  const std::vector<std::string> keys{
      "l_orderkey", "o_orderdate", "o_shippriority"};
  const std::vector<std::string> orderKeys{"revenue DESC", "o_orderdate"};
  auto plan =
      scan(
          context,
          "lineitem",
          {"l_orderkey", "l_extendedprice", "l_discount"},
          {"l_shipdate > '1995-03-15'::DATE"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              ordersJoinCustomer,
              "",
              {"l_orderkey",
               "o_orderdate",
               "o_shippriority",
               "l_extendedprice",
               "l_discount"})
          .project(
              {"l_orderkey",
               "o_orderdate",
               "o_shippriority",
               "l_extendedprice * (1.00 - l_discount) AS volume"})
          .partialAggregation(keys, {"sum(volume) AS revenue"})
          .localPartition(keys)
          .finalAggregation()
          .project({"l_orderkey", "revenue", "o_orderdate", "o_shippriority"})
          .topN(orderKeys, 10, true)
          .localMerge(orderKeys)
          .limit(0, 10, false)
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ4Plan() const {
  Context context;
  auto orders =
      scan(
          context,
          "orders",
          {"o_orderkey", "o_orderpriority"},
          {"o_orderdate BETWEEN '1993-07-01'::DATE AND '1993-09-30'::DATE"})
          .planNode();

  const std::vector<std::string> keys{"o_orderpriority"};
  auto plan =
      scan(context, "lineitem", {"l_orderkey", "l_commitdate", "l_receiptdate"})
          .filter("l_commitdate < l_receiptdate")
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"o_orderpriority"},
              core::JoinType::kRightSemiFilter)
          .partialAggregation(keys, {"count(1) AS order_count"})
          .localPartition(keys)
          .finalAggregation()
          .orderBy(keys, false)
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ5Plan() const {
  Context context;
  auto region =
      scan(context, "region", {"r_regionkey"}, {"r_name = 'ASIA'"}).planNode();
  auto nationJoinRegion =
      scan(context, "nation", {"n_nationkey", "n_name", "n_regionkey"})
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"n_nationkey", "n_name"})
          .planNode();
  auto customerJoinNation =
      scan(context, "customer", {"c_custkey", "c_nationkey"})
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              {"c_custkey", "c_nationkey", "n_name"})
          .planNode();
  auto ordersJoinCustomer =
      scan(
          context,
          "orders",
          {"o_orderkey", "o_custkey"},
          {"o_orderdate BETWEEN '1994-01-01'::DATE AND '1994-12-31'::DATE"})
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customerJoinNation,
              "",
              {"o_orderkey", "c_nationkey", "n_name"})
          .planNode();
  auto supplier =
      scan(context, "supplier", {"s_suppkey", "s_nationkey"}).planNode();

  const std::vector<std::string> keys{"n_name"};
  auto plan =
      scan(
          context,
          "lineitem",
          {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              ordersJoinCustomer,
              "",
              {"l_suppkey",
               "l_extendedprice",
               "l_discount",
               "c_nationkey",
               "n_name"})
          .hashJoin(
              {"c_nationkey", "l_suppkey"},
              {"s_nationkey", "s_suppkey"},
              supplier,
              "",
              {"n_name", "l_extendedprice", "l_discount"})
          .project(
              {"n_name", "l_extendedprice * (1.00 - l_discount) AS volume"})
          .partialAggregation(keys, {"sum(volume) AS revenue"})
          .localPartition(keys)
          .finalAggregation()
          .orderBy({"revenue DESC"}, false)
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ6Plan() const {
  Context context;
  auto plan =
      scan(
          context,
          "lineitem",
          {"l_extendedprice", "l_discount"},
          {"l_quantity < 24.00::DECIMAL(15, 2)",
           "l_discount BETWEEN 0.05::DECIMAL(15, 2) AND 0.07::DECIMAL(15, 2)",
           "l_shipdate BETWEEN '1994-01-01'::DATE AND '1994-12-31'::DATE"})
          .project({"l_extendedprice * l_discount AS revenue"})
          .partialAggregation({}, {"sum(revenue) AS revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ7Plan() const {
  Context context;
  auto supplierNation = scan(
                            context,
                            "nation",
                            {"n_nationkey", "n_name"},
                            {"n_name IN ('FRANCE', 'GERMANY')"})
                            .project({"n_nationkey", "n_name AS supp_nation"})
                            .planNode();
  auto supplierJoinNation =
      scan(context, "supplier", {"s_suppkey", "s_nationkey"})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              supplierNation,
              "",
              {"s_suppkey", "supp_nation"})
          .planNode();
  auto lineitemJoinSupplier =
      scan(
          context,
          "lineitem",
          {"l_orderkey",
           "l_suppkey",
           "l_extendedprice",
           "l_discount",
           "l_shipdate"},
          {"l_shipdate BETWEEN '1995-01-01'::DATE AND '1996-12-31'::DATE"})
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"l_orderkey",
               "l_extendedprice",
               "l_discount",
               "l_shipdate",
               "supp_nation"})
          .planNode();
  auto customerNation = scan(
                            context,
                            "nation",
                            {"n_nationkey", "n_name"},
                            {"n_name IN ('FRANCE', 'GERMANY')"})
                            .project({"n_nationkey", "n_name AS cust_nation"})
                            .planNode();
  auto customerJoinNation =
      scan(context, "customer", {"c_custkey", "c_nationkey"})
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              customerNation,
              "",
              {"c_custkey", "cust_nation"})
          .planNode();

  const std::vector<std::string> keys{"supp_nation", "cust_nation", "l_year"};
  auto plan = scan(context, "orders", {"o_orderkey", "o_custkey"})
                  .hashJoin(
                      {"o_orderkey"},
                      {"l_orderkey"},
                      lineitemJoinSupplier,
                      "",
                      {"o_custkey",
                       "l_extendedprice",
                       "l_discount",
                       "l_shipdate",
                       "supp_nation"})
                  .hashJoin(
                      {"o_custkey"},
                      {"c_custkey"},
                      customerJoinNation,
                      "(supp_nation = 'FRANCE' AND cust_nation = 'GERMANY') OR "
                      "(supp_nation = 'GERMANY' AND cust_nation = 'FRANCE')",
                      {"supp_nation",
                       "cust_nation",
                       "l_shipdate",
                       "l_extendedprice",
                       "l_discount"})
                  .project(
                      {"supp_nation",
                       "cust_nation",
                       "year(l_shipdate) AS l_year",
                       "l_extendedprice * (1.00 - l_discount) AS volume"})
                  .partialAggregation(keys, {"sum(volume) AS revenue"})
                  .localPartition(keys)
                  .finalAggregation()
                  .orderBy(keys, false)
                  .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ8Plan() const {
  Context context;
  auto region = scan(context, "region", {"r_regionkey"}, {"r_name = 'AMERICA'"})
                    .planNode();
  auto nationJoinRegion =
      scan(context, "nation", {"n_nationkey", "n_regionkey"})
          .hashJoin(
              {"n_regionkey"}, {"r_regionkey"}, region, "", {"n_nationkey"})
          .planNode();
  auto customerJoinNation =
      scan(context, "customer", {"c_custkey", "c_nationkey"})
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              {"c_custkey"})
          .planNode();
  auto ordersJoinCustomer =
      scan(
          context,
          "orders",
          {"o_orderkey", "o_custkey", "o_orderdate"},
          {"o_orderdate BETWEEN '1995-01-01'::DATE AND '1996-12-31'::DATE"})
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customerJoinNation,
              "",
              {"o_orderkey", "o_orderdate"})
          .planNode();
  auto nation = scan(context, "nation", {"n_nationkey", "n_name"}).planNode();
  auto supplierJoinNation =
      scan(context, "supplier", {"s_suppkey", "s_nationkey"})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"s_suppkey", "n_name"})
          .planNode();
  auto part =
      scan(
          context, "part", {"p_partkey"}, {"p_type = 'ECONOMY ANODIZED STEEL'"})
          .planNode();
  auto plan =
      scan(
          context,
          "lineitem",
          {"l_orderkey",
           "l_partkey",
           "l_suppkey",
           "l_extendedprice",
           "l_discount"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              ordersJoinCustomer,
              "",
              {"l_partkey",
               "l_suppkey",
               "l_extendedprice",
               "l_discount",
               "o_orderdate"})
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"l_partkey",
               "l_extendedprice",
               "l_discount",
               "o_orderdate",
               "n_name"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"l_extendedprice", "l_discount", "o_orderdate", "n_name"})
          .project(
              {"year(o_orderdate) AS o_year",
               "l_extendedprice * (1.00 - l_discount) AS volume",
               "n_name AS nation"})
          .project(
              {"o_year",
               "CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0::DECIMAL(31, 4) END AS brazil_volume",
               "volume"})
          .partialAggregation(
              {"o_year"},
              {"sum(brazil_volume) AS sum_brazil_volume",
               "sum(volume) AS sum_volume"})
          .localPartition({"o_year"})
          .finalAggregation()
          // DuckDB divides DECIMAL by DECIMAL in DOUBLE.
          .project(
              {"o_year",
               "CAST(sum_brazil_volume AS DOUBLE) / CAST(sum_volume AS DOUBLE) AS mkt_share"})
          .orderBy({"o_year"}, false)
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ9Plan() const {
  Context context;
  auto orders =
      scan(context, "orders", {"o_orderkey", "o_orderdate"}).planNode();
  auto nation = scan(context, "nation", {"n_nationkey", "n_name"}).planNode();
  auto supplierJoinNation =
      scan(context, "supplier", {"s_suppkey", "s_nationkey"})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"s_suppkey", "n_name"})
          .planNode();
  auto part = scan(context, "part", {"p_partkey"}, {}, "p_name LIKE '%green%'")
                  .planNode();
  auto partsuppJoinPart =
      scan(context, "partsupp", {"ps_partkey", "ps_suppkey", "ps_supplycost"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"ps_partkey", "ps_supplycost", "s_suppkey", "n_name"})
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              part,
              "",
              {"p_partkey", "s_suppkey", "ps_supplycost", "n_name"})
          .planNode();
  auto plan =
      scan(
          context,
          "lineitem",
          {"l_orderkey",
           "l_partkey",
           "l_suppkey",
           "l_quantity",
           "l_extendedprice",
           "l_discount"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"l_partkey",
               "l_suppkey",
               "l_quantity",
               "l_extendedprice",
               "l_discount",
               "o_orderdate"})
          .hashJoin(
              {"l_partkey", "l_suppkey"},
              {"p_partkey", "s_suppkey"},
              partsuppJoinPart,
              "",
              {"l_quantity",
               "l_extendedprice",
               "l_discount",
               "o_orderdate",
               "ps_supplycost",
               "n_name"})
          .project(
              {"n_name AS nation",
               "year(o_orderdate) AS o_year",
               "l_extendedprice * (1.00 - l_discount) - ps_supplycost * l_quantity AS amount"})
          .partialAggregation(
              {"nation", "o_year"}, {"sum(amount) AS sum_profit"})
          .localPartition({"nation", "o_year"})
          .finalAggregation()
          .orderBy({"nation", "o_year DESC"}, false)
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ10Plan() const {
  Context context;
  const std::vector<std::string> keys{
      "c_custkey",
      "c_name",
      "c_acctbal",
      "c_phone",
      "n_name",
      "c_address",
      "c_comment"};
  auto nation = scan(context, "nation", {"n_nationkey", "n_name"}).planNode();
  auto orders =
      scan(
          context,
          "orders",
          {"o_orderkey", "o_custkey"},
          {"o_orderdate BETWEEN '1993-10-01'::DATE AND '1993-12-31'::DATE"})
          .planNode();
  auto customerJoinOrders = scan(
                                context,
                                "customer",
                                {"c_custkey",
                                 "c_name",
                                 "c_address",
                                 "c_nationkey",
                                 "c_phone",
                                 "c_acctbal",
                                 "c_comment"})
                                .hashJoin(
                                    {"c_nationkey"},
                                    {"n_nationkey"},
                                    nation,
                                    "",
                                    {"c_custkey",
                                     "c_name",
                                     "c_address",
                                     "c_phone",
                                     "c_acctbal",
                                     "c_comment",
                                     "n_name"})
                                .hashJoin(
                                    {"c_custkey"},
                                    {"o_custkey"},
                                    orders,
                                    "",
                                    {"c_custkey",
                                     "c_name",
                                     "c_address",
                                     "c_phone",
                                     "c_acctbal",
                                     "c_comment",
                                     "n_name",
                                     "o_orderkey"})
                                .planNode();
  auto plan = scan(
                  context,
                  "lineitem",
                  {"l_orderkey", "l_extendedprice", "l_discount"},
                  {"l_returnflag = 'R'"})
                  .hashJoin(
                      {"l_orderkey"},
                      {"o_orderkey"},
                      customerJoinOrders,
                      "",
                      {"c_custkey",
                       "c_name",
                       "c_address",
                       "c_phone",
                       "c_acctbal",
                       "c_comment",
                       "n_name",
                       "l_extendedprice",
                       "l_discount"})
                  .project(
                      {"c_custkey",
                       "c_name",
                       "c_acctbal",
                       "c_phone",
                       "n_name",
                       "c_address",
                       "c_comment",
                       "l_extendedprice * (1.00 - l_discount) AS volume"})
                  .partialAggregation(keys, {"sum(volume) AS revenue"})
                  .localPartition(keys)
                  .finalAggregation()
                  .project(
                      {"c_custkey",
                       "c_name",
                       "revenue",
                       "c_acctbal",
                       "n_name",
                       "c_address",
                       "c_phone",
                       "c_comment"})
                  .topN({"revenue DESC"}, 20, true)
                  .localMerge({"revenue DESC"})
                  .limit(0, 20, false)
                  .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ11Plan() const {
  Context context;
  auto subqueryNation =
      scan(context, "nation", {"n_nationkey"}, {"n_name = 'GERMANY'"})
          .planNode();
  auto subquerySupplierJoinNation =
      scan(context, "supplier", {"s_suppkey", "s_nationkey"})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              subqueryNation,
              "",
              {"s_suppkey"})
          .planNode();
  auto threshold =
      scan(context, "partsupp", {"ps_suppkey", "ps_availqty", "ps_supplycost"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              subquerySupplierJoinNation,
              "",
              {"ps_availqty", "ps_supplycost"})
          .project(
              {"ps_supplycost * CAST(ps_availqty AS DECIMAL(18, 0)) AS value"})
          .partialAggregation({}, {"sum(value) AS total_value"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"total_value * 0.0001000000 AS threshold"})
          .planNode();
  auto nation = scan(context, "nation", {"n_nationkey"}, {"n_name = 'GERMANY'"})
                    .planNode();
  auto supplierJoinNation =
      scan(context, "supplier", {"s_suppkey", "s_nationkey"})
          .hashJoin({"s_nationkey"}, {"n_nationkey"}, nation, "", {"s_suppkey"})
          .planNode();
  auto plan =
      scan(
          context,
          "partsupp",
          {"ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"ps_partkey", "ps_availqty", "ps_supplycost"})
          .project(
              {"ps_partkey",
               "ps_supplycost * CAST(ps_availqty AS DECIMAL(18, 0)) AS value"})
          .partialAggregation({"ps_partkey"}, {"sum(value) AS value"})
          .localPartition({"ps_partkey"})
          .finalAggregation()
          .nestedLoopJoin(
              threshold,
              "CAST(value AS DECIMAL(38, 12)) > threshold",
              {"ps_partkey", "value"})
          .orderBy({"value DESC"}, true)
          .localMerge({"value DESC"})
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ12Plan() const {
  Context context;
  // DuckDB pushes l_shipmode IN (...) into the scan only as an optional
  // (pruning) filter and evaluates it again in its FILTER operator; Velox
  // evaluates it once, in the scan.
  auto lineitem =
      scan(
          context,
          "lineitem",
          {"l_orderkey",
           "l_shipdate",
           "l_commitdate",
           "l_receiptdate",
           "l_shipmode"},
          {"l_shipdate < '1995-01-01'::DATE",
           "l_commitdate < '1995-01-01'::DATE",
           "l_receiptdate BETWEEN '1994-01-01'::DATE AND '1994-12-31'::DATE",
           "l_shipmode IN ('MAIL', 'SHIP')"})
          .filter("l_commitdate < l_receiptdate AND l_shipdate < l_commitdate")
          .project({"l_orderkey", "l_shipmode"})
          .planNode();
  auto plan =
      scan(context, "orders", {"o_orderkey", "o_orderpriority"})
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              lineitem,
              "",
              {"l_shipmode", "o_orderpriority"})
          .project(
              {"l_shipmode",
               "CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END AS high_line",
               "CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END AS low_line"})
          .partialAggregation(
              {"l_shipmode"},
              {"sum(high_line) AS high_line_count",
               "sum(low_line) AS low_line_count"})
          .localPartition({"l_shipmode"})
          .finalAggregation()
          .orderBy({"l_shipmode"}, false)
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ13Plan() const {
  Context context;
  auto customer = scan(context, "customer", {"c_custkey"}).planNode();
  auto plan =
      scan(
          context,
          "orders",
          {"o_orderkey", "o_custkey"},
          {},
          "o_comment NOT LIKE '%special%requests%'")
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customer,
              "",
              {"c_custkey", "o_orderkey"},
              core::JoinType::kRight)
          .partialAggregation({"c_custkey"}, {"count(o_orderkey) AS c_count"})
          .localPartition({"c_custkey"})
          .finalAggregation()
          .partialAggregation({"c_count"}, {"count(1) AS custdist"})
          .localPartition({"c_count"})
          .finalAggregation()
          .orderBy({"custdist DESC", "c_count DESC"}, false)
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ14Plan() const {
  Context context;
  auto part = scan(context, "part", {"p_partkey", "p_type"}).planNode();
  auto plan =
      scan(
          context,
          "lineitem",
          {"l_partkey", "l_extendedprice", "l_discount"},
          {"l_shipdate BETWEEN '1995-09-01'::DATE AND '1995-09-30'::DATE"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"l_extendedprice", "l_discount", "p_type"})
          .project(
              {"l_extendedprice * (1.00 - l_discount) AS disc_price", "p_type"})
          // The ELSE literal takes the type of 'disc_price'; SWITCH requires
          // all branches to have the same type.
          .project(
              {"CASE WHEN p_type LIKE 'PROMO%' THEN disc_price ELSE 0::DECIMAL(31, 4) END AS promo_price",
               "disc_price"})
          .partialAggregation(
              {},
              {"sum(promo_price) AS promo_sum", "sum(disc_price) AS total_sum"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"100.00 * promo_sum / total_sum AS promo_revenue"})
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ15Plan() const {
  Context context;
  // DuckDB materializes 'revenue' once as a CTE and scans it twice: for the
  // maximum revenue and for the suppliers that have it. Velox computes it once
  // and gets the maximum from a window over all its rows.
  auto supplier =
      scan(context, "supplier", {"s_suppkey", "s_name", "s_address", "s_phone"})
          .planNode();
  auto plan =
      scan(
          context,
          "lineitem",
          {"l_suppkey", "l_extendedprice", "l_discount"},
          {"l_shipdate BETWEEN '1996-01-01'::DATE AND '1996-03-31'::DATE"})
          .project(
              {"l_suppkey AS supplier_no",
               "l_extendedprice * (1.00 - l_discount) AS disc_price"})
          .partialAggregation(
              {"supplier_no"}, {"sum(disc_price) AS total_revenue"})
          .localPartition({"supplier_no"})
          .finalAggregation()
          .localGather()
          .window({"max(total_revenue) OVER () AS max_revenue"})
          .filter("total_revenue = max_revenue")
          .hashJoin(
              {"supplier_no"},
              {"s_suppkey"},
              supplier,
              "",
              {"s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"})
          .orderBy({"s_suppkey"}, false)
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ16Plan() const {
  Context context;
  const std::vector<std::string> distinctKeys{
      "p_brand", "p_type", "p_size", "ps_suppkey"};
  const std::vector<std::string> keys{"p_brand", "p_type", "p_size"};
  // DuckDB evaluates the p_size IN list as a mark join against the list
  // followed by a filter; Velox pushes it into the scan.
  auto part =
      scan(
          context,
          "part",
          {"p_partkey", "p_brand", "p_type", "p_size"},
          {"p_brand <> 'Brand#45'",
           "p_size IN (49::INTEGER, 14::INTEGER, 23::INTEGER, 45::INTEGER, "
           "19::INTEGER, 3::INTEGER, 36::INTEGER, 9::INTEGER)"},
          "p_type NOT LIKE 'MEDIUM POLISHED%'")
          .planNode();
  auto supplier = scan(
                      context,
                      "supplier",
                      {"s_suppkey"},
                      {},
                      "s_comment LIKE '%Customer%Complaints%'")
                      .planNode();
  // DuckDB's mark join followed by 'NOT SUBQUERY' is a null-aware anti join.
  auto plan =
      scan(context, "partsupp", {"ps_partkey", "ps_suppkey"})
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              part,
              "",
              {"ps_suppkey", "p_brand", "p_type", "p_size"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplier,
              "",
              {"p_brand", "p_type", "p_size", "ps_suppkey"},
              core::JoinType::kAnti,
              true)
          .partialAggregation(distinctKeys, {})
          .localPartition(distinctKeys)
          .finalAggregation()
          .partialAggregation(keys, {"count(ps_suppkey) AS supplier_cnt"})
          .localPartition(keys)
          .finalAggregation()
          .orderBy({"supplier_cnt DESC", "p_brand", "p_type", "p_size"}, true)
          .localMerge({"supplier_cnt DESC", "p_brand", "p_type", "p_size"})
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ17Plan() const {
  Context context;
  const std::vector<std::string> partFilters{
      "p_brand = 'Brand#23'", "p_container = 'MED BOX'"};
  // DuckDB decorrelates the subquery with a delim join: it materializes
  // lineitem joined with part as a CTE and semi joins lineitem with the
  // distinct p_partkey of that CTE. Velox plans cannot share a subtree, so
  // the semi join builds on the filtered part scan instead; the parts it adds
  // have no lineitem rows and produce no groups. As in DuckDB, the average
  // and the comparison are in DOUBLE.
  auto partForAverage =
      scan(context, "part", {"p_partkey"}, partFilters).planNode();
  auto averageQuantity =
      scan(context, "lineitem", {"l_partkey", "l_quantity"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              partForAverage,
              "",
              {"l_partkey", "l_quantity"},
              core::JoinType::kLeftSemiFilter)
          .project({"l_partkey", "CAST(l_quantity AS DOUBLE) AS quantity"})
          .partialAggregation({"l_partkey"}, {"avg(quantity) AS avg_quantity"})
          .localPartition({"l_partkey"})
          .finalAggregation()
          .project(
              {"l_partkey", "0.2::DOUBLE * avg_quantity AS quantity_limit"})
          .planNode();
  auto part = scan(context, "part", {"p_partkey"}, partFilters).planNode();
  auto plan =
      scan(context, "lineitem", {"l_partkey", "l_quantity", "l_extendedprice"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"p_partkey", "l_quantity", "l_extendedprice"})
          .hashJoin(
              {"p_partkey"},
              {"l_partkey"},
              averageQuantity,
              "CAST(l_quantity AS DOUBLE) < quantity_limit",
              {"l_extendedprice"})
          .partialAggregation({}, {"sum(l_extendedprice) AS total_price"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"total_price / 7.0 AS avg_yearly"})
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ18Plan() const {
  Context context;
  auto largeOrders =
      scan(context, "lineitem", {"l_orderkey", "l_quantity"})
          .partialAggregation({"l_orderkey"}, {"sum(l_quantity) AS quantity"})
          .localPartition({"l_orderkey"})
          .finalAggregation()
          .filter("quantity > 300.00::DECIMAL(38, 2)")
          .project({"l_orderkey"})
          .planNode();
  auto customer = scan(context, "customer", {"c_custkey", "c_name"}).planNode();
  auto customerLargeOrders =
      scan(
          context,
          "orders",
          {"o_orderkey", "o_custkey", "o_totalprice", "o_orderdate"})
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              largeOrders,
              "",
              {"o_orderkey", "o_custkey", "o_totalprice", "o_orderdate"},
              core::JoinType::kLeftSemiFilter)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customer,
              "",
              {"c_name",
               "c_custkey",
               "o_orderkey",
               "o_orderdate",
               "o_totalprice"})
          .planNode();
  const std::vector<std::string> keys{
      "c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"};
  const std::vector<std::string> sortingKeys{
      "o_totalprice DESC", "o_orderdate"};
  auto plan =
      scan(context, "lineitem", {"l_orderkey", "l_quantity"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              customerLargeOrders,
              "",
              {"c_name",
               "c_custkey",
               "o_orderkey",
               "o_orderdate",
               "o_totalprice",
               "l_quantity"})
          .partialAggregation(keys, {"sum(l_quantity) AS \"sum(l_quantity)\""})
          .localPartition(keys)
          .finalAggregation()
          .topN(sortingKeys, 100, true)
          .localMerge(sortingKeys)
          .limit(0, 100, false)
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ19Plan() const {
  Context context;
  // DuckDB also pushes '(p_size >= 1 OR p_size <= 5 OR ...)' into the part
  // scan. It accepts every row, so it is left out.
  auto part =
      scan(
          context,
          "part",
          {"p_partkey", "p_brand", "p_size", "p_container"},
          {"p_brand IN ('Brand#12', 'Brand#23', 'Brand#34')",
           "p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG', "
           "'MED BAG', 'MED BOX', 'MED PKG', 'MED PACK', "
           "'LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')"},
          "(p_brand = 'Brand#12' AND p_size BETWEEN 1 AND 5 "
          "AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')) "
          "OR (p_brand = 'Brand#23' AND p_size BETWEEN 1 AND 10 "
          "AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')) "
          "OR (p_brand = 'Brand#34' AND p_size BETWEEN 1 AND 15 "
          "AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'))")
          .planNode();
  // DuckDB uses the l_shipmode IN list only to skip row groups and evaluates
  // it in a FILTER above the scan. Velox applies it in the scan.
  auto plan =
      scan(
          context,
          "lineitem",
          {"l_partkey", "l_quantity", "l_extendedprice", "l_discount"},
          {"l_quantity BETWEEN 1.00::DECIMAL(15, 2) AND 11.00::DECIMAL(15, 2) "
           "OR l_quantity BETWEEN 10.00::DECIMAL(15, 2) AND 20.00::DECIMAL(15, 2) "
           "OR l_quantity BETWEEN 20.00::DECIMAL(15, 2) AND 30.00::DECIMAL(15, 2)",
           "l_shipinstruct = 'DELIVER IN PERSON'",
           "l_shipmode IN ('AIR', 'AIR REG')"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "(l_quantity BETWEEN 1.00::DECIMAL(15, 2) AND 11.00::DECIMAL(15, 2) "
              "AND p_size BETWEEN 1 AND 5 AND p_brand = 'Brand#12' "
              "AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')) "
              "OR (l_quantity BETWEEN 10.00::DECIMAL(15, 2) AND 20.00::DECIMAL(15, 2) "
              "AND p_size BETWEEN 1 AND 10 AND p_brand = 'Brand#23' "
              "AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')) "
              "OR (l_quantity BETWEEN 20.00::DECIMAL(15, 2) AND 30.00::DECIMAL(15, 2) "
              "AND p_size BETWEEN 1 AND 15 AND p_brand = 'Brand#34' "
              "AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'))",
              {"l_extendedprice", "l_discount"})
          .project({"l_extendedprice * (1.00 - l_discount) AS revenue"})
          .partialAggregation({}, {"sum(revenue) AS revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ20Plan() const {
  Context context;
  // DuckDB computes partsupp semi-joined with the 'forest%' parts once, as a
  // CTE that both sides of the decorrelated subquery read. Velox has no CTE,
  // so the fragment is built for each reader.
  // A scan takes one range filter per column, so the upper bound of DuckDB's
  // p_name range is a remaining filter.
  auto forestPartsupp = [&]() {
    auto forestPart = scan(
                          context,
                          "part",
                          {"p_partkey"},
                          {"p_name >= 'forest'"},
                          "p_name < 'foresu'")
                          .planNode();
    return scan(
               context, "partsupp", {"ps_partkey", "ps_suppkey", "ps_availqty"})
        .hashJoin(
            {"ps_partkey"},
            {"p_partkey"},
            forestPart,
            "",
            {"ps_partkey", "ps_suppkey", "ps_availqty"},
            core::JoinType::kLeftSemiFilter);
  };
  auto partsuppKeys = forestPartsupp()
                          .partialAggregation({"ps_suppkey", "ps_partkey"}, {})
                          .localPartition({"ps_suppkey", "ps_partkey"})
                          .finalAggregation()
                          .planNode();
  auto halfShippedQuantity =
      scan(
          context,
          "lineitem",
          {"l_partkey", "l_suppkey", "l_quantity"},
          {"l_shipdate BETWEEN '1994-01-01'::DATE AND '1994-12-31'::DATE"})
          .hashJoin(
              {"l_partkey", "l_suppkey"},
              {"ps_partkey", "ps_suppkey"},
              partsuppKeys,
              "",
              {"l_suppkey", "l_partkey", "l_quantity"},
              core::JoinType::kLeftSemiFilter)
          .partialAggregation(
              {"l_suppkey", "l_partkey"}, {"sum(l_quantity) AS quantity"})
          .localPartition({"l_suppkey", "l_partkey"})
          .finalAggregation()
          .project(
              {"0.5 * quantity AS half_quantity", "l_suppkey", "l_partkey"})
          .planNode();
  auto nation = scan(context, "nation", {"n_nationkey"}, {"n_name = 'CANADA'"})
                    .planNode();
  auto canadianSupplier =
      scan(
          context,
          "supplier",
          {"s_suppkey", "s_name", "s_address", "s_nationkey"})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"s_suppkey", "s_name", "s_address"})
          .planNode();
  auto plan = forestPartsupp()
                  .hashJoin(
                      {"ps_suppkey", "ps_partkey"},
                      {"l_suppkey", "l_partkey"},
                      halfShippedQuantity,
                      "",
                      {"ps_suppkey", "ps_availqty", "half_quantity"})
                  .filter("CAST(ps_availqty AS DECIMAL(38, 3)) > half_quantity")
                  .project({"ps_suppkey"})
                  .hashJoin(
                      {"ps_suppkey"},
                      {"s_suppkey"},
                      canadianSupplier,
                      "",
                      {"s_name", "s_address"},
                      core::JoinType::kRightSemiFilter)
                  .orderBy({"s_name"}, true)
                  .localMerge({"s_name"})
                  .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ21Plan() const {
  Context context;
  // Lineitems of the same order from another supplier that were received late.
  auto otherLateLineitem =
      scan(
          context,
          "lineitem",
          {"l_orderkey", "l_suppkey"},
          {"l_orderkey IS NOT NULL", "l_suppkey IS NOT NULL"},
          "l_receiptdate > l_commitdate")
          .project({"l_suppkey AS late_suppkey", "l_orderkey AS late_orderkey"})
          .planNode();
  auto nation =
      scan(context, "nation", {"n_nationkey"}, {"n_name = 'SAUDI ARABIA'"})
          .planNode();
  auto saudiSupplier =
      scan(context, "supplier", {"s_suppkey", "s_name", "s_nationkey"})
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"s_suppkey", "s_name"})
          .planNode();
  // Late lineitems of Saudi suppliers that are the only late ones of their
  // order.
  auto onlyLateLineitem = scan(
                              context,
                              "lineitem",
                              {"l_orderkey", "l_suppkey"},
                              {},
                              "l_receiptdate > l_commitdate")
                              .hashJoin(
                                  {"l_orderkey"},
                                  {"late_orderkey"},
                                  otherLateLineitem,
                                  "late_suppkey <> l_suppkey",
                                  {"l_orderkey", "l_suppkey"},
                                  core::JoinType::kAnti)
                              .hashJoin(
                                  {"l_suppkey"},
                                  {"s_suppkey"},
                                  saudiSupplier,
                                  "",
                                  {"l_orderkey", "l_suppkey", "s_name"})
                              .planNode();
  // Keeps the lineitems whose order has a lineitem from another supplier.
  auto multiSupplierLineitem =
      scan(
          context,
          "lineitem",
          {"l_orderkey", "l_suppkey"},
          {"l_orderkey IS NOT NULL", "l_suppkey IS NOT NULL"})
          .project(
              {"l_suppkey AS other_suppkey", "l_orderkey AS other_orderkey"})
          .hashJoin(
              {"other_orderkey"},
              {"l_orderkey"},
              onlyLateLineitem,
              "other_suppkey <> l_suppkey",
              {"l_orderkey", "s_name"},
              core::JoinType::kRightSemiFilter)
          .planNode();
  const std::vector<std::string> sortingKeys{"numwait DESC", "s_name"};
  auto plan = scan(context, "orders", {"o_orderkey"}, {"o_orderstatus = 'F'"})
                  .hashJoin(
                      {"o_orderkey"},
                      {"l_orderkey"},
                      multiSupplierLineitem,
                      "",
                      {"s_name"})
                  .partialAggregation({"s_name"}, {"count(1) AS numwait"})
                  .localPartition({"s_name"})
                  .finalAggregation()
                  .topN(sortingKeys, 100, true)
                  .localMerge(sortingKeys)
                  .limit(0, 100, false)
                  .planNode();
  return finish(context, plan);
}

TpchPlan DuckDbTpchQueryBuilder::getQ22Plan() const {
  Context context;
  // DuckDB evaluates the country code IN list as a mark join against a
  // constant table; Velox evaluates the IN expression directly. DuckDB
  // computes the customers above the average balance once, as a CTE that
  // both sides of the decorrelated NOT EXISTS read. Velox has no CTE, so the
  // fragment is built for each reader. DuckDB divides the balance sum by the
  // count as DOUBLE; Velox divides as DECIMAL with scale 10. avg() would round
  // the average to the scale of c_acctbal, which drops a customer whose
  // balance equals the rounded average.
  const std::string countryCodeFilter =
      "substr(c_phone, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17')";
  auto customerAboveAverage = [&]() {
    auto averageBalance =
        scan(
            context,
            "customer",
            {"c_acctbal"},
            {"c_acctbal > 0.00::DECIMAL(15, 2)"},
            countryCodeFilter)
            .partialAggregation(
                {},
                {"sum(c_acctbal) AS total_balance",
                 "count(1) AS num_customers"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .project(
                {"CAST(total_balance AS DECIMAL(38, 10)) / "
                 "CAST(num_customers AS DECIMAL(19, 0)) AS average_balance"})
            .planNode();
    return scan(
               context,
               "customer",
               {"c_custkey", "c_phone", "c_acctbal"},
               {},
               countryCodeFilter)
        .nestedLoopJoin(
            averageBalance,
            "CAST(c_acctbal AS DECIMAL(38, 10)) > average_balance",
            {"c_custkey", "c_phone", "c_acctbal"});
  };
  auto customerKeys = customerAboveAverage()
                          .partialAggregation({"c_custkey"}, {})
                          .localPartition({"c_custkey"})
                          .finalAggregation()
                          .planNode();
  auto ordersOfCustomers = scan(context, "orders", {"o_custkey"})
                               .hashJoin(
                                   {"o_custkey"},
                                   {"c_custkey"},
                                   customerKeys,
                                   "",
                                   {"o_custkey"},
                                   core::JoinType::kLeftSemiFilter)
                               .planNode();
  auto plan = customerAboveAverage()
                  .hashJoin(
                      {"c_custkey"},
                      {"o_custkey"},
                      ordersOfCustomers,
                      "",
                      {"c_phone", "c_acctbal"},
                      core::JoinType::kAnti)
                  .project({"substr(c_phone, 1, 2) AS cntrycode", "c_acctbal"})
                  .partialAggregation(
                      {"cntrycode"},
                      {"count(1) AS numcust", "sum(c_acctbal) AS totacctbal"})
                  .localPartition({"cntrycode"})
                  .finalAggregation()
                  .orderBy({"cntrycode"}, false)
                  .planNode();
  return finish(context, plan);
}

} // namespace facebook::velox::exec::test
