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
#include "velox/exec/tests/utils/TpchQueryBuilder.h"

#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/tpch/gen/TpchGen.h"

#include <gflags/gflags.h>
#include <fstream>

DEFINE_bool(
    tpch_q17_filter_aggregation,
    false,
    "Restrict Q17's average calculation to the parts selected by the query.");
DEFINE_bool(
    tpch_q19_pushdown,
    false,
    "Push necessary Q19 join predicates into scans and prune unused output columns.");
DEFINE_bool(
    tpch_q19_numeric_predicate,
    false,
    "Compute Q19 per-part quantity bounds before joining instead of rechecking strings.");
DEFINE_bool(
    tpch_q19_late_revenue,
    false,
    "Compute Q19 revenue only after the selective join and its residual predicate.");
DEFINE_bool(
    tpch_q21_candidate_summary,
    false,
    "Evaluate Q21 supplier existence predicates with candidate-order summaries.");
DEFINE_bool(
    tpch_q21_prune_candidate_dates,
    false,
    "Read Q21 candidate dates for the scan predicate without gathering them into output.");
DEFINE_bool(
    tpch_q21_late_supplier_projection,
    false,
    "Compute Q21's nullable late-supplier field after joining candidate orders.");
DEFINE_bool(
    tpch_q17_window,
    false,
    "Compute Q17's selected-part averages with one scan and a partition window.");
DEFINE_bool(
    tpch_q15_window,
    false,
    "Compute Q15's maximum revenue with one aggregation and a global window.");
DEFINE_bool(
    tpch_q1_share_aggregates,
    false,
    "Share Q1 SUM states with AVG and compute averages after aggregation.");
DEFINE_bool(
    tpch_q13_preaggregate,
    false,
    "Aggregate Q13 orders by customer before joining the customer dimension.");
DEFINE_bool(
    tpch_q9_dimension_first,
    false,
    "Build filtered Q9 part/supplier dimensions first and aggregate before nation names.");
DEFINE_bool(
    tpch_q10_late_payload,
    false,
    "Use TPC-H key constraints to aggregate and limit Q10 before customer payload joins.");
DEFINE_bool(
    tpch_q18_late_customer,
    false,
    "Use TPC-H key constraints to limit Q18 before looking up customer names.");
DEFINE_bool(
    tpch_q18_complete_groups,
    false,
    "Use complete lineitem order-key batches for Q18; requires runtime-validated GPU complete-batch groupby.");
DEFINE_bool(
    tpch_q3_topn,
    false,
    "Use TopN instead of full sort plus limit in Q3.");
DEFINE_bool(
    tpch_q4_join_first,
    false,
    "Join filtered Q4 orders before deduplicating matching order keys (TPC-H primary keys).");
DEFINE_bool(
    tpch_q5_customer_first,
    false,
    "Join Q5 filtered customers and orders before lineitem, then check supplier nation.");
DEFINE_bool(
    tpch_q8_part_first,
    false,
    "Apply Q8's selective part join before its order and supplier joins.");
DEFINE_bool(
    tpch_q13_count_rows,
    false,
    "Count Q13 order rows without reading the non-null TPC-H order primary key.");
DEFINE_bool(
    tpch_q13_raw_single,
    false,
    "Gather filtered order keys into a single raw SUM(BIGINT one) aggregation.");
DEFINE_bool(
    tpch_q13_single_count,
    false,
    "Use COUNT(*) instead of SUM(materialized one) in Q13's raw single aggregation.");
DEFINE_bool(
    tpch_q13_prune_comment,
    false,
    "Read Q13 comments for the scan predicate but omit them from scan output.");
DEFINE_bool(
    tpch_prune_filter_only_columns,
    false,
    "Omit filter-only scan outputs in Q1, Q6, Q13 and Q19 when filters stay inside scans.");
DEFINE_bool(
    tpch_q7_numeric_nations,
    false,
    "Carry Q7's two filtered nation labels as booleans through joins and aggregation.");

namespace facebook::velox::exec::test {

namespace {

/// DWRF does not support Date type and Varchar is used.
/// Return the Date filter expression as per data format.
std::string formatDateFilter(
    const std::string& stringDate,
    const RowTypePtr& rowType,
    const std::string& lowerBound,
    const std::string& upperBound) {
  bool isDwrf = rowType->findChild(stringDate)->isVarchar();
  auto suffix = isDwrf ? "" : "::DATE";

  if (!lowerBound.empty() && !upperBound.empty()) {
    return fmt::format(
        "{} between {}{} and {}{}",
        stringDate,
        lowerBound,
        suffix,
        upperBound,
        suffix);
  } else if (!lowerBound.empty()) {
    return fmt::format("{} > {}{}", stringDate, lowerBound, suffix);
  } else if (!upperBound.empty()) {
    return fmt::format("{} < {}{}", stringDate, upperBound, suffix);
  }

  VELOX_FAIL(
      "Date range check expression must have either a lower or an upper bound");
}

std::vector<std::string> mergeColumnNames(
    const std::vector<std::string>& firstColumnVector,
    const std::vector<std::string>& secondColumnVector) {
  std::vector<std::string> mergedColumnVector = std::move(firstColumnVector);
  mergedColumnVector.insert(
      mergedColumnVector.end(),
      secondColumnVector.begin(),
      secondColumnVector.end());
  return mergedColumnVector;
};
} // namespace

void TpchQueryBuilder::readFileSchema(
    const std::string& tableName,
    const std::string& filePath,
    const std::vector<std::string>& columns) {
  dwio::common::ReaderOptions readerOptions(pool_.get());
  readerOptions.setDataIoStats(dataIoStats_);
  readerOptions.setMetadataIoStats(metadataIoStats_);
  readerOptions.setFileFormat(format_);
  auto uniqueReadFile =
      filesystems::getFileSystem(filePath, nullptr)->openFileForRead(filePath);
  std::shared_ptr<ReadFile> readFile;
  readFile.reset(uniqueReadFile.release());
  auto input = std::make_unique<dwio::common::BufferedInput>(
      readFile, readerOptions.memoryPool());
  std::unique_ptr<dwio::common::Reader> reader =
      dwio::common::getReaderFactory(readerOptions.fileFormat())
          ->createReader(std::move(input), readerOptions);
  const auto fileType = reader->rowType();
  const auto fileColumnNames = fileType->names();
  // There can be extra columns in the file towards the end.
  VELOX_CHECK_GE(fileColumnNames.size(), columns.size());
  std::unordered_map<std::string, std::string> fileColumnNamesMap(
      columns.size());
  std::transform(
      columns.begin(),
      columns.end(),
      fileColumnNames.begin(),
      std::inserter(fileColumnNamesMap, fileColumnNamesMap.begin()),
      [](std::string a, std::string b) { return std::make_pair(a, b); });
  auto columnNames = columns;
  auto types = fileType->children();
  types.resize(columnNames.size());
  tableMetadata_[tableName].type =
      std::make_shared<RowType>(std::move(columnNames), std::move(types));
  tableMetadata_[tableName].fileColumnNames = std::move(fileColumnNamesMap);
}

void TpchQueryBuilder::initialize(const std::string& dataPath) {
  for (const auto& [tableName, columns] : kTables_) {
    const fs::path tablePath{dataPath + "/" + tableName};
    std::error_code error;
    bool anyFound = false;
    for (auto const& dirEntry : fs::directory_iterator{
             tablePath, std::filesystem::directory_options(), error}) {
      if (!dirEntry.is_regular_file()) {
        continue;
      }
      // Ignore hidden files.
      if (dirEntry.path().filename().c_str()[0] == '.') {
        continue;
      }
      if (tableMetadata_[tableName].dataFiles.empty()) {
        anyFound = true;
        readFileSchema(tableName, dirEntry.path().string(), columns);
      }
      tableMetadata_[tableName].dataFiles.push_back(dirEntry.path());
    }
    if (!anyFound && error) {
      std::ifstream file(tablePath);
      std::string line;
      while (std::getline(file, line)) {
        if (tableMetadata_[tableName].dataFiles.empty()) {
          readFileSchema(tableName, line, columns);
        }
        tableMetadata_[tableName].dataFiles.push_back(line);
      }
    }
  }
}

const std::vector<std::string>& TpchQueryBuilder::getTableNames() {
  return kTableNames_;
}

TpchPlan TpchQueryBuilder::getQueryPlan(int queryId) const {
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
      return getQ9Plan("p_name like '%green%'");
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
      VELOX_NYI("TPC-H query {} is not supported yet", queryId);
  }
}

TpchPlan TpchQueryBuilder::getAltPlan(int queryId) const {
  switch (queryId) {
    case 9:
      return getQ9Plan("p_size <= 3");
    default:
      VELOX_NYI("TPC-H query {} has no alternate plan", queryId);
  }
}

TpchPlan TpchQueryBuilder::getQ1Plan() const {
  std::vector<std::string> selectedColumns = {
      "l_returnflag",
      "l_linestatus",
      "l_quantity",
      "l_extendedprice",
      "l_discount",
      "l_tax",
      "l_shipdate"};

  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);

  // shipdate <= '1998-09-02'
  const auto shipDate = "l_shipdate";
  auto filter = formatDateFilter(shipDate, selectedRowType, "", "'1998-09-03'");

  core::PlanNodeId lineitemPlanNodeId;

  auto builder = PlanBuilder(pool_.get());
  const bool pruneFilterColumns =
      FLAGS_tpch_prune_filter_only_columns && !filtersAsNode_;
  const auto scanOutputType = pruneFilterColumns ? getRowType(
                                                       kLineitem,
                                                       {"l_returnflag",
                                                        "l_linestatus",
                                                        "l_quantity",
                                                        "l_extendedprice",
                                                        "l_discount",
                                                        "l_tax"})
                                                 : selectedRowType;
  builder.filtersAsNode(filtersAsNode_)
      .tableScan(
          kLineitem,
          scanOutputType,
          fileColumnNames,
          {filter},
          "",
          pruneFilterColumns ? selectedRowType : nullptr)
      .captureScanNodeId(lineitemPlanNodeId)
      .project(
          {"l_returnflag",
           "l_linestatus",
           "l_quantity",
           "l_extendedprice",
           "l_extendedprice * (1.0 - l_discount) AS l_sum_disc_price",
           "l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax) AS l_sum_charge",
           "l_discount"});
  if (FLAGS_tpch_q1_share_aggregates) {
    builder
        .partialAggregation(
            {"l_returnflag", "l_linestatus"},
            {"sum(l_quantity) AS sum_qty",
             "sum(l_extendedprice) AS sum_price",
             "sum(l_sum_disc_price) AS sum_disc_price",
             "sum(l_sum_charge) AS sum_charge",
             "sum(l_discount) AS sum_discount",
             "count(l_quantity) AS count_qty",
             "count(l_extendedprice) AS count_price",
             "count(l_discount) AS count_discount",
             "count(0) AS count_order"})
        .localPartition(std::vector<std::string>{})
        .finalAggregation()
        .project(
            {"l_returnflag",
             "l_linestatus",
             "sum_qty AS a0",
             "sum_price AS a1",
             "sum_disc_price AS a2",
             "sum_charge AS a3",
             "sum_qty / cast(count_qty as double) AS a4",
             "sum_price / cast(count_price as double) AS a5",
             "sum_discount / cast(count_discount as double) AS a6",
             "count_order AS a7"});
  } else {
    builder
        .partialAggregation(
            {"l_returnflag", "l_linestatus"},
            {"sum(l_quantity)",
             "sum(l_extendedprice)",
             "sum(l_sum_disc_price)",
             "sum(l_sum_charge)",
             "avg(l_quantity)",
             "avg(l_extendedprice)",
             "avg(l_discount)",
             "count(0)"})
        .localPartition(std::vector<std::string>{})
        .finalAggregation();
  }
  auto plan =
      builder.orderBy({"l_returnflag", "l_linestatus"}, false).planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ2Plan() const {
  std::vector<std::string> supplierColumnsSubQuery = {
      "s_suppkey", "s_nationkey"};
  std::vector<std::string> nationColumnsSubQuery = {
      "n_nationkey", "n_regionkey"};
  std::vector<std::string> supplierColumns = {
      "s_acctbal",
      "s_name",
      "s_address",
      "s_phone",
      "s_comment",
      "s_suppkey",
      "s_nationkey"};
  std::vector<std::string> partColumns = {
      "p_partkey", "p_mfgr", "p_size", "p_type"};
  std::vector<std::string> partsuppColumns = {
      "ps_partkey", "ps_suppkey", "ps_supplycost"};
  std::vector<std::string> nationColumns = {
      "n_nationkey", "n_name", "n_regionkey"};
  std::vector<std::string> regionColumns = {"r_regionkey", "r_name"};

  auto supplierSelectedRowTypeSubQuery =
      getRowType(kSupplier, supplierColumnsSubQuery);
  const auto& supplierFileColumnsSubQuery = getFileColumnNames(kSupplier);
  auto nationSelectedRowTypeSubQuery =
      getRowType(kNation, nationColumnsSubQuery);
  const auto& nationFileColumnsSubQuery = getFileColumnNames(kNation);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  auto regionSelectedRowType = getRowType(kRegion, regionColumns);
  const auto& regionFileColumns = getFileColumnNames(kRegion);

  const std::string regionNameFilter = "r_name = 'EUROPE'";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId supplierScanIdSubQuery;
  core::PlanNodeId partsuppScanIdSubQuery;
  core::PlanNodeId nationScanIdSubQuery;
  core::PlanNodeId regionScanIdSubQuery;
  core::PlanNodeId partScanId;
  core::PlanNodeId supplierScanId;
  core::PlanNodeId partsuppScanId;
  core::PlanNodeId nationScanId;
  core::PlanNodeId regionScanId;

  auto regionSubQuery = PlanBuilder(planNodeIdGenerator)
                            .filtersAsNode(filtersAsNode_)
                            .tableScan(
                                kRegion,
                                regionSelectedRowType,
                                regionFileColumns,
                                {regionNameFilter})
                            .captureScanNodeId(regionScanIdSubQuery)
                            .planNode();

  auto nationJoinRegionSubQuery =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowTypeSubQuery, nationFileColumnsSubQuery)
          .captureScanNodeId(nationScanIdSubQuery)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              regionSubQuery,
              "",
              {"n_nationkey"})
          .planNode();

  auto supplierJoinNationJoinRegionSubQuery =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kSupplier,
              supplierSelectedRowTypeSubQuery,
              supplierFileColumnsSubQuery)
          .captureScanNodeId(supplierScanIdSubQuery)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationJoinRegionSubQuery,
              "",
              {"s_suppkey"})
          .planNode();

  auto part = PlanBuilder(planNodeIdGenerator)
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partSelectedRowType,
                      partFileColumns,
                      {},
                      "p_type like '%BRASS'")
                  .captureScanNodeId(partScanId)
                  .filter("p_size = 15")
                  .planNode();

  auto region = PlanBuilder(planNodeIdGenerator)
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kRegion,
                        regionSelectedRowType,
                        regionFileColumns,
                        {regionNameFilter})
                    .captureScanNodeId(regionScanId)
                    .planNode();

  auto nationJoinRegion =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanId)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"n_nationkey", "n_name"})
          .planNode();

  auto supplierJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              mergeColumnNames(supplierColumns, {"s_suppkey", "n_name"}))
          .planNode();

  auto partsuppJoinPartJoinSupplierJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanId)
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              part,
              "",
              {"ps_suppkey", "ps_supplycost", "p_partkey", "p_mfgr"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNationJoinRegion,
              "",
              mergeColumnNames(
                  supplierColumns,
                  {"ps_supplycost", "p_partkey", "p_mfgr", "n_name"}))
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanIdSubQuery)
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNationJoinRegionSubQuery,
              "",
              {"ps_supplycost", "ps_partkey"})
          .partialAggregation(
              {"ps_partkey"}, {"min(ps_supplycost) AS min_supplycost"})
          .localPartition({"ps_partkey"})
          .finalAggregation()
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              partsuppJoinPartJoinSupplierJoinNationJoinRegion,
              "ps_supplycost = min_supplycost",
              mergeColumnNames(
                  supplierColumns, {"p_partkey", "p_mfgr", "n_name"}))
          .orderBy({"s_acctbal DESC", "n_name", "s_name", "p_partkey"}, false)
          .project(
              {"s_acctbal",
               "s_name",
               "n_name",
               "p_partkey",
               "p_mfgr",
               "s_address",
               "s_phone",
               "s_comment"})
          .limit(0, 100, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[supplierScanIdSubQuery] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanIdSubQuery] = getTableFilePaths(kPartsupp);
  context.dataFiles[nationScanIdSubQuery] = getTableFilePaths(kNation);
  context.dataFiles[regionScanIdSubQuery] = getTableFilePaths(kRegion);
  context.dataFiles[partScanId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanId] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanId] = getTableFilePaths(kPartsupp);
  context.dataFiles[nationScanId] = getTableFilePaths(kNation);
  context.dataFiles[regionScanId] = getTableFilePaths(kRegion);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ3Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_shipdate", "l_orderkey", "l_extendedprice", "l_discount"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_shippriority", "o_custkey", "o_orderkey"};
  std::vector<std::string> customerColumns = {"c_custkey", "c_mktsegment"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  const auto orderDate = "o_orderdate";
  const auto shipDate = "l_shipdate";
  auto orderDateFilter =
      formatDateFilter(orderDate, ordersSelectedRowType, "", "'1995-03-15'");
  auto shipDateFilter =
      formatDateFilter(shipDate, lineitemSelectedRowType, "'1995-03-15'", "");
  auto customerFilter = "c_mktsegment = 'BUILDING'";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  core::PlanNodeId ordersPlanNodeId;
  core::PlanNodeId customerPlanNodeId;

  auto customers = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(
                           kCustomer,
                           customerSelectedRowType,
                           customerFileColumns,
                           {customerFilter})
                       .captureScanNodeId(customerPlanNodeId)
                       .planNode();

  auto custkeyJoinNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              ordersSelectedRowType,
              ordersFileColumns,
              {orderDateFilter})
          .captureScanNodeId(ordersPlanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"o_orderdate", "o_shippriority", "o_orderkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter})
          .captureScanNodeId(lineitemPlanNodeId)
          .project(
              {"l_extendedprice * (1.0 - l_discount) AS part_revenue",
               "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              custkeyJoinNode,
              "",
              {"l_orderkey", "o_orderdate", "o_shippriority", "part_revenue"})
          .partialAggregation(
              {"l_orderkey", "o_orderdate", "o_shippriority"},
              {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"l_orderkey", "revenue", "o_orderdate", "o_shippriority"})
          .planNode();
  auto orderBuilder = PlanBuilder(plan, planNodeIdGenerator, pool_.get());
  if (FLAGS_tpch_q3_topn) {
    orderBuilder.topN({"revenue DESC", "o_orderdate"}, 10, false);
  } else {
    orderBuilder.orderBy({"revenue DESC", "o_orderdate"}, false)
        .limit(0, 10, false);
  }
  plan = orderBuilder.planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersPlanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerPlanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ4Plan() const {
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_orderpriority", "o_orderkey"};
  std::vector<std::string> lineitemColumns = {
      "l_orderkey", "l_commitdate", "l_receiptdate"};

  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);

  std::string orderDateFilter = formatDateFilter(
      "o_orderdate", ordersSelectedRowType, "'1993-07-01'", "'1993-09-30'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  core::PlanNodeId ordersPlanNodeId;

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        ordersSelectedRowType,
                        ordersFileColumns,
                        {orderDateFilter})
                    .captureScanNodeId(ordersPlanNodeId)
                    .planNode();

  if (FLAGS_tpch_q4_join_first) {
    // TPC-H declares o_orderkey a primary key. Deduplicating matching keys
    // after the selective orders join is equivalent to the original EXISTS,
    // without materializing all qualifying lineitem order keys globally.
    auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kLineitem,
                        lineitemSelectedRowType,
                        lineitemFileColumns,
                        {},
                        "l_commitdate < l_receiptdate")
                    .captureScanNodeId(lineitemPlanNodeId)
                    .hashJoin(
                        {"l_orderkey"},
                        {"o_orderkey"},
                        orders,
                        "",
                        {"o_orderkey", "o_orderpriority"})
                    .partialAggregation({"o_orderkey", "o_orderpriority"}, {})
                    .localPartition({"o_orderkey", "o_orderpriority"})
                    .finalAggregation()
                    .partialAggregation(
                        {"o_orderpriority"}, {"count(0) AS order_count"})
                    .localPartition(std::vector<std::string>{})
                    .finalAggregation()
                    .orderBy({"o_orderpriority"}, false)
                    .planNode();
    TpchPlan context;
    context.plan = std::move(plan);
    context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
    context.dataFiles[ordersPlanNodeId] = getTableFilePaths(kOrders);
    context.dataFileFormat = format_;
    return context;
  }

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {},
              "l_commitdate < l_receiptdate")
          .captureScanNodeId(lineitemPlanNodeId)
          .partialAggregation({"l_orderkey"}, {}, {})
          .localPartition({"l_orderkey"})
          .finalAggregation()
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"o_orderpriority"},
              core::JoinType::kRightSemiFilter)
          .partialAggregation({"o_orderpriority"}, {"count(0) AS order_count"})
          .finalAggregation()
          .orderBy({"o_orderpriority"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersPlanNodeId] = getTableFilePaths(kOrders);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ5Plan() const {
  std::vector<std::string> customerColumns = {"c_custkey", "c_nationkey"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_custkey", "o_orderkey"};
  std::vector<std::string> lineitemColumns = {
      "l_suppkey", "l_orderkey", "l_discount", "l_extendedprice"};
  std::vector<std::string> supplierColumns = {"s_nationkey", "s_suppkey"};
  std::vector<std::string> nationColumns = {
      "n_nationkey", "n_name", "n_regionkey"};
  std::vector<std::string> regionColumns = {"r_regionkey", "r_name"};

  auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  auto regionSelectedRowType = getRowType(kRegion, regionColumns);
  const auto& regionFileColumns = getFileColumnNames(kRegion);

  std::string regionNameFilter = "r_name = 'ASIA'";
  const auto orderDate = "o_orderdate";
  std::string orderDateFilter = formatDateFilter(
      orderDate, ordersSelectedRowType, "'1994-01-01'", "'1994-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId regionScanNodeId;

  auto region = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kRegion,
                        regionSelectedRowType,
                        regionFileColumns,
                        {regionNameFilter})
                    .captureScanNodeId(regionScanNodeId)
                    .planNode();

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        ordersSelectedRowType,
                        ordersFileColumns,
                        {orderDateFilter})
                    .captureScanNodeId(ordersScanNodeId)
                    .planNode();

  auto customer =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .planNode();

  auto nationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"n_nationkey", "n_name"})
          .planNode();

  if (FLAGS_tpch_q5_customer_first) {
    // All joins are inner joins. Since c_nationkey = s_nationkey, the
    // region/nation restriction can be applied on customers first without
    // assuming key uniqueness. Filter orders before touching the fact table
    // and defer revenue computation until both fact joins have succeeded.
    auto selectedCustomers =
        PlanBuilder(customer, planNodeIdGenerator, pool_.get())
            .hashJoin(
                {"c_nationkey"},
                {"n_nationkey"},
                nationJoinRegion,
                "",
                {"c_custkey", "c_nationkey", "n_name"})
            .planNode();
    auto selectedOrders = PlanBuilder(orders, planNodeIdGenerator, pool_.get())
                              .hashJoin(
                                  {"o_custkey"},
                                  {"c_custkey"},
                                  selectedCustomers,
                                  "",
                                  {"o_orderkey", "c_nationkey", "n_name"})
                              .planNode();
    auto supplier =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
            .captureScanNodeId(supplierScanNodeId)
            .planNode();
    auto plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
            .captureScanNodeId(lineitemScanNodeId)
            .hashJoin(
                {"l_orderkey"},
                {"o_orderkey"},
                selectedOrders,
                "",
                {"l_suppkey",
                 "l_extendedprice",
                 "l_discount",
                 "c_nationkey",
                 "n_name"})
            .hashJoin(
                {"l_suppkey", "c_nationkey"},
                {"s_suppkey", "s_nationkey"},
                supplier,
                "",
                {"l_extendedprice", "l_discount", "n_name"})
            .project(
                {"n_name",
                 "l_extendedprice * (1.0 - l_discount) AS part_revenue"})
            .partialAggregation({"n_name"}, {"sum(part_revenue) as revenue"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .orderBy({"revenue DESC"}, false)
            .project({"n_name", "revenue"})
            .planNode();
    TpchPlan context;
    context.plan = std::move(plan);
    context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
    context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
    context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
    context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
    context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
    context.dataFiles[regionScanNodeId] = getTableFilePaths(kRegion);
    context.dataFileFormat = format_;
    return context;
  }

  auto supplierJoinNationRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              {"s_suppkey", "n_name", "s_nationkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .project(
              {"l_extendedprice * (1.0 - l_discount) AS part_revenue",
               "l_orderkey",
               "l_suppkey"})
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplierJoinNationRegion,
              "",
              {"n_name", "part_revenue", "s_nationkey", "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"n_name", "part_revenue", "s_nationkey", "o_custkey"})
          .hashJoin(
              {"s_nationkey", "o_custkey"},
              {"c_nationkey", "c_custkey"},
              customer,
              "",
              {"n_name", "part_revenue"})
          .partialAggregation({"n_name"}, {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"revenue DESC"}, false)
          .project({"n_name", "revenue"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[regionScanNodeId] = getTableFilePaths(kRegion);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ6Plan() const {
  std::vector<std::string> selectedColumns = {
      "l_shipdate", "l_extendedprice", "l_quantity", "l_discount"};

  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);

  const auto shipDate = "l_shipdate";
  auto shipDateFilter = formatDateFilter(
      shipDate, selectedRowType, "'1994-01-01'", "'1994-12-31'");

  core::PlanNodeId lineitemPlanNodeId;
  const bool pruneFilterColumns =
      FLAGS_tpch_prune_filter_only_columns && !filtersAsNode_;
  const auto scanOutputType = pruneFilterColumns
      ? getRowType(kLineitem, {"l_extendedprice", "l_discount"})
      : selectedRowType;
  auto plan = PlanBuilder(pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kLineitem,
                      scanOutputType,
                      fileColumnNames,
                      {shipDateFilter,
                       "l_discount between 0.05 and 0.07",
                       "l_quantity < 24.0"},
                      "",
                      pruneFilterColumns ? selectedRowType : nullptr)
                  .captureScanNodeId(lineitemPlanNodeId)
                  .project({"l_extendedprice * l_discount"})
                  .partialAggregation({}, {"sum(p0)"})
                  .localPartition(std::vector<std::string>{})
                  .finalAggregation()
                  .planNode();
  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ7Plan() const {
  std::vector<std::string> supplierColumns = {"s_nationkey", "s_suppkey"};
  std::vector<std::string> lineitemColumns = {
      "l_shipdate", "l_suppkey", "l_orderkey", "l_discount", "l_extendedprice"};
  std::vector<std::string> ordersColumns = {"o_custkey", "o_orderkey"};
  std::vector<std::string> customerColumns = {"c_custkey", "c_nationkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  const std::string nationFilter = "n_name IN ('FRANCE', 'GERMANY')";
  auto shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "'1995-01-01'", "'1996-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId suppNationScanNodeId;
  core::PlanNodeId custNationScanNodeId;

  auto custNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowType, nationFileColumns, {nationFilter})
          .captureScanNodeId(custNationScanNodeId)
          .planNode();

  auto customerJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              custNation,
              "",
              {"n_name", "c_custkey"})
          .project(
              {FLAGS_tpch_q7_numeric_nations
                   ? "n_name = 'FRANCE' AS cust_nation"
                   : "n_name AS cust_nation",
               "c_custkey"})
          .planNode();

  auto ordersJoinCustomer =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customerJoinNation,
              "",
              {"cust_nation", "o_orderkey"})
          .planNode();

  auto suppNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowType, nationFileColumns, {nationFilter})
          .captureScanNodeId(suppNationScanNodeId)
          .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              suppNation,
              "",
              {"n_name", "s_suppkey"})
          .project(
              {FLAGS_tpch_q7_numeric_nations
                   ? "n_name = 'FRANCE' AS supp_nation"
                   : "n_name AS supp_nation",
               "s_suppkey"})
          .planNode();

  auto builder = PlanBuilder(planNodeIdGenerator, pool_.get());
  builder.filtersAsNode(filtersAsNode_)
      .tableScan(
          kLineitem,
          lineitemSelectedRowType,
          lineitemFileColumns,
          {shipDateFilter})
      .captureScanNodeId(lineitemScanNodeId)
      .hashJoin(
          {"l_suppkey"},
          {"s_suppkey"},
          supplierJoinNation,
          "",
          {"supp_nation",
           "l_extendedprice",
           "l_discount",
           "l_shipdate",
           "l_orderkey"})
      .hashJoin(
          {"l_orderkey"},
          {"o_orderkey"},
          ordersJoinCustomer,
          FLAGS_tpch_q7_numeric_nations
              ? "cust_nation <> supp_nation"
              : "(((cust_nation = 'FRANCE') AND (supp_nation = 'GERMANY')) OR "
                "((cust_nation = 'GERMANY') AND (supp_nation = 'FRANCE')))",
          {"supp_nation",
           "cust_nation",
           "l_extendedprice",
           "l_discount",
           "l_shipdate"})
      .project(
          {"cust_nation",
           "supp_nation",
           "l_extendedprice * (1.0 - l_discount) as part_revenue",
           "year(l_shipdate) as l_year"})
      .partialAggregation(
          {"supp_nation", "cust_nation", "l_year"},
          {"sum(part_revenue) as revenue"})
      .localPartition(std::vector<std::string>{})
      .finalAggregation();
  if (FLAGS_tpch_q7_numeric_nations) {
    // Both nation scans already admit only the two literal labels, excluding
    // nulls. Decode after aggregation so the output and ordering stay
    // unchanged.
    builder.project(
        {"CASE WHEN supp_nation THEN 'FRANCE' ELSE 'GERMANY' END AS supp_nation",
         "CASE WHEN cust_nation THEN 'FRANCE' ELSE 'GERMANY' END AS cust_nation",
         "l_year",
         "revenue"});
  }
  auto plan = builder.orderBy({"supp_nation", "cust_nation", "l_year"}, false)
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[suppNationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[custNationScanNodeId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ8Plan() const {
  std::vector<std::string> partColumns = {"p_partkey", "p_type"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_nationkey"};
  std::vector<std::string> lineitemColumns = {
      "l_suppkey", "l_orderkey", "l_partkey", "l_extendedprice", "l_discount"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_orderkey", "o_custkey"};
  std::vector<std::string> customerColumns = {"c_nationkey", "c_custkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_regionkey"};
  std::vector<std::string> nationColumnsWithName = {
      "n_name", "n_nationkey", "n_regionkey"};
  std::vector<std::string> regionColumns = {"r_name", "r_regionkey"};

  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  const auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  const auto nationSelectedRowTypeWithName =
      getRowType(kNation, nationColumnsWithName);
  const auto& nationFileColumnsWithName = getFileColumnNames(kNation);
  const auto regionSelectedRowType = getRowType(kRegion, regionColumns);
  const auto& regionFileColumns = getFileColumnNames(kRegion);

  const auto orderDateFilter = formatDateFilter(
      "o_orderdate", ordersSelectedRowType, "'1995-01-01'", "'1996-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId partScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId nationScanNodeIdWithName;
  core::PlanNodeId regionScanNodeId;

  auto nationWithName =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowTypeWithName, nationFileColumnsWithName)
          .captureScanNodeId(nationScanNodeIdWithName)
          .planNode();

  auto region = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kRegion,
                        regionSelectedRowType,
                        regionFileColumns,
                        {"r_name = 'AMERICA'"})
                    .captureScanNodeId(regionScanNodeId)
                    .planNode();

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partSelectedRowType,
                      partFileColumns,
                      {"p_type = 'ECONOMY ANODIZED STEEL'"})
                  .captureScanNodeId(partScanNodeId)
                  .planNode();

  auto nationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .hashJoin(
              {"n_regionkey"}, {"r_regionkey"}, region, "", {"n_nationkey"})
          .planNode();

  auto customerJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              {"c_custkey"})
          .planNode();

  auto ordersJoinCustomerJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              ordersSelectedRowType,
              ordersFileColumns,
              {orderDateFilter})
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customerJoinNationJoinRegion,
              "",
              {"o_orderkey", "o_orderdate"})
          .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationWithName,
              "",
              {"s_suppkey", "n_name"})
          .planNode();

  auto factBuilder = PlanBuilder(planNodeIdGenerator, pool_.get());
  factBuilder.filtersAsNode(filtersAsNode_)
      .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
      .captureScanNodeId(lineitemScanNodeId);
  if (FLAGS_tpch_q8_part_first) {
    // Inner-join reassociation preserves duplicates and null semantics. The
    // selected part type is far more selective than the selected orders.
    factBuilder.hashJoin(
        {"l_partkey"},
        {"p_partkey"},
        part,
        "",
        {"l_orderkey",
         "l_partkey",
         "l_suppkey",
         "l_extendedprice",
         "l_discount"});
  }
  factBuilder
      .hashJoin(
          {"l_orderkey"},
          {"o_orderkey"},
          ordersJoinCustomerJoinNationJoinRegion,
          "",
          {"l_partkey",
           "l_suppkey",
           "o_orderdate",
           "l_extendedprice",
           "l_discount"})
      .hashJoin(
          {"l_suppkey"},
          {"s_suppkey"},
          supplierJoinNation,
          "",
          {"n_name",
           "o_orderdate",
           "l_partkey",
           "l_extendedprice",
           "l_discount"});
  if (!FLAGS_tpch_q8_part_first) {
    factBuilder.hashJoin(
        {"l_partkey"},
        {"p_partkey"},
        part,
        "",
        {"n_name", "o_orderdate", "l_extendedprice", "l_discount"});
  }
  auto plan =
      factBuilder
          .project(
              {"l_extendedprice * (1.0 - l_discount) as volume",
               "n_name",
               "o_orderdate"})
          .project(
              {"volume",
               "(CASE WHEN n_name = 'BRAZIL' THEN volume ELSE 0.0 END) as brazil_volume",
               "year(o_orderdate) AS o_year"})
          .partialAggregation(
              {"o_year"},
              {"sum(brazil_volume) as volume_brazil",
               "sum(volume) as volume_all"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"o_year"}, false)
          .project({"o_year", "(volume_brazil / volume_all) as mkt_share"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[nationScanNodeIdWithName] = getTableFilePaths(kNation);
  context.dataFiles[regionScanNodeId] = getTableFilePaths(kRegion);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ9Plan(const std::string& partFilter) const {
  std::vector<std::string> lineitemColumns = {
      "l_suppkey",
      "l_partkey",
      "l_discount",
      "l_extendedprice",
      "l_orderkey",
      "l_quantity"};
  std::vector<std::string> partColumns = {"p_name", "p_size", "p_partkey"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_nationkey"};
  std::vector<std::string> partsuppColumns = {
      "ps_partkey", "ps_suppkey", "ps_supplycost"};
  std::vector<std::string> ordersColumns = {"o_orderkey", "o_orderdate"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId partScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId partsuppScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId nationScanNodeId;

  const std::vector<std::string> lineitemCommonColumns = {
      "l_extendedprice", "l_discount", "l_quantity"};

  auto part =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kPart, partSelectedRowType, partFileColumns, {}, partFilter)
          .captureScanNodeId(partScanNodeId)
          .planNode();

  auto supplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .planNode();

  auto nation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .planNode();

  if (FLAGS_tpch_q9_dimension_first) {
    auto selectedPartsupp =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
            .captureScanNodeId(partsuppScanNodeId)
            .hashJoin({"ps_partkey"}, {"p_partkey"}, part, "", partsuppColumns)
            .hashJoin(
                {"ps_suppkey"},
                {"s_suppkey"},
                supplier,
                "",
                {"ps_partkey", "ps_suppkey", "ps_supplycost", "s_nationkey"})
            .planNode();
    auto profits =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
            .captureScanNodeId(lineitemScanNodeId)
            .hashJoin(
                {"l_partkey", "l_suppkey"},
                {"ps_partkey", "ps_suppkey"},
                selectedPartsupp,
                "",
                {"l_orderkey",
                 "s_nationkey",
                 "l_extendedprice",
                 "l_discount",
                 "l_quantity",
                 "ps_supplycost"})
            .project(
                {"l_orderkey",
                 "s_nationkey",
                 "l_extendedprice * (1.0 - l_discount) - ps_supplycost * l_quantity AS amount"})
            .planNode();
    TpchPlan context;
    context.plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
            .captureScanNodeId(ordersScanNodeId)
            .hashJoin(
                {"o_orderkey"},
                {"l_orderkey"},
                profits,
                "",
                {"s_nationkey", "amount", "o_orderdate"})
            .project({"s_nationkey", "year(o_orderdate) AS o_year", "amount"})
            .partialAggregation(
                {"s_nationkey", "o_year"}, {"sum(amount) AS profit"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .hashJoin(
                {"s_nationkey"},
                {"n_nationkey"},
                nation,
                "",
                {"n_name", "o_year", "profit"})
            // Keep the final grouping by name: distinct nation keys with the
            // same name must still contribute to the same output group.
            .project({"n_name AS nation", "o_year", "profit"})
            .singleAggregation(
                {"nation", "o_year"}, {"sum(profit) AS sum_profit"})
            .orderBy({"nation", "o_year DESC"}, false)
            .planNode();
    context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
    context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
    context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
    context.dataFiles[partsuppScanNodeId] = getTableFilePaths(kPartsupp);
    context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
    context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
    context.dataFileFormat = format_;
    return context;
  }

  auto lineitemJoinPartJoinSupplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .hashJoin({"l_partkey"}, {"p_partkey"}, part, "", lineitemColumns)
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplier,
              "",
              mergeColumnNames(lineitemColumns, {"s_nationkey"}))
          .planNode();

  auto partsuppJoinLineitemJoinPartJoinSupplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanNodeId)
          .hashJoin(
              {"ps_partkey", "ps_suppkey"},
              {"l_partkey", "l_suppkey"},
              lineitemJoinPartJoinSupplier,
              "",
              mergeColumnNames(
                  lineitemCommonColumns,
                  {"l_orderkey", "s_nationkey", "ps_supplycost"}))
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              partsuppJoinLineitemJoinPartJoinSupplier,
              "",
              mergeColumnNames(
                  lineitemCommonColumns,
                  {"s_nationkey", "ps_supplycost", "o_orderdate"}))
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              mergeColumnNames(
                  lineitemCommonColumns,
                  {"ps_supplycost", "o_orderdate", "n_name"}))
          .project(
              {"n_name AS nation",
               "year(o_orderdate) AS o_year",
               "l_extendedprice * (1.0 - l_discount) - ps_supplycost * l_quantity AS amount"})
          .partialAggregation(
              {"nation", "o_year"}, {"sum(amount) AS sum_profit"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"nation", "o_year DESC"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[partsuppScanNodeId] = getTableFilePaths(kPartsupp);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ10Plan() const {
  std::vector<std::string> customerColumns = {
      "c_nationkey",
      "c_custkey",
      "c_acctbal",
      "c_name",
      "c_address",
      "c_phone",
      "c_comment"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};
  std::vector<std::string> lineitemColumns = {
      "l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_orderkey", "o_custkey"};

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  const auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  const auto lineitemReturnFlagFilter = "l_returnflag = 'R'";
  const auto orderDate = "o_orderdate";
  auto orderDateFilter = formatDateFilter(
      orderDate, ordersSelectedRowType, "'1993-10-01'", "'1993-12-31'");

  const std::vector<std::string> customerOutputColumns = {
      "c_name", "c_acctbal", "c_phone", "c_address", "c_custkey", "c_comment"};

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId ordersScanNodeId;

  auto nation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .planNode();

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        ordersSelectedRowType,
                        ordersFileColumns,
                        {orderDateFilter})
                    .captureScanNodeId(ordersScanNodeId)
                    .planNode();

  if (FLAGS_tpch_q10_late_payload) {
    // TPC-H guarantees a unique customer and nation for each order. Aggregate
    // and limit narrow fact rows before fetching the functionally dependent
    // customer payload. This optimization relies on those PK/FK constraints.
    auto topCustomers =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(
                kLineitem,
                lineitemSelectedRowType,
                lineitemFileColumns,
                {lineitemReturnFlagFilter})
            .captureScanNodeId(lineitemScanNodeId)
            .project(
                {"l_orderkey",
                 "l_extendedprice * (1.0 - l_discount) AS part_revenue"})
            .hashJoin(
                {"l_orderkey"},
                {"o_orderkey"},
                orders,
                "",
                {"o_custkey", "part_revenue"})
            .partialAggregation({"o_custkey"}, {"sum(part_revenue) AS revenue"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .topN({"revenue DESC"}, 20, false)
            .planNode();
    TpchPlan context;
    context.plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
            .captureScanNodeId(customerScanNodeId)
            .hashJoin(
                {"c_custkey"},
                {"o_custkey"},
                topCustomers,
                "",
                mergeColumnNames(
                    customerOutputColumns, {"c_nationkey", "revenue"}))
            .hashJoin(
                {"c_nationkey"},
                {"n_nationkey"},
                nation,
                "",
                mergeColumnNames(customerOutputColumns, {"n_name", "revenue"}))
            .localPartition(std::vector<std::string>{})
            .orderBy({"revenue DESC"}, false)
            .project(
                {"c_custkey",
                 "c_name",
                 "revenue",
                 "c_acctbal",
                 "n_name",
                 "c_address",
                 "c_phone",
                 "c_comment"})
            .planNode();
    context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
    context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
    context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
    context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
    context.dataFileFormat = format_;
    return context;
  }

  auto partialPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              orders,
              "",
              mergeColumnNames(
                  customerOutputColumns, {"c_nationkey", "o_orderkey"}))
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              mergeColumnNames(customerOutputColumns, {"n_name", "o_orderkey"}))
          .planNode();

  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kLineitem,
                      lineitemSelectedRowType,
                      lineitemFileColumns,
                      {lineitemReturnFlagFilter})
                  .captureScanNodeId(lineitemScanNodeId)
                  .project(
                      {"l_extendedprice * (1.0 - l_discount) AS part_revenue",
                       "l_orderkey"})
                  .hashJoin(
                      {"l_orderkey"},
                      {"o_orderkey"},
                      partialPlan,
                      "",
                      mergeColumnNames(
                          customerOutputColumns, {"part_revenue", "n_name"}))
                  .partialAggregation(
                      {"c_custkey",
                       "c_name",
                       "c_acctbal",
                       "n_name",
                       "c_address",
                       "c_phone",
                       "c_comment"},
                      {"sum(part_revenue) as revenue"})
                  .localPartition(std::vector<std::string>{})
                  .finalAggregation()
                  .orderBy({"revenue DESC"}, false)
                  .project(
                      {"c_custkey",
                       "c_name",
                       "revenue",
                       "c_acctbal",
                       "n_name",
                       "c_address",
                       "c_phone",
                       "c_comment"})
                  .limit(0, 20, false)
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ11Plan() const {
  // Column definitions.
  std::vector<std::string> partsuppColumns = {
      "ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"};
  std::vector<std::string> partsuppColumnsSubQuery = {
      "ps_availqty", "ps_suppkey", "ps_supplycost"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_nationkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  // Row type definitions.
  auto partsuppRowType = getRowType(kPartsupp, partsuppColumns);
  auto partsuppRowTypeSubQuery = getRowType(kPartsupp, partsuppColumnsSubQuery);
  auto supplierRowType = getRowType(kSupplier, supplierColumns);
  auto nationRowType = getRowType(kNation, nationColumns);

  // File columns.
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  const auto& partsuppFileColumnsSubQuery = getFileColumnNames(kPartsupp);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  // Filter for nation name.
  const std::string nationNameFilter = "n_name = 'GERMANY'";

  // Plan node ID generator.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  // Subquery plan nodes.
  core::PlanNodeId partsuppScanIdSubQuery;
  core::PlanNodeId supplierScanIdSubQuery;
  core::PlanNodeId nationScanIdSubQuery;

  // Main query plan nodes.
  core::PlanNodeId partsuppScanId;
  core::PlanNodeId supplierScanId;
  core::PlanNodeId nationScanId;

  // Subquery plan to calculate filter cost.
  auto nationSubQuery =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationRowType, nationFileColumns, {nationNameFilter})
          .captureScanNodeId(nationScanIdSubQuery)
          .planNode();

  auto supplierJoinNationSubQuery =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanIdSubQuery)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationSubQuery,
              "",
              {"s_suppkey"})
          .planNode();

  auto subQueryPlan =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kPartsupp, partsuppRowTypeSubQuery, partsuppFileColumnsSubQuery)
          .captureScanNodeId(partsuppScanIdSubQuery)
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNationSubQuery,
              "",
              {"ps_availqty", "ps_suppkey", "ps_supplycost"})
          .project({"ps_supplycost * ps_availqty AS product_cost_qty"})
          .partialAggregation({}, {"sum(product_cost_qty) AS sum_cost_qty"})
          .localPartition({})
          .finalAggregation()
          .project({"sum_cost_qty * 0.0001 AS filter_cost_qty"})
          .planNode();

  // Query plan.
  auto nation =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationRowType, nationFileColumns, {nationNameFilter})
          .captureScanNodeId(nationScanId)
          .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanId)
          .hashJoin({"s_nationkey"}, {"n_nationkey"}, nation, "", {"s_suppkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanId)
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"ps_partkey", "ps_availqty", "ps_supplycost"})
          .project(
              {"ps_supplycost * ps_availqty AS product_cost_qty", "ps_partkey"})
          .partialAggregation(
              {"ps_partkey"}, {"sum(product_cost_qty) AS value"})
          .localPartition({"ps_partkey"})
          .finalAggregation()
          .nestedLoopJoin(
              subQueryPlan, {"ps_partkey", "value", "filter_cost_qty"})
          .filter("value > filter_cost_qty")
          .project({"ps_partkey", "value"})
          .orderBy({"value DESC"}, false)
          .planNode();

  // Create context and return.
  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partsuppScanIdSubQuery] = getTableFilePaths(kPartsupp);
  context.dataFiles[supplierScanIdSubQuery] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanIdSubQuery] = getTableFilePaths(kNation);
  context.dataFiles[partsuppScanId] = getTableFilePaths(kPartsupp);
  context.dataFiles[supplierScanId] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ12Plan() const {
  std::vector<std::string> ordersColumns = {"o_orderkey", "o_orderpriority"};
  std::vector<std::string> lineitemColumns = {
      "l_receiptdate",
      "l_orderkey",
      "l_commitdate",
      "l_shipmode",
      "l_shipdate"};

  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;

  const std::string receiptDateFilter = formatDateFilter(
      "l_receiptdate", lineitemSelectedRowType, "'1994-01-01'", "'1994-12-31'");
  const std::string shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "", "'1995-01-01'");
  const std::string commitDateFilter = formatDateFilter(
      "l_commitdate", lineitemSelectedRowType, "", "'1995-01-01'");

  auto lineitem = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kLineitem,
                          lineitemSelectedRowType,
                          lineitemFileColumns,
                          {receiptDateFilter,
                           "l_shipmode IN ('MAIL', 'SHIP')",
                           shipDateFilter,
                           commitDateFilter},
                          "l_commitdate < l_receiptdate")
                      .captureScanNodeId(lineitemScanNodeId)
                      .filter("l_shipdate < l_commitdate")
                      .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns, {})
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              lineitem,
              "",
              {"l_shipmode", "o_orderpriority"})
          .project(
              {"l_shipmode",
               "(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count_partial",
               "(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count_partial"})
          .partialAggregation(
              {"l_shipmode"},
              {"sum(high_line_count_partial) as high_line_count",
               "sum(low_line_count_partial) as low_line_count"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"l_shipmode"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ13Plan() const {
  const bool countOrderRows =
      FLAGS_tpch_q13_preaggregate && FLAGS_tpch_q13_count_rows;
  std::vector<std::string> ordersColumns = {"o_custkey", "o_comment"};
  if (!countOrderRows) {
    ordersColumns.push_back("o_orderkey");
  }
  std::vector<std::string> customerColumns = {"c_custkey"};

  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;

  auto customers =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .planNode();

  if (FLAGS_tpch_q13_preaggregate) {
    const bool pruneComment = (FLAGS_tpch_q13_prune_comment ||
                               FLAGS_tpch_prune_filter_only_columns) &&
        !filtersAsNode_;
    const auto orderOutputType = pruneComment
        ? getRowType(
              kOrders,
              countOrderRows
                  ? std::vector<std::string>{"o_custkey"}
                  : std::vector<std::string>{"o_custkey", "o_orderkey"})
        : ordersSelectedRowType;
    auto orderBuilder = PlanBuilder(planNodeIdGenerator, pool_.get());
    orderBuilder.filtersAsNode(filtersAsNode_)
        .tableScan(
            kOrders,
            orderOutputType,
            ordersFileColumns,
            {},
            "o_comment not like '%special%requests%'",
            pruneComment ? ordersSelectedRowType : nullptr)
        .captureScanNodeId(ordersScanNodeId);
    if (FLAGS_tpch_q13_raw_single) {
      VELOX_USER_CHECK(
          countOrderRows,
          "Q13 raw single aggregation requires tpch_q13_count_rows");
      if (FLAGS_tpch_q13_single_count) {
        orderBuilder.localPartition(std::vector<std::string>{})
            .singleAggregation({"o_custkey"}, {"count(0) AS order_count"});
      } else {
        orderBuilder.project({"o_custkey", "cast(1 as bigint) AS order_one"})
            .localPartition(std::vector<std::string>{})
            .singleAggregation(
                {"o_custkey"}, {"sum(order_one) AS order_count"});
      }
    } else {
      orderBuilder
          .partialAggregation(
              {"o_custkey"},
              {countOrderRows ? "count(0) AS order_count"
                              : "count(o_orderkey) AS order_count"})
          .localPartition({"o_custkey"})
          .finalAggregation();
    }
    auto orderCounts = orderBuilder.planNode();
    TpchPlan context;
    // c_custkey is the TPC-H customer primary key. Preaggregating the many-side
    // preserves each customer's count, including customers with no matching
    // orders, without materializing one joined row per order.
    context.plan =
        PlanBuilder(customers, planNodeIdGenerator, pool_.get())
            .hashJoin(
                {"c_custkey"},
                {"o_custkey"},
                orderCounts,
                "",
                {"c_custkey", "order_count"},
                core::JoinType::kLeft)
            .project({"coalesce(order_count, cast(0 as bigint)) AS c_count"})
            .partialAggregation({"c_count"}, {"count(0) AS custdist"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .orderBy({"custdist DESC", "c_count DESC"}, false)
            .planNode();
    context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
    context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
    context.dataFileFormat = format_;
    return context;
  }

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              ordersSelectedRowType,
              ordersFileColumns,
              {},
              "o_comment not like '%special%requests%'")
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"c_custkey", "o_orderkey"},
              core::JoinType::kRight)
          .partialAggregation({"c_custkey"}, {"count(o_orderkey) as c_count"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .singleAggregation({"c_count"}, {"count(0) as custdist"})
          .orderBy({"custdist DESC", "c_count DESC"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ14Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_partkey", "l_extendedprice", "l_discount", "l_shipdate"};
  std::vector<std::string> partColumns = {"p_partkey", "p_type"};

  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  const std::string shipDate = "l_shipdate";
  const std::string shipDateFilter = formatDateFilter(
      shipDate, lineitemSelectedRowType, "'1995-09-01'", "'1995-09-30'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId partScanNodeId;

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(kPart, partSelectedRowType, partFileColumns)
                  .captureScanNodeId(partScanNodeId)
                  .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {},
              shipDateFilter)
          .captureScanNodeId(lineitemScanNodeId)
          .project(
              {"l_extendedprice * (1.0 - l_discount) as part_revenue",
               "l_shipdate",
               "l_partkey"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"part_revenue", "p_type"})
          .project(
              {"(CASE WHEN (p_type LIKE 'PROMO%') THEN part_revenue ELSE 0.0 END) as filter_revenue",
               "part_revenue"})
          .partialAggregation(
              {},
              {"sum(part_revenue) as total_revenue",
               "sum(filter_revenue) as total_promo_revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project(
              {"100.00 * total_promo_revenue/total_revenue as promo_revenue"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ15Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"};
  std::vector<std::string> supplierColumns = {
      "s_suppkey", "s_name", "s_address", "s_phone"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  const std::string shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "'1996-01-01'", "'1996-03-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanNodeIdSubQuery;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId supplierScanNodeId;

  if (FLAGS_tpch_q15_window) {
    auto revenue =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(
                kLineitem,
                lineitemSelectedRowType,
                lineitemFileColumns,
                {shipDateFilter})
            .captureScanNodeId(lineitemScanNodeId)
            .project(
                {"l_suppkey AS supplier_no",
                 "l_extendedprice * (1.0 - l_discount) AS part_revenue"})
            .partialAggregation(
                {"supplier_no"}, {"sum(part_revenue) AS total_revenue"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .window({"max(total_revenue) OVER () AS max_revenue"})
            .filter("total_revenue = max_revenue")
            .project({"supplier_no", "total_revenue"})
            .planNode();
    TpchPlan context;
    context.plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
            .captureScanNodeId(supplierScanNodeId)
            .hashJoin(
                {"s_suppkey"},
                {"supplier_no"},
                revenue,
                "",
                {"s_suppkey",
                 "s_name",
                 "s_address",
                 "s_phone",
                 "total_revenue"})
            .orderBy({"s_suppkey"}, false)
            .planNode();
    context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
    context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
    context.dataFileFormat = format_;
    return context;
  }

  auto maxRevenue =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter})
          .captureScanNodeId(lineitemScanNodeId)
          .project(
              {"l_suppkey",
               "l_extendedprice * (1.0 - l_discount) as part_revenue"})
          .partialAggregation(
              {"l_suppkey"}, {"sum(part_revenue) as total_revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .singleAggregation({}, {"max(total_revenue) as max_revenue"})
          .planNode();

  auto supplierWithMaxRevenue =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter})
          .captureScanNodeId(lineitemScanNodeIdSubQuery)
          .project(
              {"l_suppkey as supplier_no",
               "l_extendedprice * (1.0 - l_discount) as part_revenue"})
          .partialAggregation(
              {"supplier_no"}, {"sum(part_revenue) as total_revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .hashJoin(
              {"total_revenue"},
              {"max_revenue"},
              maxRevenue,
              "",
              {"supplier_no", "total_revenue"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_suppkey"},
              {"supplier_no"},
              supplierWithMaxRevenue,
              "",
              {"s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"})
          .orderBy({"s_suppkey"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeIdSubQuery] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ16Plan() const {
  std::vector<std::string> partColumns = {
      "p_brand", "p_type", "p_size", "p_partkey"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_comment"};
  std::vector<std::string> partsuppColumns = {"ps_partkey", "ps_suppkey"};

  const auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  const auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId partScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId partsuppScanNodeId;

  // Keep the IN-list literals as INTEGER so they match the INTEGER p_size
  // column; widening them to BIGINT would cast the column and block the
  // subfield filter pushdown.
  parse::ParseOptions parseOptions;
  parseOptions.parseIntegerAsBigint = false;
  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .setParseOptions(parseOptions)
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partSelectedRowType,
                      partFileColumns,
                      {"p_size in (49, 14, 23, 45, 19, 3, 36, 9)"},
                      "p_type NOT LIKE 'MEDIUM POLISHED%'")
                  .captureScanNodeId(partScanNodeId)
                  // Neq is unsupported as a tableScan subfield filter for
                  // Parquet source.
                  .filter("p_brand <> 'Brand#45'")
                  .planNode();

  auto supplier = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kSupplier,
                          supplierSelectedRowType,
                          supplierFileColumns,
                          {},
                          "s_comment LIKE '%Customer%Complaints%'")
                      .captureScanNodeId(supplierScanNodeId)
                      .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanNodeId)
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
              {"ps_suppkey", "p_brand", "p_type", "p_size"},
              core::JoinType::kAnti,
              true /*nullAware*/)
          // Empty aggregate is used here to get the distinct count of
          // ps_suppkey.
          // approx_distinct could be used instead for getting the count of
          // distinct ps_suppkey but since approx_distinct is non deterministic
          // and the standard error can not be set to 0, it is not used here.
          .partialAggregation({"p_brand", "p_type", "p_size", "ps_suppkey"}, {})
          .localPartition({"p_brand", "p_type", "p_size", "ps_suppkey"})
          .finalAggregation()
          .partialAggregation(
              {"p_brand", "p_type", "p_size"},
              {"count(ps_suppkey) as supplier_cnt"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"supplier_cnt DESC", "p_brand", "p_type", "p_size"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanNodeId] = getTableFilePaths(kPartsupp);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ17Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_partkey", "l_extendedprice", "l_quantity"};
  std::vector<std::string> partColumns = {
      "p_partkey", "p_brand", "p_container"};

  auto lineitemRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanId;
  core::PlanNodeId lineitemAggScanId;
  core::PlanNodeId partScanId;
  core::PlanNodeId partAggScanId;

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partRowType,
                      partFileColumns,
                      {"p_brand = 'Brand#23'", "p_container = 'MED BOX'"})
                  .captureScanNodeId(partScanId)
                  .planNode();

  auto partAgg =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kPart,
              partRowType,
              partFileColumns,
              FLAGS_tpch_q17_filter_aggregation
                  ? std::vector<
                        std::
                            string>{"p_brand = 'Brand#23'", "p_container = 'MED BOX'"}
                  : std::vector<std::string>{})
          .captureScanNodeId(partAggScanId)
          .planNode();

  auto lineitemJoinPart =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanId)
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"l_quantity", "p_partkey", "l_extendedprice"})
          .planNode();

  if (FLAGS_tpch_q17_window) {
    TpchPlan context;
    // The selected part key is unique in part, so this window sees exactly the
    // same lineitems as the correlated AVG, without rescanning lineitem.
    context.plan =
        PlanBuilder(lineitemJoinPart, planNodeIdGenerator, pool_.get())
            .localPartition(std::vector<std::string>{})
            .window(
                {"avg(l_quantity) OVER (PARTITION BY p_partkey) AS avg_quantity"})
            .filter("l_quantity < 0.2 * avg_quantity")
            .singleAggregation({}, {"sum(l_extendedprice) AS partial_sum"})
            .project({"partial_sum / 7.0 AS avg_yearly"})
            .planNode();
    context.dataFiles[lineitemScanId] = getTableFilePaths(kLineitem);
    context.dataFiles[partScanId] = getTableFilePaths(kPart);
    context.dataFileFormat = format_;
    return context;
  }

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemAggScanId)
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              partAgg,
              "",
              {"l_partkey", "l_quantity"})
          .partialAggregation({"l_partkey"}, {"avg(l_quantity) as avg_"})
          .localPartition({"l_partkey"})
          .finalAggregation()
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              lineitemJoinPart,
              "l_quantity < 0.2 * avg_",
              {"l_extendedprice"})
          .partialAggregation({}, {"sum(l_extendedprice) as partial_sum"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"(partial_sum / 7.0) as avg_yearly"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanId] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitemAggScanId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanId] = getTableFilePaths(kPart);
  context.dataFiles[partAggScanId] = getTableFilePaths(kPart);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ18Plan() const {
  std::vector<std::string> lineitemColumns = {"l_orderkey", "l_quantity"};
  std::vector<std::string> ordersColumns = {
      "o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"};
  std::vector<std::string> customerColumns = {"c_name", "c_custkey"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);

  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;

  auto bigOrdersBuilder =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId);
  if (FLAGS_tpch_q18_complete_groups) {
    std::string completeBatchOption;
    VELOX_USER_CHECK(
        gflags::GetCommandLineOption(
            "cudf_groupby_complete_batches", &completeBatchOption) &&
            completeBatchOption == "true",
        "Q18 complete groups requires --cudf_groupby_complete_batches=true for runtime validation");
    bigOrdersBuilder
        .singleAggregation({"l_orderkey"}, {"sum(l_quantity) AS quantity"})
        .addNode([](std::string, core::PlanNodePtr node) {
          auto aggregation =
              std::dynamic_pointer_cast<const core::AggregationNode>(node);
          return core::AggregationNode::Builder(*aggregation)
              .preGroupedKeys(aggregation->groupingKeys())
              .noGroupsSpanBatches(true)
              .build();
        });
  } else {
    bigOrdersBuilder
        .partialAggregation({"l_orderkey"}, {"sum(l_quantity) AS quantity"})
        .localPartition({"l_orderkey"})
        .finalAggregation();
  }
  auto bigOrders = bigOrdersBuilder.filter("quantity > 300.0").planNode();

  if (FLAGS_tpch_q18_late_customer) {
    // TPC-H orders reference exactly one customer. The ORDER BY uses only
    // order attributes, so the top 100 can be chosen before fetching names.
    auto topOrders =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
            .captureScanNodeId(ordersScanNodeId)
            .hashJoin(
                {"o_orderkey"},
                {"l_orderkey"},
                bigOrders,
                "",
                {"o_orderkey",
                 "o_custkey",
                 "o_orderdate",
                 "o_totalprice",
                 "quantity"})
            .localPartition(std::vector<std::string>{})
            .topN({"o_totalprice DESC", "o_orderdate"}, 100, false)
            .planNode();
    TpchPlan context;
    context.plan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
            .captureScanNodeId(customerScanNodeId)
            .hashJoin(
                {"c_custkey"},
                {"o_custkey"},
                topOrders,
                "",
                {"c_name",
                 "c_custkey",
                 "o_orderkey",
                 "o_orderdate",
                 "o_totalprice",
                 "quantity"})
            .localPartition(std::vector<std::string>{})
            .orderBy({"o_totalprice DESC", "o_orderdate"}, false)
            .planNode();
    context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
    context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
    context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
    context.dataFileFormat = format_;
    return context;
  }

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              bigOrders,
              "",
              {"o_orderkey",
               "o_custkey",
               "o_orderdate",
               "o_totalprice",
               "l_orderkey",
               "quantity"})
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kCustomer, customerSelectedRowType, customerFileColumns)
                  .captureScanNodeId(customerScanNodeId)
                  .planNode(),
              "",
              {"c_name",
               "c_custkey",
               "o_orderkey",
               "o_orderdate",
               "o_totalprice",
               "quantity"})
          .localPartition(std::vector<std::string>{})
          .orderBy({"o_totalprice DESC", "o_orderdate"}, false)
          .limit(0, 100, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ19Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_partkey",
      "l_shipmode",
      "l_shipinstruct",
      "l_extendedprice",
      "l_discount",
      "l_quantity"};
  std::vector<std::string> partColumns = {
      "p_partkey", "p_brand", "p_container", "p_size"};

  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId partScanNodeId;

  const std::string shipModeFilter = "l_shipmode IN ('AIR', 'AIR REG')";
  const std::string shipInstructFilter =
      "(l_shipinstruct = 'DELIVER IN PERSON')";
  // These are necessary conditions of the complete residual predicate below.
  // Retain that predicate to preserve each brand's exact quantity interval.
  const bool pushdown =
      FLAGS_tpch_q19_pushdown || FLAGS_tpch_q19_numeric_predicate;
  const bool pruneFilterColumns =
      FLAGS_tpch_prune_filter_only_columns && !filtersAsNode_;
  const auto lineitemOutputType = pruneFilterColumns && pushdown
      ? getRowType(
            kLineitem,
            {"l_partkey", "l_extendedprice", "l_discount", "l_quantity"})
      : lineitemSelectedRowType;
  const auto partOutputType =
      pruneFilterColumns && FLAGS_tpch_q19_numeric_predicate
      ? getRowType(kPart, {"p_partkey", "p_brand"})
      : partSelectedRowType;
  const std::string partFilter = pushdown
      ? "((p_brand = 'Brand#12' AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG') AND p_size BETWEEN 1 AND 5)"
        " OR (p_brand = 'Brand#23' AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK') AND p_size BETWEEN 1 AND 10)"
        " OR (p_brand = 'Brand#34' AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG') AND p_size BETWEEN 1 AND 15))"
      : "";
  std::vector<std::string> lineitemFilters{shipModeFilter, shipInstructFilter};
  std::vector<std::string> lineitemProjection{
      "l_extendedprice * (1.0 - l_discount) as part_revenue",
      "l_shipmode",
      "l_shipinstruct",
      "l_partkey",
      "l_quantity"};
  if (pushdown) {
    lineitemFilters.push_back("l_quantity between 1.0 and 30.0");
    lineitemProjection = {
        "l_extendedprice * (1.0 - l_discount) as part_revenue",
        "l_partkey",
        "l_quantity"};
  }
  std::string joinFilterExpr =
      "     ((p_brand = 'Brand#12')"
      "     AND (l_quantity between 1.0 and 11.0)"
      "     AND (p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG'))"
      "     AND (p_size BETWEEN 1 AND 5))"
      " OR  ((p_brand ='Brand#23')"
      "     AND (p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK'))"
      "     AND (l_quantity between 10.0 and 20.0)"
      "     AND (p_size BETWEEN 1 AND 10))"
      " OR  ((p_brand = 'Brand#34')"
      "     AND (p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG'))"
      "     AND (l_quantity between 20.0 and 30.0)"
      "     AND (p_size BETWEEN 1 AND 15))";

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partOutputType,
                      partFileColumns,
                      {},
                      partFilter,
                      pruneFilterColumns ? partSelectedRowType : nullptr)
                  .captureScanNodeId(partScanNodeId)
                  .planNode();

  if (FLAGS_tpch_q19_numeric_predicate) {
    // The pushed filter has already checked the disjoint brand/container/size
    // branches. Only each surviving part's quantity interval remains to be
    // checked on the lineitem side. Drop all strings from the join input.
    part =
        PlanBuilder(part, planNodeIdGenerator, pool_.get())
            .project(
                {"p_partkey",
                 "CASE WHEN p_brand = 'Brand#12' THEN 1.0 WHEN p_brand = 'Brand#23' THEN 10.0 ELSE 20.0 END AS min_qty",
                 "CASE WHEN p_brand = 'Brand#12' THEN 11.0 WHEN p_brand = 'Brand#23' THEN 20.0 ELSE 30.0 END AS max_qty"})
            .planNode();
    joinFilterExpr = "l_quantity >= min_qty AND l_quantity <= max_qty";
  }

  auto factBuilder = PlanBuilder(planNodeIdGenerator, pool_.get());
  factBuilder.filtersAsNode(filtersAsNode_)
      .tableScan(
          kLineitem,
          lineitemOutputType,
          lineitemFileColumns,
          lineitemFilters,
          "",
          pruneFilterColumns ? lineitemSelectedRowType : nullptr)
      .captureScanNodeId(lineitemScanNodeId);
  if (!FLAGS_tpch_q19_late_revenue) {
    factBuilder.project(lineitemProjection);
  }
  factBuilder.hashJoin(
      {"l_partkey"},
      {"p_partkey"},
      part,
      joinFilterExpr,
      FLAGS_tpch_q19_late_revenue
          ? std::vector<std::string>{"l_extendedprice", "l_discount"}
          : std::vector<std::string>{"part_revenue"});
  if (FLAGS_tpch_q19_late_revenue) {
    factBuilder.project(
        {"l_extendedprice * (1.0 - l_discount) as part_revenue"});
  }
  auto plan =
      factBuilder.partialAggregation({}, {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ20Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_shipdate", "l_suppkey", "l_partkey", "l_quantity"};
  std::vector<std::string> partColumns = {"p_partkey", "p_name"};
  std::vector<std::string> supplierColumns = {
      "s_nationkey", "s_address", "s_name", "s_suppkey"};
  std::vector<std::string> partsuppColumns = {
      "ps_availqty", "ps_partkey", "ps_suppkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  const std::string shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "'1994-01-01'", "'1994-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanId;
  core::PlanNodeId partScanId;
  core::PlanNodeId partAggScanId;
  core::PlanNodeId supplierScanId;
  core::PlanNodeId partsuppScanId;
  core::PlanNodeId nationScanId;

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partSelectedRowType,
                      partFileColumns,
                      {},
                      "p_name like 'forest%'")
                  .captureScanNodeId(partScanId)
                  .planNode();

  auto partAgg = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .filtersAsNode(filtersAsNode_)
                     .tableScan(
                         kPart,
                         partSelectedRowType,
                         partFileColumns,
                         {},
                         "p_name like 'forest%'")
                     .captureScanNodeId(partAggScanId)
                     .planNode();

  auto nation = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kNation,
                        nationSelectedRowType,
                        nationFileColumns,
                        {"n_name = 'CANADA'"})
                    .captureScanNodeId(nationScanId)
                    .planNode();

  auto partsuppJoinPart =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanId)
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              part,
              "",
              {"ps_partkey", "ps_suppkey", "ps_availqty"},
              core::JoinType::kLeftSemiFilter)
          .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"s_name", "s_address", "s_suppkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter})
          .captureScanNodeId(lineitemScanId)
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              partAgg,
              "",
              {"l_partkey", "l_suppkey", "l_quantity"})
          .partialAggregation(
              {"l_partkey", "l_suppkey"}, {"sum(l_quantity) AS sum_qty"})
          .localPartition({"l_partkey", "l_suppkey"})
          .finalAggregation()
          .project({"l_partkey", "l_suppkey", "0.5 * sum_qty AS filter_qty"})
          .hashJoin(
              {"l_partkey", "l_suppkey"},
              {"ps_partkey", "ps_suppkey"},
              partsuppJoinPart,
              "ps_availqty > filter_qty",
              {"ps_suppkey"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"s_name", "s_address"},
              core::JoinType::kRightSemiFilter)
          .orderBy({"s_name"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanId] = getTableFilePaths(kPart);
  context.dataFiles[partAggScanId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanId] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanId] = getTableFilePaths(kPartsupp);
  context.dataFiles[nationScanId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ21Plan() const {
  if (FLAGS_tpch_q21_candidate_summary) {
    return getQ21CandidateSummaryPlan();
  }
  std::vector<std::string> supplierColumns = {
      "s_nationkey", "s_name", "s_suppkey"};
  std::vector<std::string> lineitemColumnsWithDates = {
      "l_suppkey", "l_commitdate", "l_orderkey", "l_receiptdate"};
  std::vector<std::string> lineitemColumns = {"l_suppkey", "l_orderkey"};
  std::vector<std::string> ordersColumns = {"o_orderkey", "o_orderstatus"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto supplierRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto lineitem1RowType = getRowType(kLineitem, lineitemColumnsWithDates);
  const auto& lineitem1FileColumns = getFileColumnNames(kLineitem);
  auto lineitem2RowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitem2FileColumns = getFileColumnNames(kLineitem);
  auto lineitem3RowType = getRowType(kLineitem, lineitemColumnsWithDates);
  const auto& lineitem3FileColumns = getFileColumnNames(kLineitem);
  auto ordersRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto nationRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitem1ScanNodeId;
  core::PlanNodeId lineitem2ScanNodeId;
  core::PlanNodeId lineitem3ScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId nationScanNodeId;
  const std::string receiptCommitFilter = "l_receiptdate > l_commitdate";

  auto lineitem3 =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitem3RowType,
              lineitem3FileColumns,
              {},
              receiptCommitFilter)
          .captureScanNodeId(lineitem3ScanNodeId)
          .project({"l_orderkey as l_orderkey_3", "l_suppkey as l_suppkey_3"})
          .planNode();

  auto nation = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kNation,
                        nationRowType,
                        nationFileColumns,
                        {"n_name = 'SAUDI ARABIA'"})
                    .captureScanNodeId(nationScanNodeId)
                    .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"s_suppkey", "s_name", "s_nationkey"})
          .planNode();

  auto lineitemJoinSupplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitem1RowType,
              lineitem1FileColumns,
              {},
              receiptCommitFilter)
          .captureScanNodeId(lineitem1ScanNodeId)
          .project({"l_orderkey as l_orderkey_1", "l_suppkey as l_suppkey_1"})
          .hashJoin(
              {"l_suppkey_1"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"l_orderkey_1", "s_nationkey", "l_suppkey_1", "s_name"})
          .planNode();

  auto ordersJoinLineitem1 =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              ordersRowType,
              ordersFileColumns,
              {"o_orderstatus = 'F'"})
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey_1"},
              lineitemJoinSupplier,
              "",
              {"s_nationkey", "l_orderkey_1", "l_suppkey_1", "s_name"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitem2RowType, lineitem2FileColumns)
          .captureScanNodeId(lineitem2ScanNodeId)
          .project({"l_orderkey as l_orderkey_2", "l_suppkey as l_suppkey_2"})
          .hashJoin(
              {"l_orderkey_2"},
              {"l_orderkey_1"},
              ordersJoinLineitem1,
              "l_suppkey_2 <> l_suppkey_1",
              {"l_orderkey_1", "l_suppkey_1", "s_name"},
              core::JoinType::kRightSemiFilter)
          .hashJoin(
              {"l_orderkey_1"},
              {"l_orderkey_3"},
              lineitem3,
              "l_suppkey_3 <> l_suppkey_1",
              {"s_name"},
              core::JoinType::kAnti,
              false /*nullAware*/)
          .partialAggregation({"s_name"}, {"count(1) as numwait"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"numwait DESC", "s_name"}, false)
          .limit(0, 100, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[lineitem1ScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitem2ScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitem3ScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ21CandidateSummaryPlan() const {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  TpchPlan context;
  context.dataFileFormat = format_;
  core::PlanNodeId nationScan;
  core::PlanNodeId supplierScan;
  core::PlanNodeId candidateScan;
  core::PlanNodeId ordersScan;
  core::PlanNodeId summaryScan;
  core::PlanNodeId namesScan;

  auto nation = PlanBuilder(ids, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kNation,
                        getRowType(kNation, {"n_nationkey", "n_name"}),
                        getFileColumnNames(kNation),
                        {"n_name = 'SAUDI ARABIA'"})
                    .captureScanNodeId(nationScan)
                    .planNode();
  auto saudiSuppliers =
      PlanBuilder(ids, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kSupplier,
              getRowType(kSupplier, {"s_suppkey", "s_nationkey"}),
              getFileColumnNames(kSupplier))
          .captureScanNodeId(supplierScan)
          .hashJoin({"s_nationkey"}, {"n_nationkey"}, nation, "", {"s_suppkey"})
          .planNode();
  const std::vector<std::string> lineColumns{
      "l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"};
  const bool pruneCandidateDates =
      FLAGS_tpch_q21_prune_candidate_dates && !filtersAsNode_;
  auto candidates =
      PlanBuilder(ids, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              getRowType(
                  kLineitem,
                  pruneCandidateDates
                      ? std::vector<std::string>{"l_orderkey", "l_suppkey"}
                      : lineColumns),
              getFileColumnNames(kLineitem),
              {},
              "l_receiptdate > l_commitdate",
              pruneCandidateDates ? getRowType(kLineitem, lineColumns)
                                  : nullptr)
          .captureScanNodeId(candidateScan)
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              saudiSuppliers,
              "",
              {"l_orderkey", "l_suppkey"})
          .planNode();
  auto candidateOrders =
      PlanBuilder(ids, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              getRowType(kOrders, {"o_orderkey", "o_orderstatus"}),
              getFileColumnNames(kOrders),
              {"o_orderstatus = 'F'"})
          .captureScanNodeId(ordersScan)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              candidates,
              "",
              {"l_orderkey", "l_suppkey"})
          .partialAggregation(
              {"l_orderkey", "l_suppkey"}, {"count(1) AS candidate_count"})
          .localPartition({"l_orderkey", "l_suppkey"})
          .finalAggregation()
          .project(
              {"l_orderkey AS c_orderkey",
               "l_suppkey AS c_suppkey",
               "candidate_count"})
          .planNode();

  auto names = PlanBuilder(ids, pool_.get())
                   .filtersAsNode(filtersAsNode_)
                   .tableScan(
                       kSupplier,
                       getRowType(kSupplier, {"s_suppkey", "s_name"}),
                       getFileColumnNames(kSupplier))
                   .captureScanNodeId(namesScan)
                   .planNode();

  // For each (order, candidate supplier), preserve the number of original late
  // lineitems. Joining a grouped candidate to all order lines cannot multiply
  // that count. A different supplier exists iff min(supp) != max(supp), and no
  // different late supplier exists iff min(late_supp) == max(late_supp). The
  // candidate itself is late, so the late summary is never empty. TPC-H
  // supplier and order keys are non-null. This avoids buffering the entire
  // probe relation in a right-semi join or building all late lineitems for an
  // anti join.
  auto summaryBuilder = PlanBuilder(ids, pool_.get());
  summaryBuilder.filtersAsNode(filtersAsNode_)
      .tableScan(
          kLineitem,
          getRowType(kLineitem, lineColumns),
          getFileColumnNames(kLineitem))
      .captureScanNodeId(summaryScan);
  if (!FLAGS_tpch_q21_late_supplier_projection) {
    summaryBuilder.project(
        {"l_orderkey",
         "l_suppkey",
         "if(l_receiptdate > l_commitdate, l_suppkey, "
         "cast(null as bigint)) AS late_suppkey"});
  }
  std::vector<std::string> summaryJoinOutput{
      "c_orderkey", "c_suppkey", "candidate_count", "l_suppkey"};
  if (FLAGS_tpch_q21_late_supplier_projection) {
    summaryJoinOutput.push_back("l_receiptdate");
    summaryJoinOutput.push_back("l_commitdate");
  } else {
    summaryJoinOutput.push_back("late_suppkey");
  }
  summaryBuilder.hashJoin(
      {"l_orderkey"}, {"c_orderkey"}, candidateOrders, "", summaryJoinOutput);
  if (FLAGS_tpch_q21_late_supplier_projection) {
    summaryBuilder.project(
        {"c_orderkey",
         "c_suppkey",
         "candidate_count",
         "l_suppkey",
         "if(l_receiptdate > l_commitdate, l_suppkey, "
         "cast(null as bigint)) AS late_suppkey"});
  }
  context.plan =
      summaryBuilder
          .partialAggregation(
              {"c_orderkey", "c_suppkey", "candidate_count"},
              {"min(l_suppkey) AS min_supp",
               "max(l_suppkey) AS max_supp",
               "min(late_suppkey) AS min_late",
               "max(late_suppkey) AS max_late"})
          .localPartition({"c_orderkey", "c_suppkey", "candidate_count"})
          .finalAggregation()
          .filter("min_supp <> max_supp AND min_late = max_late")
          .partialAggregation(
              {"c_suppkey"}, {"sum(candidate_count) AS numwait"})
          .localPartition({"c_suppkey"})
          .finalAggregation()
          .hashJoin(
              {"c_suppkey"}, {"s_suppkey"}, names, "", {"s_name", "numwait"})
          .partialAggregation({"s_name"}, {"sum(numwait) AS numwait"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"numwait DESC", "s_name"}, false)
          .limit(0, 100, false)
          .planNode();
  context.dataFiles[nationScan] = getTableFilePaths(kNation);
  context.dataFiles[supplierScan] = getTableFilePaths(kSupplier);
  context.dataFiles[candidateScan] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScan] = getTableFilePaths(kOrders);
  context.dataFiles[summaryScan] = getTableFilePaths(kLineitem);
  context.dataFiles[namesScan] = getTableFilePaths(kSupplier);
  return context;
}

TpchPlan TpchQueryBuilder::getQ22Plan() const {
  std::vector<std::string> ordersColumns = {"o_custkey"};
  std::vector<std::string> customerColumns = {"c_acctbal", "c_phone"};
  std::vector<std::string> customerColumnsWithKey = {
      "c_custkey", "c_acctbal", "c_phone"};

  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  const auto customerSelectedRowTypeWithKey =
      getRowType(kCustomer, customerColumnsWithKey);
  const auto& customerFileColumnsWithKey = getFileColumnNames(kCustomer);

  const std::string phoneFilter =
      "substr(c_phone, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17')";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId customerScanNodeIdWithKey;
  core::PlanNodeId ordersScanNodeId;

  auto orders =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .planNode();

  auto customerAvgAccountBalance =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kCustomer,
              customerSelectedRowType,
              customerFileColumns,
              {"c_acctbal > 0.0"},
              phoneFilter)
          .captureScanNodeId(customerScanNodeId)
          .partialAggregation({}, {"avg(c_acctbal) as avg_acctbal"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kCustomer,
              customerSelectedRowTypeWithKey,
              customerFileColumnsWithKey,
              {},
              phoneFilter)
          .captureScanNodeId(customerScanNodeIdWithKey)
          .nestedLoopJoin(
              customerAvgAccountBalance,
              {"c_acctbal", "avg_acctbal", "c_custkey", "c_phone"})
          .filter("c_acctbal > avg_acctbal")
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              orders,
              "",
              {"c_acctbal", "c_phone"},
              core::JoinType::kAnti,
              false /*nullAware*/)
          .project({"substr(c_phone, 1, 2) AS country_code", "c_acctbal"})
          .partialAggregation(
              {"country_code"},
              {"count(0) AS numcust", "sum(c_acctbal) AS totacctbal"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"country_code"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[customerScanNodeIdWithKey] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getIoMeterPlan(int columnPct) const {
  VELOX_CHECK(columnPct > 0 && columnPct <= 100);
  auto columns = getFileColumnNames(kLineitem);
  std::vector<std::string> names;
  for (auto& pair : columns) {
    names.push_back(pair.first);
  }
  std::sort(names.begin(), names.end());
  names.resize(names.size() * columnPct / 100);
  if (std::find(names.begin(), names.end(), "l_partkey") == names.end()) {
    names.push_back("l_partkey");
  }

  const auto selectedRowType = getRowType(kLineitem, names);
  std::vector<std::string> aggregates;
  std::vector<std::string> projectExprs;

  for (auto i = 0; i < selectedRowType->size(); ++i) {
    if (selectedRowType->childAt(i)->kind() == TypeKind::VARCHAR) {
      projectExprs.push_back(
          fmt::format("length({}) as l{}", selectedRowType->nameOf(i), i));
      aggregates.push_back(fmt::format("max(l{})", i));
    } else {
      projectExprs.push_back(selectedRowType->nameOf(i));
      aggregates.push_back(fmt::format("max({})", selectedRowType->nameOf(i)));
    }
  }

  std::string filter = "l_partkey between 2000000 and 2500000";

  core::PlanNodeId lineitemPlanNodeId;
  std::unordered_map<std::string, std::string> aliases;
  for (auto& name : names) {
    aliases[name] = name;
  }
  auto plan = PlanBuilder(pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(kLineitem, selectedRowType, aliases, {filter})
                  .captureScanNodeId(lineitemPlanNodeId)
                  .project(projectExprs)
                  .partialAggregation({}, aggregates)
                  .localPartition(std::vector<std::string>{})
                  .finalAggregation()
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

const std::vector<std::string> TpchQueryBuilder::kTableNames_ = {
    kLineitem,
    kOrders,
    kCustomer,
    kNation,
    kRegion,
    kPart,
    kSupplier,
    kPartsupp};

const std::unordered_map<std::string, std::vector<std::string>>
    TpchQueryBuilder::kTables_ = {
        std::make_pair(
            "lineitem",
            tpch::getTableSchema(tpch::Table::TBL_LINEITEM)->names()),
        std::make_pair(
            "orders",
            tpch::getTableSchema(tpch::Table::TBL_ORDERS)->names()),
        std::make_pair(
            "customer",
            tpch::getTableSchema(tpch::Table::TBL_CUSTOMER)->names()),
        std::make_pair(
            "nation",
            tpch::getTableSchema(tpch::Table::TBL_NATION)->names()),
        std::make_pair(
            "region",
            tpch::getTableSchema(tpch::Table::TBL_REGION)->names()),
        std::make_pair(
            "part",
            tpch::getTableSchema(tpch::Table::TBL_PART)->names()),
        std::make_pair(
            "supplier",
            tpch::getTableSchema(tpch::Table::TBL_SUPPLIER)->names()),
        std::make_pair(
            "partsupp",
            tpch::getTableSchema(tpch::Table::TBL_PARTSUPP)->names())};

} // namespace facebook::velox::exec::test
