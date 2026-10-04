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

// Normalization of self-describing decimal aggregate state where batches meet:
// a logical VARBINARY column may be physically a STRING blob or a STRUCT of
// any shape, and the concat funnel (getConcatenatedTable,
// getConcatenatedTableBatched, getConcatenatedCudfVectorsBatched,
// normalizeDecimalStateBatches) and the FINAL aggregation must accept every
// mix of forms.

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/tests/DecimalStateTestColumns.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox::test {
namespace {

using exec::test::AssertQueryBuilder;
using exec::test::PlanBuilder;

constexpr int32_t kScale = 2;

rmm::device_async_resource_ref testMr() {
  return cudf::get_current_device_resource_ref();
}

// Deterministic rows with nonzero overflows; every fourth row (from
// `nullPhase`) is null.
DecimalStateRows makeRows(int64_t base, size_t numRows, size_t nullPhase = 3) {
  DecimalStateRows rows;
  for (size_t i = 0; i < numRows; ++i) {
    const auto value = base + static_cast<int64_t>(i);
    rows.sums.push_back(static_cast<int128_t>(value) * 1'000'003 - 77);
    rows.counts.push_back(2 + (value % 5));
    rows.overflows.push_back(value % 3);
    rows.valid.push_back((i % 4) != nullPhase);
  }
  return rows;
}

// True if plan node `planNodeId` of `task` recorded the custom stat `name`.
bool hasCustomStat(
    const std::shared_ptr<exec::Task>& task,
    const core::PlanNodeId& planNodeId,
    std::string_view name) {
  const auto planStats = exec::toPlanStats(task->taskStats());
  const auto it = planStats.find(planNodeId);
  return it != planStats.end() &&
      it->second.customStats.count(std::string{name}) > 0;
}

// Physical form of one batch's state column. A blob carries every field.
struct Form {
  bool blob;
  DecimalStateShape shape;
};

constexpr Form kBlob{true, DecimalStateShape::kSum64};

constexpr Form structOf(DecimalStateShape shape) {
  return Form{false, shape};
}

std::unique_ptr<cudf::column> makeStateColumn(
    const DecimalStateRows& rows,
    Form form,
    cuda::stream_ref stream) {
  return form.blob
      ? makeDecimalStateBlob(rows, stream)
      : makeDecimalStateStruct(rows, form.shape, kScale, stream, testMr());
}

// The rows as a column of `form` reports them.
DecimalStateRows observedAs(const DecimalStateRows& rows, Form form) {
  return form.blob ? rows : rows.as(form.shape);
}

class DecimalStateConcatTest : public exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    exec::test::OperatorTestBase::SetUp();
    parse::registerTypeResolver();
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    if (!initCudaDevice()) {
      GTEST_SKIP() << "No CUDA device available";
    }
    savedConfig_ = CudfConfig::getInstance();
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
  }

  void TearDown() override {
    if (savedConfig_.has_value()) {
      unregisterCudf();
      CudfConfig::getInstance() = *savedConfig_;
    }
    exec::test::OperatorTestBase::TearDown();
  }

  // ROW(k BIGINT, s VARBINARY).
  RowTypePtr keyStateType() const {
    return ROW({"k", "s"}, {BIGINT(), VARBINARY()});
  }

  // One batch of keyStateType() with keys [keyBase, keyBase + rows.size()).
  CudfVectorPtr
  makeKeyStateBatch(int64_t keyBase, const DecimalStateRows& rows, Form form) {
    std::vector<int64_t> keys(rows.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      keys[i] = keyBase + static_cast<int64_t>(i);
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(makeInt64Column(keys, nullptr, stream_));
    columns.push_back(makeStateColumn(rows, form, stream_));
    return makeVector(keyStateType(), std::move(columns));
  }

  CudfVectorPtr makeVector(
      const RowTypePtr& type,
      std::vector<std::unique_ptr<cudf::column>> columns) {
    auto table = std::make_unique<cudf::table>(std::move(columns));
    const auto size = static_cast<vector_size_t>(table->num_rows());
    return std::make_shared<CudfVector>(
        pool(), type, size, std::move(table), stream_);
  }

  // A zero-row keyStateType() table. VARBINARY is STRING, as CudfFromVelox and
  // the operators' empty outputs build it.
  std::unique_ptr<cudf::table> makeEmptyKeyStateTable() {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(cudf::make_empty_column(cudf::type_id::INT64));
    columns.push_back(cudf::make_empty_column(cudf::type_id::STRING));
    return std::make_unique<cudf::table>(std::move(columns));
  }

  CudfVectorPtr makeEmptyBatch() {
    return std::make_shared<CudfVector>(
        pool(), keyStateType(), 0, makeEmptyKeyStateTable(), stream_);
  }

  enum class Funnel { kTable, kTableBatched, kCudfVectorsBatched };

  // Runs `batches` through one funnel entry point and returns its output
  // tables.
  std::vector<std::unique_ptr<cudf::table>> runFunnel(
      Funnel funnel,
      std::vector<CudfVectorPtr> batches,
      const RowTypePtr& type) {
    std::vector<std::unique_ptr<cudf::table>> tables;
    switch (funnel) {
      case Funnel::kTable:
        tables.push_back(
            getConcatenatedTable(std::move(batches), type, stream_, testMr()));
        break;
      case Funnel::kTableBatched:
        tables = getConcatenatedTableBatched(
            std::move(batches), type, stream_, testMr());
        break;
      case Funnel::kCudfVectorsBatched: {
        auto vectors = getConcatenatedCudfVectorsBatched(
            pool(), std::move(batches), type, stream_, testMr());
        for (auto& vector : vectors) {
          EXPECT_EQ(vector->size(), vector->getTableView().num_rows());
          tables.push_back(vector->release());
        }
        break;
      }
    }
    stream_.sync();
    return tables;
  }

  static std::vector<Funnel> allFunnels() {
    return {Funnel::kTable, Funnel::kTableBatched, Funnel::kCudfVectorsBatched};
  }

  DecimalStateRows readState(const cudf::column_view& column) {
    return readDecimalState(column, kScale, stream_, testMr());
  }

  // Builds key/state batches from (rows, form) pairs, runs each funnel and
  // checks the keys (order), the state values and nulls, and the result form:
  // STRING when `expectedShape` is not set, else a struct of that shape.
  void checkMix(
      const std::vector<std::pair<DecimalStateRows, Form>>& inputs,
      std::optional<DecimalStateShape> expectedShape) {
    for (auto funnel : allFunnels()) {
      SCOPED_TRACE(static_cast<int>(funnel));
      std::vector<CudfVectorPtr> batches;
      DecimalStateRows expected;
      std::vector<int64_t> expectedKeys;
      int64_t keyBase = 0;
      for (const auto& [rows, form] : inputs) {
        batches.push_back(makeKeyStateBatch(keyBase, rows, form));
        expected.append(observedAs(rows, form));
        for (size_t i = 0; i < rows.size(); ++i) {
          expectedKeys.push_back(keyBase + static_cast<int64_t>(i));
        }
        keyBase += 1000;
      }
      auto tables = runFunnel(funnel, std::move(batches), keyStateType());
      ASSERT_EQ(tables.size(), 1);
      const auto view = tables[0]->view();
      ASSERT_EQ(view.num_columns(), 2);
      EXPECT_EQ(copyColumnData<int64_t>(view.column(0), stream_), expectedKeys);
      const auto state = view.column(1);
      if (expectedShape) {
        ASSERT_TRUE(isDecimalStateStruct(state));
        EXPECT_EQ(decimalStateShapeOf(state), *expectedShape);
        expectDecimalStateRowsEqual(
            expected.as(*expectedShape), readState(state));
      } else {
        EXPECT_EQ(state.type().id(), cudf::type_id::STRING);
        expectDecimalStateRowsEqual(expected, readState(state));
      }
    }
  }

  // Input for the plan tests: 64 rows over 7 keys with nulls, a DECIMAL64
  // column and a DECIMAL128 column.
  RowVectorPtr makeInput(int32_t seed, vector_size_t numRows = 64) {
    return makeRowVector(
        {"k", "d64", "d128"},
        {
            makeFlatVector<int32_t>(numRows, [](auto row) { return row % 7; }),
            makeFlatVector<int64_t>(
                numRows,
                [seed](auto row) { return (row * 37 + seed) % 10'000 - 5'000; },
                [](auto row) { return row % 11 == 0; },
                DECIMAL(12, 2)),
            makeFlatVector<int128_t>(
                numRows,
                [seed](auto row) {
                  return static_cast<int128_t>(row * 1'234'567 + seed) *
                      1'000'000'007;
                },
                [](auto row) { return row % 13 == 0; },
                DECIMAL(30, 3)),
        });
  }

  // Runs `plan` (rooted at its FINAL aggregation) on the GPU with one driver
  // per pipeline and every input vector converted as its own GPU batch, with
  // the CudfBatchConcat funnel on and off. With `streamingCapable` it also
  // runs with the streaming FINAL group-by enabled and asserts the FINAL
  // really streamed, so a silent fallback is caught. Plans with a decimal avg
  // are not streaming capable: toStreamingGroupbyAggregators has no decimal
  // avg aggregator and falls back for the whole FINAL.
  void assertMatchesForEachFinalPath(
      const core::PlanNodePtr& plan,
      const RowVectorPtr& expected,
      bool streamingCapable) {
    auto& config = CudfConfig::getInstance();
    const auto streamingModes = streamingCapable
        ? std::vector<bool>{false, true}
        : std::vector<bool>{false};
    for (const bool concatOptimization : {true, false}) {
      for (const bool streaming : streamingModes) {
        SCOPED_TRACE(
            fmt::format(
                "concatOptimization {} streaming {}",
                concatOptimization,
                streaming));
        config.concatOptimizationEnabled = concatOptimization;
        config.batchSizeMinThreshold = 1'000'000;
        config.streamingGroupbyEnabled = streaming;
        auto task = AssertQueryBuilder(plan)
                        .maxDrivers(1)
                        .config(CudfFromVelox::kGpuBatchSizeRows, "1")
                        .assertResults(expected);
        EXPECT_EQ(
            hasCustomStat(task, plan->id(), kStreamingGroupbyUsedStat),
            streaming);
      }
    }
  }

  // GPU partial states (structs) and CPU partial states (blobs, as from an
  // HTTP exchange or a CPU-fallback producer) of `aggregates` meet in front of
  // one GPU FINAL; the result must match a CPU single aggregation over all
  // inputs.
  void checkGpuAndCpuPartialsToFinal(
      const std::vector<std::string>& aggregates,
      bool streamingCapable) {
    const std::vector<RowVectorPtr> gpuInputs = {
        makeInput(1), makeInput(2), makeInput(3)};
    const std::vector<RowVectorPtr> cpuInputs = {makeInput(4), makeInput(5)};

    unregisterCudf();
    // One blob batch per CPU input, so several reach the FINAL.
    std::vector<RowVectorPtr> cpuPartials;
    for (const auto& input : cpuInputs) {
      cpuPartials.push_back(AssertQueryBuilder(
                                PlanBuilder()
                                    .values({input})
                                    .partialAggregation({"k"}, aggregates)
                                    .planNode())
                                .copyResults(pool()));
    }
    std::vector<RowVectorPtr> allInputs = gpuInputs;
    allInputs.insert(allInputs.end(), cpuInputs.begin(), cpuInputs.end());
    const auto expected = AssertQueryBuilder(
                              PlanBuilder()
                                  .values(allInputs)
                                  .singleAggregation({"k"}, aggregates)
                                  .planNode())
                              .copyResults(pool());
    registerCudf();

    auto generator = std::make_shared<core::PlanNodeIdGenerator>();
    std::vector<core::PlanNodePtr> sources;
    // The first source must be the GPU partial aggregation:
    // finalAggregation() derives the final node from it.
    for (const auto& input : gpuInputs) {
      sources.push_back(PlanBuilder(generator)
                            .values({input})
                            .partialAggregation({"k"}, aggregates)
                            .planNode());
    }
    sources.push_back(PlanBuilder(generator).values(cpuPartials).planNode());
    const auto plan = PlanBuilder(generator)
                          .localPartitionRoundRobin(sources)
                          .finalAggregation()
                          .planNode();
    assertMatchesForEachFinalPath(plan, expected, streamingCapable);
  }

  std::optional<CudfConfig> savedConfig_;
  cuda::stream_ref stream_{cudf::get_default_stream()};
};

TEST_F(DecimalStateConcatTest, allBlobsUnchanged) {
  checkMix({{makeRows(0, 5), kBlob}, {makeRows(10, 7), kBlob}}, std::nullopt);
}

TEST_F(DecimalStateConcatTest, allStructSameShape) {
  for (auto shape : kAllDecimalStateShapes) {
    SCOPED_TRACE(DecimalStateShapeName::toName(shape));
    checkMix(
        {{makeRows(0, 5), structOf(shape)},
         {makeRows(10, 6), structOf(shape)},
         {makeRows(20, 3), structOf(shape)}},
        shape);
  }
}

// Blobs are unpacked to the struct shape; their rows read back as that shape
// reports them.
TEST_F(DecimalStateConcatTest, blobAndStructMixToStruct) {
  for (auto shape : kAllDecimalStateShapes) {
    SCOPED_TRACE(DecimalStateShapeName::toName(shape));
    checkMix(
        {{makeRows(0, 5), kBlob},
         {makeRows(10, 6), structOf(shape)},
         {makeRows(20, 4, 0), kBlob},
         {makeRows(30, 2), structOf(shape)}},
        shape);
  }
}

TEST_F(DecimalStateConcatTest, twoStructShapesWidened) {
  checkMix(
      {{makeRows(0, 5), structOf(DecimalStateShape::kSum64)},
       {makeRows(10, 6), structOf(DecimalStateShape::kAvg64)},
       {makeRows(20, 3), structOf(DecimalStateShape::kSum128)}},
      DecimalStateShape::kAvg128);
}

TEST_F(DecimalStateConcatTest, blobAndTwoStructShapesWidened) {
  checkMix(
      {{makeRows(0, 5), kBlob},
       {makeRows(10, 6), structOf(DecimalStateShape::kSum64)},
       {makeRows(20, 3), structOf(DecimalStateShape::kAvg128)}},
      DecimalStateShape::kAvg128);
}

// An empty STRING batch must not reach cudf::concatenate next to struct
// batches.
TEST_F(DecimalStateConcatTest, zeroRowBlobWithStructBatches) {
  const auto first = makeRows(0, 5);
  const auto second = makeRows(10, 3);
  for (auto funnel : allFunnels()) {
    SCOPED_TRACE(static_cast<int>(funnel));
    std::vector<CudfVectorPtr> batches;
    batches.push_back(makeEmptyBatch());
    batches.push_back(
        makeKeyStateBatch(0, first, structOf(DecimalStateShape::kAvg64)));
    batches.push_back(makeEmptyBatch());
    batches.push_back(
        makeKeyStateBatch(1000, second, structOf(DecimalStateShape::kAvg64)));
    batches.push_back(makeEmptyBatch());

    auto tables = runFunnel(funnel, std::move(batches), keyStateType());
    ASSERT_EQ(tables.size(), 1);
    const auto state = tables[0]->view().column(1);
    ASSERT_TRUE(isDecimalStateStruct(state));
    EXPECT_EQ(decimalStateShapeOf(state), DecimalStateShape::kAvg64);
    auto expected = first;
    expected.append(second);
    expectDecimalStateRowsEqual(
        expected.as(DecimalStateShape::kAvg64), readState(state));
  }
}

TEST_F(DecimalStateConcatTest, zeroRowStructWithBlobBatches) {
  checkMix(
      {{makeRows(0, 0), structOf(DecimalStateShape::kSum128)},
       {makeRows(0, 5), kBlob},
       {makeRows(10, 4), kBlob},
       {makeRows(0, 0), structOf(DecimalStateShape::kAvg64)}},
      std::nullopt);
}

TEST_F(DecimalStateConcatTest, allEmptyInputs) {
  for (auto funnel : allFunnels()) {
    SCOPED_TRACE(static_cast<int>(funnel));
    // No inputs at all: a typed empty table.
    {
      auto tables = runFunnel(funnel, {}, keyStateType());
      ASSERT_EQ(tables.size(), 1);
      EXPECT_EQ(tables[0]->num_rows(), 0);
      EXPECT_EQ(tables[0]->num_columns(), 2);
    }
    // Zero-row inputs of both forms: exactly one zero-row output.
    {
      std::vector<CudfVectorPtr> batches;
      batches.push_back(makeEmptyBatch());
      batches.push_back(makeKeyStateBatch(
          0, makeRows(0, 0), structOf(DecimalStateShape::kAvg128)));
      batches.push_back(makeEmptyBatch());
      auto tables = runFunnel(funnel, std::move(batches), keyStateType());
      ASSERT_EQ(tables.size(), 1);
      EXPECT_EQ(tables[0]->num_rows(), 0);
      ASSERT_EQ(tables[0]->num_columns(), 2);
      EXPECT_TRUE(isDecimalStateColumn(tables[0]->view().column(1)));
    }
  }
}

// ROW(k BIGINT, mixed VARBINARY, blobs VARBINARY): `mixed` mixes forms,
// `blobs` is all STRING and must stay STRING with identical content.
TEST_F(DecimalStateConcatTest, onlyMixedVarbinaryColumnIsTouched) {
  const auto type =
      ROW({"k", "mixed", "blobs"}, {BIGINT(), VARBINARY(), VARBINARY()});
  const auto mixedBlob = makeRows(0, 5);
  const auto blobsFirst = makeRows(100, 5, 1);
  const auto mixedStruct = makeRows(10, 6);
  const auto blobsSecond = makeRows(200, 6, 2);
  for (auto funnel : allFunnels()) {
    SCOPED_TRACE(static_cast<int>(funnel));
    std::vector<CudfVectorPtr> batches;
    {
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(makeInt64Column({0, 1, 2, 3, 4}, nullptr, stream_));
      columns.push_back(makeDecimalStateBlob(mixedBlob, stream_));
      columns.push_back(makeDecimalStateBlob(blobsFirst, stream_));
      batches.push_back(makeVector(type, std::move(columns)));
    }
    {
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(makeInt64Column({5, 6, 7, 8, 9, 10}, nullptr, stream_));
      columns.push_back(makeStateColumn(
          mixedStruct, structOf(DecimalStateShape::kSum64), stream_));
      columns.push_back(makeDecimalStateBlob(blobsSecond, stream_));
      batches.push_back(makeVector(type, std::move(columns)));
    }
    auto tables = runFunnel(funnel, std::move(batches), type);
    ASSERT_EQ(tables.size(), 1);
    const auto view = tables[0]->view();
    EXPECT_EQ(
        copyColumnData<int64_t>(view.column(0), stream_),
        (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    auto expectedMixed = mixedBlob;
    expectedMixed.append(mixedStruct);
    ASSERT_TRUE(isDecimalStateStruct(view.column(1)));
    EXPECT_EQ(decimalStateShapeOf(view.column(1)), DecimalStateShape::kSum64);
    expectDecimalStateRowsEqual(
        expectedMixed.as(DecimalStateShape::kSum64), readState(view.column(1)));

    auto expectedBlobs = blobsFirst;
    expectedBlobs.append(blobsSecond);
    EXPECT_EQ(view.column(2).type().id(), cudf::type_id::STRING);
    expectDecimalStateRowsEqual(expectedBlobs, readState(view.column(2)));
  }
}

// Two output batches of 4 rows; each mixes a blob with a struct.
TEST_F(DecimalStateConcatTest, batchedFunnelNormalizesPerOutputBatch) {
  CudfConfig::getInstance().batchSizeMaxThreshold = 4;
  const std::vector<DecimalStateRows> rows = {
      makeRows(0, 2), makeRows(10, 2), makeRows(20, 2, 1), makeRows(30, 2, 0)};
  const auto sumOnly = structOf(DecimalStateShape::kSum64);
  for (auto funnel : {Funnel::kTableBatched, Funnel::kCudfVectorsBatched}) {
    SCOPED_TRACE(static_cast<int>(funnel));
    std::vector<CudfVectorPtr> batches;
    for (size_t batch = 0; batch < rows.size(); ++batch) {
      batches.push_back(makeKeyStateBatch(
          static_cast<int64_t>(2 * batch),
          rows[batch],
          batch % 2 == 0 ? kBlob : sumOnly));
    }
    auto tables = runFunnel(funnel, std::move(batches), keyStateType());
    ASSERT_EQ(tables.size(), 2);
    for (size_t table = 0; table < 2; ++table) {
      const auto view = tables[table]->view();
      std::vector<int64_t> keys;
      for (auto key = static_cast<int64_t>(4 * table);
           key < static_cast<int64_t>(4 * table + 4);
           ++key) {
        keys.push_back(key);
      }
      EXPECT_EQ(copyColumnData<int64_t>(view.column(0), stream_), keys);
      auto expected = rows[2 * table];
      expected.append(rows[2 * table + 1]);
      ASSERT_TRUE(isDecimalStateStruct(view.column(1)));
      expectDecimalStateRowsEqual(
          expected.as(DecimalStateShape::kSum64), readState(view.column(1)));
    }
  }
}

TEST_F(DecimalStateConcatTest, normalizeLeavesOtherColumnsIdentical) {
  const auto type =
      ROW({"k", "mixed", "blobs"}, {BIGINT(), VARBINARY(), VARBINARY()});
  std::vector<std::unique_ptr<cudf::table>> owners;
  auto makeTable = [&](std::vector<int64_t> keys,
                       const DecimalStateRows& mixed,
                       Form mixedForm,
                       const DecimalStateRows& blobs) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(makeInt64Column(keys, nullptr, stream_));
    columns.push_back(makeStateColumn(mixed, mixedForm, stream_));
    columns.push_back(makeDecimalStateBlob(blobs, stream_));
    owners.push_back(std::make_unique<cudf::table>(std::move(columns)));
    return owners.back()->view();
  };
  std::vector<cudf::table_view> views{
      makeTable({0, 1, 2}, makeRows(0, 3), kBlob, makeRows(50, 3)),
      makeTable(
          {3, 4},
          makeRows(10, 2),
          structOf(DecimalStateShape::kAvg64),
          makeRows(60, 2))};
  const auto original = views;

  auto replacements =
      normalizeDecimalStateTableViews(views, type, stream_, testMr());
  stream_.sync();
  ASSERT_EQ(views.size(), 2);
  // The blob batch of `mixed` was unpacked into a replacement column.
  EXPECT_FALSE(replacements.empty());
  for (size_t batch = 0; batch < views.size(); ++batch) {
    // BIGINT and the uniform VARBINARY column keep the same device memory.
    EXPECT_EQ(views[batch].column(0).head(), original[batch].column(0).head());
    EXPECT_EQ(views[batch].column(2).head(), original[batch].column(2).head());
    EXPECT_EQ(views[batch].column(2).type(), original[batch].column(2).type());
    ASSERT_TRUE(isDecimalStateStruct(views[batch].column(1)));
    EXPECT_EQ(
        decimalStateShapeOf(views[batch].column(1)), DecimalStateShape::kAvg64);
  }
  // The struct batch is left as is (no copy).
  EXPECT_EQ(
      views[1].column(1).child(0).head(),
      original[1].column(1).child(0).head());
}

TEST_F(DecimalStateConcatTest, normalizeIsNoOpWithoutMix) {
  const auto type = keyStateType();
  std::vector<std::unique_ptr<cudf::table>> owners;
  owners.push_back(makeEmptyKeyStateTable());
  for (int i = 0; i < 2; ++i) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(makeInt64Column({1, 2}, nullptr, stream_));
    columns.push_back(makeDecimalStateBlob(makeRows(i, 2), stream_));
    owners.push_back(std::make_unique<cudf::table>(std::move(columns)));
  }
  std::vector<cudf::table_view> views;
  for (const auto& owner : owners) {
    views.push_back(owner->view());
  }
  auto replacements =
      normalizeDecimalStateTableViews(views, type, stream_, testMr());
  EXPECT_TRUE(replacements.empty());
  // Zero-row views are only dropped when a mix must be resolved.
  ASSERT_EQ(views.size(), 3);
  for (size_t batch = 0; batch < views.size(); ++batch) {
    EXPECT_EQ(
        views[batch].column(1).head(), owners[batch]->view().column(1).head());
  }

  // Only VARBINARY columns are normalized: a BIGINT-typed column with
  // differing physical types is never inspected.
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(makeInt64Column({3, 4}, nullptr, stream_));
    columns.push_back(makeStateColumn(
        makeRows(5, 2), structOf(DecimalStateShape::kSum64), stream_));
    owners.push_back(std::make_unique<cudf::table>(std::move(columns)));
  }
  const auto bigintType = ROW({"k", "s"}, {BIGINT(), BIGINT()});
  std::vector<cudf::table_view> mixed{owners[1]->view(), owners[3]->view()};
  EXPECT_TRUE(
      normalizeDecimalStateTableViews(mixed, bigintType, stream_, testMr())
          .empty());
  EXPECT_EQ(mixed[0].column(1).type().id(), cudf::type_id::STRING);
  EXPECT_EQ(mixed[1].column(1).type().id(), cudf::type_id::STRUCT);
}

// Zero-row batches are wildcards: the only non-empty shape wins, with no
// widening, and the empty struct of another shape is re-typed.
TEST_F(DecimalStateConcatTest, normalizeBatchesRetypesZeroRowStruct) {
  const auto rows = makeRows(0, 5);
  auto sumOnly =
      makeStateColumn(rows, structOf(DecimalStateShape::kSum64), stream_);
  auto emptyAvg = makeStateColumn(
      DecimalStateRows{}, structOf(DecimalStateShape::kAvg128), stream_);
  std::vector<cudf::column_view> views = {emptyAvg->view(), sumOnly->view()};
  auto replacements = normalizeDecimalStateBatches(views, stream_, testMr());
  ASSERT_EQ(replacements.size(), 1);
  EXPECT_EQ(views[1].head(), sumOnly->view().head());
  for (const auto& view : views) {
    ASSERT_TRUE(isDecimalStateStruct(view));
    EXPECT_EQ(decimalStateShapeOf(view), DecimalStateShape::kSum64);
  }
  auto concatenated = cudf::concatenate(views, stream_, testMr());
  expectDecimalStateRowsEqual(
      rows.as(DecimalStateShape::kSum64), readState(concatenated->view()));
}

TEST_F(DecimalStateConcatTest, normalizeBatchesAllZeroRowMixedForms) {
  auto emptyBlob = cudf::make_empty_column(cudf::type_id::STRING);
  auto emptySum = makeStateColumn(
      DecimalStateRows{}, structOf(DecimalStateShape::kSum64), stream_);
  auto emptyAvg = makeStateColumn(
      DecimalStateRows{}, structOf(DecimalStateShape::kAvg64), stream_);
  std::vector<cudf::column_view> views = {
      emptyBlob->view(), emptySum->view(), emptyAvg->view()};
  auto replacements = normalizeDecimalStateBatches(views, stream_, testMr());
  for (const auto& view : views) {
    ASSERT_TRUE(isDecimalStateStruct(view));
    EXPECT_EQ(decimalStateShapeOf(view), DecimalStateShape::kAvg128);
  }
  EXPECT_EQ(cudf::concatenate(views, stream_, testMr())->size(), 0);

  // Uniform empties and no batches at all are left alone.
  std::vector<cudf::column_view> blobs = {emptyBlob->view(), emptyBlob->view()};
  EXPECT_TRUE(normalizeDecimalStateBatches(blobs, stream_, testMr()).empty());
  std::vector<cudf::column_view> none;
  EXPECT_TRUE(normalizeDecimalStateBatches(none, stream_, testMr()).empty());
}

TEST_F(DecimalStateConcatTest, normalizeBatchesRejectsNonStateBatch) {
  auto sumOnly = makeStateColumn(
      makeRows(0, 3), structOf(DecimalStateShape::kSum64), stream_);
  auto int32Column = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32},
      3,
      cudf::mask_state::UNALLOCATED,
      stream_,
      testMr());
  std::vector<cudf::column_view> views = {sumOnly->view(), int32Column->view()};
  VELOX_ASSERT_THROW(
      normalizeDecimalStateBatches(views, stream_, testMr()),
      "not a decimal aggregate state column");
}

// Struct and blob partial states of SUM and AVG meet in front of one GPU
// FINAL. With the CudfBatchConcat funnel off, the FINAL's own buffered-state
// concat sees the mix (struct buffer plus blob input).
TEST_F(DecimalStateConcatTest, gpuAndCpuPartialStatesToFinalMatchCpu) {
  checkGpuAndCpuPartialsToFinal(
      {"sum(d64) AS s64",
       "avg(d64) AS a64",
       "sum(d128) AS s128",
       "avg(d128) AS a128"},
      false);
}

// SUM only, so the streaming FINAL group-by also runs and its decimal SUM
// aggregator sees struct and blob batches.
TEST_F(DecimalStateConcatTest, gpuAndCpuPartialSumStatesToFinalMatchCpu) {
  checkGpuAndCpuPartialsToFinal({"sum(d64) AS s64", "sum(d128) AS s128"}, true);
}

// Version skew: two producers emit different struct shapes (kAvg128 from avg,
// kSum128 from sum) for the same DECIMAL(30, 3) state column. Both carry the
// sum and overflow a SUM FINAL reads, so the result is the plain sum.
TEST_F(DecimalStateConcatTest, differentStructShapesToFinalMatchCpu) {
  const std::vector<RowVectorPtr> inputs = {
      makeInput(1), makeInput(2), makeInput(3), makeInput(4)};
  unregisterCudf();
  const auto expected =
      AssertQueryBuilder(
          PlanBuilder()
              .values(inputs)
              .singleAggregation({"k"}, {"sum(d128) AS total"})
              .planNode())
          .copyResults(pool());
  registerCudf();

  auto generator = std::make_shared<core::PlanNodeIdGenerator>();
  const std::vector<core::PlanNodePtr> sources = {
      PlanBuilder(generator)
          .values({inputs[0], inputs[1]})
          .partialAggregation({"k"}, {"avg(d128) AS s"})
          .planNode(),
      PlanBuilder(generator)
          .values({inputs[2], inputs[3]})
          .partialAggregation({"k"}, {"sum(d128) AS s"})
          .planNode(),
  };
  const auto plan =
      PlanBuilder(generator)
          .localPartitionRoundRobin(sources)
          .finalAggregation({"k"}, {"sum(s) AS total"}, {{DECIMAL(30, 3)}})
          .planNode();
  assertMatchesForEachFinalPath(plan, expected, true);
}

// Zero-row state batches cannot reach the funnel through a plan: the Driver
// rejects an empty operator output ("Operator::getOutput() must return nullptr
// or a non-empty vector"), and CudfFromVelox, CudfLocalPartition,
// CudfBatchConcat and CudfGroupby drop empty inputs or partitions. So the
// zero-row cases are exercised on the table-view funnel directly, as the
// group-by FINAL calls it before concatenating its buffered state with an
// input batch (normalizeDecimalStateTableViews, then cudf::concatenate).
TEST_F(DecimalStateConcatTest, normalizeTableViewsDropsZeroRowViews) {
  const auto type = keyStateType();
  const auto rows = makeRows(0, 5);
  const auto blobRows = makeRows(10, 3);
  std::vector<std::unique_ptr<cudf::table>> owners;
  auto makeTable = [&](const DecimalStateRows& stateRows, Form form) {
    std::vector<int64_t> keys(stateRows.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      keys[i] = static_cast<int64_t>(i);
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(makeInt64Column(keys, nullptr, stream_));
    columns.push_back(makeStateColumn(stateRows, form, stream_));
    owners.push_back(std::make_unique<cudf::table>(std::move(columns)));
    return owners.back()->view();
  };
  auto emptyBlobTable = [&]() {
    owners.push_back(makeEmptyKeyStateTable());
    return owners.back()->view();
  };
  const auto avg64 = structOf(DecimalStateShape::kAvg64);
  const auto sum128 = structOf(DecimalStateShape::kSum128);

  struct Case {
    std::string name;
    std::vector<cudf::table_view> views;
    // Expected content of the concatenated state column, and its shape (a
    // STRING blob when unset).
    DecimalStateRows expected;
    std::optional<DecimalStateShape> shape;
  };
  std::vector<Case> cases;
  // Buffered struct state plus an empty blob input: the empty view is
  // dropped and the struct is concatenated alone.
  cases.push_back(
      {"struct buffer, empty blob input",
       {makeTable(rows, avg64), emptyBlobTable()},
       rows.as(DecimalStateShape::kAvg64),
       DecimalStateShape::kAvg64});
  // Empty struct buffer of another shape plus a blob input: the blob is kept
  // as is.
  cases.push_back(
      {"empty struct buffer, blob input",
       {makeTable(DecimalStateRows{}, sum128), makeTable(blobRows, kBlob)},
       blobRows,
       std::nullopt});
  // Empty views of both forms around a struct and a blob: the empties are
  // dropped, the blob is unpacked to the struct's shape.
  {
    auto expected = rows;
    expected.append(blobRows);
    cases.push_back(
        {"empties around struct and blob",
         {emptyBlobTable(),
          makeTable(rows, avg64),
          makeTable(DecimalStateRows{}, sum128),
          makeTable(blobRows, kBlob),
          emptyBlobTable()},
         expected.as(DecimalStateShape::kAvg64),
         DecimalStateShape::kAvg64});
  }

  for (auto& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    auto views = testCase.views;
    auto replacements =
        normalizeDecimalStateTableViews(views, type, stream_, testMr());
    for (const auto& view : views) {
      EXPECT_GT(view.num_rows(), 0);
    }
    auto concatenated = cudf::concatenate(views, stream_, testMr());
    stream_.sync();
    const auto state = concatenated->view().column(1);
    if (testCase.shape.has_value()) {
      ASSERT_TRUE(isDecimalStateStruct(state));
      EXPECT_EQ(decimalStateShapeOf(state), *testCase.shape);
    } else {
      EXPECT_EQ(state.type().id(), cudf::type_id::STRING);
    }
    expectDecimalStateRowsEqual(testCase.expected, readState(state));
  }

  // Only empty views of mixed forms: all are kept and re-typed to one form,
  // so the concatenation is a typed zero-row column.
  std::vector<cudf::table_view> empties{
      emptyBlobTable(),
      makeTable(DecimalStateRows{}, avg64),
      makeTable(DecimalStateRows{}, sum128)};
  auto replacements =
      normalizeDecimalStateTableViews(empties, type, stream_, testMr());
  ASSERT_EQ(empties.size(), 3);
  auto concatenated = cudf::concatenate(empties, stream_, testMr());
  EXPECT_EQ(concatenated->num_rows(), 0);
  EXPECT_TRUE(isDecimalStateColumn(concatenated->view().column(1)));
}

} // namespace
} // namespace facebook::velox::cudf_velox::test
