/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/Expr.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <cuda/stream_ref>

#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Re-runs a filter through Velox after a GPU SFI kernel declined a row.
///
/// A declined row is one whose precondition check failed on the device --
/// division by zero, an overflow -- where the device cannot raise. Velox is
/// what raises: this hands the rows back to Velox's own evaluator, which
/// produces the error, its class and its message, and none of that is
/// reimplemented here.
///
/// The whole filter expression is re-run, not the subexpression that declined.
/// That is deliberate and load-bearing: a GPU conditional computes both of its
/// branches for every row and selects afterwards, so the device can decline a
/// row that Velox's SwitchExpr would never have evaluated. Only the full tree
/// carries the conditional that masks it, so only the full tree gives the same
/// answer -- including no error at all.
///
/// Returns the verdict as a boolean cudf column, so the caller applies it the
/// same way it applies the device one. Throws whatever Velox throws.
///
/// `cached` holds the compiled expression across calls; it is built on first
/// use, because most queries never reach this path and compiling does constant
/// folding.
std::unique_ptr<cudf::column> reevaluateFilterOnCpu(
    const core::TypedExprPtr& filter,
    const RowTypePtr& rowType,
    const std::vector<cudf::column_view>& columns,
    std::unique_ptr<velox::exec::ExprSet>& cached,
    core::ExecCtx* execCtx,
    memory::MemoryPool* pool,
    cuda::stream_ref stream);

/// The connector form, for a scan's remaining filter.
///
/// Connectors reach Velox's evaluator through core::ExpressionEvaluator rather
/// than holding an ExecCtx of their own, so the compile and evaluate calls
/// differ; everything the comment above says applies unchanged.
std::unique_ptr<cudf::column> reevaluateFilterOnCpu(
    const core::TypedExprPtr& filter,
    const RowTypePtr& rowType,
    const std::vector<cudf::column_view>& columns,
    std::unique_ptr<velox::exec::ExprSet>& cached,
    core::ExpressionEvaluator* evaluator,
    memory::MemoryPool* pool,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox
