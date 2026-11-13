/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "velox/type/Filter.h"

#include <cudf/column/column.hpp>

#include <memory>

namespace facebook::velox::cudf_velox::connector::hive {

/// Marks 'filter' as advisory: the operator that produced it still drops
/// every row the filter rejects, so a scan may keep any of those rows. A scan
/// uses an advisory filter to prune row groups and, when that pays, rows.
/// Shared copies stay advisory. A clone or a merge with another filter is not
/// advisory, and a scan applies it to every row.
///
/// 'deviceBitmap', if given, is what a scan that filters rows on the device
/// tests instead of 'filter', a BigintRange or a set of integers: UINT32 words
/// with bit 'value - smallest value the filter passes' set for each value the
/// producer has. It narrows a range to the producer's values, and saves the
/// scan from unpacking a set. Other readers of the filter see 'filter'.
common::FilterPtr makeAdvisoryFilter(
    std::unique_ptr<common::Filter> filter,
    std::shared_ptr<const cudf::column> deviceBitmap = nullptr);

/// Whether a scan may keep rows that 'filter' rejects.
bool isAdvisoryFilter(const common::FilterPtr& filter);

/// Returns the device bitmap of an advisory filter, or null if it has none.
const cudf::column* advisoryFilterBitmap(const common::FilterPtr& filter);

} // namespace facebook::velox::cudf_velox::connector::hive
