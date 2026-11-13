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

#include "velox/type/Filter.h"
#include "velox/type/Type.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <cuda/stream>

#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// A filter that passes the join keys of a build side, for the scans that
/// feed the probe side.
struct BuildKeyFilter {
  /// Passes every non-null build key and rejects null: AlwaysFalse for a
  /// build side without a non-null key. Null if there is no filter: the key
  /// type carries none, or building it failed.
  std::unique_ptr<common::Filter> filter;

  /// Whether 'filter' rejects every value that is not a build key. True for a
  /// build side of at most CudfConfig::dynamicFilterMaxValues distinct keys
  /// in at most 64 times as many rows. A larger build side gives the
  /// range from its smallest to its largest key.
  bool exact{false};

  /// Whether no two build rows have the same non-null key. Known, and
  /// possibly true, only if 'exact'.
  bool uniqueKeys{false};

  /// If CudfConfig::dynamicFilterMaxBitmapBytes allows it and 'filter' is not
  /// a plain range of all its keys: UINT32 words with bit 'key - smallest
  /// key' set for each build key. A cuDF scan tests rows with it when it
  /// comes attached to the filter (connector::hive::makeAdvisoryFilter): it
  /// narrows the range of a 'filter' that is not exact, and spares the scan
  /// unpacking an exact one.
  std::shared_ptr<const cudf::column> deviceBitmap;
};

/// Builds the filter for one join key from 'keyColumns', the column of that
/// key in each table of the build side, of Velox type 'type'. Supports the
/// integer types and DATE. Runs on 'stream' and returns after the device work
/// has finished. Never throws: a dynamic filter is an optimization, and a
/// build side it cannot be built for gives no filter.
BuildKeyFilter makeBuildKeyFilter(
    const std::vector<cudf::column_view>& keyColumns,
    const TypePtr& type,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox
