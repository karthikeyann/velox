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
#include "velox/exec/DynamicFilterCollector.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec {
namespace {

// The values an integer filter passes: the bounds and, if the filter is a set
// of at most 'maxValues' values, the values.
struct IntegerValues {
  int64_t min;
  int64_t max;
  std::optional<std::vector<int64_t>> values;
};

std::optional<IntegerValues> integerValues(
    const common::Filter& filter,
    int32_t maxValues) {
  switch (filter.kind()) {
    case common::FilterKind::kBigintRange: {
      const auto& range = static_cast<const common::BigintRange&>(filter);
      IntegerValues result{range.lower(), range.upper(), std::nullopt};
      if (range.lower() == range.upper()) {
        result.values = std::vector<int64_t>{range.lower()};
      }
      return result;
    }
    case common::FilterKind::kBigintValuesUsingHashTable: {
      const auto& set =
          static_cast<const common::BigintValuesUsingHashTable&>(filter);
      IntegerValues result{set.min(), set.max(), std::nullopt};
      if (set.values().size() <= maxValues) {
        result.values = set.values();
      }
      return result;
    }
    case common::FilterKind::kBigintValuesUsingBitmask: {
      const auto& set =
          static_cast<const common::BigintValuesUsingBitmask&>(filter);
      IntegerValues result{set.min(), set.max(), std::nullopt};
      // A bitmask holds at most one value per bit of its range. Listing its
      // values costs a pass over the range.
      constexpr int64_t kMaxBitsPerValue = 1'024;
      if ((set.max() - set.min()) / kMaxBitsPerValue <= maxValues) {
        auto values = set.values();
        if (values.size() <= maxValues) {
          result.values = std::move(values);
        }
      }
      return result;
    }
    default:
      return std::nullopt;
  }
}

} // namespace

DynamicFilterCollector::DynamicFilterCollector(
    BuildDistribution distribution,
    int32_t numTasks,
    const Options& options)
    : distribution_{distribution}, numTasks_{numTasks}, options_{options} {
  VELOX_CHECK_GT(numTasks_, 0);
}

bool DynamicFilterCollector::add(const common::FilterPtr& filter) {
  VELOX_CHECK_LT(
      numAdded_, numTasks_, "More dynamic filters added than tasks make");
  ++numAdded_;
  if (done_) {
    return true;
  }
  if (filter == nullptr) {
    result_ = nullptr;
    done_ = true;
    return true;
  }
  if (numAdded_ == 1) {
    result_ = filter;
  } else {
    result_ = unionOf(result_, filter, options_);
  }
  done_ = result_ == nullptr || numAdded_ == numTasks_ ||
      distribution_ == BuildDistribution::kBroadcast;
  return done_;
}

const common::FilterPtr& DynamicFilterCollector::result() const {
  VELOX_CHECK(done_, "Dynamic filters of some tasks are missing");
  return result_;
}

// static
common::FilterPtr DynamicFilterCollector::unionOf(
    const common::FilterPtr& left,
    const common::FilterPtr& right,
    const Options& options) {
  VELOX_CHECK_NOT_NULL(left);
  VELOX_CHECK_NOT_NULL(right);
  if (left->kind() == common::FilterKind::kAlwaysFalse) {
    return right;
  }
  if (right->kind() == common::FilterKind::kAlwaysFalse) {
    return left;
  }
  const bool nullAllowed = left->nullAllowed() || right->nullAllowed();

  if (left->kind() == common::FilterKind::kBytesValues &&
      right->kind() == common::FilterKind::kBytesValues) {
    const auto& leftValues =
        static_cast<const common::BytesValues&>(*left).values();
    const auto& rightValues =
        static_cast<const common::BytesValues&>(*right).values();
    std::vector<std::string> values(leftValues.begin(), leftValues.end());
    for (const auto& value : rightValues) {
      if (!leftValues.contains(value)) {
        values.push_back(value);
      }
    }
    if (values.size() > options.maxValues) {
      return nullptr;
    }
    return std::make_shared<common::BytesValues>(values, nullAllowed);
  }

  auto leftValues = integerValues(*left, options.maxValues);
  auto rightValues = integerValues(*right, options.maxValues);
  if (!leftValues.has_value() || !rightValues.has_value()) {
    return nullptr;
  }
  if (leftValues->values.has_value() && rightValues->values.has_value() &&
      leftValues->values->size() + rightValues->values->size() <=
          options.maxValues) {
    auto& values = leftValues->values.value();
    values.insert(
        values.end(), rightValues->values->begin(), rightValues->values->end());
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return common::createBigintValues(values, nullAllowed);
  }
  return std::make_shared<common::BigintRange>(
      std::min(leftValues->min, rightValues->min),
      std::max(leftValues->max, rightValues->max),
      nullAllowed);
}

} // namespace facebook::velox::exec
