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

#include "velox/experimental/cudf/benchmarks/KvikioReadBenchmark.h"

#include "velox/common/base/Exceptions.h"

#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include <fstream>
#include <iostream>

DEFINE_string(
    paths,
    "",
    "Path to a manifest file holding one object URI per line. Blank lines "
    "and lines starting with '#' are ignored. Credentials come from the "
    "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION and "
    "AWS_SESSION_TOKEN environment variables; set AWS_ENDPOINT_URL to target "
    "a non-AWS S3 server.");

DEFINE_bool(
    list_targets,
    false,
    "Open every target, print its size, and exit without reading payload.");

namespace facebook::velox::cudf_velox {

RemoteTargets::RemoteTargets(const std::vector<std::string>& uris) {
  infos_.reserve(uris.size());
  handles_.reserve(uris.size());
  for (const auto& uri : uris) {
    try {
      auto handle = kvikio::RemoteHandle::open(uri);
      infos_.push_back(TargetInfo{uri, handle.nbytes()});
      handles_.push_back(std::move(handle));
    } catch (const std::exception& e) {
      VELOX_USER_FAIL(
          "Failed to open remote target. URI: {}, error: {}", uri, e.what());
    }
  }
}

uint64_t RemoteTargets::totalBytes() const {
  uint64_t total{0};
  for (const auto& info : infos_) {
    total += info.size;
  }
  return total;
}

std::vector<std::string> readManifestFile(const std::string& path) {
  std::ifstream in(path);
  VELOX_USER_CHECK(in.is_open(), "Cannot open manifest file: {}", path);
  auto uris = parseManifest(in);
  VELOX_USER_CHECK(!uris.empty(), "Manifest file holds no URIs: {}", path);
  return uris;
}

} // namespace facebook::velox::cudf_velox

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Measures read throughput from S3 through KvikIO's remote path. "
      "Run with --helpon=KvikioReadBenchmark for the full flag list.");
  folly::Init init{&argc, &argv, false};

  using namespace facebook::velox::cudf_velox;

  VELOX_USER_CHECK(!FLAGS_paths.empty(), "--paths is required");
  RemoteTargets targets{readManifestFile(FLAGS_paths)};

  if (FLAGS_list_targets) {
    for (const auto& info : targets.infos()) {
      std::cout << info.size << '\t' << info.uri << std::endl;
    }
    std::cout << "total_bytes=" << targets.totalBytes() << std::endl;
    return 0;
  }

  std::cout << "Opened " << targets.infos().size() << " targets, "
            << targets.totalBytes() << " bytes total" << std::endl;
  return 0;
}
