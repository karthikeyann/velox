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

#include "velox/experimental/cudf/benchmarks/CudfBenchmarkLifecycle.h"

#include "velox/benchmarks/tpch/TpchBenchmark.h"
#include "velox/common/base/Exceptions.h"
#include "velox/connectors/Connector.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class CudfTpchBenchmark : public TpchBenchmark {
 public:
  /// Validates an enabled I/O benchmark mode before taking anything, then
  /// acquires what the run needs. The validation reads only the local
  /// manifest, so a misconfigured run is rejected without having registered a
  /// file system, created a connector or opened a remote object.
  void initialize() override;

  std::shared_ptr<facebook::velox::config::ConfigBase> makeConnectorProperties()
      override;

  std::vector<std::shared_ptr<facebook::velox::connector::ConnectorSplit>>
  listSplits(
      const std::string& path,
      int32_t numSplitsPerFile,
      const facebook::velox::exec::test::TpchPlan& plan) override;

  /// Releases the resources initialize() took, in CudfBenchmarkResource order:
  /// the Hive connector, then the query builder and the base's resources, then
  /// the I/O executor the connector borrowed, then cuDF, and last the S3 file
  /// system. Process-global state the benchmark found already in place is left
  /// as it was: the connector is erased only while the registry still maps the
  /// id to the one this benchmark registered, and cuDF and the S3 file system
  /// are released only if this benchmark is what put them there. Attempts every
  /// resource and rethrows the first failure, keeping whatever it could not
  /// release for a later call. Each resource is released exactly once, so a
  /// repeated call after a clean shutdown does nothing.
  void shutdown() override;

  /// Dispatches to the I/O benchmark loop when an I/O mode is enabled;
  /// otherwise delegates unchanged to TpchBenchmark::runMain.
  void runMain(std::ostream& out, facebook::velox::RunStats& runStats) override;

 private:
  // Acquires the resources shutdown() releases, recording each with
  // 'lifecycle_' as it is taken. Runs under initialize()'s cleanup handler.
  void initializeResources(bool ioEnabled);

  // Records that 'installed' is the connector this benchmark put in the
  // registry, replacing any connector it recorded before, and arranges for
  // shutdown to erase it. The erase is skipped when the registry no longer
  // maps the id to 'installed', so a connector something else registered
  // afterwards is left alone.
  void ownConnector(
      std::shared_ptr<facebook::velox::connector::Connector> installed);

  // Resources this benchmark currently owns. Empty before initialize() takes
  // the first one and empty again once shutdown() has released them all.
  facebook::velox::cudf_velox::CudfBenchmarkLifecycle lifecycle_;

  // Connector this benchmark registered, or null when it registered none.
  // Held so that shutdown can tell it apart from a connector the host process
  // or a later caller registered under the same id.
  std::shared_ptr<facebook::velox::connector::Connector> benchmarkConnector_;
};

namespace facebook::velox::cudf_velox {

/// Parse a properties file into a key-value map.
/// Each line should be key=value. Lines starting with '#' and blank lines are
/// skipped. Lines without '=' are logged as warnings and ignored.
inline std::unordered_map<std::string, std::string> loadPropertiesFile(
    const std::string& path) {
  auto fsPath = std::filesystem::path(path);
  VELOX_USER_CHECK(
      std::filesystem::exists(fsPath), "Properties file not found: {}", path);
  std::unordered_map<std::string, std::string> properties;
  std::string line;
  std::ifstream configFile(fsPath);
  int lineNum = 0;
  while (std::getline(configFile, line)) {
    ++lineNum;
    line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto delimiterPos = line.find('=');
    if (delimiterPos == std::string::npos) {
      LOG(WARNING) << "Skipping malformed config line in " << path
                   << " at line " << lineNum << " (no '=')";
      continue;
    }
    LOG(INFO) << "Setting property " << line.substr(0, delimiterPos);
    properties.emplace(
        line.substr(0, delimiterPos), line.substr(delimiterPos + 1));
  }
  return properties;
}

} // namespace facebook::velox::cudf_velox
