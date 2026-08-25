# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- TPC-H benchmarks ---

add_executable(velox_cudf_tpch_benchmark CudfTpchBenchmarkMain.cpp)

# The benchmark itself, plus the resource lifecycle, I/O mode configuration,
# discarding sink, and raw read runner it is built from. Only the entry point
# above is exclusive to the executable, so a test can compile these same
# sources and drive the benchmark directly.
target_sources(
  velox_cudf_tpch_benchmark
  PRIVATE
    CudfTpchBenchmark.cpp
    CudfBenchmarkLifecycle.cpp
    CudfTpchIoConfig.cpp
    CudfBenchmarkDiscard.cpp
    CudfRawReadBenchmark.cpp
)

velox_add_test_headers(
  velox_cudf_tpch_benchmark
  CudfBenchmarkDiscard.h
  CudfBenchmarkLifecycle.h
  CudfRawReadBenchmark.h
  CudfTpchBenchmark.h
  CudfTpchIoConfig.h
)

target_link_libraries(
  velox_cudf_tpch_benchmark
  velox_cudf_exec
  velox_cudf_exec_test_lib
  velox_tpch_benchmark_lib
  velox_s3fs
  velox_cudf_hive_connector
)

# --- TPC-DS benchmarks ---

add_executable(velox_cudf_tpcds_benchmark CudfTpcdsBenchmark.cpp)

target_link_libraries(
  velox_cudf_tpcds_benchmark
  velox_cudf_exec
  velox_cudf_exec_test_lib
  velox_tpcds_benchmark_lib
)

# --- other benchmarks ---

add_executable(velox_cudf_interop_benchmark CudfInteropBenchmark.cpp)

target_link_libraries(
  velox_cudf_interop_benchmark
  velox_cudf_exec
  velox_memory
  velox_vector
  velox_vector_fuzzer
  Folly::folly
  Folly::follybenchmark
)
