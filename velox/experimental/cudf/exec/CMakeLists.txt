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

find_package(Threads REQUIRED)

set(PERFETTO_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../external/perfetto/sdk")

add_library(velox_cudf_perfetto STATIC ${PERFETTO_SDK_DIR}/perfetto.cc)
target_include_directories(velox_cudf_perfetto PUBLIC ${PERFETTO_SDK_DIR})
target_compile_features(velox_cudf_perfetto PRIVATE cxx_std_17)
target_compile_options(
  velox_cudf_perfetto
  PRIVATE -Wno-pragmas -Wno-deprecated-declarations -Wno-error
)
set_target_properties(velox_cudf_perfetto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(velox_cudf_perfetto PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

add_library(
  velox_cudf_exec
  AggregationRegistry.cpp
  CudfAggregation.cpp
  CudfAssignUniqueId.cpp
  CudfBatchConcat.cpp
  CudfConversion.cpp
  CudfDistinct.cpp
  CudfEnforceSingleRow.cpp
  CudfFilterProject.cpp
  CudfGroupId.cpp
  CudfGroupby.cpp
  CudfHashJoin.cpp
  CudfLimit.cpp
  CudfLocalPartition.cpp
  CudfMarkDistinct.cpp
  CudfNestedLoopJoin.cpp
  CudfOrderBy.cpp
  CudfReduce.cpp
  CudfTopN.cpp
  CudfWindow.cpp
  DebugUtil.cpp
  DecimalAggregationDevice.cu
  DecimalAggregationHostOps.cpp
  DecimalAggregationState.cpp
  GpuMemoryCapture.cpp
  GpuMemoryTrace.cpp
  GpuResources.cpp
  OperatorAdapters.cpp
  PrestoAggregateFunctions.cpp
  SparkAggregateFunctions.cpp
  ToCudf.cpp
  Utilities.cpp
  VeloxCudfInterop.cpp
)

target_link_libraries(
  velox_cudf_exec
  PUBLIC cudf::cudf nvtx3::nvtx3-cpp
  PRIVATE
    arrow
    velox_arrow_bridge
    velox_common_base
    velox_cudf_expression
    velox_cudf_hive_connector
    velox_cudf_iceberg_connector
    velox_cudf_vector
    velox_cudf_perfetto
    velox_exception
    velox_exec
    velox_test_util
    ${CMAKE_DL_LIBS}
)

target_compile_options(velox_cudf_exec PRIVATE -Wno-missing-field-initializers)
