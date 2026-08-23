#!/bin/bash
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

# Sweeps request size, reader threads and KvikIO task size against one
# manifest, printing one result line per combination.
#
# Credentials come from the environment. For a non-AWS server also export
# AWS_ENDPOINT_URL.
#
# Usage: ./kvikio_sweep.sh <manifest> [measurement_bytes]

set -euo pipefail

MANIFEST="${1:?usage: kvikio_sweep.sh <manifest> [measurement_bytes]}"
MEASUREMENT_BYTES="${2:-$((4 * 1024 * 1024 * 1024))}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
BIN="${KVIKIO_BENCHMARK_BIN:-${REPO_ROOT}/_build/release/velox/experimental/cudf/benchmarks/velox_cudf_kvikio_read_benchmark}"

if [[ ! -x ${BIN} ]]; then
  echo "Benchmark binary not found at ${BIN}." >&2
  echo "Build it or set KVIKIO_BENCHMARK_BIN." >&2
  exit 1
fi

REQUEST_SIZES="${REQUEST_SIZES:-1048576 4194304 8388608 16777216 67108864}"
READER_THREADS="${READER_THREADS:-1 4 8 16 32}"
TASK_SIZES="${TASK_SIZES:-0 4194304}"
DEVICE_MEMORY="${DEVICE_MEMORY:-false}"

for request_bytes in ${REQUEST_SIZES}; do
  for threads in ${READER_THREADS}; do
    for task_size in ${TASK_SIZES}; do
      "${BIN}" \
        --paths="${MANIFEST}" \
        --mode=cold \
        --measurement_bytes="${MEASUREMENT_BYTES}" \
        --request_bytes="${request_bytes}" \
        --reader_threads="${threads}" \
        --kvikio_task_size="${task_size}" \
        --device_memory="${DEVICE_MEMORY}"
    done
  done
done
