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
# manifest, printing one result line per combination and teeing every line to
# a results file.
#
# Credentials come from the environment. For a non-AWS server also export
# AWS_ENDPOINT_URL.
#
# Overrides, all optional:
#   REQUEST_SIZES         Range request sizes to sweep, in bytes.
#   READER_THREADS        Reader thread counts to sweep.
#   TASK_SIZES            KvikIO task sizes to sweep; 0 means one range GET.
#   KVIKIO_NTHREADS       Width of KvikIO's internal thread pool. Defaults to
#                         the reader thread count of each combination, so the
#                         pool never becomes the bottleneck for the row.
#   DEVICE_MEMORY         'true' reads into device memory instead of host.
#   KVIKIO_BENCHMARK_BIN  Path to velox_cudf_kvikio_read_benchmark.
#   RESULTS_FILE          Where to tee the result lines.
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

if [[ ! -r ${MANIFEST} ]]; then
  echo "Manifest not readable at ${MANIFEST}." >&2
  exit 1
fi

REQUEST_SIZES="${REQUEST_SIZES:-1048576 4194304 8388608 16777216 67108864}"
READER_THREADS="${READER_THREADS:-1 4 8 16 32}"
TASK_SIZES="${TASK_SIZES:-0 4194304}"
DEVICE_MEMORY="${DEVICE_MEMORY:-false}"
RESULTS_FILE="${RESULTS_FILE:-${PWD}/kvikio_sweep_$(date +%Y%m%d-%H%M%S).txt}"

# An unwritable results file is a configuration error, so surface it now rather
# than after the first run has already burned its egress.
if ! (: >"${RESULTS_FILE}") 2>/dev/null; then
  echo "Cannot write the results file at ${RESULTS_FILE}." >&2
  exit 1
fi

cat >&2 <<EOF
Only the FIRST run below is a true cold read. Every run walks the same byte
range from the start of the manifest, so later runs may be served from the S3
server's cache and read faster for that reason alone. Compare rows against
each other; do not quote any single row as an absolute cold-read number.

Writing results to ${RESULTS_FILE}
EOF

for request_bytes in ${REQUEST_SIZES}; do
  for threads in ${READER_THREADS}; do
    for task_size in ${TASK_SIZES}; do
      # A transient 503 partway through must not discard the rows still to
      # come, so record the gap and carry on. Configuration errors are caught
      # by the guards above, before the first run.
      if ! "${BIN}" \
        --paths="${MANIFEST}" \
        --mode=cold \
        --measurement_bytes="${MEASUREMENT_BYTES}" \
        --request_bytes="${request_bytes}" \
        --reader_threads="${threads}" \
        --kvikio_task_size="${task_size}" \
        --kvikio_nthreads="${KVIKIO_NTHREADS:-${threads}}" \
        --device_memory="${DEVICE_MEMORY}"; then
        combination="request=${request_bytes} threads=${threads} kvikio_task_size=${task_size} device=${DEVICE_MEMORY}"
        # The stdout marker keeps the results file from silently omitting the
        # row; the stderr line stays visible when stdout is redirected.
        echo "FAILED ${combination}"
        echo "Benchmark exited non-zero for ${combination}; continuing." >&2
      fi
    done
  done
done | tee "${RESULTS_FILE}"
