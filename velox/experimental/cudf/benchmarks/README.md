# CuDF Benchmarks

Benchmark binaries for TPC-H and TPC-DS queries with optional CuDF GPU acceleration.

## Binaries

| Binary | Benchmark | Mode | Source |
|--------|-----------|------|--------|
| `velox_tpch_benchmark` | TPC-H (Q1-Q22) | CPU | `velox/benchmarks/tpch/` |
| `velox_cudf_tpch_benchmark` | TPC-H (Q1-Q22) | GPU | `CudfTpchBenchmark.cpp` |
| `velox_tpcds_benchmark` | TPC-DS (Q1-Q99) | CPU | `velox/benchmarks/tpcds/` |
| `velox_cudf_tpcds_benchmark` | TPC-DS (Q1-Q99) | GPU | `CudfTpcdsBenchmark.cpp` |
| `velox_cudf_kvikio_read_benchmark` | Raw S3 read throughput | GPU/CPU | `KvikioReadBenchmark.cpp` |

CPU binaries use HiveConnector. GPU binaries use CudfHiveConnector and register
cuDF GPU operator replacements.

## Build

```bash
# GPU binaries (requires CUDA)
CUDA_ARCHITECTURES="native" EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_BENCHMARKS=ON" make cudf
cd _build/release
ninja velox_cudf_tpch_benchmark velox_cudf_tpcds_benchmark

# CPU-only binaries (no CUDA required)
ninja velox_tpch_benchmark velox_tpcds_benchmark
```

---

## TPC-H Benchmark

### Data

Generate TPC-H parquet data using the [velox-testing](https://github.com/rapidsai/velox-testing) data generation tool (see [Data Generation](#data-generation) below), or the standard `dbgen` tool and convert decimal columns to float.

### Run

```bash
# CPU - all queries
./velox_tpch_benchmark --data_path=/path/to/tpch/sf100 --data_format=parquet

# GPU (CuDF) - all queries
./velox_cudf_tpch_benchmark --data_path=/path/to/tpch/sf100 --data_format=parquet
```

---

## TPC-DS Benchmark

TPC-DS plans are loaded from pre-dumped Velox plan JSON files (serialized from Presto).

### 1. Get Plan JSON Files

Clone the plans repository:

```bash
git clone https://github.com/karthikeyann/VeloxPlans.git
# Plans are at: VeloxPlans/presto/tpcds/sf100/
```

The directory contains `Q1.json`, `Q2.json`, ..., `Q99.json`.

### 2. Get TPC-DS Data

Generate TPC-DS parquet data using the [velox-testing](https://github.com/rapidsai/velox-testing) data generation tool (see [Data Generation](#data-generation) below). The data directory must have one subdirectory per table:

```
/path/to/tpcds/sf100/
  store_sales/
  customer/
  date_dim/
  item/
  ...
```

Each subdirectory contains parquet files for that table.

### 3. Run

**CPU - all queries (folly benchmark mode):**

```bash
./velox_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet
```

**CPU - single query with stats:**

```bash
./velox_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet \
  --run_query_verbose=1
```

**GPU (CuDF) - all queries:**

```bash
./velox_cudf_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet
```

**GPU (CuDF) - single query with stats:**

```bash
./velox_cudf_tpcds_benchmark \
  --data_path=/path/to/tpcds/sf100 \
  --plan_path=/path/to/VeloxPlans/presto/tpcds/sf100 \
  --data_format=parquet \
  --run_query_verbose=1
```

### TPC-DS Flags

These flags are shared by both CPU and GPU binaries:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | (required) | Root directory of TPC-DS table data |
| `--plan_path` | (required) | Directory containing Q*.json plan files |
| `--data_format` | `parquet` | Data file format |
| `--run_query_verbose` | `-1` | Run single query with stats (`-1` = run all) |
| `--num_drivers` | `4` | Number of parallel drivers |
| `--include_results` | `false` | Print query results |

### CuDF Flags (GPU binaries only)

These flags apply to `velox_cudf_tpch_benchmark` and `velox_cudf_tpcds_benchmark`:

| Flag | Default | Description |
|------|---------|-------------|
| `--cudf_chunk_read_limit` | `0` | Chunk read limit for cuDF parquet reader |
| `--cudf_pass_read_limit` | `0` | Pass read limit for cuDF parquet reader |
| `--cudf_gpu_batch_size_rows` | `100000` | GPU batch size in rows |
| `--velox_cudf_table_scan` | `true` | Use CuDF table scan |
| `--cudf_properties` | `""` | Path to a CudfConfig properties file (key=value per line). See `CudfConfig.h` for available keys |

---

## KvikIO Read Benchmark

Measures raw read throughput from S3 through `kvikio::RemoteHandle`, with no
parquet decode. Use it to find the transport ceiling before asking how much of
it the TPC-H scan reaches.

### Build

```bash
CUDA_ARCHITECTURES="native" EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_BENCHMARKS=ON" make cudf
cd _build/release && ninja velox_cudf_kvikio_read_benchmark
```

The binary lands in `_build/release/velox/experimental/cudf/benchmarks/`.

### Credentials

KvikIO reads credentials from these environment variables and from nowhere
else. It never queries the EC2 instance metadata service, so an instance
profile by itself leaves it unauthenticated:

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
# Required when using temporary credentials from an IAM role, STS, or SSO.
export AWS_SESSION_TOKEN=...
# Only for a non-AWS S3 server such as MinIO.
export AWS_ENDPOINT_URL=http://my-s3-host:9000
```

On EC2, materialize the instance profile into the environment first (AWS CLI
2.9 or newer):

```bash
eval "$(aws configure export-credentials --format env)"
export AWS_DEFAULT_REGION=us-east-1  # The bucket's region.
```

The AWS CLI resolves instance-profile credentials on its own, so `aws s3 cp`
succeeding on the same box says nothing about whether the benchmark can read
the bucket. Only the exported variables above matter to it.

Those exported keys are temporary. If a long sweep starts failing partway
through, re-run the export and restart from the combination that failed.

Missing credentials do not surface as an authentication error. KvikIO falls
back to a public-bucket endpoint built from the original `s3://` URL, which
libcurl cannot speak, so the first symptom is `Unsupported URL scheme` or a
failed HEAD request. The benchmark's error message says as much.

### Manifest

`--paths` names a manifest holding one object URI per line. Blank lines and
lines whose first non-whitespace character is `#` are ignored:

```
# TPC-H SF1000 lineitem, first three files.
s3://my-bucket/tpch/sf1000/lineitem/part-00000.parquet
s3://my-bucket/tpch/sf1000/lineitem/part-00001.parquet
s3://my-bucket/tpch/sf1000/lineitem/part-00002.parquet
```

Generate one for a whole prefix:

```bash
aws s3 ls --recursive s3://my-bucket/tpch/sf1000/lineitem/ \
  | awk '{print "s3://my-bucket/" $4}' > /tmp/lineitem.manifest
```

### Run

Start with `--list_targets`. It opens every target and exits without moving
payload, so it costs one HEAD request per object and no egress, which makes it
the cheapest check that credentials, region and object paths are all right:

```bash
# From: _build/release/velox/experimental/cudf/benchmarks/
./velox_cudf_kvikio_read_benchmark \
  --paths=/tmp/lineitem.manifest \
  --list_targets
```

It prints one `<size>\t<uri>` line per target and a final `total_bytes=`. Cold
mode refuses to read more than `total_bytes`, so that figure is also the upper
bound on `--measurement_bytes`.

Then take one measurement:

```bash
./velox_cudf_kvikio_read_benchmark \
  --paths=/tmp/lineitem.manifest \
  --mode=cold \
  --request_bytes=$((8 * 1024 * 1024)) \
  --measurement_bytes=$((4 * 1024 * 1024 * 1024)) \
  --reader_threads=16
```

Opening the manifest takes one HEAD request per object and prints its progress
to stderr, so a few hundred objects means tens of seconds before the transfer
starts. That is not a hang.

### Sweep

`kvikio_sweep.sh` walks request size × reader threads × KvikIO task size,
printing one result line per combination and teeing every line to a results
file whose path it prints when it starts. One combination failing does not stop
the sweep; the row is replaced by a `FAILED` line naming the combination.

```bash
# From: velox/experimental/cudf/benchmarks/ (source tree)

# Default: 4 GiB measurement.  The manifest must total at least 4 GiB.
./kvikio_sweep.sh /tmp/lineitem.manifest

# Smaller manifest: pass measurement_bytes explicitly.
./kvikio_sweep.sh /tmp/lineitem.manifest $((1 * 1024 * 1024 * 1024))
```

The results file lands in the current directory, which is inside the source
tree if the sweep is launched from there. Set `RESULTS_FILE` to put it
somewhere else.

It opens with a `#`-commented header naming the manifest, the measurement size,
the three axes swept, the pool width used and the cold-read caveat below. The
benchmark's own warnings go to stderr and so are absent from the file; the
header is what keeps the file interpretable once it has been copied off the
machine.

**Only the first run of a sweep is a true cold read.** Every run walks the same
byte range from the start of the manifest, so later runs may be served from the
S3 server's cache and read faster for that reason alone, even though every row
prints `mode=cold`. Compare rows against each other; do not quote any single
row as an absolute cold-read number. To make a specific row cold again, run it
by itself against a manifest whose objects the machine has not touched.

The default matrix is 5 request sizes × 5 thread counts × 2 task sizes, so 50
runs. At the default 4 GiB per run that is 200 GiB of egress, and at a few
GB/s the wall clock runs to an hour or more before the per-run HEAD requests
are counted. Trim the matrix through the environment:

| Variable | Default | Meaning |
|----------|---------|---------|
| `REQUEST_SIZES` | `1048576 4194304 8388608 16777216 67108864` | Range request sizes to sweep, in bytes |
| `READER_THREADS` | `1 4 8 16 32` | Reader thread counts to sweep |
| `TASK_SIZES` | `0 4194304` | KvikIO task sizes to sweep; `0` is one range GET per request |
| `KVIKIO_NTHREADS` | the row's thread count | Width of KvikIO's internal pool, on `task_size != 0` rows only |
| `DEVICE_MEMORY` | `false` | `true` reads into device memory |
| `KVIKIO_BENCHMARK_BIN` | `_build/release/.../velox_cudf_kvikio_read_benchmark` | Path to the binary |
| `RESULTS_FILE` | `./kvikio_sweep_<timestamp>.txt` | Where to tee the result lines |

```bash
# A four-run smoke sweep at 1 GiB each.
REQUEST_SIZES="8388608" READER_THREADS="8 16" TASK_SIZES="0 4194304" \
  ./kvikio_sweep.sh /tmp/lineitem.manifest $((1024 * 1024 * 1024))
```

On rows with a non-zero task size the sweep passes `--kvikio_nthreads` equal to
the row's reader thread count, because KvikIO's own default pool is one thread
wide: with a non-zero task size and an unwidened pool, every reader queues
behind a single worker and the whole `--reader_threads` axis measures one
serialized stream. A `task_size=0` row never touches the pool, so the sweep
leaves it at the KvikIO default and those rows print `kvikio_nthreads=1`.
Reading `1` on a `task_size=0` row is not a misconfiguration; a widened pool
there would only have printed a number that served no request.

Each reader thread owns one destination buffer of `--request_bytes`, so the
largest default row, 32 threads at 64 MiB, holds 2 GiB. In host mode that is
resident RSS, first-touched before the clock starts; in device mode it is the
same 2 GiB of VRAM. Raising both axes together is what runs a machine out of
memory, not either one alone.

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--paths` | (required) | Manifest file of object URIs, one per line |
| `--list_targets` | `false` | Print each target's size and exit without reading |
| `--mode` | `cold` | `cold` reads each byte at most once; `warm` re-reads at random offsets |
| `--request_bytes` | `8388608` | Size of each range request |
| `--measurement_bytes` | `1073741824` | Total payload bytes per run |
| `--reader_threads` | `8` | Concurrent readers, external to KvikIO |
| `--kvikio_task_size` | `0` | `0` is one range GET per request; otherwise KvikIO's split granularity |
| `--kvikio_nthreads` | `0` | Width of KvikIO's internal pool; `0` keeps the KvikIO default of 1 |
| `--kvikio_bounce_buffer_bytes` | `0` | Bounce buffer size for device reads; `0` keeps the KvikIO default |
| `--device_memory` | `false` | Read into device instead of host memory |
| `--seed` | `0` | Seed for warm-mode offsets |

### Interpreting the output

One line per run on stdout, in the same MB/s units `velox_read_benchmark`
prints, echoing the knobs it ran with. Progress and warnings go to stderr, so
redirecting stdout captures results only:

```
1843.2 MB/s mode=cold request=8388608 threads=16 kvikio_task_size=0 kvikio_nthreads=1 device=false bytes=4294967296 requests=512 elapsed_s=2.330
```

- `kvikio_nthreads=` is the live pool width, not the flag. It reads `1` unless
  `--kvikio_nthreads` or the `KVIKIO_NTHREADS` environment variable set it, and
  it is meaningful only when `kvikio_task_size=` is non-zero, since nothing
  else uses the pool.
- `requests=` counts the range requests the benchmark issued, which is not the
  number of range GETs. With a non-zero `--kvikio_task_size`, KvikIO splits
  each one into `ceil(size / task_size)` GETs, so per-request latency cannot be
  derived from `requests=` in that mode.
- `elapsed_s=` covers the transfer loop, not the whole process. Deliberately
  outside it: parsing the manifest and its HEAD request per target, building
  the read plan, allocating and first-touching the destination buffers, the
  CUDA primary context that the first device allocation creates, pre-warming
  KvikIO's pinned bounce-buffer pool in device mode, loading the CUDA driver,
  starting and joining the reader threads, and freeing the buffers.
- Two costs are still inside `elapsed_s=`, and both make the first requests of
  a run slower than its steady state. Opening the targets warms DNS and TLS
  only for the calling thread's curl handle, so every reader thread pays its
  own first DNS, TCP and TLS handshake in the window; this hits host and device
  mode equally. In device mode each reader thread additionally pays for the one
  CUDA stream KvikIO creates per thread on its first transfer, which is small
  next to a handshake. Both argue for a longer run rather than a shorter one:
  at a `--measurement_bytes` small enough that the ramp is a visible fraction
  of the total, the reported figure sits below the sustained rate.

Two conditions produce a number that is real but does not mean what the line
says, so the run warns on stderr and continues:

- The KvikIO pool is one thread wide while a non-zero task size and more than
  one reader thread are in play, which serializes the readers.
- The plan holds fewer than four requests per reader thread, which measures
  request latency and ramp rather than sustained throughput. Raise
  `--measurement_bytes` or lower `--request_bytes`.

Cold mode refuses to run when `--measurement_bytes` exceeds the manifest total,
because wrapping around would serve the excess from the server's cache and
report a warm result as a cold one. Either shrink the measurement or add
objects to the manifest.

### KvikIO environment variables

These change throughput and are worth knowing about even where a flag covers
the same ground:

| Variable | Default | Effect |
|----------|---------|--------|
| `KVIKIO_NTHREADS` | `1` | Width of KvikIO's internal thread pool. `--kvikio_nthreads` overrides it when non-zero |
| `KVIKIO_NUM_THREADS` | `1` | Alias for `KVIKIO_NTHREADS`. Setting both to different values is an error |
| `KVIKIO_TASK_SIZE` | 4 MiB | KvikIO's own split granularity. This binary always passes `--kvikio_task_size` explicitly, so the variable has no effect here |
| `KVIKIO_BOUNCE_BUFFER_SIZE` | 16 MiB | Size of the pinned host buffer each device transfer stages through. `--kvikio_bounce_buffer_bytes` overrides it when non-zero |
| `KVIKIO_HTTP_MAX_ATTEMPTS` | `3` | Attempts per transfer before KvikIO gives up |
| `KVIKIO_HTTP_TIMEOUT` | `60` | Per-transfer timeout, in seconds |

KvikIO retries HTTP 429, 500, 502, 503 and 504 up to `KVIKIO_HTTP_MAX_ATTEMPTS`
times, each with a `KVIKIO_HTTP_TIMEOUT` budget. S3 throttling therefore shows
up as a low throughput figure rather than as an error, which is worth checking
before believing a number that comes back well under the instance's network
ceiling.

The benchmark never calls `cudaSetDevice`, so on a multi-GPU instance
`--device_memory=true` always reads into device 0.

---

## Data Generation

Both TPC-H and TPC-DS parquet data can be generated using the
[velox-testing](https://github.com/rapidsai/velox-testing) repository.
Full instructions are also available in the
[VeloxPlans TPC-DS README](https://github.com/karthikeyann/VeloxPlans/tree/main/presto/tpcds/sf100).

### Quick Start

```bash
# 1. Clone velox-testing
git clone https://github.com/rapidsai/velox-testing.git
cd velox-testing

# 2. Install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r benchmark_data_tools/requirements.txt

# 3. Generate TPC-DS data (sf100)
python benchmark_data_tools/generate_data_files.py \
  --benchmark-type tpcds \
  --data-dir-path /path/to/tpcds/sf100/data \
  --scale-factor 100 \
  --convert-decimals-to-floats

# 4. Generate TPC-H data (sf100)
python benchmark_data_tools/generate_data_files.py \
  --benchmark-type tpch \
  --data-dir-path /path/to/tpch/sf100/data \
  --scale-factor 100 \
  --convert-decimals-to-floats
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--benchmark-type` | `tpcds` or `tpch` |
| `--data-dir-path` | Output directory for parquet files |
| `--scale-factor` | Scale factor (e.g. `1`, `10`, `100`) |
| `--convert-decimals-to-floats` | Convert decimal columns to double (recommended for Velox) |

The output directory will contain one subdirectory per table, each with `.parquet` files.
For a quick sanity check, use `--scale-factor 1` first.
