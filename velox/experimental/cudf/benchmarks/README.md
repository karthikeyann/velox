# CuDF Benchmarks

Benchmark binaries for TPC-H and TPC-DS queries with optional CuDF GPU acceleration.

## Binaries

| Binary | Benchmark | Mode | Source |
|--------|-----------|------|--------|
| `velox_tpch_benchmark` | TPC-H (Q1-Q22) | CPU | `velox/benchmarks/tpch/` |
| `velox_cudf_tpch_benchmark` | TPC-H (Q1-Q22) | GPU | `CudfTpchBenchmark.cpp`, `CudfTpchBenchmarkMain.cpp` |
| `velox_tpcds_benchmark` | TPC-DS (Q1-Q99) | CPU | `velox/benchmarks/tpcds/` |
| `velox_cudf_tpcds_benchmark` | TPC-DS (Q1-Q99) | GPU | `CudfTpcdsBenchmark.cpp` |

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

### S3 I/O modes

`--cudf_io_mode` turns `velox_cudf_tpch_benchmark` into an I/O benchmark: it
scans one TPC-H table out of S3 and reports what the read cost instead of
running Q1-Q22. The three modes read the same objects through the same KvikIO
transport and differ only in how much of each object they touch and how far up
the stack the bytes travel, so their per-mode figures are directly comparable.

| `--cudf_io_mode` | What it reads | What the timed window covers |
|------------------|---------------|------------------------------|
| `decode_discard` | The compressed column chunks a `TableScan` of every column of the table needs | Task start to task finish: range fetch, GPU Parquet decode, and dropping each batch in a sink that produces no output |
| `raw_parquet_ranges` | The same compressed column chunks, selected from the Parquet footer but never decoded | The payload pass only. Opening each object, reading its footer and selecting ranges are timed separately as `setup_s` |
| `raw_file` | Every byte of every object, in `--cudf_io_read_size_bytes` pieces | The payload pass only, with object opening and size discovery in `setup_s`. Its runner reads no Parquet metadata |

Before any of them runs, the TPC-H query builder reads the schema of the first
object listed for each table it finds under `--data_path`. That read goes
through the Velox connector rather than KvikIO and happens outside every timing
and byte counter reported below, so `raw_file` still pays for one Parquet footer
even though its payload runner never looks at one.

The empty default runs the ordinary TPC-H queries. It ignores the `--cudf_io_*`
flags, while `--num_drivers` and `--num_repeats` keep their ordinary meaning.

#### Preflight

An enabled mode is checked before anything is registered, opened or requested,
because the schema read above would otherwise be the first thing to reach
storage. A run is rejected up front unless all of the following hold:

- `--cudf_io_mode` names a known mode;
- `--data_format=parquet`;
- `--run_query_verbose=-1` and `--io_meter_column_pct=0`;
- `--num_drivers`, `--num_repeats` and `--cudf_io_read_size_bytes` are
  positive;
- `--cudf_io_table` is one of the eight TPC-H tables;
- `--velox_cudf_table_scan=true`;
- the connector endpoint and `AWS_ENDPOINT_URL` agree, or neither is set;
- `--data_path/<table>` is a local file, not a directory, and lists at least
  one object;
- every line of that file is non-empty and begins with `s3:` or `s3a:`.

Only the manifest is read, from the local file system, so a run that cannot
produce a meaningful measurement costs nothing and leaves the process as it
found it.

`--velox_cudf_table_scan=true` is required rather than merely recommended: all
three modes measure the direct cuDF/KvikIO read path, and the CPU Hive scan
publishes none of the payload counters they report.

The binary must be built with S3 support:

```bash
CUDA_ARCHITECTURES="native" \
  EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_BENCHMARKS=ON -DVELOX_ENABLE_S3=ON" make cudf
cd _build/release && ninja velox_cudf_tpch_benchmark
```

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--cudf_io_mode` | `""` | `""`, `decode_discard`, `raw_parquet_ranges`, or `raw_file` |
| `--cudf_io_table` | `lineitem` | TPC-H table to scan |
| `--cudf_io_read_size_bytes` | `134217728` | Bound on each `raw_file` read and on each of its destination buffers. Only `raw_file` reads the value, but every enabled mode requires it to be positive |
| `--connector_properties` | `""` | Path to a `key=value` file of Hive connector properties, one per line |
| `--num_drivers` | `4` | Scan drivers in `decode_discard`, reader threads in the raw modes |
| `--num_repeats` | `1` | Times to repeat the run, one output line each |

`--data_format=parquet` and `--velox_cudf_table_scan=true` are required.
`--run_query_verbose` and `--io_meter_column_pct` are rejected, because neither
means anything without a query. See [Preflight](#preflight) for the full list
and for when it is checked.

#### Manifest

Point `--data_path` at a directory holding one file per table, named after the
table, whose lines are the object URIs of that table:

```
/tmp/tpch-s3/
  lineitem
```

```
s3://my-bucket/tpch/sf1000/lineitem/part-00000.parquet
s3://my-bucket/tpch/sf1000/lineitem/part-00001.parquet
s3://my-bucket/tpch/sf1000/lineitem/part-00002.parquet
```

Every line is taken as a URI, so unlike `--paths` on the standalone KvikIO
benchmark, blank lines and `#` comments are rejected rather than skipped. A
table with no file in the directory is left empty and costs nothing, so a
one-table directory is enough. All paths must be `s3://` or `s3a://`; the
benchmark rejects local paths rather than quietly measuring the local disk.

`decode_discard` builds a Velox task and gives it one split per object. The raw
modes build neither a task nor a split: their runner schedules whole objects
across reader threads directly. Either way the object is the unit of work, so
concurrency is capped by the number of objects: a manifest with fewer objects
than `--num_drivers` leaves drivers or workers idle, and the output reports
`effective_drivers` (or `effective_workers`) alongside the requested count so
that is visible.

#### Endpoint and credentials

Two independent stacks reach S3 and each is configured separately. The Velox
connector reads the object's schema and, in `decode_discard`, resolves the
split; KvikIO moves every payload byte in all three modes and reads its
configuration only from the environment.

```
# /tmp/minio.properties, passed as --connector_properties
hive.s3.endpoint=my-s3-host:9000
hive.s3.ssl.enabled=false
hive.s3.path-style-access=true
hive.s3.aws-access-key=<access-key>
hive.s3.aws-secret-key=<secret-key>
```

```bash
# The same server, as KvikIO reads it.
export AWS_ENDPOINT_URL=http://my-s3-host:9000
export AWS_ACCESS_KEY_ID=<access-key>
export AWS_SECRET_ACCESS_KEY=<secret-key>
export AWS_DEFAULT_REGION=us-east-1
export AWS_EC2_METADATA_DISABLED=true
```

`AWS_ENDPOINT_URL` must carry an explicit `http://` or `https://` scheme, while
`hive.s3.endpoint` must not. The benchmark normalizes the connector endpoint
using `hive.s3.ssl.enabled` and refuses to start when the two sides disagree,
or when only one of them names an endpoint. Against real AWS S3, set neither.

Exporting `AWS_ENDPOINT_URL` is also what selects path-style addressing:
KvikIO fetches `<AWS_ENDPOINT_URL>/<bucket>/<key>` whenever it is set, and only
builds a virtual-host URL when it is not. Set `hive.s3.path-style-access=true`
to match on the connector side.

Missing or wrong credentials do not surface as an authentication error. KvikIO
falls back to an unauthenticated public-bucket endpoint built from the original
`s3://` URL, which libcurl cannot speak, so the symptom is `Unsupported URL
scheme` or a failed HEAD request. A refused connection reads the same way, so
check that the server is up before suspecting the keys.

All three modes go straight to KvikIO: the benchmark forces
`cudf.hive.use-buffered-input=false`, and `decode_discard` additionally forces
`cudf.hive.use-experimental-reader=true` so the scan fetches exact column-chunk
ranges rather than whole row groups. Both override anything
`--connector_properties` sets.

#### Run

```bash
# From: _build/release/velox/experimental/cudf/benchmarks/

# Decode to the GPU and discard.
./velox_cudf_tpch_benchmark \
  --data_path=/tmp/tpch-s3 \
  --data_format=parquet \
  --cudf_io_mode=decode_discard \
  --cudf_io_table=lineitem \
  --connector_properties=/tmp/minio.properties \
  --num_drivers=8 \
  --num_repeats=3

# Fetch the same column chunks, decode nothing.
./velox_cudf_tpch_benchmark \
  --data_path=/tmp/tpch-s3 \
  --data_format=parquet \
  --cudf_io_mode=raw_parquet_ranges \
  --cudf_io_table=lineitem \
  --connector_properties=/tmp/minio.properties \
  --num_drivers=8 \
  --num_repeats=3

# Fetch whole objects in bounded pieces.
./velox_cudf_tpch_benchmark \
  --data_path=/tmp/tpch-s3 \
  --data_format=parquet \
  --cudf_io_mode=raw_file \
  --cudf_io_table=lineitem \
  --connector_properties=/tmp/minio.properties \
  --cudf_io_read_size_bytes=$((16 * 1024 * 1024)) \
  --num_drivers=8 \
  --num_repeats=3
```

#### Interpreting the output

One line per repeat, prefixed by a `note:` line naming the counters that are
diagnostics rather than rates.

Every byte counter is payload: what the reader asked storage for and what came
back. Each mode reports both, separately:

| Mode | Requested | Completed |
|------|-----------|-----------|
| `decode_discard` | `compressed_requested_bytes` | `compressed_completed_bytes` |
| `raw_parquet_ranges` | `requested_bytes` | `completed_bytes` |
| `raw_file` | `requested_bytes` | `completed_bytes` |

Request and response headers, TLS, retried transfers, and the footer reads that
`raw_parquet_ranges` charges to `setup_s` are all outside them, so an S3 server
or NIC counter will always read higher than these numbers. The two are expected
to be equal; a short read fails the run rather than lowering the completed
count, which is why they are reported apart rather than as one figure.

`compressed_completed_bytes_per_s` and `completed_bytes_per_s` divide completed
payload bytes by the wall time of the whole pass, so they are the throughput
figures. `column_chunk_read_wall_s`, `parquet_decode_gpu_s` and `read_wall_s`
are summed over every driver or worker and overlap in wall time, so they can
exceed `elapsed_s`; use them to see where the time went, never as a
denominator.

`raw_file` bounds its memory: each worker holds one `--cudf_io_read_size_bytes`
buffer at a time, so that size times `effective_workers` is an upper bound on
the device memory held for read destinations, however large the objects are.
Actual usage is lower whenever a worker's final piece is short or workers are
not all fetching at once. `raw_parquet_ranges` instead fetches all of a
file's selected chunks at once, allocating one compressed buffer per file
exactly as the decode path does, which is what makes its byte counts and its
memory profile comparable to `decode_discard` rather than to `raw_file`.

`--num_repeats` rereads the same objects, so only the first repeat can be cold,
and even that depends on what the machine and the server have already cached.
Compare repeats against each other rather than quoting one as a cold-read
number.

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
| `--velox_cudf_table_scan` | `true` | Use CuDF table scan. Required by every `--cudf_io_mode` |
| `--cudf_properties` | `""` | Path to a CudfConfig properties file (key=value per line). See `CudfConfig.h` for available keys |

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
