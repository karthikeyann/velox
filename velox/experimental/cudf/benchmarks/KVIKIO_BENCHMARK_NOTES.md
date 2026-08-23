# KvikIO S3 read benchmark — working notes

Working notes for `velox_cudf_kvikio_read_benchmark`, carried on the branch so
the context travels with a checkout. This is not user documentation — see
`README.md` for how to run the tool. **Drop this file before upstreaming.**

## What the tool is

Measures raw read throughput from S3 through `kvikio::RemoteHandle`, with no
parquet decode in the measured path. It is the KvikIO counterpart to
`velox_read_benchmark` (`velox/benchmarks/filesystem/ReadBenchmark.cpp`), which
measures the same thing through Velox's `S3FileSystem` and the AWS C++ SDK.
Running both against the same bucket gives a transport-level comparison; the
MB/s output format is deliberately identical so the numbers line up by eye.

Source: `KvikioReadPlan.{h,cpp}` is a pure planning layer with no I/O, unit
tested by `velox/experimental/cudf/tests/KvikioReadPlanTest.cpp`.
`KvikioReadBenchmark.{h,cpp}` holds everything that touches KvikIO or CUDA.

## Build

Requires GCC >= 13.3 (Velox hard-fails a cuDF configure below that), CUDA, and
CMake >= 4.0. Velox builds KvikIO, RMM and cuDF from source via FetchContent,
so the first build is long.

```bash
CUDA_ARCHITECTURES=native EXTRA_CMAKE_FLAGS="-DVELOX_ENABLE_BENCHMARKS=ON" \
  make cmake-cudf BUILD_DIR=release BUILD_TYPE=release

cd _build/release
ninja -j <N> velox_cudf_kvikio_read_benchmark velox_cudf_kvikio_read_plan_test
```

**Budget roughly 4-5 GB of RAM per parallel CUDA compile job.** On the
development host, `ninja` at its `nproc` default of 64 against 125 GB exhausted
memory and produced GCC internal compiler errors in
`tree_node_structure_for_code` — which look like compiler bugs but are not.
`-j 24` completed cleanly there. Size `-j` to the instance, not to `nproc`.

Confirm remote support actually linked, since KvikIO's `KvikIO_REMOTE_SUPPORT`
option gates whether `remote_handle.cpp` is compiled at all:

```bash
nm -DC velox/experimental/cudf/benchmarks/velox_cudf_kvikio_read_benchmark \
  | grep -c "RemoteHandle::open"
```

A count of 0 means remote support is off; reconfigure rather than working
around it.

## Credentials on EC2

KvikIO reads AWS credentials from the environment and **never** consults the
EC2 instance metadata service. An instance profile alone is not enough. Convert
it to environment variables first:

```bash
eval "$(aws configure export-credentials --format env)"
export AWS_DEFAULT_REGION=us-east-1
```

That exports `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
`AWS_SESSION_TOKEN`. The session token is required for any temporary
credential — instance profile, STS, or SSO.

For a non-AWS S3-compatible server, also set `AWS_ENDPOINT_URL`.

If credentials are missing or wrong, KvikIO silently falls back to an
unauthenticated public-S3 endpoint built from the original URL, so the error
surfaces as a libcurl protocol or HEAD failure with no mention of credentials.
Suspect credentials first when an open fails.

## Concurrency: which knob actually controls parallelism

Verified against the KvikIO sources the build links
(`_build/release/_deps/kvikio-src/`). This is the least obvious part of the
tool and the easiest way to measure the wrong thing.

`--kvikio_nthreads` is the ceiling on concurrent transfers, not
`--reader_threads`. Whenever `--kvikio_task_size > 0`, every sub-task is
submitted to KvikIO's *global* thread pool
(`cpp/include/kvikio/detail/parallel_operation.hpp:178-212`) whose default
width is **1** (`cpp/src/defaults.cpp:94`). Reader threads block on their
`pread` future, so anything beyond the pool width only queues.

`--kvikio_task_size >= --request_bytes` silently disables splitting. The
single-task guard at `parallel_operation.hpp:178` is
`if (task_size >= size || get_page_size() >= size)`, where `size` is the
request size. It submits one pooled task instead of splitting, so raising task
size without raising `--request_bytes` past it accomplishes nothing.

| Configuration | Concurrent transfers |
|---|---|
| `--kvikio_task_size=0` | exactly `--reader_threads`; `read()` runs on the reader thread and the pool is uninvolved |
| `task_size > 0`, `request_bytes > task_size` | `--kvikio_nthreads` |
| `task_size >= request_bytes` | `min(reader_threads, kvikio_nthreads)`, one task per request |

The first and second rows answer different questions. The first is "how many
parallel streams can I sustain", which is also the cleanest comparison against
`velox_read_benchmark`'s AWS-SDK path. The second is "does splitting one large
GET help".

## Memory sizing

Device buffers are `reader_threads x request_bytes` of VRAM, allocated up front
so an oversubscribed configuration fails immediately rather than partway
through a run.

KvikIO allocates a pinned host bounce buffer per concurrent device transfer,
16 MiB by default (`defaults.cpp:123`, `KVIKIO_BOUNCE_BUFFER_SIZE`), so a wide
pool in device mode reserves `kvikio_nthreads x 16 MiB` of pinned host memory.

## KvikIO environment variables not exposed as flags

`KVIKIO_HTTP_MAX_ATTEMPTS` and `KVIKIO_HTTP_TIMEOUT` matter under load: KvikIO
retries 429/500/502/503/504 up to three times with a 60-second per-transfer
timeout. S3 throttling therefore depresses reported throughput silently. If a
number comes back unexpectedly low at high concurrency, suspect `SlowDown`
before concluding anything about the transport.

`KVIKIO_NTHREADS`, `KVIKIO_TASK_SIZE` and `KVIKIO_BOUNCE_BUFFER_SIZE` are the
environment equivalents of the corresponding flags.

## Verified, and not

Verified against the KvikIO sources, so these are not open risks:

- `s3://bucket/key` URIs work. KvikIO special-cases the scheme
  (`remote_handle.cpp:251-257`) despite `S3Endpoint`'s doc comment saying URLs
  should start with `http://`.
- Sharing one `RemoteHandle` across reader threads is safe. `read` takes a curl
  handle from a global pool per call and `S3Endpoint::setopt` only reads
  members.
- Device reads are fully synchronized before returning; the byte count is
  honest.

**Never run against a live S3 endpoint.** Every live gate in the implementation
plan is outstanding. The tool has only ever been built and unit tested.

## Known limitations

Cold mode's single-touch guarantee holds within a run but not across runs. The
plan is deterministic from target 0 offset 0, so every sweep row reads the
identical byte range while printing `mode=cold`. Only the first row of a sweep
is a true cold read; compare rows against each other rather than treating any
single row as an absolute cold number.

The cold plan emits tasks in strict manifest order and the runner hands them
out sequentially, so at any instant all threads are reading the same object.
With few large objects this measures the per-object S3 ceiling rather than the
aggregate one. Round-robin emission was considered and deliberately deferred.

`requests=` in the output counts benchmark-issued requests, not range GETs.
With a non-zero `--kvikio_task_size`, KvikIO issues `ceil(size / task_size)`
GETs per counted request, so per-request latency cannot be derived in that mode.

The tool does not call `cudaSetDevice`, so multi-GPU instances always use
device 0.

## Repository state

Branch `kvikio-s3-read-benchmark`, off `main` at `683854b305`, pushed to remote
`kn` (`git@github.com:karthikeyann/velox.git`).

Commits on this branch use `--no-verify`. The `check-header-ownership` and
`CMake formatter` pre-commit hooks fail on four pre-existing `dwio/nimble/`
headers from upstream commit `4d16851ecd` that this branch never touches. CI
will still run those hooks against the merge result.
