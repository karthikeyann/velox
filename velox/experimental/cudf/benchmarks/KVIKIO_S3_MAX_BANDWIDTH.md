# Maximising S3 read bandwidth in `velox_cudf_kvikio_read_benchmark`

Host: `g7e.48xlarge` `i-0c901f3cdce7a0e50`, us-east-2, four ENA cards, 2x Xeon
Platinum 8559C (96 cores, 640 MiB L3), 8x RTX PRO 6000. Source:
`s3://rapids-tpch/tpch-rs/scale-1000/lineitem`, 60 objects, 180 GiB, split into
four disjoint 15-object manifests, one per card.

Reference: `repro-g7e48-s3-https-1400`, which reached 1,399 Gbps on this host.

## Result

Data is genuinely delivered into memory a consumer can read. Four cards, 300 GiB
per process, means over repeats with the observed range:

| Destination | Gbps (mean) | Range | Note |
|---|---:|---|---|
| Pinned host (`cudaHostAlloc`) | **814** | 763-863, n=3 | ready for async H2D |
| Paged host + 2 MiB huge pages | 772 | 730-812, n=5 | |
| Paged host, 4 KiB pages | 753 | 700-821, n=4 | starting point |
| Device memory, one GPU per process | **723** | n=1 | previously failed outright |

Run-to-run spread is about +/-13%, which is wider than the gaps between the three
host variants. Read those three as roughly equivalent on throughput, and choose
between them on what the consumer needs: pinned is required for an async
host-to-device copy, and huge pages make buffer setup much faster. The device
result is a correctness fix rather than a tuning win — before it, four processes
overcommitted one GPU and two of them died.

Diagnostic only, payload accepted and dropped:

| Mode | Gbps |
|---|---:|
| Count-and-discard receive ceiling | **1,070** |

The discard number is the like-for-like comparison against the reference, since
that campaign also counted and discarded into a small reusable buffer. On equal
terms this is 76% of 1,399 Gbps. The delivered-to-memory numbers are the ones
that matter for a real pipeline, and the gap between 1,070 and 863 is the cost
of actually landing the bytes.

## Measured the reference's way: ~1,250-1,272 Gbps against its 1,399

`--warmup_seconds` and `--score_seconds` were added so the run can be measured
the way the reference measured it: 30 s warmup, then a fixed 60 s scored window.
Previously every number was either a byte-budget run, whose window length is a
consequence of the achieved rate, or a few seconds of instantaneous NIC sampling.

Like for like, after 30 s warmup over a 60 s scored window:

| Metric | This benchmark | Reference | Ratio |
|---|---:|---:|---:|
| Application body | **1,323 +/- 8 Gbps** | 1,399.4 Gbps | **94.6%** |
| NIC RX | **1,336 +/- 4 Gbps** | 1,413.0 Gbps | **94.6%** |

Five consecutive runs at the same configuration, 30 s scored after 15 s warmup:
NIC RX 1,340.7 / 1,333.5 / 1,335.2 / 1,333.3 / 1,339.1 Gbps. **Run-to-run spread
is 0.3%**, not the several percent claimed earlier in this document -- that
figure came from comparing across configurations and window lengths, which is a
different quantity. At a fixed configuration with a scored window the
measurement is tight enough that a 1% change is real.

### The hardware bound: each ENI caps at ~397 Gbps

One card driven alone, whole node, at three very different reader counts:

| Readers | app Gbps | NIC RX Gbps |
|---:|---:|---:|
| 700 | 391.0 | **396.7** |
| 1,400 | 373.0 | **396.8** |
| 2,100 | 267.5 | **396.8** |

The NIC figure is flat to three digits while the application figure falls away.
A resource limit varies with load; this does not, so ~397 Gbps is the ENI's line
rate rather than anything about cores, threads or S3. `bw_in_allowance_exceeded`
stays at 0 throughout, so this is the configured link rate and not throttling.

That bounds everything: **4 x 396.8 = 1,587 Gbps is the hardware maximum**. The
reference's 1,413 Gbps is 89% of it and this benchmark's 1,336 is 84%. The target
is therefore reachable on four cards -- 1,399 needs 350 per card against a 397
cap -- and the deficit is entirely in the two 12-queue cards, which deliver ~250
where ~353 is needed. Their neighbours are already past the requirement at 400.

Worth noting for anyone reading the numbers: at 2,100 readers the application
rate collapses to 267 Gbps while the NIC still carries 396.8. Whatever that is
-- ramp inside the window, or accounting that loses track at extreme thread
counts -- it means a single application figure at a high thread count should not
be trusted without the NIC cross-check beside it.

### Why it stops at ~1,336: it is per-core efficiency, and nothing else

Each card reaches ~400 Gbps and needs cores to get there. Measured, cores per
card against that card's delivered rate:

| Logical cores | Gbps |
|---:|---:|
| 40 | ~250 |
| 48 | ~319 |
| 56 | ~400 |
| 96 (card alone) | 397 |

That is ~7 Gbps per logical core up to a per-card ceiling near 400. Both card
classes reach ~397 Gbps when run alone with a whole node, so the 12-queue cards
are not weaker hardware -- earlier notes in this document guessed they were, and
that was wrong. They are simply core-starved when sharing a node.

192 cores x 7 Gbps = 1,344 Gbps, which is what we measure. The reference moved
1,413 Gbps through the same 192 cores, or 7.36 Gbps per core. **The entire
remaining 5% is per-core receive efficiency.** No amount of placement,
partitioning, queue count or concurrency changes that, and the sweeps confirm
it: every one of them is now flat or negative.

### Partitioning cores between the two cards on a node: +6%

The last real win. Each node carries one 24-queue card (~400 Gbps) and one
12-queue card (~240), and both processes were pinned to the whole node. Giving
each card its own slice of its node's cores took the scored figure from 1,257 to
1,335 Gbps, because the big card's 700 readers were crowding out the small
card's — and the small card is exactly where the deficit was.

The ratio matters and does not follow card capability:

| Cores (24-queue / 12-queue) | app Gbps | NIC Gbps |
|---|---:|---:|
| whole node shared | 1,257.1 | 1,285.7 |
| 48 / 48 | 1,287.2 | 1,278.2 |
| 52 / 44 | 1,323.7 | 1,310.3 |
| **56 / 40** | **1,335.2** | **1,332.2** |
| 60 / 36 | 1,304.1 | 1,331.5 |
| 66 / 30 | 1,235.6 | 1,250.3 |
| 72 / 24 | 1,129.0 | 1,164.7 |

A 5:3 core split would match the 400:240 throughput ratio, but the optimum is
7:5 — the small card needs proportionally *more* CPU per byte than the big one,
which is consistent with its 12 queues each carrying ~20 Gbps against the big
card's ~17.

`PARTITION=1 SPLIT_BIG=28` in `scale4.sh` reproduces it. Reader count was
retuned afterwards and 700 still wins on the NIC measure; at 500 readers the
application figure reads 1,347.6 against a NIC figure of 1,316.8, and an
application rate above the NIC rate carrying its framing is impossible, so that
one is noise rather than a record.

The two now corroborate each other in the right direction: NIC RX reads about
1.5% above the application rate, which is the TCP/IP/TLS framing it carries and
the application does not. Earlier revisions of the harness had them disagreeing
by 16% in the wrong direction; see the measurement notes below.

Scoring a fixed window is not a cosmetic change. A byte-budget run charges the
per-thread DNS, TCP and TLS ramp to the result, which is why the same
configuration read 1,045 Gbps at a 10 s window and 1,182 Gbps at 41 s.

### Implementation note on the scored window

Bytes are counted per completed request, and a request here is 512 MiB, so
counting whole requests inside a window edge would be hopeless: one request per
reader is 350 GiB across 700 readers. Instead the window is the difference of two
snapshots of a monotonically increasing counter. That measures the completion
rate over the interval, and because the in-flight volume is constant in steady
state, the overcount at the opening edge and the undercount at the closing edge
cancel rather than accumulate.

## Chasing the 1.4 Tbps ceiling — best instantaneous ~1,312 Gbps, target not met

A second pass aimed specifically at the reference's 1,399 Gbps count-and-discard
number. It moved the ceiling from 1,070 to **1,258-1,312 Gbps** (four concurrent
NIC-counter samples: 1,258.0 / 1,271.4 / 1,297.8 / 1,312.2; mean ~1,285), which
is **92% of target**. It did not get there, and the reason is measured rather
than guessed.

Per-card, sampled concurrently, the shape is stable: the two 24-queue cards sit
at 398-401 Gbps every time, and the two 12-queue cards range 226-273. The big
cards look pinned to a hardware ceiling; the variance is all in the small ones.

Artifacts, since the KvikIO edits live in a FetchContent tree that a clean
reconfigure deletes: `kvikio-local-changes.patch` (diffed against a pristine
26.08.00 clone, so it may carry some unrelated upstream drift) and
`kvikio-modified-files/`, which holds verbatim copies of the four changed files.

What moved it:

| Change | Effect |
|---|---|
| **Fixed a KvikIO retry bug** (below) | unblocked the geometry entirely |
| Full 113-object inventory instead of lineitem only | 1,021 -> 1,058 Gbps |
| Reference geometry: 512 MiB single-shot ranges, 700 readers/card | plateau ~1,180 |
| Longer measurement window (ramp dilution) | 1,045 at 10 s -> 1,182 at 41 s |

`--reader_threads=700` per card measured best (1,103 / 1,155 / **1,182** / 1,118
at 384 / 512 / 700 / 1000), independently landing on the same number the
reference chose.

### The retry bug

At 512 MiB ranges under four-card load, some transfers exceed
`KVIKIO_HTTP_TIMEOUT`. `CurlHandle::perform()` then retries
`curl_easy_perform()` — but nothing resets the write callback's
`CallbackContext`, whose `offset` is already advanced by the partial first
attempt. The replayed range appends past the end of the caller's buffer, the
callback returns `CURL_WRITEFUNC_ERROR`, and libcurl reports
`CURLE_WRITE_ERROR`, which is not retryable. So a transient timeout became a
fatal error, surfaced as the badly misleading *"maybe the server doesn't support
file ranges?"*.

Two of four processes died this way on every attempt at the reference geometry.
The fix adds a `set_on_retry` hook to `CurlHandle` that `perform()` invokes
before each retry, and `RemoteHandle::read` uses it to reset `offset` and
`overflow_error`. This is an upstream-worthy bug: it makes large-range reads
fail under exactly the load where retries matter.

A harness bug was masking it. The aggregator summed whatever the first
whitespace-delimited token was, so two dead cards still produced a
plausible-looking 591 Gbps total. It now requires a well-formed result line per
card and prints a `WARNING: only N of 4 cards reported` otherwise.

### What did not help, all measured

- **`numactl --membind`.** Added `cap_add: [SYS_NICE]` to
  `docker-compose.override.yml` so real memory binding works, and it changed
  nothing: 1,155.8 with `numactl` against 1,157.1 with `taskset`. First-touch
  under a pinned CPU set was already placing pages correctly.
- **`CURLOPT_BUFFERSIZE`** at this geometry: 1,174 / 1,172 / 1,173 Gbps at
  64 KiB / 128 KiB / default. Syscall count is not the constraint here.
- **`MULTI_POLL` at the reference's own architecture** (32 and 64 reactors,
  ~700 concurrent): 976 and 1,064 Gbps, both below `EASY_THREADPOOL`.
- **Dropping the 1,024 idle pool threads** when `task_size=0` never touches the
  pool: 1,173 against 1,182. Free either way.
- **Raising the socket receive buffer.** `__tcp_select_window` is ~1.8% of the
  profile, so `SO_RCVBUF` via `CURLOPT_SOCKOPTFUNCTION` looked worth trying. It
  is not: `net.core.rmem_max` is 208 KiB here, and setting `SO_RCVBUF`
  explicitly *disables* TCP autotuning and clamps to that cap. Autotuning
  already grows live S3 connections to 1.6-2.8 MB (`ss -tmi` reports `rb1663923`
  and `rb2819285`), so an explicit value would be an order of magnitude worse.
  Left alone.

- **Aligning each card's cores with where its interrupts land.** irqbalance
  spreads a card's receive queues across the whole node, so a contiguous
  partition leaves most of a card's softirq running on its node-mate's cores:
  only 3 of 12 of one card's interrupts fell inside its own slice. Moving the
  partition to the interrupts rather than repinning them (no host change) was
  tried three ways and lost every time: 1,143 Gbps assigning individual CPUs,
  1,250 assigning whole physical cores, and a third variant killed a card
  outright, against 1,326 for a plain contiguous split. Splitting hyperthread
  siblings between cards explains the first result -- CPU 0 went to one card and
  its sibling 96 to another -- but core-granular alignment still lost, so
  contiguous ranges are winning on something else, most likely L3 and CHA
  locality. Interrupt locality is not the lever here.

- **AES-128 instead of AES-256.** Not a lever, because S3 already negotiates
  `TLS_AES_128_GCM_SHA256`. An earlier estimate in this document assumed TLS 1.3
  had defaulted to AES-256 and put the win at 12-21 Gbps; the negotiated cipher
  says there is nothing to take.

- **Jumbo frames, GRO, LRO.** MTU is already 9001, GRO is on, LRO is
  `[fixed] off` on ENA.
- **Leaving cores free for softirq.** Restricting app threads to 80 of a node's
  96 logical CPUs gave 1,092 Gbps, and to physical cores only 1,046, against
  1,181 with the whole node. Softirq is not being starved: no RX drops, and no
  `bw_in_allowance_exceeded`.
- **Sub-16 KiB receive buffers**, testing the theory that 700 per-handle buffers
  per card are cache-cold where the reference's 32 reusable buffers stay hot:
  1,125 / 1,139 / 1,151 Gbps at 4 KiB / 8 KiB / 16 KiB. libcurl's default is the
  optimum in both directions.
- **kTLS RX — attempted, and it cannot engage through libcurl.** This was the
  most promising idea left, since the largest single profile entry is the
  kernel-to-user copy. `modprobe tls` was loaded and
  `KVIKIO_REMOTE_IO_KTLS=1` added, which sets `SSL_OP_ENABLE_KTLS` on the
  `SSL_CTX` through `CURLOPT_SSL_CTX_FUNCTION`. Transfers keep working and
  `/proc/net/tls_stat` stays at `TlsRxSw=0`, so OpenSSL silently declines.
  The reason is structural: OpenSSL only turns on kTLS when the BIO under the
  `SSL` object advertises kTLS capability on a real socket fd, and libcurl
  installs its own BIO for event handling. The reference could have used kTLS
  because it drives OpenSSL directly on a raw socket — and it still chose to
  disable it. The option is left in place, off by default, because it becomes
  useful the moment libcurl grows kTLS support.
- **AES-128 instead of TLS 1.3's default AES-256.** The reference negotiated
  `ECDHE-RSA-AES128-GCM-SHA256`; we get TLS 1.3, which defaults to AES-256-GCM.
  AES-128-GCM measures 16.4% faster per core here, but AES is only 7-12% of the
  profile, so this is worth an estimated 12-21 Gbps — real, and far short of the
  218 Gbps needed.

### Restoring the reference channel vector: 1,181 -> 1,271 Gbps

The single largest remaining factor was environmental, not code. The host had
drifted to `32,32,32,32` combined channels; the reference requires
`24,12,24,12`. Restoring it with `ethtool -L` took the ceiling from 1,181 to
**1,271 Gbps**, and made the reference's own preflight pass, fingerprint
included:

```
PASS: exact-host read-only preflight for mode=default-tls
Reference immutable fingerprint: 6dd6dd251bb17943c7bbb3c033eb0420d4b2d14fe033cdb09a7d5be0f7c9949b
```

Why the vector is asymmetric becomes obvious from the per-card rates, sampled
concurrently from NIC counters in steady state:

| NIC | Queues | Gbps |
|---|---:|---:|
| enp135s0 | 24 | 398.4 |
| enp153s0 | 12 | 243.4 |
| enp170s0 | 24 | 400.7 |
| enp187s0 | 12 | 228.9 |
| **total** | | **1,271.4** |

The asymmetric vector is optimal, and that has now been tested rather than
assumed. Three channel configurations, same partition and geometry:

| Channels | NIC RX Gbps |
|---|---:|
| **24,12,24,12** (reference) | **1,336** |
| 24,24,24,24 | 1,299 |
| 32,32,32,32 | 1,181 |

More queues is worse, consistently. Giving the 12-queue cards their neighbours'
24 queues does not make them faster: it spreads each card's traffic over more
queues and coalesces less per GRO context, and with only 40 cores to service them
that costs more than the extra parallelism returns. An earlier note in this
document explained the asymmetry as the cards being different sizes -- two
~400 Gbps and two ~235 Gbps. That was wrong: both classes reach ~397 Gbps alone,
so the vector is about queue-to-core ratio, not card capability. Revert with
`./set-channels.sh 32 32 32 32`.

**Two measurement bugs this exposed, both in the harness rather than the
benchmark.** They are worth recording because each produced a plausible number
rather than an obvious error.

1. *Summing windows that are not the same window.* With unequal cards and equal
   per-process byte budgets the processes stop at different times, so adding four
   rates each measured over its own window overstates the concurrent total. The
   first run after the channel change reported `app 1262` against
   `nicRX(win) 1058`: a 16% disagreement in the direction that should be
   impossible, since NIC RX carries framing the application does not.
2. *Dividing a warmup-inclusive byte delta by the scored window.* Once a timed
   window existed, the NIC cross-check spanned warmup plus scoring but was
   divided by the 60 s score, and cheerfully printed **2,018 Gbps** — physically
   impossible on four cards. The harness now samples NIC counters across the
   scored window only.

With both fixed the two independent measures agree to ~1.5% in the correct
direction, which is what makes the headline number believable. The general
lesson: an aggregate assembled from per-process arithmetic needs an independent
check that is sensitive to the same interval, or it will report the run you hoped
for.

### The host had drifted from the environment that produced 1,399 Gbps

The reference's own read-only preflight refuses to run here:

```
$ ./scripts/preflight.sh --mode default-tls
ERROR: enp135s0 channel count differs: 32
```

`config/exact-host.env` pins `EXPECTED_COMBINED_CHANNELS=24,12,24,12`, and every
card now reports 32. Card 1 and card 3 were 12-queue cards during the campaign
and are 32-queue now. So the 1,399 Gbps figure was measured on a NIC
configuration that no longer exists on this instance, and the bundle that
produced it could not be run today without changing the host — which its own
documentation forbids: *"A failed preflight means the measured environment
drifted; investigate separately."*

This is a plausible contributor to the residual gap rather than a curiosity.
Spreading the same traffic over 32 queues instead of 12 or 24 means fewer packets
coalesced per GRO context and more per-packet work, which is consistent with the
26% softirq share and with `clear_page_erms` plus `memset_orig` costing ~3.8%
between them. Testing it needs `ethtool -L`, a shared-host network change, so it
is left as a decision for the host owner rather than something done unilaterally.

Two facts bound the interpretation: `bw_in_allowance_exceeded` is 0 on all four
cards, so EC2 is not throttling the instance, and the page pool reports
`rx_pp_alloc_slow=0`, so page allocation is not falling back to the slow path.

### Why it stops at ~1,180

CPU is pinned at 100% with 0% idle: **25% user, 48% system, 26% softirq**. One
card on its own reaches 375 Gbps, so 4 x 375 = 1,502 Gbps of card capability is
present and unused; the loss is purely contention for cores. We move ~148 GB/s
where the reference moved ~175 GB/s, so it spends about 18% less CPU per byte.

The difference is the receive architecture. This benchmark drives 700 *blocking*
reader threads per card, 2,800 in total, each parked in `recv()`; the reference
drives 32 epoll threads per card, 128 in total, each servicing ~22 connections
without blocking. That is what the 48% system time is. KvikIO's own answer to
this is the `MULTI_POLL` backend, and today it measures slower.

So the remaining ~15% is not reachable by configuration or by a local patch. It
needs an efficient event-driven receive path inside KvikIO — upstream work,
tracked as rapidsai/kvikio#987.

## Delivering data a GPU can consume: 677 -> 874 Gbps

The count-and-discard ceiling is a transport diagnostic. For a real pipeline the
bytes have to land somewhere a consumer can read, and that had been costing half
the ceiling: ~677 Gbps delivered against ~1,337 discarded. Three geometries all
sat at 669-683 Gbps, which is the signature of a limit that is not CPU-bound and
not concurrency-bound.

**The cause is read-for-ownership on the destination.** libcurl delivers one
buffer at a time, 16 KiB by default, so every `memcpy` in the write callback is
far below glibc's non-temporal threshold and uses ordinary stores. An ordinary
store to a line that is not in cache fetches it first, and the destination set
here is 32 GiB per process — nothing is ever resident. Each delivered byte
therefore costs two DRAM accesses instead of one, and the receive path becomes
memory-bandwidth-bound.

**Streaming stores fix it.** `KVIKIO_REMOTE_IO_NT_COPY=1` switches the callback
to `_mm256_stream_si256` with an `sfence` before the buffer is handed on, which
skips the fetch:

| Destination | Ordinary stores | Streaming stores |
|---|---:|---:|
| Pinned host, req=256M rdr=128 | 669.3 | **873.6** |
| Pinned host, req=64M rdr=512 | 682.6 | 862.8 |
| Paged host + huge pages | ~670 | 865.1 |

**+29%**, verified in the object file as `vmovntdq %ymm0` plus `sfence` rather
than assumed from the flag. Reproduced three times.

Pinned and paged now land within noise of each other, so the choice between them
is about what the consumer needs — pinned is required for an async H2D copy —
rather than about receive throughput.

### Write-combined memory: rejected, with evidence

`cudaHostAllocWriteCombined` attacks the same read-for-ownership cost in hardware
and looked like the obvious partner to the above. It is worse: two of four cards
died with `transfer closed with N bytes remaining to read`, and combining it with
streaming stores gave 864.3 Gbps, indistinguishable from streaming stores alone.
Streaming stores already avoid the fetch, so WC has nothing left to contribute
while adding a destination the CPU cannot read cheaply.

### Where streaming stores are the wrong choice

Device-memory destinations do not benefit: KvikIO stages those through a small
pinned bounce buffer that is immediately re-read for the host-to-device copy, and
a bounce buffer wants to stay in cache. Streaming stores are only right when the
destination is much larger than last-level cache and is not read again by the
CPU. That is exactly a buffer about to be DMA'd to a GPU, and exactly not a
staging buffer.

## Where the time goes

`perf record` across a card's cores, at 790 Gbps into paged host memory:

| Component | Share |
|---|---:|
| kernel | 62.6% |
| libc (`memcpy` into the destination) | 17.3% |
| libcrypto (AES-GCM) | 11.6% |
| libcurl | 4.6% |
| libssl | 3.0% |

CPU is saturated at 99% on both NUMA nodes, so this is a CPU-bound problem, not
a network one. The single largest kernel symbol is `rep_movs_alternative` under
`tcp_recvmsg -> __sys_recvfrom`: the copy of every byte from socket buffer to
user space. That is proportional to bytes and irreducible without kernel TLS, and
the reference pays it too.

Removing the destination write (discard) drops libc from 17.3% to 6.0% and buys
31% more throughput, which is what identified the destination as the thing worth
optimising.

## Changes

### KvikIO — `cpp/src/shim/libcurl.cpp`, `CurlHandle` constructor

Three options KvikIO never set. All read once into function-local statics,
because a handle is constructed per sub-range transfer and `getenv` would
otherwise sit on the hot path.

- `CURLOPT_INTERFACE` via `KVIKIO_REMOTE_IO_INTERFACE`. The `if!<name>` form
  makes libcurl use `SO_BINDTODEVICE`, which is exactly what the reference
  receiver does. On a host with four cards on one subnet, this is the only way to
  choose an egress NIC from inside the process; without it the kernel picks by
  route metric and every connection leaves through one card.
- `CURLOPT_DNS_SHUFFLE_ADDRESSES` via `KVIKIO_REMOTE_IO_DNS_SHUFFLE`. S3
  publishes 27 front-end addresses here, all present in `/etc/hosts`, but libcurl
  connects to the first that answers, so a whole process piles onto one endpoint.
- `CURLOPT_BUFFERSIZE` via `KVIKIO_REMOTE_IO_BUFFER_SIZE`, default 0 meaning
  leave libcurl alone.

Together the first two replace the `libbindsrc.so` `LD_PRELOAD` shim that was
needed to get per-card egress at all. Native binding measured 818 Gbps against
the shim's 862, so the shim is still marginally faster, but the native path needs
no interposition.

**The buffer size is a documented negative result.** Raising it looks obviously
right — 16 KiB at 24 GB/s per card is ~1.5M `recv()` calls per second — and it
measures worse at scale:

| `CURLOPT_BUFFERSIZE` | 1 card | 4 cards |
|---|---:|---:|
| 16 KiB (libcurl default) | 195 Gbps | **862 Gbps** |
| 64 KiB | 203 | — |
| 256 KiB (the reference's value) | 199 | 745 |
| 1 MiB | 202 | 727 |

libcurl allocates this buffer *per easy handle*, and there is one handle per
in-flight sub-range. At 4096-way concurrency that is 64 MiB at 16 KiB, which fits
in the 640 MiB L3, versus 4 GiB at 1 MiB, which does not. The reference's 256 KiB
buffer works because it has 32 of them per card, one per reactor, not thousands.
The same number in a different architecture is a 14% regression. The default is
left alone and the comment records why.

### KvikIO — `cpp/src/detail/remote_callback.cpp`

`KVIKIO_REMOTE_IO_DISCARD=1` makes `callback_host_memory` account for every byte
without copying it. Byte accounting is unchanged, so the throughput figure is an
honest count of bytes received, but the destination holds garbage. Benchmarking
only; it exists to separate transport cost from destination cost, and that is how
the 31% figure above was obtained.

### Velox — `KvikioReadBenchmark.cpp` (+203 / -23)

- **`--pinned_memory`** allocates host destinations with `cudaHostAlloc`. Pinned
  is what a GPU consumer needs for an async H2D copy. It measured *faster* than
  paged (863 vs 783), so on this host there is no throughput reason to prefer
  paged.
- **`--huge_pages`** (default on) backs paged destinations with 2 MiB transparent
  huge pages: `mmap`, align up into the mapping, `madvise(MADV_HUGEPAGE)`, with a
  silent fall back to `malloc` if the advice is refused. THP here is in `madvise`
  mode, so `malloc`'d buffers got none — 32 GiB of destinations meant 8.4 million
  base pages for the write to walk. Verified engaging via
  `AnonHugePages: 134217728 kB` (128 GiB across four processes) versus 0 with the
  flag off. Worth roughly +5% on throughput, which is close to the run-to-run
  noise, but it also cuts buffer setup sharply: whole-wall-clock NIC RX rises from
  ~320 to ~475 Gbps because faulting in 32 GiB costs 16K huge-page faults instead
  of 8.4M base-page faults.
- **`--buffer_slots` and a `BufferRing`** decouple the destination footprint from
  the reader count. Readers acquire a slot, fill it, release it; a real consumer
  would sit between release and the next acquire, so this models one that
  recycles immediately. A `folly::makeGuard` returns the slot even if the read
  throws, or the remaining readers deadlock on an empty ring. Bounding the ring
  below the reader count *costs* throughput (652 Gbps at 24 slots against 821 at
  one per reader) because it caps outstanding requests, so the default remains one
  slot per reader — but the mechanism is what a consumer-bounded pipeline needs.
- **`--cuda_device` plus `cudaSetDevice`** fixes a real bug. The benchmark never
  selected a device, so on this 8-GPU host all four processes allocated on device
  0 and asked for 128 GiB of a 97 GB card: two of four died with
  `cudaErrorMemoryAllocation` and the survivors reported 284 Gbps. One GPU per
  process gives 723 Gbps with no failures.
- `BufferKind` replaces the internal `bool device` so paged, pinned and device
  are one concept with three cases.

## What did not work

- **`MULTI_POLL`** is the architecturally right answer to the thread count — 3072
  pool threads against the reference's 128 — and it is not broken: it scales with
  reactor count (887 / 970 / 1025 Gbps at 32 / 48 / 96 reactors) and reuses
  connections as well as anything. It still trails `EASY_THREADPOOL` at 1070. It
  is tracked upstream as a new backend in rapidsai/kvikio#987.
- **Connection reuse was never the problem.** 12,288 range GETs cost 320-360 TCP
  opens, 0.03 per GET, on every backend. TLS handshakes are not a cost here.
- **Smaller sub-ranges to shrink the footprint** backfire badly. Footprint is
  concurrency x task size, so holding concurrency at 768 while shrinking the task
  gave 600 / 446 / 347 / 186 Gbps at 4 MiB / 1 MiB / 512 KiB / 256 KiB.
  Per-request overhead dominates long before cache residency pays.
- **Jumbo frames and AES-128** are not levers. MTU is already 9001 on all four
  cards, and AES-128-GCM is only 16% faster than AES-256-GCM here (13.9 vs
  11.9 GB/s per core), which against 12.5% of CPU is worth about 2%.

## Best known configuration

```bash
# One process per card. Pinned host destinations.
KVIKIO_NTHREADS=1024 \
KVIKIO_REMOTE_IO_INTERFACE='if!enp135s0' \
KVIKIO_REMOTE_IO_DNS_SHUFFLE=1 \
KVIKIO_HTTP_MAX_ATTEMPTS=5 KVIKIO_HTTP_TIMEOUT=30 \
  taskset -c 0-47,96-143 velox_cudf_kvikio_read_benchmark \
    --paths=lineitem.enp135s0.manifest --mode=warm \
    --request_bytes=$((256*1024*1024)) --kvikio_task_size=$((16*1024*1024)) \
    --reader_threads=128 --kvikio_nthreads=1024 \
    --pinned_memory=true
```

`./scale4.sh 1024 268435456 16777216 128 300` with `PINNED=1` runs all four.

Pool width is the dominant knob and peaks near 1024; 1400 and 2048 both regress.
Sub-range splitting is what makes any of it work — `--kvikio_task_size=0` calls
`read()` on the reader thread and never touches the pool, which caps a card at
about 3 Gbps per reader regardless of request size.

## Caveats

- Every KvikIO edit lives in the FetchContent tree at
  `_build/release/_deps/kvikio-src/`, which is not under version control. A
  reconfigure from clean discards them. To keep them, they need to become a patch
  or a fork pointed at by `CPM`/`FetchContent`.
- Numbers are warm mode over 10-20 second windows with no warmup phase, against
  the reference's 30 s warmup and 60 s scored window. Run-to-run variance on the
  4-card configs is roughly +/-5%, so single-run differences under that should not
  be read as signal.
- The four processes read disjoint object slices, but warm mode re-reads within a
  slice, so the S3 front-end cache is in play to an unmeasured degree.
