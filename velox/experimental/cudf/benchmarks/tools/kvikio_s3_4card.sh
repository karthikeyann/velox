#!/usr/bin/env bash
# Runs one configuration across all four NICs and prints the aggregate
# application throughput plus the NIC RX cross-check.
#
#   ./scale4.sh <pool> <request_bytes> <task_bytes> <readers> <gib_per_proc> [backend] [reactors] [maxconc]

set -Eeuo pipefail

POOL=${1:?pool}; REQ=${2:?request bytes}; TASK=${3:?task bytes}
READERS=${4:?readers}; GIB=${5:?GiB per process}
BACKEND=${6:-EASY_THREADPOOL}; REACTORS=${7:-1}; MAXCONC=${8:-256}

BIN=${BIN:-/velox/_build/release/velox/experimental/cudf/benchmarks/velox_cudf_kvikio_read_benchmark}
KD=${KD:-/velox/_build/kvikio}          # manifests + shim, as seen inside the container
S3IPS=${S3IPS:-$(grep -i "s3\.${AWS_REGION:-us-east-2}\.amazonaws\.com" /etc/hosts | awk '{print $1}' | sort -u | paste -sd,)}
G=$((1024 * 1024 * 1024))

# nic:source_ip:cpu_list:cuda_device. Each process gets its own GPU so that
# four device-memory processes do not collectively overcommit card 0.
ALL_NICS=(
  "enp135s0:172.31.241.234:0-47,96-143:0:0"
  "enp153s0:172.31.241.193:0-47,96-143:1:0"
  "enp170s0:172.31.241.146:48-95,144-191:4:1"
  "enp187s0:172.31.241.204:48-95,144-191:5:1"
)
# ONLY=enp170s0,enp187s0 restricts the run to a subset of cards, so one node can
# be measured without the other competing for cores or S3 capacity.
NICS=()
for spec in "${ALL_NICS[@]}"; do
  if [[ -z ${ONLY:-} || ,${ONLY}, == *,${spec%%:*},* ]]; then NICS+=("$spec"); fi
done
[[ ${#NICS[@]} -gt 0 ]] || { echo "ONLY matched no cards" >&2; exit 1; }

eval "$(AWS_PROFILE=${AWS_PROFILE_NAME:-cloud-benchmark} aws configure export-credentials --format env)"

# KvikIO receive-buffer size, and the two native egress controls added to the
# CurlHandle constructor. NATIVE_BIND=on uses KvikIO's own CURLOPT_INTERFACE and
# CURLOPT_DNS_SHUFFLE_ADDRESSES instead of the libbindsrc.so shim, so the two
# mechanisms can be compared directly.
BIND=${BIND:-numactl}
CPUS0=${CPUS0:-}
PARTITION=${PARTITION:-0}
RDR_BIG=${RDR_BIG:-}
RDR_SMALL=${RDR_SMALL:-}
CPUS1=${CPUS1:-}
BUFFER_SIZE=${BUFFER_SIZE:-0}
NATIVE_BIND=${NATIVE_BIND:-off}

# DISCARD=1 accepts payload without landing it in the destination buffer, which
# isolates transport cost from the cost of the destination write. Byte accounting
# is unaffected; the buffer contents are not valid.
DISCARD=${DISCARD:-0}
KTLS=${KTLS:-0}
WARMUP=${WARMUP:-0}
SCORE=${SCORE:-0}

# Destination ring depth (0 = one slot per reader) and destination kind. Device
# reads have no GDS path from a remote source, so they stage through KvikIO's
# pinned bounce buffer and then copy host-to-device.
SLOTS=${SLOTS:-0}
PINNED=${PINNED:-0}
DEVICE_MEMORY=${DEVICE_MEMORY:-false}
HUGE=${HUGE:-true}

# MANIFEST selects the object set: "lineitem" is 15 lineitem objects per card,
# "full" is 22 objects per card spanning five tables, matching the reference's
# 113-object inventory. USHARE=1 gives every reader one shared destination.
MANIFEST=${MANIFEST:-lineitem}
USHARE=${USHARE:-0}

declare -A rx0
for spec in "${NICS[@]}"; do
  nic=${spec%%:*}
  rx0[$nic]=$(<"/sys/class/net/$nic/statistics/rx_bytes")
done

tmp=$(mktemp -d)
declare -A rxw0
sample_window() {
  # Runs concurrently with the benchmark: wait out warmup, then measure the NIC
  # counters over the scored window only.
  sleep $((WARMUP + 3))
  for spec in "${NICS[@]}"; do
    nic=${spec%%:*}; rxw0[$nic]=$(<"/sys/class/net/$nic/statistics/rx_bytes")
  done
  local s=$((SCORE > 6 ? SCORE - 6 : SCORE))
  sleep "$s"
  local tot=0
  for spec in "${NICS[@]}"; do
    nic=${spec%%:*}
    tot=$((tot + $(<"/sys/class/net/$nic/statistics/rx_bytes") - rxw0[$nic]))
  done
  echo "scale=2; $tot * 8 / $s / 1000000000" | bc > "$tmp/nicwin"
}
t0=$(date +%s.%N)
if [[ $SCORE -gt 0 ]]; then sample_window & fi
for spec in "${NICS[@]}"; do
  IFS=: read -r nic ip cpus gpu node <<<"$spec"
  if [[ $node == 0 && -n $CPUS0 ]]; then cpus=$CPUS0; fi
  if [[ $node == 1 && -n $CPUS1 ]]; then cpus=$CPUS1; fi
  if [[ $PARTITION == 1 ]]; then
    # SPLIT_BIG cores are taken from each of the node's two CPU ranges and given
    # to the 24-queue card; the 12-queue card gets the remainder. Each range is
    # 48 wide, so SPLIT_BIG=30 is a 60/36 logical-core split.
    B=${SPLIT_BIG:-30}
    case $nic in
      enp135s0) cpus="0-$((B-1)),96-$((96+B-1))" ;;
      enp153s0) cpus="$B-47,$((96+B))-143" ;;
      enp170s0) cpus="48-$((48+B-1)),144-$((144+B-1))" ;;
      enp187s0) cpus="$((48+B))-95,$((144+B))-191" ;;
    esac
  fi
  readers=$READERS
  if [[ -n $RDR_BIG || -n $RDR_SMALL ]]; then
    q=$(ethtool -l "$nic" 2>/dev/null | awk '/Current hardware settings/{f=1} f&&/Combined/{print $2}')
    if [[ $q -ge 24 ]]; then readers=${RDR_BIG:-$READERS}; else readers=${RDR_SMALL:-$READERS}; fi
  fi
  if [[ $NATIVE_BIND == on ]]; then
    egress=(-e KVIKIO_REMOTE_IO_INTERFACE="if!$nic" -e KVIKIO_REMOTE_IO_DNS_SHUFFLE=1)
  else
    egress=(-e FANOUT_DST_IPS="$S3IPS" -e BIND_SRC_IP="$ip"
            -e LD_PRELOAD="$KD/libbindsrc.so")
  fi
  docker exec \
    -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN \
    -e AWS_DEFAULT_REGION=us-east-2 \
    -e KVIKIO_NTHREADS="$POOL" \
    -e KVIKIO_HTTP_MAX_ATTEMPTS=5 -e KVIKIO_HTTP_TIMEOUT=30 \
    -e KVIKIO_REMOTE_IO_BACKEND="$BACKEND" \
    -e KVIKIO_REMOTE_IO_NUM_REACTORS="$REACTORS" \
    -e KVIKIO_REMOTE_IO_MAX_CONCURRENT_REQUESTS="$MAXCONC" \
    -e KVIKIO_REMOTE_IO_BUFFER_SIZE="$BUFFER_SIZE" \
    -e KVIKIO_REMOTE_IO_DISCARD="$DISCARD" \
    -e KVIKIO_REMOTE_IO_KTLS="$KTLS" \
    "${egress[@]}" \
    ${CONTAINER:-velox-adapters-cuda-ubuntu} \
    $([[ $BIND == numactl ]] && echo "numactl --cpunodebind=$node --membind=$node" || echo "taskset -c $cpus") "$BIN" \
      --paths="$KD/$MANIFEST.$nic.manifest" --mode=warm \
      --measurement_bytes=$((GIB * G)) --request_bytes="$REQ" \
      --reader_threads="$readers" --kvikio_task_size="$TASK" \
      --buffer_slots="$SLOTS" \
      --pinned_memory="$([[ $PINNED == 1 ]] && echo true || echo false)" \
      --device_memory="$DEVICE_MEMORY" --cuda_device="$gpu" \
      --huge_pages="$HUGE" \
      --unsafe_shared_buffer="$([[ $USHARE == 1 ]] && echo true || echo false)" \
      --warmup_seconds="$WARMUP" --score_seconds="$SCORE" \
      --kvikio_nthreads="$POOL" >"$tmp/$nic" 2>"$tmp/$nic.err" &
done
wait
t1=$(date +%s.%N)

# Aggregate the application rate over each process's own measured window, which
# is the number the benchmark itself stands behind. The NIC figure below spans
# the whole wall clock including opens, so it reads lower by construction.
mbps=0; slowest=0; ok=0
for spec in "${NICS[@]}"; do
  nic=${spec%%:*}
  # Accept only a well-formed result line. Anything else is a failure worth
  # printing verbatim, because silently folding it into the total produced a
  # believable-looking aggregate from a run where cards had died.
  line=$(grep -m1 -E '^[0-9]+(\.[0-9]+)? MB/s ' "$tmp/$nic" 2>/dev/null || true)
  if [[ -z $line ]]; then
    echo "  FAILED on $nic: $(tail -2 "$tmp/$nic.err" 2>/dev/null | tr '\n' ' ')"
    echo "                 stdout: $(tail -1 "$tmp/$nic" 2>/dev/null)"
    continue
  fi
  rate=${line%% *}
  el=$(sed -n 's/.*elapsed_s=\([0-9.]*\).*/\1/p' <<<"$line")
  mbps=$(echo "$mbps + $rate" | bc)
  slowest=$(echo "if ($el > $slowest) $el else $slowest" | bc)
  ok=$((ok + 1))
done
[[ $ok -eq ${#NICS[@]} ]] || echo "  WARNING: only $ok of ${#NICS[@]} cards reported; the total below is not a $((${#NICS[@]}))-card number"

rxtot=0
for spec in "${NICS[@]}"; do
  nic=${spec%%:*}
  now=$(<"/sys/class/net/$nic/statistics/rx_bytes")
  rxtot=$((rxtot + now - rx0[$nic]))
done

wall=$(echo "$t1 - $t0" | bc)
# Report the NIC figure over the measured window as well as over the wall clock.
# Over the wall clock it is diluted by target opening, buffer first-touch and
# thread setup, which made it read far below the application rate even though it
# counts strictly more bytes (it includes TCP/IP/TLS framing). Over the window it
# is directly comparable, and should sit a few percent ABOVE the application rate.
nic_wall=$(echo "scale=2; $rxtot * 8 / $wall / 1000000000" | bc)
if [[ $SCORE -gt 0 && -s $tmp/nicwin ]]; then
  nic_win=$(<"$tmp/nicwin")   # sampled across the scored window only
else
  nic_win=$(echo "scale=2; if ($slowest > 0) $rxtot * 8 / $slowest / 1000000000 else 0" | bc)
fi
dest=$([[ $DEVICE_MEMORY == true ]] && echo device || { [[ $PINNED == 1 ]] && echo pinned || echo paged; })
printf "req=%-5s task=%-6s rdr=%-4s pool=%-4s slots=%-4s dest=%-6s man=%-8s bind=%-8s ushare=%s disc=%s | app %7.1f Gbps | nicRX(win) %7.1f | nicRX(wall) %6.1f | win %.1fs wall %.1fs\n" \
  "$((REQ / 1024 / 1024))M" "$((TASK / 1024))K" "$READERS" "$POOL" \
  "$([[ $SLOTS == 0 ]] && echo "=rdr" || echo "$SLOTS")" "$dest" "$MANIFEST" "$BIND" "$USHARE" "$DISCARD" \
  "$(echo "scale=2; $mbps * 8 / 1000" | bc)" \
  "$nic_win" "$nic_wall" "$slowest" "$wall"

rm -rf "$tmp"
