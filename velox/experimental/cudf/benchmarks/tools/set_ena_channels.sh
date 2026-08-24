#!/usr/bin/env bash
# Sets the ENA combined-channel vector. The reference campaign ran with
# 24,12,24,12 and its preflight fails closed on anything else; this host had
# drifted to 32,32,32,32.
#
# ethtool -L reinitialises the device, which briefly interrupts the interface,
# so this is meant to be launched detached (setsid) and its log inspected after.
#
#   ./set-channels.sh 24 12 24 12    # reference vector
#   ./set-channels.sh 32 32 32 32    # revert to as-found
set -uo pipefail

LOG=/home/ubuntu/knataraj/velox/_build/kvikio/set-channels.log
CARDS=(enp135s0 enp153s0 enp170s0 enp187s0)
WANT=("${1:?card0}" "${2:?card1}" "${3:?card2}" "${4:?card3}")

exec >>"$LOG" 2>&1
echo "=== $(date -Is) requesting ${WANT[*]} ==="

for i in "${!CARDS[@]}"; do
  nic=${CARDS[$i]}
  now=$(ethtool -l "$nic" 2>/dev/null | awk '/Current hardware settings/{f=1} f&&/Combined/{print $2}')
  echo "$nic: current=$now want=${WANT[$i]}"
  if [[ $now == "${WANT[$i]}" ]]; then
    echo "$nic: already correct, skipping"
    continue
  fi
  # The primary interface is reconfigured last, so a blip on it cannot leave the
  # other three half-done.
  sudo ethtool -L "$nic" combined "${WANT[$i]}" && echo "$nic: set ok" || echo "$nic: SET FAILED"
  sleep 2
done

echo "--- resulting vector ---"
for nic in "${CARDS[@]}"; do
  printf '%s=%s ' "$nic" \
    "$(ethtool -l "$nic" 2>/dev/null | awk '/Current hardware settings/{f=1} f&&/Combined/{print $2}')"
done
echo
echo "--- link state ---"
for nic in "${CARDS[@]}"; do
  printf '%s:%s ' "$nic" "$(cat /sys/class/net/"$nic"/operstate)"
done
echo
echo "=== $(date -Is) done ==="
