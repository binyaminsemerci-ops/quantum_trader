#!/usr/bin/env bash
set -euo pipefail
LOG="/var/log/quantum-healthcheck.log"
LOCK="/tmp/quantum-healthcheck.alert.lock"
COOLDOWN=600

mkdir -p /var/log

if /opt/quantum/ops/healthcheck.sh >>"$LOG" 2>&1; then
  echo "$(TZ=Europe/Oslo date -Is) OK" >>"$LOG"
  exit 0
fi

echo "$(TZ=Europe/Oslo date -Is) FAIL" >>"$LOG"

now=$(date +%s)
last=0
[[ -f "$LOCK" ]] && last=$(cat "$LOCK" 2>/dev/null || echo 0)

if (( now - last >= COOLDOWN )); then
  echo "$now" >"$LOCK"
  /opt/quantum/ops/healthcheck.sh >>"$LOG" 2>&1 || true
fi

exit 1
