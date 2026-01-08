#!/usr/bin/env bash
set -euo pipefail

SERVICE="quantum-market-publisher.service"
STREAM_TICK="quantum:stream:market.tick"
STREAM_KLINES="quantum:stream:market.klines"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

fail=0

check() {
  local name="$1"; shift
  log "TEST: $name"
  if "$@"; then
    log "✅ PASS: $name"
  else
    log "❌ FAIL: $name"
    fail=1
  fi
  echo
}

check "service is active" systemctl is-active --quiet "$SERVICE"

check "redis ping" bash -c 'redis-cli PING | grep -q PONG'

tick_before=$(redis-cli XLEN "$STREAM_TICK" || echo 0)
kline_before=$(redis-cli XLEN "$STREAM_KLINES" || echo 0)
log "Before: tick=$tick_before klines=$kline_before"
echo

check "wait for klines to increase within 90s" bash -c "
  before=\$(redis-cli XLEN $STREAM_KLINES)
  for i in {1..9}; do
    sleep 10
    now=\$(redis-cli XLEN $STREAM_KLINES)
    if [ \$now -gt \$before ]; then exit 0; fi
  done
  exit 1
"

log "Latest kline (raw):"
redis-cli XREVRANGE "$STREAM_KLINES" + - COUNT 1 | head -40
echo

log "Latest HEALTH line:"
journalctl -u "$SERVICE" --since "5 minutes ago" --no-pager | grep HEALTH | tail -1 || true
echo

if [ "$fail" -eq 0 ]; then
  log "🎉 ALL TESTS PASSED"
  exit 0
else
  log "❌ SOME TESTS FAILED"
  exit 1
fi
