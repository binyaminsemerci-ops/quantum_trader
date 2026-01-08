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

tick_before=$(redis-cli XLEN "$STREAM_TICK" 2>/dev/null || echo 0)
kline_before=$(redis-cli XLEN "$STREAM_KLINES" 2>/dev/null || echo 0)
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

# Smoke alarm: check kline freshness (detect stuck streams)
check "kline freshness (< 120s old)" bash -c "
  health_line=\$(journalctl -u $SERVICE --since '5 minutes ago' --no-pager | grep HEALTH | tail -1 || echo '')
  if [ -z \"\$health_line\" ]; then
    echo 'No HEALTH line found'
    exit 1
  fi
  
  # Extract last_kline age (e.g. '36.5s ago' or '1.2m ago')
  last_kline_age=\$(echo \"\$health_line\" | grep -oP 'last_kline=\K[0-9.]+[sm]' || echo '999s')
  
  # Convert to seconds
  if [[ \$last_kline_age == *m ]]; then
    seconds=\$(echo \"\$last_kline_age\" | sed 's/m//' | awk '{print int(\$1 * 60)}')
  else
    seconds=\$(echo \"\$last_kline_age\" | sed 's/s//' | awk '{print int(\$1)}')
  fi
  
  echo \"Last kline age: \${seconds}s (threshold: 120s)\"
  
  if [ \$seconds -lt 120 ]; then
    exit 0
  else
    echo \"⚠️  Kline stream is stuck (last kline \${seconds}s ago)\"
    exit 1
  fi
"

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
