#!/usr/bin/env bash
set -euo pipefail

SERVICE="quantum-ai-engine.service"
STREAM_INTENT="quantum:stream:trade.intent"
TIMEOUT=60

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

# Pre-checks
check "ai-engine service is active" systemctl is-active --quiet "$SERVICE"

check "redis ping" bash -c 'redis-cli PING | grep -q PONG'

# Check governor kill switch (must be 0 for test)
kill_switch=$(redis-cli GET quantum:kill 2>/dev/null || echo "1")
if [ "$kill_switch" != "0" ]; then
  log "❌ FAIL: Governor kill switch is ON (quantum:kill=$kill_switch)"
  log "   Run: redis-cli SET quantum:kill 0"
  exit 1
fi
log "Governor kill switch: OFF (quantum:kill=0)"
echo

# Main test: wait for trade.intent to increase
intent_before=$(redis-cli XLEN "$STREAM_INTENT" 2>/dev/null || echo 0)
log "Before: trade.intent=$intent_before"
echo

check "wait for trade.intent to increase within ${TIMEOUT}s" bash -c "
  before=\$(redis-cli XLEN $STREAM_INTENT 2>/dev/null || echo 0)
  end_time=\$((SECONDS + $TIMEOUT))
  
  while [ \$SECONDS -lt \$end_time ]; do
    sleep 5
    now=\$(redis-cli XLEN $STREAM_INTENT 2>/dev/null || echo 0)
    if [ \$now -gt \$before ]; then
      echo \"Stream increased: \$before -> \$now (+\$((now - before)) events)\"
      exit 0
    fi
  done
  
  echo \"Stream did not increase within ${TIMEOUT}s\"
  exit 1
"

# Verify last event is NOT from manual/ops test
log "Last trade.intent event:"
last_event=$(redis-cli XREVRANGE "$STREAM_INTENT" + - COUNT 1)
echo "$last_event"
echo

check "verify event source is NOT manual/ops test" bash -c "
  source=\$(redis-cli XREVRANGE $STREAM_INTENT + - COUNT 1 | grep -oP 'source \K[^ ]+' || echo 'unknown')
  confidence=\$(redis-cli XREVRANGE $STREAM_INTENT + - COUNT 1 | grep -oP 'confidence \K[^ ]+' || echo '0')
  
  echo \"Event source: \$source, confidence: \$confidence\"
  
  if [[ \$source == *manual* ]] || [[ \$source == *ops-test* ]] || [[ \$source == *inject* ]]; then
    echo \"⚠️  Event source is manual/test (\$source)\"
    exit 1
  fi
  
  exit 0
"

log "Latest AI Engine HEALTH:"
journalctl -u "$SERVICE" --since "5 minutes ago" --no-pager | grep -E "HEALTH|signals_generated|drift_passed" | tail -5 || true
echo

if [ "$fail" -eq 0 ]; then
  log "🎉 ALL TESTS PASSED - AI Engine is publishing real signals"
  exit 0
else
  log "❌ SOME TESTS FAILED"
  exit 1
fi
