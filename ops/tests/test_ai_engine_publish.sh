#!/usr/bin/env bash
set -euo pipefail

# Set PATH for non-interactive execution (systemd oneshot)
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

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

# Verify last event is real signal (fail-closed parsing)
log "Last trade.intent event (raw):"
last_raw=$(redis-cli XREVRANGE "$STREAM_INTENT" + - COUNT 1)
echo "$last_raw"
echo

log "Parsing source and confidence..."

# Extract source field - it's on the line AFTER "source" line
src=$(echo "$last_raw" | grep -A1 "^source$" | tail -1 | tr -d '[:space:]')

# Extract payload field - it's on the line AFTER "payload" line
payload=$(echo "$last_raw" | grep -A1 "^payload$" | tail -1)

# FAIL if source missing (fail-closed)
if [[ -z "$src" ]]; then
  log "❌ FAIL: source field missing in stream entry"
  log "❌ SOME TESTS FAILED"
  exit 1
fi

log "Source: $src"

# FAIL if payload missing (fail-closed)
if [[ -z "$payload" ]]; then
  log "❌ FAIL: payload field missing in stream entry"
  log "❌ SOME TESTS FAILED"
  exit 1
fi

# FAIL if source is manual/ops-test (block known non-real sources)
if echo "$src" | grep -Eq '^(manual-inject|ops-test|inject)$'; then
  log "❌ FAIL: non-real source detected: $src"
  log "❌ SOME TESTS FAILED"
  exit 1
fi

# Parse confidence from JSON payload (fail-closed)
conf=$(echo "$payload" | python3 -c 'import json,sys; obj=json.loads(sys.stdin.read()); print(obj.get("confidence", ""))')

# FAIL if confidence parsing failed (fail-closed)
if [[ -z "$conf" ]]; then
  log "❌ FAIL: could not parse confidence from payload"
  log "❌ SOME TESTS FAILED"
  exit 1
fi

log "Confidence: $conf"

# Validate confidence > 0
if ! python3 -c "conf=float('$conf'); assert conf > 0.0, 'confidence must be > 0'; print('✅ Confidence OK:', conf)"; then
  log "❌ FAIL: confidence must be > 0, got: $conf"
  log "❌ SOME TESTS FAILED"
  exit 1
fi

log "✅ PASS: real signal detected (source=$src, confidence=$conf)"
echo

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
