#!/usr/bin/env bash
set -euo pipefail

# Set PATH for non-interactive execution (systemd oneshot)
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

SERVICE="quantum-ai-engine.service"
STREAM_INTENT="quantum:stream:trade.intent"

# Proof mode: activity (stable, checks last event freshness) or generate (strict, requires new event)
AI_PROOF_MODE="${AI_PROOF_MODE:-activity}"
AI_PROOF_MAX_AGE_SECONDS="${AI_PROOF_MAX_AGE_SECONDS:-600}"  # 10 minutes for activity mode
AI_PROOF_WAIT_SECONDS="${AI_PROOF_WAIT_SECONDS:-60}"          # timeout for generate mode

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

# Get current stream length
intent_before=$(redis-cli XLEN "$STREAM_INTENT" 2>/dev/null || echo 0)
log "Current stream length: trade.intent=$intent_before"
log "Proof mode: $AI_PROOF_MODE"
echo

# Mode-specific tests
if [[ "$AI_PROOF_MODE" == "generate" ]]; then
  # GENERATE MODE: Strict - requires new event within timeout
  # Check governor kill switch (must be 0 for generate mode)
  kill_switch=$(redis-cli GET quantum:kill 2>/dev/null || echo "1")
  if [ "$kill_switch" != "0" ]; then
    log "❌ FAIL: Governor kill switch is ON (quantum:kill=$kill_switch)"
    log "   Generate mode requires kill=0. Run: redis-cli SET quantum:kill 0"
    log "   Or use AI_PROOF_MODE=activity for stable proof (doesn't require new events)"
    exit 1
  fi
  log "Governor kill switch: OFF (quantum:kill=0) - required for generate mode"
  echo

  check "wait for trade.intent to increase within ${AI_PROOF_WAIT_SECONDS}s" bash -c "
    before=\$(redis-cli XLEN $STREAM_INTENT 2>/dev/null || echo 0)
    end_time=\$((SECONDS + $AI_PROOF_WAIT_SECONDS))
    
    while [ \$SECONDS -lt \$end_time ]; do
      sleep 5
      now=\$(redis-cli XLEN $STREAM_INTENT 2>/dev/null || echo 0)
      if [ \$now -gt \$before ]; then
        echo \"Stream increased: \$before -> \$now (+\$((now - before)) events)\"
        exit 0
      fi
    done
    
    echo \"Stream did not increase within ${AI_PROOF_WAIT_SECONDS}s\"
    echo \"This is expected if AI Engine is in cooldown, filters block signals, or market has no opportunities\"
    exit 1
  "
else
  # ACTIVITY MODE (default): Stable - checks last event freshness
  log "Activity mode: checking last event freshness (max age: ${AI_PROOF_MAX_AGE_SECONDS}s)"
  log "This mode does NOT require new events - it validates existing activity"
  log "Governor kill switch: ignored in activity mode (safe to run with kill=1)"
  echo
fi

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

# Freshness check: extract timestamp from stream entry and calculate age
event_timestamp=$(echo "$last_raw" | grep -A1 "^timestamp$" | tail -1 | tr -d '[:space:]')

if [[ -z "$event_timestamp" ]]; then
  log "⚠️  WARNING: timestamp field missing, cannot validate freshness"
else
  # Parse timestamp and calculate age in seconds
  event_epoch=$(date -d "$event_timestamp" +%s 2>/dev/null || echo "0")
  now_epoch=$(date +%s)
  age_seconds=$((now_epoch - event_epoch))
  
  log "Event timestamp: $event_timestamp"
  log "Event age: ${age_seconds}s (threshold: ${AI_PROOF_MAX_AGE_SECONDS}s)"
  
  if [[ "$AI_PROOF_MODE" == "activity" ]]; then
    # In activity mode, FAIL if event too old
    if [ $age_seconds -gt $AI_PROOF_MAX_AGE_SECONDS ]; then
      log "❌ FAIL: Last event is too old (${age_seconds}s > ${AI_PROOF_MAX_AGE_SECONDS}s threshold)"
      log "   This suggests AI Engine has not published recently"
      log "❌ SOME TESTS FAILED"
      exit 1
    fi
    log "✅ Event is fresh (${age_seconds}s < ${AI_PROOF_MAX_AGE_SECONDS}s)"
  else
    # In generate mode, just log age for info
    log "Event age: ${age_seconds}s (info only in generate mode)"
  fi
fi

log "✅ PASS: real signal detected (source=$src, confidence=$conf)"
echo

log "Latest AI Engine HEALTH:"
journalctl -u "$SERVICE" --since "5 minutes ago" --no-pager | grep -E "HEALTH|signals_generated|drift_passed" | tail -5 || true
echo

if [ "$fail" -eq 0 ]; then
  if [[ "$AI_PROOF_MODE" == "activity" ]]; then
    log "🎉 ALL TESTS PASSED - AI Engine activity validated (fresh real signals)"
  else
    log "🎉 ALL TESTS PASSED - AI Engine generated new signal on demand"
  fi
  exit 0
else
  log "❌ SOME TESTS FAILED"
  exit 1
fi
