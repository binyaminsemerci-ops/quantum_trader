#!/bin/bash
set -euo pipefail

TS=$(date -u +%Y%m%d_%H%M%S)
REPORT="/tmp/p0d4_full_pipeline_FIXED_${TS}.txt"
EXIT_CODE=0

log(){ echo "$@" | tee -a "$REPORT"; }
hr(){ log "============================================================"; }

need_cmd(){
  local bin="$1" pkg="$2"
  if ! command -v "$bin" >/dev/null 2>&1; then
    log "Installing missing: $bin (pkg=$pkg)"
    apt-get update -qq
    apt-get install -y "$pkg" 2>&1 | tee -a "$REPORT"
  fi
}

redis_ok(){ redis-cli PING >/dev/null 2>&1; }

xlen_safe(){
  local key="$1"
  local out
  out=$(redis-cli XLEN "$key" 2>&1 || true)
  if echo "$out" | grep -qiE "(no such key|ERR .*key)"; then
    echo "MISSING"
  else
    echo "$out" | tr -d "\r"
  fi
}

xinfo_stream_lastid(){
  local key="$1"
  local out
  out=$(redis-cli XINFO STREAM "$key" 2>/dev/null || true)
  if [ -z "$out" ]; then
    echo "MISSING"
    return
  fi

  local last
  last=$(echo "$out" | awk 'BEGIN{v=""} $0=="last-generated-id"{getline; v=$0} END{print v}')
  if [ -z "$last" ]; then
    last=$(echo "$out" | awk 'BEGIN{v=""} $0=="last-entry"{getline; v=$0} END{print v}')
  fi

  if [ -z "$last" ]; then
    echo "UNKNOWN"
  else
    echo "$last"
  fi
}

group_field(){
  local stream="$1" groupname="$2" field="$3"
  redis-cli XINFO GROUPS "$stream" 2>/dev/null | awk -v g="$groupname" -v f="$field" '
    $0=="name" { getline; cur=$0 }
    cur==g && $0==f { getline; print $0; found=1 }
    END { if(!found) print "NA" }'
}

svc_active(){
  local svc="$1"
  if systemctl list-unit-files --no-legend 2>/dev/null | awk '{print $1}' | grep -qx "$svc"; then
    systemctl is-active "$svc" 2>/dev/null || echo "unknown"
  else
    echo "not-installed"
  fi
}

curl_health(){
  local port="$1"
  local tmp="/tmp/p0d4_body_${port}.txt"
  local code body
  code=$(curl -s -m 2 -o "$tmp" -w "%{http_code}" "http://127.0.0.1:${port}/health" 2>/dev/null || echo "000")
  body=$(head -c 180 "$tmp" 2>/dev/null | tr -d "\r" || true)
  echo "${code}|${body}"
}

STREAMS=(
  "quantum:stream:ai.decision"
  "quantum:stream:ai.signal_generated"
  "quantum:stream:trade.intent"
  "quantum:stream:execution.result"
  "quantum:stream:trade.execution.res"
  "quantum:stream:trade.intent.dlq"
)

SERVICES=(
  "quantum-ai-engine.service"
  "quantum-strategy-router.service"
  "quantum-risk-approval.service"
  "quantum-risk-safety.service"
  "quantum-execution.service"
  "quantum-execution-result-bridge.service"
  "quantum-exit-brain.service"
  "quantum-exit-monitor.service"
  "quantum-position-monitor.service"
  "quantum-trading_bot.service"
)

PORTS=(8001 8002 8003 8006 8007)

hr
log "P0.D.4 FULL PIPELINE VERIFIER (READ-ONLY) — FIXED"
hr
log "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
log "Host: $(hostname)"
log "Report: $REPORT"
log "Duration: 120s (25 samples @ 5s)"
log ""

hr
log "PREFLIGHT"
hr
need_cmd redis-cli redis-tools
need_cmd curl curl
if ! redis_ok; then
  log "FAIL: Redis PING failed"
  log "Exit code: 3"
  exit 3
fi
log "✅ redis-cli: $(command -v redis-cli)"
log "✅ Redis PING: OK"
log ""

hr
log "SERVICE STATUS"
hr
for s in "${SERVICES[@]}"; do
  st=$(svc_active "$s")
  log "$s: $st"
done
log ""

hr
log "HEALTH ENDPOINTS (best-effort)"
hr
for p in "${PORTS[@]}"; do
  res=$(curl_health "$p")
  code="${res%%|*}"
  body="${res#*|}"
  if [ "$code" = "200" ]; then
    log "127.0.0.1:$p/health: ✅ 200 | ${body}"
  else
    log "127.0.0.1:$p/health: ⚠️ $code"
  fi
done
log ""

hr
log "STREAM SNAPSHOT BEFORE"
hr
declare -A XLEN_BEFORE LASTID_BEFORE
for k in "${STREAMS[@]}"; do
  XLEN_BEFORE["$k"]=$(xlen_safe "$k")
  LASTID_BEFORE["$k"]=$(xinfo_stream_lastid "$k")
  log "$k | XLEN=${XLEN_BEFORE[$k]} | LAST_ID=${LASTID_BEFORE[$k]}"
done
log ""

GROUP_STREAM="quantum:stream:trade.intent"
GROUP_NAME="quantum:group:execution:trade.intent"

PENDING_BEFORE=$(group_field "$GROUP_STREAM" "$GROUP_NAME" "pending")
LAG_BEFORE=$(group_field "$GROUP_STREAM" "$GROUP_NAME" "lag")
ENTRIES_READ_BEFORE=$(group_field "$GROUP_STREAM" "$GROUP_NAME" "entries-read")

log "GROUP BEFORE ($GROUP_NAME on $GROUP_STREAM): pending=$PENDING_BEFORE lag=$LAG_BEFORE entries-read=$ENTRIES_READ_BEFORE"
log ""

hr
log "120s SAMPLING (25 samples @ 5s)"
hr
SAMPLES="/tmp/p0d4_samples_FIXED_${TS}.csv"
echo "ts,pending,entries_read,dlq_len,exec_lastid_changed,bridge_lastid_changed" > "$SAMPLES"

START=$(date -u +%s)
BASE_EXEC_LAST="${LASTID_BEFORE[quantum:stream:execution.result]}"
BASE_BRIDGE_LAST="${LASTID_BEFORE[quantum:stream:trade.execution.res]}"

for i in $(seq 1 25); do
  now=$(date -u +%s)
  p=$(group_field "$GROUP_STREAM" "$GROUP_NAME" "pending")
  er=$(group_field "$GROUP_STREAM" "$GROUP_NAME" "entries-read")
  dlq=$(xlen_safe "quantum:stream:trade.intent.dlq")
  exec_last=$(xinfo_stream_lastid "quantum:stream:execution.result")
  br_last=$(xinfo_stream_lastid "quantum:stream:trade.execution.res")

  exec_delta=0
  br_delta=0
  [ "$exec_last" != "$BASE_EXEC_LAST" ] && exec_delta=1
  [ "$br_last" != "$BASE_BRIDGE_LAST" ] && br_delta=1

  log "S$i ts=$now pending=$p entries-read=$er dlq=$dlq exec_lastid_changed=$exec_delta bridge_lastid_changed=$br_delta"
  echo "$now,$p,$er,$dlq,$exec_delta,$br_delta" >> "$SAMPLES"
  sleep 5
done

END=$(date -u +%s)
ELAPSED=$((END-START))
log ""
log "Sampling elapsed: ${ELAPSED}s"
log ""

hr
log "STREAM SNAPSHOT AFTER"
hr
declare -A XLEN_AFTER LASTID_AFTER
for k in "${STREAMS[@]}"; do
  XLEN_AFTER["$k"]=$(xlen_safe "$k")
  LASTID_AFTER["$k"]=$(xinfo_stream_lastid "$k")
  log "$k | XLEN=${XLEN_AFTER[$k]} | LAST_ID=${LASTID_AFTER[$k]}"
done
log ""

PENDING_AFTER=$(group_field "$GROUP_STREAM" "$GROUP_NAME" "pending")
LAG_AFTER=$(group_field "$GROUP_STREAM" "$GROUP_NAME" "lag")
ENTRIES_READ_AFTER=$(group_field "$GROUP_STREAM" "$GROUP_NAME" "entries-read")
DLQ_AFTER=$(xlen_safe "quantum:stream:trade.intent.dlq")

log "GROUP AFTER ($GROUP_NAME on $GROUP_STREAM): pending=$PENDING_AFTER lag=$LAG_AFTER entries-read=$ENTRIES_READ_AFTER"
log ""

hr
log "ANALYSIS + PASS/FAIL"
hr

EXEC_SVC=$(svc_active "quantum-execution.service")
BRIDGE_SVC=$(svc_active "quantum-execution-result-bridge.service")

if [ "$EXEC_SVC" != "active" ]; then
  log "❌ FAIL: quantum-execution.service is $EXEC_SVC"
  EXIT_CODE=3
fi

PDELTA=0; PRATE=0
ERDELTA=0; THR=0

if [[ "$PENDING_BEFORE" =~ ^[0-9]+$ ]] && [[ "$PENDING_AFTER" =~ ^[0-9]+$ ]]; then
  PDELTA=$((PENDING_AFTER - PENDING_BEFORE))
  PRATE=$(( (PDELTA * 60) / (ELAPSED>0?ELAPSED:1) ))
else
  log "⚠️ WARN: Could not parse pending integers (before=$PENDING_BEFORE after=$PENDING_AFTER)"
  [ $EXIT_CODE -eq 0 ] && EXIT_CODE=2
fi

if [[ "$ENTRIES_READ_BEFORE" =~ ^[0-9]+$ ]] && [[ "$ENTRIES_READ_AFTER" =~ ^[0-9]+$ ]]; then
  ERDELTA=$((ENTRIES_READ_AFTER - ENTRIES_READ_BEFORE))
  THR=$(( (ERDELTA * 60) / (ELAPSED>0?ELAPSED:1) ))
else
  log "⚠️ WARN: Could not parse entries-read integers (before=$ENTRIES_READ_BEFORE after=$ENTRIES_READ_AFTER)"
  [ $EXIT_CODE -eq 0 ] && EXIT_CODE=2
fi

AI_DEC_MOVED=0
TI_MOVED=0
EX_MOVED=0
BR_MOVED=0

[ "${LASTID_AFTER[quantum:stream:ai.decision]}" != "${LASTID_BEFORE[quantum:stream:ai.decision]}" ] && AI_DEC_MOVED=1
[ "${LASTID_AFTER[quantum:stream:trade.intent]}" != "${LASTID_BEFORE[quantum:stream:trade.intent]}" ] && TI_MOVED=1
[ "${LASTID_AFTER[quantum:stream:execution.result]}" != "${LASTID_BEFORE[quantum:stream:execution.result]}" ] && EX_MOVED=1
[ "${LASTID_AFTER[quantum:stream:trade.execution.res]}" != "${LASTID_BEFORE[quantum:stream:trade.execution.res]}" ] && BR_MOVED=1

log "Movement flags (0/1): ai.decision=$AI_DEC_MOVED trade.intent=$TI_MOVED execution.result=$EX_MOVED trade.execution.res=$BR_MOVED"
log "Rates: pending_delta=$PDELTA pending_rate=${PRATE}/min entries_read_delta=$ERDELTA throughput=${THR} msg/min"
log "Bridge service: $BRIDGE_SVC"
log ""

if [ $TI_MOVED -eq 1 ] && [ $EX_MOVED -eq 0 ]; then
  log "❌ FAIL: trade.intent moved but execution.result did not (execution pipeline broken OR stream missing)"
  EXIT_CODE=3
fi
if [ $EX_MOVED -eq 1 ] && [ $BR_MOVED -eq 0 ]; then
  log "❌ FAIL: execution.result moved but trade.execution.res did not (bridge broken OR stream missing)"
  EXIT_CODE=3
fi

if [ $EXIT_CODE -lt 3 ]; then
  if [ $PRATE -gt 1000 ]; then
    log "❌ FAIL: pending rate too high (+${PRATE}/min > 1000/min)"
    EXIT_CODE=3
  elif [ $PRATE -gt 200 ]; then
    log "⚠️ DEGRADED: pending rate elevated (+${PRATE}/min)"
    [ $EXIT_CODE -eq 0 ] && EXIT_CODE=2
  else
    log "✅ OK: pending rate within healthy band (${PRATE}/min)"
  fi
fi

DLQDELTA=0
if [[ "$DLQ_AFTER" =~ ^[0-9]+$ ]] && [[ "${XLEN_BEFORE[quantum:stream:trade.intent.dlq]}" =~ ^[0-9]+$ ]]; then
  DLQDELTA=$((DLQ_AFTER - XLEN_BEFORE[quantum:stream:trade.intent.dlq]))
fi
if [ $DLQDELTA -gt 0 ]; then
  log "⚠️ DEGRADED: DLQ grew by +$DLQDELTA during window"
  [ $EXIT_CODE -eq 0 ] && EXIT_CODE=2
else
  log "✅ OK: DLQ stable"
fi

hr
log "SIGN-OFF"
hr
if [ $EXIT_CODE -eq 0 ]; then
  log "CONCLUSION: 🟢 HEALTHY"
elif [ $EXIT_CODE -eq 2 ]; then
  log "CONCLUSION: 🟡 DEGRADED (operational but stressed)"
else
  log "CONCLUSION: 🔴 FAIL"
fi
log "Exit code: $EXIT_CODE (0=HEALTHY, 2=DEGRADED, 3=FAIL)"
log "Report: $REPORT"
log "Samples: $SAMPLES"
hr

echo ""
echo "✅ P0.D.4 COMPLETE (FIXED)"
echo "Report: $REPORT"
echo "Exit code: $EXIT_CODE"
echo ""
echo "=== LAST 120 LINES ==="
tail -120 "$REPORT"

exit $EXIT_CODE
