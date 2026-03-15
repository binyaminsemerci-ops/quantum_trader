#!/bin/bash
# =============================================================================
# FULL LAYER AUDIT — quantum_trader pipeline A to Z
# Checks: service running, heartbeat, Redis keys, last activity
# =============================================================================
QT="/home/qt/quantum_trader"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
OK="${GREEN}[OK]${NC}"; FAIL="${RED}[FAIL]${NC}"; WARN="${YELLOW}[WARN]${NC}"

svc_ok()  { systemctl is-active "$1" --quiet 2>/dev/null && echo -e "$OK  $1" || echo -e "$FAIL $1 ($(systemctl is-active $1 2>/dev/null))"; }
key_age() { # key_age KEY_PATTERN label max_age_sec
  local k="$1" lbl="$2" max="$3"
  local ts=$(redis-cli get "$k" 2>/dev/null | head -1)
  if [ -z "$ts" ] || [ "$ts" = "" ]; then
    echo -e "$FAIL $lbl → key missing ($k)"
  else
    local age=$(( $(date +%s) - ts ))
    [ "$age" -le "$max" ] && echo -e "$OK  $lbl → ${age}s ago" || echo -e "$WARN $lbl → ${age}s ago (max=${max}s)"
  fi
}
stream_len() { # stream_len STREAM label
  local s="$1" lbl="$2"
  local n=$(redis-cli xlen "$s" 2>/dev/null)
  [ -z "$n" ] && echo -e "$FAIL $lbl → stream missing ($s)" || echo -e "$OK  $lbl → $n msgs in $s"
}
last_entry() { # last_entry STREAM label
  local s="$1" lbl="$2"
  local last=$(redis-cli xrevrange "$s" + - COUNT 1 2>/dev/null | head -1)
  if [ -z "$last" ]; then
    echo -e "$FAIL $lbl → no recent entries ($s)"
  else
    # Extract millisecond timestamp from stream ID
    local ms=$(echo "$last" | grep -oE '^[0-9]+' | head -1)
    if [ -n "$ms" ]; then
      local age=$(( $(date +%s) - ms/1000 ))
      [ "$age" -le 120 ] && echo -e "$OK  $lbl → ${age}s ago" || echo -e "$WARN $lbl → ${age}s ago (stale?)"
    else
      echo -e "$WARN $lbl → entries exist, id=$last"
    fi
  fi
}

echo "============================================================"
echo " QUANTUM TRADER — FULL LAYER AUDIT  $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

echo ""
echo "━━━ LAYER 0: RAW DATA SOURCES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
svc_ok quantum-layer1-data-sink
svc_ok quantum-exchange-stream-bridge
svc_ok quantum-cross-exchange-aggregator
# Check if price feed data is flowing
last_entry "quantum:stream:market.klines" "price feed (klines)"
last_entry "quantum:stream:features.live" "live features"
# Check Layer1 persistence
redis-cli keys "quantum:ohlcv:*" 2>/dev/null | wc -l | xargs -I{} echo -e "      ohlcv keys in Redis: {}"

echo ""
echo "━━━ LAYER 1: FEATURES & SIGNALS ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
svc_ok quantum-feature-publisher
svc_ok quantum-layer2-research-sandbox
# Feature publisher heartbeat
redis_hb=$(redis-cli get "quantum:svc:feature_publisher:heartbeat" 2>/dev/null)
[ -n "$redis_hb" ] && ts_diff=$(( $(date +%s) - redis_hb )) && [ "$ts_diff" -le 60 ] \
  && echo -e "$OK  feature_publisher heartbeat → ${ts_diff}s ago" \
  || echo -e "$WARN feature_publisher heartbeat → ${ts_diff:-missing}s ago"
last_entry "quantum:stream:features.raw" "features.raw stream"

echo ""
echo "━━━ LAYER 2: AI PREDICTION (ENSEMBLE) ━━━━━━━━━━━━━━━━━━━━━"
svc_ok quantum-ai-engine
svc_ok quantum-ensemble-predictor
svc_ok quantum-ai-strategy-router
# AI engine heartbeat
redis-cli keys "quantum:svc:ai_engine*" 2>/dev/null | head -3 | xargs -I{} sh -c 'v=$(redis-cli get "{}" 2>/dev/null); age=$(( $(date +%s) - ${v:-0} )); echo "      ai_engine hb: {}: ${age}s ago"'
last_entry "quantum:stream:trade.intent" "trade.intent (AI signals)"
# Check intent volume
redis-cli xlen "quantum:stream:trade.intent" 2>/dev/null | xargs -I{} echo "      trade.intent length: {}"

echo ""
echo "━━━ LAYER 2.5: HARVEST / SHADOW / PAPER TRADE ━━━━━━━━━━━━━"
svc_ok quantum-harvest-v2
svc_ok quantum-harvest-brain
svc_ok quantum-harvest-proposal
# Shadow mode status
shadow=$(redis-cli hget quantum:shadow:controller:state phase 2>/dev/null)
dag8_phase=$(redis-cli get quantum:dag8:phase 2>/dev/null)
gate=$(redis-cli get quantum:dag8:gate 2>/dev/null)
echo "      shadow phase: ${shadow:-missing}"
echo "      DAG8 phase: ${dag8_phase:-missing}  gate: ${gate:-missing}"
last_entry "quantum:stream:harvest.proposals" "harvest proposals"

echo ""
echo "━━━ LAYER 3: INTENT BRIDGE (trade.intent→apply.plan) ━━━━━━"
svc_ok quantum-intent-bridge
# Build tag check
journalctl -u quantum-intent-bridge --no-pager -n 200 -q 2>/dev/null | grep "Intent Bridge -" | tail -1 | grep -o '\[.*\]'
# Anti-churn keys
n_churn=$(redis-cli keys "quantum:intent_bridge:last_close:*" 2>/dev/null | wc -l)
echo "      anti-churn keys active: $n_churn"
# Recent bridge activity
last_entry "quantum:stream:apply.plan" "apply.plan (post-bridge)"
redis-cli xlen "quantum:stream:apply.plan" 2>/dev/null | xargs -I{} echo "      apply.plan length: {}"

echo ""
echo "━━━ LAYER 3.2: GOVERNOR (permits) ━━━━━━━━━━━━━━━━━━━━━━━━━"
svc_ok quantum-governor
# Governor activity
journalctl -u quantum-governor --no-pager --since "2 minutes ago" -q 2>/dev/null | grep -c "ALLOW\|DENY\|Evaluating" | xargs -I{} echo "      governor decisions last 2min: {}"
# P3.3/P2.6 permits
p33=$(redis-cli keys "quantum:permit:p33:*" 2>/dev/null | wc -l)
p26=$(redis-cli keys "quantum:permit:p26:*" 2>/dev/null | wc -l)
echo "      p33 permits in Redis: $p33"
echo "      p26 permits in Redis: $p26"

echo ""
echo "━━━ LAYER 3.5: APPLY LAYER (execution gate) ━━━━━━━━━━━━━━━"
svc_ok quantum-apply-layer
# Governor mode
journalctl -u quantum-apply-layer --no-pager -n 500 -q 2>/dev/null | grep -E "Governor active|GOVERNOR_BYPASS|TESTNET execution" | tail -2
# Exit ownership
journalctl -u quantum-apply-layer --no-pager -n 500 -q 2>/dev/null | grep -E "EXIT_OWNER|exit_own|WARN.*exit" | tail -3
# Recent apply result
last_entry "quantum:stream:apply.result" "apply.result stream"
exec_ok=$(redis-cli xrevrange quantum:stream:apply.result + - COUNT 20 2>/dev/null | grep '"executed"' | grep -c '"True"' 2>/dev/null || echo "0")
echo "      last 20 results with executed=True: $exec_ok"

echo ""
echo "━━━ LAYER 4: EXECUTION (Binance orders) ━━━━━━━━━━━━━━━━━━━"
svc_ok quantum-execution
svc_ok quantum-intent-executor
# Last fill event
last_entry "quantum:stream:execution.fills" "execution fills"
# Open positions in local ledger
pos_count=$(redis-cli keys "quantum:state:positions:*" 2>/dev/null | wc -l)
echo "      open position keys: $pos_count"
# Ledger
led_count=$(redis-cli keys "quantum:ledger:*" 2>/dev/null | wc -l)
echo "      ledger keys: $led_count"

echo ""
echo "━━━ LAYER 5: EXIT SYSTEM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
svc_ok quantum-exit-intelligence
svc_ok quantum-exit-monitor
# Exitbrain
ls /home/qt/quantum_trader/microservices/ 2>/dev/null | grep -i exit | while read d; do
  svc_name="quantum-$(echo $d | tr '_' '-')"
  systemctl is-active "$svc_name" --quiet 2>/dev/null && echo -e "$OK  $svc_name" || echo -e "$WARN $svc_name (check name)"
done

echo ""
echo "━━━ LAYER 6: HARDWARE STOPS (DAG3) ━━━━━━━━━━━━━━━━━━━━━━━━"
svc_ok quantum-dag3-hw-stops
hw_keys=$(redis-cli keys "quantum:position:*:hardware_stops" 2>/dev/null | wc -l)
echo "      hardware stop keys: $hw_keys"

echo ""
echo "━━━ LAYER 7: GUARDS & SAFETY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
svc_ok quantum-anti-churn-guard
svc_ok quantum-dag4-deadlock-guard
svc_ok quantum-dag5-lockdown-guard
svc_ok quantum-dag8-freeze-exit
svc_ok quantum-emergency-exit-worker
# Kill switch
ks=$(redis-cli get "quantum:global:kill_switch" 2>/dev/null)
echo "      kill_switch: ${ks:-off}"

echo ""
echo "━━━ LAYER 8: LEARNING PLANE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# RL trainer / CLM
redis-cli keys "quantum:svc:rl*" 2>/dev/null | head -5 | while read k; do
  v=$(redis-cli get "$k" 2>/dev/null)
  age=$(( $(date +%s) - ${v:-0} ))
  echo "      $k: ${age}s ago"
done
redis-cli keys "quantum:svc:clm*" 2>/dev/null | head -3 | while read k; do
  v=$(redis-cli get "$k" 2>/dev/null)
  age=$(( $(date +%s) - ${v:-0} ))
  echo "      $k: ${age}s ago"
done

echo ""
echo "━━━ LAYER 9: RECONCILE / POSITION SYNC ━━━━━━━━━━━━━━━━━━━━"
# Check if reconcile_engine service exists
systemctl is-active quantum-reconcile --quiet 2>/dev/null \
  && echo -e "$OK  quantum-reconcile" \
  || (systemctl list-units --all | grep -q reconcile \
    && echo -e "$WARN quantum-reconcile (exists but not active)" \
    || echo -e "$WARN reconcile service not found (manually patched?)")
last_entry "quantum:stream:reconcile.close" "reconcile.close stream"

echo ""
echo "━━━ PIPELINE SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Services total running:"
systemctl list-units --state=running | grep quantum | wc -l | xargs echo "  quantum services active:"
echo ""
echo "Last 5 executed trades:"
redis-cli xrevrange quantum:stream:apply.result + - COUNT 20 2>/dev/null \
  | grep -A1 "executed" | grep "True" | head -5 \
  || redis-cli xrevrange quantum:stream:execution.fills + - COUNT 5 2>/dev/null | head -20
echo ""
echo "============================================================"
echo " DONE"
echo "============================================================"
