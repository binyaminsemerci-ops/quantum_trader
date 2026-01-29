#!/bin/bash
set -euo pipefail
echo "=== P0.PASS-PROOF HARVEST (UTC) ==="
date -u
echo

echo "=== 1) SERVICES ==="
systemctl is-active quantum-portfolio-heat-gate quantum-apply-layer || true
echo

echo "=== 2) HEAT GATE METRICS (8056) ==="
curl -s http://127.0.0.1:8056/metrics | egrep "p26_(enforce_mode|proposals_processed_total|hash_writes_total|hash_write_fail_total|actions_downgraded_total|failures_total|heat_value|bucket)" || true
echo

echo "=== 3) STREAM HEALTH ==="
redis-cli PING
echo -n "harvest.proposal XLEN: "; redis-cli XLEN quantum:stream:harvest.proposal 2>/dev/null || echo "ERR"
echo -n "harvest.calibrated XLEN: "; redis-cli XLEN quantum:stream:harvest.calibrated 2>/dev/null || echo "ERR"
echo

echo "=== 4) TEST INJECT (ETHUSDT FULL_CLOSE) ==="
NOW=$(date +%s)
redis-cli HMSET quantum:state:portfolio equity_usd 15000 >/dev/null
redis-cli XADD quantum:stream:harvest.proposal "*" plan_id "TEST_PASS_${NOW}" trace_id "TEST_PASS_${NOW}" symbol "ETHUSDT" action "FULL_CLOSE" reason "P0_PASS_PROOF" timestamp "${NOW}" decision "EXECUTE" kill_score "0.75" >/dev/null
sleep 3

echo "=== 5) HASH PROOF (ETHUSDT) ==="
redis-cli HGET quantum:harvest:proposal:ETHUSDT calibrated | xargs echo "calibrated="
redis-cli HGET quantum:harvest:proposal:ETHUSDT original_action | xargs echo "original_action="
redis-cli HGET quantum:harvest:proposal:ETHUSDT action | xargs echo "action="
redis-cli HGET quantum:harvest:proposal:ETHUSDT downgrade_reason | xargs echo "downgrade_reason="
redis-cli TTL quantum:harvest:proposal:ETHUSDT | xargs echo "ttl="
echo

echo "=== 6) APPLY LAYER LOG PROOF (last 60s) ==="
journalctl -u quantum-apply-layer --since "90 seconds ago" --no-pager | egrep -i "ETHUSDT|calibrated|PARTIAL_25|FULL_CLOSE|blocking non-close action" | tail -30 || true
echo

echo "=== 7) RESTORE EQUITY ==="
redis-cli HMSET quantum:state:portfolio equity_usd 50000 >/dev/null
echo "OK"
echo

echo "=== VERDICT RULE ==="
C=$(redis-cli HGET quantum:harvest:proposal:ETHUSDT calibrated || true)
A=$(redis-cli HGET quantum:harvest:proposal:ETHUSDT action || true)
if [[ "$C" == "1" && -n "$A" ]]; then
  echo "VERDICT: PASS (calibrated=1 and action present; Apply Layer should consume calibrated action)"
else
  echo "VERDICT: FAIL (missing calibrated/action in proposal hash)"
fi
