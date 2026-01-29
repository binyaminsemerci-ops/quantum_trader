#!/bin/bash
NOW=$(date +%s)
date -u
systemctl is-active quantum-portfolio-heat-gate quantum-apply-layer || true
redis-cli HMSET quantum:state:portfolio equity_usd 15000 >/dev/null
redis-cli XADD quantum:stream:harvest.proposal '*' plan_id PASS_$NOW trace_id PASS_$NOW symbol ETHUSDT action FULL_CLOSE reason P0_PASS_PROOF timestamp $NOW decision EXECUTE kill_score 0.75 >/dev/null
sleep 3
C=$(redis-cli HGET quantum:harvest:proposal:ETHUSDT calibrated || true)
A=$(redis-cli HGET quantum:harvest:proposal:ETHUSDT action || true)
O=$(redis-cli HGET quantum:harvest:proposal:ETHUSDT original_action || true)
T=$(redis-cli TTL quantum:harvest:proposal:ETHUSDT || true)
echo calibrated=$C action=$A original_action=$O ttl=$T
redis-cli HMSET quantum:state:portfolio equity_usd 50000 >/dev/null
if [ "$C" = "1" ] && [ -n "$A" ]; then
  echo VERDICT:PASS
else
  echo VERDICT:FAIL
fi
