#!/bin/bash
TIMESTAMP=$(date +%s)
redis-cli HSET quantum:harvest:proposal:BTCUSDT \
  harvest_action PARTIAL_75 \
  kill_score 0.30 \
  k_regime_flip 0.05 \
  k_sigma_spike 0.05 \
  k_ts_drop 0.10 \
  k_age_penalty 0.04 \
  new_sl_proposed 100.0 \
  R_net 8.0 \
  last_update_epoch $TIMESTAMP \
  computed_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  reason_codes test_low_kill

echo "✓ Proposal injected (kill_score=0.3)"
sleep 7
echo ""
echo "EXECUTE check:"
journalctl -u quantum-apply-layer --since "6 seconds ago" --no-pager | grep "BTCUSDT.*Plan" | grep "decision=EXECUTE"
