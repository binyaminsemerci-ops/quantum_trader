#!/bin/bash
# Synthetic E2E test - inject proposal and verify permit chain

echo "=========================================="
echo "E2E TESTNET - Synthetic EXECUTE Test"
echo "=========================================="
echo ""

TIMESTAMP=$(date +%s)
echo "[$(date)] Step 1: Injecting synthetic proposal..."

redis-cli HSET quantum:harvest:proposal:BTCUSDT \
  harvest_action PARTIAL_75 \
  kill_score 0.45 \
  k_regime_flip 0.1 \
  k_sigma_spike 0.15 \
  k_ts_drop 0.2 \
  k_age_penalty 0.04 \
  new_sl_proposed 100.0 \
  R_net 7.5 \
  last_update_epoch $TIMESTAMP \
  computed_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  reason_codes synthetic_test

echo "✓ Proposal injected"
echo ""

echo "[$(date)] Step 2: Waiting for Apply Layer cycle (7s)..."
sleep 7

echo ""
echo "[$(date)] Step 3: Checking for EXECUTE plan..."
PLAN=$(journalctl -u quantum-apply-layer --since "6 seconds ago" --no-pager | grep "decision=EXECUTE" | grep -oE "[0-9a-f]{16}" | head -1)

if [ -z "$PLAN" ]; then
  echo "❌ No EXECUTE plan found"
  exit 1
fi

echo "✓ EXECUTE DETECTED: $PLAN"
echo ""

sleep 1

echo "[$(date)] Step 4: Checking Governor permit (P3.2)..."
GOV=$(redis-cli GET quantum:permit:$PLAN)
if [ -n "$GOV" ]; then
  echo "✓ Governor permit found:"
  echo "  $GOV" | head -c 100
  echo ""
else
  echo "❌ Governor permit MISSING"
fi

echo ""
echo "[$(date)] Step 5: Checking P3.3 permit..."
P33=$(redis-cli GET quantum:permit:p33:$PLAN)
if [ -n "$P33" ]; then
  echo "✓ P3.3 permit found:"
  echo "  $P33" | head -c 100
  echo ""
else
  echo "❌ P3.3 permit MISSING"
fi

echo ""
echo "[$(date)] Step 6: Checking execution logs..."
journalctl -u quantum-apply-layer --since "5 seconds ago" --no-pager | grep "$PLAN" | grep -E "executed=|permits ready" | head -5

echo ""
echo "=========================================="
echo "E2E Test Complete"
echo "=========================================="
