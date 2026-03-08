#!/bin/bash
# PATCH-6 end-to-end testnet test
# Injects a synthetic SOLUSDT LONG position with SL already breached
# (entry=90.00, stop_loss=88.00, current mark ~83.09)
# This triggers DecisionEngine Rule 2 (Hard SL breach → FULL_CLOSE EMERGENCY)
# on the very first EMA tick after injection.
#
# Flow being tested:
#   quantum:position:SOLUSDT   (EMA source)
#   quantum:stream:position.snapshot (AT source)
#     EMA tick (5s) → FULL_CLOSE EMERGENCY decision
#     EMA → quantum:stream:exit.intent  [PATCH-5A live write]
#     exit_intent_gateway → quantum:stream:trade.intent [PATCH-5B]
#     AT cycle (~30s) → EXIT_OWNERSHIP_SUSPENDED [PATCH-2 kill-switch]
#     No direct writes to quantum:stream:apply.plan from EMA/gateway

set -e

echo "============================================================"
echo "PATCH-6 E2E TESTNET TEST — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── STEP 0: Baselines ────────────────────────────────────────────
echo ""
echo "=== STEP 0: BASELINES ==="
EI_BEFORE=$(redis-cli XLEN quantum:stream:exit.intent)
TI_BEFORE=$(redis-cli XLEN quantum:stream:trade.intent)
AP_BEFORE=$(redis-cli XLEN quantum:stream:apply.plan)
EA_BEFORE=$(redis-cli XLEN quantum:stream:exit.audit)
NOW_TS=$(date +%s)

echo "exit.intent:  $EI_BEFORE"
echo "trade.intent: $TI_BEFORE"
echo "apply.plan:   $AP_BEFORE"
echo "exit.audit:   $EA_BEFORE"

# ── STEP 1: Inject synthetic position (EMA source) ───────────────
echo ""
echo "=== STEP 1: INJECT quantum:position:SOLUSDT ==="
# Derived mark_price = entry + unrealized_pnl/qty = 90 + (-2.07/0.3) = 83.1
# Validator V6: notional = mark_price * qty = 83.1 * 0.3 = 24.93 >= 20 PASS
# Rule 2 SL breach: mark_price(ticker ~83.09) < stop_loss(88.00) → FULL_CLOSE EMERGENCY
redis-cli HSET quantum:position:SOLUSDT \
    side LONG \
    quantity 0.3 \
    entry_price 90.00 \
    stop_loss 88.00 \
    take_profit 0 \
    leverage 1 \
    unrealized_pnl -2.07 \
    entry_risk_usdt 5.0 \
    sync_timestamp $NOW_TS \
    source e2e_test_p6

echo "Position hash written. Verifying:"
redis-cli HGETALL quantum:position:SOLUSDT

# ── STEP 2: Inject position.snapshot for AT ──────────────────────
echo ""
echo "=== STEP 2: INJECT quantum:stream:position.snapshot (AT source) ==="
SNAP_ID=$(redis-cli XADD quantum:stream:position.snapshot '*' \
    event_type position.snapshot \
    symbol SOLUSDT \
    side LONG \
    position_qty 0.3 \
    entry_price 90.00 \
    mark_price 83.10 \
    unrealized_pnl -2.07 \
    leverage 1 \
    isolated False \
    liquidation_price 0 \
    margin_type cross \
    stop_loss 88.00 \
    take_profit 0 \
    entry_timestamp $((NOW_TS - 300)) \
    entry_regime UNKNOWN \
    entry_confidence 0.5 \
    timestamp $NOW_TS \
    source e2e_test_p6)
echo "position.snapshot published: $SNAP_ID"

# ── STEP 3: Wait for EMA tick + gateway forward ──────────────────
echo ""
echo "=== STEP 3: WAITING 12s FOR EMA TICK + GATEWAY FORWARD ==="
sleep 12

echo ""
echo "=== STEP 3a: EMA JOURNAL — (last 30 lines, PATCH-6 events) ==="
journalctl -u quantum-exit-management-agent -n 30 --no-pager | grep -E 'OWNERSHIP_FLAG|EXIT_DECISION|TICK|_EXIT|intent|AUDIT|PATCH' | tail -15

echo ""
echo "=== STEP 3b: exit.intent XLEN (expected > $EI_BEFORE) ==="
EI_AFTER=$(redis-cli XLEN quantum:stream:exit.intent)
echo "exit.intent: was=$EI_BEFORE now=$EI_AFTER delta=$((EI_AFTER - EI_BEFORE))"
if [ "$((EI_AFTER - EI_BEFORE))" -gt 0 ]; then
    echo "RESULT: exit.intent INCREASED — EMA wrote exit decision"
    echo "--- Latest exit.intent entry ---"
    redis-cli XREVRANGE quantum:stream:exit.intent + - COUNT 1
else
    echo "RESULT: exit.intent NOT increased yet — EMA tick may not have fired"
fi

echo ""
echo "=== STEP 3c: trade.intent XLEN (expected > $TI_BEFORE) ==="
TI_AFTER=$(redis-cli XLEN quantum:stream:trade.intent)
echo "trade.intent: was=$TI_BEFORE now=$TI_AFTER delta=$((TI_AFTER - TI_BEFORE))"
if [ "$((TI_AFTER - TI_BEFORE))" -gt 0 ]; then
    echo "RESULT: trade.intent INCREASED — gateway forwarded"
    echo "--- Latest trade.intent entry ---"
    redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1
else
    echo "RESULT: trade.intent NOT increased yet"
fi

# ── STEP 4: Wait for AT cycle ────────────────────────────────────
echo ""
echo "=== STEP 4: WAITING 35s MORE FOR AT CYCLE (total ~47s from injection) ==="
sleep 35

echo ""
echo "=== STEP 4a: AT JOURNAL — EXIT_OWNERSHIP_SUSPENDED ==="
AT_SUSPEND=$(journalctl -u quantum-autonomous-trader --since "$(date -u -d '-90 seconds' +%Y-%m-%d\ %H:%M:%S)" --no-pager | grep EXIT_OWNERSHIP_SUSPENDED)
if [ -n "$AT_SUSPEND" ]; then
    echo "FOUND:"
    echo "$AT_SUSPEND"
else
    echo "NOT FOUND in last 90s — checking 3min window..."
    journalctl -u quantum-autonomous-trader -n 200 --no-pager | grep EXIT_OWNERSHIP_SUSPENDED | tail -5
fi

echo ""
echo "=== STEP 4b: EMA TICK with SOLUSDT — confirm actionable=1 ==="
journalctl -u quantum-exit-management-agent --since "$(date -u -d '-90 seconds' +%Y-%m-%d\ %H:%M:%S)" --no-pager | grep 'TICK' | tail -5

echo ""
echo "=== STEP 4c: exit.audit stream (EMA creates audit entry per decision) ==="
EA_AFTER=$(redis-cli XLEN quantum:stream:exit.audit)
echo "exit.audit: was=$EA_BEFORE now=$EA_AFTER delta=$((EA_AFTER - EA_BEFORE))"
if [ "$((EA_AFTER - EA_BEFORE))" -gt 0 ]; then
    echo "--- Latest exit.audit entry ---"
    redis-cli XREVRANGE quantum:stream:exit.audit + - COUNT 1
fi

# ── STEP 5: Final stream checks ──────────────────────────────────
echo ""
echo "=== STEP 5: FINAL STREAM CHECKS ==="
EI_FINAL=$(redis-cli XLEN quantum:stream:exit.intent)
TI_FINAL=$(redis-cli XLEN quantum:stream:trade.intent)
AP_FINAL=$(redis-cli XLEN quantum:stream:apply.plan)

echo "exit.intent:  was=$EI_BEFORE final=$EI_FINAL delta=$((EI_FINAL - EI_BEFORE))"
echo "trade.intent: was=$TI_BEFORE final=$TI_FINAL delta=$((TI_FINAL - TI_BEFORE))"
echo "apply.plan:   was=$AP_BEFORE final=$AP_FINAL delta=$((AP_FINAL - AP_BEFORE))"

echo ""
echo "=== STEP 5a: Latest exit.intent (full) ==="
redis-cli XREVRANGE quantum:stream:exit.intent + - COUNT 1

echo ""
echo "=== STEP 5b: Latest trade.intent (full) ==="
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1

echo ""
echo "=== STEP 5c: Latest apply.plan (full — should NOT be from EMA/gateway) ==="
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 1

# ── STEP 6: Cleanup ──────────────────────────────────────────────
echo ""
echo "=== STEP 6: CLEANUP — removing synthetic position ==="
redis-cli DEL quantum:position:SOLUSDT
echo "quantum:position:SOLUSDT deleted"
echo "Verifying deletion:"
redis-cli HGETALL quantum:position:SOLUSDT

echo ""
echo "============================================================"
echo "E2E TEST COMPLETE — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
