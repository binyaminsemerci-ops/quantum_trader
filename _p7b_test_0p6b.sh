#!/usr/bin/env bash
# PATCH-7B shadow test with qwen3:0.6b
# Injects a safe test position then monitors the next tick for qwen3 audit fields

RC=$(which redis-cli 2>/dev/null || echo /usr/bin/redis-cli)

echo "=== P7B SHADOW TEST: qwen3:0.6b ==="
echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- 1. Inject position ----
MARK=1950.65
ENTRY=2000.00
QTY=0.01
SL=1800.00
TP=2400.00
ENTRY_RISK=5.0
UPNL=$(python3 -c "print(round(($MARK - $ENTRY) * $QTY, 4))")
R_NET=$(python3 -c "print(round(($MARK - $ENTRY) * $QTY / $ENTRY_RISK, 4))")

echo ""
echo "--- Position parameters ---"
echo "  mark=$MARK  entry=$ENTRY  R_net=$R_NET  SL=$SL (safe: mark > SL)"

$RC HSET quantum:position:ETHUSDT \
  symbol ETHUSDT \
  side LONG \
  quantity $QTY \
  entry_price $ENTRY \
  stop_loss $SL \
  take_profit $TP \
  entry_risk_usdt $ENTRY_RISK \
  unrealized_pnl $UPNL \
  leverage 1.0 \
  source p7b_0p6b_test > /dev/null

echo "  INJECTED quantum:position:ETHUSDT (10 fields)"

$RC HSET quantum:ticker:ETHUSDT \
  symbol ETHUSDT \
  price $MARK \
  markPrice $MARK \
  timestamp $(date +%s%3N) > /dev/null
echo "  TICKER refreshed: markPrice=$MARK"

# ---- 2. Capture baseline audit ID ----
BEFORE=$($RC XREVRANGE quantum:stream:exit.audit + - COUNT 1 | head -1)
echo ""
echo "BASELINE_LAST_AUDIT_ID: $BEFORE"
echo "Service status: $(systemctl is-active quantum-exit-management-agent)"

# ---- 3. Wait for 3 tick cycles (15s) ----
echo ""
echo "Waiting 20s for service tick (loop_sec=5, allow 4 cycles for 0.6b inference)..."
sleep 20

# ---- 4. Capture new audit entries ----
echo ""
echo "=== NEW AUDIT ENTRIES (after baseline) ==="
ENTRIES=$($RC XRANGE quantum:stream:exit.audit "$BEFORE" + 2>/dev/null)
echo "$ENTRIES"

# ---- 5. Filter key fields ----
echo ""
echo "=== PATCH-7B + QWEN3 FIELDS ==="
echo "$ENTRIES" | grep -E "patch|qwen3_action|qwen3_confidence|qwen3_reason|qwen3_fallback|qwen3_latency|formula_action|^action$"

# ---- 6. Service log tail ----
echo ""
echo "=== SERVICE LOG (last 20 lines) ==="
journalctl -u quantum-exit-management-agent --no-pager -n 20 | grep -E "TICK|ETHUSDT|qwen3|Qwen3|scoring|PATCH|ERROR|WARNING"

# ---- 7. Cleanup ----
$RC DEL quantum:position:ETHUSDT > /dev/null
echo ""
echo "CLEANUP: quantum:position:ETHUSDT removed"
echo "=== DONE ==="
