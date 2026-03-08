#!/usr/bin/env bash
# PATCH-7B shadow validation: inject a controlled non-emergency position into Redis
# Parameters chosen so:
#   - R_net = (1950.49 - 2000.00) * 0.01 / 5.0 = -0.0990  [well above -1.5 emergency threshold]
#   - SL=1800  mark=1950.49 > SL → not breached for LONG
#   - TP=2400  not reached
# The formula engine will score this as HOLD or PARTIAL_CLOSE_25
# Qwen3 will be called (ai mode, not TIGHTEN_TRAIL/MOVE_TO_BREAKEVEN)

MARK=$(redis-cli HGET quantum:ticker:ETHUSDT markPrice)
ENTRY="2000.00"
QTY="0.01"
SL="1800.00"
TP="2400.00"
RISK="5.00"
NOW=$(date +%s)

# Compute unrealized PNL
UPNL=$(python3 -c "print(round(float('$QTY') * (float('$MARK') - float('$ENTRY')), 4))")
R_NET=$(python3 -c "print(round(float('$UPNL') / float('$RISK'), 4))")

echo "=== P7B POSITION INJECTION ==="
echo "mark_price  : $MARK"
echo "entry_price : $ENTRY"
echo "quantity    : $QTY  LONG"
echo "stop_loss   : $SL   (mark > SL → NOT breached)"
echo "unrealized  : $UPNL USDT"
echo "R_net       : $R_NET  (threshold: -1.5 → SAFE)"
echo ""

redis-cli HSET quantum:position:ETHUSDT \
    side LONG \
    quantity "$QTY" \
    entry_price "$ENTRY" \
    unrealized_pnl "$UPNL" \
    leverage 1.0 \
    stop_loss "$SL" \
    take_profit "$TP" \
    entry_risk_usdt "$RISK" \
    sync_timestamp "$NOW" \
    source "p7b_shadow_test"

echo "INJECTED: quantum:position:ETHUSDT"
echo "Verify: $(redis-cli EXISTS quantum:position:ETHUSDT) key(s) in Redis"

# Also ensure ticker is fresh (update timestamp)
redis-cli HSET quantum:ticker:ETHUSDT markPrice "$MARK" price "$MARK" timestamp "$(date +%s%3N)" symbol ETHUSDT
echo "TICKER refreshed: quantum:ticker:ETHUSDT markPrice=$MARK"
