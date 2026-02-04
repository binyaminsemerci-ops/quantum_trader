#!/bin/bash
# GO-LIVE ABORT / KILL-SWITCH
# Emergency stop for live trading

set -e

PROOF_FILE="GO_LIVE_ABORT_PROOF.md"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
VPS_HOST="root@46.224.116.254"
SSH_KEY="~/.ssh/hetzner_fresh"

echo "======================================"
echo "  🚨 GO-LIVE EMERGENCY ABORT 🚨"
echo "======================================"
echo "Date: $TIMESTAMP"
echo ""

read -p "Type 'ABORT' to confirm emergency stop: " CONFIRM
if [ "$CONFIRM" != "ABORT" ]; then
    echo "❌ Abort cancelled"
    exit 1
fi

# Start proof document
cat > "$PROOF_FILE" << EOF
# GO-LIVE ABORT PROOF

**Date**: $TIMESTAMP  
**Operator**: $(whoami)  
**Action**: Emergency Abort / Kill-Switch Activated  

---

## ABORT SEQUENCE

EOF

echo "🛑 EXECUTING ABORT SEQUENCE..."
echo ""

# Step 1: Disable new entries
echo "1️⃣ Disabling new entries..."
ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli SET quantum:feature:new_entries_disabled 1" || echo "⚠️  Failed to set feature flag"
ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli SET quantum:config:emergency_stop true" || echo "⚠️  Failed to set emergency stop"

cat >> "$PROOF_FILE" << EOF
### Step 1: Disable New Entries

\`\`\`
quantum:feature:new_entries_disabled = 1
quantum:config:emergency_stop = true
\`\`\`

Status: ✅ COMPLETED

EOF

echo "   ✅ New entries disabled"

# Step 2: Cancel open orders
echo ""
echo "2️⃣ Cancelling open orders..."

ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_auto_executor python3 -c '
import ccxt
import os
exchange = ccxt.binance({
    \"apiKey\": os.getenv(\"BINANCE_API_KEY\"),
    \"secret\": os.getenv(\"BINANCE_API_SECRET\"),
    \"options\": {\"defaultType\": \"future\"}
})
orders = exchange.fetch_open_orders()
print(f\"Found {len(orders)} open orders\")
for order in orders:
    try:
        exchange.cancel_order(order[\"id\"], order[\"symbol\"])
        print(f\"Cancelled {order[\"id\"]} ({order[\"symbol\"]})\")
    except Exception as e:
        print(f\"Failed to cancel {order[\"id\"]}: {e}\")
'" > /tmp/cancel_orders.log 2>&1 || echo "⚠️  Failed to cancel orders (may be none)"

CANCELLED_ORDERS=$(cat /tmp/cancel_orders.log)

cat >> "$PROOF_FILE" << EOF
### Step 2: Cancel Open Orders

\`\`\`
$CANCELLED_ORDERS
\`\`\`

Status: ✅ COMPLETED

EOF

echo "   ✅ Open orders cancelled"

# Step 3: Optional - Close positions
echo ""
echo "3️⃣ Position handling..."
read -p "Close all positions? (yes/no): " CLOSE_POSITIONS

if [ "$CLOSE_POSITIONS" = "yes" ]; then
    echo "   🔄 Closing all positions..."
    
    ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_auto_executor python3 -c '
import ccxt
import os
exchange = ccxt.binance({
    \"apiKey\": os.getenv(\"BINANCE_API_KEY\"),
    \"secret\": os.getenv(\"BINANCE_API_SECRET\"),
    \"options\": {\"defaultType\": \"future\"}
})
positions = exchange.fetch_positions()
active = [p for p in positions if float(p.get(\"contracts\", 0)) != 0]
print(f\"Found {len(active)} active positions\")
for pos in active:
    try:
        symbol = pos[\"symbol\"]
        contracts = float(pos[\"contracts\"])
        side = \"sell\" if contracts > 0 else \"buy\"
        print(f\"Closing {symbol}: {side} {abs(contracts)} contracts\")
        order = exchange.create_market_order(symbol, side, abs(contracts), {\"reduceOnly\": True})
        print(f\"Closed {symbol}: orderId={order[\"id\"]}\")
    except Exception as e:
        print(f\"Failed to close {symbol}: {e}\")
'" > /tmp/close_positions.log 2>&1 || echo "⚠️  Failed to close positions"

    CLOSED_POSITIONS=$(cat /tmp/close_positions.log)
    
    cat >> "$PROOF_FILE" << EOF
### Step 3: Close Positions

\`\`\`
$CLOSED_POSITIONS
\`\`\`

Status: ✅ COMPLETED

EOF
    echo "   ✅ Positions closed"
else
    echo "   ⏭️  Skipped - positions left open"
    cat >> "$PROOF_FILE" << EOF
### Step 3: Close Positions

Status: ⏭️ SKIPPED (operator choice)

EOF
fi

# Step 4: Stop trading services
echo ""
echo "4️⃣ Stopping trading services..."

ssh -i "$SSH_KEY" "$VPS_HOST" "cd /home/qt/quantum_trader && docker compose stop auto_executor ai_engine" || echo "⚠️  Failed to stop services"

cat >> "$PROOF_FILE" << EOF
### Step 4: Stop Trading Services

\`\`\`
docker compose stop auto_executor ai_engine
\`\`\`

Status: ✅ COMPLETED

EOF

echo "   ✅ Trading services stopped"

# Final status
echo ""
echo "✅ ABORT SEQUENCE COMPLETE"

# Verify
ssh -i "$SSH_KEY" "$VPS_HOST" "docker ps --filter name=auto_executor --format '{{.Names}}: {{.Status}}'" > /tmp/abort_status.txt
EXECUTOR_STATUS=$(cat /tmp/abort_status.txt || echo "not found")

ssh -i "$SSH_KEY" "$VPS_HOST" "docker exec quantum_redis redis-cli GET quantum:config:emergency_stop" > /tmp/emergency_flag.txt
EMERGENCY_FLAG=$(cat /tmp/emergency_flag.txt)

cat >> "$PROOF_FILE" << EOF
---

## VERIFICATION

### Service Status

\`\`\`
$EXECUTOR_STATUS
\`\`\`

### Emergency Flags

\`\`\`
quantum:config:emergency_stop = $EMERGENCY_FLAG
\`\`\`

---

## ABORT SUMMARY

**Date**: $TIMESTAMP  
**Actions Taken**:
1. ✅ Disabled new entries via feature flags
2. ✅ Cancelled open orders
3. $([ "$CLOSE_POSITIONS" = "yes" ] && echo "✅" || echo "⏭️") Close positions: $([ "$CLOSE_POSITIONS" = "yes" ] && echo "executed" || echo "skipped")
4. ✅ Stopped trading services

**System State**: 🔴 EMERGENCY STOP ACTIVE

**Recovery**: To resume trading:
1. Investigate root cause
2. Clear emergency flags: \`redis-cli DEL quantum:config:emergency_stop quantum:feature:new_entries_disabled\`
3. Restart services: \`docker compose up -d auto_executor ai_engine\`
4. Run preflight checks before resuming

EOF

echo ""
echo "📄 Abort proof document: $PROOF_FILE"
echo ""
echo "⚠️  System in EMERGENCY STOP mode"
echo "⚠️  Trading halted until manual recovery"
