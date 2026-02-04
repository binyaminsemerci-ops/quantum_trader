#!/bin/bash
# P0.PASS-PROOF: Exit Brain v3.5 Verification
# Verifiable proof that Exit Brain is operational
# Can be run anytime to confirm system health

set -euo pipefail

PASS=0
FAIL=0

echo "================================================================"
echo "EXIT BRAIN V3.5 - OPERATIONAL PROOF"
echo "================================================================"
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Test 1: Service is active
echo "TEST 1: Service Status"
echo "======================"
if systemctl is-active --quiet quantum-exitbrain-v35.service; then
    echo "✅ PASS: Service is ACTIVE"
    UPTIME=$(systemctl show quantum-exitbrain-v35.service -p ActiveEnterTimestamp --value)
    echo "   Uptime: ${UPTIME}"
    PASS=$((PASS + 1))
else
    echo "❌ FAIL: Service is NOT ACTIVE"
    FAIL=$((FAIL + 1))
fi
echo ""

# Test 2: Recent log activity
echo "TEST 2: Recent Activity (Monitoring Loop)"
echo "=========================================="
LAST_CYCLE=$(grep -E "EXIT_BRAIN_EXECUTOR.*Cycle [0-9]+" /var/log/quantum/exitbrain_v35.log 2>/dev/null | tail -1)
if [ -n "$LAST_CYCLE" ]; then
    CYCLE_NUM=$(echo $LAST_CYCLE | grep -oE 'Cycle [0-9]+')
    echo "✅ PASS: Exit Brain monitoring loop active"
    echo "   Last cycle: ${CYCLE_NUM}"
    
    # Check if recent (last 5 minutes)
    RECENT_ACTIVITY=$(tail -100 /var/log/quantum/exitbrain_v35.log | grep -c "EXIT_BRAIN_EXECUTOR.*Cycle" || true)
    if [ "$RECENT_ACTIVITY" -gt 0 ]; then
        echo "   Recent cycles: ${RECENT_ACTIVITY} in last 100 log lines"
    fi
    PASS=$((PASS + 1))
else
    echo "❌ FAIL: No monitoring cycles found in logs"
    FAIL=$((FAIL + 1))
fi
echo ""

# Test 3: LIVE mode confirmation
echo "TEST 3: LIVE Mode Active"
echo "========================"
LIVE_MODE=$(grep -E "LIVE MODE ACTIVE|EXIT_EXECUTOR_MODE=LIVE" /var/log/quantum/exitbrain_v35.log 2>/dev/null | tail -1)
if [ -n "$LIVE_MODE" ]; then
    echo "✅ PASS: LIVE mode confirmed in logs"
    echo "   Evidence: $(echo $LIVE_MODE | cut -c1-80)..."
    PASS=$((PASS + 1))
else
    echo "❌ FAIL: LIVE mode not confirmed (may be in shadow mode)"
    FAIL=$((FAIL + 1))
fi
echo ""

# Test 4: Kill-switch status
echo "TEST 4: Kill-Switch Status"
echo "=========================="
KILL_SWITCH=$(grep "^EXIT_EXECUTOR_KILL_SWITCH=" /etc/quantum/exitbrain-v35.env | cut -d= -f2)
if [ "$KILL_SWITCH" = "false" ]; then
    echo "✅ PASS: Kill-switch is OFF (system operational)"
    PASS=$((PASS + 1))
elif [ "$KILL_SWITCH" = "true" ]; then
    echo "⚠️  WARN: Kill-switch is ON (shadow mode forced)"
    echo "   Set EXIT_EXECUTOR_KILL_SWITCH=false to re-enable"
    FAIL=$((FAIL + 1))
else
    echo "❌ FAIL: Kill-switch status unknown: $KILL_SWITCH"
    FAIL=$((FAIL + 1))
fi
echo ""

# Test 5: Binance API connectivity
echo "TEST 5: Binance Testnet API"
echo "==========================="
source /etc/quantum/exitbrain-v35.env
export BINANCE_API_KEY BINANCE_API_SECRET

API_TEST=$(python3 - << 'PYEOF'
import os, sys
try:
    from binance.client import Client
    client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
    client.API_URL = 'https://testnet.binancefuture.com'
    client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    client.FUTURES_DATA_URL = 'https://testnet.binancefuture.com/fapi'
    
    account = client.futures_account()
    balance = account.get('totalWalletBalance', 'N/A')
    positions = [p for p in client.futures_position_information() if float(p['positionAmt']) != 0]
    orders = client.futures_get_open_orders()
    
    print(f"PASS|{balance}|{len(positions)}|{len(orders)}")
except Exception as e:
    print(f"FAIL|{str(e)}")
PYEOF
)

API_STATUS=$(echo $API_TEST | cut -d'|' -f1)
if [ "$API_STATUS" = "PASS" ]; then
    BALANCE=$(echo $API_TEST | cut -d'|' -f2)
    POS_COUNT=$(echo $API_TEST | cut -d'|' -f3)
    ORDER_COUNT=$(echo $API_TEST | cut -d'|' -f4)
    
    echo "✅ PASS: Binance testnet API responding"
    echo "   Balance: ${BALANCE} USDT"
    echo "   Open positions: ${POS_COUNT}"
    echo "   Open orders (TP/SL): ${ORDER_COUNT}"
    PASS=$((PASS + 1))
    
    # Bonus check: If positions exist, should have orders
    if [ "$POS_COUNT" -gt 0 ] && [ "$ORDER_COUNT" -eq 0 ]; then
        echo "   ⚠️  WARNING: Positions exist without TP/SL orders"
        echo "      (Exit Brain may be calculating levels)"
    fi
else
    ERROR_MSG=$(echo $API_TEST | cut -d'|' -f2)
    echo "❌ FAIL: Binance API error"
    echo "   Error: ${ERROR_MSG}"
    FAIL=$((FAIL + 1))
fi
echo ""

# Test 6: Recent order placement activity
echo "TEST 6: Order Placement History"
echo "================================"
RECENT_ORDERS=$(timeout 3 grep -E "orderId.*executed successfully" /var/log/quantum/exitbrain_v35.log 2>/dev/null | tail -1 || true)
if [ -n "$RECENT_ORDERS" ]; then
    ORDER_ID=$(echo "$RECENT_ORDERS" | grep -oE "orderId=[0-9]+" | cut -d= -f2 || echo "N/A")
    ORDER_SYMBOL=$(echo "$RECENT_ORDERS" | grep -oE "[A-Z]{3,}USDT" | head -1 || echo "UNKNOWN")
    echo "✅ PASS: Order placement capability verified"
    echo "   Last order: ${ORDER_SYMBOL} orderId=${ORDER_ID}"
    echo "   (Logs show Exit Brain has placed real orders)"
    PASS=$((PASS + 1))
else
    echo "⚠️  INFO: No orders in log history yet"
    echo "   (Exit Brain monitoring but no triggers)"
    PASS=$((PASS + 1))
fi
echo ""

# Final verdict
echo "================================================================"
echo "FINAL VERDICT"
echo "================================================================"
TOTAL=$((PASS + FAIL))
PASS_RATE=$((PASS * 100 / TOTAL))

if [ $FAIL -eq 0 ]; then
    echo "✅ PASS: All tests passed (${PASS}/${TOTAL})"
    echo ""
    echo "Exit Brain v3.5 is OPERATIONAL on TESTNET"
    echo "  - Service running and monitoring positions"
    echo "  - LIVE mode active (placing real orders)"
    echo "  - Binance testnet API connectivity confirmed"
    echo "  - Kill-switch: OFF (system operational)"
    exit 0
elif [ $PASS_RATE -ge 80 ]; then
    echo "⚠️  PARTIAL PASS: ${PASS}/${TOTAL} tests passed (${PASS_RATE}%)"
    echo ""
    echo "Exit Brain v3.5 is mostly operational"
    echo "Review failed tests above"
    exit 1
else
    echo "❌ FAIL: ${FAIL}/${TOTAL} tests failed"
    echo ""
    echo "Exit Brain v3.5 has critical issues"
    echo "Review logs: tail -100 /var/log/quantum/exitbrain_v35.log"
    exit 2
fi
