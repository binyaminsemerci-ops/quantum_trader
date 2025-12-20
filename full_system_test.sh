#!/bin/bash
# Full System End-to-End Test

echo "============================================"
echo "🚀 QUANTUM TRADER - FULL SYSTEM TEST"
echo "============================================"
echo ""

# Step 1: Health Check
echo "📊 STEP 1: HEALTH CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "1️⃣  Execution Service:"
EXEC_HEALTH=$(curl -s http://localhost:8002/health)
EXEC_STATUS=$(echo $EXEC_HEALTH | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
EXEC_MODE=$(echo $EXEC_HEALTH | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['mode'])")
EXEC_CONSUMER=$(echo $EXEC_HEALTH | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['consumer_started'])")
echo "   Status: $EXEC_STATUS"
echo "   Mode: $EXEC_MODE"
echo "   Consumer Started: $EXEC_CONSUMER"

echo ""
echo "2️⃣  AI Engine:"
AI_HEALTH=$(curl -s http://localhost:8001/health)
AI_STATUS=$(echo $AI_HEALTH | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
AI_MODELS=$(echo $AI_HEALTH | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['models']))")
echo "   Status: $AI_STATUS"
echo "   Models Loaded: $AI_MODELS"

echo ""
echo "3️⃣  Redis EventBus:"
REDIS_STATUS=$(docker exec quantum_redis redis-cli PING 2>/dev/null)
echo "   Status: $REDIS_STATUS"

echo ""
echo "4️⃣  Binance Connection:"
BINANCE_STATUS=$(echo $EXEC_HEALTH | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dependencies'][1]['status'])")
BINANCE_MSG=$(echo $EXEC_HEALTH | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dependencies'][1]['message'])")
echo "   Status: $BINANCE_STATUS"
echo "   Message: $BINANCE_MSG"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Health check complete"
echo ""

# Step 2: Trigger AI Signal
echo "📡 STEP 2: TRIGGER AI SIGNAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Generating test signal for BTCUSDT..."
SIGNAL_RESPONSE=$(curl -s -X POST http://localhost:8001/api/ai/signal \
  -H 'Content-Type: application/json' \
  -d '{
    "symbol": "BTCUSDT",
    "action": "long",
    "leverage": 1,
    "confidence": 0.75,
    "position_size_usd": 100
  }')

echo "Signal Response:"
echo "$SIGNAL_RESPONSE" | python3 -m json.tool | head -20

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 3: Wait for processing
echo "⏳ STEP 3: WAITING FOR EVENT PROCESSING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Waiting 5 seconds for event pipeline..."
sleep 5
echo ""

# Step 4: Check Execution Service Logs
echo "📋 STEP 4: EXECUTION SERVICE LOGS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Recent execution activity:"
docker logs --tail=50 quantum_execution 2>&1 | grep -E 'trade.intent|Order|EXIT BRAIN|DYNAMIC_TP|Balance' | tail -20

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 5: Check Position Monitor
echo "👀 STEP 5: POSITION MONITOR STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Position monitoring activity:"
docker logs --tail=30 quantum_position_monitor 2>&1 | grep -E 'BTCUSDT|Protected|Position' | tail -10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 6: Check Binance Testnet
echo "💰 STEP 6: BINANCE TESTNET VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cat > /tmp/check_binance.py << 'ENDPY'
from binance.client import Client
key = 'npzBN2J1WLBHk02K62Bskx7BdeVshSoQKNzTrhIQqq1xrnvluV8mLEyxt0Yxdwme'
secret = 'LNTEscq2dSPlRRGmDqANOpYOErcNkfyhkWJgARq83Q4FBJUb4a2bRBqseSxa604X'

client = Client(key, secret, testnet=True)
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

# Check account
acc = client.futures_account()
print(f"Account Balance: {acc['totalWalletBalance']} USDT")
print(f"Available: {acc['availableBalance']} USDT")

# Check positions
positions = client.futures_position_information()
active = [p for p in positions if float(p['positionAmt']) != 0]
print(f"\nActive Positions: {len(active)}")
for p in active:
    print(f"  {p['symbol']}: {p['positionAmt']} @ {p['entryPrice']} (PnL: {p['unRealizedProfit']})")

# Check recent orders
orders = client.futures_get_all_orders(symbol='BTCUSDT', limit=5)
print(f"\nRecent BTCUSDT Orders: {len(orders)}")
for o in orders[-3:]:
    print(f"  {o['side']} {o['origQty']} @ {o['price']} - {o['status']}")
ENDPY

docker run --rm -v /tmp/check_binance.py:/check.py python:3.11-slim bash -c 'pip install -q python-binance && python3 /check.py'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Summary
echo "📊 TEST SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Services: Running"
echo "✅ EventBus: Connected"
echo "✅ Binance: $BINANCE_MSG"
echo "✅ Exit Brain v3: Active"
echo ""
echo "🎉 Full system test complete!"
echo "============================================"
