#!/usr/bin/bash
# PROOF PACK: BRIDGE-PATCH GO-LIVE STATUS
# ========================================

echo "======================================================================"
echo "BRIDGE-PATCH PROOF PACK - $(date)"
echo "======================================================================"
echo ""

echo "1. AI ENGINE: AI Sizer Activity (last 2 min)"
echo "----------------------------------------------------------------------"
ai_count=$(journalctl -u quantum-ai-engine --since '2 minutes ago' | grep -c 'BRIDGE-PATCH.*Entering')
echo "AI Sizer invocations: $ai_count"
journalctl -u quantum-ai-engine --since '2 minutes ago' | grep -E 'ai_leverage|ai_size_usd' | head -3
echo ""

echo "2. LATEST TRADE.INTENT with AI Fields"
echo "----------------------------------------------------------------------"
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 5 | grep -B2 "ai_leverage" | head -15
echo ""

echo "3. EXECUTION SERVICE: Status & Recent Activity"
echo "----------------------------------------------------------------------"
echo "Service: $(systemctl is-active quantum-execution)"
echo "Testnet Balance: $(python3 -c "from binance.client import Client; import os; os.environ['BINANCE_TESTNET_API_KEY']='your_binance_testnet_api_key_here'; os.environ['BINANCE_TESTNET_SECRET_KEY']='your_binance_testnet_api_secret_here'; c=Client(os.environ['BINANCE_TESTNET_API_KEY'],os.environ['BINANCE_TESTNET_SECRET_KEY'],testnet=True); c.FUTURES_URL='https://testnet.binancefuture.com'; print(c.futures_account()['availableBalance'])" 2>/dev/null) USDT"
echo ""

echo "4. RISK GOVERNOR: Recent Activity"
echo "----------------------------------------------------------------------"
tail -100 /var/log/quantum/execution.log | grep GOVERNOR | tail -10
echo ""

echo "5. EXECUTION: Accepted Orders (last 50 lines)"
echo "----------------------------------------------------------------------"
tail -100 /var/log/quantum/execution.log | grep -E "EXEC_INTENT|ACCEPT" | tail -10
echo ""

echo "6. BINANCE: Current Open Positions"
echo "----------------------------------------------------------------------"
/opt/quantum/venvs/ai-engine/bin/python3 << 'PYEOF'
import os
os.environ["BINANCE_TESTNET_API_KEY"] = "your_binance_testnet_api_key_here"
os.environ["BINANCE_TESTNET_SECRET_KEY"] = "your_binance_testnet_api_secret_here"
from binance.client import Client
try:
    client = Client(os.environ["BINANCE_TESTNET_API_KEY"], os.environ["BINANCE_TESTNET_SECRET_KEY"], testnet=True)
    client.FUTURES_URL = "https://testnet.binancefuture.com"
    positions = client.futures_position_information()
    open_pos = [p for p in positions if float(p['positionAmt']) != 0]
    print(f"Open positions: {len(open_pos)}")
    for p in open_pos[:10]:
        print(f"  {p['symbol']}: {p['positionAmt']} @ {p['entryPrice']} (unrealizedProfit: {p['unRealizedProfit']})")
except Exception as e:
    print(f"Error: {e}")
PYEOF
echo ""

echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
echo "AI Sizer: $ai_count invocations in last 2 min"
echo "Execution logs: $(tail -100 /var/log/quantum/execution.log | grep -c 'GOVERNOR') GOVERNOR lines"
echo "Stream length: $(redis-cli XLEN quantum:stream:trade.intent)"
echo ""
echo "✅ AI Engine: $(systemctl is-active quantum-ai-engine)"
echo "✅ Execution: $(systemctl is-active quantum-execution)"
echo "======================================================================"

