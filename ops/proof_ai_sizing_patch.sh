#!/bin/bash
set -euo pipefail

echo "=== TIME ==="
date

echo ""
echo "=== 1) ACCOUNT SNAPSHOT (TOTAL vs AVAILABLE + OPEN POSITIONS) ==="
set -a
. /etc/quantum/testnet.env
set +a
/opt/quantum/venvs/ai-engine/bin/python3 - <<'PY'
import os
from binance.client import Client

api=os.getenv("BINANCE_TESTNET_API_KEY")
sec=os.getenv("BINANCE_TESTNET_SECRET_KEY")
c=Client(api,sec,testnet=True)
c.FUTURES_URL="https://testnet.binancefuture.com"

acct=c.futures_account()
print("TotalWalletBalance:", acct.get("totalWalletBalance"))
print("AvailableBalance:", acct.get("availableBalance"))
pos=[p for p in acct.get("positions",[]) if float(p.get("positionAmt",0))!=0.0]
print("OpenPositions:", len(pos))
PY

echo ""
echo "=== 2) REDIS: VERIFY ai_* EXISTS IN trade.intent (last 50) ==="
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 50 | \
  grep -E "symbol|confidence|ai_leverage|ai_size_usd|ai_harvest_policy" | head -80

echo ""
echo "=== 3) EXECUTION LOG: prove parsing + source selection ==="
echo "(Looking for src=AI or ai_* in EXEC_INTENT lines)"
tail -n 800 /var/log/quantum/execution.log | \
  grep -E "EXEC_INTENT|src=AI|src=LEGACY|ai_leverage|ai_size_usd|TradeIntent APPROVED|MARGIN" | tail -120

echo ""
echo "=== 4) SERVICE HEALTH ==="
systemctl is-active quantum-execution && systemctl is-active quantum-ai-engine && echo "✅ services active"
