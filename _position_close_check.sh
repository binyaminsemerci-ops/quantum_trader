#!/bin/bash
echo "=== INTENT EXECUTOR LOGS (last 25 lines) ==="
journalctl -u quantum-intent-executor.service -n 25 --no-pager 2>/dev/null

echo ""
echo "=== apply.result stream — last 10 (full content) ==="
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 2>/dev/null | head -80

echo ""
echo "=== apply.plan stream — last 3 (full details) ==="
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3 2>/dev/null | head -60

echo ""
echo "=== MIN_ORDER_USD in apply-layer config ==="
grep -i 'min_order\|MIN_ORDER' /etc/quantum/apply-layer.env 2>/dev/null
grep -i 'min_order\|MIN_ORDER' /opt/quantum/microservices/apply_layer/main.py 2>/dev/null | head -10

echo ""
echo "=== REAL BINANCE POSITIONS (via python testnet) ==="
source /etc/quantum/testnet.env
python3 /opt/quantum/venvs/ai-client-base/bin/python -c "
import os,sys
sys.path.insert(0,'/opt/quantum')
k=os.environ.get('BINANCE_API_KEY','')
s=os.environ.get('BINANCE_SECRET_KEY','')
print('KEY:',k[:6],'...' if k else 'MISSING')
" 2>/dev/null

/opt/quantum/venvs/ai-client-base/bin/python -c "
import os,sys
sys.path.insert(0,'/opt/quantum')
from dotenv import load_dotenv
load_dotenv('/etc/quantum/testnet.env')
from binance.client import Client
k=os.environ.get('BINANCE_API_KEY','')
s=os.environ.get('BINANCE_SECRET_KEY','')
if not k: print('NO KEY'); exit()
c=Client(k,s,testnet=True)
pos=[p for p in c.futures_position_information() if abs(float(p['positionAmt']))>0]
print('OPEN POSITIONS:',len(pos))
for p in pos:
    notional=abs(float(p['positionAmt']))*float(p['markPrice'])
    print(f'  {p[\"symbol\"]} {\"LONG\" if float(p[\"positionAmt\"])>0 else \"SHORT\"} qty={p[\"positionAmt\"]} notional=\${notional:.2f} upnl={p[\"unRealizedProfit\"]}')
" 2>/dev/null || echo "python failed"

echo ""
echo "=== CHECK if close is blocked by min_order in apply.layer code ==="
grep -n "MIN_ORDER\|min_order\|CLOSE\|close" /opt/quantum/microservices/apply_layer/main.py 2>/dev/null | grep -iE "min_order|MIN_ORDER_USD|close.*skip|skip.*close" | head -10

echo ""
echo "=== Auto permit P33: run it now ==="
/opt/quantum/venvs/ai-client-base/bin/python /opt/quantum/scripts/auto_permit_p33.py 2>&1 | tail -10

echo ""
echo "=== P33 permits after auto run ==="
redis-cli KEYS 'quantum:permit:p33:*' 2>/dev/null | wc -l
redis-cli KEYS 'quantum:permit:p33:*' 2>/dev/null | head -5
