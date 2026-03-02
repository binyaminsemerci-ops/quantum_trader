#!/bin/bash
echo "=== PERMIT STATUS ==="
redis-cli TTL quantum:permit:governor
redis-cli TTL quantum:permit:p33
redis-cli TTL quantum:permit:p26
redis-cli GET quantum:permit:governor
redis-cli GET quantum:permit:p33

echo "=== PERMIT SERVICES (active) ==="
systemctl list-units 'quantum-*.service' --state=active --no-pager 2>/dev/null | grep -iE 'permit|p33|p26|governor|regime|feature'

echo "=== PERMIT SERVICES (inactive) ==="
systemctl list-units 'quantum-*.service' --state=inactive --no-pager 2>/dev/null | grep -iE 'permit|p33|p26|governor|regime|feature'

echo "=== FEATURE PUBLISHER STATUS ==="
systemctl status quantum-feature-publisher.service --no-pager -l 2>/dev/null | head -25

echo "=== P33 SERVICE STATUS ==="
systemctl status quantum-p33*service --no-pager -l 2>/dev/null | head -20

echo "=== Who sets permit:governor ==="
grep -r 'permit:governor' /opt/quantum --include="*.py" -l 2>/dev/null | head -10
grep -r 'permit:p33' /opt/quantum --include="*.py" -l 2>/dev/null | head -10

echo "=== SIGNAL STREAM LATEST ==="
redis-cli XREVRANGE quantum:stream:signal.score + - COUNT 2 2>/dev/null

echo "=== FEATURES STREAM ==="
redis-cli XLEN quantum:stream:features.BTCUSDT 2>/dev/null
redis-cli XLEN quantum:stream:features.ETHUSDT 2>/dev/null
redis-cli XREVRANGE quantum:stream:features.BTCUSDT + - COUNT 1 2>/dev/null

echo "=== CURRENT REAL POSITIONS (Binance) ==="
source /etc/quantum/testnet.env 2>/dev/null || source /etc/quantum/apply-layer.env 2>/dev/null
python3 -c "
import os,sys
sys.path.insert(0,'/opt/quantum')
from binance.client import Client
k=os.environ.get('BINANCE_API_KEY','')
s=os.environ.get('BINANCE_SECRET_KEY','')
if not k:
    print('NO KEY')
    exit(0)
c=Client(k,s,testnet=True)
pos=[p for p in c.futures_position_information() if abs(float(p['positionAmt']))>0]
for p in pos:
    print(p['symbol'],p['positionSide'],p['positionAmt'],p['unRealizedProfit'])
if not pos:
    print('NO OPEN POSITIONS')
" 2>/dev/null || echo "python fetch failed"
