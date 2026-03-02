#!/bin/bash
# Verification script v2 - 2026-03-02
echo "=== CURRENT POSITIONS (what harvest_v2 tracks) ==="
redis-cli KEYS "quantum:position:*" | grep -v snapshot | grep -v ledger | sort

echo ""
echo "=== BNBUSDT position data ==="
redis-cli HGETALL quantum:position:BNBUSDT

echo ""
echo "=== XRPUSDT check (was SHORT 73.9) ==="
redis-cli HGETALL quantum:position:XRPUSDT

echo ""
echo "=== LINKUSDT position ==="
redis-cli HGETALL quantum:position:LINKUSDT

echo ""
echo "=== Query Binance testnet for current positions ==="
source /etc/quantum/testnet.env
python3 -c "
import requests, hmac, hashlib, time
api_key = '$BINANCE_TESTNET_API_KEY'
secret = '$BINANCE_TESTNET_SECRET_KEY'
if not api_key or not secret:
    import os
    api_key = os.environ.get('BINANCE_TESTNET_API_KEY', '')
    secret = os.environ.get('BINANCE_TESTNET_API_SECRET', os.environ.get('BINANCE_TESTNET_SECRET_KEY', ''))
ts = int(time.time()*1000)
msg = f'timestamp={ts}'
sig = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
r = requests.get(
    'https://testnet.binancefuture.com/fapi/v2/positionRisk',
    headers={'X-MBX-APIKEY': api_key},
    params={'timestamp': ts, 'signature': sig},
    timeout=10
)
positions = [p for p in r.json() if abs(float(p.get('positionAmt', 0))) > 0]
if positions:
    for p in positions:
        pamt = float(p['positionAmt'])
        upnl = float(p['unRealizedProfit'])
        entry = float(p['entryPrice'])
        lev = p['leverage']
        notional = abs(pamt) * entry
        print(f'  {p[\"symbol\"]}: amt={pamt:.4f} entry={entry:.4f} upnl={upnl:.4f} notional={notional:.1f} USDT lev={lev}x')
else:
    print('  NO OPEN POSITIONS ON BINANCE TESTNET')
" 2>&1

echo ""
echo "=== Kelly sizing verification (3 samples) ==="
redis-cli HMGET quantum:layer4:sizing:BNBUSDT size_usdt recommendation
redis-cli HMGET quantum:layer4:sizing:XRPUSDT size_usdt recommendation
redis-cli HMGET quantum:layer4:sizing:BTCUSDT size_usdt recommendation

echo ""
echo "=== apply-layer.env critical lines ==="
grep -E "SYMBOLS=|RISK_DAILY|RISK_MAX|K_OPEN|MIN_POSITION|MAX_POSITION" /etc/quantum/apply-layer.env

echo ""
echo "=== Services status ==="
for svc in quantum-apply-layer quantum-harvest-v2 quantum-intent-executor quantum-ensemble-predictor; do
    status=$(systemctl is-active $svc 2>/dev/null || echo "unknown")
    echo "  $svc: $status"
done

echo ""
echo "=== harvest_v2 shadow stream (last 2 since config change) ==="
redis-cli XREVRANGE quantum:stream:harvest.v2.shadow + - COUNT 2
