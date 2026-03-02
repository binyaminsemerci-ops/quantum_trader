#!/bin/bash
echo "=== EXCHANGE BRIDGE SYMBOLS ==="
journalctl -u quantum-exchange-stream-bridge.service -n 100 --no-pager 2>/dev/null | grep -iE "symbol|btc|subscribe|websocket|loaded|universe" | head -30

echo ""
echo "=== EXCHANGE BRIDGE ENV ==="
cat /etc/quantum/exchange-stream-bridge.env 2>/dev/null || echo "NO ENV FILE"

echo ""
echo "=== UNIVERSE SERVICE KEYS ==="
redis-cli KEYS 'quantum:universe:*' 2>/dev/null | head -20

echo ""
echo "=== UNIVERSE SYMBOLS (first 30) ==="
redis-cli SMEMBERS quantum:universe:symbols 2>/dev/null | head -30

echo ""
echo "=== APPLY_LAYER KELLY SEARCH ==="
grep -n "kelly_200usdt\|kelly.*200\|200usdt\|kelly_floor\|min_notional\|SKIP.*kelly\|kelly.*SKIP\|reason_codes.*kelly\|kelly.*reason" /opt/quantum/microservices/apply_layer/main.py 2>/dev/null | head -30

echo ""
echo "=== APPLY_LAYER DECISION=SKIP SEARCH ==="
grep -n "decision.*SKIP\|SKIP.*decision\|\"SKIP\"\|= .SKIP." /opt/quantum/microservices/apply_layer/main.py 2>/dev/null | head -30

echo ""
echo "=== APPLY_LAYER FILE SIZE ==="
wc -l /opt/quantum/microservices/apply_layer/main.py 2>/dev/null

echo ""
echo "=== BTCUSDT IN EXCHANGE.RAW (last 1000) ==="
redis-cli XREVRANGE quantum:stream:exchange.raw + - COUNT 1000 2>/dev/null | grep -c "BTCUSDT" || echo "0"

echo ""
echo "=== WHAT SYMBOLS IN EXCHANGE.RAW (last 200) ==="
redis-cli XREVRANGE quantum:stream:exchange.raw + - COUNT 200 2>/dev/null | grep -oE '"symbol","[A-Z]+"' | sort | uniq -c | sort -rn | head -20
