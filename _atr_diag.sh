#!/bin/bash
echo "=== harvest-v2 ATR / market data source code ==="
grep -n "atr\|market\|OHLCV\|kline\|data_sink\|market.data\|quantum:market\|quantum:history" \
  /opt/quantum/microservices/harvest_v2/harvest_v2.py \
  /opt/quantum/microservices/harvest_v2/engine/*.py 2>/dev/null | head -60

echo ""
echo "=== harvest-v2 config.py ==="
cat /opt/quantum/microservices/harvest_v2/engine/config.py 2>/dev/null

echo ""
echo "=== harvest-v2 feeds dir ==="
ls -la /opt/quantum/microservices/harvest_v2/feeds/ 2>/dev/null || echo "no feeds dir"

echo ""
echo "=== position feed source ==="
cat /opt/quantum/microservices/harvest_v2/feeds/position.py 2>/dev/null || \
  grep -rn "atr\|ATR\|market_data\|data_sink" /opt/quantum/microservices/harvest_v2/ 2>/dev/null | grep -v ".pyc" | head -40

echo ""
echo "=== quantum:market keys ==="
redis-cli keys "quantum:market:*" | head -20
redis-cli hgetall quantum:market:BTCUSDT 2>/dev/null | head -20

echo ""
echo "=== quantum:history keys ==="
redis-cli keys "quantum:history:ohlcv:*" | head -10
redis-cli hgetall "$(redis-cli keys 'quantum:history:ohlcv:*' | head -1)" 2>/dev/null | head -10

echo ""
echo "=== layer1 data sink service ==="
systemctl list-units --type=service --all | grep -iE "layer1|data.sink|market.data|ohlcv|kline" | head -10
