#!/bin/bash
echo "=== quantum:market:BTCUSDT:binance_main ==="
redis-cli hgetall quantum:market:BTCUSDT:binance_main 2>/dev/null | head -30

echo ""
echo "=== quantum:market:APTUSDT:binance_main ==="
redis-cli hgetall quantum:market:APTUSDT:binance_main 2>/dev/null | head -30

echo ""
echo "=== layer1 data sink LAST LOGS ==="
journalctl -u quantum-layer1-data-sink -n 20 --no-pager

echo ""
echo "=== market publisher LAST LOGS ==="
journalctl -u quantum-market-publisher -n 15 --no-pager

echo ""
echo "=== quantum:position:APTUSDT full ==="
redis-cli hgetall quantum:position:APTUSDT

echo ""
echo "=== WHO WRITES atr_value to positions (grep in all microservices) ==="
grep -rn "atr_value\|HSET.*atr" /opt/quantum/microservices/ 2>/dev/null \
  | grep -v ".pyc" | grep -v "position_provider\|harvest_v2\|layer2" | head -40

echo ""
echo "=== OHLCV zset sample APTUSDT ==="
redis-cli zrange quantum:history:ohlcv:APTUSDT:1m -5 -1 WITHSCORES | head -20
