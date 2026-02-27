#!/bin/bash
echo "=== SAMPLE POSITION LEDGER (ADAUSDT) ==="
redis-cli hgetall quantum:position:ledger:ADAUSDT

echo ""
echo "=== SAMPLE POSITION LEDGER (SUIUSDT) ==="
redis-cli hgetall quantum:position:ledger:SUIUSDT

echo ""
echo "=== SAMPLE POSITION LEDGER (ACEUSDT) ==="
redis-cli hgetall quantum:position:ledger:ACEUSDT

echo ""
echo "=== POSITION PROVIDER SOURCE ==="
cat /opt/quantum/microservices/harvest_v2/feeds/position_provider.py

echo ""
echo "=== ATR PROVIDER SOURCE ==="
cat /opt/quantum/microservices/harvest_v2/feeds/atr_provider.py

echo ""
echo "=== LAYER1 DATA SINK LOGS ==="
journalctl -u quantum-layer1-data-sink -n 10 --no-pager

echo ""
echo "=== MARKET PUBLISHER LOGS ==="
journalctl -u quantum-market-publisher -n 10 --no-pager
