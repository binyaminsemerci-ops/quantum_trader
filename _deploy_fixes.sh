#!/bin/bash
set -e

echo "============================================"
echo " FIX 1 — Exchange Bridge: Force BTCUSDT"
echo "============================================"

# Patch env file to use explicit major symbols instead of Universe service
cat > /etc/quantum/exchange-stream-bridge.env << 'ENVEOF'
PYTHONPATH=/home/qt/quantum_trader
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_URL=redis://127.0.0.1:6379
TZ=Europe/Oslo
LOG_LEVEL=INFO
EXCHANGE_BRIDGE_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,DOTUSDT,LINKUSDT,MATICUSDT,ATOMUSDT,UNIUSDT,LTCUSDT,NEARUSDT,APTUSDT,ARBUSDT,OPUSDT,AAVEUSDT,FETUSDT
EXCHANGE_BRIDGE_MAX_SYMBOLS=20
ENVEOF

echo "[OK] exchange-stream-bridge.env updated with explicit symbols"
echo ""

echo "============================================"
echo " FIX 2 — apply_layer: PARTIAL_25 support"
echo "============================================"

/opt/quantum/venvs/ai-client-base/bin/python3 /tmp/patch_partial25.py
echo ""

echo "============================================"
echo " Restarting services"
echo "============================================"

systemctl restart quantum-exchange-stream-bridge.service
echo "[OK] quantum-exchange-stream-bridge restarted"
sleep 3

systemctl restart quantum-apply-layer.service
echo "[OK] quantum-apply-layer restarted"
sleep 3

echo ""
echo "============================================"
echo " Verification"
echo "============================================"

echo "--- Bridge status ---"
systemctl is-active quantum-exchange-stream-bridge.service

echo "--- Apply-layer status ---"
systemctl is-active quantum-apply-layer.service

echo ""
echo "--- Waiting 15s for exchange.raw to fill ---"
sleep 15

echo ""
echo "--- BTCUSDT entries in exchange.raw (last 500) ---"
redis-cli XREVRANGE quantum:stream:exchange.raw + - COUNT 500 | grep -c "BTCUSDT" || echo "0"

echo ""
echo "--- Symbols in exchange.raw last 100 entries ---"
redis-cli XREVRANGE quantum:stream:exchange.raw + - COUNT 100 | grep "^symbol" -A1 | grep -v "^symbol" | grep -v "^--" | sort | uniq -c | sort -rn | head -15

echo ""
echo "--- apply_layer PARTIAL_25 in source ---"
grep -c "PARTIAL_25" /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "--- apply_layer last 20 log lines ---"
journalctl -u quantum-apply-layer.service -n 20 --no-pager 2>/dev/null | tail -10

echo ""
echo "[DONE] Both fixes deployed"
