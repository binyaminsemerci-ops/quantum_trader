#!/bin/bash
echo "=== P3.5 GUARD FULL SECTION (lines 788-840) ==="
sed -n '788,845p' /opt/quantum/microservices/intent_executor/main.py 2>/dev/null

echo ""
echo "=== Where kelly_200usdt reason is set ==="
grep -n "kelly_200usdt\|kelly_200\|KELLY.*200\|200usdt" \
    /opt/quantum/microservices/intent_executor/main.py 2>/dev/null

echo ""
echo "=== Exchange stream bridge env ==="
cat /etc/quantum/exchange-stream-bridge.env 2>/dev/null

echo ""
echo "=== Exchange stream bridge source (symbols config) ==="
grep -n "BTCUSDT\|symbol\|SYMBOL\|SYMBOLS\|pairs\|PAIRS\|subscribe" \
    /home/qt/quantum_trader/microservices/data_collector/exchange_stream_bridge.py 2>/dev/null | head -30

echo ""
echo "=== Exchange stream bridge: is it actually running? ==="
systemctl status quantum-exchange-stream-bridge.service --no-pager -l 2>/dev/null | head -20

echo ""
echo "=== ADA: trigger P3.3 permit now and watch ==="
/opt/quantum/venvs/ai-client-base/bin/python /opt/quantum/scripts/auto_permit_p33.py 2>&1 | grep -i ada
sleep 10
journalctl -u quantum-intent-executor.service --since '1 minute ago' --no-pager 2>/dev/null | grep -E "ADA|ada|execut.*True|FILLED|ERROR" | tail -10
