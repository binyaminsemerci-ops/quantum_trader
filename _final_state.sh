#!/bin/bash
echo "============ FINAL STATE ============"

echo ""
echo "[FIX 1] BTCUSDT in exchange.raw:"
redis-cli XREVRANGE quantum:stream:exchange.raw + - COUNT 500 2>/dev/null | grep -c "BTCUSDT"

echo ""
echo "[FIX 1] Meta-regime BTC samples:"
journalctl -u quantum-meta-regime.service -n 3 --no-pager 2>/dev/null | grep -oE '"found": [0-9]+|"required": [0-9]+' | tail -4

echo ""
echo "[FIX 1] Meta-regime current value:"
redis-cli GET quantum:meta:regime

echo ""
echo "[FIX 2] PARTIAL_25 all 5 locations:"
grep -n "PARTIAL_25" /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "[FIX 2] Step builder at line 1393-1405:"
sed -n '1393,1405p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "[SERVICES]"
for svc in quantum-exchange-stream-bridge quantum-apply-layer quantum-meta-regime quantum-intent-executor; do
    echo "  $svc: $(systemctl is-active $svc)"
done
