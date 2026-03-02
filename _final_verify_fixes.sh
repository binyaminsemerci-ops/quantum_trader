#!/bin/bash
echo "=== [FIX 1] BTCUSDT in exchange.raw (last 500) ==="
redis-cli XREVRANGE quantum:stream:exchange.raw + - COUNT 500 2>/dev/null | grep -c "BTCUSDT"

echo ""
echo "=== [FIX 1] Top symbols in exchange.raw last 200 ==="
redis-cli XREVRANGE quantum:stream:exchange.raw + - COUNT 200 2>/dev/null | grep -E "^symbol$" -A1 | grep -v "^symbol$" | grep -v "^--" | sort | uniq -c | sort -rn | head -10

echo ""
echo "=== [FIX 1] Meta-regime status ==="
redis-cli GET quantum:meta:regime 2>/dev/null
journalctl -u quantum-meta-regime.service -n 5 --no-pager 2>/dev/null | tail -5

echo ""
echo "=== [FIX 1] Signal risk_context samples (last 5 signals) ==="
redis-cli XREVRANGE quantum:stream:signals + - COUNT 5 2>/dev/null | grep "risk_context"

echo ""
echo "=== [FIX 2] PARTIAL_25 in apply_layer (count) ==="
grep -c "PARTIAL_25" /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== [FIX 2] PARTIAL_25 lines in apply_layer ==="
grep -n "PARTIAL_25" /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== Service statuses ==="
for svc in quantum-exchange-stream-bridge quantum-apply-layer quantum-meta-regime quantum-intent-executor; do
    echo "$svc: $(systemctl is-active $svc)"
done

echo ""
echo "=== Exchange bridge env ==="
grep "EXCHANGE_BRIDGE_SYMBOLS" /etc/quantum/exchange-stream-bridge.env
