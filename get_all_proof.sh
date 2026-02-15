#!/bin/bash
echo "========================================"
echo "QUANTUM TRADER SYSTEM PROOF"
echo "Generated: $(date)"
echo "========================================"

echo ""
echo "=== 1. AKTIVE POSISJONER (Redis Ledger) ==="
redis-cli KEYS "quantum:position:ledger:*"

echo ""
echo "=== 2. AUTONOMOUS TRADER SERVICE ==="
systemctl is-active quantum-autonomous-trader

echo ""
echo "=== 3. UNIVERSE OS CONFIG ==="
grep "USE_UNIVERSE_OS\|UNIVERSE_MAX" /etc/quantum/autonomous-trader.env

echo ""
echo "=== 4. UNIVERSE DATA (fra Redis) ==="
redis-cli GET "quantum:cfg:universe:active" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Total symbols: {len(d[\"symbols\"])}'); print(f'First 10: {d[\"symbols\"][:10]}')" 2>/dev/null || echo "Could not parse universe"

echo ""
echo "=== 5. AI ENGINE HEALTH ==="
curl -s http://localhost:8001/health 2>/dev/null | head -5

echo ""
echo "=== 6. ENSEMBLE PREDICTIONS (stream length) ==="
redis-cli XLEN "quantum:stream:ensemble.prediction"

echo ""
echo "=== 7. AI SIGNALS (stream length) ==="  
redis-cli XLEN "quantum:stream:ai.signal_generated"

echo ""
echo "=== 8. TRADE OUTCOMES/LEARNING (stream length) ==="
redis-cli XLEN "quantum:stream:trade.outcome"

echo ""
echo "=== 9. SISTE LOGGING FRA AUTONOMOUS TRADER ==="
journalctl -u quantum-autonomous-trader --since "5 min ago" --no-pager 2>&1 | grep -E "UNIVERSE|Universe|symbols|positions|CYCLE|Trading with" | tail -15

echo ""
echo "=== 10. BINANCE TESTNET POSITIONS (via API) ==="
curl -s http://localhost:8085/api/v1/positions 2>/dev/null | head -20

echo ""
echo "========================================"
echo "END OF PROOF"
echo "========================================"
