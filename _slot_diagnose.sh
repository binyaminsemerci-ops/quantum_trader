#!/bin/bash
echo "=== TESTNET ENV ==="
cat /etc/quantum/testnet.env 2>/dev/null || echo "MISSING"

echo ""
echo "=== SLOT STATE ==="
redis-cli hgetall quantum:state:slots 2>/dev/null

echo ""
echo "=== ACTIVE POSITION COUNT ==="
redis-cli get quantum:state:active_positions 2>/dev/null
redis-cli scard quantum:positions:active 2>/dev/null

echo ""
echo "=== POSITION KEYS COUNT ==="
redis-cli keys "quantum:position:*" 2>/dev/null | wc -l

echo ""
echo "=== ACTIVE POSITIONS SET ==="
redis-cli smembers quantum:positions:active 2>/dev/null

echo ""
echo "=== SLOT KEYS ==="
redis-cli keys "quantum:slots*" 2>/dev/null
redis-cli keys "quantum:state:slot*" 2>/dev/null

echo ""
echo "=== APPLY LAYER LEDGER COUNT ==="
redis-cli keys "quantum:ledger:position:*" 2>/dev/null | wc -l

echo ""
echo "=== AUTONOMOUS TRADER CONFIG ==="
redis-cli hgetall quantum:config:autonomous_trader 2>/dev/null

echo ""
echo "=== INTENT EXECUTOR ENV BASE URL ==="
grep -E "BINANCE_BASE_URL|BINANCE_USE_TESTNET|TESTNET" /etc/quantum/intent-executor.env 2>/dev/null

echo ""
echo "=== EXECUTION SERVICE ENV ==="
grep -E "BINANCE_BASE_URL|BINANCE_USE_TESTNET|TESTNET" /etc/quantum/ai-engine.env 2>/dev/null
grep -E "BINANCE_BASE_URL|BINANCE_USE_TESTNET|TESTNET" /etc/quantum/testnet.env 2>/dev/null
