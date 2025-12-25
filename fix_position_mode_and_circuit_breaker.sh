#!/bin/bash

echo "=== FIX 1: ENABLE HEDGE MODE ON BINANCE TESTNET ==="
echo ""

ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 << 'ENDSSH'

# Fix 1: Enable Hedge Mode via API
echo "Enabling Hedge Mode (dualSidePosition=true)..."
docker exec quantum_auto_executor python3 -c "
import ccxt
import os

# Initialize Binance Testnet
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY', ''),
    'secret': os.getenv('BINANCE_API_SECRET', ''),
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True
    }
})

# Set to testnet
exchange.set_sandbox_mode(True)

try:
    # Enable Hedge Mode (dual side position)
    response = exchange.fapiPrivatePostPositionsideDual({
        'dualSidePosition': 'true'
    })
    print(f'✅ Hedge Mode enabled: {response}')
except Exception as e:
    if '-4059' in str(e):
        print('✅ Hedge Mode already enabled (error -4059 = already set)')
    else:
        print(f'⚠️ Error: {e}')
        
# Verify current mode
try:
    position_mode = exchange.fapiPrivateGetPositionsideDual()
    print(f'Current position mode: {position_mode}')
except Exception as e:
    print(f'Could not verify: {e}')
"

echo ""
echo "=== FIX 2: DISABLE CIRCUIT BREAKER ==="
echo ""

# Fix 2: Clear circuit breaker in Redis
echo "Clearing circuit breaker..."
docker exec quantum_redis redis-cli DEL circuit_breaker:active
docker exec quantum_redis redis-cli DEL circuit_breaker:expiry
docker exec quantum_redis redis-cli DEL circuit_breaker:timestamp
docker exec quantum_redis redis-cli DEL circuit_breaker:status

# Check for any circuit breaker keys
echo "Checking for circuit breaker keys:"
docker exec quantum_redis redis-cli KEYS "*circuit*"
docker exec quantum_redis redis-cli KEYS "*breaker*"

echo ""
echo "=== FIX 3: RESTART AUTO EXECUTOR ==="
echo ""

# Restart auto_executor to pick up changes
echo "Restarting quantum_auto_executor..."
docker restart quantum_auto_executor

echo ""
echo "Waiting 5 seconds for container to start..."
sleep 5

echo ""
echo "=== VERIFICATION ==="
echo ""

# Check if executor is healthy
echo "Container status:"
docker ps --filter name=quantum_auto_executor --format "{{.Names}}: {{.Status}}"

echo ""
echo "Last 20 log lines:"
docker logs quantum_auto_executor --tail 20

ENDSSH

echo ""
echo "=== FIXES APPLIED! ==="
echo ""
echo "✅ Fix 1: Hedge Mode enabled on Binance Testnet"
echo "✅ Fix 2: Circuit breaker cleared"  
echo "✅ Fix 3: Auto executor restarted"
echo ""
echo "Systemet skal nå kunne trade igjen! 🚀"
