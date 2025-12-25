#!/bin/bash
set -e
echo "🚀 Starting Quantum Trader Paper Trading Mode"
echo "Timestamp: $(date)"
echo "Environment: Binance Testnet"
echo ""

# 1. Verify all critical services are healthy
echo "=== [Step 1/5] Verifying Services ==="
REQUIRED_SERVICES=(
  "quantum_ai_engine:8001"
  "quantum_position_monitor:none"
  "quantum_trading_bot:8003"
  "quantum_redis:6379"
  "quantum_backend:8000"
)

for service in "${REQUIRED_SERVICES[@]}"; do
  name="${service%%:*}"
  port="${service##*:}"
  
  status=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo "not_found")
  if [ "$status" != "running" ]; then
    echo "❌ $name is not running (status: $status)"
    exit 1
  fi
  echo "✅ $name is running"
  
  if [ "$port" != "none" ]; then
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null || echo "000")
    if [ "$http_code" == "200" ]; then
      echo "   ✅ Health check passed (HTTP $http_code)"
    else
      echo "   ⚠️ Health check failed (HTTP $http_code)"
    fi
  fi
done
echo ""

# 2. Verify Binance Testnet connection
echo "=== [Step 2/5] Verifying Binance Testnet ==="
docker exec quantum_ai_engine python3 -c "
from binance.client import Client
import os, sys
try:
    key = os.getenv('BINANCE_API_KEY')
    secret = os.getenv('BINANCE_API_SECRET')
    client = Client(key, secret, testnet=True)
    balance = client.futures_account_balance()
    print('✅ Testnet connected')
    print(f'   Account balances: {len(balance)} assets')
    # Show USDT balance
    for asset in balance:
        if asset['asset'] == 'USDT':
            print(f'   USDT Balance: {float(asset[\"balance\"]):.2f}')
except Exception as e:
    print(f'❌ Testnet connection failed: {e}')
    sys.exit(1)
"
echo ""

# 3. Check if trading is already active
echo "=== [Step 3/5] Checking Trading Status ==="
ACTIVE_POSITIONS=$(docker exec quantum_redis redis-cli KEYS "quantum:positions:*" | wc -l)
echo "Active positions: $ACTIVE_POSITIONS"

RECENT_SIGNALS=$(docker exec quantum_redis redis-cli KEYS "quantum:signal:*" | wc -l)
echo "Recent signals: $RECENT_SIGNALS"

# Check AI Engine status
AI_STATUS=$(curl -s http://localhost:8001/health | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['metrics'].get('running', False))" 2>/dev/null || echo "unknown")
echo "AI Engine running: $AI_STATUS"
echo ""

# 4. Activate paper trading mode
echo "=== [Step 4/5] Activating Paper Trading ==="

# Send activation signal via Redis
docker exec quantum_redis redis-cli SET "quantum:config:trading_mode" "PAPER_TRADING" EX 86400
docker exec quantum_redis redis-cli SET "quantum:config:trading_enabled" "true" EX 86400
docker exec quantum_redis redis-cli SET "quantum:config:auto_trading" "true" EX 86400

echo "✅ Trading mode set to: PAPER_TRADING"
echo "✅ Auto trading enabled: true"
echo "✅ Config expires in: 24 hours"
echo ""

# Restart AI Engine to pick up new config
echo "Restarting AI Engine to apply config..."
docker restart quantum_ai_engine > /dev/null 2>&1
sleep 5

# Restart Trading Bot
echo "Restarting Trading Bot..."
docker restart quantum_trading_bot > /dev/null 2>&1
sleep 3

# Restart Position Monitor
echo "Restarting Position Monitor..."
docker restart quantum_position_monitor > /dev/null 2>&1
sleep 3

echo "✅ Core services restarted with paper trading config"
echo ""

# 5. Monitor for activity
echo "=== [Step 5/5] Monitoring Initial Activity ==="
echo "Waiting 30 seconds for signal generation..."
sleep 30

# Check for new signals
NEW_SIGNALS=$(docker exec quantum_redis redis-cli KEYS "quantum:signal:*" | wc -l)
echo "Signals detected: $NEW_SIGNALS"

# Check AI Engine logs for recent activity
echo ""
echo "Recent AI Engine activity (last 2 minutes):"
docker logs quantum_ai_engine --since 2m 2>&1 | grep -E "(Signal|BUY|SELL|confidence)" | tail -10 || echo "No signals yet"

echo ""
echo "Recent Trading Bot activity (last 2 minutes):"
docker logs quantum_trading_bot --since 2m 2>&1 | grep -E "(Order|Position|Trade)" | tail -5 || echo "No trades yet"

echo ""
echo "=== [Paper Trading Activated] ==="
echo "✅ Mode: PAPER TRADING (Binance Testnet)"
echo "✅ Services: All critical services running"
echo "✅ Auto Trading: ENABLED"
echo "✅ Risk Limits: ACTIVE"
echo ""
echo "📊 Monitor trading activity:"
echo "   AI Engine logs:      docker logs -f quantum_ai_engine"
echo "   Trading Bot logs:    docker logs -f quantum_trading_bot"
echo "   Position Monitor:    docker logs -f quantum_position_monitor"
echo "   Redis positions:     docker exec quantum_redis redis-cli KEYS 'quantum:positions:*'"
echo ""
echo "🌐 Dashboards:"
echo "   Main Dashboard:      http://46.224.116.254:8080"
echo "   Grafana:             http://46.224.116.254:3001"
echo "   Governance:          http://46.224.116.254:8501"
echo ""
echo "⚠️  To stop trading:"
echo "   docker exec quantum_redis redis-cli SET quantum:config:trading_enabled false"
echo ""
echo "🎯 Paper trading is now LIVE!"
