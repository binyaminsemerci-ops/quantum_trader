#!/bin/bash
# Real-time Paper Trading Monitor
# Shows live trading activity on Binance Testnet

echo "📊 Quantum Trader - Live Paper Trading Monitor"
echo "Testnet Account: 7,846.55 USDT"
echo "Press Ctrl+C to stop monitoring"
echo "================================================"
echo ""

while true; do
  clear
  echo "🎯 QUANTUM TRADER - PAPER TRADING MONITOR"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Mode: PAPER TRADING (Binance Testnet)"
  echo "Account: 7,846.55 USDT"
  echo "================================================"
  echo ""
  
  # Active Positions
  echo "📈 ACTIVE POSITIONS:"
  POSITIONS=$(docker exec quantum_redis redis-cli KEYS "quantum:positions:*" 2>/dev/null)
  if [ -z "$POSITIONS" ]; then
    echo "   No active positions"
  else
    echo "$POSITIONS" | while read -r key; do
      if [ ! -z "$key" ]; then
        data=$(docker exec quantum_redis redis-cli GET "$key" 2>/dev/null)
        echo "   • $key"
        echo "     $data" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'     Symbol: {d.get(\"symbol\",\"N/A\")}, Side: {d.get(\"side\",\"N/A\")}, Size: {d.get(\"size\",\"N/A\")}');" 2>/dev/null || echo "     $data"
      fi
    done
  fi
  echo ""
  
  # Recent Signals (last 5)
  echo "🎯 RECENT SIGNALS:"
  docker logs quantum_ai_engine --since 5m 2>&1 | grep -E "🎯 Ensemble returned|Signal generated" | tail -5 | while read line; do
    echo "   $line"
  done
  [ -z "$(docker logs quantum_ai_engine --since 5m 2>&1 | grep -E '🎯 Ensemble returned|Signal generated' | tail -5)" ] && echo "   No signals in last 5 minutes"
  echo ""
  
  # Recent Trades (last 5)
  echo "💰 RECENT TRADES:"
  docker logs quantum_trading_bot --since 5m 2>&1 | grep -E "Order|Trade|Position opened|Position closed" | tail -5 | while read line; do
    echo "   $line"
  done
  [ -z "$(docker logs quantum_trading_bot --since 5m 2>&1 | grep -E 'Order|Trade|Position opened|Position closed' | tail -5)" ] && echo "   No trades in last 5 minutes"
  echo ""
  
  # System Health
  echo "🏥 SYSTEM HEALTH:"
  AI_STATUS=$(curl -s http://localhost:8001/health 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'Running: {d[\"metrics\"][\"running\"]}, Models: {d[\"metrics\"][\"models_loaded\"]}');" 2>/dev/null || echo "Status unavailable")
  echo "   AI Engine: $AI_STATUS"
  
  BOT_STATUS=$(docker ps --filter name=quantum_trading_bot --format "{{.Status}}" | head -1)
  echo "   Trading Bot: $BOT_STATUS"
  
  MONITOR_STATUS=$(docker ps --filter name=quantum_position_monitor --format "{{.Status}}" | head -1)
  echo "   Position Monitor: $MONITOR_STATUS"
  echo ""
  
  # Drift Warnings
  echo "⚠️  RECENT WARNINGS:"
  docker logs quantum_ai_engine --since 2m 2>&1 | grep -E "WARNING|DRIFT|ERROR" | tail -3 | while read line; do
    echo "   $line"
  done
  [ -z "$(docker logs quantum_ai_engine --since 2m 2>&1 | grep -E 'WARNING|DRIFT|ERROR' | tail -3)" ] && echo "   No warnings"
  echo ""
  
  echo "================================================"
  echo "📊 Dashboards: http://46.224.116.254:8080"
  echo "📈 Grafana: http://46.224.116.254:3001"
  echo "⏸️  Stop trading: redis-cli SET quantum:config:trading_enabled false"
  echo ""
  
  sleep 10
done
