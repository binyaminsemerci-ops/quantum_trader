# ⚡ PHASE 6: AUTO EXECUTION LAYER - QUICK REFERENCE

**Fast commands for managing auto execution system**

---

## 🚀 STARTUP COMMANDS

### Start Executor (Paper Trading)
```bash
cd ~/quantum_trader
docker compose --profile microservices up -d auto-executor
```

### View Startup Logs
```bash
docker logs quantum_auto_executor --tail 50
```

### Check Health
```bash
docker ps | grep auto_executor
```

---

## 📊 MONITORING COMMANDS

### Live Logs (Follow Mode)
```bash
docker logs quantum_auto_executor -f
```

### Last 20 Log Lines
```bash
docker logs quantum_auto_executor --tail 20
```

### Check Processing Status
```bash
docker logs quantum_auto_executor | grep "Cycle"
```

---

## 💹 TRADE HISTORY

### Last 10 Trades
```bash
docker exec quantum_redis redis-cli LRANGE trade_log 0 9 | python3 -m json.tool
```

### Last 5 Trades (Quick)
```bash
docker exec quantum_redis redis-cli LRANGE trade_log 0 4
```

### Total Trade Count
```bash
docker exec quantum_redis redis-cli GET total_trades
```

### View All Trades
```bash
docker exec quantum_redis redis-cli LRANGE trade_log 0 -1 | python3 -m json.tool
```

---

## 📈 METRICS COMMANDS

### Execution Metrics
```bash
docker exec quantum_redis redis-cli HGETALL execution_metrics
```

### Executor Metrics (Detailed)
```bash
docker exec quantum_redis redis-cli GET executor_metrics | python3 -m json.tool
```

### Current Balance
```bash
docker exec quantum_redis redis-cli HGET execution_metrics balance
```

### Circuit Breaker Status
```bash
docker exec quantum_redis redis-cli HGET execution_metrics circuit_breaker
```

---

## 🎯 SIGNAL COMMANDS

### View Current Signals
```bash
docker exec quantum_redis redis-cli GET live_signals | python3 -m json.tool
```

### Create Test Signal (BUY)
```bash
docker exec quantum_redis redis-cli SET live_signals '[
  {
    "symbol": "BTCUSDT",
    "action": "BUY",
    "confidence": 0.78,
    "price": 50000.0,
    "pnl": 0.0,
    "drawdown": 2.0
  }
]'
```

### Create Test Signal (SELL)
```bash
docker exec quantum_redis redis-cli SET live_signals '[
  {
    "symbol": "ETHUSDT",
    "action": "SELL",
    "confidence": 0.65,
    "price": 3500.0,
    "pnl": 0.0,
    "drawdown": 1.5
  }
]'
```

### Create Multiple Signals
```bash
docker exec quantum_redis redis-cli SET live_signals '[
  {"symbol":"BTCUSDT","action":"BUY","confidence":0.78,"drawdown":2.0},
  {"symbol":"ETHUSDT","action":"SELL","confidence":0.65,"drawdown":1.5}
]'
```

### Clear Signals
```bash
docker exec quantum_redis redis-cli DEL live_signals
```

---

## 🔄 CONTROL COMMANDS

### Restart Executor
```bash
docker restart quantum_auto_executor
```

### Stop Executor
```bash
docker stop quantum_auto_executor
```

### Start Executor (If Stopped)
```bash
docker start quantum_auto_executor
```

### Rebuild Executor
```bash
cd ~/quantum_trader
docker compose build auto-executor --no-cache
docker compose --profile microservices up -d auto-executor
```

---

## 🛑 EMERGENCY COMMANDS

### Stop All Trading Immediately
```bash
docker stop quantum_auto_executor
```

### Clear All Signals
```bash
docker exec quantum_redis redis-cli DEL live_signals
```

### Reset Metrics
```bash
docker exec quantum_redis redis-cli DEL execution_metrics
docker exec quantum_redis redis-cli DEL executor_metrics
docker exec quantum_redis redis-cli SET total_trades 0
```

### Circuit Breaker Override (CAREFUL!)
```bash
docker exec quantum_redis redis-cli HSET execution_metrics circuit_breaker false
```

---

## 🔍 DEBUGGING COMMANDS

### Check Redis Connection
```bash
docker exec quantum_auto_executor python3 -c "import redis; r=redis.Redis(host='quantum_redis'); print('Connected:', r.ping())"
```

### Verify Environment Variables
```bash
docker exec quantum_auto_executor env | grep -E "(EXCHANGE|TESTNET|PAPER|MAX_|CONFIDENCE)"
```

### Check Network Connectivity
```bash
docker network inspect quantum_trader_quantum_trader | grep -A 5 quantum_auto_executor
```

### Test Binance Connection (If Real Trading)
```bash
docker exec quantum_auto_executor python3 -c "from binance.client import Client; c=Client(); print(c.ping())"
```

---

## 📊 ONE-LINER STATUS CHECK

### Complete System Status
```bash
echo "=== AUTO EXECUTOR STATUS ===" && \
docker ps --format "{{.Names}}: {{.Status}}" | grep auto_executor && \
echo -e "\n=== RECENT ACTIVITY ===" && \
docker logs quantum_auto_executor --tail 10 && \
echo -e "\n=== METRICS ===" && \
echo "Total Trades: $(docker exec quantum_redis redis-cli GET total_trades)" && \
echo "Balance: $(docker exec quantum_redis redis-cli HGET execution_metrics balance)" && \
echo "Circuit Breaker: $(docker exec quantum_redis redis-cli HGET execution_metrics circuit_breaker)"
```

---

## 🎛️ CONFIGURATION QUICK CHANGE

### Switch to Production Mode (CAREFUL!)
```bash
# Edit docker-compose.yml
nano ~/quantum_trader/docker-compose.yml

# Change:
TESTNET=false
PAPER_TRADING=false

# Restart
docker restart quantum_auto_executor
```

### Adjust Risk Parameters
```bash
# Edit docker-compose.yml
nano ~/quantum_trader/docker-compose.yml

# Modify:
MAX_RISK_PER_TRADE=0.001  # Lower risk
MAX_LEVERAGE=1            # No leverage
MAX_POSITION_SIZE=100     # Smaller positions

# Restart
docker restart quantum_auto_executor
```

---

## 📱 DASHBOARD ACCESS

### View Governance Dashboard
```
http://46.224.116.254:8501
```

### Check Execution Metrics on Dashboard
```
http://46.224.116.254:8501/metrics
```

---

## 🚨 ALERT VERIFICATION

### Check Recent Alerts
```bash
docker exec quantum_redis redis-cli LRANGE governance_alerts 0 9 | python3 -m json.tool
```

### Watch for Execution Alerts
```bash
docker logs quantum_governance_alerts -f | grep -i execution
```

---

## 📝 COMMON WORKFLOWS

### Workflow 1: Deploy and Monitor
```bash
# 1. Start executor
docker compose --profile microservices up -d auto-executor

# 2. Watch logs
docker logs quantum_auto_executor -f

# 3. Check trades after 2 minutes
docker exec quantum_redis redis-cli LRANGE trade_log 0 9
```

### Workflow 2: Test New Signal
```bash
# 1. Create test signal
docker exec quantum_redis redis-cli SET live_signals '[{"symbol":"BTCUSDT","action":"BUY","confidence":0.75,"drawdown":2.0}]'

# 2. Wait 15 seconds
sleep 15

# 3. Check execution
docker logs quantum_auto_executor --tail 20

# 4. Verify trade logged
docker exec quantum_redis redis-cli LRANGE trade_log 0 0 | python3 -m json.tool
```

### Workflow 3: Daily Health Check
```bash
# Check all executor components
docker ps | grep auto_executor && \
docker logs quantum_auto_executor --tail 5 && \
echo "Total Trades: $(docker exec quantum_redis redis-cli GET total_trades)" && \
echo "Balance: $(docker exec quantum_redis redis-cli GET executor_metrics | python3 -c 'import json,sys; print(json.load(sys.stdin)["balance"])')"
```

---

## 🔧 TROUBLESHOOTING QUICK FIXES

### Issue: Executor Not Processing
```bash
# 1. Check for signals
docker exec quantum_redis redis-cli GET live_signals

# 2. If empty, create test signal
docker exec quantum_redis redis-cli SET live_signals '[{"symbol":"BTCUSDT","action":"BUY","confidence":0.70,"drawdown":2.0}]'

# 3. Watch logs
docker logs quantum_auto_executor -f
```

### Issue: Circuit Breaker Stuck
```bash
# 1. Check current drawdown
docker exec quantum_redis redis-cli GET live_signals | python3 -m json.tool | grep drawdown

# 2. Create signal with low drawdown
docker exec quantum_redis redis-cli SET live_signals '[{"symbol":"BTCUSDT","action":"BUY","confidence":0.75,"drawdown":1.0}]'
```

### Issue: No Trades Logging
```bash
# 1. Check confidence threshold
docker logs quantum_auto_executor | grep "rejected"

# 2. Increase signal confidence
docker exec quantum_redis redis-cli SET live_signals '[{"symbol":"BTCUSDT","action":"BUY","confidence":0.80,"drawdown":2.0}]'
```

---

## 📖 RESOURCE LINKS

- **Full Documentation:** `AI_PHASE_6_AUTO_EXECUTION_COMPLETE.md`
- **Architecture Overview:** `AI_FULL_SYSTEM_OVERVIEW_DEC13.md`
- **Dashboard Guide:** `AI_PHASE_4H_DASHBOARD_COMPLETE.md`
- **Alert System:** `AI_PHASE_4I_ALERTS_COMPLETE.md`

---

## 💡 PRO TIPS

### Tip 1: Test Before Production
Always run paper trading for at least 100 trades before real trading.

### Tip 2: Monitor Circuit Breaker
If circuit breaker triggers frequently, adjust MAX_DRAWDOWN threshold.

### Tip 3: Start Conservative
Begin with low MAX_RISK_PER_TRADE (0.001) and small MAX_POSITION_SIZE ($50).

### Tip 4: Watch Win Rate
Track successful_trades/total_trades ratio. Target >60% before scaling.

### Tip 5: Regular Backups
Export trade logs weekly:
```bash
docker exec quantum_redis redis-cli LRANGE trade_log 0 -1 > trades_backup_$(date +%Y%m%d).json
```

---

**Quick Reference Version:** 1.0  
**Last Updated:** 2025-12-20  
**For:** Phase 6 Auto Execution Layer  
