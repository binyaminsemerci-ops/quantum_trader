# P0 RUNBOOK: Execution Stuck

**Severity**: P0 - CRITICAL  
**Category**: Trading Execution  
**Owner**: On-Call Engineer

---

## 🚨 SYMPTOM

One or more of:
- Alert: `OrderSubmitWithoutResponse` firing
- Alert: `NoOrdersSubmitted` firing for >15min
- Auto-executor logs show `ORDER_SUBMIT` without matching `ORDER_RESPONSE`
- Grafana shows orders submitted count > orders responded count

---

## 🔍 CHECKS

### 1. Check Auto-Executor Health

```bash
# Container status
docker ps | grep quantum_auto_executor

# Recent logs (JSON format with correlation_id)
docker logs quantum_auto_executor --tail 100 | grep ORDER_

# Check for GATE_BLOCKED (execution policy rejecting orders)
docker logs quantum_auto_executor --tail 200 | grep GATE_BLOCKED
```

**Expected**: Container UP, logs show `ORDER_SUBMIT` followed by `ORDER_RESPONSE` or `ORDER_ERROR`

**If not**: Executor crashed or execution policy blocking all orders

### 2. Check Binance API Connectivity

```bash
# Test from inside executor container
docker exec quantum_auto_executor curl -s "https://fapi.binance.com/fapi/v1/time"

# Check if testnet mode (should see BINANCE_USE_TESTNET=true in preflight)
docker exec quantum_auto_executor env | grep BINANCE_USE_TESTNET
```

**Expected**: JSON response with serverTime

**If timeout**: Network issue or Binance API down

**If 418/429**: Rate limited (check IP ban)

### 3. Check Redis Event Bus

```bash
# Redis connectivity
docker exec quantum_redis redis-cli PING

# Check intent stream (should have events)
docker exec quantum_redis redis-cli XLEN quantum:stream:intent

# Check execution stream (should have responses)
docker exec quantum_redis redis-cli XLEN quantum:stream:execution
```

**Expected**: PONG, stream lengths > 0

**If not**: Event bus down, signals not reaching executor

### 4. Trace Specific Order by correlation_id

```bash
# Get correlation_id from alert or logs
CORR_ID="<correlation_id from alert>"

# Search all logs for this correlation_id
docker logs quantum_auto_executor 2>&1 | grep "$CORR_ID"
docker logs quantum_ai_engine 2>&1 | grep "$CORR_ID"

# Or use Loki (if available)
# Grafana → Explore → Loki → {container=~"quantum_.*"} |= "CORR_ID"
```

**Expected**: See full flow:
1. `INTENT_RECEIVED` (ai_engine or executor)
2. `ORDER_SUBMIT` (executor)
3. `ORDER_RESPONSE` or `ORDER_ERROR` (executor)

**If missing step 3**: Order stuck at Binance or network timeout

---

## 🛠️ MITIGATION

### Quick Fix 1: Restart Auto-Executor

```bash
cd /home/qt/quantum_trader
docker compose restart auto-executor

# Wait 15s for startup
sleep 15

# Verify health
docker logs quantum_auto_executor --tail 20
```

**When to use**: Executor crashed or in bad state

**Risk**: Low (restarts consumer, doesn't lose data)

### Quick Fix 2: Cancel Stuck Orders

```bash
# Get open orders from Binance
docker exec quantum_auto_executor python3 -c "
from binance.client import Client
import os
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
orders = client.futures_get_open_orders()
print(orders)
"

# Cancel all open orders (ONLY if confirmed stuck)
docker exec quantum_auto_executor python3 -c "
from binance.client import Client
import os
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
client.futures_cancel_all_open_orders()
print('All orders cancelled')
"
```

**When to use**: Orders stuck in Binance for >5min

**Risk**: MEDIUM (cancels live orders, may exit positions unexpectedly)

### Quick Fix 3: Emergency Stop

```bash
# Use abort script (kills all trading)
cd /home/qt/quantum_trader
bash scripts/go_live_abort.sh
# Type 'ABORT' when prompted
```

**When to use**: Cannot diagnose issue, need to stop bleeding

**Risk**: HIGH (stops all trading, closes positions if flag set)

---

## 🔄 ROLLBACK

### Rollback to Testnet

```bash
# Stop executor
docker compose stop auto-executor

# Edit .env or docker-compose.yml
echo "BINANCE_USE_TESTNET=true" >> .env
echo "PAPER_TRADING=false" >> .env

# Restart with testnet config
docker compose up -d auto-executor

# Verify testnet mode
docker exec quantum_auto_executor env | grep BINANCE_USE_TESTNET
# Should see: BINANCE_USE_TESTNET=true
```

### Rollback to Paper Trading

```bash
# Edit .env
echo "PAPER_TRADING=true" >> .env

# Restart
docker compose restart auto-executor

# Verify logs show "WOULD_SUBMIT" instead of actual orders
docker logs quantum_auto_executor --tail 50 | grep WOULD_SUBMIT
```

---

## 📋 POST-MORTEM

After resolution, document:

1. **Root Cause**: What caused the stuck execution?
2. **Detection Time**: How long until alert fired?
3. **Resolution Time**: How long to fix?
4. **Orders Affected**: How many orders were stuck?
5. **PnL Impact**: Any losses due to stuck orders?
6. **Prevention**: What can prevent this in future?

Save to: `RUNBOOKS/postmortems/YYYY-MM-DD_execution_stuck.md`

---

## 🔗 RELATED

- `RUNBOOKS/P1B_logging_stack.md` - Logging infrastructure
- `RUNBOOKS/alerts.md` - Alert definitions
- `GO_LIVE_ABORT.md` - Emergency rollback procedure
- `scripts/go_live_abort.sh` - Emergency stop script
