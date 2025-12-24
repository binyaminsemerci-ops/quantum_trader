# 🔧 MANUAL VPS RUNBOOK - QUANTUM TRADER
**For when Copilot/Sonnet is rate-limited or unavailable**

---

## 1️⃣ CONTAINER HEALTH CHECK

```bash
# All containers status
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

# Expected: All containers "Up" with "(healthy)" where applicable
# Critical containers: quantum_backend, quantum_redis, quantum_nginx, quantum_trading_bot
# Shows ports to verify services are exposed correctly
```

---

## 2️⃣ TRADE.INTENT LAG + PENDING

```bash
# Consumer group status (lag = unprocessed events)
docker exec quantum_redis redis-cli XINFO GROUPS quantum:stream:trade.intent

# Expected output:
# 1) 1) "name"
#    2) "quantum:group:execution:trade.intent"
#    3) "consumers"
#    4) "1"  # Should be 1+ (consumer active)
#    5) "pending"
#    6) "0"  # Should be 0 or low (<10)
#    7) "lag"
#    8) "0"  # Should be 0 or low (<50)
```

```bash
# Detailed pending events (shows which events are stuck)
docker exec quantum_redis redis-cli XPENDING quantum:stream:trade.intent quantum:group:execution:trade.intent

# Expected output if healthy:
# 1) (integer) 0  # No pending events
# 2) (nil)        # No start ID
# 3) (nil)        # No end ID
# 4) (nil)        # No consumers
#
# If pending > 0: Shows event IDs that are stuck in processing
```

---

## 3️⃣ TAIL LOGS (CONSUMER + BACKEND)

```bash
# Backend logs - last 200 lines
docker logs --tail 200 quantum_backend

# Look for:
# ✅ "Phase 3.5" or "TradeIntentSubscriber" = subscriber initialized
# ❌ "ERROR" or "Exception" = problems
# ✅ "Processing trade intent" = consumer working
# ✅ "EXIT_GATEWAY" with "reduceOnly=True" = exit orders using fix
```

```bash
# Trading bot logs - last 200 lines
docker logs --tail 200 quantum_trading_bot

# Look for:
# ✅ "Signal published" or "Trade intent" = generating signals
# ⚠️ "Fallback signal" = AI Engine not responding (404)
# ❌ "ERROR" = problems
# ✅ "regime": "RANGE" (or TREND/BULL/BEAR) = regime integration working
```

---

## 4️⃣ NGINX HEALTH

```bash
# Nginx container health (detailed)
docker inspect quantum_nginx | jq '.[0].State.Health'

# Expected: "Status": "healthy", "FailingStreak": 0
# Shows full health check details including last check time
```

```bash
# Nginx logs - last 120 lines
docker logs --tail 120 quantum_nginx

# Look for:
# ❌ "connection refused" = backend not reachable
# ❌ "502 Bad Gateway" = backend crashed
# ❌ "504 Gateway Timeout" = backend too slow
# ✅ "GET /health HTTP/1.1" 200 = health checks passing
```

```bash
# Test backend health directly
curl -sS http://localhost:8000/health

# Expected: {"status":"healthy"} or similar JSON
# -sS = silent but show errors
```

---

## 5️⃣ RAM/DISK + OOM CHECK

```bash
# Memory usage (system)
free -h

# Expected output example:
#               total        used        free      shared  buff/cache   available
# Mem:           15Gi        12Gi        1.5Gi       200Mi       2.0Gi       3.0Gi
# 
# ✅ Good: available > 2Gi
# ⚠️ Warning: available < 1Gi
# ❌ Critical: available < 500Mi
```

```bash
# Disk usage (all mounted filesystems)
df -h

# Expected output example:
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/sda1       150G  111G   32G  78% /
# 
# ✅ Good: Use% < 80%
# ⚠️ Warning: Use% 80-90%
# ❌ Critical: Use% > 90%
```

```bash
# Check for OOM killer events with timestamps (last 80 lines)
dmesg -T | egrep -i 'oom|killed process|out of memory' | tail -80

# Expected: No output (no OOM events)
# If output exists: Shows which processes were killed by OOM with exact timestamps
# -T = human-readable timestamps
# egrep = extended grep for multiple patterns
```

---

## 🚨 QUICK PROBLEM RESOLUTION

### If lag is high (>50):
```bash
# Check if consumer is stuck
docker exec quantum_redis redis-cli XINFO CONSUMERS quantum:stream:trade.intent quantum:group:execution:trade.intent

# If idle > 300000 (5 minutes), restart backend:
docker restart quantum_backend

# Wait 15 seconds, then check lag again
sleep 15
docker exec quantum_redis redis-cli XINFO GROUPS quantum:stream:trade.intent
```

### If backend unhealthy:
```bash
# Check logs for errors
docker logs --tail 100 quantum_backend | grep -i "error\|exception"

# Restart backend
docker restart quantum_backend

# Wait and verify
sleep 15
docker inspect quantum_backend --format='{{.State.Health.Status}}'
```

### If nginx unhealthy:
```bash
# Test backend directly (bypass nginx)
curl -s http://localhost:8000/health

# If backend OK, reload nginx:
docker exec quantum_nginx nginx -s reload

# If still unhealthy, restart:
docker restart quantum_nginx
```

### If OOM detected:
```bash
# Identify memory hog
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}" | sort -k2 -hr

# Consider restarting high-memory container
# OR add swap if system memory is genuinely exhausted
```

---

## 📊 FULL STATUS ONE-LINER

```bash
echo "=== CONTAINERS ===" && docker ps --format "table {{.Names}}\t{{.Status}}" | grep quantum && \
echo -e "\n=== TRADE.INTENT LAG ===" && docker exec quantum_redis redis-cli XINFO GROUPS quantum:stream:trade.intent | grep -A1 "lag" && \
echo -e "\n=== MEMORY ===" && free -h | grep "Mem:" && \
echo -e "\n=== DISK ===" && df -h / | grep "/$"

# Quick sanity check - all critical metrics in one command
```

---

## 📝 NOTES

- **SSH command:** `ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254`
- **Project path:** `/home/qt/quantum_trader`
- **All commands assume you're already SSH'd into VPS**
- **Normal operation:** lag=0, pending=0, all containers healthy, memory >2Gi available
- **Critical threshold:** lag>100, pending>50, memory <500Mi available, disk >90%

---

**Last Updated:** 2025-12-24  
**VPS:** 46.224.116.254 (Hetzner)  
**Project:** Quantum Trader Hedge Fund OS
