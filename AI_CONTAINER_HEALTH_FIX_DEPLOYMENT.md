# Container Health Fixes - Manual VPS Deployment Guide

## ✅ FIXES APPLIED LOCALLY

**Commit**: `c68d8258`  
**GitHub**: https://github.com/binyaminsemerci-ops/quantum_trader

### Changes Made:

1. **Risk Safety Dockerfile** (`microservices/risk_safety/Dockerfile`)
   - Fixed: CMD path to use correct working directory
   - Before: `CMD ["sh", "-c", "python /app/microservices/risk_safety/stub_main.py"]`
   - After: `CMD ["python", "stub_main.py"]` with `WORKDIR /app/microservices/risk_safety`

2. **Meta Regime Service** (`microservices/meta_regime/meta_regime_service.py`)
   - Fixed: Read from correct Redis stream
   - Before: `quantum:market:{symbol}:prices` (doesn't exist)
   - After: `quantum:stream:exchange.raw` (populated by cross-exchange collector)
   - Added: Symbol filtering and data extraction from exchange stream

3. **Portfolio Governance**
   - No changes needed - "0 samples" is normal behavior
   - Waiting for trade events from execution system

---

## 🚀 MANUAL VPS DEPLOYMENT (Run as root)

```bash
# 1. Navigate to project directory
cd /root/quantum_trader

# 2. Pull latest code
git stash  # Save any local changes
git pull origin main
git stash pop  # Restore if needed

# 3. Stop affected services
docker compose -f systemctl.vps.yml stop risk-safety meta-regime

# 4. Rebuild with no cache to ensure latest code
docker compose -f systemctl.vps.yml build --no-cache risk-safety meta-regime

# 5. Start services
docker compose -f systemctl.vps.yml up -d risk-safety meta-regime

# 6. Wait for startup
sleep 20

# 7. Verify Risk Safety (should see FastAPI startup)
docker logs --tail 20 quantum_risk_safety

# 8. Verify Meta Regime (should see market data processing)
docker logs --tail 20 quantum_meta_regime

# 9. Check container health
systemctl list-units | grep -E "risk_safety|meta_regime|portfolio_governance"
```

---

## 🔍 EXPECTED RESULTS

### Risk Safety:
```
✅ Container running (not restarting)
✅ Logs show: "Uvicorn running on http://0.0.0.0:8005"
✅ No "No such file or directory" errors
```

### Meta Regime:
```
✅ Container healthy
✅ Logs show: Market data processing for BTCUSDT, ETHUSDT, SOLUSDT
✅ Regime detection active (even if insufficient samples initially)
✅ No more "No market data available" warnings after ~30 seconds
```

### Portfolio Governance:
```
✅ Container healthy (healthcheck passes)
⚠️  Still shows "0 samples" - THIS IS NORMAL
📝 Waiting for trade execution to produce PnL events
```

---

## 🐛 TROUBLESHOOTING

### If Permission Denied Errors:
```bash
# Run commands with sudo
sudo docker compose -f systemctl.vps.yml ...
```

### If Git Pull Fails:
```bash
# Force reset to remote
git fetch origin main
git reset --hard origin/main
```

### If Containers Still Unhealthy After 60 Seconds:
```bash
# Check Redis connectivity
docker exec quantum_meta_regime python -c "import redis; r=redis.from_url('redis://redis:6379/0'); print('Redis OK' if r.ping() else 'Redis FAIL')"

# Check stream data
redis-cli XLEN quantum:stream:exchange.raw
# Should show > 0

# Check meta regime can read stream
docker exec quantum_meta_regime python -c "
import redis
r = redis.from_url('redis://redis:6379/0')
data = r.xrevrange('quantum:stream:exchange.raw', count=1)
print(f'Stream accessible: {len(data) > 0}')
"
```

---

## 📊 VERIFICATION COMMANDS

```bash
# Full system health check
systemctl list-units --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | grep quantum

# Check all Phase 4 systems
redis-cli MGET \
  quantum:feedback:strategic_memory \
  quantum:evolution:selected \
  quantum:consensus:signal \
  quantum:governance:policy

# Monitor Meta Regime real-time
docker logs -f quantum_meta_regime

# Check market data flow
watch -n 5 'redis-cli XLEN quantum:stream:exchange.raw'
```

---

## ✅ SUCCESS CRITERIA

- [ ] Risk Safety: No restart loop, FastAPI running
- [ ] Meta Regime: Processing market data from 3+ symbols
- [ ] Portfolio Governance: Healthcheck passing (0 samples OK)
- [ ] All 3 containers: Status "Up X hours (healthy)"

---

## 📝 NOTES

1. **Portfolio Governance "0 samples"** is expected until live trading produces PnL events
2. **Meta Regime** needs ~30-60 seconds to collect sufficient market data for regime detection
3. **Risk Safety** should start immediately with no errors


