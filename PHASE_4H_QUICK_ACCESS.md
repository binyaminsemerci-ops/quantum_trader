# 🎛️ PHASE 4H: GOVERNANCE DASHBOARD - QUICK ACCESS GUIDE

---

## 🌐 WEB ACCESS

### Main Dashboard
```
🔗 http://46.224.116.254:8501
```
**Features:**
- ✅ Real-time model weights display
- ✅ System status monitoring
- ✅ Validation events log
- ✅ System metrics
- ✅ Auto-refresh every 2 seconds
- ✅ Green terminal theme

---

## 📡 API ENDPOINTS

### Health Check
```bash
curl http://46.224.116.254:8501/health
```
**Response:**
```json
{
    "status": "healthy",
    "service": "governance_dashboard",
    "timestamp": "2025-12-20T08:36:10"
}
```

### System Status
```bash
curl http://46.224.116.254:8501/status | python3 -m json.tool
```
**Response:**
```json
{
    "models_loaded": 12,
    "governance_active": true,
    "retrainer_enabled": true,
    "validator_enabled": true,
    "ai_engine_health": "OK"
}
```

### Model Weights (LIVE)
```bash
curl http://46.224.116.254:8501/weights | python3 -m json.tool
```
**Response:**
```json
{
    "PatchTST": "1.0",
    "NHiTS": "0.5",
    "XGBoost": "0.3333",
    "LightGBM": "0.25"
}
```

### Validation Events
```bash
curl http://46.224.116.254:8501/events | python3 -m json.tool
```
**Response:**
```json
[]  # Will populate after Phase 4G validator runs
```

### System Metrics
```bash
curl http://46.224.116.254:8501/metrics | python3 -m json.tool
```
**Response:**
```json
{
    "timestamp": "2025-12-20T08:36:10",
    "redis_connected": true,
    "cpu_usage": "N/A",
    "memory": "N/A",
    "uptime": "N/A"
}
```

---

## 🐳 CONTAINER MANAGEMENT

### Check Status
```bash
ssh qt@46.224.116.254 'docker ps --filter name=quantum_governance_dashboard'
```

### View Logs
```bash
ssh qt@46.224.116.254 'docker logs quantum_governance_dashboard -f'
```

### Restart Container
```bash
ssh qt@46.224.116.254 'docker restart quantum_governance_dashboard'
```

### Stop Container
```bash
ssh qt@46.224.116.254 'docker stop quantum_governance_dashboard'
```

### Start Container
```bash
ssh qt@46.224.116.254 'docker start quantum_governance_dashboard'
```

### Container Shell Access
```bash
ssh qt@46.224.116.254 'docker exec -it quantum_governance_dashboard bash'
```

---

## 🔄 UPDATE PROCEDURE

### 1. Update Code Locally
Edit `c:\quantum_trader\backend\microservices\governance_dashboard\app.py`

### 2. Deploy to VPS
```bash
scp -i C:\Users\belen\.ssh\hetzner_fresh backend/microservices/governance_dashboard/app.py qt@46.224.116.254:~/quantum_trader/backend/microservices/governance_dashboard/
```

### 3. Rebuild Container
```bash
ssh -i C:\Users\belen\.ssh\hetzner_fresh qt@46.224.116.254 'cd ~/quantum_trader && docker compose build governance-dashboard'
```

### 4. Recreate Container
```bash
ssh -i C:\Users\belen\.ssh\hetzner_fresh qt@46.224.116.254 'docker stop quantum_governance_dashboard && docker rm quantum_governance_dashboard && docker run -d --name quantum_governance_dashboard --network quantum_trader_quantum_trader -e REDIS_HOST=quantum_redis -e REDIS_PORT=6379 -p 8501:8501 -v ~/quantum_trader/logs:/app/logs --restart unless-stopped quantum_trader-governance-dashboard:latest'
```

### 5. Verify
```bash
ssh -i C:\Users\belen\.ssh\hetzner_fresh qt@46.224.116.254 'curl -s http://localhost:8501/health && echo -e "\n" && curl -s http://localhost:8501/status | python3 -m json.tool'
```

---

## 🔍 TROUBLESHOOTING

### Dashboard Not Loading
```bash
# Check if container is running
ssh qt@46.224.116.254 'docker ps --filter name=quantum_governance_dashboard'

# Check logs for errors
ssh qt@46.224.116.254 'docker logs quantum_governance_dashboard --tail 50'

# Restart container
ssh qt@46.224.116.254 'docker restart quantum_governance_dashboard'
```

### Connection Refused Errors
```bash
# Verify network configuration
ssh qt@46.224.116.254 'docker network inspect quantum_trader_quantum_trader | grep -A5 quantum_governance_dashboard'

# Test AI Engine connectivity
ssh qt@46.224.116.254 'docker exec quantum_governance_dashboard python3 -c "import httpx; r=httpx.get(\"http://quantum_ai_engine:8001/health\"); print(r.status_code)"'

# Test Redis connectivity
ssh qt@46.224.116.254 'docker exec quantum_governance_dashboard python3 -c "import redis; r=redis.Redis(host=\"quantum_redis\", port=6379); print(r.ping())"'
```

### Weights Not Displaying
```bash
# Check Redis for governance weights
ssh qt@46.224.116.254 'docker exec quantum_redis redis-cli HGETALL governance_weights'

# Check AI Engine health
ssh qt@46.224.116.254 'curl -s http://localhost:8001/health | python3 -m json.tool'

# Rebuild and restart dashboard
ssh qt@46.224.116.254 'cd ~/quantum_trader && docker compose build governance-dashboard && docker restart quantum_governance_dashboard'
```

### Events Not Showing
```bash
# Check if validation log exists
ssh qt@46.224.116.254 'ls -lh ~/quantum_trader/logs/model_validation.log'

# Check log contents
ssh qt@46.224.116.254 'tail -20 ~/quantum_trader/logs/model_validation.log'

# Verify log mount in container
ssh qt@46.224.116.254 'docker exec quantum_governance_dashboard ls -lh /app/logs/model_validation.log'
```

---

## 📊 CURRENT STATUS

### Container Health
```
✅ quantum_governance_dashboard - Up 3 minutes - Port 8501
✅ quantum_ai_engine - Up 14 minutes (healthy) - Port 8001
✅ quantum_redis - Up 6 minutes (healthy) - Port 6379
✅ quantum_backend - Up 4 hours - Port 8000
```

### API Status
```
✅ /health - Responding
✅ /status - Responding (12 models loaded)
✅ /weights - Responding (4 models with weights)
✅ /events - Responding (empty - awaiting validations)
✅ /metrics - Responding (Redis connected)
```

### Integration Status
```
✅ AI Engine Connection - Working
✅ Redis Connection - Working
✅ Log File Access - Working
✅ Network Configuration - Correct (quantum_trader_quantum_trader)
✅ Auto-Refresh - Working (2 second interval)
```

---

## 🎯 MONITORING CHECKLIST

### Daily Checks
- [ ] Visit dashboard: http://46.224.116.254:8501
- [ ] Verify model weights are updating
- [ ] Check for new validation events
- [ ] Confirm all cards loading properly

### Weekly Checks
- [ ] Review container logs for errors
- [ ] Check Redis memory usage
- [ ] Verify auto-refresh still working
- [ ] Test all API endpoints

### Monthly Maintenance
- [ ] Review and archive old validation logs
- [ ] Check for dashboard updates/improvements
- [ ] Verify container restart policy still set
- [ ] Test full update procedure

---

## 📞 SUPPORT INFORMATION

### Container Details
- **Name:** quantum_governance_dashboard
- **Image:** quantum_trader-governance-dashboard:latest
- **Network:** quantum_trader_quantum_trader
- **Port:** 8501:8501
- **Restart Policy:** unless-stopped

### Dependencies
- **AI Engine:** quantum_ai_engine:8001
- **Redis:** quantum_redis:6379
- **Log Files:** ~/quantum_trader/logs/

### Documentation
- Full deployment guide: `AI_PHASE_4H_DASHBOARD_COMPLETE.md`
- Phase 4 overview: `AI_PHASE_4_STACK_OVERVIEW.md`
- AI integration: `AI_INTEGRATION_COMPLETE.md`

---

## 🚀 WHAT'S NEXT?

### When Validator Runs
1. Validation events will populate in dashboard
2. You'll see Sharpe ratio, MAPE, and training dates
3. Real-time model performance tracking available

### When Retrainer Runs
1. Dashboard will show retraining status
2. New model versions will be validated automatically
3. Weight adjustments will be visible immediately

### Future Enhancements
1. Historical weight charts
2. Performance trend graphs
3. Manual governance controls
4. Alert configuration
5. Export/reporting functionality

---

**PHASE 4H IS COMPLETE AND OPERATIONAL** ✅

Access your dashboard at: **http://46.224.116.254:8501**

---

*Last Updated: 2025-12-20*
*Status: Production Ready*
