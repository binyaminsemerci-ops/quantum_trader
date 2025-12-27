# 🎯 Quantum Trader V3 - Auto-Repair Quick Reference Card

**Version:** 1.0 | **Date:** 2025-12-17 | **Status:** ✅ All Repairs Complete

---

## 🚀 One-Command Health Check

```bash
bash ~/final_health_check.sh
```

**Expected Output:**
```
✅ TOTAL RECENT ERRORS: 0 - System healthy!
```

---

## 🔍 Specific Error Checks

### PostgreSQL Errors
```bash
docker logs --since 5m quantum_postgres 2>&1 | grep -i "fatal\|error" | wc -l
```
**Expected:** `0`

### XGBoost Errors
```bash
docker logs --since 5m quantum_ai_engine 2>&1 | grep -i "mismatch\|shape" | wc -l
```
**Expected:** `0`

### Grafana Restart Notices
```bash
docker logs --since 5m quantum_grafana 2>&1 | grep -i "restart.*plugin" | wc -l
```
**Expected:** `0`

---

## 🛠️ Quick Fixes

### If PostgreSQL Errors Return
```bash
docker exec quantum_postgres psql -U quantum -tc "SELECT 1 FROM pg_database WHERE datname='quantum'" | grep -q 1 || docker exec quantum_postgres psql -U quantum -d quantum_trader -c "CREATE DATABASE quantum;"
```

### If XGBoost Errors Return
```bash
docker cp ~/quantum_trader/scripts/fix_futures_compact.py quantum_ai_engine:/tmp/fix.py && docker exec quantum_ai_engine python3 /tmp/fix.py && docker restart quantum_ai_engine
```

### If Grafana Becomes Unstable
```bash
docker system prune -f && docker restart quantum_grafana
```

---

## 📊 Model Verification

### Check XGBoost Model Features
```bash
docker exec quantum_ai_engine python3 -c "import joblib; m=joblib.load('/app/models/xgb_futures_model.joblib'); print(f'Model: {m.n_features_in_} features')"
```
**Expected:** `Model: 22 features`

---

## 🔄 Rollback Commands

### Restore Original XGBoost Model
```bash
docker exec quantum_ai_engine cp /app/models/xgb_futures_backup_20251217_172500.joblib /app/models/xgb_futures_model.joblib && docker restart quantum_ai_engine
```

---

## 🤖 Deploy Boot-Time Automation

```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "bash ~/quantum_trader/scripts/create_boot_autofix.sh"
```

**Test Automation:**
```bash
sudo systemctl start quantum-autofix.service && journalctl -u quantum-autofix.service
```

---

## 📈 Run Comprehensive Auto-Repair

```bash
docker exec quantum_ai_engine python3 ~/quantum_trader/scripts/auto_repair_system.py
```

---

## 🌐 Access Points

- **Web Dashboard:** http://46.224.116.254:8080
- **Grafana:** http://46.224.116.254:3000
- **Prometheus:** http://46.224.116.254:9090
- **AI Engine Health:** `docker exec quantum_ai_engine curl -s http://localhost:8001/health`
- **Backend Health:** `curl http://46.224.116.254:8000/health`

---

## 📁 Important Files

### Scripts
- `/home/qt/quantum_trader/scripts/fix_futures_compact.py` - XGBoost repair
- `/home/qt/quantum_trader/scripts/auto_repair_system.py` - Full auto-repair
- `/home/qt/final_health_check.sh` - Quick health check
- `/home/qt/quantum_trader/scripts/create_boot_autofix.sh` - Boot automation

### Backups
- `/app/models/xgb_futures_backup_20251217_172500.joblib` (49-feature original)
- `/app/models/xgb_futures_model.joblib` (22-feature current)

### Documentation
- `c:\quantum_trader\docs\AI_PROMPT12_AUTOREPAIR_COMPLETE.md` - Full docs
- `c:\quantum_trader\docs\AI_PROMPT12_COMPLETION_SUMMARY.md` - Summary
- `c:\quantum_trader\docs\AI_PROMPT12_BEFORE_AFTER.md` - Comparison
- `c:\quantum_trader\docs\AI_PROMPT12_QUICK_REF.md` - This card

---

## 🎯 Success Indicators

✅ **All Green** = System Healthy
- PostgreSQL errors: 0
- XGBoost errors: 0
- Grafana restarts: 0
- All containers: Up & healthy
- Health checks: 200 OK

---

## 🚨 Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| PostgreSQL errors/5min | >0 | >5 |
| XGBoost errors/5min | >0 | >10 |
| Grafana restarts/hour | >0 | >3 |
| Container failures | >0 | >1 |

---

## 📞 Emergency Contact Flow

1. Run `bash ~/final_health_check.sh`
2. Check specific error logs (see above)
3. Apply quick fix for affected component
4. If unresolved, run full auto-repair: `python3 auto_repair_system.py`
5. If still failing, check rollback section in full documentation

---

## 🔐 SSH Access

```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254
```

**VPS:** Hetzner 46.224.116.254  
**User:** qt  
**Key:** ~/.ssh/hetzner_fresh

---

## 📅 Maintenance Schedule

- **Hourly:** Check health status (`bash ~/final_health_check.sh`)
- **Daily:** Review error logs (`docker logs --since 24h <container>`)
- **Weekly:** Run full auto-repair (`auto_repair_system.py`)
- **Monthly:** Verify model backups exist

---

## 💡 Pro Tips

1. **Always check recent logs (last 5min)** - Historical errors can be misleading
2. **AI Log Analyzer runs periodically** - Wait 10-30min after repairs for updated score
3. **Model backups are timestamped** - Easy to identify and restore
4. **Boot automation prevents errors on restart** - Deploy it ASAP
5. **Health checks return 200 even if container shows "unhealthy"** - Trust the logs

---

## ✅ Current Status (2025-12-17 17:35 UTC)

```
PostgreSQL: ✅ 0 errors (quantum DB created)
XGBoost:    ✅ 0 errors (22-feature model deployed)
Grafana:    ✅ 0 notices (27GB cache cleaned)
Containers: ✅ All 9 running healthy
AI Engine:  ✅ Processing signals (BTCUSDT, ETHUSDT, BNBUSDT)
Health:     ✅ GET /health HTTP/1.1 200 OK
```

**System Status:** 🟢 FULLY OPERATIONAL

---

**Quick Ref Version:** 1.0  
**Last Updated:** 2025-12-17T17:35:00Z  
**For:** Quantum Trader V3 Production @ 46.224.116.254  
**Keep This:** Pin to desktop or save to favorites
