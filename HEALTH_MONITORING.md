# SYSTEM HEALTH MONITORING
## Automatisk oppdagelse av feil og problemer

---

## 🏥 HEALTH MONITOR OVERSIKT

Health Monitor er et automatisert system som kontinuerlig overvåker:

1. **Model Supervisor** - Detekterer feil mode (OBSERVE vs ENFORCED)
2. **AI Modeller** - Oppdager 100% bias eller crashed modeller
3. **Retraining** - Verifiserer at modeller lærer av nye data
4. **Execution Layer** - Sjekker at bias blocking er aktiv

### **Automatisk sjekkes hver 5. minutt**

---

## 🚨 PROBLEMER SOM OPPDAGES

### 1. **Model Supervisor i feil mode**
**Problem:**
```
Mode mismatch: Expected ENFORCED, but running in OBSERVE
```

**Forklaring:**
- Model Supervisor detekterer bias men blokkerer IKKE trades
- Systemet vil fortsette å ta >70% SHORT trades
- `.env` sier ENFORCED men koden kjører OBSERVE

**Auto-fix:**
```bash
# Health Monitor anbefaler:
Restart backend with QT_MODEL_SUPERVISOR_MODE=ENFORCED
```

**Manuell fix:**
```bash
systemctl restart backend
```

---

### 2. **LightGBM 100% SHORT bias**
**Problem:**
```
LightGBM extremely biased: 95% SELL predictions
```

**Forklaring:**
- LightGBM modell har lært BARE fra bearish data
- Gir SELL signal på 95-100% av symboler
- Portfolio blir 80-90% SHORT

**Auto-fix:**
```bash
# Health Monitor kan trigger:
Force retrain LightGBM with balanced historical data
```

**Manuell fix:**
```bash
# Trigger retraining via API:
curl -X POST http://localhost:8000/api/retrain/lgbm
```

---

### 3. **Model ikke lastet**
**Problem:**
```
lgbm_agent model not loaded
```

**Forklaring:**
- Modellen crashed under startup
- Mangler training data eller model file
- Kan ikke gi predictions

**Auto-fix:**
```bash
# Health Monitor anbefaler:
Retrain lgbm_agent or restore from backup
```

**Manuell fix:**
```bash
# Retrain fra scratch:
python retrain_models.py --model lgbm
```

---

### 4. **Retraining stoppet**
**Problem:**
```
No retraining for 48.0 hours (expected every 24h)
```

**Forklaring:**
- Retraining Orchestrator har crashed
- Modeller lærer ikke fra nye trades
- Bias vil ikke bli fikset automatisk

**Auto-fix:**
```bash
# Health Monitor trigger:
Check retraining logs for errors, may need manual trigger
```

**Manuell fix:**
```bash
# Restart retraining:
journalctl -u quantum_backend.service | grep "Retraining"
systemctl restart backend
```

---

### 5. **Bias blocking disabled**
**Problem:**
```
Model Supervisor not attached to executor - bias blocking disabled!
```

**Forklaring:**
- Event-driven executor mangler Model Supervisor reference
- Bias sjekk kjører IKKE før trades
- System kan ta ubalansert portfolio

**Auto-fix:**
```bash
# Health Monitor anbefaler:
Restart backend to reinitialize executor with Model Supervisor
```

**Manuell fix:**
```bash
systemctl restart backend
```

---

## 🔧 HVORDAN BRUKE HEALTH MONITOR

### **1. Quick Check (Python)**
```bash
python check_system_health.py
```

Output:
```
🏥 QUANTUM TRADER - SYSTEM HEALTH CHECK
================================================================================
Timestamp: 2025-11-26 23:56:47
================================================================================

✅ OVERALL STATUS: HEALTHY

✅ All systems healthy - no issues detected
```

### **2. Quick Check (PowerShell)**
```powershell
.\Check-SystemHealth.ps1
```

### **3. Auto-fix problemer**
```bash
python check_system_health.py --fix
```

Output hvis problemer:
```
🔧 Attempting to auto-fix 2 issues...
✅ Successfully fixed 2 issues

⚠️  1 issues require manual intervention:
   - [Retraining Orchestrator] No retraining for 48 hours
```

### **4. Continuous Monitoring**
```bash
python check_system_health.py --watch --interval 60
```

```powershell
.\Check-SystemHealth.ps1 -Watch -Interval 60
```

Viser status hver 60. sekund:
```
👁️  Watching system health (checking every 60 seconds)
Press Ctrl+C to stop

[... health status ...]

⏰ Next check in 60 seconds...
```

---

## 🌐 API ENDPOINTS

### **GET /health/monitor**
Hent komplett health status:

```bash
curl http://localhost:8000/health/monitor | jq
```

Response:
```json
{
  "status": "ok",
  "overall_health": "HEALTHY",
  "summary": {
    "timestamp": "2025-11-26T23:56:47+01:00",
    "overall_status": "HEALTHY",
    "total_issues": 0,
    "issues_by_severity": {
      "CRITICAL": 0,
      "DEGRADED": 0,
      "FAILED": 0
    },
    "expected_config": {
      "model_supervisor_mode": "ENFORCED",
      "bias_threshold": 0.7,
      "min_samples": 20
    }
  }
}
```

### **POST /health/monitor/auto-heal**
Trigger auto-healing manuelt:

```bash
curl -X POST http://localhost:8000/health/monitor/auto-heal | jq
```

Response:
```json
{
  "status": "ok",
  "message": "Auto-healing completed",
  "issues_detected": 2,
  "fixes_applied": 1,
  "remaining_issues": [
    {
      "component": "Retraining Orchestrator",
      "problem": "No retraining for 48 hours",
      "requires_manual_fix": true
    }
  ]
}
```

---

## ⚙️ KONFIGURASJON

I `.env` eller `systemctl.yml`:

```bash
# Health Monitor
QT_HEALTH_MONITOR_ENABLED=true          # Enable/disable
QT_HEALTH_CHECK_INTERVAL=300            # Check every 5 minutes

# Expected configuration (Health Monitor validerer mot disse)
QT_MODEL_SUPERVISOR_MODE=ENFORCED       # Required mode
QT_MODEL_SUPERVISOR_BIAS_THRESHOLD=0.70 # Max 70% bias allowed
QT_MODEL_SUPERVISOR_MIN_SAMPLES=20      # Need 20 signals before check
```

---

## 📊 HEALTH STATUS LEVELS

| Status | Betydning | Handling |
|--------|-----------|----------|
| ✅ **HEALTHY** | All systems working | Ingen handling nødvendig |
| 🟡 **DEGRADED** | Minor issues detected | Monitor, auto-fix attempts |
| 🔴 **CRITICAL** | Major problems | Immediate action required |
| ❌ **FAILED** | Component crashed | Manual intervention needed |

---

## 🔄 AUTO-HEALING

Health Monitor kan automatisk fikse:

1. **Model bias** → Trigger emergency retraining
2. **Missing models** → Schedule model rebuild
3. **Config drift** → Log warning (requires restart)

**Hva som IKKE kan auto-fixes:**
- Mode mismatch → Requires backend restart
- Crashed services → Requires docker restart
- Missing training data → Requires manual data collection

---

## 📈 INTEGRERING MED OVERVÅKNING

### **Grafana Dashboard**
```bash
# Add health metrics to Prometheus:
health_overall_status{status="HEALTHY"} 1
health_issues_total{severity="CRITICAL"} 0
health_issues_total{severity="DEGRADED"} 0
```

### **Slack/Discord Alerts**
```python
# Webhook når CRITICAL issues oppdages:
if overall_status == "CRITICAL":
    send_slack_alert(issues)
```

### **Email Notifications**
```python
# Daglig health rapport:
schedule.daily(time="08:00", func=send_health_report)
```

---

## 🚀 BESTE PRAKSIS

### **1. Sjekk health etter hver deploy**
```bash
systemctl up -d backend
sleep 10
python check_system_health.py --fix
```

### **2. Sett opp cron job for continuous monitoring**
```bash
# Crontab: Check every 5 minutes
*/5 * * * * cd /app && python check_system_health.py --fix >> health.log
```

### **3. Integrer i CI/CD pipeline**
```yaml
# .github/workflows/deploy.yml
- name: Health Check
  run: |
    python check_system_health.py
    if [ $? -ne 0 ]; then
      echo "Health check failed!"
      exit 1
    fi
```

### **4. Monitor logs for health issues**
```bash
journalctl -u quantum_backend.service --follow | grep "HEALTH\|CRITICAL"
```

---

## 🛠️ TROUBLESHOOTING

### **Problem: Health Monitor ikke initialisert**
```json
{
  "status": "disabled",
  "message": "Health Monitor not initialized"
}
```

**Løsning:**
```bash
# Enable i systemctl.yml:
QT_HEALTH_MONITOR_ENABLED=true

systemctl restart backend
```

### **Problem: Health check tar for lang tid**
```
Timeout after 10 seconds
```

**Løsning:**
```bash
# Øk timeout:
requests.get(url, timeout=30)
```

### **Problem: False positives**
```
CRITICAL: Mode mismatch
# Men mode er faktisk riktig
```

**Løsning:**
```bash
# Sjekk at .env matcher systemctl.yml:
grep MODEL_SUPERVISOR_MODE .env systemctl.yml
```

---

## 📝 EKSEMPEL OUTPUT

### **Når alt er OK:**
```
🏥 QUANTUM TRADER - SYSTEM HEALTH CHECK
================================================================================

✅ OVERALL STATUS: HEALTHY

✅ All systems healthy - no issues detected
```

### **Når problemer oppdages:**
```
🏥 QUANTUM TRADER - SYSTEM HEALTH CHECK
================================================================================

🔴 OVERALL STATUS: CRITICAL

⚠️  2 issues detected:
   🔴 CRITICAL: 1
   🟡 DEGRADED: 1

📋 Expected Configuration:
   Model Supervisor Mode: ENFORCED
   Bias Threshold: 70%
   Min Samples: 20

🔍 DETECTED ISSUES:
--------------------------------------------------------------------------------

1. [Model Supervisor] 🔴 CRITICAL
   Problem: Mode mismatch: Expected ENFORCED, but running in OBSERVE
   ✅ Auto-fixable: Restart backend with QT_MODEL_SUPERVISOR_MODE=ENFORCED

2. [AI Model: LightGBM] 🟡 DEGRADED
   Problem: LightGBM extremely biased: 95% SELL predictions
   ✅ Auto-fixable: Force retrain LightGBM with balanced historical data

--------------------------------------------------------------------------------
```

---

## 🎯 SAMMENDRAG

**Health Monitor løser ditt problem:**

1. ✅ **Oppdager når Model Supervisor kjører i OBSERVE mode**
   - Sjekker mode hver 5. minutt
   - Sammenligner med forventet config
   - Gir clear fix instructions

2. ✅ **Oppdager når modeller har 100% bias**
   - Tracker prediction distribution
   - Alert når >70% samme retning
   - Trigger automatic retraining

3. ✅ **Oppdager når modeller feiler**
   - Sjekker om modeller er loaded
   - Verifiserer at retraining kjører
   - Alert hvis modeller crasher

4. ✅ **Automatisk healing hvor mulig**
   - Trigger retraining
   - Schedule model rebuilds
   - Log warnings for manual fixes

**Nå har du full oversikt over systemets helse! 🏥**

