# BLOCKER RESOLUTION COMMANDS

**Date**: December 4, 2025  
**Purpose**: Manual commands to resolve GO-LIVE blockers

---

## AUTOMATED RESOLUTION (RECOMMENDED)

```powershell
# Run automated blocker resolution script
.\scripts\resolve_blockers.ps1
```

**This script will**:
1. Start backend service
2. Wait for health endpoints
3. Verify metrics endpoint
4. Re-run preflight checks
5. Report status

---

## MANUAL RESOLUTION

### BLOCKER 1: Start Backend Service

**Option A: Start in Terminal (Foreground)**
```powershell
# Start backend service
cd c:\quantum_trader
python -m uvicorn backend.main:app --host localhost --port 8000 --reload

# Keep terminal open - service runs in foreground
# Press Ctrl+C to stop
```

**Option B: Start in New Window (Background)**
```powershell
# Start backend in new window
Start-Process python -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "localhost", "--port", "8000" -WindowStyle Normal

# Service runs in separate window
# Close window to stop service
```

**Option C: Start as Background Job**
```powershell
# Start backend as PowerShell job
$job = Start-Job -ScriptBlock {
    cd c:\quantum_trader
    python -m uvicorn backend.main:app --host localhost --port 8000
}

# Check job status
Get-Job

# View job output
Receive-Job $job

# Stop job
Stop-Job $job
```

---

### BLOCKER 2: Verify Health Endpoints

**Check Liveness Endpoint**
```powershell
# Should return: {"status": "ok"} or similar
Invoke-WebRequest -Uri "http://localhost:8000/health/live" -UseBasicParsing
```

**Check Readiness Endpoint**
```powershell
# Should return: {"status": "ready"} or similar
Invoke-WebRequest -Uri "http://localhost:8000/health/ready" -UseBasicParsing
```

**Expected Output**:
```
StatusCode        : 200
StatusDescription : OK
Content           : {"status":"ok"}
```

---

### BLOCKER 3: Verify Metrics Endpoint

**Check Metrics Endpoint**
```powershell
# Should return Prometheus metrics format
Invoke-WebRequest -Uri "http://localhost:8000/metrics" -UseBasicParsing
```

**Verify Required Metrics Present**
```powershell
$response = Invoke-WebRequest -Uri "http://localhost:8000/metrics" -UseBasicParsing
$content = $response.Content

# Check for required metrics
$content | Select-String "risk_gate_decisions_total"
$content | Select-String "ess_triggers_total"
$content | Select-String "exchange_failover_events_total"
$content | Select-String "stress_scenario_executions_total"
```

**Expected Output** (example):
```
risk_gate_decisions_total{decision="allowed"} 0.0
risk_gate_decisions_total{decision="blocked"} 0.0
ess_triggers_total 0.0
exchange_failover_events_total 0.0
stress_scenario_executions_total{scenario="capital_loss"} 0.0
```

---

### STEP 4: Re-Run Preflight Checks

```powershell
# Re-run preflight checks
python scripts/preflight_check.py

# Check exit code
echo $LASTEXITCODE
# Expected: 0 (success)
```

**Expected Output**:
```
==============================================================================
QUANTUM TRADER v2.0 - PRE-FLIGHT CHECK
==============================================================================

Running pre-flight checks...

✅ PASS | check_health_endpoints
✅ PASS | check_risk_system
✅ PASS | check_exchange_connectivity
✅ PASS | check_database_redis
✅ PASS | check_observability
✅ PASS | check_stress_scenarios

==============================================================================
RESULTS: 6 passed, 0 failed
==============================================================================

✅ PRE-FLIGHT CHECK PASSED - System ready for trading
```

---

## TROUBLESHOOTING

### Issue: Backend fails to start

**Check Python environment**:
```powershell
# Verify Python installed
python --version

# Verify uvicorn installed
python -m pip show uvicorn

# Install if missing
python -m pip install uvicorn
```

**Check port availability**:
```powershell
# Check if port 8000 is already in use
netstat -ano | findstr :8000

# If port in use, kill process or use different port
# Kill process: taskkill /PID <PID> /F
# Or start on different port: --port 8001
```

**Check for Python errors**:
```powershell
# Run with verbose logging
python -m uvicorn backend.main:app --host localhost --port 8000 --log-level debug
```

---

### Issue: Health endpoints return 404

**Possible causes**:
1. Backend service not fully started (wait 10-30 seconds)
2. Endpoints not implemented
3. Wrong URL or port

**Verify endpoints exist**:
```powershell
# Check backend code for health routes
Get-Content backend\main.py | Select-String "health"
Get-Content backend\infra\health\*.py | Select-String "live|ready"
```

---

### Issue: Metrics endpoint returns 404

**Possible causes**:
1. Metrics not enabled
2. Prometheus client not installed
3. Metrics route not registered

**Verify metrics setup**:
```powershell
# Check if prometheus_client installed
python -m pip show prometheus-client

# Install if missing
python -m pip install prometheus-client

# Check metrics configuration
Get-Content backend\infra\observability\metrics.py
```

---

### Issue: Required metrics missing from endpoint

**Possible causes**:
1. Metrics not instrumented
2. No activity yet (counters at 0 but present)
3. Metrics not registered

**Verify metrics instrumentation**:
```powershell
# Check if metrics defined
Get-Content backend\infra\observability\metrics.py | Select-String "risk_gate_decisions_total|ess_triggers_total"

# Check if metrics exported
Get-Content backend\infra\observability\metrics.py | Select-String "__all__"
```

---

## VERIFICATION CHECKLIST

After running commands, verify:

- [ ] Backend service running (check Task Manager or `Get-Process`)
- [ ] Health liveness endpoint returns 200
- [ ] Health readiness endpoint returns 200
- [ ] Metrics endpoint returns 200
- [ ] Required metrics present in metrics output:
  - [ ] risk_gate_decisions_total
  - [ ] ess_triggers_total
  - [ ] exchange_failover_events_total
  - [ ] stress_scenario_executions_total
- [ ] Preflight checks pass (6/6)
- [ ] Preflight script exits with code 0

---

## NEXT STEPS

Once all blockers resolved:

1. **Proceed to Phase 2**: Config Freeze
   ```powershell
   # Config freeze will be automated
   # Review CONFIG_FREEZE_SNAPSHOT.yaml
   ```

2. **Manual Verifications** (if needed):
   ```powershell
   # Verify exchange connectivity
   # Test Binance API
   
   # Verify database connection
   # Test PostgreSQL
   
   # Verify Redis connection
   # Test Redis ping
   ```

3. **Proceed to Phase 3**: GO-LIVE Activation
   ```powershell
   # Edit config/go_live.yaml
   # Set: activation_enabled: true
   
   # Run activation script
   python scripts/go_live_activate.py
   ```

---

## QUICK REFERENCE

| Task | Command |
|------|---------|
| **Start backend** | `python -m uvicorn backend.main:app --host localhost --port 8000` |
| **Check health** | `Invoke-WebRequest http://localhost:8000/health/live` |
| **Check metrics** | `Invoke-WebRequest http://localhost:8000/metrics` |
| **Run preflight** | `python scripts/preflight_check.py` |
| **Stop backend** | Ctrl+C (if foreground) or close window (if background) |
| **Automated resolution** | `.\scripts\resolve_blockers.ps1` |

---

**Document Version**: 1.0  
**Last Updated**: December 4, 2025
