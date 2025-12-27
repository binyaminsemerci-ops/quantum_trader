# Exit Brain V3 LIVE Mode Deployment Guide

**Date**: December 10, 2025  
**Status**: Ready for Deployment  
**Phase**: 2B - LIVE Mode with Triple-Layer Safety

---

## 🎯 Quick Start

### Option 1: Automated Deployment (Recommended)

```powershell
# Step 1: Deploy to SHADOW mode (observation only - 24-48h)
.\scripts\deploy_exit_brain_v3_live.ps1 -Mode shadow

# Step 2: Analyze shadow logs after 24-48h
python backend\tools\analyze_exit_brain_shadow.py

# Step 3: Pre-LIVE validation (test kill-switch)
.\scripts\deploy_exit_brain_v3_live.ps1 -Mode prelive

# Step 4: Enable LIVE mode (AI controls exits)
.\scripts\deploy_exit_brain_v3_live.ps1 -Mode live
```

### Option 2: Manual Deployment

See full runbook: `docs/EXIT_BRAIN_V3_ACTIVATION_RUNBOOK.md`

---

## 📊 Deployment Scripts

### Main Deployment Script
**File**: `scripts/deploy_exit_brain_v3_live.ps1`

**Modes**:
- `shadow`: AI observes, logs decisions, NO orders
- `prelive`: Test LIVE config with kill-switch active  
- `live`: AI places real orders, legacy modules blocked

**Features**:
- ✅ Automatic .env backup
- ✅ Environment variable configuration
- ✅ Backend restart with health checks
- ✅ Exit Brain status diagnostics
- ✅ Mode-specific next steps guidance
- ✅ Safety confirmation for LIVE mode

**Usage**:
```powershell
.\scripts\deploy_exit_brain_v3_live.ps1 -Mode <shadow|prelive|live>
```

### Emergency Rollback Script
**File**: `scripts/rollback_exit_brain_v3.ps1`

**Rollback Levels**:
- `kill-switch`: Disable LIVE rollout only (~5s fastest)
- `shadow`: Return to shadow mode (~10s)
- `legacy`: Complete rollback to legacy system (~30s)

**Usage**:
```powershell
.\scripts\rollback_exit_brain_v3.ps1 -RollbackLevel <kill-switch|shadow|legacy>
```

### Shadow Log Analyzer
**File**: `backend/tools/analyze_exit_brain_shadow.py`

**Analyses**:
- ✅ Decision type distribution
- ✅ Confidence scores by decision type
- ✅ Emergency exit patterns
- ✅ Symbol-specific behavior
- ✅ Temporal patterns
- ✅ LIVE mode readiness recommendation

**Usage**:
```powershell
python backend\tools\analyze_exit_brain_shadow.py
```

---

## 🛡️ Triple-Layer Safety System

Exit Brain V3 uses three independent toggles:

| Toggle | Values | Purpose |
|--------|--------|---------|
| `EXIT_MODE` | LEGACY / EXIT_BRAIN_V3 | Which system owns exits |
| `EXIT_EXECUTOR_MODE` | SHADOW / LIVE | Executor behavior |
| `EXIT_BRAIN_V3_LIVE_ROLLOUT` | DISABLED / ENABLED | Kill-switch |

**For AI to place orders, ALL THREE must be:**
```
EXIT_MODE=EXIT_BRAIN_V3
EXIT_EXECUTOR_MODE=LIVE
EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED
```

**Any misalignment = automatic fallback to safe mode**

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Phase 2B implementation verified (all tests passed)
- [ ] Backend running without errors
- [ ] Diagnostic tools available
- [ ] Team ready to monitor
- [ ] Rollback plan understood

### SHADOW Mode (24-48h)
- [ ] Deploy to shadow mode
- [ ] Verify shadow logs being written
- [ ] Monitor `[EXIT_BRAIN_SHADOW]` logs
- [ ] Collect shadow log file
- [ ] No executor crashes
- [ ] AI decisions look reasonable

### Pre-LIVE Validation
- [ ] Run pre-LIVE dry run
- [ ] Verify kill-switch works (stays in SHADOW)
- [ ] Check fallback message in logs
- [ ] Confirm safety system operational

### LIVE Mode Activation
- [ ] Shadow analysis confirms improvement
- [ ] Team ready for close monitoring
- [ ] Low-volume period selected
- [ ] Rollback scripts tested
- [ ] Enable LIVE mode
- [ ] Monitor 1-2 hours closely
- [ ] Verify AI execution logs
- [ ] Confirm legacy modules blocked
- [ ] Check metrics dashboard

### Post-Activation
- [ ] 24h stable operation
- [ ] No dual positions
- [ ] All positions have SL/TP
- [ ] PnL tracking positive
- [ ] Document any issues

---

## 🔍 Monitoring Commands

### Real-time Log Monitoring
```powershell
# Watch all EXIT-related logs
tail -f backend/logs/quantum_trader.log | grep -E "EXIT_BRAIN|EXIT_GUARD|EXIT_MODE"

# SHADOW mode - watch AI decisions
tail -f backend/logs/quantum_trader.log | grep EXIT_BRAIN_SHADOW

# LIVE mode - watch AI executions
tail -f backend/logs/quantum_trader.log | grep EXIT_BRAIN_LIVE

# Watch blocked legacy modules
tail -f backend/logs/quantum_trader.log | grep "BLOCKED"
```

### Status Checks
```powershell
# CLI diagnostic tool
python backend\tools\print_exit_status.py

# HTTP health endpoint
curl http://localhost:8000/health/exit_brain_status | jq

# General health
curl http://localhost:8000/health | jq
```

### Log File Locations
- Main logs: `backend/logs/quantum_trader.log`
- Shadow logs: `backend/data/exit_brain_shadow.jsonl`
- Deployment logs: `backend/logs/deployment_*.log`
- Rollback logs: `backend/logs/rollback_*.log`

---

## 🚨 Emergency Rollback

### Quick Rollback (choose one)

**Option A: Kill-Switch (5s)**
```powershell
.\scripts\rollback_exit_brain_v3.ps1 -RollbackLevel kill-switch
```
Sets `EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED`

**Option B: Shadow Mode (10s)**
```powershell
.\scripts\rollback_exit_brain_v3.ps1 -RollbackLevel shadow
```
Sets `EXIT_EXECUTOR_MODE=SHADOW`

**Option C: Legacy Mode (30s)**
```powershell
.\scripts\rollback_exit_brain_v3.ps1 -RollbackLevel legacy
```
Sets `EXIT_MODE=LEGACY`

### Manual Rollback
1. Edit `.env` file
2. Change appropriate toggle(s)
3. Restart backend: `pkill -f "python.*main.py"; python backend/main.py`

---

## 📈 Success Criteria

### SHADOW Mode Success
- ✅ System runs 24-48h without crashes
- ✅ Shadow logs populated with diverse decisions
- ✅ Average confidence > 0.7
- ✅ Emergency exit rate < 30%
- ✅ Decisions appear reasonable
- ✅ No obvious logic errors

### LIVE Mode Success  
- ✅ AI successfully places exit orders
- ✅ Legacy modules blocked (no conflicts)
- ✅ No dual positions
- ✅ All positions protected with SL/TP
- ✅ PnL improvement vs legacy
- ✅ No system crashes or errors
- ✅ Metrics show expected behavior

---

## 📚 Documentation

- **Activation Runbook**: `docs/EXIT_BRAIN_V3_ACTIVATION_RUNBOOK.md` (650+ lines)
- **Phase 2B Technical**: `AI_EXIT_BRAIN_PHASE2B_COMPLETE.md`
- **Test Results**: `test_exit_brain_phase2b.py` (all passed)
- **This Guide**: `docs/EXIT_BRAIN_V3_DEPLOYMENT_GUIDE.md`

---

## 🎯 Current Status

**Implementation**: ✅ COMPLETE  
**Testing**: ✅ ALL TESTS PASSED  
**Documentation**: ✅ COMPLETE  
**Deployment**: ⏳ READY TO START

**Next Step**: Deploy to SHADOW mode for 24-48h validation

```powershell
.\scripts\deploy_exit_brain_v3_live.ps1 -Mode shadow
```

---

## 💡 Tips

1. **Start Conservative**: Always begin with SHADOW mode
2. **Monitor Closely**: First 1-2 hours of LIVE mode are critical
3. **Use Rollback Scripts**: Pre-tested emergency procedures
4. **Analyze Before LIVE**: Review shadow logs thoroughly
5. **Low-Volume Activation**: Enable LIVE during quiet market periods
6. **Keep Backups**: Deployment script auto-backs up .env
7. **Document Issues**: Track any unexpected behavior
8. **Gradual Confidence**: Success in SHADOW → Pre-LIVE → LIVE

---

## ❓ Troubleshooting

### Backend Won't Start
- Check logs: `backend/logs/*.log`
- Verify Python environment: `python --version`
- Check port availability: `netstat -an | findstr 8000`

### Shadow Logs Not Populating
- Verify EXIT_MODE=EXIT_BRAIN_V3
- Check executor is running: `curl http://localhost:8000/health/exit_brain_status`
- Ensure open positions exist to monitor
- Check file permissions on `backend/data/`

### LIVE Mode Not Activating
- Verify ALL THREE toggles aligned
- Check diagnostic endpoint
- Review startup logs for fallback messages
- Confirm no typos in .env values

### Legacy Modules Not Blocked
- Verify `EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED`
- Check gateway logs for `[EXIT_GUARD]` messages
- Restart backend if config changed after startup

---

**End of Deployment Guide**
