# Manual VPS Deployment for AdaptiveLeverageEngine

## Issue Encountered
The VPS git repository and service files are owned by `root`, preventing automatic deployment. Files have been uploaded to `/tmp/` on the VPS and require manual steps with sudo access.

## Files Ready on VPS (/tmp/)
- ✅ exit_brain.py (with monitoring logs + Redis streaming)
- ✅ adaptive_leverage_config.py (tunable configuration)
- ✅ monitor_adaptive_leverage.py (monitoring script)
- ✅ ADAPTIVE_LEVERAGE_USAGE_GUIDE.md (documentation)
- ✅ deploy_adaptive.sh (deployment script)

## Manual Deployment Steps

### Step 1: SSH to VPS
```bash
ssh qt@46.224.116.254
```

### Step 2: Stop Services (if running)
```bash
sudo systemctl stop exitbrain_v3
# or stop all services:
# sudo systemctl stop quantum_trader
```

### Step 3: Backup Existing Files
```bash
cd ~/quantum_trader
sudo cp microservices/exitbrain_v3_5/exit_brain.py microservices/exitbrain_v3_5/exit_brain.py.backup
sudo cp microservices/exitbrain_v3_5/adaptive_leverage_engine.py microservices/exitbrain_v3_5/adaptive_leverage_engine.py.backup
```

### Step 4: Deploy New Files
```bash
sudo cp /tmp/exit_brain.py microservices/exitbrain_v3_5/exit_brain.py
sudo cp /tmp/adaptive_leverage_config.py microservices/exitbrain_v3_5/adaptive_leverage_config.py
sudo cp /tmp/monitor_adaptive_leverage.py monitor_adaptive_leverage.py
sudo cp /tmp/ADAPTIVE_LEVERAGE_USAGE_GUIDE.md ADAPTIVE_LEVERAGE_USAGE_GUIDE.md
```

### Step 5: Set Permissions
```bash
sudo chmod 644 microservices/exitbrain_v3_5/exit_brain.py
sudo chmod 644 microservices/exitbrain_v3_5/adaptive_leverage_config.py
sudo chmod 755 monitor_adaptive_leverage.py
sudo chown -R qt:qt microservices/exitbrain_v3_5/
sudo chown qt:qt monitor_adaptive_leverage.py ADAPTIVE_LEVERAGE_USAGE_GUIDE.md
```

### Step 6: Validate Imports
```bash
cd ~/quantum_trader
python3 -c "from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine; print('✅ Import successful')"
python3 -c "from microservices.exitbrain_v3_5.adaptive_leverage_config import get_config; print('✅ Config valid')"
```

Expected output:
```
✅ Import successful
✅ Config valid
```

### Step 7: Run Unit Tests
```bash
python3 -m pytest microservices/exitbrain_v3_5/tests/test_adaptive_leverage_engine.py -v
```

Expected output:
```
test_lsf_decreases_with_leverage PASSED
test_low_leverage_higher_tp PASSED
test_high_leverage_wider_sl PASSED
test_clamps_work PASSED
test_harvest_schemes PASSED
test_tp_progression PASSED
test_volatility_adjustment PASSED
test_funding_adjustment PASSED
========== 8 passed in 0.05s ==========
```

### Step 8: Restart Services
```bash
sudo systemctl restart exitbrain_v3
# or restart all:
# sudo systemctl restart quantum_trader
```

### Step 9: Verify Service is Running
```bash
sudo systemctl status exitbrain_v3
```

Expected output:
```
● exitbrain_v3.service - ExitBrain v3 Service
   Loaded: loaded
   Active: active (running) since...
```

### Step 10: Check Logs for Adaptive Levels
```bash
tail -f ~/quantum_trader/logs/exitbrain_v3.log | grep "Adaptive Levels"
```

Expected output (when signals processed):
```
[INFO] [ExitBrain-v3.5] Adaptive Levels | BTCUSDT 20.0x | LSF=0.2472 | TP1=0.85% TP2=1.32% TP3=1.86% | SL=0.80% | Harvest=[0.4, 0.4, 0.2]
```

### Step 11: Verify Redis Stream
```bash
redis-cli XINFO STREAM quantum:stream:adaptive_levels
```

Expected output:
```
 1) "length"
 2) (integer) X  # (some number > 0 after signals processed)
 3) "first-entry"
 4) ...
```

### Step 12: Start Monitoring (24h Watch Mode)
```bash
cd ~/quantum_trader
python3 monitor_adaptive_leverage.py watch
```

This will run in real-time monitoring mode, showing:
- Recent adaptive level calculations
- SL clamps and TP minimums alerts
- Harvest scheme distribution
- Per-symbol statistics

**Press Ctrl+C to stop monitoring**

---

## Alternative: Scripted Deployment

If you have sudo access without password, run:
```bash
cd /tmp
sudo bash deploy_adaptive.sh
```

---

## Monitoring After Deployment

### Quick Analysis (Last 100 Calculations)
```bash
python3 monitor_adaptive_leverage.py 100
```

### Watch Mode (Real-Time)
```bash
python3 monitor_adaptive_leverage.py watch
```

### Check Specific Symbol
```bash
redis-cli XREVRANGE quantum:stream:adaptive_levels + - COUNT 10 | grep BTCUSDT
```

---

## Tuning Parameters (After 24h)

If you need to adjust parameters based on production data:

1. **Edit configuration:**
   ```bash
   sudo vim ~/quantum_trader/microservices/exitbrain_v3_5/adaptive_leverage_config.py
   ```

2. **Common adjustments:**
   - Increase `BASE_TP_PCT` from 0.01 to 0.012 if TP levels too tight
   - Increase `BASE_SL_PCT` from 0.005 to 0.007 if SL too tight
   - Adjust harvest schemes if TP3 rarely hit

3. **Validate changes:**
   ```bash
   python3 ~/quantum_trader/microservices/exitbrain_v3_5/adaptive_leverage_config.py
   ```

4. **Restart service:**
   ```bash
   sudo systemctl restart exitbrain_v3
   ```

---

## Troubleshooting

### Issue: Import errors
**Solution:** Check Python path and ensure all files copied correctly
```bash
cd ~/quantum_trader
python3 -c "import sys; print('\n'.join(sys.path))"
```

### Issue: Redis stream not populating
**Solution:** Verify ExitBrain is processing signals
```bash
tail -f logs/exitbrain_v3.log | grep "build_exit_plan"
```

### Issue: Service won't start
**Solution:** Check service logs
```bash
sudo journalctl -u exitbrain_v3 -n 50 --no-pager
```

### Issue: Permission denied errors
**Solution:** Fix ownership
```bash
sudo chown -R qt:qt ~/quantum_trader/microservices/exitbrain_v3_5/
sudo chown qt:qt ~/quantum_trader/monitor_adaptive_leverage.py
```

---

## Integration Extensions (Optional)

### Position Monitor Integration
File: `backend/services/monitoring/position_monitor.py`

Add dynamic TP/SL updates using adaptive levels:
```python
from backend.domains.exits.exit_brain_v3.v35_integration import get_v35_integration

v35 = get_v35_integration()
adaptive_levels = v35.compute_adaptive_levels(
    leverage=position.leverage,
    volatility_factor=self._calculate_volatility()
)
# Use adaptive_levels for dynamic TP/SL adjustment
```

### Event-Driven Executor Integration
File: `backend/services/execution/event_driven_executor.py`

Verify adaptive levels flow through execution pipeline:
```bash
grep -n "ExitRouter\|adaptive" backend/services/execution/event_driven_executor.py
```

---

## Success Indicators

After 24 hours of monitoring, check for:
- ✅ SL clamps <5% of calculations
- ✅ TP minimums <3% of calculations
- ✅ Harvest schemes match leverage tiers
- ✅ LSF decreases with leverage
- ✅ No runtime errors in logs
- ✅ Redis stream populated continuously

---

## Contact & Support

For issues, check:
1. Logs: `~/quantum_trader/logs/exitbrain_v3.log`
2. Redis: `redis-cli XINFO STREAM quantum:stream:adaptive_levels`
3. Tests: `pytest -v test_adaptive_leverage_engine.py`
4. Config: `python3 adaptive_leverage_config.py`

**Status:** ✅ Files ready for deployment on VPS
**Next:** Execute manual steps above with sudo access
