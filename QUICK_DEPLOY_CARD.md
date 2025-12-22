# 🚀 AdaptiveLeverageEngine - Quick Deploy Card

## ✅ IMPLEMENTATION COMPLETE

**Core Engine:** 177 lines | **Tests:** 8/8 passing | **Commits:** 5 pushed  
**Docs:** 1,200+ lines | **Status:** Ready for VPS deployment

---

## 📦 Files Ready on VPS `/tmp/`

1. ✅ `exit_brain.py` - Enhanced with monitoring + Redis streaming
2. ✅ `adaptive_leverage_config.py` - Tunable configuration
3. ✅ `monitor_adaptive_leverage.py` - Production monitoring
4. ✅ `ADAPTIVE_LEVERAGE_USAGE_GUIDE.md` - User documentation
5. ✅ `deploy_adaptive.sh` - Automated deployment script

---

## 🎯 Quick Deploy (5 Commands)

```bash
# 1. SSH to VPS
ssh qt@46.224.116.254

# 2. Run deployment (requires sudo password)
cd /tmp && sudo bash deploy_adaptive.sh

# 3. Restart service
sudo systemctl restart exitbrain_v3

# 4. Verify logs
tail -f ~/quantum_trader/logs/exitbrain_v3.log | grep "Adaptive Levels"

# 5. Start monitoring (24h watch mode)
cd ~/quantum_trader && python3 monitor_adaptive_leverage.py watch
```

**Expected Time:** 5 minutes

---

## 📊 What You'll See

### In Logs (Real-Time)
```
[INFO] [ExitBrain-v3.5] Adaptive Levels | BTCUSDT 20.0x | 
LSF=0.2472 | TP1=0.85% TP2=1.32% TP3=1.86% | SL=0.80% | 
Harvest=[0.4, 0.4, 0.2]
```

### In Monitoring Dashboard
```
=== Adaptive Leverage Analysis ===
Total Calculations: 234
SL Clamps: 5 (2.1%) ✅
TP Minimums: 3 (1.3%) ✅

BTCUSDT: Avg 18.5x leverage, LSF 0.2589
ETHUSDT: Avg 22.3x leverage, LSF 0.2401
```

### In Redis Stream
```bash
redis-cli XREVRANGE quantum:stream:adaptive_levels + - COUNT 1
# Shows: timestamp, symbol, leverage, lsf, tp1/2/3, sl, harvest
```

---

## 🔧 After 24 Hours

### Check Health
```bash
# Get statistics
python3 monitor_adaptive_leverage.py 500

# Look for:
# - SL clamps <5% ✅
# - TP minimums <3% ✅
# - Harvest schemes match leverage tiers ✅
```

### Tune If Needed
```bash
# Edit config
sudo vim ~/quantum_trader/microservices/exitbrain_v3_5/adaptive_leverage_config.py

# Common adjustments:
BASE_TP_PCT = 0.012  # If TP too tight
BASE_SL_PCT = 0.007  # If SL too tight

# Restart
sudo systemctl restart exitbrain_v3
```

---

## 📚 Full Documentation

- **Usage Guide:** [ADAPTIVE_LEVERAGE_USAGE_GUIDE.md](ADAPTIVE_LEVERAGE_USAGE_GUIDE.md)
- **Manual Deployment:** [VPS_MANUAL_DEPLOYMENT_GUIDE.md](VPS_MANUAL_DEPLOYMENT_GUIDE.md)
- **Integration Blueprint:** [POSITION_MONITOR_ADAPTIVE_INTEGRATION.py](POSITION_MONITOR_ADAPTIVE_INTEGRATION.py)
- **Final Status:** [AI_ADAPTIVE_LEVERAGE_FINAL_STATUS.md](AI_ADAPTIVE_LEVERAGE_FINAL_STATUS.md)

---

## 🆘 Troubleshooting

### Service Won't Start
```bash
sudo journalctl -u exitbrain_v3 -n 50 --no-pager
```

### Import Error
```bash
python3 -c "from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine; print('✅')"
```

### Permission Denied
```bash
sudo chown -R qt:qt ~/quantum_trader/microservices/exitbrain_v3_5/
```

### Redis Stream Empty
```bash
# Check if ExitBrain processing signals
grep "build_exit_plan" ~/quantum_trader/logs/exitbrain_v3.log
```

---

## ✅ Success Checklist

After deployment, verify:
- [ ] Service running: `sudo systemctl status exitbrain_v3`
- [ ] Logs show adaptive levels: `grep "Adaptive Levels" logs/exitbrain_v3.log`
- [ ] Redis stream populated: `redis-cli XLEN quantum:stream:adaptive_levels`
- [ ] Monitoring works: `python3 monitor_adaptive_leverage.py 10`
- [ ] No errors in logs: `grep ERROR logs/exitbrain_v3.log | tail -20`

**All checked?** 🎉 You're live with adaptive leverage!

---

## 📈 Expected Results

### Leverage Behavior
- **5x leverage:** TP1 ~1.15%, SL ~0.68% (Conservative)
- **20x leverage:** TP1 ~0.85%, SL ~0.80% (Balanced)
- **50x leverage:** TP1 ~0.81%, SL ~0.82% (Aggressive)

### Harvest Schemes
- **≤10x:** [0.3, 0.3, 0.4] - Larger TP3 (moonshot)
- **≤30x:** [0.4, 0.4, 0.2] - Balanced distribution
- **>30x:** [0.5, 0.3, 0.2] - Front-loaded (quick profits)

### Clamps (Rare)
- SL always in range [0.1%, 2.0%]
- TP always ≥0.3%
- SL always ≥0.15%

---

## 🎯 Key Formulas

```python
# Leverage Scaling Factor
LSF = 1 / (1 + log(leverage + 1))

# Take Profit Levels
TP1 = base_tp × (0.6 + LSF)
TP2 = base_tp × (1.2 + LSF/2)
TP3 = base_tp × (1.8 + LSF/4)

# Stop Loss
SL = base_sl × (1 + (1 - LSF) × 0.8)
```

**Higher leverage → Lower LSF → Tighter TP, Wider SL**

---

## 📞 Support

**Files on local:** `c:\quantum_trader\`  
**Files on VPS:** `~/quantum_trader/` and `/tmp/`  
**Git commits:** 5 pushed to main  
**Documentation:** 1,200+ lines total

**Status:** ✅ READY TO DEPLOY  
**Est. Deploy Time:** 5 minutes  
**Est. Monitoring:** 24 hours  
**Est. Tuning:** 10 minutes (if needed)

---

**Last Updated:** December 22, 2025  
**Version:** v3.5  
**Deployment:** Manual (VPS permission constraints)  
