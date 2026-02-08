# Calibration-Only Learning System

## 🎯 **Philosophy**

> *"Vi forbedrer beslutningskvalitet uten å endre modeller"*  
> *(We improve decision quality without changing models)*

This system improves AI trading performance through **config-only calibration** - no model retraining, no feature changes, instant rollback.

---

## 📋 **System Overview**

### **What It Does**

1. **Confidence Calibration** - Maps predicted confidence → actual win rate (isotonic regression)
2. **Ensemble Weight Adjustment** - Micro-adjusts model weights (±10% max) based on performance
3. **HOLD Bias Calibration** - [Optional, currently disabled] Adjusts HOLD signal confidence

### **What It Doesn't Do**

❌ **NO** model retraining  
❌ **NO** architecture changes  
❌ **NO** new features  
❌ **NO** deployment downtime

---

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────┐
│              SimpleCLM (Continuous Learning Monitoring)  │
│              Records every trade to clm_trades.jsonl     │
└────────────────────┬────────────────────────────────────┘
                     │ (50+ trades)
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Learning Cadence Monitor                    │
│              Gates access: must have 50 trades           │
└────────────────────┬────────────────────────────────────┘
                     │ (authorization: 'calibration')
                     ▼
┌─────────────────────────────────────────────────────────┐
│              CalibrationAnalyzer                         │
│  ├─ Load CLM data (clm_trades.jsonl)                    │
│  ├─ Calibrate confidence (isotonic regression)          │
│  ├─ Calibrate ensemble weights (±10% max)               │
│  └─ Validate safety checks (5 levels)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              CalibrationReportGenerator                  │
│  ├─ Generate markdown report (human-readable)           │
│  └─ Generate JSON export (machine-readable)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ [HUMAN APPROVAL REQUIRED]
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              CalibrationConfigDeployer                   │
│  ├─ Atomic deployment (tmp → rename)                    │
│  ├─ Version archiving (full history)                    │
│  └─ Rollback support (<2 minutes)                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              AI Engine (EnsembleManager)                 │
│  ├─ CalibrationLoader reads calibration.json            │
│  ├─ apply_confidence_calibration()                      │
│  └─ get_ensemble_weights()                              │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 **File Structure**

```
microservices/learning/
  ├─ calibration_types.py         # Data structures (400 lines)
  ├─ calibration_analyzer.py      # Core analysis engine (500 lines)
  ├─ calibration_report.py        # Report generation (400 lines)
  ├─ calibration_deployer.py      # Deployment system (350 lines)
  ├─ calibration_orchestrator.py  # Main coordinator (450 lines)
  └─ calibration_cli.py            # Command-line interface (400 lines)

ai_engine/
  └─ calibration_loader.py         # AI Engine integration (250 lines)

config/
  ├─ calibration.json              # Active configuration
  └─ calibration_archive/          # Version history
      ├─ cal_20260210_001_backup_*.json
      └─ cal_20260210_001_deployed.json
```

**Total:** ~2750 lines of production code

---

## 🚀 **Usage**

### **1. Check Readiness**

```bash
python microservices/learning/calibration_cli.py check
```

**Output:**
```
📊 LEARNING CADENCE READINESS CHECK
═══════════════════════════════════════════════════════════

🟢 Status: READY
   Reason: Trade count sufficient (52/50)
   Allowed actions: ['calibration']

✅ CALIBRATION AUTHORIZED
   You can run: python calibration_cli.py run
```

---

### **2. Run Calibration Analysis**

```bash
python microservices/learning/calibration_cli.py run
```

**Process:**
1. Checks Learning Cadence authorization
2. Loads SimpleCLM trade data (`clm_trades.jsonl`)
3. Runs confidence calibration (isotonic regression)
4. Runs ensemble weight calibration (±10% max adjustments)
5. Validates safety checks (5 levels)
6. Generates report (`/tmp/calibration_<version>.md`)

**Output:**
```
✅ CALIBRATION ANALYSIS COMPLETE
   Job ID: cal_20260210_143022
   Status: pending_approval
   Report: /tmp/calibration_cal_20260210_143022.md
   Validation: ✅ PASSED
   Risk: 12.5%

📋 NEXT STEPS:
1. Review report: cat /tmp/calibration_cal_20260210_143022.md
2. Approve: python calibration_cli.py approve cal_20260210_143022
3. Monitor for 24-48 hours
4. Rollback if needed: python calibration_cli.py rollback
```

---

### **3. Review Report**

```bash
cat /tmp/calibration_cal_20260210_143022.md
```

**Report Contents:**
- **Data Summary**: Number of trades, win rate, date range
- **Confidence Calibration**: Mapping table (predicted → calibrated)
- **Ensemble Weights**: Per-model adjustments with rationale
- **Validation Results**: Safety check results
- **Deployment Instructions**: CLI commands

**Example Confidence Table:**
```
| Predicted Confidence | Calibrated | Adjustment |
|---------------------|------------|------------|
| 0.70                | 0.679      | -0.021 ⬇️  |
| 0.80                | 0.788      | -0.012 ⬇️  |
| 0.90                | 0.897      | -0.003     |
```

**Example Weight Changes:**
```
| Model    | Before | After | Delta    | Change % | Reason                  |
|----------|--------|-------|----------|----------|-------------------------|
| xgb      | 0.300  | 0.274 | -0.026 ⬇️ | -8.7%    | Precision: 0.48, lower  |
| lgbm     | 0.300  | 0.326 | +0.026 ⬆️ | +8.7%    | Precision: 0.62, higher |
| nhits    | 0.200  | 0.200 | +0.000   | +0.0%    | Stable performance      |
| patchtst | 0.200  | 0.200 | +0.000   | +0.0%    | Stable performance      |
```

---

### **4. Approve and Deploy**

```bash
python microservices/learning/calibration_cli.py approve cal_20260210_143022
```

**What Happens:**
1. Shows summary and asks for confirmation
2. Creates backup of current config
3. Writes new config atomically (tmp → rename)
4. Archives deployed version
5. Signals AI Engine to reload (sentinel file)
6. Marks completion in Learning Cadence

**Deployment Time:** < 5 seconds  
**AI Engine Reload:** Automatic (watches sentinel file)

---

### **5. Monitor Performance**

**What to Watch:**
- **Win Rate**: Should stay stable or improve (target: >50%)
- **Confidence Alignment**: Check if predictions match outcomes better
- **Drawdown**: Should not increase significantly
- **META-V2 Logs**: Check for `[Calibration]` entries

**Monitoring Period:** 24-48 hours minimum

```bash
# Check AI Engine logs
tail -f /home/qt/quantum_trader/logs/ai_engine.log | grep -i calibration

# Check SimpleCLM data
tail -5 /home/qt/quantum_trader/data/clm_trades.jsonl
```

---

### **6. Rollback (If Needed)**

```bash
# Rollback to most recent backup
python microservices/learning/calibration_cli.py rollback

# Rollback to specific version
python microservices/learning/calibration_cli.py rollback cal_20260210_120000
```

**Rollback Time:** < 2 minutes  
**Effect:** AI Engine immediately uses previous configuration

---

## 🔒 **Safety Guarantees**

### **Validation Checks (5 Levels)**

1. **Confidence Improvement Check**
   - MSE must improve by ≥5%
   - If not, calibration disabled

2. **Weight Sum Check**
   - Ensemble weights must sum to 1.0 (±0.001 tolerance)
   - Critical failure if violated

3. **Weight Bounds Check**
   - No weight < 0.15 (15%)
   - No weight > 0.40 (40%)
   - Prevents over-reliance on single model

4. **Max Change Check**
   - No weight change > ±10%
   - Prevents extreme adjustments

5. **Outcome Diversity Check**
   - Must have ≥15% wins AND ≥15% losses
   - Ensures calibration uses diverse outcomes

### **Risk Scoring**

```python
risk_score = (
    critical_failures * 0.5 +
    error_failures * 0.3 +
    warnings * 0.1
) / total_checks

# Risk categories:
# 0.0-0.20: Low risk ✅
# 0.21-0.40: Medium risk ⚠️
# 0.41+: High risk ❌ (deployment blocked)
```

### **Rollback Capability**

- **Time:** < 2 minutes
- **Method:** Atomic rename (Unix guarantee)
- **History:** Full version archive maintained
- **Automation:** Manual approval required (no auto-rollback)

---

## 📊 **Performance Metrics**

### **Expected Improvements**

**Confidence Calibration:**
- **Target:** 5-15% MSE reduction
- **Impact:** Better risk management, more accurate position sizing

**Ensemble Weights:**
- **Target:** 2-6% total weight delta
- **Impact:** Better model utilization, reduced bias

**Overall:**
- **Win Rate:** +1-3% (52% → 53-55%)
- **Confidence Alignment:** Improved correlation with actual outcomes
- **Drawdown:** No increase (safety constraint)

---

## 🧪 **Testing**

### **Unit Tests**

```bash
# Test analyzer
python -m pytest tests/test_calibration_analyzer.py -v

# Test deployer
python -m pytest tests/test_calibration_deployer.py -v

# Test orchestrator
python -m pytest tests/test_calibration_orchestrator.py -v
```

### **Integration Test (Dry-Run)**

```bash
# Force run without authorization (testing only)
python microservices/learning/calibration_cli.py run --force
```

---

## 🔧 **Configuration**

### **Environment Variables**

```bash
# Calibration config path (default: /home/qt/quantum_trader/config/calibration.json)
export CALIBRATION_CONFIG_PATH=/path/to/calibration.json

# Learning Cadence API (default: http://127.0.0.1:8003)
export LEARNING_CADENCE_API=http://127.0.0.1:8003

# SimpleCLM data path (default: /home/qt/quantum_trader/data/clm_trades.jsonl)
export CLM_DATA_PATH=/path/to/clm_trades.jsonl

# Report output directory (default: /tmp)
export CALIBRATION_REPORT_DIR=/path/to/reports
```

### **Safety Constraints (Hardcoded)**

```python
MIN_TRADES_FOR_CALIBRATION = 50
MIN_CONFIDENCE_IMPROVEMENT = 0.05  # 5%
MAX_WEIGHT_CHANGE = 0.10  # ±10%
MIN_WEIGHT = 0.15  # 15%
MAX_WEIGHT = 0.40  # 40%
MIN_OUTCOME_DIVERSITY = 0.15  # 15%
```

**Note:** These are intentionally hardcoded to prevent accidental misconfiguration.

---

## 📈 **Roadmap**

### **Phase 1: Calibration-Only (CURRENT)**
- ✅ Confidence calibration
- ✅ Ensemble weight micro-adjustments
- ✅ Safety validation
- ✅ Manual approval workflow

### **Phase 2: Feedback Loop (FUTURE)**
- [ ] Automated A/B testing
- [ ] Performance metric tracking
- [ ] Automatic rollback on regression
- [ ] Multi-version comparison

### **Phase 3: Meta-Learning (FUTURE)**
- [ ] Learn calibration strategy itself
- [ ] Adaptive safety thresholds
- [ ] Market regime-specific calibration
- [ ] Continuous background calibration

---

## 🐛 **Troubleshooting**

### **"Learning Cadence not ready"**

**Cause:** Fewer than 50 trades recorded  
**Solution:** Wait for more trades to accumulate

```bash
# Check trade count
wc -l /home/qt/quantum_trader/data/clm_trades.jsonl
```

### **"Calibration validation failed"**

**Common Causes:**
1. **MSE Worsened:** Confidence calibration made predictions worse
   - **Solution:** Check data quality, may need more trades
   
2. **Weights Out of Bounds:** A model weight exceeded limits
   - **Solution:** Review model performance, may be outlier trades
   
3. **Insufficient Diversity:** Not enough wins/losses
   - **Solution:** Wait for more balanced outcomes

### **"Deployment failed"**

**Possible Causes:**
1. **File Permission Error:** Config file not writable
   - **Solution:** `chmod 664 /home/qt/quantum_trader/config/calibration.json`
   
2. **JSON Syntax Error:** Malformed configuration
   - **Solution:** Validate JSON with `jq . calibration.json`

### **"AI Engine not reloading config"**

**Debug Steps:**
1. Check sentinel file: `ls -la /home/qt/quantum_trader/config/.calibration_reload`
2. Check AI Engine logs: `tail -f logs/ai_engine.log | grep -i calibration`
3. Manually restart AI Engine: `systemctl restart quantum-ai-engine`

---

## 👥 **Contributing**

### **Code Style**

- **Python:** PEP 8 with 100-char line limit
- **Logging:** Use `logger.info()` for important events, `logger.debug()` for details
- **Type Hints:** Required for all public methods
- **Docstrings:** Google-style docstrings required

### **Adding New Calibration Types**

1. Add data structure to `calibration_types.py`
2. Implement analysis in `calibration_analyzer.py`
3. Add validation checks
4. Update report generator
5. Update AI Engine integration

---

## 📜 **License**

Proprietary - Quantum Trader System  
© 2026 - All Rights Reserved

---

## 📞 **Support**

For issues or questions:
- **GitHub Issues:** [Not yet public]
- **Internal Docs:** `/docs/calibration_system.md`
- **System Logs:** `/home/qt/quantum_trader/logs/`

---

**Last Updated:** 2026-02-10  
**Version:** 1.0.0  
**Status:** ✅ Production Ready (Pending VPS Deployment)
