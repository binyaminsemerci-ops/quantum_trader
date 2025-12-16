# RL v2 Operational Tools

**Quick access to monitoring, tuning, and testing tools for RL v2 deployment.**

---

## 🚀 Quick Start

### Deploy RL v2

```powershell
# Full deployment with verification
.\scripts\deploy_rl_v2.ps1

# Test only (no deployment)
.\scripts\deploy_rl_v2.ps1 -TestOnly

# Monitor only (existing deployment)
.\scripts\deploy_rl_v2.ps1 -MonitorOnly
```

---

## 📊 Monitoring Tools

### 1. Real-Time Dashboard

```powershell
# One-time snapshot
python scripts/monitor_rl_v2.py

# Continuous monitoring (60s updates)
python scripts/monitor_rl_v2.py --continuous

# Custom interval (30s updates)
python scripts/monitor_rl_v2.py --continuous --interval 30
```

**What you see:**
- Q-table growth (states, state-action pairs)
- Update counts
- Exploration rate (epsilon)
- Q-value statistics (min/max/avg)
- Learning progress
- Health checks

---

## 🔧 Hyperparameter Tuning

### 2. Get Recommendations

```powershell
# Analyze and get recommendations
python scripts/tune_rl_v2_hyperparams.py
```

### 3. Apply Hyperparameters

```powershell
# Apply recommended changes
python scripts/tune_rl_v2_hyperparams.py --apply \
  --meta-alpha 0.015 \
  --meta-epsilon 0.08

# Apply custom hyperparameters (all agents)
python scripts/tune_rl_v2_hyperparams.py --apply \
  --meta-alpha 0.01 \
  --meta-gamma 0.99 \
  --meta-epsilon 0.1 \
  --sizing-alpha 0.01 \
  --sizing-gamma 0.99 \
  --sizing-epsilon 0.1
```

**Hyperparameter reference:**

| Parameter | Early (< 100 updates) | Active (100-1000) | Mature (> 1000) |
|-----------|----------------------|-------------------|-----------------|
| Alpha (α) | 0.01-0.02 | 0.01-0.015 | 0.001-0.005 |
| Gamma (γ) | 0.95-0.99 | 0.98-0.99 | 0.99-0.999 |
| Epsilon (ε) | 0.2-0.3 | 0.1-0.2 | 0.01-0.05 |

---

## 🔬 A/B Testing

### 4. Compare RL v1 vs v2

```powershell
# Run comparison
python scripts/ab_test_rl_v1_vs_v2.py \
  --v1-pnl 1000 --v1-winrate 0.55 --v1-sharpe 1.2 --v1-drawdown 150 \
  --v2-pnl 1200 --v2-winrate 0.58 --v2-sharpe 1.5 --v2-drawdown 120
```

### 5. View Historical Tests

```powershell
# Show all A/B test history
python scripts/ab_test_rl_v1_vs_v2.py --history
```

---

## 🧪 Testing

### 6. Run Integration Tests

```powershell
# Full test suite
$env:PYTHONPATH="c:\quantum_trader"
python tests/integration/test_rl_v2_pipeline.py

# Via deployment script
.\scripts\deploy_rl_v2.ps1 -TestOnly
```

---

## 📋 Common Workflows

### Daily Monitoring

```powershell
# Check Q-table growth
python scripts/monitor_rl_v2.py
```

### Weekly Maintenance

```powershell
# 1. Check hyperparameter recommendations
python scripts/tune_rl_v2_hyperparams.py

# 2. Apply if needed
python scripts/tune_rl_v2_hyperparams.py --apply <params>

# 3. Backup Q-tables
$date = Get-Date -Format 'yyyy-MM-dd'
Copy-Item data/rl_v2/*.json data/rl_v2/backups/weekly_$date/ -Force
```

### Monthly Review

```powershell
# 1. Run A/B test
python scripts/ab_test_rl_v1_vs_v2.py <metrics>

# 2. View historical trends
python scripts/ab_test_rl_v1_vs_v2.py --history

# 3. Consider advanced features if outperforming RL v1
```

---

## 📊 Monitoring Output Examples

### Initial State (No Training)

```
📊 META STRATEGY AGENT
  ⚠️  Q-table not found - agent not yet trained

📈 POSITION SIZING AGENT
  ⚠️  Q-table not found - agent not yet trained

🎯 OVERALL STATISTICS
  Total Updates:         0
  Total States:          0
  Total Q-Table Size:    0.00 KB
```

### After 1 Hour

```
📊 META STRATEGY AGENT
  States Learned:        24
  State-Action Pairs:    156
  Total Updates:         47
  Exploration Rate (ε):  0.0950
  Q-Value Range:         [-0.12, 2.46]
  Q-Table Size:          8.45 KB

📈 LEARNING PROGRESS
  Meta Agent:
    New States:          +8
    New Updates:         +15
```

---

## 🔧 Troubleshooting

### Q-Tables Not Growing

```powershell
# Check backend health
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Verify signals being generated
Get-Content logs/*.log | Select-String "SIGNAL_GENERATED"

# Check RL subscriber active
Get-Content logs/*.log | Select-String "RL v2 Subscriber"
```

### High Q-Value Variance

```powershell
# Lower gamma and alpha
python scripts/tune_rl_v2_hyperparams.py --apply \
  --meta-gamma 0.95 \
  --meta-alpha 0.005 \
  --sizing-gamma 0.95 \
  --sizing-alpha 0.005
```

### Agent Not Exploiting

```powershell
# Manually reduce epsilon
python scripts/tune_rl_v2_hyperparams.py --apply \
  --meta-epsilon 0.05 \
  --sizing-epsilon 0.05
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `RL_V2_IMPLEMENTATION.md` | Technical implementation details |
| `RL_V2_VERIFICATION_REPORT.md` | Quality assessment and verification |
| `RL_V2_QUICK_REFERENCE.md` | Quick usage guide |
| `RL_V2_DEPLOYMENT_OPERATIONS.md` | Comprehensive deployment guide |
| `RL_V2_TOOLS.md` | This file - operational tools reference |

---

## 🎯 Tool Summary

| Tool | Purpose | Usage Frequency |
|------|---------|----------------|
| `deploy_rl_v2.ps1` | Initial deployment | Once |
| `monitor_rl_v2.py` | Q-table monitoring | Daily |
| `tune_rl_v2_hyperparams.py` | Hyperparameter tuning | Weekly |
| `ab_test_rl_v1_vs_v2.py` | Performance comparison | Monthly |
| `test_rl_v2_pipeline.py` | Integration testing | On-demand |

---

## 📞 Quick Reference Commands

```powershell
# Deploy
.\scripts\deploy_rl_v2.ps1

# Monitor
python scripts/monitor_rl_v2.py --continuous

# Tune
python scripts/tune_rl_v2_hyperparams.py

# Test
.\scripts\deploy_rl_v2.ps1 -TestOnly

# Compare
python scripts/ab_test_rl_v1_vs_v2.py --history
```

---

**System Status**: 🟢 PRODUCTION READY  
**Last Updated**: December 2, 2025  
**Version**: 2.0
