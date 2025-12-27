# 📊 TFT Model Monitoring Guide

**Model Version**: v1.1  
**Deployed**: 2025-11-19  
**Next Review**: 2025-11-26

---

## 🔍 Daily Monitoring

### Check Live Signals
Monitor TFT predictions and R/R ratios:

```powershell
python scripts/monitor_tft_signals.py
```

**What to look for**:
- ✅ Average R/R ratio ≥ 1.2:1
- ✅ ≥20% of signals with excellent R/R (>2.0)
- ⚠️ <30% of signals with poor R/R (0.7-1.3)
- ❌ Average R/R < 1.0 → Consider retraining

**Frequency**: Daily or whenever market is active

---

## 📈 Weekly Performance Review

### Manual Review
Run comprehensive performance analysis:

```powershell
python scripts/performance_review.py
```

**What to check**:
1. **Win Rate**: Target ≥50%
2. **Sharpe Ratio**: Target ≥1.0
3. **Average Return**: Should be positive
4. **Quantile Calibration**: P10 coverage 8-12%

### Automated Review (Recommended)
Schedule weekly reviews every Tuesday at 9 AM:

```powershell
# Run once to set up automation
.\scripts\schedule_review.ps1
```

This creates a Windows scheduled task that runs automatically.

**Review Schedule**:
- First review: 2025-11-26 (Tuesday 9:00 AM)
- Frequency: Weekly (every Tuesday)
- Duration: ~1 minute

---

## 🎯 Success Criteria (Week 1)

Model is performing well if:

| Metric | Target | Action if Below |
|--------|--------|-----------------|
| Win Rate | ≥50% | Monitor another week |
| Sharpe Ratio | ≥1.0 | Check risk management |
| Avg R/R Ratio | ≥1.2:1 | Increase quantile_weight |
| P10 Coverage | 8-12% | Retrain with higher weight |
| Daily Loss | <$250 | Review risk limits |

---

## 🚨 Warning Signs

### Immediate Action Required
❌ **Win rate < 40%** for 3+ days
- Review recent predictions vs outcomes
- Check if market regime changed
- Consider rolling back to v1.0

❌ **Multiple consecutive losing days**
- Enable kill switch if losses > $250/day
- Review position sizes
- Check for model errors in logs

❌ **Model crashes or errors**
- Check backend logs: `docker logs quantum_backend`
- Verify model file integrity
- Rollback if necessary

### Monitor Closely
⚠️ **Average R/R < 1.0** for 2+ days
- Quantile predictions may be off
- Consider increasing quantile_weight to 0.7-0.8

⚠️ **Excellent R/R signals < 10%**
- Model not capturing asymmetric upside
- May need retraining with more data

⚠️ **Poor R/R signals > 40%**
- Too many symmetric predictions
- Increase quantile_weight or retrain

---

## 🔧 Troubleshooting

### No Signals in Execution Journal

**Problem**: `monitor_tft_signals.py` shows no signals

**Solutions**:
1. Check if backend is running:
   ```powershell
   docker ps | Select-String quantum_backend
   ```

2. Verify backend health:
   ```powershell
   Invoke-RestMethod http://localhost:8000/health
   ```

3. Check if event-driven mode is active:
   ```powershell
   # Should see "event_driven_active": true
   ```

4. Monitor backend logs:
   ```powershell
   docker logs quantum_backend --tail 100 --follow
   ```

### Predictions All Identical

**Problem**: Model produces same prediction for different symbols

**Root Cause**: Normalization stats not loaded or corrupted

**Solutions**:
1. Verify checkpoint has normalization stats:
   ```powershell
   python scripts/check_checkpoint.py
   ```

2. Check agent loads stats correctly:
   ```powershell
   # Look for "✅ Loaded normalization stats from checkpoint"
   docker logs quantum_backend | Select-String "normalization"
   ```

3. If missing, restore from backup or retrain

### Poor Quantile Calibration

**Problem**: P10/P90 coverage at 85% instead of 10%

**Solutions**:
1. Increase quantile_weight in training:
   ```python
   # In train_tft_quantile.py line 51:
   self.quantile_weight = 0.7  # Up from 0.5
   ```

2. Retrain model:
   ```powershell
   python scripts/train_tft_quantile.py
   ```

3. This is not critical for classification (BUY/SELL/HOLD)
   - Can wait until weekly review
   - Only retrain if R/R analysis clearly failing

---

## 📊 Database Queries

### Check Recent Signals (SQL)
```sql
SELECT 
    timestamp,
    symbol,
    action,
    confidence,
    json_extract(metadata, '$.risk_reward_ratio') as rr_ratio,
    json_extract(metadata, '$.upside') as upside,
    json_extract(metadata, '$.downside') as downside
FROM execution_journal 
WHERE agent = 'TFTAgent'
    AND timestamp > datetime('now', '-24 hours')
ORDER BY timestamp DESC
LIMIT 20;
```

### Count Signals by Action
```sql
SELECT 
    action,
    COUNT(*) as count,
    AVG(confidence) as avg_conf,
    AVG(CAST(json_extract(metadata, '$.risk_reward_ratio') AS REAL)) as avg_rr
FROM execution_journal
WHERE agent = 'TFTAgent'
    AND timestamp > datetime('now', '-7 days')
GROUP BY action;
```

---

## 🔄 Retraining Decision Tree

```
Is win rate < 40%?
├─ YES → Consider retraining
│   ├─ Are predictions accurate but timing off?
│   │   └─ NO → Retrain with more recent data
│   └─ Are predictions consistently wrong?
│       └─ YES → Retrain with higher quantile_weight (0.7-0.8)
│
└─ NO → Is avg R/R < 1.0?
    ├─ YES → Increase quantile_weight and retrain
    └─ NO → Monitor for another week
```

### When to Retrain

**Immediate** (within 24-48 hours):
- Win rate < 30% for 3+ days
- Model crashes or produces errors
- Predictions clearly degraded vs v1.0

**Scheduled** (next week):
- Win rate 40-50% consistently
- Avg R/R < 1.2:1
- Poor quantile calibration (P10 coverage >20%)

**Optional** (consider after 2-3 weeks):
- Win rate 50-60% but want to improve
- Market regime shift (new patterns emerging)
- More training data available

---

## 📅 Review Timeline

| Date | Milestone | Action |
|------|-----------|--------|
| 2025-11-19 | Deployment | Monitor daily signals |
| 2025-11-26 | Week 1 Review | Run performance_review.py |
| 2025-12-03 | Week 2 Review | Decide: continue or retrain |
| 2025-12-10 | Week 3 Review | Assess long-term performance |
| 2025-12-17 | Month Review | Consider model updates |

---

## 💾 Backup and Rollback

### Create Backup
Before any changes:
```powershell
Copy-Item ai_engine/models/tft_model.pth `
    ai_engine/models/tft_model.pth.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')
```

### Rollback to v1.0
If v1.1 underperforms:
```powershell
# Find backup
Get-ChildItem ai_engine/models/*.backup* | Sort-Object LastWriteTime

# Restore (replace YYYYMMDD_HHMMSS with actual timestamp)
Copy-Item ai_engine/models/tft_model.pth.backup_YYYYMMDD_HHMMSS `
    ai_engine/models/tft_model.pth -Force

# Restart backend
docker-compose restart backend
```

---

## 📞 Quick Reference

**Daily monitoring**:
```powershell
python scripts/monitor_tft_signals.py
```

**Weekly review**:
```powershell
python scripts/performance_review.py
```

**Check backend**:
```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

**View logs**:
```powershell
docker logs quantum_backend --tail 50 --follow
```

**Retrain model**:
```powershell
python scripts/train_tft_quantile.py
```

---

**Last Updated**: 2025-11-19  
**Maintainer**: AI Trading Team  
**Model Version**: v1.1
