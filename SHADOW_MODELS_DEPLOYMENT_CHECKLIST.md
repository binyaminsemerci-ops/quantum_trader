# SHADOW MODELS: DEPLOYMENT CHECKLIST

**Deployment Date:** _____________  
**Deployed By:** _____________  
**Environment:** Production / Staging (circle one)

---

## PRE-DEPLOYMENT (Complete Before Deployment)

### 1. Code Review ✓
- [ ] Review `backend/services/ai/shadow_model_manager.py` (1,050 lines)
- [ ] Review statistical tests (t-test, bootstrap, Sharpe, WR, MDD)
- [ ] Review promotion criteria (5 primary + 3 secondary)
- [ ] Review Thompson sampling implementation
- [ ] Review rollback mechanism (<30s requirement)
- [ ] Code approved by: ________________

### 2. Testing ✓
- [ ] Run unit tests: `pytest backend/tests/test_shadow_model_manager.py -v`
- [ ] All tests passed (25+ tests): ____/25 ✓
- [ ] Performance benchmarks met:
  - [ ] Statistical test latency <200ms: _____ ms
  - [ ] Promotion decision <500ms: _____ ms
  - [ ] Rollback <30s: _____ s
- [ ] Integration test passed (full promotion cycle)
- [ ] Rollback test passed (restoration verified)

### 3. Configuration ✓
- [ ] Set environment variables in `.env`:
  ```bash
  SHADOW_MIN_TRADES=500
  SHADOW_MDD_TOLERANCE=1.20
  SHADOW_ALPHA=0.05
  SHADOW_N_BOOTSTRAP=10000
  SHADOW_CHECK_INTERVAL=100
  ```
- [ ] Create checkpoint directory: `mkdir -p data/`
- [ ] Set checkpoint path: `data/shadow_models_checkpoint.json`
- [ ] Configure notification system for alerts
- [ ] Backup notification tested: _____ ✓

### 4. Documentation ✓
- [ ] Read `SHADOW_MODELS_SIMPLE_EXPLANATION.md`
- [ ] Read `SHADOW_MODELS_TECHNICAL_FRAMEWORK.md`
- [ ] Read `SHADOW_MODELS_INTEGRATION_GUIDE.md`
- [ ] Read `SHADOW_MODELS_RISK_ANALYSIS.md`
- [ ] Team trained on shadow model concept
- [ ] Runbook created for manual interventions

---

## DEPLOYMENT STEPS

### Phase 1: Infrastructure Setup (15 min)

#### Step 1.1: Install Dependencies
```bash
cd /quantum_trader
pip install scipy statsmodels numpy
```
- [ ] Dependencies installed
- [ ] Versions verified: `pip list | grep -E "scipy|statsmodels|numpy"`

#### Step 1.2: Deploy Shadow Manager
```bash
cp backend/services/ai/shadow_model_manager.py backend/services/ai/
```
- [ ] File copied
- [ ] Imports verified: `python -c "from backend.services.ai.shadow_model_manager import ShadowModelManager; print('OK')"`

#### Step 1.3: Deploy Tests
```bash
cp backend/tests/test_shadow_model_manager.py backend/tests/
pytest backend/tests/test_shadow_model_manager.py --co
```
- [ ] Tests discoverable (25+ tests listed)

---

### Phase 2: Integration (30 min)

#### Step 2.1: Backup Current Code
```bash
git add -A
git commit -m "Pre-shadow-models backup"
git push origin main
```
- [ ] Backup committed
- [ ] Commit SHA: ________________

#### Step 2.2: Integrate with Ensemble Manager
**Edit:** `ai_engine/ensemble_manager.py`

Add import:
```python
from backend.services.ai.shadow_model_manager import ShadowModelManager, ModelRole
```

Add to `__init__`:
```python
# Shadow model manager (optional, disabled by default)
self.shadow_manager = None
if os.getenv('ENABLE_SHADOW_MODELS', 'false').lower() == 'true':
    self.shadow_manager = ShadowModelManager(
        min_trades_for_promotion=500,
        mdd_tolerance=1.20,
        alpha=0.05,
        n_bootstrap=10000,
        checkpoint_path='data/shadow_models_checkpoint.json'
    )
    logger.info("[Shadow] Shadow model manager enabled")
```

- [ ] Code added to `ensemble_manager.py`
- [ ] Syntax verified: `python -m py_compile ai_engine/ensemble_manager.py`

#### Step 2.3: Add Shadow Prediction Tracking
**Edit:** `ai_engine/ensemble_manager.py` in `predict()` method

After existing prediction logic, add:
```python
# Record for shadow testing (if enabled)
if self.shadow_manager is not None:
    try:
        # Record champion prediction (this is tracked for comparison)
        # Actual outcome recorded later in trade callback
        pass  # Prediction stored, outcome recorded when trade closes
    except Exception as e:
        logger.error(f"[Shadow] Tracking failed: {e}")
```

- [ ] Shadow tracking added
- [ ] Error handling verified

#### Step 2.4: Add API Endpoints
**Edit:** `backend/routes/ai.py`

Add routes (copy from `SHADOW_MODELS_INTEGRATION_GUIDE.md` Section 3):
- [ ] `GET /shadow/status` - View all shadow models
- [ ] `GET /shadow/comparison/{challenger}` - Compare metrics
- [ ] `POST /shadow/deploy` - Deploy new challenger
- [ ] `POST /shadow/promote/{challenger}` - Manual promotion
- [ ] `POST /shadow/rollback` - Emergency rollback
- [ ] `GET /shadow/history` - Promotion history

Verification:
```bash
curl http://localhost:8000/shadow/status
```
- [ ] Endpoints accessible

---

### Phase 3: Initial Deployment (Test Mode)

#### Step 3.1: Enable Shadow Models (Staging Only)
```bash
# In .env file
ENABLE_SHADOW_MODELS=true
```
- [ ] Environment variable set
- [ ] Verified: `echo $ENABLE_SHADOW_MODELS`

#### Step 3.2: Register Champion Model
```bash
curl -X POST http://localhost:8000/shadow/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "ensemble_xgb_v1",
    "model_type": "xgboost",
    "description": "Current production ensemble (champion)"
  }'
```
- [ ] Champion registered
- [ ] Response: `{"status": "success", ...}`

#### Step 3.3: Verify Champion Status
```bash
curl http://localhost:8000/shadow/status | jq .
```
Expected response:
```json
{
  "status": "success",
  "data": {
    "champion": {
      "model_name": "ensemble_xgb_v1",
      "metrics": {...},
      "trade_count": 0
    },
    "challengers": []
  }
}
```
- [ ] Champion visible in status
- [ ] Metrics structure correct

#### Step 3.4: Deploy First Challenger (Test)
```bash
curl -X POST http://localhost:8000/shadow/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lightgbm_test_v1",
    "model_type": "lightgbm",
    "description": "Test challenger for shadow validation"
  }'
```
- [ ] Challenger deployed
- [ ] Allocation confirmed 0% (shadow mode)

---

### Phase 4: Monitoring Setup (20 min)

#### Step 4.1: Create Dashboard Script
**File:** `scripts/shadow_dashboard.py`

```python
#!/usr/bin/env python3
"""Real-time shadow model dashboard"""
import requests
import time
from datetime import datetime

def print_dashboard():
    resp = requests.get('http://localhost:8000/shadow/status')
    data = resp.json()['data']
    
    print("\n" + "="*80)
    print(f"SHADOW MODEL DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Champion
    champ = data['champion']
    print(f"\n🏆 CHAMPION: {champ['model_name']}")
    if champ['metrics']:
        print(f"   WR: {champ['metrics']['win_rate']:.2%}")
        print(f"   Sharpe: {champ['metrics']['sharpe_ratio']:.2f}")
        print(f"   Trades: {champ['trade_count']}")
    
    # Challengers
    print(f"\n🎯 CHALLENGERS: {len(data['challengers'])}")
    for chal in data['challengers']:
        print(f"\n   {chal['model_name']}:")
        print(f"   Status: {chal['promotion_status']}")
        print(f"   Score: {chal['promotion_score']:.1f}/100")
        print(f"   Trades: {chal['trade_count']}/500")
        if chal['metrics']:
            print(f"   WR: {chal['metrics']['win_rate']:.2%}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    while True:
        try:
            print_dashboard()
            time.sleep(60)  # Refresh every minute
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)
```

- [ ] Dashboard script created
- [ ] Executable: `chmod +x scripts/shadow_dashboard.py`
- [ ] Test run: `python scripts/shadow_dashboard.py`

#### Step 4.2: Configure Alerts
Edit notification system to include shadow events:
- [ ] Promotion alerts (priority: HIGH)
- [ ] Manual review alerts (priority: MEDIUM)
- [ ] Rollback alerts (priority: CRITICAL)
- [ ] Degradation alerts (priority: HIGH)

#### Step 4.3: Set Up Logging
Add to logging configuration:
```python
'shadow_models': {
    'level': 'INFO',
    'handlers': ['file', 'console'],
    'propagate': False
}
```
- [ ] Logging configured
- [ ] Log file created: `logs/shadow_models.log`
- [ ] Test log: `tail -f logs/shadow_models.log`

---

### Phase 5: Validation (2-3 weeks)

#### Step 5.1: Monitor First 100 Trades
Track champion performance:
- [ ] Day 1: ___ trades, WR: ____%
- [ ] Day 3: ___ trades, WR: ____%
- [ ] Day 7: ___ trades, WR: ____%
- [ ] Day 14: ___ trades, WR: ____%
- [ ] Day 21: ___ trades, WR: ____%

#### Step 5.2: Challenger Testing
Monitor challenger shadow predictions:
- [ ] Week 1: Challenger at ___ trades
- [ ] Week 2: Challenger at ___ trades
- [ ] Week 3: Challenger at 500+ trades ✓

#### Step 5.3: First Promotion Test
When challenger reaches 500 trades:
```bash
curl http://localhost:8000/shadow/comparison/lightgbm_test_v1 | jq .
```
- [ ] Promotion criteria checked
- [ ] Score calculated: ___/100
- [ ] Decision: APPROVED / PENDING / REJECTED

If APPROVED:
```bash
curl -X POST http://localhost:8000/shadow/promote/lightgbm_test_v1
```
- [ ] Promotion executed
- [ ] Champion updated
- [ ] Old champion archived
- [ ] Team notified

#### Step 5.4: Rollback Test (if promoted)
Monitor first 100 trades post-promotion:
- [ ] Trade 1-25: WR ____%
- [ ] Trade 26-50: WR ____%
- [ ] Trade 51-75: WR ____%
- [ ] Trade 76-100: WR ____%

If WR drops >5pp:
```bash
curl -X POST http://localhost:8000/shadow/rollback \
  -d '{"reason": "WR dropped >5pp post-promotion"}'
```
- [ ] Rollback triggered
- [ ] Previous champion restored
- [ ] Rollback time: ___ seconds (<30s required)

---

## POST-DEPLOYMENT VERIFICATION

### 6. System Health Checks

#### 6.1: Performance Metrics
- [ ] Statistical test latency <200ms: _____ ms ✓
- [ ] Promotion decision latency <500ms: _____ ms ✓
- [ ] Rollback speed <30s: _____ s ✓
- [ ] Memory usage <500MB: _____ MB ✓
- [ ] CPU overhead <5%: _____ % ✓

#### 6.2: Data Integrity
- [ ] Checkpoint file exists: `ls -lh data/shadow_models_checkpoint.json`
- [ ] Checkpoint backup created: `ls -lh data/shadow_models_checkpoint.backup`
- [ ] Trade history accurate: Verify sample trades
- [ ] Metrics computation correct: Manual verification

#### 6.3: Integration Tests
- [ ] Ensemble predictions unaffected (latency same as baseline)
- [ ] Shadow predictions generated (verify logs)
- [ ] No errors in application logs
- [ ] API endpoints responsive

#### 6.4: Notification System
- [ ] Test promotion alert: Manual trigger
- [ ] Test rollback alert: Manual trigger
- [ ] Alert delivery confirmed: Email/Slack/SMS
- [ ] Alert format correct: Readable, actionable

---

## ROLLBACK PLAN (If Issues Arise)

### Emergency Rollback Steps

**Symptom:** Shadow models causing errors/latency/crashes

**Action:**
1. **Disable shadow models immediately:**
   ```bash
   # In .env
   ENABLE_SHADOW_MODELS=false
   
   # Restart service
   docker restart quantum_backend
   ```
   - [ ] Shadow models disabled
   - [ ] Service restarted
   - [ ] Errors stopped

2. **Restore previous code:**
   ```bash
   git revert <commit_sha>
   git push origin main
   ```
   - [ ] Code reverted
   - [ ] Deployed to production

3. **Verify system health:**
   ```bash
   curl http://localhost:8000/health
   ```
   - [ ] System healthy
   - [ ] Predictions working
   - [ ] No shadow-related logs

4. **Post-mortem:**
   - [ ] Root cause identified: ________________
   - [ ] Fix implemented: ________________
   - [ ] Re-deployment scheduled: ________________

---

## SUCCESS CRITERIA (After 3 Months)

### Quantitative Metrics
- [ ] Models tested: ≥20 (target: 24-26)
- [ ] Promotions: ≥6 (target: 8-10)
- [ ] False promotion rate: <5% (target: <5%)
- [ ] Rollback rate: <10% (target: <10%)
- [ ] Prevented bad deployments: ≥3 ($60K+ saved)
- [ ] WR improvement: +3pp minimum (target: +5pp)
- [ ] Team time saved: ≥200 hours (target: 300+)

### Qualitative Metrics
- [ ] Team confidence increased (survey)
- [ ] Faster iteration perceived (survey)
- [ ] Zero production incidents from bad models
- [ ] Documentation complete and useful
- [ ] Process integrated into workflow

### Financial Metrics
- [ ] Prevented losses: $__,___ (target: $100K+)
- [ ] Incremental revenue: $__,___ (target: $70K+)
- [ ] ROI: ___% (target: >500%)
- [ ] Payback achieved: YES / NO (target: 2-3 months)

---

## SIGN-OFF

### Deployment Team
- [ ] **Developer:** ________________ Date: ______
- [ ] **QA Engineer:** ________________ Date: ______
- [ ] **DevOps:** ________________ Date: ______
- [ ] **Team Lead:** ________________ Date: ______

### Approvals
- [ ] **Technical Lead:** ________________ Date: ______
- [ ] **Product Owner:** ________________ Date: ______

### Post-Deployment Review (After 1 Month)
- [ ] **Review Date:** ________________
- [ ] **Attendees:** ________________
- [ ] **Status:** SUCCESS / PARTIAL / ISSUES
- [ ] **Action Items:** ________________

---

## NOTES / ISSUES

**Deployment Notes:**
_________________________________________________________________________
_________________________________________________________________________
_________________________________________________________________________

**Issues Encountered:**
_________________________________________________________________________
_________________________________________________________________________
_________________________________________________________________________

**Lessons Learned:**
_________________________________________________________________________
_________________________________________________________________________
_________________________________________________________________________

---

**Checklist Version:** 1.0  
**Last Updated:** November 26, 2025  
**Next Review:** After first 3 months of operation

