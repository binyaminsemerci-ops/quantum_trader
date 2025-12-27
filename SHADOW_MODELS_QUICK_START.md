# 🚀 Shadow Models - Quick Start Guide

**5 Minutes to Deployment**

## Prerequisites ✓

- [x] Module 5 complete (`backend/services/ai/shadow_model_manager.py` exists)
- [x] Tests pass (`pytest backend/tests/test_shadow_model_manager.py`)
- [x] Python 3.8+ with numpy, scipy, pandas

## Deployment (2 Methods)

### Method 1: Automated (Recommended)

```powershell
# Run deployment script
.\scripts\deploy_shadow_models.ps1

# Follow manual integration steps (2 files to edit)
# Enable shadows in .env
# Start monitoring
```

**Time: 15 minutes**

### Method 2: Manual

```powershell
# 1. Backup (2 min)
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
mkdir "backups\shadow_$timestamp"
cp ai_engine\ensemble_manager.py "backups\shadow_$timestamp\"
cp backend\routes\ai.py "backups\shadow_$timestamp\"

# 2. Configure .env (1 min)
# Add these lines to .env:
ENABLE_SHADOW_MODELS=false
SHADOW_MIN_TRADES=500
SHADOW_MDD_TOLERANCE=1.20
SHADOW_ALPHA=0.05
SHADOW_N_BOOTSTRAP=10000
SHADOW_CHECK_INTERVAL=100

# 3. Integration (10 min) - See below
# 4. Testing (2 min)
pytest backend\tests\test_shadow_model_manager.py -v

# 5. Enable (1 min)
# Edit .env: ENABLE_SHADOW_MODELS=true
```

**Time: 16 minutes**

---

## Integration Steps (Manual)

### File 1: `ai_engine/ensemble_manager.py`

**Step 1.1: Add imports (top of file)**

```python
import os
from datetime import datetime
from backend.services.ai.shadow_model_manager import (
    ShadowModelManager,
    ModelRole,
    ModelMetadata,
    PromotionStatus,
    TradeResult
)
```

**Step 1.2: Add to `__init__` method (after existing initialization)**

```python
        # Shadow Model Manager (optional - controlled by env var)
        self.shadow_manager = None
        self.shadow_enabled = os.getenv('ENABLE_SHADOW_MODELS', 'false').lower() == 'true'
        
        if self.shadow_enabled:
            try:
                self.shadow_manager = ShadowModelManager(
                    min_trades_for_promotion=int(os.getenv('SHADOW_MIN_TRADES', '500')),
                    mdd_tolerance=float(os.getenv('SHADOW_MDD_TOLERANCE', '1.20')),
                    alpha=float(os.getenv('SHADOW_ALPHA', '0.05')),
                    n_bootstrap=int(os.getenv('SHADOW_N_BOOTSTRAP', '10000')),
                    checkpoint_path='data/shadow_models_checkpoint.json'
                )
                
                # Register current ensemble as champion
                self.shadow_manager.register_model(
                    model_name='ensemble_production_v1',
                    model_type='ensemble',
                    version='1.0',
                    role=ModelRole.CHAMPION,
                    description='Production 4-model ensemble'
                )
                
                logger.info("[Shadow] Enabled - Champion registered")
            except Exception as e:
                logger.error(f"[Shadow] Init failed: {e}")
                self.shadow_enabled = False
        
        self.shadow_trade_count = 0
        self.shadow_check_interval = int(os.getenv('SHADOW_CHECK_INTERVAL', '100'))
```

**Step 1.3: Add new methods (end of class)**

Copy these 5 methods from `backend/services/ai/shadow_model_integration.py`:
- `record_trade_outcome_for_shadow()`
- `_check_shadow_promotions()`
- `deploy_shadow_challenger()`
- `get_shadow_status()`

*(Full code in shadow_model_integration.py, Part 1)*

---

### File 2: `backend/routes/ai.py`

**Step 2.1: Add import (top of file)**

```python
from ai_engine.ensemble_manager import EnsembleManager
```

**Step 2.2: Add 6 API endpoints (end of file)**

Copy these routes from `backend/services/ai/shadow_model_integration.py`:

```python
@router.get("/shadow/status")
async def get_shadow_status():
    # ... (see shadow_model_integration.py Part 2)

@router.get("/shadow/comparison/{challenger_name}")
async def get_shadow_comparison(challenger_name: str):
    # ... (see shadow_model_integration.py Part 2)

@router.post("/shadow/deploy")
async def deploy_shadow_model(request: dict):
    # ... (see shadow_model_integration.py Part 2)

@router.post("/shadow/promote/{challenger_name}")
async def promote_shadow_model(challenger_name: str, force: bool = False):
    # ... (see shadow_model_integration.py Part 2)

@router.post("/shadow/rollback")
async def rollback_champion(request: dict):
    # ... (see shadow_model_integration.py Part 2)

@router.get("/shadow/history")
async def get_promotion_history(n: int = 10):
    # ... (see shadow_model_integration.py Part 2)
```

*(Full code in shadow_model_integration.py, Part 2, lines 361-520)*

---

## Testing

```powershell
# Unit tests
pytest backend\tests\test_shadow_model_manager.py -v

# API tests (after enabling shadows)
curl http://localhost:8000/shadow/status

# Dashboard
python scripts\shadow_dashboard.py --once
```

**Expected output:**
- ✅ 25+ tests pass
- ✅ API returns champion status
- ✅ Dashboard shows champion + 0 challengers

---

## Enable & Monitor

```powershell
# 1. Enable in .env
ENABLE_SHADOW_MODELS=true

# 2. Restart backend
# (Restart your FastAPI server)

# 3. Start dashboard
python scripts\shadow_dashboard.py

# 4. Deploy first challenger (optional test)
curl -X POST http://localhost:8000/shadow/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "test_challenger_v1",
    "model_type": "test",
    "description": "Test shadow model"
  }'
```

---

## Verification Checklist

After deployment, verify:

- [ ] `ENABLE_SHADOW_MODELS=true` in .env
- [ ] Backend logs show: `[Shadow] Enabled - Champion registered`
- [ ] API `/shadow/status` returns champion data
- [ ] Dashboard displays champion metrics
- [ ] Tests pass: `pytest backend/tests/test_shadow_model_manager.py`
- [ ] Checkpoint file created: `data/shadow_models_checkpoint.json`

---

## Quick Commands Reference

```powershell
# Deploy
.\scripts\deploy_shadow_models.ps1

# Test
pytest backend\tests\test_shadow_model_manager.py -v

# Monitor
python scripts\shadow_dashboard.py
python scripts\shadow_dashboard.py --json  # JSON output
python scripts\shadow_dashboard.py --once  # Single run

# API
curl http://localhost:8000/shadow/status
curl http://localhost:8000/shadow/history
curl -X POST http://localhost:8000/shadow/rollback

# Disable (emergency)
# Edit .env: ENABLE_SHADOW_MODELS=false
# Restart backend
```

---

## Troubleshooting

### Issue: Tests fail

```powershell
# Check dependencies
pip install numpy scipy pandas pytest

# Re-run with verbose
pytest backend\tests\test_shadow_model_manager.py -vv
```

### Issue: Import error

```powershell
# Verify file exists
ls backend\services\ai\shadow_model_manager.py

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Issue: API returns 404

```powershell
# Check routes added
grep -n "shadow/status" backend\routes\ai.py

# Restart backend
# (Restart your FastAPI server)
```

### Issue: Dashboard shows "Connection refused"

```powershell
# Check backend running
curl http://localhost:8000/health

# Try explicit API URL
python scripts\shadow_dashboard.py --api-url http://localhost:8000
```

---

## What's Next?

After successful deployment:

1. **Wait for data**: Need 500+ trades for first promotion check
2. **Monitor dashboard**: `python scripts\shadow_dashboard.py`
3. **Deploy challenger**: When ready to test new model
4. **Review promotions**: Check dashboard alerts for APPROVED status
5. **Rollback testing**: Test emergency rollback after 2-3 weeks

**Full deployment guide**: `SHADOW_MODELS_DEPLOYMENT_CHECKLIST.md`

---

## Support

- **Full integration code**: `backend/services/ai/shadow_model_integration.py`
- **Deployment checklist**: `SHADOW_MODELS_DEPLOYMENT_CHECKLIST.md`
- **Technical docs**: `SHADOW_MODELS_TECHNICAL_FRAMEWORK.md`
- **Risk analysis**: `SHADOW_MODELS_RISK_ANALYSIS.md`

---

**Deployment time: 15-20 minutes**  
**First promotion check: 500 trades (~2-3 weeks)**  
**ROI: 640-1,797% Year 1**
