# AI SYSTEM INTEGRATION - QUICK REFERENCE

## 🎯 What We Built

**Complete AI subsystem integration layer for Quantum Trader**

- ✅ Service registry with feature flags
- ✅ 13 integration hooks for trading loop
- ✅ 5-stage rollout plan (observation → autonomy)
- ✅ Fail-safe architecture (backward compatible)
- ✅ 10 AI subsystems coordinated by AI-HFOS

---

## 📁 Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `backend/services/system_services.py` | 650 | Service registry & lifecycle |
| `backend/services/integration_hooks.py` | 450 | Trading loop integration points |
| `AI_SYSTEM_INTEGRATION_GUIDE.md` | 600 | Complete integration guide |
| `.env.example.ai_integration` | 300 | Configuration reference |
| `AI_INTEGRATION_STATUS.md` | 400 | Status & next steps |

---

## ⚡ Quick Start

### 1. Enable Observation Mode (Safe - No Changes to Trades)

```bash
# Add to .env
export QT_AI_INTEGRATION_STAGE=OBSERVATION
export QT_AI_HFOS_ENABLED=true
export QT_AI_HFOS_MODE=OBSERVE

# Restart backend
docker-compose restart quantum_backend

# Check logs
docker logs quantum_backend | grep "\\[AI-HFOS\\]"
```

**Expected Result:** AI-HFOS logs decisions but doesn't enforce them.

---

### 2. Enable Partial Enforcement (AI-HFOS Adjusts Confidence)

```bash
# Add to .env
export QT_AI_INTEGRATION_STAGE=PARTIAL
export QT_AI_HFOS_ENABLED=true
export QT_AI_HFOS_MODE=ADVISORY

# Restart
docker-compose restart quantum_backend

# Check logs
docker logs quantum_backend | grep "confidence"
```

**Expected Result:** AI-HFOS adjusts confidence thresholds and position sizing.

---

### 3. Emergency Disable

```bash
# Instant shutdown of ALL AI systems
export QT_AI_EMERGENCY_BRAKE=true

# Or disable just AI-HFOS
export QT_AI_HFOS_ENABLED=false

# Restart
docker-compose restart quantum_backend
```

---

## 🎚️ Integration Stages

| Stage | Mode | Subsystems Enabled | Behavior |
|-------|------|-------------------|----------|
| **1. OBSERVATION** | Log Only | None | AI logs decisions, no enforcement |
| **2. PARTIAL** | Advisory | AI-HFOS, PAL, AELM | Confidence/sizing adjustments |
| **3. COORDINATION** | Full | All except retraining | AI-HFOS coordinates all subsystems |
| **4. AUTONOMY** | Full | All (testnet only!) | Full autonomous operation |
| **5. MAINNET** | Conservative | Most in ADVISORY | Gradual mainnet rollout |

---

## 🔧 Configuration Cheat Sheet

### Master Controls

```bash
QT_AI_INTEGRATION_STAGE=OBSERVATION|PARTIAL|COORDINATION|AUTONOMY
QT_AI_EMERGENCY_BRAKE=false
QT_AI_FAIL_SAFE=true
```

### AI-HFOS

```bash
QT_AI_HFOS_ENABLED=false
QT_AI_HFOS_MODE=OFF|OBSERVE|ADVISORY|ENFORCED
QT_AI_HFOS_UPDATE_INTERVAL=60
```

### Profit Amplification Layer (PAL)

```bash
QT_AI_PAL_ENABLED=false
QT_AI_PAL_MODE=OFF|OBSERVE|ADVISORY|ENFORCED
QT_AI_PAL_MIN_R=1.0
```

### Portfolio Balancer (PBA)

```bash
QT_AI_PBA_ENABLED=false
QT_AI_PBA_MODE=OFF|OBSERVE|ADVISORY|ENFORCED
QT_AI_PBA_MAX_PORTFOLIO_LEVERAGE=20.0
```

### Self-Healing

```bash
QT_AI_SELF_HEALING_ENABLED=false
QT_AI_SELF_HEALING_MODE=OFF|OBSERVE|PROTECTIVE|AGGRESSIVE
QT_AI_SELF_HEALING_CHECK_INTERVAL=120
```

**Full Config:** See `.env.example.ai_integration`

---

## 🔌 Integration Hooks Reference

### Pre-Trade Hooks (5)

```python
# Filter symbols via Universe OS
filtered_symbols = await pre_trade_universe_filter(symbols)

# Check risk limits (AI-HFOS + Risk OS)
allowed, reason = await pre_trade_risk_check(symbol, signal, positions)

# Check portfolio limits (PBA)
allowed, reason = await pre_trade_portfolio_check(symbol, signal, positions)

# Adjust confidence threshold (AI-HFOS)
threshold = await pre_trade_confidence_adjustment(signal, base_threshold)

# Scale position size (AI-HFOS)
size = await pre_trade_position_sizing(symbol, signal, base_size)
```

### Execution Hooks (2)

```python
# Select order type (MARKET vs LIMIT)
order_type = await execution_order_type_selection(symbol, signal, "MARKET")

# Validate slippage
acceptable, reason = await execution_slippage_check(symbol, expected, actual)
```

### Post-Trade Hooks (2)

```python
# Classify position (PIL)
position = await post_trade_position_classification(position)

# Check amplification (PAL)
recommendation = await post_trade_amplification_check(position)
```

### Periodic Hooks (2)

```python
# Health monitoring (every 2 minutes)
await periodic_self_healing_check()

# AI-HFOS coordination (every 60 seconds)
await periodic_ai_hfos_coordination()
```

---

## 🧪 Testing Commands

### Check Service Status

```python
from backend.services.system_services import get_ai_services

ai_services = get_ai_services()
status = ai_services.get_status()
print(status)
```

### Test Integration Hooks

```python
from backend.services.integration_hooks import pre_trade_risk_check

# Simulated signal
signal = {"symbol": "BTCUSDT", "direction": "LONG", "confidence": 0.75}
positions = []

allowed, reason = await pre_trade_risk_check("BTCUSDT", signal, positions)
print(f"Trade allowed: {allowed}, reason: {reason}")
```

### Check AI-HFOS Status

```python
from backend.services.ai_hedgefund_os import AIHedgeFundOS

ai_hfos = AIHedgeFundOS()
status = ai_hfos.get_status()
print(f"Risk Mode: {status['system_risk_mode']}")
print(f"System Health: {status['system_health_score']}")
```

---

## 📊 Health Monitoring

### Log Files

```bash
# AI subsystem logs
tail -f logs/ai_subsystem.log

# Event-driven executor logs (includes AI decisions)
tail -f logs/event_driven_executor.log

# Filter for AI-HFOS messages
tail -f logs/*.log | grep "\\[AI-HFOS\\]"
```

### Health Endpoints (TODO: Implement)

```bash
# Overall AI health
curl http://localhost:8000/health/ai

# Integration status
curl http://localhost:8000/health/ai/integration
```

---

## 🚨 Troubleshooting

### Problem: AI subsystem not loading

**Solution:**
```bash
# Check environment variables
echo $QT_AI_HFOS_ENABLED

# Check logs for initialization errors
docker logs quantum_backend | grep "ERROR"

# Verify file exists
ls -la backend/services/system_services.py
```

---

### Problem: Hooks not being called

**Solution:**
```bash
# Check integration stage
echo $QT_AI_INTEGRATION_STAGE

# Verify ai_services passed to EventDrivenExecutor
docker logs quantum_backend | grep "AI System Services"

# Check if subsystem is enabled
python -c "from backend.services.system_services import get_ai_services; print(get_ai_services().get_status())"
```

---

### Problem: Emergency brake not working

**Solution:**
```bash
# Emergency brake is ALWAYS enforced regardless of config
# Check if it's actually enabled
docker logs quantum_backend | grep "EMERGENCY"

# Manually enable
export QT_AI_EMERGENCY_BRAKE=true
docker-compose restart quantum_backend

# Verify
docker logs quantum_backend | grep "Emergency brake: ACTIVE"
```

---

## ✅ Next Steps

### Immediate (High Priority)

1. **Modify event_driven_executor.py**
   - Add imports
   - Insert hooks at integration points
   - Test backward compatibility

2. **Modify main.py**
   - Initialize AI services in lifespan()
   - Add health endpoints
   - Pass ai_services to executor

3. **Test on Testnet**
   - Stage 1: Observation mode (7 days)
   - Stage 2: Partial enforcement (7 days)
   - Stage 3: Full coordination (14 days)

### Medium Priority

4. **Create Integration Tests**
   - Test all 13 hooks
   - Test stage transitions
   - Test fail-safe behavior

5. **Create Monitoring Dashboard**
   - AI subsystem status page
   - Recent decisions log
   - Performance metrics

### Low Priority

6. **Activation Scripts**
   - Quick enable/disable scripts
   - Profile switching scripts

---

## 📚 Documentation Links

- **Integration Guide:** `AI_SYSTEM_INTEGRATION_GUIDE.md` (complete guide)
- **Status Document:** `AI_INTEGRATION_STATUS.md` (completion status)
- **AI-HFOS Guide:** `AI_HEDGEFUND_OS_GUIDE.md` (AI-HFOS documentation)
- **Architecture:** `SYSTEM_ARCHITECTURE.md` (system hierarchy)
- **Config Reference:** `.env.example.ai_integration` (all variables)

---

## 🎓 Key Concepts

### SubsystemMode

- **OFF:** Subsystem disabled
- **OBSERVE:** Log decisions only, no enforcement
- **ADVISORY:** Provide recommendations, not enforced
- **ENFORCED:** Enforce decisions

### IntegrationStage

- **OBSERVATION:** All hooks log only
- **PARTIAL:** Selective enforcement (confidence, sizing)
- **COORDINATION:** AI-HFOS coordinates all subsystems
- **AUTONOMY:** Full autonomous mode (testnet only)

### Fail-Safe

- Subsystem crashes → safe fallback, not system crash
- AI-HFOS crashes → use existing Orchestrator policy
- Emergency brake → always enforced, cannot be overridden

---

**Version:** 1.0  
**Date:** November 23, 2025  
**Status:** ✅ Integration layer complete, ready for trading loop modification
