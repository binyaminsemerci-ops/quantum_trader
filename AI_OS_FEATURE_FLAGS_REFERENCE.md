# AI-OS FEATURE FLAGS QUICK REFERENCE

## 🎚️ INTEGRATION STAGE CONTROL

```bash
# Global integration stage (controls overall behavior)
QT_AI_INTEGRATION_STAGE=observation    # Stage 1: Log only, no enforcement
QT_AI_INTEGRATION_STAGE=partial        # Stage 2: Selective enforcement
QT_AI_INTEGRATION_STAGE=coordination   # Stage 3: AI-HFOS coordinates all
QT_AI_INTEGRATION_STAGE=autonomy       # Stage 4: Full AI autonomy
```

**Recommendation**: Start with `observation`, progress through stages over 30+ days

---

## 🚨 EMERGENCY CONTROLS

```bash
# Emergency brake (blocks ALL new trades immediately)
QT_EMERGENCY_BRAKE=true     # ACTIVE - no new trades
QT_EMERGENCY_BRAKE=false    # NORMAL - trading allowed (default)
```

**Use Case**: Critical drawdown, system issue, or need to stop trading instantly

---

## 🌍 UNIVERSE OS (Symbol Filtering)

```bash
# Enable/disable Universe OS
QT_AI_UNIVERSE_OS_ENABLED=true    # Enable symbol filtering
QT_AI_UNIVERSE_OS_ENABLED=false   # Disable (default)

# Operating mode
QT_AI_UNIVERSE_OS_MODE=off        # Disabled (even if enabled=true)
QT_AI_UNIVERSE_OS_MODE=observe    # Log what would be filtered (default)
QT_AI_UNIVERSE_OS_MODE=advisory   # Suggest filtering (not enforced)
QT_AI_UNIVERSE_OS_MODE=enforced   # Apply filtering
```

**Recommendation**: Start with `observe`, move to `enforced` after 7 days

---

## 🎯 AI-HFOS (Supreme Coordinator)

```bash
# Enable/disable AI-HFOS
QT_AI_HFOS_ENABLED=true     # Enable supreme coordination
QT_AI_HFOS_ENABLED=false    # Disable (default)

# Operating mode
QT_AI_HFOS_MODE=off         # Disabled
QT_AI_HFOS_MODE=observe     # Log decisions only (default)
QT_AI_HFOS_MODE=advisory    # Suggest adjustments (not enforced)
QT_AI_HFOS_MODE=enforced    # Full enforcement (HEDGEFUND MODE active)
```

**Recommendation**: Stay in `observe` for 14 days, then move to `enforced` in Stage 3

**HEDGEFUND MODE**: When `enforced`, AI-HFOS manages 4-tier risk system:
- NORMAL: Base parameters (4 positions, 100% size, 0.72 confidence)
- OPTIMISTIC: 1.25x scaling (5 positions, 115% size, 0.68 confidence)
- AGGRESSIVE: 2.5x scaling (10 positions, 130% size, 0.65 confidence)
- CRITICAL: 0.5x scaling (2 positions, 70% size, 0.80 confidence)

---

## ⚖️ PORTFOLIO BALANCER AI (PBA)

```bash
# Enable/disable PBA
QT_AI_PBA_ENABLED=true      # Enable portfolio-level constraints
QT_AI_PBA_ENABLED=false     # Disable (default)

# Operating mode
QT_AI_PBA_MODE=off          # Disabled
QT_AI_PBA_MODE=observe      # Log decisions only (default)
QT_AI_PBA_MODE=advisory     # Suggest adjustments
QT_AI_PBA_MODE=enforced     # Block overexposure trades
```

**Recommendation**: Move to `enforced` in Stage 3 alongside AI-HFOS

**Purpose**: Prevents portfolio concentration risk (e.g., too many correlated positions)

---

## 📊 POSITION INTELLIGENCE LAYER (PIL)

```bash
# Enable/disable PIL
QT_AI_PIL_ENABLED=true      # Enable position classification
QT_AI_PIL_ENABLED=false     # Disable (default)

# Operating mode
QT_AI_PIL_MODE=off          # Disabled
QT_AI_PIL_MODE=observe      # Log classifications only (default)
QT_AI_PIL_MODE=advisory     # Suggest actions
QT_AI_PIL_MODE=enforced     # Classifications used by other subsystems
```

**Recommendation**: Safe to enable in `enforced` mode at Stage 2

**Purpose**: Classifies positions as WINNER, LOSER, BREAKEVEN, STRUGGLING, POTENTIAL_WINNER

---

## 💰 PROFIT AMPLIFICATION LAYER (PAL)

```bash
# Enable/disable PAL
QT_AI_PAL_ENABLED=true      # Enable profit amplification
QT_AI_PAL_ENABLED=false     # Disable (default)

# Operating mode
QT_AI_PAL_MODE=off          # Disabled
QT_AI_PAL_MODE=observe      # Log opportunities only (default)
QT_AI_PAL_MODE=advisory     # Suggest amplifications
QT_AI_PAL_MODE=enforced     # Execute scale-ins/extend-holds
```

**Recommendation**: Keep in `advisory` until Stage 4 (30+ days), then move to `enforced`

**Purpose**: Scales into winning positions and extends holds on strong performers

**⚠️ CAUTION**: This is the most aggressive subsystem - only enable when confident

---

## 🔍 MODEL SUPERVISOR

```bash
# Enable/disable Model Supervisor
QT_AI_MODEL_SUPERVISOR_ENABLED=true    # Enable signal quality monitoring
QT_AI_MODEL_SUPERVISOR_ENABLED=false   # Disable (default)

# Operating mode
QT_AI_MODEL_SUPERVISOR_MODE=off        # Disabled
QT_AI_MODEL_SUPERVISOR_MODE=observe    # Log quality metrics (default)
QT_AI_MODEL_SUPERVISOR_MODE=advisory   # Flag degrading models
QT_AI_MODEL_SUPERVISOR_MODE=enforced   # Block signals from bad models
```

**Recommendation**: Safe to enable in `enforced` mode at Stage 2

**Purpose**: Monitors AI model performance and flags degrading signal quality

---

## ⚡ ADVANCED EXECUTION LAYER MANAGER (AELM)

```bash
# Enable/disable AELM
QT_AI_AELM_ENABLED=true     # Enable execution optimization
QT_AI_AELM_ENABLED=false    # Disable (default)

# Operating mode
QT_AI_AELM_MODE=off         # Disabled
QT_AI_AELM_MODE=observe     # Log decisions only (default)
QT_AI_AELM_MODE=advisory    # Suggest order type changes
QT_AI_AELM_MODE=enforced    # Optimize order execution
```

**Recommendation**: Enable in `observe` at Stage 2, move to `enforced` at Stage 3

**Purpose**: Optimizes order types (MARKET vs LIMIT) and validates slippage

---

## 🏥 SELF-HEALING SYSTEM

```bash
# Enable/disable Self-Healing
QT_AI_SELF_HEALING_ENABLED=true     # Enable health monitoring
QT_AI_SELF_HEALING_ENABLED=false    # Disable (default)

# Operating mode
QT_AI_SELF_HEALING_MODE=off         # Disabled
QT_AI_SELF_HEALING_MODE=observe     # Log issues only (default)
QT_AI_SELF_HEALING_MODE=protective  # Apply recovery actions
```

**Recommendation**: Enable in `protective` mode at Stage 3

**Purpose**: Detects system issues and applies recovery actions (e.g., trigger emergency brake on critical errors)

---

## 🎓 RETRAINING ORCHESTRATOR

```bash
# Enable/disable Retraining
QT_AI_RETRAINING_ENABLED=true      # Enable automated model updates
QT_AI_RETRAINING_ENABLED=false     # Disable (default)
```

**Recommendation**: Keep disabled until Stage 4 (requires extensive monitoring)

**Purpose**: Automatically retrains AI models based on performance feedback

**⚠️ CAUTION**: Only enable after 60+ days of stable operation

---

## 🎚️ RISK MANAGEMENT (Already in Production)

```bash
# Max daily drawdown (always enforced)
QT_MAX_DAILY_DD_PCT=0.05           # 5% daily drawdown limit (default)

# Max positions (can be scaled by AI-HFOS)
QT_MAX_POSITIONS=4                 # Base max positions (default)

# Max positions per symbol
QT_MAX_POSITIONS_PER_SYMBOL=2      # Prevent stacking (default)
```

**Note**: These are ALWAYS enforced, even when AI is disabled

---

## 🚀 QUICK START CONFIGURATIONS

### Configuration 1: OBSERVE EVERYTHING (Safe Start)
```bash
# Stage 1: Pure observation
QT_AI_INTEGRATION_STAGE=observation

# Enable all subsystems in OBSERVE mode
QT_AI_UNIVERSE_OS_ENABLED=true
QT_AI_UNIVERSE_OS_MODE=observe

QT_AI_HFOS_ENABLED=true
QT_AI_HFOS_MODE=observe

QT_AI_PBA_ENABLED=true
QT_AI_PBA_MODE=observe

QT_AI_PIL_ENABLED=true
QT_AI_PIL_MODE=observe

QT_AI_PAL_ENABLED=true
QT_AI_PAL_MODE=observe

QT_AI_MODEL_SUPERVISOR_ENABLED=true
QT_AI_MODEL_SUPERVISOR_MODE=observe

QT_AI_AELM_ENABLED=true
QT_AI_AELM_MODE=observe

QT_AI_SELF_HEALING_ENABLED=true
QT_AI_SELF_HEALING_MODE=observe
```

**Result**: All AI decisions logged, zero impact on trading

---

### Configuration 2: PARTIAL ENFORCEMENT (After 7 Days)
```bash
# Stage 2: Selective enforcement
QT_AI_INTEGRATION_STAGE=partial

# Enforce low-risk subsystems
QT_AI_UNIVERSE_OS_ENABLED=true
QT_AI_UNIVERSE_OS_MODE=enforced      # ✅ Safe: Just filters symbols

QT_AI_MODEL_SUPERVISOR_ENABLED=true
QT_AI_MODEL_SUPERVISOR_MODE=enforced # ✅ Safe: Just monitors quality

QT_AI_PIL_ENABLED=true
QT_AI_PIL_MODE=enforced              # ✅ Safe: Just classifies positions

# Keep high-impact subsystems in OBSERVE
QT_AI_HFOS_ENABLED=true
QT_AI_HFOS_MODE=observe              # ⚠️ Still learning

QT_AI_PBA_ENABLED=true
QT_AI_PBA_MODE=observe               # ⚠️ Still learning

QT_AI_PAL_ENABLED=true
QT_AI_PAL_MODE=observe               # ⚠️ Still learning

QT_AI_AELM_ENABLED=true
QT_AI_AELM_MODE=observe              # ⚠️ Still learning
```

**Result**: Symbol filtering active, position classification active, rest still learning

---

### Configuration 3: AI-HFOS COORDINATION (After 14 Days)
```bash
# Stage 3: Full coordination
QT_AI_INTEGRATION_STAGE=coordination

# AI-HFOS takes control
QT_AI_HFOS_ENABLED=true
QT_AI_HFOS_MODE=enforced             # 🚀 HEDGEFUND MODE ACTIVE

# PBA enforces portfolio limits
QT_AI_PBA_ENABLED=true
QT_AI_PBA_MODE=enforced              # ⚖️ Portfolio protection

# AELM optimizes execution
QT_AI_AELM_ENABLED=true
QT_AI_AELM_MODE=enforced             # ⚡ Smart order routing

# Self-Healing becomes protective
QT_AI_SELF_HEALING_ENABLED=true
QT_AI_SELF_HEALING_MODE=protective   # 🏥 Auto-recovery

# PAL suggests amplifications (advisory)
QT_AI_PAL_ENABLED=true
QT_AI_PAL_MODE=advisory              # 💰 Suggests, doesn't execute

# Already enforced from Stage 2
QT_AI_UNIVERSE_OS_ENABLED=true
QT_AI_UNIVERSE_OS_MODE=enforced
QT_AI_MODEL_SUPERVISOR_ENABLED=true
QT_AI_MODEL_SUPERVISOR_MODE=enforced
QT_AI_PIL_ENABLED=true
QT_AI_PIL_MODE=enforced
```

**Result**: Full AI coordination active, HEDGEFUND MODE managing risk tiers, PAL still advisory

---

### Configuration 4: FULL AUTONOMY (After 30+ Days)
```bash
# Stage 4: Maximum AI autonomy
QT_AI_INTEGRATION_STAGE=autonomy

# ALL subsystems enforced
QT_AI_HFOS_MODE=enforced
QT_AI_PBA_MODE=enforced
QT_AI_PAL_MODE=enforced              # 🚀 PAL now executes amplifications
QT_AI_PIL_MODE=enforced
QT_AI_UNIVERSE_OS_MODE=enforced
QT_AI_MODEL_SUPERVISOR_MODE=enforced
QT_AI_AELM_MODE=enforced
QT_AI_SELF_HEALING_MODE=protective
```

**Result**: Full AI autonomy - system adapts in real-time

**⚠️ REQUIREMENT**: Only enable after 30+ days of profitable Stage 3 operation

---

## 🆘 EMERGENCY ROLLBACK

If anything goes wrong, immediately revert to pure observation:

```bash
# Emergency: Disable ALL AI enforcement
QT_AI_INTEGRATION_STAGE=observation

# OR just stop all trading
QT_EMERGENCY_BRAKE=true

# OR restart backend to default config
docker restart quantum_backend
```

---

## 📊 MONITORING COMMANDS

```bash
# Check current configuration
journalctl -u quantum_backend.service | grep "AI-HFOS\|PBA\|PAL\|PIL"

# Check for blocks/modifications
journalctl -u quantum_backend.service | grep "BLOCKED\|MODIFIED"

# Check SafetyGovernor decisions
journalctl -u quantum_backend.service | grep "SAFETY GOVERNOR"

# Check HEDGEFUND MODE transitions
journalctl -u quantum_backend.service | grep "HEDGEFUND MODE\|AGGRESSIVE\|CRITICAL"
```

---

## ✅ RECOMMENDED PROGRESSION TIMELINE

| Days | Stage | Configuration | Goal |
|------|-------|---------------|------|
| **1-7** | OBSERVATION | All OBSERVE | Verify integration, collect metrics |
| **8-14** | PARTIAL | Selective ENFORCED | Enable low-risk subsystems |
| **15-30** | COORDINATION | AI-HFOS ENFORCED | Full coordination, HEDGEFUND MODE |
| **31+** | AUTONOMY | PAL ENFORCED | Maximum AI autonomy |

**Critical Success Factors**:
- ✅ No integration errors in logs
- ✅ AI decisions improve performance metrics
- ✅ No emergency brake activations
- ✅ Profitability maintained or improved at each stage

---

## 📚 REFERENCES

- **Full Integration Report**: `AI_OS_FULL_INTEGRATION_REPORT.md`
- **Verification Script**: `verify_ai_integration.py`
- **Service Registry**: `backend/services/system_services.py`
- **Integration Hooks**: `backend/services/integration_hooks.py`
- **Trading Loop**: `backend/services/event_driven_executor.py`

---

**Last Updated**: 2025-01-XX  
**Status**: Production-Ready

