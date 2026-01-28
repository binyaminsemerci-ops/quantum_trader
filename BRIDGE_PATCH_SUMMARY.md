# BRIDGE-PATCH Implementation Summary

**Status**: ✅ **COMPLETE & READY FOR DEPLOYMENT**  
**Date**: 2026-01-21  
**Design**: Fail-Closed (conservative), SHADOW-first (safe default)

---

## What BRIDGE-PATCH Does

Connects AI sizing decisions directly to execution while maintaining ironclad safety.

**Flow**:
```
AI Engine computes sizing → Injects into trade.intent → Execution reads AI fields → 
Risk Governor enforces bounds → Order placed with final clamped values
```

**Key principle**: AI proposes, Governor disposes. All bounds are safety-enforced.

---

## Files Modified/Created

### A) Schema Evolution (Backwards Compatible v1.0 → v1.1)

**Modified**: `ai_engine/services/eventbus_bridge.py`
- TradeIntent dataclass now accepts optional: `ai_size_usd`, `ai_leverage`, `ai_harvest_policy`, `risk_budget_usd`
- Added `HarvestPolicy` dataclass with modes: scalper, swing, trend_runner
- Added `normalized()` method for schema cleanup + leverage clamping [5..80]
- Updated `validate_trade_intent()` to make position_size_usd/leverage optional (AI-injectable)

**Result**: v1.0 trade.intent still accepted, v1.1 with AI fields now supported

### B) AI Engine Sizing (NEW)

**Created**: `microservices/ai_engine/ai_sizer_policy.py`
- `AISizerPolicy` class: Computes size/leverage/policy from signal confidence + volatility
- `HarvestMode` enum: SCALPER (tight, fast) | SWING (balanced) | TREND_RUNNER (aggressive)
- `SizingConfig`: Configurable via environment variables (MIN_LEVERAGE, MAX_LEVERAGE, etc.)
- **Modes**:
  - **SHADOW**: AI proposes sizing but doesn't inject (dry-run, log what WOULD happen)
  - **LIVE**: AI sizing injected into trade.intent as primary fields (actual execution)

**Sizing formula**:
- Base: 0.5% to 2% of account based on confidence
- Volatility adjusted
- Leverage: 5x (low conf) to 80x (high conf)
- Policy: SCALPER (<60%), SWING (60-80%), TREND_RUNNER (>80%)

### C) AI Engine Integration (MODIFIED)

**Modified**: `microservices/ai_engine/service.py` (lines ~2330)
- Added import + call to `AISizerPolicy.inject_into_payload()` after rate limiting passes
- Inject AI fields before publishing trade.intent to Redis
- Fail-graceful: If sizer fails, continue with original payload (not blocking)

### D) Risk Governor (NEW)

**Created**: `services/risk_governor.py`
- `RiskGovernor` class: Enforces safety bounds before order execution
- Policies:
  1. Size bounds: [$MIN_ORDER_USD .. $MAX_POSITION_USD]
  2. Leverage bounds: [5..80]x
  3. Notional: size × leverage ≤ $MAX_NOTIONAL_USD
  4. Confidence floor (optional)
  5. Risk budget (optional)
- Returns: (approved, reason, metadata with clamped values)
- Logging: Every ACCEPT/REJECT decision logged at INFO level

### E) Execution Service Integration (MODIFIED)

**Modified**: `services/execution_service.py` (lines ~560)
- Added RiskGovernor evaluation after margin check, before order placement
- Governor returns clamped size/leverage
- Uses clamped values for final order (safety-enforced)
- Logs governance decision: `[GOVERNOR] ✅ ACCEPT` or `❌ REJECT`

### F) Schema Contract (UPDATED)

**Modified**: `TRADE_INTENT_SCHEMA_CONTRACT.md`
- Documented v1.1 changes
- Explained optional AI fields vs legacy fields
- Harvest policy modes and configurations
- Backwards compatibility notes

### G) Tests (NEW)

**Created**: `tests/test_bridge_patch.py`
- Smoke tests for AI sizer (sizing, confidence-based modes, harvest policy)
- Governor tests (accept, clamp, reject)
- End-to-end flow test
- All tests use local objects (no Redis, no Binance)

**Created**: `tests/test_trade_intent_schema.py` (already existed, schema tests pass v1.1)

### H) Documentation (NEW)

**Created**: `BRIDGE_PATCH_RUNBOOK.md`
- Configuration guide
- Deployment checklist
- SHADOW → LIVE progression
- Safety rollback procedures
- Monitoring commands
- Troubleshooting guide

---

## Configuration (Environment Variables)

### AI Sizer Config
```bash
AI_SIZING_MODE=shadow|live              # Default: shadow
AI_MAX_LEVERAGE=80                       # Max AI can recommend
AI_MIN_LEVERAGE=5                        # Min AI can recommend
MAX_POSITION_USD=10000                   # Max size per trade
MAX_NOTIONAL_USD=100000                  # Max notional exposure
MIN_ORDER_USD=50                         # Min size per trade
```

### Governor Config (same vars used)
```bash
MIN_CONFIDENCE=0.0                       # 0=disabled, >0=confidence floor
GOVERNOR_FAIL_OPEN=false                 # false=hard-fail, true=soft-fail
```

### Suggested `/etc/quantum/bridge-patch.env`
```bash
# Start in SHADOW (safe default)
AI_SIZING_MODE=shadow
AI_MAX_LEVERAGE=80
AI_MIN_LEVERAGE=5
MAX_POSITION_USD=10000
MAX_NOTIONAL_USD=100000
MIN_ORDER_USD=50
MIN_CONFIDENCE=0.0
GOVERNOR_FAIL_OPEN=false
```

---

## Deployment Steps

### 1. Verify Tests Pass
```bash
python tests/test_bridge_patch.py
python tests/test_trade_intent_schema.py
```
Expected: ✅ All tests PASSED

### 2. Deploy Code Files
```bash
scp -i ~/.ssh/hetzner_fresh \
    microservices/ai_engine/ai_sizer_policy.py \
    root@46.224.116.254:/opt/quantum/microservices/ai_engine/

scp -i ~/.ssh/hetzner_fresh \
    services/risk_governor.py \
    root@46.224.116.254:/opt/quantum/services/

scp -i ~/.ssh/hetzner_fresh \
    ai_engine/services/eventbus_bridge.py \
    root@46.224.116.254:/opt/quantum/ai_engine/services/

scp -i ~/.ssh/hetzner_fresh \
    microservices/ai_engine/service.py \
    root@46.224.116.254:/opt/quantum/microservices/ai_engine/

scp -i ~/.ssh/hetzner_fresh \
    services/execution_service.py \
    root@46.224.116.254:/opt/quantum/services/
```

### 3. Create Environment File
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

cat > /etc/quantum/bridge-patch.env << 'EOF'
AI_SIZING_MODE=shadow
AI_MAX_LEVERAGE=80
AI_MIN_LEVERAGE=5
MAX_POSITION_USD=10000
MAX_NOTIONAL_USD=100000
MIN_ORDER_USD=50
MIN_CONFIDENCE=0.0
GOVERNOR_FAIL_OPEN=false
EOF

chmod 644 /etc/quantum/bridge-patch.env
```

### 4. Update Systemd Services
```bash
# Edit both service files
nano /etc/systemd/system/quantum-ai-engine.service
# Add under [Service]: EnvironmentFile=/etc/quantum/bridge-patch.env

nano /etc/systemd/system/quantum-execution.service
# Add under [Service]: EnvironmentFile=/etc/quantum/bridge-patch.env

systemctl daemon-reload
```

### 5. Restart in SHADOW Mode
```bash
systemctl restart quantum-ai-engine
sleep 2
systemctl restart quantum-execution
sleep 2

# Verify startup
systemctl status quantum-ai-engine --no-pager
systemctl status quantum-execution --no-pager

# Check initialization logs
journalctl -u quantum-ai-engine -n 20 | grep -E "AI.Sizer|initialized"
journalctl -u quantum-execution -n 20 | grep -E "Risk.Governor|initialized"
```

Expected:
```
[AI-SIZER] ✅ AI Sizer Policy initialized: mode=shadow
[GOVERNOR] ✅ Risk Governor initialized: mode=fail-closed
```

### 6. Validate SHADOW Mode (24 hours)
```bash
# Run in shadow for 24 hours, monitor:

# AI sizing recommendations
journalctl -u quantum-ai-engine --since='24 hours ago' | grep SHADOW | tail -20

# Governor accept decisions
journalctl -u quantum-execution --since='24 hours ago' | grep "GOVERNOR.*ACCEPT" | wc -l

# Governor reject decisions (should be rare)
journalctl -u quantum-execution --since='24 hours ago' | grep "GOVERNOR.*REJECT" | wc -l
```

### 7. Switch to LIVE Mode (if confident)
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Update config
sed -i 's/AI_SIZING_MODE=shadow/AI_SIZING_MODE=live/' /etc/quantum/bridge-patch.env

# Restart
systemctl restart quantum-ai-engine
systemctl restart quantum-execution

# Verify
journalctl -u quantum-ai-engine -n 5 | grep initialized
# Should show: mode=live
```

---

## Safety Features

### Fail-Closed Design
- **Defaults**: Conservative (1x leverage, 10% of MAX_POSITION_USD)
- **Clamping**: All values bounded before order
- **Hard stops**: Rejects if notional > MAX_NOTIONAL_USD
- **Audit trail**: Every decision logged (ACCEPT/REJECT reason)

### Graceful Degradation
- If AI sizer fails → use original payload (not blocking)
- If governor fails → reject order (fail-safe)
- If confidence floor set too high → soft-warn in fail-open mode, reject in fail-closed

### Backwards Compatible
- v1.0 trade.intent still works (all new fields optional)
- Schema validation relaxed for size/leverage (now optional)
- Existing orders not affected if BRIDGE-PATCH disabled

---

## Monitoring & Observability

### Health Checks
```bash
# AI Sizer status
journalctl -u quantum-ai-engine -n 5 | grep "AI-SIZER"

# Governor status
journalctl -u quantum-execution -n 5 | grep "GOVERNOR"

# Recent decisions
journalctl -u quantum-execution --since='1 hour ago' | grep EXEC_INTENT
```

### Metrics
```bash
# Decision volume
journalctl -u quantum-execution --since='1 hour ago' | grep GOVERNOR | wc -l

# Acceptance rate
ACCEPTS=$(journalctl -u quantum-execution --since='1 hour ago' | grep "GOVERNOR.*ACCEPT" | wc -l)
REJECTS=$(journalctl -u quantum-execution --since='1 hour ago' | grep "GOVERNOR.*REJECT" | wc -l)
echo "Acceptance rate: $ACCEPTS / ($ACCEPTS + $REJECTS)"
```

---

## Rollback Procedure

If issues:
```bash
# Immediate: Back to SHADOW
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254
sed -i 's/AI_SIZING_MODE=live/AI_SIZING_MODE=shadow/' /etc/quantum/bridge-patch.env
systemctl restart quantum-ai-engine quantum-execution

# Full rollback: Remove BRIDGE-PATCH
git checkout HEAD microservices/ai_engine/service.py
git checkout HEAD services/execution_service.py
systemctl restart quantum-ai-engine quantum-execution
```

---

## Next Steps (After LIVE Mode)

1. **Monitor for 48+ hours**: Ensure sizing reasonable, acceptance rate stable
2. **Harvest policy**: Exit-brain should read `harvest_policy` from execution.result
3. **Tuning**: Adjust AI_MAX_LEVERAGE, MAX_POSITION_USD based on real behavior
4. **Advanced**: Implement per-symbol leverage caps based on volatility

---

## Key Differences from P0.D.5 BACKLOG HARDENING

| Feature | P0.D.5 | BRIDGE-PATCH |
|---------|--------|-------------|
| Focus | Backlog clearing (TTL, throughput) | AI sizing integration |
| Safety | TTL drops, rate limiting | Size/leverage bounds enforcement |
| Schema | Added P0.D.5 fields | Added AI fields (v1.1) |
| Governor | Per-stream | Per-trade |
| Default | N/A | SHADOW mode |
| Integration | Execution service | AI engine + Execution service |

---

## Commit Message

```
bridge: AI sizing/leverage/harvest policy injection (fail-closed)

BRIDGE-PATCH connects AI engine sizing decisions to execution while maintaining
fail-closed safety through Risk Governor enforcement.

Features:
- AI Sizer: Confidence-based sizing, leverage, harvest policy computation
- Risk Governor: Safety bounds enforcement (size, leverage, notional)
- Schema v1.1: Optional AI fields, HarvestPolicy, normalized()
- SHADOW mode: Safe default (no execution effect, dry-run logging)
- LIVE mode: Actual AI sizing injection into trade.intent

Safety:
- All size/leverage clamped to [MIN..MAX] before order
- Notional exposure capped at MAX_NOTIONAL_USD
- Confidence floor optional
- Fail-closed defaults (conservative)

Environment variables:
- AI_SIZING_MODE=shadow|live (default: shadow)
- AI_MAX_LEVERAGE, AI_MIN_LEVERAGE, MAX_POSITION_USD, MAX_NOTIONAL_USD
- MIN_CONFIDENCE, GOVERNOR_FAIL_OPEN

Tests:
- test_bridge_patch.py: Smoke tests for sizer + governor
- test_trade_intent_schema.py: Schema validation (v1.1 compatible)

Backwards compatible: v1.0 trade.intent still accepted.
```

---

**Ready for deployment to VPS: 46.224.116.254**
