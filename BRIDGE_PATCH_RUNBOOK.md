# BRIDGE-PATCH: AI Sizing/Leverage/Policy Injection (v1.0)

**Date**: 2026-01-21  
**Status**: ✅ Implementation Complete, Ready for Deployment  
**Mode**: FAIL-CLOSED (conservative defaults, enforced safety bounds)

---

## Overview

BRIDGE-PATCH connects AI engine decisions (position sizing, leverage, harvest policy) to execution service while maintaining fail-closed safety.

**Design principle**: AI *proposes*, Governor *disposes*.

1. **AI Engine** (microservices/ai_engine/service.py):
   - Computes recommended size/leverage/policy based on signal confidence + volatility
   - Injects into trade.intent payload as `ai_size_usd`, `ai_leverage`, `ai_harvest_policy`
   - Mode: SHADOW (dry-run) or LIVE (actual sizing)

2. **Execution Service** (services/execution_service.py):
   - Consumes trade.intent, parses AI fields
   - Runs Risk Governor to enforce safety bounds
   - Clamps size/leverage to safe ranges before order
   - Logs decision (ACCEPT/REJECT + reason)

3. **Risk Governor** (services/risk_governor.py):
   - Enforces:
     * Size bounds: [$MIN_ORDER_USD .. $MAX_POSITION_USD]
     * Leverage bounds: [5 .. 80]x
     * Notional: size × leverage ≤ $MAX_NOTIONAL_USD
     * Confidence floor (optional)
     * Risk budget (optional)
   - Returns: (approved, reason, metadata with clamped values)

4. **Schema** (ai_engine/services/eventbus_bridge.py):
   - TradeIntent v1.1 with optional AI fields
   - HarvestPolicy dataclass for exit strategies
   - normalized() method for backwards compatibility
   - validate_trade_intent() relaxed to allow optional sizing

---

## Configuration (Environment Variables)

### AI Engine Sizing (microservices/ai_engine/ai_sizer_policy.py)
```bash
AI_SIZING_MODE=shadow|live              # Default: shadow (dry-run)
AI_MAX_LEVERAGE=80                       # Max leverage AI can request
AI_MIN_LEVERAGE=5                        # Min leverage AI can request
MAX_POSITION_USD=10000                   # Max position size per trade
MAX_NOTIONAL_USD=100000                  # Max total notional exposure
MIN_ORDER_USD=50                         # Min position size
```

### Risk Governor (services/risk_governor.py)
```bash
MIN_ORDER_USD=50
MAX_POSITION_USD=10000
MAX_NOTIONAL_USD=100000
MIN_CONFIDENCE=0.0                       # 0 = disabled, >0 = confidence floor
AI_MAX_LEVERAGE=80
AI_MIN_LEVERAGE=5
GOVERNOR_FAIL_OPEN=false                # true = soft failures, false = hard failures
```

### Default Config
Create `/etc/quantum/bridge-patch.env`:
```bash
# BRIDGE-PATCH Configuration
# Start with SHADOW mode to validate before going LIVE

# AI Sizing
AI_SIZING_MODE=shadow
AI_MAX_LEVERAGE=80
AI_MIN_LEVERAGE=5

# Sizing Bounds
MAX_POSITION_USD=10000
MAX_NOTIONAL_USD=100000
MIN_ORDER_USD=50

# Governor Policy
MIN_CONFIDENCE=0.0
GOVERNOR_FAIL_OPEN=false
```

Load in systemd:
```ini
[Service]
EnvironmentFile=/etc/quantum/bridge-patch.env
```

---

## Deployment Checklist

### 1. Schema Tests
```bash
# Test schema changes
python tests/test_trade_intent_schema.py
```

Expected: ✅ All tests pass (v1.1 optional fields accepted)

### 2. Deploy Code
```bash
# SCP to VPS
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

# Create config file
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

# Add to systemd services:
# Edit /etc/systemd/system/quantum-ai-engine.service
# Add: EnvironmentFile=/etc/quantum/bridge-patch.env

# Edit /etc/systemd/system/quantum-execution.service
# Add: EnvironmentFile=/etc/quantum/bridge-patch.env

systemctl daemon-reload
```

### 3. Restart Services (SHADOW MODE)
```bash
systemctl restart quantum-ai-engine
systemctl restart quantum-execution

# Verify startup
sleep 5
systemctl status quantum-ai-engine -l
systemctl status quantum-execution -l

# Check logs for AI Sizer initialization
journalctl -u quantum-ai-engine -n 20 | grep "AI Sizer"
journalctl -u quantum-execution -n 20 | grep "Risk Governor"
```

Expected output:
```
[AI-SIZER] ✅ AI Sizer Policy initialized: mode=shadow
[GOVERNOR] ✅ Risk Governor initialized: mode=fail-closed
```

### 4. Verify SHADOW Mode
```bash
# Let system run for 5 minutes in shadow mode
# Then check logs

# AI Sizer shadow logs
journalctl -u quantum-ai-engine -n 50 | grep "SHADOW"

# Governor accept logs
journalctl -u quantum-execution -n 50 | grep "ACCEPT\|PASS"

# Execution logs (should show proposed sizes)
journalctl -u quantum-execution -n 50 | grep "EXEC_INTENT"
```

Expected: No actual trades executed (shadow mode), but sizing logged.

---

## Mode Progression: SHADOW → LIVE

### Phase 1: SHADOW Validation (24 hours)
- `AI_SIZING_MODE=shadow`
- System proposes sizes but doesn't execute
- Monitor: AI sizing recommendations, governor decisions
- Check: No unexpected rejections, sizing reasonable

Commands:
```bash
# Count accept vs reject decisions
journalctl -u quantum-execution --since='24 hours ago' | \
  grep GOVERNOR | grep -c ACCEPT
journalctl -u quantum-execution --since='24 hours ago' | \
  grep GOVERNOR | grep -c REJECT

# Check AI sizing recommendations
journalctl -u quantum-ai-engine --since='24 hours ago' | \
  grep "SHADOW:" | head -20
```

### Phase 2: Switch to LIVE
Once comfortable:
```bash
# Update config
sed -i 's/AI_SIZING_MODE=shadow/AI_SIZING_MODE=live/' /etc/quantum/bridge-patch.env

# Reload and restart
systemctl daemon-reload
systemctl restart quantum-ai-engine
systemctl restart quantum-execution

# Verify switch
journalctl -u quantum-ai-engine -n 5 | grep "initialized"
```

Expected:
```
[AI-SIZER] ✅ AI Sizer Policy initialized: mode=live
```

### Phase 3: Monitor LIVE Mode
```bash
# Watch for actual trade execution with AI sizing
journalctl -u quantum-execution -f | grep "EXEC_INTENT\|GOVERNOR"

# Example output (LIVE mode):
# [EXEC_INTENT] BTCUSDT BUY: size_usd=500, leverage=15.5x, conf=0.85, governor=PASS
```

---

## Safety Rollback

If issues:
```bash
# Immediate: Switch back to shadow
sed -i 's/AI_SIZING_MODE=live/AI_SIZING_MODE=shadow/' /etc/quantum/bridge-patch.env
systemctl restart quantum-ai-engine
systemctl restart quantum-execution

# Full rollback: Remove all BRIDGE-PATCH changes
git checkout HEAD microservices/ai_engine/service.py
git checkout HEAD services/execution_service.py
systemctl restart quantum-ai-engine
systemctl restart quantum-execution
```

---

## Monitoring Commands

### AI Sizing Metrics
```bash
# Count shadow vs live decisions
journalctl -u quantum-ai-engine --since='1 hour ago' | grep -c "SHADOW"
journalctl -u quantum-ai-engine --since='1 hour ago' | grep -c "LIVE"

# View recent sizing decisions
journalctl -u quantum-ai-engine -n 20 | grep "AI-SIZER"
```

### Governor Metrics
```bash
# Count governance decisions
journalctl -u quantum-execution --since='1 hour ago' | \
  grep GOVERNOR | awk -F'[[]' '{print $2}' | sort | uniq -c

# View rejections (error analysis)
journalctl -u quantum-execution --since='1 hour ago' | grep REJECT

# View acceptance reasons
journalctl -u quantum-execution --since='1 hour ago' | grep PASS | head -10
```

### End-to-End Audit
```bash
# Trace single order from AI decision through execution
SYMBOL=BTCUSDT
journalctl -u quantum-ai-engine --since='5 minutes ago' | grep $SYMBOL | grep "PUBLISHED"
journalctl -u quantum-execution --since='5 minutes ago' | grep $SYMBOL | grep -E "GOVERNOR|EXEC_INTENT"
```

---

## Harvest Policy Integration

Exit-brain (when enabled) should read `harvest_policy` from execution.result:

```python
# In exit-brain consumer:
if 'harvest_policy' in execution_result:
    policy = HarvestPolicy(**execution_result['harvest_policy'])
    apply_policy(position, policy)
else:
    # Fallback to default behavior
    pass
```

Exit-brain modes:
- **scalper**: Close 50% at 1% profit, trail stops tight
- **swing**: Hold for swings, trail at 1% below high
- **trend_runner**: Let winners run, partial close at 3% profit

---

## Unit Tests / Pre-Deploy Check

Run schema tests before deploying:
```bash
python tests/test_trade_intent_schema.py

# Expected:
# ✅ test_valid_trade_intent PASSED
# ✅ test_optional_ai_fields PASSED
# ✅ test_harvest_policy PASSED
# ✅ All schema validation tests PASSED - safe to deploy
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `AI Sizer not initialized` | Module not installed | Ensure `ai_sizer_policy.py` deployed |
| `Governor fails on startup` | Config not loaded | Check `bridge-patch.env` sourced by systemd |
| `All trades rejected` | Notional bounds too tight | Increase `MAX_NOTIONAL_USD` in env |
| `Leverage not changing` | AI_SIZING_MODE=shadow | Set `AI_SIZING_MODE=live` |
| `Confidence floor filtering` | MIN_CONFIDENCE > 0 | Either disable (set to 0) or increase signal quality |

---

## Testing Locally

```bash
# Import and test AI sizer
python3 << 'EOF'
from microservices.ai_engine.ai_sizer_policy import AISizerPolicy, SizingConfig

config = SizingConfig(ai_sizing_mode='shadow')
sizer = AISizerPolicy(config)

# Test sizing for high-confidence signal
lev, size, policy = sizer.compute_size_and_leverage(
    signal_confidence=0.95,
    volatility_factor=1.0,
    account_equity=10000.0
)

print(f"Size: ${size:.0f}, Leverage: {lev:.1f}x, Policy: {policy['mode']}")

# Test Governor
from services.risk_governor import RiskGovernor

gov = RiskGovernor()
approved, reason, meta = gov.evaluate(
    symbol='BTCUSDT',
    action='BUY',
    confidence=0.95,
    position_size_usd=size,
    leverage=lev
)

print(f"Approved: {approved}, Reason: {reason}")
EOF
```

---

## Key Differences from P0.D.5

| Aspect | P0.D.5 | BRIDGE-PATCH |
|--------|--------|-------------|
| Focus | Backlog clearing (TTL, throughput) | AI sizing integration |
| Fields modified | Added TTL checking | Added ai_* fields, HarvestPolicy |
| Governor | None | New RiskGovernor enforces bounds |
| Default mode | N/A | Shadow (safe default) |
| Backwards compat | Yes | Yes (v1.0 payloads still work) |

---

## Next Steps

1. ✅ Schema updated (v1.1)
2. ✅ AI Sizer module created
3. ✅ Risk Governor enforcer created
4. ✅ Integration points added (AI engine + execution)
5. ⏳ Deploy to VPS
6. ⏳ Run in SHADOW mode (24 hours)
7. ⏳ Switch to LIVE mode
8. ⏳ Monitor harvest policy adoption by exit-brain

---

**Created**: 2026-01-21  
**Author**: AI Agent  
**Status**: Implementation complete, awaiting deployment
