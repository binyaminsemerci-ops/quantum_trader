# EXIT BRAIN CONTROL LAYER V1

**Central control for Exit Brain v3.5 execution without code changes.**

---

## Overview

Control Layer v1 enables dynamic runtime control of Exit Brain behavior:
- **LIVE/SHADOW mode switching** via environment variables
- **Percentage-based rollout** (e.g., 20% symbols LIVE, 80% SHADOW)
- **Emergency kill-switch** (fail-closed)
- **Redis audit trail** for all state changes
- **No code deployments** required for mode changes

---

## Architecture

### Enforcement Hierarchy
```
KILL_SWITCH > MODE > ROLLOUT > DEFAULT
```

1. **KILL_SWITCH=true** → Force SHADOW (fail-closed)
2. **EXIT_MODE≠EXIT_BRAIN_V3** → SHADOW
3. **EXIT_EXECUTOR_MODE≠LIVE** → SHADOW
4. **ROLLOUT_PCT** → Symbol-hash % 100 < PCT → LIVE, else SHADOW

### Files

| File | Purpose |
|------|---------|
| `/etc/quantum/exitbrain-control.env` | Control variables (SHADOW/LIVE/KILL/PCT) |
| `/etc/systemd/system/quantum-exitbrain-v35.service.d/control.conf` | systemd drop-in (loads control env) |
| `backend/config/exit_mode.py` | Enforcement logic (hash-based rollout) |
| `scripts/proof_exitbrain_control.sh` | Verification (exit codes 0/2/9) |

---

## Configuration

### Control Variables (`/etc/quantum/exitbrain-control.env`)

```bash
# Core mode
EXIT_MODE=EXIT_BRAIN_V3

# Executor: SHADOW (logs only) or LIVE (place orders)
EXIT_EXECUTOR_MODE=SHADOW

# Kill-switch: true = force SHADOW (fail-closed)
EXIT_EXECUTOR_KILL_SWITCH=false

# Rollout: 0-100% (what % of symbols get LIVE mode)
EXIT_LIVE_ROLLOUT_PCT=0

# Feature flags
EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED
EXIT_BRAIN_V35_ENABLED=true
PYTHONUNBUFFERED=1
```

### systemd Drop-In (`control.conf`)

```ini
[Service]
EnvironmentFile=/etc/quantum/exitbrain-control.env
```

**Location**: `/etc/systemd/system/quantum-exitbrain-v35.service.d/control.conf`

---

## Rollout Logic

Percentage-based rollout uses **deterministic symbol hashing**:

```python
def is_symbol_in_live_rollout(symbol: str) -> bool:
    rollout_pct = get_exit_rollout_pct()  # 0-100
    if rollout_pct == 0: return False
    if rollout_pct == 100: return True
    
    symbol_hash = hash(symbol) % 100
    return symbol_hash < rollout_pct
```

**Example** (`EXIT_LIVE_ROLLOUT_PCT=20`):
- Symbol hash % 100 < 20 → **LIVE mode**
- Symbol hash % 100 ≥ 20 → **SHADOW mode**
- ~20% symbols get LIVE, consistent across restarts

---

## Operations

### 1. Deploy Control Layer

```bash
# 1) Create control env
cat > /etc/quantum/exitbrain-control.env << 'EOF'
EXIT_MODE=EXIT_BRAIN_V3
EXIT_EXECUTOR_MODE=SHADOW
EXIT_EXECUTOR_KILL_SWITCH=false
EXIT_LIVE_ROLLOUT_PCT=0
EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED
EXIT_BRAIN_V35_ENABLED=true
PYTHONUNBUFFERED=1
EOF

chmod 600 /etc/quantum/exitbrain-control.env
chown root:root /etc/quantum/exitbrain-control.env

# 2) Create systemd drop-in
mkdir -p /etc/systemd/system/quantum-exitbrain-v35.service.d
cat > /etc/systemd/system/quantum-exitbrain-v35.service.d/control.conf << 'EOF'
[Service]
EnvironmentFile=/etc/quantum/exitbrain-control.env
EOF

# 3) Reload and restart
systemctl daemon-reload
systemctl restart quantum-exitbrain-v35.service

# 4) Verify
bash /home/qt/quantum_trader/scripts/proof_exitbrain_control.sh
```

### 2. Activate LIVE Mode (10% Rollout)

```bash
sed -i 's/EXIT_EXECUTOR_MODE=.*/EXIT_EXECUTOR_MODE=LIVE/' /etc/quantum/exitbrain-control.env
sed -i 's/EXIT_LIVE_ROLLOUT_PCT=.*/EXIT_LIVE_ROLLOUT_PCT=10/' /etc/quantum/exitbrain-control.env
systemctl restart quantum-exitbrain-v35.service
```

### 3. Scale to 50% Rollout

```bash
sed -i 's/EXIT_LIVE_ROLLOUT_PCT=.*/EXIT_LIVE_ROLLOUT_PCT=50/' /etc/quantum/exitbrain-control.env
systemctl restart quantum-exitbrain-v35.service
```

### 4. Emergency Kill-Switch

```bash
# Activate (immediate SHADOW)
sed -i 's/EXIT_EXECUTOR_KILL_SWITCH=.*/EXIT_EXECUTOR_KILL_SWITCH=true/' /etc/quantum/exitbrain-control.env
systemctl restart quantum-exitbrain-v35.service

# Deactivate
sed -i 's/EXIT_EXECUTOR_KILL_SWITCH=.*/EXIT_EXECUTOR_KILL_SWITCH=false/' /etc/quantum/exitbrain-control.env
systemctl restart quantum-exitbrain-v35.service
```

---

## Verification

### Proof Script

```bash
bash /home/qt/quantum_trader/scripts/proof_exitbrain_control.sh
```

**Exit Codes**:
- `0` = PASS (all tests, LIVE mode operational)
- `2` = SHADOW (tests pass but SHADOW mode active)
- `9` = KILL (kill-switch active, forced SHADOW)

### Redis Audit Trail

```bash
redis-cli LRANGE quantum:ops:exitbrain:control 0 9
```

Shows last 10 control state changes with timestamps.

---

## Deployment Steps (VPS)

```bash
# 1) Copy files to VPS
scp exitbrain-control.env root@VPS:/etc/quantum/
scp systemd-drop-in-control.conf root@VPS:/etc/systemd/system/quantum-exitbrain-v35.service.d/control.conf
scp scripts/proof_exitbrain_control.sh root@VPS:/home/qt/quantum_trader/scripts/

# 2) Set permissions
ssh root@VPS 'chmod 600 /etc/quantum/exitbrain-control.env && \
  chown root:root /etc/quantum/exitbrain-control.env && \
  chmod +x /home/qt/quantum_trader/scripts/proof_exitbrain_control.sh'

# 3) Deploy code changes
ssh root@VPS 'cd /home/qt/quantum_trader && git pull origin main'

# 4) Reload and verify
ssh root@VPS 'systemctl daemon-reload && \
  systemctl restart quantum-exitbrain-v35.service && \
  sleep 5 && \
  bash /home/qt/quantum_trader/scripts/proof_exitbrain_control.sh'
```

---

## Code Changes

### `backend/config/exit_mode.py` Patch

**Added Functions**:
- `get_exit_rollout_pct()` → Returns 0-100 (clamped)
- `is_symbol_in_live_rollout(symbol)` → Hash-based rollout check
- `log_control_state_to_redis()` → Audit trail to Redis

**Modified Function**:
- `is_exit_brain_live_fully_enabled(symbol=None)` → Now accepts optional symbol parameter for per-symbol rollout check

**Enforcement Order**:
1. Kill-switch check (fail-closed)
2. Base mode checks (EXIT_MODE, EXECUTOR_MODE, LIVE_ROLLOUT)
3. Symbol rollout (if symbol provided)

---

## Safety

- **Fail-closed**: Kill-switch forces SHADOW regardless of other settings
- **Deterministic**: Same symbol always gets same mode (hash-based)
- **Auditable**: All state changes logged to Redis
- **Restart-safe**: systemd drop-in loads control env before service start
- **No secrets**: Control env has no credentials (references base env)

---

**Status**: Ready for deployment | **Version**: 1.0 | **Date**: 2026-01-29
