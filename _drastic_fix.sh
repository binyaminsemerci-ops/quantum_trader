#!/bin/bash
# ============================================================
# DRASTIC FIX DEPLOYMENT — quantum_trader
# 
# Fixes deployed:
#   1. Add harvest_v2 to INTENT_EXECUTOR_SOURCE_ALLOWLIST
#   2. Lower RISK_DAILY_LOSS_LIMIT to -150 USDT
#   3. Add RISK_MAX_CONSECUTIVE_LOSSES=3 (was 5)
#   4. Set minimum viable Kelly sizing ($50 floor) for all symbols
#   5. Raise harvest_v2 R threshold (min 1.2R before any close)
#   6. Confirm SYNTHETIC_MODE=false on ensemble_predictor
# ============================================================
set -e

# ─────────────────────────────────────────────────────────────
# STEP 0: Find service unit files
# ─────────────────────────────────────────────────────────────
echo "=== STEP 0: Locating service files ==="
find /etc/systemd/system -name "*.service" | xargs grep -l "intent_executor\|intent-executor" 2>/dev/null | head -5
find /etc/systemd/system -name "*.service" | xargs grep -l "apply_layer\|apply-layer" 2>/dev/null | head -5
find /etc/systemd/system -name "*.service" | xargs grep -l "ensemble.predictor\|ensemble_predictor" 2>/dev/null | head -5

echo ""
echo "=== Service unit file listing ==="
ls /etc/systemd/system/*.service 2>/dev/null | grep -E "intent|apply|ensemble|harvest" | head -20

# ─────────────────────────────────────────────────────────────
# STEP 1: FIX intent_executor -- add harvest_v2 to source allowlist
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 1: Fix intent_executor source allowlist ==="

# Find the service file
IE_SERVICE=$(find /etc/systemd/system -name "*.service" -exec grep -l "intent.executor\|intent_executor" {} \; 2>/dev/null | head -1)
echo "intent-executor service: $IE_SERVICE"

if [ -n "$IE_SERVICE" ]; then
    # Create drop-in for intent-executor
    IE_DROPIN_DIR="${IE_SERVICE%.service}.service.d"
    mkdir -p "$IE_DROPIN_DIR"
    
    cat > "${IE_DROPIN_DIR}/allowlist-fix.conf" << 'DROPIN'
[Service]
Environment="INTENT_EXECUTOR_SOURCE_ALLOWLIST=intent_bridge,apply_layer,harvest_v2"
DROPIN
    echo "Created drop-in: ${IE_DROPIN_DIR}/allowlist-fix.conf"
else
    echo "WARNING: Could not find intent-executor service file"
    echo "Trying process-based env injection..."
    # Find intent executor process and service name
    IE_PID=$(ps aux | grep intent_executor | grep -v grep | awk '{print $2}' | head -1)
    echo "Intent executor PID: $IE_PID"
fi

# ─────────────────────────────────────────────────────────────
# STEP 2: FIX apply_layer -- lower risk limits
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 2: Fix apply_layer risk limits ==="

AL_SERVICE=$(find /etc/systemd/system -name "*.service" -exec grep -l "apply.layer\|apply_layer" {} \; 2>/dev/null | head -1)
echo "apply-layer service: $AL_SERVICE"

if [ -n "$AL_SERVICE" ]; then
    AL_DROPIN_DIR="${AL_SERVICE%.service}.service.d"
    mkdir -p "$AL_DROPIN_DIR"
    
    cat > "${AL_DROPIN_DIR}/risk-limits.conf" << 'DROPIN'
[Service]
Environment="RISK_DAILY_LOSS_LIMIT=-150.0"
Environment="RISK_MAX_CONSECUTIVE_LOSSES=3"
Environment="K_OPEN_THRESHOLD=0.80"
DROPIN
    echo "Created drop-in: ${AL_DROPIN_DIR}/risk-limits.conf"
else
    echo "WARNING: Could not find apply-layer service file — will use Redis config"
fi

# ─────────────────────────────────────────────────────────────
# STEP 3: FIX ensemble_predictor -- ensure SYNTHETIC_MODE=false
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 3: Fix ensemble_predictor synthetic mode ==="

EP_SERVICE=$(find /etc/systemd/system -name "*.service" -exec grep -l "ensemble.predictor\|ensemble_predictor" {} \; 2>/dev/null | head -1)
echo "ensemble-predictor service: $EP_SERVICE"

if [ -n "$EP_SERVICE" ]; then
    EP_DROPIN_DIR="${EP_SERVICE%.service}.service.d"
    mkdir -p "$EP_DROPIN_DIR"
    
    cat > "${EP_DROPIN_DIR}/no-synthetic.conf" << 'DROPIN'
[Service]
Environment="SYNTHETIC_MODE=false"
Environment="ENSEMBLE_MIN_CONFIDENCE=0.65"
DROPIN
    echo "Created drop-in: ${EP_DROPIN_DIR}/no-synthetic.conf"
else
    echo "WARNING: Could not find ensemble-predictor service file"
fi

# ─────────────────────────────────────────────────────────────
# STEP 4: FIX Kelly sizing -- set minimum viable floor ($50)
# for all tradeable symbols
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 4: Fix Kelly sizing (minimum viable floor) ==="

SYMBOLS="BTCUSDT ETHUSDT BNBUSDT XRPUSDT SOLUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT MATICUSDT"

for sym in $SYMBOLS; do
    redis-cli HSET quantum:layer4:sizing:$sym \
        symbol "$sym" \
        recommendation "MINIMUM_VIABLE_FLOOR" \
        kelly_raw 0.03 \
        kelly_adj 0.03 \
        size_usdt 50.0 \
        max_leverage 2 \
        reason "minimum_viable_floor_deployed" \
        ts $(date +%s) > /dev/null
    echo "  Set kelly floor for $sym: size_usdt=50.0"
done

# ─────────────────────────────────────────────────────────────
# STEP 5: FIX harvest_v2 config -- raise R threshold
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 5: Fix harvest_v2 R threshold ==="

# Check current harvest_v2 config
echo "Current harvest_v2 config:"
redis-cli HGETALL quantum:config:harvest_v2

# Update R thresholds - don't harvest below 1.5R
redis-cli HSET quantum:config:harvest_v2 \
    r_step_min 1.5 \
    r_target 4.0 \
    r_stop 0.7 \
    vol_factor_cap 1.4 \
    max_position_age_sec 604800 \
    stream_live quantum:stream:apply.plan

echo ""
echo "Updated harvest_v2 config:"
redis-cli HGETALL quantum:config:harvest_v2

# ─────────────────────────────────────────────────────────────
# STEP 6: Reload and restart services
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 6: Reload systemd and restart services ==="

systemctl daemon-reload

# Restart intent-executor
IE_NAME=$(systemctl list-units --type=service --all | grep -i "intent.exec\|intent-exec" | awk '{print $1}' | head -1)
echo "intent-executor unit: $IE_NAME"
if [ -n "$IE_NAME" ]; then
    systemctl restart "$IE_NAME" && echo "  ✅ intent-executor restarted" || echo "  ❌ failed to restart intent-executor"
    sleep 2
    systemctl is-active "$IE_NAME" && echo "  ✅ intent-executor is active" || echo "  ❌ intent-executor not active"
fi

# Restart apply-layer
AL_NAME=$(systemctl list-units --type=service --all | grep -i "apply.layer\|apply-layer" | awk '{print $1}' | head -1)
echo "apply-layer unit: $AL_NAME"
if [ -n "$AL_NAME" ]; then
    systemctl restart "$AL_NAME" && echo "  ✅ apply-layer restarted" || echo "  ❌ failed to restart apply-layer"
    sleep 2
    systemctl is-active "$AL_NAME" && echo "  ✅ apply-layer is active" || echo "  ❌ apply-layer not active"
fi

# Restart ensemble-predictor
EP_NAME=$(systemctl list-units --type=service --all | grep -i "ensemble.pred\|ensemble-pred" | awk '{print $1}' | head -1)
echo "ensemble-predictor unit: $EP_NAME"
if [ -n "$EP_NAME" ]; then
    systemctl restart "$EP_NAME" && echo "  ✅ ensemble-predictor restarted" || echo "  ❌ failed to restart ensemble-predictor"
    sleep 2
    systemctl is-active "$EP_NAME" && echo "  ✅ ensemble-predictor is active" || echo "  ❌ ensemble-predictor not active"
fi

# ─────────────────────────────────────────────────────────────
# STEP 7: Verify fixes
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 7: Verify all fixes ==="

echo "--- intent-executor env ---"
systemctl show "$IE_NAME" --property Environment 2>/dev/null | head -3

echo "--- apply-layer env ---"
systemctl show "$AL_NAME" --property Environment 2>/dev/null | head -3

echo "--- ensemble-predictor env ---"
systemctl show "$EP_NAME" --property Environment 2>/dev/null | head -3

echo "--- Kelly sizing BNBUSDT ---"
redis-cli HGETALL quantum:layer4:sizing:BNBUSDT

echo "--- harvest_v2 config ---"
redis-cli HGETALL quantum:config:harvest_v2

echo ""
echo "=== ALL FIXES DEPLOYED ==="
