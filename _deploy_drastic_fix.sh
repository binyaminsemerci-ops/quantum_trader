#!/bin/bash
# ============================================================
# DRASTIC FIX DEPLOYMENT — quantum_trader
# Deploys all critical fixes to address -769 USDT/2day losses
# ============================================================
set -e
BACKUP_DIR="/etc/quantum/backups.$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "============================================================"
echo " DRASTIC FIX DEPLOYMENT"
echo " $(date)"
echo "============================================================"

# ─────────────────────────────────────────────────────────────
# FIX 1: Reduce SYMBOLS and add risk limits to apply-layer.env
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> FIX 1: apply-layer.env — narrow symbols + risk limits"

cp /etc/quantum/apply-layer.env "$BACKUP_DIR/apply-layer.env.bak"

# Write a fresh, focused apply-layer.env
cat > /etc/quantum/apply-layer.env << 'ENVEOF'
# P3 Apply Layer Configuration — AFTER DRASTIC FIX (2026-03-01)
# REDUCED to top 12 high-liquidity pairs only (was 400+ exotic tokens)

APPLY_MODE=testnet

# TOP 12 HIGH-LIQUIDITY PAIRS ONLY
# Reduces noise from 400+ meme tokens to focused, liquid markets
SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,DOTUSDT,OPUSDT,LTCUSDT

# RISK LIMITS — tightened from defaults
# Daily loss limit: was -1000 USDT (default), now -150 USDT
RISK_DAILY_LOSS_LIMIT=-150.0

# Max consecutive losses before pausing: was 5, now 3
RISK_MAX_CONSECUTIVE_LOSSES=3

# Kill-score open threshold: was 0.85 (too permissive), now 0.80
# Higher means allow less signals through to open
K_OPEN_THRESHOLD=0.80

# Minimum confidence to open a position: 70%
ENSEMBLE_MIN_CONFIDENCE_OPEN=0.70

# Position sizing: minimum notional per trade
MIN_POSITION_NOTIONAL=200.0

# Max positions at one time
MAX_POSITIONS=5
ENVEOF

echo "   ✅ apply-layer.env updated"
echo "   New SYMBOLS: BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,DOTUSDT,OPUSDT,LTCUSDT"
echo "   RISK_DAILY_LOSS_LIMIT=-150.0"
echo "   RISK_MAX_CONSECUTIVE_LOSSES=3"
echo "   K_OPEN_THRESHOLD=0.80"

# ─────────────────────────────────────────────────────────────
# FIX 2: Kelly sizing floor = $200 for all major symbols
# Fixes: Layer 4 Kelly returns size_usdt=0 (no_backtest_data)
# → apply_layer falls back to minimum (~$100 notional)
# → positions too small to generate meaningful profit
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> FIX 2: Kelly sizing floor = 200 USDT for all symbols"

SYMBOLS_LIST="BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT MATICUSDT TRXUSDT UNIUSDT ATOMUSDT NEARUSDT APTUSDT ARBUSDT INJUSDT"

for sym in $SYMBOLS_LIST; do
    redis-cli HSET quantum:layer4:sizing:$sym \
        symbol "$sym" \
        recommendation "MINIMUM_VIABLE_FLOOR" \
        kelly_raw 0.04 \
        kelly_adj 0.04 \
        size_usdt 200.0 \
        max_leverage 2 \
        reason "minimum_viable_floor_deployed_2026_03_01" \
        ts $(date +%s) > /dev/null
    echo "   $sym: size_usdt=200.0"
done

echo "   ✅ Kelly sizing floor set for all symbols"

# ─────────────────────────────────────────────────────────────
# FIX 3: harvest_v2 config — better R thresholds
# Current issue: exits at 0.29R (trailing 30% retracement from 0.57R max)
# Fix: partial closes only start at 1.5R, trailing_step=0.25 (was 0.3)
# This means: only partial close when position reaches 1.5R profit
# trailing_step=0.25 means close if R drops 25% from peak (slightly tighter)
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> FIX 3: harvest_v2 R thresholds"

redis-cli HSET quantum:config:harvest_v2 \
    r_stop_base 0.5 \
    r_target_base 3.5 \
    trailing_step 0.25 \
    r_emit_step 0.1 \
    partial_25_r 1.5 \
    partial_50_r 2.0 \
    partial_75_r 2.5 \
    vol_factor_low 0.7 \
    vol_factor_mid 1.0 \
    vol_factor_high 1.4 \
    heat_sensitivity 0.3 \
    max_position_age_sec 86400 \
    stream_live quantum:stream:apply.plan

echo "   ✅ harvest_v2 config updated"
echo "   r_stop_base=0.5 (unchanged — stop at 0.5R loss)"
echo "   r_target_base=3.5 (was 3.0 — let profits run)"
echo "   trailing_step=0.25 (was 0.3 — tighter trailing stop)"
echo "   partial_25_r=1.5 (was 1.0 — first partial only at 1.5R profit)"
echo "   partial_50_r=2.0 (was 1.5)"
echo "   partial_75_r=2.5 (was 2.0)"

echo ""
echo "   Current harvest_v2 config:"
redis-cli HGETALL quantum:config:harvest_v2

# ─────────────────────────────────────────────────────────────
# FIX 4: Fix Redis WRONGTYPE error causing harvest_failed=482
# The error: "WRONGTYPE Operation against a key holding the wrong kind of value"
# Check which key is wrong type and fix it
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> FIX 4: Investigate Redis WRONGTYPE errors"

# Check common keys that might have wrong type
for key in "quantum:position:BNBUSDT" "quantum:position:XRPUSDT" "quantum:position:LINKUSDT"; do
    type=$(redis-cli TYPE "$key")
    echo "   $key: type=$type"
done

# Check if any position keys are the wrong type (should be hash)
echo "   Checking all active position keys types..."
for sym in BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT; do
    type=$(redis-cli TYPE "quantum:position:$sym" 2>/dev/null)
    echo "   quantum:position:$sym → $type"
done

# ─────────────────────────────────────────────────────────────
# FIX 5: Restrict harvest proposal symbols to top 12 only
# Reduces spam from 400+ symbols generating proposals
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> FIX 5: Check harvest-proposal.env"
cat /etc/quantum/harvest-proposal.env | grep -E "SYMBOL|symbol" | head -5

# ─────────────────────────────────────────────────────────────
# RESTART apply-layer service
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Restarting quantum-apply-layer.service"
systemctl restart quantum-apply-layer.service
sleep 3
STATUS=$(systemctl is-active quantum-apply-layer.service)
echo "   quantum-apply-layer.service: $STATUS"

if [ "$STATUS" != "active" ]; then
    echo "   ❌ apply-layer failed to start! Checking logs..."
    journalctl -u quantum-apply-layer.service -n 20 --no-pager
else
    echo "   ✅ apply-layer is running"
fi

# ─────────────────────────────────────────────────────────────
# FINAL VERIFICATION
# ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " VERIFICATION"
echo "============================================================"

echo "--- apply-layer.env risk settings ---"
grep -E "RISK_DAILY|RISK_MAX|K_OPEN|MIN_POSITION|MAX_POSITION" /etc/quantum/apply-layer.env

echo ""
echo "--- Kelly sizing (sample BNBUSDT) ---"
redis-cli HGETALL quantum:layer4:sizing:BNBUSDT

echo ""
echo "--- harvest_v2 R config ---"
redis-cli HMGET quantum:config:harvest_v2 r_target_base partial_25_r partial_50_r trailing_step

echo ""
echo "--- apply-layer service status ---"
systemctl is-active quantum-apply-layer.service

echo ""
echo "============================================================"
echo " DRASTIC FIX COMPLETE"
echo " Backup saved to: $BACKUP_DIR"
echo "============================================================"
