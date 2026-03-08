#!/bin/bash
# Quantum Trader — Full Model Retraining Pipeline
# Fetches fresh data from Binance mainnet and retrains all base models
# Run as: bash /opt/quantum/ops/retrain/_retrain_all.sh >> /var/log/quantum/retrain.log 2>&1

set -e
LOGFILE="/var/log/quantum/retrain.log"
MODELS_DIR="/opt/quantum/ai_engine/models"
RETRAIN_DIR="/opt/quantum/ops/retrain"
VENV="/opt/quantum/venvs/ai-engine/bin/python"
QT_ROOT="/opt/quantum"

mkdir -p /var/log/quantum

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=========================================="
log "  QUANTUM TRADER — MODEL RETRAINING START"
log "=========================================="
log "Models dir: $MODELS_DIR"
log "Python: $VENV"

# ── 1. XGBoost ──────────────────────────────────────────────────────────────
log ""
log "━━ STEP 1/3: XGBoost v5 (Binance mainnet public API) ━━"
cd "$QT_ROOT"
$VENV "$RETRAIN_DIR/fetch_and_train_xgb_v5.py"
XGB_STATUS=$?

if [ $XGB_STATUS -eq 0 ]; then
    # Find the newest xgb_v* file
    NEWEST_XGB=$(ls -t "$MODELS_DIR"/xgb_v*_v5.pkl 2>/dev/null | head -1)
    NEWEST_XGB_SCALER=$(ls -t "$MODELS_DIR"/xgb_v*_v5_scaler.pkl 2>/dev/null | head -1)
    if [ -n "$NEWEST_XGB" ]; then
        ln -sf "$(basename $NEWEST_XGB)" "$MODELS_DIR/xgb_model.pkl"
        ln -sf "$(basename $NEWEST_XGB)" "$MODELS_DIR/xgboost_v_prod.pkl"
        log "[OK] xgb_model.pkl → $(basename $NEWEST_XGB)"
    fi
    if [ -n "$NEWEST_XGB_SCALER" ]; then
        ln -sf "$(basename $NEWEST_XGB_SCALER)" "$MODELS_DIR/xgboost_v_prod_scaler.pkl"
        cp "$NEWEST_XGB_SCALER" "$MODELS_DIR/scaler.pkl"
        log "[OK] scaler.pkl updated from $(basename $NEWEST_XGB_SCALER)"
    fi
else
    log "[WARN] XGBoost training failed (exit $XGB_STATUS) — keeping existing model"
fi

# ── 2. LightGBM v6 ──────────────────────────────────────────────────────────
log ""
log "━━ STEP 2/3: LightGBM v6 (49 features) ━━"
cd "$QT_ROOT"
$VENV "$RETRAIN_DIR/train_lightgbm_v6.py"
LGBM_STATUS=$?

if [ $LGBM_STATUS -eq 0 ]; then
    NEWEST_LGBM=$(ls -t "$MODELS_DIR"/lgbm_v6_*.txt 2>/dev/null | head -1)
    NEWEST_LGBM_SCALER=$(ls -t "$MODELS_DIR"/lgbm_v6_*_scaler.pkl 2>/dev/null | head -1)
    if [ -n "$NEWEST_LGBM" ]; then
        ln -sf "$(basename $NEWEST_LGBM)" "$MODELS_DIR/lgbm_model.pkl"
        ln -sf "$(basename $NEWEST_LGBM)" "$MODELS_DIR/lightgbm_v_prod.pkl"
        log "[OK] lgbm_model.pkl → $(basename $NEWEST_LGBM)"
    fi
    if [ -n "$NEWEST_LGBM_SCALER" ]; then
        ln -sf "$(basename $NEWEST_LGBM_SCALER)" "$MODELS_DIR/lgbm_scaler.pkl"
        log "[OK] lgbm_scaler.pkl updated"
    fi
else
    log "[WARN] LightGBM training failed (exit $LGBM_STATUS) — keeping existing model"
fi

# ── 3. TFT v6 ───────────────────────────────────────────────────────────────
log ""
log "━━ STEP 3/3: TFT v6 (neural network — may take 20-40 min) ━━"
cd "$QT_ROOT"
$VENV "$RETRAIN_DIR/train_tft_v6.py"
TFT_STATUS=$?

if [ $TFT_STATUS -eq 0 ]; then
    NEWEST_TFT=$(ls -t "$MODELS_DIR"/tft_v6_*.pth 2>/dev/null | head -1)
    if [ -n "$NEWEST_TFT" ]; then
        ln -sf "$(basename $NEWEST_TFT)" "$MODELS_DIR/tft_model.pth"
        log "[OK] tft_model.pth → $(basename $NEWEST_TFT)"
    fi
else
    log "[WARN] TFT training failed (exit $TFT_STATUS) — keeping existing model"
fi

# ── Copy approved models to registry ────────────────────────────────────────
log ""
log "━━ Syncing to model_registry/approved/ ━━"
APPROVED="/opt/quantum/model_registry/approved"
[ -n "$NEWEST_XGB" ] && cp "$NEWEST_XGB" "$APPROVED/" && log "[OK] XGB → registry"
[ -n "$NEWEST_LGBM" ] && cp "$NEWEST_LGBM" "$APPROVED/" && log "[OK] LGBM → registry"
[ -n "$NEWEST_TFT" ] && cp "$NEWEST_TFT" "$APPROVED/" && log "[OK] TFT → registry"

# ── Restart AI engine ────────────────────────────────────────────────────────
log ""
log "━━ Restarting AI engine ━━"
systemctl restart quantum-ai-engine.service 2>/dev/null || \
    docker restart quantum-ai-engine 2>/dev/null || \
    log "[WARN] Could not restart AI engine — restart manually"
sleep 5
systemctl is-active quantum-ai-engine.service 2>/dev/null && log "[OK] AI engine running" || \
    docker inspect --format='{{.State.Status}}' quantum-ai-engine 2>/dev/null | grep -q running && log "[OK] AI engine container running"

log ""
log "=========================================="
log "  RETRAINING COMPLETE"
log "  XGB: $XGB_STATUS | LGBM: $LGBM_STATUS | TFT: $TFT_STATUS"
log "=========================================="
