#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

echo "=== Last 10 minutes: XGB + LGBM prediction results ==="

$SSH '
# Update metadata JSON to match v6 features
cat > /opt/quantum/ai_engine/models/xgboost_v_prod_meta.json << ENDJSON
{"features": ["returns", "log_returns", "price_range", "body_size", "upper_wick", "lower_wick", "is_doji", "is_hammer", "is_engulfing", "gap_up", "gap_down", "rsi", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "roc", "ema_9", "ema_9_dist", "ema_21", "ema_21_dist", "ema_50", "ema_50_dist", "ema_200", "ema_200_dist", "sma_20", "sma_50", "adx", "plus_di", "minus_di", "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position", "atr", "atr_pct", "volatility", "volume_sma", "volume_ratio", "obv", "obv_ema", "vpt", "momentum_5", "momentum_10", "momentum_20", "acceleration", "relative_spread"], "model_type": "xgboost", "n_features": 49}
ENDJSON
echo "Updated xgboost_v_prod_meta.json to v6 (49 features)"

echo ""
echo "--- Last 10 min: XGB/LGBM prediction errors ---"
journalctl -u quantum-ai-engine --no-pager --since "-10m" | \
    grep -iE "xgboost prediction failed|lightgbm prediction failed|xgb.*error|lgbm.*error|features.*expecting|scaler.*none" | \
    tail -20

echo ""
echo "--- Last 10 min: XGB/LGBM successes ---"
journalctl -u quantum-ai-engine --no-pager --since "-10m" | \
    grep -iE "xgb .*BUY|xgb .*SELL|xgb .*HOLD|LGBM.*BUY|LGBM.*SELL|LGBM.*HOLD|\[XGB\]|\[LGBM\]" | \
    tail -20

echo ""
echo "--- Ensemble active agents (last 10 min) ---"
journalctl -u quantum-ai-engine --no-pager --since "-10m" | \
    grep "ACTIVE.*INACTIVE" | \
    tail -5
'
