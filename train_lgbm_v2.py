"""
TRAIN LGBM V2 — 18-24 Months Binance Data
==========================================
Downloads 18 months of 1h OHLCV from Binance public API (no auth needed).
Computes all 49 unified features via UnifiedFeatureEngineer.
Trains a real LightGBMClassifier: SELL=0, HOLD=1, BUY=2.
Saves: ai_engine/models/lgbm_model.pkl + lgbm_scaler.pkl + lgbm_metadata.json

Run:
    cd c:/quantum_trader
    .\.venv\Scripts\Activate.ps1
    python train_lgbm_v2.py
"""

import sys
import os
import json
import pickle
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_lgbm_v2")

# ─── Config ───────────────────────────────────────────────────────────────────
# v3: 12 symbols + percentile labels (same approach as TFT v3)
SYMBOLS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
             "DOGEUSDT", "AVAXUSDT", "LTCUSDT", "TRXUSDT", "UNIUSDT",
             "INJUSDT", "SUIUSDT"]
INTERVAL  = "1h"
MONTHS    = 18          # download window (months)
LABEL_PERCENTILE = 25  # top/bottom 25% per-symbol → BUY/SELL, middle 50% → HOLD
FORWARD_K = 4           # candles ahead for label (4h lookahead has more predictable signal than 1h)
TEST_FRAC = 0.15        # hold-out fraction (time-based, NOT shuffled)

MODEL_DIR = Path(__file__).parent / "ai_engine" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BINANCE_BASE = "https://api.binance.com/api/v3/klines"

# ─── Data download ────────────────────────────────────────────────────────────

def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    """Paginated fetch from Binance public klines endpoint."""
    all_rows = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  INTERVAL,
            "startTime": current,
            "endTime":   end_ms,
            "limit":     1000,
        }
        for attempt in range(5):
            try:
                r = requests.get(BINANCE_BASE, params=params, timeout=15)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                log.warning(f"  Retry {attempt+1}/5 for {symbol}: {e}")
                time.sleep(2 ** attempt)
        else:
            log.error(f"  ❌ Gave up fetching {symbol} @ {current}")
            break

        if not batch:
            break

        all_rows.extend(batch)
        # next page starts after last candle open time
        current = batch[-1][0] + 1
        time.sleep(0.12)   # stay within rate limit (weight ~2 per call)

    return all_rows


def download_ohlcv(symbol: str, months: int) -> pd.DataFrame:
    """Download `months` of 1h candles. Returns tidy DataFrame."""
    log.info(f"  Downloading {symbol} ({months}mo × 1h) …")
    end_dt   = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=months * 30)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    rows = _fetch_klines(symbol, start_ms, end_ms)
    if not rows:
        raise RuntimeError(f"No data returned for {symbol}")

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.drop(columns=["open_time"], inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info(f"    → {len(df):,} candles  ({df['timestamp'].iloc[0].date()} … {df['timestamp'].iloc[-1].date()})")
    return df


# ─── Feature engineering ─────────────────────────────────────────────────────

FEATURE_NAMES = [
    # price / candle
    "returns", "log_returns", "price_range", "body_size", "upper_wick", "lower_wick",
    "is_doji", "is_hammer", "is_engulfing", "gap_up", "gap_down",
    # oscillators
    "rsi", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "roc",
    # EMA + distance
    "ema_9", "ema_9_dist", "ema_21", "ema_21_dist",
    "ema_50", "ema_50_dist", "ema_200", "ema_200_dist",
    # SMA
    "sma_20", "sma_50",
    # ADX
    "adx", "plus_di", "minus_di",
    # Bollinger Bands
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
    # Volatility
    "atr", "atr_pct", "volatility",
    # Volume
    "volume_sma", "volume_ratio", "obv", "obv_ema", "vpt",
    # Momentum / microstructure
    "momentum_5", "momentum_10", "momentum_20", "acceleration", "relative_spread",
]
N_FEATURES = len(FEATURE_NAMES)   # must be 49


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 49 unified features via UnifiedFeatureEngineer,
    then select exactly FEATURE_NAMES columns in order.
    Missing columns are filled with 0.
    """
    from backend.shared.unified_features import get_feature_engineer
    eng = get_feature_engineer()
    df_feat = eng.compute_features(df.copy())

    # Select / fill in order
    out = pd.DataFrame(index=df_feat.index)
    for col in FEATURE_NAMES:
        out[col] = df_feat[col] if col in df_feat.columns else 0.0

    # Carry forward the close for labeling
    out["close"]     = df_feat["close"]
    out["timestamp"] = df_feat["timestamp"] if "timestamp" in df_feat.columns else df["timestamp"].values
    return out


def make_labels(df: pd.DataFrame, k: int = 1, percentile: int = 20) -> pd.Series:
    """
    Percentile-based forward-return label (v3):
      forward_ret = (close[t+k] - close[t]) / close[t] * 100
      BUY=2  if forward_ret >= top percentile threshold (per symbol)
      SELL=0 if forward_ret <= bottom percentile threshold (per symbol)
      HOLD=1 otherwise
    Guarantees balanced labels: bottom 20% = SELL, top 20% = BUY, middle 60% = HOLD.
    """
    fwd = (df["close"].shift(-k) / df["close"] - 1.0) * 100.0
    buy_thresh  = np.nanpercentile(fwd.values, 100 - percentile)
    sell_thresh = np.nanpercentile(fwd.values, percentile)
    label = np.where(fwd >= buy_thresh, 2, np.where(fwd <= sell_thresh, 0, 1))
    log.info(f"  Percentile labels: sell_thresh={sell_thresh:.4f}%, buy_thresh={buy_thresh:.4f}%")
    log.info(f"  Distribution: SELL={int((label==0).sum())} HOLD={int((label==1).sum())} BUY={int((label==2).sum())}")
    return pd.Series(label, index=df.index, name="label")


# ─── Training ─────────────────────────────────────────────────────────────────

def train_lgbm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray,   y_val: np.ndarray) -> object:
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm not installed — run: pip install lightgbm")
        raise

    log.info(f"  Training LightGBM  (train={len(X_train):,}  val={len(X_val):,}  features={X_train.shape[1]})")
    log.info(f"  Label distribution (train): { {int(v): int(c) for v,c in zip(*np.unique(y_train, return_counts=True))} }")

    model = lgb.LGBMClassifier(
        n_estimators      = 3000,
        learning_rate     = 0.03,     # slower LR → better generalisation
        max_depth         = 5,        # shallower trees prevent overfit
        num_leaves        = 24,
        min_child_samples = 150,      # larger leaf size → smoother predictions
        subsample         = 0.75,
        subsample_freq    = 1,
        colsample_bytree  = 0.70,
        reg_alpha         = 0.3,
        reg_lambda        = 2.0,      # stronger L2
        min_gain_to_split = 0.02,     # only split on meaningful gain
        # Data is 25/50/25 (SELL/HOLD/BUY). Tested weights:
        #   None           → HOLD recall=86%, dominates like XGB (bad)
        #   {1:0.5}        → HOLD prob 11-19%, effective share=25%, never wins (bad)
        #   {1:0.75}       → HOLD effective share=37.5% → should win in neutral markets
        class_weight      = {0: 1.0, 1: 0.75, 2: 1.0},
        n_jobs            = -1,
        random_state      = 42,
        verbose           = -1,
    )

    model.fit(
        X_train, y_train,
        eval_set     = [(X_val, y_val)],
        eval_metric  = "multi_logloss",
        callbacks    = [lgb.early_stopping(stopping_rounds=150, verbose=True),
                        lgb.log_evaluation(period=100)],
    )

    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_val)
    acc    = accuracy_score(y_val, y_pred)
    log.info(f"\n  Val accuracy: {acc*100:.2f}%")
    log.info("\n" + classification_report(
        y_val, y_pred,
        target_names=["SELL", "HOLD", "BUY"],
        zero_division=0,
    ))

    # Save raw LGBMClassifier — class_weight={0:1, 1:0.75, 2:1}: HOLD effective
    # share 37.5% (vs 50% raw). Allows HOLD to win in neutral markets without
    # dominating. Tuned from: None→86% recall, 0.5→never wins, 0.75→balanced.
    return model, float(acc)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  LGBM V2 — 18-20 month training run")
    log.info("=" * 60)

    assert N_FEATURES == 49, f"Feature list must have 49 entries, got {N_FEATURES}"

    # 1) Download & featurise all symbols
    all_X, all_y = [], []
    for sym in SYMBOLS:
        try:
            raw   = download_ohlcv(sym, MONTHS)
            feats = build_features(raw)
            feats["label"] = make_labels(feats, k=FORWARD_K, percentile=LABEL_PERCENTILE)

            # Drop tail (no forward label) and any NaN
            feats = feats.dropna(subset=FEATURE_NAMES + ["label"])
            feats = feats[:-FORWARD_K]  # remove last k rows (no future close)

            X = feats[FEATURE_NAMES].values.astype(np.float32)
            y = feats["label"].values.astype(int)

            all_X.append(X)
            all_y.append(y)
            log.info(f"  {sym}: {len(X):,} samples  (BUY={int((y==2).sum())}  HOLD={int((y==1).sum())}  SELL={int((y==0).sum())})")
        except Exception as e:
            log.error(f"  ❌ {sym} skipped: {e}")

    if not all_X:
        raise RuntimeError("No data could be built — aborting.")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    log.info(f"\n  Total rows: {len(X_all):,}  features: {X_all.shape[1]}")

    # 2) Time-based split (NOT shuffled — respect temporal order)
    split = int(len(X_all) * (1 - TEST_FRAC))
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    # 3) Scale features
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # 4) Train
    model, val_acc = train_lgbm(X_train, y_train, X_val, y_val)

    # 5) Save artifacts
    timestamp   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path  = MODEL_DIR / f"lightgbm_v{timestamp}_v3.pkl"
    scaler_path = MODEL_DIR / f"lightgbm_v{timestamp}_v3_scaler.pkl"
    meta_path   = MODEL_DIR / f"lightgbm_v{timestamp}_v3_metadata.json"

    with open(model_path,  "wb") as f: pickle.dump(model,  f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)

    feature_imp = {}
    if hasattr(model, "feature_importances_"):
        feature_imp = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))

    metadata = {
        "version":           "v3",
        "trained_at":        timestamp,
        "training_months":   MONTHS,
        "symbols":           SYMBOLS,
        "interval":          INTERVAL,
        "num_features":      N_FEATURES,
        "feature_names":     FEATURE_NAMES,
        "label_percentile":  LABEL_PERCENTILE,
        "forward_k":         FORWARD_K,
        "train_samples":     int(len(X_train)),
        "val_samples":       int(len(X_val)),
        "val_accuracy":      round(val_acc * 100, 2),
        "best_iteration":    int(getattr(model, "best_iteration_", -1)),
        "feature_importance": feature_imp,
        "note":              "LightGBM v3 — 49 features — 18mo — 4h lookahead — 25pct labels — raw (no calibration)",
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("")
    log.info("  ✅  Saved:")
    log.info(f"      {model_path}")
    log.info(f"      {scaler_path}")
    log.info(f"      {meta_path}")
    log.info(f"  Val accuracy : {val_acc*100:.2f}%")
    log.info(f"  Best iteration: {metadata['best_iteration']}")
    log.info("  Done.")


if __name__ == "__main__":
    main()
