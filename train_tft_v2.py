"""
TRAIN TFT V2 — 20 Months Binance Data
=======================================
Downloads 20 months of 1h OHLCV from Binance public API (no auth needed).
Computes all 49 unified features via UnifiedFeatureEngineer.
Trains the TFTModel from unified_agents.py: SELL=0, HOLD=1, BUY=2.
Saves checkpoint that TFTAgent._load_pytorch_model() can load directly.

Run:
    cd c:/quantum_trader
    .venv\\Scripts\\Activate.ps1
    python train_tft_v2.py
"""

import sys
import os
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_tft_v2")

# ─── Dependencies check ───────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    log.info(f"PyTorch {torch.__version__} — device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except ImportError:
    log.error("PyTorch not installed. Run: pip install torch")
    sys.exit(1)

# ─── Config ───────────────────────────────────────────────────────────────────
SYMBOLS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
             "DOGEUSDT", "AVAXUSDT", "LTCUSDT", "TRXUSDT",
             "UNIUSDT", "INJUSDT", "SUIUSDT"]
INTERVAL  = "1h"
MONTHS    = 20
LABEL_PERCENTILE = 20   # top/bottom 20% of forward returns = BUY/SELL (remaining 60% = HOLD)
FORWARD_K = 1           # candles ahead for label
TEST_FRAC = 0.15
VAL_FRAC  = 0.10

# Training hyperparameters
BATCH_SIZE  = 128
EPOCHS      = 50
LR          = 1e-4
PATIENCE    = 15        # early stopping — percentile labels is harder task
HIDDEN_SIZE = 128
NUM_HEADS   = 8
NUM_LAYERS  = 3
DROPOUT     = 0.1

MODEL_DIR    = Path(__file__).parent / "ai_engine" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BINANCE_BASE = "https://api.binance.com/api/v3/klines"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Data download ────────────────────────────────────────────────────────────

def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
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
        current = batch[-1][0] + 1
        time.sleep(0.12)

    return all_rows


def download_symbol(symbol: str) -> pd.DataFrame:
    now_ms  = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    # 20 months back (approx 20 * 30 * 24 * 3600 * 1000 ms)
    start_ms = now_ms - int(MONTHS * 30.44 * 24 * 3600 * 1000)

    log.info(f"  Downloading {symbol} ({MONTHS} months)…")
    rows = _fetch_klines(symbol, start_ms, now_ms)
    if not rows:
        raise RuntimeError(f"No data for {symbol}")

    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_vol","n_trades","tbb","tbq","ig"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    log.info(f"  {symbol}: {len(df)} candles")
    return df


# ─── Features ─────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    from backend.shared.unified_features import get_feature_engineer
    eng = get_feature_engineer()
    return eng.compute_features(df)


# ─── Labels ───────────────────────────────────────────────────────────────────

def make_labels(df: pd.DataFrame) -> pd.Series:
    fwd = df["close"].shift(-FORWARD_K) / df["close"] - 1.0
    # Percentile-based labeling: top/bottom 20% = BUY/SELL, middle 60% = HOLD
    # This ensures balanced classes regardless of market volatility regime
    buy_thresh  = np.nanpercentile(fwd.dropna().values, 100 - LABEL_PERCENTILE)
    sell_thresh = np.nanpercentile(fwd.dropna().values, LABEL_PERCENTILE)
    log.info(f"  Label thresholds: SELL<{sell_thresh*100:.4f}%  BUY>{buy_thresh*100:.4f}%")
    labels = pd.Series(1, index=df.index, dtype=int)  # 1 = HOLD
    labels[fwd >  buy_thresh]  = 2   # BUY
    labels[fwd <  sell_thresh] = 0   # SELL
    return labels


# ─── TFT Model (import from production code) ─────────────────────────────────

def get_tft_model(input_size=49) -> nn.Module:
    """Import and instantiate the exact same TFTModel used in production."""
    from ai_engine.agents.unified_agents import TFTModel
    model = TFTModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=3,
        dropout=DROPOUT,
    )
    log.info(f"TFTModel created: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        opt.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    return total_loss / total, correct / total


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("TFT V3 TRAINING — 20 months Binance data, 12 symbols, percentile labels")
    log.info("=" * 60)

    # ── Step 1: Download & feature-engineer all symbols ──────────────────
    FEATURE_NAMES = [
        'returns','log_returns','price_range','body_size','upper_wick','lower_wick',
        'is_doji','is_hammer','is_engulfing','gap_up','gap_down',
        'rsi','macd','macd_signal','macd_hist','stoch_k','stoch_d','roc',
        'ema_9','ema_9_dist','ema_21','ema_21_dist','ema_50','ema_50_dist',
        'ema_200','ema_200_dist','sma_20','sma_50','adx','plus_di','minus_di',
        'bb_middle','bb_upper','bb_lower','bb_width','bb_position',
        'atr','atr_pct','volatility','volume_sma','volume_ratio',
        'obv','obv_ema','vpt','momentum_5','momentum_10','momentum_20',
        'acceleration','relative_spread',
    ]

    all_X, all_y = [], []

    for sym in SYMBOLS:
        try:
            raw = download_symbol(sym)
            feat_df = build_features(raw)
            labels  = make_labels(feat_df)

            # Drop NaN rows
            feat_df = feat_df.ffill().bfill()
            valid   = feat_df[FEATURE_NAMES].notna().all(axis=1) & labels.notna()
            feat_df = feat_df[valid].reset_index(drop=True)
            labels  = labels[valid].reset_index(drop=True)
            labels  = labels.iloc[:len(feat_df)]  # align after forward shift

            X = feat_df[FEATURE_NAMES].values.astype(np.float32)
            y = labels.values.astype(np.int64)

            # Remove last FORWARD_K rows (no valid label)
            X = X[:-FORWARD_K]
            y = y[:-FORWARD_K]

            if len(X) < 500:
                log.warning(f"  {sym}: only {len(X)} samples — skipping (< 500)")
                continue

            all_X.append(X)
            all_y.append(y)
            log.info(f"  {sym}: {len(X)} samples — class dist: {np.bincount(y)}")
        except Exception as e:
            log.warning(f"  {sym}: failed ({e}) — skipping")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    log.info(f"\nTotal: {len(X_all):,} samples × {X_all.shape[1]} features")
    log.info(f"Class distribution: SELL={np.sum(y_all==0):,}  HOLD={np.sum(y_all==1):,}  BUY={np.sum(y_all==2):,}")

    # ── Step 2: Time-based split ──────────────────────────────────────────
    n = len(X_all)
    n_test = int(n * TEST_FRAC)
    n_val  = int(n * VAL_FRAC)
    n_train = n - n_test - n_val

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val   = X_all[n_train:n_train+n_val]
    y_val   = y_all[n_train:n_train+n_val]
    X_test  = X_all[n_train+n_val:]
    y_test  = y_all[n_train+n_val:]

    log.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── Step 3: Normalize features (sklearn StandardScaler — matches inference code) ─
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    log.info(f"Scaler fitted: mean={scaler.mean_[:3].round(4)} std={scaler.scale_[:3].round(4)} ...")

    # ── Step 4: DataLoaders ───────────────────────────────────────────────
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.LongTensor(y_val))
    test_ds  = TensorDataset(torch.FloatTensor(X_test),  torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Step 5: Model, optimizer, loss ───────────────────────────────────
    model = get_tft_model(input_size=49).to(DEVICE)

    # Mild class weights: slightly upweight BUY/SELL vs HOLD (2:1:2 ratio)
    # Do NOT use inverse-frequency weights — they cause SELL/BUY bias with 20/60/20 labels
    class_weights = torch.FloatTensor([2.0, 1.0, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Step 6: Training loop with early stopping ─────────────────────────
    best_val_loss = float("inf")
    best_state    = None
    best_val_acc  = 0.0
    patience_ctr  = 0

    log.info(f"\nTraining on {DEVICE} for up to {EPOCHS} epochs (patience={PATIENCE})…")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        log.info(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # ── Step 7: Test evaluation ───────────────────────────────────────────
    model.load_state_dict(best_state)
    test_loss, test_acc = eval_epoch(model, test_loader, criterion)
    log.info(f"\n✅ Best val_loss={best_val_loss:.4f} val_acc={best_val_acc:.4f}")
    log.info(f"✅ Test accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")

    # ── Step 8: Verify model is NOT constant output ───────────────────────
    model.eval()
    with torch.no_grad():
        x1 = torch.FloatTensor(X_test[:1]).to(DEVICE)
        x2 = torch.FloatTensor(X_test[-1:]).to(DEVICE)
        logits1 = model(x1)
        logits2 = model(x2)
        if torch.allclose(logits1, logits2, atol=1e-4):
            log.error("❌ ABORT: Model output is constant — did not learn properly")
            sys.exit(1)

    probs_test = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in test_loader:
            probs = torch.softmax(model(X_batch.to(DEVICE)), dim=1).cpu().numpy()
            probs_test.append(probs)
    probs_test = np.concatenate(probs_test, axis=0)
    pred_classes = probs_test.argmax(axis=1)
    class_names = ["SELL","HOLD","BUY"]
    log.info("Prediction distribution on test set:")
    for i, name in enumerate(class_names):
        pct = (pred_classes == i).mean() * 100
        log.info(f"  {name}: {pct:.1f}%")

    # Step 9: Logit calibration -- de-bias HOLD
    # TFT learns unconditional HOLD bias at data mean (zeros after scaling).
    # We use zero-input logit as calibration: subtracting it at inference
    # centers model so mean input -> equal probs ~[0.33, 0.33, 0.33].
    model.eval()
    with torch.no_grad():
        X_zero_cal = torch.FloatTensor(scaler.transform([np.zeros(49)])).to(DEVICE)
        calibration_bias = model(X_zero_cal).cpu()[0]  # shape [3]
    log.info(f"Calibration bias (zero-input logits): {calibration_bias.numpy().round(3)}")

    # Verify calibration effect
    with torch.no_grad():
        raw_logits_v = model(X_zero_cal)
        cal_logits_v = raw_logits_v - calibration_bias.to(DEVICE)
        raw_probs = torch.softmax(raw_logits_v, dim=1)
        cal_probs = torch.softmax(cal_logits_v, dim=1)
        log.info(f"Before calibration -- SELL={raw_probs[0,0]:.3f} HOLD={raw_probs[0,1]:.3f} BUY={raw_probs[0,2]:.3f}")
        log.info(f"After  calibration -- SELL={cal_probs[0,0]:.3f} HOLD={cal_probs[0,1]:.3f} BUY={cal_probs[0,2]:.3f}  (should be ~0.333)")

    # ── Step 10: Save checkpoint ────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"tft_v6_{ts}_v3.pth"
    meta_filename  = f"tft_v6_{ts}_v3_metadata.json"
    model_path = MODEL_DIR / model_filename
    meta_path  = MODEL_DIR / meta_filename

    checkpoint = {
        # Architecture params (read by TFTAgent._load_pytorch_model)
        "model_state_dict": best_state,
        "input_size":    49,
        "num_features":  49,
        "hidden_size":   HIDDEN_SIZE,
        "num_heads":     NUM_HEADS,
        "num_layers":    NUM_LAYERS,
        "num_classes":   3,
        "dropout":       DROPOUT,
        # Logit calibration bias (subtract at inference for HOLD de-bias)
        "calibration_bias": calibration_bias.numpy().tolist(),
        # Training stats
        "val_acc":       best_val_acc,
        "test_acc":      test_acc,
        "training_samples": len(X_train),
        "trained_at":    datetime.now(tz=timezone.utc).isoformat(),
    }

    torch.save(checkpoint, model_path)
    log.info(f"✅ Checkpoint saved: {model_path}")

    # Save sklearn StandardScaler — REQUIRED by TFTAgent.predict() in inference
    scaler_filename = f"tft_v6_{ts}_v3_scaler.pkl"
    scaler_path_out = MODEL_DIR / scaler_filename
    joblib.dump(scaler, scaler_path_out)
    log.info(f"✅ Scaler saved:     {scaler_path_out}")

    metadata = {
        "version":          f"tft_v6_{ts}_v3",
        "features":         FEATURE_NAMES,
        "num_features":     49,
        "feature_schema":   "FEATURES_V6",
        "model_type":       "TFT",
        "architecture": {
            "input_size":  49,
            "hidden_size": HIDDEN_SIZE,
            "num_heads":   NUM_HEADS,
            "num_layers":  NUM_LAYERS,
            "num_classes": 3,
            "dropout":     DROPOUT,
        },
        "training_date":      datetime.now(tz=timezone.utc).isoformat(),
        "training_samples":   int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples":       int(len(X_test)),
        "test_accuracy":      float(test_acc),
        "best_val_loss":      float(best_val_loss),
        "total_parameters":   sum(p.numel() for p in model.parameters()),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"✅ Metadata saved: {meta_path}")

    # Also save as _meta.json — BaseAgent._load() looks for this suffix
    meta_short_path = MODEL_DIR / f"tft_v6_{ts}_v3_meta.json"
    with open(meta_short_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"✅ Meta (short) saved: {meta_short_path}")

    log.info(f"\n{'='*60}")
    log.info(f"TRAINING COMPLETE")
    log.info(f"  Model:         {model_filename}")
    log.info(f"  Test accuracy: {test_acc*100:.2f}%")
    log.info(f"  Parameters:    {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"{'='*60}")
    log.info("\nNeste steg:")
    log.info(f"  scp ai_engine/models/{model_filename}  root@46.224.116.254:/home/qt/quantum_trader/ai_engine/models/")
    log.info(f"  scp ai_engine/models/{scaler_filename} root@46.224.116.254:/home/qt/quantum_trader/ai_engine/models/")
    log.info(f"  scp ai_engine/models/{meta_filename}   root@46.224.116.254:/home/qt/quantum_trader/ai_engine/models/")
    log.info(f"  scp ai_engine/models/tft_v6_{ts}_v2_meta.json root@46.224.116.254:/home/qt/quantum_trader/ai_engine/models/")
    log.info("  ssh root@46.224.116.254 'systemctl restart quantum-ensemble-predictor'")


if __name__ == "__main__":
    main()
