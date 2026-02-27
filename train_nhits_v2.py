"""
TRAIN NHITS V2 — 18-20 Months Binance Data
===========================================
Downloads the same 20 months of 1h OHLCV (5 symbols, same as LGBM).
Builds sliding-window sequences of shape [batch, 120, 49].
Trains SimpleNHiTS (from nhits_simple.py) with num_features=49.
Saves: ai_engine/models/nhits_model.pth + nhits_metadata.json

The checkpoint stores {"num_features": 49, ...} so nhits_agent.py
(line 96: checkpoint.get('num_features', 23)) will automatically
instantiate SimpleNHiTS(num_features=49) — no agent code change needed.

Run:
    cd c:/quantum_trader
    .\.venv\Scripts\Activate.ps1
    python train_nhits_v2.py
"""

import sys
import os
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_nhits_v2")

# ─── Config ───────────────────────────────────────────────────────────────────
SYMBOLS      = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
                "DOGEUSDT", "AVAXUSDT", "LTCUSDT", "TRXUSDT", "UNIUSDT",
                "INJUSDT", "SUIUSDT"]
INTERVAL     = "1h"
MONTHS       = 18
SEQ_LEN      = 120        # lookback window in candles
LABEL_PERCENTILE = 25    # top/bottom 25% per-symbol → BUY/SELL, middle 50% → HOLD
FORWARD_K    = 4          # 4h lookahead (less noise than 1h)
BATCH_SIZE   = 256
EPOCHS       = 60
LR           = 3e-4
WEIGHT_DECAY = 1e-4
TEST_FRAC    = 0.15
HIDDEN_SIZE  = 256

MODEL_DIR    = Path(__file__).parent / "ai_engine" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BINANCE_BASE = "https://api.binance.com/api/v3/klines"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Exact 49 feature names (MUST match lgbm_agent.py / unified_features.py) ─
FEATURE_NAMES = [
    "returns", "log_returns", "price_range", "body_size", "upper_wick", "lower_wick",
    "is_doji", "is_hammer", "is_engulfing", "gap_up", "gap_down",
    "rsi", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "roc",
    "ema_9", "ema_9_dist", "ema_21", "ema_21_dist",
    "ema_50", "ema_50_dist", "ema_200", "ema_200_dist",
    "sma_20", "sma_50",
    "adx", "plus_di", "minus_di",
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
    "atr", "atr_pct", "volatility",
    "volume_sma", "volume_ratio", "obv", "obv_ema", "vpt",
    "momentum_5", "momentum_10", "momentum_20", "acceleration", "relative_spread",
]
N_FEATURES = len(FEATURE_NAMES)   # 49


# ─── Data download (same logic as train_lgbm_v2) ─────────────────────────────

def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    all_rows = []
    current  = start_ms
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


def download_ohlcv(symbol: str, months: int) -> pd.DataFrame:
    log.info(f"  Downloading {symbol} ({months}mo × 1h) …")
    end_dt   = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=months * 30)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    rows     = _fetch_klines(symbol, start_ms, end_ms)

    if not rows:
        raise RuntimeError(f"No data for {symbol}")

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
    log.info(f"    → {len(df):,} candles")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    from backend.shared.unified_features import get_feature_engineer
    eng     = get_feature_engineer()
    df_feat = eng.compute_features(df.copy())
    out     = pd.DataFrame(index=df_feat.index)
    for col in FEATURE_NAMES:
        out[col] = df_feat[col] if col in df_feat.columns else 0.0
    out["close"] = df_feat["close"]
    return out


# ─── Sequence dataset ─────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Returns (seq [SEQ_LEN, 49], label [int]) pairs."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        """
        X: [T, 49]  already scaled float32
        y: [T]      int labels
        """
        assert len(X) == len(y)
        self.X       = torch.from_numpy(X)
        self.y       = torch.from_numpy(y).long()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        seq   = self.X[idx : idx + self.seq_len]          # [seq_len, 49]
        label = self.y[idx + self.seq_len]                 # scalar label at t+seq_len
        return seq, label


def build_sequences_from_symbol(df_feat: pd.DataFrame) -> tuple:
    """Build (X [T,49],  y [T]) for one symbol using percentile labels."""
    feat_arr = df_feat[FEATURE_NAMES].values.astype(np.float32)
    close    = df_feat["close"].values

    fwd_ret  = (np.roll(close, -FORWARD_K) - close) / (close + 1e-10) * 100
    fwd_ret[-FORWARD_K:] = np.nan  # mask tail

    # Percentile-based labels (same as LGBM/XGB): guaranteed balance per symbol
    valid    = fwd_ret[~np.isnan(fwd_ret)]
    buy_thresh  = np.nanpercentile(valid, 100 - LABEL_PERCENTILE)
    sell_thresh = np.nanpercentile(valid, LABEL_PERCENTILE)

    y = np.where(fwd_ret >= buy_thresh, 2,
        np.where(fwd_ret <= sell_thresh, 0, 1)).astype(np.int64)
    # Set tail rows to HOLD (will be excluded via seq_len offset anyway)
    y[np.isnan(fwd_ret)] = 1

    sell_c = int((y == 0).sum()); hold_c = int((y == 1).sum()); buy_c = int((y == 2).sum())
    log.info(f"    Labels: SELL={sell_c} HOLD={hold_c} BUY={buy_c}  sell_thresh={sell_thresh:.3f}% buy_thresh={buy_thresh:.3f}%")
    return feat_arr, y


# ─── Training loop ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(seqs)
        loss      = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        logits, _   = model(seqs)
        total_loss  += criterion(logits, labels).item() * len(labels)
        correct     += (logits.argmax(dim=1) == labels).sum().item()
        total       += len(labels)
    return total_loss / total, correct / total


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  NHiTS V2 — 18-20 month training run")
    log.info(f"  Device: {DEVICE}   SEQ_LEN={SEQ_LEN}   FEATURES={N_FEATURES}")
    log.info("=" * 60)

    assert N_FEATURES == 49, f"Feature count mismatch: {N_FEATURES}"

    # 1) Download & featurise
    all_X, all_y = [], []
    for sym in SYMBOLS:
        try:
            raw    = download_ohlcv(sym, MONTHS)
            feats  = build_features(raw)
            feats  = feats.dropna(subset=FEATURE_NAMES)
            X, y   = build_sequences_from_symbol(feats)
            all_X.append(X)
            all_y.append(y)
            log.info(f"  {sym}: {len(X):,} rows  (BUY={int((y==2).sum())}  HOLD={int((y==1).sum())}  SELL={int((y==0).sum())})")
        except Exception as e:
            log.error(f"  ❌ {sym} skipped: {e}")

    if not all_X:
        raise RuntimeError("No data built — aborting.")

    X_all = np.vstack(all_X).astype(np.float32)
    y_all = np.concatenate(all_y).astype(np.int64)
    log.info(f"\n  Total rows: {len(X_all):,}  features: {X_all.shape[1]}")

    # 2) Time-based split
    split   = int(len(X_all) * (1 - TEST_FRAC))
    X_train = X_all[:split];  y_train = y_all[:split]
    X_val   = X_all[split:];  y_val   = y_all[split:]

    # 3) Scale feature-wise (fit only on train)
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)

    # 4) Datasets & loaders
    ds_train = SequenceDataset(X_train, y_train, SEQ_LEN)
    ds_val   = SequenceDataset(X_val,   y_val,   SEQ_LEN)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    log.info(f"  Train sequences: {len(ds_train):,}   Val sequences: {len(ds_val):,}")

    # 5) Build model with num_features=49
    from ai_engine.nhits_simple import SimpleNHiTS
    model = SimpleNHiTS(
        input_size   = SEQ_LEN,
        hidden_size  = HIDDEN_SIZE,
        num_features = N_FEATURES,   # 49 — this is what matters
        dropout      = 0.15,
    ).to(DEVICE)

    # Class weights to counteract HOLD dominance
    counts      = np.bincount(y_train, minlength=3).astype(np.float32)
    weights     = (counts.sum() / (3.0 * counts + 1e-6))
    class_wt    = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion   = nn.CrossEntropyLoss(weight=class_wt)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_state   = None
    patience     = 15
    no_improve   = 0

    log.info(f"\n  Training {EPOCHS} epochs …")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, dl_train, optimizer, criterion, DEVICE)
        va_loss, va_acc = eval_epoch(model,  dl_val,   criterion, DEVICE)
        scheduler.step()

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
            marker       = " ← best"
        else:
            no_improve  += 1

        if epoch % 5 == 0 or epoch == 1 or marker:
            log.info(
                f"  Epoch {epoch:3d}/{EPOCHS}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  "
                f"val_loss={va_loss:.4f}  val_acc={va_acc*100:.1f}%{marker}"
            )

        if no_improve >= patience:
            log.info(f"  Early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # 6) Restore best and save
    if best_state is not None:
        model.load_state_dict(best_state)

    timestamp  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # Timestamped name so nhits_agent._find_latest_model('nhits_v*_v2.pth') picks it up
    model_path  = MODEL_DIR / f"nhits_v{timestamp}_v2.pth"
    scaler_path = MODEL_DIR / f"nhits_v{timestamp}_v2_scaler.pkl"
    meta_path   = MODEL_DIR / f"nhits_v{timestamp}_v2_metadata.json"
    # Also keep a stable alias for backward compat
    alias_path  = MODEL_DIR / "nhits_model.pth"

    # Checkpoint format nhits_agent.py expects:
    #   checkpoint['model_state_dict'], checkpoint['num_features'], etc.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_features":     N_FEATURES,      # ← nhits_agent line 96 reads this
        "sequence_length":  SEQ_LEN,
        "seq_len":          SEQ_LEN,
        "hidden_size":      HIDDEN_SIZE,
        "version":          "v2",
        "trained_at":       timestamp,
        "val_accuracy":     round(best_val_acc * 100, 2),
        # Scaler stats for inference normalisation (nhits_agent uses these)
        "feature_mean":     scaler.mean_.astype(np.float32),
        "feature_std":      scaler.scale_.astype(np.float32),
    }
    torch.save(checkpoint, str(model_path))
    torch.save(checkpoint, str(alias_path))  # stable alias

    # Save scaler for inference-time normalization
    import pickle as pkl
    with open(scaler_path, "wb") as fh:
        pkl.dump(scaler, fh)

    metadata = {
        "version":           "v2",
        "trained_at":        timestamp,
        "training_months":   MONTHS,
        "symbols":           SYMBOLS,
        "interval":          INTERVAL,
        "num_features":      N_FEATURES,
        "feature_names":     FEATURE_NAMES,
        "sequence_length":   SEQ_LEN,
        "hidden_size":       HIDDEN_SIZE,
        "label_percentile":  LABEL_PERCENTILE,
        "forward_k":         FORWARD_K,
        "epochs_trained":    epoch,
        "val_accuracy":      round(best_val_acc * 100, 2),
        "note":              "NHiTS v2 — 49 unified features — 18 months — 4h lookahead — 25pct percentile labels",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("")
    log.info("  ✅  Saved:")
    log.info(f"      {model_path}")
    log.info(f"      {scaler_path}")
    log.info(f"      {meta_path}")
    log.info(f"  Best val accuracy: {best_val_acc*100:.2f}%")
    log.info("  Done.")


if __name__ == "__main__":
    main()
