#!/usr/bin/env python3
"""
NHiTS v6 — Improved training vs v2
====================================
Key changes vs train_nhits_v2.py:
  1. LABEL_PERCENTILE = 33  →  equal 33/34/33 split instead of 25/50/25
     (50% HOLD in v2 trained the model that neutral-input = HOLD with 90% confidence)
  2. label_smoothing = 0.1  →  prevents overconfident HOLD at neutral inputs
  3. dropout = 0.20         →  slightly more regularization (was 0.15)
  4. EPOCHS = 80            →  longer training with cosine annealing
  5. MONTHS = 20            →  20 months data (was 18)
  Everything else identical to train_nhits_v2.py (4h lookahead, 12 symbols,
  120-candle sequences, class weights, SimpleNHiTS architecture).

Output: nhits_v{timestamp}_v3.pth  (agent autodiscover picks up 'nhits_v*_v2.pth'
  — update nhits_agent glob if you want 'v3')
"""
import sys, os, json, time, logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_nhits_v6")

# ─── Config ──────────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "AVAXUSDT", "LTCUSDT", "TRXUSDT", "UNIUSDT",
    "INJUSDT", "SUIUSDT",
]
INTERVAL         = "1h"
MONTHS           = 20               # 20 months history (was 18)
SEQ_LEN          = 120              # lookback window in candles
LABEL_PERCENTILE = 33               # ← KEY FIX (vs 25): equal 33/34/33 split
FORWARD_K        = 4                # 4h lookahead
LABEL_SMOOTHING  = 0.10             # ← NEW: prevents over-99% confidence
BATCH_SIZE       = 256
EPOCHS           = 80               # more epochs (was 60) — cos annealing
LR               = 3e-4
WEIGHT_DECAY     = 1e-4
DROPOUT          = 0.20             # slightly higher (was 0.15)
TEST_FRAC        = 0.15
HIDDEN_SIZE      = 256
PATIENCE         = 18

MODEL_DIR    = Path(__file__).parent.parent.parent / "ai_engine" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BINANCE_BASE = "https://api.binance.com/api/v3/klines"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 49 features — must match unified_features / nhits_agent
FEATURE_NAMES = [
    "returns", "log_returns", "price_range", "body_size", "upper_wick", "lower_wick",
    "is_doji", "is_hammer", "is_engulfing", "gap_up", "gap_down",
    "rsi", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "roc",
    "ema_9", "ema_9_dist", "ema_21", "ema_21_dist",
    "ema_50", "ema_50_dist", "ema_200", "ema_200_dist",
    "sma_20", "sma_50",
    "adx", "plus_di", "minus_di",
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
    "atr", "atr_pct", "volatility",
    "volume_sma", "volume_ratio", "obv", "obv_ema", "vpt",
    "momentum_5", "momentum_10", "momentum_20", "acceleration", "relative_spread",
]
N_FEATURES = 49
assert len(FEATURE_NAMES) == N_FEATURES


# ─── Data download ────────────────────────────────────────────────────────────

def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    all_rows, current = [], start_ms
    while current < end_ms:
        params = {"symbol": symbol, "interval": INTERVAL,
                  "startTime": current, "endTime": end_ms, "limit": 1000}
        for attempt in range(5):
            try:
                r = requests.get(BINANCE_BASE, params=params, timeout=15)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                log.warning(f"  Retry {attempt+1}/5 {symbol}: {e}")
                time.sleep(2 ** attempt)
        else:
            break
        if not batch:
            break
        all_rows.extend(batch)
        current = batch[-1][0] + 1
        time.sleep(0.12)
    return all_rows


def download_ohlcv(symbol: str, months: int) -> pd.DataFrame:
    log.info(f"  Downloading {symbol} ({months}mo × 1h)…")
    end_dt   = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=months * 30)
    rows     = _fetch_klines(
        symbol,
        int(start_dt.timestamp() * 1000),
        int(end_dt.timestamp()   * 1000),
    )
    if not rows:
        raise RuntimeError(f"No data for {symbol}")
    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","num_trades","taker_buy_base","taker_buy_quote","ignore",
    ])
    df = df[["open_time","open","high","low","close","volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.drop(columns=["open_time"], inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
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


def build_labels(df_feat: pd.DataFrame):
    """Per-symbol percentile labels: SELL=0 HOLD=1 BUY=2.
    LABEL_PERCENTILE=33 → ~33/34/33 split (vs 25→25/50/25).
    """
    close   = df_feat["close"].values
    fwd_ret = (np.roll(close, -FORWARD_K) - close) / (close + 1e-10) * 100
    fwd_ret[-FORWARD_K:] = np.nan

    valid        = fwd_ret[~np.isnan(fwd_ret)]
    buy_thresh   = np.nanpercentile(valid, 100 - LABEL_PERCENTILE)
    sell_thresh  = np.nanpercentile(valid, LABEL_PERCENTILE)

    y = np.where(fwd_ret >= buy_thresh, 2,
        np.where(fwd_ret <= sell_thresh, 0, 1)).astype(np.int64)
    y[np.isnan(fwd_ret)] = 1

    feat_arr = df_feat[FEATURE_NAMES].values.astype(np.float32)
    s, h, b  = int((y==0).sum()), int((y==1).sum()), int((y==2).sum())
    log.info(f"    Labels → SELL={s} HOLD={h} BUY={b}  "
             f"(sell≤{sell_thresh:.2f}% buy≥{buy_thresh:.2f}%)")
    return feat_arr, y


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X, self.y, self.seq_len = (
            torch.from_numpy(X),
            torch.from_numpy(y).long(),
            seq_len,
        )
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, i):
        return self.X[i : i + self.seq_len], self.y[i + self.seq_len]


# ─── Train / eval loops ───────────────────────────────────────────────────────

def train_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optim.zero_grad()
        logits, _ = model(seqs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        logits, _    = model(seqs)
        total_loss   += criterion(logits, labels).item() * len(labels)
        preds         = logits.argmax(1)
        correct      += (preds == labels).sum().item()
        total        += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  NHiTS v6  —  20mo  -  33pct-labels  -  label_smoothing=0.10")
    log.info(f"  Device: {DEVICE}   SEQ_LEN={SEQ_LEN}   FEATURES={N_FEATURES}")
    log.info("=" * 60)

    # 1) Download & featurise
    all_X, all_y = [], []
    for sym in SYMBOLS:
        try:
            raw   = download_ohlcv(sym, MONTHS)
            feats = build_features(raw).dropna(subset=FEATURE_NAMES)
            X, y  = build_labels(feats)
            all_X.append(X); all_y.append(y)
            log.info(f"  {sym}: {len(X):,} rows")
        except Exception as e:
            log.error(f"  ❌ {sym} skipped: {e}")

    if not all_X:
        raise RuntimeError("No data — aborting")

    X_all = np.vstack(all_X).astype(np.float32)
    y_all = np.concatenate(all_y).astype(np.int64)
    log.info(f"\n  Total: {len(X_all):,} rows  (SELL={int((y_all==0).sum())} "
             f"HOLD={int((y_all==1).sum())} BUY={int((y_all==2).sum())})")

    # 2) Split (time-based)
    split   = int(len(X_all) * (1 - TEST_FRAC))
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    # 3) Scale (feature-wise, fit on train only)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)

    # 4) Datasets & loaders
    ds_train = SequenceDataset(X_train, y_train, SEQ_LEN)
    ds_val   = SequenceDataset(X_val,   y_val,   SEQ_LEN)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    log.info(f"  Train seqs: {len(ds_train):,}   Val seqs: {len(ds_val):,}")

    # 5) Model
    from ai_engine.nhits_simple import SimpleNHiTS
    model = SimpleNHiTS(
        input_size   = SEQ_LEN,
        hidden_size  = HIDDEN_SIZE,
        num_features = N_FEATURES,
        dropout      = DROPOUT,
    ).to(DEVICE)

    # Class weights (counteract residual imbalance after 33pct percentile)
    counts   = np.bincount(y_train, minlength=3).astype(np.float32)
    weights  = counts.sum() / (3.0 * counts + 1e-6)
    wt       = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    log.info(f"  Class weights: SELL={wt[0]:.2f} HOLD={wt[1]:.2f} BUY={wt[2]:.2f}")

    # CrossEntropyLoss with label_smoothing — key fix for overconfident HOLD
    criterion = nn.CrossEntropyLoss(weight=wt, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 6) Train
    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    log.info(f"\n  Training {EPOCHS} epochs  (patience={PATIENCE})…")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, dl_train, optimizer, criterion, DEVICE)
        va_loss, va_acc, va_preds, va_labels = eval_epoch(model, dl_val, criterion, DEVICE)
        scheduler.step()

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
            marker       = " ← best"
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or marker:
            # Per-class recall
            from sklearn.metrics import classification_report
            rep  = classification_report(va_labels, va_preds, target_names=["SELL","HOLD","BUY"],
                                          output_dict=True, zero_division=0)
            s_rc = rep.get("SELL",{}).get("recall",0)
            h_rc = rep.get("HOLD",{}).get("recall",0)
            b_rc = rep.get("BUY", {}).get("recall",0)
            log.info(
                f"  Ep {epoch:3d}  train={tr_acc*100:.1f}%  "
                f"val={va_acc*100:.1f}%  "
                f"recall SELL={s_rc:.2f} HOLD={h_rc:.2f} BUY={b_rc:.2f}{marker}"
            )

        if no_improve >= PATIENCE:
            log.info(f"  Early stop at epoch {epoch}")
            break

    # 7) Restore & save
    if best_state:
        model.load_state_dict(best_state)

    # Quick sanity: random N(0,1) → expect ~33/33/34
    model.eval()
    np.random.seed(42); cnts = [0,0,0]
    with torch.no_grad():
        for _ in range(1000):
            X = torch.FloatTensor(np.random.randn(1, SEQ_LEN, N_FEATURES).astype("f")).to(DEVICE)
            p = torch.softmax(model(X)[0], dim=1)[0]
            cnts[p.argmax().item()] += 1
    log.info(f"\n  Sanity random N(0,1) x1000: SELL={cnts[0]} HOLD={cnts[1]} BUY={cnts[2]}")

    ts         = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path  = MODEL_DIR / f"nhits_v{ts}_v3.pth"   # v3 suffix = v6 training
    scaler_path = MODEL_DIR / f"nhits_v{ts}_v3_scaler.pkl"
    meta_path   = MODEL_DIR / f"nhits_v{ts}_v3_metadata.json"
    alias_path  = MODEL_DIR / "nhits_model.pth"         # stable alias

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_features":     N_FEATURES,
        "sequence_length":  SEQ_LEN,
        "seq_len":          SEQ_LEN,
        "hidden_size":      HIDDEN_SIZE,
        "version":          "v6-train",
        "trained_at":       ts,
        "val_accuracy":     round(best_val_acc * 100, 2),
        "feature_mean":     scaler.mean_.astype(np.float32),
        "feature_std":      scaler.scale_.astype(np.float32),
        "label_percentile": LABEL_PERCENTILE,
        "forward_k":        FORWARD_K,
        "sanity_random":    cnts,
    }
    torch.save(checkpoint, str(model_path))
    torch.save(checkpoint, str(alias_path))

    import pickle as pkl
    with open(scaler_path, "wb") as fh:
        pkl.dump(scaler, fh)

    with open(meta_path, "w") as f:
        json.dump({
            "version": f"nhits_v6_{ts}",
            "trained_at": ts, "months": MONTHS, "symbols": SYMBOLS,
            "num_features": N_FEATURES, "sequence_length": SEQ_LEN,
            "label_percentile": LABEL_PERCENTILE, "label_smoothing": LABEL_SMOOTHING,
            "forward_k": FORWARD_K, "dropout": DROPOUT,
            "val_accuracy": round(best_val_acc * 100, 2),
            "sanity_random_1000": {"SELL": cnts[0], "HOLD": cnts[1], "BUY": cnts[2]},
            "note": "v6: 33pct-labels + label_smoothing=0.10 — reduces neutral→HOLD overconfidence",
        }, f, indent=2)

    log.info(f"\n  ✅  Saved: {model_path.name}")
    log.info(f"  Best val accuracy: {best_val_acc*100:.2f}%")
    log.info(f"  Sanity: SELL={cnts[0]} HOLD={cnts[1]} BUY={cnts[2]}  (target ~333 each)")


if __name__ == "__main__":
    main()
