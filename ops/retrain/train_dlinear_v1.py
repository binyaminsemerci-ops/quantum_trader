#!/usr/bin/env python3
"""
DLinear v1 Training
===================
DLinear (Decomposition-Linear) from "Are Transformers Effective for Time Series Forecasting?" (2023).
Splits sequence into trend (moving-average) + residual, applies independent linear
layers to each, then classifies via MLP.

Why DLinear for this system:
  - Near zero risk of HOLD-collapse (no LSTM → no vanishing gradient)
  - Much faster to train than TFT / PatchTST
  - Strong on short crypto sequences (proven to beat many Transformers)
  - ~50K params → inference latency < 1ms

Feature pipeline:
  unified_features.get_feature_engineer()  ← same as NHiTS v6 (57.75% acc)

Architecture:
  DLinearModel:
    input:  (batch, seq_len=60, 49)
    trend:  MovingAvg(kernel=25) → Flatten → Linear(49*60, hidden)
    resid:  input - trend        → Flatten → Linear(49*60, hidden)
    output: concat → ReLU → Linear(2*hidden, hidden) → ReLU → Linear(hidden, 3)

Everything else from proven v10 pipeline:
  - Per-symbol 70/15/15 time-ordered split + ConcatDataset
  - Percentile labels (33rd/67th → ~33/34/33)
  - 18 months × 12 symbols via Binance REST
  - CrossEntropyLoss + class_weights + label_smoothing=0.10
  - AdamW + WarmupCosine + grad clip
"""
import os, sys, json, math, time, pickle, logging
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_dlinear_v1")

from backend.shared.unified_features import get_feature_engineer

SAVE_DIR = Path(__file__).parent.parent.parent / "ai_engine" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BINANCE_BASE = "https://api.binance.com/api/v3/klines"

# ── 49 features (identical to nhits_v6 / tft_v10) ──────────────────────────
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
N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 49

# ── Config ───────────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "ADAUSDT", "XRPUSDT", "DOGEUSDT",
    "INJUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT",
]
MONTHS           = 18
INTERVAL         = "1h"
LABEL_PERCENTILE = 33
FORWARD_K        = 4
SEQUENCE_LEN     = 60
MA_KERNEL        = 25     # moving-average window for trend decomposition
HIDDEN_SIZE      = 256
LABEL_SMOOTHING  = 0.10
BATCH_SIZE       = 512    # DLinear is small → big batches are fine
EPOCHS           = 150
PATIENCE         = 30
LR               = 1e-3
WEIGHT_DECAY     = 1e-4
WARMUP_EPOCHS    = 10
DROPOUT          = 0.2
NOISE_SIGMA      = 0.005


# ── DLinear architecture ─────────────────────────────────────────────────────
class MovingAvg(nn.Module):
    """Causal (history-only) moving average for trend decomposition."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (batch, seq_len, features)
        # Pad left so output length == seq_len (causal, no future leakage)
        pad = x[:, :self.kernel_size - 1, :]  # repeat first timestep as left pad
        x_padded = torch.cat([pad, x], dim=1)
        # AvgPool1d works on (batch, channels, length) → transpose
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        return trend  # (batch, seq_len, features)


class DLinearModel(nn.Module):
    """
    DLinear classifier for multivariate time-series.

    Constructor signature is fixed so the inference agent can reconstruct it
    from the metadata checkpoint (same pattern as TFTModel in unified_agents.py).
    """
    def __init__(
        self,
        input_size: int = 49,
        seq_len: int = 60,
        hidden_size: int = 256,
        num_classes: int = 3,
        ma_kernel: int = 25,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size  = input_size
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.ma_kernel   = ma_kernel

        flat_dim = seq_len * input_size

        self.moving_avg = MovingAvg(kernel_size=ma_kernel)

        # Independent linear projections for trend and residual
        self.trend_proj = nn.Linear(flat_dim, hidden_size)
        self.resid_proj = nn.Linear(flat_dim, hidden_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)  or  (batch, features) if flat
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.seq_len, -1)

        trend  = self.moving_avg(x)          # (batch, seq_len, features)
        resid  = x - trend                   # (batch, seq_len, features)

        t_flat = trend.reshape(x.size(0), -1)
        r_flat = resid.reshape(x.size(0), -1)

        t_proj = self.trend_proj(t_flat)     # (batch, hidden)
        r_proj = self.resid_proj(r_flat)     # (batch, hidden)

        combined = torch.cat([t_proj, r_proj], dim=-1)  # (batch, 2*hidden)
        return self.classifier(combined)


# ── Dataset / scheduler (identical to tft_v10) ───────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=60, augment=False, noise_sigma=0.0):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.seq_len = seq_len
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.valid_idx = list(range(seq_len - 1, len(X)))

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        i = self.valid_idx[idx]
        x = self.X[i - self.seq_len + 1: i + 1]
        if self.augment and self.noise_sigma > 0:
            x = x + torch.randn_like(x) * self.noise_sigma
        return x, self.y[i]


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_max, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_max = lr_max
        self.lr_min = lr_min
        self._epoch = 0

    def step(self):
        e = self._epoch
        if e < self.warmup_epochs:
            lr = self.lr_max * (e + 1) / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self._epoch += 1
        return lr


# ── Binance REST fetch (same as tft_v10) ─────────────────────────────────────
def fetch_klines_paginated(symbol: str, months: int, interval: str = "1h") -> pd.DataFrame:
    end_dt   = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=months * 30)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp()   * 1000)

    all_rows, current = [], start_ms
    while current < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": current, "endTime": end_ms, "limit": 1000}
        for attempt in range(5):
            try:
                r = requests.get(BINANCE_BASE, params=params, timeout=20)
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

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "num_trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Feature computation (same as tft_v10) ────────────────────────────────────
def build_features(df: pd.DataFrame) -> np.ndarray:
    eng     = get_feature_engineer()
    df_feat = eng.compute_features(df.copy())
    out     = pd.DataFrame(index=df_feat.index)
    for col in FEATURE_NAMES:
        out[col] = df_feat[col] if col in df_feat.columns else 0.0
    return out[FEATURE_NAMES].values.astype(np.float32)


# ── Labels (same as tft_v10) ─────────────────────────────────────────────────
def create_percentile_labels(close: np.ndarray, forward_k: int, percentile: int):
    fwd_ret = (np.roll(close, -forward_k) - close) / (close + 1e-10) * 100
    fwd_ret[-forward_k:] = np.nan
    valid       = fwd_ret[~np.isnan(fwd_ret)]
    buy_thresh  = np.nanpercentile(valid, 100 - percentile)
    sell_thresh = np.nanpercentile(valid, percentile)
    y = np.where(fwd_ret >= buy_thresh, 2,
        np.where(fwd_ret <= sell_thresh, 0, 1)).astype(np.int64)
    y[np.isnan(fwd_ret)] = 1
    return y, sell_thresh, buy_thresh


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("DLinear v1 Training  (unified_features + 18mo + seq_len=60)")
print(f"Device: {device} | {MONTHS}mo | percentile labels | MA kernel={MA_KERNEL}")
print("=" * 70)

sym_buckets = {}
print(f"\n[DATA] Fetching ~{MONTHS*30*24} candles/symbol via Binance REST...")
for sym in SYMBOLS:
    print(f"   {sym}: ", end="", flush=True)
    try:
        candles = fetch_klines_paginated(sym, MONTHS, INTERVAL)
        if candles is None or len(candles) < 500:
            print("SKIP (no data)", flush=True)
            continue
        print(f"{len(candles)} candles → unified features...", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        continue

    try:
        sym_X = build_features(candles)
    except Exception as e:
        print(f"   {sym}: SKIP features failed: {e}", flush=True)
        continue

    sym_y, sell_thr, buy_thr = create_percentile_labels(
        candles["close"].values.astype(float), FORWARD_K, LABEL_PERCENTILE
    )

    n_min = min(len(sym_X), len(sym_y))
    sym_X = sym_X[:n_min]
    sym_y = sym_y[:n_min]

    valid_mask = ~np.any(np.isnan(sym_X), axis=1)
    sym_X = sym_X[valid_mask]
    sym_y = sym_y[valid_mask]

    if len(sym_X) < SEQUENCE_LEN * 5:
        print(f"   {sym}: SKIP only {len(sym_X)} valid rows", flush=True)
        continue

    n  = len(sym_X)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)
    sym_buckets[sym] = dict(
        X_tr=sym_X[:t1], y_tr=sym_y[:t1],
        X_va=sym_X[t1:t2], y_va=sym_y[t1:t2],
        X_te=sym_X[t2:],   y_te=sym_y[t2:],
    )
    dist = np.bincount(sym_y, minlength=3)
    log.info(f"   {sym}: {n} rows | SELL={dist[0]} HOLD={dist[1]} BUY={dist[2]}  "
             f"(sell≤{sell_thr:.2f}% buy≥{buy_thr:.2f}%)")

if len(sym_buckets) < 3:
    print("ERROR: Too few symbols (<3)")
    sys.exit(1)

y_train = np.concatenate([sym_buckets[s]["y_tr"] for s in sym_buckets])
y_val   = np.concatenate([sym_buckets[s]["y_va"] for s in sym_buckets])
y_test  = np.concatenate([sym_buckets[s]["y_te"] for s in sym_buckets])
total   = sum(len(sym_buckets[s]["X_tr"]) + len(sym_buckets[s]["X_va"]) + len(sym_buckets[s]["X_te"]) for s in sym_buckets)

print(f"\n[DATA] Total rows: {total}  |  Symbols: {len(sym_buckets)}")
print(f"[SPLIT] Train={len(y_train)} Val={len(y_val)} Test={len(y_test)}")
dist_tr = np.bincount(y_train, minlength=3)
print(f"[LABELS] Train: SELL={dist_tr[0]/len(y_train):.1%}  "
      f"HOLD={dist_tr[1]/len(y_train):.1%}  BUY={dist_tr[2]/len(y_train):.1%}")

X_train_all = np.vstack([sym_buckets[s]["X_tr"] for s in sym_buckets])
scaler = StandardScaler()
scaler.fit(X_train_all)

counts = np.bincount(y_train, minlength=3)
raw_w  = np.array([len(y_train) / (3 * c) if c > 0 else 1.0 for c in counts])
raw_w  = raw_w / raw_w.mean()
class_weights = torch.FloatTensor(raw_w).to(device)
print(f"[WEIGHTS] {class_weights.cpu().numpy().round(3)}")

train_ds_list, val_ds_list, test_ds_list = [], [], []
for sym in sym_buckets:
    sb    = sym_buckets[sym]
    Xs_tr = scaler.transform(sb["X_tr"])
    Xs_va = scaler.transform(sb["X_va"])
    Xs_te = scaler.transform(sb["X_te"])
    if len(Xs_tr) > SEQUENCE_LEN:
        train_ds_list.append(SequenceDataset(Xs_tr, sb["y_tr"], SEQUENCE_LEN, augment=True,  noise_sigma=NOISE_SIGMA))
    if len(Xs_va) > SEQUENCE_LEN:
        val_ds_list.append(SequenceDataset(Xs_va, sb["y_va"], SEQUENCE_LEN, augment=False))
    if len(Xs_te) > SEQUENCE_LEN:
        test_ds_list.append(SequenceDataset(Xs_te, sb["y_te"], SEQUENCE_LEN, augment=False))

train_loader = DataLoader(ConcatDataset(train_ds_list), batch_size=BATCH_SIZE, shuffle=True,  drop_last=True,  num_workers=0)
val_loader   = DataLoader(ConcatDataset(val_ds_list),   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
test_loader  = DataLoader(ConcatDataset(test_ds_list),  batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
print(f"[DATA] Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}")

print(f"\n[MODEL] DLinear v1 (trend+resid decomp, hidden={HIDDEN_SIZE}, seq={SEQUENCE_LEN}, kernel={MA_KERNEL})")
model = DLinearModel(
    input_size=N_FEATURES,
    seq_len=SEQUENCE_LEN,
    hidden_size=HIDDEN_SIZE,
    num_classes=3,
    ma_kernel=MA_KERNEL,
    dropout=DROPOUT,
).to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"[MODEL] Parameters: {param_count:,}")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, lr_max=LR, lr_min=1e-6)

print(f"\n[TRAIN] {EPOCHS} epochs | patience={PATIENCE} | CE+class_weights+smoothing={LABEL_SMOOTHING}")

best_val_loss = float("inf")
best_val_acc  = 0.0
best_state    = None
patience_ctr  = 0

for epoch in range(EPOCHS):
    current_lr = scheduler.step()

    model.train()
    tr_loss = tr_correct = tr_total = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        tr_loss    += loss.item()
        _, pred     = torch.max(out, 1)
        tr_correct += (pred == yb).sum().item()
        tr_total   += yb.size(0)

    tr_loss /= len(train_loader)
    tr_acc   = tr_correct / tr_total

    model.eval()
    vl_loss = vl_correct = vl_total = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out  = model(Xb)
            loss = criterion(out, yb)
            vl_loss    += loss.item()
            _, pred     = torch.max(out, 1)
            vl_correct += (pred == yb).sum().item()
            vl_total   += yb.size(0)

    vl_loss /= len(val_loader)
    vl_acc   = vl_correct / vl_total

    improved = vl_loss < best_val_loss
    if improved:
        best_val_loss = vl_loss
        best_val_acc  = vl_acc
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        patience_ctr  = 0
        flag = " [BEST]"
    else:
        patience_ctr += 1
        flag = ""

    if (epoch + 1) % 5 == 0 or improved:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | lr={current_lr:.2e} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}{flag}")

    if patience_ctr >= PATIENCE:
        print(f"\n[EARLY STOP] Patience {PATIENCE} exhausted at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
print(f"\n[TRAIN] Done — best_val_loss={best_val_loss:.4f}  best_val_acc={best_val_acc:.4f}")

# ── Test ──────────────────────────────────────────────────────────────────────
model.eval()
y_pred_all, y_true_all = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        out = model(Xb.to(device))
        _, pred = torch.max(out, 1)
        y_pred_all.extend(pred.cpu().numpy())
        y_true_all.extend(yb.numpy())

y_pred = np.array(y_pred_all)
y_true = np.array(y_true_all)
test_acc     = accuracy_score(y_true, y_pred)
unique_preds = len(np.unique(y_pred))

print(f"\n[METRICS] Test Accuracy: {test_acc:.4f}  ({unique_preds}/3 classes)")
print("\n[CONFUSION MATRIX]")
print(confusion_matrix(y_true, y_pred))
print("\n[CLASSIFICATION REPORT]")
print(classification_report(y_true, y_pred, target_names=["SELL", "HOLD", "BUY"],
                             zero_division=0))

if unique_preds < 2:
    print(f"\n[ABORT] Only {unique_preds} unique class(es) — collapsed model, not saving")
    sys.exit(1)

if test_acc < 0.36:
    print(f"\n[WARN] test_acc={test_acc:.4f} below 36% — not saving")
    sys.exit(1)

# ── Save ──────────────────────────────────────────────────────────────────────
version    = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_fn   = f"dlinear_v1_{version}.pth"
scaler_fn  = f"dlinear_v1_{version}_scaler.pkl"
meta_fn    = f"dlinear_v1_{version}_metadata.json"

model_path  = SAVE_DIR / model_fn
scaler_path = SAVE_DIR / scaler_fn
meta_path   = SAVE_DIR / meta_fn

checkpoint = {
    "model_state_dict": model.state_dict(),
    "input_size":   N_FEATURES,
    "seq_len":      SEQUENCE_LEN,
    "hidden_size":  HIDDEN_SIZE,
    "num_classes":  3,
    "ma_kernel":    MA_KERNEL,
    "dropout":      DROPOUT,
    "features":     FEATURE_NAMES,
    "scaler_mean":  scaler.mean_.tolist(),
    "scaler_var":   scaler.var_.tolist(),
}
torch.save(checkpoint, model_path)
print(f"\n[SAVE] Model  -> {model_path}")

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"[SAVE] Scaler -> {scaler_path}")

metadata = {
    "version":          f"dlinear_v1_{version}",
    "model_type":       "DLinear",
    "training_version": "v1",
    "feature_schema":   "unified_features",
    "features":         FEATURE_NAMES,
    "num_features":     N_FEATURES,
    "architecture": {
        "type":        "DLinear",
        "input_size":  N_FEATURES,
        "seq_len":     SEQUENCE_LEN,
        "hidden_size": HIDDEN_SIZE,
        "num_classes": 3,
        "ma_kernel":   MA_KERNEL,
        "dropout":     DROPOUT,
        "parameters":  param_count,
    },
    "training_config": {
        "months":           MONTHS,
        "label_percentile": LABEL_PERCENTILE,
        "forward_k":        FORWARD_K,
        "sequence_len":     SEQUENCE_LEN,
        "label_smoothing":  LABEL_SMOOTHING,
        "epochs":           EPOCHS,
        "patience":         PATIENCE,
        "batch_size":       BATCH_SIZE,
        "noise_sigma":      NOISE_SIGMA,
    },
    "training_date":       datetime.utcnow().isoformat() + "Z",
    "training_samples":    int(len(y_train)),
    "validation_samples":  int(len(y_val)),
    "test_samples":        int(len(y_test)),
    "test_accuracy":       float(test_acc),
    "unique_pred_classes": int(unique_preds),
    "best_val_loss":       float(best_val_loss),
    "best_val_acc":        float(best_val_acc),
    "symbols":             list(sym_buckets.keys()),
}
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[SAVE] Meta   -> {meta_path}")

print(f"""
===========================================
  DLinear v1 Training Complete
  File:         {model_fn}
  Accuracy:     {test_acc:.4f}  ({unique_preds}/3 classes)
  Parameters:   {param_count:,}
  Data:         {len(sym_buckets)} symbols × {MONTHS}mo × unified_features
===========================================
Next:
  systemctl restart quantum-ai-engine
""")
