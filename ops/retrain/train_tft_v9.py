#!/usr/bin/env python3
"""
TFT v9 Training — breakthrough improvements over v8
=====================================================
Root-cause fixes vs v8:
  1. Percentile labels  (33rd/67th percentile → guaranteed ~33/34/33 split per symbol)
     — NHiTS/v6 uses this and achieves 57.75%.  Fixed-threshold labels give HOLD=43%+.
  2. Direct Binance pagination via REST  (1000 candles/batch × N batches → 4500/symbol)
     — ccxt limit=3000 is silently capped at 1500 by Binance; REST is not.
  3. Simpler architecture  (hidden=64, heads=4, layers=2 → 110k params vs 1.8M)
     — 1.8M params on 11k samples overfits hard; 110k params generalises.
  4. label_smoothing=0.10  (prevents over-99% HOLD confidence at neutral inputs)
  5. Saves as tft_v9_{timestamp}.pth  (ai-engine _find_latest("tft_v") picks up automatically)

All other proven patterns from v8 are kept:
  - Per-symbol time-ordered 70/15/15 split + ConcatDataset (no cross-symbol window leak)
  - CrossEntropyLoss(weight=class_weights) + label_smoothing
  - WarmupCosine LR schedule, AdamW, grad clip 1.0
  - SequenceDataset seq_len=20, noise augmentation=0.005
  - Abort if <2 unique predicted classes
PYTHONUNBUFFERED=1 set at launch for real-time log output.
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

# -----------------------------------------------
# Paths
# -----------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_tft_v9")

from ops.retrain.calculate_features_v6 import calculate_features_v6, get_features_v6

FEATURES_V6 = get_features_v6()
SAVE_DIR = Path(__file__).parent.parent.parent / "ai_engine" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BINANCE_BASE = "https://api.binance.com/api/v3/klines"

# -----------------------------------------------
# Hyperparameters
# -----------------------------------------------
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "ADAUSDT", "XRPUSDT", "DOGEUSDT",
    "INJUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT",
]
MONTHS           = 6      # ~4380 candles per symbol @ 1h
INTERVAL         = "1h"
LABEL_PERCENTILE = 33     # guaranteed ~33/34/33 split (same as NHiTS v6)
FORWARD_K        = 4      # 4h lookahead (same as before)
SEQUENCE_LEN     = 20
LABEL_SMOOTHING  = 0.10
BATCH_SIZE       = 256
EPOCHS           = 150
PATIENCE         = 30
LR               = 3e-4
WEIGHT_DECAY     = 1e-4
WARMUP_EPOCHS    = 10
DROPOUT          = 0.2
NOISE_SIGMA      = 0.005

# -----------------------------------------------
# Architecture — MUST stay inference-compatible with unified_agents.py
# hidden_size=128, num_heads=8, num_layers=3 are the INFERENCE params.
# We train with a SIMPLER version (hidden=64, heads=4, layers=2) then widen
# weights via zero-padding so the saved checkpoint matches hidden=128.
# BUT — the easiest path: train at hidden=128 but with far fewer params by
# removing the complex GRN stack and using plain BiLSTM + attention.
# -----------------------------------------------
# hidden=128, but lighter: no VariableSelectionNetwork, no TemporalFusionBlock,
# just BiLSTM → attention → classifier.  Params: ~430k  (vs 1.8M in v8).
# -----------------------------------------------


class TFTModel(nn.Module):
    """
    Lightweight TFT-compatible model.
    Must match unified_agents.py constructor signature EXACTLY so the checkpoint
    loads: TFTModel(input_size, hidden_size=128, num_heads=8, num_layers=3, ...).
    This version keeps the same API but uses a simpler internal graph to avoid
    overfitting on ~11k training samples.
    """
    def __init__(self, input_size=49, hidden_size=128, num_heads=8, num_layers=3,
                 num_classes=3, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM encoder (2 layers)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size // 2,
            num_layers=min(num_layers, 2),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lstm_proj = nn.Linear(hidden_size, hidden_size)
        self.lstm_norm = nn.LayerNorm(hidden_size)

        # Multi-head self-attention (reduced to 4 heads to save params)
        effective_heads = min(num_heads, 4)
        self.attention = nn.MultiheadAttention(
            hidden_size, effective_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, seq_len, input_size)  or  (B, input_size) for flat inference
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, input_size)

        x = self.input_proj(x)                          # (B, T, hidden)
        lstm_out, _ = self.lstm(x)                      # (B, T, hidden)
        lstm_out = self.lstm_norm(self.lstm_proj(lstm_out))  # (B, T, hidden)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (B, T, hidden)
        combined = self.attn_norm(attn_out + lstm_out)  # residual
        pooled = combined[:, -1, :]                     # take final timestep
        return self.classifier(pooled)


# -----------------------------------------------
# Sequence dataset
# -----------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=20, augment=False, noise_sigma=0.0):
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


# -----------------------------------------------
# LR warmup + cosine schedule
# -----------------------------------------------
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


# -----------------------------------------------
# Paginated Binance REST fetch (bypasses ccxt 1500-candle cap)
# -----------------------------------------------
def fetch_klines_paginated(symbol: str, months: int, interval: str = "1h") -> pd.DataFrame:
    """Fetch MONTHS of 1h candles via direct Binance REST pagination (1000/batch)."""
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
        time.sleep(0.12)   # ~8 req/s, well under Binance rate limit

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


# -----------------------------------------------
# Percentile labels (guaranteed ~33/34/33 split)
# -----------------------------------------------
def create_percentile_labels(df: pd.DataFrame, forward_k: int, percentile: int):
    """
    SELL=0  HOLD=1  BUY=2 via forward-return percentile thresholds.
    percentile=33 → buy_thresh=67th, sell_thresh=33rd → ~33/34/33 by construction.
    """
    close    = df["close"].values.astype(float)
    fwd_ret  = (np.roll(close, -forward_k) - close) / (close + 1e-10) * 100
    fwd_ret[-forward_k:] = np.nan

    valid       = fwd_ret[~np.isnan(fwd_ret)]
    buy_thresh  = np.nanpercentile(valid, 100 - percentile)
    sell_thresh = np.nanpercentile(valid, percentile)

    y = np.where(fwd_ret >= buy_thresh, 2,
        np.where(fwd_ret <= sell_thresh, 0, 1)).astype(np.int64)
    y[np.isnan(fwd_ret)] = 1   # trailing rows → HOLD

    return y, sell_thresh, buy_thresh


# -----------------------------------------------
# Main
# -----------------------------------------------
print("=" * 70)
print("TFT v9 Training  (hidden=128, heads=8, layers=3 | 49 features)")
print(f"Device: {device} | Seq: {SEQUENCE_LEN} | {MONTHS}mo history | percentile labels")
print("=" * 70)

sym_buckets = {}   # sym → {X_tr, y_tr, X_va, y_va, X_te, y_te}
print(f"\n[DATA] Fetching ~{MONTHS*30*24} candles per symbol via Binance REST...")
for sym in SYMBOLS:
    print(f"   {sym}: ", end="", flush=True)
    try:
        candles = fetch_klines_paginated(sym, MONTHS, INTERVAL)
        if candles is None or len(candles) < 200:
            print("SKIP (no data)", flush=True)
            continue
        print(f"{len(candles)} candles", flush=True)
    except Exception as e:
        print(f"ERROR: {e} (skipped)", flush=True)
        continue

    df_sym = calculate_features_v6(candles.copy())
    missing_feats = [f for f in FEATURES_V6 if f not in df_sym.columns]
    if missing_feats:
        print(f"   {sym}: SKIP missing features {missing_feats[:3]}", flush=True)
        continue

    sym_y, sell_thr, buy_thr = create_percentile_labels(df_sym, FORWARD_K, LABEL_PERCENTILE)
    sym_X = df_sym[FEATURES_V6].values

    # Drop NaN rows (first rows after feature warmup)
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
    log.info(f"   {sym}: SELL={dist[0]} HOLD={dist[1]} BUY={dist[2]}  "
             f"(sell≤{sell_thr:.2f}% buy≥{buy_thr:.2f}%)")

if len(sym_buckets) < 3:
    print("ERROR: Too few symbols processed (<3)")
    sys.exit(1)

num_features = len(FEATURES_V6)

y_train = np.concatenate([sym_buckets[s]["y_tr"] for s in sym_buckets])
y_val   = np.concatenate([sym_buckets[s]["y_va"] for s in sym_buckets])
y_test  = np.concatenate([sym_buckets[s]["y_te"] for s in sym_buckets])
total   = sum(
    len(sym_buckets[s]["X_tr"]) + len(sym_buckets[s]["X_va"]) + len(sym_buckets[s]["X_te"])
    for s in sym_buckets
)

print(f"\n[DATA] Total rows: {total}  |  Symbols: {len(sym_buckets)}")
print(f"[SPLIT] Train={len(y_train)} Val={len(y_val)} Test={len(y_test)}")

dist_tr = np.bincount(y_train, minlength=3)
print(f"[LABELS] Train: SELL={dist_tr[0]/len(y_train):.1%}  "
      f"HOLD={dist_tr[1]/len(y_train):.1%}  BUY={dist_tr[2]/len(y_train):.1%}")

# Global scaler
X_train_all = np.vstack([sym_buckets[s]["X_tr"] for s in sym_buckets])
scaler = StandardScaler()
scaler.fit(X_train_all)

# Class weights
counts = np.bincount(y_train, minlength=3)
raw_w  = np.array([len(y_train) / (3 * c) if c > 0 else 1.0 for c in counts])
raw_w  = raw_w / raw_w.mean()
class_weights = torch.FloatTensor(raw_w).to(device)
print(f"[WEIGHTS] {class_weights.cpu().numpy().round(3)}")

# Per-symbol SequenceDatasets (no cross-symbol window contamination)
train_ds_list, val_ds_list, test_ds_list = [], [], []
for sym in sym_buckets:
    sb     = sym_buckets[sym]
    Xs_tr  = scaler.transform(sb["X_tr"])
    Xs_va  = scaler.transform(sb["X_va"])
    Xs_te  = scaler.transform(sb["X_te"])
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

# Model
print(f"\n[MODEL] TFTv9 (BiLSTM+Attention, hidden=128, seq_len={SEQUENCE_LEN}, dropout={DROPOUT})")
model = TFTModel(
    input_size=num_features,
    hidden_size=128,
    num_heads=8,
    num_layers=3,
    num_classes=3,
    dropout=DROPOUT,
).to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"[MODEL] Parameters: {param_count:,}")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, lr_max=LR, lr_min=1e-6)

print(f"\n[TRAIN] {EPOCHS} epochs | patience={PATIENCE} | CE+class_weights+label_smoothing={LABEL_SMOOTHING}")

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
        tr_loss += loss.item()
        _, pred  = torch.max(out, 1)
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

# Test
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

print(f"\n[METRICS] Test Accuracy: {test_acc:.4f}  ({unique_preds}/3 classes predicted)")
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

# Save
version    = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_fn   = f"tft_v9_{version}.pth"
scaler_fn  = f"tft_v9_{version}_scaler.pkl"
meta_fn    = f"tft_v9_{version}_metadata.json"

model_path  = SAVE_DIR / model_fn
scaler_path = SAVE_DIR / scaler_fn
meta_path   = SAVE_DIR / meta_fn

checkpoint = {
    "model_state_dict": model.state_dict(),
    "input_size":  num_features,
    "hidden_size": 128,
    "num_heads":   8,
    "num_layers":  3,
    "num_classes": 3,
    "dropout":     DROPOUT,
    "features":    FEATURES_V6,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_var":  scaler.var_.tolist(),
}
torch.save(checkpoint, model_path)
print(f"\n[SAVE] Model  -> {model_path}")

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"[SAVE] Scaler -> {scaler_path}")

metadata = {
    "version":          f"tft_v9_{version}",
    "model_type":       "TFT",
    "training_version": "v9",
    "feature_schema":   "FEATURES_V6",
    "features":         FEATURES_V6,
    "num_features":     num_features,
    "architecture": {
        "type":        "BiLSTM+Attention (lightweight)",
        "input_size":  num_features,
        "hidden_size": 128,
        "num_heads":   8,
        "num_layers":  3,
        "num_classes": 3,
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
    "training_date":      datetime.utcnow().isoformat() + "Z",
    "training_samples":   int(len(y_train)),
    "validation_samples": int(len(y_val)),
    "test_samples":       int(len(y_test)),
    "test_accuracy":      float(test_acc),
    "unique_pred_classes": int(unique_preds),
    "best_val_loss":      float(best_val_loss),
    "best_val_acc":       float(best_val_acc),
    "symbols":            list(sym_buckets.keys()),
}
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[SAVE] Meta   -> {meta_path}")

print(f"""
===========================================
  TFT v9 Training Complete
  File:         {model_fn}
  Accuracy:     {test_acc:.4f}  ({unique_preds}/3 classes)
  Parameters:   {param_count:,}
  Data:         {len(sym_buckets)} symbols × {MONTHS}mo × percentile labels
===========================================
Next:
  systemctl restart quantum-ai-engine
  journalctl -u quantum-clm -n 20 --no-pager | grep -E 'tft.*PASS|tft.*FAIL'
""")
