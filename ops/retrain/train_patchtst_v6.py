#!/usr/bin/env python3
"""
PatchTST v6 — True sequence-based transformer training
=======================================================
CRITICAL BUG in v3: SimplePatchTST received (batch, 49) → unsqueeze(1) →
(batch, 1, d_model) → Transformer over seq_len=1 — attention over a single
token literally does nothing. The model degraded to a fancy linear layer.

v6 fixes:
  1. Real sequence input: 30 candles × 49 features per sample
  2. Patch processing: patch_size=5 → 6 patches → Transformer sees 6 tokens
     and can learn temporal momentum/reversal over ~150-min windows
  3. 4h lookahead + percentile labels (same as XGB/NHiTS/LGBM baseline)
  4. 12 symbols × 5000 candles each (~60k training rows)
  5. No StandardScaler at INFERENCE (bypass = same fix as XGB). The agent
     uses in-sequence z-score normalization (per-sequence, not global scaler).
  6. label_smoothing=0.10 — prevents overconfident HOLD

Architecture (SequencePatchTST):
  Input:  (batch, seq_len=30, n_features=49)
  Patch:  patch_size=5 → 6 patches → project to d_model=128
  Pos enc: learnable (6 positions)
  Encoder: 3 layers, 4 heads, dim_ff=256
  Head:   GlobalAvgPool → LayerNorm → Linear(128, 3)

Agent update needed:
  patchtst_agent_v3.py must add a 30-candle history buffer (like nhits_agent).
  The script prints a reminder at the end.
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
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_patchtst_v6")

# ─── Config ──────────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
    "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LTCUSDT", "TRXUSDT",
    "INJUSDT", "SUIUSDT",
]
INTERVAL         = "1h"
LIMIT_PER_SYMBOL = 5000            # candles per symbol
LABEL_PERCENTILE = 25              # top/bottom 25% → BUY/SELL, middle 50% HOLD
LOOKAHEAD        = 4               # 4h forward return
SEQ_LEN          = 30              # ← KEY: 30-candle input window
PATCH_SIZE       = 5               # 5-candle patch → 6 patches per window
LABEL_SMOOTHING  = 0.10
BATCH_SIZE       = 256
EPOCHS           = 80
LR               = 3e-4
WEIGHT_DECAY     = 1e-4
PATIENCE         = 15
D_MODEL          = 128
NUM_HEADS        = 4
NUM_LAYERS       = 3
DIM_FF           = 256
DROPOUT          = 0.15
TEST_FRAC        = 0.15

MODEL_DIR    = Path(__file__).parent.parent.parent / "ai_engine" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BINANCE_BASE = "https://api.binance.com/api/v3/klines"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_NAMES = [
    "returns", "log_returns", "price_range", "body_size", "upper_wick", "lower_wick",
    "is_doji", "is_hammer", "is_engulfing", "gap_up", "gap_down",
    "rsi", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "roc",
    "ema_9", "ema_9_dist", "ema_21", "ema_21_dist",
    "ema_50", "ema_50_dist", "ema_200", "ema_200_dist",
    "sma_20", "sma_50", "adx", "plus_di", "minus_di",
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
    "atr", "atr_pct", "volatility", "volume_sma", "volume_ratio",
    "obv", "obv_ema", "vpt",
    "momentum_5", "momentum_10", "momentum_20", "acceleration", "relative_spread",
]
N_FEATURES = 49
assert len(FEATURE_NAMES) == N_FEATURES
NUM_PATCHES = SEQ_LEN // PATCH_SIZE   # 30 // 5 = 6


# ─── Model ───────────────────────────────────────────────────────────────────

class SequencePatchTST(nn.Module):
    """
    True PatchTST: divides input sequence into non-overlapping patches,
    projects each patch, adds positional encoding, runs Transformer encoder,
    then classifies via global average pool.

    Input:  (batch, seq_len,   n_features)   e.g. (B, 30, 49)
    Output: (batch, num_classes)
    """

    def __init__(self, n_features=49, patch_size=5, d_model=128,
                 num_heads=4, num_layers=3, dim_ff=256,
                 dropout=0.15, num_classes=3):
        super().__init__()
        self.patch_size  = patch_size
        self.patch_dim   = patch_size * n_features   # 5 × 49 = 245
        self.num_patches = None                       # set dynamically

        # Patch embedding: flatten each patch → d_model
        self.patch_embed = nn.Sequential(
            nn.Linear(self.patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable positional encoding (up to 64 patches — flexible)
        self.pos_emb = nn.Embedding(64, d_model)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = num_heads,
            dim_feedforward= dim_ff,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # Init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        """
        B, T, F = x.shape
        assert T % self.patch_size == 0, f"seq_len {T} must be divisible by patch_size {self.patch_size}"
        num_patches = T // self.patch_size

        # Reshape into patches: (B, num_patches, patch_size * F)
        x = x.reshape(B, num_patches, self.patch_size * F)

        # Patch embedding
        x = self.patch_embed(x)   # (B, num_patches, d_model)

        # Positional encoding
        pos = torch.arange(num_patches, device=x.device)
        x   = x + self.pos_emb(pos).unsqueeze(0)   # broadcast over batch
        x   = self.dropout(x)

        # Transformer
        x = self.encoder(x)   # (B, num_patches, d_model)

        # Global average pooling over patches
        x = x.mean(dim=1)     # (B, d_model)

        return self.head(x)   # (B, num_classes)


# ─── Data ────────────────────────────────────────────────────────────────────

def _fetch_klines(symbol: str, limit: int) -> list:
    params = {"symbol": symbol, "interval": INTERVAL, "limit": limit}
    for attempt in range(5):
        try:
            r = requests.get(BINANCE_BASE, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"  Retry {attempt+1}/5 {symbol}: {e}")
            time.sleep(2 ** attempt)
    return []


def download_ohlcv(symbol: str, limit: int) -> pd.DataFrame:
    log.info(f"  Downloading {symbol} ({limit} candles)…")
    rows = _fetch_klines(symbol, limit)
    if not rows:
        raise RuntimeError(f"No data: {symbol}")
    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qv","nt","tbb","tbq","ig",
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
    return out.dropna(subset=FEATURE_NAMES)


def build_labels_and_features(df_feat: pd.DataFrame):
    """Per-symbol percentile labels + normalize features per-sequence (no global scaler)."""
    close   = df_feat["close"].values
    fwd_ret = (np.roll(close, -LOOKAHEAD) - close) / (close + 1e-10) * 100
    fwd_ret[-LOOKAHEAD:] = np.nan

    valid       = fwd_ret[~np.isnan(fwd_ret)]
    buy_thresh  = np.nanpercentile(valid, 100 - LABEL_PERCENTILE)
    sell_thresh = np.nanpercentile(valid, LABEL_PERCENTILE)

    y = np.where(fwd_ret >= buy_thresh, 2,
        np.where(fwd_ret <= sell_thresh, 0, 1)).astype(np.int64)
    y[np.isnan(fwd_ret)] = 1

    feat_arr = df_feat[FEATURE_NAMES].values.astype(np.float32)
    s, h, b  = int((y==0).sum()), int((y==1).sum()), int((y==2).sum())
    log.info(f"    Labels → SELL={s} HOLD={h} BUY={b}  "
             f"sell≤{sell_thresh:.2f}% buy≥{buy_thresh:.2f}%")
    return feat_arr, y


class SequenceDataset(Dataset):
    """
    Returns (seq [SEQ_LEN, 49], label) pairs.
    Normalization: per-sequence z-score (mean/std over the window).
    This avoids global scaler OOD issues — each window is self-normalized.
    """

    def __init__(self, feat: np.ndarray, labels: np.ndarray, seq_len: int):
        self.feat    = torch.from_numpy(feat)
        self.labels  = torch.from_numpy(labels).long()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.feat) - self.seq_len

    def __getitem__(self, i):
        seq   = self.feat[i : i + self.seq_len].clone()   # [seq_len, 49]
        label = self.labels[i + self.seq_len]

        # Per-sequence normalization: mean/std over SEQ_LEN timesteps per feature
        m = seq.mean(dim=0, keepdim=True)
        s = seq.std(dim=0, keepdim=True).clamp(min=1e-6)
        seq = (seq - m) / s

        # Clip extreme z-scores (ADX artifact protection)
        seq = seq.clamp(-5.0, 5.0)

        return seq, label


# ─── Train / eval ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optim.zero_grad()
        logits = model(seqs)
        loss   = criterion(logits, labels)
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
        logits        = model(seqs)
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
    log.info("  PatchTST v6  —  Real 30-candle sequences + 6 patches")
    log.info(f"  Device: {DEVICE}   SEQ={SEQ_LEN}   PATCH={PATCH_SIZE}   PATCHES={NUM_PATCHES}")
    log.info("=" * 60)

    # 1) Data
    all_feat, all_y = [], []
    for sym in SYMBOLS:
        try:
            raw  = download_ohlcv(sym, LIMIT_PER_SYMBOL)
            feat = build_features(raw)
            X, y = build_labels_and_features(feat)
            all_feat.append(X); all_y.append(y)
            log.info(f"  {sym}: {len(X):,} rows")
        except Exception as e:
            log.error(f"  ❌ {sym} skipped: {e}")

    if not all_feat:
        raise RuntimeError("No data")

    X_all = np.vstack(all_feat).astype(np.float32)
    y_all = np.concatenate(all_y).astype(np.int64)
    log.info(f"\n  Total: {len(X_all):,} rows  "
             f"SELL={int((y_all==0).sum())} HOLD={int((y_all==1).sum())} BUY={int((y_all==2).sum())}")

    # 2) Split
    split   = int(len(X_all) * (1 - TEST_FRAC))
    X_tr, X_va = X_all[:split], X_all[split:]
    y_tr, y_va = y_all[:split], y_all[split:]

    # 3) Datasets (NO global scaler — per-sequence normalization inside Dataset)
    ds_tr = SequenceDataset(X_tr, y_tr, SEQ_LEN)
    ds_va = SequenceDataset(X_va, y_va, SEQ_LEN)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    log.info(f"  Train seqs: {len(ds_tr):,}   Val seqs: {len(ds_va):,}")

    # 4) Model
    model = SequencePatchTST(
        n_features  = N_FEATURES,
        patch_size  = PATCH_SIZE,
        d_model     = D_MODEL,
        num_heads   = NUM_HEADS,
        num_layers  = NUM_LAYERS,
        dim_ff      = DIM_FF,
        dropout     = DROPOUT,
        num_classes = 3,
    ).to(DEVICE)
    log.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    counts  = np.bincount(y_tr, minlength=3).astype(np.float32)
    weights = counts.sum() / (3.0 * counts + 1e-6)
    wt      = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    log.info(f"  Class weights: SELL={wt[0]:.2f} HOLD={wt[1]:.2f} BUY={wt[2]:.2f}")

    criterion = nn.CrossEntropyLoss(weight=wt, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5) Train
    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    log.info(f"\n  Training {EPOCHS} epochs…")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, dl_tr, optimizer, criterion, DEVICE)
        va_loss, va_acc, va_preds, va_labels = eval_epoch(model, dl_va, criterion, DEVICE)
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
            rep  = classification_report(va_labels, va_preds, target_names=["SELL","HOLD","BUY"],
                                          output_dict=True, zero_division=0)
            s_rc = rep.get("SELL",{}).get("recall",0)
            h_rc = rep.get("HOLD",{}).get("recall",0)
            b_rc = rep.get("BUY", {}).get("recall",0)
            log.info(
                f"  Ep {epoch:3d}  tr={tr_acc*100:.1f}%  va={va_acc*100:.1f}%  "
                f"recall S={s_rc:.2f} H={h_rc:.2f} B={b_rc:.2f}{marker}"
            )

        if no_improve >= PATIENCE:
            log.info(f"  Early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # 6) Sanity check — random per-sequence-normalized input
    model.eval()
    np.random.seed(42); cnts = [0,0,0]
    with torch.no_grad():
        for _ in range(1000):
            raw = torch.randn(1, SEQ_LEN, N_FEATURES)
            m   = raw.mean(dim=1, keepdim=True)
            s   = raw.std(dim=1,  keepdim=True).clamp(min=1e-6)
            inp = ((raw - m) / s).clamp(-5, 5).to(DEVICE)
            p   = torch.softmax(model(inp), dim=1)[0]
            cnts[p.argmax().item()] += 1
    log.info(f"\n  Sanity random x1000: SELL={cnts[0]} HOLD={cnts[1]} BUY={cnts[2]}")

    # 7) Save
    ts          = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path  = MODEL_DIR / f"patchtst_v6_{ts}.pth"
    meta_path   = MODEL_DIR / f"patchtst_v6_{ts}_metadata.json"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "n_features": N_FEATURES, "patch_size": PATCH_SIZE,
            "d_model": D_MODEL, "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS, "dim_ff": DIM_FF,
            "dropout": DROPOUT, "num_classes": 3,
            "seq_len": SEQ_LEN,
        },
        "features": FEATURE_NAMES,
        "num_features": N_FEATURES,
        "seq_len": SEQ_LEN,
        "patch_size": PATCH_SIZE,
        "lookahead": LOOKAHEAD,
        "label_percentile": LABEL_PERCENTILE,
        "test_accuracy": round(best_val_acc * 100, 2),
        "trained_at": ts,
        "normalization": "per_sequence_zscore",  # NO global scaler
        "sanity_random": cnts,
    }
    torch.save(checkpoint, str(model_path))

    with open(meta_path, "w") as f:
        json.dump({
            "version": f"patchtst_v6_{ts}", "trained_at": ts,
            "symbols": SYMBOLS, "limit_per_symbol": LIMIT_PER_SYMBOL,
            "num_features": N_FEATURES, "feature_schema": "FEATURES_V6",
            "seq_len": SEQ_LEN, "patch_size": PATCH_SIZE, "num_patches": NUM_PATCHES,
            "lookahead": LOOKAHEAD, "label_percentile": LABEL_PERCENTILE,
            "label_smoothing": LABEL_SMOOTHING,
            "val_accuracy": round(best_val_acc * 100, 2),
            "sanity_random_1000": {"SELL": cnts[0], "HOLD": cnts[1], "BUY": cnts[2]},
            "normalization": "per_sequence_zscore",
            "architecture": {
                "d_model": D_MODEL, "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS, "dim_ff": DIM_FF,
                "patch_size": PATCH_SIZE, "num_patches": NUM_PATCHES,
            },
            "note": "v6: true 30-candle sequences, 6 patches of 5, per-seq zscore — no OOD scaler issue",
        }, f, indent=2)

    log.info(f"\n  ✅  Saved: {model_path.name}")
    log.info(f"  Val accuracy: {best_val_acc*100:.2f}%")
    log.info(f"  Sanity: SELL={cnts[0]} HOLD={cnts[1]} BUY={cnts[2]}")
    log.info("\n  ⚠️  NEXT: Update patchtst_agent_v3.py to:")
    log.info("       1. Add 30-candle history_buffer (like nhits_agent)")
    log.info("       2. Load SequencePatchTST architecture (not SimplePatchTST)")
    log.info("       3. Apply per-sequence z-score normalization at inference")
    log.info("       4. Use: from ops.retrain.train_patchtst_v6 import SequencePatchTST")
    log.info("      (or copy SequencePatchTST to ai_engine/patchtst_v6.py)")


if __name__ == "__main__":
    main()
