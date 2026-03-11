#!/usr/bin/env python3
"""
DLinear v1 Training Script
===========================
Trains DLinear classifier using 49-feature OHLCV schema (FEATURES_V6).

Data:   Binance USDM futures public API (no auth required)
Output: ai_engine/models/dlinear_v{timestamp}_v1.pth
        ai_engine/models/dlinear_v{timestamp}_v1_scaler.pkl
        ai_engine/models/dlinear_v{timestamp}_v1_metadata.json

Usage (on VPS):
    cd /home/qt/quantum_trader
    /home/qt/quantum_trader_venv/bin/python /tmp/train_dlinear_v1.py
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── PyTorch ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    logger.error("PyTorch not available. Run: pip install torch")
    sys.exit(1)

try:
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    logger.error("scikit-learn / joblib not available.")
    sys.exit(1)

# ── DLinear architecture (must match unified_agents.py exactly) ───────────────
class DLinearMovingAvg(nn.Module):
    """Causal moving average — no future leakage."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (batch, seq_len, features)
        pad   = x[:, :self.kernel_size - 1, :]
        x_pad = torch.cat([pad, x], dim=1)
        return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)


class DLinearModel(nn.Module):
    def __init__(self, input_size=49, seq_len=60, hidden_size=256,
                 num_classes=3, ma_kernel=25, dropout=0.2):
        super().__init__()
        self.input_size  = input_size
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.ma_kernel   = ma_kernel
        flat_dim = seq_len * input_size
        self.moving_avg  = DLinearMovingAvg(kernel_size=ma_kernel)
        self.trend_proj  = nn.Linear(flat_dim, hidden_size)
        self.resid_proj  = nn.Linear(flat_dim, hidden_size)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            # Single feature vector → expand to seq (matches predict() in unified_agents.py)
            x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        trend    = self.moving_avg(x)
        resid    = x - trend
        t_proj   = self.trend_proj(trend.reshape(x.size(0), -1))
        r_proj   = self.resid_proj(resid.reshape(x.size(0), -1))
        combined = torch.cat([t_proj, r_proj], dim=-1)
        return self.classifier(combined)


# ── Feature list (must match FEATURES_V6 in common_features.py) ───────────────
FEATURES_V6 = [
    "returns", "log_returns", "price_range", "body_size", "upper_wick", "lower_wick",
    "is_doji", "is_hammer", "is_engulfing", "gap_up", "gap_down",
    "rsi", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "roc",
    "ema_9", "ema_9_dist", "ema_21", "ema_21_dist", "ema_50", "ema_50_dist",
    "ema_200", "ema_200_dist",
    "sma_20", "sma_50", "adx", "plus_di", "minus_di",
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
    "atr", "atr_pct", "volatility",
    "volume_sma", "volume_ratio", "obv", "obv_ema", "vpt",
    "momentum_5", "momentum_10", "momentum_20", "acceleration", "relative_spread",
]
assert len(FEATURES_V6) == 49, f"Expected 49 features, got {len(FEATURES_V6)}"


# ── Feature engineering ───────────────────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Returns
    d["returns"]     = d["close"].pct_change()
    d["log_returns"] = np.log(d["close"] / d["close"].shift(1))

    # Candle structure
    d["price_range"] = (d["high"] - d["low"]) / d["close"]
    d["body_size"]   = abs(d["close"] - d["open"]) / d["close"]
    hi = d[["open", "close"]].max(axis=1)
    lo = d[["open", "close"]].min(axis=1)
    d["upper_wick"]  = (d["high"] - hi) / d["close"]
    d["lower_wick"]  = (lo - d["low"])  / d["close"]

    # Patterns
    d["is_doji"]      = (d["body_size"] < 0.001).astype(float)
    d["is_hammer"]    = ((d["lower_wick"] > 2 * d["body_size"]) &
                         (d["upper_wick"] < d["body_size"])).astype(float)
    d["is_engulfing"] = (abs(d["close"] - d["open"]) >
                          abs(d["close"].shift(1) - d["open"].shift(1))).astype(float)
    d["gap_up"]   = ((d["open"] > d["close"].shift(1)) *
                     (d["open"] - d["close"].shift(1)) / d["close"]).clip(lower=0)
    d["gap_down"]  = ((d["open"] < d["close"].shift(1)) *
                      (d["close"].shift(1) - d["open"]) / d["close"]).clip(lower=0)

    # RSI
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"]        = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]

    # Stochastic
    low14  = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["stoch_k"] = 100 * (d["close"] - low14) / (high14 - low14 + 1e-10)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # ROC
    d["roc"] = d["close"].pct_change(10) * 100

    # EMAs
    for p in [9, 21, 50, 200]:
        e = d["close"].ewm(span=p, adjust=False).mean()
        d[f"ema_{p}"]      = e
        d[f"ema_{p}_dist"] = (d["close"] - e) / (e + 1e-10)

    # SMAs
    d["sma_20"] = d["close"].rolling(20).mean()
    d["sma_50"] = d["close"].rolling(50).mean()

    # ATR / ADX
    tr  = pd.concat([
        d["high"] - d["low"],
        abs(d["high"] - d["close"].shift(1)),
        abs(d["low"]  - d["close"].shift(1)),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up  = d["high"] - d["high"].shift(1)
    dn  = d["low"].shift(1) - d["low"]
    pdm = up.where((up > dn) & (up > 0), 0).rolling(14).mean()
    mdm = dn.where((dn > up) & (dn > 0), 0).rolling(14).mean()
    d["plus_di"]  = 100 * pdm / (atr + 1e-10)
    d["minus_di"] = 100 * mdm / (atr + 1e-10)
    dx = 100 * abs(d["plus_di"] - d["minus_di"]) / (d["plus_di"] + d["minus_di"] + 1e-10)
    d["adx"]      = dx.rolling(14).mean()
    d["atr"]      = atr
    d["atr_pct"]  = atr / d["close"]

    # Bollinger Bands
    d["bb_middle"]   = d["close"].rolling(20).mean()
    bb_std           = d["close"].rolling(20).std()
    d["bb_upper"]    = d["bb_middle"] + 2 * bb_std
    d["bb_lower"]    = d["bb_middle"] - 2 * bb_std
    d["bb_width"]    = (d["bb_upper"] - d["bb_lower"]) / (d["bb_middle"] + 1e-10)
    d["bb_position"] = (d["close"] - d["bb_lower"]) / (
        d["bb_upper"] - d["bb_lower"] + 1e-10)

    # Volatility
    d["volatility"] = d["close"].pct_change().rolling(20).std()

    # Volume
    d["volume_sma"]   = d["volume"].rolling(20).mean()
    d["volume_ratio"] = d["volume"] / (d["volume_sma"] + 1e-10)
    d["obv"]          = (np.sign(d["close"].diff()) * d["volume"]).cumsum()
    d["obv_ema"]      = d["obv"].ewm(span=20, adjust=False).mean()
    d["vpt"]          = (d["volume"] * d["close"].pct_change()).cumsum()

    # Momentum
    for p in [5, 10, 20]:
        d[f"momentum_{p}"] = d["close"].pct_change(p)
    d["acceleration"]    = d["momentum_5"] - d["momentum_5"].shift(1)
    d["relative_spread"] = (d["high"] - d["low"]) / (d["close"] + 1e-10)

    return d


# ── Binance USDM futures klines (no auth) ────────────────────────────────────
def fetch_klines(symbol: str, interval: str = "5m", limit: int = 1000) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()


# ── Training ─────────────────────────────────────────────────────────────────
def main():
    # ── Hyperparameters ──
    SYMBOLS   = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"]
    INTERVAL  = "5m"
    LIMIT     = 1000       # candles per symbol
    THRESHOLD = 0.003      # 0.3% threshold for BUY/SELL label
    HORIZON   = 5          # candles ahead for label
    SEQ_LEN   = 60
    INPUT_SIZE = 49
    HIDDEN    = 256
    MA_KERNEL = 25
    DROPOUT   = 0.2
    EPOCHS    = 40
    BATCH     = 128
    LR        = 1e-3
    MODEL_DIR = Path("/home/qt/quantum_trader/ai_engine/models")

    logger.info("=" * 60)
    logger.info("DLinear v1 Training")
    logger.info("=" * 60)

    # ── Collect data ──
    all_X, all_y = [], []
    for sym in SYMBOLS:
        try:
            logger.info(f"Fetching {sym} {INTERVAL} ({LIMIT} candles)...")
            df   = fetch_klines(sym, INTERVAL, LIMIT)
            df   = compute_features(df)
            # Label: future return horizon candles out
            df["_future_ret"] = df["close"].shift(-HORIZON) / df["close"] - 1
            df["_label"]      = 1  # HOLD
            df.loc[df["_future_ret"] >  THRESHOLD, "_label"] = 2  # BUY
            df.loc[df["_future_ret"] < -THRESHOLD, "_label"] = 0  # SELL
            df = df.dropna(subset=FEATURES_V6 + ["_label"])
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES_V6)

            X = df[FEATURES_V6].values.astype(np.float32)
            y = df["_label"].values.astype(np.int64)
            all_X.append(X)
            all_y.append(y)
            buy  = int((y == 2).sum())
            sell = int((y == 0).sum())
            hold = int((y == 1).sum())
            logger.info(f"  {sym}: {len(X)} samples  BUY={buy}  SELL={sell}  HOLD={hold}")
            time.sleep(0.3)  # rate-limit
        except Exception as e:
            logger.warning(f"  {sym}: FAILED — {e}")

    if not all_X:
        logger.error("No data fetched from Binance. Exiting.")
        sys.exit(1)

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    logger.info(f"\nTotal dataset: {len(X)} samples, {INPUT_SIZE} features")
    logger.info(f"Labels: BUY={int((y==2).sum())}  SELL={int((y==0).sum())}  HOLD={int((y==1).sum())}")

    # ── Scale ──
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # ── Split ──
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Tensors — expand to (N, seq_len, 49) matching predict() in DLinearAgent ──
    def to_seq(arr):
        t = torch.FloatTensor(arr)                      # (N, 49)
        return t.unsqueeze(1).expand(-1, SEQ_LEN, -1).contiguous()  # (N, 60, 49)

    train_dl = DataLoader(
        TensorDataset(to_seq(X_tr), torch.LongTensor(y_tr)),
        batch_size=BATCH, shuffle=True
    )
    val_dl = DataLoader(
        TensorDataset(to_seq(X_val), torch.LongTensor(y_val)),
        batch_size=BATCH
    )

    # ── Model ──
    model = DLinearModel(
        input_size=INPUT_SIZE, seq_len=SEQ_LEN, hidden_size=HIDDEN,
        num_classes=3, ma_kernel=MA_KERNEL, dropout=DROPOUT
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    # ── Class weights (handle HOLD dominance) ──
    counts       = np.bincount(y_tr.astype(np.int64), minlength=3).astype(float)
    weights      = 1.0 / (counts + 1.0)
    weights     /= weights.sum()
    weights     *= 3.0
    class_w      = torch.FloatTensor(weights)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    best_val_acc = 0.0
    best_state   = None

    logger.info("")
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        t_loss = t_correct = t_total = 0
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * len(yb)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total   += len(yb)

        # Val
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for Xb, yb in val_dl:
                logits     = model(Xb)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total   += len(yb)

        scheduler.step()
        t_acc = t_correct / t_total
        v_acc = v_correct / v_total

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{EPOCHS}  "
                f"train_loss={t_loss/t_total:.4f}  "
                f"train_acc={t_acc:.3f}  "
                f"val_acc={v_acc:.3f}  "
                f"best={best_val_acc:.3f}"
            )

    logger.info(f"\nBest val_acc: {best_val_acc:.3f}")

    # ── Save ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ts        = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = f"dlinear_v{ts}_v1"

    # Checkpoint (.pth) — includes architecture params for _load_pytorch_model
    ckpt_path = MODEL_DIR / f"{base_name}.pth"
    torch.save({
        "model_state_dict": best_state,
        "input_size":       INPUT_SIZE,
        "seq_len":          SEQ_LEN,
        "hidden_size":      HIDDEN,
        "num_classes":      3,
        "ma_kernel":        MA_KERNEL,
        "dropout":          DROPOUT,
        "version":          "v1",
        "val_accuracy":     best_val_acc,
        "trained_at":       ts,
    }, str(ckpt_path))
    logger.info(f"[SAVED] Model:   {ckpt_path}")

    # Scaler
    scaler_path = MODEL_DIR / f"{base_name}_scaler.pkl"
    joblib.dump(scaler, str(scaler_path))
    logger.info(f"[SAVED] Scaler:  {scaler_path}")

    # Metadata
    meta_path = MODEL_DIR / f"{base_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "version":      "v1",
            "trained_at":   ts,
            "num_features": INPUT_SIZE,
            "features":     FEATURES_V6,
            "num_classes":  3,
            "classes":      ["SELL", "HOLD", "BUY"],
            "val_accuracy": best_val_acc,
            "symbols":      SYMBOLS,
            "interval":     INTERVAL,
            "architecture": {
                "input_size":  INPUT_SIZE,
                "seq_len":     SEQ_LEN,
                "hidden_size": HIDDEN,
                "ma_kernel":   MA_KERNEL,
                "dropout":     DROPOUT,
            },
        }, f, indent=2)
    logger.info(f"[SAVED] Metadata: {meta_path}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ DLinear v1 training COMPLETE")
    logger.info(f"   Prefix match: dlinear_v  →  {base_name}.pth")
    logger.info(f"   Val accuracy: {best_val_acc:.1%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
