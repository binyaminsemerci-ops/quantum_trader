#!/usr/bin/env python3
"""
TFT (Temporal Fusion Transformer) v7 Training
Same architecture as unified_agents.py (hidden_size=128, num_heads=8, num_layers=3)
Improved training: AdamW, gradient clipping, 150 epochs, cosine LR, variety check.
ccxt exchange.timeout=15000 prevents hanging on delisted/slow symbols.
PYTHONUNBUFFERED=1 set at launch for real-time log output.
"""
Same architecture as unified_agents.py (hidden_size=128, num_heads=8, num_layers=3)
Improved training: AdamW, gradient clipping, 150 epochs, cosine LR, variety check.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# --
# Paths
# --
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.clients.binance_market_data_client import BinanceMarketDataClient
from ops.retrain.calculate_features_v6 import calculate_features_v6, get_features_v6, create_labels

FEATURES_V6 = get_features_v6()
BASE_DIR = Path(__file__).parent.parent.parent
SAVE_DIR = BASE_DIR / "ai_engine" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --
# TFT architecture (must match unified_agents.py EXACTLY)
# --

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(input_size + hidden_size, output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        residual = self.skip(x)
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        gate_input = torch.cat([x, torch.relu(self.fc1(x))], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        return self.layer_norm(gate * h + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.feature_transform = nn.Linear(input_size, hidden_size)
        self.gating = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=-1)
        )
        self.grn = GatedResidualNetwork(hidden_size, hidden_size)

    def forward(self, x):
        weights = self.gating(x)
        weighted = x * weights
        transformed = self.feature_transform(weighted)
        return self.grn(transformed)


class TemporalFusionBlock(nn.Module):
    def __init__(self, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.enrichment_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x, hidden=None):
        if x.dim() == 3:
            static = self.static_encoder(x[:, -1, :])
            out, hidden = self.decoder_lstm(x, hidden)
            gate_input = torch.cat([out[:, -1, :], static], dim=-1)
            gate = self.enrichment_gate(gate_input)
            return out[:, -1, :] * gate
        else:
            return self.static_encoder(x)


class TFTModel(nn.Module):
    """Must match unified_agents.py exactly: hidden_size=128, num_heads=8, num_layers=3"""
    def __init__(self, input_size=49, hidden_size=128, num_heads=8, num_layers=3,
                 num_classes=3, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.vsn = VariableSelectionNetwork(input_size, hidden_size)
        self.encoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.encoder_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.grn1 = GatedResidualNetwork(hidden_size, hidden_size * 2, hidden_size, dropout)
        self.grn2 = GatedResidualNetwork(hidden_size, hidden_size * 2, hidden_size, dropout)
        self.fusion = TemporalFusionBlock(hidden_size, num_layers=2, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_len, _ = x.shape
        vsn_out = []
        for t in range(seq_len):
            vsn_out.append(self.vsn(x[:, t, :]))
        x = torch.stack(vsn_out, dim=1)
        enc_out, _ = self.encoder_lstm(x)
        enc_out = self.encoder_projection(enc_out)
        attn_out, _ = self.attention(enc_out, enc_out, enc_out)
        attn_out = self.grn1(attn_out[:, -1, :])
        processed = self.grn2(attn_out)
        fused = self.fusion(enc_out)
        combined = self.layer_norm(processed + fused)
        return self.classifier(combined)


class TradingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --
# Main
# --
print("=" * 70)
print("TFT v7 Training  (hidden=128, heads=8, layers=3 | 49 features)")
print(f"Device: {device}")
print("=" * 70)

# -- Data --
print("\n[DATA] Fetching from Binance (1500 candles per symbol, 30s timeout)...")
symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "ADAUSDT", "XRPUSDT", "DOGEUSDT",
]
bc = BinanceMarketDataClient()
bc.exchange.timeout = 15000   # 15-second socket timeout per request
dfs = []

for sym in symbols:
    print(f"   {sym}: ", end="", flush=True)
    try:
        candles = bc.get_latest_candles(sym, "1h", limit=1500)
        if candles is not None and len(candles) > 50:
            print(f"{len(candles)} candles", flush=True)
            dfs.append(candles)
        else:
            print("SKIP (no data)", flush=True)
    except Exception as e:
        print(f"ERROR: {e} (skipped)", flush=True)

if not dfs:
    print("ERROR: No data fetched")
    sys.exit(1)

df = pd.concat(dfs, ignore_index=True)
df_orig = df.copy()
print(f"[DATA] Total raw rows: {len(df)}")

# -- Features --
print("\n[FEATURES] Calculating v6 features (49)...")
df = calculate_features_v6(df)
available = [f for f in FEATURES_V6 if f in df.columns]
missing = [f for f in FEATURES_V6 if f not in df.columns]
if missing:
    print(f"WARNING: missing features: {missing}")
    FEATURES_V6 = available
else:
    print(f"[FEATURES] All 49 OK, {len(df)} samples")

# -- Labels --
print("\n[LABELS] Creating labels (threshold=1.5%, lookahead=5)...")
df, y = create_labels(df, df_orig, threshold=0.015, lookahead=5)
X = df[FEATURES_V6].values
num_features = len(FEATURES_V6)

class_dist = np.bincount(y.astype(int))
print(f"[LABELS] Distribution: SELL={class_dist[0]}, HOLD={class_dist[1]}, BUY={class_dist[2]}")
print(f"[LABELS] Total samples: {len(X)}")

# -- Split --
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.176, random_state=42, stratify=y_tmp)
print(f"[SPLIT] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# -- Scale --
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# -- Class weights (squared inverse frequency for stronger minority upweighting) --
counts = np.bincount(y_train.astype(int))
# Inverse-frequency weights, then normalize so mean=1
raw_w = np.array([len(y_train) / (len(counts) * c) for c in counts])
raw_w = raw_w / raw_w.mean()
class_weights = torch.FloatTensor(raw_w).to(device)
print(f"[WEIGHTS] {class_weights.cpu().numpy().round(3)}")

# -- DataLoaders --
batch_size = 64   # smaller batch - more gradient steps per epoch
train_loader = DataLoader(TradingDataset(X_train_s, y_train), batch_size=batch_size, shuffle=True,  drop_last=True)
val_loader   = DataLoader(TradingDataset(X_val_s,   y_val),   batch_size=batch_size)
test_loader  = DataLoader(TradingDataset(X_test_s,  y_test),  batch_size=batch_size)

# -- Model --
print(f"\n[MODEL] TFT(input={num_features}, hidden=128, heads=8, layers=3)")
model = TFTModel(
    input_size=num_features,
    hidden_size=128,
    num_heads=8,
    num_layers=3,
    num_classes=3,
    dropout=0.1,
).to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"[MODEL] Parameters: {param_count:,}")

# -- Optimizer / scheduler --
# Lower LR + weight decay + cosine annealing - stable convergence for large models
EPOCHS   = 150
PATIENCE = 30
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# -- Training --
print(f"\n[TRAIN] {EPOCHS} epochs, patience={PATIENCE}, AdamW lr=3e-4, cosine LR")
best_val_loss = float('inf')
best_val_acc  = 0.0
best_state    = None
patience_ctr  = 0

for epoch in range(EPOCHS):
    # - Train -
    model.train()
    tr_loss = tr_correct = tr_total = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        # Gradient clipping: critical for LSTM/attention stacks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        tr_loss += loss.item()
        _, pred = torch.max(out, 1)
        tr_correct += (pred == yb).sum().item()
        tr_total += yb.size(0)

    scheduler.step()
    tr_loss /= len(train_loader)
    tr_acc   = tr_correct / tr_total

    # - Val -
    model.eval()
    vl_loss = vl_correct = vl_total = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out  = model(Xb)
            loss = criterion(out, yb)
            vl_loss += loss.item()
            _, pred = torch.max(out, 1)
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
        flag = " -"
    else:
        patience_ctr += 1
        flag = ""

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
          f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}{flag}")

    if patience_ctr >= PATIENCE:
        print(f"\n[EARLY STOP] No val improvement for {PATIENCE} epochs")
        break

# -- Restore best --
model.load_state_dict(best_state)
print(f"\n[TRAIN] Done - best_val_loss={best_val_loss:.4f}, best_val_acc={best_val_acc:.4f}")

# -- Evaluate --
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
test_acc = accuracy_score(y_true, y_pred)
unique_preds = len(np.unique(y_pred))

print(f"\n[METRICS] Test Accuracy: {test_acc:.4f}")
print(f"[METRICS] Unique predicted classes: {unique_preds}/3")
print("\n[CONFUSION MATRIX]")
print(confusion_matrix(y_true, y_pred))
print("\n[CLASSIFICATION REPORT]")
print(classification_report(y_true, y_pred, target_names=["SELL", "HOLD", "BUY"]))

# -- Variety guard --
if unique_preds < 2:
    print(f"\n- ABORT: Only {unique_preds} unique prediction class(es) - model collapsed to single class")
    print("   Not saving degenerate model. Try again or investigate data balance.")
    sys.exit(1)

if test_acc < 0.36:
    print(f"\n--  WARNING: test_acc={test_acc:.4f}, below 36% (near random)")
    print("   Saving model anyway, but consider retraining with more data.")

# -- Save --
version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_fn    = f"tft_v7_{version}.pth"
scaler_fn   = f"tft_v7_{version}_scaler.pkl"
meta_fn     = f"tft_v7_{version}_metadata.json"

model_path  = SAVE_DIR / model_fn
scaler_path = SAVE_DIR / scaler_fn
meta_path   = SAVE_DIR / meta_fn

checkpoint = {
    'model_state_dict': model.state_dict(),
    'input_size':  num_features,
    'hidden_size': 128,
    'num_heads':   8,
    'num_layers':  3,
    'num_classes': 3,
    'dropout':     0.1,
    'features':    FEATURES_V6,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_var':  scaler.var_.tolist(),
}
torch.save(checkpoint, model_path)
print(f"\n[SAVE] Model  - {model_path}")

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"[SAVE] Scaler - {scaler_path}")

metadata = {
    "version":       f"tft_v7_{version}",
    "model_type":    "TFT",
    "feature_schema": "FEATURES_V6",
    "features":      FEATURES_V6,
    "num_features":  num_features,
    "architecture": {
        "input_size":  num_features,
        "hidden_size": 128,
        "num_heads":   8,
        "num_layers":  3,
        "num_classes": 3,
        "dropout":     0.1,
    },
    "training_date":      datetime.utcnow().isoformat() + "Z",
    "training_samples":   int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples":       int(len(X_test)),
    "test_accuracy":      float(test_acc),
    "unique_pred_classes": int(unique_preds),
    "best_val_loss":      float(best_val_loss),
    "best_val_acc":       float(best_val_acc),
    "total_parameters":   int(param_count),
}
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"[SAVE] Meta   - {meta_path}")

print(f"""
===========================================
  TFT v7 Training Complete
  File:     {model_fn}
  Accuracy: {test_acc:.4f} ({unique_preds}/3 classes)
  Params:   {param_count:,}
===========================================
Next: systemctl restart quantum-ai-engine
""")
