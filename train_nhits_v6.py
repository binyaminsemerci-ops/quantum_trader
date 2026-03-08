#!/usr/bin/env python3
"""
N-HiTS v6 Training - 49 features (aligned with XGBoost v6 / LGBM v6)
Fetches fresh Binance OHLCV data, calculates v6 features, trains NHiTS.
Saves in checkpoint format compatible with NHiTSAgent._load_pytorch_model().
"""
import os, sys, json, joblib, torch
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime, timezone
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ops.retrain.calculate_features_v6 import get_features_v6
FEATURES_V6 = get_features_v6()

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = Path("/opt/quantum/ai_engine/models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 120
LR = 1e-3
PATIENCE = 20

print("=" * 70)
print("N-HiTS v6 Training - 49 features")
print(f"Device: {DEVICE}")
print("=" * 70)

# ---- Fetch data ----
from ops.retrain.calculate_features_v6 import calculate_features_v6, create_labels
from backend.clients.binance_market_data_client import BinanceMarketDataClient

bc = BinanceMarketDataClient()
dfs = []
for symbol in SYMBOLS:
    print(f"  {symbol}: ", end="", flush=True)
    candles = bc.get_latest_candles(symbol, "1h", limit=1000)
    if candles is not None and len(candles) > 0:
        print(f"{len(candles)} candles")
        dfs.append(candles)
    else:
        print("FAILED")

if not dfs:
    print("ERROR: No data")
    sys.exit(1)

df = pd.concat(dfs, ignore_index=True)
df_original = df.copy()
print(f"[INFO] Total raw samples: {len(df)}")

df = calculate_features_v6(df)
available = [f for f in FEATURES_V6 if f in df.columns]
missing = [f for f in FEATURES_V6 if f not in df.columns]
if missing:
    print(f"WARNING: Missing features: {missing}")
FEATURES_V6 = available
print(f"[FEATURES] {len(FEATURES_V6)}/49 features available")

df, y = create_labels(df, df_original, threshold=0.015, lookahead=5)
X = df[FEATURES_V6].values
print(f"[LABELS] Samples: {len(X)}, SELL={sum(y==0)}, HOLD={sum(y==1)}, BUY={sum(y==2)}")

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
print(f"[SPLIT] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# ---- Model ----
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class NHiTSModel(nn.Module):
    def __init__(self, input_dim=49, hidden_dim=128, n_blocks=3, output_dim=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_d = input_dim
        for i in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
            ))
            in_d = hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)

input_dim = len(FEATURES_V6)
model = NHiTSModel(input_dim=input_dim, hidden_dim=128, n_blocks=3, output_dim=3).to(DEVICE)
print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Class weights for imbalanced data
class_counts = np.bincount(y_train.astype(int))
class_weights = torch.FloatTensor([max(class_counts) / c for c in class_counts]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

train_ds = SequenceDataset(X_train_s, y_train)
val_ds = SequenceDataset(X_val_s, y_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

best_acc, patience_counter = 0.0, 0
best_state = None

print(f"\n[TRAIN] NHiTS v6 (epochs={EPOCHS}, patience={PATIENCE})")
for epoch in range(EPOCHS):
    model.train()
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    preds, labels = [], []
    val_loss_total = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out = model(Xb)
            val_loss_total += criterion(out, yb).item()
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            labels.extend(yb.cpu().numpy())

    val_acc = accuracy_score(labels, preds)
    scheduler.step(val_loss_total)

    if (epoch + 1) % 20 == 0:
        unique = len(set(preds))
        print(f"  Epoch {epoch+1:3d} | Val Acc: {val_acc*100:.2f}% | Variety: {unique}/3")

    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

model.load_state_dict(best_state)
model.eval()

# Final eval
preds, labels = [], []
with torch.no_grad():
    for Xb, yb in val_loader:
        out = model(Xb.to(DEVICE))
        preds.extend(torch.argmax(out, 1).cpu().numpy())
        labels.extend(yb.numpy())

accuracy = accuracy_score(labels, preds)
unique_preds = len(set(preds))
print(f"\nAccuracy: {accuracy*100:.2f}%  Variety: {unique_preds}/3")
print(classification_report(labels, preds, target_names=["SELL", "HOLD", "BUY"]))

if unique_preds < 2:
    print("ERROR: Model is degenerate (< 2 unique predictions) - aborting save")
    sys.exit(1)

# Save in checkpoint format (NHiTSAgent._load_pytorch_model expects this)
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
model_path = SAVE_DIR / f"nhits_v6_{timestamp}.pth"
scaler_path = SAVE_DIR / f"nhits_v6_{timestamp}_scaler.pkl"
meta_path = SAVE_DIR / f"nhits_v6_{timestamp}_meta.json"

checkpoint = {
    'model_state_dict': best_state,
    'input_dim': input_dim,
    'hidden_dim': 128,
    'n_blocks': 3,
    'output_dim': 3,
    'features': FEATURES_V6,
    'version': f'v6_{timestamp}',
    'accuracy': float(accuracy),
}
torch.save(checkpoint, model_path)
joblib.dump(scaler, scaler_path)

with open(meta_path, 'w') as f:
    json.dump({
        "version": f"v6_{timestamp}",
        "features": FEATURES_V6,
        "num_features": len(FEATURES_V6),
        "accuracy": float(accuracy),
        "unique_predictions": unique_preds,
        "architecture": {"hidden_dim": 128, "n_blocks": 3, "output_dim": 3}
    }, f, indent=2)

print(f"\nModel  -> {model_path}")
print(f"Scaler -> {scaler_path}")
print(f"Meta   -> {meta_path}")
print("DONE")
