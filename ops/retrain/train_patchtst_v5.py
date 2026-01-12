#!/usr/bin/env python3
"""
PatchTST v5 Training – aligned with XGBoost v5 (18 features)
Real data distribution, no synthetic balancing
"""
import os, sys, json, pickle, torch
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader

# v5 Feature set (aligned with XGBoost v5)
FEATURES_V5 = [
    "price_change", "high_low_range", "volume_change", "volume_ma_ratio",
    "ema_10", "ema_20", "ema_50", "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_position", "volatility_20",
    "momentum_10", "momentum_20",
    "ema_10_20_cross", "ema_10_50_cross", "volume_ratio"
]

LABEL = "signal_class"  # 0=SELL, 1=HOLD, 2=BUY
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_FILE = BASE_DIR / "datasets" / "training_data_v5.csv"
SAVE_DIR = BASE_DIR / "ai_engine" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 15

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class PatchTST(nn.Module):
    def __init__(self, input_dim=18, patch_len=16, stride=8, d_model=128, n_heads=8, n_layers=4, dropout=0.1, num_classes=3):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # Embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x: [batch, features]
        x = x.unsqueeze(1)  # [batch, 1, features]
        x = self.embedding(x)  # [batch, 1, d_model]
        x = self.transformer(x)  # [batch, 1, d_model]
        x = x.mean(dim=1)  # [batch, d_model]
        return self.fc(x)  # [batch, num_classes]

print("=" * 70)
print("PatchTST v5 Training - Aligned with XGBoost v5 (18 features)")
print(f"Device: {DEVICE}")
print("=" * 70)

# Load data
print(f"\n📊 Loading data from {DATA_FILE}")
if not DATA_FILE.exists():
    print(f"❌ ERROR: Data file not found: {DATA_FILE}")
    sys.exit(1)

df = pd.read_csv(DATA_FILE)
print(f"   Loaded {len(df)} rows")

# Check features
missing = [f for f in FEATURES_V5 if f not in df.columns]
if missing:
    print(f"❌ ERROR: Missing features: {missing}")
    sys.exit(1)

# Drop NaN
df = df.dropna(subset=FEATURES_V5 + [LABEL])
print(f"   After dropna: {len(df)} rows")

# Extract
X = df[FEATURES_V5].values
y = df[LABEL].values

# Class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"\n📈 Class distribution (REAL DATA - no balancing):")
for cls, cnt in zip(unique, counts):
    label_name = {0: "SELL", 1: "HOLD", 2: "BUY"}.get(int(cls), str(cls))
    print(f"   {label_name} ({cls}): {cnt} ({cnt/len(y)*100:.1f}%)")

# Split
print(f"\n🔀 Splitting train/val (80/20, stratified)")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train)}, Val: {len(X_val)}")

# Scale
print(f"\n⚙️ Scaling features")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Datasets
train_dataset = SequenceDataset(X_train_scaled, y_train)
val_dataset = SequenceDataset(X_val_scaled, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
print(f"\n🏗️ Building PatchTST model")
model = PatchTST(input_dim=len(FEATURES_V5), d_model=128, n_heads=8, n_layers=4, dropout=0.1).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
print(f"\n🚀 Training PatchTST v5 (epochs={EPOCHS}, patience={PATIENCE})")
best_acc = 0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    val_acc = accuracy_score(all_labels, all_preds)
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc*100:.2f}%")
    
    # Early stopping
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"   Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(best_model_state)

# Final evaluation
print(f"\n📊 Final Evaluation:")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(DEVICE)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"   Accuracy: {accuracy*100:.2f}%")

print(f"\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["SELL", "HOLD", "BUY"]))

print(f"\n📊 Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)
print(f"   Rows: Actual | Columns: Predicted")

# Check variety (anti-degeneracy)
unique_preds = len(np.unique(all_preds))
print(f"\n✅ Variety Check: {unique_preds}/3 unique predictions")
if unique_preds < 2:
    print(f"⚠️ WARNING: Low variety detected!")

# Save model, scaler, metadata
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_file = SAVE_DIR / f"patchtst_v{timestamp}_v5.pth"
scaler_file = SAVE_DIR / f"patchtst_v{timestamp}_v5_scaler.pkl"
meta_file = SAVE_DIR / f"patchtst_v{timestamp}_v5_meta.json"

print(f"\n💾 Saving model to {model_file}")
torch.save(model.state_dict(), model_file)

print(f"💾 Saving scaler to {scaler_file}")
with open(scaler_file, "wb") as f:
    pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"💾 Saving metadata to {meta_file}")
metadata = {
    "version": "v5",
    "timestamp": timestamp,
    "features": FEATURES_V5,
    "num_features": len(FEATURES_V5),
    "class_mapping": {0: "SELL", 1: "HOLD", 2: "BUY"},
    "accuracy": float(accuracy),
    "training_samples": len(X_train),
    "val_samples": len(X_val),
    "epochs_trained": epoch + 1,
    "best_accuracy": float(best_acc),
    "unique_predictions": int(unique_preds),
    "architecture": {
        "d_model": 128,
        "n_heads": 8,
        "n_layers": 4,
        "dropout": 0.1
    }
}
with open(meta_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n🎉 PatchTST v5 training complete!")
print(f"   Model: {model_file.name}")
print(f"   Accuracy: {accuracy*100:.2f}%")
print(f"   Variety: {unique_preds}/3 classes")
print("=" * 70)
