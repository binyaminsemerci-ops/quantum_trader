#!/usr/bin/env python3
"""
PatchTST v3 Training Script - 49 Features (FEATURES_V6 schema)
Trains SimplePatchTST transformer model with 49-feature input

Architecture:
- SimplePatchTST: Linear projection → Transformer encoder → Classification head
- Input: (batch, 49 features)
- Output: (batch, 3 classes) - SELL/HOLD/BUY

New in v3:
- 49 features (vs 23 in v2)
- Uses calculate_features_v6() from canonical module
- Aligned with ai_engine/common_features.py schema
- Matches patchtst_agent_v3.py SimplePatchTST architecture
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.clients.binance_market_data_client import BinanceMarketDataClient
from ops.retrain.calculate_features_v6 import calculate_features_v6, get_features_v6, create_labels

# v6 Feature set (49 features)
FEATURES_V6 = get_features_v6()

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
SAVE_DIR = BASE_DIR / "ai_engine" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimplePatchTST(nn.Module):
    """SimplePatchTST architecture matching patchtst_agent_v3.py"""
    def __init__(self, num_features=49, d_model=128, num_heads=4, num_layers=2, num_classes=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_features)
        Returns:
            (batch, num_classes)
        """
        x = self.input_proj(x).unsqueeze(1)  # (batch, 1, d_model)
        x = self.dropout(x)
        x = self.encoder(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        x = self.head(x)  # (batch, num_classes)
        return x


class TradingDataset(Dataset):
    """PyTorch Dataset for trading data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ============================================================================
# MAIN TRAINING
# ============================================================================

print("=" * 70)
print("PatchTST v3 Training - SimplePatchTST (49 features)")
print("=" * 70)

# Fetch data from Binance
print(f"\n📊 Fetching data from Binance...")
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "DOTUSDT", "MATICUSDT"]
bc = BinanceMarketDataClient()
dfs = []

for symbol in symbols:
    print(f"   {symbol}: ", end="", flush=True)
    candles = bc.get_latest_candles(symbol, "1h", limit=1000)
    if candles is not None and len(candles) > 0:
        print(f"{len(candles)} candles")
        dfs.append(candles)
    else:
        print("FAILED")

if not dfs:
    print("❌ ERROR: No data fetched from Binance")
    sys.exit(1)

df = pd.concat(dfs, ignore_index=True)
df_original = df.copy()
print(f"[INFO] Total raw samples: {len(df)}")

# Calculate v6 features
print(f"\n[FEATURES] Calculating 49 v6 features...")
df = calculate_features_v6(df)

# Verify features
available_features = [f for f in FEATURES_V6 if f in df.columns]
if len(available_features) < 49:
    print(f"⚠️ WARNING: Using {len(available_features)}/49 features")
    FEATURES_V6_USE = available_features
else:
    print(f"[FEATURES] ✅ All 49 features calculated, {len(df)} valid samples")
    FEATURES_V6_USE = FEATURES_V6

# Create labels
print(f"\n[LABELS] Creating labels...")
df, y = create_labels(df, df_original, threshold=0.015, lookahead=5)

# Extract features
X = df[FEATURES_V6_USE].values
print(f"[DATA] Feature shape: {X.shape}, Labels shape: {y.shape}")

# Train/validation/test split (70/15/15)
print(f"\n[SPLIT] Splitting data...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)
print(f"[SPLIT] Train: {len(X_train)}, Valid: {len(X_val)}, Test: {len(X_test)}")

# Scale features
print(f"\n⚙️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print(f"[SCALER] Feature dimension: {scaler.n_features_in_}")

# Class weights for imbalanced data
class_counts = np.bincount(y_train.astype(int))
class_weights = torch.FloatTensor([max(class_counts) / c for c in class_counts]).to(DEVICE)
print(f"\n⚖️ Class weights: {class_weights.cpu().numpy()}")

# Create datasets and dataloaders
batch_size = 128
train_dataset = TradingDataset(X_train_scaled, y_train)
val_dataset = TradingDataset(X_val_scaled, y_val)
test_dataset = TradingDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
num_features = len(FEATURES_V6_USE)
model = SimplePatchTST(
    num_features=num_features,
    d_model=128,
    num_heads=4,
    num_layers=2,
    num_classes=3,
    dropout=0.1
).to(DEVICE)

print(f"\n[MODEL] SimplePatchTST initialized")
print(f"[MODEL] Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"[MODEL] Device: {DEVICE}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training loop
print(f"\n[TRAIN] Starting training...")
num_epochs = 50
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch:3d}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model state
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n[TRAIN] Early stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(best_model_state)
print(f"\n[TRAIN] ✅ Training complete")

# Evaluate on test set
print(f"\n[EVAL] Evaluating on test set...")
test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, DEVICE)

print(f"\n[METRICS] Test Loss: {test_loss:.4f}")
print(f"[METRICS] Test Accuracy: {test_acc:.4f}")
print(f"\n[CONFUSION MATRIX]")
print(confusion_matrix(y_true, y_pred))
print(f"\n[CLASSIFICATION REPORT]")
print(classification_report(y_true, y_pred, target_names=['SELL', 'HOLD', 'BUY']))

# Save model
version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_filename = f"patchtst_v3_{version}.pth"
scaler_filename = f"patchtst_v3_{version}_scaler.pkl"
metadata_filename = f"patchtst_v3_{version}_metadata.json"

model_path = SAVE_DIR / model_filename
scaler_path = SAVE_DIR / scaler_filename
metadata_path = SAVE_DIR / metadata_filename

# Save model checkpoint
print(f"\n[SAVE] Saving model...")
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_config': {
        'num_features': num_features,
        'd_model': 128,
        'num_heads': 4,
        'num_layers': 2,
        'num_classes': 3,
        'dropout': 0.1
    },
    'features': FEATURES_V6_USE,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_var': scaler.var_.tolist(),
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss)
}
torch.save(checkpoint, model_path)
print(f"[SAVE] Model → {model_path}")

# Save scaler
joblib.dump(scaler, scaler_path)
print(f"[SAVE] Scaler → {scaler_path}")

# Save metadata
metadata = {
    "version": f"patchtst_v3_{version}",
    "model_type": "SimplePatchTST",
    "architecture": "PatchTST-v3",
    "features": FEATURES_V6_USE,
    "num_features": num_features,
    "feature_schema": "FEATURES_V6",
    "training_date": datetime.utcnow().isoformat() + "Z",
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "hyperparameters": {
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "num_epochs_trained": epoch
    },
    "scaler_n_features": int(scaler.n_features_in_),
    "device": str(DEVICE)
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"[SAVE] Metadata → {metadata_path}")

print(f"\n✅ PatchTST v3 Training Complete!")
print(f"   Model: {model_filename}")
print(f"   Scaler: {scaler_filename}")
print(f"   Metadata: {metadata_filename}")
print(f"   Test Accuracy: {test_acc:.4f}")
print(f"   Features: {num_features} (v6 schema)")
print(f"\n💡 Next steps:")
print(f"   1. Copy {model_filename} to ai_engine/models/")
print(f"   2. Copy {scaler_filename} to ai_engine/models/")
print(f"   3. Test with patchtst_agent_v3.py")
print(f"   4. Verify in ensemble_predictor_service")
