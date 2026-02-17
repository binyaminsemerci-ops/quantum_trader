#!/usr/bin/env python3
"""
TFT (Temporal Fusion Transformer) v6 Training – 49 features (FEATURES_V6 schema)
Fetches fresh data from Binance, trains TFT model with PyTorch

New in v6:
- 49 features (SPOT trading, no futures data)
- Uses calculate_features_v6() from canonical module
- TFT architecture: VSN + BiLSTM + Attention + GRN + Temporal Fusion
- Class-weighted loss for imbalanced data
"""
import os
import sys
import json
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

# v6 Feature set (49 features - canonical schema)
FEATURES_V6 = get_features_v6()

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
SAVE_DIR = BASE_DIR / "ai_engine" / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== TFT ARCHITECTURE (copied from unified_agents.py) ==========

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for TFT"""
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
    """Variable Selection Network for TFT"""
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
    """Temporal Fusion Block for TFT"""
    def __init__(self, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, 
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0)
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
    """Temporal Fusion Transformer model with 49-feature support (Feb 2026)"""
    def __init__(self, input_size=49, hidden_size=128, num_heads=8, num_layers=3, 
                 num_classes=3, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.vsn = VariableSelectionNetwork(input_size, hidden_size)
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                                    batch_first=True, bidirectional=True, dropout=dropout)
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
            x = x.unsqueeze(1)  # [batch, 1, features]
        
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
        logits = self.classifier(combined)
        
        return logits


class TradingDataset(Dataset):
    """PyTorch Dataset for trading data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ========== TRAINING SCRIPT ==========

print("=" * 70)
print("TFT v6 Training - Fresh from Binance (49 features)")
print(f"Device: {device}")
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

# Calculate v6 features (49 features)
print(f"\n[FEATURES] Calculating 49 v6 features...")
df = calculate_features_v6(df)

# Verify features
available_features = [f for f in FEATURES_V6 if f in df.columns]
missing_features = [f for f in FEATURES_V6 if f not in df.columns]

if missing_features:
    print(f"⚠️ WARNING: Missing features: {missing_features}")
    print(f"[FEATURES] Using {len(available_features)}/49 features")
    FEATURES_V6 = available_features
else:
    print(f"[FEATURES] ✅ All 49 features calculated, {len(df)} valid samples")

# Create labels
print(f"\n[LABELS] Creating labels...")
df, y = create_labels(df, df_original, threshold=0.015, lookahead=5)

# Extract features
X = df[FEATURES_V6].values
num_features = len(FEATURES_V6)

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

# Compute class weights
class_counts = np.bincount(y_train.astype(int))
class_weights = torch.FloatTensor([max(class_counts) / c for c in class_counts]).to(device)
print(f"\n⚖️ Class weights: {class_weights.cpu().numpy()}")

# Create datasets and loaders
train_dataset = TradingDataset(X_train_scaled, y_train)
val_dataset = TradingDataset(X_val_scaled, y_val)
test_dataset = TradingDataset(X_test_scaled, y_test)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
print(f"\n[MODEL] Initializing TFT (input_size={num_features}, hidden_size=128)...")
model = TFTModel(
    input_size=num_features,
    hidden_size=128,
    num_heads=8,
    num_layers=3,
    num_classes=3,
    dropout=0.1
).to(device)

param_count = sum(p.numel() for p in model.parameters())
print(f"[MODEL] Total parameters: {param_count:,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
print(f"\n[TRAIN] Training TFT v6...")
num_epochs = 50
best_val_loss = float('inf')
best_model_state = None
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()
    
    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f} ✅")
    else:
        patience_counter += 1
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f}")
        
        if patience_counter >= patience:
            print(f"\n[EARLY STOP] No improvement for {patience} epochs")
            break

# Load best model
model.load_state_dict(best_model_state)
print(f"\n[TRAIN] ✅ Training complete (best_val_loss={best_val_loss:.4f})")

# Evaluate on test set
print(f"\n[EVAL] Evaluating on test set...")
model.eval()
y_pred_list = []
y_true_list = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred_list.extend(predicted.cpu().numpy())
        y_true_list.extend(y_batch.numpy())

y_pred = np.array(y_pred_list)
y_true = np.array(y_true_list)

test_accuracy = accuracy_score(y_true, y_pred)
print(f"\n[METRICS] Test Accuracy: {test_accuracy:.4f}")
print(f"\n[CONFUSION MATRIX]")
print(confusion_matrix(y_true, y_pred))
print(f"\n[CLASSIFICATION REPORT]")
print(classification_report(y_true, y_pred, target_names=['SELL', 'HOLD', 'BUY']))

# Save model
version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
model_filename = f"tft_v6_{version}.pth"
scaler_filename = f"tft_v6_{version}_scaler.pkl"
metadata_filename = f"tft_v6_{version}_metadata.json"

model_path = SAVE_DIR / model_filename
scaler_path = SAVE_DIR / scaler_filename
metadata_path = SAVE_DIR / metadata_filename

print(f"\n[SAVE] Saving model...")

# Save model state dict and config
checkpoint = {
    'model_state_dict': model.state_dict(),
    'input_size': num_features,
    'hidden_size': 128,
    'num_heads': 8,
    'num_layers': 3,
    'num_classes': 3,
    'dropout': 0.1,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_var': scaler.var_.tolist(),
    'features': FEATURES_V6
}
torch.save(checkpoint, model_path)
print(f"[SAVE] Model → {model_path}")

# Save scaler separately (for compatibility)
import pickle
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"[SAVE] Scaler → {scaler_path}")

# Save metadata
metadata = {
    "version": f"tft_v6_{version}",
    "features": FEATURES_V6,
    "num_features": num_features,
    "feature_schema": "FEATURES_V6",
    "model_type": "TFT",
    "architecture": {
        "input_size": num_features,
        "hidden_size": 128,
        "num_heads": 8,
        "num_layers": 3,
        "num_classes": 3,
        "dropout": 0.1
    },
    "training_date": datetime.utcnow().isoformat() + "Z",
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "test_accuracy": float(test_accuracy),
    "best_val_loss": float(best_val_loss),
    "scaler_n_features": int(scaler.n_features_in_),
    "total_parameters": param_count
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"[SAVE] Metadata → {metadata_path}")

print(f"\n✅ TFT v6 Training Complete!")
print(f"   Model: {model_filename}")
print(f"   Scaler: {scaler_filename}")
print(f"   Metadata: {metadata_filename}")
print(f"   Test Accuracy: {test_accuracy:.4f}")
print(f"   Features: {num_features} (v6 schema)")
print(f"   Parameters: {param_count:,}")
print(f"\n💡 Next steps:")
print(f"   1. Copy all 3 files to VPS: /home/qt/quantum_trader/ai_engine/models/")
print(f"   2. Update ensemble_predictor_service.py to enable TFT")
print(f"   3. Restart quantum-ensemble-predictor service")
print(f"   4. Monitor logs for 'Models loaded (5 agents: ..., TFT)'")
