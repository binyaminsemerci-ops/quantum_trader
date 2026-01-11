"""
PatchTST v3 retraining script — 23-feature normalized version
Author: Quantum Trader AI Core Team
Date: 2026-01-11
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# === Paths ===
DATA_PATH = "ops/retrain/train_full.csv"
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = f"patchtst_v{datetime.now().strftime('%Y%m%d_%H%M%S')}_v3"
MODEL_PATH = OUTPUT_DIR / f"{MODEL_NAME}.pth"
SCALER_PATH = OUTPUT_DIR / f"{MODEL_NAME}_scaler.pkl"
META_PATH = OUTPUT_DIR / f"{MODEL_NAME}_meta.json"

# === Load data ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label"]).astype(np.float32)
y = df["label"].astype(np.int64)
features = list(X.columns)

print(f"📊 Loaded {len(X)} samples with {len(features)} features for PatchTST v3 training.")

# === Normalize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print(f"🧩 Scaler saved to {SCALER_PATH}")

# === Train/test split ===
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"🧠 Train: {len(X_train)}, Val: {len(X_val)}")

# === Tensor datasets ===
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

# === Define PatchTST model ===
# This model structure should match ai_engine/patchtst_simple.py
class SimplePatchTST(torch.nn.Module):
    def __init__(self, num_features: int, d_model=128, num_heads=4, num_layers=2, num_classes=3):
        super().__init__()
        self.input_proj = torch.nn.Linear(num_features, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=256, dropout=0.1, activation="gelu"
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (batch, features)
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, d_model)
        x = self.encoder(x)
        x = x.mean(dim=1)  # global average pooling
        return self.head(x)

# === Initialize model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimplePatchTST(num_features=len(features)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
criterion = torch.nn.CrossEntropyLoss()

print(f"🚀 Starting training on {device} ...")

# === Training loop with early stopping ===
epochs = 100
patience = 10
best_val_loss = np.inf
patience_counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train_t.to(device))
    loss = criterion(out, y_train_t.to(device))
    loss.backward()
    optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t.to(device))
        val_loss = criterion(val_out, y_val_t.to(device)).item()

    print(f"Epoch {epoch+1:03d} | Train: {loss.item():.4f} | Val: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("⏹️ Early stopping triggered.")
        break

print(f"✅ Training complete. Best val loss: {best_val_loss:.4f}")

# === Save metadata ===
metadata = {
    "timestamp": datetime.now().isoformat(),
    "features": features,
    "num_features": len(features),
    "model_type": "PatchTST-v3",
    "best_val_loss": float(best_val_loss),
    "device": device,
}
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"💾 Model saved to {MODEL_PATH}")
print(f"🧠 Metadata saved to {META_PATH}")
print(f"✅ PatchTST v3 retraining finished successfully.")
