import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# === Konfigurasjon ===
DATA_PATH = "ops/retrain/train_full.csv"
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
N_EPOCHS = 50
HIDDEN_SIZE = 256  # Must match nhits_agent.py (line 94)
NUM_CLASSES = 3
LEARNING_RATE = 0.001
DROPOUT = 0.2
BATCH_SIZE = 64

# === Last treningsdata ===
df = pd.read_csv(DATA_PATH)
X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values.astype(np.int64)
input_size = X.shape[1]

print(f"📊 Training SimpleNHiTS v2")
print(f"Samples: {len(X)}, Features: {input_size}, Classes: {df['label'].nunique()}")

# === Definer SimpleNHiTS arkitektur (MUST match ai_engine/nhits_simple.py) ===
class SimpleNHiTS(nn.Module):
    """Training uses seq_len=1, Production uses seq_len=120
    
    Training: input_size=1 (single candle) × num_features=23 = 23 input neurons
    Production: Agent loads with input_size=120 (120 candles) × 23 = 2760 input neurons
    
    Solution: Train with input_size=1, then agent will resize first layer on load
    """
    def __init__(self, seq_len=1, hidden_size=256, num_features=23, num_classes=3, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.flatten = nn.Flatten()
        
        # Block-based MLP (matches nhits_simple.py ModuleList structure)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len * num_features, hidden_size),  # 1×23=23 -> 256
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),  # 256 -> 128
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 64),  # 128 -> 64
                nn.LayerNorm(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(dropout)
            )
        ])
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        # Training: x shape is (batch, features) where features=23
        # We need to treat it as (batch, seq_len=1, num_features=23) for compatibility
        # Production: x shape is (batch, seq_len=120, num_features=23)
        
        if len(x.shape) == 2:
            # Training mode: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
        
        # Now x is (batch, seq_len, num_features)
        x = self.flatten(x)  # (batch, seq_len * num_features)
        
        for block in self.blocks:
            x = block(x)
        logits = self.output_layer(x)
        
        # Return tuple (logits, dummy_forecast) for compatibility with nhits_agent.py
        dummy_forecast = logits[:, :1]  # Just take first logit
        return logits, dummy_forecast

# === Initialiser model ===
# Train with seq_len=1 (single candle), Agent will load with seq_len=120 (120 candles)
model = SimpleNHiTS(seq_len=1, hidden_size=HIDDEN_SIZE, num_features=input_size, num_classes=NUM_CLASSES, dropout=DROPOUT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Til Tensor ===
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# === Treningsløkke ===
model.train()
for epoch in range(N_EPOCHS):
    perm = torch.randperm(len(X_tensor))
    epoch_loss = 0.0
    for i in range(0, len(X_tensor), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        xb, yb = X_tensor[idx], y_tensor[idx]
        optimizer.zero_grad()
        logits, _ = model(xb)  # Model returns (logits, dummy_forecast)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{N_EPOCHS} - Loss: {epoch_loss/len(X_tensor):.6f}")

# === Beregn normaliseringsdata ===
feature_mean = torch.FloatTensor(X.mean(axis=0))
feature_std = torch.FloatTensor(X.std(axis=0) + 1e-8)

# === Lagre model-checkpoint ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = OUTPUT_DIR / f"nhits_v{timestamp}_v2.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "seq_len": 1,  # Training seq_len (agent loads with 120)
    "hidden_size": HIDDEN_SIZE,
    "num_classes": NUM_CLASSES,
    "num_features": input_size,  # Feature dimension (23)
    "feature_mean": feature_mean,
    "feature_std": feature_std
}, model_path)

print("\n✅ Model saved:", model_path)
print("🧠 Architecture: seq_len=1 (train) × num_features={} → {} → 64 → 3 (agent loads with seq_len=120)".format(input_size, HIDDEN_SIZE))
print("📈 feature_mean/std:", feature_mean.shape)
