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
    def __init__(self, input_size=23, hidden_size=64, num_features=23, num_classes=3, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.flatten = nn.Flatten()
        
        # Block-based MLP (matches nhits_simple.py ModuleList structure)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size * num_features, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 64),
                nn.LayerNorm(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(dropout)
            )
        ])
        self.output_layer = nn.Linear(64, 3)

    def forward(self, x):
        # x shape: (batch, features) - no sequence dimension for simplified version
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # (features,) -> (1, features)
        
        # For sequence-based: flatten (batch, seq, feat) -> (batch, seq*feat)
        # For single-step: treat input_size as 1, expand to (batch, 1, features)
        if len(x.shape) == 2:
            batch_size, features = x.shape
            x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
        
        x = self.flatten(x)  # (batch, 1, features) -> (batch, 1*features)
        
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

# === Initialiser model ===
# Note: input_size is sequence length (1 for single-step), num_features is feature dimension
model = SimpleNHiTS(input_size=1, hidden_size=HIDDEN_SIZE, num_features=input_size, num_classes=NUM_CLASSES, dropout=DROPOUT)
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
        outputs = model(xb)
        loss = criterion(outputs, yb)
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
    "input_size": 1,  # Sequence length (single-step classification)
    "hidden_size": HIDDEN_SIZE,
    "num_classes": NUM_CLASSES,
    "num_features": input_size,  # Feature dimension (23)
    "feature_mean": feature_mean,
    "feature_std": feature_std
}, model_path)

print("\n✅ Model saved:", model_path)
print("🧠 Architecture: input_size=1 × num_features={} → {} → 64 → 3".format(input_size, HIDDEN_SIZE))
print("📈 feature_mean/std:", feature_mean.shape)
