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
HIDDEN_SIZE = 64
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

# === Definer SimpleNHiTS arkitektur ===
class SimpleNHiTS(nn.Module):
    def __init__(self, input_size=23, hidden_size=64, num_classes=3, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.num_features = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.blocks = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

# === Initialiser model ===
model = SimpleNHiTS(input_size=input_size, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES, dropout=DROPOUT)
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
    "input_size": input_size,
    "hidden_size": HIDDEN_SIZE,
    "num_classes": NUM_CLASSES,
    "num_features": input_size,
    "feature_mean": feature_mean,
    "feature_std": feature_std
}, model_path)

print("\n✅ Model saved:", model_path)
print("🧠 Architecture:", f"{input_size} → {HIDDEN_SIZE} → {HIDDEN_SIZE} → {NUM_CLASSES}")
print("📈 feature_mean/std:", feature_mean.shape)
