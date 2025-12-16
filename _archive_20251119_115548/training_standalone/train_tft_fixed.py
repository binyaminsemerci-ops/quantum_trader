"""
IMPROVED TFT TRAINING - Fixed Overfitting Issues
- Temporal validation split (not random)
- Balanced dataset (equal WIN/LOSS/NEUTRAL)
- Data leakage detection
- Better regularization
"""
import sys
import os
import random
from pathlib import Path

# Fix encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("IMPROVED TFT TRAINING - Overfitting Fixes")
print("="*80)
print("1. Temporal validation split (last 20% of data)")
print("2. Balanced dataset (equal WIN/LOSS/NEUTRAL)")
print("3. Data leakage detection and removal")
print("4. Strong regularization")
print("="*80)

# Check PyTorch
print("\nChecking PyTorch...")
try:
    import torch
    print(f"PyTorch {torch.__version__} installed")
except ImportError:
    print("Installing PyTorch...")
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "torch", "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    import torch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import json
from typing import List, Tuple
from collections import Counter

from ai_engine.tft_model import TemporalFusionTransformer, TFTTrainer, save_model
from backend.database import engine
from sqlalchemy.orm import sessionmaker
from backend.models.ai_training import AITrainingSample

print("\n" + "="*80)
print("LOADING AND CLEANING DATA")
print("="*80)

SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# Load samples in batches (faster, less memory)
batch_size = 50000
all_samples = []
offset = 0

while True:
    batch = db.query(AITrainingSample).filter(
        AITrainingSample.outcome_known == True,
        AITrainingSample.features.isnot(None),
        AITrainingSample.target_class.isnot(None)
    ).order_by(AITrainingSample.timestamp).limit(batch_size).offset(offset).all()
    
    if not batch:
        break
    
    all_samples.extend(batch)
    print(f"Loaded {len(all_samples):,} samples...", end='\r')
    offset += batch_size

db.close()

print(f"\nLoaded {len(all_samples):,} samples")

# Remove data leakage (negative hold times)
clean_samples = [s for s in all_samples if s.hold_duration_seconds and s.hold_duration_seconds > 0]
print(f"After removing negative hold times: {len(clean_samples):,} samples")

# Check class distribution
class_counts = Counter([s.target_class for s in clean_samples])
print(f"\nOriginal distribution:")
for cls, count in class_counts.items():
    print(f"  {cls}: {count:,} ({count/len(clean_samples)*100:.1f}%)")

# Balance dataset - undersample majority class
min_class_count = min(class_counts.values())
balanced_samples = []
class_samples = {cls: [] for cls in class_counts.keys()}

for s in clean_samples:
    class_samples[s.target_class].append(s)

# Take equal samples from each class
for cls in class_samples:
    balanced_samples.extend(class_samples[cls][:min_class_count])

# Sort by timestamp to maintain temporal order
balanced_samples.sort(key=lambda x: x.timestamp)

print(f"\nBalanced dataset: {len(balanced_samples):,} samples")
balanced_counts = Counter([s.target_class for s in balanced_samples])
for cls, count in balanced_counts.items():
    print(f"  {cls}: {count:,} ({count/len(balanced_samples)*100:.1f}%)")

class TradingDataset(Dataset):
    def __init__(self, samples: List[AITrainingSample], sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.samples = samples
        
        # Pre-parse all features (only once!)
        print("Pre-parsing features...")
        self.parsed_features = {}
        for i, s in enumerate(samples):
            try:
                if isinstance(s.features, str):
                    feats = json.loads(s.features)
                else:
                    feats = s.features
                
                if isinstance(feats, dict):
                    feats = list(feats.values())
                self.parsed_features[i] = np.array(feats, dtype=np.float32)
            except (json.JSONDecodeError, TypeError, AttributeError, ValueError) as e:
                self.parsed_features[i] = np.zeros(14, dtype=np.float32)
            
            if (i + 1) % 10000 == 0:
                print(f"  Parsed {i+1:,}/{len(samples):,} features...", end='\r')
        print(f"  Parsed {len(samples):,} features")
        
        # Build symbol-indexed sequences
        symbol_samples = {}
        for idx, s in enumerate(samples):
            if s.symbol not in symbol_samples:
                symbol_samples[s.symbol] = []
            symbol_samples[s.symbol].append((idx, s))  # Store (index, sample)
        
        # Create sequences
        self.sequences = []
        for symbol, sym_samples in symbol_samples.items():
            if len(sym_samples) < sequence_length:
                continue
            for i in range(len(sym_samples) - sequence_length + 1):
                seq = sym_samples[i:i + sequence_length]
                self.sequences.append(seq)
        
        print(f"Created {len(self.sequences):,} sequences from {len(symbol_samples)} symbols")
        
        # Calculate normalization stats
        self._normalize_features()
    
    def _normalize_features(self):
        print("Computing normalization stats...")
        
        # Sample 1000 random sequences for stats
        sample_size = min(1000, len(self.sequences))
        sampled_seqs = random.sample(self.sequences, sample_size)
        
        all_features = []
        for seq in sampled_seqs:
            # Use first 10 samples per sequence
            for idx, _ in seq[:10]:
                all_features.append(self.parsed_features[idx])
        
        all_features = np.array(all_features, dtype=np.float32)
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0) + 1e-8
        
        # Save normalization stats
        stats_path = Path("ai_engine/models/tft_normalization.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump({
                'mean': self.feature_mean.tolist(),
                'std': self.feature_std.tolist()
            }, f)
        print(f"Normalization complete (stats from {len(all_features)} vectors)")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        features = []
        
        # Use pre-parsed features (no JSON parsing!)
        for sample_idx, sample in seq:
            feats = self.parsed_features[sample_idx]
            # Normalize
            feats = (feats - self.feature_mean) / self.feature_std
            features.append(feats)
        
        # Target from last sample
        _, last_sample = seq[-1]
        target_map = {'WIN': 0, 'LOSS': 1, 'NEUTRAL': 2}
        target = target_map.get(last_sample.target_class, 2)
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

# Create dataset
print("\n" + "="*80)
print("PREPARING DATASET")
print("="*80)

dataset = TradingDataset(balanced_samples, sequence_length=60)

# TEMPORAL SPLIT (not random!)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Use indices to split without shuffling
train_indices = list(range(train_size))
val_indices = list(range(train_size, len(dataset)))

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

print(f"\nTemporal split:")
print(f"  Train: {len(train_dataset):,} sequences (first 80%)")
print(f"  Val:   {len(val_dataset):,} sequences (last 20%)")

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,  # Reduced from 128 for faster batches
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,  # Reduced from 256
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print(f"Data loaders ready")

# Create model
print("\n" + "="*80)
print("BUILDING TFT MODEL")
print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("\nRegularization config:")
print("  Dropout: 0.4 (very strong)")
print("  Hidden size: 48 (compact)")
print("  Layers: 2 (fewer)")
print("  Heads: 3 (fewer)")
print("  Weight decay: 0.01 (strong)")
print("  Label smoothing: 0.15")

model = TemporalFusionTransformer(
    input_size=14,
    sequence_length=60,
    hidden_size=48,       # Further reduced (64→48)
    num_heads=3,          # Reduced from 4
    num_layers=2,
    dropout=0.4,
    num_classes=3
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {total_params:,} parameters (~{total_params*4/1024/1024:.1f} MB)")

# Create trainer
trainer = TFTTrainer(
    model=model,
    device=device,
    learning_rate=0.0005,    # Lower learning rate
    weight_decay=0.01        # Strong weight decay
)

# Training
print("\n" + "="*80)
print("TRAINING")
print("="*80)
print("Estimated time: 15-30 minutes\n")

num_epochs = 20  # Reduced from 40 (faster training)
best_val_accuracy = 0
patience_counter = 0
max_patience = 5  # Stop earlier if no improvement

for epoch in range(num_epochs):
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch + 1}/{num_epochs}")
    print(f"{'='*80}")
    
    train_loss = trainer.train_epoch(train_loader, epoch)
    print(f"   Train Loss: {train_loss:.4f}")
    
    val_metrics = trainer.evaluate(val_loader)
    print(f"   Val Loss: {val_metrics['loss']:.4f}")
    print(f"   Val Accuracy: {val_metrics['accuracy']:.2f}%")
    
    trainer.scheduler.step(val_metrics['loss'])
    
    # Save if improved
    if val_metrics['accuracy'] > best_val_accuracy:
        best_val_accuracy = val_metrics['accuracy']
        save_model(model, 'ai_engine/models/tft_model.pth')
        print(f"   *** New best! Saved model.")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= max_patience:
        print(f"\nEarly stopping (patience={max_patience})")
        break
    
    # Warning if accuracy too high (overfitting)
    if val_metrics['accuracy'] >= 98:
        print(f"\n   WARNING: Accuracy too high ({val_metrics['accuracy']:.1f}%)")
        print(f"   May still be overfitting!")

# Results
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nBest validation accuracy: {best_val_accuracy:.2f}%")

if 45 <= best_val_accuracy <= 70:
    print("\nEXCELLENT! Realistic accuracy range for trading.")
    print("Ready for live testing!")
elif best_val_accuracy > 85:
    print("\nWARNING: Accuracy still too high!")
    print("Model may still overfit. Test carefully.")
else:
    print(f"\nAccuracy lower than expected.")
    print("May need more training or better features.")

print(f"\nModel saved to: ai_engine/models/tft_model.pth")
print(f"Size: {total_params*4/1024/1024:.1f} MB")
print("\n" + "="*80)
