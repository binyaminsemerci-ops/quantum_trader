"""
Train PatchTST model for crypto trading
Patch-based transformer (SOTA 2023)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import logging

from ai_engine.patchtst_model import PatchTST, PatchTSTTrainer, save_model
from backend.shared.unified_features import get_feature_engineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize feature engineer
feature_engineer = get_feature_engineer()


def prepare_sequences(df: pd.DataFrame, sequence_length: int = 120):
    """Prepare sequences for PatchTST training using unified features."""
    # Use existing unified features - already has ema_9, ema_21, ema_50, ema_200
    # Map to simplified 12-feature set
    
    df['price_change'] = df['returns']
    df['high_low_range'] = df['price_range']
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    df['volume_ma_ratio'] = df['volume_ratio']
    
    # Use existing EMAs and create crosses
    df['ema_10'] = df['ema_9']  # Use existing ema_9 as proxy for ema_10
    df['ema_20'] = df['ema_21']  # Use existing ema_21 as proxy for ema_20
    # ema_50 already exists
    df['ema_10_20_cross'] = df['ema_9_dist']  # Distance from ema_9
    df['ema_10_50_cross'] = df['ema_50_dist']  # Distance from ema_50
    
    df['rsi_14'] = df['rsi']
    df['volatility_20'] = df['volatility']
    # macd already exists
    
    features = [
        'price_change', 'high_low_range', 'volume_change', 'volume_ma_ratio',
        'ema_10', 'ema_20', 'ema_50', 'ema_10_20_cross', 'ema_10_50_cross',
        'rsi_14', 'volatility_20', 'macd'
    ]
    
    # Verify all features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}. Available: {list(df.columns[:20])}")
    
    sequences = []
    targets = []
    
    for i in range(len(df) - sequence_length - 5):  # -5 for forward looking
        seq = df[features].iloc[i:i+sequence_length].values
        
        # Target: future price movement
        future_return = (df['close'].iloc[i+sequence_length+5] - df['close'].iloc[i+sequence_length]) / df['close'].iloc[i+sequence_length]
        
        if future_return > 0.005:
            target = 2  # BUY
        elif future_return < -0.005:
            target = 0  # SELL
        else:
            target = 1  # HOLD
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def main():
    logger.info("=" * 60)
    logger.info("[ROCKET] PatchTST TRAINING FOR CRYPTO TRADING")
    logger.info("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Load training data (try full dataset first)
    data_path = Path("data/binance_training_data_full.csv")
    if not data_path.exists():
        data_path = Path("data/binance_training_data.csv")
        if not data_path.exists():
            logger.error(f"❌ Training data not found: {data_path}")
            logger.info("   Run: python scripts/combine_training_data.py first")
            return
    
    logger.info(f"📂 Loading training data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"   Loaded {len(df)} rows")
    
    # Get unique symbols
    symbols = df['symbol'].unique()
    logger.info(f"   Found {len(symbols)} symbols")
    
    # Use only top 20 symbols to reduce memory usage
    symbols = symbols[:20]
    logger.info(f"   Training on {len(symbols)} symbols (memory optimization)")
    
    # Process each symbol
    all_sequences = []
    all_targets = []
    
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = feature_engineer.compute_features(symbol_df)
        
        if len(symbol_df) < 150:
            continue
        
        # Debug: show available columns
        if symbol == symbols[0]:  # Only for first symbol
            logger.info(f"[DEBUG] Columns after compute_features: {list(symbol_df.columns)}")
        
        seqs, tgts = prepare_sequences(symbol_df, sequence_length=120)
        all_sequences.append(seqs)
        all_targets.append(tgts)
    
    # Concatenate
    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_targets, axis=0)
    
    logger.info(f"[OK] Created {len(X)} sequences")
    logger.info(f"   Shape: {X.shape}")
    logger.info(f"   Labels: BUY={sum(y==2)}, HOLD={sum(y==1)}, SELL={sum(y==0)}")
    
    # Normalize
    feature_mean = X.mean(axis=(0, 1))
    feature_std = X.std(axis=(0, 1))
    X_norm = (X - feature_mean) / (feature_std + 1e-8)
    
    logger.info(f"[OK] Normalization stats saved")
    
    # Train/val split
    split_idx = int(0.8 * len(X_norm))
    X_train, X_val = X_norm[:split_idx], X_norm[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"[CHART] Train: {len(X_train)} | Val: {len(y_val)}")
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True  # Reduced for memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=False
    )
    
    # Create model - reduced parameters for CPU training
    model = PatchTST(
        input_size=120,
        patch_len=12,  # 120/12 = 10 patches
        d_model=64,    # Reduced from 128
        nhead=4,       # Reduced from 8
        num_layers=2,  # Reduced from 3
        dim_feedforward=256,  # Reduced from 512
        dropout=0.1,
        num_features=12  # Changed from 14 to match actual features
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[OK] Model initialized with {num_params:,} parameters")
    
    # Create trainer
    trainer = PatchTSTTrainer(model, device=device, learning_rate=0.001)
    
    # Training loop
    logger.info("\n🏋️ Starting training for 25 epochs (memory optimized)\n")
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(25):
        logger.info(f"[CHART_UP] Epoch {epoch+1}/25")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        logger.info(f"[OK] Epoch complete: Loss={train_loss:.4f}")
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        logger.info(f"[CHART] Validation metrics:")
        logger.info(f"   Loss: {val_metrics['loss']:.4f}")
        logger.info(f"   Accuracy: {val_metrics['accuracy']:.2f}%")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            model_path = Path("ai_engine/models/patchtst_model.pth")
            model_path.parent.mkdir(exist_ok=True, parents=True)
            save_model(model, str(model_path), feature_mean=feature_mean, feature_std=feature_std)
            
            logger.info(f"[OK] Best model saved (val_loss={val_metrics['loss']:.4f})")
            
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'accuracy': val_metrics['accuracy'],
                'num_params': num_params,
                'sequence_length': 120,
                'patch_len': 12,
                'd_model': 128
            }
            
            metadata_path = Path("ai_engine/models/patchtst_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        else:
            patience_counter += 1
            logger.info(f"⏸️  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                logger.info(f"🛑 Early stopping triggered")
                break
        
        logger.info("")
    
    print("\n" + "="*60)
    print("[OK] PatchTST TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: ai_engine/models/patchtst_model.pth")
    print("\n[ROCKET] Ready to deploy!")
    print("   Restart backend to load new model")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
