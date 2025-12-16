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
from ai_engine.feature_engineer import compute_all_indicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_sequences(df: pd.DataFrame, sequence_length: int = 120):
    """Prepare sequences for PatchTST training."""
    # Derive needed features from compute_all_indicators output
    df['price_change'] = df['close'].pct_change().fillna(0.0)
    df['volume_change'] = df['volume'].pct_change().fillna(0.0)
    
    # Use existing MA columns and compute simple indicators
    df['volume_ma_ratio'] = (df['volume'] / df['volume_ma_20']).fillna(1.0) if 'volume_ma_20' in df.columns else 1.0
    
    # Compute EMAs from close price
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_10_20_cross'] = ((df['ema_10'] - df['ema_20']) / df['close']).fillna(0.0)
    df['ema_10_50_cross'] = ((df['ema_10'] - df['ema_50']) / df['close']).fillna(0.0)
    
    # Use existing volatility or compute it
    if 'hist_vol_20' in df.columns:
        df['volatility_20'] = df['hist_vol_20']
    else:
        df['volatility_20'] = df['close'].pct_change().rolling(20).std().fillna(0.0)
    
    # Compute MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = (df['macd'] - df['macd_signal']).fillna(0.0)
    
    # Fill any remaining NaN
    df = df.ffill().bfill().fillna(0)
    
    features = [
        'price_change', 'volume_change', 'volume_ma_ratio',
        'ema_10', 'ema_20', 'ema_50', 'ema_10_20_cross', 'ema_10_50_cross',
        'volatility_20', 'macd', 'macd_signal', 'macd_hist'
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
    
    # Process each symbol
    all_sequences = []
    all_targets = []
    
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = compute_all_indicators(symbol_df)
        
        if len(symbol_df) < 150:
            continue
        
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
        train_dataset, batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )
    
    # Create model
    model = PatchTST(
        input_size=120,
        patch_len=12,  # 120/12 = 10 patches
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        num_features=12  # Changed from 14 to match actual features
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[OK] Model initialized with {num_params:,} parameters")
    
    # Create trainer
    trainer = PatchTSTTrainer(model, device=device, learning_rate=0.001)
    
    # Training loop
    logger.info("\n🏋️ Starting training for 50 epochs\n")
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        logger.info(f"[CHART_UP] Epoch {epoch+1}/50")
        
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
