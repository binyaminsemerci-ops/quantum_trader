#!/usr/bin/env python3
"""
PatchTST Retraining Script - P0.4 SAFE CONTROLLED RETRAIN
- Trains in isolated directory
- Resource controls (nice/ionice)
- Disk checks before training
- Comprehensive metrics and evaluation
- Saves timestamped model + metadata
"""
import sys
import os
import json
import time
import shutil
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import PatchTST model architecture
sys.path.insert(0, "/home/qt/quantum_trader")
from ai_engine.agents.patchtst_agent import PatchTSTModel

# Configuration
TRAINING_WINDOW_DAYS = 30  # Use last 30 days
MIN_SAMPLES = 1000
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0005
MIN_DISK_FREE_GB = 5.0
SEQUENCE_LENGTH = 128
NUM_FEATURES = 8

def check_disk_space():
    """Ensure sufficient disk space before training"""
    st = os.statvfs('/tmp')
    free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
    print(f"[DISK] Free space: {free_gb:.2f} GB")
    
    if free_gb < MIN_DISK_FREE_GB:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GB < {MIN_DISK_FREE_GB} GB required")
    
    return free_gb

def set_process_priority():
    """Set process to low priority to avoid impacting live services"""
    try:
        # Nice to 19 (lowest CPU priority)
        os.nice(19)
        print("[PRIORITY] Set to nice 19 (lowest CPU priority)")
        
        # Ionice to idle class (won't impact disk I/O)
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=False)
        print("[PRIORITY] Set to ionice idle class (no disk I/O impact)")
    except Exception as e:
        print(f"[PRIORITY] Warning: Could not set priority: {e}")

def load_training_data(db_path: str, window_days: int):
    """Load training data from database with time window"""
    print(f"\n{'='*60}")
    print(f"LOADING TRAINING DATA")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Window: Last {window_days} days")
    
    conn = sqlite3.connect(db_path)
    
    # Calculate cutoff timestamp
    cutoff_time = datetime.utcnow() - timedelta(days=window_days)
    cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Cutoff: {cutoff_str}")
    
    # Query with time filter
    query = f"""
    SELECT features, target_class, timestamp 
    FROM ai_training_samples 
    WHERE timestamp >= '{cutoff_str}'
    AND target_class IS NOT NULL
    ORDER BY timestamp DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"\nLoaded {len(df)} samples")
    
    if len(df) < MIN_SAMPLES:
        raise ValueError(f"Insufficient samples: {len(df)} < {MIN_SAMPLES} required")
    
    return df

def prepare_features(df):
    """Parse features from JSON and prepare training matrices"""
    print(f"\n{'='*60}")
    print(f"PREPARING FEATURES")
    print(f"{'='*60}")
    
    # Feature keys available in database
    feature_keys = ['rsi', 'ma_cross', 'volatility', 'returns_1h']
    
    features_list = []
    labels = []
    
    for idx, row in df.iterrows():
        try:
            features = json.loads(row['features']) if isinstance(row['features'], str) else row['features']
            
            # Extract 4 available features
            feat_vec = [features.get(k, 0.0) for k in feature_keys]
            
            # Pad to 8 features (duplicate to match model architecture)
            feat_vec_padded = feat_vec + feat_vec
            
            features_list.append(feat_vec_padded)
            
            # Binary classification: WIN=1, LOSS=0
            labels.append(1 if row['target_class'] == 'WIN' else 0)
        except Exception as e:
            # Skip malformed rows
            continue
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    print(f"Feature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
    
    # Data quality checks
    win_count = int(y.sum())
    loss_count = len(y) - win_count
    win_rate = win_count / len(y) if len(y) > 0 else 0
    
    print(f"\nTarget Distribution:")
    print(f"  WIN:  {win_count:5d} ({win_rate*100:5.1f}%)")
    print(f"  LOSS: {loss_count:5d} ({(1-win_rate)*100:5.1f}%)")
    
    # Check for class imbalance
    if win_rate < 0.3 or win_rate > 0.7:
        print(f"\n⚠️  WARNING: Class imbalance detected (win_rate={win_rate:.1%})")
        print("   Consider using class weights or resampling")
    
    return X, y

def create_sequences(X, y, sequence_length):
    """Create sequences by repeating tabular features (not true time series)"""
    print(f"\nCreating sequences (length={sequence_length})...")
    
    # Tile features to create sequences: (batch, seq_len, features)
    X_seq = np.tile(X[:, np.newaxis, :], (1, sequence_length, 1))
    
    print(f"Sequence shape: {X_seq.shape}")
    
    return X_seq

def train_model(X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """Train PatchTST model with validation"""
    print(f"\n{'='*60}")
    print(f"TRAINING PATCHTST")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    
    # Initialize model
    model = PatchTSTModel(
        input_dim=NUM_FEATURES,
        output_dim=1,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        patch_len=16,
        num_patches=SEQUENCE_LENGTH // 16
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Create data loaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_f1 = 0.0
    best_model_state = None
    
    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<12}")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        y_pred_list = []
        y_true_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
                
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                y_pred_list.extend(predictions.cpu().numpy())
                y_true_list.extend(y_batch.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Calculate metrics
        val_acc = accuracy_score(y_true_list, y_pred_list)
        val_f1 = f1_score(y_true_list, y_pred_list, zero_division=0)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
        
        print(f"{epoch+1:<8d} {avg_train_loss:<12.4f} {avg_val_loss:<12.4f} {val_acc*100:<11.2f}% {val_f1:<12.4f}")
    
    print("-" * 60)
    print(f"Best Val F1: {best_val_f1:.4f}")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    return model, history, best_val_f1

def evaluate_model(model, X_test, y_test):
    """Comprehensive evaluation on test set"""
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*60}")
    
    model.eval()
    
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    y_pred_list = []
    y_true_list = []
    y_prob_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            
            y_pred_list.extend(predictions.cpu().numpy())
            y_true_list.extend(y_batch.cpu().numpy())
            y_prob_list.extend(probs.cpu().numpy())
    
    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Accuracy:  {acc*100:6.2f}%")
    print(f"Precision: {prec*100:6.2f}%")
    print(f"Recall:    {rec*100:6.2f}%")
    print(f"F1 Score:  {f1:6.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"           LOSS    WIN")
    print(f"  Actual LOSS  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"         WIN   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Confidence distribution
    conf_bins = [0.0, 0.4, 0.45, 0.55, 0.6, 1.0]
    hist, _ = np.histogram(y_prob, bins=conf_bins)
    
    print(f"\nConfidence Distribution:")
    for i in range(len(conf_bins)-1):
        pct = (hist[i] / len(y_prob)) * 100
        print(f"  [{conf_bins[i]:.2f}, {conf_bins[i+1]:.2f}): {hist[i]:4d} ({pct:5.1f}%)")
    
    # Check for flatline
    unique_probs = len(np.unique(y_prob))
    print(f"\nUnique confidence values: {unique_probs}")
    
    if unique_probs <= 3:
        print("⚠️  WARNING: Low confidence diversity (possible flatline)")
    else:
        print("✅ Good confidence diversity")
    
    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'unique_confidences': int(unique_probs),
        'conf_min': float(y_prob.min()),
        'conf_max': float(y_prob.max()),
        'conf_mean': float(y_prob.mean()),
        'conf_std': float(y_prob.std())
    }
    
    return metrics

def save_model_checkpoint(model, metrics, history, output_dir):
    """Save model checkpoint with metadata"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"patchtst_v{timestamp}.pth"
    model_path = output_dir / model_filename
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': 'PatchTSTModel',
        'input_dim': NUM_FEATURES,
        'sequence_length': SEQUENCE_LENGTH,
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 4,
        'patch_len': 16,
        'num_patches': SEQUENCE_LENGTH // 16,
        'timestamp': timestamp,
        'metrics': metrics,
        'training_config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'training_window_days': TRAINING_WINDOW_DAYS
        }
    }
    
    torch.save(checkpoint, model_path)
    print(f"\n✅ Model saved: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024:.1f} KB")
    
    # Save metrics as JSON
    metrics_path = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'history': history,
            'config': checkpoint['training_config']
        }, f, indent=2)
    
    print(f"✅ Metrics saved: {metrics_path}")
    
    # Save training summary
    summary_path = output_dir / f"summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write(f"PatchTST Retraining Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_filename}\n")
        f.write(f"\nTest Metrics:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Precision: {metrics['precision']*100:.2f}%\n")
        f.write(f"  Recall:    {metrics['recall']*100:.2f}%\n")
        f.write(f"  F1 Score:  {metrics['f1']:.4f}\n")
        f.write(f"\nConfidence Stats:\n")
        f.write(f"  Min:  {metrics['conf_min']:.4f}\n")
        f.write(f"  Max:  {metrics['conf_max']:.4f}\n")
        f.write(f"  Mean: {metrics['conf_mean']:.4f}\n")
        f.write(f"  Std:  {metrics['conf_std']:.4f}\n")
        f.write(f"  Unique values: {metrics['unique_confidences']}\n")
    
    print(f"✅ Summary saved: {summary_path}")
    
    return model_path, metrics_path, summary_path

def main():
    print(f"\n{'#'*60}")
    print(f"# PATCHTST RETRAINING - P0.4 SAFE CONTROLLED RETRAIN")
    print(f"{'#'*60}")
    
    start_time = time.time()
    
    # Setup
    check_disk_space()
    set_process_priority()
    
    # Create isolated output directory
    timestamp_start = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"/tmp/patchtst_retrain/{timestamp_start}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[OUTPUT] Directory: {output_dir}")
    
    # Load data
    db_path = "/opt/quantum/data/quantum_trader.db"
    
    df = load_training_data(db_path, TRAINING_WINDOW_DAYS)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Create sequences
    X_seq = create_sequences(X, y, SEQUENCE_LENGTH)
    
    # Split data: 70% train, 15% val, 15% test
    n = len(X_seq)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    
    X_train = X_seq[:train_end]
    y_train = y[:train_end]
    X_val = X_seq[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X_seq[val_end:]
    y_test = y[val_end:]
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train):5d} ({len(X_train)/n*100:.1f}%)")
    print(f"  Val:   {len(X_val):5d} ({len(X_val)/n*100:.1f}%)")
    print(f"  Test:  {len(X_test):5d} ({len(X_test)/n*100:.1f}%)")
    
    # Train
    model, history, best_val_f1 = train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save checkpoint
    model_path, metrics_path, summary_path = save_model_checkpoint(
        model, metrics, history, output_dir
    )
    
    # Print final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Elapsed time: {elapsed/60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_path.name}")
    print(f"\nNext steps:")
    print(f"1. Review metrics in {metrics_path}")
    print(f"2. Compare with baseline model")
    print(f"3. If acceptable, deploy via PATCHTST_MODEL_PATH env var")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
