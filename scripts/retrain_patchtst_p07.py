#!/usr/bin/env python3
"""
PatchTST P0.7 Retraining Script — ANTI-COLLAPSE FIX
 
P0.6 FAILURE:
- Model collapsed to constant output (prob=0.5239 for ALL inputs)
- 100% HOLD actions (0.5239 in HOLD range [0.4, 0.6])
- Zero confidence spread (stddev=0.0000)
- Gates 1+2 FAILED

P0.7 FIXES:
1. Variance penalty loss (encourage prediction diversity)
2. Early-stop if val_std < 0.02 (catch collapse during training)
3. Reduced learning rate (0.0001 for stable convergence)
4. Increased epochs (40 for better learning)
5. NO label smoothing (was pushing outputs to 0.5)
6. Balanced sampling (keep 50/50 WIN/LOSS)

ENVIRONMENT: Systemd-only (NO DOCKER)
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
from sklearn.utils.class_weight import compute_class_weight

# Import PatchTST model architecture
sys.path.insert(0, "/home/qt/quantum_trader")
from ai_engine.agents.patchtst_agent import PatchTSTModel

# Configuration
TRAINING_WINDOW_DAYS = 30
MIN_SAMPLES = 1000
BATCH_SIZE = 32
EPOCHS = 40  # P0.7: Increased from 20
LEARNING_RATE = 0.0001  # P0.7: Reduced from 0.0003 for stability
MIN_DISK_FREE_GB = 5.0
SEQUENCE_LENGTH = 128
NUM_FEATURES = 4  # P0.7 FIX: Actual feature count (rsi, ma_cross, volatility, returns_1h)
LABEL_SMOOTHING = 0.0  # P0.7 FIX: DISABLED (was pushing outputs to 0.5)
VARIANCE_PENALTY_WEIGHT = 0.1  # P0.7: NEW anti-collapse regularization
EARLY_STOP_VAL_STD_THRESHOLD = 0.02  # P0.7: Stop if collapse detected

# Sanity check thresholds (HARD FAILS)
MAX_ACTION_PCT = 0.70  # No single action >70%
MIN_CLASSES_ABOVE_10 = 2  # At least 2 classes >10%
MIN_CONFIDENCE_STD = 0.02  # Confidence stddev ≥0.02 (Gate 2 threshold is 0.05, but we'll be conservative)
MIN_P10_P90_RANGE = 0.05  # P10-P90 range ≥0.05 (Gate 2 threshold is 0.12, but incremental improvement)


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
        os.nice(19)
        print("[PRIORITY] Set to nice 19 (lowest CPU priority)")
        
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
    
    cutoff_time = datetime.utcnow() - timedelta(days=window_days)
    cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Cutoff: {cutoff_str}")
    
    query = f"""
    SELECT features, target_class, timestamp, symbol
    FROM ai_training_samples 
    WHERE timestamp >= '{cutoff_str}'
    AND target_class IN ('WIN', 'LOSS')
    ORDER BY timestamp DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"\nLoaded {len(df)} samples")
    
    if len(df) < MIN_SAMPLES:
        raise ValueError(f"Insufficient samples: {len(df)} < {MIN_SAMPLES} required")
    
    return df


def prepare_features(df, balance_classes=True):
    """
    Parse features from JSON and prepare training matrices.
    
    P0.7 FIX: Use 4 real features (NO padding to 8).
    """
    print(f"\n{'='*60}")
    print(f"PREPARING FEATURES (P0.7 ANTI-COLLAPSE)")
    print(f"{'='*60}")
    
    feature_keys = ['rsi', 'ma_cross', 'volatility', 'returns_1h']
    
    features_list = []
    labels = []
    symbols = []
    
    for idx, row in df.iterrows():
        try:
            features = json.loads(row['features']) if isinstance(row['features'], str) else row['features']
            
            # P0.7: Use 4 real features (no padding to 8 - that confused P0.6 model)
            feat_vec = [features.get(k, 0.0) for k in feature_keys]
            
            features_list.append(feat_vec)
            
            # Binary classification: WIN=1, LOSS=0
            labels.append(1 if row['target_class'] == 'WIN' else 0)
            symbols.append(row.get('symbol', 'UNKNOWN'))
        except Exception as e:
            continue
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    symbols_arr = np.array(symbols)
    
    print(f"Feature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
    
    # BEFORE balancing stats
    win_count_before = int(y.sum())
    loss_count_before = len(y) - win_count_before
    win_rate_before = win_count_before / len(y) if len(y) > 0 else 0
    
    print(f"\n📊 BEFORE BALANCING:")
    print(f"  WIN:  {win_count_before:5d} ({win_rate_before*100:5.1f}%)")
    print(f"  LOSS: {loss_count_before:5d} ({(1-win_rate_before)*100:5.1f}%)")
    
    # P0.6 FIX: Stratified balanced sampling
    if balance_classes and abs(win_rate_before - 0.5) > 0.05:
        print(f"\n🔧 APPLYING BALANCED SAMPLING (P0.6 FIX)...")
        
        win_indices = np.where(y == 1)[0]
        loss_indices = np.where(y == 0)[0]
        
        # Undersample majority class
        n_minority = min(len(win_indices), len(loss_indices))
        
        np.random.seed(42)
        win_sampled = np.random.choice(win_indices, size=n_minority, replace=False)
        loss_sampled = np.random.choice(loss_indices, size=n_minority, replace=False)
        
        # Combine and shuffle
        balanced_indices = np.concatenate([win_sampled, loss_sampled])
        np.random.shuffle(balanced_indices)
        
        X = X[balanced_indices]
        y = y[balanced_indices]
        symbols_arr = symbols_arr[balanced_indices]
        
        # AFTER balancing stats
        win_count_after = int(y.sum())
        loss_count_after = len(y) - win_count_after
        win_rate_after = win_count_after / len(y) if len(y) > 0 else 0
        
        print(f"\n📊 AFTER BALANCING:")
        print(f"  WIN:  {win_count_after:5d} ({win_rate_after*100:5.1f}%)")
        print(f"  LOSS: {loss_count_after:5d} ({(1-win_rate_after)*100:5.1f}%)")
        print(f"  Total: {len(y)} samples")
        
        if abs(win_rate_after - 0.5) > 0.02:
            print(f"\n⚠️  WARNING: Balance not perfect ({win_rate_after:.1%}), but proceeding")
    else:
        print(f"\n✓ Classes already balanced, no sampling needed")
    
    return X, y, symbols_arr


def smooth_labels(y, epsilon=0.1):
    """
    P0.6 FIX: Label smoothing to prevent overconfidence.
    
    Instead of [0, 1], use [0.1, 0.9] to encourage wider confidence range.
    """
    return y * (1.0 - epsilon) + 0.5 * epsilon


def create_sequences(X, y, sequence_length):
    """Create sequences by repeating tabular features"""
    print(f"\nCreating sequences (length={sequence_length})...")
    
    X_seq = np.tile(X[:, np.newaxis, :], (1, sequence_length, 1))
    
    print(f"Sequence shape: {X_seq.shape}")
    
    return X_seq


def train_model(X_train, y_train, X_val, y_val, epochs, batch_size, lr, use_label_smoothing=True, use_class_weights=True):
    """
    Train PatchTST model with P0.6 improvements.
    
    NEW:
    - Label smoothing
    - Class weights (backup if balance insufficient)
    - Confidence tracking
    """
    print(f"\n{'='*60}")
    print(f"TRAINING PATCHTST P0.6")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Label smoothing: {use_label_smoothing} (epsilon={LABEL_SMOOTHING})")
    print(f"Class weights: {use_class_weights}")
    
    # P0.6 FIX: Apply label smoothing
    if use_label_smoothing:
        y_train_smooth = smooth_labels(y_train, epsilon=LABEL_SMOOTHING)
        y_val_smooth = smooth_labels(y_val, epsilon=LABEL_SMOOTHING)
        print(f"\n✓ Label smoothing applied: [0,1] → [{LABEL_SMOOTHING},{1-LABEL_SMOOTHING}]")
    else:
        y_train_smooth = y_train
        y_val_smooth = y_val
    
    # Initialize model
    model = PatchTSTModel(
        input_dim=NUM_FEATURES,  # P0.7: Now 4 (not 8)
        output_dim=1,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.2,  # P0.7: Increased from 0.1 for regularization
        patch_len=16,
        num_patches=SEQUENCE_LENGTH // 16
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Create data loaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_smooth))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val_smooth))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # P0.6 FIX: Class weights (backup if sampling insufficient)
    if use_class_weights:
        # Calculate weights from original (non-smoothed) labels
        class_weights_vals = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=y_train
        )
        pos_weight = torch.tensor([class_weights_vals[1] / class_weights_vals[0]])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"\n✓ Class weights: LOSS={class_weights_vals[0]:.2f}, WIN={class_weights_vals[1]:.2f}")
        print(f"  Pos weight: {pos_weight.item():.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_conf_mean': [],
        'val_conf_std': []
    }
    
    best_val_f1 = 0.0
    best_model_state = None
    
    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<12} {'Val F1':<12} {'Conf Std':<12}")
    print("-" * 80)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            
            # P0.7: Base BCE loss
            bce_loss = criterion(outputs, y_batch)
            
            # P0.7 FIX: Variance penalty (encourage spread)
            preds_prob = torch.sigmoid(outputs)
            variance_penalty = -torch.std(preds_prob)  # Negative = penalize low variance
            
            # P0.7: Combined loss
            loss = bce_loss + VARIANCE_PENALTY_WEIGHT * variance_penalty
            
            loss.backward()
            optimizer.step()
            total_train_loss += bce_loss.item()  # Track BCE for history
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        y_pred_list = []
        y_true_list = []
        y_conf_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                y_pred_list.extend(predictions.cpu().numpy())
                y_true_list.extend((y_batch > 0.5).cpu().numpy())  # De-smooth for metrics
                y_conf_list.extend(probs.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Calculate metrics
        val_acc = accuracy_score(y_true_list, y_pred_list)
        val_f1 = f1_score(y_true_list, y_pred_list, zero_division=0)
        val_conf_mean = np.mean(y_conf_list)
        val_conf_std = np.std(y_conf_list)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_conf_mean'].append(val_conf_mean)
        history['val_conf_std'].append(val_conf_std)
        
        # P0.7: EARLY ABORT CHECK (catch collapse during training)
        if val_conf_std < EARLY_STOP_VAL_STD_THRESHOLD:
            print(f"\n❌ MODEL COLLAPSE DETECTED!")
            print(f"Val confidence stddev: {val_conf_std:.6f} < {EARLY_STOP_VAL_STD_THRESHOLD}")
            print(f"Aborting training at epoch {epoch+1}/{epochs}")
            raise ValueError(f"Model collapsing: val_conf_std={val_conf_std:.6f} < {EARLY_STOP_VAL_STD_THRESHOLD}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
        
        print(f"{epoch+1:<8d} {avg_train_loss:<12.4f} {avg_val_loss:<12.4f} {val_acc*100:<11.2f}% {val_f1:<12.4f} {val_conf_std:<12.4f}")
    
    print("-" * 80)
    print(f"Best Val F1: {best_val_f1:.4f}")
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    return model, history, best_val_f1


def evaluate_model(model, X_test, y_test):
    """Comprehensive evaluation on test set with confidence analysis"""
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION (P0.7)")
    print(f"{'='*60}")
    
    model.eval()
    
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    y_pred_list = []
    y_true_list = []
    y_prob_list = []
    y_action_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            
            # Map to actions (same logic as patchtst_agent.py)
            for prob in probs.cpu().numpy():
                if prob > 0.6:
                    y_action_list.append('BUY')
                elif prob < 0.4:
                    y_action_list.append('SELL')
                else:
                    y_action_list.append('HOLD')
            
            y_pred_list.extend(predictions.cpu().numpy())
            y_true_list.extend(y_batch.cpu().numpy())
            y_prob_list.extend(probs.cpu().numpy())
    
    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    y_action = np.array(y_action_list)
    
    # Calculate classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"             LOSS    WIN")
    print(f"Actual LOSS  {conf_matrix[0][0]:4d}   {conf_matrix[0][1]:4d}")
    print(f"       WIN   {conf_matrix[1][0]:4d}   {conf_matrix[1][1]:4d}")
    
    # P0.6 NEW: Action distribution analysis
    action_counts = pd.Series(y_action).value_counts()
    action_pct = action_counts / len(y_action)
    
    print(f"\n📊 ACTION DISTRIBUTION (P0.6):")
    for action in ['BUY', 'SELL', 'HOLD']:
        count = action_counts.get(action, 0)
        pct = action_pct.get(action, 0) * 100
        print(f"  {action:<6s}: {count:4d} ({pct:5.1f}%)")
    
    # P0.6 NEW: Confidence statistics
    conf_mean = np.mean(y_prob)
    conf_std = np.std(y_prob)
    conf_min = np.min(y_prob)
    conf_max = np.max(y_prob)
    conf_p10 = np.percentile(y_prob, 10)
    conf_p50 = np.percentile(y_prob, 50)
    conf_p90 = np.percentile(y_prob, 90)
    conf_range = conf_p90 - conf_p10
    
    print(f"\n📈 CONFIDENCE STATISTICS (P0.6):")
    print(f"  Mean:       {conf_mean:.4f}")
    print(f"  Stddev:     {conf_std:.4f}")
    print(f"  Min:        {conf_min:.4f}")
    print(f"  P10:        {conf_p10:.4f}")
    print(f"  P50:        {conf_p50:.4f}")
    print(f"  P90:        {conf_p90:.4f}")
    print(f"  Max:        {conf_max:.4f}")
    print(f"  P10-P90:    {conf_range:.4f}")
    
    # Calculate unique confidences (like P0.4 report)
    unique_confs = np.unique(np.round(y_prob, 4))
    print(f"  Unique:     {len(unique_confs)}")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'action_distribution': action_pct.to_dict(),
        'action_counts': action_counts.to_dict(),
        'confidence': {
            'mean': conf_mean,
            'std': conf_std,
            'min': conf_min,
            'p10': conf_p10,
            'p50': conf_p50,
            'p90': conf_p90,
            'max': conf_max,
            'p10_p90_range': conf_range,
            'unique_count': len(unique_confs)
        }
    }


def sanity_checks(eval_results):
    """
    P0.6 HARD FAILS: Validate action diversity and confidence spread.
    
    FAIL conditions:
    - Any action >70% → BUY bias not fixed
    - <2 actions >10% → No diversity
    - Conf stddev <0.02 → Confidence collapse
    - P10-P90 range <0.05 → Flatlined
    """
    print(f"\n{'='*60}")
    print(f"SANITY CHECKS (P0.6 HARD FAILS)")
    print(f"{'='*60}")
    
    failures = []
    
    # CHECK 1: Action diversity
    action_dist = eval_results['action_distribution']
    max_action = max(action_dist.values())
    max_action_name = max(action_dist, key=action_dist.get)
    
    classes_above_10 = sum(1 for pct in action_dist.values() if pct > 0.10)
    
    print(f"\n1. ACTION DIVERSITY:")
    print(f"   Max action: {max_action_name} ({max_action*100:.1f}%)")
    print(f"   Threshold: ≤70%")
    
    if max_action > MAX_ACTION_PCT:
        print(f"   ❌ FAIL: {max_action_name} bias ({max_action*100:.1f}% > {MAX_ACTION_PCT*100:.0f}%)")
        failures.append(f"Action bias: {max_action_name} {max_action*100:.1f}%")
    else:
        print(f"   ✅ PASS")
    
    print(f"\n   Classes >10%: {classes_above_10}")
    print(f"   Threshold: ≥2")
    
    if classes_above_10 < MIN_CLASSES_ABOVE_10:
        print(f"   ❌ FAIL: Only {classes_above_10} classes >10%")
        failures.append(f"Low diversity: {classes_above_10} classes >10%")
    else:
        print(f"   ✅ PASS")
    
    # CHECK 2: Confidence spread
    conf_std = eval_results['confidence']['std']
    conf_range = eval_results['confidence']['p10_p90_range']
    
    print(f"\n2. CONFIDENCE SPREAD:")
    print(f"   Stddev: {conf_std:.4f}")
    print(f"   Threshold: ≥{MIN_CONFIDENCE_STD:.2f}")
    
    if conf_std < MIN_CONFIDENCE_STD:
        print(f"   ❌ FAIL: Confidence collapse (std={conf_std:.4f} < {MIN_CONFIDENCE_STD:.2f})")
        failures.append(f"Confidence collapse: std={conf_std:.4f}")
    else:
        print(f"   ✅ PASS")
    
    print(f"\n   P10-P90 range: {conf_range:.4f}")
    print(f"   Threshold: ≥{MIN_P10_P90_RANGE:.2f}")
    
    if conf_range < MIN_P10_P90_RANGE:
        print(f"   ❌ FAIL: Narrow range ({conf_range:.4f} < {MIN_P10_P90_RANGE:.2f})")
        failures.append(f"Narrow confidence: P10-P90={conf_range:.4f}")
    else:
        print(f"   ✅ PASS")
    
    # FINAL VERDICT
    print(f"\n{'='*60}")
    if failures:
        print(f"❌ SANITY CHECKS FAILED ({len(failures)} issues)")
        for i, failure in enumerate(failures, 1):
            print(f"   {i}. {failure}")
        print(f"\n🔧 RECOMMENDATION:")
        if 'Action bias' in failures[0]:
            print("   - Increase label smoothing (epsilon=0.15)")
            print("   - Verify balanced sampling worked")
            print("   - Try class weights adjustment")
        if 'Confidence collapse' in str(failures):
            print("   - Add confidence regularization loss term")
            print("   - Increase dropout (0.2)")
            print("   - Check for saturation in sigmoid")
        print(f"\n⚠️  RETRAIN REQUIRED BEFORE DEPLOYMENT")
        return False
    else:
        print(f"✅ ALL SANITY CHECKS PASSED")
        print(f"   Model ready for shadow deployment")
        return True


def gate_evaluation(eval_results):
    """
    Evaluate P0.6 model against Gates 1-2 (Gate 3-4 need production data).
    """
    print(f"\n{'='*60}")
    print(f"GATE EVALUATION (P0.6)")
    print(f"{'='*60}")
    
    action_dist = eval_results['action_distribution']
    conf_stats = eval_results['confidence']
    
    # Gate 1: Action Diversity
    max_action_pct = max(action_dist.values())
    classes_above_10 = sum(1 for pct in action_dist.values() if pct > 0.10)
    
    gate1_check1 = max_action_pct <= 0.70
    gate1_check2 = classes_above_10 >= 2
    gate1_pass = gate1_check1 and gate1_check2
    
    # Gate 2: Confidence Spread
    gate2_check1 = conf_stats['std'] >= 0.05
    gate2_check2 = conf_stats['p10_p90_range'] >= 0.12
    gate2_pass = gate2_check1 and gate2_check2
    
    print(f"\n📋 GATE RESULTS:")
    print(f"\n  Gate 1 - Action Diversity:")
    print(f"    Check 1 (max ≤70%): {gate1_check1} ({max_action_pct*100:.1f}%)")
    print(f"    Check 2 (≥2 classes >10%): {gate1_check2} ({classes_above_10} classes)")
    print(f"    Result: {'✅ PASS' if gate1_pass else '❌ FAIL'}")
    
    print(f"\n  Gate 2 - Confidence Spread:")
    print(f"    Check 1 (std ≥0.05): {gate2_check1} ({conf_stats['std']:.4f})")
    print(f"    Check 2 (p10-p90 ≥0.12): {gate2_check2} ({conf_stats['p10_p90_range']:.4f})")
    print(f"    Result: {'✅ PASS' if gate2_pass else '❌ FAIL'}")
    
    print(f"\n  Gate 3 - Agreement (shadow mode data needed):")
    print(f"    Status: ⏳ DEFERRED (need production data)")
    
    print(f"\n  Gate 4 - Calibration (outcome data needed):")
    print(f"    Status: ⏳ DEFERRED (need 1h forward returns)")
    
    gates_passed = sum([gate1_pass, gate2_pass])
    gates_total = 2  # Only 2 testable now
    
    print(f"\n{'='*60}")
    print(f"GATES PASSED: {gates_passed}/{gates_total} (testable now)")
    print(f"{'='*60}")
    
    if gates_passed == gates_total:
        print(f"✅ P0.6 MODEL READY FOR SHADOW DEPLOYMENT")
        print(f"   Next: Deploy to VPS, collect data for Gates 3-4")
    else:
        print(f"⚠️  P0.6 MODEL NOT READY")
        print(f"   Need {gates_total - gates_passed} more gate(s) to pass")
    
    return {
        'gate1': gate1_pass,
        'gate2': gate2_pass,
        'gate3': None,  # Deferred
        'gate4': None,  # Deferred
        'gates_passed': gates_passed,
        'gates_total': gates_total
    }


def main():
    """Main retraining workflow"""
    print(f"\n{'='*70}")
    print(f"PATCHTST P0.6 RETRAINING — FIX BUY BIAS + CONFIDENCE COLLAPSE")
    print(f"{'='*70}")
    print(f"Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Setup
    check_disk_space()
    set_process_priority()
    
    # Create training directory
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    train_dir = Path(f"/tmp/patchtst_retrain_p06/{timestamp}")
    train_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[TRAIN DIR] {train_dir}")
    
    # Load data (FIXED: Use correct database path with 6,000 training samples)
    db_path = "/opt/quantum/data/quantum_trader.db"
    df = load_training_data(db_path, TRAINING_WINDOW_DAYS)
    
    # Prepare features with P0.6 balanced sampling
    X, y, symbols = prepare_features(df, balance_classes=True)
    
    # Create sequences
    X_seq = create_sequences(X, y, SEQUENCE_LENGTH)
    
    # Split data (70/15/15)
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
    print(f"  Train: {len(X_train)} ({len(X_train)/n*100:.0f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/n*100:.0f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/n*100:.0f}%)")
    
    # Train model with P0.7 anti-collapse fixes
    start_time = time.time()
    model, history, best_f1 = train_model(
        X_train, y_train, X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        use_label_smoothing=False,  # P0.7: DISABLED (was pushing outputs to 0.5)
        use_class_weights=True
    )
    train_time = time.time() - start_time
    
    print(f"\nTraining completed in {train_time:.1f}s")
    
    # Evaluate
    eval_results = evaluate_model(model, X_test, y_test)
    
    # P0.7 SANITY CHECKS (HARD FAILS)
    sanity_pass = sanity_checks(eval_results)
    
    if not sanity_pass:
        print(f"\n{'='*70}")
        print(f"❌ P0.7 TRAINING FAILED SANITY CHECKS")
        print(f"{'='*70}")
        print(f"\nExiting without saving model.")
        return 1
    
    # Gate evaluation
    gate_results = gate_evaluation(eval_results)
    
    # Save model
    model_filename = f"patchtst_v{timestamp}_p0_7.pth"
    model_path = train_dir / model_filename
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'sequence_length': SEQUENCE_LENGTH,
            'num_features': NUM_FEATURES,
            'label_smoothing': LABEL_SMOOTHING,
            'variance_penalty': VARIANCE_PENALTY_WEIGHT,
            'balanced_sampling': True,
            'class_weights': True
        },
        'eval_results': eval_results,
        'gate_results': gate_results,
        'training_history': history,
        'timestamp': timestamp
    }, model_path)
    
    print(f"\n✅ Model saved: {model_path}")
    print(f"   Size: {model_path.stat().st_size / (1024**2):.1f} MB")
    
    # Generate summary report
    summary_path = train_dir / "training_summary_p0_6.md"
    with open(summary_path, 'w') as f:
        f.write(f"# PatchTST P0.6 Training Summary\n\n")
        f.write(f"**Date**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"**Model**: {model_filename}\n")
        f.write(f"**Training Time**: {train_time:.1f}s\n\n")
        
        f.write(f"## Configuration\n\n")
        f.write(f"- Window: {TRAINING_WINDOW_DAYS} days\n")
        f.write(f"- Samples: {len(X)}\n")
        f.write(f"- Epochs: {EPOCHS}\n")
        f.write(f"- Batch size: {BATCH_SIZE}\n")
        f.write(f"- Learning rate: {LEARNING_RATE}\n")
        f.write(f"- Label smoothing: {LABEL_SMOOTHING}\n")
        f.write(f"- Balanced sampling: YES\n")
        f.write(f"- Class weights: YES\n\n")
        
        f.write(f"## Action Distribution\n\n")
        for action, pct in eval_results['action_distribution'].items():
            f.write(f"- {action}: {pct*100:.1f}%\n")
        
        f.write(f"\n## Confidence Statistics\n\n")
        conf = eval_results['confidence']
        f.write(f"- Mean: {conf['mean']:.4f}\n")
        f.write(f"- Stddev: {conf['std']:.4f}\n")
        f.write(f"- P10: {conf['p10']:.4f}\n")
        f.write(f"- P50: {conf['p50']:.4f}\n")
        f.write(f"- P90: {conf['p90']:.4f}\n")
        f.write(f"- P10-P90 Range: {conf['p10_p90_range']:.4f}\n")
        f.write(f"- Unique: {conf['unique_count']}\n\n")
        
        f.write(f"## Gate Results\n\n")
        f.write(f"- Gate 1 (Action Diversity): {'PASS' if gate_results['gate1'] else 'FAIL'}\n")
        f.write(f"- Gate 2 (Confidence Spread): {'PASS' if gate_results['gate2'] else 'FAIL'}\n")
        f.write(f"- Gate 3 (Agreement): DEFERRED\n")
        f.write(f"- Gate 4 (Calibration): DEFERRED\n")
        f.write(f"- **Total**: {gate_results['gates_passed']}/{gate_results['gates_total']} (testable)\n\n")
        
        f.write(f"## Next Steps\n\n")
        if gate_results['gates_passed'] == gate_results['gates_total']:
            f.write(f"✅ Deploy to VPS in shadow mode for Gates 3-4 evaluation\n")
        else:
            f.write(f"⚠️ Retrain with adjusted hyperparameters\n")
    
    print(f"✅ Summary saved: {summary_path}")
    
    print(f"\n{'='*70}")
    print(f"P0.6 RETRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n📦 Deliverables:")
    print(f"   - Model: {model_path}")
    print(f"   - Summary: {summary_path}")
    print(f"\n🚀 Next: Deploy to VPS shadow mode")
    print(f"   1. Copy model to /opt/quantum/ai_engine/models/")
    print(f"   2. Update PATCHTST_MODEL_PATH in env")
    print(f"   3. Keep PATCHTST_SHADOW_ONLY=true")
    print(f"   4. Restart quantum-ai-engine.service")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
