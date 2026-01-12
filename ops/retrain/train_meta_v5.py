#!/usr/bin/env python3
"""
MetaPredictorAgent v5 Training Script
-------------------------------------
Trener meta-learning modellen på ensemble-output data.
Kjøres etter at XGBoost, LightGBM, PatchTST og N-HiTS v5 er deployet.

Input features (8):
    - xgb_conf, lgbm_conf, patch_conf, nhits_conf (confidences)
    - xgb_action, lgbm_action, patch_action, nhits_action (encoded as 0/1/2)

Output:
    - 3-class classification: SELL (0), HOLD (1), BUY (2)

Training strategy:
    - Generate synthetic ensemble data (or use real ensemble logs)
    - Simple neural network (8 → 32 → 32 → 3)
    - Train for 100 epochs with early stopping
"""

import os, sys, joblib, json, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ai_engine.agents.meta_agent import MetaNet

# ========== DATA GENERATION ==========
def generate_training_data(n_samples=6000, seed=42):
    """
    Generate synthetic ensemble output data for training.
    
    In production, replace this with real ensemble logs:
    - Parse /var/log/quantum/ensemble-*.log
    - Extract actual predictions from XGBoost, LightGBM, PatchTST, N-HiTS
    - Label with actual market outcomes (+1.5% = BUY, -1.5% = SELL, else HOLD)
    
    For now: synthetic data with reasonable patterns
    """
    rng = np.random.default_rng(seed)
    
    # Generate random ensemble outputs
    # [xgb_conf, lgbm_conf, patch_conf, nhits_conf, xgb_act, lgbm_act, patch_act, nhits_act]
    X = np.zeros((n_samples, 8))
    
    # Confidences: 0.3 to 0.95
    X[:, 0:4] = rng.uniform(0.3, 0.95, (n_samples, 4))
    
    # Actions: 0 (SELL), 1 (HOLD), 2 (BUY)
    X[:, 4:8] = rng.integers(0, 3, (n_samples, 4))
    
    # Generate labels based on ensemble consensus
    y = []
    for row in X:
        confs = row[:4]
        acts = row[4:8]
        
        # Calculate weighted average action
        weighted_action = np.sum(confs * acts) / np.sum(confs)
        
        # Convert to class
        if weighted_action > 1.6:  # Strong BUY signal
            y.append(2)
        elif weighted_action < 0.8:  # Strong SELL signal
            y.append(0)
        else:  # HOLD
            y.append(1)
    
    return X, np.array(y)

# ========== MODEL TRAINING ==========
def train_meta_v5():
    """Train MetaPredictorAgent v5"""
    
    print("=" * 60)
    print("MetaPredictorAgent v5 Training")
    print("=" * 60)
    
    # Generate training data
    print("\n[DATA] Generating synthetic ensemble data...")
    X, y = generate_training_data(n_samples=6000, seed=42)
    print(f"[DATA] Total samples: {len(X)}")
    print(f"[DATA] Class distribution: SELL={np.sum(y==0)}, HOLD={np.sum(y==1)}, BUY={np.sum(y==2)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.176, random_state=42, stratify=y_train
    )
    
    print(f"[SPLIT] Train: {len(X_train)}, Valid: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    model = MetaNet(input_dim=8, hidden=32, output_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training loop
    print("\n[TRAIN] Training MetaPredictor...")
    best_val_acc = 0
    patience = 20
    no_improve = 0
    
    for epoch in range(200):
        # Train
        model.train()
        X_batch = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_batch = torch.tensor(y_train, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        # Validate
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(torch.tensor(X_val_scaled, dtype=torch.float32))
                val_pred = torch.argmax(val_outputs, dim=1).numpy()
                val_acc = accuracy_score(y_val, val_pred)
            
            print(f"[EPOCH {epoch:3d}] Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[TRAIN] Early stopping at epoch {epoch}")
                    break
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test_scaled, dtype=torch.float32))
        test_pred = torch.argmax(test_outputs, dim=1).numpy()
    
    test_acc = accuracy_score(y_test, test_pred)
    print(f"\n📊 Test Accuracy: {test_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=["SELL", "HOLD", "BUY"]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))
    
    # Check class variety
    unique_preds = np.unique(test_pred)
    print(f"\n[CHECK] Unique predictions: {unique_preds}")
    if len(unique_preds) == 3:
        print("[SUCCESS] All 3 classes predicted ✓")
    else:
        print(f"[WARNING] Only {len(unique_preds)} classes predicted")
    
    # Save model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("ai_engine/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"meta_v{timestamp}_v5.pth"
    scaler_path = model_dir / f"meta_v{timestamp}_v5_scaler.pkl"
    meta_path = model_dir / f"meta_v{timestamp}_v5_meta.json"
    
    # Save model state
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved: {scaler_path}")
    
    # Save metadata
    metadata = {
        "version": "v5",
        "timestamp": timestamp,
        "input_dim": 8,
        "hidden_dim": 32,
        "output_dim": 3,
        "train_samples": len(X_train),
        "test_accuracy": float(test_acc),
        "best_val_accuracy": float(best_val_acc),
        "features": [
            "xgb_conf", "lgbm_conf", "patch_conf", "nhits_conf",
            "xgb_action", "lgbm_action", "patch_action", "nhits_action"
        ],
        "classes": ["SELL", "HOLD", "BUY"]
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata saved: {meta_path}")
    
    print("\n" + "=" * 60)
    print("🎉 MetaPredictorAgent v5 Training Complete!")
    print("=" * 60)
    print("\n📦 Ready for deployment:")
    print(f"  1. Copy models: sudo cp {model_path} {scaler_path} {meta_path} /opt/quantum/ai_engine/models/")
    print(f"  2. Set ownership: sudo chown qt:qt /opt/quantum/ai_engine/models/meta_v*_v5*")
    print(f"  3. Restart service: sudo systemctl restart quantum-ai-engine.service")
    print(f"  4. Check logs: journalctl -u quantum-ai-engine.service -f | grep Meta-Agent")

if __name__ == "__main__":
    train_meta_v5()
