"""
Retrain PatchTST on synthetic data (tabular approach like N-HiTS)
"""
import sys
import torch
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import PatchTST model
sys.path.insert(0, "/app")
from ai_engine.agents.patchtst_agent import PatchTSTModel

def main():
    # Load data
    db_path = "/app/data/quantum_trader.db"
    print(f"Loading training data from {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Table is ai_training_samples, features are stored as JSON blob
    df_meta = pd.read_sql("SELECT features, feature_names, target_class FROM ai_training_samples", conn)
    conn.close()
    
    print(f"\nLoaded {len(df_meta)} training samples")
    
    # Parse JSON features
    import json
    features_list = []
    labels = []
    
    # Define feature keys we expect (database has 4, we need 8 for PatchTST)
    feature_keys = ['rsi', 'ma_cross', 'volatility', 'returns_1h']
    
    for _, row in df_meta.iterrows():
        try:
            features = json.loads(row['features']) if isinstance(row['features'], str) else row['features']
            
            # Extract 4 available features
            feat_vec = [features.get(k, 0.0) for k in feature_keys]
            
            # Pad to 8 features (repeat the 4 features for simplicity)
            feat_vec_padded = feat_vec + feat_vec  # [rsi, ma_cross, vol, ret, rsi, ma_cross, vol, ret]
            
            features_list.append(feat_vec_padded)
            labels.append(1 if row['target_class'] == 'WIN' else 0)
        except Exception as e:
            # Skip malformed rows
            print(f"Skipping row: {e}")
            continue
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    print(f"\nExtracted features from {len(X)} samples")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Count outcomes
    win_count = int(y.sum())
    loss_count = len(y) - win_count
    print(f"WIN: {win_count}, LOSS: {loss_count}, WIN Rate: {win_count/len(y):.1%}")
    
    # Initialize model (same architecture as agent)
    model = PatchTSTModel(
        input_dim=8,
        output_dim=1,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        patch_len=16,
        num_patches=8
    )
    
    print(f"\nModel initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Reshape to (batch, seq_len=128, features=8) by repeating
    # This is tabular approach (not true time series)
    X_train_seq = np.tile(X_train[:, np.newaxis, :], (1, 128, 1))
    X_test_seq = np.tile(X_test[:, np.newaxis, :], (1, 128, 1))
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    # Train
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("\n" + "="*60)
    print("TRAINING PATCHTST")
    print("="*60)
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch).squeeze()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += len(y_batch)
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Final evaluation
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            y_pred.extend(predictions.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print("="*60)
    
    # Save checkpoint
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"/app/models/patchtst_v{timestamp}.pth"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': 128,
        'hidden_size': 128,
        'num_features': 8,
        'model_class': 'PatchTSTModel',
        'val_accuracy': acc,
        'val_f1': f1
    }
    
    torch.save(checkpoint, save_path)
    print(f"\n✅ Model checkpoint saved: {save_path}")
    print(f"   Val Accuracy: {acc*100:.2f}%")
    print(f"   Val F1 Score: {f1*100:.2f}%")

if __name__ == "__main__":
    main()
