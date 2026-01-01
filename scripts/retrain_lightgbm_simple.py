"""
Retrain LightGBM on synthetic data
"""
import sys
import sqlite3
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# Import LightGBM
try:
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except ImportError:
    print("Installing lightgbm...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "scikit-learn"])
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    
    # Define feature keys we expect
    feature_keys = ['rsi', 'ma_cross', 'volatility', 'returns_1h']
    
    for _, row in df_meta.iterrows():
        try:
            features = json.loads(row['features']) if isinstance(row['features'], str) else row['features']
            
            # Extract 4 available features
            feat_vec = [features.get(k, 0.0) for k in feature_keys]
            
            features_list.append(feat_vec)
            labels.append(1 if row['target_class'] == 'WIN' else 0)
        except Exception as e:
            # Skip malformed rows
            print(f"Skipping row: {e}")
            continue
    
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    print(f"\nExtracted features from {len(X)} samples")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Count outcomes
    win_count = int(y.sum())
    loss_count = len(y) - win_count
    print(f"WIN: {win_count}, LOSS: {loss_count}, WIN Rate: {win_count/len(y):.1%}")
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train LightGBM
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM")
    print("="*60)
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print("="*60)
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"/app/models/lightgbm_v{timestamp}.pkl"
    scaler_path = f"/app/models/lightgbm_scaler_v{timestamp}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n✅ Model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    print(f"   Val Accuracy: {acc*100:.2f}%")
    print(f"   Val F1 Score: {f1*100:.2f}%")

if __name__ == "__main__":
    main()
