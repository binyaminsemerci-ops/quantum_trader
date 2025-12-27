#!/usr/bin/env python3
"""
Custom AI Training Script - Tune hyperparameters and experiment
"""

import asyncio
from train_ai import MODEL_DIR, SimpleScaler, _is_stub_xgb, collect_training_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


async def train_with_custom_params():
    """Train model with custom hyperparameters"""
    
    print("🤖 Samler training data...")
    X, y = await collect_training_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize
    scaler = SimpleScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[CHART] Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"[CHART] Class distribution - BUY: {sum(y_train)}, SELL: {len(y_train) - sum(y_train)}")
    
    # Try different models
    models = {}
    
    # XGBoost - Best for crypto trading
    try:
        from xgboost import XGBClassifier
        if _is_stub_xgb(XGBClassifier):
            raise ImportError("Native XGBoost missing")
        print("\n🔧 Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200,      # Flere trees = bedre, men tregere
            max_depth=8,           # Dypere trees = mer komplekse mønstre
            learning_rate=0.05,    # Lavere = mer forsiktig læring
            subsample=0.8,         # 80% av data per tree
            colsample_bytree=0.8,  # 80% av features per tree
            random_state=42
        )
        xgb.fit(X_train_scaled, y_train)
        y_pred = xgb.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"[OK] XGBoost accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred, target_names=['SELL', 'BUY']))
        models['xgboost'] = (xgb, accuracy)
    except ImportError:
        print("[WARNING]  XGBoost ikke tilgjengelig")
    
    # LightGBM - Raskere alternativ
    try:
        from lightgbm import LGBMClassifier
        print("\n🔧 Training LightGBM...")
        lgbm = LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        lgbm.fit(X_train_scaled, y_train)
        y_pred = lgbm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"[OK] LightGBM accuracy: {accuracy:.3f}")
        models['lightgbm'] = (lgbm, accuracy)
    except ImportError:
        print("[WARNING]  LightGBM ikke tilgjengelig")
    
    # Random Forest - Backup
    try:
        from sklearn.ensemble import RandomForestClassifier
        print("\n🔧 Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        y_pred = rf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"[OK] Random Forest accuracy: {accuracy:.3f}")
        models['random_forest'] = (rf, accuracy)
    except ImportError:
        pass
    
    # Velg beste modell
    if models:
        best_name = max(models, key=lambda k: models[k][1])
        best_model, best_accuracy = models[best_name]
        
        print(f"\n🏆 Beste modell: {best_name} (accuracy: {best_accuracy:.3f})")
        
        # Lagre
        model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"💾 Modell lagret: {model_path}")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"💾 Scaler lagret: {scaler_path}")
        
        print("\n[OK] Training ferdig! Restart backend for å bruke ny modell.")
        return best_model, scaler
    else:
        print("❌ Ingen modeller kunne trenes")
        return None, None


if __name__ == "__main__":
    asyncio.run(train_with_custom_params())
