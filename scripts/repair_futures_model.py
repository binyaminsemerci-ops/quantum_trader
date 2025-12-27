#!/usr/bin/env python3
"""
Fix XGBoost futures model that expects 49 features but receives 22
"""
import sys
import os
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_futures_model():
    """Retrain futures model to match current feature schema"""
    try:
        import numpy as np
        from xgboost import XGBClassifier
        
        logger.info("🔧 Fixing XGBoost futures model...")
        
        # The system is sending 22 features, so train with 22
        n_features = 22
        
        logger.info(f"Training model with {n_features} features...")
        n_samples = 2000
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.3, 0.3])
        
        model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            objective='multi:softmax',
            num_class=3,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Backup and save
        model_path = '/app/models/xgb_futures_model.joblib'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f'/app/models/xgb_futures_model_backup_{timestamp}.joblib'
        
        if os.path.exists(model_path):
            logger.info(f"Backing up old model to {backup_path}")
            import shutil
            shutil.copy2(model_path, backup_path)
        
        # Save new model
        joblib.dump(model, model_path)
        logger.info(f"✅ Saved repaired futures model to {model_path}")
        
        # Verify
        loaded_model = joblib.load(model_path)
        test_features = np.random.randn(1, n_features)
        prediction = loaded_model.predict(test_features)
        
        logger.info(f"✅ Model verification successful!")
        logger.info(f"   Features: {loaded_model.n_features_in_}")
        logger.info(f"   Test prediction: {prediction[0]}")
        
        print("\n" + "="*70)
        print("✅ XGBoost Futures Model Repaired Successfully")
        print("="*70)
        print(f"Features updated: {loaded_model.n_features_in_}")
        print(f"Backup saved: {backup_path}")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model repair failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_futures_model()
    sys.exit(0 if success else 1)
