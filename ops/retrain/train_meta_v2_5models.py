#!/usr/bin/env python3
"""
MetaAgentV2 Training Script — 5 Models (XGB, LGBM, NHiTS, PatchTST, TFT)
-------------------------------------------------------------------------
Trains a Logistic Regression meta-model that combines signals from 5 base agents.

Feature vector (30 features per sample):
  - 5 base models × 4 features (3 one-hot action + 1 confidence) = 20 features
  - 4 aggregate stats: mean_conf, max_conf, min_conf, std_conf
  - 4 voting stats: vote_buy%, vote_sell%, vote_hold%, disagreement
  - 2 regime padding: volatility=0.5, trend=0.0
  Total: 30 features

Output: Saves model to ai_engine/models/meta_v2/
"""

import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Output directory
SAVE_DIR = Path(__file__).parent.parent.parent / "ai_engine" / "models" / "meta_v2"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODELS = ['xgb', 'lgbm', 'nhits', 'patchtst', 'tft']
ACTIONS = ['SELL', 'HOLD', 'BUY']

print("=" * 60)
print("MetaAgentV2 Training — 5 Models")
print(f"Output: {SAVE_DIR}")
print("=" * 60)


def encode_action(action: str) -> list:
    """One-hot encode action [SELL=0, HOLD=1, BUY=2]"""
    vec = [0, 0, 0]
    idx = {'SELL': 0, 'HOLD': 1, 'BUY': 2}.get(action, 1)
    vec[idx] = 1
    return vec


def build_feature_vector(predictions: dict) -> np.ndarray:
    """
    Build 30-feature vector from 5-model predictions.
    predictions: {model_name: {'action': str, 'confidence': float}}
    """
    features = []

    # 1. Base agent signals (5 × 4 = 20 features)
    for model in BASE_MODELS:
        if model in predictions:
            action = predictions[model].get('action', 'HOLD')
            conf = predictions[model].get('confidence', 0.5)
        else:
            action = 'HOLD'
            conf = 0.5
        features.extend(encode_action(action))
        features.append(conf)

    # 2. Aggregate stats (4 features)
    confidences = [predictions[m].get('confidence', 0.5) for m in BASE_MODELS if m in predictions]
    if confidences:
        features.extend([
            float(np.mean(confidences)),
            float(np.max(confidences)),
            float(np.min(confidences)),
            float(np.std(confidences)) if len(confidences) > 1 else 0.0
        ])
    else:
        features.extend([0.5, 0.5, 0.5, 0.0])

    # 3. Voting stats (4 features)
    actions = [predictions[m].get('action', 'HOLD') for m in BASE_MODELS if m in predictions]
    total = len(actions) if actions else 5
    from collections import Counter
    action_counts = Counter(actions)
    num_buy = action_counts.get('BUY', 0)
    num_sell = action_counts.get('SELL', 0)
    num_hold = action_counts.get('HOLD', 0)

    if actions:
        most_common_count = action_counts.most_common(1)[0][1]
        disagreement = 1.0 - (most_common_count / total)
    else:
        disagreement = 0.0

    features.extend([num_buy / total, num_sell / total, num_hold / total, disagreement])

    # 4. Regime padding (2 features)
    features.extend([0.5, 0.0])

    return np.array(features, dtype=np.float32)


def generate_training_data(n_samples: int = 12000, seed: int = 42) -> tuple:
    """
    Generate synthetic ensemble output data for 5 models.
    Simulates realistic agent behavior patterns:
    - Strong consensus → label follows consensus
    - Mixed signals → HOLD biased
    - High confidence BUY/SELL → typically correct
    """
    rng = np.random.default_rng(seed)
    X_list = []
    y_list = []

    action_options = ['SELL', 'HOLD', 'BUY']

    for _ in range(n_samples):
        # Simulate a "ground truth" market outcome
        true_outcome = rng.choice([0, 1, 2], p=[0.25, 0.50, 0.25])  # SELL/HOLD/BUY

        predictions = {}
        for model in BASE_MODELS:
            # TFT: slightly noisier (newer model)
            noise_factor = 0.35 if model == 'tft' else 0.25

            # With probability 0.7, agent predicts correctly + noise
            if rng.random() < 0.70:
                # Mostly correct with noise
                action_probs = np.ones(3) * noise_factor / 2
                action_probs[true_outcome] = 1.0 - noise_factor
                action_probs /= action_probs.sum()
                action_idx = rng.choice(3, p=action_probs)
                conf = float(rng.uniform(0.60, 0.90))
            else:
                # Random prediction
                action_idx = rng.integers(0, 3)
                conf = float(rng.uniform(0.35, 0.65))

            predictions[model] = {
                'action': action_options[action_idx],
                'confidence': conf
            }

        X_list.append(build_feature_vector(predictions))
        y_list.append(true_outcome)

    return np.array(X_list), np.array(y_list)


print("\n[DATA] Generating 12,000 synthetic training samples...")
X, y = generate_training_data(n_samples=12000)
print(f"[DATA] X shape: {X.shape}, y distribution: {dict(zip(['SELL','HOLD','BUY'], np.bincount(y)))}")

# Scale features
print("\n[SCALER] Fitting StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validation
print("\n[CV] Running 5-fold stratified cross-validation...")
base_lr = LogisticRegression(
    C=0.1,  # Strong L2 regularization (conservative meta-agent)
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs'
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(base_lr, X_scaled, y, cv=cv, scoring='accuracy')
print(f"[CV] Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train final model with calibration
print("\n[TRAIN] Training calibrated Logistic Regression...")
calibrated_model = CalibratedClassifierCV(base_lr, cv=5, method='sigmoid')
calibrated_model.fit(X_scaled, y)

# Evaluate on training set (for sanity check)
y_pred = calibrated_model.predict(X_scaled)
train_accuracy = accuracy_score(y, y_pred)
print(f"[TRAIN] Training accuracy: {train_accuracy:.4f}")
print("\n[REPORT] Classification Report:")
print(classification_report(y, y_pred, target_names=['SELL', 'HOLD', 'BUY']))

# Build feature names
feature_names = []
for model in BASE_MODELS:
    feature_names.extend([
        f'{model}_sell', f'{model}_hold', f'{model}_buy', f'{model}_conf'
    ])
feature_names.extend(['mean_conf', 'max_conf', 'min_conf', 'std_conf'])
feature_names.extend(['vote_buy_pct', 'vote_sell_pct', 'vote_hold_pct', 'disagreement'])
feature_names.extend(['volatility', 'trend_strength'])

assert len(feature_names) == X.shape[1], f"Feature name count mismatch: {len(feature_names)} vs {X.shape[1]}"

# Save model artifacts
print(f"\n[SAVE] Saving to {SAVE_DIR}...")

model_path = SAVE_DIR / "meta_model.pkl"
scaler_path = SAVE_DIR / "scaler.pkl"
metadata_path = SAVE_DIR / "metadata.json"

with open(model_path, 'wb') as f:
    pickle.dump(calibrated_model, f)
print(f"[SAVE] Model → {model_path}")

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"[SAVE] Scaler → {scaler_path}")

metadata = {
    "version": "2.1.0",
    "training_date": datetime.utcnow().isoformat() + "Z",
    "n_samples": len(X),
    "n_features": X.shape[1],
    "feature_names": feature_names,
    "base_models": BASE_MODELS,
    "cv_accuracy": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
    "train_accuracy": float(train_accuracy),
    "base_weights": {
        "xgb": 0.20,
        "lgbm": 0.20,
        "nhits": 0.25,
        "patchtst": 0.20,
        "tft": 0.15
    }
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"[SAVE] Metadata → {metadata_path}")

print(f"""
✅ MetaAgentV2 Training Complete (5 models)
   Model: {model_path.name}
   Features: {X.shape[1]}
   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
   Base models: {', '.join(BASE_MODELS)}
""")
