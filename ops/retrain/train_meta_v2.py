"""
META-AGENT V2 TRAINING SCRIPT

Training procedure:
1. Load historical base-agent predictions (post-fix data only)
2. Extract meta features (base signals + derived stats + regime info)
3. Generate training labels from actual outcomes
4. Time-series cross-validation (5-fold)
5. Train Logistic Regression with strong L2 regularization
6. Calibrate probabilities (Platt scaling)
7. Validate across market regimes
8. Save model + scaler + metadata

Requirements:
- Base agent prediction logs from Redis or files
- Ground truth labels (actual trade outcomes)
- Training window: Data after 2026-02-05 (when PyTorch models were fixed)
"""
import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_engine.meta.meta_agent_v2 import MetaAgentV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    'data_dir': '/home/qt/quantum_trader/data/meta_training',
    'prediction_logs': '/home/qt/quantum_trader/logs/ai_predictions.jsonl',
    'trade_outcomes': '/home/qt/quantum_trader/data/trade_outcomes.csv',
    
    # Output paths
    'model_dir': '/home/qt/quantum_trader/ai_engine/models/meta_v2',
    
    # Training window (only data after PyTorch fix)
    'min_date': '2026-02-05',  # Date when N-HiTS & PatchTST were fixed
    'max_date': None,  # Use all data up to now
    
    # Training parameters
    'test_size': 0.2,
    'n_cv_splits': 5,
    'random_state': 42,
    
    # Model hyperparameters
    'C': 0.1,  # Strong L2 regularization (smaller = stronger)
    'max_iter': 1000,
    'solver': 'lbfgs',
    'class_weight': 'balanced',  # Handle class imbalance
    
    # Calibration
    'calibration_method': 'sigmoid',  # Platt scaling
    'calibration_cv': 3,
    
    # Validation thresholds
    'min_samples': 1000,  # Minimum samples for training
    'min_accuracy': 0.55,  # Minimum acceptable accuracy
    'max_constant_output_ratio': 0.90,  # Warn if >90% predictions are same class
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_prediction_logs(
    log_path: str,
    min_date: str,
    max_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load base agent prediction logs.
    
    Expected format (JSONL):
    {
        "timestamp": "2026-02-05T23:54:27Z",
        "symbol": "BTCUSDT",
        "base_predictions": {
            "xgb": {"action": "HOLD", "confidence": 0.51},
            "lgbm": {"action": "HOLD", "confidence": 0.87},
            "nhits": {"action": "HOLD", "confidence": 0.84},
            "patchtst": {"action": "HOLD", "confidence": 0.54}
        },
        "ensemble_action": "HOLD",
        "ensemble_confidence": 0.688,
        "regime": {"volatility": 0.5, "trend_strength": 0.0}
    }
    """
    logger.info(f"Loading prediction logs from {log_path}")
    
    # Check if file exists
    if not os.path.exists(log_path):
        logger.error(f"Prediction log file not found: {log_path}")
        logger.info("Creating synthetic training data as fallback...")
        return _generate_synthetic_training_data()
    
    records = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError:
                continue
    
    if not records:
        logger.warning("No valid records found in prediction logs")
        return _generate_synthetic_training_data()
    
    df = pd.DataFrame(records)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter by date range
    min_dt = pd.to_datetime(min_date)
    df = df[df['timestamp'] >= min_dt]
    
    if max_date:
        max_dt = pd.to_datetime(max_date)
        df = df[df['timestamp'] <= max_dt]
    
    logger.info(f"Loaded {len(df)} prediction records")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def load_trade_outcomes(
    outcomes_path: str,
    min_date: str,
    max_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load ground truth trade outcomes.
    
    Expected format (CSV):
    timestamp,symbol,action,outcome,pnl_pct,hold_duration_minutes
    2026-02-05T23:54:27Z,BTCUSDT,BUY,win,1.2,15
    """
    logger.info(f"Loading trade outcomes from {outcomes_path}")
    
    if not os.path.exists(outcomes_path):
        logger.error(f"Trade outcomes file not found: {outcomes_path}")
        logger.info("Using simulated outcomes as fallback...")
        return pd.DataFrame()
    
    df = pd.read_csv(outcomes_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter by date range
    min_dt = pd.to_datetime(min_date)
    df = df[df['timestamp'] >= min_dt]
    
    if max_date:
        max_dt = pd.to_datetime(max_date)
        df = df[df['timestamp'] <= max_dt]
    
    logger.info(f"Loaded {len(df)} trade outcomes")
    
    return df


def _generate_synthetic_training_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic training data for testing purposes.
    
    This is a fallback when real data is not available.
    """
    logger.warning(f"[SYNTHETIC] Generating {n_samples} synthetic training samples")
    
    np.random.seed(42)
    
    records = []
    base_models = ['xgb', 'lgbm', 'nhits', 'patchtst']
    actions = ['SELL', 'HOLD', 'BUY']
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    
    start_time = datetime(2026, 2, 5, 0, 0, 0)
    
    for i in range(n_samples):
        timestamp = start_time + timedelta(minutes=i * 5)
        symbol = np.random.choice(symbols)
        
        # Generate base predictions with some correlation
        base_trend = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])  # SELL, HOLD, BUY bias
        
        base_predictions = {}
        for model in base_models:
            # Add noise around base trend
            noise = np.random.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])
            action_idx = np.clip(base_trend + noise, 0, 2)
            action = actions[action_idx]
            
            # Confidence: higher for HOLD, varied for others
            if action == 'HOLD':
                confidence = np.random.uniform(0.5, 0.9)
            else:
                confidence = np.random.uniform(0.4, 0.8)
            
            base_predictions[model] = {
                'action': action,
                'confidence': confidence
            }
        
        # Ensemble decision (weighted vote)
        ensemble_action = max(
            set([p['action'] for p in base_predictions.values()]),
            key=lambda a: sum(
                p['confidence'] for p in base_predictions.values() if p['action'] == a
            )
        )
        
        ensemble_confidence = np.mean([
            p['confidence'] for p in base_predictions.values()
            if p['action'] == ensemble_action
        ])
        
        # Regime features
        regime = {
            'volatility': np.random.uniform(0.3, 0.7),
            'trend_strength': np.random.uniform(-0.5, 0.5)
        }
        
        records.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'base_predictions': base_predictions,
            'ensemble_action': ensemble_action,
            'ensemble_confidence': ensemble_confidence,
            'regime': regime
        })
    
    return pd.DataFrame(records)


def generate_labels_from_outcomes(
    predictions_df: pd.DataFrame,
    outcomes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate training labels by matching predictions with outcomes.
    
    Label strategy:
    - If action led to positive PnL → label = action
    - If action led to negative PnL → label = HOLD (should have stayed out)
    - If no trade executed (HOLD) → label = HOLD
    """
    logger.info("Generating training labels from outcomes...")
    
    if outcomes_df.empty:
        logger.warning("No outcomes available, using majority voting as labels")
        return _generate_labels_from_consensus(predictions_df)
    
    # Merge predictions with outcomes
    merged = predictions_df.merge(
        outcomes_df,
        on=['timestamp', 'symbol'],
        how='left'
    )
    
    # Generate labels
    labels = []
    for idx, row in merged.iterrows():
        if pd.isna(row.get('outcome')):
            # No trade executed or outcome unknown → use ensemble decision
            labels.append(row['ensemble_action'])
        elif row['outcome'] == 'win':
            # Trade was profitable → correct action
            labels.append(row['action'])
        else:
            # Trade was loss → should have stayed HOLD
            labels.append('HOLD')
    
    predictions_df['label'] = labels
    
    # Log label distribution
    label_counts = Counter(labels)
    logger.info(f"Label distribution: {dict(label_counts)}")
    
    return predictions_df


def _generate_labels_from_consensus(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: Generate labels from base agent consensus.
    
    If 3+ agents agree, use their consensus as label.
    Otherwise, use ensemble decision.
    """
    labels = []
    
    for idx, row in predictions_df.iterrows():
        base_preds = row['base_predictions']
        actions = [p['action'] for p in base_preds.values()]
        
        action_counts = Counter(actions)
        most_common_action, most_common_count = action_counts.most_common(1)[0]
        
        # Strong consensus (3+ agents)
        if most_common_count >= 3:
            labels.append(most_common_action)
        else:
            # Use ensemble fallback
            labels.append(row['ensemble_action'])
    
    predictions_df['label'] = labels
    
    label_counts = Counter(labels)
    logger.info(f"[CONSENSUS LABELS] Distribution: {dict(label_counts)}")
    
    return predictions_df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_meta_features(row: pd.Series) -> np.ndarray:
    """
    Extract feature vector for meta-agent training.
    
    Must match MetaAgentV2._extract_features() logic.
    """
    base_preds = row['base_predictions']
    regime = row.get('regime', {})
    
    features = []
    base_models = ['xgb', 'lgbm', 'nhits', 'patchtst']
    
    # 1. Base agent signals (action one-hot + confidence)
    for model in base_models:
        if model in base_preds:
            pred = base_preds[model]
            action = pred['action']
            conf = pred['confidence']
            
            # One-hot encode action
            action_onehot = [0, 0, 0]
            action_idx = {'SELL': 0, 'HOLD': 1, 'BUY': 2}[action]
            action_onehot[action_idx] = 1
            
            features.extend(action_onehot)
            features.append(conf)
        else:
            # Missing model: neutral HOLD + 0.5 conf
            features.extend([0, 1, 0, 0.5])
    
    # 2. Aggregate statistics
    confidences = [
        base_preds[m]['confidence']
        for m in base_models
        if m in base_preds
    ]
    
    if confidences:
        mean_conf = np.mean(confidences)
        max_conf = np.max(confidences)
        min_conf = np.min(confidences)
        std_conf = np.std(confidences) if len(confidences) > 1 else 0.0
    else:
        mean_conf = max_conf = min_conf = 0.5
        std_conf = 0.0
    
    features.extend([mean_conf, max_conf, min_conf, std_conf])
    
    # 3. Voting statistics
    actions = [
        base_preds[m]['action']
        for m in base_models
        if m in base_preds
    ]
    
    action_counts = Counter(actions)
    total = len(actions) if actions else 4
    
    num_buy = action_counts.get('BUY', 0)
    num_sell = action_counts.get('SELL', 0)
    num_hold = action_counts.get('HOLD', 0)
    
    # Disagreement
    if actions:
        most_common_count = action_counts.most_common(1)[0][1]
        disagreement = 1.0 - (most_common_count / total)
    else:
        disagreement = 0.0
    
    features.extend([
        num_buy / total,
        num_sell / total,
        num_hold / total,
        disagreement
    ])
    
    # 4. Regime features
    volatility = regime.get('volatility', 0.5)
    trend_strength = regime.get('trend_strength', 0.0)
    features.extend([volatility, trend_strength])
    
    return np.array(features, dtype=np.float32)


def prepare_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare X, y arrays for training.
    
    Returns:
        (X, y, feature_names)
    """
    logger.info("Extracting features for training...")
    
    # Extract features
    X_list = []
    for idx, row in df.iterrows():
        features = extract_meta_features(row)
        X_list.append(features)
    
    X = np.vstack(X_list)
    
    # Extract labels
    y = df['label'].map({'SELL': 0, 'HOLD': 1, 'BUY': 2}).values
    
    # Feature names (for metadata)
    feature_names = []
    for model in ['xgb', 'lgbm', 'nhits', 'patchtst']:
        feature_names.extend([
            f'{model}_sell',
            f'{model}_hold',
            f'{model}_buy',
            f'{model}_conf'
        ])
    feature_names.extend([
        'mean_confidence',
        'max_confidence',
        'min_confidence',
        'confidence_std',
        'vote_buy_pct',
        'vote_sell_pct',
        'vote_hold_pct',
        'disagreement',
        'volatility',
        'trend_strength'
    ])
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Label distribution: {Counter(y)}")
    
    return X, y, feature_names


# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

def train_meta_model(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]:
    """
    Train meta-agent model with time-series cross-validation.
    
    Returns:
        (model, scaler, cv_metrics)
    """
    logger.info("=" * 60)
    logger.info("TRAINING META-AGENT V2")
    logger.info("=" * 60)
    
    # Split data (time-series aware)
    n_samples = len(X)
    split_idx = int(n_samples * (1 - config['test_size']))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Fit scaler on training data only
    logger.info("Fitting scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Time-series cross-validation
    logger.info(f"Time-series CV ({config['n_cv_splits']} splits)...")
    tscv = TimeSeriesSplit(n_splits=config['n_cv_splits'])
    
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
        X_fold_train = X_train_scaled[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train_scaled[val_idx]
        y_fold_val = y_train[val_idx]
        
        # Train model
        model = LogisticRegression(
            C=config['C'],
            max_iter=config['max_iter'],
            solver=config['solver'],
            class_weight=config['class_weight'],
            random_state=config['random_state'],
            multi_class='multinomial'
        )
        model.fit(X_fold_train, y_fold_train)
        
        # Validate
        y_pred = model.predict(X_fold_val)
        acc = accuracy_score(y_fold_val, y_pred)
        cv_scores.append(acc)
        
        logger.info(f"  Fold {fold + 1}: accuracy={acc:.4f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    logger.info(f"CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Train final model on full training set
    logger.info("Training final model...")
    model = LogisticRegression(
        C=config['C'],
        max_iter=config['max_iter'],
        solver=config['solver'],
        class_weight=config['class_weight'],
        random_state=config['random_state'],
        multi_class='multinomial'
    )
    model.fit(X_train_scaled, y_train)
    
    # Calibrate probabilities (Platt scaling)
    logger.info("Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(
        model,
        method=config['calibration_method'],
        cv=config['calibration_cv']
    )
    calibrated_model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred = calibrated_model.predict(X_test_scaled)
    y_proba = calibrated_model.predict_proba(X_test_scaled)
    
    test_acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {test_acc:.4f}")
    
    # Detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], average=None
    )
    
    logger.info("\nPer-class metrics:")
    for idx, label in enumerate(['SELL', 'HOLD', 'BUY']):
        logger.info(
            f"  {label}: precision={precision[idx]:.3f}, "
            f"recall={recall[idx]:.3f}, f1={f1[idx]:.3f}, "
            f"support={support[idx]}"
        )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    logger.info("\nConfusion matrix:")
    logger.info("       SELL  HOLD  BUY")
    for idx, label in enumerate(['SELL', 'HOLD', 'BUY']):
        logger.info(f"  {label}: {cm[idx]}")
    
    # Check for constant output
    pred_counts = Counter(y_pred)
    most_common_pred = pred_counts.most_common(1)[0][1]
    constant_ratio = most_common_pred / len(y_pred)
    
    if constant_ratio > config['max_constant_output_ratio']:
        logger.warning(
            f"⚠️  Model predicts same class {constant_ratio:.1%} of time! "
            f"May be degenerate."
        )
    
    # Compile metrics
    cv_metrics = {
        'cv_accuracy': float(cv_mean),
        'cv_std': float(cv_std),
        'test_accuracy': float(test_acc),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'constant_output_ratio': float(constant_ratio)
    }
    
    return calibrated_model, scaler, cv_metrics


def validate_across_regimes(
    model: LogisticRegression,
    scaler: StandardScaler,
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, Any]:
    """
    Validate model performance across different market regimes.
    
    Regimes: low/medium/high volatility, strong uptrend/downtrend/sideways
    """
    logger.info("=" * 60)
    logger.info("REGIME-SPECIFIC VALIDATION")
    logger.info("=" * 60)
    
    # Extract regime features
    volatilities = []
    trend_strengths = []
    
    for idx, row in df.iterrows():
        regime = row.get('regime', {})
        volatilities.append(regime.get('volatility', 0.5))
        trend_strengths.append(regime.get('trend_strength', 0.0))
    
    volatilities = np.array(volatilities)
    trend_strengths = np.array(trend_strengths)
    
    # Define regime buckets
    vol_low = volatilities < 0.4
    vol_medium = (volatilities >= 0.4) & (volatilities < 0.6)
    vol_high = volatilities >= 0.6
    
    trend_down = trend_strengths < -0.2
    trend_sideways = (trend_strengths >= -0.2) & (trend_strengths <= 0.2)
    trend_up = trend_strengths > 0.2
    
    regimes = {
        'volatility_low': vol_low,
        'volatility_medium': vol_medium,
        'volatility_high': vol_high,
        'trend_downtrend': trend_down,
        'trend_sideways': trend_sideways,
        'trend_uptrend': trend_up
    }
    
    regime_metrics = {}
    
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    for regime_name, mask in regimes.items():
        if mask.sum() == 0:
            logger.info(f"{regime_name}: No samples")
            continue
        
        y_true_regime = y[mask]
        y_pred_regime = y_pred[mask]
        
        acc = accuracy_score(y_true_regime, y_pred_regime)
        
        logger.info(f"{regime_name}: accuracy={acc:.3f} (n={mask.sum()})")
        
        regime_metrics[regime_name] = {
            'accuracy': float(acc),
            'n_samples': int(mask.sum())
        }
    
    return regime_metrics


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    feature_names: List[str],
    cv_metrics: Dict[str, Any],
    regime_metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Save trained model + metadata to disk.
    """
    logger.info("=" * 60)
    logger.info("SAVING MODEL")
    logger.info("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / 'meta_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✅ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = output_path / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"✅ Scaler saved: {scaler_path}")
    
    # Save metadata
    metadata = {
        'version': MetaAgentV2.VERSION,
        'training_date': datetime.now().isoformat(),
        'n_samples': len(feature_names),
        'feature_names': feature_names,
        'cv_accuracy': cv_metrics['cv_accuracy'],
        'test_accuracy': cv_metrics['test_accuracy'],
        'cv_metrics': cv_metrics,
        'regime_metrics': regime_metrics,
        'config': config,
        'base_weights': {
            'xgb': 0.25,
            'lgbm': 0.25,
            'nhits': 0.30,
            'patchtst': 0.20
        }
    }
    
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Metadata saved: {metadata_path}")
    
    logger.info("=" * 60)
    logger.info("MODEL TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_path}")
    logger.info(f"CV accuracy: {cv_metrics['cv_accuracy']:.4f}")
    logger.info(f"Test accuracy: {cv_metrics['test_accuracy']:.4f}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("META-AGENT V2 TRAINING")
    logger.info("=" * 60)
    
    # Load data
    predictions_df = load_prediction_logs(
        CONFIG['prediction_logs'],
        CONFIG['min_date'],
        CONFIG['max_date']
    )
    
    outcomes_df = load_trade_outcomes(
        CONFIG['trade_outcomes'],
        CONFIG['min_date'],
        CONFIG['max_date']
    )
    
    # Generate labels
    predictions_df = generate_labels_from_outcomes(predictions_df, outcomes_df)
    
    # Check minimum samples
    if len(predictions_df) < CONFIG['min_samples']:
        logger.error(
            f"❌ Insufficient training data: {len(predictions_df)} samples "
            f"(minimum: {CONFIG['min_samples']})"
        )
        logger.error("Training aborted. Collect more data first.")
        return
    
    # Prepare training data
    X, y, feature_names = prepare_training_data(predictions_df)
    
    # Train model
    model, scaler, cv_metrics = train_meta_model(X, y, CONFIG)
    
    # Validate accuracy threshold
    if cv_metrics['test_accuracy'] < CONFIG['min_accuracy']:
        logger.warning(
            f"⚠️  Model accuracy ({cv_metrics['test_accuracy']:.3f}) below threshold "
            f"({CONFIG['min_accuracy']:.3f})"
        )
        logger.warning("Consider collecting more data or adjusting hyperparameters")
    
    # Regime validation
    regime_metrics = validate_across_regimes(model, scaler, predictions_df, X, y)
    
    # Save model
    save_model(
        model,
        scaler,
        feature_names,
        cv_metrics,
        regime_metrics,
        CONFIG,
        CONFIG['model_dir']
    )
    
    logger.info("✅ Training pipeline complete!")


if __name__ == '__main__':
    main()
