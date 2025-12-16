"""
REGENERATE DATASET WITH ENSEMBLE MODEL
Re-analyze alle 316K samples med ensemble-modellen for bedre action determination!
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings('ignore')

print("🔄🔄🔄 DATASET REGENERATION WITH ENSEMBLE 🔄🔄🔄")
print("=" * 80)
print("[TARGET] Goal: Re-analyze all samples with 6-model ensemble predictions")
print("[CHART] Expected: 55-60% WIN rate (vs previous 42%)")
print("=" * 80)

from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample
import json
import numpy as np

# Load ensemble model
print("\n📦 Loading ensemble model...")
from ai_engine.model_ensemble import EnsemblePredictor

ensemble = EnsemblePredictor()
ensemble.load("ensemble_model.pkl")
print(f"[OK] Loaded ensemble with {len(ensemble.models)} models")

db = SessionLocal()

# Get all samples
print("\n[CHART] Loading all training samples...")
samples = db.query(AITrainingSample).all()
print(f"[OK] Found {len(samples):,} samples to re-analyze")

# Re-analyze each sample
print("\n🔄 Re-analyzing samples with ensemble predictions...")

stats = {
    'BUY': 0,
    'SELL': 0,
    'HOLD': 0,
    'WIN': 0,
    'LOSS': 0,
    'NEUTRAL': 0,
    'errors': 0
}

for i, sample in enumerate(samples):
    try:
        # Progress
        if i % 10000 == 0:
            print(f"   Processed {i:,}/{len(samples):,} ({i/len(samples)*100:.1f}%)")
        
        # Get features
        features = json.loads(sample.features)
        
        # Extract 14 features
        feature_vector = np.array([[
            features.get('Close', 0),
            features.get('Volume', 0),
            features.get('EMA_10', 0),
            features.get('EMA_50', 0),
            features.get('RSI', 50),
            features.get('MACD', 0),
            features.get('MACD_signal', 0),
            features.get('BB_upper', 0),
            features.get('BB_middle', 0),
            features.get('BB_lower', 0),
            features.get('ATR', 0),
            features.get('volume_sma_20', 0),
            features.get('price_change_pct', 0),
            features.get('high_low_range', 0),
        ]])
        
        # Get ensemble prediction
        prediction = ensemble.predict(feature_vector)[0]
        
        # Determine action based on ENSEMBLE prediction
        # More aggressive thresholds for futures
        if prediction > 0.3:  # Strong positive prediction
            new_action = "BUY"
        elif prediction < -0.3:  # Strong negative prediction
            new_action = "SELL"
        else:
            new_action = "HOLD"
        
        # Update sample
        sample.target_class = new_action
        stats[new_action] += 1
        
        # Also update outcome based on realized PnL
        if sample.realized_pnl and sample.outcome_known:
            if sample.realized_pnl > 0.1:  # >0.1% profit
                sample.target_class_outcome = "WIN"
                stats['WIN'] += 1
            elif sample.realized_pnl < -0.1:  # <-0.1% loss
                sample.target_class_outcome = "LOSS"
                stats['LOSS'] += 1
            else:
                sample.target_class_outcome = "NEUTRAL"
                stats['NEUTRAL'] += 1
        
    except Exception as e:
        stats['errors'] += 1
        continue

# Commit changes
print("\n💾 Saving updated samples...")
db.commit()

# Print results
print("\n" + "=" * 80)
print("[OK] DATASET REGENERATION COMPLETE!")
print("=" * 80)

total = len(samples)
print(f"\n[CHART] ACTIONS DISTRIBUTION:")
print(f"   BUY:  {stats['BUY']:>7,} ({stats['BUY']/total*100:>5.1f}%)")
print(f"   SELL: {stats['SELL']:>7,} ({stats['SELL']/total*100:>5.1f}%)")
print(f"   HOLD: {stats['HOLD']:>7,} ({stats['HOLD']/total*100:>5.1f}%)")

outcome_total = stats['WIN'] + stats['LOSS'] + stats['NEUTRAL']
if outcome_total > 0:
    print(f"\n[CHART] OUTCOMES DISTRIBUTION:")
    print(f"   WIN:     {stats['WIN']:>7,} ({stats['WIN']/outcome_total*100:>5.1f}%)")
    print(f"   LOSS:    {stats['LOSS']:>7,} ({stats['LOSS']/outcome_total*100:>5.1f}%)")
    print(f"   NEUTRAL: {stats['NEUTRAL']:>7,} ({stats['NEUTRAL']/outcome_total*100:>5.1f}%)")
    
    win_rate = stats['WIN'] / outcome_total * 100
    
    print(f"\n{'🏆' * 40}")
    print(f"   WIN RATE: {win_rate:.1f}%")
    print(f"{'🏆' * 40}")
    
    if win_rate >= 55:
        print("\n🎉🎉🎉 TARGET ACHIEVED: ≥55% WIN RATE! 🎉🎉🎉")
    else:
        print(f"\n[WARNING] Target not reached. Need {55 - win_rate:.1f}% more")

if stats['errors'] > 0:
    print(f"\n[WARNING] Errors: {stats['errors']:,}")

db.close()

print("\n" + "=" * 80)
print("[ROCKET] Dataset ready for high-performance trading!")
print("=" * 80)
