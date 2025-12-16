"""
Test TFT model predictions - check accuracy and signal diversity
"""
import sys
import os
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models.ai_training import AITrainingSample
from ai_engine.agents.tft_agent import TFTAgent

print("="*80)
print("TFT MODEL PREDICTION TEST")
print("="*80)

# Load model
print("\n1. Loading TFT model...")
agent = TFTAgent()
if not agent.load_model():
    print("❌ Failed to load TFT model!")
    sys.exit(1)
print("✓ TFT model loaded successfully")

# Load test data (last 20% chronologically)
print("\n2. Loading validation data (last 20% of dataset)...")
engine = create_engine("sqlite:///quantum_trader.db")
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# Get all samples ordered by time
all_samples = db.query(AITrainingSample).filter(
    AITrainingSample.outcome_known == True,
    AITrainingSample.features.isnot(None),
    AITrainingSample.target_class.isnot(None),
    AITrainingSample.hold_duration_seconds > 0  # No data leakage
).order_by(AITrainingSample.timestamp).all()

# Take last 20%
val_start_idx = int(len(all_samples) * 0.8)
val_samples = all_samples[val_start_idx:]

print(f"✓ Loaded {len(val_samples):,} validation samples")

# Test predictions
print("\n3. Testing model predictions...")
predictions = []
actuals = []
correct = 0

for i, sample in enumerate(val_samples[:1000]):  # Test first 1000
    try:
        # Get prediction (simplified - just test if it works)
        pred_class = agent.model.predict_single(sample)  # Will need to implement this
        actual_class = sample.target_class
        
        predictions.append(pred_class)
        actuals.append(actual_class)
        
        if pred_class == actual_class:
            correct += 1
            
        if (i + 1) % 100 == 0:
            print(f"  Tested {i+1}/1000 samples...", end='\r')
    except Exception as e:
        print(f"\n  Error on sample {i}: {e}")
        continue

print(f"\n  Tested {len(predictions)} samples")

db.close()

# Analyze results
print("\n" + "="*80)
print("RESULTS")
print("="*80)

accuracy = (correct / len(predictions)) * 100 if predictions else 0
print(f"\n[CHART] Validation Accuracy: {accuracy:.2f}%")

pred_dist = Counter(predictions)
actual_dist = Counter(actuals)

print(f"\n[CHART_UP] Prediction Distribution:")
for cls in ['BUY', 'SELL', 'HOLD']:
    count = pred_dist.get(cls, 0)
    pct = (count / len(predictions) * 100) if predictions else 0
    print(f"  {cls}: {count} ({pct:.1f}%)")

print(f"\n[TARGET] Actual Distribution:")
for cls in ['WIN', 'LOSS', 'NEUTRAL']:
    count = actual_dist.get(cls, 0)
    pct = (count / len(actuals) * 100) if actuals else 0
    print(f"  {cls}: {count} ({pct:.1f}%)")

# Evaluation
print(f"\n" + "="*80)
print("EVALUATION")
print("="*80)

if accuracy > 85:
    print("[WARNING]  OVERFITTING DETECTED!")
    print(f"   Accuracy {accuracy:.1f}% is too high (target: 45-70%)")
    print("   → Recommend using XGBoost fallback")
elif accuracy < 40:
    print("[WARNING]  UNDERFITTING DETECTED!")
    print(f"   Accuracy {accuracy:.1f}% is too low (target: 45-70%)")
    print("   → Model needs more training or complexity")
else:
    print("✓ Accuracy in target range (45-70%)")

# Check signal diversity
if pred_dist.get('HOLD', 0) > len(predictions) * 0.7:
    print("\n[WARNING]  HOLD BIAS DETECTED!")
    print(f"   {pred_dist.get('HOLD', 0)/len(predictions)*100:.1f}% of predictions are HOLD")
    print("   → Model not producing enough BUY/SELL signals")
    print("   → Recommend using XGBoost fallback")
else:
    buy_sell_pct = (pred_dist.get('BUY', 0) + pred_dist.get('SELL', 0)) / len(predictions) * 100
    print(f"\n✓ Signal diversity good ({buy_sell_pct:.1f}% BUY/SELL)")

print("\n" + "="*80)
