# Simple test of raw XGBoost model predictions
import pickle
import numpy as np

print("=" * 70)
print("XGBoost Model Prediction Analysis")
print("=" * 70)

try:
    # Load model
    print("\n1. Loading model...")
    with open('ai_engine/models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("   ✓ Model loaded successfully")
    
    # Load scaler
    print("\n2. Loading scaler...")
    with open('ai_engine/models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("   ✓ Scaler loaded successfully")
    
    # Generate random features (simulating market data)
    print("\n3. Generating test features...")
    np.random.seed(42)
    n_samples = 20
    
    # Check how many features scaler expects
    n_features = scaler.n_features_in_
    print(f"   Model expects {n_features} features")
    
    # Simulate realistic market features (normalized around 0)
    test_features = np.random.randn(n_samples, n_features) * 0.5
    
    # Scale features
    test_features_scaled = scaler.transform(test_features)
    print(f"   ✓ Generated {n_samples} test samples with {n_features} features")
    
    # Make predictions
    print("\n4. Making predictions...")
    predictions = model.predict(test_features_scaled)
    
    print("\n" + "=" * 70)
    print("RAW PREDICTIONS")
    print("=" * 70)
    print(f"\n{'Sample':<10} {'Prediction':>12} {'Interpretation':<15}")
    print("-" * 70)
    
    buy_count = 0
    sell_count = 0
    hold_count = 0
    
    for i, pred in enumerate(predictions):
        # Apply same logic as xgb_agent.py (lines 267-285)
        if pred > 0.01:
            interpretation = "BUY"
            buy_count += 1
        elif pred < -0.01:
            interpretation = "SELL"
            sell_count += 1
        else:
            interpretation = "HOLD"
            hold_count += 1
        
        print(f"Sample {i+1:<3} {pred:>12.6f}   {interpretation:<15}")
    
    print("-" * 70)
    
    # Statistics
    print(f"\nPREDICTION STATISTICS:")
    print(f"  Min:     {predictions.min():>10.6f}")
    print(f"  Max:     {predictions.max():>10.6f}")
    print(f"  Mean:    {predictions.mean():>10.6f}")
    print(f"  Median:  {np.median(predictions):>10.6f}")
    print(f"  Std Dev: {predictions.std():>10.6f}")
    
    print(f"\nINTERPRETATION DISTRIBUTION:")
    print(f"  BUY:  {buy_count:2d} ({100*buy_count/n_samples:5.1f}%)")
    print(f"  SELL: {sell_count:2d} ({100*sell_count/n_samples:5.1f}%)")
    print(f"  HOLD: {hold_count:2d} ({100*hold_count/n_samples:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)
    print("\nCurrent thresholds in xgb_agent.py:")
    print("  BUY:  prediction > 0.01")
    print("  SELL: prediction < -0.01")
    print("  HOLD: -0.01 <= prediction <= 0.01")
    
    # Count how many predictions fall in each range
    in_hold_range = np.sum((predictions >= -0.01) & (predictions <= 0.01))
    above_buy = np.sum(predictions > 0.01)
    below_sell = np.sum(predictions < -0.01)
    
    print(f"\nWith current thresholds:")
    print(f"  {in_hold_range} predictions fall in HOLD range ({100*in_hold_range/n_samples:.1f}%)")
    print(f"  {above_buy} predictions are > 0.01 (BUY)")
    print(f"  {below_sell} predictions are < -0.01 (SELL)")
    
    # Suggest better thresholds
    if in_hold_range > n_samples * 0.8:  # More than 80% are HOLD
        print("\n[WARNING]  PROBLEM DETECTED:")
        print("   Most predictions fall in HOLD range!")
        print("   Model predictions are too small/centered near zero")
        
        print("\n💡 SOLUTIONS:")
        print("\n   Option 1: Make thresholds more sensitive")
        # Find thresholds that would give 33% BUY, 33% SELL, 33% HOLD
        sorted_preds = np.sort(predictions)
        buy_threshold = sorted_preds[int(len(sorted_preds) * 0.66)]
        sell_threshold = sorted_preds[int(len(sorted_preds) * 0.33)]
        
        print(f"      Suggested BUY threshold:  > {buy_threshold:.6f}")
        print(f"      Suggested SELL threshold: < {sell_threshold:.6f}")
        print(f"\n      In xgb_agent.py line 267-277, change:")
        print(f"        if v > {buy_threshold:.6f}:  # was 0.01")
        print(f"            return BUY")
        print(f"        if v < {sell_threshold:.6f}:  # was -0.01")
        print(f"            return SELL")
        
        print("\n   Option 2: Retrain model with real market data")
        print("      POST /ai/train with real USDC pairs and recent data")
        print("      This will create a model that learns from actual market patterns")
    else:
        print("\n✓ Thresholds seem reasonable for this model")
    
    print("\n" + "=" * 70)

except FileNotFoundError as e:
    print(f"\n❌ Error: Model file not found")
    print(f"   {e}")
    print("\n   Run training first:")
    print("   POST http://localhost:8000/ai/train")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
