# Test XGBoost model predictions
import sys
sys.path.insert(0, '.')

import asyncio
import pickle
import numpy as np
from ai_engine.agents.xgb_agent import XGBAgent

async def test_model():
    print("Loading XGBoost model...")
    agent = XGBAgent(use_advanced_features=True)
    
    # Test symbols
    test_symbols = ['BTCUSDC', 'ETHUSDC', 'SOLUSDC', 'BNBUSDC', 'ADAUSDC']
    
    print(f"\nTesting predictions for {len(test_symbols)} symbols...")
    print("=" * 70)
    
    try:
        # Get predictions
        results = await agent.scan_top_by_volume_from_api(
            test_symbols, 
            top_n=len(test_symbols),
            limit=100
        )
        
        print(f"\n{'Symbol':<12} {'Action':<6} {'Score':>8} {'Confidence':>10} {'Model':<12}")
        print("-" * 70)
        
        buy_count = 0
        sell_count = 0
        hold_count = 0
        
        for symbol, pred in results.items():
            action = pred.get('action', 'UNKNOWN')
            score = pred.get('score', 0.0)
            confidence = pred.get('confidence', 0.0)
            model = pred.get('model', 'unknown')
            
            print(f"{symbol:<12} {action:<6} {score:>8.4f} {confidence:>10.4f} {model:<12}")
            
            if action == 'BUY':
                buy_count += 1
            elif action == 'SELL':
                sell_count += 1
            else:
                hold_count += 1
        
        print("-" * 70)
        print(f"\nSummary:")
        print(f"  BUY:  {buy_count} ({100*buy_count/len(test_symbols):.1f}%)")
        print(f"  SELL: {sell_count} ({100*sell_count/len(test_symbols):.1f}%)")
        print(f"  HOLD: {hold_count} ({100*hold_count/len(test_symbols):.1f}%)")
        
        # Load model directly to inspect raw predictions
        print("\n" + "=" * 70)
        print("Raw model output inspection:")
        print("=" * 70)
        
        try:
            with open('ai_engine/models/xgb_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Generate some dummy features
            dummy_features = np.random.randn(5, 100)  # 5 samples, 100 features
            
            raw_preds = model.predict(dummy_features)
            print(f"\nRaw predictions (5 random samples):")
            for i, pred in enumerate(raw_preds):
                print(f"  Sample {i+1}: {pred:.6f}")
            
            print(f"\nPrediction range: [{raw_preds.min():.6f}, {raw_preds.max():.6f}]")
            print(f"Prediction mean: {raw_preds.mean():.6f}")
            print(f"Prediction std: {raw_preds.std():.6f}")
            
            print("\nThresholds in xgb_agent.py:")
            print("  BUY:  prediction > 0.01")
            print("  SELL: prediction < -0.01")
            print("  HOLD: -0.01 <= prediction <= 0.01")
            
            if abs(raw_preds.mean()) < 0.01:
                print("\n[WARNING]  PROBLEM DETECTED:")
                print("   Model predictions are too small (near zero)")
                print("   This means the model is not confident about anything")
                print("\nSOLUTION:")
                print("   1. Retrain model with real market data")
                print("   2. Or adjust thresholds in xgb_agent.py (lines 267-285)")
                print("      Change: v > 0.01 -> v > 0.001 (more sensitive)")
                print("              v < -0.01 -> v < -0.001")
            
        except Exception as e:
            print(f"Could not load model directly: {e}")
        
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_model())
