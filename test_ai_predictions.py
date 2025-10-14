import sys
import os
sys.path.append(os.path.dirname(__file__))

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

def load_artifacts():
    """Load trained model and scaler"""
    try:
        models_dir = os.path.join(os.path.dirname(__file__), "ai_engine", "models")
        
        # Load model
        model_path = os.path.join(models_dir, "xgb_model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        return model, scaler
        
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None, None

def test_ai_predictions():
    """Test AI model predictions with sample data"""
    print("🤖 Testing AI Model Predictions...")
    
    try:
        # Load trained model and scaler
        model, scaler = load_artifacts()
        
        if model is None or scaler is None:
            print("❌ Could not load model artifacts")
            return False
            
        print(f"✅ Loaded model: {type(model).__name__}")
        
        # Create sample features (matching training data structure)
        sample_data = {
            'open': 45000.0,
            'high': 45500.0,
            'low': 44800.0,
            'close': 45200.0,
            'volume': 1200.5,
            'sma_20': 45100.0,
            'ema_12': 45150.0,
            'rsi': 65.5,
            'macd': 150.2,
            'bb_upper': 45800.0,
            'bb_lower': 44600.0,
            'sentiment_score': 0.7
        }
        
        # Convert to DataFrame and scale
        features_df = pd.DataFrame([sample_data])
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            prediction_proba = max(probabilities)
            
        # Map prediction to signal
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        signal = signal_map.get(prediction, 'UNKNOWN')
        
        print(f"📊 Sample Input: BTC at ${sample_data['close']:,.2f}")
        print(f"🎯 AI Prediction: {signal}")
        if prediction_proba:
            print(f"🎲 Confidence: {prediction_proba:.1%}")
            
        # Test with different market conditions
        test_scenarios = [
            {'name': 'Bull Market', 'close': 46000, 'rsi': 75, 'sentiment_score': 0.8},
            {'name': 'Bear Market', 'close': 44000, 'rsi': 30, 'sentiment_score': 0.3},
            {'name': 'Sideways', 'close': 45000, 'rsi': 50, 'sentiment_score': 0.5}
        ]
        
        print(f"\n📈 Testing Different Market Scenarios:")
        for scenario in test_scenarios:
            test_data = sample_data.copy()
            test_data.update(scenario)
            
            test_df = pd.DataFrame([test_data])
            test_scaled = scaler.transform(test_df)
            
            pred = model.predict(test_scaled)[0]
            signal = signal_map.get(pred, 'UNKNOWN')
            
            if hasattr(model, 'predict_proba'):
                proba = max(model.predict_proba(test_scaled)[0])
                print(f"  {scenario['name']:12} → {signal:4} ({proba:.1%})")
            else:
                print(f"  {scenario['name']:12} → {signal}")
                
    except Exception as e:
        print(f"❌ Error testing AI model: {e}")
        return False
        
    print(f"✅ AI model test completed successfully!")
    return True

if __name__ == "__main__":
    test_ai_predictions()