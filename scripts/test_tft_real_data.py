"""
Test TFT with REAL Binance data to validate predictions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from ai_engine.agents.tft_agent import TFTAgent


def test_with_real_data():
    """Test TFT with real Binance candles"""
    
    print("\n" + "="*60)
    print("[TEST_TUBE] TESTING TFT WITH REAL BINANCE DATA")
    print("="*60 + "\n")
    
    # Load training data
    data_path = "data/binance_training_data.csv"
    if not os.path.exists(data_path):
        print(f"❌ Training data not found: {data_path}")
        return
    
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"[OK] Loaded {len(df)} candles from {df['symbol'].nunique()} symbols")
    
    # Initialize agent
    agent = TFTAgent(sequence_length=120)
    if not agent.load_model():
        print("❌ Failed to load model")
        return
    
    print(f"[OK] Model loaded successfully\n")
    
    # Test on 3 symbols with real data
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    for symbol in test_symbols:
        symbol_df = df[df['symbol'] == symbol].sort_values('timestamp')
        
        if len(symbol_df) < 120:
            print(f"[WARNING] {symbol}: Not enough data ({len(symbol_df)} candles)")
            continue
        
        print(f"\n{'='*60}")
        print(f"[SEARCH] Testing {symbol}")
        print(f"{'='*60}")
        print(f"📅 Date range: {symbol_df['timestamp'].min()} → {symbol_df['timestamp'].max()}")
        print(f"[CHART] Candles: {len(symbol_df)}")
        
        # Feed 120 historical candles
        for idx, row in symbol_df.head(120).iterrows():
            features = {
                'Close': row['Close'],
                'Volume': row['Volume'],
                'EMA_10': row['Close'],  # Simplified
                'EMA_50': row['Close'],
                'RSI': 50.0,
                'MACD': 0.0,
                'MACD_signal': 0.0,
                'BB_upper': row['Close'] * 1.02,
                'BB_middle': row['Close'],
                'BB_lower': row['Close'] * 0.98,
                'ATR': (row['High'] - row['Low']),
                'volume_sma_20': row['Volume'],
                'price_change_pct': 0.0,
                'high_low_range': (row['High'] - row['Low'])
            }
            agent.add_to_history(symbol, features)
        
        # Make prediction on 120th candle
        last_row = symbol_df.iloc[119]
        last_features = {
            'Close': last_row['Close'],
            'Volume': last_row['Volume'],
            'EMA_10': last_row['Close'],
            'EMA_50': last_row['Close'],
            'RSI': 50.0,
            'MACD': 0.0,
            'MACD_signal': 0.0,
            'BB_upper': last_row['Close'] * 1.02,
            'BB_middle': last_row['Close'],
            'BB_lower': last_row['Close'] * 0.98,
            'ATR': (last_row['High'] - last_row['Low']),
            'volume_sma_20': last_row['Volume'],
            'price_change_pct': 0.0,
            'high_low_range': (last_row['High'] - last_row['Low'])
        }
        
        action, confidence, metadata = agent.predict(symbol, last_features)
        
        print(f"\n[CHART_UP] PREDICTION (on candle 120):")
        print(f"   Action: {action}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Price: ${last_row['Close']:,.2f}")
        
        print(f"\n[CHART] PROBABILITY DISTRIBUTION:")
        print(f"   BUY:  {metadata.get('buy_prob', 0):.4f}")
        print(f"   SELL: {metadata.get('sell_prob', 0):.4f}")
        print(f"   HOLD: {metadata.get('hold_prob', 0):.4f}")
        
        print(f"\n⭐ QUANTILE PREDICTIONS:")
        q10 = metadata.get('q10', 0)
        q50 = metadata.get('q50', 0)
        q90 = metadata.get('q90', 0)
        print(f"   P10 (10% worst):  {q10:.6f}")
        print(f"   P50 (median):     {q50:.6f}")
        print(f"   P90 (10% best):   {q90:.6f}")
        
        print(f"\n[TARGET] RISK/REWARD ANALYSIS:")
        upside = metadata.get('upside', 0)
        downside = metadata.get('downside', 0)
        rr = metadata.get('risk_reward_ratio', 0)
        print(f"   Upside potential:   {upside:.6f} ({upside*100:.2f}%)")
        print(f"   Downside risk:      {downside:.6f} ({downside*100:.2f}%)")
        print(f"   Risk/Reward ratio:  {rr:.2f}:1")
        
        if rr > 2.0:
            print(f"   [OK] EXCELLENT - Strong asymmetric upside!")
        elif rr > 1.5:
            print(f"   [OK] GOOD - Favorable risk/reward")
        elif rr > 0.7 and rr < 1.3:
            print(f"   [WARNING] POOR - Symmetric/uncertain")
        elif rr < 0.5:
            print(f"   ❌ BEARISH - High downside risk")
        else:
            print(f"   ℹ️ NEUTRAL")
    
    print("\n" + "="*60)
    print("[OK] REAL DATA TEST COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    test_with_real_data()
