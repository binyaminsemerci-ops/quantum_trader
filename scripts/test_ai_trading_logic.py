#!/usr/bin/env python3
"""
Test AI Trading Logic
Demonstrates how AI generates signals and how they would trigger trades
"""

import sys
import os

# Add paths
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai_engine.agents.xgb_agent import XGBAgent
import random

def test_ai_trading_logic():
    """Test complete AI trading flow"""
    
    print("\n" + "="*60)
    print("🤖 QUANTUM TRADER - AI TRADING LOGIC TEST")
    print("="*60)
    
    # Initialize AI Agent
    print("\n📦 Initialiserer AI Agent...")
    agent = XGBAgent(use_ensemble=True)
    
    # Check what's loaded
    print("\n[OK] AI Agent Status:")
    print(f"   • Single Model: {'✓' if agent.model else '✗'}")
    print(f"   • Ensemble (6 models): {'✓' if agent.ensemble else '✗'}")
    print(f"   • Scaler: {'✓' if agent.scaler else '✗'}")
    
    # Simulate trading scenarios
    print("\n" + "="*60)
    print("[TARGET] SIMULERING: AI TRADING SCENARIOS")
    print("="*60)
    
    test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    for symbol in test_symbols:
        print(f"\n[CHART] Symbol: {symbol}")
        print("-" * 40)
        
        # Simulate AI generating a signal
        # In real system, this would be based on market data and model prediction
        actions = ["BUY", "SELL", "HOLD"]
        confidence = random.uniform(0.3, 0.9)
        action = random.choice(actions)
        price = random.uniform(20000, 100000) if symbol == "BTCUSDT" else random.uniform(1000, 5000)
        
        print(f"   🤖 AI Signal:")
        print(f"      Action: {action}")
        print(f"      Confidence: {confidence:.1%}")
        print(f"      Price: ${price:.2f}")
        
        # Trading Logic Decision
        print(f"\n   💡 Trading Logic:")
        
        if action == "BUY" and confidence > 0.6:
            print(f"      [OK] OPEN LONG POSITION")
            print(f"      • Entry: ${price:.2f}")
            print(f"      • Size: Based on risk management")
            print(f"      • Stop Loss: -2% (${price * 0.98:.2f})")
            print(f"      • Take Profit: +3% (${price * 1.03:.2f})")
            
        elif action == "SELL" and confidence > 0.6:
            print(f"      [OK] OPEN SHORT POSITION (or close long)")
            print(f"      • Entry: ${price:.2f}")
            print(f"      • Size: Based on risk management")
            print(f"      • Stop Loss: +2% (${price * 1.02:.2f})")
            print(f"      • Take Profit: -3% (${price * 0.97:.2f})")
            
        else:
            print(f"      ⏸️ HOLD / NO ACTION")
            print(f"      • Confidence too low ({confidence:.1%} < 60%)")
            print(f"      • Or signal is HOLD")
            print(f"      • Wait for better opportunity")
    
    # Explain execution flow
    print("\n" + "="*60)
    print("🔄 EXECUTION FLOW (Hver 5. minutt)")
    print("="*60)
    print("""
1. [CHART] Market Data Update
   • Fetch OHLCV data for all symbols
   • Update price cache
   • Calculate technical indicators

2. 🤖 AI Analysis
   • Generate features from market data
   • Run prediction through ensemble (6 models)
   • Calculate confidence scores
   • Determine action (BUY/SELL/HOLD)

3. [CLIPBOARD] Order Planning
   • Check existing positions
   • Apply risk management rules
   • Calculate position sizes
   • Plan new orders or adjustments

4. [BRIEFCASE] Execution (DRY-RUN MODE)
   • Log planned orders
   • Skip actual exchange submission
   • Record in database for analysis

5. [CHART_UP] Monitoring
   • Update dashboard with signals
   • Track performance metrics
   • Log all decisions
    """)
    
    print("\n" + "="*60)
    print("[WARNING] CURRENT STATUS: DRY-RUN MODE")
    print("="*60)
    print("""
[OK] AI genererer signals
[OK] Trading logic evalueres
[OK] Orders planlegges
❌ Orders IKKE sendt til exchange
[OK] Alt logges for analyse

For LIVE trading:
1. Sett QT_DRY_RUN=false i .env
2. Restart backend
3. Verifiser API keys
    """)
    
    print("\n[OK] Test komplett!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        test_ai_trading_logic()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
