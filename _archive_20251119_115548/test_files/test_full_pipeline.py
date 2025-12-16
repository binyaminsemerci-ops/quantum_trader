"""Test full pipeline fra AITradingEngine"""
import asyncio
import sys
sys.path.insert(0, '/app')

from backend.services.ai_trading_engine import AITradingEngine

async def test_full_pipeline():
    engine = AITradingEngine()
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    current_positions = {}
    
    print("🔄 Testing full AI trading pipeline...\n")
    
    signals = await engine.get_trading_signals(symbols, current_positions)
    
    print(f"[CHART] Got {len(signals)} signals:\n")
    
    for sig in signals:
        print(f"{sig['symbol']}:")
        print(f"  Action: {sig['action']}")
        print(f"  Score: {sig.get('score', 0):.3f}")
        print(f"  Confidence: {sig.get('confidence', 0):.3f}")
        print(f"  Model: {sig.get('model', 'unknown')}")
        print()

asyncio.run(test_full_pipeline())
