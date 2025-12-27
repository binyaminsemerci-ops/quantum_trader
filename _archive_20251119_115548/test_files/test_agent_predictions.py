#!/usr/bin/env python3
"""Quick test to verify agent can generate predictions with real market data"""

import sys
import os
sys.path.insert(0, r"c:\quantum_trader")

import asyncio
from ai_engine.agents.xgb_agent import make_default_agent


async def test_agent_predictions():
    print("\n=== Testing XGBAgent Real Predictions ===")
    
    # Load agent
    agent = make_default_agent()
    print(f"✓ Agent loaded: {agent is not None}")
    print(f"✓ Has model: {agent.model is not None}")
    print(f"✓ Has scaler: {agent.scaler is not None}")
    
    # Test with small symbol set
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    print(f"\n[CHART] Fetching predictions for: {symbols}")
    
    try:
        # Use the agent's scan method with timeout
        results = await asyncio.wait_for(
            agent.scan_top_by_volume_from_api(symbols, top_n=3, limit=100),
            timeout=30.0
        )
        
        print(f"✓ Received {len(results)} predictions")
        
        for symbol, prediction in results.items():
            action = prediction.get("action", "HOLD")
            score = prediction.get("score", 0.0)
            confidence = prediction.get("confidence", 0.0)
            model = prediction.get("model", "unknown")
            
            print(f"  {symbol}: {action} (score={score:.3f}, conf={confidence:.3f}, model={model})")
        
        return len(results) > 0
        
    except asyncio.TimeoutError:
        print("✗ Agent scan timed out after 30s")
        return False
    except Exception as e:
        print(f"✗ Agent scan failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    success = await test_agent_predictions()
    
    if success:
        print("\n[OK] Agent predictions working!")
        return 0
    else:
        print("\n[WARNING]  Agent predictions failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
