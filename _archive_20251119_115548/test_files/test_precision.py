#!/usr/bin/env python3
"""Test AI model predictions directly."""
import asyncio
import sys
sys.path.insert(0, '.')
from ai_engine.agents.xgb_agent import make_default_agent

async def test():
    agent = make_default_agent()
    symbols = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "DOGEUSDC", "XRPUSDC"]
    
    print("\n=== Testing AI Model Predictions ===\n")
    results = await agent.scan_top_by_volume_from_api(symbols, top_n=5, limit=100)
    
    for sym, pred in results.items():
        action = pred.get('action', 'HOLD')
        score = pred.get('score', 0.0)
        conf = pred.get('confidence', 0.0)
        model = pred.get('model', 'unknown')
        print(f"{sym:12} {action:5} score={score:.4f} conf={conf:.4f} model={model}")

asyncio.run(test())
