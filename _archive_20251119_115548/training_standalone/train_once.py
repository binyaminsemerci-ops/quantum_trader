"""Train model once with new bootstrap data"""
import asyncio
import sys
sys.path.insert(0, '/app')

from backend.services.ai_trading_engine import AITradingEngine

async def train_once():
    engine = AITradingEngine()
    result = await engine._retrain_model(min_samples=1)
    
    if result.get('status') == 'success':
        print('[OK] TRAINING SUCCESS!')
        print(f"[CHART] Train samples: {result.get('train_samples', 0)}")
        print(f"[CHART] Val samples: {result.get('val_samples', 0)}")
        print(f"[TARGET] Train accuracy: {result.get('train_accuracy', 0)*100:.2f}%")
        print(f"[TARGET] Val accuracy: {result.get('val_accuracy', 0)*100:.2f}%")
        print(f"💾 Model version: {result.get('model_version', 'unknown')}")
    else:
        print(f"❌ Training failed: {result.get('error', 'Unknown')}")

asyncio.run(train_once())
