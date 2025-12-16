"""
Retrain AI model with updated training data
"""
import sys
import warnings
warnings.filterwarnings('ignore')

from ai_engine.agents.xgb_agent import XGBAgent

print('🤖 RETRAINING MED OPPDATERT DATA')
print('=' * 50)

agent = XGBAgent()
result = agent.train()

if result and result.get('success'):
    print(f"\n[OK] TRAINING SUCCESS!")
    print(f"   Train samples: {result.get('train_samples', 0)}")
    print(f"   Val samples: {result.get('val_samples', 0)}")
    print(f"   Accuracy: {result.get('accuracy', 0):.2%}")
    print(f"   Model version: {result.get('model_version', 'unknown')}")
    print(f"\n[ROCKET] Modellen er nå trent på 154 samples!")
else:
    print('\n❌ Training failed')
    if result:
        print(f"   Error: {result.get('error', 'Unknown error')}")
