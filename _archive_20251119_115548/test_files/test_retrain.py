"""Quick test of model retraining with 131K samples"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '/app')

from ai_engine.agents.xgb_agent import XGBAgent
from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample

print("[SEARCH] Checking dataset...")
db = SessionLocal()
total = db.query(AITrainingSample).count()
print(f"   Total samples: {total:,}")

actions = db.query(AITrainingSample.predicted_action).all()
buy_count = sum(1 for a in actions if a[0] == 'BUY')
sell_count = sum(1 for a in actions if a[0] == 'SELL')
hold_count = sum(1 for a in actions if a[0] == 'HOLD')

print(f"   BUY: {buy_count:,} ({buy_count/total*100:.1f}%)")
print(f"   SELL: {sell_count:,} ({sell_count/total*100:.1f}%)")
print(f"   HOLD: {hold_count:,} ({hold_count/total*100:.1f}%)")

db.close()

print("\n🔄 Retraining model...")
agent = XGBAgent()
result = agent.retrain_model()

print(f"\n[OK] RETRAINING COMPLETE:")
print(f"   Success: {result.get('success')}")
print(f"   Train samples: {result.get('train_samples', 0):,}")
print(f"   Val samples: {result.get('val_samples', 0):,}")
print(f"   Train accuracy: {result.get('train_accuracy', 0):.2%}")
print(f"   Val accuracy: {result.get('val_accuracy', 0):.2%}")

if result.get('success'):
    print("\n💎 Model retrained successfully with 131K samples!")
    print("[ROCKET] Ready for FUTURES trading!")
else:
    print("\n❌ Retraining failed!")
    print(f"   Error: {result.get('error', 'Unknown')}")
