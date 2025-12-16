from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample
import json

db = SessionLocal()
count = db.query(AITrainingSample).count()
print(f"Total samples: {count}")

recent = db.query(AITrainingSample).order_by(AITrainingSample.id.desc()).first()
if recent:
    features = json.loads(recent.features) if recent.features else {}
    print(f"\nMost recent sample (ID {recent.id}):")
    print(f"  Symbol: {recent.symbol}")
    print(f"  Timestamp: {recent.timestamp}")
    print(f"  Features count: {len(features)}")
    if features:
        print(f"  Feature keys (first 10): {list(features.keys())[:10]}")
        print(f"  [OK] HAS FEATURES!")
    else:
        print(f"  [WARNING] NO FEATURES")

db.close()
