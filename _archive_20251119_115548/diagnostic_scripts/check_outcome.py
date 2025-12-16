from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample

db = SessionLocal()
samples = db.query(AITrainingSample).all()

print(f"Total samples: {len(samples)}")

known = [s for s in samples if s.outcome_known]
print(f"Samples with outcome_known=True: {len(known)}")

print("\nFirst 3 samples:")
for s in samples[:3]:
    print(f"  ID {s.id}: outcome_known={s.outcome_known}, outcome={s.outcome}")

db.close()
