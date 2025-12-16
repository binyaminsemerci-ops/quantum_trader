"""Promote top candidate strategies to LIVE status"""
from backend.database import SessionLocal
from sqlalchemy import text
from datetime import datetime

session = SessionLocal()

# Get top 5 CANDIDATE strategies
result = session.execute(text("""
    SELECT strategy_id, name, min_confidence 
    FROM sg_strategies 
    WHERE status = 'CANDIDATE' 
    ORDER BY min_confidence DESC 
    LIMIT 5
""")).fetchall()

print("\n" + "="*80)
print("Promoting Top 5 Strategies to LIVE")
print("="*80 + "\n")

if not result:
    print("[ERROR] No CANDIDATE strategies found!")
    session.close()
    exit(1)

promoted = []
for strategy_id, name, confidence in result:
    print(f"[PROMOTING] {strategy_id} | {name[:50]} | conf={confidence:.2f}")
    
    # Update status to LIVE
    session.execute(text("""
        UPDATE sg_strategies 
        SET status = 'LIVE', updated_at = :now 
        WHERE strategy_id = :id
    """), {"id": strategy_id, "now": datetime.utcnow()})
    
    promoted.append(strategy_id)

session.commit()

print(f"\n[SUCCESS] Promoted {len(promoted)} strategies to LIVE status")

# Verify
result = session.execute(text(
    "SELECT COUNT(*) FROM sg_strategies WHERE status = 'LIVE'"
)).fetchone()

print(f"[VERIFY] Total LIVE strategies: {result[0]}")

# Show LIVE strategies
result = session.execute(text("""
    SELECT strategy_id, name, min_confidence 
    FROM sg_strategies 
    WHERE status = 'LIVE'
""")).fetchall()

print("\n" + "="*80)
print("Current LIVE Strategies:")
print("="*80 + "\n")

for i, (strategy_id, name, confidence) in enumerate(result, 1):
    print(f"{i}. {strategy_id:30} | conf={confidence:.2f}")

print("\n" + "="*80)
print("[NEXT STEP] Test the Strategy Runtime Engine:")
print("            python test_integration_simple.py")
print("="*80)

session.close()
