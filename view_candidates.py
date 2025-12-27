"""View and promote candidate strategies"""
from backend.database import SessionLocal
from sqlalchemy import text

session = SessionLocal()

# Get CANDIDATE strategies
result = session.execute(text("""
    SELECT strategy_id, name, min_confidence, regime_filter 
    FROM sg_strategies 
    WHERE status = 'CANDIDATE' 
    ORDER BY min_confidence DESC 
    LIMIT 10
""")).fetchall()

print("\n" + "="*80)
print("Top 10 CANDIDATE Strategies (Ready for Promotion)")
print("="*80 + "\n")

for i, (strategy_id, name, confidence, regime) in enumerate(result, 1):
    print(f"{i:2}. ID: {strategy_id[:40]:40} | Conf: {confidence:.2f} | Regime: {regime or 'ALL'}")

print("\n" + "="*80)
print(f"Total CANDIDATE strategies: {len(result)}")
print("="*80)

# Recommend promoting the best ones
if result:
    top_5_ids = [r[0] for r in result[:5]]
    
    print("\n[RECOMMENDATION] Promote top 5 strategies to LIVE:")
    for i, strategy_id in enumerate(top_5_ids, 1):
        print(f"  {i}. {strategy_id}")
    
    print("\nTo promote these strategies:")
    print("  python promote_strategies.py")

session.close()
