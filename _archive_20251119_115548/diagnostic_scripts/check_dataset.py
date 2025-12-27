from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample
from sqlalchemy import func

db = SessionLocal()
total = db.query(AITrainingSample).count()
actions = db.query(AITrainingSample.predicted_action, func.count(AITrainingSample.id)).group_by(AITrainingSample.predicted_action).all()
outcomes = db.query(AITrainingSample.target_class, func.count(AITrainingSample.id)).group_by(AITrainingSample.target_class).all()
sources = db.query(AITrainingSample.model_version, func.count(AITrainingSample.id)).group_by(AITrainingSample.model_version).all()

profits = db.query(AITrainingSample.realized_pnl).filter(AITrainingSample.realized_pnl.isnot(None)).all()
profit_values = [p[0] for p in profits if p[0] is not None]
avg_profit = sum(profit_values) / len(profit_values) if profit_values else 0

print(f'🎉 MASSIV DATASET!')
print(f'═══════════════════')
print(f'Total samples: {total}')
print(f'')
print(f'[CHART] Actions:')
for action, cnt in actions:
    print(f'  {action}: {cnt} ({cnt/total*100:.1f}%)')
print(f'')
print(f'[CHART_UP] Outcomes:')
for outcome, cnt in outcomes:
    if outcome:
        print(f'  {outcome}: {cnt} ({cnt/total*100:.1f}%)')
print(f'')
print(f'[MONEY] Avg profit per trade: {avg_profit:.3f}%')
print(f'')
print(f'📦 Data sources:')
for source, cnt in sources:
    src_name = source if source else 'bootstrap'
    print(f'  {src_name}: {cnt}')

db.close()
