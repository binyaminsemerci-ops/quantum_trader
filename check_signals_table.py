from backend.database import SessionLocal
from sqlalchemy import text, inspect

db = SessionLocal()

# Check if ai_signals table exists
inspector = inspect(db.bind)
tables = inspector.get_table_names()
print(f"Tables in database: {tables}")
print(f"Has 'ai_signals' table: {'ai_signals' in tables}")

if 'ai_signals' in tables:
    # Get table structure
    columns = inspector.get_columns('ai_signals')
    print(f"\nai_signals columns: {[col['name'] for col in columns]}")
    
    # Try to count recent signals
    try:
        result = db.execute(text("SELECT COUNT(*) FROM ai_signals WHERE timestamp > datetime('now', '-1 hour')"))
        count = result.scalar()
        print(f"Signals in last hour: {count}")
    except Exception as e:
        print(f"Error querying signals: {e}")
        
    # Try total count
    try:
        result = db.execute(text("SELECT COUNT(*) FROM ai_signals"))
        total = result.scalar()
        print(f"Total signals: {total}")
    except Exception as e:
        print(f"Error counting total: {e}")

db.close()
