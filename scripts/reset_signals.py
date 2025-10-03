"""Reset signals table safely for local development.
Drops the `signals` table if it exists and recreates it from the model in
`scripts/scheduler.py`. Run this from the repository root with the virtualenv
activated used for the project.
"""
from backend.database import engine
from sqlalchemy import text

print("Dropping signals table if exists...")
with engine.connect() as conn:
    try:
        conn.execute(text('DROP TABLE IF EXISTS signals'))
        conn.commit()
        print("Dropped signals table")
    except Exception as e:
        print("Drop failed:", e)

# Import model and recreate
from scripts.scheduler import Signal
print("Recreating signals table if missing...")
Signal.__table__.create(bind=engine, checkfirst=True)
print("Done")
