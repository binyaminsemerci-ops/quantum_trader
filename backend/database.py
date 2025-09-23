from typing import Any, cast
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base


# Sør for at database-mappa finnes
DB_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DB_DIR, exist_ok=True)

# Allow overriding the database URL for tests/CI via environment variable.
# If QUANTUM_TRADER_DATABASE_URL is set, use that. Otherwise fall back to the
# default file-based sqlite DB under backend/data/trades.db.
if "QUANTUM_TRADER_DATABASE_URL" in os.environ:
    DATABASE_URL = os.environ["QUANTUM_TRADER_DATABASE_URL"]
else:
    DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'trades.db')}"

# For SQLite we need check_same_thread=False so SQLAlchemy works with FastAPI
# and pytest's test client in the same thread; for other DBs the connect_args
# can remain empty.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite:///") else {}

# Opprett engine
engine = create_engine(DATABASE_URL, connect_args=connect_args)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base ORM
# Use typing.cast to tell MyPy to treat the declarative base as Any for
# type-checking purposes. This avoids redeclaration issues while preserving
# runtime behavior of SQLAlchemy's declarative_base().
Base = cast(Any, declarative_base())


# Tabeller / modeller
class TradeLog(Base):  # type: ignore[valid-type,misc]
    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)
    qty = Column(Float)
    price = Column(Float)
    status = Column(String)
    reason = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Settings(Base):  # type: ignore[valid-type,misc]
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String)
    api_secret = Column(String)


# Opprett tabellene hvis de ikke finnes
Base.metadata.create_all(bind=engine)


# Dependency for å hente en DB-session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
