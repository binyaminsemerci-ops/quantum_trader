from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import Any
from datetime import datetime
import os

# Sørg for at database-mappa finnes
DB_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DB_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'trades.db')}"

# Opprett engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base ORM
Base: Any = declarative_base()  # type: ignore


# Tabeller / modeller
class TradeLog(Base):
    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)
    qty = Column(Float)
    price = Column(Float)
    status = Column(String)
    reason = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Settings(Base):
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
