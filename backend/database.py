"""Database utilities for Quantum Trader.

This module centralises SQLAlchemy setup so the application can run against
SQLite (default) or PostgreSQL by overriding `QUANTUM_TRADER_DATABASE_URL`.
It also exposes convenience helpers for seeding/demo scripts and background
jobs that need direct access to the session.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Resolve default SQLite path under backend/data/
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_SQLITE_PATH = os.path.join(DATA_DIR, "trades.db")

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_URL = os.environ.get(
    "QUANTUM_TRADER_DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}"
)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class TradeLog(Base):
    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    symbol = Column(String(50), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    status = Column(String(32), nullable=False)
    reason = Column(Text, nullable=True)


class Candle(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


class EquityPoint(Base):
    __tablename__ = "equity_curve"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    equity = Column(Float, nullable=False)


class TrainingTask(Base):
    __tablename__ = "training_tasks"

    id = Column(Integer, primary_key=True, index=True)
    symbols = Column(Text, nullable=False)
    limit = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)


class Settings(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String(255))
    api_secret = Column(String(255))


Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    # Backwards compatible alias used across the codebase
    yield from get_session()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_training_task(session: Session, symbols: str, limit: int) -> TrainingTask:
    task = TrainingTask(symbols=symbols, limit=limit, status="pending")
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


def update_training_task(
    session: Session,
    task_id: int,
    status: str,
    details: Optional[str] = None,
) -> None:
    task: TrainingTask | None = session.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        return
    task.status = status
    task.details = details
    if status in {"completed", "failed"}:
        task.completed_at = datetime.now(timezone.utc)
    session.add(task)
    session.commit()


__all__ = [
    "Base",
    "Trade",
    "TradeLog",
    "Candle",
    "EquityPoint",
    "TrainingTask",
    "Settings",
    "get_session",
    "get_db",
    "session_scope",
    "create_training_task",
    "update_training_task",
    "engine",
    "SessionLocal",
]
