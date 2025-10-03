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
from typing import Iterator, Any

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
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Resolve default SQLite path under backend/data/
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_SQLITE_PATH = os.path.join(DATA_DIR, "trades.db")

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_URL = os.environ.get(
    "QUANTUM_TRADER_DATABASE_URL",
    f"sqlite:///{DEFAULT_SQLITE_PATH}",
)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# SQLAlchemy Base - annotate as Any for static type checkers to avoid
# "Variable ... is not valid as a type" complaints while preserving runtime
# behavior.
Base: Any = declarative_base()


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    def __init__(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        timestamp: datetime | None = None,
        **kwargs,
    ) -> None:
        # Provide an explicit constructor for static type checkers. SQLAlchemy
        # will still populate columns as usual and these assignments are
        # lightweight and preserve runtime behaviour.
        super().__init__(**kwargs)
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.price = price
        if timestamp is not None:
            self.timestamp = timestamp


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

    def __init__(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        status: str,
        reason: str | None = None,
        timestamp: datetime | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.price = price
        self.status = status
        self.reason = reason
        if timestamp is not None:
            self.timestamp = timestamp


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

    def __init__(
        self,
        *,
        symbol: str,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.symbol = symbol
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    # New fields added to align with CI/test expectations
    tag = Column(String(64), nullable=True, index=True)
    metrics_json = Column(Text, nullable=True)
    is_active = Column(Integer, default=0, index=True)
    created_at = Column(DateTime, default=func.now())
    accuracy = Column(Float, nullable=True)
    path = Column(String(500), nullable=True)

    def __init__(
        self,
        *,
        model_name: str,
        version: str,
        tag: str | None = None,
        is_active: int = 0,
        metrics_json: str | None = None,
        accuracy: float | None = None,
        path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.version = version
        self.tag = tag
        self.is_active = is_active
        self.metrics_json = metrics_json
        self.accuracy = accuracy
        self.path = path


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    condition = Column(String(500), nullable=False)
    threshold = Column(Float, nullable=False)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())

    def __init__(
        self,
        *,
        symbol: str,
        condition: str,
        threshold: float,
        is_active: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.symbol = symbol
        self.condition = condition
        self.threshold = threshold
        self.is_active = is_active


class WatchlistEntry(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True)
    name = Column(String(100))
    category = Column(String(50))
    added_at = Column(DateTime, default=func.now())

    def __init__(
        self,
        *,
        symbol: str,
        name: str | None = None,
        category: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.symbol = symbol
        self.name = name
        self.category = category


class EquityPoint(Base):
    __tablename__ = "equity_curve"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    equity = Column(Float, nullable=False)

    def __init__(self, *, date: datetime, equity: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.date = date
        self.equity = equity


class TrainingTask(Base):
    __tablename__ = "training_tasks"

    id = Column(Integer, primary_key=True, index=True)
    symbols = Column(Text, nullable=False)
    limit = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    details = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __init__(
        self,
        *,
        symbols: str,
        limit: int,
        status: str = "pending",
        details: str | None = None,
        created_at: datetime | None = None,
        completed_at: datetime | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.symbols = symbols
        self.limit = limit
        self.status = status
        self.details = details
        if created_at is not None:
            self.created_at = created_at
        if completed_at is not None:
            self.completed_at = completed_at


class Settings(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String(255))
    api_secret = Column(String(255))

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_secret = api_secret


Base.metadata.create_all(bind=engine)


class AlertEvent(Base):
    """Simple event log for alerts (activation, trigger, disable).

    Added to satisfy CI/tests expecting `AlertEvent` export. Can be expanded later.
    """

    __tablename__ = "alert_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(Integer, index=True, nullable=False)
    event = Column(String(64), nullable=False, default="trigger")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    def __init__(self, *, alert_id: int, event: str = "trigger", **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.alert_id = alert_id
        self.event = event


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
    details: str | None = None,
) -> None:
    task: TrainingTask | None = (
        session.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    )
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
    "ModelRegistry",
    "Alert",
    "WatchlistEntry",
    "AlertEvent",
    "get_session",
    "get_db",
    "session_scope",
    "create_training_task",
    "update_training_task",
    "engine",
    "SessionLocal",
]
