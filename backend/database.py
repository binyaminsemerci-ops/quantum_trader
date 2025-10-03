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

    def __init__(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        timestamp: Optional[datetime] = None,
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
        reason: Optional[str] = None,
        timestamp: Optional[datetime] = None,
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
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __init__(
        self,
        *,
        symbols: str,
        limit: int,
        status: str = "pending",
        details: Optional[str] = None,
        created_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
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
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_secret = api_secret


class WatchlistEntry(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __init__(
        self, *, symbol: str, created_at: Optional[datetime] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.symbol = symbol
        if created_at is not None:
            self.created_at = created_at


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    condition = Column(
        String(32), nullable=False
    )  # e.g., 'price_above', 'price_below', 'change_pct'
    threshold = Column(Float, nullable=False)
    enabled = Column(Integer, nullable=False, default=1)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __init__(
        self,
        *,
        symbol: str,
        condition: str,
        threshold: float,
        enabled: int = 1,
        created_at: Optional[datetime] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.symbol = symbol
        self.condition = condition
        self.threshold = threshold
        self.enabled = enabled
        if created_at is not None:
            self.created_at = created_at


class AlertEvent(Base):
    __tablename__ = "alert_events"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, nullable=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    condition = Column(String(32), nullable=False)
    threshold = Column(Float, nullable=False)
    value = Column(Float, nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __init__(
        self,
        *,
        alert_id: int | None,
        symbol: str,
        condition: str,
        threshold: float,
        value: float,
        created_at: Optional[datetime] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alert_id = alert_id
        self.symbol = symbol
        self.condition = condition
        self.threshold = threshold
        self.value = value
        if created_at is not None:
            self.created_at = created_at


class ModelRegistry(Base):
    """Registered models with metadata & promotion flag.

    The active model is the single row with is_active=1. All training runs
    should insert a new row (is_active=0) with metrics snapshot; promotion
    toggles flags. Simplicity first: JSON blobs for params & metrics.
    """

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(64), nullable=False, index=True)
    tag = Column(String(128), nullable=True, index=True)
    path = Column(Text, nullable=False)  # relative or absolute path to artifact
    params_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    trained_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    is_active = Column(Integer, nullable=False, default=0, index=True)

    def __init__(
        self,
        *,
        version: str,
        path: str,
        tag: str | None = None,
        params_json: str | None = None,
        metrics_json: str | None = None,
        trained_at: datetime | None = None,
        is_active: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.version = version
        self.tag = tag
        self.path = path
        self.params_json = params_json
        self.metrics_json = metrics_json
        if trained_at is not None:
            self.trained_at = trained_at
        self.is_active = is_active


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
    "WatchlistEntry",
    "Alert",
    "AlertEvent",
    "ModelRegistry",
    "get_session",
    "get_db",
    "session_scope",
    "create_training_task",
    "update_training_task",
    "engine",
    "SessionLocal",
]
