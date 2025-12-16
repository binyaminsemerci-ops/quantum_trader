"""SQLAlchemy models for liquidity snapshots and portfolio selections."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from backend.database import Base


class LiquidityRun(Base):
    __tablename__ = "liquidity_runs"

    id = Column(Integer, primary_key=True, index=True)
    fetched_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    universe_size = Column(Integer, nullable=False)
    selection_size = Column(Integer, nullable=False)
    provider_primary = Column(String(32), nullable=True)
    status = Column(String(32), nullable=False, default="completed")
    message = Column(Text, nullable=True)

    snapshots = relationship("LiquiditySnapshot", back_populates="run", cascade="all, delete-orphan")
    allocations = relationship("PortfolioAllocation", back_populates="run", cascade="all, delete-orphan")
    executions = relationship("ExecutionJournal", back_populates="run")


class LiquiditySnapshot(Base):
    __tablename__ = "liquidity_snapshots"
    __table_args__ = (
        UniqueConstraint("run_id", "symbol", name="uq_liquidity_snapshot_symbol"),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("liquidity_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    rank = Column(Integer, nullable=False)
    symbol = Column(String(24), nullable=False, index=True)
    price = Column(Float, nullable=True)
    change_percent = Column(Float, nullable=True)
    base_volume = Column(Float, nullable=True)
    quote_volume = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)
    liquidity_score = Column(Float, nullable=False)
    momentum_score = Column(Float, nullable=True)
    aggregate_score = Column(Float, nullable=False)
    payload = Column(Text, nullable=True)

    run = relationship("LiquidityRun", back_populates="snapshots")


class PortfolioAllocation(Base):
    __tablename__ = "portfolio_allocations"
    __table_args__ = (
        UniqueConstraint("run_id", "symbol", name="uq_portfolio_allocation_symbol"),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("liquidity_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol = Column(String(24), nullable=False, index=True)
    weight = Column(Float, nullable=False)
    score = Column(Float, nullable=False)
    reason = Column(Text, nullable=True)

    run = relationship("LiquidityRun", back_populates="allocations")


class ExecutionJournal(Base):
    __tablename__ = "execution_journal"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("liquidity_runs.id", ondelete="SET NULL"), nullable=True, index=True)
    symbol = Column(String(24), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    target_weight = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    status = Column(String(16), nullable=False, default="pending")
    reason = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    executed_at = Column(DateTime, nullable=True)

    run = relationship("LiquidityRun", back_populates="executions")


__all__ = [
    "LiquidityRun",
    "LiquiditySnapshot",
    "PortfolioAllocation",
    "ExecutionJournal",
]
