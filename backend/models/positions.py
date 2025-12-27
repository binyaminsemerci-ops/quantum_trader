"""SQLAlchemy models for portfolio position tracking."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, UniqueConstraint

from backend.database import Base


class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"
    __table_args__ = (
        UniqueConstraint("symbol", name="uq_portfolio_position_symbol"),
        {'extend_existing': True}
    )

    id = Column(Integer, primary_key=True)
    symbol = Column(String(24), nullable=False, index=True)
    quantity = Column(Float, nullable=False, default=0.0)
    notional = Column(Float, nullable=False, default=0.0)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


__all__ = ["PortfolioPosition"]
