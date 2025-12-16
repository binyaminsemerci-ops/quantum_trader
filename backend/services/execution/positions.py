"""Portfolio position persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Mapping

from sqlalchemy.orm import Session

from backend.models.positions import PortfolioPosition


@dataclass(slots=True)
class PositionSnapshot:
    symbol: str
    quantity: float
    notional: float
    updated_at: datetime


class PortfolioPositionService:
    """Prime DB-backed portfolio exposures for risk and monitoring."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def all(self) -> Iterable[PortfolioPosition]:
        return (
            self._db.query(PortfolioPosition)
            .order_by(PortfolioPosition.symbol.asc())
            .all()
        )

    def snapshot(self) -> Dict[str, object]:
        items = []
        total_notional = 0.0
        for pos in self.all():
            entry = PositionSnapshot(
                symbol=pos.symbol,
                quantity=float(pos.quantity or 0.0),
                notional=float(pos.notional or 0.0),
                updated_at=pos.updated_at or datetime.now(timezone.utc),
            )
            items.append(
                {
                    "symbol": entry.symbol,
                    "quantity": entry.quantity,
                    "notional": entry.notional,
                    "updated_at": entry.updated_at.isoformat(),
                }
            )
            total_notional += abs(entry.notional)
        return {
            "positions": items,
            "total_notional": total_notional,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

    def sync_from_holdings(
        self,
        holdings: Mapping[str, float],
        prices: Mapping[str, float],
    ) -> Dict[str, object]:
        existing = {
            pos.symbol.upper(): pos for pos in self.all()
        }
        now = datetime.now(timezone.utc)
        symbols_seen = set()
        for raw_symbol, qty in holdings.items():
            symbol = raw_symbol.upper()
            quantity = float(qty)
            price = float(prices.get(symbol, 0.0))
            notional = abs(quantity) * price
            symbols_seen.add(symbol)
            instance = existing.get(symbol)
            if instance is None:
                instance = PortfolioPosition(symbol=symbol, quantity=quantity, notional=notional, updated_at=now)
                self._db.add(instance)
            else:
                instance.quantity = quantity
                instance.notional = notional
                instance.updated_at = now
        for symbol, instance in list(existing.items()):
            if symbol not in symbols_seen:
                self._db.delete(instance)
        self._db.commit()
        return self.snapshot()


__all__ = ["PortfolioPositionService", "PositionSnapshot"]
