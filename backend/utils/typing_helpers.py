"""Central typing helpers and Protocols to reduce scattered type: ignore usage.

Phase 0: Minimal, safe abstractions.

Additive only; no runtime imports of heavy libs to keep import cost low.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, Sequence, TypedDict

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | "JSONDict" | list[JSONScalar] | list["JSONDict"]
class JSONDict(TypedDict, total=False):
    # Flexible catch-all mapping; tighten keys in future phases.
    pass

@runtime_checkable
class ConfigLike(Protocol):
    """Minimal interface extracted from dynamic config usages.
    Extend incrementally instead of relying on Any.
    """
    def __getattr__(self, name: str) -> Any: ...  # fallback; to be narrowed

@runtime_checkable
class ExchangeClientProto(Protocol):
    """Surface used from dynamically loaded exchange clients (binance/ccxt)."""
    def get_recent_trades(self, symbol: str, limit: int = 100) -> Any: ...

class RiskMetric(TypedDict, total=False):
    name: str
    value: float
    threshold: float
    status: str

NumberLike = int | float
VectorLike = Sequence[NumberLike]

__all__ = [
    "JSONScalar",
    "JSONValue",
    "JSONDict",
    "ConfigLike",
    "ExchangeClientProto",
    "RiskMetric",
    "NumberLike",
    "VectorLike",
]
