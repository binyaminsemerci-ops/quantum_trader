from typing import Iterator, Optional, Any
from datetime import datetime


class Trade:
    id: Optional[int]
    symbol: Any
    side: Any
    qty: Any
    price: Any
    timestamp: Any
    def __init__(self, *, symbol: str, side: str, qty: float, price: float, timestamp: Optional[datetime] = ...) -> None: ...

class TradeLog:
    id: Optional[int]
    symbol: Any
    side: Any
    qty: Any
    price: Any
    status: Any
    reason: Optional[Any]
    timestamp: Any
    def __init__(self, *, symbol: str, side: str, qty: float, price: float, status: str, reason: Optional[str] = ..., timestamp: Optional[datetime] = ...) -> None: ...

class Candle:
    id: Optional[int]
    symbol: Any
    timestamp: Any
    open: Any
    high: Any
    low: Any
    close: Any
    volume: Any
    def __init__(self, *, symbol: str, timestamp: datetime, open: float, high: float, low: float, close: float, volume: float, ) -> None: ...

class EquityPoint:
    id: Optional[int]
    date: Any
    equity: Any
    def __init__(self, *, date: datetime, equity: float) -> None: ...

class TrainingTask:
    id: Optional[int]
    symbols: Any
    limit: Any
    status: Any
    details: Optional[Any]
    created_at: Optional[Any]
    completed_at: Optional[Any]
    def __init__(
        self,
        *,
        symbols: str,
        limit: int,
        status: str = ...,
        details: Optional[str] = ...,
        created_at: Optional[datetime] = ...,
        completed_at: Optional[datetime] = ...,
    ) -> None: ...

class Session:
    def add(self, obj): ...
    def commit(self) -> None: ...
    def refresh(self, obj): ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...
    def query(self, model): ...
    def execute(self, query): ...


def get_session() -> Iterator[Session]: ...

def get_db() -> Iterator[Session]: ...

def session_scope(): ...

def create_training_task(session: Session, symbols: str, limit: int) -> TrainingTask: ...

def update_training_task(session: Session, task_id: int, status: str, details: Optional[str] = ...) -> None: ...
