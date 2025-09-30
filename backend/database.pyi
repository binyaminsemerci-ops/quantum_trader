from typing import Iterator, Optional
from datetime import datetime

class Trade:
    id: Optional[int]
    symbol: str
    side: str
    qty: float
    price: float
    timestamp: Optional[datetime]

class TradeLog:
    id: Optional[int]
    symbol: str
    side: str
    qty: float
    price: float
    status: str
    reason: Optional[str]
    timestamp: Optional[datetime]

class Candle:
    id: Optional[int]
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class EquityPoint:
    id: Optional[int]
    date: datetime
    equity: float

class TrainingTask:
    id: Optional[int]
    symbols: str
    limit: int
    status: str
    details: Optional[str]
    created_at: Optional[datetime]
    completed_at: Optional[datetime]

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
