from typing import Iterator, Optional, Any, List
from datetime import datetime


class TradeLog:
    id: int
    symbol: str
    side: str
    qty: float
    price: float
    status: str
    reason: Optional[str]
    timestamp: Optional[datetime]

    def __init__(self, *, symbol: str = ..., side: str = ..., qty: float = ..., price: float = ..., status: str = ..., reason: Optional[str] = ..., timestamp: Optional[datetime] = ...) -> None: ...


class Settings:
    id: int
    api_key: str
    api_secret: str


class TrainingTask:
    id: int
    symbols: str
    limit: int
    status: str
    created_at: Any
    completed_at: Any
    details: Optional[str]


class Query:
    def order_by(self, *args: Any) -> 'Query': ...

    def limit(self, n: int) -> 'Query': ...

    def offset(self, n: int) -> 'Query': ...

    def all(self) -> List[Any]: ...

    def filter(self, *args: Any) -> 'Query': ...

    def first(self) -> Optional[Any]: ...


class Session:
    def add(self, obj: Any) -> None: ...

    def commit(self) -> None: ...

    def refresh(self, obj: Any) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...

    def query(self, model: Any) -> Query: ...


def create_training_task(db: Session, symbols: str, limit: int) -> TrainingTask: ...


def update_training_task(db: Session, task_id: int, status: str, details: Optional[str] = ...) -> None: ...


def get_db() -> Iterator[Session]: ...
