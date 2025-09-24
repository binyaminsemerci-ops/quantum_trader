from typing import Iterator, Optional
from datetime import datetime

class TrainingTask:
    id: int
    symbols: str
    limit: int
    status: str
    created_at: Optional[datetime]
    completed_at: Optional[datetime]
    details: Optional[str]


def create_training_task(db: object, symbols: str, limit: int) -> TrainingTask: ...

def update_training_task(db: object, task_id: int, status: str, details: Optional[str] = ...) -> None: ...

def get_db() -> Iterator[object]: ...
