from typing import Any, cast
import os
import json
import importlib
import logging
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

logger = logging.getLogger(__name__)

# Sør for at database-mappa finnes
DB_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DB_DIR, exist_ok=True)

# Allow overriding the database URL for tests/CI via environment variable.
# If QUANTUM_TRADER_DATABASE_URL is set, use that. Otherwise fall back to the
# default file-based sqlite DB under backend/data/trades.db.
if "QUANTUM_TRADER_DATABASE_URL" in os.environ:
    DATABASE_URL = os.environ["QUANTUM_TRADER_DATABASE_URL"]
else:
    DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'trades.db')}"

# For SQLite we need check_same_thread=False so SQLAlchemy works with FastAPI
# and pytest's test client in the same thread; for other DBs the connect_args
# can remain empty.
# BULLETPROOF: Add timeout to prevent hanging on locked database
connect_args = (
    {"check_same_thread": False, "timeout": 10} 
    if DATABASE_URL.startswith("sqlite:///") 
    else {}
)

# BULLETPROOF: Create engine with connection pooling and health checks
# - pool_pre_ping: Verify connections before using (prevents stale connections)
# - pool_recycle: Recycle connections after 1 hour (prevents connection leaks)
# - pool_size/max_overflow: Control connection pool (non-SQLite only)
try:
    if DATABASE_URL.startswith("sqlite:///"):
        engine = create_engine(
            DATABASE_URL,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    else:
        engine = create_engine(
            DATABASE_URL,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
        )
    logger.info(f"[OK] Database engine created: {DATABASE_URL.split('/')[-1]}")
except Exception as e:
    logger.critical(f"❌ CRITICAL: Failed to create database engine: {e}")
    # Re-raise to prevent system from starting with broken database
    raise

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base ORM
# Use typing.cast to tell MyPy to treat the declarative base as Any for
# type-checking purposes. This avoids redeclaration issues while preserving
# runtime behavior of SQLAlchemy's declarative_base().
Base = cast(Any, declarative_base())


# Tabeller / modeller
class TradeLog(Base):  # type: ignore[valid-type,misc]
    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    side = Column(String)
    qty = Column(Float)
    price = Column(Float)
    status = Column(String)
    reason = Column(String, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Analytics fields
    realized_pnl = Column(Float, nullable=True)
    realized_pnl_pct = Column(Float, nullable=True)
    equity_after = Column(Float, nullable=True)
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    strategy_id = Column(String, nullable=True)


class Settings(Base):  # type: ignore[valid-type,misc]
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String)
    api_secret = Column(String)


class TrainingTask(Base):  # type: ignore[valid-type,misc]
    __tablename__ = "training_tasks"

    id = Column(Integer, primary_key=True, index=True)
    symbols = Column(String, nullable=False)
    limit = Column(Integer, nullable=False)
    status = Column(String, default="pending", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    details = Column(Text, nullable=True)


class ModelTrainingRun(Base):  # type: ignore[valid-type,misc]
    __tablename__ = "model_training_runs"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, nullable=False, unique=True)
    status = Column(String, nullable=False, default="running")
    started_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    dataset_path = Column(String, nullable=True)
    symbol_count = Column(Integer, nullable=True)
    sample_count = Column(Integer, nullable=True)
    feature_count = Column(Integer, nullable=True)
    model_path = Column(String, nullable=True)
    scaler_path = Column(String, nullable=True)
    metrics = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)


# Note: Tables are now created via Alembic migrations
# Run `alembic upgrade head` to create/update the database schema


# Dependency for å hente en DB-session
# BULLETPROOF: Now includes error handling and logging
def get_db():
    """
    FastAPI dependency for database sessions.
    
    BULLETPROOF: Handles errors gracefully, logs issues, ensures cleanup.
    """
    db = None
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        logger.error(f"❌ Database session error: {e}", exc_info=True)
        if db:
            try:
                db.rollback()
            except Exception as rollback_error:
                logger.error(f"❌ Rollback failed: {rollback_error}")
        raise
    finally:
        if db:
            try:
                db.close()
            except Exception as close_error:
                logger.error(f"❌ Session close failed: {close_error}")


def create_training_task(db: Session, symbols: str, limit: int) -> TrainingTask:
    """
    Create a training task record used to track AI retraining jobs.
    
    BULLETPROOF: Handles database errors gracefully.
    """
    try:
        task = TrainingTask(symbols=symbols, limit=limit, status="pending")
        db.add(task)
        db.commit()
        db.refresh(task)
        return task
    except Exception as e:
        logger.error(f"❌ Failed to create training task: {e}", exc_info=True)
        db.rollback()
        raise


def update_training_task(
    db: Session,
    task_id: int,
    status: str,
    *,
    details: str | None = None,
) -> TrainingTask:
    """Update an existing training task status with optional details.
    
    BULLETPROOF: Handles database errors gracefully.
    """
    try:
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if task is None:
            raise ValueError(f"TrainingTask {task_id} not found")

        task.status = status
        if details is not None:
            task.details = details

        if status in {"completed", "failed"}:
            task.completed_at = datetime.now(timezone.utc)

        db.add(task)
        db.commit()
        db.refresh(task)
        return task
    except ValueError:
        raise  # Re-raise not found errors
    except Exception as e:
        logger.error(f"❌ Failed to update training task {task_id}: {e}", exc_info=True)
        db.rollback()
        raise


def start_model_run(
    db: Session,
    *,
    version: str,
    dataset_path: str | None = None,
    symbol_count: int | None = None,
    sample_count: int | None = None,
    feature_count: int | None = None,
    notes: str | None = None,
) -> ModelTrainingRun:
    """Persist the start of a model training run.
    
    BULLETPROOF: Handles database errors gracefully.
    """
    try:
        run = ModelTrainingRun(
            version=version,
            status="running",
            dataset_path=dataset_path,
            symbol_count=symbol_count,
            sample_count=sample_count,
            feature_count=feature_count,
            notes=notes,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run
    except Exception as e:
        logger.error(f"❌ Failed to start model run {version}: {e}", exc_info=True)
        db.rollback()
        raise


def complete_model_run(
    db: Session,
    run_id: int,
    *,
    status: str = "completed",
    model_path: str | None = None,
    scaler_path: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> ModelTrainingRun:
    """Mark an existing model training run as completed/failed.
    
    BULLETPROOF: Handles database errors gracefully.
    """
    try:
        run = db.query(ModelTrainingRun).filter(ModelTrainingRun.id == run_id).first()
        if run is None:
            raise ValueError(f"ModelTrainingRun {run_id} not found")

        run.status = status
        run.model_path = model_path or run.model_path
        run.scaler_path = scaler_path or run.scaler_path
        if metrics is not None:
            run.metrics = json.dumps(metrics)
        run.completed_at = datetime.now(timezone.utc)

        db.add(run)
        db.commit()
        db.refresh(run)
        return run
    except ValueError:
        raise  # Re-raise not found errors
    except Exception as e:
        logger.error(f"❌ Failed to complete model run {run_id}: {e}", exc_info=True)
        db.rollback()
        raise


def fail_model_run(
    db: Session,
    run_id: int,
    *,
    metrics: dict[str, Any] | None = None,
) -> ModelTrainingRun:
    """Convenience wrapper to mark a run as failed.
    
    BULLETPROOF: Handles database errors gracefully.
    """
    try:
        return complete_model_run(
            db,
            run_id,
            status="failed",
            metrics=metrics,
        )
    except Exception as e:
        logger.error(f"❌ Failed to mark model run {run_id} as failed: {e}", exc_info=True)
        raise


# Ensure modular SQLAlchemy models register with Base.metadata so Alembic
# migrations and metadata.create_all() are aware of every table.
_MODEL_MODULES = [
    "backend.models.ai_training",
    "backend.models.liquidity",
    "backend.models.positions",
]

for _module in _MODEL_MODULES:
    try:
        importlib.import_module(_module)
    except ImportError:
        try:
            importlib.import_module(_module.replace("backend.", ""))  # type: ignore[arg-type]
        except ImportError:
            # Missing modules are expected during minimal test setups; skip silently.
            pass
