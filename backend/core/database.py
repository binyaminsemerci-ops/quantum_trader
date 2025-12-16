"""
Database connection utilities for backend core.

Provides SQLAlchemy session management for database operations.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import os

# Use standard SQLite (not async) to avoid aiosqlite dependency
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/backend/data/trades.db")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(
    engine,
    autocommit=False,
    autoflush=False
)

Base = declarative_base()


def get_db_session() -> Generator[Session, None, None]:
    """Get database session for dependency injection."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_database():
    """Initialize database schema."""
    Base.metadata.create_all(bind=engine)


def close_database():
    """Close database connection pool."""
    engine.dispose()
