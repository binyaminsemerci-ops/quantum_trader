"""Lightweight migration: ensure new ModelRegistry columns exist.

Usage (one-off):
    python -m backend.scripts.migrate_add_model_registry_columns

This script is intentionally simple and idempotent; it checks for the
`tag` and `metrics_json` columns on the `model_registry` table and adds
them if they are missing (SQLite / PostgreSQL minimal support).

For production-grade migrations prefer Alembic. This exists only to
unblock CI/local after introducing new fields without a full migration
framework in place.
"""
from __future__ import annotations

import logging
from typing import Iterable

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from backend.database import engine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TABLE = "model_registry"
NEW_COLUMNS: dict[str, str] = {
    # name: DDL fragment (generic subset understood by SQLite & Postgres)
    "tag": "VARCHAR(64)",
    "metrics_json": "TEXT",
}


def existing_columns() -> set[str]:
    insp = inspect(engine)
    cols = {c["name"].lower() for c in insp.get_columns(TABLE)}
    return cols


def add_column(name: str, ddl_type: str) -> None:
    ddl = f"ALTER TABLE {TABLE} ADD COLUMN {name} {ddl_type}"
    logger.info("Adding column %s ...", name)
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Added column %s", name)


def migrate() -> None:
    try:
        cols = existing_columns()
    except SQLAlchemyError as exc:  # pragma: no cover - introspection failure
        logger.error("Could not inspect table '%s': %s", TABLE, exc)
        return

    to_add: Iterable[tuple[str, str]] = [
        (n, t) for n, t in NEW_COLUMNS.items() if n.lower() not in cols
    ]
    if not to_add:
        logger.info("All target columns already present; nothing to do.")
        return

    for name, ddl_type in to_add:
        try:
            add_column(name, ddl_type)
        except SQLAlchemyError as exc:
            logger.error("Failed adding column %s: %s", name, exc)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    migrate()
