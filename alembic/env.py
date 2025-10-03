from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from sqlalchemy import create_engine
from alembic import context

# Import metadata
try:
    from backend.database import Base  # type: ignore
except Exception as exc:  # pragma: no cover
    Base = None  # type: ignore
    print(f"[alembic] Warning: could not import Base: {exc}")

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Provide metadata for 'autogenerate'
target_metadata = getattr(Base, 'metadata', None)

def get_url() -> str:
    return os.environ.get(
        "QUANTUM_TRADER_DATABASE_URL",
        "sqlite:///backend/data/trades.db",
    )

def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    connectable = create_engine(get_url())
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
