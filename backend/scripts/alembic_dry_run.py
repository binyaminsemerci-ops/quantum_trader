"""Utility to validate Alembic migrations against a database snapshot.

This script copies a provided SQLite snapshot into a temporary directory and
executes `alembic upgrade head` against the copy. Optionally it can emit the
SQL statements that would be applied during the upgrade, which is useful for
change-management review before the migration reaches production.

Usage (from repo root):

    python backend/scripts/alembic_dry_run.py --snapshot backups/prod/trades.db \
        --sql-output artifacts/alembic-upgrade.sql

The snapshot file is never modified; the migration runs against a disposable
copy.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from alembic import command
from alembic.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alembic upgrade against a database snapshot copy.")
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Path to the SQLite database snapshot to validate (e.g. backups/prod/trades.db).",
    )
    parser.add_argument(
        "--sql-output",
        help="Optional path to write the generated SQL for offline review (runs Alembic in offline mode).",
    )
    return parser.parse_args()


def configure_alembic(database_url: str, stdout_path: Path | None = None) -> Config:
    """Prepare an Alembic Config bound to the supplied database URL."""

    backend_dir = Path(__file__).resolve().parents[1]
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.attributes["stdout"] = stdout_path.open("w", encoding="utf-8")
    return cfg


def close_stdout(cfg: Config) -> None:
    stream = cfg.attributes.get("stdout")
    if stream is not None:
        stream.close()


def main() -> int:
    args = parse_args()
    snapshot = Path(args.snapshot).expanduser().resolve()
    if not snapshot.is_file():
        print(f"Snapshot not found: {snapshot}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="alembic-dry-run-") as tmp_dir:
        working_copy = Path(tmp_dir) / snapshot.name
        shutil.copy2(snapshot, working_copy)

        database_url = f"sqlite:///{working_copy}"
        previous_url = os.environ.get("QUANTUM_TRADER_DATABASE_URL")
        os.environ["QUANTUM_TRADER_DATABASE_URL"] = database_url

        sql_output_path: Path | None = None
        if args.sql_output:
            sql_output_path = Path(args.sql_output).expanduser().resolve()

        cfg_online: Config | None = None
        try:
            if sql_output_path is not None:
                cfg_offline = configure_alembic(database_url, sql_output_path)
                # Generate the upgrade script without mutating the snapshot copy.
                command.upgrade(cfg_offline, "head", sql=True)
                close_stdout(cfg_offline)

            cfg_online = configure_alembic(database_url)
            # Run the migrations against the disposable copy to ensure they succeed online.
            command.upgrade(cfg_online, "head")
        finally:
            if cfg_online is not None:
                close_stdout(cfg_online)

            if previous_url is None:
                os.environ.pop("QUANTUM_TRADER_DATABASE_URL", None)
            else:
                os.environ["QUANTUM_TRADER_DATABASE_URL"] = previous_url

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
