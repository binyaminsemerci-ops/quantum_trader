#!/usr/bin/env python
"""qtctl - Minimal CLI for model registry & training orchestration.

Usage examples:
  python scripts/qtctl.py list
  python scripts/qtctl.py register --version v2025.10.03.1 --path ai_engine/models/model.bin --tag exp-sharpe --metrics metrics.json
  python scripts/qtctl.py promote --id 7
  python scripts/qtctl.py active

This is a thin convenience layer; training pipeline should call `register` automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

from backend.database import SessionLocal, ModelRegistry, Base, engine


def ensure_tables() -> None:
    # In case the new table hasn't been created yet (first run after pull)
    Base.metadata.create_all(bind=engine)


def cmd_list(args: argparse.Namespace) -> int:
    with SessionLocal() as session:
        rows = (
            session.query(ModelRegistry)
            .order_by(ModelRegistry.id.desc())
            .limit(args.limit)
            .all()
        )
        if not rows:
            print("No models registered.")
            return 0
        for r in rows:
            marker = "*" if r.is_active else "-"
            print(
                f"{marker} id={r.id} ver={r.version} tag={r.tag or '-'} active={r.is_active} "
                f"trained_at={r.trained_at} path={r.path}"
            )
    return 0


def cmd_active(_: argparse.Namespace) -> int:
    with SessionLocal() as session:
        row = session.query(ModelRegistry).filter(ModelRegistry.is_active == 1).first()
        if not row:
            print("No active model.")
            return 1
        print(json.dumps({
            "id": row.id,
            "version": row.version,
            "tag": row.tag,
            "path": row.path,
            "trained_at": row.trained_at.isoformat() if row.trained_at else None,
        }, indent=2))
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    metrics_json: str | None = None
    if args.metrics:
        if os.path.isfile(args.metrics):
            with open(args.metrics, "r", encoding="utf-8") as fh:
                metrics_json = json.dumps(json.load(fh))
        else:
            metrics_json = args.metrics  # allow raw JSON string
    params_json = args.params if args.params else None
    with SessionLocal() as session:
        entry = ModelRegistry(
            version=args.version,
            tag=args.tag,
            path=args.path,
            params_json=params_json,
            metrics_json=metrics_json,
            trained_at=datetime.now(timezone.utc),
            is_active=1 if args.activate else 0,
        )
        if args.activate:
            # demote existing
            session.query(ModelRegistry).filter(ModelRegistry.is_active == 1).update({"is_active": 0})
        session.add(entry)
        session.commit()
        session.refresh(entry)
        print(f"Registered model id={entry.id} version={entry.version} active={entry.is_active}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    with SessionLocal() as session:
        target = session.query(ModelRegistry).filter(ModelRegistry.id == args.id).first()
        if not target:
            print(f"Model id={args.id} not found", file=sys.stderr)
            return 1
        session.query(ModelRegistry).filter(ModelRegistry.is_active == 1).update({"is_active": 0})
        target.is_active = 1
        session.add(target)
        session.commit()
        print(f"Promoted model id={target.id} version={target.version}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qtctl", description="Quantum Trader control CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="List recent models")
    sp.add_argument("--limit", type=int, default=25)
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("active", help="Show active model metadata")
    sp.set_defaults(func=cmd_active)

    sp = sub.add_parser("register", help="Register a new model artifact")
    sp.add_argument("--version", required=True)
    sp.add_argument("--path", required=True, help="Path to model artifact")
    sp.add_argument("--tag")
    sp.add_argument("--params", help="Raw JSON string of hyperparameters")
    sp.add_argument("--metrics", help="Path to metrics JSON file or raw JSON string")
    sp.add_argument("--activate", action="store_true", help="Set as active on register")
    sp.set_defaults(func=cmd_register)

    sp = sub.add_parser("promote", help="Promote an existing model id to active")
    sp.add_argument("--id", type=int, required=True)
    sp.set_defaults(func=cmd_promote)

    return p


def main(argv: list[str] | None = None) -> int:
    ensure_tables()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
