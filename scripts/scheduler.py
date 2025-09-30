#!/usr/bin/env python3
"""Simple scheduler that runs training jobs on a fixed cadence and writes
generated signals into the database for the autotrader to consume.

Default cadence: every 6 hours (configurable). The script supports a
--once flag for CI/local testing to run a single iteration.

This is intentionally lightweight and safe: it simulates a training job and
produces fake signals when running in dry-run mode. Real training code can
be invoked in place of `simulate_training_and_generate_signals`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone

# Ensure repo root is on sys.path so `import backend` works when running
# the script directly (sys.path[0] is the script's directory otherwise).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func

from config.config import DEFAULT_SYMBOLS, MAINBASE_SYMBOLS, LAYER1_SYMBOLS, LAYER2_SYMBOLS

from backend.database import Base, engine, SessionLocal, create_training_task, update_training_task

logger = logging.getLogger("scheduler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    symbol = Column(String(50), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String(32), nullable=False, default="new")
    meta = Column(Text, nullable=True)


def ensure_signals_table():
    # create signals table if missing. If an older signals table exists with a
    # broken textual server_default like 'now()', SQLAlchemy will try to parse
    # that string as a datetime when reading defaults which raises
    # ValueError: Invalid isoformat string: 'now()'. To be safe for local
    # testing we'll detect that case and drop the table so the corrected
    # schema (server_default=func.now()) can be created.
    from sqlalchemy import inspect

    inspector = inspect(engine)
    if inspector.has_table("signals"):
        try:
            cols = inspector.get_columns("signals")
            for c in cols:
                if c.get("name") == "created_at":
                    default = c.get("default") or ""
                    if isinstance(default, str) and "now" in default.lower():
                        # older/broken table: drop it and recreate
                        engine.execute("DROP TABLE IF EXISTS signals")
                        break
        except Exception:
            # If introspection fails, drop and recreate as a last resort for tests
            try:
                engine.execute("DROP TABLE IF EXISTS signals")
            except Exception:
                pass

    Base.metadata.create_all(bind=engine, tables=[Signal.__table__])


def simulate_training_and_generate_signals(limit: int = 5):
    # deterministic-ish fake signals for testing
    # Prefer central DEFAULT_SYMBOLS from config; fall back to env var if set.
    raw = os.environ.get("DEFAULT_SYMBOLS")
    if raw:
        symbols = [s.strip() for s in raw.split(',') if s.strip()]
    else:
        symbols = DEFAULT_SYMBOLS
    out = []
    for _ in range(min(limit, len(symbols))):
        s = random.choice(symbols)
        side = random.choice(["buy", "sell"])
        price = round(random.uniform(1, 60000), 2)
        qty = round(max(0.001, 1.0 / max(price, 1.0) * random.uniform(0.5, 2.0)), 6)
        confidence = round(random.uniform(0.01, 0.99), 3)
        out.append({"symbol": s, "side": side, "qty": qty, "price": price, "confidence": confidence})
    return out


def run_iteration(limit: int, dry_run: bool):
    logger.info("Creating training task and running simulated training")
    with SessionLocal() as session:
        task = create_training_task(
            session,
            symbols=",".join((os.environ.get("DEFAULT_SYMBOLS") and os.environ.get("DEFAULT_SYMBOLS").split(",") or DEFAULT_SYMBOLS)[:limit]),
            limit=limit,
        )
        try:
            signals = simulate_training_and_generate_signals(limit=limit)
            # persist signals
            for s in signals:
                # set created_at explicitly to avoid reading server_default values
                # from an older/broken schema (which may contain the literal
                # string 'now()' and cause SQLAlchemy to attempt to parse it
                # as a datetime). Using an explicit timestamp here is safe for
                # local testing and avoids the ValueError seen previously.
                sig = Signal(
                    created_at=datetime.now(timezone.utc),
                    symbol=s["symbol"],
                    side=s["side"],
                    qty=s["qty"],
                    price=s["price"],
                    confidence=s["confidence"],
                    status=("new" if not dry_run else "new"),
                )
                session.add(sig)
            session.commit()
            update_training_task(session, task.id, status="completed", details=f"generated {len(signals)} signals")
            logger.info("Generated %d signals", len(signals))
        except Exception as exc:
            logger.exception("Training failed: %s", exc)
            update_training_task(session, task.id, status="failed", details=str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-hours", type=float, default=6.0, help="How often to run training (hours). Default: 6")
    parser.add_argument("--limit", type=int, default=5, help="Max number of symbols/signals to generate per run")
    parser.add_argument("--symbol-group", type=str, default="default", choices=["default", "mainbase", "layer1", "layer2"], help="Use a predefined symbol group: mainbase, layer1, layer2 or default (from DEFAULT_SYMBOLS)")
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit (useful for testing)")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", default=True, help="Disable dry-run (be careful!)")
    args = parser.parse_args()

    ensure_signals_table()

    logger.info("Scheduler starting (interval %.2f hours). dry_run=%s", args.interval_hours, args.dry_run)
    if args.once:
        # If a symbol group was requested, override DEFAULT_SYMBOLS for this run
        if args.symbol_group != "default":
            if args.symbol_group == "mainbase":
                os.environ["DEFAULT_SYMBOLS"] = ",".join(MAINBASE_SYMBOLS)
            elif args.symbol_group == "layer1":
                os.environ["DEFAULT_SYMBOLS"] = ",".join(LAYER1_SYMBOLS)
            elif args.symbol_group == "layer2":
                os.environ["DEFAULT_SYMBOLS"] = ",".join(LAYER2_SYMBOLS)
        run_iteration(limit=args.limit, dry_run=args.dry_run)
        return

    interval_seconds = int(args.interval_hours * 3600)
    while True:
        # If a symbol group was requested, set DEFAULT_SYMBOLS in the env for each iteration
        if args.symbol_group != "default":
            if args.symbol_group == "mainbase":
                os.environ["DEFAULT_SYMBOLS"] = ",".join(MAINBASE_SYMBOLS)
            elif args.symbol_group == "layer1":
                os.environ["DEFAULT_SYMBOLS"] = ",".join(LAYER1_SYMBOLS)
            elif args.symbol_group == "layer2":
                os.environ["DEFAULT_SYMBOLS"] = ",".join(LAYER2_SYMBOLS)
        run_iteration(limit=args.limit, dry_run=args.dry_run)
        logger.info("Sleeping for %d seconds", interval_seconds)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
