#!/usr/bin/env python3
"""Autotrader that listens for new signals and places orders according to
risk rules. This implementation is a safe, dry-run-first prototype that
records order attempts to `trade_logs` in the database for audit.

Real order sending is implemented behind a `send_order` function that can
be swapped for real Binance client calls once keys and staging are in place.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import argparse
import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger("autotrader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# When running this file directly (python scripts/autotrader.py) the
# interpreter sets sys.path[0] to the scripts/ directory which makes
# package-level imports like `config.config` fail. Ensure the repo root
# is on sys.path so top-level packages can be imported the same way as
# when running via `python -m scripts.autotrader` or via the app entry.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def ensure_trade_tables():
    # import backend models at runtime to avoid top-level import-order issues
    from backend.database import Base, engine, Trade, TradeLog

    Base.metadata.create_all(bind=engine, tables=[Trade.__table__, TradeLog.__table__])


def send_order(symbol: str, side: str, qty: float, price: float, dry_run: bool = True, exchange_name: str | None = None) -> dict:
    """Send an order using the exchange adapter.

    By default this function simulates orders. To allow real orders, set the
    environment variable AUTOTRADER_ALLOW_REAL_ORDERS=1 and ensure the exchange
    credentials are available via config or env vars. The adapter factory will
    pick up configured credentials.
    """
    logger.info("send_order: %s %s %s @ %s (dry_run=%s)", symbol, side, qty, price, dry_run)
    allow_real = bool(os.environ.get("AUTOTRADER_ALLOW_REAL_ORDERS") in ("1", "true", "True"))
    if dry_run or not allow_real:
        return {"status": "simulated", "order_id": "sim-123", "filled_qty": 0.0}

    # attempt to resolve an exchange client and submit the order
    try:
        from backend.utils.exchanges import get_exchange_client

        client = get_exchange_client(name=exchange_name)
        # client.create_order expects symbol, side, qty (adapter normalizes type)
        res = client.create_order(symbol=symbol, side=side, qty=float(qty))
        return res
    except Exception as exc:
        logger.exception("Real order failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def process_new_signals(max_symbols: int = 5, dry_run: bool = True, notional_usd: float = 100.0):
    # import SessionLocal locally to avoid top-level import-order issues
    from backend.database import SessionLocal
    from sqlalchemy import text

    with SessionLocal() as session:
        # use raw SQL quickly to select new signals for prototype
        # Optionally filter by DEFAULT_SYMBOLS env if set
        raw = os.environ.get("DEFAULT_SYMBOLS")
        if raw:
            allowed = [s.strip() for s in raw.split(',') if s.strip()]
            placeholder = ','.join([f"'{s}'" for s in allowed])
            sql = f"SELECT id, symbol, side, qty, price, confidence FROM signals WHERE status='new' AND symbol IN ({placeholder})"
            res = session.execute(text(sql))
        else:
            res = session.execute(text("SELECT id, symbol, side, qty, price, confidence FROM signals WHERE status='new'"))
        rows = res.fetchall()
        if not rows:
            logger.info("No new signals")
            return
        # process up to max_symbols
        for r in rows[:max_symbols]:
            # r is a Row; map by index
            sig_id = r[0]
            sym = r[1]
            side = r[2]
            qty = float(r[3] or 0.0)
            price = float(r[4] or 0.0)
            confidence = float(r[5] or 0.0)
            # simple risk filter
            if confidence < float(os.environ.get("AUTO_TRADE_MIN_CONFIDENCE", 0.02)):
                logger.info("Skipping %s due to low confidence %s", sym, confidence)
                # mark as skipped
                session.execute(text("UPDATE signals SET status='skipped' WHERE id=:id"), {"id": sig_id})
                session.commit()
                continue
            # compute actual qty or use notional
            if qty <= 0:
                qty = max(0.0001, notional_usd / max(price, 1.0))
            resp = send_order(sym, side, qty, price, dry_run=dry_run)
            status = resp.get("status", "unknown")
            # write to trade_logs and mark processed
            try:
                from backend.database import TradeLog

                tl = TradeLog(
                    symbol=sym,
                    side=side,
                    qty=qty,
                    price=price,
                    status=status,
                    reason=str(resp),
                    timestamp=datetime.now(timezone.utc),
                )
                session.add(tl)
                session.execute(text("UPDATE signals SET status='processed' WHERE id=:id"), {"id": sig_id})
                session.commit()
            except Exception as e:
                logger.exception("Failed to record trade log: %s", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True, help="Run in dry-run mode (simulate orders).")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Disable dry-run; allow sending real orders when AUTOTRADER_ALLOW_REAL_ORDERS is set.")
    parser.add_argument("--once", dest="once", action="store_true", default=False)
    parser.add_argument("--max-symbols", type=int, default=5)
    parser.add_argument("--notional-usd", type=float, default=100.0)
    parser.add_argument("--symbol-group", type=str, default="default", choices=["default", "mainbase", "layer1", "layer2"], help="Use a predefined symbol group: mainbase, layer1, layer2 or default (from DEFAULT_SYMBOLS)")
    args = parser.parse_args()

    # import symbol lists locally to avoid top-level import-order issues
    from config.config import MAINBASE_SYMBOLS, LAYER1_SYMBOLS, LAYER2_SYMBOLS

    ensure_trade_tables()
    if args.once:
        # Apply symbol group override if requested
        if args.symbol_group != "default":
            if args.symbol_group == "mainbase":
                os.environ["DEFAULT_SYMBOLS"] = ",".join(MAINBASE_SYMBOLS)
            elif args.symbol_group == "layer1":
                os.environ["DEFAULT_SYMBOLS"] = ",".join(LAYER1_SYMBOLS)
            elif args.symbol_group == "layer2":
                os.environ["DEFAULT_SYMBOLS"] = ",".join(LAYER2_SYMBOLS)
        process_new_signals(max_symbols=args.max_symbols, dry_run=args.dry_run, notional_usd=args.notional_usd)
        return
    interval = int(os.environ.get("AUTO_TRADE_INTERVAL_SECONDS", "300"))
    logger.info("AutoTrader starting (dry_run=%s) interval=%s", args.dry_run, interval)
    while True:
        process_new_signals(max_symbols=args.max_symbols, dry_run=args.dry_run, notional_usd=args.notional_usd)
        time.sleep(interval)


if __name__ == "__main__":
    main()
