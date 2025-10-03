from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from backend.database import session_scope, Alert, AlertEvent

logger = logging.getLogger(__name__)

# Basic in-memory websocket registry for alert events
_clients: List = []


def register_ws(ws):
    _clients.append(ws)


def unregister_ws(ws):
    try:
        _clients.remove(ws)
    except Exception:
        pass


async def _notify_clients(event: Dict):
    to_remove = []
    for ws in list(_clients):
        try:
            await ws.send_json(event)
        except Exception:
            to_remove.append(ws)
    for r in to_remove:
        unregister_ws(r)


def evaluate_alert_for_symbol(alert: Alert) -> None:
    # simple evaluator: fetch recent candles and compute latest price
    try:
        # Import market data helper lazily to avoid circular imports at module import time
        from backend.utils.market_data import fetch_recent_candles

        candles = fetch_recent_candles(symbol=alert.symbol, limit=24)
        closes = [c.get("close") for c in candles if c.get("close") is not None]
        if not closes:
            return
        latest = float(closes[-1])
        # determine value based on condition
        value = latest
        triggered = False
        if alert.condition == "price_above" and latest > alert.threshold:
            triggered = True
        elif alert.condition == "price_below" and latest < alert.threshold:
            triggered = True
        elif alert.condition == "change_pct":
            first = float(closes[0]) if closes[0] else latest
            change = (latest - first) / (first or 1.0)
            value = change
            if abs(change) >= alert.threshold:
                triggered = True

        if triggered:
            # record event
            with session_scope() as session:
                ev = AlertEvent(
                    alert_id=alert.id,
                    symbol=alert.symbol,
                    condition=alert.condition,
                    threshold=alert.threshold,
                    value=value,
                )
                session.add(ev)
                session.commit()
                session.refresh(ev)
                event_payload = {
                    "type": "alert_event",
                    "id": ev.id,
                    "alert_id": ev.alert_id,
                    "symbol": ev.symbol,
                    "condition": ev.condition,
                    "threshold": ev.threshold,
                    "value": ev.value,
                    "created_at": ev.created_at.isoformat() if ev.created_at else None,
                }
            # notify ws clients. If we're in an async event loop (normal runtime)
            # schedule the notification as a task. During synchronous unit tests
            # there may be no running loop, so run the notifier synchronously to
            # avoid "coroutine was never awaited" warnings.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop:
                loop.create_task(_notify_clients(event_payload))
            else:
                # Run synchronously in a temporary event loop for tests / CLI
                try:
                    asyncio.run(_notify_clients(event_payload))
                except Exception:
                    # Best-effort: don't let notifier failures break evaluation
                    logger.exception("Failed to run notifier synchronously")
    except Exception as exc:
        logger.exception(
            "Error evaluating alert %s: %s", getattr(alert, "id", None), exc
        )


async def evaluator_loop(poll_interval: float = 5.0):
    """Continuously evaluate all enabled alerts every poll_interval seconds."""
    while True:
        try:
            with session_scope() as session:
                alerts = session.query(Alert).filter(Alert.enabled == 1).all()
                for a in alerts:
                    evaluate_alert_for_symbol(a)
        except Exception as exc:
            logger.exception("Alert evaluator top-level failure: %s", exc)
        await asyncio.sleep(poll_interval)
