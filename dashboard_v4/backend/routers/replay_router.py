"""Replay Router — Historical trade decision chain replay via Redis streams"""
from typing import Optional
from fastapi import APIRouter, Query
import redis
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/replay", tags=["Trade Replay"])

STREAM_MAP = {
    "trade.intent": "quantum:stream:trade.intent",
    "apply.plan": "quantum:stream:apply.plan",
    "apply.result": "quantum:stream:apply.result",
    "exit.intent": "quantum:stream:exit.intent",
    "harvest.intent": "quantum:stream:harvest.intent",
    "trade.closed": "quantum:stream:trade.closed",
}


def _get_redis():
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    return redis.Redis(host=host, port=port, decode_responses=True, socket_timeout=3)


def _decode_entry(entry_id: str, fields: dict) -> dict:
    """Convert Redis stream entry to JSON-friendly dict"""
    return {"id": entry_id, **{k: v for k, v in fields.items()}}


@router.get("/streams")
def list_streams():
    """List available streams and their lengths"""
    try:
        r = _get_redis()
        result = {}
        for label, key in STREAM_MAP.items():
            try:
                length = r.xlen(key)
                result[label] = {"key": key, "length": length}
            except Exception:
                result[label] = {"key": key, "length": 0}
        return result
    except Exception as e:
        return {"error": str(e)}


@router.get("/stream/{stream_name}")
def read_stream(
    stream_name: str,
    symbol: Optional[str] = Query(None),
    count: int = Query(50, ge=1, le=500),
    start: str = Query("-"),
    end: str = Query("+"),
):
    """Read entries from a stream with optional symbol filter.
    Use start/end for time-range queries (Redis stream IDs or '-'/'+')."""
    if stream_name not in STREAM_MAP:
        return {"error": f"Unknown stream. Available: {list(STREAM_MAP.keys())}"}
    try:
        r = _get_redis()
        # XREVRANGE returns newest first
        raw = r.xrevrange(STREAM_MAP[stream_name], max=end, min=start, count=count)
        entries = [_decode_entry(eid, fields) for eid, fields in raw]
        if symbol:
            s = symbol.upper()
            entries = [e for e in entries if e.get("symbol", "").upper() == s]
        return {"stream": stream_name, "count": len(entries), "entries": entries}
    except Exception as e:
        return {"stream": stream_name, "error": str(e), "entries": []}


@router.get("/trade/{symbol}")
def replay_trade(
    symbol: str,
    count: int = Query(20, ge=1, le=200),
):
    """Get the full decision chain for a symbol across all streams.
    Returns events from trade.intent → apply.plan → apply.result → trade.closed
    and exit.intent → harvest.intent in chronological order."""
    s = symbol.upper()
    try:
        r = _get_redis()
        all_events = []
        for label, key in STREAM_MAP.items():
            try:
                raw = r.xrevrange(key, count=count)
                for eid, fields in raw:
                    if fields.get("symbol", "").upper() == s:
                        all_events.append({
                            "stream": label,
                            "id": eid,
                            **{k: v for k, v in fields.items()},
                        })
            except Exception:
                continue
        # Sort by Redis stream ID (timestamp-based) ascending
        all_events.sort(key=lambda e: e["id"])
        return {"symbol": s, "count": len(all_events), "chain": all_events}
    except Exception as e:
        return {"symbol": s, "error": str(e), "chain": []}


@router.get("/symbols")
def active_symbols():
    """Get list of symbols that appear in recent trade.intent stream"""
    try:
        r = _get_redis()
        raw = r.xrevrange(STREAM_MAP["trade.intent"], count=200)
        symbols = sorted(set(
            fields.get("symbol", "").upper()
            for _, fields in raw
            if fields.get("symbol")
        ))
        return {"symbols": symbols}
    except Exception as e:
        return {"symbols": [], "error": str(e)}
