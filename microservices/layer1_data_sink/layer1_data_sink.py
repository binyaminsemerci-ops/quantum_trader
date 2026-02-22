#!/usr/bin/env python3
"""
layer1_data_sink.py — Persistent OHLCV & Feature Data Store

Layer 1: Data Ingestion & Management

Subscribes to two Redis streams:
  quantum:stream:market.klines   → OHLCV candles (all symbols, 1m)
  quantum:stream:features        → Computed feature vectors (all symbols)

Writes to disk (Parquet, daily-partitioned):
  /opt/quantum/data/ohlcv/{SYMBOL}/1m/{YYYY-MM-DD}.parquet
  /opt/quantum/data/features/{SYMBOL}/{YYYY-MM-DD}.parquet

Also maintains Redis fast-lookback ZSET (last N candles per symbol):
  quantum:history:ohlcv:{SYMBOL}:1m  (score=open_time_ms, value=JSON, capped at LOOKBACK_ROWS)

This service is the ONLY component that touches disk. All other services
read from Redis. Disk = permanent archive. Redis = fast operational cache.

State published to:
  quantum:layer1:data_sink:latest  (hash, no TTL)
"""

import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import redis as redis_lib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s sink %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("layer1")

REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
DATA_ROOT   = os.getenv("QUANTUM_DATA_ROOT", "/opt/quantum/data")
LOOKBACK_ROWS = int(os.getenv("OHLCV_LOOKBACK_ROWS", "500"))   # per symbol in Redis ZSET
FLUSH_EVERY   = int(os.getenv("OHLCV_FLUSH_ROWS", "60"))        # write Parquet every N candles per symbol
STATE_KEY = "quantum:layer1:data_sink:latest"

KLINES_STREAM   = "quantum:stream:market.klines"
FEATURES_STREAM = "quantum:stream:features"
CONSUMER_GROUP  = "layer1_data_sink"
CONSUMER_ID     = "sink_worker_1"

# ── Graceful shutdown ─────────────────────────────────────────────────────
_RUNNING = True

def _handle_signal(sig, frame):
    global _RUNNING
    logger.info("Signal %s — flushing and stopping", sig)
    _RUNNING = False

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── Helpers ───────────────────────────────────────────────────────────────

def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode()
    return str(v) if v is not None else ""


def _date_from_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _parquet_path(base: str, symbol: str, interval: str, date_str: str) -> str:
    d = os.path.join(base, symbol, interval)
    _ensure_dir(d)
    return os.path.join(d, f"{date_str}.parquet")


def _feature_path(symbol: str, date_str: str) -> str:
    d = os.path.join(DATA_ROOT, "features", symbol)
    _ensure_dir(d)
    return os.path.join(d, f"{date_str}.parquet")


def _append_parquet(path: str, df_new: pd.DataFrame):
    """Append rows to parquet file, dedup on open_time or timestamp."""
    if os.path.isfile(path):
        df_existing = pd.read_parquet(path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        dedup_col = "open_time" if "open_time" in df_combined.columns else "timestamp"
        df_combined = df_combined.drop_duplicates(subset=[dedup_col], keep="last")
        df_combined = df_combined.sort_values(dedup_col)
    else:
        df_combined = df_new
    df_combined.to_parquet(path, index=False, compression="snappy")


def _update_redis_zset(r: redis_lib.Redis, symbol: str, interval: str, row: dict):
    """Keep last LOOKBACK_ROWS candles per symbol in a Redis ZSET for fast read access."""
    key = f"quantum:history:ohlcv:{symbol}:{interval}"
    score = row.get("open_time", row.get("timestamp", 0))
    r.zadd(key, {json.dumps(row): float(score)})
    # Keep only last N entries
    count = r.zcard(key)
    if count > LOOKBACK_ROWS:
        r.zremrangebyrank(key, 0, count - LOOKBACK_ROWS - 1)


# ── Stream consumer setup ─────────────────────────────────────────────────

def _ensure_consumer_groups(r: redis_lib.Redis):
    for stream in [KLINES_STREAM, FEATURES_STREAM]:
        try:
            r.xgroup_create(stream, CONSUMER_GROUP, id="$", mkstream=False)
            logger.info("Created consumer group %s on %s", CONSUMER_GROUP, stream)
        except redis_lib.ResponseError as e:
            if "BUSYGROUP" in str(e):
                pass  # already exists
            else:
                logger.warning("Group create error on %s: %s", stream, e)


# ── Processing ────────────────────────────────────────────────────────────

class OHLCVBuffer:
    """Accumulates candles in memory, flushes to Parquet every FLUSH_EVERY rows."""

    def __init__(self):
        # symbol → list of row dicts
        self._buf: dict[str, list[dict]] = defaultdict(list)
        self._counts: dict[str, int] = defaultdict(int)

    def add(self, r: redis_lib.Redis, row: dict):
        symbol   = row.get("symbol", "UNKNOWN")
        interval = row.get("interval", "1m")

        # Only store closed candles
        if not row.get("is_closed", True):
            return

        self._buf[symbol].append(row)
        self._counts[symbol] += 1
        _update_redis_zset(r, symbol, interval, row)

        if len(self._buf[symbol]) >= FLUSH_EVERY:
            self.flush_symbol(symbol, interval)

    def flush_symbol(self, symbol: str, interval: str = "1m"):
        rows = self._buf.pop(symbol, [])
        if not rows:
            return 0
        df = pd.DataFrame(rows)
        # Group by date and write each day's data
        if "open_time" in df.columns:
            df["_date"] = df["open_time"].apply(lambda ms: _date_from_ts(int(ms)))
        else:
            df["_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        written = 0
        for date_str, group in df.groupby("_date"):
            group_clean = group.drop(columns=["_date"])
            path = _parquet_path(os.path.join(DATA_ROOT, "ohlcv"), symbol, interval, str(date_str))
            _append_parquet(path, group_clean)
            written += len(group_clean)
        logger.debug("Flushed %d candles for %s → Parquet", written, symbol)
        return written

    def flush_all(self):
        symbols = list(self._buf.keys())
        total = 0
        for sym in symbols:
            total += self.flush_symbol(sym)
        return total

    def pending_count(self) -> int:
        return sum(len(v) for v in self._buf.values())


class FeatureBuffer:
    """Accumulates feature rows, flushes to Parquet by date."""

    def __init__(self):
        self._buf: dict[str, list[dict]] = defaultdict(list)

    def add(self, row: dict):
        symbol = row.get("symbol", "UNKNOWN")
        self._buf[symbol].append(row)
        if len(self._buf[symbol]) >= FLUSH_EVERY:
            self.flush_symbol(symbol)

    def flush_symbol(self, symbol: str):
        rows = self._buf.pop(symbol, [])
        if not rows:
            return 0
        df = pd.DataFrame(rows)
        ts_col = "timestamp"
        if ts_col in df.columns:
            try:
                df["_date"] = pd.to_datetime(df[ts_col]).dt.strftime("%Y-%m-%d")
            except Exception:
                df["_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        else:
            df["_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        written = 0
        for date_str, group in df.groupby("_date"):
            group_clean = group.drop(columns=["_date"])
            path = _feature_path(symbol, str(date_str))
            _append_parquet(path, group_clean)
            written += len(group_clean)
        return written

    def flush_all(self):
        symbols = list(self._buf.keys())
        for sym in symbols:
            self.flush_symbol(sym)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    logger.info("[L1] Data Sink starting — data_root=%s lookback=%d flush_every=%d",
                DATA_ROOT, LOOKBACK_ROWS, FLUSH_EVERY)
    _ensure_dir(DATA_ROOT)
    _ensure_dir(os.path.join(DATA_ROOT, "ohlcv"))
    _ensure_dir(os.path.join(DATA_ROOT, "features"))

    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try:
        r.ping()
        logger.info("[L1] Redis OK")
    except redis_lib.ConnectionError as e:
        logger.error("[L1] Redis FAILED: %s", e)
        sys.exit(1)

    _ensure_consumer_groups(r)

    ohlcv_buf   = OHLCVBuffer()
    feature_buf = FeatureBuffer()

    total_klines   = 0
    total_features = 0
    last_state_ts  = time.monotonic()
    symbols_seen: set[str] = set()

    while _RUNNING:
        # Read from both streams
        try:
            results = r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_ID,
                streams={KLINES_STREAM: ">", FEATURES_STREAM: ">"},
                count=100,
                block=2000,  # 2s block
            )
        except Exception as e:
            logger.error("[L1] XREADGROUP error: %s", e)
            time.sleep(5)
            continue

        if not results:
            # No new messages — periodic flush
            ohlcv_buf.flush_all()
            feature_buf.flush_all()
            continue

        ids_to_ack: dict[str, list] = {}

        for stream_bytes, messages in results:
            stream = _decode(stream_bytes)
            ids_to_ack[stream] = []

            for msg_id, fields in messages:
                ids_to_ack[stream].append(msg_id)
                flds = {_decode(k): _decode(v) for k, v in fields.items()}

                if stream == KLINES_STREAM:
                    # payload is a JSON string
                    try:
                        payload_raw = flds.get("payload", "{}")
                        row = json.loads(payload_raw)
                        if row.get("is_closed", False):
                            ohlcv_buf.add(r, row)
                            total_klines += 1
                            symbols_seen.add(row.get("symbol", "?"))
                    except Exception as e:
                        logger.warning("[L1] klines parse error: %s  raw=%s", e, flds)

                elif stream == FEATURES_STREAM:
                    try:
                        # features stream fields ARE the feature values directly
                        symbol_raw = flds.get("symbol", "UNKNOWN")
                        if symbol_raw and symbol_raw != "UNKNOWN":
                            feature_buf.add(flds)
                            total_features += 1
                    except Exception as e:
                        logger.warning("[L1] features parse error: %s", e)

        # ACK processed messages
        for stream, id_list in ids_to_ack.items():
            if id_list:
                r.xack(stream, CONSUMER_GROUP, *id_list)

        # Publish state every 30s
        if time.monotonic() - last_state_ts > 30:
            # Compute disk usage
            def disk_usage(path: str) -> int:
                total = 0
                for root, _, files in os.walk(path):
                    for f in files:
                        try:
                            total += os.path.getsize(os.path.join(root, f))
                        except Exception:
                            pass
                return total

            ohlcv_size    = disk_usage(os.path.join(DATA_ROOT, "ohlcv"))
            feature_size  = disk_usage(os.path.join(DATA_ROOT, "features"))
            pending       = ohlcv_buf.pending_count()

            state = {
                "status":           "RUNNING",
                "total_klines":     str(total_klines),
                "total_features":   str(total_features),
                "symbols_tracked":  str(len(symbols_seen)),
                "ohlcv_disk_mb":    f"{ohlcv_size / 1_048_576:.2f}",
                "feature_disk_mb":  f"{feature_size / 1_048_576:.2f}",
                "pending_flush":    str(pending),
                "lookback_rows":    str(LOOKBACK_ROWS),
                "ts":               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            r.hset(STATE_KEY, mapping=state)
            logger.info("[L1_STATUS] klines=%d features=%d symbols=%d ohlcv=%.1fMB features=%.1fMB",
                        total_klines, total_features, len(symbols_seen),
                        ohlcv_size / 1_048_576, feature_size / 1_048_576)
            last_state_ts = time.monotonic()

    # Final flush on exit
    logger.info("[L1] Shutting down — final flush ...")
    flushed = ohlcv_buf.flush_all()
    feature_buf.flush_all()
    r.hset(STATE_KEY, "status", "STOPPED")
    logger.info("[L1] Flushed %d remaining candles on shutdown", flushed)


if __name__ == "__main__":
    main()
