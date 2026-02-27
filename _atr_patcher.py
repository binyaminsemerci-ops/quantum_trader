#!/usr/bin/env python3
"""
atr_patcher.py — Backfills atr_value into quantum:position:* keys.

Reads OHLCV data from quantum:history:ohlcv:SYMBOL:1m (sorted set),
computes ATR-14 (mean of last 14 true ranges), and patches any position
key that has atr_value == '0' or missing.

Safe: only writes atr_value to existing position hashes. Never creates keys.
"""

import json
import sys
import time
import redis
import numpy as np

r = redis.Redis(host="localhost", port=6379, db=0)

OHLCV_PREFIX = "quantum:history:ohlcv"
POSITION_PREFIX = "quantum:position:"
ATR_PERIOD = 14
MIN_CANDLES = 10

def compute_atr(symbol: str) -> float | None:
    """Compute ATR-14 from 1m OHLCV zset for symbol."""
    key = f"{OHLCV_PREFIX}:{symbol}:1m"
    try:
        # Get last 50 candles (sorted by score = open_time ms)
        raw_items = r.zrange(key, -50, -1)
        if not raw_items:
            return None

        candles = []
        seen_times = set()
        for item in reversed(raw_items):  # newest last → iterate newest first
            try:
                c = json.loads(item)
                ot = c.get("open_time", 0)
                if ot in seen_times:
                    continue
                seen_times.add(ot)
                h = float(c.get("high", 0))
                l = float(c.get("low", 0))
                cl = float(c.get("close", 0))
                if h > 0 and l > 0:
                    candles.append((h, l, cl))
            except Exception:
                continue

        if len(candles) < MIN_CANDLES:
            return None

        # True range = max(H-L, |H-prev_close|, |L-prev_close|)
        trs = []
        for i in range(1, len(candles)):
            h, l, _ = candles[i]
            _, _, prev_close = candles[i - 1]
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            trs.append(tr)

        if not trs:
            return None

        # Use last ATR_PERIOD values
        atr = float(np.mean(trs[-ATR_PERIOD:]))
        return atr if atr > 0 else None

    except Exception as e:
        print(f"  [ERROR] ATR compute for {symbol}: {e}")
        return None


def patch_positions():
    pos_keys = r.keys(f"{POSITION_PREFIX}*")
    print(f"Found {len(pos_keys)} quantum:position:* keys")

    patched = 0
    skipped_has_atr = 0
    skipped_no_ohlcv = 0
    skipped_wrong_type = 0
    errors = 0

    for key_bytes in pos_keys:
        key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes

        # Only patch hash keys
        try:
            key_type = r.type(key)
            if key_type != b"hash":
                skipped_wrong_type += 1
                continue
        except Exception:
            errors += 1
            continue

        try:
            raw = r.hgetall(key)
            raw = {k.decode(): v.decode() for k, v in raw.items()}
        except Exception as e:
            print(f"  [ERROR] hgetall {key}: {e}")
            errors += 1
            continue

        # Extract symbol from key or hash
        symbol = raw.get("symbol", "")
        if not symbol:
            # Try to extract from key name: quantum:position:SYMBOL or quantum:position:ledger:SYMBOL etc
            parts = key.split(":")
            # quantum:position:SYMBOL → parts[-1]
            # quantum:position:ledger:SYMBOL → parts[-1]
            # quantum:position:snapshot:SYMBOL → parts[-1]
            symbol = parts[-1] if len(parts) >= 3 else ""

        if not symbol:
            continue

        # Check if atr_value already valid
        existing_atr = float(raw.get("atr_value", 0) or 0)
        if existing_atr > 0:
            skipped_has_atr += 1
            continue

        # Compute ATR from OHLCV
        atr = compute_atr(symbol)
        if atr is None:
            skipped_no_ohlcv += 1
            print(f"  [SKIP] {key}: no OHLCV data for {symbol}")
            continue

        # Patch
        try:
            r.hset(key, "atr_value", str(atr))
            print(f"  [PATCH] {key}: atr_value={atr:.6f} (symbol={symbol})")
            patched += 1
        except Exception as e:
            print(f"  [ERROR] hset {key}: {e}")
            errors += 1

    print()
    print("=== ATR PATCHER SUMMARY ===")
    print(f"  Patched:         {patched}")
    print(f"  Already had ATR: {skipped_has_atr}")
    print(f"  No OHLCV data:   {skipped_no_ohlcv}")
    print(f"  Wrong type:      {skipped_wrong_type}")
    print(f"  Errors:          {errors}")
    print(f"  Total keys:      {len(pos_keys)}")


if __name__ == "__main__":
    print(f"ATR Patcher starting at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    patch_positions()
    print("Done.")
