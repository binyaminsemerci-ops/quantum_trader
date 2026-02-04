#!/usr/bin/env python3
"""
Generate TOP 10 Universe based on volume, volatility, and trend strength.
Run this script periodically (via cron/systemd timer) to update the top 10 symbols.
"""

import redis
import json
import sys
from typing import List, Dict
from datetime import datetime

# Configuration
MIN_VOLUME_USDT = 10_000_000  # 10M USDT 24h volume
MIN_SIGMA = 0.005  # Minimum volatility
MIN_TS = 0.3  # Minimum trend strength
MAX_SYMBOLS = 10

# Core symbols (always included if they pass filters)
CORE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]


def get_symbol_metrics(r: redis.Redis, symbol: str) -> Dict:
    """Get metrics for a symbol from Redis."""
    try:
        # Market state
        ms_key = f"quantum:marketstate:{symbol}"
        if not r.exists(ms_key):
            return None
        
        sigma = float(r.hget(ms_key, "sigma") or "0")
        ts = float(r.hget(ms_key, "ts") or "0")
        p_trend = float(r.hget(ms_key, "p_trend") or "0")
        
        # Position snapshot (if exists)
        snap_key = f"quantum:position:snapshot:{symbol}"
        has_position = r.exists(snap_key)
        
        # TODO: Get 24h volume from Binance API or cache
        # For now, use placeholder
        volume_24h = 100_000_000  # Placeholder
        
        return {
            "symbol": symbol,
            "sigma": sigma,
            "ts": ts,
            "p_trend": p_trend,
            "volume_24h": volume_24h,
            "has_position": has_position,
            "score": calculate_score(sigma, ts, p_trend, has_position)
        }
    except Exception as e:
        print(f"Error getting metrics for {symbol}: {e}", file=sys.stderr)
        return None


def calculate_score(sigma: float, ts: float, p_trend: float, has_position: bool) -> float:
    """Calculate composite score for symbol ranking."""
    # Volatility score (0-1, higher is better)
    vol_score = min(sigma / 0.02, 1.0)
    
    # Trend strength score (0-1, higher is better)
    trend_score = min(ts / 0.8, 1.0)
    
    # Trend probability score (0-1, higher is better)
    prob_score = p_trend
    
    # Composite score
    score = (
        0.3 * vol_score +
        0.4 * trend_score +
        0.3 * prob_score
    )
    
    # Boost if we already have a position (continuity)
    if has_position:
        score *= 1.2
    
    return score


def generate_top10(r: redis.Redis) -> List[str]:
    """Generate top 10 symbols based on metrics."""
    # Get all symbols from universe
    universe_data = r.get("quantum:cfg:universe:active")
    if not universe_data:
        print("ERROR: quantum:cfg:universe:active not found!", file=sys.stderr)
        return []
    
    universe = json.loads(universe_data)
    all_symbols = universe.get("symbols", [])
    
    print(f"Scanning {len(all_symbols)} symbols from universe...")
    
    # Get metrics for all symbols
    candidates = []
    for symbol in all_symbols:
        metrics = get_symbol_metrics(r, symbol)
        if metrics:
            # Apply filters
            if metrics["sigma"] < MIN_SIGMA:
                continue
            if metrics["ts"] < MIN_TS:
                continue
            # if metrics["volume_24h"] < MIN_VOLUME_USDT:  # Disabled for now
            #     continue
            
            candidates.append(metrics)
    
    print(f"Found {len(candidates)} candidates after filtering")
    
    # Sort by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Ensure core symbols are included if they passed filters
    top10 = []
    for core in CORE_SYMBOLS:
        for c in candidates:
            if c["symbol"] == core:
                top10.append(core)
                break
    
    # Fill remaining slots
    for c in candidates:
        if c["symbol"] not in top10:
            top10.append(c["symbol"])
        if len(top10) >= MAX_SYMBOLS:
            break
    
    return top10


def main():
    r = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
    
    print("=" * 80)
    print("TOP 10 UNIVERSE GENERATOR")
    print("=" * 80)
    print()
    
    top10 = generate_top10(r)
    
    # Fallback: Use CORE_SYMBOLS if no candidates found
    if not top10:
        print("⚠️  No candidates found, using CORE_SYMBOLS fallback")
        top10 = CORE_SYMBOLS.copy()
    
    print(f"\n✅ TOP {len(top10)} SYMBOLS:")
    for i, symbol in enumerate(top10, 1):
        print(f"  {i:2d}. {symbol}")
    
    # Save to Redis
    top10_config = {
        "symbols": top10,
        "generated_at": datetime.utcnow().isoformat(),
        "criteria": {
            "min_sigma": MIN_SIGMA,
            "min_ts": MIN_TS,
            "max_symbols": MAX_SYMBOLS
        }
    }
    
    r.set("quantum:cfg:universe:top10", json.dumps(top10_config))
    print(f"\n✅ Saved to quantum:cfg:universe:top10")
    print("=" * 80)


if __name__ == "__main__":
    main()
