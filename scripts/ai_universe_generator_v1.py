#!/usr/bin/env python3
"""
AI Universe Generator v1 - Dynamic Top-10 symbol selection (NO HARDCODING!)

Fetches ~566 USDT perpetual symbols from Binance mainnet,
computes features from 15m/1h klines (volatility, trend, momentum),
ranks all symbols by score, selects Top-10.

Writes to PolicyStore with audit trail.

Usage:
    python scripts/ai_universe_generator_v1.py [--dry-run]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Check dependencies
try:
    import numpy as np
except ImportError:
    print("[AI-UNIVERSE] ERROR: numpy not installed. Run: pip install numpy")
    sys.exit(1)

from lib.policy_store import save_policy
import requests
import time
import json
import hashlib
from datetime import datetime


# Guardrails configuration (env override supported)
MIN_QUOTE_VOL_USDT_24H = int(os.getenv("MIN_QUOTE_VOL_USDT_24H", "20000000"))  # $20M default
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "15"))  # 15 bps default
MIN_AGE_DAYS = int(os.getenv("MIN_AGE_DAYS", "30"))  # 30 days default

# Cache for base universe (10min TTL to avoid excessive API calls)
UNIVERSE_CACHE = {"data": None, "timestamp": 0, "ttl": 600}


def fetch_base_universe():
    """Fetch ~566 USDT perpetual symbols from Binance mainnet (NOT hardcoded!)"""
    
    # Check cache first
    now = time.time()
    if UNIVERSE_CACHE["data"] and (now - UNIVERSE_CACHE["timestamp"]) < UNIVERSE_CACHE["ttl"]:
        print(f"[AI-UNIVERSE] Using cached base universe ({len(UNIVERSE_CACHE['data']['symbols'])} symbols)")
        return UNIVERSE_CACHE["data"]["symbols"], UNIVERSE_CACHE["data"]["exchange_info"]
    
    print("[AI-UNIVERSE] Fetching base universe from Binance mainnet...")
    
    try:
        # Binance Futures API
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        symbols = []
        
        for symbol_info in data.get("symbols", []):
            symbol = symbol_info.get("symbol", "")
            status = symbol_info.get("status", "")
            contract_type = symbol_info.get("contractType", "")
            
            # Filter: USDT perpetual, TRADING status, exclude leveraged tokens
            if (
                symbol.endswith("USDT") and
                status == "TRADING" and
                contract_type == "PERPETUAL" and
                not any(x in symbol for x in ["UP", "DOWN", "BULL", "BEAR"])  # Exclude leveraged tokens
            ):
                symbols.append(symbol)
        
        # Update cache (store both symbols and exchange_info for age lookup)
        UNIVERSE_CACHE["data"] = {"symbols": symbols, "exchange_info": data}
        UNIVERSE_CACHE["timestamp"] = now
        
        print(f"[AI-UNIVERSE] ✅ Fetched {len(symbols)} tradable symbols from Binance")
        return symbols, data
    
    except Exception as e:
        print(f"[AI-UNIVERSE] ❌ ERROR fetching universe: {e}")
        
        # FAIL-CLOSED: Do NOT fallback to hardcoded symbols!
        # If data fetch fails, policy generation should fail
        raise RuntimeError(f"Failed to fetch base universe: {e}")


def fetch_24h_stats_bulk():
    """Fetch 24h ticker stats for ALL symbols in one call (bulk endpoint)"""
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Map by symbol for O(1) lookup
        stats_map = {}
        for ticker in data:
            symbol = ticker.get("symbol", "")
            if symbol:
                stats_map[symbol] = {
                    "quoteVolume": float(ticker.get("quoteVolume", 0)),
                    "priceChangePercent": float(ticker.get("priceChangePercent", 0)),
                    "lastPrice": float(ticker.get("lastPrice", 0))
                }
        
        print(f"[AI-UNIVERSE] Fetched 24h stats for {len(stats_map)} symbols (bulk)")
        return stats_map
    except Exception as e:
        print(f"[AI-UNIVERSE] ERROR fetching 24h stats: {e}")
        raise RuntimeError(f"Failed to fetch 24h stats (fail-closed): {e}")


def fetch_orderbook_spread(symbol):
    """Fetch orderbook top for spread calculation"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/depth"
        params = {"symbol": symbol, "limit": 5}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        
        if not bids or not asks:
            return None
        
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        
        # Calculate spread in bps
        if best_bid > 0:
            mid = (best_bid + best_ask) / 2
            spread_bps = ((best_ask - best_bid) / mid) * 10000
            return spread_bps
        
        return None
    except Exception as e:
        return None


def get_symbol_age_days(symbol, exchange_info):
    """Extract listing age from exchangeInfo onboardDate if available"""
    try:
        for sym_info in exchange_info.get("symbols", []):
            if sym_info.get("symbol") == symbol:
                onboard_date = sym_info.get("onboardDate")
                if onboard_date:
                    # onboardDate is timestamp in milliseconds
                    age_sec = (time.time() * 1000 - onboard_date) / 1000
                    return age_sec / 86400  # days
        return None  # Unknown age
    except:
        return None


def fetch_klines(symbol, interval="15m", limit=100):
    """Fetch recent klines for a symbol"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        klines = response.json()
        
        # Extract close prices
        closes = np.array([float(k[4]) for k in klines])
        highs = np.array([float(k[2]) for k in klines])
        lows = np.array([float(k[3]) for k in klines])
        
        return closes, highs, lows
    
    except Exception as e:
        # Return None for failed fetches (will be excluded from ranking)
        return None, None, None


def compute_features(symbol):
    """Compute AI features for a symbol: volatility, trend, momentum"""
    
    # Fetch 15m and 1h klines
    closes_15m, highs_15m, lows_15m = fetch_klines(symbol, "15m", 100)
    closes_1h, highs_1h, lows_1h = fetch_klines(symbol, "1h", 100)
    
    if closes_15m is None or closes_1h is None:
        return None  # Skip symbols with data issues
    
    features = {}
    
    # 1. Volatility (ATR% on 15m)
    try:
        tr_15m = np.maximum(highs_15m - lows_15m, np.abs(highs_15m - np.roll(closes_15m, 1)))
        tr_15m = np.maximum(tr_15m, np.abs(lows_15m - np.roll(closes_15m, 1)))
        atr_15m = np.mean(tr_15m[-14:])  # 14-period ATR
        atr_pct_15m = (atr_15m / closes_15m[-1]) * 100 if closes_15m[-1] > 0 else 0
        features["volatility_15m"] = atr_pct_15m
    except:
        features["volatility_15m"] = 0
    
    # 2. Trend (EMA slope on 1h)
    try:
        ema_20 = closes_1h[-20:].mean()  # Simple approximation
        ema_50 = closes_1h[-50:].mean()
        ema_slope = ((ema_20 - ema_50) / ema_50) * 100 if ema_50 > 0 else 0
        features["trend_1h"] = ema_slope
    except:
        features["trend_1h"] = 0
    
    # 3. Momentum (ROC on 15m)
    try:
        roc_15m = ((closes_15m[-1] - closes_15m[-20]) / closes_15m[-20]) * 100 if closes_15m[-20] > 0 else 0
        features["momentum_15m"] = abs(roc_15m)  # Use absolute for ranking (momentum strength)
    except:
        features["momentum_15m"] = 0
    
    # 4. Trend strength (1h)
    try:
        roc_1h = ((closes_1h[-1] - closes_1h[-20]) / closes_1h[-20]) * 100 if closes_1h[-20] > 0 else 0
        features["momentum_1h"] = abs(roc_1h)
    except:
        features["momentum_1h"] = 0
    
    return features


def rank_symbols(symbols, exchange_info=None):
    """Compute features for all symbols and rank by score with liquidity guardrails
    
    Pipeline:
    A) Fetch 24h stats for ALL symbols (bulk)
    B) Filter by volume >= MIN_QUOTE_VOL_USDT_24H
    C) Filter by spread_bps <= MAX_SPREAD_BPS (only check spread for vol_ok)
    D) Filter by age >= MIN_AGE_DAYS (apply penalty if unknown)
    E) Rank by score with liquidity/spread factors
    """
    
    print(f"[AI-UNIVERSE] Computing features with guardrails for {len(symbols)} symbols...")
    print(f"[AI-UNIVERSE] Guardrails: vol>=${MIN_QUOTE_VOL_USDT_24H/1e6:.0f}M, age>={MIN_AGE_DAYS}d, spread<={MAX_SPREAD_BPS}bps")
    
    # Pipeline A: Fetch 24h stats in bulk (one API call for all symbols)
    stats_map = fetch_24h_stats_bulk()
    
    # Pipeline B: Filter by volume
    candidates_vol_ok = []
    excluded_volume = 0
    
    for symbol in symbols:
        if symbol not in stats_map:
            excluded_volume += 1
            continue
        
        quote_volume = stats_map[symbol]["quoteVolume"]
        
        if quote_volume < MIN_QUOTE_VOL_USDT_24H:
            excluded_volume += 1
            continue
        
        candidates_vol_ok.append({
            "symbol": symbol,
            "quote_volume": quote_volume,
            "stats": stats_map[symbol]
        })
    
    vol_ok = len(candidates_vol_ok)
    print(f"[AI-UNIVERSE] Volume filter: {vol_ok}/{len(symbols)} pass (excluded {excluded_volume})")
    
    # Pipeline C: Filter by spread (only check spread for vol_ok candidates)
    candidates_spread_ok = []
    excluded_spread = 0
    
    for i, candidate in enumerate(candidates_vol_ok):
        if i % 20 == 0:
            print(f"[AI-UNIVERSE] Spread check: {i}/{vol_ok}...")
        
        symbol = candidate["symbol"]
        spread_bps = fetch_orderbook_spread(symbol)
        
        if spread_bps is None or spread_bps > MAX_SPREAD_BPS:
            excluded_spread += 1
            continue
        
        candidate["spread_bps"] = spread_bps
        candidates_spread_ok.append(candidate)
    
    spread_ok = len(candidates_spread_ok)
    print(f"[AI-UNIVERSE] Spread filter: {spread_ok}/{vol_ok} pass (excluded {excluded_spread})")
    
    # Pipeline D: Filter by age
    candidates_age_ok = []
    excluded_age = 0
    unknown_age = 0
    
    for candidate in candidates_spread_ok:
        symbol = candidate["symbol"]
        age_days = get_symbol_age_days(symbol, exchange_info) if exchange_info else None
        
        candidate["age_days"] = age_days
        candidate["age_unknown_penalty"] = 1.0
        
        if age_days is not None:
            if age_days < MIN_AGE_DAYS:
                excluded_age += 1
                continue
        else:
            # Age unknown - apply penalty but don't exclude
            unknown_age += 1
            candidate["age_unknown_penalty"] = 0.85
        
        candidates_age_ok.append(candidate)
    
    age_ok = len(candidates_age_ok)
    print(f"[AI-UNIVERSE] Age filter: {age_ok}/{spread_ok} pass (excluded {excluded_age}, unknown_age {unknown_age})")
    
    # Log guardrails summary
    total = len(symbols)
    print(f"[AI-UNIVERSE] AI_UNIVERSE_GUARDRAILS total={total} vol_ok={vol_ok} spread_ok={spread_ok} age_ok={age_ok} excluded_vol={excluded_volume} excluded_spread={excluded_spread} excluded_age={excluded_age} unknown_age={unknown_age} min_qv={MIN_QUOTE_VOL_USDT_24H} max_spread_bps={MAX_SPREAD_BPS} min_age_days={MIN_AGE_DAYS}")
    
    # Pipeline E: Compute features and rank
    print(f"[AI-UNIVERSE] Computing features for {age_ok} eligible symbols...")
    
    scored_symbols = []
    all_volumes = [c["quote_volume"] for c in candidates_age_ok]
    
    for i, candidate in enumerate(candidates_age_ok):
        if i % 20 == 0:
            print(f"[AI-UNIVERSE] Features: {i}/{age_ok}...")
        
        symbol = candidate["symbol"]
        features = compute_features(symbol)
        
        if features is None:
            continue  # Skip symbols with data issues
        
        # Base score (existing logic)
        base_score = (
            abs(features["trend_1h"]) * 1.0 +
            features["momentum_15m"] * 0.8 +
            features["momentum_1h"] * 0.6 +
            features["volatility_15m"] * 0.4
        )
        
        # Liquidity factor (volume percentile, clamped 0.5..1.0)
        volume_rank = sum(1 for v in all_volumes if v < candidate["quote_volume"]) / max(len(all_volumes), 1)
        liquidity_factor = max(0.5, min(1.0, 0.5 + volume_rank * 0.5))
        
        # Spread factor (clamped 0.5..1.0)
        spread_factor = max(0.5, min(1.0, 1.0 - (candidate["spread_bps"] / MAX_SPREAD_BPS) * 0.5))
        
        # Apply penalties
        age_penalty = candidate["age_unknown_penalty"]
        
        # Final score
        score = base_score * liquidity_factor * spread_factor * age_penalty
        
        scored_symbols.append({
            "symbol": symbol,
            "score": score,
            "base_score": base_score,
            "liquidity_factor": liquidity_factor,
            "spread_factor": spread_factor,
            "features": features,
            "quote_volume": candidate["quote_volume"],
            "spread_bps": candidate["spread_bps"],
            "age_days": candidate["age_days"]
        })
    
    # Sort by score (descending)
    scored_symbols.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"[AI-UNIVERSE] Ranked {len(scored_symbols)} eligible symbols")
    
    return scored_symbols


def generate_ai_universe(dry_run=False):
    """Generate dynamic AI-selected Top-10 universe"""
    
    print("\n" + "="*70)
    print("  AI UNIVERSE GENERATOR v1 - Dynamic Top-10 Selection")
    if dry_run:
        print("  MODE: DRY-RUN (no policy save)")
    print("="*70)
    
    # Step 1: Fetch base universe (~566 symbols, NOT hardcoded!)
    base_universe, exchange_info = fetch_base_universe()
    
    if len(base_universe) < 10:
        raise RuntimeError(f"Base universe too small: {len(base_universe)} symbols (need at least 10)")
    
    # Step 2: Rank all symbols by AI features with liquidity guardrails
    ranked = rank_symbols(base_universe, exchange_info)
    
    # Step 3: Select Top-N (up to 10, but may be fewer if guardrails filter aggressively)
    top_n = min(10, len(ranked))
    
    if top_n < 1:
        raise RuntimeError(f"No symbols passed guardrails! Eligible: {len(ranked)}")
    
    if top_n < 10:
        print(f"[AI-UNIVERSE] WARNING: Only {top_n} symbols passed guardrails (expected 10)")
    
    top_symbols = ranked[:top_n]
    universe_symbols = [x["symbol"] for x in top_symbols]
    
    # Generate universe hash (for change detection)
    universe_str = ",".join(sorted(universe_symbols))
    universe_hash = hashlib.sha256(universe_str.encode()).hexdigest()[:16]
    
    print(f"\n[AI-UNIVERSE] TOP-{top_n} SELECTED:")
    
    # Per-symbol logging (grep-friendly)
    for i, entry in enumerate(top_symbols, 1):
        age_str = f"{entry['age_days']:.0f}" if entry['age_days'] is not None else "NA"
        print(f"[AI-UNIVERSE] AI_UNIVERSE_PICK symbol={entry['symbol']} score={entry['score']:.2f} qv24h={entry['quote_volume']:.0f} spread_bps={entry['spread_bps']:.2f} age_days={age_str} lf={entry['liquidity_factor']:.3f} sf={entry['spread_factor']:.3f}")
        
        # Also show human-readable summary
        print(f"  {i:2d}. {entry['symbol']:15s} score={entry['score']:8.2f} vol=${entry['quote_volume']/1e6:6.1f}M spread={entry['spread_bps']:4.1f}bps age={age_str:>4s}")
        age_str = f"{entry['age_days']:.0f}d" if entry['age_days'] is not None else "NA"
        print(f"  {i:2d}. {entry['symbol']:15s} score={entry['score']:8.2f} "
              f"liq_factor={entry['liquidity_factor']:.3f} "
              f"vol24h=${entry['quote_volume']/1e6:6.1f}M "
              f"spread={entry['spread_bps']:4.1f}bps "
              f"age={age_str:>4s}")
    
    print(f"\n[AI-UNIVERSE] Universe hash: {universe_hash}")
    print(f"[AI-UNIVERSE] Generator: ai_universe_v1")
    print(f"[AI-UNIVERSE] Features window: 15m,1h")
    
    # Step 4: Generate leverage by symbol (AI-adjusted based on volatility)
    leverage_by_symbol = {}
    for entry in top_symbols:
        symbol = entry["symbol"]
        vol = entry["features"]["volatility_15m"]
        
        # Higher volatility → lower leverage (6x-15x range)
        if vol > 3.0:
            lev = 6.0
        elif vol > 2.0:
            lev = 8.0
        elif vol > 1.5:
            lev = 10.0
        elif vol > 1.0:
            lev = 12.0
        else:
            lev = 15.0
        
        leverage_by_symbol[symbol] = lev
    
    # Step 5: AI harvest/kill params (sample - can be enhanced)
    harvest_params = {
        "T1_R": 1.8,
        "T2_R": 3.5,
        "T3_R": 5.8,
        "lock_R": 1.2,
        "be_plus_pct": 0.003,
        "cost_bps": 12.0,
        "k_regime_flip": 0.8,
        "k_sigma_spike": 0.6,
        "k_ts_drop": 0.4,
        "k_age_penalty": 0.2,
        "kill_threshold": 0.55
    }
    
    kill_params = {
        "trend_min": 0.25,
        "sigma_ref": 0.012,
        "sigma_spike_cap": 1.8,
        "ts_ref": 0.35,
        "ts_drop_cap": 0.45,
        "max_age_sec": 72000.0,
        "k_close_threshold": 0.62
    }
    
    # Step 6: Save to PolicyStore (skip if dry-run)
    if dry_run:
        print(f"\n[AI-UNIVERSE] DRY-RUN: Skipping policy save")
        print(f"[AI-UNIVERSE] Would save: {len(universe_symbols)} symbols, hash={universe_hash}")
        return True
    
    print(f"\n[AI-UNIVERSE] Writing policy to PolicyStore...")
    
    success = save_policy(
        universe_symbols=universe_symbols,
        leverage_by_symbol=leverage_by_symbol,
        harvest_params=harvest_params,
        kill_params=kill_params,
        valid_for_seconds=3600,
        policy_version="1.0.0-ai-v1",
        generator="ai_universe_v1",
        features_window="15m,1h",
        universe_hash=universe_hash
    )
    
    if success:
        print(f"[AI-UNIVERSE] AI policy generated successfully!")
        print(f"[AI-UNIVERSE] Universe: {len(universe_symbols)} symbols")
        print(f"[AI-UNIVERSE] Leverage range: {min(leverage_by_symbol.values()):.1f}x - {max(leverage_by_symbol.values()):.1f}x")
        print(f"[AI-UNIVERSE] Valid for: 60 minutes")
        print(f"[AI-UNIVERSE] Next refresh: {datetime.fromtimestamp(time.time() + 3600).strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    else:
        print(f"[AI-UNIVERSE] Failed to save policy to PolicyStore")
        return False


if __name__ == "__main__":
    # Parse arguments
    dry_run = "--dry-run" in sys.argv
    
    try:
        success = generate_ai_universe(dry_run=dry_run)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[AI-UNIVERSE] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
