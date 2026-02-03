#!/usr/bin/env python3
"""
AI Universe Generator v1 - Dynamic Top-10 symbol selection (NO HARDCODING!)

Fetches ~566 USDT perpetual symbols from Binance mainnet,
computes features from 15m/1h klines (volatility, trend, momentum),
ranks all symbols by score, selects Top-10.

Writes to PolicyStore with audit trail.

Usage:
    python scripts/ai_universe_generator_v1.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Check dependencies
try:
    import numpy as np
except ImportError:
    print("[AI-UNIVERSE] ❌ ERROR: numpy not installed. Run: pip install numpy")
    sys.exit(1)

from lib.policy_store import save_policy
import requests
import time
import json
import hashlib
from datetime import datetime


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


def fetch_24h_stats(symbol):
    """Fetch 24h ticker stats for liquidity/volume check"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr"
        params = {"symbol": symbol}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        return {
            "quoteVolume": float(data.get("quoteVolume", 0)),
            "priceChangePercent": float(data.get("priceChangePercent", 0)),
            "lastPrice": float(data.get("lastPrice", 0))
        }
    except Exception as e:
        return None


def fetch_orderbook_top(symbol):
    """Fetch top of orderbook for spread calculation"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/depth"
        params = {"symbol": symbol, "limit": 5}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        
        if not bids or not asks:
            return None, None
        
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        
        return best_bid, best_ask
    except Exception as e:
        return None, None


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
    """Compute features for all symbols and rank by score with liquidity guardrails"""
    
    # Guardrails configuration
    MIN_QUOTE_VOLUME_USDT_24H = 20_000_000  # $20M/day
    MIN_AGE_DAYS = 30
    MAX_SPREAD_BPS = 15  # 15 basis points
    
    print(f"[AI-UNIVERSE] Computing features for {len(symbols)} symbols...")
    print(f"[AI-UNIVERSE] Guardrails: vol≥${MIN_QUOTE_VOLUME_USDT_24H/1e6:.0f}M, age≥{MIN_AGE_DAYS}d, spread≤{MAX_SPREAD_BPS}bps")
    
    scored_symbols = []
    excluded_volume = 0
    excluded_spread = 0
    excluded_age = 0
    unknown_age = 0
    
    # Collect all quote volumes for percentile calculation
    all_volumes = []
    
    for i, symbol in enumerate(symbols):
        if i % 50 == 0:
            print(f"[AI-UNIVERSE] Progress: {i}/{len(symbols)} symbols processed...")
        
        # Fetch 24h stats
        stats_24h = fetch_24h_stats(symbol)
        if stats_24h is None:
            continue
        
        quote_volume = stats_24h["quoteVolume"]
        all_volumes.append(quote_volume)
        
        # Guardrail 1: Volume filter
        if quote_volume < MIN_QUOTE_VOLUME_USDT_24H:
            excluded_volume += 1
            continue
        
        # Fetch orderbook for spread
        best_bid, best_ask = fetch_orderbook_top(symbol)
        if best_bid is None or best_ask is None or best_bid <= 0:
            excluded_spread += 1
            continue
        
        # Calculate spread in bps
        spread_bps = ((best_ask - best_bid) / best_bid) * 10000
        
        # Guardrail 2: Spread filter
        if spread_bps > MAX_SPREAD_BPS:
            excluded_spread += 1
            continue
        
        # Check age
        age_days = None
        age_penalty = 1.0
        if exchange_info:
            age_days = get_symbol_age_days(symbol, exchange_info)
        
        # Guardrail 3: Age filter
        if age_days is not None:
            if age_days < MIN_AGE_DAYS:
                excluded_age += 1
                continue
        else:
            # Age unknown - apply penalty but don't exclude
            unknown_age += 1
            age_penalty = 0.85
        
        # Now compute features
        features = compute_features(symbol)
        
        if features is None:
            continue  # Skip symbols with data issues
        
        # Score = trend + momentum + volatility_quality
        # Higher score = better candidate
        base_score = (
            abs(features["trend_1h"]) * 1.0 +        # Trend strength
            features["momentum_15m"] * 0.8 +         # 15m momentum
            features["momentum_1h"] * 0.6 +          # 1h momentum
            features["volatility_15m"] * 0.4         # Volatility (liquidity proxy)
        )
        
        # Compute liquidity_factor based on volume percentile and spread quality
        # Volume percentile (clamp 0.5..1.0)
        volume_percentile = 0.5
        if all_volumes:
            volume_rank = sum(1 for v in all_volumes if v < quote_volume) / max(len(all_volumes), 1)
            volume_percentile = max(0.5, min(1.0, 0.5 + volume_rank * 0.5))
        
        # Spread quality (clamp 0.5..1.0)
        spread_quality = max(0.5, min(1.0, 1.0 - (spread_bps / MAX_SPREAD_BPS) * 0.5))
        
        # Combined liquidity factor
        liquidity_factor = (volume_percentile * 0.7 + spread_quality * 0.3) * age_penalty
        
        # Final score
        score = base_score * liquidity_factor
        
        scored_symbols.append({
            "symbol": symbol,
            "score": score,
            "base_score": base_score,
            "liquidity_factor": liquidity_factor,
            "features": features,
            "quote_volume": quote_volume,
            "spread_bps": spread_bps,
            "age_days": age_days
        })
    
    # Sort by score (descending)
    scored_symbols.sort(key=lambda x: x["score"], reverse=True)
    
    eligible = len(scored_symbols)
    total = len(symbols)
    
    print(f"[AI-UNIVERSE] ✅ Ranked {eligible} symbols")
    print(f"[AI-UNIVERSE] AI_UNIVERSE_GUARDRAILS total={total} eligible={eligible} "
          f"excluded_volume={excluded_volume} excluded_spread={excluded_spread} "
          f"excluded_age={excluded_age} unknown_age={unknown_age}")
    
    return scored_symbols


def generate_ai_universe():
    """Generate dynamic AI-selected Top-10 universe"""
    
    print("\n" + "="*70)
    print("  AI UNIVERSE GENERATOR v1 - Dynamic Top-10 Selection")
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
        print(f"[AI-UNIVERSE] ⚠️  WARNING: Only {top_n} symbols passed guardrails (expected 10)")
    
    top_symbols = ranked[:top_n]
    universe_symbols = [x["symbol"] for x in top_symbols]
    
    # Generate universe hash (for change detection)
    universe_str = ",".join(sorted(universe_symbols))
    universe_hash = hashlib.sha256(universe_str.encode()).hexdigest()[:16]
    
    print(f"\n[AI-UNIVERSE] 🎯 TOP-{top_n} SELECTED:")
    for i, entry in enumerate(top_symbols, 1):
        age_str = f"{entry['age_days']:.0f}d" if entry['age_days'] is not None else "NA"
        print(f"  {i:2d}. {entry['symbol']:15s} score={entry['score']:8.2f} "
              f"liq_factor={entry['liquidity_factor']:.3f} "
              f"vol24h=${entry['quote_volume']/1e6:6.1f}M "
              f"spread={entry['spread_bps']:4.1f}bps "
              f"age={age_str:>4s}")
    
    print(f"\n[AI-UNIVERSE] Universe hash: {universe_hash}")
    print(f"[AI-UNIVERSE] Generator: ai_universe_v1")
    print(f"[AI-UNIVERSE] Features window: 15m,1h")
    
    # Log detailed Top-N with all metrics
    top_details = []
    for entry in top_symbols:
        age_str = f"{entry['age_days']:.0f}" if entry['age_days'] is not None else "NA"
        top_details.append(
            f"{entry['symbol']}(score={entry['score']:.1f},vol={entry['quote_volume']/1e6:.0f}M,"
            f"spread={entry['spread_bps']:.1f}bps,age={age_str},liq={entry['liquidity_factor']:.2f})"
        )
    
    print(f"\n[AI-UNIVERSE] AI_UNIVERSE_TOP10 {' '.join(top_details[:3])}...")
    
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
    
    # Step 6: Save to PolicyStore
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
        print(f"[AI-UNIVERSE] ✅ AI policy generated successfully!")
        print(f"[AI-UNIVERSE] Universe: {len(universe_symbols)} symbols")
        print(f"[AI-UNIVERSE] Leverage range: {min(leverage_by_symbol.values()):.1f}x - {max(leverage_by_symbol.values()):.1f}x")
        print(f"[AI-UNIVERSE] Valid for: 60 minutes")
        print(f"[AI-UNIVERSE] Next refresh: {datetime.fromtimestamp(time.time() + 3600).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Log structured audit trail
        print(f"\n[AI-UNIVERSE] AI_UNIVERSE_SELECTED hash={universe_hash} symbols={','.join(universe_symbols[:3])}... top_score={top_10[0]['score']:.2f}")
        return True
    else:
        print(f"[AI-UNIVERSE] ❌ Failed to save policy to PolicyStore")
        return False


if __name__ == "__main__":
    try:
        success = generate_ai_universe()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[AI-UNIVERSE] ❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
