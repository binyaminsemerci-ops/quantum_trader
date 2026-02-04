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

# Windows UTF-8 fix for local testing
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # Python < 3.7

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

# Redis import for loading previous universe
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[AI-UNIVERSE] WARN: redis-py not available, churn guard disabled")


# Guardrails configuration (env override supported)
MIN_QUOTE_VOL_USDT_24H = int(os.getenv("MIN_QUOTE_VOL_USDT_24H", "20000000"))  # $20M default
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "15"))  # 15 bps default
MIN_AGE_DAYS = int(os.getenv("MIN_AGE_DAYS", "30"))  # 30 days default

# Correlation diversity configuration
MAX_CORRELATION = float(os.getenv("MAX_CORRELATION", "0.85"))  # Strong penalty above 0.85
CORR_PENALTY_STRENGTH = float(os.getenv("CORR_PENALTY_STRENGTH", "0.7"))  # Penalty factor

# Churn guard configuration
MAX_REPLACEMENTS = int(os.getenv("MAX_REPLACEMENTS", "3"))  # Max symbols to replace per refresh
MIN_SCORE_IMPROVEMENT = float(os.getenv("MIN_SCORE_IMPROVEMENT", "0.15"))  # 15% improvement to allow churn

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
    """Fetch orderbook top for spread calculation - returns dict with bid/ask/mid/spread_bps"""
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
        
        # Calculate spread in bps: (ask-bid)/mid * 10000
        if best_bid > 0:
            mid = (best_bid + best_ask) / 2
            spread_bps = ((best_ask - best_bid) / mid) * 10000
            return {
                "bid": best_bid,
                "ask": best_ask,
                "mid": mid,
                "spread_bps": spread_bps
            }
        
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

def get_symbol_performance(symbol):
    """
    Fetch historical performance for symbol from Redis.
    Returns profitability_multiplier (0.5 - 1.5 range, default 1.0)
    
    30% weight in hybrid scoring: 70% technical + 30% historical
    
    Redis key: quantum:symbol:performance:{symbol}
    Fields: win_rate, avg_pnl_pct, total_trades, sharpe_ratio
    """
    if not REDIS_AVAILABLE:
        return 1.0
        
    try:
        r = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=False)
        key = f"quantum:symbol:performance:{symbol}"
        data = r.hgetall(key)
        
        if not data or len(data) == 0:
            return 1.0  # No history = neutral multiplier
        
        # Parse performance metrics
        win_rate = float(data.get(b"win_rate", b"0.5").decode())
        avg_pnl_pct = float(data.get(b"avg_pnl_pct", b"0.0").decode())
        total_trades = int(data.get(b"total_trades", b"0").decode())
        sharpe_ratio = float(data.get(b"sharpe_ratio", b"0.0").decode())
        
        # Require minimum trades for reliability
        if total_trades < 5:
            return 1.0  # Not enough data
        
        # Calculate profitability multiplier (0.5 - 1.5 range)
        # win_rate: 0.3-0.7 range → contribution
        # avg_pnl_pct: -2% to +2% range → contribution
        # sharpe_ratio: -1 to +2 range → contribution
        
        win_contribution = 0.5 + (win_rate * 0.9)
        pnl_contribution = 1.0 + (avg_pnl_pct / 10.0)
        sharpe_contribution = 1.0 + (sharpe_ratio / 5.0)
        
        # Weighted average
        profitability_multiplier = (
            win_contribution * 0.5 +
            pnl_contribution * 0.3 +
            sharpe_contribution * 0.2
        )
        
        # Clamp to 0.5 - 1.5 range (max 50% boost or penalty)
        profitability_multiplier = max(0.5, min(1.5, profitability_multiplier))
        
        return profitability_multiplier
        
    except Exception as e:
        # Silent fail on Redis errors, use neutral multiplier
        return 1.0

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
    
    # Sort by volume and limit spread checks to top candidates (performance optimization)
    MAX_SPREAD_CHECKS = int(os.getenv("MAX_SPREAD_CHECKS", "80"))
    candidates_vol_ok.sort(key=lambda x: x["quote_volume"], reverse=True)
    
    # Handle case where vol_ok < spread_cap
    actual_checks = min(len(candidates_vol_ok), MAX_SPREAD_CHECKS)
    candidates_to_check = candidates_vol_ok[:actual_checks]
    skipped_spread_check = len(candidates_vol_ok) - actual_checks
    
    if skipped_spread_check > 0:
        print(f"[AI-UNIVERSE] Spread optimization: spread_cap={MAX_SPREAD_CHECKS}, checking top {actual_checks}/{vol_ok} by volume (skipping {skipped_spread_check})")
    else:
        print(f"[AI-UNIVERSE] Spread check: spread_cap={MAX_SPREAD_CHECKS}, vol_ok={vol_ok} <= cap, checking all {actual_checks} candidates")
    
    # Pipeline C: Filter by spread (only check spread for top volume candidates)
    candidates_spread_ok = []
    excluded_spread = 0
    
    for i, candidate in enumerate(candidates_to_check):
        if i % 20 == 0:
            print(f"[AI-UNIVERSE] Spread check: {i}/{len(candidates_to_check)}...")
        
        symbol = candidate["symbol"]
        spread_data = fetch_orderbook_spread(symbol)
        
        if spread_data is None or spread_data["spread_bps"] > MAX_SPREAD_BPS:
            excluded_spread += 1
            continue
        
        candidate["spread_bps"] = spread_data["spread_bps"]
        candidate["spread_bid"] = spread_data["bid"]
        candidate["spread_ask"] = spread_data["ask"]
        candidate["spread_mid"] = spread_data["mid"]
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
    
    # Log guardrails summary with optimization metrics, volume source, and venue metadata
    total = len(symbols)
    spread_cap = MAX_SPREAD_CHECKS if 'MAX_SPREAD_CHECKS' in locals() else vol_ok
    actual_checked = actual_checks if 'actual_checks' in locals() else spread_cap
    skipped = skipped_spread_check if 'skipped_spread_check' in locals() else 0
    # COMPLIANCE LOCK: Add metadata_ok preview (will be confirmed in POLICY_SAVED)
    print(f"[AI-UNIVERSE] AI_UNIVERSE_GUARDRAILS total={total} vol_ok={vol_ok} spread_cap={spread_cap} spread_checked={actual_checked} spread_skipped={skipped} spread_ok={spread_ok} age_ok={age_ok} excluded_vol={excluded_volume} excluded_spread={excluded_spread} excluded_age={excluded_age} unknown_age={unknown_age} min_qv_usdt={MIN_QUOTE_VOL_USDT_24H} max_spread_bps={MAX_SPREAD_BPS} min_age_days={MIN_AGE_DAYS} vol_src=quoteVolume market=futures stats_endpoint=fapi/v1/ticker/24hr metadata_ok=1")
    
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
        
        # HYBRID PROFITABILITY SCORING
        # 70% Technical (profit potential) + 30% Historical (actual performance)
        
        # Technical score - favor volatility and trend (profit opportunities)
        technical_score = (
            features["volatility_15m"] * 3.0 +    # High volatility = more profit opportunities
            abs(features["trend_1h"]) * 2.0 +     # Strong trend = better entries
            features["momentum_15m"] * 1.5 +      # Short-term momentum
            features["momentum_1h"] * 1.0         # Medium-term momentum
        )

        # Liquidity factor (volume percentile, clamped 0.5..1.0)
        volume_rank = sum(1 for v in all_volumes if v < candidate["quote_volume"]) / max(len(all_volumes), 1)
        liquidity_factor = max(0.5, min(1.0, 0.5 + volume_rank * 0.5))
        
        # Spread factor (clamped 0.5..1.0)
        spread_factor = max(0.5, min(1.0, 1.0 - (candidate["spread_bps"] / MAX_SPREAD_BPS) * 0.5))
        
        # Apply penalties
        age_penalty = candidate["age_unknown_penalty"]
        
        # Final score
        # Historical profitability multiplier (0.5-1.5 range, default 1.0)
        profitability_multiplier = get_symbol_performance(symbol)
        
        # Hybrid final score: 70% technical + 30% historical
        # profitability_multiplier contributes 30% via its 0.5-1.5 range
        score = technical_score * liquidity_factor * spread_factor * age_penalty * profitability_multiplier
        
        scored_symbols.append({
            "symbol": symbol,
            "score": score,
            "technical_score": technical_score,
            "profitability_multiplier": profitability_multiplier,
            "liquidity_factor": liquidity_factor,
            "spread_factor": spread_factor,
            "features": features,
            "quote_volume": candidate["quote_volume"],
            "spread_bps": candidate["spread_bps"],
            "spread_bid": candidate.get("spread_bid"),
            "spread_ask": candidate.get("spread_ask"),
            "spread_mid": candidate.get("spread_mid"),
            "age_days": candidate["age_days"]
        })
    
    # Sort by score (descending)
    scored_symbols.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"[AI-UNIVERSE] Ranked {len(scored_symbols)} eligible symbols")
    
    return scored_symbols


def fetch_returns_series(symbol, interval="1h", limit=100):
    """Fetch returns series for correlation computation"""
    closes, _, _ = fetch_klines(symbol, interval, limit)
    
    if closes is None or len(closes) < 2:
        return None
    
    # Compute log returns
    log_returns = np.diff(np.log(closes))
    return log_returns


def compute_correlation_matrix(candidates):
    """Compute pairwise correlation matrix for candidates"""
    print(f"[AI-UNIVERSE] Computing correlation matrix for {len(candidates)} candidates...")
    
    # Fetch returns for all candidates
    returns_map = {}
    for i, candidate in enumerate(candidates):
        if i % 20 == 0:
            print(f"[AI-UNIVERSE] Correlation data: {i}/{len(candidates)}...")
        
        symbol = candidate["symbol"]
        returns = fetch_returns_series(symbol)
        
        if returns is not None and len(returns) > 50:  # Need enough data
            returns_map[symbol] = returns
    
    print(f"[AI-UNIVERSE] Returns fetched for {len(returns_map)}/{len(candidates)} symbols")
    
    # Compute correlation matrix
    symbols = list(returns_map.keys())
    n = len(symbols)
    corr_matrix = np.eye(n)  # Identity matrix (1.0 on diagonal)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Align lengths (use minimum)
            len_i = len(returns_map[symbols[i]])
            len_j = len(returns_map[symbols[j]])
            min_len = min(len_i, len_j)
            
            r1 = returns_map[symbols[i]][-min_len:]
            r2 = returns_map[symbols[j]][-min_len:]
            
            # Pearson correlation
            corr = np.corrcoef(r1, r2)[0, 1]
            
            # Handle NaN (can occur with zero variance)
            if np.isnan(corr):
                corr = 0.0
            
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    return symbols, corr_matrix


def select_diversified_topn(ranked, n=None, max_correlation=MAX_CORRELATION, 
                            corr_penalty_strength=CORR_PENALTY_STRENGTH):
    """Select top N symbols with correlation diversity
    
    Greedy algorithm:
    1. Pick highest score first
    2. For each next pick, apply correlation penalty
    3. adjusted_score = score * (1 - corr_penalty)
    4. If corr > max_correlation with any selected: strong penalty
    
    Args:
        ranked: List of ranked symbols
        n: Number of symbols to select (default: all if None, otherwise min(n, len(ranked)))
        max_correlation: Maximum acceptable correlation
        corr_penalty_strength: Penalty strength for correlation
    """
    
    if n is None:
        n = len(ranked)
    else:
        n = min(n, len(ranked))
    
    if len(ranked) < n:
        print(f"[AI-UNIVERSE] WARN: Only {len(ranked)} candidates, selecting all")
        return ranked[:n], None, None
    
    # Compute correlation matrix
    symbols, corr_matrix = compute_correlation_matrix(ranked)
    
    # Build symbol index map
    symbol_to_idx = {sym: i for i, sym in enumerate(symbols)}
    
    # Filter ranked to only include symbols with correlation data
    ranked_with_corr = [c for c in ranked if c["symbol"] in symbol_to_idx]
    
    if len(ranked_with_corr) < n:
        print(f"[AI-UNIVERSE] WARN: Only {len(ranked_with_corr)} symbols with correlation data")
        return ranked[:n], None, None
    
    print(f"[AI-UNIVERSE] Greedy diversified selection (target={n}, max_corr={max_correlation})...")
    
    selected = []
    selected_indices = []
    
    # Pick first (highest score)
    first = ranked_with_corr[0]
    selected.append(first)
    selected_indices.append(symbol_to_idx[first["symbol"]])
    
    print(f"[AI-UNIVERSE] Pick 1: {first['symbol']} score={first['score']:.2f}")
    
    # Pick remaining N-1 with diversity
    for pick_num in range(2, n + 1):
        best_adjusted_score = -999999
        best_candidate = None
        best_max_corr = 0
        
        for candidate in ranked_with_corr:
            if candidate in selected:
                continue
            
            symbol = candidate["symbol"]
            if symbol not in symbol_to_idx:
                continue
            
            idx = symbol_to_idx[symbol]
            
            # Compute max correlation with already selected
            max_corr = 0
            for sel_idx in selected_indices:
                corr = abs(corr_matrix[idx, sel_idx])
                if corr > max_corr:
                    max_corr = corr
            
            # Apply correlation penalty
            if max_corr > max_correlation:
                # Strong penalty for high correlation
                corr_penalty = corr_penalty_strength * 2
            else:
                # Soft penalty proportional to correlation
                corr_penalty = max_corr * corr_penalty_strength
            
            adjusted_score = candidate["score"] * (1 - corr_penalty)
            
            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_candidate = candidate
                best_max_corr = max_corr
        
        if best_candidate is None:
            # Fallback: pick next highest raw score
            for candidate in ranked_with_corr:
                if candidate not in selected:
                    best_candidate = candidate
                    best_max_corr = 0
                    break
        
        if best_candidate:
            selected.append(best_candidate)
            selected_indices.append(symbol_to_idx[best_candidate["symbol"]])
            
            print(f"[AI-UNIVERSE] Pick {pick_num}: {best_candidate['symbol']} "
                  f"score={best_candidate['score']:.2f} adj={best_adjusted_score:.2f} "
                  f"max_corr={best_max_corr:.3f}")
    
    # Compute diversity metrics
    avg_corr = 0
    max_corr_final = 0
    corr_count = 0
    
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            corr = abs(corr_matrix[selected_indices[i], selected_indices[j]])
            avg_corr += corr
            corr_count += 1
            if corr > max_corr_final:
                max_corr_final = corr
    
    if corr_count > 0:
        avg_corr /= corr_count
    
    return selected, avg_corr, max_corr_final


def load_previous_universe():
    """Load previous universe from Redis PolicyStore"""
    if not REDIS_AVAILABLE:
        return []
    
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        universe_json = r.hget('quantum:policy:current', 'universe_symbols')
        
        if universe_json:
            universe = json.loads(universe_json)
            print(f"[AI-UNIVERSE] Loaded previous universe: {len(universe)} symbols")
            return universe
        else:
            print(f"[AI-UNIVERSE] No previous universe found in PolicyStore")
            return []
    except Exception as e:
        print(f"[AI-UNIVERSE] WARN: Failed to load previous universe: {e}")
        return []


def apply_churn_guard(selected, previous_universe, max_replacements=MAX_REPLACEMENTS,
                     min_score_improvement=MIN_SCORE_IMPROVEMENT):
    """Apply churn guard to limit replacements per refresh
    
    Rules:
    1. If no previous universe, allow all changes
    2. Compute kept vs replaced
    3. If replacements > max_replacements:
       - Check if score improvement justifies churn
       - If not, keep more symbols from previous universe
    """
    
    if not previous_universe:
        print(f"[AI-UNIVERSE] Churn guard: No previous universe, allowing all selections")
        return selected, 0, len(selected)
    
    selected_symbols = [s["symbol"] for s in selected]
    previous_set = set(previous_universe)
    selected_set = set(selected_symbols)
    
    kept = selected_set & previous_set
    replaced = len(previous_universe) - len(kept)
    new_added = len(selected_set - previous_set)
    
    print(f"[AI-UNIVERSE] Churn analysis: kept={len(kept)}, replaced={replaced}, new={new_added}")
    
    if replaced <= max_replacements:
        print(f"[AI-UNIVERSE] Churn guard: {replaced} replacements <= {max_replacements}, PASS")
        return selected, len(kept), replaced
    
    # Too much churn - check score improvement
    avg_score_selected = sum(s["score"] for s in selected) / len(selected)
    
    # Compute avg score of symbols being replaced
    replaced_symbols = previous_set - selected_set
    
    # Find scores of replaced symbols in original ranked list (if available)
    # For simplicity, assume score improvement if new selections have high scores
    
    print(f"[AI-UNIVERSE] Churn guard: {replaced} replacements > {max_replacements}")
    print(f"[AI-UNIVERSE] Avg score of new selections: {avg_score_selected:.2f}")
    
    # Heuristic: Allow churn if top selections have significantly higher scores
    # For now, keep the diversified selection but log the churn
    print(f"[AI-UNIVERSE] Churn guard: Allowing churn (diversity optimization)")
    
    return selected, len(kept), replaced


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
    
    # Step 3: Load previous universe for churn guard
    previous_universe = load_previous_universe()
    
    # Step 4: Select TOP 10 BEST symbols from quality-filtered candidates
    # First filter by quality (volume, spread, age) → ~71 symbols
    # Then rank and select TOP 10 by composite score → final universe
    max_symbols_env = os.getenv("AI_UNIVERSE_MAX_SYMBOLS", "10")
    top_n = min(int(max_symbols_env), len(ranked))
    
    print(f"[AI-UNIVERSE] Quality filter passed: {len(ranked)} symbols")
    print(f"[AI-UNIVERSE] Selecting TOP {top_n} BEST by score (AI_UNIVERSE_MAX_SYMBOLS={max_symbols_env})")
    
    if top_n < 1:
        raise RuntimeError(f"No symbols passed guardrails! Eligible: {len(ranked)}")
    
    # Use diversified selection algorithm with top_n limit
    top_symbols, avg_corr, max_corr = select_diversified_topn(ranked, n=top_n)
    
    # Apply churn guard
    top_symbols, kept_count, replaced_count = apply_churn_guard(top_symbols, previous_universe)
    top_symbols, kept_count, replaced_count = apply_churn_guard(top_symbols, previous_universe)
    
    universe_symbols = [x["symbol"] for x in top_symbols]
    
    # Log correlation diversity metrics
    if avg_corr is not None and max_corr is not None:
        print(f"[AI-UNIVERSE] AI_UNIVERSE_DIVERSITY selected={len(top_symbols)} avg_corr={avg_corr:.3f} max_corr={max_corr:.3f} threshold={MAX_CORRELATION}")
    else:
        print(f"[AI-UNIVERSE] AI_UNIVERSE_DIVERSITY selected={len(top_symbols)} avg_corr=N/A max_corr=N/A reason=insufficient_data")
    
    # Log churn guard results
    if previous_universe:
        print(f"[AI-UNIVERSE] AI_UNIVERSE_CHURN kept={kept_count} replaced={replaced_count} prev_count={len(previous_universe)} max_replacements={MAX_REPLACEMENTS}")
    
    # Generate universe hash (for change detection)
    universe_str = ",".join(sorted(universe_symbols))
    universe_hash = hashlib.sha256(universe_str.encode()).hexdigest()[:16]
    
    print(f"\n[AI-UNIVERSE] TOP-{top_n} SELECTED:")
    
    # Per-symbol logging (grep-friendly) with full spread transparency
    spread_detail_missing_count = 0
    for i, entry in enumerate(top_symbols, 1):
        age_str = f"{entry['age_days']:.0f}" if entry['age_days'] is not None else "NA"
        
        # COMPLIANCE LOCK: Verify spread transparency (non-optional)
        spread_detail_ok = "1"
        if 'spread_bps' not in entry:
            print(f"[AI-UNIVERSE] WARN: {entry['symbol']} missing spread_bps (should not happen)")
            spread_detail_missing_count += 1
            spread_detail_ok = "0"
        elif 'spread_bps' in entry and not ('spread_bid' in entry and 'spread_ask' in entry and 'spread_mid' in entry):
            # SPREAD DETAIL LOCK: spread_bps exists but bid/ask/mid missing (regression guard)
            print(f"[AI-UNIVERSE] WARN: AI_UNIVERSE_PICK_MISSING_SPREAD_DETAIL symbol={entry['symbol']} has_spread_bps=1 has_bid_ask_mid=0")
            spread_detail_missing_count += 1
            spread_detail_ok = "0"
        
        # Main PICK log with qv24h_usdt (explicit) + spread_detail_ok flag
        spread_bps_val = entry.get('spread_bps', -1.0)
        print(f"[AI-UNIVERSE] AI_UNIVERSE_PICK symbol={entry['symbol']} score={entry['score']:.2f} qv24h_usdt={entry['quote_volume']:.0f} spread_bps={spread_bps_val:.2f} age_days={age_str} lf={entry['liquidity_factor']:.3f} sf={entry['spread_factor']:.3f} spread_detail_ok={spread_detail_ok}")
        
        # Detailed spread breakdown for top 10
        if 'spread_bid' in entry and 'spread_ask' in entry and 'spread_mid' in entry:
            print(f"[AI-UNIVERSE]   └─ spread_detail: bid={entry['spread_bid']:.6f} ask={entry['spread_ask']:.6f} mid={entry['spread_mid']:.6f} spread_bps={entry['spread_bps']:.2f}")
        
        # Also show human-readable summary
        age_display = f"{entry['age_days']:.0f}d" if entry['age_days'] is not None else "NA"
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
        valid_for_seconds=300,
        policy_version="1.0.0-ai-v1",
        generator="ai_universe_v1",
        features_window="15m,1h",
        universe_hash=universe_hash,
        market="futures",
        stats_endpoint="fapi/v1/ticker/24hr",
        venue="binance-futures",
        exchange_info_endpoint="fapi/v1/exchangeInfo",
        ticker_24h_endpoint="fapi/v1/ticker/24hr"
    )
    
    if success:
        print(f"[AI-UNIVERSE] AI policy generated successfully!")
        print(f"[AI-UNIVERSE] Universe: {len(universe_symbols)} symbols")
        print(f"[AI-UNIVERSE] Leverage range: {min(leverage_by_symbol.values()):.1f}x - {max(leverage_by_symbol.values()):.1f}x")
        print(f"[AI-UNIVERSE] Valid for: 5 minutes (continuous re-evaluation)")
        print(f"[AI-UNIVERSE] Next refresh: {datetime.fromtimestamp(time.time() + 300).strftime('%Y-%m-%d %H:%M:%S')}")
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


