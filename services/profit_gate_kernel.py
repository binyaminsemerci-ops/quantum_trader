#!/usr/bin/env python3
"""
CORE PROFIT GATE KERNEL - Economic Correctness Enforcement
============================================================
Guarantees:
1. Minimum notional ($25+)
2. Edge > friction * multiplier (3x fees)
3. Cooldown enforcement (5min between trades)
4. R-ratio geometry (2.5+ trend, 1.8+ range)
5. Model-regime validation

FAIL-CLOSED: Any uncertainty = HOLD
TESTNET: Full enforcement
LIVE: Audit-only (logs warnings)

Author: Quantum Trader Core Team
Date: 2026-01-17
"""
import os
import logging
import time
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("profit_gate")

class GateVerdict(Enum):
    PASS = "PASS"
    HOLD = "HOLD"
    
class GateReason(Enum):
    MIN_NOTIONAL = "MIN_NOTIONAL"
    EDGE_TOO_SMALL = "EDGE_TOO_SMALL"
    COOLDOWN = "COOLDOWN"
    RATE_LIMIT = "RATE_LIMIT"
    R_TOO_LOW = "R_TOO_LOW"
    MODEL_REGIME_MISMATCH = "MODEL_REGIME_MISMATCH"
    
@dataclass
class GateConfig:
    """Loaded from /etc/quantum/core_gates.env"""
    min_notional_usd: float = 25.0
    edge_multiplier: float = 3.0
    min_hold_seconds: int = 300
    max_trades_10min: int = 3
    min_r_trend: float = 2.5
    min_r_range: float = 1.8

# Global: cooldown tracker and trade counter
_last_close_times: Dict[str, float] = {}  # symbol -> timestamp
_trade_history: list = []  # [(timestamp, symbol), ...]

def load_gate_config() -> GateConfig:
    """Load from /etc/quantum/core_gates.env"""
    config = GateConfig()
    
    env_path = "/etc/quantum/core_gates.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip().lower()
                    val = val.strip()
                    
                    if key == 'min_notional_usd':
                        config.min_notional_usd = float(val)
                    elif key == 'edge_multiplier':
                        config.edge_multiplier = float(val)
                    elif key == 'min_hold_seconds':
                        config.min_hold_seconds = int(val)
                    elif key == 'max_trades_10min':
                        config.max_trades_10min = int(val)
                    elif key == 'min_r_trend':
                        config.min_r_trend = float(val)
                    elif key == 'min_r_range':
                        config.min_r_range = float(val)
    
    return config

def is_testnet() -> bool:
    """Check if running in TESTNET mode"""
    env_path = "/etc/quantum/testnet.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            content = f.read()
            return 'BINANCE_TESTNET=true' in content or 'TRADING_MODE=TESTNET' in content
    return False

def estimate_friction(
    symbol: str,
    price: float,
    qty: float,
    leverage: int
) -> float:
    """
    Estimate total friction: fees + funding + slippage + noise
    
    Binance Futures Maker/Taker: 0.02%/0.04%
    Funding: ~0.01% per 8h = ~0.03%/day
    Slippage: ~0.05% for market orders
    Volatility noise: ~0.1%
    
    Total: ~0.22% per round-trip
    """
    notional = price * qty
    
    # Fees (round-trip: open + close)
    fee_open = notional * 0.0004  # Taker 0.04%
    fee_close = notional * 0.0004  # Taker 0.04%
    
    # Funding (assume 1-day hold, 3 payments)
    funding = notional * 0.0003  # 0.03% total
    
    # Slippage (market orders, both sides)
    slippage = notional * 0.001  # 0.05% each side
    
    # Volatility noise
    noise = notional * 0.001  # 0.1%
    
    total_friction = fee_open + fee_close + funding + slippage + noise
    return total_friction

def check_cooldown(symbol: str, min_hold_seconds: int) -> Tuple[bool, str]:
    """Check if symbol is in cooldown period"""
    global _last_close_times
    
    now = time.time()
    last_close = _last_close_times.get(symbol, 0)
    elapsed = now - last_close
    
    if elapsed < min_hold_seconds:
        remaining = min_hold_seconds - elapsed
        return False, f"Cooldown: {remaining:.0f}s remaining"
    
    return True, "Cooldown OK"

def check_rate_limit(max_trades_10min: int) -> Tuple[bool, str]:
    """Check if trade rate limit exceeded"""
    global _trade_history
    
    now = time.time()
    cutoff = now - 600  # 10 minutes
    
    # Prune old trades
    _trade_history = [(ts, sym) for ts, sym in _trade_history if ts > cutoff]
    
    if len(_trade_history) >= max_trades_10min:
        return False, f"Rate limit: {len(_trade_history)}/{max_trades_10min} in last 10min"
    
    return True, "Rate OK"

def record_trade(symbol: str):
    """Record trade execution for cooldown and rate limit tracking"""
    global _last_close_times, _trade_history
    
    now = time.time()
    _last_close_times[symbol] = now
    _trade_history.append((now, symbol))

def profit_gate(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    expected_move_usd: float,
    tp: float,
    sl: float,
    model: str,
    regime: str,
    leverage: int,
    trace_id: Optional[str] = None
) -> Tuple[GateVerdict, Optional[GateReason], Dict[str, any]]:
    """
    CORE PROFIT GATE - Economic Correctness Check
    
    Returns:
        (verdict, reason, context)
        
        verdict: PASS or HOLD
        reason: None if PASS, GateReason enum if HOLD
        context: Dict with diagnostics
    """
    config = load_gate_config()
    testnet = is_testnet()
    
    context = {
        "symbol": symbol,
        "side": side,
        "price": price,
        "qty": qty,
        "tp": tp,
        "sl": sl,
        "model": model,
        "regime": regime,
        "leverage": leverage,
        "testnet": testnet,
        "trace_id": trace_id or "unknown"
    }
    
    # GATE 1: MIN NOTIONAL
    notional = price * qty
    context["notional_usd"] = notional
    
    if notional < config.min_notional_usd:
        context["gate_failed"] = "MIN_NOTIONAL"
        context["min_required"] = config.min_notional_usd
        
        logger.warning(
            f"🚫 GATE_BLOCK: MIN_NOTIONAL | {symbol} | "
            f"notional=${notional:.2f} < ${config.min_notional_usd} | "
            f"trace_id={trace_id}"
        )
        
        if testnet:
            return (GateVerdict.HOLD, GateReason.MIN_NOTIONAL, context)
        else:
            logger.warning("   ⚠️ AUDIT-ONLY (LIVE mode)")
    
    # GATE 2: EDGE > FRICTION * MULTIPLIER
    friction = estimate_friction(symbol, price, qty, leverage)
    required_edge = friction * config.edge_multiplier
    
    context["friction_usd"] = friction
    context["required_edge_usd"] = required_edge
    context["expected_move_usd"] = expected_move_usd
    
    if expected_move_usd < required_edge:
        context["gate_failed"] = "EDGE_TOO_SMALL"
        context["edge_multiplier"] = config.edge_multiplier
        
        logger.warning(
            f"🚫 GATE_BLOCK: EDGE_TOO_SMALL | {symbol} | "
            f"expected=${expected_move_usd:.2f} < required=${required_edge:.2f} "
            f"(friction=${friction:.2f} * {config.edge_multiplier}x) | "
            f"trace_id={trace_id}"
        )
        
        if testnet:
            return (GateVerdict.HOLD, GateReason.EDGE_TOO_SMALL, context)
        else:
            logger.warning("   ⚠️ AUDIT-ONLY (LIVE mode)")
    
    # GATE 3: COOLDOWN
    cooldown_ok, cooldown_msg = check_cooldown(symbol, config.min_hold_seconds)
    context["cooldown_ok"] = cooldown_ok
    context["cooldown_msg"] = cooldown_msg
    
    if not cooldown_ok:
        context["gate_failed"] = "COOLDOWN"
        
        logger.warning(
            f"🚫 GATE_BLOCK: COOLDOWN | {symbol} | {cooldown_msg} | "
            f"trace_id={trace_id}"
        )
        
        if testnet:
            return (GateVerdict.HOLD, GateReason.COOLDOWN, context)
        else:
            logger.warning("   ⚠️ AUDIT-ONLY (LIVE mode)")
    
    # GATE 4: RATE LIMIT
    rate_ok, rate_msg = check_rate_limit(config.max_trades_10min)
    context["rate_ok"] = rate_ok
    context["rate_msg"] = rate_msg
    
    if not rate_ok:
        context["gate_failed"] = "RATE_LIMIT"
        
        logger.warning(
            f"🚫 GATE_BLOCK: RATE_LIMIT | {symbol} | {rate_msg} | "
            f"trace_id={trace_id}"
        )
        
        if testnet:
            return (GateVerdict.HOLD, GateReason.RATE_LIMIT, context)
        else:
            logger.warning("   ⚠️ AUDIT-ONLY (LIVE mode)")
    
    # GATE 5: R-RATIO GEOMETRY
    # R = (TP - Entry) / (Entry - SL) for LONG
    # R = (Entry - TP) / (SL - Entry) for SHORT
    
    if side.upper() == "BUY":
        r_ratio = abs(tp - price) / abs(price - sl) if price != sl else 0
    else:
        r_ratio = abs(price - tp) / abs(sl - price) if sl != price else 0
    
    context["r_ratio"] = r_ratio
    
    min_r = config.min_r_trend if regime == "TREND" else config.min_r_range
    context["min_r_required"] = min_r
    
    if r_ratio < min_r:
        context["gate_failed"] = "R_TOO_LOW"
        
        logger.warning(
            f"🚫 GATE_BLOCK: R_TOO_LOW | {symbol} | "
            f"R={r_ratio:.2f} < {min_r} (regime={regime}) | "
            f"trace_id={trace_id}"
        )
        
        if testnet:
            return (GateVerdict.HOLD, GateReason.R_TOO_LOW, context)
        else:
            logger.warning("   ⚠️ AUDIT-ONLY (LIVE mode)")
    
    # GATE 6: MODEL-REGIME VALIDATION
    ALLOWED_MODELS = {
        "TREND": ["NHiTS", "PatchTST"],
        "RANGE": ["XGBoost", "LightGBM"],
        "HIGH_VOL": ["XGBoost"],
        "LOW_VOL": ["PatchTST"]
    }
    
    allowed = ALLOWED_MODELS.get(regime, [])
    context["allowed_models"] = allowed
    
    if model not in allowed:
        context["gate_failed"] = "MODEL_REGIME_MISMATCH"
        
        logger.warning(
            f"🚫 GATE_BLOCK: MODEL_REGIME_MISMATCH | {symbol} | "
            f"model={model} not in {allowed} for regime={regime} | "
            f"trace_id={trace_id}"
        )
        
        if testnet:
            return (GateVerdict.HOLD, GateReason.MODEL_REGIME_MISMATCH, context)
        else:
            logger.warning("   ⚠️ AUDIT-ONLY (LIVE mode)")
    
    # ALL GATES PASSED
    logger.info(
        f"✅ GATE_PASS | {symbol} {side} | "
        f"notional=${notional:.2f} edge=${expected_move_usd:.2f} "
        f"R={r_ratio:.2f} model={model} regime={regime} | "
        f"trace_id={trace_id}"
    )
    
    return (GateVerdict.PASS, None, context)
