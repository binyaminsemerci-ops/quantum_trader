"""
Trading Profile System - Quantum Trader

Complete trading profile layer with:
- Liquidity & Universe filtering (volume, spread, depth scoring)
- Position sizing with AI risk factors
- Dynamic TP/SL engine (ATR-based, multi-targets, trailing)
- Funding protection (timing + rate filtering)
- Spread & liquidity guards

Ensures AI trades only on high-quality symbols with optimal risk/reward.

Author: Quantum Trader Team
Date: 2025-11-26
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import datetime as dt
import math
import logging
from enum import Enum

log = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class ShiftType(str, Enum):
    """Covariate shift severity levels."""
    NO_SHIFT = "no_shift"
    MILD_SHIFT = "mild_shift"
    MODERATE_SHIFT = "moderate_shift"
    SEVERE_SHIFT = "severe_shift"


class UniverseTier(str, Enum):
    """Symbol universe tiers based on market cap and liquidity."""
    MAIN = "main"          # BTC, ETH
    L1 = "l1"              # Top L1s (SOL, BNB, ADA, AVAX, DOT, etc.)
    L2 = "l2"              # L2s and major DeFi (ARB, OP, MATIC, UNI, LINK)
    DEFI = "defi"          # DeFi protocols
    INFRASTRUCTURE = "infrastructure"  # Oracles, data, etc.
    MEME = "meme"          # Meme coins (high risk)
    EXCLUDED = "excluded"  # Blacklisted or too risky


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SymbolMetrics:
    """Market data for a single symbol."""
    symbol: str
    quote_volume_24h: float  # USDT volume in 24h
    bid: float  # Best bid price
    ask: float  # Best ask price
    depth_notional_5bps: float  # Orderbook depth within ±0.5% of mid
    funding_rate: float  # Current funding rate (e.g., 0.0001 = 0.01%)
    next_funding_time: dt.datetime  # Next funding timestamp
    mark_price: float  # Mark price
    index_price: float  # Index price
    open_interest: float  # Open interest in USDT
    universe_tier: UniverseTier = UniverseTier.EXCLUDED
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price from bid/ask."""
        return (self.bid + self.ask) / 2.0 if (self.bid > 0 and self.ask > 0) else self.mark_price


@dataclass
class LiquidityConfig:
    """Configuration for liquidity & universe filtering."""
    # Minimum thresholds
    min_quote_volume_24h: float = 5_000_000.0  # $5M minimum daily volume
    max_spread_bps: float = 3.0  # 0.03% = 3 basis points
    min_depth_notional: float = 200_000.0  # $200k depth within ±0.5%
    
    # Liquidity score weights (should sum to ~1.0)
    w_volume: float = 0.5
    w_spread: float = 0.3
    w_depth: float = 0.2
    
    # Score threshold
    min_liquidity_score: float = 0.0  # Can be tuned empirically
    
    # Universe tier filters
    allowed_tiers: List[UniverseTier] = field(default_factory=lambda: [
        UniverseTier.MAIN,
        UniverseTier.L1,
        UniverseTier.L2
    ])
    
    # Top N symbols to trade
    max_universe_size: int = 20  # Trade only top 20 most liquid symbols


@dataclass
class RiskConfig:
    """Configuration for position sizing and risk management."""
    # Base risk per trade
    base_risk_frac: float = 0.01  # 1% of equity per trade
    max_risk_frac: float = 0.03  # 3% maximum per trade
    
    # Position limits
    min_margin: float = 10.0  # Minimum margin in USDT
    max_margin: float = 1000.0  # Maximum margin per position
    
    # Portfolio limits
    max_total_risk_frac: float = 0.15  # 15% total exposure
    max_positions: int = 8  # Maximum concurrent positions
    
    # AI risk factor bounds
    min_ai_risk_factor: float = 0.5  # Conservative AI signal
    max_ai_risk_factor: float = 1.5  # Aggressive AI signal
    
    # Leverage settings (Binance allows up to 125x, we cap lower)
    default_leverage: int = 30  # What we set on Binance
    effective_leverage_main: float = 15.0  # BTC, ETH
    effective_leverage_l1: float = 12.0  # Top L1s
    effective_leverage_l2: float = 10.0  # L2s and DeFi
    effective_leverage_min: float = 8.0  # Minimum for any symbol


@dataclass
class TpslConfig:
    """Configuration for dynamic TP/SL system."""
    # ATR multipliers (R = ATR * atr_mult_base)
    atr_mult_base: float = 1.0  # Base R = 1.0 * ATR
    
    # Stop Loss
    atr_mult_sl: float = 1.0  # SL at 1R
    
    # Take Profit levels
    atr_mult_tp1: float = 1.5  # First target at 1.5R
    atr_mult_tp2: float = 2.5  # Second target / trailing start at 2.5R
    atr_mult_tp3: float = 4.0  # Optional third target at 4R
    
    # Break-even move
    atr_mult_be: float = 1.0  # Move SL to BE when price hits 1R
    be_buffer_bps: float = 5.0  # Add 5 bps buffer above entry for BE
    
    # Trailing stop
    trail_dist_mult: float = 0.8  # Trailing distance = 0.8R
    trail_activation_mult: float = 2.5  # Activate trailing at TP2
    
    # Partial position closes
    partial_close_tp1: float = 0.5  # Close 50% at TP1
    partial_close_tp2: float = 0.3  # Close 30% at TP2 (20% remains for trailing)
    
    # ATR calculation
    atr_period: int = 14  # ATR period
    atr_timeframe: str = "15m"  # Timeframe for ATR calculation


@dataclass
class FundingConfig:
    """Configuration for funding rate protection."""
    # Time windows around funding (Binance funding every 8h)
    pre_window_minutes: int = 40  # Don't enter 40min before funding
    post_window_minutes: int = 20  # Don't enter 20min after funding
    
    # Funding rate thresholds (as decimal, e.g., 0.0003 = 0.03%)
    min_long_funding: float = -0.0003  # Don't LONG if funding < -0.03%
    max_short_funding: float = 0.0003  # Don't SHORT if funding > +0.03%
    
    # Funding rate quality filters
    extreme_funding_threshold: float = 0.001  # 0.1% = extreme
    high_funding_threshold: float = 0.0005  # 0.05% = high


@dataclass
class DynamicTpslLevels:
    """Calculated TP/SL levels for a position."""
    # Stop Loss (required)
    sl_init: float  # Initial stop loss
    
    # Take Profit targets (required)
    tp1: float  # First target (partial close)
    tp2: float  # Second target (partial close)
    
    # Break-even (required)
    be_trigger: float  # Price level that triggers BE move
    be_price: float  # Break-even price (entry + small buffer)
    
    # Trailing stop (required)
    trail_activation: float  # Price level that activates trailing
    trail_distance: float  # Distance from price to maintain
    
    # Optional third target (default)
    tp3: Optional[float] = None  # Optional third target
    
    # Position management (defaults)
    partial_close_frac_tp1: float = 0.5  # Fraction to close at TP1
    partial_close_frac_tp2: float = 0.3  # Fraction to close at TP2
    
    # Risk metrics (defaults)
    risk_r: float = 1.0  # Risk in R units (always 1R)
    reward_r_tp1: float = 1.5  # Reward at TP1 in R units
    reward_r_tp2: float = 2.5  # Reward at TP2 in R units
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API/logging."""
        return {
            'sl_init': round(self.sl_init, 8),
            'tp1': round(self.tp1, 8),
            'tp2': round(self.tp2, 8),
            'tp3': round(self.tp3, 8) if self.tp3 else None,
            'be_trigger': round(self.be_trigger, 8),
            'be_price': round(self.be_price, 8),
            'trail_activation': round(self.trail_activation, 8),
            'trail_distance': round(self.trail_distance, 8),
            'risk_reward_tp1': f"1:{round(self.reward_r_tp1, 2)}",
            'risk_reward_tp2': f"1:{round(self.reward_r_tp2, 2)}",
        }


# ============================================================================
# LIQUIDITY & UNIVERSE FILTERING
# ============================================================================

def compute_spread(symbol: SymbolMetrics) -> float:
    """
    Calculate bid-ask spread as percentage of mid price.
    
    Returns:
        Spread in decimal (e.g., 0.0003 = 0.03% = 3 bps)
    """
    mid = symbol.mid_price
    if mid <= 0 or symbol.bid <= 0 or symbol.ask <= 0:
        return math.inf
    
    spread = (symbol.ask - symbol.bid) / mid
    return max(0.0, spread)


def compute_spread_bps(symbol: SymbolMetrics) -> float:
    """Calculate spread in basis points."""
    return compute_spread(symbol) * 10000


def compute_liquidity_score(symbol: SymbolMetrics, cfg: LiquidityConfig) -> float:
    """
    Calculate composite liquidity score.
    
    Higher is better. Uses log-normalization for volume and depth,
    and inverted log for spread (smaller spread = better).
    
    Returns:
        Liquidity score (unbounded, higher = more liquid)
    """
    spread = compute_spread(symbol)
    
    # Log-normalize volume (higher = better)
    vol_term = math.log10(max(symbol.quote_volume_24h, 1.0))
    
    # Log-normalize depth (higher = better)
    depth_term = math.log10(max(symbol.depth_notional_5bps, 1.0))
    
    # Inverted log spread (smaller spread = better)
    # Add small epsilon to avoid log(0)
    spread_term = -math.log10(max(spread, 1e-6))
    
    score = (
        cfg.w_volume * vol_term +
        cfg.w_depth * depth_term +
        cfg.w_spread * spread_term
    )
    
    return score


def is_symbol_tradeable(
    symbol: SymbolMetrics,
    cfg: LiquidityConfig
) -> Tuple[bool, Optional[str]]:
    """
    Check if symbol meets all liquidity and universe requirements.
    
    Returns:
        (is_tradeable, rejection_reason)
    """
    # Check universe tier
    if symbol.universe_tier not in cfg.allowed_tiers:
        return False, f"Universe tier {symbol.universe_tier.value} not in allowed tiers"
    
    # Check spread
    spread_bps = compute_spread_bps(symbol)
    if spread_bps > cfg.max_spread_bps:
        return False, f"Spread {spread_bps:.2f} bps > {cfg.max_spread_bps:.2f} bps"
    
    # Check volume
    if symbol.quote_volume_24h < cfg.min_quote_volume_24h:
        vol_m = symbol.quote_volume_24h / 1_000_000
        min_m = cfg.min_quote_volume_24h / 1_000_000
        return False, f"Volume ${vol_m:.1f}M < ${min_m:.1f}M"
    
    # Check depth
    if symbol.depth_notional_5bps < cfg.min_depth_notional:
        depth_k = symbol.depth_notional_5bps / 1_000
        min_k = cfg.min_depth_notional / 1_000
        return False, f"Depth ${depth_k:.0f}k < ${min_k:.0f}k"
    
    # Check liquidity score
    score = compute_liquidity_score(symbol, cfg)
    if score < cfg.min_liquidity_score:
        return False, f"Liquidity score {score:.2f} < {cfg.min_liquidity_score:.2f}"
    
    return True, None


def filter_and_rank_universe(
    symbols: List[SymbolMetrics],
    cfg: LiquidityConfig
) -> List[SymbolMetrics]:
    """
    Filter symbols by liquidity criteria and rank by score.
    
    Returns:
        Top N tradeable symbols ranked by liquidity score
    """
    tradeable = []
    
    for symbol in symbols:
        is_ok, reason = is_symbol_tradeable(symbol, cfg)
        if is_ok:
            # Calculate score for ranking
            symbol_with_score = symbol
            score = compute_liquidity_score(symbol, cfg)
            tradeable.append((symbol_with_score, score))
        else:
            log.debug(f"Symbol {symbol.symbol} rejected: {reason}")
    
    # Sort by liquidity score (descending)
    tradeable.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N
    top_symbols = [s for s, _ in tradeable[:cfg.max_universe_size]]
    
    log.info(f"Universe: {len(top_symbols)}/{len(symbols)} symbols tradeable")
    return top_symbols


# ============================================================================
# POSITION SIZING
# ============================================================================

def compute_position_margin(
    equity: float,
    ai_risk_factor: float,
    cfg: RiskConfig
) -> float:
    """
    Calculate position margin in USDT.
    
    Args:
        equity: Account equity in USDT
        ai_risk_factor: AI conviction (0.5 = conservative, 1.5 = aggressive)
        cfg: Risk configuration
    
    Returns:
        Position margin in USDT
    """
    # Clamp AI risk factor to allowed range
    ai_factor = max(
        cfg.min_ai_risk_factor,
        min(cfg.max_ai_risk_factor, ai_risk_factor)
    )
    
    # Calculate raw risk fraction
    raw_risk = cfg.base_risk_frac * ai_factor
    
    # Clamp to [0.5 * base_risk, max_risk]
    risk_frac = max(
        cfg.base_risk_frac * 0.5,
        min(cfg.max_risk_frac, raw_risk)
    )
    
    # Calculate margin
    margin = equity * risk_frac
    
    # Apply min/max bounds
    margin = max(cfg.min_margin, min(cfg.max_margin, margin))
    
    log.debug(
        f"Position sizing: equity=${equity:.0f}, ai_factor={ai_factor:.2f}, "
        f"risk_frac={risk_frac*100:.2f}%, margin=${margin:.2f}"
    )
    
    return margin


def compute_effective_leverage(
    symbol: SymbolMetrics,
    cfg: RiskConfig
) -> float:
    """
    Get effective leverage based on symbol tier.
    
    Higher quality symbols (BTC, ETH) get higher leverage.
    """
    tier_leverage = {
        UniverseTier.MAIN: cfg.effective_leverage_main,
        UniverseTier.L1: cfg.effective_leverage_l1,
        UniverseTier.L2: cfg.effective_leverage_l2,
    }
    
    leverage = tier_leverage.get(symbol.universe_tier, cfg.effective_leverage_min)
    
    # Cap at default Binance leverage setting
    return min(leverage, float(cfg.default_leverage))


def compute_position_size(
    margin: float,
    effective_leverage: float,
    entry_price: float
) -> float:
    """
    Calculate position size in base currency.
    
    Args:
        margin: Position margin in USDT
        effective_leverage: Effective leverage multiplier
        entry_price: Entry price
    
    Returns:
        Position size in base currency (e.g., BTC quantity)
    """
    notional = margin * effective_leverage
    quantity = notional / entry_price
    return quantity


# ============================================================================
# DYNAMIC TP/SL ENGINE
# ============================================================================

def compute_dynamic_tpsl_long(
    entry_price: float,
    atr: float,
    cfg: TpslConfig,
    regime: Optional[str] = None
) -> DynamicTpslLevels:
    """
    Calculate dynamic TP/SL levels for LONG position.
    
    [CRITICAL FIX #2] Dynamic SL widening based on volatility regime:
    - HIGH_VOL: 1.5x stop distance (avoid premature stops)
    - EXTREME_VOL: 2.5x stop distance (flash crash protection)
    
    Args:
        entry_price: Entry price
        atr: Average True Range
        cfg: TP/SL configuration
        regime: Market regime (NORMAL_VOL, HIGH_VOL, EXTREME_VOL)
    
    Returns:
        DynamicTpslLevels with all calculated levels
    """
    # [CRITICAL FIX #2] Apply regime-based SL widening
    sl_multiplier = 1.0
    if regime == "HIGH_VOL":
        sl_multiplier = 1.5
        log.info(f"[FIX #2] HIGH_VOL detected - widening SL by 1.5x")
    elif regime == "EXTREME_VOL":
        sl_multiplier = 2.5
        log.info(f"[FIX #2] EXTREME_VOL detected - widening SL by 2.5x (flash crash protection)")
    
    # Base risk unit
    R = atr * cfg.atr_mult_base
    
    # Stop Loss (with regime-based widening)
    sl = entry_price - (R * cfg.atr_mult_sl * sl_multiplier)
    
    # Take Profit levels
    tp1 = entry_price + (atr * cfg.atr_mult_tp1)
    tp2 = entry_price + (atr * cfg.atr_mult_tp2)
    tp3 = entry_price + (atr * cfg.atr_mult_tp3) if cfg.atr_mult_tp3 else None
    
    # Break-even
    be_trigger = entry_price + (atr * cfg.atr_mult_be)
    be_buffer = entry_price * (cfg.be_buffer_bps / 10000)
    be_price = entry_price + be_buffer
    
    # Trailing stop
    trail_activation = entry_price + (atr * cfg.trail_activation_mult)
    trail_distance = atr * cfg.trail_dist_mult
    
    return DynamicTpslLevels(
        sl_init=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        be_trigger=be_trigger,
        be_price=be_price,
        trail_activation=trail_activation,
        trail_distance=trail_distance,
        partial_close_frac_tp1=cfg.partial_close_tp1,
        partial_close_frac_tp2=cfg.partial_close_tp2,
        risk_r=cfg.atr_mult_sl,
        reward_r_tp1=cfg.atr_mult_tp1,
        reward_r_tp2=cfg.atr_mult_tp2,
    )


def compute_dynamic_tpsl_short(
    entry_price: float,
    atr: float,
    cfg: TpslConfig,
    regime: Optional[str] = None
) -> DynamicTpslLevels:
    """
    Calculate dynamic TP/SL levels for SHORT position.
    
    [CRITICAL FIX #2] Dynamic SL widening based on volatility regime:
    - HIGH_VOL: 1.5x stop distance (avoid premature stops)
    - EXTREME_VOL: 2.5x stop distance (flash crash protection)
    
    Args:
        entry_price: Entry price
        atr: Average True Range
        cfg: TP/SL configuration
        regime: Market regime (NORMAL_VOL, HIGH_VOL, EXTREME_VOL)
    
    Returns:
        DynamicTpslLevels with all calculated levels (inverted for SHORT)
    """
    # [CRITICAL FIX #2] Apply regime-based SL widening
    sl_multiplier = 1.0
    if regime == "HIGH_VOL":
        sl_multiplier = 1.5
        log.info(f"[FIX #2] HIGH_VOL detected - widening SL by 1.5x")
    elif regime == "EXTREME_VOL":
        sl_multiplier = 2.5
        log.info(f"[FIX #2] EXTREME_VOL detected - widening SL by 2.5x (flash crash protection)")
    
    # Base risk unit
    R = atr * cfg.atr_mult_base
    
    # Stop Loss (above entry for SHORT, with regime-based widening)
    sl = entry_price + (R * cfg.atr_mult_sl * sl_multiplier)
    
    # Take Profit levels (below entry for SHORT)
    tp1 = entry_price - (atr * cfg.atr_mult_tp1)
    tp2 = entry_price - (atr * cfg.atr_mult_tp2)
    tp3 = entry_price - (atr * cfg.atr_mult_tp3) if cfg.atr_mult_tp3 else None
    
    # Break-even
    be_trigger = entry_price - (atr * cfg.atr_mult_be)
    be_buffer = entry_price * (cfg.be_buffer_bps / 10000)
    be_price = entry_price - be_buffer
    
    # Trailing stop (below price for SHORT)
    trail_activation = entry_price - (atr * cfg.trail_activation_mult)
    trail_distance = atr * cfg.trail_dist_mult
    
    return DynamicTpslLevels(
        sl_init=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        be_trigger=be_trigger,
        be_price=be_price,
        trail_activation=trail_activation,
        trail_distance=trail_distance,
        partial_close_frac_tp1=cfg.partial_close_tp1,
        partial_close_frac_tp2=cfg.partial_close_tp2,
        risk_r=cfg.atr_mult_sl,
        reward_r_tp1=cfg.atr_mult_tp1,
        reward_r_tp2=cfg.atr_mult_tp2,
    )


# ============================================================================
# FUNDING PROTECTION
# ============================================================================

def is_funding_window_blocked(
    now: dt.datetime,
    symbol: SymbolMetrics,
    cfg: FundingConfig
) -> Tuple[bool, Optional[str]]:
    """
    Check if current time is too close to funding event.
    
    Returns:
        (is_blocked, reason)
    """
    # Calculate time until next funding
    delta = symbol.next_funding_time - now
    minutes_to_funding = delta.total_seconds() / 60.0
    
    # Check pre-window (before funding)
    if 0 <= minutes_to_funding <= cfg.pre_window_minutes:
        return True, f"{minutes_to_funding:.0f}min until funding (pre-window)"
    
    # Check post-window (after funding)
    if -cfg.post_window_minutes <= minutes_to_funding < 0:
        return True, f"{abs(minutes_to_funding):.0f}min after funding (post-window)"
    
    return False, None


def is_funding_rate_unfavourable(
    side: str,
    symbol: SymbolMetrics,
    cfg: FundingConfig
) -> Tuple[bool, Optional[str]]:
    """
    Check if funding rate is unfavourable for the trade direction.
    
    Args:
        side: "LONG" or "SHORT"
        symbol: Symbol metrics with funding rate
        cfg: Funding configuration
    
    Returns:
        (is_unfavourable, reason)
    """
    fr = symbol.funding_rate
    
    # LONG positions pay when funding is positive, receive when negative
    if side.upper() == "LONG":
        if fr < cfg.min_long_funding:
            return True, f"Funding {fr*100:.3f}% too negative for LONG"
    
    # SHORT positions receive when funding is positive, pay when negative
    elif side.upper() == "SHORT":
        if fr > cfg.max_short_funding:
            return True, f"Funding {fr*100:.3f}% too positive for SHORT"
    
    return False, None


def check_funding_protection(
    side: str,
    symbol: SymbolMetrics,
    cfg: FundingConfig,
    now: Optional[dt.datetime] = None
) -> Tuple[bool, Optional[str]]:
    """
    Comprehensive funding protection check.
    
    Returns:
        (is_allowed, rejection_reason)
    """
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    
    # Check timing window
    blocked, reason = is_funding_window_blocked(now, symbol, cfg)
    if blocked:
        return False, reason
    
    # Check funding rate
    unfavourable, reason = is_funding_rate_unfavourable(side, symbol, cfg)
    if unfavourable:
        return False, reason
    
    return True, None


# ============================================================================
# INTEGRATED TRADE VALIDATION
# ============================================================================

def validate_trade(
    symbol: SymbolMetrics,
    side: str,
    liquidity_cfg: LiquidityConfig,
    funding_cfg: FundingConfig,
    now: Optional[dt.datetime] = None
) -> Tuple[bool, Optional[str]]:
    """
    Run all validation checks before allowing a trade.
    
    Returns:
        (is_valid, rejection_reason)
    """
    # Liquidity check
    tradeable, reason = is_symbol_tradeable(symbol, liquidity_cfg)
    if not tradeable:
        return False, f"Liquidity: {reason}"
    
    # Funding check
    allowed, reason = check_funding_protection(side, symbol, funding_cfg, now)
    if not allowed:
        return False, f"Funding: {reason}"
    
    return True, None


# ============================================================================
# UNIVERSE CLASSIFICATION
# ============================================================================

def classify_symbol_tier(symbol: str) -> UniverseTier:
    """
    Classify symbol into universe tier based on predefined lists.
    
    This is a simple rule-based classifier. In production, could be
    enhanced with dynamic market cap / liquidity data.
    """
    symbol = symbol.upper().replace('USDT', '').replace('BUSD', '').replace('USDC', '')
    
    # MAIN tier: BTC, ETH
    if symbol in {'BTC', 'ETH'}:
        return UniverseTier.MAIN
    
    # L1 tier: Major L1 blockchains
    l1_symbols = {
        'SOL', 'BNB', 'ADA', 'AVAX', 'DOT', 'ATOM', 'NEAR', 'FTM', 'ALGO',
        'XRP', 'TRX', 'TON', 'APT', 'SUI', 'SEI', 'INJ'
    }
    if symbol in l1_symbols:
        return UniverseTier.L1
    
    # L2 tier: L2s and major DeFi
    l2_symbols = {
        'ARB', 'OP', 'MATIC', 'IMX', 'METIS', 'MANTA',
        'UNI', 'AAVE', 'LINK', 'MKR', 'CRV', 'LDO', 'SNX'
    }
    if symbol in l2_symbols:
        return UniverseTier.L2
    
    # DeFi tier
    defi_symbols = {
        'COMP', 'SUSHI', 'YFI', 'BAL', 'PERP', 'GMX', 'GNS', 'PENDLE'
    }
    if symbol in defi_symbols:
        return UniverseTier.DEFI
    
    # Infrastructure tier
    infra_symbols = {
        'GRT', 'FIL', 'AR', 'BAND', 'API3', 'OCEAN'
    }
    if symbol in infra_symbols:
        return UniverseTier.INFRASTRUCTURE
    
    # Meme tier (usually excluded)
    meme_symbols = {
        'DOGE', 'SHIB', 'PEPE', 'WIF', 'BONK', 'FLOKI'
    }
    if symbol in meme_symbols:
        return UniverseTier.MEME
    
    # Known problematic symbols (from user's losses)
    excluded_symbols = {
        'TAO', 'PUNDIX', 'ZEC', 'JUP', 'DYM', 'IO', 'NOT', 'LISTA', 'ZRO',
        'OMNI', 'SAGA', 'BB', 'REZ', 'AEVO', 'PORTAL', 'PIXEL', 'STRK'
    }
    if symbol in excluded_symbols:
        return UniverseTier.EXCLUDED
    
    # Default to excluded for unknown symbols
    return UniverseTier.EXCLUDED


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_price(price: float, decimals: int = 8) -> str:
    """Format price with appropriate decimals."""
    return f"{price:.{decimals}f}".rstrip('0').rstrip('.')


def calculate_risk_reward_ratio(
    entry: float,
    sl: float,
    tp: float,
    side: str
) -> float:
    """Calculate risk:reward ratio."""
    if side.upper() == "LONG":
        risk = abs(entry - sl)
        reward = abs(tp - entry)
    else:
        risk = abs(sl - entry)
        reward = abs(entry - tp)
    
    if risk == 0:
        return 0.0
    
    return reward / risk


def estimate_position_pnl(
    entry_price: float,
    current_price: float,
    quantity: float,
    side: str
) -> float:
    """Estimate unrealized PnL."""
    if side.upper() == "LONG":
        pnl = (current_price - entry_price) * quantity
    else:
        pnl = (entry_price - current_price) * quantity
    
    return pnl
