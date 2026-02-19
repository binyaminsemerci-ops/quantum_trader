"""
Exit Math Module - Formula-Based Exit Logic

This module provides leverage-aware, risk-normalized exit calculations
that replace all hardcoded percentage thresholds.

NO HARDCODED PERCENTAGES ALLOWED.
ALL exit decisions must be based on:
- Account equity
- Position leverage
- Market volatility (ATR)
- Risk capital allocation
- Time in trade
- Distance to liquidation

Author: Quantum Trader System
Created: 2026-02-18
"""

from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    side: str  # "BUY" or "SELL" (or "LONG"/"SHORT")
    entry_price: float
    size: float  # Position size in contracts/units
    leverage: float
    highest_price: float  # For trailing stop (LONG)
    lowest_price: float  # For trailing stop (SHORT)
    time_in_trade: float  # Seconds since entry
    distance_to_liq: float  # % distance to liquidation price


@dataclass
class Account:
    """Account data structure"""
    equity: float  # Total account equity in USDT


@dataclass
class Market:
    """Market data structure"""
    current_price: float
    atr: float  # Average True Range


@dataclass
class RiskSettings:
    """Risk configuration - NO HARDCODED VALUES IN CODE"""
    RISK_FRACTION: float = 0.005  # 0.5% of equity per trade
    STOP_ATR_MULT: float = 1.2  # Stop distance = 1.2x ATR minimum
    TRAILING_ATR_MULT: float = 1.5  # Trailing distance = 1.5x ATR
    TRAILING_ACTIVATION_R: float = 1.0  # Activate trailing at 1R profit
    MAX_HOLD_TIME: float = 3600  # Maximum 1 hour hold (configurable)
    LIQ_BUFFER_PCT: float = 0.02  # Close if within 2% of liquidation
    

def compute_dynamic_stop(
    position: Position,
    account: Account,
    market: Market,
    settings: RiskSettings
) -> float:
    """
    Compute dynamic stop loss based on risk capital and volatility.
    
    Formula Logic:
    1. Risk capital = equity * risk_fraction
    2. Price risk distance = risk_capital / (size * leverage)
    3. Volatility floor = ATR * stop_atr_mult
    4. Stop distance = max(price_risk_distance, volatility_floor)
    
    This ensures:
    - Stop loss respects account risk limits
    - Stop loss adapts to market volatility
    - Higher leverage = tighter stop (automatic risk normalization)
    
    Args:
        position: Position data
        account: Account equity
        market: Market data (ATR)
        settings: Risk configuration
    
    Returns:
        Stop loss price level
    """
    # Calculate risk capital allocation
    risk_capital = account.equity * settings.RISK_FRACTION
    
    # Calculate price distance needed to lose risk_capital
    # loss = distance * size * leverage
    # distance = loss / (size * leverage)
    if position.size <= 0 or position.leverage <= 0:
        logger.warning(f"Invalid position size ({position.size}) or leverage ({position.leverage})")
        # Fallback to ATR-based stop
        price_risk_distance = market.atr * settings.STOP_ATR_MULT
    else:
        price_risk_distance = risk_capital / (position.size * position.leverage)
    
    # Volatility floor: stop must be at least N x ATR away
    volatility_floor = market.atr * settings.STOP_ATR_MULT
    
    # Use the larger of the two (wider stop)
    stop_distance = max(price_risk_distance, volatility_floor)
    
    # Calculate actual stop price
    if position.side in ("BUY", "LONG"):
        stop_price = position.entry_price - stop_distance
    else:  # SELL/SHORT
        stop_price = position.entry_price + stop_distance
    
    logger.debug(
        f"[{position.symbol}] Dynamic stop calculated: "
        f"risk_capital=${risk_capital:.2f}, "
        f"price_risk_dist=${price_risk_distance:.4f}, "
        f"volatility_floor=${volatility_floor:.4f}, "
        f"stop_distance=${stop_distance:.4f}, "
        f"stop_price=${stop_price:.4f}"
    )
    
    return stop_price


def compute_R(position: Position, price: float, stop_distance: float) -> float:
    """
    Compute R-multiple (risk-normalized profit).
    
    R = profit / initial_risk
    
    For LONG: R = (price - entry) / stop_distance
    For SHORT: R = (entry - price) / stop_distance
    
    Examples:
    - R = 0.0 → at entry
    - R = 1.0 → profit = 1x initial risk
    - R = 2.5 → profit = 2.5x initial risk
    - R = -1.0 → loss = initial risk (at stop)
    
    Args:
        position: Position data
        price: Current market price
        stop_distance: Initial stop distance
    
    Returns:
        R-multiple
    """
    if stop_distance <= 0:
        logger.warning(f"Invalid stop_distance: {stop_distance}")
        return 0.0
    
    if position.side in ("BUY", "LONG"):
        r_value = (price - position.entry_price) / stop_distance
    else:  # SELL/SHORT
        r_value = (position.entry_price - price) / stop_distance
    
    return r_value


def compute_trailing_hit(
    position: Position,
    market: Market,
    settings: RiskSettings
) -> bool:
    """
    Check if trailing stop has been hit.
    
    Formula Logic:
    1. Trail distance = ATR * trailing_atr_mult
    2. For LONG: price dropped more than trail_distance from highest
    3. For SHORT: price rose more than trail_distance from lowest
    
    This creates a volatility-adaptive trailing stop that:
    - Widens in volatile markets (larger ATR)
    - Tightens in calm markets (smaller ATR)
    - No hardcoded percentages
    
    Args:
        position: Position data with highest/lowest price
        market: Market data (ATR, current price)
        settings: Risk configuration
    
    Returns:
        True if trailing stop hit
    """
    trail_distance = market.atr * settings.TRAILING_ATR_MULT
    
    if position.side in ("BUY", "LONG"):
        # Check if price fell more than trail_distance from peak
        trailing_level = position.highest_price - trail_distance
        hit = market.current_price <= trailing_level
        
        if hit:
            logger.info(
                f"[{position.symbol}] LONG trailing stop HIT: "
                f"highest=${position.highest_price:.4f}, "
                f"current=${market.current_price:.4f}, "
                f"trail_dist=${trail_distance:.4f}"
            )
    else:  # SELL/SHORT
        # Check if price rose more than trail_distance from bottom
        trailing_level = position.lowest_price + trail_distance
        hit = market.current_price >= trailing_level
        
        if hit:
            logger.info(
                f"[{position.symbol}] SHORT trailing stop HIT: "
                f"lowest=${position.lowest_price:.4f}, "
                f"current=${market.current_price:.4f}, "
                f"trail_dist=${trail_distance:.4f}"
            )
    
    return hit


def near_liquidation(position: Position, settings: RiskSettings) -> bool:
    """
    Check if position is dangerously close to liquidation.
    
    Formula Logic:
    If distance_to_liquidation <= LIQ_BUFFER_PCT, close immediately.
    
    Example: If LIQ_BUFFER_PCT = 2% and position is within 2% of
    liquidation price, emergency close is triggered.
    
    Args:
        position: Position data with distance_to_liq
        settings: Risk configuration
    
    Returns:
        True if too close to liquidation
    """
    # Skip check if liquidation distance not available
    if position.distance_to_liq is None:
        return False
    
    if position.distance_to_liq <= settings.LIQ_BUFFER_PCT:
        logger.warning(
            f"[{position.symbol}] LIQUIDATION RISK: "
            f"distance_to_liq={position.distance_to_liq*100:.2f}% "
            f"(threshold={settings.LIQ_BUFFER_PCT*100:.2f}%)"
        )
        return True
    
    return False


def should_activate_trailing(
    position: Position,
    market: Market,
    stop_distance: float,
    settings: RiskSettings
) -> bool:
    """
    Check if trailing stop should be activated.
    
    Trailing only activates when position is profitable beyond
    TRAILING_ACTIVATION_R (e.g., 1.0R = 100% of initial risk).
    
    Args:
        position: Position data
        market: Market data
        stop_distance: Initial stop distance
        settings: Risk configuration
    
    Returns:
        True if should activate trailing
    """
    current_r = compute_R(position, market.current_price, stop_distance)
    
    if current_r >= settings.TRAILING_ACTIVATION_R:
        logger.debug(
            f"[{position.symbol}] Trailing stop ACTIVATED at {current_r:.2f}R"
        )
        return True
    
    return False


def evaluate_exit(
    position: Position,
    account: Account,
    market: Market,
    settings: RiskSettings
) -> Optional[str]:
    """
    Master exit evaluation function.
    
    Evaluates ALL exit conditions in priority order:
    1. Liquidation protection (highest priority)
    2. Dynamic stop loss
    3. Trailing stop (if activated)
    4. Time exit
    
    NO TAKE PROFIT - let trailing stops capture upside.
    
    Args:
        position: Position data
        account: Account equity
        market: Market data
        settings: Risk configuration
    
    Returns:
        Exit reason string if should close, None otherwise
    """
    # 1. LIQUIDATION PROTECTION (highest priority)
    if near_liquidation(position, settings):
        return "liq_protection"
    
    # 2. DYNAMIC STOP LOSS
    dynamic_stop = compute_dynamic_stop(position, account, market, settings)
    stop_distance = abs(position.entry_price - dynamic_stop)
    
    if position.side in ("BUY", "LONG"):
        if market.current_price <= dynamic_stop:
            return "risk_stop"
    else:  # SELL/SHORT
        if market.current_price >= dynamic_stop:
            return "risk_stop"
    
    # 3. TRAILING STOP (if in profit)
    if should_activate_trailing(position, market, stop_distance, settings):
        if compute_trailing_hit(position, market, settings):
            return "trailing_stop"
    
    # 4. TIME EXIT
    if position.time_in_trade > settings.MAX_HOLD_TIME:
        logger.info(
            f"[{position.symbol}] Time exit triggered: "
            f"{position.time_in_trade:.0f}s > {settings.MAX_HOLD_TIME:.0f}s"
        )
        return "time_exit"
    
    # No exit condition met
    return None


def get_exit_metrics(
    position: Position,
    account: Account,
    market: Market,
    settings: RiskSettings
) -> Dict:
    """
    Get all exit-related metrics for logging/monitoring.
    
    Returns dictionary with:
    - dynamic_stop: Current stop price
    - stop_distance: Distance from entry to stop
    - current_r: Current R-multiple
    - trailing_active: Whether trailing is active
    - distance_to_liq: Distance to liquidation
    - time_in_trade: Seconds in trade
    """
    dynamic_stop = compute_dynamic_stop(position, account, market, settings)
    stop_distance = abs(position.entry_price - dynamic_stop)
    current_r = compute_R(position, market.current_price, stop_distance)
    trailing_active = should_activate_trailing(position, market, stop_distance, settings)
    
    return {
        "dynamic_stop": dynamic_stop,
        "stop_distance": stop_distance,
        "current_r": current_r,
        "trailing_active": trailing_active,
        "distance_to_liq": position.distance_to_liq,
        "time_in_trade": position.time_in_trade,
        "atr": market.atr,
    }
