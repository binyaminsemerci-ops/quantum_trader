"""
Risk Settings - Centralized Configuration for Exit Logic

ALL exit parameters must be defined here.
NO HARDCODED VALUES allowed in exit logic code.

This file is the SINGLE SOURCE OF TRUTH for:
- Risk capital allocation
- Stop loss calculations
- Trailing stop behavior
- Time-based exits
- Liquidation protection

Modify these values to tune system-wide exit behavior.

Author: Quantum Trader System
Created: 2026-02-18
"""

import os
from dataclasses import dataclass


@dataclass
class ExitRiskSettings:
    """
    Exit risk configuration parameters.
    
    All values can be overridden via environment variables.
    """
    
    # === RISK CAPITAL ALLOCATION ===
    # Fraction of equity to risk per trade
    # Default: 0.5% of account per trade
    # Higher = wider stops, lower = tighter stops
    RISK_FRACTION: float = float(os.getenv("EXIT_RISK_FRACTION", "0.005"))
    
    # === DYNAMIC STOP LOSS ===
    # Stop distance multiplier relative to ATR
    # Stop will be max(risk_capital_distance, ATR * STOP_ATR_MULT)
    # Default: 1.2x ATR minimum
    # Higher = wider stops in volatile markets
    STOP_ATR_MULT: float = float(os.getenv("EXIT_STOP_ATR_MULT", "1.2"))
    
    # === TRAILING STOP ===
    # Trailing stop distance multiplier relative to ATR
    # Trail distance = ATR * TRAILING_ATR_MULT
    # Default: 1.5x ATR
    # Higher = wider trailing stops (let profits run more)
    TRAILING_ATR_MULT: float = float(os.getenv("EXIT_TRAILING_ATR_MULT", "1.5"))
    
    # R-multiple threshold to activate trailing stop
    # Default: 1.0R (activate after 100% of initial risk is profit)
    # Lower = activate trailing sooner
    TRAILING_ACTIVATION_R: float = float(os.getenv("EXIT_TRAILING_ACTIVATION_R", "1.0"))
    
    # === TIME-BASED EXIT ===
    # Maximum hold time in seconds
    # Default: 3600 seconds (1 hour)
    # Increase for swing trading, decrease for scalping
    MAX_HOLD_TIME: float = float(os.getenv("EXIT_MAX_HOLD_TIME", "3600"))
    
    # === LIQUIDATION PROTECTION ===
    # Emergency close if within X% of liquidation price
    # Default: 2% (close if liquidation price is within 2%)
    # Higher = earlier emergency exits
    LIQ_BUFFER_PCT: float = float(os.getenv("EXIT_LIQ_BUFFER_PCT", "0.02"))
    
    # === SHADOW VALIDATION MODE ===
    # Enable logging of old vs new exit decisions
    # Set to True during validation period
    SHADOW_MODE: bool = os.getenv("EXIT_SHADOW_MODE", "False").lower() == "true"
    
    # === ATR CALCULATION ===
    # Period for ATR calculation
    # Default: 14 periods
    ATR_PERIOD: int = int(os.getenv("EXIT_ATR_PERIOD", "14"))
    
    # Timeframe for ATR calculation
    # Default: 5m (5-minute candles)
    ATR_TIMEFRAME: str = os.getenv("EXIT_ATR_TIMEFRAME", "5m")
    
    # === HARVEST BRAIN CONFIGURATION ===
    # Dynamic R-ladder for partial profit taking
    # These scale inversely with leverage (higher lev = lower R targets)
    
    # Partial 25% profit take at T1_R
    # Default: 2.0R baseline (at 1x leverage)
    # Scales as: T1_R_effective = T1_R_BASE / sqrt(leverage)
    HARVEST_T1_R_BASE: float = float(os.getenv("HARVEST_T1_R_BASE", "2.0"))
    
    # Partial 50% profit take at T2_R
    # Default: 4.0R baseline (at 1x leverage)
    HARVEST_T2_R_BASE: float = float(os.getenv("HARVEST_T2_R_BASE", "4.0"))
    
    # Partial 75% profit take at T3_R
    # Default: 6.0R baseline (at 1x leverage)
    HARVEST_T3_R_BASE: float = float(os.getenv("HARVEST_T3_R_BASE", "6.0"))
    
    # Move stop to breakeven at lock_R
    # Default: 1.5R baseline (at 1x leverage)
    HARVEST_LOCK_R_BASE: float = float(os.getenv("HARVEST_LOCK_R_BASE", "1.5"))
    
    # Breakeven plus offset
    # Default: 0.2% above breakeven
    HARVEST_BE_PLUS_PCT: float = float(os.getenv("HARVEST_BE_PLUS_PCT", "0.002"))
    
    # Kill threshold (close entire position when kill_score exceeds this)
    # Default: 0.6 (60% kill score triggers full close)
    HARVEST_KILL_THRESHOLD: float = float(os.getenv("HARVEST_KILL_THRESHOLD", "0.6"))


# Global instance - import this in exit logic modules
DEFAULT_SETTINGS = ExitRiskSettings()


def get_settings() -> ExitRiskSettings:
    """Get current exit risk settings"""
    return DEFAULT_SETTINGS


def validate_settings(settings: ExitRiskSettings) -> bool:
    """
    Validate settings are within safe bounds.
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    errors = []
    
    # Risk fraction must be positive and reasonable
    if not (0.001 <= settings.RISK_FRACTION <= 0.02):
        errors.append(
            f"RISK_FRACTION must be between 0.1% and 2% (got {settings.RISK_FRACTION*100:.2f}%)"
        )
    
    # ATR multipliers must be positive
    if settings.STOP_ATR_MULT <= 0:
        errors.append(f"STOP_ATR_MULT must be positive (got {settings.STOP_ATR_MULT})")
    
    if settings.TRAILING_ATR_MULT <= 0:
        errors.append(f"TRAILING_ATR_MULT must be positive (got {settings.TRAILING_ATR_MULT})")
    
    # Trailing activation R must be non-negative
    if settings.TRAILING_ACTIVATION_R < 0:
        errors.append(
            f"TRAILING_ACTIVATION_R must be non-negative (got {settings.TRAILING_ACTIVATION_R})"
        )
    
    # Max hold time must be positive
    if settings.MAX_HOLD_TIME <= 0:
        errors.append(f"MAX_HOLD_TIME must be positive (got {settings.MAX_HOLD_TIME})")
    
    # Liquidation buffer must be positive
    if not (0.01 <= settings.LIQ_BUFFER_PCT <= 0.10):
        errors.append(
            f"LIQ_BUFFER_PCT must be between 1% and 10% (got {settings.LIQ_BUFFER_PCT*100:.2f}%)"
        )
    
    # Harvest R-targets must be positive and in ascending order
    if settings.HARVEST_T1_R_BASE <= 0:
        errors.append(f"HARVEST_T1_R_BASE must be positive (got {settings.HARVEST_T1_R_BASE})")
    
    if settings.HARVEST_T2_R_BASE <= settings.HARVEST_T1_R_BASE:
        errors.append(
            f"HARVEST_T2_R_BASE ({settings.HARVEST_T2_R_BASE}) must be > "
            f"HARVEST_T1_R_BASE ({settings.HARVEST_T1_R_BASE})"
        )
    
    if settings.HARVEST_T3_R_BASE <= settings.HARVEST_T2_R_BASE:
        errors.append(
            f"HARVEST_T3_R_BASE ({settings.HARVEST_T3_R_BASE}) must be > "
            f"HARVEST_T2_R_BASE ({settings.HARVEST_T2_R_BASE})"
        )
    
    if settings.HARVEST_LOCK_R_BASE <= 0:
        errors.append(f"HARVEST_LOCK_R_BASE must be positive (got {settings.HARVEST_LOCK_R_BASE})")
    
    if not (0.0001 <= settings.HARVEST_BE_PLUS_PCT <= 0.01):
        errors.append(
            f"HARVEST_BE_PLUS_PCT must be between 0.01% and 1% "
            f"(got {settings.HARVEST_BE_PLUS_PCT*100:.2f}%)"
        )
    
    if not (0.3 <= settings.HARVEST_KILL_THRESHOLD <= 1.0):
        errors.append(
            f"HARVEST_KILL_THRESHOLD must be between 0.3 and 1.0 "
            f"(got {settings.HARVEST_KILL_THRESHOLD})"
        )
    
    if errors:
        raise ValueError("Invalid exit risk settings:\n" + "\n".join(errors))
    
    return True


def compute_harvest_r_targets(leverage: float, settings: ExitRiskSettings = None) -> dict:
    """
    Compute leverage-aware R-targets for harvest brain.
    
    R-targets scale inversely with leverage:
        R_effective = R_base / sqrt(leverage)
    
    This ensures higher leverage positions take profits sooner
    to protect against rapid liquidation risk.
    
    Args:
        leverage: Position leverage (e.g., 10.0)
        settings: Risk settings (uses DEFAULT_SETTINGS if None)
    
    Returns:
        dict with keys: T1_R, T2_R, T3_R, lock_R, be_plus_pct, kill_threshold
    
    Example:
        At 1x leverage: T1=2.0R, T2=4.0R, T3=6.0R
        At 4x leverage: T1=1.0R, T2=2.0R, T3=3.0R
        At 10x leverage: T1=0.63R, T2=1.26R, T3=1.90R
    """
    if settings is None:
        settings = DEFAULT_SETTINGS
    
    if leverage <= 0:
        leverage = 1.0  # Fallback to 1x
    
    # Scale factor: higher leverage = lower R-targets
    import math
    scale = 1.0 / math.sqrt(leverage)
    
    return {
        "T1_R": settings.HARVEST_T1_R_BASE * scale,
        "T2_R": settings.HARVEST_T2_R_BASE * scale,
        "T3_R": settings.HARVEST_T3_R_BASE * scale,
        "lock_R": settings.HARVEST_LOCK_R_BASE * scale,
        "be_plus_pct": settings.HARVEST_BE_PLUS_PCT,
        "kill_threshold": settings.HARVEST_KILL_THRESHOLD
    }


# Validate on import
validate_settings(DEFAULT_SETTINGS)


if __name__ == "__main__":
    """Print current settings"""
    settings = get_settings()
    print("=" * 60)
    print("EXIT RISK SETTINGS")
    print("=" * 60)
    print(f"Risk Fraction:         {settings.RISK_FRACTION*100:.2f}% of equity per trade")
    print(f"Stop ATR Multiplier:   {settings.STOP_ATR_MULT}x ATR")
    print(f"Trailing ATR Mult:     {settings.TRAILING_ATR_MULT}x ATR")
    print(f"Trailing Activation:   {settings.TRAILING_ACTIVATION_R}R profit")
    print(f"Max Hold Time:         {settings.MAX_HOLD_TIME:.0f} seconds")
    print(f"Liquidation Buffer:    {settings.LIQ_BUFFER_PCT*100:.1f}%")
    print(f"Shadow Mode:           {settings.SHADOW_MODE}")
    print(f"ATR Period:            {settings.ATR_PERIOD}")
    print(f"ATR Timeframe:         {settings.ATR_TIMEFRAME}")
    print()
    print("HARVEST BRAIN CONFIGURATION")
    print("=" * 60)
    print(f"T1_R Baseline:         {settings.HARVEST_T1_R_BASE}R (partial 25%)")
    print(f"T2_R Baseline:         {settings.HARVEST_T2_R_BASE}R (partial 50%)")
    print(f"T3_R Baseline:         {settings.HARVEST_T3_R_BASE}R (partial 75%)")
    print(f"Lock_R Baseline:       {settings.HARVEST_LOCK_R_BASE}R (move to BE+)")
    print(f"BE Plus Offset:        {settings.HARVEST_BE_PLUS_PCT*100:.2f}%")
    print(f"Kill Threshold:        {settings.HARVEST_KILL_THRESHOLD}")
    print()
    print("LEVERAGE-SCALED R-TARGETS EXAMPLES:")
    print("=" * 60)
    for lev in [1, 5, 10, 20]:
        targets = compute_harvest_r_targets(lev, settings)
        print(f"{lev:2}x Leverage: T1={targets['T1_R']:.2f}R | "
              f"T2={targets['T2_R']:.2f}R | T3={targets['T3_R']:.2f}R | "
              f"Lock={targets['lock_R']:.2f}R")
    print("=" * 60)
