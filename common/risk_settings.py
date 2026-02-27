"""
common/risk_settings.py

Central risk configuration for the Quantum Trader system.
Used by:
  - harvest_brain._get_harvest_theta()
  - exit_math.compute_dynamic_stop()

R-targets scale INVERSELY with sqrt(leverage):
  R_effective = R_base / sqrt(leverage)
  At leverage=1:  T1=2R, T2=4R, T3=6R
  At leverage=50: T1=0.28R, T2=0.57R, T3=0.85R
"""

import math
from dataclasses import dataclass


@dataclass
class RiskSettings:
    # Position sizing
    RISK_FRACTION: float = 0.005       # 0.5% of equity per trade

    # Stop distance
    STOP_ATR_MULT: float = 1.2         # ATR multiplier for stop distance
    TRAILING_ATR_MULT: float = 1.5     # ATR multiplier for trailing stop
    TRAILING_ACTIVATION_R: float = 1.0 # Activate trailing at 1R profit
    MAX_HOLD_TIME: float = 86400.0     # 24h max hold (seconds)
    LIQ_BUFFER_PCT: float = 0.02       # Close if within 2% of liquidation

    # Harvest R-targets at leverage=1x (scaled down for higher leverage)
    HARVEST_T1_R_BASE: float = 2.0     # Trigger PARTIAL_25
    HARVEST_T2_R_BASE: float = 4.0     # Trigger PARTIAL_50
    HARVEST_T3_R_BASE: float = 6.0     # Trigger PARTIAL_75
    HARVEST_LOCK_R_BASE: float = 1.5   # Move SL to break-even+
    HARVEST_BE_PLUS_PCT: float = 0.002 # 0.2% above breakeven
    HARVEST_KILL_THRESHOLD: float = 0.6


DEFAULT_SETTINGS = RiskSettings()


def compute_harvest_r_targets(leverage: float, settings=None) -> dict:
    """
    Return leverage-scaled harvest R-targets.

    At leverage= 1: T1=2.00R, T2=4.00R, T3=6.00R
    At leverage= 9: T1=0.67R, T2=1.33R, T3=2.00R
    At leverage=10: T1=0.63R, T2=1.26R, T3=1.90R
    At leverage=30: T1=0.37R, T2=0.73R, T3=1.10R
    At leverage=50: T1=0.28R, T2=0.57R, T3=0.85R
    """
    if settings is None:
        settings = DEFAULT_SETTINGS

    scale = math.sqrt(max(float(leverage), 1.0))

    return {
        "T1_R":           settings.HARVEST_T1_R_BASE  / scale,
        "T2_R":           settings.HARVEST_T2_R_BASE  / scale,
        "T3_R":           settings.HARVEST_T3_R_BASE  / scale,
        "lock_R":         settings.HARVEST_LOCK_R_BASE / scale,
        "be_plus_pct":    settings.HARVEST_BE_PLUS_PCT,
        "kill_threshold": settings.HARVEST_KILL_THRESHOLD,
    }
