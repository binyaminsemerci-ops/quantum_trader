"""Exit Policy Regime-Specific Configuration.

This module provides regime-specific exit policy parameters that adapt
stop loss and take profit multipliers based on market conditions.

Usage:
    from backend.services.execution.exit_policy_regime_config import get_exit_params
    
    params = get_exit_params("NORMAL_VOL")
    stop_loss = entry - (params.k1_SL * atr)
    take_profit = entry + (params.k2_TP * atr)
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RegimeExitConfig:
    """Exit policy parameters for a specific regime.
    
    Attributes:
        k1_SL: Stop loss ATR multiplier (e.g., 1.5 = 1.5x ATR below entry for LONG)
        k2_TP: Take profit ATR multiplier (e.g., 3.0 = 3x ATR above entry for LONG)
        breakeven_R: R-multiple threshold to move SL to breakeven (e.g., 0.5R)
        trailing_multiplier: ATR multiplier for trailing stop (e.g., 2.0)
        max_duration_hours: Maximum trade duration before forced exit
        description: Human-readable description of the regime strategy
    """
    k1_SL: float
    k2_TP: float
    breakeven_R: float
    trailing_multiplier: float
    max_duration_hours: int
    description: str


# Default exit configurations per regime
DEFAULT_REGIME_CONFIGS: Dict[str, RegimeExitConfig] = {
    "LOW_VOL": RegimeExitConfig(
        k1_SL=1.2,  # Tighter stop in low volatility
        k2_TP=3.5,  # Larger targets (price moves slowly)
        breakeven_R=0.3,  # Quick breakeven protection
        trailing_multiplier=1.8,
        max_duration_hours=48,  # Longer holds in low volatility
        description="Low volatility: tight stops, large targets, patient holds"
    ),
    
    "NORMAL_VOL": RegimeExitConfig(
        k1_SL=1.5,  # Standard stop
        k2_TP=3.0,  # Standard 2:1 risk/reward
        breakeven_R=0.5,  # Standard breakeven
        trailing_multiplier=2.0,
        max_duration_hours=36,
        description="Normal volatility: balanced risk/reward with standard parameters"
    ),
    
    "HIGH_VOL": RegimeExitConfig(
        k1_SL=2.0,  # Wider stop to avoid noise
        k2_TP=4.0,  # Larger targets (price moves fast)
        breakeven_R=0.8,  # Wait for more profit before breakeven
        trailing_multiplier=2.5,
        max_duration_hours=24,  # Shorter holds in high volatility
        description="High volatility: wide stops, large targets, faster exits"
    ),
    
    "EXTREME_VOL": RegimeExitConfig(
        k1_SL=2.5,  # Very wide stop for extreme swings
        k2_TP=5.0,  # Aggressive targets
        breakeven_R=1.0,  # Only move to breakeven after 1R
        trailing_multiplier=3.0,
        max_duration_hours=12,  # Very fast exits
        description="Extreme volatility: very wide stops, aggressive targets, quick exits"
    ),
    
    "TRENDING": RegimeExitConfig(
        k1_SL=1.8,  # Wider stop to stay in trends
        k2_TP=4.5,  # Large targets (trend continuation)
        breakeven_R=0.4,  # Quick protection then let it run
        trailing_multiplier=2.5,  # Wider trailing for trend continuation
        max_duration_hours=72,  # Much longer holds for trends
        description="Trending: wide stops, large targets, long holds for trend continuation"
    ),
    
    "RANGING": RegimeExitConfig(
        k1_SL=1.0,  # Tight stop (quick reversal likely)
        k2_TP=2.0,  # Smaller targets (range-bound)
        breakeven_R=0.3,  # Quick breakeven in ranges
        trailing_multiplier=1.5,
        max_duration_hours=18,  # Shorter holds in ranges
        description="Ranging: tight stops, small targets, quick exits in range-bound markets"
    ),
}


def get_exit_params(
    regime: str,
    custom_configs: Optional[Dict[str, RegimeExitConfig]] = None
) -> RegimeExitConfig:
    """Get exit policy parameters for a given regime.
    
    Args:
        regime: Regime name (e.g., "NORMAL_VOL", "HIGH_VOL", "TRENDING")
        custom_configs: Optional custom configurations to override defaults
        
    Returns:
        RegimeExitConfig with parameters for the regime
        
    Raises:
        ValueError: If regime is not recognized
        
    Example:
        >>> params = get_exit_params("HIGH_VOL")
        >>> print(params.k1_SL, params.k2_TP)
        2.0 4.0
    """
    configs = custom_configs if custom_configs else DEFAULT_REGIME_CONFIGS
    
    if regime not in configs:
        raise ValueError(
            f"Unknown regime '{regime}'. Valid regimes: {list(configs.keys())}"
        )
    
    return configs[regime]


def get_all_regimes() -> list[str]:
    """Get list of all available regime names.
    
    Returns:
        List of regime names
        
    Example:
        >>> regimes = get_all_regimes()
        >>> print(regimes)
        ['LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL', 'EXTREME_VOL', 'TRENDING', 'RANGING']
    """
    return list(DEFAULT_REGIME_CONFIGS.keys())


def load_from_env() -> Dict[str, RegimeExitConfig]:
    """Load regime configurations from environment variables.
    
    Supports overriding default configs via environment variables:
        EXIT_REGIME_NORMAL_VOL_K1_SL=1.5
        EXIT_REGIME_NORMAL_VOL_K2_TP=3.0
        EXIT_REGIME_HIGH_VOL_K1_SL=2.0
        etc.
        
    Returns:
        Dictionary of regime configurations (defaults + env overrides)
        
    Example:
        >>> os.environ["EXIT_REGIME_NORMAL_VOL_K1_SL"] = "2.0"
        >>> configs = load_from_env()
        >>> print(configs["NORMAL_VOL"].k1_SL)
        2.0
    """
    configs = DEFAULT_REGIME_CONFIGS.copy()
    
    # Try to load overrides from environment
    for regime in configs.keys():
        prefix = f"EXIT_REGIME_{regime}_"
        
        # Check for each parameter
        k1_sl_key = f"{prefix}K1_SL"
        k2_tp_key = f"{prefix}K2_TP"
        breakeven_key = f"{prefix}BREAKEVEN_R"
        trailing_key = f"{prefix}TRAILING_MULT"
        max_dur_key = f"{prefix}MAX_DURATION_HOURS"
        
        # Apply overrides if present
        overrides = {}
        if k1_sl_key in os.environ:
            overrides["k1_SL"] = float(os.environ[k1_sl_key])
        if k2_tp_key in os.environ:
            overrides["k2_TP"] = float(os.environ[k2_tp_key])
        if breakeven_key in os.environ:
            overrides["breakeven_R"] = float(os.environ[breakeven_key])
        if trailing_key in os.environ:
            overrides["trailing_multiplier"] = float(os.environ[trailing_key])
        if max_dur_key in os.environ:
            overrides["max_duration_hours"] = int(os.environ[max_dur_key])
        
        # Create new config with overrides if any exist
        if overrides:
            original = configs[regime]
            configs[regime] = RegimeExitConfig(
                k1_SL=overrides.get("k1_SL", original.k1_SL),
                k2_TP=overrides.get("k2_TP", original.k2_TP),
                breakeven_R=overrides.get("breakeven_R", original.breakeven_R),
                trailing_multiplier=overrides.get("trailing_multiplier", original.trailing_multiplier),
                max_duration_hours=overrides.get("max_duration_hours", original.max_duration_hours),
                description=f"{original.description} (env override)"
            )
    
    return configs


def get_risk_reward_ratio(regime: str, custom_configs: Optional[Dict[str, RegimeExitConfig]] = None) -> float:
    """Calculate risk/reward ratio for a regime.
    
    Args:
        regime: Regime name
        custom_configs: Optional custom configurations
        
    Returns:
        Risk/reward ratio (e.g., 2.0 for 2:1 R/R)
        
    Example:
        >>> rr = get_risk_reward_ratio("NORMAL_VOL")
        >>> print(rr)
        2.0
    """
    params = get_exit_params(regime, custom_configs)
    return params.k2_TP / params.k1_SL


def validate_config(config: RegimeExitConfig) -> list[str]:
    """Validate a regime configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
        
    Example:
        >>> config = RegimeExitConfig(k1_SL=1.5, k2_TP=3.0, breakeven_R=0.5,
        ...                           trailing_multiplier=2.0, max_duration_hours=24,
        ...                           description="Test")
        >>> errors = validate_config(config)
        >>> print(errors)
        []
    """
    errors = []
    
    # Check k1_SL
    if config.k1_SL <= 0:
        errors.append("k1_SL must be positive")
    elif config.k1_SL < 0.5 or config.k1_SL > 5.0:
        errors.append("k1_SL should be between 0.5 and 5.0 (typical range: 1.0-2.5)")
    
    # Check k2_TP
    if config.k2_TP <= 0:
        errors.append("k2_TP must be positive")
    elif config.k2_TP < 1.0 or config.k2_TP > 10.0:
        errors.append("k2_TP should be between 1.0 and 10.0 (typical range: 2.0-5.0)")
    
    # Check risk/reward ratio
    if config.k2_TP <= config.k1_SL:
        errors.append(f"Risk/reward ratio {config.k2_TP/config.k1_SL:.2f} is too low (should be >1.0)")
    
    # Check breakeven_R
    if config.breakeven_R < 0:
        errors.append("breakeven_R cannot be negative")
    elif config.breakeven_R > 2.0:
        errors.append("breakeven_R should be <= 2.0 (typical range: 0.3-1.0)")
    
    # Check trailing_multiplier
    if config.trailing_multiplier <= 0:
        errors.append("trailing_multiplier must be positive")
    elif config.trailing_multiplier < 0.5 or config.trailing_multiplier > 5.0:
        errors.append("trailing_multiplier should be between 0.5 and 5.0")
    
    # Check max_duration_hours
    if config.max_duration_hours <= 0:
        errors.append("max_duration_hours must be positive")
    elif config.max_duration_hours > 168:  # 1 week
        errors.append("max_duration_hours should be <= 168 (1 week)")
    
    return errors


def print_all_configs():
    """Print all regime configurations in a readable format.
    
    Example:
        >>> print_all_configs()
        Regime: LOW_VOL
          Description: Low volatility: tight stops, large targets, patient holds
          k1_SL: 1.2, k2_TP: 3.5 (R/R: 2.92)
          Breakeven: 0.3R, Trailing: 1.8x ATR
          Max Duration: 48 hours
        ...
    """
    configs = load_from_env()
    
    print("\n=== Exit Policy Regime Configurations ===\n")
    for regime, config in configs.items():
        rr = config.k2_TP / config.k1_SL
        print(f"Regime: {regime}")
        print(f"  Description: {config.description}")
        print(f"  k1_SL: {config.k1_SL}, k2_TP: {config.k2_TP} (R/R: {rr:.2f})")
        print(f"  Breakeven: {config.breakeven_R}R, Trailing: {config.trailing_multiplier}x ATR")
        print(f"  Max Duration: {config.max_duration_hours} hours")
        
        # Show any validation warnings
        errors = validate_config(config)
        if errors:
            print(f"  [WARNING]  Warnings: {', '.join(errors)}")
        
        print()


if __name__ == "__main__":
    # Demo usage
    print_all_configs()
    
    # Example integration
    print("\n=== Example Integration ===\n")
    regime = "HIGH_VOL"
    params = get_exit_params(regime)
    
    entry_price = 50000.0
    atr = 1000.0
    
    stop_loss = entry_price - (params.k1_SL * atr)
    take_profit = entry_price + (params.k2_TP * atr)
    
    print(f"Trade Setup for {regime}:")
    print(f"  Entry: ${entry_price:,.0f}")
    print(f"  Stop Loss: ${stop_loss:,.0f} ({params.k1_SL}x ATR)")
    print(f"  Take Profit: ${take_profit:,.0f} ({params.k2_TP}x ATR)")
    print(f"  Risk/Reward: {params.k2_TP/params.k1_SL:.2f}")
    print(f"  Move to breakeven at: {params.breakeven_R}R profit")
    print(f"  Max hold time: {params.max_duration_hours} hours")
