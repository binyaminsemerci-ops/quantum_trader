"""
AdaptiveLeverageEngine Configuration
Tunable parameters for production optimization
"""

# Base TP/SL percentages (decimal format: 0.01 = 1%)
BASE_TP_PCT = 0.01    # 1.0% - Base take profit
BASE_SL_PCT = 0.005   # 0.5% - Base stop loss

# Fail-safe clamps (DO NOT change unless absolutely necessary)
SL_CLAMP_MIN = 0.001  # 0.1% - Minimum stop loss
SL_CLAMP_MAX = 0.02   # 2.0% - Maximum stop loss
TP_MIN = 0.003        # 0.3% - Minimum take profit
SL_MIN = 0.0015       # 0.15% - Minimum stop loss

# Scaling factors for adjustments (conservative defaults)
FUNDING_TP_SCALE = 0.8      # How much funding rate affects TP (0.8 = 80%)
DIVERGENCE_SL_SCALE = 0.4   # How much divergence widens SL (0.4 = 40%)
VOLATILITY_SL_SCALE = 0.2   # How much volatility widens SL (0.2 = 20%)

# Harvest schemes per leverage tier
# Format: [TP1_ratio, TP2_ratio, TP3_ratio] (must sum to 1.0)
HARVEST_LOW_LEVERAGE = [0.3, 0.3, 0.4]    # ≤10x: Conservative
HARVEST_MID_LEVERAGE = [0.4, 0.4, 0.2]    # ≤30x: Aggressive
HARVEST_HIGH_LEVERAGE = [0.5, 0.3, 0.2]   # >30x: Ultra-aggressive

# Leverage tier thresholds
LEVERAGE_TIER_1 = 10   # Low leverage threshold
LEVERAGE_TIER_2 = 30   # Mid leverage threshold

# Monitoring and alerting
ENABLE_ADAPTIVE_STREAM = True   # Publish to Redis stream
ENABLE_CLAMP_WARNINGS = True    # Log warnings when clamps trigger
ENABLE_LSF_LOGGING = True       # Log LSF calculations

# Performance tuning (advanced)
# These affect how aggressive the adaptive engine is
LSF_SENSITIVITY = 1.0           # Multiplier for LSF effect (1.0 = default)
TP_PROGRESSION_FACTOR = [0.6, 1.2, 1.8]  # TP level progression multipliers
SL_LEVERAGE_FACTOR = 0.8        # How much (1-LSF) affects SL widening


def get_config() -> dict:
    """Get configuration as dictionary"""
    return {
        'base_tp_pct': BASE_TP_PCT,
        'base_sl_pct': BASE_SL_PCT,
        'sl_clamp_min': SL_CLAMP_MIN,
        'sl_clamp_max': SL_CLAMP_MAX,
        'tp_min': TP_MIN,
        'sl_min': SL_MIN,
        'funding_tp_scale': FUNDING_TP_SCALE,
        'divergence_sl_scale': DIVERGENCE_SL_SCALE,
        'volatility_sl_scale': VOLATILITY_SL_SCALE,
        'harvest_schemes': {
            'low': HARVEST_LOW_LEVERAGE,
            'mid': HARVEST_MID_LEVERAGE,
            'high': HARVEST_HIGH_LEVERAGE
        },
        'leverage_tiers': {
            'tier1': LEVERAGE_TIER_1,
            'tier2': LEVERAGE_TIER_2
        },
        'monitoring': {
            'enable_adaptive_stream': ENABLE_ADAPTIVE_STREAM,
            'enable_clamp_warnings': ENABLE_CLAMP_WARNINGS,
            'enable_lsf_logging': ENABLE_LSF_LOGGING
        },
        'advanced': {
            'lsf_sensitivity': LSF_SENSITIVITY,
            'tp_progression_factor': TP_PROGRESSION_FACTOR,
            'sl_leverage_factor': SL_LEVERAGE_FACTOR
        }
    }


def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    # Check base percentages
    if BASE_TP_PCT <= 0 or BASE_TP_PCT > 0.5:
        errors.append("BASE_TP_PCT must be between 0 and 0.5 (0-50%)")
    if BASE_SL_PCT <= 0 or BASE_SL_PCT > 0.5:
        errors.append("BASE_SL_PCT must be between 0 and 0.5 (0-50%)")
    
    # Check clamps
    if not (SL_CLAMP_MIN < SL_CLAMP_MAX):
        errors.append("SL_CLAMP_MIN must be less than SL_CLAMP_MAX")
    if TP_MIN <= 0 or SL_MIN <= 0:
        errors.append("TP_MIN and SL_MIN must be positive")
    
    # Check harvest schemes sum to 1.0
    for name, scheme in [
        ('HARVEST_LOW_LEVERAGE', HARVEST_LOW_LEVERAGE),
        ('HARVEST_MID_LEVERAGE', HARVEST_MID_LEVERAGE),
        ('HARVEST_HIGH_LEVERAGE', HARVEST_HIGH_LEVERAGE)
    ]:
        if abs(sum(scheme) - 1.0) > 0.001:
            errors.append(f"{name} must sum to 1.0 (current: {sum(scheme)})")
    
    # Check leverage tiers
    if LEVERAGE_TIER_1 >= LEVERAGE_TIER_2:
        errors.append("LEVERAGE_TIER_1 must be less than LEVERAGE_TIER_2")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# Validate on import
validate_config()


if __name__ == "__main__":
    import json
    print("AdaptiveLeverageEngine Configuration")
    print("=" * 60)
    config = get_config()
    print(json.dumps(config, indent=2))
    print("\n✅ Configuration valid")
