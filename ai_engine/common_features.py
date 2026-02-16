"""
Common Feature Schema - Feb 2026 Ensemble Uniformity
=====================================================

This module defines the canonical 49-feature set used across:
- Feature Publisher Service (feeds Redis streams)
- ML Agents (LightGBM, XGBoost, N-HiTS, PatchTST)
- Retraining Scripts (train_*_v6.py)

CRITICAL: All components MUST use this exact feature list and order.

Feature Categories:
- Candlestick (11): Price action patterns
- Oscillators (7): RSI, MACD, Stochastic, ROC
- EMAs (8): Exponential MAs with distance metrics
- SMAs (2): Simple moving averages
- ADX (3): Average Directional Index system
- Bollinger (5): Bollinger Bands with position
- Volatility (3): ATR and volatility measures
- Volume (5): Volume indicators and OBV
- Momentum (5): Momentum and acceleration
"""

# Complete 49-feature schema (Feb 2026)
FEATURES_V6 = [
    # Candlestick patterns (11 features)
    'returns', 
    'log_returns', 
    'price_range', 
    'body_size', 
    'upper_wick', 
    'lower_wick',
    'is_doji', 
    'is_hammer', 
    'is_engulfing', 
    'gap_up', 
    'gap_down',
    
    # Oscillators (7 features)
    'rsi', 
    'macd', 
    'macd_signal', 
    'macd_hist', 
    'stoch_k', 
    'stoch_d', 
    'roc',
    
    # EMAs with distance (8 features)
    'ema_9', 
    'ema_9_dist', 
    'ema_21', 
    'ema_21_dist', 
    'ema_50', 
    'ema_50_dist', 
    'ema_200', 
    'ema_200_dist',
    
    # SMAs (2 features)
    'sma_20', 
    'sma_50',
    
    # ADX system (3 features)
    'adx', 
    'plus_di', 
    'minus_di',
    
    # Bollinger Bands (5 features)
    'bb_middle', 
    'bb_upper', 
    'bb_lower', 
    'bb_width', 
    'bb_position',
    
    # Volatility measures (3 features)
    'atr', 
    'atr_pct', 
    'volatility',
    
    # Volume indicators (5 features)
    'volume_sma', 
    'volume_ratio', 
    'obv', 
    'obv_ema', 
    'vpt',
    
    # Momentum (5 features)
    'momentum_5', 
    'momentum_10', 
    'momentum_20', 
    'acceleration', 
    'relative_spread'
]

# Safe defaults for missing features
FEATURE_DEFAULTS = {
    # Boolean features
    'is_doji': 0,
    'is_hammer': 0,
    'is_engulfing': 0,
    'gap_up': 0,
    'gap_down': 0,
    
    # RSI neutral
    'rsi': 50,
    
    # Volatility small positive
    'atr_pct': 0.01,
    'volatility': 0.01,
    'relative_spread': 0.01,
}

def get_feature_default(feature_name: str) -> float:
    """Get safe default for a feature."""
    if feature_name in FEATURE_DEFAULTS:
        return FEATURE_DEFAULTS[feature_name]
    
    # Auto-detect
    if feature_name.startswith('is_'):
        return 0  # Boolean features
    elif feature_name == 'rsi':
        return 50  # Neutral
    elif feature_name in ['atr_pct', 'volatility', 'relative_spread']:
        return 0.01  # Small positive
    else:
        return 0.0  # Safe zero

# Validation
assert len(FEATURES_V6) == 49, f"Expected 49 features, got {len(FEATURES_V6)}"
assert len(FEATURES_V6) == len(set(FEATURES_V6)), "Duplicate features detected!"

print(f"✅ Common feature schema loaded: {len(FEATURES_V6)} features")
