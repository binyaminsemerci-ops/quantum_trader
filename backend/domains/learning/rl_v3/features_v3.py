"""
Feature extraction for RL v3 observations.
"""

import numpy as np
from typing import Dict, Any


def build_feature_vector(obs_dict: Dict[str, Any]) -> np.ndarray:
    """
    Build feature vector from observation dictionary.
    
    Args:
        obs_dict: Dictionary with keys:
            - price_change_1m, price_change_5m, price_change_15m
            - volatility, rsi, macd
            - position_size, position_side
            - balance, equity
            - regime (int or str)
            
    Returns:
        np.ndarray of shape (feature_dim,) as float32
    """
    features = []
    
    # Price changes (3)
    features.append(obs_dict.get("price_change_1m", 0.0))
    features.append(obs_dict.get("price_change_5m", 0.0))
    features.append(obs_dict.get("price_change_15m", 0.0))
    
    # Technical indicators (3)
    features.append(obs_dict.get("volatility", 0.02))
    features.append(obs_dict.get("rsi", 50.0) / 100.0)  # Normalize to [0,1]
    features.append(obs_dict.get("macd", 0.0))
    
    # Position info (2)
    features.append(obs_dict.get("position_size", 0.0))
    features.append(obs_dict.get("position_side", 0.0))  # -1, 0, +1
    
    # Account metrics (2)
    balance = obs_dict.get("balance", 10000.0)
    equity = obs_dict.get("equity", 10000.0)
    features.append(equity / balance if balance > 0 else 1.0)
    features.append((equity - balance) / balance if balance > 0 else 0.0)  # PnL ratio
    
    # Regime encoding (4) - one-hot for 4 regimes
    regime = obs_dict.get("regime", 0)
    if isinstance(regime, str):
        regime_map = {"TREND": 0, "RANGE": 1, "BREAKOUT": 2, "MEAN_REVERSION": 3}
        regime = regime_map.get(regime, 0)
    
    regime_onehot = [0.0, 0.0, 0.0, 0.0]
    if 0 <= regime < 4:
        regime_onehot[regime] = 1.0
    features.extend(regime_onehot)
    
    # Additional market features (4)
    features.append(obs_dict.get("trend_strength", 0.0))
    features.append(obs_dict.get("volume_ratio", 1.0))
    features.append(obs_dict.get("bid_ask_spread", 0.001))
    features.append(obs_dict.get("time_of_day", 0.5))  # 0.0-1.0
    
    # Padding to reach 64 dimensions
    while len(features) < 64:
        features.append(0.0)
    
    # Clip to first 64 features
    features = features[:64]
    
    return np.array(features, dtype=np.float32)


def normalize_features(features: np.ndarray, clip_range: float = 5.0) -> np.ndarray:
    """
    Normalize and clip features.
    
    Args:
        features: Raw feature vector
        clip_range: Clip values to [-clip_range, +clip_range]
        
    Returns:
        Normalized features
    """
    # Clip outliers
    features = np.clip(features, -clip_range, clip_range)
    
    # Simple standardization (mean=0, std≈1 for typical values)
    # This is a simplified version; in production, use running statistics
    mean = 0.0
    std = 1.0
    
    normalized = (features - mean) / (std + 1e-8)
    
    return normalized.astype(np.float32)
