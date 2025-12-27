"""
Dashboard API Utilities
Sprint 4 Del 3

Helper functions for dashboard data processing and formatting.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ========== ROUNDING & FORMATTING ==========

def safe_round(value: Optional[float], decimals: int = 2) -> float:
    """
    Safely round a float value, handling None.
    
    Args:
        value: Float value to round (or None)
        decimals: Number of decimal places
        
    Returns:
        Rounded float, or 0.0 if None
    """
    if value is None:
        return 0.0
    return round(float(value), decimals)


def safe_percentage(numerator: float, denominator: float, decimals: int = 2) -> float:
    """
    Safely calculate percentage, handling zero division.
    
    Args:
        numerator: Top value
        denominator: Bottom value
        decimals: Decimal places
        
    Returns:
        (numerator / denominator) * 100, or 0.0 if denominator is 0
    """
    if denominator == 0 or denominator is None:
        return 0.0
    return round((numerator / denominator) * 100, decimals)


# ========== TIMESTAMP HELPERS ==========

def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def is_valid_timestamp(timestamp: Optional[str]) -> bool:
    """Check if timestamp string is valid ISO format."""
    if not timestamp:
        return False
    try:
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False


# ========== DATA EXTRACTION ==========

def safe_get(data: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    """
    Safely get value from dict, handling None dict.
    
    Args:
        data: Dictionary (or None)
        key: Key to extract
        default: Default value if key missing or data is None
        
    Returns:
        data[key] or default
    """
    if data is None:
        return default
    return data.get(key, default)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float, handling NaN and Infinity.
    
    Args:
        value: Value to convert
        default: Default if conversion fails or value is NaN/Inf
        
    Returns:
        Float value or default
    """
    import math
    try:
        result = float(value)
        # Check for NaN or Infinity
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


# ========== FIELD NAME STANDARDIZATION ==========

def standardize_pnl_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize PnL field names to consistent format.
    
    Converts:
    - pnl -> *_pnl
    - pnl_percent -> *_pnl_pct
    - drawdown -> *_drawdown_pct
    
    Args:
        data: Raw data dict
        
    Returns:
        Dict with standardized field names
    """
    # Map old names to new standard names
    field_mapping = {
        "unrealized_pnl": "unrealized_pnl",
        "realized_pnl": "realized_pnl",
        "daily_pnl": "daily_pnl",
        "weekly_pnl": "weekly_pnl",
        "monthly_pnl": "monthly_pnl",
        "total_pnl": "total_pnl",
        # Percentages
        "unrealized_pnl_percent": "unrealized_pnl_pct",
        "daily_pnl_percent": "daily_pnl_pct",
        # Drawdowns
        "daily_dd": "daily_drawdown_pct",
        "weekly_dd": "weekly_drawdown_pct",
        "max_dd": "max_drawdown_pct",
        "daily_dd_pct": "daily_drawdown_pct",
        "weekly_dd_pct": "weekly_drawdown_pct",
        "max_dd_pct": "max_drawdown_pct",
    }
    
    standardized = {}
    for key, value in data.items():
        new_key = field_mapping.get(key, key)
        standardized[new_key] = value
    
    return standardized


# ========== VALIDATION ==========

def validate_snapshot_structure(snapshot_dict: Dict[str, Any]) -> bool:
    """
    Validate that snapshot has required top-level keys.
    
    Args:
        snapshot_dict: Dashboard snapshot as dict
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["timestamp", "portfolio", "positions", "signals", "risk", "system"]
    
    for key in required_keys:
        if key not in snapshot_dict:
            logger.error(f"[VALIDATION] Missing required key: {key}")
            return False
    
    return True
