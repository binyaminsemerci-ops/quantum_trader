"""
TP Dashboard API Models

Response models for TP analytics dashboard endpoints.
Provides clean, frontend-ready data structures for TP performance metrics,
profiles, and optimizer recommendations.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime


class TPLegInfo(BaseModel):
    """
    Single leg in a TP profile.
    
    Frontend can render as a table row or card showing:
    - TP1 @ 0.5R (15% hard)
    - TP2 @ 1.0R (20% soft)
    etc.
    """
    label: str = Field(..., description="Human-readable label (e.g., 'TP1', 'TP2')")
    r_multiple: float = Field(..., description="Risk multiple (e.g., 1.0 = 1R)")
    size_fraction: float = Field(..., description="Portion of position (0.0-1.0)")
    kind: Literal["HARD", "SOFT"] = Field(..., description="Execution type: HARD (market) or SOFT (limit)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "TP1",
                "r_multiple": 1.0,
                "size_fraction": 0.30,
                "kind": "HARD"
            }
        }


class TrailingInfo(BaseModel):
    """
    Trailing stop configuration.
    
    Frontend can show trailing parameters and tightening curve.
    """
    callback_pct: float = Field(..., description="Initial trailing callback percentage (e.g., 0.02 = 2%)")
    activation_r: float = Field(..., description="R multiple to start trailing (e.g., 0.5)")
    tightening_curve: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Steps to tighten callback: [{r_threshold: 2.0, callback_pct: 0.01}]"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "callback_pct": 0.02,
                "activation_r": 0.5,
                "tightening_curve": [
                    {"r_threshold": 2.0, "callback_pct": 0.015},
                    {"r_threshold": 4.0, "callback_pct": 0.010}
                ]
            }
        }


class TPProfileInfo(BaseModel):
    """
    Complete TP profile information.
    
    Frontend can render:
    - Profile name header
    - Table of TP legs
    - Trailing configuration panel
    """
    profile_id: str = Field(..., description="Profile identifier (e.g., 'TREND_DEFAULT')")
    legs: List[TPLegInfo] = Field(..., description="Ordered list of TP legs")
    trailing: Optional[TrailingInfo] = Field(None, description="Trailing configuration (if enabled)")
    description: str = Field("", description="Profile purpose description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "profile_id": "TREND_DEFAULT",
                "legs": [
                    {"label": "TP1", "r_multiple": 0.5, "size_fraction": 0.15, "kind": "SOFT"},
                    {"label": "TP2", "r_multiple": 1.0, "size_fraction": 0.20, "kind": "HARD"},
                    {"label": "TP3", "r_multiple": 2.0, "size_fraction": 0.30, "kind": "HARD"}
                ],
                "trailing": {
                    "callback_pct": 0.02,
                    "activation_r": 1.5,
                    "tightening_curve": []
                },
                "description": "Trend-following: Let profits run with wide trailing"
            }
        }


class TPMetricsInfo(BaseModel):
    """
    TP performance metrics for a strategy/symbol pair.
    
    Frontend can render as dashboard table row with columns:
    - Strategy | Symbol | Hit Rate | Attempts | Avg R | Slippage | Time to TP | Profit
    """
    strategy_id: str = Field(..., description="Strategy identifier")
    symbol: str = Field(..., description="Trading symbol")
    
    # Hit rate metrics
    tp_hit_rate: float = Field(..., description="TP hit rate (0.0-1.0)")
    tp_attempts: int = Field(..., description="Total TP attempts")
    tp_hits: int = Field(0, description="Successful TP hits")
    tp_misses: int = Field(0, description="TP misses")
    
    # R multiple (estimated from hit rate or actual)
    avg_r_multiple_winners: Optional[float] = Field(
        None,
        description="Average R multiple per winning trade"
    )
    
    # Slippage metrics
    avg_slippage_pct: Optional[float] = Field(
        None,
        description="Average slippage percentage"
    )
    max_slippage_pct: Optional[float] = Field(
        None,
        description="Maximum slippage observed"
    )
    
    # Timing metrics
    avg_time_to_tp_minutes: Optional[float] = Field(
        None,
        description="Average time to reach TP (minutes)"
    )
    
    # Premature exits
    premature_exit_rate: Optional[float] = Field(
        None,
        description="Rate of premature exits before TP (0.0-1.0)"
    )
    
    # Profit metrics
    total_tp_profit_usd: Optional[float] = Field(
        None,
        description="Total profit from TP hits (USD)"
    )
    
    # Metadata
    last_updated: Optional[datetime] = Field(
        None,
        description="Timestamp of last metric update"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "strategy_id": "RL_V3",
                "symbol": "BTCUSDT",
                "tp_hit_rate": 0.62,
                "tp_attempts": 45,
                "tp_hits": 28,
                "tp_misses": 17,
                "avg_r_multiple_winners": 1.61,
                "avg_slippage_pct": 0.008,
                "max_slippage_pct": 0.025,
                "avg_time_to_tp_minutes": 38.5,
                "premature_exit_rate": 0.12,
                "total_tp_profit_usd": 1245.50,
                "last_updated": "2025-12-10T15:30:00Z"
            }
        }


class TPRecommendationInfo(BaseModel):
    """
    TPOptimizer adjustment recommendation.
    
    Frontend can render as:
    - Badge or alert showing recommendation exists
    - Detail panel with reason, scale factor, metrics snapshot
    """
    profile_id: str = Field(..., description="Current profile being used")
    suggested_scale_factor: float = Field(
        ...,
        description="Suggested multiplier for R multiples (0.9=closer, 1.1=further)"
    )
    direction: Literal["CLOSER", "FURTHER", "NO_CHANGE"] = Field(
        ...,
        description="Direction to adjust TPs"
    )
    reason: str = Field(..., description="Human-readable explanation")
    confidence: float = Field(..., description="Confidence in recommendation (0.0-1.0)")
    metrics_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metrics that led to recommendation"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "profile_id": "TREND_DEFAULT",
                "suggested_scale_factor": 0.95,
                "direction": "CLOSER",
                "reason": "Hit rate 42% below target 45%, but avg R 2.38 acceptable. Bringing TPs closer.",
                "confidence": 0.67,
                "metrics_snapshot": {
                    "tp_hit_rate": 0.42,
                    "avg_r_multiple": 2.38,
                    "tp_attempts": 45,
                    "tp_hits": 19,
                    "tp_misses": 26
                }
            }
        }


class TPDashboardRow(BaseModel):
    """
    Complete TP analytics for a single strategy/symbol pair.
    
    Frontend can render as:
    - Table row with key metrics
    - Expandable detail showing profile legs and recommendations
    """
    strategy_id: str = Field(..., description="Strategy identifier")
    symbol: str = Field(..., description="Trading symbol")
    regime: str = Field("NORMAL", description="Market regime used for profile selection")
    
    metrics: TPMetricsInfo = Field(..., description="Performance metrics")
    profile: Optional[TPProfileInfo] = Field(
        None,
        description="Current TP profile (null if not configured)"
    )
    recommendation: Optional[TPRecommendationInfo] = Field(
        None,
        description="Optimizer recommendation (null if no adjustment needed)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "strategy_id": "RL_V3",
                "symbol": "BTCUSDT",
                "regime": "TREND",
                "metrics": {
                    "strategy_id": "RL_V3",
                    "symbol": "BTCUSDT",
                    "tp_hit_rate": 0.62,
                    "tp_attempts": 45,
                    "tp_hits": 28,
                    "tp_misses": 17,
                    "avg_r_multiple_winners": 1.61
                },
                "profile": {
                    "profile_id": "TREND_DEFAULT",
                    "legs": [
                        {"label": "TP1", "r_multiple": 0.5, "size_fraction": 0.15, "kind": "SOFT"}
                    ],
                    "trailing": None,
                    "description": "Trend profile"
                },
                "recommendation": None
            }
        }


class TPDashboardSummary(BaseModel):
    """
    Complete TP dashboard summary for all tracked pairs.
    
    Frontend can render as:
    - Table with all rows
    - Filters for strategy/symbol/regime
    - Sort by hit rate, attempts, profit, etc.
    """
    rows: List[TPDashboardRow] = Field(..., description="List of strategy/symbol pairs")
    total_pairs: int = Field(..., description="Total number of tracked pairs")
    generated_at: datetime = Field(..., description="Response generation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rows": [],
                "total_pairs": 5,
                "generated_at": "2025-12-10T15:30:00Z"
            }
        }
