"""Risk Management Configuration - ATR-based, profit-optimized settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import List


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: str | None, *, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_list(value: str | None, *, default: List[str]) -> List[str]:
    if not value:
        return default
    items = [item.strip().upper() for item in value.split(",") if item.strip()]
    return items or default


class ConsensusType(str, Enum):
    """Ensemble consensus quality levels."""
    UNANIMOUS = "UNANIMOUS"  # All 4 models agree
    STRONG = "STRONG"         # 3/4 models agree
    WEAK = "WEAK"             # Only 2/4 models agree
    SPLIT = "SPLIT"           # 2-2 disagreement


@dataclass(frozen=True)
class TradeFilterConfig:
    """Configuration for trade opportunity filtering."""
    
    # Consensus requirements
    min_consensus_types: List[ConsensusType]  # e.g., [UNANIMOUS, STRONG]
    min_confidence: float                      # 0.70 = 70% confidence required
    
    # Trend alignment
    require_trend_alignment: bool              # True = must align with 200 EMA trend
    ema_period: int                            # 200 for long-term trend
    
    # Volatility gates
    enable_volatility_gate: bool               # True = reject if too volatile
    max_atr_ratio: float                       # 0.05 = max 5% ATR/price ratio
    high_volatility_confidence_boost: float    # 0.80 = require 80% confidence if volatile
    
    # Additional filters
    min_volume_24h: float                      # Minimum 24h volume in USD
    max_spread_bps: int                        # Maximum spread in basis points


@dataclass(frozen=True)
class PositionSizingConfig:
    """Configuration for ATR-based position sizing."""
    
    # Risk per trade
    risk_per_trade_pct: float    # 0.01 = 1% of equity per trade
    min_risk_pct: float           # 0.005 = 0.5% minimum
    max_risk_pct: float           # 0.015 = 1.5% maximum
    
    # Signal quality adjustment
    enable_signal_quality_adjustment: bool
    high_confidence_multiplier: float  # 1.5x size for confidence ≥85%
    low_confidence_multiplier: float   # 0.5x size for confidence <60%
    
    # ATR settings
    atr_period: int              # 14 periods for ATR calculation
    atr_multiplier_sl: float     # k1 = 1.2 for stop loss distance (tighter SL)
    
    # Leverage & constraints
    max_leverage: float          # 30x for Binance Futures
    min_position_usd: float      # $5 minimum position size
    max_position_usd: float      # $500 maximum position size


@dataclass(frozen=True)
class ExitPolicyConfig:
    """Configuration for ATR-based exit management."""
    
    # ATR multipliers
    atr_period: int              # 14 periods
    sl_multiplier: float         # k1 = 1.5 (stop loss at 1.5 ATR)
    tp_multiplier: float         # k2 = 3.75 (take profit at 3.75 ATR)
    
    # Risk-reward target
    target_rr_ratio: float       # 2.5 (TP = 2.5x SL distance)
    
    # Partial exits
    enable_partial_tp: bool      # True = take partial profits
    partial_tp_at_r: float       # 2.0 = partial TP at +2R
    partial_tp_percent: float    # 0.5 = close 50% of position
    
    # Breakeven
    enable_breakeven: bool       # True = move SL to breakeven
    breakeven_at_r: float        # 1.0 = move to BE at +1R
    breakeven_offset_pct: float  # 0.001 = 0.1% above entry (small profit lock)
    
    # Trailing stop
    enable_trailing: bool        # True = trail after partial TP
    trailing_start_r: float      # 2.0 = start trailing at +2R
    trailing_distance_atr: float # 1.0 = trail at 1 ATR from peak
    
    # Time-based exit
    enable_time_exit: bool       # True = close if no progress
    max_hours_no_progress: int   # 24 hours with no profit → close


@dataclass(frozen=True)
class GlobalRiskConfig:
    """Configuration for global risk controls."""
    
    # Daily drawdown limits
    max_daily_drawdown_pct: float      # 0.03 = 3% max daily DD
    max_weekly_drawdown_pct: float     # 0.10 = 10% max weekly DD
    
    # Position limits
    max_concurrent_trades: int         # 4 maximum open positions
    max_exposure_pct: float            # 0.80 = max 80% of equity exposed
    max_correlation: float             # 0.70 = max correlation between positions
    
    # Losing streak protection
    enable_streak_protection: bool     # True = reduce risk after losses
    losing_streak_threshold: int       # 3 consecutive losses
    streak_risk_reduction: float       # 0.5 = reduce risk by 50%
    
    # Recovery mode
    enable_recovery_mode: bool         # True = conservative mode after DD
    recovery_threshold_pct: float      # 0.02 = enter recovery at 2% DD
    recovery_risk_multiplier: float    # 0.5 = half size in recovery mode
    
    # Circuit breaker
    enable_circuit_breaker: bool       # True = pause trading on extreme conditions
    circuit_breaker_loss_pct: float    # 0.05 = 5% loss triggers breaker
    circuit_breaker_cooldown_hours: int # 4 hours before resuming


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for auto-training logging."""
    
    # What to log
    log_trade_decisions: bool          # True = log every trade decision
    log_filter_rejections: bool        # True = log why trades rejected
    log_position_sizing: bool          # True = log size calculations
    log_exit_decisions: bool           # True = log exit reasons
    
    # Log detail level
    include_market_data: bool          # True = include price/ATR/EMA in logs
    include_signal_breakdown: bool     # True = include model votes
    include_risk_metrics: bool         # True = include R-multiple, MFE/MAE
    
    # Storage
    log_to_database: bool              # True = store in PostgreSQL
    log_to_file: bool                  # True = write to trade_decisions.log


@dataclass(frozen=True)
class RiskManagementConfig:
    """Complete risk management configuration."""
    
    trade_filter: TradeFilterConfig
    position_sizing: PositionSizingConfig
    exit_policy: ExitPolicyConfig
    global_risk: GlobalRiskConfig
    logging: LoggingConfig


def load_risk_management_config() -> RiskManagementConfig:
    """Load risk management configuration from environment variables with sensible defaults."""
    
    # Trade Filter Config
    consensus_types_str = os.getenv("RM_MIN_CONSENSUS_TYPES", "UNANIMOUS,STRONG")
    min_consensus_types = [
        ConsensusType[ct.strip()] 
        for ct in consensus_types_str.split(",") 
        if ct.strip() in ConsensusType.__members__
    ]
    
    trade_filter = TradeFilterConfig(
        min_consensus_types=min_consensus_types,
        min_confidence=_parse_float(os.getenv("RM_MIN_CONFIDENCE"), default=0.40),
        require_trend_alignment=_parse_bool(os.getenv("RM_REQUIRE_TREND"), default=True),
        ema_period=_parse_int(os.getenv("RM_EMA_PERIOD"), default=200),
        enable_volatility_gate=_parse_bool(os.getenv("RM_VOLATILITY_GATE"), default=True),
        max_atr_ratio=_parse_float(os.getenv("RM_MAX_ATR_RATIO"), default=0.05),
        high_volatility_confidence_boost=_parse_float(os.getenv("RM_HIGH_VOL_CONFIDENCE"), default=0.80),
        min_volume_24h=_parse_float(os.getenv("RM_MIN_VOLUME_24H"), default=0),  # TESTNET: Disabled volume check (data unavailable)
        max_spread_bps=_parse_int(os.getenv("RM_MAX_SPREAD_BPS"), default=50),
    )
    
    # Position Sizing Config
    position_sizing = PositionSizingConfig(
        risk_per_trade_pct=_parse_float(os.getenv("RM_RISK_PER_TRADE_PCT"), default=0.01),
        min_risk_pct=_parse_float(os.getenv("RM_MIN_RISK_PCT"), default=0.005),
        max_risk_pct=_parse_float(os.getenv("RM_MAX_RISK_PCT"), default=0.80),  # 🔥 80% to match Math AI aggressive mode
        enable_signal_quality_adjustment=_parse_bool(os.getenv("RM_SIGNAL_QUALITY_ADJ"), default=True),
        high_confidence_multiplier=_parse_float(os.getenv("RM_HIGH_CONF_MULT"), default=1.5),
        low_confidence_multiplier=_parse_float(os.getenv("RM_LOW_CONF_MULT"), default=0.5),
        atr_period=_parse_int(os.getenv("RM_ATR_PERIOD"), default=14),
        atr_multiplier_sl=_parse_float(os.getenv("RM_ATR_MULT_SL"), default=1.2),
        max_leverage=_parse_float(os.getenv("RM_MAX_LEVERAGE"), default=30.0),
        min_position_usd=_parse_float(os.getenv("RM_MIN_POSITION_USD"), default=10.0),
        max_position_usd=_parse_float(os.getenv("RM_MAX_POSITION_USD"), default=1250.0),  # 25% of $5000 balance
    )
    
    # Exit Policy Config
    exit_policy = ExitPolicyConfig(
        atr_period=_parse_int(os.getenv("RM_EXIT_ATR_PERIOD"), default=14),
        sl_multiplier=_parse_float(os.getenv("RM_SL_MULTIPLIER"), default=1.2),
        tp_multiplier=_parse_float(os.getenv("RM_TP_MULTIPLIER"), default=3.0),
        target_rr_ratio=_parse_float(os.getenv("RM_TARGET_RR"), default=2.5),
        enable_partial_tp=_parse_bool(os.getenv("RM_ENABLE_PARTIAL_TP"), default=True),
        partial_tp_at_r=_parse_float(os.getenv("RM_PARTIAL_TP_AT_R"), default=2.0),
        partial_tp_percent=_parse_float(os.getenv("RM_PARTIAL_TP_PERCENT"), default=0.5),
        enable_breakeven=_parse_bool(os.getenv("RM_ENABLE_BREAKEVEN"), default=True),
        breakeven_at_r=_parse_float(os.getenv("RM_BREAKEVEN_AT_R"), default=1.0),
        breakeven_offset_pct=_parse_float(os.getenv("RM_BREAKEVEN_OFFSET_PCT"), default=0.001),
        enable_trailing=_parse_bool(os.getenv("RM_ENABLE_TRAILING"), default=True),
        trailing_start_r=_parse_float(os.getenv("RM_TRAILING_START_R"), default=2.0),
        trailing_distance_atr=_parse_float(os.getenv("RM_TRAILING_DISTANCE_ATR"), default=1.0),
        enable_time_exit=_parse_bool(os.getenv("RM_ENABLE_TIME_EXIT"), default=True),
        max_hours_no_progress=_parse_int(os.getenv("RM_MAX_HOURS_NO_PROGRESS"), default=24),
    )
    
    # Global Risk Config
    global_risk = GlobalRiskConfig(
        max_daily_drawdown_pct=_parse_float(os.getenv("RM_MAX_DAILY_DD_PCT"), default=0.08),
        max_weekly_drawdown_pct=_parse_float(os.getenv("RM_MAX_WEEKLY_DD_PCT"), default=0.15),
        max_concurrent_trades=_parse_int(os.getenv("RM_MAX_CONCURRENT_TRADES"), default=20),
        max_exposure_pct=_parse_float(os.getenv("RM_MAX_EXPOSURE_PCT"), default=1.10),  # TESTNET: Allow 110% exposure (for rounding)
        max_correlation=_parse_float(os.getenv("RM_MAX_CORRELATION"), default=0.70),
        enable_streak_protection=_parse_bool(os.getenv("RM_ENABLE_STREAK_PROTECTION"), default=True),
        losing_streak_threshold=_parse_int(os.getenv("RM_LOSING_STREAK_THRESHOLD"), default=3),
        streak_risk_reduction=_parse_float(os.getenv("RM_STREAK_RISK_REDUCTION"), default=0.5),
        enable_recovery_mode=_parse_bool(os.getenv("RM_ENABLE_RECOVERY_MODE"), default=True),
        recovery_threshold_pct=_parse_float(os.getenv("RM_RECOVERY_THRESHOLD_PCT"), default=0.02),
        recovery_risk_multiplier=_parse_float(os.getenv("RM_RECOVERY_RISK_MULT"), default=0.5),
        enable_circuit_breaker=_parse_bool(os.getenv("RM_ENABLE_CIRCUIT_BREAKER"), default=True),
        circuit_breaker_loss_pct=_parse_float(os.getenv("RM_CIRCUIT_BREAKER_LOSS_PCT"), default=0.12),
        circuit_breaker_cooldown_hours=_parse_int(os.getenv("RM_CIRCUIT_BREAKER_COOLDOWN_HOURS"), default=2),
    )
    
    # Logging Config
    logging = LoggingConfig(
        log_trade_decisions=_parse_bool(os.getenv("RM_LOG_TRADE_DECISIONS"), default=True),
        log_filter_rejections=_parse_bool(os.getenv("RM_LOG_FILTER_REJECTIONS"), default=True),
        log_position_sizing=_parse_bool(os.getenv("RM_LOG_POSITION_SIZING"), default=True),
        log_exit_decisions=_parse_bool(os.getenv("RM_LOG_EXIT_DECISIONS"), default=True),
        include_market_data=_parse_bool(os.getenv("RM_LOG_MARKET_DATA"), default=True),
        include_signal_breakdown=_parse_bool(os.getenv("RM_LOG_SIGNAL_BREAKDOWN"), default=True),
        include_risk_metrics=_parse_bool(os.getenv("RM_LOG_RISK_METRICS"), default=True),
        log_to_database=_parse_bool(os.getenv("RM_LOG_TO_DATABASE"), default=True),
        log_to_file=_parse_bool(os.getenv("RM_LOG_TO_FILE"), default=True),
    )
    
    return RiskManagementConfig(
        trade_filter=trade_filter,
        position_sizing=position_sizing,
        exit_policy=exit_policy,
        global_risk=global_risk,
        logging=logging,
    )
