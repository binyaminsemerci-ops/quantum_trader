"""
Orchestrator Configuration - Feature flags and modes for policy integration.

This module provides configuration for how the OrchestratorPolicy is integrated
into the trading system.

Includes SAFE and AGGRESSIVE profiles for different trading scenarios.
"""
import os
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

# [TARGET] CURRENT ACTIVE PROFILE
# Can be overridden by environment variable: ORCH_PROFILE=SAFE or ORCH_PROFILE=AGGRESSIVE
CURRENT_PROFILE = os.getenv("ORCH_PROFILE", "SAFE").upper()

import logging
logger = logging.getLogger(__name__)


# ===========================================================
# ORCHESTRATOR PROFILES
# ===========================================================

SAFE_PROFILE = {
    # === BASE PARAMETERS ===
    "base_confidence": 0.65,  # Quality threshold: 65%+ confidence trades
    "base_risk_pct": 0.8,     # Conservative base risk (0.8% vs 1.0%)
    
    # === RISK LIMITS ===
    "daily_dd_limit": 2.5,             # Strict daily drawdown limit (%)
    "losing_streak_limit": 4,          # Stop after 4 consecutive losses
    "max_open_positions": 5,           # Fewer simultaneous positions
    "total_exposure_limit": 10.0,      # Lower total portfolio exposure (%)
    
    # === VOLATILITY THRESHOLDS ===
    "extreme_vol_threshold": 0.05,     # Lower threshold = more sensitive (ATR/price)
    "high_vol_threshold": 0.03,        # Trigger high-vol mode earlier
    
    # === COST THRESHOLDS ===
    "high_spread_bps": 8.0,            # More sensitive to spreads
    "high_slippage_bps": 6.0,          # More sensitive to slippage
    
    # === RISK SCALING MULTIPLIERS ===
    "risk_multipliers": {
        # Regime-based scaling
        "BULL": 0.9,                    # Slightly conservative even in bull
        "BEAR": 0.3,                    # Very defensive in bear markets
        "HIGH_VOL": 0.4,                # Aggressive cut in high volatility
        "CHOP": 0.5,                    # Reduce in choppy conditions
        "NORMAL": 0.8,                  # Default conservative multiplier
        
        # Risk state adjustments
        "losing_streak_per_loss": 0.15,  # -15% per consecutive loss
        "drawdown_per_percent": 0.10,    # -10% per 1% drawdown
        "high_exposure": 0.6,            # Reduce when portfolio too concentrated
    },
    
    # === CONFIDENCE ADJUSTMENTS ===
    "confidence_adjustments": {
        "BULL": 0.00,                   # No reduction in bull
        "BEAR": +0.08,                  # Require +8% higher confidence in bear
        "HIGH_VOL": +0.10,              # Require +10% higher confidence in volatility
        "CHOP": +0.05,                  # Slight increase in choppy markets
        "NORMAL": 0.00,
        
        # Cost-based adjustments
        "high_spread": +0.03,           # Require better signals when spreads high
        "high_slippage": +0.03,
    },
    
    # === SYMBOL FILTERING ===
    "symbol_performance_thresholds": {
        "min_winrate": 0.45,            # Block symbols with <45% win rate
        "min_avg_R": 0.6,               # Block symbols with <0.6 R-multiple
        "bad_streak_limit": 3,          # Block after 3 consecutive losses on symbol
    },
    
    # === EXIT MODE PREFERENCES ===
    "exit_mode_bias": {
        "BULL": "TREND_FOLLOW",        # Follow trends in bull markets
        "BEAR": "FAST_TP",             # Take profits quickly in bear
        "HIGH_VOL": "DEFENSIVE_TRAIL",  # Tight stops in volatility
        "CHOP": "FAST_TP",             # Quick exits in choppy conditions
        "NORMAL": "TREND_FOLLOW",
    },
    
    # === ENTRY MODE PREFERENCES ===
    "entry_mode_bias": {
        "BULL": "NORMAL",
        "BEAR": "DEFENSIVE",           # Very selective in bear
        "HIGH_VOL": "DEFENSIVE",
        "CHOP": "DEFENSIVE",
        "NORMAL": "NORMAL",
    },
    
    # === RECOVERY THRESHOLDS ===
    "recovery_multiplier": 1.1,         # Slow recovery: +10% risk after profit
    "recovery_after_streak": 2,         # Need 2 wins to recover from losses
    
    # === SPREAD/SLIPPAGE SENSITIVITY ===
    "cost_sensitivity": "HIGH",       # React strongly to trading costs
    "max_cost_in_R": 0.15,              # Block trades with >0.15R in costs
}


AGGRESSIVE_PROFILE = {
    # === BASE PARAMETERS ===
    "base_confidence": 0.30,  # TESTNET: Lower threshold for testing
    "base_risk_pct": 1.2,     # Higher base risk (1.2% vs 1.0%)
    
    # === RISK LIMITS ===
    "daily_dd_limit": 4.5,             # Higher DD tolerance (%)
    "losing_streak_limit": 7,          # More losses allowed before stopping
    "max_open_positions": 10,          # More simultaneous positions
    "total_exposure_limit": 20.0,      # Higher total portfolio exposure (%)
    
    # === VOLATILITY THRESHOLDS ===
    "extreme_vol_threshold": 0.08,     # Higher threshold = less sensitive
    "high_vol_threshold": 0.05,        # Allow more volatility before reacting
    
    # === COST THRESHOLDS ===
    "high_spread_bps": 15.0,           # More tolerant of spreads
    "high_slippage_bps": 12.0,         # More tolerant of slippage
    
    # === RISK SCALING MULTIPLIERS ===
    "risk_multipliers": {
        # Regime-based scaling
        "BULL": 1.3,                    # Capitalize on bull runs
        "BEAR": 0.6,                    # Still trade in bears (reduced)
        "HIGH_VOL": 0.8,                # Less reduction in volatility
        "CHOP": 0.7,                    # More active in chop
        "NORMAL": 1.0,                  # Full risk in normal conditions
        
        # Risk state adjustments
        "losing_streak_per_loss": 0.08,  # -8% per loss (softer)
        "drawdown_per_percent": 0.05,    # -5% per 1% DD (softer)
        "high_exposure": 0.8,            # Less concerned about concentration
    },
    
    # === CONFIDENCE ADJUSTMENTS ===
    "confidence_adjustments": {
        "BULL": -0.02,                  # Actually LOWER threshold in bull (more trades)
        "BEAR": +0.05,                  # Only +5% higher in bear
        "HIGH_VOL": +0.05,              # Only +5% in volatility
        "CHOP": +0.02,                  # Slight increase
        "NORMAL": 0.00,
        
        # Cost-based adjustments
        "high_spread": +0.01,           # Minimal adjustment
        "high_slippage": +0.01,
    },
    
    # === SYMBOL FILTERING ===
    "symbol_performance_thresholds": {
        "min_winrate": 0.35,            # Allow symbols with 35%+ win rate
        "min_avg_R": 0.3,               # Allow lower R-multiples
        "bad_streak_limit": 5,          # Tolerate 5 losses before blocking
    },
    
    # === EXIT MODE PREFERENCES ===
    "exit_mode_bias": {
        "BULL": "TREND_FOLLOW",        # Maximize bull moves
        "BEAR": "TREND_FOLLOW",        # Still follow trends in bear (contrarian)
        "HIGH_VOL": "TREND_FOLLOW",    # Ride volatility
        "CHOP": "TREND_FOLLOW",        # Try to catch breakouts
        "NORMAL": "TREND_FOLLOW",
    },
    
    # === ENTRY MODE PREFERENCES ===
    "entry_mode_bias": {
        "BULL": "AGGRESSIVE",          # Capitalize on momentum
        "BEAR": "NORMAL",              # Normal entry in bear
        "HIGH_VOL": "NORMAL",
        "CHOP": "AGGRESSIVE",          # Hunt for breakouts
        "NORMAL": "NORMAL",
    },
    
    # === RECOVERY THRESHOLDS ===
    "recovery_multiplier": 1.3,         # Fast recovery: +30% risk after profit
    "recovery_after_streak": 1,         # Just 1 win needed to recover
    
    # === SPREAD/SLIPPAGE SENSITIVITY ===
    "cost_sensitivity": "LOW",        # Less concerned about costs
    "max_cost_in_R": 0.30,              # Allow up to 0.30R in costs
}


def load_profile(profile_name: str) -> Dict[str, Any]:
    """
    Load orchestrator profile by name.
    
    Args:
        profile_name: "SAFE" or "AGGRESSIVE"
        
    Returns:
        Dictionary containing profile configuration
        
    Raises:
        ValueError: If profile_name is invalid
        
    Example:
        >>> profile = load_profile("SAFE")
        >>> profile["base_confidence"]
        0.55
    """
    profile_name = profile_name.upper()
    
    if profile_name == "SAFE":
        logger.info("[SHIELD] Loading SAFE profile: Conservative risk, higher thresholds")
        return SAFE_PROFILE.copy()
    elif profile_name == "AGGRESSIVE":
        logger.info("⚡ Loading AGGRESSIVE profile: Higher risk, lower thresholds")
        return AGGRESSIVE_PROFILE.copy()
    else:
        raise ValueError(
            f"Invalid profile name: {profile_name}. "
            f"Valid options are: 'SAFE', 'AGGRESSIVE'. "
            f"Use: export ORCH_PROFILE=SAFE or --profile AGGRESSIVE"
        )


def get_active_profile() -> Dict[str, Any]:
    """
    Get the currently active profile based on CURRENT_PROFILE.
    
    Returns:
        Active profile configuration dictionary
    """
    return load_profile(CURRENT_PROFILE)


class OrchestratorMode(str, Enum):
    """Operating mode for orchestrator integration."""
    OBSERVE = "OBSERVE"  # Log policy decisions, don't enforce
    LIVE = "LIVE"        # Enforce policy decisions on trading


@dataclass
class OrchestratorIntegrationConfig:
    """
    Configuration for orchestrator integration.
    
    Controls how OrchestratorPolicy interacts with other subsystems.
    """
    
    # Master switch
    enable_orchestrator: bool = True
    
    # Operating mode
    mode: OrchestratorMode = OrchestratorMode.OBSERVE
    
    # Individual subsystem overrides (only apply in LIVE mode)
    use_for_signal_filter: bool = False    # Filter signals by policy.disallowed_symbols
    use_for_confidence_threshold: bool = False  # Use policy.min_confidence
    use_for_risk_sizing: bool = False      # Use policy.risk_per_trade_pct
    use_for_position_limits: bool = False  # Use policy.max_open_positions
    use_for_trading_gate: bool = False     # Use policy.allow_new_trades
    use_for_exit_mode: bool = False        # Use policy.exit_mode_override
    
    # Policy update frequency
    policy_update_interval_sec: int = 60  # Update policy every 60 seconds
    
    # Observation mode settings
    log_all_signals: bool = True          # Log every signal decision in OBSERVE mode
    observation_log_dir: str = "data/policy_observations"
    
    def is_observe_mode(self) -> bool:
        """Check if in observation mode."""
        return self.mode == OrchestratorMode.OBSERVE
    
    def is_live_mode(self) -> bool:
        """Check if in live enforcement mode."""
        return self.mode == OrchestratorMode.LIVE
    
    def should_enforce_any(self) -> bool:
        """Check if any enforcement is enabled."""
        if not self.enable_orchestrator:
            return False
        if self.is_observe_mode():
            return False
        return (
            self.use_for_signal_filter or
            self.use_for_confidence_threshold or
            self.use_for_risk_sizing or
            self.use_for_position_limits or
            self.use_for_trading_gate or
            self.use_for_exit_mode
        )
    
    @classmethod
    def create_observe_mode(cls) -> "OrchestratorIntegrationConfig":
        """Create config for pure observation mode (no enforcement)."""
        return cls(
            enable_orchestrator=True,
            mode=OrchestratorMode.OBSERVE,
            use_for_signal_filter=False,
            use_for_confidence_threshold=False,
            use_for_risk_sizing=False,
            use_for_position_limits=False,
            use_for_trading_gate=False,
            use_for_exit_mode=False,
            log_all_signals=True
        )
    
    @classmethod
    def create_live_mode_gradual(cls) -> "OrchestratorIntegrationConfig":
        """
        Create config for FULL LIVE MODE - ALL SUBSYSTEMS ACTIVE.
        
        [ROCKET] FULLY AUTONOMOUS TRADING:
        [OK] Step 1: Signal filtering (symbols + confidence threshold)
        [OK] Step 2: Risk scaling (dynamic position sizing)
        [OK] Step 3: Exit mode override (regime-aware exits)
        [OK] Step 4: Trade shutdown gates (DD/volatility protection)
        [OK] Step 5: Position limits (per-symbol exposure control)
        
        ALL orchestrator controls are now ENFORCED in LIVE mode.
        System is fully autonomous with comprehensive safety mechanisms.
        """
        return cls(
            enable_orchestrator=True,
            mode=OrchestratorMode.LIVE,
            use_for_signal_filter=True,           # [OK] ACTIVE: Filter blocked symbols
            use_for_confidence_threshold=True,    # [OK] ACTIVE: Apply min_confidence
            use_for_risk_sizing=True,             # [OK] ACTIVE: Dynamic position sizing
            use_for_exit_mode=True,               # [OK] ACTIVE: Regime-aware exit strategy
            use_for_trading_gate=True,            # [OK] ACTIVE: Shutdown gates (DD/vol)
            use_for_position_limits=True,         # [OK] ACTIVE: Per-symbol limits
            log_all_signals=True                  # [OK] ACTIVE: Full observability
        )
    
    @classmethod
    def create_live_mode_full(cls) -> "OrchestratorIntegrationConfig":
        """Create config for full LIVE orchestrator control."""
        return cls(
            enable_orchestrator=True,
            mode=OrchestratorMode.LIVE,
            use_for_signal_filter=True,
            use_for_confidence_threshold=True,
            use_for_risk_sizing=True,
            use_for_position_limits=True,
            use_for_trading_gate=True,
            use_for_exit_mode=True,
            log_all_signals=True
        )
    
    def get_summary(self) -> str:
        """Get human-readable summary of config."""
        if not self.enable_orchestrator:
            return "❌ Orchestrator DISABLED"
        
        if self.is_observe_mode():
            return "[EYE] Orchestrator in OBSERVE mode (logging only, no enforcement)"
        
        enforcing = []
        if self.use_for_signal_filter:
            enforcing.append("signal_filter")
        if self.use_for_confidence_threshold:
            enforcing.append("confidence")
        if self.use_for_risk_sizing:
            enforcing.append("risk_sizing")
        if self.use_for_position_limits:
            enforcing.append("position_limits")
        if self.use_for_trading_gate:
            enforcing.append("trading_gate")
        if self.use_for_exit_mode:
            enforcing.append("exit_mode")
        
        if not enforcing:
            return "[WARNING] Orchestrator in LIVE mode but NO enforcement enabled"
        
        return f"[OK] Orchestrator LIVE enforcing: {', '.join(enforcing)}"
