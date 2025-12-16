"""
Orchestrator Policy Engine

This module acts as the central "Conductor" that unifies outputs from
all subsystems (regime, risk, performance, cost) and produces one single
POLICY that the entire trading stack follows.

It controls:
- Trade permission
- Risk scaling
- Confidence thresholds
- Symbol selection
- Entry/exit style
- Adaptability based on market conditions

Author: Quantum Trader Team
Date: 2025-01-22
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import json
from pathlib import Path

from config.config import (
    get_policy_min_confidence_trending,
    get_policy_min_confidence_ranging,
    get_policy_min_confidence_normal
)

# MSC AI INTEGRATION
try:
    from backend.services.msc_ai_integration import QuantumPolicyStoreMSC
    MSC_AI_AVAILABLE = True
    logger_msc = logging.getLogger(__name__)
    logger_msc.info("[OK] MSC AI Policy Reader available")
except ImportError as e:
    MSC_AI_AVAILABLE = False
    logger_msc = logging.getLogger(__name__)
    logger_msc.warning(f"[WARNING] MSC AI not available: {e}")

# TRADING PROFILE INTEGRATION
try:
    from backend.services.ai.trading_profile import (
        validate_trade,
        classify_symbol_tier,
    )
    from backend.services.binance_market_data import create_market_data_fetcher
    from backend.config.trading_profile import get_trading_profile_config
    TRADING_PROFILE_AVAILABLE = True
    logger_tp = logging.getLogger(__name__)
    logger_tp.info("✅ Trading Profile integration available")
except ImportError as e:
    TRADING_PROFILE_AVAILABLE = False
    logger_tp = logging.getLogger(__name__)
    logger_tp.warning(f"⚠️ Trading Profile not available: {e}")

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for Orchestrator Policy Engine."""
    
    # Base parameters
    base_confidence: float = 0.50
    base_risk_pct: float = 1.0
    
    # Risk limits
    daily_dd_limit: float = 3.0  # %
    losing_streak_limit: int = 5
    max_open_positions: int = 8
    total_exposure_limit: float = 15.0  # %
    
    # Volatility thresholds
    extreme_vol_threshold: float = 0.06  # ATR/price ratio
    high_vol_threshold: float = 0.04
    
    # Cost thresholds
    high_spread_bps: float = 10.0
    high_slippage_bps: float = 8.0
    
    # Stability parameters
    policy_update_interval_sec: int = 60
    similarity_threshold: float = 0.95  # How similar policies must be to skip update
    
    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        """Load config from environment variables."""
        import os
        return cls(
            base_confidence=float(os.getenv("ORCH_BASE_CONFIDENCE", "0.50")),
            base_risk_pct=float(os.getenv("ORCH_BASE_RISK_PCT", "1.0")),
            daily_dd_limit=float(os.getenv("ORCH_DAILY_DD_LIMIT", "3.0")),
            losing_streak_limit=int(os.getenv("ORCH_LOSING_STREAK_LIMIT", "5")),
            max_open_positions=int(os.getenv("ORCH_MAX_OPEN_POSITIONS", "8")),
            total_exposure_limit=float(os.getenv("ORCH_TOTAL_EXPOSURE_LIMIT", "15.0"))
        )
    
    @classmethod
    def from_profile(cls, profile_name: Optional[str] = None) -> "OrchestratorConfig":
        """
        Load config from orchestrator profile (SAFE or AGGRESSIVE).
        
        Args:
            profile_name: "SAFE" or "AGGRESSIVE". If None, uses CURRENT_PROFILE from env.
            
        Returns:
            OrchestratorConfig instance with profile parameters
            
        Example:
            >>> config = OrchestratorConfig.from_profile("SAFE")
            >>> config.base_confidence
            0.55
        """
        try:
            from backend.services.governance.orchestrator_config import load_profile, CURRENT_PROFILE
            
            if profile_name is None:
                profile_name = CURRENT_PROFILE
            
            profile = load_profile(profile_name)
            
            return cls(
                base_confidence=profile["base_confidence"],
                base_risk_pct=profile["base_risk_pct"],
                daily_dd_limit=profile["daily_dd_limit"],
                losing_streak_limit=profile["losing_streak_limit"],
                max_open_positions=profile["max_open_positions"],
                total_exposure_limit=profile["total_exposure_limit"],
                extreme_vol_threshold=profile["extreme_vol_threshold"],
                high_vol_threshold=profile["high_vol_threshold"],
                high_spread_bps=profile["high_spread_bps"],
                high_slippage_bps=profile["high_slippage_bps"]
            )
        except ImportError as e:
            logger.warning(f"[WARNING] Could not load profile: {e}. Falling back to defaults.")
            return cls()


@dataclass
class RiskState:
    """Current risk state of the trading system."""
    daily_pnl_pct: float
    current_drawdown_pct: float
    losing_streak: int
    open_trades_count: int
    total_exposure_pct: float


@dataclass
class SymbolPerformanceData:
    """Performance data for a symbol."""
    symbol: str
    winrate: float
    avg_R: float
    cumulative_pnl: float
    performance_tag: str  # "GOOD" | "NEUTRAL" | "BAD"


@dataclass
class CostMetrics:
    """Cost-related metrics."""
    spread_level: str  # "LOW" | "NORMAL" | "HIGH"
    slippage_level: str  # "LOW" | "NORMAL" | "HIGH"
    funding_cost_estimate: Optional[float] = None


@dataclass
class TradingPolicy:
    """Complete trading policy from Orchestrator."""
    allow_new_trades: bool
    risk_profile: str  # "NORMAL" | "REDUCED" | "NO_NEW_TRADES"
    max_risk_pct: float
    min_confidence: float
    entry_mode: str  # "NORMAL" | "DEFENSIVE" | "AGGRESSIVE"
    exit_mode: str  # "FAST_TP" | "TREND_FOLLOW" | "DEFENSIVE_TRAIL"
    allowed_symbols: List[str]
    disallowed_symbols: List[str]
    note: str
    timestamp: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def similarity_score(self, other: "TradingPolicy") -> float:
        """
        Calculate similarity with another policy (0-1).
        Used to prevent oscillation.
        """
        if other is None:
            return 0.0
        
        score = 0.0
        weights = {
            'allow_new_trades': 0.20,
            'risk_profile': 0.15,
            'max_risk_pct': 0.15,
            'min_confidence': 0.15,
            'entry_mode': 0.10,
            'exit_mode': 0.10,
            'allowed_symbols': 0.10,
            'disallowed_symbols': 0.05
        }
        
        # Boolean comparison
        if self.allow_new_trades == other.allow_new_trades:
            score += weights['allow_new_trades']
        
        # String comparisons
        if self.risk_profile == other.risk_profile:
            score += weights['risk_profile']
        if self.entry_mode == other.entry_mode:
            score += weights['entry_mode']
        if self.exit_mode == other.exit_mode:
            score += weights['exit_mode']
        
        # Numeric comparisons (within 10% tolerance)
        if abs(self.max_risk_pct - other.max_risk_pct) / max(other.max_risk_pct, 0.01) < 0.10:
            score += weights['max_risk_pct']
        if abs(self.min_confidence - other.min_confidence) / max(other.min_confidence, 0.01) < 0.10:
            score += weights['min_confidence']
        
        # List comparisons (set overlap)
        if set(self.allowed_symbols) == set(other.allowed_symbols):
            score += weights['allowed_symbols']
        if set(self.disallowed_symbols) == set(other.disallowed_symbols):
            score += weights['disallowed_symbols']
        
        return score


class OrchestratorPolicy:
    """
    Top-level control module that unifies all subsystem outputs
    into a single authoritative trading policy.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None, profile_name: Optional[str] = None, policy_store = None):
        """
        Initialize Orchestrator Policy Engine.
        
        Args:
            config: Configuration object. If None, loads from profile.
            profile_name: "SAFE" or "AGGRESSIVE". If None, uses ORCH_PROFILE env var.
            policy_store: PolicyStore for dynamic thresholds and rankings
        
        Example:
            >>> # Use active profile (from ORCH_PROFILE env)
            >>> orchestrator = OrchestratorPolicy()
            >>> 
            >>> # Explicitly use SAFE profile
            >>> orchestrator = OrchestratorPolicy(profile_name="SAFE")
            >>>
            >>> # Use custom config
            >>> config = OrchestratorConfig(base_confidence=0.60)
            >>> orchestrator = OrchestratorPolicy(config=config)
        """
        # Priority: explicit config > profile_name > ORCH_PROFILE env > from_profile default
        if config is None:
            self.config = OrchestratorConfig.from_profile(profile_name)
            logger.info(
                f"[TARGET] Using profile: {profile_name or 'CURRENT_PROFILE from env'}"
            )
        else:
            self.config = config
        
        self.policy_store = policy_store  # [NEW] Store reference for dynamic reads
        self.current_policy: Optional[TradingPolicy] = None
        self.last_update_time: Optional[datetime] = None
        self.policy_history: List[TradingPolicy] = []
        
        # Store profile for risk multipliers (loaded separately)
        try:
            from backend.services.governance.orchestrator_config import load_profile, CURRENT_PROFILE
            self.profile = load_profile(profile_name or CURRENT_PROFILE)
        except (ImportError, Exception) as e:
            logger.warning(f"Could not load orchestrator profile: {e}")
            self.profile = {}
        
        # PolicyStore integration
        if self.policy_store:
            logger.info("PolicyStore integration enabled in OrchestratorPolicy (dynamic filtering)")
        
        # MSC AI POLICY READER (Legacy - kept for backward compatibility)
        if MSC_AI_AVAILABLE:
            try:
                self.msc_policy_store = QuantumPolicyStoreMSC()
                logger.info("✅ MSC AI Policy Reader initialized in OrchestratorPolicy")
            except Exception as e:
                logger.error(f"⚠️ MSC AI Policy Reader initialization failed: {e}")
                self.msc_policy_store = None
        else:
            self.msc_policy_store = None
        
        # TRADING PROFILE INTEGRATION
        if TRADING_PROFILE_AVAILABLE:
            try:
                self.tp_config = get_trading_profile_config()
                self.tp_market_data = create_market_data_fetcher()
                self.tp_enabled = self.tp_config.enabled
                logger.info(f"✅ Trading Profile enabled: {self.tp_enabled}")
            except Exception as e:
                logger.warning(f"⚠️ Trading Profile initialization failed: {e}")
                self.tp_config = None
                self.tp_market_data = None
                self.tp_enabled = False
        else:
            self.tp_config = None
            self.tp_market_data = None
            self.tp_enabled = False
        
        # Initialize with default safe policy
        self._initialize_default_policy()
        
        logger.info(
            f"[OK] OrchestratorPolicy initialized: "
            f"Base confidence={self.config.base_confidence:.2f}, "
            f"Base risk={self.config.base_risk_pct:.2%}, "
            f"DD limit={self.config.daily_dd_limit:.1f}%"
        )
    
    def _initialize_default_policy(self) -> None:
        """Initialize with a safe default policy."""
        self.current_policy = TradingPolicy(
            allow_new_trades=True,
            risk_profile="NORMAL",
            max_risk_pct=self.config.base_risk_pct,
            min_confidence=self.config.base_confidence,
            entry_mode="NORMAL",
            exit_mode="TREND_FOLLOW",
            allowed_symbols=[],
            disallowed_symbols=[],
            note="Default initialization policy",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def get_dynamic_confidence_threshold(self) -> float:
        """
        Get confidence threshold from PolicyStore or fall back to config.
        
        Returns:
            Confidence threshold (0-1)
        """
        if self.policy_store:
            try:
                policy = self.policy_store.get()
                threshold = policy.get('global_min_confidence')
                if threshold is not None:
                    logger.debug(f"[PolicyStore] Using dynamic confidence: {threshold:.2f}")
                    return threshold
            except Exception as e:
                logger.error(f"[ERROR] Failed to read PolicyStore confidence: {e}")
        
        # Fallback to config
        return self.config.base_confidence
    
    def get_opportunity_rankings(self) -> dict[str, float]:
        """
        Get opportunity rankings from PolicyStore.
        
        Returns:
            Dictionary of {symbol: score} sorted by score descending
        """
        if self.policy_store:
            try:
                policy = self.policy_store.get()
                rankings = policy.get('opp_rankings', {})
                if rankings:
                    logger.debug(f"[PolicyStore] Retrieved {len(rankings)} opportunity rankings")
                    return rankings
            except Exception as e:
                logger.error(f"[ERROR] Failed to read PolicyStore rankings: {e}")
        
        return {}
    
    def update_policy(
        self,
        regime_tag: str,
        vol_level: str,
        risk_state: RiskState,
        symbol_performance: List[SymbolPerformanceData],
        ensemble_quality: Optional[float] = None,
        cost_metrics: Optional[CostMetrics] = None
    ) -> TradingPolicy:
        """
        Compute new trading policy based on all subsystem inputs.
        
        Args:
            regime_tag: "TRENDING" or "RANGING"
            vol_level: "LOW", "NORMAL", "HIGH", "EXTREME"
            risk_state: Current risk state
            symbol_performance: List of per-symbol performance data
            ensemble_quality: Recent ensemble performance (0-1)
            cost_metrics: Cost-related metrics
        
        Returns:
            TradingPolicy object
        """
        # Check if update is needed (stability mechanism)
        now = datetime.now(timezone.utc)
        if self.last_update_time and self.current_policy:
            elapsed = (now - self.last_update_time).total_seconds()
            if elapsed < self.config.policy_update_interval_sec:
                logger.debug(f"Policy update skipped - only {elapsed:.0f}s since last update")
                return self.current_policy
        
        # =============================================================
        # REGIME-BASED CONFIDENCE THRESHOLDS
        # =============================================================
        # Why regime-based thresholds?
        # - TRENDING markets: Higher signal reliability → Lower threshold (0.38)
        # - NORMAL markets: Standard conditions → Medium threshold (0.40)
        # - RANGING markets: Choppy signals → Higher threshold (0.43)
        # - HIGH_VOL: Increased noise → Even higher threshold (0.45)
        #
        # Why >= instead of >?
        # - Signals with exactly threshold confidence should be allowed
        # - Prevents artificial blocking of borderline high-quality signals
        # - More aligned with statistical confidence intervals
        # =============================================================
        
        # Calculate base confidence from regime
        if regime_tag == "TRENDING":
            base_conf = get_policy_min_confidence_trending()  # Lower bar for trending markets (clearer signals)
            regime_note = "TRENDING regime (clear directional moves)"
        elif regime_tag == "RANGING":
            base_conf = get_policy_min_confidence_ranging()  # Higher bar for ranging markets (choppy signals)
            regime_note = "RANGING regime (choppy conditions)"
        else:
            base_conf = get_policy_min_confidence_normal()  # Default for undefined regimes
            regime_note = "NORMAL regime (standard conditions)"
        
        # Adjust for volatility level
        vol_adjustment = 0.0
        if vol_level == "LOW":
            vol_adjustment = -0.02  # Lower threshold in calm markets
            vol_note = "LOW volatility (calm markets)"
        elif vol_level == "NORMAL":
            vol_adjustment = 0.0  # No change
            vol_note = "NORMAL volatility"
        elif vol_level == "HIGH":
            vol_adjustment = 0.02  # Higher threshold in volatile markets
            vol_note = "HIGH volatility (increased noise)"
        elif vol_level == "EXTREME":
            vol_adjustment = 0.07  # Much higher threshold in extreme conditions
            vol_note = "EXTREME volatility (very noisy)"
        else:
            vol_note = "Unknown volatility level"
        
        # Final confidence threshold
        min_confidence = base_conf + vol_adjustment
        min_confidence = max(0.15, min(0.60, min_confidence))  # 🔥 LOWERED min from 0.30 to 0.15 (15%)
        
        logger.info(
            f"[CONFIDENCE] Regime-based threshold calculation:\n"
            f"   Regime: {regime_tag} → base={base_conf:.2f} ({regime_note})\n"
            f"   Vol Level: {vol_level} → adjustment={vol_adjustment:+.2f} ({vol_note})\n"
            f"   Final min_confidence: {min_confidence:.2f}\n"
            f"   Comparison logic: signal_confidence >= {min_confidence:.2f}"
        )
        
        # [MSC AI] Read supreme policy from Meta Strategy Controller
        msc_policy = None
        if MSC_AI_AVAILABLE and self.msc_policy_store:
            try:
                msc_policy = self.msc_policy_store.read_policy()
                if msc_policy:
                    logger.info(
                        f"[MSC AI] Supreme policy loaded: "
                        f"risk_mode={msc_policy.get('risk_mode')}, "
                        f"max_risk={msc_policy.get('max_risk_per_trade', 0)*100:.2f}%"
                    )
            except Exception as e:
                logger.error(f"[ERROR] Failed to read MSC AI policy: {e}", exc_info=True)
        
        # Start with base parameters (may be overridden by MSC AI)
        policy_data = {
            "allow_new_trades": True,
            "risk_profile": "NORMAL",
            "max_risk_pct": self.config.base_risk_pct,
            "min_confidence": min_confidence,  # Use regime-calculated threshold
            "entry_mode": "NORMAL",
            "exit_mode": "TREND_FOLLOW",
            "allowed_symbols": [],
            "disallowed_symbols": [],
            "note": "",
            "timestamp": now.isoformat()
        }
        
        notes = []
        
        # [MSC AI] APPLY SUPREME RISK MODE (HIGHEST PRIORITY)
        if msc_policy:
            msc_risk_mode = msc_policy.get('risk_mode', 'NORMAL')
            msc_max_risk = msc_policy.get('max_risk_per_trade', self.config.base_risk_pct)
            msc_confidence = msc_policy.get('global_min_confidence')
            
            # MSC AI risk mode overrides everything
            if msc_risk_mode == 'DEFENSIVE':
                policy_data['risk_profile'] = 'REDUCED'
                policy_data['max_risk_pct'] = msc_max_risk
                policy_data['entry_mode'] = 'DEFENSIVE'
                if msc_confidence:
                    policy_data['min_confidence'] = msc_confidence
                notes.append(f"MSC AI: DEFENSIVE mode (risk={msc_max_risk*100:.2f}%)")
            elif msc_risk_mode == 'AGGRESSIVE':
                policy_data['risk_profile'] = 'NORMAL'
                policy_data['max_risk_pct'] = msc_max_risk
                policy_data['entry_mode'] = 'AGGRESSIVE'
                if msc_confidence:
                    policy_data['min_confidence'] = msc_confidence
                notes.append(f"MSC AI: AGGRESSIVE mode (risk={msc_max_risk*100:.2f}%)")
            else:  # NORMAL
                policy_data['max_risk_pct'] = msc_max_risk
                if msc_confidence:
                    policy_data['min_confidence'] = msc_confidence
                notes.append(f"MSC AI: NORMAL mode (risk={msc_max_risk*100:.2f}%)")
            
            # Apply MSC AI position limits
            msc_max_positions = msc_policy.get('max_positions')
            if msc_max_positions and risk_state.open_trades_count >= msc_max_positions:
                policy_data['allow_new_trades'] = False
                notes.append(f"MSC AI: Position limit reached ({risk_state.open_trades_count}/{msc_max_positions})")
        
        # =============================================================
        # A) REGIME + VOLATILITY RULES
        # =============================================================
        
        if vol_level == "EXTREME":
            policy_data["allow_new_trades"] = False
            policy_data["risk_profile"] = "NO_NEW_TRADES"
            notes.append("EXTREME volatility - no new trades")
        
        elif vol_level == "HIGH":
            policy_data["risk_profile"] = "REDUCED"
            policy_data["max_risk_pct"] *= 0.5
            # min_confidence already adjusted above
            notes.append("HIGH volatility - risk reduced 50%")
        
        if regime_tag == "TRENDING" and vol_level == "NORMAL":
            policy_data["entry_mode"] = "AGGRESSIVE"
            policy_data["exit_mode"] = "TREND_FOLLOW"
            # min_confidence already set to 0.38 for TRENDING above
            notes.append("TRENDING + NORMAL_VOL - aggressive trend following")
        
        elif regime_tag == "RANGING":
            policy_data["entry_mode"] = "DEFENSIVE"
            policy_data["exit_mode"] = "FAST_TP"
            policy_data["max_risk_pct"] *= 0.7
            # min_confidence already set to 0.43 for RANGING above
            notes.append("RANGING market - defensive scalping")
        
        # =============================================================
        # B) RISK-STATE RULES
        # =============================================================
        
        # Daily drawdown protection
        if risk_state.current_drawdown_pct <= -self.config.daily_dd_limit:
            policy_data["allow_new_trades"] = False
            policy_data["risk_profile"] = "NO_NEW_TRADES"
            notes.append(f"Daily DD limit hit ({risk_state.current_drawdown_pct:.2f}%)")
        
        # Losing streak protection
        if risk_state.losing_streak >= self.config.losing_streak_limit:
            policy_data["max_risk_pct"] *= 0.3
            policy_data["entry_mode"] = "DEFENSIVE"
            policy_data["min_confidence"] += 0.05
            notes.append(f"Losing streak {risk_state.losing_streak} - reduced to 30% risk")
        
        # Position count limit
        if risk_state.open_trades_count >= self.config.max_open_positions:
            policy_data["allow_new_trades"] = False
            notes.append(f"Max positions reached ({risk_state.open_trades_count})")
        
        # Total exposure limit
        if risk_state.total_exposure_pct >= self.config.total_exposure_limit:
            policy_data["allow_new_trades"] = False
            notes.append(f"Exposure limit hit ({risk_state.total_exposure_pct:.1f}%)")
        
        # =============================================================
        # C) SYMBOL PERFORMANCE RULES
        # =============================================================
        
        for perf in symbol_performance:
            if perf.performance_tag == "BAD":
                policy_data["disallowed_symbols"].append(perf.symbol)
            else:
                policy_data["allowed_symbols"].append(perf.symbol)
        
        if policy_data["disallowed_symbols"]:
            notes.append(f"Excluded {len(policy_data['disallowed_symbols'])} poor performers")
        
        # =============================================================
        # D) COST MODEL RULES
        # =============================================================
        
        if cost_metrics:
            if cost_metrics.spread_level == "HIGH" or cost_metrics.slippage_level == "HIGH":
                policy_data["entry_mode"] = "DEFENSIVE"
                policy_data["min_confidence"] += 0.03
                notes.append("High costs - tighter entry filter")
        
        # =============================================================
        # E) ENSEMBLE QUALITY ADJUSTMENT (OPTIONAL)
        # =============================================================
        
        if ensemble_quality is not None and ensemble_quality < 0.40:
            policy_data["min_confidence"] += 0.05
            notes.append(f"Low ensemble quality ({ensemble_quality:.2f}) - higher confidence required")
        
        # Combine notes
        policy_data["note"] = "; ".join(notes) if notes else "Normal conditions"
        
        # Create new policy
        new_policy = TradingPolicy(**policy_data)
        
        # =============================================================
        # F) STABILITY CHECK
        # =============================================================
        
        if self.current_policy:
            similarity = new_policy.similarity_score(self.current_policy)
            if similarity >= self.config.similarity_threshold:
                logger.debug(f"Policy unchanged (similarity={similarity:.2%}) - keeping previous")
                return self.current_policy
            else:
                logger.info(f"Policy updated (similarity={similarity:.2%}): {policy_data['note']}")
        else:
            logger.info(f"Initial policy set: {policy_data['note']}")
        
        # Save and return
        self.current_policy = new_policy
        self.last_update_time = now
        self.policy_history.append(new_policy)
        
        # Trim history to last 100 policies
        if len(self.policy_history) > 100:
            self.policy_history = self.policy_history[-100:]
        
        self._log_policy_change(new_policy)
        
        return new_policy
    
    def get_policy(self) -> TradingPolicy:
        """
        Get current trading policy.
        
        Returns:
            Current TradingPolicy object
        """
        if self.current_policy is None:
            self._initialize_default_policy()
        return self.current_policy
    
    def can_trade_symbol(self, symbol: str, side: str) -> Tuple[bool, str]:
        """
        Check if a symbol can be traded based on Trading Profile filters.
        
        This is the PRIMARY FILTER before any signal is considered.
        Checks:
        - Liquidity (volume, spread, depth)
        - Funding window timing
        - Funding rate favorability
        - Universe tier classification
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: Trade direction ('LONG' or 'SHORT')
        
        Returns:
            Tuple of (can_trade: bool, reason: str)
        
        Example:
            >>> can_trade, reason = orchestrator.can_trade_symbol('BTCUSDT', 'LONG')
            >>> if not can_trade:
            >>>     logger.info(f"Trade rejected: {reason}")
        """
        # If Trading Profile is disabled, allow all symbols
        if not self.tp_enabled or not self.tp_config or not self.tp_market_data:
            return True, "Trading Profile disabled - all symbols allowed"
        
        try:
            # Fetch symbol metrics
            metrics = self.tp_market_data.fetch_symbol_metrics(symbol)
            
            if not metrics:
                logger.warning(f"❌ Failed to fetch metrics for {symbol}")
                return False, f"Failed to fetch market data for {symbol}"
            
            # Validate trade using Trading Profile
            valid, reason = validate_trade(
                metrics,
                side,
                self.tp_config.liquidity,
                self.tp_config.funding
            )
            
            if not valid:
                logger.info(f"🚫 Trading Profile rejected {symbol} {side}: {reason}")
            else:
                logger.debug(f"✅ Trading Profile approved {symbol} {side}")
            
            return valid, reason
            
        except Exception as e:
            logger.error(f"❌ Error validating symbol {symbol}: {e}", exc_info=True)
            # On error, be conservative and reject the trade
            return False, f"Validation error: {str(e)}"
    
    def filter_symbols(self, symbols: List[str], side: str) -> List[str]:
        """
        Filter a list of symbols through Trading Profile checks.
        
        Returns only symbols that pass liquidity and funding filters.
        
        Args:
            symbols: List of symbols to filter
            side: Trade direction ('LONG' or 'SHORT')
        
        Returns:
            List of approved symbols
        
        Example:
            >>> candidates = ['BTCUSDT', 'TAOUSDT', 'ETHUSDT']
            >>> approved = orchestrator.filter_symbols(candidates, 'LONG')
            >>> # Result: ['BTCUSDT', 'ETHUSDT'] (TAO excluded due to low liquidity)
        """
        if not self.tp_enabled:
            return symbols
        
        approved = []
        rejected_summary = []
        
        for symbol in symbols:
            can_trade, reason = self.can_trade_symbol(symbol, side)
            if can_trade:
                approved.append(symbol)
            else:
                rejected_summary.append(f"{symbol}: {reason}")
        
        if rejected_summary:
            logger.info(
                f"📊 Trading Profile Filter Results:\n"
                f"   Total: {len(symbols)}\n"
                f"   Approved: {len(approved)}\n"
                f"   Rejected: {len(rejected_summary)}\n"
                f"   Details:\n      " + "\n      ".join(rejected_summary)
            )
        
        return approved
    
    def reset_daily(self) -> None:
        """
        Reset internal counters for new trading day.
        Reinitializes to default safe policy.
        """
        logger.info("🔄 Daily reset - reinitializing orchestrator policy")
        self._initialize_default_policy()
        self.last_update_time = None
        # Keep policy_history for analysis
    
    def _log_policy_change(self, policy: TradingPolicy) -> None:
        """Log policy change with key details."""
        logger.info(
            f"[CLIPBOARD] POLICY UPDATE: "
            f"allow_trades={policy.allow_new_trades}, "
            f"risk_profile={policy.risk_profile}, "
            f"max_risk={policy.max_risk_pct:.2%}, "
            f"min_conf={policy.min_confidence:.2f}, "
            f"entry={policy.entry_mode}, "
            f"exit={policy.exit_mode}"
        )
        
        if policy.disallowed_symbols:
            logger.warning(f"[WARNING] Disallowed symbols: {', '.join(policy.disallowed_symbols)}")
    
    def get_policy_history(self, limit: int = 10) -> List[TradingPolicy]:
        """
        Get recent policy history.
        
        Args:
            limit: Maximum number of policies to return
        
        Returns:
            List of recent TradingPolicy objects
        """
        return self.policy_history[-limit:]
    
    def save_policy_history(self, filepath: str = "data/orchestrator_policy_history.json") -> None:
        """
        Save policy history to file.
        
        Args:
            filepath: Path to save history
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        history_data = [p.to_dict() for p in self.policy_history]
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"💾 Saved {len(history_data)} policies to {filepath}")
    
    def load_policy_history(self, filepath: str = "data/orchestrator_policy_history.json") -> None:
        """
        Load policy history from file.
        
        Args:
            filepath: Path to load history from
        """
        try:
            with open(filepath, 'r') as f:
                history_data = json.load(f)
            
            self.policy_history = [TradingPolicy(**p) for p in history_data]
            
            if self.policy_history:
                self.current_policy = self.policy_history[-1]
                logger.info(f"📂 Loaded {len(self.policy_history)} policies from {filepath}")
            
        except FileNotFoundError:
            logger.debug(f"No policy history file found at {filepath}")
        except Exception as e:
            logger.error(f"Error loading policy history: {e}")


# Convenience functions

def create_risk_state(
    daily_pnl_pct: float,
    current_drawdown_pct: float,
    losing_streak: int,
    open_trades_count: int,
    total_exposure_pct: float
) -> RiskState:
    """Create RiskState object."""
    return RiskState(
        daily_pnl_pct=daily_pnl_pct,
        current_drawdown_pct=current_drawdown_pct,
        losing_streak=losing_streak,
        open_trades_count=open_trades_count,
        total_exposure_pct=total_exposure_pct
    )


def create_symbol_performance(
    symbol: str,
    winrate: float,
    avg_R: float,
    cumulative_pnl: float,
    performance_tag: str
) -> SymbolPerformanceData:
    """Create SymbolPerformanceData object."""
    return SymbolPerformanceData(
        symbol=symbol,
        winrate=winrate,
        avg_R=avg_R,
        cumulative_pnl=cumulative_pnl,
        performance_tag=performance_tag
    )


def create_cost_metrics(
    spread_level: str,
    slippage_level: str,
    funding_cost_estimate: Optional[float] = None
) -> CostMetrics:
    """Create CostMetrics object."""
    return CostMetrics(
        spread_level=spread_level,
        slippage_level=slippage_level,
        funding_cost_estimate=funding_cost_estimate
    )
