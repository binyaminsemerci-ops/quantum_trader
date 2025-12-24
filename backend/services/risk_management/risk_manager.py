"""Risk Manager - ATR-based position sizing engine."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from backend.config.risk_management import PositionSizingConfig

logger = logging.getLogger(__name__)

# [TARGET] ORCHESTRATOR INTEGRATION: Import for policy-based risk scaling
try:
    from backend.services.governance.orchestrator_policy import TradingPolicy
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    TradingPolicy = None
    logger.warning("[WARNING] OrchestratorPolicy not available, risk scaling will use base config only")

# [RL] REINFORCEMENT LEARNING SIZING: AI-driven position sizing
try:
    from backend.services.ai.rl_position_sizing_agent import get_rl_sizing_agent
    RL_SIZING_AVAILABLE = True
except ImportError:
    RL_SIZING_AVAILABLE = False
    get_rl_sizing_agent = None
    logger.warning("[WARNING] RL Position Sizing not available, using rule-based sizing only")

# [PHASE 3C] CONFIDENCE CALIBRATION & PERFORMANCE-BASED SIZING
try:
    from backend.services.ai.confidence_calibrator import ConfidenceCalibrator
    from backend.services.ai.performance_benchmarker import PerformanceBenchmarker
    PHASE_3C_AVAILABLE = True
except ImportError:
    PHASE_3C_AVAILABLE = False
    ConfidenceCalibrator = None
    PerformanceBenchmarker = None
    logger.warning("[WARNING] Phase 3C components not available, using standard sizing")


@dataclass
class PositionSize:
    """Calculated position size details."""
    quantity: float           # Number of contracts/coins
    notional_usd: float       # Position value in USD
    risk_usd: float           # Amount at risk (if SL hit)
    risk_pct: float           # Risk as % of equity
    leverage_used: float      # Actual leverage (notional / equity)
    sl_distance_pct: float    # Stop loss distance as %
    adjustment_reason: Optional[str] = None  # Why size was adjusted


class RiskManager:
    """
    ATR-based position sizing engine.
    
    Formula:
        risk_amount = equity * risk_per_trade_pct
        sl_distance = ATR * k1 / current_price
        raw_size = risk_amount / sl_distance / current_price
        final_size = apply_constraints(raw_size)
    
    Adjustments based on:
    - Signal confidence (high confidence → larger, low confidence → smaller)
    - Current equity
    - Min/max position size limits
    - Leverage limits
    """
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.current_policy: Optional[TradingPolicy] = None  # [TARGET] Policy cache
        
        # [PHASE 3C] Initialize confidence calibrator and performance benchmarker
        self.confidence_calibrator = None
        self.performance_benchmarker = None
        
        phase3c_enabled = os.getenv("PHASE_3C_ENABLED", "true").lower() == "true"
        if PHASE_3C_AVAILABLE and phase3c_enabled:
            try:
                # These will be injected by AI Engine, but allow lazy init
                logger.info("[PHASE3C] ✅ Confidence Calibrator & Performance Benchmarker available")
            except Exception as e:
                logger.warning(f"[PHASE3C] Failed to initialize: {e}")
        else:
            logger.info("[PHASE3C] Using standard sizing (Phase 3C disabled)")
        
        # [RL] Initialize RL sizing agent if enabled
        rl_enabled = os.getenv("RL_POSITION_SIZING_ENABLED", "true").lower() == "true"
        self.rl_agent = None
        if RL_SIZING_AVAILABLE and rl_enabled:
            try:
                self.rl_agent = get_rl_sizing_agent(
                    enabled=True,
                    learning_rate=float(os.getenv("RL_SIZING_ALPHA", "0.15")),
                    exploration_rate=float(os.getenv("RL_SIZING_EPSILON", "0.10")),
                    min_position_usd=config.min_position_usd,
                    max_position_usd=config.max_position_usd,
                    min_leverage=1.0,
                    max_leverage=config.max_leverage
                )
                logger.info("[RL-SIZING] 🤖 Reinforcement Learning sizing ENABLED")
            except Exception as e:
                logger.warning(f"[RL-SIZING] Failed to initialize: {e}")
                self.rl_agent = None
        else:
            logger.info("[RL-SIZING] Using rule-based sizing (RL disabled)")
        
        logger.info("[OK] RiskManager initialized")
        logger.info(f"   Risk per trade: {config.risk_per_trade_pct:.2%}")
        logger.info(f"   ATR SL multiplier: {config.atr_multiplier_sl}x")
        logger.info(f"   Max leverage: {config.max_leverage}x")
        logger.info(f"   Position range: ${config.min_position_usd} - ${config.max_position_usd}")
    
    def set_policy(self, policy: Optional[TradingPolicy]) -> None:
        """Update current trading policy for risk scaling."""
        self.current_policy = policy
    
    def set_phase3c_components(
        self,
        confidence_calibrator: Optional['ConfidenceCalibrator'] = None,
        performance_benchmarker: Optional['PerformanceBenchmarker'] = None
    ) -> None:
        """
        Inject Phase 3C components for adaptive sizing.
        
        Args:
            confidence_calibrator: Confidence calibration engine
            performance_benchmarker: Performance tracking engine
        """
        self.confidence_calibrator = confidence_calibrator
        self.performance_benchmarker = performance_benchmarker
        
        if confidence_calibrator or performance_benchmarker:
            logger.info(
                "[PHASE3C] ✅ Components injected: "
                f"calibrator={'YES' if confidence_calibrator else 'NO'}, "
                f"benchmarker={'YES' if performance_benchmarker else 'NO'}"
            )
    
    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        equity_usd: float,
        signal_confidence: float,
        action: str,  # "LONG" or "SHORT"
        signal_source: Optional[str] = None  # For Phase 3C tracking
    ) -> PositionSize:
        """
        Calculate position size using ATR-based risk management.
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            atr: Average True Range (14 periods)
            equity_usd: Current account equity
            signal_confidence: AI confidence (0.0 to 1.0)
            action: "LONG" or "SHORT"
            signal_source: Module that generated signal (for Phase 3C tracking)
        
        Returns:
            PositionSize with quantity and risk details
        """
        # [PHASE 3C] STEP 0: Calibrate confidence if available
        original_confidence = signal_confidence
        if self.confidence_calibrator and signal_source:
            try:
                calibration = self.confidence_calibrator.calibrate_confidence_sync(
                    signal_source=signal_source,
                    raw_confidence=signal_confidence,
                    symbol=symbol
                )
                signal_confidence = calibration.calibrated_confidence
                
                if abs(original_confidence - signal_confidence) > 0.05:
                    logger.info(
                        f"[PHASE3C-CALIB] {symbol} ({signal_source}): "
                        f"Confidence {original_confidence:.1%} → {signal_confidence:.1%} "
                        f"(factor={calibration.calibration_factor:.2f}, samples={calibration.sample_count})"
                    )
            except Exception as e:
                logger.warning(f"[PHASE3C-CALIB] Failed for {symbol}: {e}")
        
        # Step 1: Determine risk percentage
        base_risk_pct = self.config.risk_per_trade_pct
        
        # [TARGET] ORCHESTRATOR STEP 2: Apply policy-based risk scaling
        policy_risk_multiplier = 1.0
        policy_adjustments = []
        
        if ORCHESTRATOR_AVAILABLE and self.current_policy:
            policy = self.current_policy
            
            # Apply policy risk scaling
            # policy.max_risk_pct is already scaled by regime/vol/conditions
            policy_risk_multiplier = policy.max_risk_pct
            policy_adjustments.append(f"Policy base: {policy.max_risk_pct:.2%}")
            
            # Additional reduction for REDUCED risk profile
            if policy.risk_profile == "REDUCED":
                policy_risk_multiplier *= 0.7  # Extra 30% reduction
                policy_adjustments.append("REDUCED risk profile (-30%)")
            elif policy.risk_profile == "NO_NEW_TRADES":
                # Safety: If policy says no trades, reduce risk dramatically
                policy_risk_multiplier *= 0.1  # 90% reduction
                policy_adjustments.append("NO_NEW_TRADES profile (-90%)")
                logger.warning(
                    f"[WARNING] {symbol} Policy risk profile is NO_NEW_TRADES, "
                    f"reducing risk to {policy_risk_multiplier * 100:.1f}%"
                )
            
            logger.info(
                f"[TARGET] {symbol} Orchestrator Risk Scaling:\n"
                f"   Policy: {policy.risk_profile}\n"
                f"   Policy note: {policy.note}\n"
                f"   Base risk: {base_risk_pct:.2%}\n"
                f"   Policy multiplier: {policy_risk_multiplier:.2%}\n"
                f"   Adjustments: {', '.join(policy_adjustments) if policy_adjustments else 'None'}"
            )
        else:
            # Fallback if no policy available
            if ORCHESTRATOR_AVAILABLE:
                logger.debug(f"ℹ️ {symbol} No orchestrator policy set, using base risk config")
            policy_adjustments.append("No policy (using base config)")
        
        # Apply policy scaling to base risk
        risk_pct = base_risk_pct * policy_risk_multiplier
        
        # [RL] STEP 1.5: Use RL agent for intelligent sizing (if enabled)
        rl_decision = None
        if self.rl_agent:
            try:
                # Calculate current portfolio exposure
                # (This would ideally come from portfolio manager, using simple estimate for now)
                current_exposure_pct = 0.5  # TODO: Get from portfolio manager
                
                # Get RL sizing decision
                rl_decision = self.rl_agent.decide_sizing(
                    symbol=symbol,
                    confidence=signal_confidence,
                    atr_pct=atr / current_price if current_price > 0 else 0.01,
                    current_exposure_pct=current_exposure_pct,
                    equity_usd=equity_usd,
                    adx=None,  # TODO: Pass ADX if available
                    trend_strength=None  # TODO: Pass trend strength if available
                )
                
                logger.info(
                    f"[RL-SIZING] 🤖 {symbol}: ${rl_decision.position_size_usd:.0f} @ {rl_decision.leverage:.1f}x | "
                    f"{rl_decision.reasoning}"
                )
                
                # Override with RL decision
                notional_usd = rl_decision.position_size_usd
                max_leverage_override = rl_decision.leverage
                risk_pct = rl_decision.risk_pct
                adjustment_reason = f"RL: {rl_decision.reasoning}"
                
                # Skip traditional sizing calculation, jump to Step 6
                quantity = notional_usd / current_price
                # USE RL-DECIDED LEVERAGE DIRECTLY (not calculated from notional)
                leverage_used = rl_decision.leverage
                
                logger.info(
                    f"[RL-LEVERAGE-DEBUG] {symbol}: rl_decision.leverage={rl_decision.leverage:.1f}x, "
                    f"leverage_used={leverage_used:.1f}x, notional=${notional_usd:.0f}"
                )
                
                # For RL-based sizing, use estimated SL distance
                sl_distance_price = atr * self.config.atr_multiplier_sl
                sl_distance_pct = sl_distance_price / current_price if current_price > 0 else 0.01
                
                return PositionSize(
                    quantity=quantity,
                    notional_usd=notional_usd,
                    risk_usd=notional_usd * sl_distance_pct,  # Estimated risk
                    risk_pct=risk_pct,
                    leverage_used=leverage_used,
                    sl_distance_pct=sl_distance_pct,
                    adjustment_reason=adjustment_reason
                )
                
            except Exception as e:
                logger.warning(f"[RL-SIZING] Failed for {symbol}: {e}, falling back to rule-based")
                rl_decision = None
        
        # Continue with traditional rule-based sizing if RL not used
        
        # Adjust for signal confidence if enabled (additional layer)
        confidence_multiplier = 1.0
        if self.config.enable_signal_quality_adjustment:
            if signal_confidence >= 0.85:
                confidence_multiplier = self.config.high_confidence_multiplier
                policy_adjustments.append(f"High confidence {signal_confidence:.1%} (+{(confidence_multiplier-1)*100:.0f}%)")
            elif signal_confidence < 0.60:
                confidence_multiplier = self.config.low_confidence_multiplier
                policy_adjustments.append(f"Low confidence {signal_confidence:.1%} ({(confidence_multiplier-1)*100:.0f}%)")
            
            risk_pct *= confidence_multiplier
        
        # [PHASE 3C] STEP 2.5: Performance-based sizing adjustment
        performance_multiplier = 1.0
        if self.performance_benchmarker and signal_source:
            try:
                # Get module accuracy from Phase 3C-2
                benchmarks = self.performance_benchmarker.get_current_benchmarks()
                
                if signal_source in benchmarks:
                    module_perf = benchmarks[signal_source]
                    
                    if module_perf.accuracy_stats:
                        accuracy = module_perf.accuracy_stats.accuracy_pct
                        
                        # Performance-based multipliers (from enhancement plan)
                        if accuracy >= 75.0:
                            performance_multiplier = 1.2  # +20% for high accuracy
                            policy_adjustments.append(f"High accuracy {accuracy:.1f}% (+20%)")
                        elif accuracy < 60.0:
                            performance_multiplier = 0.7  # -30% for low accuracy
                            policy_adjustments.append(f"Low accuracy {accuracy:.1f}% (-30%)")
                        else:
                            policy_adjustments.append(f"Normal accuracy {accuracy:.1f}%")
                        
                        # Additional reduction if performance score is very low
                        if hasattr(module_perf, 'performance_score'):
                            perf_score = module_perf.performance_score
                            if perf_score < 50.0:
                                performance_multiplier *= 0.8  # Additional -20%
                                policy_adjustments.append(f"Low perf score {perf_score:.0f} (-20%)")
                        
                        risk_pct *= performance_multiplier
                        
                        logger.info(
                            f"[PHASE3C-PERF] {symbol} ({signal_source}): "
                            f"Accuracy={accuracy:.1f}%, Multiplier={performance_multiplier:.2f}x"
                        )
            except Exception as e:
                logger.warning(f"[PHASE3C-PERF] Failed for {symbol}: {e}")
        
        adjustment_reason = "; ".join(policy_adjustments) if policy_adjustments else None
        
        # Apply min/max constraints
        risk_pct = max(self.config.min_risk_pct, min(risk_pct, self.config.max_risk_pct))
        risk_usd = equity_usd * risk_pct
        
        # Step 2: Calculate stop loss distance
        # SL distance = ATR * k1
        sl_distance_price = atr * self.config.atr_multiplier_sl
        sl_distance_pct = sl_distance_price / current_price if current_price > 0 else 0
        
        # Safety check: prevent division by zero
        if sl_distance_pct <= 0:
            logger.error(
                f"[ERROR] {symbol} invalid SL distance: ATR={atr:.6f}, k1={self.config.atr_multiplier_sl}, "
                f"price={current_price:.4f}, sl_distance_pct={sl_distance_pct:.6f}. Using fallback 1% SL."
            )
            sl_distance_pct = 0.01  # Fallback to 1% stop loss
        
        # Step 3: Calculate position size
        # Position size = risk_amount / sl_distance
        # In USD: notional = risk_usd / sl_distance_pct
        notional_usd = risk_usd / sl_distance_pct
        
        # Step 4: Apply leverage constraint
        max_notional = equity_usd * self.config.max_leverage
        if notional_usd > max_notional:
            logger.warning(
                f"[WARNING]  {symbol} position size ${notional_usd:,.2f} exceeds max leverage "
                f"${max_notional:,.2f}, capping"
            )
            notional_usd = max_notional
            adjustment_reason = f"Leverage capped at {self.config.max_leverage}x"
        
        # Step 5: Apply min/max position size constraints
        if notional_usd < self.config.min_position_usd:
            logger.warning(
                f"[WARNING]  {symbol} position size ${notional_usd:,.2f} below minimum "
                f"${self.config.min_position_usd}, adjusting up"
            )
            notional_usd = self.config.min_position_usd
            adjustment_reason = "Raised to minimum position size"
        
        if notional_usd > self.config.max_position_usd:
            logger.info(
                f"ℹ️  {symbol} position size ${notional_usd:,.2f} above maximum "
                f"${self.config.max_position_usd}, capping"
            )
            notional_usd = self.config.max_position_usd
            adjustment_reason = "Capped at maximum position size"
        
        # Step 6: Calculate final quantity
        # quantity = notional / price
        quantity = notional_usd / current_price
        
        # Step 7: Calculate actual risk with final size
        actual_risk_usd = notional_usd * sl_distance_pct
        actual_risk_pct = actual_risk_usd / equity_usd
        
        # Step 8: Calculate leverage used
        leverage_used = notional_usd / equity_usd
        
        logger.info(
            f"[CHART] {symbol} {action} Position Sizing:\n"
            f"   Price: ${current_price:.4f}, ATR: ${atr:.4f}\n"
            f"   SL Distance: {sl_distance_pct:.2%} (${sl_distance_price:.4f})\n"
            f"   Base Risk: {base_risk_pct:.2%}\n"
            f"   [TARGET] Policy Risk Multiplier: {policy_risk_multiplier:.2%}\n"
            f"   [TARGET] Final Risk: ${actual_risk_usd:.2f} ({actual_risk_pct:.2%} of equity)\n"
            f"   Size: {quantity:.4f} units = ${notional_usd:.2f} notional\n"
            f"   Leverage: {leverage_used:.1f}x\n"
            + (f"   [TARGET] Adjustments: {adjustment_reason}\n" if adjustment_reason else "")
        )
        
        return PositionSize(
            quantity=quantity,
            notional_usd=notional_usd,
            risk_usd=actual_risk_usd,
            risk_pct=actual_risk_pct,
            leverage_used=leverage_used,
            sl_distance_pct=sl_distance_pct,
            adjustment_reason=adjustment_reason,
        )
    
    def validate_position_size(
        self,
        position_size: PositionSize,
        equity_usd: float,
    ) -> tuple[bool, Optional[str]]:
        """
        Final validation before trade execution.
        
        Returns:
            (is_valid, rejection_reason)
        """
        # Check risk percentage
        if position_size.risk_pct > self.config.max_risk_pct:
            return False, f"Risk {position_size.risk_pct:.2%} exceeds max {self.config.max_risk_pct:.2%}"
        
        # Check leverage
        if position_size.leverage_used > self.config.max_leverage:
            return False, f"Leverage {position_size.leverage_used:.1f}x exceeds max {self.config.max_leverage}x"
        
        # Check minimum position
        if position_size.notional_usd < self.config.min_position_usd:
            return False, f"Position ${position_size.notional_usd:.2f} below min ${self.config.min_position_usd}"
        
        # Check quantity is positive
        if position_size.quantity <= 0:
            return False, f"Invalid quantity: {position_size.quantity}"
        
        return True, None
