#!/usr/bin/env python3
"""
MODEL SUPERVISOR
================

Oversees AI models and ensembles for Quantum Trader.
Monitors performance, detects drift, ranks models, and decides when retraining is needed.

Mission: MONITOR MODEL PERFORMANCE, DETECT DRIFT, OPTIMIZE ENSEMBLE WEIGHTS

Author: Quantum Trader AI Team
Date: November 23, 2025
Version: 1.0
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import statistics
import math

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class ModelHealth(Enum):
    """Model health status"""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


class RetrainPriority(Enum):
    """Retraining priority levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


# Performance thresholds
MIN_WINRATE = 0.50  # 50% minimum
MIN_AVG_R = 0.0     # Break-even minimum
MIN_CALIBRATION = 0.70  # 70% calibration quality
MIN_SAMPLES = 20    # Minimum samples for evaluation

# Drift detection
MAX_PERFORMANCE_DROP = 0.15  # 15% drop triggers drift
MAX_CALIBRATION_DROP = 0.20  # 20% calibration drop


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SignalLog:
    """Individual signal prediction log"""
    timestamp: str
    model_id: str
    symbol: str
    prediction: str  # BUY/SELL/HOLD
    confidence: float
    
    # Context
    regime_tag: str = "UNKNOWN"
    vol_level: str = "NORMAL"
    
    # Outcome (filled later)
    realized_R: Optional[float] = None
    realized_pnl: Optional[float] = None
    outcome_known: bool = False
    outcome_timestamp: Optional[str] = None


@dataclass
class ModelMetrics:
    """Performance metrics for a single model"""
    model_id: str
    total_predictions: int
    predictions_with_outcome: int
    
    # Overall performance
    winrate: float = 0.0
    avg_R: float = 0.0
    median_R: float = 0.0
    total_R: float = 0.0
    profit_factor: float = 0.0
    
    # Calibration
    calibration_quality: float = 0.0
    confidence_vs_reality_error: float = 0.0
    
    # Distribution
    R_distribution: List[float] = field(default_factory=list)
    confidence_distribution: List[float] = field(default_factory=list)
    
    # Regime-specific
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Time-based
    recent_performance: Dict[str, float] = field(default_factory=dict)  # Last N days
    performance_trend: str = "STABLE"  # IMPROVING, STABLE, DEGRADING
    
    # Health
    health_status: str = "HEALTHY"
    issues: List[str] = field(default_factory=list)


@dataclass
class ModelRanking:
    """Model ranking entry"""
    model_id: str
    rank: int
    overall_score: float
    winrate: float
    avg_R: float
    calibration: float
    health: str
    recommended_weight: float


@dataclass
class EnsembleWeightSuggestion:
    """Suggested ensemble weights"""
    timestamp: str
    
    # Overall weights
    overall_weights: Dict[str, float] = field(default_factory=dict)
    
    # Regime-specific weights
    regime_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Justification
    reasoning: List[str] = field(default_factory=list)


@dataclass
class RetrainRecommendation:
    """Model retraining recommendation"""
    model_id: str
    priority: str
    reasons: List[str]
    current_performance: Dict[str, float]
    target_improvement: Dict[str, float]
    suggested_actions: List[str]


@dataclass
class SupervisorOutput:
    """Model Supervisor output"""
    timestamp: str
    analysis_period_days: int
    
    # Model metrics
    model_metrics: Dict[str, ModelMetrics]
    
    # Rankings
    model_rankings: List[ModelRanking]
    
    # Ensemble suggestions
    ensemble_weights: EnsembleWeightSuggestion
    
    # Health flags
    healthy_models: List[str]
    degraded_models: List[str]
    critical_models: List[str]
    
    # Retraining
    retrain_recommendations: List[RetrainRecommendation]
    
    # Summary
    summary: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# MODEL SUPERVISOR
# ============================================================================

class ModelSupervisor:
    """
    Model Supervisor
    
    Oversees AI models and ensembles, monitors performance, detects drift,
    ranks models, and recommends retraining or reweighting.
    """
    
    def __init__(
        self,
        data_dir: str = "/app/data",
        analysis_window_days: int = 30,
        recent_window_days: int = 7,
        event_bus = None  # [NEW] EventBus for real-time subscriptions
    ):
        """Initialize Model Supervisor"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.analysis_window = analysis_window_days
        self.recent_window = recent_window_days
        
        # Get mode from environment
        import os
        self.mode = os.getenv("QT_MODEL_SUPERVISOR_MODE", "ENFORCED").upper()
        
        # Storage
        self.signal_logs: List[SignalLog] = []
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info("=" * 80)
        logger.info("MODEL SUPERVISOR - INITIALIZING")
        logger.info("=" * 80)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Analysis Window: {analysis_window_days} days")
        logger.info(f"Recent Window: {recent_window_days} days")
        logger.info(f"Min Winrate: {MIN_WINRATE:.0%}")
        logger.info(f"Min Avg R: {MIN_AVG_R:.2f}")
        logger.info(f"Min Calibration: {MIN_CALIBRATION:.0%}")
        
        # [NEW] Real-time observation tracking
        self.realtime_signal_count = defaultdict(int)
        self.realtime_action_count = defaultdict(lambda: {"BUY": 0, "SELL": 0, "HOLD": 0})
        self.realtime_window_start = datetime.now(timezone.utc)
        
        # [FIX #1] EventBus integration for real-time drift detection
        self.event_bus = event_bus
        self.realtime_model_performance: Dict[str, List[float]] = defaultdict(list)  # Rolling R-values
        self.drift_alert_threshold = 3  # Alert after 3 consecutive losses
        
        if self.event_bus:
            # Subscribe to trade outcomes for real-time drift detection
            import asyncio
            asyncio.create_task(self._subscribe_to_trade_events())
            logger.info("[FIX #1] ✅ EventBus subscription enabled - real-time drift detection active")
        else:
            logger.warning("[FIX #1] ⚠️ EventBus not provided - drift detection will be delayed")
    
    async def _subscribe_to_trade_events(self) -> None:
        """[FIX #1] Subscribe to trade.closed events for real-time drift detection."""
        try:
            await self.event_bus.subscribe(
                stream_name="trade.closed",
                consumer_group="model_supervisor",
                handler=self._handle_trade_closed
            )
            logger.info("[FIX #1] Subscribed to trade.closed events")
        except Exception as e:
            logger.error(f"[FIX #1] Failed to subscribe to trade events: {e}")
    
    async def _handle_trade_closed(self, event: Dict[str, Any]) -> None:
        """[FIX #1] Handle trade.closed event - detect drift in real-time."""
        try:
            model_id = event.get("model", "unknown")
            r_multiple = event.get("r_multiple", 0.0)
            pnl_pct = event.get("pnl_pct", 0.0)
            
            # Track rolling performance (last 10 trades per model)
            self.realtime_model_performance[model_id].append(r_multiple)
            if len(self.realtime_model_performance[model_id]) > 10:
                self.realtime_model_performance[model_id].pop(0)
            
            # Check for drift: 3+ consecutive losses
            recent = self.realtime_model_performance[model_id][-self.drift_alert_threshold:]
            if len(recent) >= self.drift_alert_threshold and all(r < 0 for r in recent):
                avg_loss = sum(recent) / len(recent)
                logger.warning(
                    f"[FIX #1] 🚨 DRIFT DETECTED: {model_id} - {self.drift_alert_threshold} consecutive losses "
                    f"(avg R={avg_loss:.2f})"
                )
                
                # Publish drift.detected event
                if self.event_bus:
                    await self.event_bus.publish("model.drift_detected", {
                        "model_id": model_id,
                        "consecutive_losses": self.drift_alert_threshold,
                        "avg_r_multiple": avg_loss,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "severity": "HIGH" if avg_loss < -0.5 else "MEDIUM"
                    })
            
            # Update metrics for weight adjustment
            if len(self.realtime_model_performance[model_id]) >= 5:
                recent_5 = self.realtime_model_performance[model_id][-5:]
                winrate = sum(1 for r in recent_5 if r > 0) / len(recent_5)
                avg_r = sum(recent_5) / len(recent_5)
                
                # Store for dynamic weight calculation
                self.model_metadata[model_id] = {
                    "recent_winrate": winrate,
                    "recent_avg_r": avg_r,
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"[FIX #1] Error handling trade.closed event: {e}")
    
    def observe(
        self,
        signal: Optional[Dict[str, Any]] = None,
        trade_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Real-time observation method for OBSERVATION MODE.
        
        Tracks signal bias, confidence patterns, and trade outcomes in real-time.
        Does NOT enforce any decisions - logging only.
        
        Args:
            signal: Signal/decision from AI with keys:
                - symbol: str
                - action: str (BUY/SELL/HOLD)
                - confidence: float
                - model_predictions: dict (optional)
                - regime: str (optional)
            trade_result: Trade outcome with keys:
                - symbol: str
                - pnl: float
                - r_multiple: float
                - outcome: str (WIN/LOSS)
        
        Usage:
            # After signal generation:
            model_supervisor.observe(signal=decision)
            
            # After trade closes:
            model_supervisor.observe(trade_result=result)
        """
        try:
            if signal:
                self._observe_signal(signal)
            
            if trade_result:
                self._observe_trade_result(trade_result)
        
        except Exception as e:
            logger.error(f"[MODEL_SUPERVISOR] observe() error: {e}", exc_info=True)
    
    def check_bias_and_block(self, action: str, min_samples: int = 20, bias_threshold: float = 0.70) -> Tuple[bool, str]:
        """
        Check if there's excessive directional bias and block trade if needed.
        
        Args:
            action: Proposed action (BUY/SELL/LONG/SHORT)
            min_samples: Minimum signals needed to detect bias
            bias_threshold: Max allowed bias (0.70 = 70% of one direction)
        
        Returns:
            (should_block, reason)
        """
        try:
            total_signals = sum(
                self.realtime_action_count[regime].get("BUY", 0) + 
                self.realtime_action_count[regime].get("SELL", 0)
                for regime in self.realtime_action_count
            )
            
            if total_signals < min_samples:
                return False, f"Insufficient data ({total_signals}/{min_samples} signals)"
            
            total_buy = sum(
                self.realtime_action_count[regime].get("BUY", 0)
                for regime in self.realtime_action_count
            )
            total_sell = sum(
                self.realtime_action_count[regime].get("SELL", 0)
                for regime in self.realtime_action_count
            )
            
            if total_signals == 0:
                return False, "No trading signals"
            
            buy_pct = total_buy / total_signals
            sell_pct = total_sell / total_signals
            
            # Normalize action
            normalized_action = action.upper()
            if normalized_action in ["LONG", "BUY"]:
                normalized_action = "BUY"
            elif normalized_action in ["SHORT", "SELL"]:
                normalized_action = "SELL"
            
            # Block if bias exceeds threshold AND action matches bias direction
            if sell_pct > bias_threshold and normalized_action == "SELL":
                logger.warning(
                    f"[MODEL_SUPERVISOR] BLOCKING {action}: "
                    f"SHORT bias detected ({sell_pct:.1%} of {total_signals} signals). "
                    f"Threshold: {bias_threshold:.1%}"
                )
                return True, f"SHORT bias {sell_pct:.1%} exceeds threshold {bias_threshold:.1%}"
            
            if buy_pct > bias_threshold and normalized_action == "BUY":
                logger.warning(
                    f"[MODEL_SUPERVISOR] BLOCKING {action}: "
                    f"LONG bias detected ({buy_pct:.1%} of {total_signals} signals). "
                    f"Threshold: {bias_threshold:.1%}"
                )
                return True, f"LONG bias {buy_pct:.1%} exceeds threshold {bias_threshold:.1%}"
            
            # Log current bias (no blocking)
            logger.debug(
                f"[MODEL_SUPERVISOR] Bias check OK: "
                f"BUY={buy_pct:.1%}, SELL={sell_pct:.1%} ({total_signals} signals)"
            )
            return False, f"Bias OK: BUY={buy_pct:.1%}, SELL={sell_pct:.1%}"
        
        except Exception as e:
            logger.error(f"[MODEL_SUPERVISOR] check_bias_and_block() error: {e}", exc_info=True)
            return False, f"Error checking bias: {e}"
    
    def _observe_signal(self, signal: Dict[str, Any]) -> None:
        """Observe a new signal for bias detection."""
        symbol = signal.get("symbol", "UNKNOWN")
        action = signal.get("action", "UNKNOWN")
        confidence = signal.get("confidence", 0.0)
        regime = signal.get("regime", "UNKNOWN")
        
        # Track counts
        self.realtime_signal_count["total"] += 1
        self.realtime_action_count[regime][action] += 1
        
        # Check for SHORT bias in uptrend
        if regime in ["BULL", "UPTREND"] and action == "SELL":
            self.realtime_action_count["SHORT_IN_UPTREND"]["count"] += 1
            
            # Calculate bias ratio every 10 signals
            if self.realtime_signal_count["total"] % 10 == 0:
                total_in_uptrend = sum(self.realtime_action_count.get("BULL", {}).values())
                shorts_in_uptrend = self.realtime_action_count.get("BULL", {}).get("SELL", 0)
                
                if total_in_uptrend > 0:
                    short_ratio = shorts_in_uptrend / total_in_uptrend
                    
                    if short_ratio > 0.70:
                        logger.warning(
                            f"[MODEL_SUPERVISOR] SHORT BIAS DETECTED in UPTREND: "
                            f"{short_ratio:.0%} of signals are SHORT ({shorts_in_uptrend}/{total_in_uptrend}). "
                            f"Last: {symbol} SELL @ {confidence:.1%} confidence"
                        )
        
        # Log high-confidence signals
        if confidence >= 0.60:
            logger.info(
                f"[MODEL_SUPERVISOR] High-confidence signal: {symbol} {action} "
                f"@ {confidence:.1%} in {regime} regime"
            )
    
    def _observe_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """Observe a trade outcome."""
        symbol = trade_result.get("symbol", "UNKNOWN")
        r_multiple = trade_result.get("r_multiple", 0.0)
        outcome = trade_result.get("outcome", "UNKNOWN")
        pnl = trade_result.get("pnl", 0.0)
        
        logger.info(
            f"[MODEL_SUPERVISOR] Trade closed: {symbol} {outcome} "
            f"R={r_multiple:.2f} PnL=${pnl:.2f}"
        )
        
        # Track outcome
        self.realtime_signal_count[f"trade_{outcome.lower()}"] += 1
    
    def reset_observation_window(self) -> None:
        """Reset observation window (called periodically, e.g., daily)."""
        logger.info(
            f"[MODEL_SUPERVISOR] Resetting observation window. "
            f"Tracked {self.realtime_signal_count['total']} signals since "
            f"{self.realtime_window_start.isoformat()}"
        )
        self.realtime_signal_count.clear()
        self.realtime_action_count.clear()
        self.realtime_window_start = datetime.now(timezone.utc)
    
    def analyze_models(
        self,
        signal_logs: List[Dict[str, Any]],
        model_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> SupervisorOutput:
        """
        Main analysis method
        
        Args:
            signal_logs: List of signal prediction logs with outcomes
            model_metadata: Optional metadata about models (training date, etc.)
        
        Returns:
            SupervisorOutput with metrics, rankings, and recommendations
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        logger.info("=" * 80)
        logger.info("MODEL SUPERVISOR - ANALYSIS STARTING")
        logger.info("=" * 80)
        logger.info(f"Total Signal Logs: {len(signal_logs)}")
        
        # Convert to SignalLog objects
        self.signal_logs = self._parse_signal_logs(signal_logs)
        self.model_metadata = model_metadata or {}
        
        # Filter to analysis window
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.analysis_window)
        filtered_logs = [
            log for log in self.signal_logs
            if datetime.fromisoformat(log.timestamp.replace('Z', '+00:00')) >= cutoff_date
        ]
        
        logger.info(f"Signals in Analysis Window: {len(filtered_logs)}")
        
        # 1. Compute per-model metrics
        model_metrics = self._compute_model_metrics(filtered_logs)
        
        # 2. Rank models
        model_rankings = self._rank_models(model_metrics)
        
        # 3. Generate ensemble weight suggestions
        ensemble_weights = self._suggest_ensemble_weights(model_metrics, model_rankings)
        
        # 4. Categorize by health
        healthy, degraded, critical = self._categorize_by_health(model_metrics)
        
        # 5. Generate retrain recommendations
        retrain_recommendations = self._generate_retrain_recommendations(model_metrics)
        
        # 6. Create summary
        summary = self._create_summary(model_metrics, model_rankings, retrain_recommendations)
        
        # Create output
        output = SupervisorOutput(
            timestamp=timestamp,
            analysis_period_days=self.analysis_window,
            model_metrics=model_metrics,
            model_rankings=model_rankings,
            ensemble_weights=ensemble_weights,
            healthy_models=healthy,
            degraded_models=degraded,
            critical_models=critical,
            retrain_recommendations=retrain_recommendations,
            summary=summary
        )
        
        # Log summary
        self._log_summary(output)
        
        # Save to disk
        self._save_output(output)
        
        return output
    
    def _parse_signal_logs(self, raw_logs: List[Dict[str, Any]]) -> List[SignalLog]:
        """Parse raw signal logs into SignalLog objects"""
        parsed = []
        for log in raw_logs:
            try:
                parsed.append(SignalLog(
                    timestamp=log.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    model_id=log.get("model_id", "unknown"),
                    symbol=log.get("symbol", "UNKNOWN"),
                    prediction=log.get("prediction", "HOLD"),
                    confidence=float(log.get("confidence", 0.5)),
                    regime_tag=log.get("regime_tag", "UNKNOWN"),
                    vol_level=log.get("vol_level", "NORMAL"),
                    realized_R=log.get("realized_R"),
                    realized_pnl=log.get("realized_pnl"),
                    outcome_known=log.get("outcome_known", False),
                    outcome_timestamp=log.get("outcome_timestamp")
                ))
            except Exception as e:
                logger.warning(f"Failed to parse signal log: {e}")
                continue
        
        return parsed
    
    def _compute_model_metrics(
        self,
        logs: List[SignalLog]
    ) -> Dict[str, ModelMetrics]:
        """Compute comprehensive metrics for each model"""
        
        # Group by model
        logs_by_model: Dict[str, List[SignalLog]] = defaultdict(list)
        for log in logs:
            logs_by_model[log.model_id].append(log)
        
        metrics = {}
        
        for model_id, model_logs in logs_by_model.items():
            logger.info(f"Computing metrics for {model_id}...")
            
            # Filter to logs with known outcomes
            outcome_logs = [log for log in model_logs if log.outcome_known and log.realized_R is not None]
            
            if len(outcome_logs) < MIN_SAMPLES:
                logger.warning(f"  [WARNING] {model_id}: Only {len(outcome_logs)} samples (min {MIN_SAMPLES})")
            
            # Overall performance
            R_values = [log.realized_R for log in outcome_logs]
            wins = sum(1 for r in R_values if r > 0)
            losses = sum(1 for r in R_values if r < 0)
            
            winrate = wins / len(outcome_logs) if outcome_logs else 0.0
            avg_R = statistics.mean(R_values) if R_values else 0.0
            median_R = statistics.median(R_values) if R_values else 0.0
            total_R = sum(R_values)
            
            # Profit factor
            gross_wins = sum(r for r in R_values if r > 0)
            gross_losses = abs(sum(r for r in R_values if r < 0))
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0.0
            
            # Calibration
            calibration_quality, conf_vs_reality_error = self._compute_calibration(outcome_logs)
            
            # Regime-specific performance
            regime_perf = self._compute_regime_performance(outcome_logs)
            
            # Recent performance
            recent_perf = self._compute_recent_performance(outcome_logs, model_logs)
            
            # Performance trend
            trend = self._detect_performance_trend(outcome_logs)
            
            # Health assessment
            health, issues = self._assess_model_health(
                winrate, avg_R, calibration_quality, len(outcome_logs)
            )
            
            metrics[model_id] = ModelMetrics(
                model_id=model_id,
                total_predictions=len(model_logs),
                predictions_with_outcome=len(outcome_logs),
                winrate=winrate,
                avg_R=avg_R,
                median_R=median_R,
                total_R=total_R,
                profit_factor=profit_factor,
                calibration_quality=calibration_quality,
                confidence_vs_reality_error=conf_vs_reality_error,
                R_distribution=R_values,
                confidence_distribution=[log.confidence for log in model_logs],
                regime_performance=regime_perf,
                recent_performance=recent_perf,
                performance_trend=trend,
                health_status=health.value,
                issues=issues
            )
            
            logger.info(f"  Metrics: WR={winrate:.1%}, AvgR={avg_R:.3f}, Cal={calibration_quality:.1%}, Health={health.value}")
        
        return metrics
    
    def _compute_calibration(
        self,
        logs: List[SignalLog]
    ) -> Tuple[float, float]:
        """
        Compute calibration quality
        
        Calibration = how well confidence scores match actual outcomes
        Perfect calibration: 70% confidence → 70% success rate
        """
        if not logs:
            return 0.0, 1.0
        
        # Group by confidence buckets
        buckets = defaultdict(list)
        for log in logs:
            bucket = int(log.confidence * 10) / 10  # Round to 0.1
            buckets[bucket].append(log.realized_R > 0)
        
        # Compute calibration error
        errors = []
        for conf, outcomes in buckets.items():
            actual_success_rate = sum(outcomes) / len(outcomes)
            error = abs(conf - actual_success_rate)
            errors.append(error)
        
        if not errors:
            return 0.0, 1.0
        
        avg_error = statistics.mean(errors)
        calibration_quality = 1.0 - avg_error  # Higher = better
        
        return calibration_quality, avg_error
    
    def _compute_regime_performance(
        self,
        logs: List[SignalLog]
    ) -> Dict[str, Dict[str, float]]:
        """Compute performance per regime"""
        regime_logs: Dict[str, List[SignalLog]] = defaultdict(list)
        for log in logs:
            regime_logs[log.regime_tag].append(log)
        
        performance = {}
        for regime, regime_log_list in regime_logs.items():
            R_values = [log.realized_R for log in regime_log_list if log.realized_R is not None]
            wins = sum(1 for r in R_values if r > 0)
            
            performance[regime] = {
                "count": len(regime_log_list),
                "winrate": wins / len(R_values) if R_values else 0.0,
                "avg_R": statistics.mean(R_values) if R_values else 0.0,
                "total_R": sum(R_values)
            }
        
        return performance
    
    def _compute_recent_performance(
        self,
        outcome_logs: List[SignalLog],
        all_logs: List[SignalLog]
    ) -> Dict[str, float]:
        """Compute performance in recent window"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.recent_window)
        
        recent_logs = [
            log for log in outcome_logs
            if datetime.fromisoformat(log.timestamp.replace('Z', '+00:00')) >= cutoff
        ]
        
        if not recent_logs:
            return {"count": 0, "winrate": 0.0, "avg_R": 0.0}
        
        R_values = [log.realized_R for log in recent_logs if log.realized_R is not None]
        wins = sum(1 for r in R_values if r > 0)
        
        return {
            "count": len(recent_logs),
            "winrate": wins / len(R_values) if R_values else 0.0,
            "avg_R": statistics.mean(R_values) if R_values else 0.0
        }
    
    def _detect_performance_trend(
        self,
        logs: List[SignalLog]
    ) -> str:
        """Detect if performance is improving, stable, or degrading"""
        if len(logs) < 40:  # Need enough samples
            return "STABLE"
        
        # Split into first half and second half
        mid = len(logs) // 2
        first_half = logs[:mid]
        second_half = logs[mid:]
        
        first_R = [log.realized_R for log in first_half if log.realized_R is not None]
        second_R = [log.realized_R for log in second_half if log.realized_R is not None]
        
        if not first_R or not second_R:
            return "STABLE"
        
        first_avg = statistics.mean(first_R)
        second_avg = statistics.mean(second_R)
        
        change = (second_avg - first_avg) / (abs(first_avg) + 0.01)  # Avoid div by zero
        
        if change > 0.15:  # 15% improvement
            return "IMPROVING"
        elif change < -0.15:  # 15% degradation
            return "DEGRADING"
        else:
            return "STABLE"
    
    def _assess_model_health(
        self,
        winrate: float,
        avg_R: float,
        calibration: float,
        sample_count: int
    ) -> Tuple[ModelHealth, List[str]]:
        """Assess model health status"""
        issues = []
        
        # Check critical failures
        critical_issues = 0
        
        if winrate < MIN_WINRATE:
            issues.append(f"Winrate below minimum: {winrate:.1%} < {MIN_WINRATE:.1%}")
            critical_issues += 1
        
        if avg_R < MIN_AVG_R:
            issues.append(f"Avg R below break-even: {avg_R:.3f} < {MIN_AVG_R:.3f}")
            critical_issues += 1
        
        if calibration < MIN_CALIBRATION:
            issues.append(f"Poor calibration: {calibration:.1%} < {MIN_CALIBRATION:.1%}")
            if calibration < 0.5:  # Very poor
                critical_issues += 1
        
        if sample_count < MIN_SAMPLES:
            issues.append(f"Insufficient samples: {sample_count} < {MIN_SAMPLES}")
        
        # Determine health
        if critical_issues >= 2:
            return ModelHealth.CRITICAL, issues
        elif critical_issues == 1 or len(issues) >= 2:
            return ModelHealth.DEGRADED, issues
        else:
            return ModelHealth.HEALTHY, issues
    
    def _rank_models(
        self,
        metrics: Dict[str, ModelMetrics]
    ) -> List[ModelRanking]:
        """Rank models by overall performance"""
        
        rankings = []
        
        for model_id, metric in metrics.items():
            # Compute overall score (weighted combination)
            score = (
                metric.winrate * 0.35 +           # 35% weight on winrate
                max(0, metric.avg_R) * 0.35 +     # 35% weight on avg R
                metric.calibration_quality * 0.20 + # 20% weight on calibration
                (metric.profit_factor / 10) * 0.10  # 10% weight on profit factor
            )
            
            # Penalty for insufficient samples
            if metric.predictions_with_outcome < MIN_SAMPLES:
                score *= 0.5
            
            # Penalty for degraded health
            if metric.health_status == "DEGRADED":
                score *= 0.8
            elif metric.health_status == "CRITICAL":
                score *= 0.5
            
            rankings.append(ModelRanking(
                model_id=model_id,
                rank=0,  # Will be set after sorting
                overall_score=score,
                winrate=metric.winrate,
                avg_R=metric.avg_R,
                calibration=metric.calibration_quality,
                health=metric.health_status,
                recommended_weight=0.0  # Will be computed from rank
            ))
        
        # Sort by score (highest first)
        rankings.sort(key=lambda r: r.overall_score, reverse=True)
        
        # Assign ranks and weights
        total_score = sum(r.overall_score for r in rankings) or 1.0
        
        for i, ranking in enumerate(rankings, 1):
            ranking.rank = i
            ranking.recommended_weight = ranking.overall_score / total_score
        
        return rankings
    
    def _suggest_ensemble_weights(
        self,
        metrics: Dict[str, ModelMetrics],
        rankings: List[ModelRanking]
    ) -> EnsembleWeightSuggestion:
        """Generate ensemble weight suggestions"""
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Overall weights from rankings
        overall_weights = {
            r.model_id: round(r.recommended_weight, 3)
            for r in rankings
        }
        
        # Regime-specific weights
        regime_weights = {}
        
        # Get all regimes
        all_regimes = set()
        for metric in metrics.values():
            all_regimes.update(metric.regime_performance.keys())
        
        for regime in all_regimes:
            regime_scores = {}
            for model_id, metric in metrics.items():
                if regime in metric.regime_performance:
                    perf = metric.regime_performance[regime]
                    # Score based on regime performance
                    score = perf["winrate"] * 0.5 + max(0, perf["avg_R"]) * 0.5
                    regime_scores[model_id] = score
            
            # Normalize to weights
            total = sum(regime_scores.values()) or 1.0
            regime_weights[regime] = {
                model_id: round(score / total, 3)
                for model_id, score in regime_scores.items()
            }
        
        # Reasoning
        reasoning = []
        reasoning.append(f"Overall weights based on {len(rankings)} models")
        
        if rankings:
            best = rankings[0]
            reasoning.append(
                f"Best model: {best.model_id} "
                f"(WR:{best.winrate:.1%}, R:{best.avg_R:.3f}, weight:{best.recommended_weight:.1%})"
            )
        
        # Regime-specific notes
        for regime, weights in regime_weights.items():
            if weights:
                best_in_regime = max(weights.items(), key=lambda x: x[1])
                reasoning.append(
                    f"{regime}: {best_in_regime[0]} strongest (weight:{best_in_regime[1]:.1%})"
                )
        
        return EnsembleWeightSuggestion(
            timestamp=timestamp,
            overall_weights=overall_weights,
            regime_weights=regime_weights,
            reasoning=reasoning
        )
    
    def _categorize_by_health(
        self,
        metrics: Dict[str, ModelMetrics]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Categorize models by health status"""
        healthy = []
        degraded = []
        critical = []
        
        for model_id, metric in metrics.items():
            if metric.health_status == "HEALTHY":
                healthy.append(model_id)
            elif metric.health_status == "DEGRADED":
                degraded.append(model_id)
            else:
                critical.append(model_id)
        
        return healthy, degraded, critical
    
    def _generate_retrain_recommendations(
        self,
        metrics: Dict[str, ModelMetrics]
    ) -> List[RetrainRecommendation]:
        """Generate model retraining recommendations"""
        
        recommendations = []
        
        for model_id, metric in metrics.items():
            # Skip healthy models with good performance
            if (metric.health_status == "HEALTHY" and 
                metric.performance_trend != "DEGRADING"):
                continue
            
            # Determine priority
            if metric.health_status == "CRITICAL":
                priority = RetrainPriority.URGENT
            elif metric.health_status == "DEGRADED":
                priority = RetrainPriority.HIGH
            elif metric.performance_trend == "DEGRADING":
                priority = RetrainPriority.MEDIUM
            else:
                priority = RetrainPriority.LOW
            
            # Collect reasons
            reasons = metric.issues.copy()
            if metric.performance_trend == "DEGRADING":
                reasons.append("Performance degrading over time")
            
            # Current performance
            current_perf = {
                "winrate": metric.winrate,
                "avg_R": metric.avg_R,
                "calibration": metric.calibration_quality
            }
            
            # Target improvement
            target_improvement = {
                "winrate": max(MIN_WINRATE, metric.winrate * 1.1),
                "avg_R": max(MIN_AVG_R, metric.avg_R * 1.2),
                "calibration": max(MIN_CALIBRATION, metric.calibration_quality * 1.1)
            }
            
            # Suggested actions
            actions = []
            if metric.winrate < MIN_WINRATE:
                actions.append("Focus on improving prediction accuracy")
            if metric.avg_R < 0:
                actions.append("Review risk/reward ratios in training")
            if metric.calibration_quality < MIN_CALIBRATION:
                actions.append("Re-calibrate confidence scores")
            if metric.predictions_with_outcome < MIN_SAMPLES * 2:
                actions.append("Collect more training data before retraining")
            
            # Check regime performance
            for regime, perf in metric.regime_performance.items():
                if perf["winrate"] < 0.45:
                    actions.append(f"Improve performance in {regime} regime")
            
            recommendations.append(RetrainRecommendation(
                model_id=model_id,
                priority=priority.value,
                reasons=reasons,
                current_performance=current_perf,
                target_improvement=target_improvement,
                suggested_actions=actions
            ))
        
        # Sort by priority
        priority_order = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 4))
        
        return recommendations
    
    def _create_summary(
        self,
        metrics: Dict[str, ModelMetrics],
        rankings: List[ModelRanking],
        recommendations: List[RetrainRecommendation]
    ) -> Dict[str, Any]:
        """Create analysis summary"""
        
        healthy_count = sum(1 for m in metrics.values() if m.health_status == "HEALTHY")
        degraded_count = sum(1 for m in metrics.values() if m.health_status == "DEGRADED")
        critical_count = sum(1 for m in metrics.values() if m.health_status == "CRITICAL")
        
        return {
            "total_models": len(metrics),
            "healthy_models": healthy_count,
            "degraded_models": degraded_count,
            "critical_models": critical_count,
            "models_needing_retrain": len(recommendations),
            "urgent_retrains": sum(1 for r in recommendations if r.priority == "URGENT"),
            "best_model": rankings[0].model_id if rankings else None,
            "worst_model": rankings[-1].model_id if rankings else None,
            "avg_winrate": statistics.mean([m.winrate for m in metrics.values()]) if metrics else 0.0,
            "avg_R": statistics.mean([m.avg_R for m in metrics.values()]) if metrics else 0.0,
            "avg_calibration": statistics.mean([m.calibration_quality for m in metrics.values()]) if metrics else 0.0
        }
    
    def _log_summary(self, output: SupervisorOutput):
        """Log analysis summary"""
        logger.info("=" * 80)
        logger.info("MODEL SUPERVISOR - ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total Models: {output.summary['total_models']}")
        logger.info(f"  [OK] Healthy: {output.summary['healthy_models']}")
        logger.info(f"  [WARNING] Degraded: {output.summary['degraded_models']}")
        logger.info(f"  🚨 Critical: {output.summary['critical_models']}")
        logger.info(f"Retrain Recommendations: {output.summary['models_needing_retrain']}")
        if output.summary['urgent_retrains'] > 0:
            logger.warning(f"  🚨 URGENT: {output.summary['urgent_retrains']} models")
        
        logger.info(f"\nModel Rankings:")
        for i, ranking in enumerate(output.model_rankings[:5], 1):
            logger.info(
                f"  {i}. {ranking.model_id}: "
                f"Score={ranking.overall_score:.3f}, "
                f"WR={ranking.winrate:.1%}, "
                f"R={ranking.avg_R:.3f}, "
                f"Weight={ranking.recommended_weight:.1%}"
            )
        
        if output.retrain_recommendations:
            logger.warning("\n[WARNING] RETRAIN RECOMMENDATIONS:")
            for rec in output.retrain_recommendations[:3]:
                logger.warning(f"  [{rec.priority}] {rec.model_id}: {', '.join(rec.reasons[:2])}")
    
    def _save_output(self, output: SupervisorOutput):
        """Save output to disk"""
        try:
            output_file = self.data_dir / "model_supervisor_output.json"
            
            # Convert to dict
            output_dict = {
                "timestamp": output.timestamp,
                "analysis_period_days": output.analysis_period_days,
                "model_metrics": {
                    model_id: asdict(metrics)
                    for model_id, metrics in output.model_metrics.items()
                },
                "model_rankings": [asdict(r) for r in output.model_rankings],
                "ensemble_weights": asdict(output.ensemble_weights),
                "healthy_models": output.healthy_models,
                "degraded_models": output.degraded_models,
                "critical_models": output.critical_models,
                "retrain_recommendations": [asdict(r) for r in output.retrain_recommendations],
                "summary": output.summary
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_dict, f, indent=2)
            
            logger.info(f"[OK] Output saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
    
    async def monitor_loop(self):
        """Continuous monitoring loop for real-time observation."""
        import asyncio
        
        logger.info("\n" + "=" * 80)
        logger.info("🔍 MODEL SUPERVISOR - STARTING CONTINUOUS MONITORING")
        logger.info("=" * 80)
        logger.info(f"Mode: {self.mode.upper()} ({'block biased trades' if self.mode == 'enforced' else 'bias detection & performance tracking'})")
        logger.info(f"Check interval: Every signal/trade")
        logger.info("=" * 80 + "\n")
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                
                # Log status every hour
                if iteration % 60 == 0:  # Assuming 1-minute interval
                    total_signals = self.realtime_signal_count.get("total", 0)
                    wins = self.realtime_signal_count.get("trade_win", 0)
                    losses = self.realtime_signal_count.get("trade_loss", 0)
                    
                    logger.info(
                        f"🔍 [MODEL_SUPERVISOR] Hourly status - "
                        f"Signals tracked: {total_signals}, "
                        f"Trades: {wins}W/{losses}L"
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                logger.info("🔍 [MODEL_SUPERVISOR] Monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"[MODEL_SUPERVISOR] Monitor loop error: {e}", exc_info=True)
                await asyncio.sleep(60)


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

def main():
    """Test the Model Supervisor"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test data
    signal_logs = []
    
    # Model A: Good performance
    for i in range(50):
        signal_logs.append({
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=i // 2)).isoformat(),
            "model_id": "xgboost_v1",
            "symbol": "BTCUSDT",
            "prediction": "BUY" if i % 3 != 0 else "SELL",
            "confidence": 0.65 + (i % 10) * 0.02,
            "regime_tag": "TRENDING" if i % 2 == 0 else "RANGING",
            "vol_level": "NORMAL",
            "realized_R": 0.5 if i % 10 < 6 else -0.3,  # 60% winrate
            "outcome_known": True
        })
    
    # Model B: Poor performance
    for i in range(40):
        signal_logs.append({
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=i // 2)).isoformat(),
            "model_id": "lstm_v1",
            "symbol": "ETHUSDT",
            "prediction": "BUY",
            "confidence": 0.70,
            "regime_tag": "RANGING",
            "vol_level": "HIGH",
            "realized_R": -0.4 if i % 10 < 6 else 0.3,  # 40% winrate (poor)
            "outcome_known": True
        })
    
    # Model C: Degrading performance
    for i in range(60):
        winrate = 0.65 if i < 30 else 0.45  # Performance drops
        signal_logs.append({
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=i // 2)).isoformat(),
            "model_id": "ensemble_v2",
            "symbol": "SOLUSDT",
            "prediction": "BUY",
            "confidence": 0.72,
            "regime_tag": "TRENDING",
            "vol_level": "NORMAL",
            "realized_R": 0.6 if i % 10 < winrate * 10 else -0.4,
            "outcome_known": True
        })
    
    # Create supervisor
    supervisor = ModelSupervisor(
        data_dir="/app/data",
        analysis_window_days=30,
        recent_window_days=7
    )
    
    # Run analysis
    output = supervisor.analyze_models(signal_logs)
    
    print("\n" + "=" * 80)
    print("ENSEMBLE WEIGHT SUGGESTIONS:")
    print("=" * 80)
    print("\nOverall Weights:")
    for model_id, weight in output.ensemble_weights.overall_weights.items():
        print(f"  {model_id}: {weight:.1%}")
    
    print("\nRegime-Specific Weights:")
    for regime, weights in output.ensemble_weights.regime_weights.items():
        print(f"\n  {regime}:")
        for model_id, weight in weights.items():
            print(f"    {model_id}: {weight:.1%}")
    
    if output.retrain_recommendations:
        print("\n" + "=" * 80)
        print("RETRAIN RECOMMENDATIONS:")
        print("=" * 80)
        for rec in output.retrain_recommendations:
            print(f"\n[{rec.priority}] {rec.model_id}")
            print(f"  Reasons: {', '.join(rec.reasons)}")
            print(f"  Actions: {', '.join(rec.suggested_actions[:2])}")


if __name__ == "__main__":
    main()
