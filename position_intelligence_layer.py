#!/usr/bin/env python3
"""
POSITION INTELLIGENCE LAYER (PIL)
==================================

Autonomous per-position lifecycle management system for Quantum Trader.

Mission: MAXIMIZE PER-POSITION OUTCOME BY DYNAMICALLY MANAGING HOLD TIME,
         SIZING ADJUSTMENTS (SCALE-IN/OUT), AND EXIT TACTICS.

This system does NOT generate signals.
It analyzes and manages EACH OPEN POSITION over its entire lifecycle.

Author: Senior Quant Developer
Date: November 23, 2025
Version: 1.0
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from enum import Enum
import math
import statistics

# ============================================================================
# CONFIGURATION & ENVIRONMENT
# ============================================================================

PIL_MODE = os.getenv("PIL_MODE", "ADVISORY")  # ADVISORY | AUTONOMOUS
PIL_UPDATE_INTERVAL_SECONDS = int(os.getenv("PIL_UPDATE_INTERVAL_SECONDS", "60"))

# Position classification thresholds
STRONG_TREND_MIN_R = float(os.getenv("STRONG_TREND_MIN_R", "1.0"))
STRONG_TREND_MIN_MOMENTUM = float(os.getenv("STRONG_TREND_MIN_MOMENTUM", "0.6"))
STALLING_MOMENTUM_THRESHOLD = float(os.getenv("STALLING_MOMENTUM_THRESHOLD", "0.3"))
REVERSAL_RISK_R_DROP = float(os.getenv("REVERSAL_RISK_R_DROP", "0.5"))
TOXIC_MIN_TIME_MINUTES = int(os.getenv("TOXIC_MIN_TIME_MINUTES", "30"))
TOXIC_MAX_R = float(os.getenv("TOXIC_MAX_R", "-0.5"))

# Scale-in/out thresholds
SCALE_IN_MIN_R = float(os.getenv("SCALE_IN_MIN_R", "0.5"))
SCALE_OUT_PEAK_R_THRESHOLD = float(os.getenv("SCALE_OUT_PEAK_R_THRESHOLD", "2.0"))
SCALE_OUT_MOMENTUM_DROP = float(os.getenv("SCALE_OUT_MOMENTUM_DROP", "0.4"))

# Risk state thresholds
CALM_VOLATILITY_MAX = float(os.getenv("CALM_VOLATILITY_MAX", "1.2"))
STRESSED_VOLATILITY_MAX = float(os.getenv("STRESSED_VOLATILITY_MAX", "2.0"))


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class PositionClassification(Enum):
    """Position quality classifications"""
    STRONG_TREND = "STRONG_TREND"
    SLOW_GRINDER = "SLOW_GRINDER"
    STALLING = "STALLING"
    REVERSAL_RISK = "REVERSAL_RISK"
    TOXIC = "TOXIC"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RiskState(Enum):
    """Position risk state"""
    CALM = "CALM"
    STRESSED = "STRESSED"
    CRITICAL = "CRITICAL"


class RecommendedAction(Enum):
    """Recommended actions for positions"""
    HOLD = "HOLD"
    HOLD_LONGER = "HOLD_LONGER"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"
    PARTIAL_TP = "PARTIAL_TP"
    TIGHTEN_SL = "TIGHTEN_SL"
    LOOSEN_TP = "LOOSEN_TP"
    ENABLE_TRAILING = "ENABLE_TRAILING"
    EXIT_SOON = "EXIT_SOON"
    EXIT_IMMEDIATELY = "EXIT_IMMEDIATELY"


@dataclass
class PositionMetrics:
    """Complete metrics for a single position"""
    symbol: str
    side: str
    size: float
    entry_price: float
    entry_time: str
    current_price: float = 0.0
    
    # Performance metrics
    unrealized_pnl: float = 0.0
    current_R: float = 0.0
    peak_R: float = 0.0
    trough_R: float = 0.0
    R_range: float = 0.0
    
    # Time metrics
    time_in_trade_minutes: int = 0
    time_in_trade_hours: float = 0.0
    
    # Momentum & volatility
    momentum_score: float = 0.0
    volatility_change_factor: float = 1.0
    trend_strength: float = 0.0
    
    # Risk assessment
    risk_state: RiskState = RiskState.CALM
    expected_R: float = 0.0
    realized_vs_expected: float = 0.0
    
    # Market context
    regime_tag: str = "UNKNOWN"
    vol_level: str = "UNKNOWN"
    spread: float = 0.0
    slippage_estimate: float = 0.0
    
    # Symbol classification
    symbol_category: str = "UNKNOWN"
    symbol_stability_score: float = 0.0
    
    # Stop-loss & take-profit
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop_active: bool = False
    
    # Performance expectations
    expected_hold_time_minutes: int = 0
    expected_max_R: float = 0.0


@dataclass
class PositionIntelligence:
    """Complete intelligence assessment for a position"""
    symbol: str
    classification: PositionClassification
    
    # Current state
    current_R: float
    peak_R: float
    time_in_trade_hours: float
    momentum_score: float
    risk_state: RiskState
    
    # Recommendations
    recommended_action: RecommendedAction
    action_rationale: str
    
    # Scale suggestions
    scale_suggestion: str  # "SCALE_IN" | "SCALE_OUT" | "NONE"
    scale_rationale: str
    
    # Exit suggestions
    exit_suggestion: str  # "HOLD" | "PARTIAL_EXIT" | "FULL_EXIT" | "EMERGENCY_EXIT"
    exit_rationale: str
    
    # Fields with defaults
    suggested_size_delta: float = 0.0
    suggested_exit_percentage: float = 0.0
    
    # Tactical adjustments
    suggested_sl_adjustment: Optional[float] = None
    suggested_tp_adjustment: Optional[float] = None
    enable_trailing: bool = False
    trailing_distance_atr: float = 0.0
    
    # Priority & urgency
    priority_score: float = 0.0  # 0-10
    urgency: str = "NORMAL"  # NORMAL | ELEVATED | URGENT | CRITICAL
    
    # Integration hints for AELM
    exit_mode_override: Optional[str] = None
    risk_override: Optional[str] = None


@dataclass
class PILSummary:
    """Overall summary of position intelligence"""
    timestamp: str
    total_positions: int
    
    # Classification breakdown
    strong_trend_count: int = 0
    slow_grinder_count: int = 0
    stalling_count: int = 0
    reversal_risk_count: int = 0
    toxic_count: int = 0
    insufficient_data_count: int = 0
    
    # Risk state breakdown
    calm_count: int = 0
    stressed_count: int = 0
    critical_count: int = 0
    
    # Aggregate metrics
    total_unrealized_pnl: float = 0.0
    total_current_R: float = 0.0
    avg_time_in_trade_hours: float = 0.0
    avg_momentum_score: float = 0.0
    
    # Recommendations
    positions_needing_attention: List[str] = field(default_factory=list)
    positions_to_scale_in: List[str] = field(default_factory=list)
    positions_to_scale_out: List[str] = field(default_factory=list)
    positions_to_exit: List[str] = field(default_factory=list)
    
    # Focus areas
    focus_risk_reduction: List[str] = field(default_factory=list)
    focus_profit_maximization: List[str] = field(default_factory=list)


# ============================================================================
# MAIN POSITION INTELLIGENCE LAYER
# ============================================================================

class PositionIntelligenceLayer:
    """
    POSITION INTELLIGENCE LAYER (PIL)
    
    Autonomous per-position lifecycle management system.
    Maximizes per-position outcomes through dynamic management.
    """
    
    def __init__(self):
        print("\n" + "="*80)
        print("POSITION INTELLIGENCE LAYER (PIL) — INITIALIZING")
        print("="*80 + "\n")
        
        # Data paths
        self.data_dir = Path("/app/data")
        self.positions_path = self.data_dir / "open_positions.json"
        self.trades_dir = self.data_dir / "trades"
        self.policy_obs_dir = self.data_dir / "policy_observations"
        self.universe_control_path = self.data_dir / "universe_control_snapshot.json"
        self.orchestrator_state_path = self.data_dir / "orchestrator_state.json"
        
        # Output paths
        self.pil_output_path = self.data_dir / "position_intelligence.json"
        self.pil_summary_path = self.data_dir / "position_intelligence_summary.json"
        self.pil_recommendations_path = self.data_dir / "position_recommendations.json"
        
        # Data storage
        self.open_positions: List[Dict] = []
        self.position_metrics: Dict[str, PositionMetrics] = {}
        self.position_intelligence: Dict[str, PositionIntelligence] = {}
        self.trade_history: Dict[str, List[Dict]] = defaultdict(list)
        self.signal_history: Dict[str, List[Dict]] = defaultdict(list)
        self.universe_data: Dict = {}
        self.orchestrator_state: Dict = {}
        
        # Tracking
        self.peak_R_tracker: Dict[str, float] = {}
        self.trough_R_tracker: Dict[str, float] = {}
        
        print(f"✓ PIL initialized")
        print(f"  Mode: {PIL_MODE}")
        print(f"  Update Interval: {PIL_UPDATE_INTERVAL_SECONDS}s")
        print()
    
    # ========================================================================
    # PHASE 1: DATA INGESTION
    # ========================================================================
    
    def load_all_data(self) -> bool:
        """Load all required data"""
        print("PHASE 1: DATA INGESTION")
        print("="*80 + "\n")
        
        success = True
        
        # Load open positions
        if self.positions_path.exists():
            with open(self.positions_path) as f:
                self.open_positions = json.load(f)
            print(f"✓ Open positions loaded: {len(self.open_positions)} positions")
        else:
            print(f"✓ No open positions found (system may be starting)")
            self.open_positions = []
        
        # Load trade history
        if self.trades_dir.exists():
            trade_files = list(self.trades_dir.glob("*.jsonl"))
            total_trades = 0
            for trade_file in trade_files:
                with open(trade_file) as f:
                    for line in f:
                        if line.strip():
                            try:
                                trade = json.loads(line)
                                symbol = trade.get("symbol")
                                if symbol:
                                    self.trade_history[symbol].append(trade)
                                    total_trades += 1
                            except json.JSONDecodeError:
                                continue
            print(f"✓ Trade history loaded: {total_trades} trades")
        else:
            print(f"ℹ No trade history available")
        
        # Load signal history
        if self.policy_obs_dir.exists():
            signal_files = list(self.policy_obs_dir.glob("*.jsonl"))
            total_signals = 0
            for signal_file in signal_files:
                with open(signal_file) as f:
                    for line in f:
                        if line.strip():
                            try:
                                signal = json.loads(line)
                                symbol = signal.get("symbol")
                                if symbol:
                                    self.signal_history[symbol].append(signal)
                                    total_signals += 1
                            except json.JSONDecodeError:
                                continue
            print(f"✓ Signal history loaded: {total_signals} signals")
        else:
            print(f"ℹ No signal history available")
        
        # Load universe data
        if self.universe_control_path.exists():
            with open(self.universe_control_path) as f:
                self.universe_data = json.load(f)
            print(f"✓ Universe data loaded")
        else:
            print(f"ℹ Universe data not available")
        
        # Load orchestrator state
        if self.orchestrator_state_path.exists():
            with open(self.orchestrator_state_path) as f:
                self.orchestrator_state = json.load(f)
            print(f"✓ Orchestrator state loaded")
        else:
            print(f"ℹ Orchestrator state not available")
        
        print()
        return success
    
    # ========================================================================
    # PHASE 2: POSITION METRICS COMPUTATION
    # ========================================================================
    
    def compute_position_metrics(self):
        """Compute comprehensive metrics for all positions"""
        print("PHASE 2: POSITION METRICS COMPUTATION")
        print("="*80 + "\n")
        
        if not self.open_positions:
            print("ℹ No open positions to analyze")
            print()
            return
        
        print(f"Computing metrics for {len(self.open_positions)} positions...")
        
        for position in self.open_positions:
            metrics = self._compute_single_position_metrics(position)
            self.position_metrics[metrics.symbol] = metrics
        
        print(f"✓ Metrics computed for {len(self.position_metrics)} positions")
        print()
    
    def _compute_single_position_metrics(self, position: Dict) -> PositionMetrics:
        """Compute metrics for a single position"""
        symbol = position.get("symbol", "")
        
        metrics = PositionMetrics(
            symbol=symbol,
            side=position.get("side", ""),
            size=position.get("size", 0.0),
            entry_price=position.get("entry_price", 0.0),
            entry_time=position.get("entry_time", ""),
            current_price=position.get("current_price", position.get("entry_price", 0.0)),
        )
        
        # === TIME METRICS ===
        if metrics.entry_time:
            try:
                entry_dt = datetime.fromisoformat(metrics.entry_time.replace('Z', '+00:00'))
                now_dt = datetime.now(timezone.utc)
                time_diff = now_dt - entry_dt
                metrics.time_in_trade_minutes = int(time_diff.total_seconds() / 60)
                metrics.time_in_trade_hours = time_diff.total_seconds() / 3600
            except:
                pass
        
        # === PERFORMANCE METRICS ===
        metrics.unrealized_pnl = position.get("unrealized_pnl", 0.0)
        metrics.current_R = position.get("current_R", 0.0)
        
        # Track peak and trough R
        position_key = f"{symbol}_{metrics.entry_time}"
        if position_key not in self.peak_R_tracker:
            self.peak_R_tracker[position_key] = metrics.current_R
            self.trough_R_tracker[position_key] = metrics.current_R
        else:
            self.peak_R_tracker[position_key] = max(self.peak_R_tracker[position_key], metrics.current_R)
            self.trough_R_tracker[position_key] = min(self.trough_R_tracker[position_key], metrics.current_R)
        
        metrics.peak_R = self.peak_R_tracker[position_key]
        metrics.trough_R = self.trough_R_tracker[position_key]
        metrics.R_range = metrics.peak_R - metrics.trough_R
        
        # === MOMENTUM SCORE ===
        # Momentum: how well is position performing relative to time?
        if metrics.time_in_trade_hours > 0:
            # Positive momentum if R is growing
            if metrics.peak_R > 0:
                metrics.momentum_score = min(1.0, metrics.current_R / (metrics.peak_R + 0.1))
            else:
                metrics.momentum_score = 0.5 if metrics.current_R >= 0 else 0.0
        
        # === VOLATILITY CHANGE FACTOR ===
        # Compare entry volatility to current volatility
        entry_vol = position.get("entry_volatility", 1.0)
        current_vol = position.get("current_volatility", 1.0)
        if entry_vol > 0:
            metrics.volatility_change_factor = current_vol / entry_vol
        
        # === TREND STRENGTH ===
        # From market data or position behavior
        if metrics.current_R > 0 and metrics.time_in_trade_hours > 0:
            metrics.trend_strength = metrics.current_R / metrics.time_in_trade_hours
        
        # === RISK STATE ===
        if metrics.volatility_change_factor <= CALM_VOLATILITY_MAX:
            metrics.risk_state = RiskState.CALM
        elif metrics.volatility_change_factor <= STRESSED_VOLATILITY_MAX:
            metrics.risk_state = RiskState.STRESSED
        else:
            metrics.risk_state = RiskState.CRITICAL
        
        # === MARKET CONTEXT ===
        metrics.regime_tag = self.orchestrator_state.get("regime_tag", "UNKNOWN")
        metrics.vol_level = self.orchestrator_state.get("vol_level", "UNKNOWN")
        metrics.spread = position.get("spread", 0.0)
        metrics.slippage_estimate = position.get("slippage_estimate", 0.0)
        
        # === SYMBOL CLASSIFICATION ===
        if self.universe_data:
            symbol_health = self.universe_data.get("symbol_health", {}).get(symbol, {})
            metrics.symbol_category = symbol_health.get("tier", "UNKNOWN")
            metrics.symbol_stability_score = symbol_health.get("stability_score", 0.0)
        
        # === STOP-LOSS & TAKE-PROFIT ===
        metrics.stop_loss = position.get("stop_loss", 0.0)
        metrics.take_profit = position.get("take_profit", 0.0)
        metrics.trailing_stop_active = position.get("trailing_stop_active", False)
        
        # === PERFORMANCE EXPECTATIONS ===
        # Get expected performance from historical trades
        recent_trades = self.trade_history.get(symbol, [])[-20:]  # Last 20 trades
        if recent_trades:
            Rs = [t.get("R", 0.0) for t in recent_trades]
            hold_times = []
            for t in recent_trades:
                if "entry_timestamp" in t and "exit_timestamp" in t:
                    try:
                        entry = datetime.fromisoformat(t["entry_timestamp"].replace('Z', '+00:00'))
                        exit = datetime.fromisoformat(t["exit_timestamp"].replace('Z', '+00:00'))
                        hold_times.append((exit - entry).total_seconds() / 60)
                    except:
                        pass
            
            if Rs:
                metrics.expected_R = statistics.mean([r for r in Rs if r > 0]) if any(r > 0 for r in Rs) else 0.0
                metrics.expected_max_R = max(Rs) if Rs else 0.0
            
            if hold_times:
                metrics.expected_hold_time_minutes = int(statistics.mean(hold_times))
        
        # === REALIZED VS EXPECTED ===
        if metrics.expected_R > 0:
            metrics.realized_vs_expected = metrics.current_R / metrics.expected_R
        
        return metrics
    
    # ========================================================================
    # PHASE 3: POSITION CLASSIFICATION
    # ========================================================================
    
    def classify_positions(self):
        """Classify all positions based on behavior"""
        print("PHASE 3: POSITION CLASSIFICATION")
        print("="*80 + "\n")
        
        if not self.position_metrics:
            print("ℹ No positions to classify")
            print()
            return
        
        for symbol, metrics in self.position_metrics.items():
            classification = self._classify_single_position(metrics)
            intelligence = self._generate_intelligence(metrics, classification)
            self.position_intelligence[symbol] = intelligence
        
        # Print classification summary
        classifications = defaultdict(int)
        for intel in self.position_intelligence.values():
            classifications[intel.classification.value] += 1
        
        print(f"✓ Positions classified:")
        for cls, count in sorted(classifications.items()):
            print(f"  {cls}: {count}")
        print()
    
    def _classify_single_position(self, metrics: PositionMetrics) -> PositionClassification:
        """Classify a single position"""
        
        # TOXIC: Poor performance after sufficient time
        if (metrics.time_in_trade_minutes >= TOXIC_MIN_TIME_MINUTES and
            metrics.current_R < TOXIC_MAX_R):
            return PositionClassification.TOXIC
        
        # REVERSAL_RISK: Dropped significantly from peak
        if metrics.peak_R > 0.5 and (metrics.peak_R - metrics.current_R) >= REVERSAL_RISK_R_DROP:
            return PositionClassification.REVERSAL_RISK
        
        # STRONG_TREND: High R, strong momentum
        if (metrics.current_R >= STRONG_TREND_MIN_R and
            metrics.momentum_score >= STRONG_TREND_MIN_MOMENTUM):
            return PositionClassification.STRONG_TREND
        
        # STALLING: Low momentum despite time
        if (metrics.time_in_trade_hours >= 1.0 and
            metrics.momentum_score < STALLING_MOMENTUM_THRESHOLD):
            return PositionClassification.STALLING
        
        # SLOW_GRINDER: Positive but slow
        if metrics.current_R > 0 and metrics.momentum_score < STRONG_TREND_MIN_MOMENTUM:
            return PositionClassification.SLOW_GRINDER
        
        # INSUFFICIENT_DATA: Too early to classify
        if metrics.time_in_trade_minutes < 15:
            return PositionClassification.INSUFFICIENT_DATA
        
        return PositionClassification.SLOW_GRINDER
    
    # ========================================================================
    # PHASE 4: INTELLIGENCE GENERATION
    # ========================================================================
    
    def _generate_intelligence(self, metrics: PositionMetrics, 
                               classification: PositionClassification) -> PositionIntelligence:
        """Generate complete intelligence assessment for a position"""
        
        intelligence = PositionIntelligence(
            symbol=metrics.symbol,
            classification=classification,
            current_R=metrics.current_R,
            peak_R=metrics.peak_R,
            time_in_trade_hours=metrics.time_in_trade_hours,
            momentum_score=metrics.momentum_score,
            risk_state=metrics.risk_state,
            recommended_action=RecommendedAction.HOLD,
            action_rationale="",
            scale_suggestion="NONE",
            scale_rationale="",
            exit_suggestion="HOLD",
            exit_rationale="",
        )
        
        # === STRONG_TREND ===
        if classification == PositionClassification.STRONG_TREND:
            intelligence.recommended_action = RecommendedAction.HOLD_LONGER
            intelligence.action_rationale = (
                f"Strong trend with R={metrics.current_R:.2f} and momentum={metrics.momentum_score:.2f}. "
                f"Let winners run."
            )
            
            # Suggest loosening TP
            if metrics.take_profit > 0 and metrics.current_R > 0.5:
                intelligence.suggested_tp_adjustment = metrics.take_profit * 1.5
            
            # Suggest trailing stop
            intelligence.enable_trailing = True
            intelligence.trailing_distance_atr = 2.0
            
            # Scale-in if CORE symbol
            if metrics.symbol_category == "CORE" and metrics.current_R >= SCALE_IN_MIN_R:
                intelligence.scale_suggestion = "SCALE_IN"
                intelligence.scale_rationale = (
                    f"CORE symbol in strong trend with R={metrics.current_R:.2f}. "
                    f"Opportunity to increase exposure."
                )
                intelligence.suggested_size_delta = metrics.size * 0.5  # 50% scale-in
            
            intelligence.exit_suggestion = "HOLD"
            intelligence.exit_rationale = "Position performing well. Hold for larger gains."
            intelligence.urgency = "NORMAL"
        
        # === SLOW_GRINDER ===
        elif classification == PositionClassification.SLOW_GRINDER:
            intelligence.recommended_action = RecommendedAction.HOLD
            intelligence.action_rationale = (
                f"Slow but positive progress (R={metrics.current_R:.2f}). "
                f"Monitor for acceleration or stalling."
            )
            intelligence.exit_suggestion = "HOLD"
            intelligence.exit_rationale = "Position moving slowly but positively. Continue monitoring."
            intelligence.urgency = "NORMAL"
        
        # === STALLING ===
        elif classification == PositionClassification.STALLING:
            intelligence.recommended_action = RecommendedAction.PARTIAL_TP
            intelligence.action_rationale = (
                f"Position stalling after {metrics.time_in_trade_hours:.1f}h with low momentum "
                f"({metrics.momentum_score:.2f}). Consider partial profit-taking."
            )
            
            # Suggest partial exit
            intelligence.exit_suggestion = "PARTIAL_EXIT"
            intelligence.exit_rationale = (
                f"Take partial profits while momentum is weak. Reduce exposure by 50%."
            )
            intelligence.suggested_exit_percentage = 50.0
            
            # Suggest tightening SL
            intelligence.recommended_action = RecommendedAction.TIGHTEN_SL
            if metrics.stop_loss > 0:
                # Move SL closer to entry
                intelligence.suggested_sl_adjustment = metrics.entry_price * 0.99
            
            intelligence.urgency = "ELEVATED"
        
        # === REVERSAL_RISK ===
        elif classification == PositionClassification.REVERSAL_RISK:
            intelligence.recommended_action = RecommendedAction.EXIT_SOON
            intelligence.action_rationale = (
                f"Significant reversal from peak R={metrics.peak_R:.2f} to "
                f"current R={metrics.current_R:.2f}. Exit to protect profits."
            )
            
            # Suggest scale-out or full exit
            if metrics.current_R > 0.5:
                intelligence.scale_suggestion = "SCALE_OUT"
                intelligence.scale_rationale = (
                    f"Take profits before further decline. Reduce position by 75%."
                )
                intelligence.suggested_size_delta = -metrics.size * 0.75
                intelligence.exit_suggestion = "PARTIAL_EXIT"
                intelligence.suggested_exit_percentage = 75.0
            else:
                intelligence.exit_suggestion = "FULL_EXIT"
                intelligence.exit_rationale = (
                    f"Reversal risk high with R dropping from {metrics.peak_R:.2f} to "
                    f"{metrics.current_R:.2f}. Exit completely."
                )
                intelligence.suggested_exit_percentage = 100.0
            
            intelligence.urgency = "URGENT"
        
        # === TOXIC ===
        elif classification == PositionClassification.TOXIC:
            intelligence.recommended_action = RecommendedAction.EXIT_IMMEDIATELY
            intelligence.action_rationale = (
                f"TOXIC position: R={metrics.current_R:.2f} after {metrics.time_in_trade_hours:.1f}h. "
                f"Immediate exit required."
            )
            
            intelligence.exit_suggestion = "EMERGENCY_EXIT"
            intelligence.exit_rationale = (
                f"Position performing very poorly. Cut losses immediately."
            )
            intelligence.suggested_exit_percentage = 100.0
            
            # Override exit mode
            intelligence.exit_mode_override = "AGGRESSIVE"
            intelligence.risk_override = "REDUCED"
            
            intelligence.urgency = "CRITICAL"
        
        # === PRIORITY SCORE ===
        # Higher score = needs more attention
        priority_factors = []
        
        if classification == PositionClassification.TOXIC:
            priority_factors.append(10.0)
        elif classification == PositionClassification.REVERSAL_RISK:
            priority_factors.append(8.0)
        elif classification == PositionClassification.STALLING:
            priority_factors.append(6.0)
        elif classification == PositionClassification.STRONG_TREND:
            priority_factors.append(4.0)  # Needs attention for profit maximization
        else:
            priority_factors.append(2.0)
        
        # Risk state factor
        if metrics.risk_state == RiskState.CRITICAL:
            priority_factors.append(9.0)
        elif metrics.risk_state == RiskState.STRESSED:
            priority_factors.append(5.0)
        
        # Time factor (older positions need more attention)
        if metrics.time_in_trade_hours > 24:
            priority_factors.append(7.0)
        elif metrics.time_in_trade_hours > 12:
            priority_factors.append(4.0)
        
        intelligence.priority_score = max(priority_factors) if priority_factors else 0.0
        
        return intelligence
    
    # ========================================================================
    # PHASE 5: SUMMARY GENERATION
    # ========================================================================
    
    def generate_summary(self) -> PILSummary:
        """Generate overall summary"""
        print("PHASE 5: SUMMARY GENERATION")
        print("="*80 + "\n")
        
        summary = PILSummary(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_positions=len(self.position_intelligence),
        )
        
        # Count classifications
        for intel in self.position_intelligence.values():
            if intel.classification == PositionClassification.STRONG_TREND:
                summary.strong_trend_count += 1
            elif intel.classification == PositionClassification.SLOW_GRINDER:
                summary.slow_grinder_count += 1
            elif intel.classification == PositionClassification.STALLING:
                summary.stalling_count += 1
            elif intel.classification == PositionClassification.REVERSAL_RISK:
                summary.reversal_risk_count += 1
            elif intel.classification == PositionClassification.TOXIC:
                summary.toxic_count += 1
            else:
                summary.insufficient_data_count += 1
        
        # Count risk states
        for metrics in self.position_metrics.values():
            if metrics.risk_state == RiskState.CALM:
                summary.calm_count += 1
            elif metrics.risk_state == RiskState.STRESSED:
                summary.stressed_count += 1
            else:
                summary.critical_count += 1
        
        # Aggregate metrics
        if self.position_metrics:
            summary.total_unrealized_pnl = sum(m.unrealized_pnl for m in self.position_metrics.values())
            summary.total_current_R = sum(m.current_R for m in self.position_metrics.values())
            summary.avg_time_in_trade_hours = statistics.mean([m.time_in_trade_hours for m in self.position_metrics.values()])
            summary.avg_momentum_score = statistics.mean([m.momentum_score for m in self.position_metrics.values()])
        
        # Collect recommendations
        for symbol, intel in self.position_intelligence.items():
            if intel.priority_score >= 7.0:
                summary.positions_needing_attention.append(symbol)
            
            if intel.scale_suggestion == "SCALE_IN":
                summary.positions_to_scale_in.append(symbol)
            elif intel.scale_suggestion == "SCALE_OUT":
                summary.positions_to_scale_out.append(symbol)
            
            if intel.exit_suggestion in ["FULL_EXIT", "EMERGENCY_EXIT"]:
                summary.positions_to_exit.append(symbol)
            
            # Focus areas
            if intel.urgency in ["URGENT", "CRITICAL"]:
                summary.focus_risk_reduction.append(symbol)
            if intel.classification == PositionClassification.STRONG_TREND:
                summary.focus_profit_maximization.append(symbol)
        
        print(f"✓ Summary generated")
        print(f"  Total Positions: {summary.total_positions}")
        print(f"  Needing Attention: {len(summary.positions_needing_attention)}")
        print(f"  Scale-In Opportunities: {len(summary.positions_to_scale_in)}")
        print(f"  Scale-Out Recommendations: {len(summary.positions_to_scale_out)}")
        print(f"  Exit Recommendations: {len(summary.positions_to_exit)}")
        print()
        
        return summary
    
    # ========================================================================
    # PHASE 6: OUTPUT GENERATION
    # ========================================================================
    
    def write_outputs(self, summary: PILSummary):
        """Write all output files"""
        print("PHASE 6: OUTPUT GENERATION")
        print("="*80 + "\n")
        
        # Position intelligence (full details)
        intelligence_output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": PIL_MODE,
            "positions": {
                symbol: asdict(intel)
                for symbol, intel in self.position_intelligence.items()
            }
        }
        
        with open(self.pil_output_path, 'w') as f:
            json.dump(intelligence_output, f, indent=2)
        print(f"✓ Position intelligence: {self.pil_output_path}")
        
        # Summary
        with open(self.pil_summary_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"✓ Summary: {self.pil_summary_path}")
        
        # Recommendations (for AELM integration)
        recommendations = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": PIL_MODE,
            "immediate_actions": [],
            "scale_recommendations": [],
            "exit_recommendations": [],
        }
        
        for symbol, intel in sorted(self.position_intelligence.items(), 
                                    key=lambda x: x[1].priority_score, 
                                    reverse=True):
            
            # Immediate actions (high priority)
            if intel.urgency in ["URGENT", "CRITICAL"]:
                recommendations["immediate_actions"].append({
                    "symbol": symbol,
                    "classification": intel.classification.value,
                    "action": intel.recommended_action.value,
                    "rationale": intel.action_rationale,
                    "urgency": intel.urgency,
                    "priority_score": intel.priority_score,
                })
            
            # Scale recommendations
            if intel.scale_suggestion != "NONE":
                recommendations["scale_recommendations"].append({
                    "symbol": symbol,
                    "suggestion": intel.scale_suggestion,
                    "rationale": intel.scale_rationale,
                    "size_delta": intel.suggested_size_delta,
                })
            
            # Exit recommendations
            if intel.exit_suggestion != "HOLD":
                recommendations["exit_recommendations"].append({
                    "symbol": symbol,
                    "suggestion": intel.exit_suggestion,
                    "rationale": intel.exit_rationale,
                    "exit_percentage": intel.suggested_exit_percentage,
                    "exit_mode_override": intel.exit_mode_override,
                })
        
        with open(self.pil_recommendations_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"✓ Recommendations: {self.pil_recommendations_path}")
        
        print()
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def run(self):
        """Execute full PIL workflow"""
        print("\n" + "="*80)
        print("POSITION INTELLIGENCE LAYER (PIL) — STARTING")
        print("="*80 + "\n")
        
        try:
            # Phase 1: Data Ingestion
            self.load_all_data()
            
            # Phase 2: Position Metrics
            self.compute_position_metrics()
            
            # Phase 3: Classification
            self.classify_positions()
            
            # Phase 4: Intelligence (integrated in Phase 3)
            
            # Phase 5: Summary
            summary = self.generate_summary()
            
            # Phase 6: Outputs
            self.write_outputs(summary)
            
            # Executive Summary
            self.print_executive_summary(summary)
            
            return True
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_executive_summary(self, summary: PILSummary):
        """Print executive summary"""
        print("\n" + "="*80)
        print("POSITION INTELLIGENCE LAYER (PIL) — EXECUTIVE SUMMARY")
        print("="*80 + "\n")
        
        print(f"Timestamp: {summary.timestamp}")
        print(f"Mode: {PIL_MODE}")
        print(f"Total Positions: {summary.total_positions}")
        print()
        
        if summary.total_positions == 0:
            print("ℹ No open positions to analyze")
            print()
            return
        
        print("POSITION CLASSIFICATIONS:")
        print("-" * 80)
        print(f"  STRONG_TREND: {summary.strong_trend_count}")
        print(f"  SLOW_GRINDER: {summary.slow_grinder_count}")
        print(f"  STALLING: {summary.stalling_count}")
        print(f"  REVERSAL_RISK: {summary.reversal_risk_count}")
        print(f"  TOXIC: {summary.toxic_count}")
        print(f"  INSUFFICIENT_DATA: {summary.insufficient_data_count}")
        print()
        
        print("RISK STATE DISTRIBUTION:")
        print("-" * 80)
        print(f"  CALM: {summary.calm_count}")
        print(f"  STRESSED: {summary.stressed_count}")
        print(f"  CRITICAL: {summary.critical_count}")
        print()
        
        print("AGGREGATE METRICS:")
        print("-" * 80)
        print(f"  Total Unrealized PnL: ${summary.total_unrealized_pnl:.2f}")
        print(f"  Total Current R: {summary.total_current_R:.2f}")
        print(f"  Avg Time in Trade: {summary.avg_time_in_trade_hours:.1f}h")
        print(f"  Avg Momentum Score: {summary.avg_momentum_score:.2f}")
        print()
        
        print("RECOMMENDATIONS:")
        print("-" * 80)
        print(f"  Positions Needing Attention: {len(summary.positions_needing_attention)}")
        if summary.positions_needing_attention:
            print(f"    {', '.join(summary.positions_needing_attention[:5])}")
        
        print(f"  Scale-In Opportunities: {len(summary.positions_to_scale_in)}")
        if summary.positions_to_scale_in:
            print(f"    {', '.join(summary.positions_to_scale_in[:5])}")
        
        print(f"  Scale-Out Recommendations: {len(summary.positions_to_scale_out)}")
        if summary.positions_to_scale_out:
            print(f"    {', '.join(summary.positions_to_scale_out[:5])}")
        
        print(f"  Exit Recommendations: {len(summary.positions_to_exit)}")
        if summary.positions_to_exit:
            print(f"    {', '.join(summary.positions_to_exit[:5])}")
        print()
        
        print("FOCUS AREAS:")
        print("-" * 80)
        print(f"  Risk Reduction: {len(summary.focus_risk_reduction)} positions")
        print(f"  Profit Maximization: {len(summary.focus_profit_maximization)} positions")
        print()
        
        print("="*80)
        print()


def main():
    """Main entry point"""
    pil = PositionIntelligenceLayer()
    success = pil.run()
    
    if success:
        print("✅ POSITION INTELLIGENCE LAYER (PIL) COMPLETED SUCCESSFULLY")
        return 0
    else:
        print("✗ POSITION INTELLIGENCE LAYER (PIL) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
