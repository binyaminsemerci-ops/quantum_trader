#!/usr/bin/env python3
"""
RISK & UNIVERSE CONTROL CENTER OS
==================================

Autonomous, high-level supervisory AI system responsible for:
- Universe Selection & Optimization
- Universe Health Monitoring & Performance Analysis
- Symbol Classification & Toxicity Detection
- Risk Oversight & Emergency Braking
- Real-time Monitoring & Scheduler Orchestration
- Visual Intelligence & Snapshot/Delta Engines
- Orchestrator + Risk Manager Integration

Mission: MAINTAIN A SAFE, PROFITABLE, STABLE, SELF-OPTIMIZING TRADING UNIVERSE
         WITHOUT HUMAN INTERVENTION.

Author: Senior Quant Developer
Date: November 23, 2025
Version: 3.0 (Control Center OS)
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

UNIVERSE_OS_MODE = os.getenv("UNIVERSE_OS_MODE", "OBSERVE")  # OBSERVE | FULL_AUTONOMY
UNIVERSE_OS_INTERVAL_HOURS = int(os.getenv("UNIVERSE_OS_INTERVAL_HOURS", "4"))
ENABLE_VISUALIZATION = os.getenv("UNIVERSE_OS_VISUALIZE", "false").lower() == "true"

# Emergency brake thresholds
EMERGENCY_SLIPPAGE_THRESHOLD = float(os.getenv("EMERGENCY_SLIPPAGE_THRESHOLD", "0.01"))  # 1%
EMERGENCY_SPREAD_THRESHOLD = float(os.getenv("EMERGENCY_SPREAD_THRESHOLD", "0.005"))  # 0.5%
EMERGENCY_LOSS_STREAK = int(os.getenv("EMERGENCY_LOSS_STREAK", "5"))
EMERGENCY_DD_THRESHOLD = float(os.getenv("EMERGENCY_DD_THRESHOLD", "-0.15"))  # -15%

# Classification thresholds
CORE_MIN_STABILITY = float(os.getenv("CORE_MIN_STABILITY", "0.20"))
CORE_MIN_QUALITY = float(os.getenv("CORE_MIN_QUALITY", "0.25"))
CORE_MIN_WINRATE = float(os.getenv("CORE_MIN_WINRATE", "0.45"))
BLACKLIST_MAX_DISALLOW = float(os.getenv("BLACKLIST_MAX_DISALLOW", "0.35"))


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class SymbolAction(Enum):
    """Recommended symbol-level actions"""
    CONTINUE = "CONTINUE"
    WATCH = "WATCH"
    PAUSE = "PAUSE_SYMBOL"
    REDUCE_RISK = "REDUCE_RISK_SYMBOL"
    BLACKLIST = "BLACKLIST_SYMBOL"


class GlobalAction(Enum):
    """Recommended global-level actions"""
    NORMAL = "NORMAL"
    NO_NEW_TRADES = "NO_NEW_TRADES"
    REDUCE_GLOBAL_RISK = "REDUCE_GLOBAL_RISK"
    DEFENSIVE_EXIT = "DEFENSIVE_EXIT_MODE"
    SAFE_UNIVERSE = "SAFE_UNIVERSE_MODE"
    EMERGENCY_BRAKE = "EMERGENCY_BRAKE"


@dataclass
class SymbolHealth:
    """Complete health profile for a symbol"""
    symbol: str
    
    # Performance
    winrate: float = 0.0
    avg_R: float = 0.0
    total_R: float = 0.0
    median_R: float = 0.0
    variance_R: float = 0.0
    profit_factor: float = 0.0
    trade_count: int = 0
    
    # Cost behavior
    avg_slippage: float = 0.0
    avg_spread: float = 0.0
    max_slippage: float = 0.0
    max_spread: float = 0.0
    slippage_spikes: int = 0
    spread_explosions: int = 0
    
    # Regime profile
    trending_R: float = 0.0
    ranging_R: float = 0.0
    mixed_R: float = 0.0
    regime_dependency: float = 0.0
    
    # Volatility profile
    high_vol_R: float = 0.0
    extreme_vol_R: float = 0.0
    normal_vol_R: float = 0.0
    vol_sensitivity: float = 0.0
    
    # Policy alignment
    disallow_rate: float = 0.0
    avg_confidence: float = 0.0
    confidence_std: float = 0.0
    
    # Composite scores
    stability_score: float = 0.0
    quality_score: float = 0.0
    toxicity_score: float = 0.0
    
    # Classification
    tier: str = "UNCLASSIFIED"
    tier_reason: str = ""
    
    # Health status
    health_status: str = "UNKNOWN"
    recommended_action: SymbolAction = SymbolAction.CONTINUE
    alerts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class UniverseHealth:
    """Overall universe health status"""
    timestamp: str
    universe_size: int
    
    # Aggregate performance
    daily_pnl: float = 0.0
    cumulative_R: float = 0.0
    rolling_winrate: float = 0.0
    rolling_avg_R: float = 0.0
    
    # Aggregate costs
    rolling_slippage: float = 0.0
    rolling_spread: float = 0.0
    
    # Risk metrics
    drawdown: float = 0.0
    max_drawdown: float = 0.0
    trade_frequency: float = 0.0
    
    # Regime health
    regime_tag: str = "UNKNOWN"
    regime_performance: float = 0.0
    
    # Volatility health
    vol_level: str = "UNKNOWN"
    vol_performance: float = 0.0
    
    # Health status
    overall_health: str = "UNKNOWN"
    health_score: float = 0.0
    
    # Alerts
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    recommended_global_action: GlobalAction = GlobalAction.NORMAL


@dataclass
class UniverseProfile:
    """Universe configuration profile"""
    name: str
    size: int
    symbols: List[str]
    composition: str
    expected_R: float
    expected_winrate: float
    expected_slippage: float
    risk_level: str
    description: str


@dataclass
class EmergencyBrake:
    """Emergency brake recommendation"""
    triggered: bool
    severity: AlertSeverity
    reason: str
    recommended_action: GlobalAction
    affected_symbols: List[str]
    duration_hours: int
    timestamp: str


# ============================================================================
# MAIN CONTROL CENTER OS
# ============================================================================

class RiskUniverseControlCenter:
    """
    RISK & UNIVERSE CONTROL CENTER OS
    
    Autonomous supervisory AI system for complete universe and risk governance.
    """
    
    def __init__(self):
        print("\n" + "="*80)
        print("RISK & UNIVERSE CONTROL CENTER OS — INITIALIZING")
        print("="*80 + "\n")
        
        # Data paths
        self.data_dir = Path("/app/data")
        self.universe_snapshot_path = self.data_dir / "universe_snapshot.json"
        self.selector_output_path = self.data_dir / "universe_selector_output.json"
        self.trades_dir = self.data_dir / "trades"
        self.policy_obs_dir = self.data_dir / "policy_observations"
        self.charts_dir = self.data_dir / "universe_charts"
        
        # Output paths
        self.health_output_path = self.data_dir / "universe_health_report.json"
        self.snapshot_output_path = self.data_dir / "universe_control_snapshot.json"
        self.delta_output_path = self.data_dir / "universe_delta.json"
        self.emergency_output_path = self.data_dir / "emergency_brake_status.json"
        
        # Create output directories
        self.charts_dir.mkdir(exist_ok=True, parents=True)
        
        # Data storage
        self.universe_snapshot = {}
        self.selector_output = {}
        self.trade_data = defaultdict(list)
        self.signal_data = defaultdict(list)
        self.policy_obs_data = []
        
        # Health tracking
        self.symbol_health: Dict[str, SymbolHealth] = {}
        self.universe_health: Optional[UniverseHealth] = None
        
        # Classifications
        self.core_symbols = []
        self.expansion_symbols = []
        self.conditional_symbols = []
        self.blacklist_symbols = []
        self.watch_list_symbols = []
        
        # Universe profiles
        self.safe_profile: Optional[UniverseProfile] = None
        self.aggressive_profile: Optional[UniverseProfile] = None
        self.experimental_profile: Optional[UniverseProfile] = None
        
        # Emergency tracking
        self.emergency_brake: Optional[EmergencyBrake] = None
        self.emergency_history = []
        
        # Performance curves
        self.performance_curves = {}
        
        # Deltas
        self.deltas = {}
        
        print(f"✓ Control Center OS initialized")
        print(f"  Mode: {UNIVERSE_OS_MODE}")
        print(f"  Interval: {UNIVERSE_OS_INTERVAL_HOURS}h")
        print(f"  Emergency Slippage Threshold: {EMERGENCY_SLIPPAGE_THRESHOLD:.2%}")
        print(f"  Emergency Spread Threshold: {EMERGENCY_SPREAD_THRESHOLD:.2%}")
        print()
    
    # ========================================================================
    # PHASE 1: DATA INGESTION
    # ========================================================================
    
    def load_all_data(self) -> bool:
        """Load all data sources"""
        print("PHASE 1: DATA INGESTION")
        print("="*80 + "\n")
        
        success = True
        
        # Load universe snapshot
        if self.universe_snapshot_path.exists():
            with open(self.universe_snapshot_path) as f:
                self.universe_snapshot = json.load(f)
            print(f"✓ Universe snapshot loaded: {self.universe_snapshot.get('symbol_count')} symbols")
        else:
            print(f"⚠ Universe snapshot not found")
            success = False
        
        # Load selector output (optional)
        if self.selector_output_path.exists():
            with open(self.selector_output_path) as f:
                self.selector_output = json.load(f)
            print(f"✓ Selector output loaded")
        else:
            print(f"ℹ Selector output not found (optional)")
        
        # Load trade data
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
                                    self.trade_data[symbol].append(trade)
                                    total_trades += 1
                            except json.JSONDecodeError:
                                continue
            print(f"✓ Trade data loaded: {total_trades} trades from {len(trade_files)} files")
        else:
            print(f"⚠ No trade data available")
        
        # Load signal data
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
                                    self.signal_data[symbol].append(signal)
                                    total_signals += 1
                            except json.JSONDecodeError:
                                continue
            print(f"✓ Signal data loaded: {total_signals} signals from {len(signal_files)} files")
        else:
            print(f"⚠ No signal data available")
        
        print()
        return success
    
    # ========================================================================
    # PHASE 2: SYMBOL HEALTH ENGINE
    # ========================================================================
    
    def compute_symbol_health(self):
        """Compute complete health profile for all symbols"""
        print("PHASE 2: SYMBOL HEALTH ENGINE")
        print("="*80 + "\n")
        
        all_symbols = set(self.universe_snapshot.get("symbols", []))
        all_symbols.update(self.trade_data.keys())
        all_symbols.update(self.signal_data.keys())
        
        print(f"Computing health for {len(all_symbols)} symbols...")
        
        for symbol in all_symbols:
            health = self._compute_single_symbol_health(symbol)
            self.symbol_health[symbol] = health
        
        print(f"✓ Health computed for {len(self.symbol_health)} symbols")
        print()
    
    def _compute_single_symbol_health(self, symbol: str) -> SymbolHealth:
        """Compute health profile for a single symbol"""
        health = SymbolHealth(symbol=symbol)
        
        trades = self.trade_data.get(symbol, [])
        signals = self.signal_data.get(symbol, [])
        
        health.trade_count = len(trades)
        
        # === PERFORMANCE METRICS ===
        if trades:
            R_values = [t.get("R", 0.0) for t in trades]
            wins = [r for r in R_values if r > 0]
            losses = [r for r in R_values if r <= 0]
            
            health.winrate = len(wins) / len(R_values) if R_values else 0.0
            health.avg_R = statistics.mean(R_values) if R_values else 0.0
            health.total_R = sum(R_values)
            health.median_R = statistics.median(R_values) if R_values else 0.0
            health.variance_R = statistics.variance(R_values) if len(R_values) > 1 else 0.0
            
            gross_win = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            health.profit_factor = gross_win / gross_loss if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0)
        
        # === COST BEHAVIOR ===
        if trades:
            slippages = [t.get("slippage", 0.0) for t in trades if "slippage" in t]
            spreads = [t.get("spread", 0.0) for t in trades if "spread" in t]
            
            if slippages:
                health.avg_slippage = statistics.mean(slippages)
                health.max_slippage = max(slippages)
                health.slippage_spikes = sum(1 for s in slippages if s > EMERGENCY_SLIPPAGE_THRESHOLD)
            
            if spreads:
                health.avg_spread = statistics.mean(spreads)
                health.max_spread = max(spreads)
                health.spread_explosions = sum(1 for s in spreads if s > EMERGENCY_SPREAD_THRESHOLD)
        
        # === REGIME PROFILE ===
        if trades:
            trending = [t for t in trades if t.get("regime_tag", "").startswith("TRENDING")]
            ranging = [t for t in trades if t.get("regime_tag", "").startswith("RANGING")]
            mixed = [t for t in trades if t.get("regime_tag", "").startswith("MIXED")]
            
            if trending:
                health.trending_R = statistics.mean([t.get("R", 0.0) for t in trending])
            if ranging:
                health.ranging_R = statistics.mean([t.get("R", 0.0) for t in ranging])
            if mixed:
                health.mixed_R = statistics.mean([t.get("R", 0.0) for t in mixed])
            
            regime_Rs = [r for r in [health.trending_R, health.ranging_R, health.mixed_R] if r != 0.0]
            if len(regime_Rs) > 1:
                health.regime_dependency = statistics.stdev(regime_Rs)
        
        # === VOLATILITY PROFILE ===
        if trades:
            high_vol = [t for t in trades if t.get("vol_level") in ["HIGH", "EXTREME"]]
            extreme_vol = [t for t in trades if t.get("vol_level") == "EXTREME"]
            normal_vol = [t for t in trades if t.get("vol_level") in ["NORMAL", "LOW"]]
            
            if high_vol:
                health.high_vol_R = statistics.mean([t.get("R", 0.0) for t in high_vol])
            if extreme_vol:
                health.extreme_vol_R = statistics.mean([t.get("R", 0.0) for t in extreme_vol])
            if normal_vol:
                health.normal_vol_R = statistics.mean([t.get("R", 0.0) for t in normal_vol])
            
            vol_Rs = [r for r in [health.high_vol_R, health.extreme_vol_R, health.normal_vol_R] if r != 0.0]
            if len(vol_Rs) > 1:
                health.vol_sensitivity = statistics.stdev(vol_Rs)
        
        # === POLICY ALIGNMENT ===
        if signals:
            allow_signals = [s for s in signals if s.get("decision") == "TRADE_ALLOWED"]
            disallow_signals = [s for s in signals if s.get("decision") == "TRADE_DISALLOWED"]
            
            health.disallow_rate = len(disallow_signals) / len(signals) if signals else 0.0
            
            confidences = [s.get("confidence", 0.0) for s in signals if "confidence" in s]
            if confidences:
                health.avg_confidence = statistics.mean(confidences)
                health.confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        
        # === COMPOSITE SCORES ===
        # Stability score: (profitability * consistency) / (costs + variance)
        if health.avg_R > 0:
            cost_penalty = health.avg_spread + health.avg_slippage + 0.01
            variance_penalty = health.variance_R + 0.01
            health.stability_score = (health.avg_R * health.winrate) / (cost_penalty + variance_penalty)
        
        # Quality score
        profitability = max(0, min(1, (health.avg_R + 1) / 3))
        reliability = 1 - health.disallow_rate
        stability = min(1, health.stability_score)
        health.quality_score = 0.4 * profitability + 0.3 * reliability + 0.3 * stability
        
        # Toxicity score (higher = more toxic)
        toxicity_components = []
        if health.slippage_spikes > 0:
            toxicity_components.append(health.slippage_spikes / max(1, health.trade_count))
        if health.spread_explosions > 0:
            toxicity_components.append(health.spread_explosions / max(1, health.trade_count))
        if health.disallow_rate > 0.5:
            toxicity_components.append(health.disallow_rate)
        if health.avg_R < 0:
            toxicity_components.append(abs(health.avg_R))
        
        health.toxicity_score = statistics.mean(toxicity_components) if toxicity_components else 0.0
        
        # === HEALTH STATUS & ALERTS ===
        health.health_status, health.recommended_action, health.alerts = self._assess_symbol_health(health)
        
        return health
    
    def _assess_symbol_health(self, health: SymbolHealth) -> Tuple[str, SymbolAction, List[Dict]]:
        """Assess symbol health and generate alerts"""
        alerts = []
        
        # Check for emergency conditions
        if health.slippage_spikes >= 3:
            alerts.append({
                "severity": AlertSeverity.CRITICAL.value,
                "message": f"Multiple slippage spikes detected ({health.slippage_spikes})",
                "metric": "slippage",
                "value": health.max_slippage,
            })
        
        if health.spread_explosions >= 3:
            alerts.append({
                "severity": AlertSeverity.CRITICAL.value,
                "message": f"Multiple spread explosions detected ({health.spread_explosions})",
                "metric": "spread",
                "value": health.max_spread,
            })
        
        if health.avg_R < -0.5 and health.trade_count >= 5:
            alerts.append({
                "severity": AlertSeverity.WARNING.value,
                "message": f"Consistently negative performance: avg_R={health.avg_R:.3f}",
                "metric": "avg_R",
                "value": health.avg_R,
            })
        
        if health.disallow_rate > 0.5:
            alerts.append({
                "severity": AlertSeverity.WARNING.value,
                "message": f"High disallow rate: {health.disallow_rate:.1%}",
                "metric": "disallow_rate",
                "value": health.disallow_rate,
            })
        
        if health.toxicity_score > 0.5:
            alerts.append({
                "severity": AlertSeverity.CRITICAL.value,
                "message": f"High toxicity score: {health.toxicity_score:.3f}",
                "metric": "toxicity",
                "value": health.toxicity_score,
            })
        
        # Determine recommended action
        if health.toxicity_score > 0.7 or health.slippage_spikes >= 5:
            return "TOXIC", SymbolAction.BLACKLIST, alerts
        elif health.slippage_spikes >= 3 or health.spread_explosions >= 3:
            return "CRITICAL", SymbolAction.PAUSE, alerts
        elif health.avg_R < 0 and health.trade_count >= 5:
            return "UNHEALTHY", SymbolAction.REDUCE_RISK, alerts
        elif health.disallow_rate > 0.5:
            return "RISKY", SymbolAction.WATCH, alerts
        elif health.quality_score > 0.25 and health.stability_score > 0.20:
            return "HEALTHY", SymbolAction.CONTINUE, alerts
        elif health.trade_count < 3:
            return "INSUFFICIENT_DATA", SymbolAction.WATCH, alerts
        else:
            return "MODERATE", SymbolAction.CONTINUE, alerts
    
    # ========================================================================
    # PHASE 3: UNIVERSE HEALTH MONITORING
    # ========================================================================
    
    def compute_universe_health(self):
        """Compute overall universe health"""
        print("PHASE 3: UNIVERSE HEALTH MONITORING")
        print("="*80 + "\n")
        
        universe_symbols = set(self.universe_snapshot.get("symbols", []))
        
        # Aggregate all trades
        all_trades = []
        for symbol in universe_symbols:
            all_trades.extend(self.trade_data.get(symbol, []))
        
        # Sort by timestamp
        all_trades.sort(key=lambda t: t.get("entry_timestamp", ""))
        
        # Compute aggregate metrics
        self.universe_health = UniverseHealth(
            timestamp=datetime.now(timezone.utc).isoformat(),
            universe_size=len(universe_symbols),
        )
        
        if all_trades:
            # Daily PnL (last 24 hours)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_trades = [t for t in all_trades if t.get("exit_timestamp", "") > cutoff.isoformat()]
            if recent_trades:
                self.universe_health.daily_pnl = sum(t.get("pnl", 0.0) for t in recent_trades)
            
            # Cumulative R
            all_Rs = [t.get("R", 0.0) for t in all_trades]
            self.universe_health.cumulative_R = sum(all_Rs)
            
            # Rolling winrate (last 100 trades)
            recent_100 = all_trades[-100:]
            recent_Rs = [t.get("R", 0.0) for t in recent_100]
            wins = [r for r in recent_Rs if r > 0]
            self.universe_health.rolling_winrate = len(wins) / len(recent_Rs) if recent_Rs else 0.0
            self.universe_health.rolling_avg_R = statistics.mean(recent_Rs) if recent_Rs else 0.0
            
            # Rolling costs
            recent_slippages = [t.get("slippage", 0.0) for t in recent_100 if "slippage" in t]
            recent_spreads = [t.get("spread", 0.0) for t in recent_100 if "spread" in t]
            if recent_slippages:
                self.universe_health.rolling_slippage = statistics.mean(recent_slippages)
            if recent_spreads:
                self.universe_health.rolling_spread = statistics.mean(recent_spreads)
            
            # Drawdown (cumulative R curve)
            cumulative_Rs = []
            running_sum = 0
            for r in all_Rs:
                running_sum += r
                cumulative_Rs.append(running_sum)
            
            if cumulative_Rs:
                peak = cumulative_Rs[0]
                max_dd = 0
                current_dd = 0
                for r in cumulative_Rs:
                    if r > peak:
                        peak = r
                    dd = r - peak
                    if dd < max_dd:
                        max_dd = dd
                    current_dd = dd
                
                self.universe_health.drawdown = current_dd
                self.universe_health.max_drawdown = max_dd
            
            # Trade frequency (trades per day)
            if len(all_trades) > 0:
                first_trade = datetime.fromisoformat(all_trades[0].get("entry_timestamp", ""))
                last_trade = datetime.fromisoformat(all_trades[-1].get("entry_timestamp", ""))
                days = max(1, (last_trade - first_trade).days)
                self.universe_health.trade_frequency = len(all_trades) / days
        
        # Health score
        health_components = []
        if self.universe_health.rolling_winrate > 0:
            health_components.append(self.universe_health.rolling_winrate)
        if self.universe_health.rolling_avg_R > 0:
            health_components.append(min(1, self.universe_health.rolling_avg_R / 2))
        if self.universe_health.drawdown > EMERGENCY_DD_THRESHOLD:
            health_components.append(0.5)  # Penalty for drawdown
        
        self.universe_health.health_score = statistics.mean(health_components) if health_components else 0.0
        
        # Overall health status
        if self.universe_health.health_score > 0.6:
            self.universe_health.overall_health = "HEALTHY"
        elif self.universe_health.health_score > 0.4:
            self.universe_health.overall_health = "MODERATE"
        elif self.universe_health.health_score > 0.2:
            self.universe_health.overall_health = "UNHEALTHY"
        else:
            self.universe_health.overall_health = "CRITICAL"
        
        # Generate alerts
        self._generate_universe_alerts()
        
        print(f"✓ Universe health computed")
        print(f"  Overall Health: {self.universe_health.overall_health}")
        print(f"  Health Score: {self.universe_health.health_score:.3f}")
        print(f"  Cumulative R: {self.universe_health.cumulative_R:.2f}")
        print(f"  Rolling Winrate: {self.universe_health.rolling_winrate:.1%}")
        print(f"  Drawdown: {self.universe_health.drawdown:.2f}")
        print()
    
    def _generate_universe_alerts(self):
        """Generate universe-level alerts"""
        alerts = []
        
        if self.universe_health.drawdown < EMERGENCY_DD_THRESHOLD:
            alerts.append({
                "severity": AlertSeverity.CRITICAL.value,
                "message": f"Severe drawdown: {self.universe_health.drawdown:.2f}",
                "metric": "drawdown",
                "value": self.universe_health.drawdown,
            })
            self.universe_health.recommended_global_action = GlobalAction.REDUCE_GLOBAL_RISK
        
        if self.universe_health.rolling_slippage > EMERGENCY_SLIPPAGE_THRESHOLD:
            alerts.append({
                "severity": AlertSeverity.WARNING.value,
                "message": f"High rolling slippage: {self.universe_health.rolling_slippage:.2%}",
                "metric": "slippage",
                "value": self.universe_health.rolling_slippage,
            })
        
        if self.universe_health.rolling_spread > EMERGENCY_SPREAD_THRESHOLD:
            alerts.append({
                "severity": AlertSeverity.WARNING.value,
                "message": f"High rolling spread: {self.universe_health.rolling_spread:.2%}",
                "metric": "spread",
                "value": self.universe_health.rolling_spread,
            })
        
        if self.universe_health.rolling_winrate < 0.35:
            alerts.append({
                "severity": AlertSeverity.WARNING.value,
                "message": f"Low rolling winrate: {self.universe_health.rolling_winrate:.1%}",
                "metric": "winrate",
                "value": self.universe_health.rolling_winrate,
            })
        
        self.universe_health.alerts = alerts
    
    # ========================================================================
    # PHASE 4: SYMBOL CLASSIFICATION
    # ========================================================================
    
    def classify_symbols(self):
        """Classify all symbols based on health"""
        print("PHASE 4: SYMBOL CLASSIFICATION")
        print("="*80 + "\n")
        
        for symbol, health in self.symbol_health.items():
            tier, reason = self._classify_symbol(health)
            health.tier = tier
            health.tier_reason = reason
            
            if tier == "CORE":
                self.core_symbols.append(symbol)
            elif tier == "EXPANSION":
                self.expansion_symbols.append(symbol)
            elif tier == "CONDITIONAL":
                self.conditional_symbols.append(symbol)
            elif tier == "BLACKLIST":
                self.blacklist_symbols.append(symbol)
            
            if health.recommended_action == SymbolAction.WATCH:
                self.watch_list_symbols.append(symbol)
        
        print(f"CORE: {len(self.core_symbols)} symbols")
        print(f"EXPANSION: {len(self.expansion_symbols)} symbols")
        print(f"CONDITIONAL: {len(self.conditional_symbols)} symbols")
        print(f"BLACKLIST: {len(self.blacklist_symbols)} symbols")
        print(f"WATCH LIST: {len(self.watch_list_symbols)} symbols")
        print()
    
    def _classify_symbol(self, health: SymbolHealth) -> Tuple[str, str]:
        """Classify a single symbol"""
        if health.trade_count < 3:
            return "INSUFFICIENT_DATA", f"Only {health.trade_count} trades"
        
        # BLACKLIST (most restrictive)
        if health.toxicity_score > 0.5:
            return "BLACKLIST", f"Toxic: toxicity={health.toxicity_score:.3f}"
        if health.avg_R < -0.3 and health.trade_count >= 5:
            return "BLACKLIST", f"Poor performance: avg_R={health.avg_R:.3f}"
        if health.disallow_rate > BLACKLIST_MAX_DISALLOW:
            return "BLACKLIST", f"High disallow: {health.disallow_rate:.1%}"
        
        # CORE (most desirable)
        if (health.stability_score >= CORE_MIN_STABILITY and
            health.quality_score >= CORE_MIN_QUALITY and
            health.winrate >= CORE_MIN_WINRATE):
            return "CORE", f"Excellent: stability={health.stability_score:.2f}, quality={health.quality_score:.2f}"
        
        # EXPANSION
        if (health.stability_score >= 0.10 and
            health.quality_score >= 0.15 and
            health.winrate >= 0.35):
            return "EXPANSION", f"Good: stability={health.stability_score:.2f}, quality={health.quality_score:.2f}"
        
        # CONDITIONAL
        if health.trending_R > 0.5 or health.normal_vol_R > 0.5:
            return "CONDITIONAL", f"Regime-specific: trending_R={health.trending_R:.2f}"
        
        return "BLACKLIST", f"Does not meet criteria"
    
    # ========================================================================
    # PHASE 5: UNIVERSE OPTIMIZATION
    # ========================================================================
    
    def optimize_universe(self):
        """Generate optimized universe profiles"""
        print("PHASE 5: UNIVERSE OPTIMIZATION")
        print("="*80 + "\n")
        
        # SAFE Profile
        safe_symbols = self.core_symbols.copy()
        expansion_sorted = sorted(
            [s for s in self.expansion_symbols if s in self.symbol_health],
            key=lambda s: self.symbol_health[s].quality_score,
            reverse=True
        )
        safe_target = 180
        if len(safe_symbols) < safe_target:
            safe_symbols.extend(expansion_sorted[:safe_target - len(safe_symbols)])
        
        safe_features = [self.symbol_health[s] for s in safe_symbols if s in self.symbol_health]
        self.safe_profile = UniverseProfile(
            name="SAFE",
            size=len(safe_symbols),
            symbols=safe_symbols,
            composition="CORE + top EXPANSION",
            expected_R=statistics.mean([f.avg_R for f in safe_features]) if safe_features else 0.0,
            expected_winrate=statistics.mean([f.winrate for f in safe_features]) if safe_features else 0.0,
            expected_slippage=statistics.mean([f.avg_slippage for f in safe_features if f.avg_slippage > 0]) if safe_features else 0.0,
            risk_level="LOW",
            description="Production/Mainnet - stable, low cost"
        )
        
        # AGGRESSIVE Profile
        aggressive_symbols = self.core_symbols + self.expansion_symbols + self.conditional_symbols[:20]
        aggressive_features = [self.symbol_health[s] for s in aggressive_symbols if s in self.symbol_health]
        self.aggressive_profile = UniverseProfile(
            name="AGGRESSIVE",
            size=len(aggressive_symbols),
            symbols=aggressive_symbols,
            composition="CORE + EXPANSION + top CONDITIONAL",
            expected_R=statistics.mean([f.avg_R for f in aggressive_features]) if aggressive_features else 0.0,
            expected_winrate=statistics.mean([f.winrate for f in aggressive_features]) if aggressive_features else 0.0,
            expected_slippage=statistics.mean([f.avg_slippage for f in aggressive_features if f.avg_slippage > 0]) if aggressive_features else 0.0,
            risk_level="MEDIUM",
            description="Testnet/Training - broader coverage"
        )
        
        # EXPERIMENTAL Profile
        experimental_symbols = list(set(self.symbol_health.keys()) - set(self.blacklist_symbols))
        experimental_features = [self.symbol_health[s] for s in experimental_symbols if s in self.symbol_health and self.symbol_health[s].trade_count > 0]
        self.experimental_profile = UniverseProfile(
            name="EXPERIMENTAL",
            size=len(experimental_symbols),
            symbols=experimental_symbols,
            composition="All except BLACKLIST",
            expected_R=statistics.mean([f.avg_R for f in experimental_features]) if experimental_features else 0.0,
            expected_winrate=statistics.mean([f.winrate for f in experimental_features]) if experimental_features else 0.0,
            expected_slippage=statistics.mean([f.avg_slippage for f in experimental_features if f.avg_slippage > 0]) if experimental_features else 0.0,
            risk_level="HIGH",
            description="Research - maximum breadth"
        )
        
        print(f"✓ Universe profiles generated:")
        print(f"  SAFE: {self.safe_profile.size} symbols")
        print(f"  AGGRESSIVE: {self.aggressive_profile.size} symbols")
        print(f"  EXPERIMENTAL: {self.experimental_profile.size} symbols")
        print()
    
    # ========================================================================
    # PHASE 6: EMERGENCY BRAKE
    # ========================================================================
    
    def evaluate_emergency_brake(self):
        """Evaluate if emergency brake should be triggered"""
        print("PHASE 6: EMERGENCY BRAKE EVALUATION")
        print("="*80 + "\n")
        
        triggered = False
        severity = AlertSeverity.INFO
        reason = ""
        action = GlobalAction.NORMAL
        affected = []
        duration = 0
        
        # Check for toxic symbols
        toxic_symbols = [s for s, h in self.symbol_health.items() if h.toxicity_score > 0.7]
        if len(toxic_symbols) > 5:
            triggered = True
            severity = AlertSeverity.CRITICAL
            reason = f"Multiple toxic symbols detected: {len(toxic_symbols)}"
            action = GlobalAction.SAFE_UNIVERSE
            affected = toxic_symbols
            duration = 24
        
        # Check for severe drawdown
        if self.universe_health and self.universe_health.drawdown < EMERGENCY_DD_THRESHOLD:
            triggered = True
            severity = AlertSeverity.EMERGENCY
            reason = f"Severe drawdown: {self.universe_health.drawdown:.2f}"
            action = GlobalAction.DEFENSIVE_EXIT
            duration = 48
        
        # Check for cost explosions
        high_cost_symbols = [
            s for s, h in self.symbol_health.items()
            if h.slippage_spikes >= 3 or h.spread_explosions >= 3
        ]
        if len(high_cost_symbols) > 10:
            triggered = True
            severity = AlertSeverity.CRITICAL
            reason = f"Cost explosions across {len(high_cost_symbols)} symbols"
            action = GlobalAction.REDUCE_GLOBAL_RISK
            affected = high_cost_symbols
            duration = 12
        
        self.emergency_brake = EmergencyBrake(
            triggered=triggered,
            severity=severity,
            reason=reason,
            recommended_action=action,
            affected_symbols=affected,
            duration_hours=duration,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        if triggered:
            print(f"⚠ EMERGENCY BRAKE TRIGGERED")
            print(f"  Severity: {severity.value}")
            print(f"  Reason: {reason}")
            print(f"  Action: {action.value}")
            print(f"  Duration: {duration}h")
        else:
            print(f"✓ No emergency conditions detected")
        
        print()
    
    # ========================================================================
    # PHASE 7: SNAPSHOTS & DELTAS
    # ========================================================================
    
    def write_snapshots(self):
        """Write all output snapshots"""
        print("PHASE 7: WRITING SNAPSHOTS & DELTAS")
        print("="*80 + "\n")
        
        # Custom JSON encoder for Enums
        class EnumEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Enum):
                    return obj.value
                return super().default(obj)
        
        # Universe health report
        health_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "universe_health": asdict(self.universe_health) if self.universe_health else None,
            "emergency_brake": asdict(self.emergency_brake) if self.emergency_brake else None,
        }
        
        with open(self.health_output_path, 'w') as f:
            json.dump(health_report, f, indent=2, cls=EnumEncoder)
        print(f"✓ Health report: {self.health_output_path}")
        
        # Control snapshot
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": UNIVERSE_OS_MODE,
            "classifications": {
                "CORE": {"count": len(self.core_symbols), "symbols": self.core_symbols},
                "EXPANSION": {"count": len(self.expansion_symbols), "symbols": self.expansion_symbols},
                "CONDITIONAL": {"count": len(self.conditional_symbols), "symbols": self.conditional_symbols},
                "BLACKLIST": {"count": len(self.blacklist_symbols), "symbols": self.blacklist_symbols},
            },
            "profiles": {
                "SAFE": asdict(self.safe_profile) if self.safe_profile else None,
                "AGGRESSIVE": asdict(self.aggressive_profile) if self.aggressive_profile else None,
                "EXPERIMENTAL": asdict(self.experimental_profile) if self.experimental_profile else None,
            },
            "symbol_health": {s: asdict(h) for s, h in self.symbol_health.items()},
        }
        
        with open(self.snapshot_output_path, 'w') as f:
            json.dump(snapshot, f, indent=2, cls=EnumEncoder)
        print(f"✓ Control snapshot: {self.snapshot_output_path}")
        
        # Delta report
        current_symbols = set(self.universe_snapshot.get("symbols", []))
        recommended_symbols = set(self.safe_profile.symbols if self.safe_profile else [])
        
        delta = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "current_size": len(current_symbols),
            "recommended_size": len(recommended_symbols),
            "to_add": list(recommended_symbols - current_symbols),
            "to_remove": list(current_symbols - recommended_symbols),
            "to_keep": list(current_symbols & recommended_symbols),
            "net_change": len(recommended_symbols) - len(current_symbols),
        }
        
        with open(self.delta_output_path, 'w') as f:
            json.dump(delta, f, indent=2, cls=EnumEncoder)
        print(f"✓ Delta report: {self.delta_output_path}")
        
        # Emergency status
        with open(self.emergency_output_path, 'w') as f:
            json.dump(asdict(self.emergency_brake) if self.emergency_brake else {}, f, indent=2, cls=EnumEncoder)
        print(f"✓ Emergency status: {self.emergency_output_path}")
        
        print()
    
    # ========================================================================
    # PHASE 8: ORCHESTRATOR INTEGRATION
    # ========================================================================
    
    def generate_orchestrator_recommendations(self) -> Dict:
        """Generate recommendations for Orchestrator integration"""
        print("PHASE 8: ORCHESTRATOR INTEGRATION")
        print("="*80 + "\n")
        
        recommendations = {
            "allow_new_trades": True,
            "risk_profile": "NORMAL",
            "disallowed_symbols": self.blacklist_symbols,
            "universe_change_required": False,
            "recommended_universe": "current",
            "emergency_override": False,
        }
        
        if self.emergency_brake and self.emergency_brake.triggered:
            if self.emergency_brake.recommended_action == GlobalAction.NO_NEW_TRADES:
                recommendations["allow_new_trades"] = False
                recommendations["emergency_override"] = True
            elif self.emergency_brake.recommended_action == GlobalAction.REDUCE_GLOBAL_RISK:
                recommendations["risk_profile"] = "REDUCED"
            elif self.emergency_brake.recommended_action == GlobalAction.SAFE_UNIVERSE:
                recommendations["universe_change_required"] = True
                recommendations["recommended_universe"] = "SAFE"
            elif self.emergency_brake.recommended_action == GlobalAction.DEFENSIVE_EXIT:
                recommendations["allow_new_trades"] = False
                recommendations["risk_profile"] = "DEFENSIVE"
                recommendations["emergency_override"] = True
        
        print(f"✓ Orchestrator recommendations:")
        print(f"  Allow new trades: {recommendations['allow_new_trades']}")
        print(f"  Risk profile: {recommendations['risk_profile']}")
        print(f"  Disallowed symbols: {len(recommendations['disallowed_symbols'])}")
        print(f"  Universe change: {recommendations['universe_change_required']}")
        print()
        
        return recommendations
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def run(self):
        """Execute full Control Center OS workflow"""
        print("\n" + "="*80)
        print("RISK & UNIVERSE CONTROL CENTER OS — STARTING")
        print("="*80 + "\n")
        
        try:
            # Phase 1: Data Ingestion
            self.load_all_data()
            
            # Phase 2: Symbol Health
            self.compute_symbol_health()
            
            # Phase 3: Universe Health
            self.compute_universe_health()
            
            # Phase 4: Classification
            self.classify_symbols()
            
            # Phase 5: Optimization
            self.optimize_universe()
            
            # Phase 6: Emergency Brake
            self.evaluate_emergency_brake()
            
            # Phase 7: Snapshots
            self.write_snapshots()
            
            # Phase 8: Orchestrator Integration
            orchestrator_recs = self.generate_orchestrator_recommendations()
            
            # Executive Summary
            self.print_executive_summary(orchestrator_recs)
            
            return True
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_executive_summary(self, orchestrator_recs: Dict):
        """Print executive summary"""
        print("\n" + "="*80)
        print("RISK & UNIVERSE CONTROL CENTER OS — EXECUTIVE SUMMARY")
        print("="*80 + "\n")
        
        print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        print(f"Mode: {UNIVERSE_OS_MODE}")
        print()
        
        if self.universe_health:
            print("UNIVERSE HEALTH:")
            print("-" * 80)
            print(f"  Overall Health: {self.universe_health.overall_health}")
            print(f"  Health Score: {self.universe_health.health_score:.3f}")
            print(f"  Universe Size: {self.universe_health.universe_size}")
            print(f"  Cumulative R: {self.universe_health.cumulative_R:.2f}")
            print(f"  Rolling Winrate: {self.universe_health.rolling_winrate:.1%}")
            print(f"  Drawdown: {self.universe_health.drawdown:.2f}")
            print(f"  Global Action: {self.universe_health.recommended_global_action.value}")
            print()
        
        print("SYMBOL CLASSIFICATIONS:")
        print("-" * 80)
        print(f"  CORE: {len(self.core_symbols)}")
        print(f"  EXPANSION: {len(self.expansion_symbols)}")
        print(f"  CONDITIONAL: {len(self.conditional_symbols)}")
        print(f"  BLACKLIST: {len(self.blacklist_symbols)}")
        print(f"  WATCH LIST: {len(self.watch_list_symbols)}")
        print()
        
        if self.emergency_brake:
            print("EMERGENCY BRAKE:")
            print("-" * 80)
            print(f"  Triggered: {self.emergency_brake.triggered}")
            if self.emergency_brake.triggered:
                print(f"  Severity: {self.emergency_brake.severity.value}")
                print(f"  Reason: {self.emergency_brake.reason}")
                print(f"  Action: {self.emergency_brake.recommended_action.value}")
                print(f"  Duration: {self.emergency_brake.duration_hours}h")
            print()
        
        print("ORCHESTRATOR RECOMMENDATIONS:")
        print("-" * 80)
        for key, value in orchestrator_recs.items():
            if key != "disallowed_symbols":
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {len(value)} symbols")
        print()
        
        print("="*80)
        print()


def main():
    """Main entry point"""
    control_center = RiskUniverseControlCenter()
    success = control_center.run()
    
    if success:
        print("✅ RISK & UNIVERSE CONTROL CENTER OS COMPLETED SUCCESSFULLY")
        return 0
    else:
        print("✗ RISK & UNIVERSE CONTROL CENTER OS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
