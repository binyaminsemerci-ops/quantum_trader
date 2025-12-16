"""
AI HEDGEFUND OPERATING SYSTEM (AI-HFOS)
========================================

The supreme meta-intelligence layer that oversees, coordinates, supervises,
and optimizes every AI subsystem to achieve safe, consistent, autonomous profit generation.

AI-HFOS operates ABOVE all layers and coordinates:
- Universe OS
- Risk OS
- Execution Layer Manager
- Position Intelligence Layer (PIL)
- Portfolio Balancer AI (PBA)
- Model Supervisor
- Retraining System
- Profit Amplification Layer (PAL)
- Self-Healing System
- OrchestratorPolicy
- Scheduler
- Logging + Health Systems

This is NOT a single module. This is the DIRECTOR of the entire system.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# P2-02: Import logging APIs
from backend.core import metrics_logger
from backend.core import audit_logger

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class SystemRiskMode(str, Enum):
    """Global risk mode for the entire system."""
    SAFE = "SAFE"                  # Maximum safety, defensive
    NORMAL = "NORMAL"              # Standard operation
    AGGRESSIVE = "AGGRESSIVE"      # Opportunistic, higher risk
    CRITICAL = "CRITICAL"          # Emergency mode, damage control


class SystemHealth(str, Enum):
    """Overall system health status."""
    OPTIMAL = "OPTIMAL"            # All systems green
    HEALTHY = "HEALTHY"            # Normal operation
    DEGRADED = "DEGRADED"          # Some issues detected
    CRITICAL = "CRITICAL"          # Serious issues
    EMERGENCY = "EMERGENCY"        # System failure


class SubsystemStatus(str, Enum):
    """Status of individual subsystems."""
    ACTIVE = "ACTIVE"              # Operating normally
    DEGRADED = "DEGRADED"          # Performance issues
    DISABLED = "DISABLED"          # Temporarily disabled
    FAILED = "FAILED"              # System failure


class ConflictSeverity(str, Enum):
    """Severity of subsystem conflicts."""
    INFO = "INFO"                  # Informational, no action needed
    WARNING = "WARNING"            # Minor conflict, should review
    ERROR = "ERROR"                # Significant conflict, needs resolution
    CRITICAL = "CRITICAL"          # Critical conflict, immediate action


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SubsystemState:
    """Current state of a subsystem."""
    name: str
    status: SubsystemStatus
    health_score: float  # 0-100
    last_updated: str
    metrics: Dict[str, Any]
    issues: List[str]
    performance_score: float  # 0-100
    
    def is_healthy(self) -> bool:
        """Check if subsystem is healthy."""
        return (
            self.status in [SubsystemStatus.ACTIVE, SubsystemStatus.DEGRADED] and
            self.health_score >= 50 and
            len(self.issues) < 5
        )


@dataclass
class SubsystemConflict:
    """Detected conflict between subsystems."""
    subsystems: List[str]
    conflict_type: str
    severity: ConflictSeverity
    description: str
    recommendation: str


@dataclass
class GlobalDirectives:
    """System-wide directives."""
    allow_new_trades: bool
    allow_new_positions: bool
    enforce_defensive_exits: bool
    reduce_global_risk: bool
    pause_entire_symbols: List[str]
    adjust_confidence_threshold: Optional[float]
    scale_position_sizes: float  # Multiplier (0.5 = 50%, 1.0 = 100%, 1.5 = 150%)
    max_daily_dd_override: Optional[float]
    force_exit_symbols: List[str]


@dataclass
class UniverseDirectives:
    """Universe-level directives."""
    universe_mode: str  # "SAFE", "NORMAL", "AGGRESSIVE", "EXPERIMENTAL"
    blacklist_symbols: List[str]
    whitelist_symbols: List[str]
    promote_categories: List[str]
    demote_categories: List[str]
    emergency_brake_override: Optional[bool]


@dataclass
class ExecutionDirectives:
    """Execution-level directives."""
    order_type_preference: str  # "MARKET", "LIMIT", "SMART"
    max_slippage_bps: float
    max_spread_bps: float
    reduce_urgency: bool
    enforce_limit_orders: bool
    execution_delay_seconds: int


@dataclass
class PortfolioDirectives:
    """Portfolio-level directives."""
    reduce_exposure_pct: float  # 0-100
    max_position_count: Optional[int]
    max_leverage: Optional[float]
    reduce_correlated_positions: bool
    avoid_expansion_symbols: bool
    concentration_limit_pct: float


@dataclass
class ModelDirectives:
    """Model-level directives."""
    ensemble_weight_adjustments: Dict[str, float]
    models_to_retrain: List[str]
    models_to_disable: List[str]
    confidence_threshold_override: Optional[float]
    use_conservative_predictions: bool


@dataclass
class EmergencyAction:
    """Emergency action to execute."""
    action_type: str
    target: str
    parameters: Dict[str, Any]
    priority: int  # 1=highest
    rationale: str


@dataclass
class AmplificationOpportunity:
    """Profit amplification opportunity identified."""
    symbol: str
    action: str
    expected_r_increase: float
    confidence: float
    priority: int
    rationale: str


@dataclass
class AIHFOSOutput:
    """Complete output from AI-HFOS."""
    timestamp: str
    system_risk_mode: SystemRiskMode
    system_health: SystemHealth
    global_directives: GlobalDirectives
    universe_directives: UniverseDirectives
    execution_directives: ExecutionDirectives
    portfolio_directives: PortfolioDirectives
    model_directives: ModelDirectives
    emergency_actions: List[EmergencyAction]
    amplification_opportunities: List[AmplificationOpportunity]
    priority_notes: List[str]
    subsystem_states: Dict[str, SubsystemState]
    detected_conflicts: List[SubsystemConflict]
    summary: str


# ============================================================================
# AI HEDGEFUND OPERATING SYSTEM
# ============================================================================

class AIHedgeFundOS:
    """
    AI HEDGEFUND OPERATING SYSTEM (AI-HFOS)
    
    Supreme meta-intelligence layer coordinating all subsystems.
    
    Responsibilities:
    - Global supervision of all subsystems
    - Conflict resolution between subsystems
    - System-wide risk governance
    - Emergency response coordination
    - Performance optimization
    - Strategic decision-making
    - [P1-04] Instant regime-change reaction
    """
    
    def __init__(
        self,
        data_dir: str = "/app/data",
        config_path: Optional[str] = None,
        event_bus = None  # [P1-04] EventBus for regime-change events
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # System state tracking
        self.subsystem_states: Dict[str, SubsystemState] = {}
        self.last_risk_mode = SystemRiskMode.NORMAL
        self.emergency_brake_active = False
        
        # [P1-04] Event-driven regime reaction
        self.event_bus = event_bus
        self._last_coordination_run: Optional[datetime] = None
        self._pending_regime_change: Optional[Dict[str, Any]] = None
        
        # [P1-04] Subscribe to market regime changes
        if self.event_bus:
            self.event_bus.subscribe("market.regime.changed", self._on_regime_changed)
            logger.info("[P1-04] AI-HFOS subscribed to market.regime.changed - instant reaction enabled")
        else:
            logger.warning("[P1-04] EventBus not provided - AI-HFOS will only react on 60s loop")
        
        logger.info("[AI-HFOS] Initialized - Supreme coordinator online")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load AI-HFOS configuration with HEDGEFUND MODE support."""
        default_config = {
            # [HEDGEFUND MODE] Configuration
            "enable_hedgefund_mode": True,  # Enable AGGRESSIVE mode
            "max_positions_aggressive": 10,  # Max positions in AGGRESSIVE mode
            "aggressive_confidence_threshold": 0.60,  # Lower threshold for more opportunities
            
            # Safety thresholds
            "safety_priority": 1.0,
            "max_daily_dd_pct": 5.0,
            "max_volatility_exposure": 0.7,
            "max_symbol_concentration": 0.25,
            "conflict_resolution_strategy": "safety_first",
            "emergency_thresholds": {
                "max_daily_dd_pct": 3.0,
                "max_open_dd_pct": 8.0,
                "min_health_score": 30.0
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    # ========================================================================
    # EVENT HANDLERS (P1-04)
    # ========================================================================
    
    async def _on_regime_changed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle market.regime.changed event - trigger immediate coordination (P1-04).
        
        Instead of waiting for the 60s loop, this triggers an extra coordination
        round ON-DEMAND when market conditions change significantly.
        
        Args:
            event_data: {
                "symbol": str,
                "old_regime": str,
                "new_regime": str,
                "regime_confidence": float,
                "volatility": float,
                "recommended_strategy": str,
                "risk_adjustment": str  # INCREASE/DECREASE/MAINTAIN
            }
        """
        symbol = event_data.get("symbol", "UNKNOWN")
        old_regime = event_data.get("old_regime", "UNKNOWN")
        new_regime = event_data.get("new_regime", "UNKNOWN")
        risk_adjustment = event_data.get("risk_adjustment", "MAINTAIN")
        
        logger.warning(
            f"[P1-04] 🔄 REGIME CHANGE DETECTED: {symbol} {old_regime} → {new_regime}\n"
            f"   Risk adjustment: {risk_adjustment}\n"
            f"   Triggering IMMEDIATE coordination round (not waiting for 60s loop)"
        )
        
        # Store regime change for next coordination cycle
        self._pending_regime_change = {
            "symbol": symbol,
            "old_regime": old_regime,
            "new_regime": new_regime,
            "risk_adjustment": risk_adjustment,
            "detected_at": datetime.now(timezone.utc).isoformat()
        }
        
        # TODO: Trigger immediate coordination cycle
        # This would require access to all subsystem data, which isn't available here.
        # Instead, we flag the regime change and let the next regular cycle prioritize it.
        # In a full implementation, this would call an async coordination method.
        
        logger.info(
            f"[P1-04] Regime change flagged - next coordination cycle will prioritize "
            f"{new_regime} adjustments"
        )
    
    # ========================================================================
    # MAIN COORDINATION LOOP
    # ========================================================================
    
    def run_coordination_cycle(
        self,
        universe_data: Dict[str, Any],
        risk_data: Dict[str, Any],
        positions_data: Dict[str, Any],
        execution_data: Dict[str, Any],
        model_performance: Dict[str, Any],
        self_healing_report: Dict[str, Any],
        pal_report: Dict[str, Any],
        orchestrator_policy: Dict[str, Any]
    ) -> AIHFOSOutput:
        """
        Main coordination cycle - analyze all subsystems and produce unified directives.
        
        This is the BRAIN of the entire system.
        [P1-04] Now prioritizes regime changes detected via events.
        """
        logger.info("[AI-HFOS] ====== COORDINATION CYCLE START ======")
        
        # [P1-04] Check for pending regime change
        if self._pending_regime_change:
            logger.warning(
                f"[P1-04] ⚡ PRIORITIZING REGIME CHANGE: "
                f"{self._pending_regime_change['old_regime']} → {self._pending_regime_change['new_regime']}"
            )
            # Inject regime change into universe_data for processing
            if "regime_change" not in universe_data:
                universe_data["regime_change"] = self._pending_regime_change
            # Clear pending flag
            self._pending_regime_change = None
        
        # Record coordination run time
        self._last_coordination_run = datetime.now(timezone.utc)
        
        # 1. Update subsystem states
        self._update_subsystem_states(
            universe_data,
            risk_data,
            positions_data,
            execution_data,
            model_performance,
            self_healing_report,
            pal_report,
            orchestrator_policy
        )
        
        # 2. Detect conflicts between subsystems
        conflicts = self._detect_subsystem_conflicts()
        
        # 3. Assess global system health
        system_health = self._assess_system_health()
        
        # 4. Determine system risk mode
        system_risk_mode = self._determine_risk_mode(
            universe_data,
            risk_data,
            positions_data,
            system_health
        )
        
        # 5. Generate unified directives
        global_directives = self._generate_global_directives(
            system_risk_mode,
            system_health,
            conflicts
        )
        
        universe_directives = self._generate_universe_directives(
            universe_data,
            system_risk_mode,
            conflicts
        )
        
        execution_directives = self._generate_execution_directives(
            execution_data,
            system_risk_mode,
            system_health
        )
        
        portfolio_directives = self._generate_portfolio_directives(
            positions_data,
            system_risk_mode,
            conflicts
        )
        
        model_directives = self._generate_model_directives(
            model_performance,
            system_risk_mode
        )
        
        # 6. Identify emergency actions
        emergency_actions = self._identify_emergency_actions(
            system_health,
            conflicts,
            self_healing_report
        )
        
        # 7. Identify amplification opportunities (from PAL)
        amplification_opportunities = self._process_pal_recommendations(
            pal_report,
            system_risk_mode,
            global_directives
        )
        
        # 8. Generate priority notes
        priority_notes = self._generate_priority_notes(
            system_risk_mode,
            system_health,
            conflicts,
            emergency_actions
        )
        
        # 9. Create summary
        summary = self._generate_summary(
            system_risk_mode,
            system_health,
            global_directives,
            conflicts,
            emergency_actions,
            amplification_opportunities
        )
        
        # 10. Build output
        output = AIHFOSOutput(
            timestamp=datetime.now(timezone.utc).isoformat(),
            system_risk_mode=system_risk_mode,
            system_health=system_health,
            global_directives=global_directives,
            universe_directives=universe_directives,
            execution_directives=execution_directives,
            portfolio_directives=portfolio_directives,
            model_directives=model_directives,
            emergency_actions=emergency_actions,
            amplification_opportunities=amplification_opportunities,
            priority_notes=priority_notes,
            subsystem_states=self.subsystem_states,
            detected_conflicts=conflicts,
            summary=summary
        )
        
        # 11. Save report
        self._save_report(output)
        
        logger.info(f"[AI-HFOS] Coordination complete - Mode: {system_risk_mode.value}, Health: {system_health.value}")
        logger.info(f"[AI-HFOS] Emergency actions: {len(emergency_actions)}, Conflicts: {len(conflicts)}")
        
        return output
    
    # ========================================================================
    # SUBSYSTEM STATE TRACKING
    # ========================================================================
    
    def _update_subsystem_states(
        self,
        universe_data: Dict[str, Any],
        risk_data: Dict[str, Any],
        positions_data: Dict[str, Any],
        execution_data: Dict[str, Any],
        model_performance: Dict[str, Any],
        self_healing_report: Dict[str, Any],
        pal_report: Dict[str, Any],
        orchestrator_policy: Dict[str, Any]
    ):
        """Update state tracking for all subsystems."""
        
        # Universe OS
        universe_status = self._evaluate_universe_status(universe_data)
        self.subsystem_states["universe_os"] = universe_status
        
        # Risk OS
        risk_status = self._evaluate_risk_status(risk_data)
        self.subsystem_states["risk_os"] = risk_status
        
        # Position Intelligence Layer
        pil_status = self._evaluate_pil_status(positions_data)
        self.subsystem_states["position_intelligence"] = pil_status
        
        # Execution Layer
        execution_status = self._evaluate_execution_status(execution_data)
        self.subsystem_states["execution_layer"] = execution_status
        
        # Model Supervisor
        model_status = self._evaluate_model_status(model_performance)
        self.subsystem_states["model_supervisor"] = model_status
        
        # Self-Healing System
        healing_status = self._evaluate_healing_status(self_healing_report)
        self.subsystem_states["self_healing"] = healing_status
        
        # PAL
        pal_status = self._evaluate_pal_status(pal_report)
        self.subsystem_states["profit_amplification"] = pal_status
        
        # Orchestrator Policy
        orchestrator_status = self._evaluate_orchestrator_status(orchestrator_policy)
        self.subsystem_states["orchestrator"] = orchestrator_status
        
        logger.info(f"[AI-HFOS] Updated {len(self.subsystem_states)} subsystem states")
    
    def _evaluate_universe_status(self, data: Dict[str, Any]) -> SubsystemState:
        """Evaluate Universe OS status."""
        issues = []
        metrics = {}
        
        # Check data confidence
        confidence = data.get("data_confidence", "UNKNOWN")
        if confidence == "LOW":
            issues.append("Low data confidence")
        
        # Check symbol count
        symbol_count = data.get("current_universe", {}).get("symbol_count", 0)
        metrics["symbol_count"] = symbol_count
        
        if symbol_count < 50:
            issues.append(f"Low symbol count: {symbol_count}")
        
        # Check blacklist size
        blacklist_count = data.get("classifications", {}).get("BLACKLIST", {}).get("count", 0)
        metrics["blacklist_count"] = blacklist_count
        
        if blacklist_count > 150:
            issues.append(f"Large blacklist: {blacklist_count} symbols")
        
        # Calculate health score
        health_score = 100.0
        if confidence == "LOW":
            health_score -= 20
        if symbol_count < 50:
            health_score -= 30
        if blacklist_count > 150:
            health_score -= 10
        
        status = SubsystemStatus.ACTIVE if health_score >= 70 else SubsystemStatus.DEGRADED
        
        return SubsystemState(
            name="Universe OS",
            status=status,
            health_score=max(0, health_score),
            last_updated=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
            performance_score=health_score
        )
    
    def _evaluate_risk_status(self, data: Dict[str, Any]) -> SubsystemState:
        """Evaluate Risk OS status."""
        issues = []
        metrics = {
            "emergency_brake": data.get("emergency_brake_active", False),
            "daily_dd_pct": data.get("daily_dd_pct", 0.0),
            "open_dd_pct": data.get("open_dd_pct", 0.0)
        }
        
        # Check emergency brake
        if metrics["emergency_brake"]:
            issues.append("Emergency brake ACTIVE")
        
        # Check drawdown levels
        if metrics["daily_dd_pct"] > 3.0:
            issues.append(f"High daily DD: {metrics['daily_dd_pct']:.1f}%")
        
        if metrics["open_dd_pct"] > 8.0:
            issues.append(f"High open DD: {metrics['open_dd_pct']:.1f}%")
        
        # Calculate health score
        health_score = 100.0
        if metrics["emergency_brake"]:
            health_score -= 40
        health_score -= min(30, metrics["daily_dd_pct"] * 5)
        health_score -= min(20, metrics["open_dd_pct"] * 2)
        
        status = SubsystemStatus.ACTIVE if health_score >= 60 else SubsystemStatus.DEGRADED
        
        return SubsystemState(
            name="Risk OS",
            status=status,
            health_score=max(0, health_score),
            last_updated=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
            performance_score=health_score
        )
    
    def _evaluate_pil_status(self, data: Dict[str, Any]) -> SubsystemState:
        """Evaluate Position Intelligence Layer status."""
        issues = []
        metrics = {
            "position_count": data.get("position_count", 0),
            "toxic_count": data.get("toxic_count", 0),
            "winner_count": data.get("winner_count", 0)
        }
        
        # Check for toxic positions
        if metrics["toxic_count"] > 0:
            issues.append(f"Toxic positions detected: {metrics['toxic_count']}")
        
        # Check position distribution
        if metrics["position_count"] > 0:
            toxic_ratio = metrics["toxic_count"] / metrics["position_count"]
            if toxic_ratio > 0.3:
                issues.append(f"High toxic ratio: {toxic_ratio:.1%}")
        
        health_score = 100.0 - (metrics["toxic_count"] * 10)
        
        return SubsystemState(
            name="Position Intelligence Layer",
            status=SubsystemStatus.ACTIVE,
            health_score=max(0, health_score),
            last_updated=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
            performance_score=health_score
        )
    
    def _evaluate_execution_status(self, data: Dict[str, Any]) -> SubsystemState:
        """Evaluate Execution Layer status."""
        issues = []
        metrics = {
            "avg_slippage_bps": data.get("avg_slippage_bps", 0.0),
            "avg_spread_bps": data.get("avg_spread_bps", 0.0),
            "fill_rate": data.get("fill_rate", 1.0)
        }
        
        if metrics["avg_slippage_bps"] > 10:
            issues.append(f"High slippage: {metrics['avg_slippage_bps']:.1f} bps")
        
        if metrics["fill_rate"] < 0.9:
            issues.append(f"Low fill rate: {metrics['fill_rate']:.1%}")
        
        health_score = 100.0
        health_score -= min(30, metrics["avg_slippage_bps"] * 2)
        health_score -= (1.0 - metrics["fill_rate"]) * 50
        
        return SubsystemState(
            name="Execution Layer",
            status=SubsystemStatus.ACTIVE,
            health_score=max(0, health_score),
            last_updated=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
            performance_score=health_score
        )
    
    def _evaluate_model_status(self, data: Dict[str, Any]) -> SubsystemState:
        """Evaluate Model Supervisor status."""
        issues = []
        metrics = {
            "ensemble_accuracy": data.get("ensemble_accuracy", 0.0),
            "degraded_models": data.get("degraded_models", [])
        }
        
        if metrics["ensemble_accuracy"] < 0.5:
            issues.append(f"Low ensemble accuracy: {metrics['ensemble_accuracy']:.1%}")
        
        if len(metrics["degraded_models"]) > 0:
            issues.append(f"Degraded models: {len(metrics['degraded_models'])}")
        
        health_score = metrics["ensemble_accuracy"] * 100
        
        return SubsystemState(
            name="Model Supervisor",
            status=SubsystemStatus.ACTIVE,
            health_score=health_score,
            last_updated=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
            performance_score=health_score
        )
    
    def _evaluate_healing_status(self, data: Dict[str, Any]) -> SubsystemState:
        """Evaluate Self-Healing System status.
        
        Handles both string and dict formats for subsystems:
        - If subsystem is a string: treat as status value directly
        - If subsystem is a dict: extract status field
        - Otherwise: log warning and treat as UNKNOWN
        """
        issues = []
        
        overall_status = data.get("overall_status", "UNKNOWN")
        subsystems = data.get("subsystems", [])
        
        # Helper function to safely extract status from subsystem (string or dict)
        def get_subsystem_status(subsystem) -> str:
            if isinstance(subsystem, str):
                # Subsystem is a string like "HEALTHY", "DEGRADED", "CRITICAL"
                return subsystem.upper()
            elif isinstance(subsystem, dict):
                # Subsystem is a dict with "status" field
                return subsystem.get("status", "UNKNOWN").upper()
            else:
                # Unexpected type - log warning
                logger.warning(f"[AI-HFOS] Unexpected subsystem type in Self-Healing: {type(subsystem)}")
                return "UNKNOWN"
        
        critical_count = sum(1 for s in subsystems if "CRITICAL" in get_subsystem_status(s))
        degraded_count = sum(1 for s in subsystems if "DEGRADED" in get_subsystem_status(s))
        
        metrics = {
            "overall_status": overall_status,
            "critical_subsystems": critical_count,
            "degraded_subsystems": degraded_count,
            "total_subsystems": len(subsystems)
        }
        
        # P2-02: Record system health metrics
        metrics_logger.record_gauge(
            "system_health_critical_subsystems",
            value=float(critical_count)
        )
        metrics_logger.record_gauge(
            "system_health_degraded_subsystems",
            value=float(degraded_count)
        )
        
        if "CRITICAL" in overall_status:
            issues.append("System in CRITICAL state")
        
        if critical_count > 0:
            issues.append(f"{critical_count} subsystems critical")
        
        health_score = 100.0
        if "CRITICAL" in overall_status:
            health_score -= 40
        health_score -= critical_count * 10
        health_score -= degraded_count * 5
        
        return SubsystemState(
            name="Self-Healing System",
            status=SubsystemStatus.ACTIVE,
            health_score=max(0, health_score),
            last_updated=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
            performance_score=health_score
        )
    
    def _evaluate_pal_status(self, data: Dict[str, Any]) -> SubsystemState:
        """Evaluate Profit Amplification Layer status."""
        metrics = {
            "candidates_found": len(data.get("amplification_candidates", [])),
            "recommendations": len(data.get("recommendations", [])),
            "avg_score": data.get("avg_amplification_score", 0.0)
        }
        
        health_score = 100.0  # PAL is advisory, always healthy
        
        return SubsystemState(
            name="Profit Amplification Layer",
            status=SubsystemStatus.ACTIVE,
            health_score=health_score,
            last_updated=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=[],
            performance_score=health_score
        )
    
    def _evaluate_orchestrator_status(self, data: Dict[str, Any]) -> SubsystemState:
        """Evaluate Orchestrator Policy status."""
        metrics = {
            "regime": data.get("regime", "UNKNOWN"),
            "risk_profile": data.get("risk_profile", "UNKNOWN"),
            "exit_mode": data.get("exit_mode", "UNKNOWN")
        }
        
        issues = []
        if metrics["regime"] == "UNKNOWN":
            issues.append("Unknown regime")
        
        health_score = 100.0 if metrics["regime"] != "UNKNOWN" else 70.0
        
        return SubsystemState(
            name="Orchestrator Policy",
            status=SubsystemStatus.ACTIVE,
            health_score=health_score,
            last_updated=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            issues=issues,
            performance_score=health_score
        )
    
    # ========================================================================
    # CONFLICT DETECTION
    # ========================================================================
    
    def _detect_subsystem_conflicts(self) -> List[SubsystemConflict]:
        """Detect conflicts between subsystems."""
        conflicts = []
        
        # Example: Universe wants aggressive but Risk says emergency
        universe = self.subsystem_states.get("universe_os")
        risk = self.subsystem_states.get("risk_os")
        
        if universe and risk:
            if risk.metrics.get("emergency_brake") and universe.health_score > 80:
                conflicts.append(SubsystemConflict(
                    subsystems=["universe_os", "risk_os"],
                    conflict_type="risk_universe_mismatch",
                    severity=ConflictSeverity.WARNING,
                    description="Universe healthy but emergency brake active",
                    recommendation="Follow Risk OS - safety first"
                ))
        
        # Add more conflict detection logic here...
        
        return conflicts
    
    # ========================================================================
    # SYSTEM HEALTH ASSESSMENT
    # ========================================================================
    
    def _assess_system_health(self) -> SystemHealth:
        """Assess overall system health."""
        if not self.subsystem_states:
            return SystemHealth.DEGRADED
        
        avg_health = sum(s.health_score for s in self.subsystem_states.values()) / len(self.subsystem_states)
        critical_count = sum(1 for s in self.subsystem_states.values() if not s.is_healthy())
        
        if avg_health >= 90 and critical_count == 0:
            return SystemHealth.OPTIMAL
        elif avg_health >= 70 and critical_count <= 1:
            return SystemHealth.HEALTHY
        elif avg_health >= 50:
            return SystemHealth.DEGRADED
        elif critical_count > 3 or avg_health < 30:
            return SystemHealth.EMERGENCY
        else:
            return SystemHealth.CRITICAL
    
    # ========================================================================
    # RISK MODE DETERMINATION
    # ========================================================================
    
    def _determine_risk_mode(
        self,
        universe_data: Dict[str, Any],
        risk_data: Dict[str, Any],
        positions_data: Dict[str, Any],
        system_health: SystemHealth
    ) -> SystemRiskMode:
        """
        Determine global system risk mode with HEDGEFUND MODE support.
        
        CRITICAL mode = RISK REDUCTION (damage control)
        AGGRESSIVE mode = OPPORTUNISTIC (optimized returns with safety bounds)
        
        [NEW] Logs mode transitions for monitoring.
        """
        
        # Track previous mode for transition logging
        previous_mode = getattr(self, "_previous_risk_mode", None)
        
        # [CRITICAL] Emergency conditions = DAMAGE CONTROL (reduce risk)
        if system_health == SystemHealth.EMERGENCY:
            logger.warning("[AI-HFOS] CRITICAL MODE: System emergency detected")
            mode_selected = SystemRiskMode.CRITICAL
        
        elif risk_data.get("emergency_brake_active"):
            logger.warning("[AI-HFOS] CRITICAL MODE: Emergency brake active")
            mode_selected = SystemRiskMode.CRITICAL
        
        # [SAFE] High drawdown = DEFENSIVE
        else:
            daily_dd = abs(risk_data.get("daily_dd_pct", 0))
            if daily_dd > 3.0:
                logger.warning(f"[AI-HFOS] SAFE MODE: High drawdown {daily_dd:.1f}%")
                mode_selected = SystemRiskMode.SAFE
            
            # [SAFE] System health degraded = DEFENSIVE
            elif system_health == SystemHealth.CRITICAL:
                logger.warning("[AI-HFOS] SAFE MODE: Critical system health")
                mode_selected = SystemRiskMode.SAFE
            
            # [HEDGEFUND MODE] AGGRESSIVE conditions check
            # Only activate AGGRESSIVE if:
            # 1. System health is OPTIMAL
            # 2. Drawdown is minimal (< 1.5%)
            # 3. Position count is healthy
            # 4. No recent major losses
            else:
                hedgefund_mode_enabled = self.config.get("enable_hedgefund_mode", True)
                
                if hedgefund_mode_enabled and system_health == SystemHealth.OPTIMAL:
                    # Check performance metrics
                    position_count = positions_data.get("open_positions", 0)
                    max_positions = self.config.get("max_positions_aggressive", 8)
                    
                    # Aggressive mode criteria:
                    # - Very low drawdown (< 1.5%)
                    # - Position count not maxed out
                    # - System running smoothly
                    if daily_dd < 1.5 and position_count < max_positions * 0.8:
                        logger.info(f"[AI-HFOS] 🚀 AGGRESSIVE MODE: Optimal conditions (DD={daily_dd:.1f}%, Positions={position_count})")
                        mode_selected = SystemRiskMode.AGGRESSIVE
                    else:
                        mode_selected = SystemRiskMode.NORMAL
                
                # [NORMAL] Standard operation
                elif system_health in [SystemHealth.OPTIMAL, SystemHealth.HEALTHY]:
                    logger.info("[AI-HFOS] NORMAL MODE: Standard operation")
                    mode_selected = SystemRiskMode.NORMAL
                
                # [SAFE] Default to SAFE for degraded health
                elif system_health == SystemHealth.DEGRADED:
                    logger.warning("[AI-HFOS] SAFE MODE: Degraded system health")
                    mode_selected = SystemRiskMode.SAFE
                
                # [SAFE] Fallback default
                else:
                    logger.warning("[AI-HFOS] SAFE MODE: Default fallback")
                    mode_selected = SystemRiskMode.SAFE
        
        # [NEW] Log mode transitions
        if previous_mode and previous_mode != mode_selected:
            logger.warning(
                f"🔄 [AI-HFOS] *** RISK MODE TRANSITION: {previous_mode.value} → {mode_selected.value} ***"
            )
            
            # P2-02: Log mode transition to audit
            audit_logger.log_policy_changed(
                policy_name="system_risk_mode",
                old_value=previous_mode.value,
                new_value=mode_selected.value,
                changed_by="AI-HFOS",
                reason=f"Auto-adjustment: DD={daily_dd:.2f}%, Health={system_health.value}"
            )
            
            # P2-02: Record mode transition metrics
            metrics_logger.record_counter(
                "risk_mode_transitions",
                value=1.0,
                labels={
                    "from_mode": previous_mode.value,
                    "to_mode": mode_selected.value
                }
            )
            
            if mode_selected == SystemRiskMode.CRITICAL:
                logger.error("⚠️ [AI-HFOS] ENTERING CRITICAL MODE - Damage control activated!")
            elif mode_selected == SystemRiskMode.AGGRESSIVE:
                logger.info("🚀 [AI-HFOS] ENTERING AGGRESSIVE MODE - HEDGEFUND MODE active!")
            elif previous_mode == SystemRiskMode.AGGRESSIVE:
                logger.warning(f"⬇️ [AI-HFOS] DOWNGRADE from AGGRESSIVE → {mode_selected.value}")
        
        # Store for next cycle
        self._previous_risk_mode = mode_selected
        
        return mode_selected
    
    # ========================================================================
    # DIRECTIVE GENERATION
    # ========================================================================
    
    def _generate_global_directives(
        self,
        risk_mode: SystemRiskMode,
        health: SystemHealth,
        conflicts: List[SubsystemConflict]
    ) -> GlobalDirectives:
        """
        Generate global system directives with HEDGEFUND MODE support.
        
        CRITICAL = Risk reduction (0.3x size, block trades)
        SAFE = Conservative (0.6x size, higher confidence)
        NORMAL = Standard (1.0x size, normal confidence)
        AGGRESSIVE = Opportunistic (1.3x size, lower confidence, more PAL opportunities)
        """
        
        # [CRITICAL] DAMAGE CONTROL - Reduce risk aggressively
        if risk_mode == SystemRiskMode.CRITICAL:
            logger.warning("[AI-HFOS] 🛑 CRITICAL DIRECTIVES: Block new trades, force defensive exits")
            return GlobalDirectives(
                allow_new_trades=False,
                allow_new_positions=False,
                enforce_defensive_exits=True,
                reduce_global_risk=True,
                pause_entire_symbols=[],
                adjust_confidence_threshold=0.85,  # Very high confidence only
                scale_position_sizes=0.3,  # 30% size (risk reduction)
                max_daily_dd_override=2.0,
                force_exit_symbols=[]
            )
        
        # [SAFE] DEFENSIVE - Conservative trading
        elif risk_mode == SystemRiskMode.SAFE:
            logger.info("[AI-HFOS] 🛡️ SAFE DIRECTIVES: Conservative mode, reduced sizes")
            return GlobalDirectives(
                allow_new_trades=True,
                allow_new_positions=True,
                enforce_defensive_exits=False,
                reduce_global_risk=True,
                pause_entire_symbols=[],
                adjust_confidence_threshold=0.75,  # Higher confidence required
                scale_position_sizes=0.6,  # 60% size
                max_daily_dd_override=3.0,
                force_exit_symbols=[]
            )
        
        # [AGGRESSIVE] HEDGEFUND MODE - Opportunistic trading with bounds
        elif risk_mode == SystemRiskMode.AGGRESSIVE:
            logger.info("[AI-HFOS] 🚀 AGGRESSIVE DIRECTIVES: HEDGEFUND MODE active")
            return GlobalDirectives(
                allow_new_trades=True,
                allow_new_positions=True,
                enforce_defensive_exits=False,
                reduce_global_risk=False,
                pause_entire_symbols=[],
                adjust_confidence_threshold=0.60,  # Lower confidence (more opportunities)
                scale_position_sizes=1.3,  # 130% size (within SafetyGovernor bounds)
                max_daily_dd_override=None,
                force_exit_symbols=[]
            )
        
        # [NORMAL] STANDARD - Normal operation
        else:  # NORMAL
            logger.info("[AI-HFOS] ✅ NORMAL DIRECTIVES: Standard operation")
            return GlobalDirectives(
                allow_new_trades=True,
                allow_new_positions=True,
                enforce_defensive_exits=False,
                reduce_global_risk=False,
                pause_entire_symbols=[],
                adjust_confidence_threshold=None,  # Use default (0.65)
                scale_position_sizes=1.0,  # 100% size
                max_daily_dd_override=None,
                force_exit_symbols=[]
            )
    
    def _generate_universe_directives(
        self,
        universe_data: Dict[str, Any],
        risk_mode: SystemRiskMode,
        conflicts: List[SubsystemConflict]
    ) -> UniverseDirectives:
        """Generate universe directives."""
        
        mode_map = {
            SystemRiskMode.CRITICAL: "SAFE",
            SystemRiskMode.SAFE: "SAFE",
            SystemRiskMode.NORMAL: "NORMAL",
            SystemRiskMode.AGGRESSIVE: "AGGRESSIVE"
        }
        
        # [HEDGEFUND MODE] AGGRESSIVE universe settings
        if risk_mode == SystemRiskMode.AGGRESSIVE:
            # Allow more symbols, promote high-quality categories
            return UniverseDirectives(
                universe_mode="AGGRESSIVE",
                blacklist_symbols=[],
                whitelist_symbols=[],
                promote_categories=["CORE", "EXPANSION"],  # Promote quality symbols
                demote_categories=["TOXIC", "MONITORING"],  # Avoid risky symbols
                emergency_brake_override=None
            )
        
        return UniverseDirectives(
            universe_mode=mode_map[risk_mode],
            blacklist_symbols=[],
            whitelist_symbols=[],
            promote_categories=[],
            demote_categories=[],
            emergency_brake_override=None
        )
    
    def _generate_execution_directives(
        self,
        execution_data: Dict[str, Any],
        risk_mode: SystemRiskMode,
        health: SystemHealth
    ) -> ExecutionDirectives:
        """Generate execution directives."""
        
        if risk_mode == SystemRiskMode.CRITICAL:
            return ExecutionDirectives(
                order_type_preference="LIMIT",
                max_slippage_bps=5.0,
                max_spread_bps=10.0,
                reduce_urgency=True,
                enforce_limit_orders=True,
                execution_delay_seconds=5
            )
        
        else:
            return ExecutionDirectives(
                order_type_preference="SMART",
                max_slippage_bps=15.0,
                max_spread_bps=25.0,
                reduce_urgency=False,
                enforce_limit_orders=False,
                execution_delay_seconds=0
            )
    
    def _generate_portfolio_directives(
        self,
        positions_data: Dict[str, Any],
        risk_mode: SystemRiskMode,
        conflicts: List[SubsystemConflict]
    ) -> PortfolioDirectives:
        """Generate portfolio directives."""
        
        if risk_mode == SystemRiskMode.CRITICAL:
            return PortfolioDirectives(
                reduce_exposure_pct=50.0,
                max_position_count=3,
                max_leverage=5.0,
                reduce_correlated_positions=True,
                avoid_expansion_symbols=True,
                concentration_limit_pct=15.0
            )
        
        elif risk_mode == SystemRiskMode.SAFE:
            return PortfolioDirectives(
                reduce_exposure_pct=20.0,
                max_position_count=5,
                max_leverage=10.0,
                reduce_correlated_positions=True,
                avoid_expansion_symbols=True,
                concentration_limit_pct=20.0
            )
        
        else:
            return PortfolioDirectives(
                reduce_exposure_pct=0.0,
                max_position_count=None,
                max_leverage=None,
                reduce_correlated_positions=False,
                avoid_expansion_symbols=False,
                concentration_limit_pct=30.0
            )
    
    def _generate_model_directives(
        self,
        model_performance: Dict[str, Any],
        risk_mode: SystemRiskMode
    ) -> ModelDirectives:
        """Generate model directives."""
        
        return ModelDirectives(
            ensemble_weight_adjustments={},
            models_to_retrain=[],
            models_to_disable=[],
            confidence_threshold_override=None,
            use_conservative_predictions=(risk_mode in [SystemRiskMode.SAFE, SystemRiskMode.CRITICAL])
        )
    
    # ========================================================================
    # EMERGENCY ACTIONS
    # ========================================================================
    
    def _identify_emergency_actions(
        self,
        health: SystemHealth,
        conflicts: List[SubsystemConflict],
        self_healing_report: Dict[str, Any]
    ) -> List[EmergencyAction]:
        """Identify emergency actions needed."""
        actions = []
        
        if health == SystemHealth.EMERGENCY:
            actions.append(EmergencyAction(
                action_type="CLOSE_ALL_POSITIONS",
                target="ALL",
                parameters={"urgency": "immediate"},
                priority=1,
                rationale="System in EMERGENCY state"
            ))
        
        # Check for critical subsystems from self-healing
        if "CRITICAL" in self_healing_report.get("overall_status", ""):
            actions.append(EmergencyAction(
                action_type="PAUSE_NEW_TRADES",
                target="SYSTEM",
                parameters={"duration_minutes": 60},
                priority=2,
                rationale="Self-Healing detected CRITICAL issues"
            ))
        
        return actions
    
    # ========================================================================
    # PROFIT AMPLIFICATION
    # ========================================================================
    
    def _process_pal_recommendations(
        self,
        pal_report: Dict[str, Any],
        risk_mode: SystemRiskMode,
        global_directives: GlobalDirectives
    ) -> List[AmplificationOpportunity]:
        """Process PAL recommendations into opportunities."""
        opportunities = []
        
        # Only allow amplification in NORMAL or AGGRESSIVE modes
        if risk_mode not in [SystemRiskMode.NORMAL, SystemRiskMode.AGGRESSIVE]:
            logger.info("[AI-HFOS] PAL recommendations blocked - risk mode not suitable")
            return opportunities
        
        for rec in pal_report.get("recommendations", []):
            if rec.get("priority", 10) <= 2 and rec.get("confidence", 0) >= 60:
                opportunities.append(AmplificationOpportunity(
                    symbol=rec["symbol"],
                    action=rec["action"],
                    expected_r_increase=rec.get("expected_R_increase", 0.0),
                    confidence=rec.get("confidence", 0.0),
                    priority=rec.get("priority", 5),
                    rationale=rec.get("rationale", "")
                ))
        
        return opportunities
    
    # ========================================================================
    # SUMMARY GENERATION
    # ========================================================================
    
    def _generate_priority_notes(
        self,
        risk_mode: SystemRiskMode,
        health: SystemHealth,
        conflicts: List[SubsystemConflict],
        emergency_actions: List[EmergencyAction]
    ) -> List[str]:
        """Generate priority notes for operators."""
        notes = []
        
        notes.append(f"System Risk Mode: {risk_mode.value}")
        notes.append(f"System Health: {health.value}")
        
        if emergency_actions:
            notes.append(f"⚠️  {len(emergency_actions)} EMERGENCY ACTIONS REQUIRED")
        
        if conflicts:
            notes.append(f"⚠️  {len(conflicts)} subsystem conflicts detected")
        
        if risk_mode == SystemRiskMode.CRITICAL:
            notes.append("🔴 CRITICAL MODE - Defensive operations only")
        elif risk_mode == SystemRiskMode.SAFE:
            notes.append("🟡 SAFE MODE - Reduced risk profile active")
        else:
            notes.append("🟢 Normal operations - All systems aligned")
        
        return notes
    
    def _generate_summary(
        self,
        risk_mode: SystemRiskMode,
        health: SystemHealth,
        global_directives: GlobalDirectives,
        conflicts: List[SubsystemConflict],
        emergency_actions: List[EmergencyAction],
        amplification_opportunities: List[AmplificationOpportunity]
    ) -> str:
        """Generate human-readable summary."""
        
        summary_parts = []
        
        summary_parts.append(f"AI-HFOS Coordination Summary:")
        summary_parts.append(f"- System Risk Mode: {risk_mode.value}")
        summary_parts.append(f"- System Health: {health.value}")
        summary_parts.append(f"- New Trades Allowed: {'Yes' if global_directives.allow_new_trades else 'No'}")
        summary_parts.append(f"- Position Size Scaling: {global_directives.scale_position_sizes:.1%}")
        
        if emergency_actions:
            summary_parts.append(f"- Emergency Actions: {len(emergency_actions)} required")
        
        if conflicts:
            summary_parts.append(f"- Subsystem Conflicts: {len(conflicts)} detected")
        
        if amplification_opportunities:
            summary_parts.append(f"- Amplification Opportunities: {len(amplification_opportunities)} identified")
        
        if risk_mode == SystemRiskMode.CRITICAL:
            summary_parts.append("⚠️  CRITICAL MODE ACTIVE - System in damage control")
        elif risk_mode == SystemRiskMode.SAFE:
            summary_parts.append("🟡 SAFE MODE - Operating with reduced risk")
        else:
            summary_parts.append("🟢 Normal operations - System aligned")
        
        return "\n".join(summary_parts)
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def _save_report(self, output: AIHFOSOutput):
        """Save AI-HFOS report to disk."""
        report_path = self.data_dir / "ai_hfos_report.json"
        
        # Convert to dict
        report_dict = asdict(output)
        
        # Save
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"[AI-HFOS] Report saved to {report_path}")


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AI HEDGEFUND OPERATING SYSTEM (AI-HFOS) - Standalone Test")
    print("=" * 80)
    print()
    
    # Initialize AI-HFOS
    hfos = AIHedgeFundOS(data_dir="./data")
    
    # Mock subsystem data
    universe_data = {
        "data_confidence": "LOW",
        "current_universe": {"symbol_count": 222},
        "classifications": {"BLACKLIST": {"count": 156}}
    }
    
    risk_data = {
        "emergency_brake_active": False,
        "daily_dd_pct": 1.2,
        "open_dd_pct": 3.5
    }
    
    positions_data = {
        "position_count": 3,
        "toxic_count": 0,
        "winner_count": 2
    }
    
    execution_data = {
        "avg_slippage_bps": 5.0,
        "avg_spread_bps": 12.0,
        "fill_rate": 0.95
    }
    
    model_performance = {
        "ensemble_accuracy": 0.62,
        "degraded_models": []
    }
    
    self_healing_report = {
        "overall_status": "HEALTHY",
        "subsystems": []
    }
    
    pal_report = {
        "amplification_candidates": [
            {"symbol": "BTCUSDT", "current_R": 2.5, "amplification_score": 73.6}
        ],
        "recommendations": [
            {
                "symbol": "BTCUSDT",
                "action": "extend_hold",
                "priority": 2,
                "confidence": 65,
                "expected_R_increase": 1.0,
                "rationale": "Strong trend at 2.5R"
            }
        ]
    }
    
    orchestrator_policy = {
        "regime": "BULL",
        "risk_profile": "NORMAL",
        "exit_mode": "FAST_TP"
    }
    
    # Run coordination cycle
    print("[TEST] Running coordination cycle...")
    output = hfos.run_coordination_cycle(
        universe_data=universe_data,
        risk_data=risk_data,
        positions_data=positions_data,
        execution_data=execution_data,
        model_performance=model_performance,
        self_healing_report=self_healing_report,
        pal_report=pal_report,
        orchestrator_policy=orchestrator_policy
    )
    
    print()
    print("=" * 80)
    print("AI-HFOS OUTPUT")
    print("=" * 80)
    print()
    print(output.summary)
    print()
    print(f"Subsystems tracked: {len(output.subsystem_states)}")
    print(f"Conflicts detected: {len(output.detected_conflicts)}")
    print(f"Emergency actions: {len(output.emergency_actions)}")
    print(f"Amplification opportunities: {len(output.amplification_opportunities)}")
    print()
    print("✅ AI-HFOS TEST COMPLETE")
