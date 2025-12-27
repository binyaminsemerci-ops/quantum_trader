"""
RL v3 Dashboard API Routes
===========================

REST API endpoints for RL v3 (PPO) dashboard and metrics visualization.

Endpoints:
- GET /api/v1/rl-v3/dashboard/summary - Get summary statistics only
- GET /api/v1/rl-v3/dashboard/full - Get summary + recent decisions

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import structlog

from backend.domains.learning.rl_v3.metrics_v3 import RLv3MetricsStore

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/rl-v3/dashboard", tags=["RL v3 Dashboard"])


class RLv3Decision(BaseModel):
    """Individual RL v3 decision record."""
    
    timestamp: str = Field(..., description="ISO8601 timestamp")
    symbol: Optional[str] = Field(None, description="Trading symbol")
    action: int = Field(..., description="Action code (0-5)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    value: float = Field(default=0.0, description="Value estimate")
    trace_id: Optional[str] = Field(None, description="Trace ID for distributed tracing")
    shadow_mode: bool = Field(default=True, description="Whether decision was in shadow mode")


class RLv3Summary(BaseModel):
    """Summary statistics for RL v3 decisions."""
    
    total_decisions: int = Field(..., description="Total decisions made")
    action_counts: Dict[int, int] = Field(..., description="Count per action")
    action_distribution: Dict[int, float] = Field(..., description="Percentage per action")
    avg_confidence: Optional[float] = Field(None, description="Average confidence")
    max_confidence: Optional[float] = Field(None, description="Maximum confidence")
    min_confidence: Optional[float] = Field(None, description="Minimum confidence")


class RLv3TrainingRun(BaseModel):
    """Individual training run record."""
    
    timestamp: str = Field(..., description="ISO8601 timestamp")
    episodes: int = Field(..., description="Number of episodes trained")
    duration_seconds: float = Field(..., description="Training duration in seconds")
    success: bool = Field(..., description="Whether training completed successfully")
    error: Optional[str] = Field(None, description="Error message if failed")
    avg_reward: Optional[float] = Field(None, description="Average reward across episodes")
    final_reward: Optional[float] = Field(None, description="Final episode reward")
    avg_policy_loss: Optional[float] = Field(None, description="Average policy loss")
    avg_value_loss: Optional[float] = Field(None, description="Average value loss")


class RLv3TrainingSummary(BaseModel):
    """Summary statistics for RL v3 training runs."""
    
    total_runs: int = Field(..., description="Total training runs")
    success_count: int = Field(..., description="Number of successful runs")
    failure_count: int = Field(..., description="Number of failed runs")
    last_run_at: Optional[str] = Field(None, description="Timestamp of last run")
    last_error: Optional[str] = Field(None, description="Last error message if any")
    avg_duration_seconds: Optional[float] = Field(None, description="Average training duration")
    avg_reward: Optional[float] = Field(None, description="Average reward across successful runs")


class RLv3TrainingSummaryResponse(BaseModel):
    """Extended training summary with config and recent runs."""
    
    enabled: bool = Field(..., description="Whether training is enabled")
    interval_minutes: int = Field(..., description="Training interval in minutes")
    episodes_per_run: int = Field(..., description="Episodes per training run")
    total_runs: int = Field(..., description="Total training runs")
    success_rate: float = Field(..., description="Success rate (0-1)")
    last_run_at: Optional[str] = Field(None, description="Timestamp of last run")
    last_error: Optional[str] = Field(None, description="Last error message if any")
    recent_runs: List[RLv3TrainingRun] = Field(..., description="Recent training runs")


class RLv3DashboardResponse(BaseModel):
    """Complete dashboard response with summary and recent decisions."""
    
    summary: RLv3Summary = Field(..., description="Decision summary statistics")
    recent_decisions: List[RLv3Decision] = Field(..., description="Recent decisions")
    training_summary: RLv3TrainingSummary = Field(..., description="Training summary statistics")
    recent_training_runs: List[RLv3TrainingRun] = Field(..., description="Recent training runs")


class RLv3LiveStatusResponse(BaseModel):
    """Live orchestrator status response."""
    
    enabled: bool = Field(..., description="Whether live orchestrator is enabled")
    mode: str = Field(..., description="Current mode (OFF/SHADOW/PRIMARY/HYBRID)")
    min_confidence: float = Field(..., description="Minimum confidence threshold")
    trade_intents_total: int = Field(..., description="Total trade intents published")
    trade_intents_executed: int = Field(..., description="Trade intents executed")
    shadow_decisions: int = Field(..., description="Decisions made in SHADOW mode")
    last_decision_at: Optional[str] = Field(None, description="Timestamp of last decision")
    last_trade_intent_at: Optional[str] = Field(None, description="Timestamp of last trade intent")


@router.get("/summary", response_model=RLv3Summary)
async def get_rl_v3_summary():
    """
    Get RL v3 summary statistics only.
    
    Returns:
        Summary with total counts, action distribution, and confidence stats
    """
    try:
        store = RLv3MetricsStore.instance()
        data = store.get_summary()
        
        return RLv3Summary(
            total_decisions=data["total_decisions"],
            action_counts=data["action_counts"],
            action_distribution=data["action_distribution"],
            avg_confidence=data["avg_confidence"],
            max_confidence=data["max_confidence"],
            min_confidence=data["min_confidence"]
        )
    except Exception as e:
        logger.error("[RL v3 Dashboard] Failed to get summary", error=str(e))
        # Return empty summary on error
        return RLv3Summary(
            total_decisions=0,
            action_counts={i: 0 for i in range(6)},
            action_distribution={i: 0.0 for i in range(6)},
            avg_confidence=None,
            max_confidence=None,
            min_confidence=None
        )


@router.get("/training-summary", response_model=RLv3TrainingSummaryResponse)
async def get_rl_v3_training_summary(request: Any = None):
    """
    Get RL v3 training summary with config and recent runs.
    
    Returns:
        Training summary with daemon config, statistics, and recent runs
    """
    try:
        from fastapi import Request
        
        store = RLv3MetricsStore.instance()
        
        # Get training summary from metrics store
        training_data = store.get_training_summary()
        
        # Get daemon status from app state (if available)
        daemon_config = {
            "enabled": True,
            "interval_minutes": 30,
            "episodes_per_run": 2
        }
        
        # Try to get live daemon status
        if hasattr(request, "app") and hasattr(request.app.state, "rl_v3_training_daemon"):
            daemon = request.app.state.rl_v3_training_daemon
            daemon_status = daemon.get_status()
            daemon_config["enabled"] = daemon_status["enabled"]
            daemon_config["interval_minutes"] = daemon_status["interval_minutes"]
            daemon_config["episodes_per_run"] = daemon_status["episodes_per_run"]
        
        # Get recent training runs
        recent_runs_data = store.get_recent_training_runs(limit=10)
        recent_runs = [
            RLv3TrainingRun(
                timestamp=run["timestamp"],
                episodes=run["episodes"],
                duration_seconds=run["duration_seconds"],
                success=run["success"],
                error=run.get("error"),
                avg_reward=run.get("avg_reward"),
                final_reward=run.get("final_reward"),
                avg_policy_loss=run.get("avg_policy_loss"),
                avg_value_loss=run.get("avg_value_loss")
            )
            for run in recent_runs_data
        ]
        
        # Calculate success rate
        total_runs = training_data["total_runs"]
        success_rate = (
            training_data["success_count"] / total_runs 
            if total_runs > 0 else 0.0
        )
        
        return RLv3TrainingSummaryResponse(
            enabled=daemon_config["enabled"],
            interval_minutes=daemon_config["interval_minutes"],
            episodes_per_run=daemon_config["episodes_per_run"],
            total_runs=training_data["total_runs"],
            success_rate=success_rate,
            last_run_at=training_data["last_run_at"],
            last_error=training_data["last_error"],
            recent_runs=recent_runs
        )
        
    except Exception as e:
        logger.error("[RL v3 Dashboard] Failed to get training summary", error=str(e))
        # Return empty summary on error
        return RLv3TrainingSummaryResponse(
            enabled=False,
            interval_minutes=0,
            episodes_per_run=0,
            total_runs=0,
            success_rate=0.0,
            last_run_at=None,
            last_error=str(e),
            recent_runs=[]
        )


@router.get("/full", response_model=RLv3DashboardResponse)
async def get_rl_v3_dashboard(limit: int = 50, training_limit: int = 10):
    """
    Get complete RL v3 dashboard with summary and recent decisions.
    
    Args:
        limit: Maximum number of recent decisions to return (default 50)
        training_limit: Maximum number of training runs to return (default 10)
        
    Returns:
        Dashboard with decision summary, recent decisions, training summary, and recent training runs
    """
    try:
        store = RLv3MetricsStore.instance()
        
        # Get decision summary
        summary_data = store.get_summary()
        summary = RLv3Summary(
            total_decisions=summary_data["total_decisions"],
            action_counts=summary_data["action_counts"],
            action_distribution=summary_data["action_distribution"],
            avg_confidence=summary_data["avg_confidence"],
            max_confidence=summary_data["max_confidence"],
            min_confidence=summary_data["min_confidence"]
        )
        
        # Get recent decisions
        recent_data = store.get_recent_decisions(limit=limit)
        recent_decisions = [
            RLv3Decision(
                timestamp=d["timestamp"],
                symbol=d.get("symbol"),
                action=d["action"],
                confidence=d["confidence"],
                value=d.get("value", 0.0),
                trace_id=d.get("trace_id"),
                shadow_mode=d.get("shadow_mode", True)
            )
            for d in recent_data
        ]
        
        # Get training summary
        training_summary_data = store.get_training_summary()
        training_summary = RLv3TrainingSummary(
            total_runs=training_summary_data["total_runs"],
            success_count=training_summary_data["success_count"],
            failure_count=training_summary_data["failure_count"],
            last_run_at=training_summary_data["last_run_at"],
            last_error=training_summary_data["last_error"],
            avg_duration_seconds=training_summary_data["avg_duration_seconds"],
            avg_reward=training_summary_data["avg_reward"]
        )
        
        # Get recent training runs
        training_runs_data = store.get_recent_training_runs(limit=training_limit)
        recent_training_runs = [
            RLv3TrainingRun(
                timestamp=run["timestamp"],
                episodes=run["episodes"],
                duration_seconds=run["duration_seconds"],
                success=run["success"],
                error=run.get("error"),
                avg_reward=run.get("avg_reward"),
                final_reward=run.get("final_reward"),
                avg_policy_loss=run.get("avg_policy_loss"),
                avg_value_loss=run.get("avg_value_loss")
            )
            for run in training_runs_data
        ]
        
        logger.debug(
            "[RL v3 Dashboard] Dashboard data retrieved",
            total_decisions=summary.total_decisions,
            recent_count=len(recent_decisions),
            total_training_runs=training_summary.total_runs,
            recent_training_count=len(recent_training_runs)
        )
        
        return RLv3DashboardResponse(
            summary=summary,
            recent_decisions=recent_decisions,
            training_summary=training_summary,
            recent_training_runs=recent_training_runs
        )
        
    except Exception as e:
        logger.error("[RL v3 Dashboard] Failed to get dashboard", error=str(e))
        # Return empty dashboard on error
        return RLv3DashboardResponse(
            summary=RLv3Summary(
                total_decisions=0,
                action_counts={i: 0 for i in range(6)},
                action_distribution={i: 0.0 for i in range(6)},
                avg_confidence=None,
                max_confidence=None,
                min_confidence=None
            ),
            recent_decisions=[],
            training_summary=RLv3TrainingSummary(
                total_runs=0,
                success_count=0,
                failure_count=0,
                last_run_at=None,
                last_error=None,
                avg_duration_seconds=None,
                avg_reward=None
            ),
            recent_training_runs=[]
        )


@router.get("/live-status", response_model=RLv3LiveStatusResponse)
async def get_rl_v3_live_status(request: Any = None):
    """
    Get RL v3 live orchestrator status.
    
    Returns:
        Live status with mode, trade counts, shadow decisions
    """
    try:
        from fastapi import Request
        
        store = RLv3MetricsStore.instance()
        live_status = store.get_live_status()
        
        # Get live config from orchestrator (via app.state)
        enabled = True
        mode = "SHADOW"
        min_confidence = 0.6
        
        if hasattr(request, "app") and hasattr(request.app.state, "rl_v3_live_orchestrator"):
            orch = request.app.state.rl_v3_live_orchestrator
            config = orch.get_config()
            enabled = config.get("enabled", True)
            mode = config.get("mode", "SHADOW")
            min_confidence = config.get("min_confidence", 0.6)
        
        logger.debug(
            "[RL v3 Dashboard] Live status retrieved",
            mode=mode,
            enabled=enabled,
            trade_intents_total=live_status["trade_intents_total"],
        )
        
        return RLv3LiveStatusResponse(
            enabled=enabled,
            mode=mode,
            min_confidence=min_confidence,
            trade_intents_total=live_status["trade_intents_total"],
            trade_intents_executed=live_status["trade_intents_executed"],
            shadow_decisions=live_status["shadow_decisions"],
            last_decision_at=live_status["last_decision_at"],
            last_trade_intent_at=live_status["last_trade_intent_at"],
        )
        
    except Exception as e:
        logger.error("[RL v3 Dashboard] Failed to get live status", error=str(e))
        # Return default status on error
        return RLv3LiveStatusResponse(
            enabled=False,
            mode="OFF",
            min_confidence=0.6,
            trade_intents_total=0,
            trade_intents_executed=0,
            shadow_decisions=0,
            last_decision_at=None,
            last_trade_intent_at=None,
        )
