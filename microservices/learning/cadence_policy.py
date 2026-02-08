"""
Learning Cadence Policy - Gate-keeper for continuous learning.

Implements three-level decision framework:
- NIVÅ 1: Data-klarhet (hard gates)
- NIVÅ 2: Lærings-trigger (when to train)
- NIVÅ 3: Lærings-type (what's allowed)

Philosophy: "Vi lærer sakte av sannhet, ikke raskt av støy."
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OutcomeLabel(str, Enum):
    """Trade outcome classification"""
    WIN = "WIN"
    LOSS = "LOSS"
    NEUTRAL = "NEUTRAL"


@dataclass
class Trade:
    """Simplified trade record for cadence evaluation"""
    timestamp: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl_percent: float
    confidence: float
    model_id: str
    outcome_label: str
    duration_seconds: Optional[float] = None
    strategy_id: Optional[str] = None
    position_size: Optional[float] = None
    exit_reason: Optional[str] = None


class DataReadinessGate:
    """
    NIVÅ 1 - Hard gates før læring tillates.
    
    ALL checks must pass for learning to be authorized.
    """
    
    def __init__(
        self,
        min_trades: int = 50,
        min_days: int = 3,
        min_win_pct: float = 0.20,
        min_loss_pct: float = 0.20
    ):
        self.min_trades = min_trades
        self.min_days = min_days
        self.min_win_pct = min_win_pct
        self.min_loss_pct = min_loss_pct
    
    def check_minimum_trades(self, trades: List[Trade]) -> Tuple[bool, str]:
        """Check if we have enough trades"""
        count = len(trades)
        passed = count >= self.min_trades
        reason = f"Trade count: {count}/{self.min_trades}"
        return passed, reason
    
    def check_time_span(self, trades: List[Trade]) -> Tuple[bool, str]:
        """Check if data spans enough days to avoid single-phase bias"""
        if not trades:
            return False, "No trades available"
        
        timestamps = [t.timestamp for t in trades]
        span = max(timestamps) - min(timestamps)
        span_days = span.total_seconds() / 86400
        
        passed = span_days >= self.min_days
        reason = f"Time span: {span_days:.1f}/{self.min_days} days"
        return passed, reason
    
    def check_outcome_diversity(self, trades: List[Trade]) -> Tuple[bool, str]:
        """Check if we have diversity in outcomes (not all WIN or all LOSS)"""
        if not trades:
            return False, "No trades available"
        
        wins = sum(1 for t in trades if t.outcome_label == OutcomeLabel.WIN)
        losses = sum(1 for t in trades if t.outcome_label == OutcomeLabel.LOSS)
        total = len(trades)
        
        win_pct = wins / total if total > 0 else 0
        loss_pct = losses / total if total > 0 else 0
        
        passed = win_pct >= self.min_win_pct and loss_pct >= self.min_loss_pct
        reason = f"Diversity: WIN={win_pct:.1%} (min {self.min_win_pct:.0%}), LOSS={loss_pct:.1%} (min {self.min_loss_pct:.0%})"
        return passed, reason
    
    def check_data_integrity(self, trades: List[Trade]) -> Tuple[bool, str]:
        """Check if all trades have required fields"""
        if not trades:
            return False, "No trades available"
        
        required_fields = ['entry_price', 'exit_price', 'pnl_percent', 'outcome_label']
        invalid_count = 0
        
        for trade in trades:
            if trade.entry_price <= 0 or trade.exit_price <= 0:
                invalid_count += 1
            elif trade.outcome_label not in [OutcomeLabel.WIN, OutcomeLabel.LOSS, OutcomeLabel.NEUTRAL]:
                invalid_count += 1
        
        passed = invalid_count == 0
        reason = f"Data integrity: {len(trades) - invalid_count}/{len(trades)} valid trades"
        return passed, reason
    
    def evaluate(self, trades: List[Trade]) -> Tuple[bool, str]:
        """
        Evaluate ALL gates. ALL must pass.
        
        Returns:
            (passed, reason) - reason contains first failure or success message
        """
        checks = [
            self.check_minimum_trades(trades),
            self.check_time_span(trades),
            self.check_outcome_diversity(trades),
            self.check_data_integrity(trades)
        ]
        
        for passed, reason in checks:
            if not passed:
                return False, f"❌ Gate failed: {reason}"
        
        return True, f"✅ All gates passed: {len(trades)} trades, {checks[1][1]}, {checks[2][1]}"


class LearningTrigger:
    """
    NIVÅ 2 - Når trening faktisk skjer.
    
    Multiple triggers can be active, but only ONE needs to fire.
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        time_interval_hours: int = 24
    ):
        self.batch_size = batch_size
        self.time_interval_hours = time_interval_hours
    
    def should_trigger_batch(self, new_trades_count: int) -> Tuple[bool, str]:
        """Trigger when we've accumulated enough new trades"""
        triggered = new_trades_count >= self.batch_size
        reason = f"Batch trigger: {new_trades_count}/{self.batch_size} new trades"
        return triggered, reason
    
    def should_trigger_time(self, last_training_ts: Optional[datetime]) -> Tuple[bool, str]:
        """Trigger if enough time has passed since last training"""
        if last_training_ts is None:
            return True, "Time trigger: Never trained before"
        
        now = datetime.now(timezone.utc)
        elapsed = now - last_training_ts
        elapsed_hours = elapsed.total_seconds() / 3600
        
        triggered = elapsed_hours >= self.time_interval_hours
        reason = f"Time trigger: {elapsed_hours:.1f}/{self.time_interval_hours}h elapsed"
        return triggered, reason
    
    def should_trigger_regime(self, current_regime: Optional[str], last_regime: Optional[str]) -> Tuple[bool, str]:
        """Trigger if market regime has changed (future feature)"""
        # Not implemented yet - requires regime detection
        return False, "Regime trigger: Not implemented"
    
    def evaluate(
        self,
        new_trades_count: int,
        last_training_ts: Optional[datetime],
        current_regime: Optional[str] = None,
        last_regime: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """
        Evaluate all triggers. ANY can fire.
        
        Returns:
            (triggered, reason, trigger_type)
        """
        # Check batch trigger
        batch_triggered, batch_reason = self.should_trigger_batch(new_trades_count)
        if batch_triggered:
            return True, batch_reason, "batch"
        
        # Check time trigger
        time_triggered, time_reason = self.should_trigger_time(last_training_ts)
        if time_triggered:
            return True, time_reason, "time"
        
        # Check regime trigger (future)
        regime_triggered, regime_reason = self.should_trigger_regime(current_regime, last_regime)
        if regime_triggered:
            return True, regime_reason, "regime"
        
        # Nothing triggered
        return False, f"No triggers fired: {batch_reason}, {time_reason}", "none"


class LearningAuthorization:
    """
    NIVÅ 3 - Hva slags læring er lov.
    
    Different learning types have different requirements.
    """
    
    def __init__(
        self,
        calibration_min_trades: int = 50,
        shadow_min_trades: int = 100,
        shadow_min_days: int = 3,
        promotion_min_trades: int = 500,
        promotion_min_days: int = 14
    ):
        self.calibration_min_trades = calibration_min_trades
        self.shadow_min_trades = shadow_min_trades
        self.shadow_min_days = shadow_min_days
        self.promotion_min_trades = promotion_min_trades
        self.promotion_min_days = promotion_min_days
    
    def authorize_calibration(self, trade_count: int, time_span_days: float) -> bool:
        """Low-risk: Adjusts thresholds, confidence scaling, ensemble weights"""
        return trade_count >= self.calibration_min_trades
    
    def authorize_shadow_training(self, trade_count: int, time_span_days: float) -> bool:
        """Zero-risk: Train models offline, no production impact"""
        return (
            trade_count >= self.shadow_min_trades and
            time_span_days >= self.shadow_min_days
        )
    
    def authorize_promotion(
        self,
        trade_count: int,
        time_span_days: float,
        manual_approval: bool = False
    ) -> bool:
        """High-risk: Actually change production decisions"""
        return (
            trade_count >= self.promotion_min_trades and
            time_span_days >= self.promotion_min_days and
            manual_approval  # ALWAYS require manual approval
        )
    
    def get_allowed_actions(self, trade_count: int, time_span_days: float) -> List[str]:
        """Get list of currently authorized learning actions"""
        actions = []
        
        if self.authorize_calibration(trade_count, time_span_days):
            actions.append("calibration")
        
        if self.authorize_shadow_training(trade_count, time_span_days):
            actions.append("shadow")
        
        # Promotion requires manual approval, so we don't auto-add it
        # It would be added via explicit API call with approval flag
        
        return actions


class LearningCadencePolicy:
    """
    Main orchestrator combining all three levels.
    
    This is the single entry point for checking learning readiness.
    """
    
    def __init__(
        self,
        clm_storage_path: str = "/home/qt/quantum_trader/data/clm_trades.jsonl",
        state_path: str = "/home/qt/quantum_trader/data/learning_cadence_state.json"
    ):
        self.clm_path = Path(clm_storage_path)
        self.state_path = Path(state_path)
        
        # Initialize three levels
        self.gate = DataReadinessGate(
            min_trades=50,
            min_days=3,
            min_win_pct=0.20,
            min_loss_pct=0.20
        )
        
        self.trigger = LearningTrigger(
            batch_size=100,
            time_interval_hours=24
        )
        
        self.auth = LearningAuthorization(
            calibration_min_trades=50,
            shadow_min_trades=100,
            shadow_min_days=3,
            promotion_min_trades=500,
            promotion_min_days=14
        )
        
        # Load state
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load persistent state (last training info)"""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                    # Parse datetime
                    if data.get("last_training_timestamp"):
                        data["last_training_timestamp"] = datetime.fromisoformat(data["last_training_timestamp"])
                    return data
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        
        # Default state
        return {
            "last_training_timestamp": None,
            "last_training_trade_count": 0,
            "last_training_action": None,
            "total_trainings": 0
        }
    
    def _save_state(self):
        """Persist state to disk"""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = self.state.copy()
            # Serialize datetime
            if data.get("last_training_timestamp"):
                data["last_training_timestamp"] = data["last_training_timestamp"].isoformat()
            
            with open(self.state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_trades(self) -> List[Trade]:
        """Load trades from SimpleCLM storage"""
        if not self.clm_path.exists():
            logger.warning(f"CLM file not found: {self.clm_path}")
            return []
        
        trades = []
        try:
            with open(self.clm_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    
                    # Parse timestamp
                    ts = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                    
                    trade = Trade(
                        timestamp=ts,
                        symbol=data["symbol"],
                        side=data["side"],
                        entry_price=data["entry_price"],
                        exit_price=data["exit_price"],
                        pnl_percent=data["pnl_percent"],
                        confidence=data["confidence"],
                        model_id=data["model_id"],
                        outcome_label=data["outcome_label"],
                        duration_seconds=data.get("duration_seconds"),
                        strategy_id=data.get("strategy_id"),
                        position_size=data.get("position_size"),
                        exit_reason=data.get("exit_reason")
                    )
                    trades.append(trade)
        
        except Exception as e:
            logger.error(f"Failed to load trades from {self.clm_path}: {e}", exc_info=True)
        
        return trades
    
    def _calculate_time_span(self, trades: List[Trade]) -> float:
        """Calculate time span in days"""
        if not trades:
            return 0.0
        timestamps = [t.timestamp for t in trades]
        span = max(timestamps) - min(timestamps)
        return span.total_seconds() / 86400
    
    def _calculate_win_rate(self, trades: List[Trade]) -> float:
        """Calculate win rate (excluding NEUTRAL)"""
        decisive = [t for t in trades if t.outcome_label in [OutcomeLabel.WIN, OutcomeLabel.LOSS]]
        if not decisive:
            return 0.0
        wins = sum(1 for t in decisive if t.outcome_label == OutcomeLabel.WIN)
        return wins / len(decisive)
    
    def _calculate_loss_rate(self, trades: List[Trade]) -> float:
        """Calculate loss rate (excluding NEUTRAL)"""
        decisive = [t for t in trades if t.outcome_label in [OutcomeLabel.WIN, OutcomeLabel.LOSS]]
        if not decisive:
            return 0.0
        losses = sum(1 for t in decisive if t.outcome_label == OutcomeLabel.LOSS)
        return losses / len(decisive)
    
    def evaluate_learning_readiness(self) -> Dict[str, Any]:
        """
        Complete three-level evaluation.
        
        Returns comprehensive readiness report:
        {
            "ready": bool,              # Overall: ready to learn?
            "gate_passed": bool,        # NIVÅ 1 passed?
            "gate_reason": str,         # Why gate passed/failed
            "trigger_fired": bool,      # NIVÅ 2 fired?
            "trigger_type": str,        # "batch" | "time" | "regime" | "none"
            "trigger_reason": str,      # Why trigger fired/not
            "allowed_actions": [],      # NIVÅ 3 authorized actions
            "stats": {...}              # Data statistics
        }
        """
        trades = self._load_trades()
        
        # NIVÅ 1: Gate check (ALL must pass)
        gate_passed, gate_reason = self.gate.evaluate(trades)
        
        if not gate_passed:
            return {
                "ready": False,
                "gate_passed": False,
                "gate_reason": gate_reason,
                "trigger_fired": False,
                "trigger_type": "none",
                "trigger_reason": "Gate not passed",
                "allowed_actions": [],
                "stats": {
                    "total_trades": len(trades),
                    "new_trades": 0,
                    "time_span_days": self._calculate_time_span(trades),
                    "win_rate": self._calculate_win_rate(trades),
                    "loss_rate": self._calculate_loss_rate(trades),
                    "last_training": self.state.get("last_training_timestamp").isoformat() if self.state.get("last_training_timestamp") else "Never",
                    "total_trainings": self.state.get("total_trainings", 0)
                }
            }
        
        # NIVÅ 2: Trigger check (ANY can fire)
        new_trades_count = len(trades) - self.state["last_training_trade_count"]
        triggered, trigger_reason, trigger_type = self.trigger.evaluate(
            new_trades_count=new_trades_count,
            last_training_ts=self.state.get("last_training_timestamp")
        )
        
        # NIVÅ 3: Authorization (what's allowed?)
        time_span_days = self._calculate_time_span(trades)
        allowed_actions = self.auth.get_allowed_actions(
            trade_count=len(trades),
            time_span_days=time_span_days
        )
        
        # Overall readiness
        ready = gate_passed and triggered and len(allowed_actions) > 0
        
        return {
            "ready": ready,
            "gate_passed": gate_passed,
            "gate_reason": gate_reason,
            "trigger_fired": triggered,
            "trigger_type": trigger_type,
            "trigger_reason": trigger_reason,
            "allowed_actions": allowed_actions,
            "stats": {
                "total_trades": len(trades),
                "new_trades": new_trades_count,
                "time_span_days": time_span_days,
                "win_rate": self._calculate_win_rate(trades),
                "loss_rate": self._calculate_loss_rate(trades),
                "last_training": self.state.get("last_training_timestamp").isoformat() if self.state.get("last_training_timestamp") else "Never",
                "total_trainings": self.state.get("total_trainings", 0)
            }
        }
    
    def mark_training_completed(self, action: str):
        """Record that training was completed (updates state)"""
        trades = self._load_trades()
        
        self.state["last_training_timestamp"] = datetime.now(timezone.utc)
        self.state["last_training_trade_count"] = len(trades)
        self.state["last_training_action"] = action
        self.state["total_trainings"] = self.state.get("total_trainings", 0) + 1
        
        self._save_state()
        
        logger.info(f"✅ Training recorded: action={action}, trades={len(trades)}, total_trainings={self.state['total_trainings']}")
