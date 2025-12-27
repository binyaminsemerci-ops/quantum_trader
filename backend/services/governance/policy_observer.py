"""
Policy Observer - Logs orchestrator policy decisions in observation mode.

This module logs what the OrchestratorPolicy WOULD do without affecting live trading.
All policy decisions are recorded for offline analysis and eventual transition to LIVE mode.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from backend.services.governance.orchestrator_policy import TradingPolicy, RiskState, SymbolPerformanceData, CostMetrics

logger = logging.getLogger(__name__)


class PolicyObserver:
    """
    Observes and logs OrchestratorPolicy decisions without enforcement.
    
    In OBSERVATION MODE:
    - Records policy updates
    - Logs what policy WOULD do vs what actually happens
    - Tracks policy effectiveness metrics
    - No interference with live trading
    """
    
    def __init__(self, log_dir: str = "data/policy_observations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Rotating log file (daily)
        self.current_date = datetime.now(timezone.utc).date()
        self.log_file = self.log_dir / f"policy_obs_{self.current_date}.jsonl"
        
        # In-memory policy history (last 100)
        self.policy_history: List[Dict[str, Any]] = []
        self.max_history = 100
        
        logger.info(f"[OK] PolicyObserver initialized: log_dir={log_dir}")
    
    def _check_rotation(self):
        """Rotate log file if date changed."""
        today = datetime.now(timezone.utc).date()
        if today != self.current_date:
            self.current_date = today
            self.log_file = self.log_dir / f"policy_obs_{self.current_date}.jsonl"
            logger.info(f"📅 Rotated policy log to: {self.log_file}")
    
    def log_policy_update(
        self,
        policy: TradingPolicy,
        regime_tag: str,
        vol_level: str,
        risk_state: RiskState,
        symbol_performance: List[SymbolPerformanceData],
        cost_metrics: CostMetrics,
        signals_before_filter: List[Dict[str, Any]],
        actual_confidence_used: float,
        actual_trading_allowed: bool
    ):
        """
        Log a policy update in observation mode.
        
        Args:
            policy: The policy computed by orchestrator
            regime_tag: Market regime used
            vol_level: Volatility level used
            risk_state: Current risk state
            symbol_performance: Symbol performance data
            cost_metrics: Cost metrics
            signals_before_filter: Trading signals before any filtering
            actual_confidence_used: The confidence threshold actually used
            actual_trading_allowed: Whether trading actually happened
        """
        self._check_rotation()
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Build observation record
        observation = {
            "timestamp": timestamp,
            "mode": "OBSERVE",
            
            # Input context
            "inputs": {
                "regime_tag": regime_tag,
                "vol_level": vol_level,
                "risk_state": asdict(risk_state),
                "symbol_performance_count": len(symbol_performance),
                "symbol_performance": [asdict(sp) for sp in symbol_performance],
                "cost_metrics": asdict(cost_metrics)
            },
            
            # Policy output
            "policy": {
                "allow_new_trades": policy.allow_new_trades,
                "min_confidence": policy.min_confidence,
                "max_risk_pct": policy.max_risk_pct,
                "risk_profile": policy.risk_profile,
                "entry_mode": policy.entry_mode,
                "exit_mode": policy.exit_mode,
                "disallowed_symbols": policy.disallowed_symbols,
                "note": policy.note
            },
            
            # What actually happened
            "actual": {
                "confidence_used": actual_confidence_used,
                "trading_allowed": actual_trading_allowed,
                "signals_received": len(signals_before_filter)
            },
            
            # Policy vs Reality comparison
            "comparison": {
                "would_block_trading": not policy.allow_new_trades and actual_trading_allowed,
                "would_raise_confidence": policy.min_confidence > actual_confidence_used,
                "confidence_delta": policy.min_confidence - actual_confidence_used,
                "blocked_symbols_count": len(policy.disallowed_symbols)
            }
        }
        
        # Add to memory
        self.policy_history.append(observation)
        if len(self.policy_history) > self.max_history:
            self.policy_history.pop(0)
        
        # Write to disk (JSONL format for streaming analysis)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(observation) + "\n")
        except Exception as e:
            logger.error(f"Failed to write policy observation: {e}")
        
        # Log summary to console
        logger.info(
            f"[CHART] POLICY OBSERVATION | "
            f"Allow={policy.allow_new_trades} | "
            f"MinConf={policy.min_confidence:.2f} (vs {actual_confidence_used:.2f}) | "
            f"Risk={policy.max_risk_pct:.1f}% | "
            f"Profile={policy.risk_profile} | "
            f"Blocked={len(policy.disallowed_symbols)} symbols | "
            f"Note: {policy.note}"
        )
    
    def log_signal_decision(
        self,
        signal: Dict[str, Any],
        policy: TradingPolicy,
        decision: str,  # "ALLOWED", "BLOCKED_BY_POLICY", "BLOCKED_BY_CONFIDENCE", etc.
        reason: str
    ):
        """
        Log individual signal decisions in observation mode.
        
        Args:
            signal: The trading signal
            policy: Current policy
            decision: What happened ("ALLOWED", "BLOCKED_BY_POLICY", etc.)
            reason: Why this decision was made
        """
        symbol = signal.get("symbol", "UNKNOWN")
        confidence = signal.get("confidence", 0.0)
        action = signal.get("action", "HOLD")
        
        # What would policy do?
        would_block_symbol = symbol in policy.disallowed_symbols
        would_block_confidence = confidence < policy.min_confidence
        would_block_trading = not policy.allow_new_trades
        
        policy_verdict = "ALLOW"
        if would_block_trading:
            policy_verdict = "BLOCK_TRADING_GATE"
        elif would_block_symbol:
            policy_verdict = "BLOCK_SYMBOL"
        elif would_block_confidence:
            policy_verdict = "BLOCK_CONFIDENCE"
        
        observation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "signal_decision",
            "signal": {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "model": signal.get("model", "unknown")
            },
            "actual_decision": decision,
            "actual_reason": reason,
            "policy_verdict": policy_verdict,
            "agreement": (decision == "ALLOWED") == (policy_verdict == "ALLOW")
        }
        
        # Write to separate signal log for detailed analysis
        signal_log = self.log_dir / f"signals_{self.current_date}.jsonl"
        try:
            with open(signal_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(observation) + "\n")
        except Exception as e:
            logger.error(f"Failed to write signal observation: {e}")
    
    def get_recent_policies(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the N most recent policy observations."""
        return self.policy_history[-count:]
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get summary statistics from recent observations."""
        if not self.policy_history:
            return {"error": "No observations yet"}
        
        recent = self.policy_history[-20:]  # Last 20 observations
        
        blocked_count = sum(1 for obs in recent if not obs["policy"]["allow_new_trades"])
        avg_min_conf = sum(obs["policy"]["min_confidence"] for obs in recent) / len(recent)
        avg_risk = sum(obs["policy"]["max_risk_pct"] for obs in recent) / len(recent)
        
        return {
            "total_observations": len(self.policy_history),
            "recent_sample_size": len(recent),
            "blocked_trading_pct": (blocked_count / len(recent)) * 100,
            "avg_min_confidence": avg_min_conf,
            "avg_risk_per_trade": avg_risk,
            "log_file": str(self.log_file)
        }
