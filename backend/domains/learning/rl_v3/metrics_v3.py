"""
RL v3 Metrics Store - In-memory metrics tracking for RL v3 decisions.

Thread-safe singleton for storing and retrieving RL v3 metrics.
Used for dashboard visualization and performance monitoring.

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0
"""

import threading
from collections import deque
from typing import Deque, Dict, Any, List, Optional
from datetime import datetime, timezone


class RLv3MetricsStore:
    """
    Singleton in-memory store for RL v3 decision metrics.
    
    Thread-safe storage of recent decisions with circular buffer.
    Provides summary statistics for dashboard and monitoring.
    """
    
    _instance: Optional["RLv3MetricsStore"] = None
    _lock = threading.Lock()
    
    def __init__(self, maxlen: int = 200):
        """
        Initialize metrics store.
        
        Args:
            maxlen: Maximum number of decisions to store (circular buffer)
        """
        self._decisions: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._training_runs: Deque[Dict[str, Any]] = deque(maxlen=50)
        self._live_decisions: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._trade_intents: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._lock = threading.Lock()
        self._total_decisions = 0
        self._action_counts = {i: 0 for i in range(6)}
        self._total_live_decisions = 0
        self._total_trade_intents = 0
    
    @classmethod
    def instance(cls) -> "RLv3MetricsStore":
        """
        Get singleton instance of metrics store.
        
        Returns:
            RLv3MetricsStore singleton
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def record_decision(self, payload: Dict[str, Any]) -> None:
        """
        Record a decision from RL v3 subscriber.
        
        Args:
            payload: Decision payload from subscriber
        """
        with self._lock:
            # Store decision with timestamp
            decision = {
                "symbol": payload.get("symbol", "UNKNOWN"),
                "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "action": payload.get("action", 0),
                "confidence": payload.get("confidence", 0.0),
                "value": payload.get("value", 0.0),
                "source": payload.get("source", "RL_V3_PPO"),
                "shadow_mode": payload.get("shadow_mode", True),
                "trace_id": payload.get("trace_id", ""),
                "raw_obs": payload.get("raw_obs", {})
            }
            
            self._decisions.append(decision)
            self._total_decisions += 1
            
            # Update action counts
            action = payload.get("action", 0)
            if action in self._action_counts:
                self._action_counts[action] += 1
    
    def get_recent_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get most recent decisions.
        
        Args:
            limit: Maximum number of decisions to return
            
        Returns:
            List of recent decisions (most recent first)
        """
        with self._lock:
            # Return copy of recent decisions (reversed for most recent first)
            decisions = list(self._decisions)
            decisions.reverse()
            return decisions[:limit]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of RL v3 decisions.
        
        Returns:
            Summary dict with counts, distributions, and averages
        """
        with self._lock:
            if not self._decisions:
                return {
                    "total_decisions": 0,
                    "buffer_size": 0,
                    "action_counts": {i: 0 for i in range(6)},
                    "action_distribution": {i: 0.0 for i in range(6)},
                    "avg_confidence": None,
                    "max_confidence": None,
                    "min_confidence": None,
                    "avg_value": 0.0
                }
            
            # Calculate statistics
            confidences = [d["confidence"] for d in self._decisions]
            values = [d["value"] for d in self._decisions]
            
            # Action distribution percentages
            total = sum(self._action_counts.values())
            action_dist = {
                k: (v / total * 100) if total > 0 else 0.0
                for k, v in self._action_counts.items()
            }
            
            return {
                "total_decisions": self._total_decisions,
                "buffer_size": len(self._decisions),
                "action_counts": self._action_counts.copy(),
                "action_distribution": action_dist,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
                "max_confidence": max(confidences) if confidences else None,
                "min_confidence": min(confidences) if confidences else None,
                "avg_value": sum(values) / len(values) if values else 0.0
            }
    
    def clear(self) -> None:
        """Clear all stored metrics (for testing)."""
        with self._lock:
            self._decisions.clear()
            self._training_runs.clear()
            self._live_decisions.clear()
            self._trade_intents.clear()
            self._total_decisions = 0
            self._action_counts = {i: 0 for i in range(6)}
            self._total_live_decisions = 0
            self._total_trade_intents = 0
    
    def record_training_run(self, run: Dict[str, Any]) -> None:
        """
        Record a training run.
        
        Args:
            run: Training run data with keys: timestamp, episodes, duration_seconds, 
                 success, error, avg_reward, final_reward, etc.
        """
        with self._lock:
            self._training_runs.append(run)
    
    def get_recent_training_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get most recent training runs.
        
        Args:
            limit: Maximum number of runs to return
            
        Returns:
            List of recent training runs (most recent first)
        """
        with self._lock:
            runs = list(self._training_runs)
            runs.reverse()
            return runs[:limit]
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of training runs.
        
        Returns:
            Summary dict with total_runs, success_count, failure_count, 
            last_run_at, last_error
        """
        with self._lock:
            if not self._training_runs:
                return {
                    "total_runs": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "last_run_at": None,
                    "last_error": None,
                    "avg_duration_seconds": None,
                    "avg_reward": None
                }
            
            total = len(self._training_runs)
            success_count = sum(1 for r in self._training_runs if r.get("success", False))
            failure_count = total - success_count
            
            # Get last run
            last_run = self._training_runs[-1]
            last_run_at = last_run.get("timestamp")
            last_error = last_run.get("error") if not last_run.get("success") else None
            
            # Calculate averages
            durations = [r.get("duration_seconds", 0) for r in self._training_runs]
            avg_duration = sum(durations) / len(durations) if durations else None
            
            successful_runs = [r for r in self._training_runs if r.get("success", False)]
            rewards = [r.get("avg_reward", 0) for r in successful_runs]
            avg_reward = sum(rewards) / len(rewards) if rewards else None
            
            return {
                "total_runs": total,
                "success_count": success_count,
                "failure_count": failure_count,
                "last_run_at": last_run_at,
                "last_error": last_error,
                "avg_duration_seconds": avg_duration,
                "avg_reward": avg_reward
            }
    
    def record_live_decision(self, payload: Dict[str, Any]) -> None:
        """
        Record a live decision from orchestrator.
        
        Args:
            payload: Live decision payload with mode, action, confidence, etc.
        """
        with self._lock:
            decision = {
                "symbol": payload.get("symbol", "UNKNOWN"),
                "mode": payload.get("mode", "SHADOW"),
                "action": payload.get("action", "HOLD"),
                "action_idx": payload.get("action_idx", 5),
                "confidence": payload.get("confidence", 0.0),
                "value": payload.get("value", 0.0),
                "signal_action": payload.get("signal_action", "HOLD"),
                "signal_confidence": payload.get("signal_confidence", 0.0),
                "published_trade_intent": payload.get("published_trade_intent", False),
                "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "trace_id": payload.get("trace_id", ""),
            }
            self._live_decisions.append(decision)
            self._total_live_decisions += 1
    
    def record_trade_intent(self, payload: Dict[str, Any]) -> None:
        """
        Record a published trade intent.
        
        Args:
            payload: Trade intent payload with symbol, side, source, etc.
        """
        with self._lock:
            intent = {
                "symbol": payload.get("symbol", "UNKNOWN"),
                "side": payload.get("side", "HOLD"),
                "source": payload.get("source", "RL_V3"),
                "confidence": payload.get("confidence", 0.0),
                "size_pct": payload.get("size_pct", 0.0),
                "executed": payload.get("executed", False),
                "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "trace_id": payload.get("trace_id", ""),
            }
            self._trade_intents.append(intent)
            self._total_trade_intents += 1
    
    def get_live_status(self) -> Dict[str, Any]:
        """
        Get live orchestrator status.
        
        Returns:
            Status dict with total counts, shadow decisions, trade intents, timestamps
        """
        with self._lock:
            shadow_count = sum(1 for d in self._live_decisions if d.get("mode") == "SHADOW")
            executed_count = sum(1 for i in self._trade_intents if i.get("executed"))
            
            last_decision_at = None
            if self._live_decisions:
                last_decision_at = self._live_decisions[-1].get("timestamp")
            
            last_trade_intent_at = None
            if self._trade_intents:
                last_trade_intent_at = self._trade_intents[-1].get("timestamp")
            
            return {
                "total_live_decisions": self._total_live_decisions,
                "shadow_decisions": shadow_count,
                "trade_intents_total": self._total_trade_intents,
                "trade_intents_executed": executed_count,
                "last_decision_at": last_decision_at,
                "last_trade_intent_at": last_trade_intent_at,
            }
    
    def get_recent_live_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get most recent live decisions.
        
        Args:
            limit: Maximum number of decisions to return
            
        Returns:
            List of recent live decisions (most recent first)
        """
        with self._lock:
            decisions = list(self._live_decisions)
            decisions.reverse()
            return decisions[:limit]
    
    def get_recent_trade_intents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get most recent trade intents.
        
        Args:
            limit: Maximum number of intents to return
            
        Returns:
            List of recent trade intents (most recent first)
        """
        with self._lock:
            intents = list(self._trade_intents)
            intents.reverse()
            return intents[:limit]
