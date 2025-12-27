"""
Portfolio Governance Controller
================================
AI-driven policy controller that adjusts risk parameters based on portfolio performance.

This controller:
- Monitors portfolio score and exposure memory
- Adjusts risk policy dynamically (CONSERVATIVE/BALANCED/AGGRESSIVE)
- Provides feedback signals to ExitBrain v3.5 and RL Sizing Agent
- Enforces position limits and risk thresholds
- Learns optimal exposure levels for different market regimes
"""

from exposure_memory import ExposureMemory
import json
import redis
import time
import logging
from typing import Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioGovernanceAgent:
    """
    Main governance agent that controls portfolio risk policy.
    
    Policy Modes:
    - CONSERVATIVE: Low risk, strict limits, high confidence required
    - BALANCED: Moderate risk, standard limits
    - AGGRESSIVE: Higher risk, relaxed limits, lower confidence threshold
    
    The agent continuously monitors portfolio performance and adjusts
    policy to optimize risk-adjusted returns.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize Portfolio Governance Agent.
        
        Args:
            redis_url: Redis connection URL
        """
        self.memory = ExposureMemory(redis_url=redis_url)
        self.redis = redis.from_url(redis_url, decode_responses=True)
        
        # Policy thresholds
        self.thresholds = {
            "max_risk": 0.85,
            "target_score": 0.7,
            "conservative_threshold": 0.3,
            "aggressive_threshold": 0.7,
            "min_samples": 50
        }
        
        # Risk parameters per policy
        self.policy_params = {
            "CONSERVATIVE": {
                "max_leverage": 10,
                "max_position_pct": 0.15,
                "min_confidence": 0.75,
                "max_concurrent_positions": 3,
                "stop_loss_multiplier": 1.5
            },
            "BALANCED": {
                "max_leverage": 20,
                "max_position_pct": 0.25,
                "min_confidence": 0.65,
                "max_concurrent_positions": 5,
                "stop_loss_multiplier": 1.0
            },
            "AGGRESSIVE": {
                "max_leverage": 30,
                "max_position_pct": 0.35,
                "min_confidence": 0.55,
                "max_concurrent_positions": 7,
                "stop_loss_multiplier": 0.8
            }
        }
        
        # Initialize with BALANCED policy
        self._initialize_policy()
        
        logger.info("Portfolio Governance Agent initialized")
    
    def _initialize_policy(self) -> None:
        """Initialize policy in Redis if not exists."""
        if not self.redis.exists("quantum:governance:policy"):
            self.redis.set("quantum:governance:policy", "BALANCED")
            logger.info("Initialized policy to BALANCED")
    
    def adjust_policy(self) -> Dict:
        """
        Adjust portfolio policy based on current performance.
        
        Decision Logic:
        - Score < 0.3: CONSERVATIVE (poor performance, reduce risk)
        - Score 0.3-0.7: BALANCED (moderate performance)
        - Score > 0.7: AGGRESSIVE (excellent performance, increase exposure)
        
        Returns:
            Dictionary with policy decision details
        """
        summary = self.memory.summarize()
        score = self.memory.get_portfolio_score()
        
        # Don't adjust if insufficient data
        if not summary or summary.get("samples", 0) < self.thresholds["min_samples"]:
            current_policy = self.redis.get("quantum:governance:policy") or "BALANCED"
            logger.info(f"Insufficient samples ({summary.get('samples', 0)}), maintaining current policy: {current_policy}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "policy": current_policy,
                "previous_policy": current_policy,
                "policy_changed": False,
                "score": 0.5,
                "summary": summary or {},
                "action": "maintain",
                "reason": "insufficient_data"
            }
        
        # Determine new policy based on score
        current_policy = self.redis.get("quantum:governance:policy") or "BALANCED"
        
        if score < self.thresholds["conservative_threshold"]:
            new_policy = "CONSERVATIVE"
            reason = f"Low score ({score}), reducing risk"
        elif score < self.thresholds["aggressive_threshold"]:
            new_policy = "BALANCED"
            reason = f"Moderate score ({score}), balanced approach"
        else:
            new_policy = "AGGRESSIVE"
            reason = f"High score ({score}), increasing exposure"
        
        # Additional safety checks
        win_rate = summary.get("win_rate", 0)
        if win_rate < 0.4 and new_policy == "AGGRESSIVE":
            new_policy = "BALANCED"
            reason = f"Win rate too low ({win_rate}), staying BALANCED"
        
        # Update policy if changed
        policy_changed = new_policy != current_policy
        if policy_changed:
            self.redis.set("quantum:governance:policy", new_policy)
            self._publish_policy_change(current_policy, new_policy, reason)
            logger.info(f"Policy changed: {current_policy} → {new_policy} ({reason})")
        
        # Store policy parameters in Redis for other services
        self._update_policy_parameters(new_policy)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "policy": new_policy,
            "previous_policy": current_policy,
            "policy_changed": policy_changed,
            "score": score,
            "summary": summary,
            "reason": reason
        }
    
    def _update_policy_parameters(self, policy: str) -> None:
        """
        Update policy parameters in Redis for consumption by other services.
        
        Args:
            policy: Current policy (CONSERVATIVE/BALANCED/AGGRESSIVE)
        """
        params = self.policy_params.get(policy, self.policy_params["BALANCED"])
        
        # Store each parameter in Redis
        for key, value in params.items():
            redis_key = f"quantum:governance:param:{key}"
            self.redis.set(redis_key, str(value))
        
        # Store all params as JSON for easy retrieval
        self.redis.set("quantum:governance:params", json.dumps(params))
    
    def _publish_policy_change(self, old_policy: str, new_policy: str, reason: str) -> None:
        """
        Publish policy change event to Redis Pub/Sub.
        
        Args:
            old_policy: Previous policy
            new_policy: New policy
            reason: Reason for change
        """
        event = {
            "event": "policy_changed",
            "timestamp": datetime.utcnow().isoformat(),
            "old_policy": old_policy,
            "new_policy": new_policy,
            "reason": reason
        }
        
        self.redis.publish("quantum:events:governance", json.dumps(event))
        
        # Also store in stream for historical tracking
        self.redis.xadd("quantum:stream:governance.events", event)
    
    def get_current_policy(self) -> Dict:
        """
        Get current policy and parameters.
        
        Returns:
            Dictionary with policy details
        """
        policy = self.redis.get("quantum:governance:policy") or "BALANCED"
        params_json = self.redis.get("quantum:governance:params")
        
        if params_json:
            params = json.loads(params_json)
        else:
            params = self.policy_params.get(policy, {})
        
        score = self.memory.get_portfolio_score()
        summary = self.memory.summarize()
        
        return {
            "policy": policy,
            "parameters": params,
            "score": score,
            "samples": summary.get("samples", 0),
            "health": self.memory.get_memory_health()
        }
    
    def should_allow_trade(self, symbol: str, leverage: int, confidence: float) -> Dict:
        """
        Check if a trade should be allowed based on current policy.
        
        Args:
            symbol: Trading symbol
            leverage: Requested leverage
            confidence: AI confidence score
            
        Returns:
            Dictionary with decision and reason
        """
        policy = self.redis.get("quantum:governance:policy") or "BALANCED"
        params = self.policy_params.get(policy, self.policy_params["BALANCED"])
        
        # Check leverage limit
        if leverage > params["max_leverage"]:
            return {
                "allowed": False,
                "reason": f"Leverage {leverage} exceeds limit {params['max_leverage']}",
                "policy": policy
            }
        
        # Check confidence threshold
        if confidence < params["min_confidence"]:
            return {
                "allowed": False,
                "reason": f"Confidence {confidence} below minimum {params['min_confidence']}",
                "policy": policy
            }
        
        # Get symbol-specific stats
        symbol_stats = self.memory.get_symbol_stats(symbol)
        if symbol_stats.get("samples", 0) > 10:
            # If symbol has poor historical performance, be cautious
            if symbol_stats.get("win_rate", 0) < 0.3:
                return {
                    "allowed": False,
                    "reason": f"Symbol {symbol} has poor win rate: {symbol_stats['win_rate']}",
                    "policy": policy
                }
        
        return {
            "allowed": True,
            "reason": "Trade approved",
            "policy": policy,
            "parameters": params
        }
    
    def get_recommended_position_size(self, base_size: float, confidence: float) -> float:
        """
        Calculate recommended position size based on policy and confidence.
        
        Args:
            base_size: Base position size
            confidence: AI confidence score
            
        Returns:
            Adjusted position size
        """
        policy = self.redis.get("quantum:governance:policy") or "BALANCED"
        params = self.policy_params.get(policy, self.policy_params["BALANCED"])
        
        # Adjust size based on confidence
        confidence_multiplier = min(confidence / 0.7, 1.5)  # Max 1.5x at high confidence
        
        # Apply max position percentage limit
        max_size = base_size * params["max_position_pct"]
        recommended_size = min(base_size * confidence_multiplier, max_size)
        
        return round(recommended_size, 2)
    
    def run(self, interval: int = 30) -> None:
        """
        Run governance agent in continuous loop.
        
        Args:
            interval: Seconds between policy adjustments
        """
        logger.info(f"Portfolio Governance Agent running (interval={interval}s)")
        
        while True:
            try:
                decision = self.adjust_policy()
                
                # Log summary
                logger.info(
                    f"[Governance] Policy={decision['policy']}, "
                    f"Score={decision['score']}, "
                    f"Samples={decision['summary'].get('samples', 0)}"
                )
                
                # Store latest decision
                self.redis.set("quantum:governance:last_decision", json.dumps(decision))
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in governance loop: {e}", exc_info=True)
                time.sleep(interval)
    
    def get_dashboard_data(self) -> Dict:
        """
        Get comprehensive data for governance dashboard.
        
        Returns:
            Dictionary with all governance metrics
        """
        policy_info = self.get_current_policy()
        summary = self.memory.summarize()
        recent_events = self.memory.get_recent_events(limit=10)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "policy": policy_info,
            "summary": summary,
            "recent_events": recent_events,
            "thresholds": self.thresholds
        }
