"""
Reinforcement Feedback - Provides meta-signals to AI Engine
"""
import redis
import json
from typing import Dict, Any
import structlog

logger = structlog.get_logger()


class ReinforcementFeedback:
    """Provides reinforcement signals to AI Engine based on pattern analysis"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis = redis.from_url(redis_url, decode_responses=True)
        logger.info("ReinforcementFeedback initialized", redis_url=redis_url)
    
    def push_feedback(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate and push feedback to AI Engine
        
        Args:
            analysis: Pattern analysis results
        
        Returns:
            Feedback payload that was pushed to Redis
        """
        try:
            # Get best regime and its performance
            best_regime_name, best_pnl = analysis.get("best_regime", ("UNKNOWN", 0.0))
            
            if best_regime_name == "UNKNOWN" or analysis.get("total_samples", 0) < 3:
                logger.warning(
                    "Insufficient data for feedback",
                    samples=analysis.get("total_samples", 0)
                )
                return self._empty_feedback()
            
            # Get regime statistics
            regimes = analysis.get("regimes", {})
            best_regime_stats = regimes.get(best_regime_name, {})
            
            # Determine recommended policy based on regime
            recommended_policy = self._recommend_policy(best_regime_name, best_regime_stats)
            
            # Calculate confidence boost (normalized PnL with safety bounds)
            confidence_boost = self._calculate_confidence(best_pnl, best_regime_stats)
            
            # Determine leverage adjustment
            leverage_hint = self._calculate_leverage_hint(best_regime_name, best_regime_stats)
            
            # Build feedback payload
            payload = {
                "preferred_regime": best_regime_name,
                "updated_policy": recommended_policy,
                "confidence_boost": confidence_boost,
                "leverage_hint": leverage_hint,
                "regime_performance": {
                    "avg_pnl": round(best_pnl, 4),
                    "win_rate": best_regime_stats.get("win_rate", 0.0),
                    "sample_count": best_regime_stats.get("count", 0)
                },
                "timestamp": analysis.get("timestamp"),
                "version": "1.0.0"
            }
            
            # Push to Redis
            self.redis.set("quantum:feedback:strategic_memory", json.dumps(payload))
            
            # Also publish event to event bus
            self.redis.publish("quantum:events:strategic_feedback", json.dumps({
                "event": "strategic_feedback_updated",
                "regime": best_regime_name,
                "policy": recommended_policy,
                "confidence": confidence_boost
            }))
            
            logger.info(
                "Feedback pushed to Redis",
                regime=best_regime_name,
                policy=recommended_policy,
                confidence=confidence_boost,
                leverage=leverage_hint
            )
            
            return payload
            
        except Exception as e:
            logger.error("Failed to push feedback", error=str(e))
            return self._empty_feedback()
    
    def _recommend_policy(self, regime: str, stats: Dict[str, Any]) -> str:
        """
        Recommend policy based on regime and performance
        
        Policy mapping:
        - BULL + good performance → AGGRESSIVE
        - BULL + weak performance → BALANCED
        - BEAR → CONSERVATIVE
        - VOLATILE → CONSERVATIVE
        - RANGE → BALANCED
        - UNCERTAIN → CONSERVATIVE
        """
        win_rate = stats.get("win_rate", 0.0)
        avg_pnl = stats.get("avg_pnl", 0.0)
        
        if regime == "BULL":
            if win_rate > 0.6 and avg_pnl > 0.2:
                return "AGGRESSIVE"
            elif win_rate > 0.5 and avg_pnl > 0.1:
                return "BALANCED"
            else:
                return "CONSERVATIVE"
        
        elif regime == "BEAR":
            return "CONSERVATIVE"
        
        elif regime == "VOLATILE":
            return "CONSERVATIVE"
        
        elif regime == "RANGE":
            if win_rate > 0.55:
                return "BALANCED"
            else:
                return "CONSERVATIVE"
        
        else:  # UNCERTAIN or UNKNOWN
            return "CONSERVATIVE"
    
    def _calculate_confidence(self, pnl: float, stats: Dict[str, Any]) -> float:
        """
        Calculate confidence boost based on PnL and win rate
        
        Returns value between 0.0 and 1.0
        """
        # Base confidence from PnL (capped at +/- 1.0)
        pnl_confidence = max(-1.0, min(1.0, pnl))
        
        # Adjust by win rate
        win_rate = stats.get("win_rate", 0.0)
        win_rate_bonus = (win_rate - 0.5) * 2  # Scale from -1.0 to +1.0
        
        # Adjust by sample count (more samples = more confidence)
        sample_count = stats.get("count", 0)
        sample_factor = min(1.0, sample_count / 20.0)  # Full confidence at 20+ samples
        
        # Combine factors
        confidence = (pnl_confidence * 0.6 + win_rate_bonus * 0.4) * sample_factor
        
        # Ensure positive and bounded
        return round(max(0.0, min(1.0, (confidence + 1.0) / 2.0)), 4)
    
    def _calculate_leverage_hint(self, regime: str, stats: Dict[str, Any]) -> float:
        """
        Calculate leverage adjustment hint
        
        Returns:
            Multiplier between 0.5 and 2.0
            - < 1.0: Reduce leverage
            - = 1.0: Keep current
            - > 1.0: Increase leverage (up to 2x)
        """
        win_rate = stats.get("win_rate", 0.0)
        avg_pnl = stats.get("avg_pnl", 0.0)
        
        # Base leverage by regime
        if regime == "BULL" and win_rate > 0.6 and avg_pnl > 0.2:
            base = 1.5
        elif regime == "BULL" and win_rate > 0.5:
            base = 1.2
        elif regime == "RANGE" and win_rate > 0.55:
            base = 1.0
        elif regime in ["BEAR", "VOLATILE", "UNCERTAIN"]:
            base = 0.7
        else:
            base = 1.0
        
        # Adjust by confidence
        confidence = stats.get("confidence", 0.0)
        confidence_factor = 0.8 + (confidence * 0.4)  # 0.8 to 1.2
        
        leverage = base * confidence_factor
        
        # Bounds: 0.5x to 2.0x
        return round(max(0.5, min(2.0, leverage)), 2)
    
    def _empty_feedback(self) -> Dict[str, Any]:
        """Return empty feedback structure"""
        from datetime import datetime
        return {
            "preferred_regime": "UNKNOWN",
            "updated_policy": "CONSERVATIVE",
            "confidence_boost": 0.0,
            "leverage_hint": 1.0,
            "regime_performance": {
                "avg_pnl": 0.0,
                "win_rate": 0.0,
                "sample_count": 0
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": "1.0.0"
        }
