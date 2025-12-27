"""
Meta-Regime Correlator - Links portfolio results to market regimes.
"""
import redis
import json
from typing import Dict, Optional
import structlog
from regime_memory import RegimeMemory

logger = structlog.get_logger()


class MetaRegimeCorrelator:
    """Links portfolio performance to market regimes"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize correlator.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis = redis.from_url(redis_url)
        self.memory = RegimeMemory(redis_url=redis_url)
        
        # Policy mappings
        self.regime_to_policy = {
            "BULL": "AGGRESSIVE",
            "RANGE": "BALANCED",
            "BEAR": "CONSERVATIVE",
            "VOLATILE": "CONSERVATIVE",
            "UNCERTAIN": "BALANCED"
        }
        
        logger.info("MetaRegimeCorrelator initialized")
    
    def correlate(self) -> Optional[Dict]:
        """
        Analyze regime correlation and update governance.
        
        Returns:
            Dictionary with correlation results or None
        """
        summary = self.memory.summarize()
        if not summary:
            logger.warning("No regime data available for correlation")
            return None
        
        # Find best performing regime
        best = self.memory.get_best_regime()
        if not best:
            logger.warning("Insufficient regime samples for correlation")
            return None
        
        regime_name, stats = best
        
        # Store preferred regime in Redis
        self.redis.set("quantum:governance:preferred_regime", regime_name)
        
        # Store regime statistics
        self.redis.set(
            "quantum:governance:regime_stats",
            json.dumps(summary)
        )
        
        logger.info(
            "Meta-Regime correlation complete",
            best_regime=regime_name,
            avg_pnl=stats["avg_pnl"],
            samples=stats["count"],
            win_rate=stats["win_rate"]
        )
        
        return {
            "best_regime": regime_name,
            "statistics": stats,
            "all_regimes": summary
        }
    
    def suggest_policy(self, current_regime: str) -> str:
        """
        Suggest governance policy based on current regime.
        
        Args:
            current_regime: Current market regime
            
        Returns:
            Suggested policy name
        """
        # Check if we have performance data for this regime
        regime_stats = self.memory.get_regime_stats(current_regime)
        
        if regime_stats and regime_stats.get("count", 0) >= 10:
            # Use historical performance to adjust policy
            avg_pnl = regime_stats.get("avg_pnl", 0)
            win_rate = regime_stats.get("win_rate", 0.5)
            
            # If this regime has been profitable for us, be more aggressive
            if avg_pnl > 0.01 and win_rate > 0.55:
                if current_regime == "BULL":
                    return "AGGRESSIVE"
                else:
                    return "BALANCED"
            
            # If unprofitable, be conservative
            elif avg_pnl < -0.005 or win_rate < 0.45:
                return "CONSERVATIVE"
        
        # Fall back to default mapping
        return self.regime_to_policy.get(current_regime, "BALANCED")
    
    def update_governance_from_regime(self, regime_data: Dict) -> bool:
        """
        Update governance policy based on detected regime.
        
        Args:
            regime_data: Current regime detection results
            
        Returns:
            True if policy was updated
        """
        current_regime = regime_data.get("regime")
        if not current_regime:
            return False
        
        # Get suggested policy
        suggested_policy = self.suggest_policy(current_regime)
        
        # Get current policy
        current_policy = self.redis.get("quantum:governance:policy")
        if current_policy:
            current_policy = current_policy.decode() if isinstance(current_policy, bytes) else current_policy
        else:
            current_policy = "BALANCED"
        
        # Update if different
        if suggested_policy != current_policy:
            confidence = regime_data.get("confidence", 0.5)
            
            # Only update if confidence is high enough
            if confidence > 0.7:
                self.redis.set("quantum:governance:policy", suggested_policy)
                
                # Publish policy change event
                event = {
                    "timestamp": regime_data.get("timestamp"),
                    "old_policy": current_policy,
                    "new_policy": suggested_policy,
                    "regime": current_regime,
                    "confidence": confidence,
                    "reason": "regime_change"
                }
                
                self.redis.publish("quantum:events:policy_change", json.dumps(event))
                
                logger.info(
                    "Policy updated from regime",
                    regime=current_regime,
                    old_policy=current_policy,
                    new_policy=suggested_policy,
                    confidence=confidence
                )
                
                return True
        
        return False
    
    def get_regime_recommendations(self) -> Dict:
        """
        Get recommendations based on regime analysis.
        
        Returns:
            Dictionary with recommendations
        """
        summary = self.memory.summarize()
        best = self.memory.get_best_regime()
        worst = self.memory.get_worst_regime()
        
        recommendations = {
            "summary": summary,
            "best_regime": best[0] if best else None,
            "worst_regime": worst[0] if worst else None,
            "recommendations": []
        }
        
        if best:
            regime_name, stats = best
            recommendations["recommendations"].append({
                "type": "opportunity",
                "message": f"Best performance in {regime_name} regime (avg PnL: {stats['avg_pnl']:.3f})",
                "suggested_policy": self.regime_to_policy.get(regime_name, "BALANCED")
            })
        
        if worst:
            regime_name, stats = worst
            recommendations["recommendations"].append({
                "type": "warning",
                "message": f"Worst performance in {regime_name} regime (avg PnL: {stats['avg_pnl']:.3f})",
                "suggested_action": "Reduce exposure or avoid trading in this regime"
            })
        
        return recommendations
