"""
Consensus Calculator - Builds collective decision based on model weights and trust.
"""
import json


class ConsensusCalculator:
    """Builds weighted consensus from multiple model signals."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
    def build_consensus(self, signals, trust):
        """
        Calculate weighted consensus from model signals.
        
        Args:
            signals: List of model signals with action and confidence
            trust: TrustMemory instance for retrieving model weights
            
        Returns:
            dict: Consensus decision with action, confidence, and metadata
        """
        if not signals:
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "reason": "no_signals",
                "models_used": 0,
                "trust_weights": {}
            }
        
        # Aggregate weighted votes
        weighted_votes = {}
        total_weight = 0
        trust_weights = {}
        
        for signal in signals:
            model = signal["model"]
            action = signal["action"].upper()
            confidence = signal.get("confidence", 0.5)
            
            # Get trust weight for this model
            weight = trust.get_weight(model)
            trust_weights[model] = weight
            
            # Weight the vote by both trust and signal confidence
            vote_strength = weight * confidence
            
            weighted_votes[action] = weighted_votes.get(action, 0) + vote_strength
            total_weight += weight
        
        if total_weight == 0:
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "reason": "zero_weight",
                "models_used": len(signals),
                "trust_weights": trust_weights
            }
        
        # Determine final action (highest weighted vote)
        final_action = max(weighted_votes, key=weighted_votes.get)
        
        # Normalize confidence
        confidence = round(weighted_votes[final_action] / total_weight, 3)
        
        # Calculate agreement percentage
        agreement = sum(1 for s in signals if s["action"].upper() == final_action)
        agreement_pct = round(agreement / len(signals), 3)
        
        return {
            "action": final_action,
            "confidence": confidence,
            "models_used": len(signals),
            "agreement_pct": agreement_pct,
            "trust_weights": trust_weights,
            "vote_distribution": weighted_votes,
            "reason": "consensus"
        }
