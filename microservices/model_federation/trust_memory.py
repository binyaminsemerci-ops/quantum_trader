"""
Trust Memory - Remembers models' historical accuracy and updates weights dynamically.
"""
import json
import time


class TrustMemory:
    """Manages model trust weights based on historical performance."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.default_weight = 1.0
        self.min_weight = 0.1
        self.max_weight = 2.0
        
    def get_weight(self, model_name):
        """
        Get current trust weight for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            float: Trust weight (0.1 to 2.0)
        """
        trust_key = f"quantum:trust:{model_name}"
        trust = self.redis.get(trust_key)
        
        if trust:
            try:
                return float(trust)
            except ValueError:
                pass
        
        # Initialize with default weight
        self.redis.set(trust_key, self.default_weight)
        return self.default_weight
    
    def update_trust(self, signals, consensus):
        """
        Update trust weights based on agreement with consensus.
        
        Models that agree with consensus get increased trust.
        Models that disagree get decreased trust.
        
        Args:
            signals: List of model signals
            consensus: Final consensus decision
        """
        consensus_action = consensus["action"]
        
        for signal in signals:
            model = signal["model"]
            signal_action = signal["action"].upper()
            
            # Calculate trust adjustment
            if signal_action == consensus_action:
                # Model agreed with consensus - reward
                delta = 0.05
            else:
                # Model disagreed - penalize (but less severely)
                delta = -0.03
            
            # Apply adjustment
            current_weight = self.get_weight(model)
            new_weight = current_weight + delta
            
            # Enforce bounds
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            
            # Store updated weight
            trust_key = f"quantum:trust:{model}"
            self.redis.set(trust_key, new_weight)
            
            # Store in history hash for easy retrieval
            self.redis.hset("quantum:trust:history", model, new_weight)
            
            # Store timestamped event
            event = {
                "model": model,
                "old_weight": current_weight,
                "new_weight": new_weight,
                "delta": delta,
                "action": signal_action,
                "consensus": consensus_action,
                "timestamp": time.time()
            }
            self.redis.lpush(f"quantum:trust:events:{model}", json.dumps(event))
            self.redis.ltrim(f"quantum:trust:events:{model}", 0, 99)  # Keep last 100
    
    def get_all_weights(self):
        """Get trust weights for all models."""
        weights = self.redis.hgetall("quantum:trust:history")
        return {
            k.decode() if isinstance(k, bytes) else k: 
            float(v.decode() if isinstance(v, bytes) else v)
            for k, v in weights.items()
        }
    
    def reset_weights(self):
        """Reset all trust weights to default (use with caution)."""
        weights = self.get_all_weights()
        for model in weights.keys():
            self.redis.set(f"quantum:trust:{model}", self.default_weight)
            self.redis.hset("quantum:trust:history", model, self.default_weight)
