"""
Model Broker - Collects predictions from all models in the ensemble.
"""
import json


class ModelBroker:
    """Fetches predictions from all active models."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        # List of all models that can provide signals
        self.models = [
            "xgb",
            "lgbm", 
            "nhits",
            "patchtst",
            "rl_sizer",
            "evo_model",
            "tft",
            "lstm",
            "transformer"
        ]
        
    def collect_signals(self):
        """
        Collect prediction signals from all active models.
        
        Returns:
            list: List of signal dictionaries with model name, action, confidence
        """
        signals = []
        
        for model in self.models:
            # Try to get signal from Redis
            signal_key = f"quantum:model:{model}:signal"
            data = self.redis.get(signal_key)
            
            if not data:
                continue
                
            try:
                signal = json.loads(data)
                
                # Validate signal structure
                if "action" not in signal or "confidence" not in signal:
                    continue
                
                # Add model identifier
                signal["model"] = model
                signals.append(signal)
                
            except (json.JSONDecodeError, TypeError):
                # Skip malformed signals
                continue
        
        return signals
    
    def get_active_models(self):
        """Get list of models that have recent signals."""
        active = []
        for model in self.models:
            if self.redis.exists(f"quantum:model:{model}:signal"):
                active.append(model)
        return active
