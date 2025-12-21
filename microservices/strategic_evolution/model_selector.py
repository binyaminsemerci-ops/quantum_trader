import json
import logging

class ModelSelector:
    """Velger beste strategier for videre evolusjon"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.top_n = 3  # Select top 3 models

    def select_best(self, ranked):
        """
        Selects top N performing models for mutation and retraining
        """
        try:
            if not ranked:
                logging.warning(json.dumps({
                    "event": "No ranked strategies to select from",
                    "level": "warning"
                }))
                return []
            
            # Select top N strategies
            top = ranked[:self.top_n]
            
            # Store selection in Redis
            selection = {
                "models": [r["name"] for r in top],
                "scores": [r["score"] for r in top],
                "timestamp": top[0]["timestamp"] if top else None
            }
            
            self.redis.set("quantum:evolution:selected", json.dumps(selection))
            
            logging.info(json.dumps({
                "event": "Model selection complete",
                "selected_models": [r["name"] for r in top],
                "scores": [r["score"] for r in top],
                "level": "info"
            }))
            
            return top
            
        except Exception as e:
            logging.error(json.dumps({
                "event": "Model selection failed",
                "error": str(e),
                "level": "error"
            }))
            return []
