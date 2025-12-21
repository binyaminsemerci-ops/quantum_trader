import random
import json
import logging
from datetime import datetime

class MutationEngine:
    """Muterer hyperparametere for evolusjonært eksperiment"""
    
    def __init__(self, redis_client):
        self.redis = redis_client

    def mutate(self, models):
        """
        Creates mutated configurations of top models
        Explores hyperparameter space for better performance
        """
        try:
            if not models:
                logging.warning(json.dumps({
                    "event": "No models to mutate",
                    "level": "warning"
                }))
                return []
            
            mutated = []
            
            for m in models:
                mutation = {
                    "name": m["name"],
                    "original_score": m["score"],
                    "learning_rate": round(random.uniform(0.0005, 0.01), 6),
                    "batch_size": random.choice([32, 64, 128]),
                    "optimizer": random.choice(["adam", "rmsprop", "sgd"]),
                    "dropout_rate": round(random.uniform(0.1, 0.5), 3),
                    "hidden_units": random.choice([64, 128, 256, 512]),
                    "mutation_strength": round(random.uniform(0.8, 1.2), 3),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                mutated.append(mutation)
            
            # Store mutations in Redis
            self.redis.set("quantum:evolution:mutated", json.dumps(mutated))
            
            logging.info(json.dumps({
                "event": "Mutation complete",
                "mutated_configs": len(mutated),
                "models": [m["name"] for m in mutated],
                "level": "info"
            }))
            
            return mutated
            
        except Exception as e:
            logging.error(json.dumps({
                "event": "Mutation failed",
                "error": str(e),
                "level": "error"
            }))
            return []
