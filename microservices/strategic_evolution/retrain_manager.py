import json
import time
import logging

class RetrainManager:
    """Planlegger retrening av modeller basert på evolusjonsdata"""
    
    def __init__(self, redis_client):
        self.redis = redis_client

    def schedule_retrain(self, mutated_models):
        """
        Schedules retraining jobs for mutated models
        Pushes to Redis stream for CLM to pick up
        """
        try:
            if not mutated_models:
                logging.warning(json.dumps({
                    "event": "No models to retrain",
                    "level": "warning"
                }))
                return
            
            scheduled_count = 0
            
            for m in mutated_models:
                payload = {
                    "model": m["name"],
                    "learning_rate": str(m["learning_rate"]),
                    "batch_size": str(m["batch_size"]),
                    "optimizer": m["optimizer"],
                    "dropout_rate": str(m["dropout_rate"]),
                    "hidden_units": str(m["hidden_units"]),
                    "mutation_strength": str(m["mutation_strength"]),
                    "timestamp": str(time.time()),
                    "source": "strategic_evolution"
                }
                
                # Push to retrain stream
                self.redis.xadd("quantum:stream:model.retrain", payload)
                
                logging.info(json.dumps({
                    "event": "Retrain scheduled",
                    "model": m["name"],
                    "learning_rate": m["learning_rate"],
                    "optimizer": m["optimizer"],
                    "level": "info"
                }))
                
                scheduled_count += 1
            
            # Update retrain counter
            self.redis.incr("quantum:evolution:retrain_count", scheduled_count)
            
            logging.info(json.dumps({
                "event": "Retrain scheduling complete",
                "scheduled": scheduled_count,
                "level": "info"
            }))
            
        except Exception as e:
            logging.error(json.dumps({
                "event": "Retrain scheduling failed",
                "error": str(e),
                "level": "error"
            }))
