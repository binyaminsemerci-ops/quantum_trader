import time
import json
import redis
import logging
from performance_evaluator import PerformanceEvaluator
from model_selector import ModelSelector
from retrain_manager import RetrainManager
from mutation_engine import MutationEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class StrategicEvolutionEngine:
    """Kjernen i strategi-evolusjonssystemet"""

    def __init__(self, redis_url="redis://redis:6379/0"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.evaluator = PerformanceEvaluator(self.redis)
        self.selector = ModelSelector(self.redis)
        self.retrainer = RetrainManager(self.redis)
        self.mutator = MutationEngine(self.redis)
        self.iteration = 0

    def run(self):
        logging.info(json.dumps({
            "event": "Strategic Evolution Engine started",
            "status": "active",
            "level": "info"
        }))
        
        while True:
            try:
                self.iteration += 1
                
                logging.info(json.dumps({
                    "iteration": self.iteration,
                    "event": "Starting evolution cycle",
                    "level": "info"
                }))
                
                # Step 1: Evaluate all strategies
                perf = self.evaluator.evaluate()
                
                # Step 2: Select best performers
                best_models = self.selector.select_best(perf)
                
                # Step 3: Mutate configurations
                mutated = self.mutator.mutate(best_models)
                
                # Step 4: Schedule retraining
                self.retrainer.schedule_retrain(mutated)
                
                logging.info(json.dumps({
                    "iteration": self.iteration,
                    "evaluated": len(perf),
                    "selected": len(best_models),
                    "mutated": len(mutated),
                    "event": "Evolution cycle complete",
                    "level": "info"
                }))
                
                # Run every 10 minutes
                time.sleep(600)
                
            except Exception as e:
                logging.error(json.dumps({
                    "event": "Evolution cycle error",
                    "error": str(e),
                    "level": "error"
                }))
                time.sleep(60)

if __name__ == "__main__":
    StrategicEvolutionEngine().run()
