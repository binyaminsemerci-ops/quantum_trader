"""
Retraining Worker - Redis Stream Listener
==========================================
Listens for retraining jobs on Redis streams and dispatches them to ModelTrainer.

Streams monitored:
- quantum:stream:model.retrain (incoming jobs)

Streams written:
- quantum:stream:learning.retraining.started
- quantum:stream:learning.retraining.completed
- quantum:stream:learning.retraining.failed
"""

import json
import redis
import time
import os
import logging
from datetime import datetime
from model_trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class RetrainWorker:
    """
    Worker service that listens for model retraining jobs and executes them.
    """
    
    def __init__(self, redis_url: str = None):
        """
        Initialize retraining worker.
        
        Args:
            redis_url: Redis connection URL (defaults to env var)
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        self.trainer = ModelTrainer()
        
        # Stream names
        self.input_stream = "quantum:stream:model.retrain"
        self.started_stream = "quantum:stream:learning.retraining.started"
        self.completed_stream = "quantum:stream:learning.retraining.completed"
        self.failed_stream = "quantum:stream:learning.retraining.failed"
        
        # Consumer group setup
        self.consumer_group = "retraining_workers"
        self.consumer_name = f"worker_{os.getpid()}"
        
        self._setup_consumer_group()
        
        logger.info(f"[RetrainWorker] Initialized")
        logger.info(f"[RetrainWorker] Redis: {self.redis_url}")
        logger.info(f"[RetrainWorker] Listening on: {self.input_stream}")
    
    def _setup_consumer_group(self):
        """Create consumer group if it doesn't exist."""
        try:
            self.redis.xgroup_create(
                self.input_stream, 
                self.consumer_group, 
                id='0', 
                mkstream=True
            )
            logger.info(f"[RetrainWorker] Created consumer group: {self.consumer_group}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"[RetrainWorker] Consumer group already exists")
            else:
                raise
    
    def listen(self):
        """
        Main listening loop - reads from Redis stream and processes jobs.
        """
        logger.info("=" * 60)
        logger.info("[RetrainWorker] 🎧 Listening for retrain jobs...")
        logger.info("=" * 60)
        
        last_id = '>'  # Start from new messages
        
        while True:
            try:
                # Read from consumer group
                messages = self.redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.input_stream: last_id},
                    count=1,
                    block=5000  # 5 second timeout
                )
                
                if not messages:
                    continue
                
                # Process each message
                for stream_name, entries in messages:
                    for entry_id, fields in entries:
                        try:
                            logger.info(f"[RetrainWorker] 📨 New retrain job: {entry_id}")
                            
                            # Parse job data
                            job_data = {k: v for k, v in fields.items()}
                            model = job_data.get("model", "unknown")
                            learning_rate = float(job_data.get("learning_rate", 0.001))
                            optimizer = job_data.get("optimizer", "adam")
                            
                            logger.info(f"[RetrainWorker] Model: {model}, LR: {learning_rate}, Opt: {optimizer}")
                            
                            # Publish started event
                            self._publish_started(model, job_data)
                            
                            # Execute training
                            result = self.trainer.run_training(
                                model=model,
                                learning_rate=learning_rate,
                                optimizer=optimizer,
                                job_data=job_data
                            )
                            
                            # Publish completion
                            if result["success"]:
                                self._publish_completed(model, result)
                                logger.info(f"[RetrainWorker] ✅ Training complete: {model}")
                            else:
                                self._publish_failed(model, result)
                                logger.error(f"[RetrainWorker] ❌ Training failed: {model}")
                            
                            # Acknowledge message
                            self.redis.xack(self.input_stream, self.consumer_group, entry_id)
                            
                        except Exception as e:
                            logger.error(f"[RetrainWorker] Error processing job {entry_id}: {e}")
                            self._publish_failed(
                                job_data.get("model", "unknown"),
                                {"error": str(e), "entry_id": entry_id}
                            )
                
            except KeyboardInterrupt:
                logger.info("[RetrainWorker] Shutting down...")
                break
            except Exception as e:
                logger.error(f"[RetrainWorker] Error in main loop: {e}")
                time.sleep(5)
    
    def _publish_started(self, model: str, job_data: dict):
        """Publish job started event."""
        event = {
            "event_type": "learning.retraining.started",
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
            "learning_rate": job_data.get("learning_rate", "unknown"),
            "optimizer": job_data.get("optimizer", "unknown")
        }
        self.redis.xadd(self.started_stream, event)
    
    def _publish_completed(self, model: str, result: dict):
        """Publish job completed event."""
        event = {
            "event_type": "learning.retraining.completed",
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
            "model_path": result.get("model_path", ""),
            "duration_seconds": result.get("duration", 0),
            "final_loss": result.get("final_loss", 0),
            "status": "success"
        }
        self.redis.xadd(self.completed_stream, event)
        
        # Update counter
        self.redis.incr("quantum:evolution:retrain_completed_count")
    
    def _publish_failed(self, model: str, result: dict):
        """Publish job failed event."""
        event = {
            "event_type": "learning.retraining.failed",
            "model": model,
            "timestamp": datetime.utcnow().isoformat(),
            "error": result.get("error", "Unknown error"),
            "status": "failed"
        }
        self.redis.xadd(self.failed_stream, event)


if __name__ == "__main__":
    worker = RetrainWorker()
    worker.listen()
