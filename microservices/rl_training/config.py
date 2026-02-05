"""
Configuration for RL Training / CLM / Shadow Models Service

Service: rl-training-service (port 8005)
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Service configuration"""
    
    # Service identity
    SERVICE_NAME: str = "rl-training"
    SERVICE_VERSION: str = "1.0.0"
    PORT: int = 8007
    
    # Redis (EventBus)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Database (for model registry, training history)
    DATABASE_URL: str = "sqlite:///data/rl_training.db"
    
    # PolicyStore (readonly access to trading policies)
    POLICY_STORE_TYPE: str = "sqlite"  # or "redis"
    POLICY_STORE_DB_PATH: str = "data/policy_store.db"
    
    # Model storage
    MODEL_SAVE_DIR: str = "data/models"
    MODEL_CHECKPOINT_DIR: str = "data/models/checkpoints"
    
    # Training configuration
    RL_TRAINING_ENABLED: bool = True
    CLM_ENABLED: bool = True
    SHADOW_TESTING_ENABLED: bool = True
    DRIFT_DETECTION_ENABLED: bool = True
    
    # Scheduling
    RL_RETRAIN_INTERVAL_HOURS: int = 168  # Weekly (7 days)
    CLM_RETRAIN_INTERVAL_HOURS: int = 168  # Weekly
    DRIFT_CHECK_INTERVAL_HOURS: int = 24  # Daily
    PERFORMANCE_CHECK_INTERVAL_HOURS: int = 6  # Every 6 hours
    
    # Thresholds
    MIN_SAMPLES_FOR_RETRAIN: int = 100
    DRIFT_TRIGGER_THRESHOLD: float = 0.05  # PSI > 0.05 triggers retraining
    PERFORMANCE_DECAY_THRESHOLD: float = 0.10  # 10% performance drop
    SHADOW_MIN_PREDICTIONS: int = 100  # Min predictions for shadow evaluation
    
    # Auto-promotion
    AUTO_PROMOTION_ENABLED: bool = True
    MIN_IMPROVEMENT_FOR_PROMOTION: float = 0.02  # 2% improvement required
    
    # Data sources
    TRADE_STORE_TYPE: str = "sqlite"
    TRADE_STORE_DB_PATH: str = "data/trades.db"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    model_config = {
        "env_prefix": "RL_TRAINING_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Ignore extra fields from .env
    }


settings = Settings()
