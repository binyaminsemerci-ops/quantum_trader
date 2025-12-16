"""
Risk & Safety Service - Configuration
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration."""
    
    # Service metadata
    SERVICE_NAME: str = "risk-safety"
    VERSION: str = "1.0.0"
    PORT: int = 8003
    
    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    
    # EventBus configuration
    EVENT_BUS_TYPE: str = "redis"  # or "memory" for testing
    EVENT_RETENTION_HOURS: int = 24
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = "logs"
    
    # ESS configuration
    ESS_MAX_DAILY_LOSS_PCT: float = 0.05  # 5%
    ESS_MAX_CONSECUTIVE_LOSSES: int = 3
    ESS_MAX_DRAWDOWN_PCT: float = 0.10  # 10%
    ESS_MIN_WIN_RATE: float = 0.30  # 30%
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
