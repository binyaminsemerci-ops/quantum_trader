"""
Execution Service - Configuration V2

Clean configuration with NO monolith dependencies.
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration."""
    
    # Service metadata
    SERVICE_NAME: str = "execution"
    VERSION: str = "2.0.0"
    PORT: int = 8002
    
    # Execution mode: PAPER, TESTNET, LIVE
    EXECUTION_MODE: str = os.getenv("EXECUTION_MODE", "PAPER")
    
    # Binance configuration
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    USE_BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    
    # Redis configuration (for EventBus)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    
    # Risk validation (RiskStub parameters)
    MAX_POSITION_USD: float = float(os.getenv("MAX_POSITION_USD", "1000"))
    MAX_LEVERAGE: int = int(os.getenv("MAX_LEVERAGE", "10"))
    
    # Rate limiting
    BINANCE_RATE_LIMIT_RPM: int = int(os.getenv("BINANCE_RATE_LIMIT_RPM", "1200"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "logs")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


settings = Settings()
