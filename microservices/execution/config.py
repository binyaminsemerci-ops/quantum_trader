"""
Execution Service - Configuration
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration."""
    
    # Service metadata
    SERVICE_NAME: str = "execution"
    VERSION: str = "1.0.0"
    PORT: int = 8002
    
    # Binance configuration
    USE_BINANCE_TESTNET: bool = os.getenv("USE_BINANCE_TESTNET", "false").lower() == "true"
    BINANCE_API_KEY: str = os.getenv("BINANCE_TESTNET_API_KEY" if os.getenv("USE_BINANCE_TESTNET", "false").lower() == "true" else "BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_TESTNET_SECRET_KEY" if os.getenv("USE_BINANCE_TESTNET", "false").lower() == "true" else "BINANCE_SECRET_KEY", "")
    
    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    
    # SQLite configuration (TradeStore fallback)
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/trades.db")
    
    # EventBus configuration
    EVENT_BUS_TYPE: str = "redis"  # or "memory" for testing
    EVENT_RETENTION_HOURS: int = 24
    
    # Risk & Safety Service (for ESS and PolicyStore)
    RISK_SAFETY_SERVICE_URL: str = os.getenv("RISK_SAFETY_SERVICE_URL", "http://localhost:8003")
    
    # Execution configuration
    POSITION_MONITOR_INTERVAL_SEC: int = int(os.getenv("POSITION_MONITOR_INTERVAL_SEC", "10"))
    MAX_CONCURRENT_ORDERS: int = int(os.getenv("MAX_CONCURRENT_ORDERS", "5"))
    ORDER_TIMEOUT_SEC: int = int(os.getenv("ORDER_TIMEOUT_SEC", "30"))
    
    # Rate limiting (D6)
    BINANCE_RATE_LIMIT_RPM: int = int(os.getenv("BINANCE_RATE_LIMIT_RPM", "1200"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = "logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
