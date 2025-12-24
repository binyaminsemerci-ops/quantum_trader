"""
AI Engine Service - Configuration
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """AI Engine Service settings."""
    
    # Service info
    SERVICE_NAME: str = "ai-engine"
    VERSION: str = "1.0.0"
    PORT: int = 8001
    
    # Redis (EventBus) - no prefix, read directly from env
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # Risk-safety-service (PolicyStore)
    RISK_SAFETY_SERVICE_URL: str = "http://localhost:8003"
    
    # Ensemble configuration
    ENSEMBLE_MODELS: List[str] = ["xgb", "lgbm", "nhits", "patchtst"]
    ENSEMBLE_WEIGHTS: dict = {
        "xgb": 0.25,
        "lgbm": 0.25,
        "nhits": 0.30,
        "patchtst": 0.20
    }
    MIN_CONSENSUS: int = 3  # 3/4 models must agree
    
    # Model paths
    MODELS_DIR: str = "models"
    XGB_MODEL_PATH: str = "models/xgb_futures_model.joblib"
    XGB_SCALER_PATH: str = "models/xgb_futures_scaler.joblib"
    LGBM_MODEL_PATH: str = "models/lgbm_model.txt"
    NHITS_MODEL_PATH: str = "models/nhits_model.pt"
    PATCHTST_MODEL_PATH: str = "models/patchtst_model.pt"
    
    # Meta-strategy configuration
    META_STRATEGY_ENABLED: bool = True
    META_STRATEGY_EPSILON: float = 0.10  # 10% exploration
    META_STRATEGY_ALPHA: float = 0.20    # Learning rate
    META_STRATEGY_DISCOUNT: float = 0.95  # Future reward discount
    META_STRATEGY_STATE_PATH: str = "/app/data/meta_strategy_q_table.json"
    
    # RL Position Sizing configuration
    RL_SIZING_ENABLED: bool = True
    RL_SIZING_EPSILON: float = 0.15      # 15% exploration
    RL_SIZING_ALPHA: float = 0.20        # Learning rate
    RL_SIZING_DISCOUNT: float = 0.95     # Future reward discount
    RL_SIZING_STATE_PATH: str = "/app/data/rl_sizing_q_table.json"
    
    # Regime detection
    REGIME_DETECTION_ENABLED: bool = True
    REGIME_UPDATE_INTERVAL_SEC: int = 60  # Update every minute
    
    # Memory state
    MEMORY_STATE_ENABLED: bool = True
    MEMORY_LOOKBACK_HOURS: int = 24       # 24-hour memory window
    
    # Model Supervisor (bias detection)
    MODEL_SUPERVISOR_ENABLED: bool = True
    MODEL_SUPERVISOR_BIAS_THRESHOLD: float = 0.70  # Block if >70% SHORT or LONG bias
    MODEL_SUPERVISOR_MIN_SAMPLES: int = 20         # Need 20 signals to detect bias
    
    # Continuous Learning
    CONTINUOUS_LEARNING_ENABLED: bool = True
    MIN_SAMPLES_FOR_RETRAIN: int = 20              # Retrain after 20 new samples (was 50)
    RETRAIN_INTERVAL_HOURS: int = 2                # Auto-retrain every 2 hours (was 24)
    
    # 🔥 PHASE 1 MODULES - FUTURES INTELLIGENCE 🔥
    
    # Cross-Exchange Normalizer (volatility_factor, divergence, lead/lag)
    CROSS_EXCHANGE_ENABLED: bool = True
    CROSS_EXCHANGE_SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    CROSS_EXCHANGE_EXCHANGES: List[str] = ["binance", "bybit", "coinbase"]
    
    # Funding Rate Filter (funding_delta, crowded_side_score, squeeze_probability)
    FUNDING_RATE_ENABLED: bool = True
    MAX_FUNDING_RATE: float = 0.001  # 0.1% per 8h = max acceptable
    WARN_FUNDING_RATE: float = 0.0005  # 0.05% per 8h = warning threshold
    
    # Drift Detection Manager (drift_flag, retrain_trigger)
    DRIFT_DETECTION_ENABLED: bool = True
    DRIFT_PSI_THRESHOLD_MODERATE: float = 0.15  # 15% = moderate drift
    DRIFT_PSI_THRESHOLD_SEVERE: float = 0.25    # 25% = severe drift
    DRIFT_CHECK_INTERVAL_SEC: int = 300         # Check every 5 minutes
    
    # Reinforcement Signal Manager (PnL feedback, confidence calibration)
    REINFORCEMENT_SIGNAL_ENABLED: bool = True
    RL_SIGNAL_LEARNING_RATE: float = 0.05       # 5% learning from each trade
    RL_SIGNAL_DISCOUNT_FACTOR: float = 0.95     # Temporal discount
    RL_SIGNAL_STATE_PATH: str = "/app/data/rl_signal_weights.json"
    
    # Confidence thresholds
    MIN_SIGNAL_CONFIDENCE: float = 0.55   # Block signals <55% (lowered for testing)
    HIGH_CONFIDENCE_THRESHOLD: float = 0.85  # Flag high-confidence signals
    
    # Event processing
    MAX_CONCURRENT_SIGNALS: int = 10      # Max 10 symbols processed concurrently
    SIGNAL_TIMEOUT_SEC: int = 30          # Timeout for signal generation
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "/app/logs"
    
    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL."""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    class Config:
        env_file = ".env"
        env_prefix = "AI_ENGINE_"
        extra = "allow"  # Allow extra env vars


settings = Settings()
