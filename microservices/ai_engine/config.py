"""
AI Engine Service - Configuration

Controlled Refactor 2026-02-21:
  All model LOAD paths → /opt/quantum/model_registry/approved/
  All model WRITE paths → /opt/quantum/model_registry/staging/  (retrain worker only)
"""
from pydantic_settings import BaseSettings
from typing import List
import os
import logging

_guard_log = logging.getLogger(__name__)


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
    
    # Model paths — Controlled Refactor 2026-02-21
    # READ (live AI engine) : /opt/quantum/model_registry/approved/
    # WRITE (retrain worker): /opt/quantum/model_registry/staging/
    # DO NOT change these defaults to point anywhere else without running
    # the CLM promotion workflow first.
    APPROVED_MODEL_DIR: str = "/opt/quantum/model_registry/approved"
    STAGING_MODEL_DIR:  str = "/opt/quantum/model_registry/staging"
    MODELS_DIR: str = "/opt/quantum/model_registry/approved"
    XGB_MODEL_PATH:     str = "/opt/quantum/model_registry/approved/xgb_model.pkl"
    XGB_SCALER_PATH:    str = "/opt/quantum/model_registry/approved/scaler.pkl"
    LGBM_MODEL_PATH:    str = "/opt/quantum/model_registry/approved/lgbm_model.txt"
    NHITS_MODEL_PATH:   str = "/opt/quantum/model_registry/approved/nhits_model.pt"
    PATCHTST_MODEL_PATH: str = "/opt/quantum/model_registry/approved/patchtst_model.pt"
    
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
    CROSS_EXCHANGE_SYMBOLS: str = "BTCUSDT,ETHUSDT,SOLUSDT"  # Comma-separated string
    CROSS_EXCHANGE_EXCHANGES: List[str] = ["binance", "bybit", "coinbase"]
    
    @property
    def cross_exchange_symbols_list(self) -> List[str]:
        """Parse CROSS_EXCHANGE_SYMBOLS into a list."""
        return [s.strip() for s in self.CROSS_EXCHANGE_SYMBOLS.split(",") if s.strip()]
    
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
    
    # Portfolio Top-N Confidence Gate
    # Filters predictions portfolio-wide to select only the top N highest-confidence signals
    # This prevents overtrading and focuses resources on highest-quality opportunities
    TOP_N_LIMIT: int = int(os.getenv("TOP_N_LIMIT", "10"))  # Max predictions to publish per cycle
    TOP_N_BUFFER_INTERVAL_SEC: float = float(os.getenv("TOP_N_BUFFER_INTERVAL_SEC", "2.0"))  # Buffer processing interval
    MAX_SYMBOL_CORRELATION: float = float(os.getenv("MAX_SYMBOL_CORRELATION", "0.80"))  # Max correlation threshold for portfolio diversification
    
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
        env_prefix = ""  # No prefix - read vars directly
        extra = "allow"  # Allow extra env vars


settings = Settings()


def validate_approved_model_paths() -> None:
    """
    Import-time assertion: all configured model paths must resolve under
    APPROVED_MODEL_DIR.  Called once at module load so any misconfiguration
    is caught before the service accepts traffic.

    Raises RuntimeError and aborts startup if any path is outside approved/.
    """
    try:
        from model_path_guard import assert_approved_load_path
    except ImportError:
        _guard_log.error(
            "[MODEL-GUARD] model_path_guard module not found — "
            "path enforcement is DISABLED. Deploy model_path_guard.py."
        )
        return

    paths_to_check = [
        (settings.XGB_MODEL_PATH,      "xgb_model"),
        (settings.XGB_SCALER_PATH,     "xgb_scaler"),
        (settings.LGBM_MODEL_PATH,     "lgbm_model"),
        (settings.NHITS_MODEL_PATH,    "nhits_model"),
        (settings.PATCHTST_MODEL_PATH, "patchtst_model"),
    ]

    violations = []
    for path, label in paths_to_check:
        try:
            assert_approved_load_path(path, label=label)
        except RuntimeError as exc:
            violations.append(str(exc))

    if violations:
        joined = "\n".join(violations)
        raise RuntimeError(
            f"[MODEL-GUARD] Config validation failed — {len(violations)} path violation(s):\n{joined}"
        )

    _guard_log.info(
        "[MODEL-GUARD] ✅ All model load paths verified under approved registry: "
        f"{settings.APPROVED_MODEL_DIR}"
    )


# Run at import time so misconfiguration is caught before service start
validate_approved_model_paths()
