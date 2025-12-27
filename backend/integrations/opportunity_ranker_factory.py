"""
OpportunityRanker Integration Factory
Wires real implementations into OpportunityRanker
"""

"""
OpportunityRanker Integration Factory
Wires real implementations into OpportunityRanker
"""

import logging
from typing import Optional
import redis

# Direct import from the comprehensive OpportunityRanker file (not the directory package)
import importlib.util
import sys
from pathlib import Path

# Load opportunity_ranker.py directly to avoid package conflict
spec = importlib.util.spec_from_file_location(
    "opportunity_ranker_direct",
    Path(__file__).parent.parent / "services" / "opportunity_ranker.py"
)
opp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opp_module)
OpportunityRanker = opp_module.OpportunityRanker

from backend.clients.binance_market_data_client import BinanceMarketDataClient
from backend.repositories.postgres_trade_log_repository import PostgresTradeLogRepository
from backend.stores.redis_opportunity_store import RedisOpportunityStore
from backend.services.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


def create_opportunity_ranker(
    binance_api_key: Optional[str] = None,
    binance_api_secret: Optional[str] = None,
    db_session_factory = None,
    redis_client: Optional[redis.Redis] = None,
    regime_detector: Optional[RegimeDetector] = None,
    custom_weights: Optional[dict[str, float]] = None,
    policy_store = None
) -> OpportunityRanker:
    """
    Create fully wired OpportunityRanker with real implementations.
    
    Args:
        binance_api_key: Binance API key (optional for public data)
        binance_api_secret: Binance API secret
        db_session_factory: Database session factory for trade logs
        redis_client: Redis client for rankings storage
        regime_detector: Existing RegimeDetector instance
        custom_weights: Optional custom metric weights
        policy_store: PolicyStore instance for writing rankings
        
    Returns:
        Configured OpportunityRanker instance
    """
    logger.info("Creating OpportunityRanker with real implementations")
    
    # Create MarketDataClient
    market_data = BinanceMarketDataClient(
        api_key=binance_api_key,
        api_secret=binance_api_secret
    )
    
    # Create TradeLogRepository
    trade_logs = PostgresTradeLogRepository(db_session_factory)
    
    # Create OpportunityStore
    opportunity_store = RedisOpportunityStore(redis_client)
    
    # Get default symbols
    symbols = get_default_symbols()
    
    # Create OpportunityRanker with correct parameter names
    ranker = OpportunityRanker(
        market_data=market_data,
        trade_logs=trade_logs,
        regime_detector=regime_detector,
        opportunity_store=opportunity_store,
        symbols=symbols,
        timeframe="1h",
        candle_limit=200,
        weights=custom_weights,
        min_score_threshold=0.3
    )
    
    # Attach PolicyStore for rankings storage
    if policy_store:
        ranker.policy_store = policy_store
        logger.info("OpportunityRanker created with PolicyStore integration")
    else:
        logger.info("OpportunityRanker created without PolicyStore integration")
    
    return ranker


def get_default_symbols() -> list[str]:
    """
    Get default symbol universe for ranking.
    
    Returns:
        List of symbols to rank
    """
    return [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
        "XRPUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT",
        "LINKUSDT", "ATOMUSDT", "NEARUSDT", "UNIUSDT", "LTCUSDT",
        "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT"
    ]
