"""
Quick-Start Integration Guide for OpportunityRanker

This file provides step-by-step instructions to integrate OpportunityRanker
into your existing Quantum Trader backend.
"""

# ============================================================================
# STEP 1: Implement Protocol Interfaces
# ============================================================================

"""
Create implementations for the four required protocols.
These should already exist in your codebase in some form.
"""

# File: backend/clients/market_data_client.py
# ============================================================================

from opportunity_ranker import MarketDataClient as IMarketDataClient
import ccxt
import pandas as pd

class BinanceMarketDataClient(IMarketDataClient):
    """Real implementation using CCXT/Binance."""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })
    
    def get_latest_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch candles from Binance."""
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    
    def get_spread(self, symbol: str) -> float:
        """Calculate spread from orderbook."""
        orderbook = self.exchange.fetch_order_book(symbol, limit=5)
        
        best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
        best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
        
        if best_bid > 0 and best_ask > 0:
            spread_pct = (best_ask - best_bid) / best_bid
            return spread_pct
        
        return 0.001  # Default 0.1%
    
    def get_liquidity(self, symbol: str) -> float:
        """Get 24h volume in USD."""
        ticker = self.exchange.fetch_ticker(symbol)
        volume_usd = ticker.get('quoteVolume', 0)
        return volume_usd


# File: backend/repositories/trade_log_repository.py
# ============================================================================

from opportunity_ranker import TradeLogRepository as ITradeLogRepository
from sqlalchemy.orm import Session
from database.models import Trade

class PostgresTradeLogRepository(ITradeLogRepository):
    """Real implementation using PostgreSQL."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_symbol_winrate(self, symbol: str, last_n: int = 200) -> float:
        """Calculate winrate from database."""
        # Query last N closed trades for symbol
        trades = self.db.query(Trade).filter(
            Trade.symbol == symbol,
            Trade.status == 'CLOSED'
        ).order_by(Trade.closed_at.desc()).limit(last_n).all()
        
        if not trades:
            return 0.5  # Default 50% for new symbols
        
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        winrate = winning_trades / len(trades)
        
        return winrate


# File: backend/stores/opportunity_store.py
# ============================================================================

from opportunity_ranker import OpportunityStore as IOpportunityStore
import redis
import json
from typing import Dict
from datetime import datetime

class RedisOpportunityStore(IOpportunityStore):
    """Real implementation using Redis."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key = "opportunity_rankings"
        self.ttl_seconds = 3600  # 1 hour TTL
    
    def update(self, rankings: Dict[str, float]) -> None:
        """Store rankings in Redis."""
        data = {
            'rankings': rankings,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        self.redis.setex(
            self.key,
            self.ttl_seconds,
            json.dumps(data)
        )
    
    def get(self) -> Dict[str, float]:
        """Retrieve rankings from Redis."""
        data = self.redis.get(self.key)
        
        if not data:
            return {}
        
        parsed = json.loads(data)
        return parsed.get('rankings', {})


# ============================================================================
# STEP 2: Update Configuration
# ============================================================================

# File: backend/config.py
# ============================================================================

class Config:
    # ... existing config ...
    
    # OpportunityRanker settings
    OPPORTUNITY_RANKER_ENABLED = True
    OPPORTUNITY_RANKER_TIMEFRAME = "1h"
    OPPORTUNITY_RANKER_CANDLE_LIMIT = 200
    OPPORTUNITY_MIN_SCORE = 0.5
    OPPORTUNITY_UPDATE_INTERVAL_MINUTES = 15
    
    # Symbols to track
    TRADEABLE_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
        "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOGEUSDT",
        "MATICUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT",
    ]
    
    # Optional: Custom weights
    OPPORTUNITY_WEIGHTS = {
        'trend_strength': 0.25,
        'volatility_quality': 0.20,
        'liquidity_score': 0.15,
        'regime_score': 0.15,
        'symbol_winrate_score': 0.10,
        'spread_score': 0.10,
        'noise_score': 0.05,
    }


# ============================================================================
# STEP 3: Initialize in FastAPI App
# ============================================================================

# File: backend/main.py
# ============================================================================

from fastapi import FastAPI
import asyncio
from opportunity_ranker import OpportunityRanker
from routes.opportunity_routes import router as opportunity_router
from config import Config

app = FastAPI(title="Quantum Trader API")

# Include opportunity routes
app.include_router(opportunity_router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    
    # Initialize dependencies (existing services)
    app.state.market_data_client = BinanceMarketDataClient()
    app.state.trade_log_repository = PostgresTradeLogRepository(db_session)
    app.state.regime_detector = global_regime_detector  # Your existing service
    app.state.opportunity_store = RedisOpportunityStore(redis_client)
    
    # Initialize OpportunityRanker
    if Config.OPPORTUNITY_RANKER_ENABLED:
        app.state.opportunity_ranker = OpportunityRanker(
            market_data=app.state.market_data_client,
            trade_logs=app.state.trade_log_repository,
            regime_detector=app.state.regime_detector,
            opportunity_store=app.state.opportunity_store,
            symbols=Config.TRADEABLE_SYMBOLS,
            timeframe=Config.OPPORTUNITY_RANKER_TIMEFRAME,
            candle_limit=Config.OPPORTUNITY_RANKER_CANDLE_LIMIT,
            weights=Config.OPPORTUNITY_WEIGHTS,
            min_score_threshold=Config.OPPORTUNITY_MIN_SCORE,
        )
        
        # Compute initial rankings
        print("Computing initial opportunity rankings...")
        app.state.opportunity_ranker.update_rankings()
        print("✅ Initial rankings computed")
        
        # Start background updater
        asyncio.create_task(periodic_ranking_updater(app))


async def periodic_ranking_updater(app):
    """Background task for periodic updates."""
    interval_minutes = Config.OPPORTUNITY_UPDATE_INTERVAL_MINUTES
    
    while True:
        try:
            rankings = app.state.opportunity_ranker.update_rankings()
            print(f"✅ Rankings updated: {len(rankings)} symbols")
            
        except Exception as e:
            print(f"❌ Ranking update failed: {e}")
        
        await asyncio.sleep(interval_minutes * 60)


# ============================================================================
# STEP 4: Integrate with Orchestrator
# ============================================================================

# File: backend/services/orchestrator_policy.py
# ============================================================================

class OrchestratorPolicy:
    """Enhanced orchestrator with opportunity filtering."""
    
    def __init__(self, opportunity_store, min_opportunity_score: float = 0.5):
        self.opportunity_store = opportunity_store
        self.min_opportunity_score = min_opportunity_score
    
    def should_allow_trade(self, signal) -> tuple[bool, str]:
        """
        Evaluate if trade should be allowed.
        
        Now includes opportunity score check.
        """
        # ... existing checks (risk, exposure, etc.) ...
        
        # NEW: Check opportunity score
        rankings = self.opportunity_store.get()
        symbol_score = rankings.get(signal.symbol, 0.0)
        
        if symbol_score < self.min_opportunity_score:
            return False, f"Low opportunity score: {symbol_score:.3f}"
        
        # All checks passed
        return True, f"Allowed (opportunity: {symbol_score:.3f})"


# ============================================================================
# STEP 5: Use in Strategy Engine
# ============================================================================

# File: backend/services/strategy_engine.py
# ============================================================================

class StrategyEngine:
    """Enhanced strategy engine with opportunity-based symbol selection."""
    
    def __init__(self, opportunity_store):
        self.opportunity_store = opportunity_store
    
    def get_active_symbols(self, max_symbols: int = 10) -> list[str]:
        """
        Get top N symbols to trade based on opportunity scores.
        
        This replaces static symbol lists.
        """
        rankings = self.opportunity_store.get()
        
        # Return top N symbols
        return list(rankings.keys())[:max_symbols]
    
    def generate_signals(self):
        """Generate signals only for high-opportunity symbols."""
        active_symbols = self.get_active_symbols(max_symbols=10)
        
        for symbol in active_symbols:
            # ... generate signal logic ...
            pass


# ============================================================================
# STEP 6: Add to MSC AI (Meta Strategy Controller)
# ============================================================================

# File: backend/services/msc_ai.py
# ============================================================================

class MetaStrategyController:
    """Enhanced MSC AI with opportunity-aware policies."""
    
    def __init__(self, opportunity_store):
        self.opportunity_store = opportunity_store
    
    def adjust_global_policy(self):
        """
        Adjust global trading policy based on opportunity landscape.
        
        Example: If few symbols have high scores, reduce risk mode.
        """
        rankings = self.opportunity_store.get()
        
        # Count high-opportunity symbols
        high_opp_count = sum(1 for score in rankings.values() if score >= 0.7)
        
        if high_opp_count >= 5:
            # Many good opportunities → AGGRESSIVE mode
            self.set_risk_mode("AGGRESSIVE")
            self.set_max_positions(15)
            
        elif high_opp_count >= 2:
            # Some opportunities → NORMAL mode
            self.set_risk_mode("NORMAL")
            self.set_max_positions(10)
            
        else:
            # Few opportunities → DEFENSIVE mode
            self.set_risk_mode("DEFENSIVE")
            self.set_max_positions(5)


# ============================================================================
# STEP 7: Testing & Validation
# ============================================================================

"""
1. Start your backend:
   python backend/main.py

2. Test API endpoints:
   curl http://localhost:8000/api/opportunities/rankings
   curl http://localhost:8000/api/opportunities/rankings/top?n=5
   curl -X POST http://localhost:8000/api/opportunities/refresh

3. Check logs for periodic updates:
   ✅ Rankings updated: 12 symbols

4. Monitor integration:
   - Check Orchestrator logs for opportunity filtering
   - Verify Strategy Engine is using top-ranked symbols
   - Confirm MSC AI is adjusting policies
"""


# ============================================================================
# STEP 8: Monitoring & Tuning
# ============================================================================

"""
After integration, monitor these metrics:

1. Ranking stability
   - How often does top 5 change?
   - Expected: Moderate stability (changes every 2-3 updates)

2. Score distribution
   - How many symbols pass threshold?
   - Expected: 40-60% of symbols

3. Correlation with profitability
   - Do high-score symbols actually perform better?
   - Expected: Positive correlation

4. Update performance
   - How long do updates take?
   - Expected: 5-20 seconds for 50 symbols

5. System impact
   - Does it improve overall PnL?
   - Expected: 10-30% improvement by avoiding poor symbols

Tuning recommendations:
- If too few symbols pass: Lower min_score_threshold
- If updates are slow: Reduce candle_limit or parallelize
- If scores don't correlate with profit: Adjust weights
"""


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Common issues and solutions:

Issue: No symbols pass threshold
Solution: Lower Config.OPPORTUNITY_MIN_SCORE to 0.4 or 0.3

Issue: Rankings never update
Solution: Check background task is running, verify no exceptions in logs

Issue: Market data fetch fails
Solution: Check API keys, rate limits, network connectivity

Issue: High computation time
Solution: Reduce candle_limit to 100, or implement caching

Issue: Scores don't make sense
Solution: Review individual metrics for specific symbols using /rankings/{symbol}/details
"""


# ============================================================================
# COMPLETE! 🎉
# ============================================================================

"""
You now have OpportunityRanker fully integrated into Quantum Trader.

The system will:
✅ Continuously scan all symbols every 15 minutes
✅ Identify high-edge opportunities objectively
✅ Filter out low-quality symbols automatically
✅ Provide rankings to Orchestrator, Strategy Engine, and MSC AI
✅ Expose REST API for monitoring and manual control

Next steps:
1. Monitor performance for 1-2 weeks
2. Tune weights based on observed results
3. Adjust update frequency if needed
4. Consider multi-timeframe enhancement (future)

Congratulations! 🚀
"""
