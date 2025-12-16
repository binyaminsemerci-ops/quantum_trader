"""
OpportunityRanker Integration Instructions for main.py

Follow these steps to integrate OpportunityRanker into Quantum Trader:

## STEP 1: Add imports at top of main.py (after existing imports)

```python
# OpportunityRanker Integration
from backend.integrations.opportunity_ranker_factory import (
    create_opportunity_ranker,
    get_default_symbols
)
from backend.routes import opportunity_routes
```

## STEP 2: Initialize OpportunityRanker in lifespan() function
Add this code in the lifespan() function around line 638 (after self_healing system):

```python
# [NEW] Initialize OpportunityRanker - Market Quality Evaluation
opportunity_ranker_enabled = os.getenv("QT_OPPORTUNITY_RANKER_ENABLED", "true").lower() == "true"
if opportunity_ranker_enabled:
    try:
        logger.info("[SEARCH] Initializing OpportunityRanker...")
        
        # Get Binance credentials from environment
        binance_api_key = os.getenv("BINANCE_API_KEY")
        binance_api_secret = os.getenv("BINANCE_API_SECRET")
        
        # Create Redis client (reuse existing if available)
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        
        import redis
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        
        # Get RegimeDetector if available
        regime_detector = None
        try:
            from backend.services.regime_detector import RegimeDetector
            regime_detector = RegimeDetector()
            logger.info("[OK] RegimeDetector loaded for OpportunityRanker")
        except Exception as e:
            logger.warning(f"[WARNING] RegimeDetector not available: {e}")
        
        # Create OpportunityRanker
        opportunity_ranker = create_opportunity_ranker(
            binance_api_key=binance_api_key,
            binance_api_secret=binance_api_secret,
            db_session_factory=SessionLocal,
            redis_client=redis_client,
            regime_detector=regime_detector
        )
        
        # Store in app state
        app_instance.state.opportunity_ranker = opportunity_ranker
        app_instance.state.redis_client = redis_client
        
        # Compute initial rankings
        symbols = get_default_symbols()
        logger.info(f"[SEARCH] Computing initial rankings for {len(symbols)} symbols...")
        
        async def compute_initial_rankings():
            """Compute initial rankings asynchronously."""
            try:
                await opportunity_ranker.rank_opportunities(symbols)
                rankings = opportunity_ranker.get_rankings()
                logger.info(f"[OK] Initial rankings computed: {len(rankings)} symbols ranked")
                
                # Log top 5
                top_5 = rankings[:5]
                for rank in top_5:
                    logger.info(
                        f"   #{rank.rank}: {rank.symbol} = {rank.overall_score:.3f}"
                    )
            except Exception as e:
                logger.error(f"[ERROR] Failed to compute initial rankings: {e}")
        
        # Start background ranking task
        asyncio.create_task(compute_initial_rankings())
        
        # Start periodic refresh (every 5 minutes)
        refresh_interval = int(os.getenv("QT_OPPORTUNITY_REFRESH_INTERVAL", "300"))
        
        async def ranking_refresh_loop():
            """Periodically refresh opportunity rankings."""
            await asyncio.sleep(60)  # Wait for system to stabilize
            
            while True:
                try:
                    logger.debug("[OpportunityRanker] Refreshing rankings...")
                    await opportunity_ranker.rank_opportunities(symbols)
                    rankings = opportunity_ranker.get_rankings()
                    logger.info(
                        f"[OpportunityRanker] Refreshed: {len(rankings)} symbols ranked, "
                        f"Top: {rankings[0].symbol if rankings else 'N/A'}"
                    )
                except Exception as e:
                    logger.error(f"[OpportunityRanker] Refresh failed: {e}")
                
                await asyncio.sleep(refresh_interval)
        
        ranking_task = asyncio.create_task(ranking_refresh_loop())
        app_instance.state.opportunity_ranking_task = ranking_task
        
        logger.info(
            f"✅ OpportunityRanker: ENABLED (refreshes every {refresh_interval}s)"
        )
    except Exception as e:
        logger.warning(f"[WARNING] Could not start OpportunityRanker: {e}")
else:
    logger.info("ℹ️ OpportunityRanker: DISABLED (set QT_OPPORTUNITY_RANKER_ENABLED=true)")
```

## STEP 3: Register OpportunityRanker routes
Add this line around line 1158 (with other app.include_router() calls):

```python
app.include_router(opportunity_routes.router, prefix="/opportunities")
```

## STEP 4: Environment Variables
Add to .env file:

```bash
# OpportunityRanker Configuration
QT_OPPORTUNITY_RANKER_ENABLED=true
QT_OPPORTUNITY_REFRESH_INTERVAL=300  # 5 minutes
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

## STEP 5: Test Integration

After modifying main.py:

1. Start backend:
```bash
python backend/main.py
```

2. Check logs for:
```
[SEARCH] Initializing OpportunityRanker...
[OK] Initial rankings computed: 20 symbols ranked
   #1: BTCUSDT = 0.753
   #2: ETHUSDT = 0.698
✅ OpportunityRanker: ENABLED (refreshes every 300s)
```

3. Test API endpoints:
```bash
# Get all rankings
curl http://localhost:8000/opportunities/rankings

# Get top 10
curl http://localhost:8000/opportunities/rankings/top?n=10

# Get specific symbol
curl http://localhost:8000/opportunities/rankings/BTCUSDT

# Get detailed breakdown
curl http://localhost:8000/opportunities/rankings/BTCUSDT/details

# Force refresh
curl -X POST http://localhost:8000/opportunities/refresh
```

## STEP 6: Integration with Orchestrator (Optional)

To filter trades by opportunity score, modify the Orchestrator:

```python
# In backend/services/orchestrator.py or similar

def should_allow_trade(self, symbol: str, side: str) -> bool:
    # Existing checks...
    
    # Check opportunity score
    if hasattr(self.app_state, 'opportunity_ranker'):
        ranker = self.app_state.opportunity_ranker
        ranking = ranker.get_ranking_for_symbol(symbol)
        
        if ranking and ranking.overall_score < 0.5:  # Below threshold
            logger.info(
                f"Trade blocked: {symbol} opportunity score too low "
                f"({ranking.overall_score:.3f})"
            )
            return False
    
    return True
```

## STEP 7: Integration with Strategy Engine (Optional)

To use top-ranked symbols for signal generation:

```python
# In backend/services/strategy_engine.py or similar

def get_active_symbols(self) -> list[str]:
    # Check if OpportunityRanker is available
    if hasattr(self.app_state, 'opportunity_ranker'):
        ranker = self.app_state.opportunity_ranker
        
        # Get top N symbols by opportunity score
        rankings = ranker.get_top_opportunities(n=10, min_score=0.6)
        symbols = [r.symbol for r in rankings]
        
        logger.info(
            f"Using OpportunityRanker top symbols: {symbols}"
        )
        return symbols
    
    # Fallback to default symbols
    return ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
```

## STEP 8: Integration with MSC AI (Optional)

To adjust risk based on opportunity landscape:

```python
# In backend/services/msc_ai_integration.py

def adjust_policy_based_on_opportunities(self):
    """Adjust risk policy based on available high-quality opportunities."""
    if not hasattr(self.app_state, 'opportunity_ranker'):
        return
    
    ranker = self.app_state.opportunity_ranker
    high_quality = ranker.get_top_opportunities(n=20, min_score=0.7)
    
    logger.info(f"[MSC AI] High-quality opportunities: {len(high_quality)}")
    
    if len(high_quality) >= 5:
        # Many good opportunities -> AGGRESSIVE
        self.set_risk_mode("AGGRESSIVE")
        logger.info("[MSC AI] Rich opportunity landscape → AGGRESSIVE mode")
    elif len(high_quality) >= 2:
        # Moderate opportunities -> NORMAL
        self.set_risk_mode("NORMAL")
    else:
        # Few opportunities -> DEFENSIVE
        self.set_risk_mode("DEFENSIVE")
        logger.info("[MSC AI] Scarce opportunities → DEFENSIVE mode")
```

## Complete!

Your OpportunityRanker is now:
✅ Integrated into Quantum Trader's main.py
✅ Computing rankings every 5 minutes
✅ Accessible via REST API
✅ Stored in Redis with 1-hour TTL
✅ Ready for Orchestrator/Strategy Engine integration
✅ Monitoring 20 default symbols
✅ Using real Binance data, PostgreSQL trade logs, and RegimeDetector

## Next Steps

1. Monitor logs during operation
2. Check `/opportunities/rankings` endpoint
3. Integrate with Orchestrator to filter low-quality trades
4. Integrate with Strategy Engine to focus on high-quality symbols
5. Integrate with MSC AI for dynamic risk adjustment
6. Add custom weights if needed: `opportunity_ranker.update_weights({...})`
7. Expand symbol universe if needed: modify `get_default_symbols()`
"""
