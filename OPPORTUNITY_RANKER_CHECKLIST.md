# OpportunityRanker Implementation Checklist

Use this checklist to track your integration progress.

## ✅ Phase 1: Review & Understanding

- [ ] Read `OPPORTUNITY_RANKER_README.md` (full documentation)
- [ ] Read `OPPORTUNITY_RANKER_SUMMARY.md` (implementation overview)
- [ ] Review `OPPORTUNITY_RANKER_ARCHITECTURE.md` (system design)
- [ ] Run examples: `python opportunity_ranker_example.py`
- [ ] Run tests: `python -m pytest test_opportunity_ranker.py -v`
- [ ] Understand the 7 metrics and their purpose
- [ ] Review default weights and scoring logic

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## ✅ Phase 2: Implement Dependencies

### MarketDataClient
- [ ] Create `BinanceMarketDataClient` class
- [ ] Implement `get_latest_candles(symbol, timeframe, limit)`
- [ ] Implement `get_spread(symbol)` using orderbook
- [ ] Implement `get_liquidity(symbol)` using 24h volume
- [ ] Test with real API (fetch BTCUSDT data)
- [ ] Add error handling and rate limiting

### TradeLogRepository
- [ ] Create `PostgresTradeLogRepository` class (or use existing)
- [ ] Implement `get_symbol_winrate(symbol, last_n=200)`
- [ ] Query closed trades from database
- [ ] Calculate winrate (winning_trades / total_trades)
- [ ] Return 0.5 for symbols with no history
- [ ] Test with existing trade data

### OpportunityStore
- [ ] Create `RedisOpportunityStore` class
- [ ] Implement `update(rankings)` with JSON serialization
- [ ] Implement `get()` with JSON deserialization
- [ ] Add TTL (1 hour recommended)
- [ ] Add timestamp metadata
- [ ] Test Redis connection and storage

### RegimeDetector (existing)
- [ ] Verify `get_global_regime()` returns string
- [ ] Supported values: "BULL", "BEAR", "CHOPPY", "RANGING"
- [ ] Test integration

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## ✅ Phase 3: Configuration

- [ ] Add OpportunityRanker settings to `config.py`
- [ ] Set `TRADEABLE_SYMBOLS` list (10-50 symbols)
- [ ] Set `OPPORTUNITY_RANKER_TIMEFRAME` (1h recommended)
- [ ] Set `OPPORTUNITY_RANKER_CANDLE_LIMIT` (200 recommended)
- [ ] Set `OPPORTUNITY_MIN_SCORE` (0.5 recommended)
- [ ] Set `OPPORTUNITY_UPDATE_INTERVAL_MINUTES` (15 recommended)
- [ ] (Optional) Customize `OPPORTUNITY_WEIGHTS` dict
- [ ] Verify all config values are reasonable

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## ✅ Phase 4: Backend Integration

### FastAPI Setup
- [ ] Copy `opportunity_routes.py` to `backend/routes/`
- [ ] Add router to FastAPI app: `app.include_router(opportunity_router)`
- [ ] Verify router imports correctly
- [ ] Test import doesn't break existing app

### Startup Hook
- [ ] Add OpportunityRanker initialization to `@app.on_event("startup")`
- [ ] Initialize all four dependencies
- [ ] Create OpportunityRanker instance
- [ ] Store in `app.state.opportunity_ranker`
- [ ] Compute initial rankings
- [ ] Start background scheduler task
- [ ] Add logging for startup process

### Background Scheduler
- [ ] Create `periodic_ranking_updater()` async function
- [ ] Use `asyncio.create_task()` to start background task
- [ ] Add error handling in scheduler loop
- [ ] Add logging for each update cycle
- [ ] Verify task runs on startup

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## ✅ Phase 5: Test Backend Integration

### Manual Testing
- [ ] Start FastAPI app: `python backend/main.py`
- [ ] Check startup logs for "Rankings computed"
- [ ] Wait 1 minute, check for first periodic update
- [ ] Test GET `/api/opportunities/rankings`
- [ ] Test GET `/api/opportunities/rankings/top?n=5`
- [ ] Test GET `/api/opportunities/rankings/BTCUSDT`
- [ ] Test GET `/api/opportunities/rankings/BTCUSDT/details`
- [ ] Test POST `/api/opportunities/refresh`

### API Response Validation
- [ ] Verify rankings are sorted by score (descending)
- [ ] Verify all scores are between 0.0 and 1.0
- [ ] Verify timestamps are recent
- [ ] Verify detailed metrics endpoint shows all 7 metrics
- [ ] Verify refresh endpoint returns execution time

### Performance Testing
- [ ] Measure initial ranking computation time
- [ ] Measure periodic update time
- [ ] Verify updates complete within 30 seconds
- [ ] Monitor Redis memory usage
- [ ] Check for memory leaks (observe over 1 hour)

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## ✅ Phase 6: Integrate with Orchestrator

- [ ] Locate your `OrchestratorPolicy` class
- [ ] Add `opportunity_store` to constructor
- [ ] Add `min_opportunity_score` config parameter
- [ ] Enhance `should_allow_trade()` method
- [ ] Add opportunity score check before allowing trade
- [ ] Return (False, reason) if score < threshold
- [ ] Add opportunity score to allow reason message
- [ ] Test with fake signal (BTCUSDT)
- [ ] Test with low-score symbol (should be blocked)
- [ ] Add logging for opportunity filtering

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## ✅ Phase 7: Integrate with Strategy Engine

- [ ] Locate your `StrategyEngine` class
- [ ] Add `opportunity_store` to constructor
- [ ] Create `get_active_symbols(max_symbols=10)` method
- [ ] Modify signal generation to use only top-ranked symbols
- [ ] Remove static symbol lists (if any)
- [ ] Test signal generation with dynamic symbol selection
- [ ] Verify only top N symbols generate signals
- [ ] Add logging for symbol selection

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## ✅ Phase 8: Integrate with MSC AI (Meta Strategy Controller)

- [ ] Locate your `MetaStrategyController` class
- [ ] Add `opportunity_store` to constructor
- [ ] Create `analyze_opportunity_landscape()` method
- [ ] Count symbols with score >= 0.7 (high opportunity)
- [ ] Adjust risk mode based on opportunity count:
  - [ ] Many opportunities (≥5) → AGGRESSIVE
  - [ ] Some opportunities (2-4) → NORMAL
  - [ ] Few opportunities (<2) → DEFENSIVE
- [ ] Adjust max_positions based on opportunities
- [ ] Add logging for policy adjustments
- [ ] Test with different market conditions

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## ✅ Phase 9: Monitoring & Logging

### Add Metrics
- [ ] Track number of symbols passing threshold
- [ ] Track average opportunity score
- [ ] Track top 5 symbol stability (how often it changes)
- [ ] Track update execution time
- [ ] Track API endpoint usage

### Add Alerts
- [ ] Alert if no symbols pass threshold
- [ ] Alert if update fails multiple times
- [ ] Alert if update takes > 60 seconds
- [ ] Alert if Redis connection fails

### Dashboard Integration (Optional)
- [ ] Add opportunity rankings to admin dashboard
- [ ] Show top 10 symbols with scores
- [ ] Show score distribution histogram
- [ ] Show metric breakdown per symbol
- [ ] Add refresh button

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## ✅ Phase 10: Validation & Tuning

### Validate Scores Make Sense
- [ ] Review scores for BTCUSDT (should be high)
- [ ] Review scores for low-volume altcoins (should be low)
- [ ] Check trending symbols have high trend_strength
- [ ] Check choppy symbols have low scores
- [ ] Verify regime score matches current regime

### Backtest Integration
- [ ] Run backtest with opportunity filtering enabled
- [ ] Compare results to baseline (no filtering)
- [ ] Expected: 10-30% improvement in Sharpe ratio
- [ ] Analyze which symbols were filtered out
- [ ] Verify filtering avoided losing trades

### Tune Weights (If Needed)
- [ ] Analyze correlation between each metric and profitability
- [ ] Increase weight for highly correlated metrics
- [ ] Decrease weight for low-correlation metrics
- [ ] Re-run backtest with tuned weights
- [ ] Document final weight configuration

### Tune Threshold (If Needed)
- [ ] If too few symbols pass: Lower min_score to 0.4
- [ ] If too many symbols pass: Raise min_score to 0.6
- [ ] Target: 40-60% of symbols pass threshold
- [ ] Document final threshold

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## ✅ Phase 11: Production Deployment

### Pre-Deployment Checklist
- [ ] All tests pass
- [ ] No critical errors in logs
- [ ] Background scheduler running smoothly
- [ ] Redis connection stable
- [ ] API endpoints respond correctly
- [ ] Orchestrator integration working
- [ ] Strategy Engine integration working
- [ ] MSC AI integration working

### Deployment
- [ ] Deploy to staging environment
- [ ] Run for 24 hours in staging
- [ ] Monitor logs and metrics
- [ ] Verify no performance issues
- [ ] Deploy to production
- [ ] Enable monitoring and alerts

### Post-Deployment
- [ ] Monitor for first 48 hours continuously
- [ ] Check logs for errors
- [ ] Verify rankings update every 15 minutes
- [ ] Verify trades are filtered correctly
- [ ] Compare performance to baseline

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## ✅ Phase 12: Ongoing Maintenance

### Weekly
- [ ] Review opportunity score distribution
- [ ] Check for symbols consistently scoring low
- [ ] Review top 10 symbol changes
- [ ] Monitor update execution time

### Monthly
- [ ] Analyze correlation between score and profitability
- [ ] Tune weights if needed
- [ ] Review and update TRADEABLE_SYMBOLS list
- [ ] Check for new symbols to add

### Quarterly
- [ ] Comprehensive backtest with current weights
- [ ] Review metric effectiveness
- [ ] Consider new metrics to add
- [ ] Update documentation

**Status**: ⬜ Not Started | 🟨 In Progress | ✅ Complete

---

## 📊 Success Metrics

After integration, you should see:

### Immediate (Day 1)
- ✅ Rankings update successfully every 15 minutes
- ✅ 40-60% of symbols pass min_score threshold
- ✅ API endpoints return valid data
- ✅ No errors in logs

### Short-term (Week 1)
- ✅ Orchestrator filters 20-40% of signals (low-opportunity)
- ✅ Strategy Engine focuses on top 10-15 symbols
- ✅ MSC AI adjusts risk mode appropriately
- ✅ System trades only high-quality symbols

### Medium-term (Month 1)
- ✅ 10-30% improvement in Sharpe ratio
- ✅ Reduction in losing trades
- ✅ Higher average win size
- ✅ Lower drawdown during choppy markets

### Long-term (Quarter 1)
- ✅ Consistent outperformance vs baseline
- ✅ Stable opportunity score correlation with profit
- ✅ Reduced trading during poor market conditions
- ✅ System automatically adapts to regime changes

---

## 🚨 Troubleshooting Guide

### Issue: No symbols pass threshold
**Solution**: Lower `OPPORTUNITY_MIN_SCORE` to 0.4 or 0.3

### Issue: Rankings never update
**Solution**: Check background task is running, verify no exceptions in logs

### Issue: All symbols have similar scores
**Solution**: Verify market data is diverse (different symbols have different trends)

### Issue: Scores don't correlate with performance
**Solution**: Tune weights based on backtest analysis

### Issue: Update takes too long (>60 seconds)
**Solution**: Reduce `CANDLE_LIMIT` to 100, implement caching, or parallelize

### Issue: Redis connection fails
**Solution**: Check Redis is running, verify connection string, check firewall

---

## 📞 Need Help?

If you encounter issues not covered in this checklist:

1. Review the detailed README: `OPPORTUNITY_RANKER_README.md`
2. Check the quick-start guide: `OPPORTUNITY_RANKER_QUICKSTART.py`
3. Review the architecture diagram: `OPPORTUNITY_RANKER_ARCHITECTURE.md`
4. Run the examples: `python opportunity_ranker_example.py`
5. Check the test file for usage patterns: `test_opportunity_ranker.py`

---

## 🎉 Completion

When all phases are complete, you will have:

✅ A fully integrated OpportunityRanker system
✅ Automatic symbol quality filtering
✅ Improved trading performance
✅ Dynamic symbol selection
✅ Regime-aware opportunity analysis
✅ REST API for monitoring
✅ System-wide intelligence enhancement

**Congratulations on implementing OpportunityRanker!** 🚀

---

**Last Updated**: November 30, 2024
**Version**: 1.0.0
