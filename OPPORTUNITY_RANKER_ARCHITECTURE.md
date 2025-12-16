# OpportunityRanker System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          QUANTUM TRADER SYSTEM                              │
│                    (Enhanced with OpportunityRanker)                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES (External)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │   Binance    │   │   OrderBook  │   │  Trade Logs  │   │   Market     │ │
│  │   OHLCV      │   │   Spread     │   │   (Postgres) │   │   Regime     │ │
│  │   Data       │   │   Depth      │   │   Winrates   │   │   Detector   │ │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘ │
└─────────┼──────────────────┼──────────────────┼──────────────────┼─────────┘
          │                  │                  │                  │
          ▼                  ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        PROTOCOL INTERFACES                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────┐│
│  │ MarketData     │  │ TradeLog       │  │ Regime         │  │Opportunity ││
│  │ Client         │  │ Repository     │  │ Detector       │  │Store       ││
│  │                │  │                │  │                │  │            ││
│  │ get_candles()  │  │ get_winrate()  │  │ get_regime()   │  │ update()   ││
│  │ get_spread()   │  │                │  │                │  │ get()      ││
│  │ get_liquidity()│  │                │  │                │  │            ││
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘  └─────┬──────┘│
└───────────┼──────────────────┼──────────────────┼─────────────────┼────────┘
            │                  │                  │                 │
            └──────────────────┴──────────────────┴─────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     OPPORTUNITY RANKER (OppRank)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      METRIC CALCULATORS                                │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │   Trend     │  │ Volatility  │  │  Liquidity  │  │   Spread    │  │ │
│  │  │  Strength   │  │   Quality   │  │    Score    │  │    Score    │  │ │
│  │  │             │  │             │  │             │  │             │  │ │
│  │  │  EMA slope  │  │  ATR range  │  │  24h volume │  │  Bid/ask    │  │ │
│  │  │  HH/HL      │  │  Stability  │  │  Depth      │  │  spread %   │  │ │
│  │  │  Alignment  │  │             │  │             │  │             │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ │
│  │                                                                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │ │
│  │  │   Symbol    │  │   Regime    │  │    Noise    │                   │ │
│  │  │  Winrate    │  │   Score     │  │    Score    │                   │ │
│  │  │             │  │             │  │             │                   │ │
│  │  │  Historical │  │  Alignment  │  │  Wick ratio │                   │ │
│  │  │  trades %   │  │  BULL/BEAR  │  │  Variance   │                   │ │
│  │  │             │  │             │  │             │                   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      SCORE AGGREGATOR                                  │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  final_score = 0.25×trend + 0.20×volatility + 0.15×liquidity +        │ │
│  │                0.15×regime + 0.10×winrate + 0.10×spread + 0.05×noise  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    RANKING ENGINE                                      │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  • Sort symbols by score (descending)                                  │ │
│  │  • Filter by min_score_threshold                                       │ │
│  │  • Store in OpportunityStore                                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
└─────────────────────────────────────┼────────────────────────────────────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      │               │               │
                      ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         OPPORTUNITY STORE (Redis)                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  Rankings: {                                                                 │
│    "BTCUSDT": 0.87,   ← Highest opportunity                                 │
│    "SOLUSDT": 0.82,                                                          │
│    "ETHUSDT": 0.79,                                                          │
│    "AVAXUSDT": 0.68,                                                         │
│    "BNBUSDT": 0.65,                                                          │
│    ...                                                                       │
│  }                                                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                      │               │               │
          ┌───────────┴───────┐   ┌───┴───────┐   ┌─┴───────────┐
          │                   │   │           │   │             │
          ▼                   ▼   ▼           ▼   ▼             ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Orchestrator   │  │  Strategy Engine │  │    MSC AI        │
│     Policy       │  │                  │  │                  │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│                  │  │                  │  │                  │
│ should_allow()   │  │ get_active_      │  │ adjust_global_   │
│                  │  │ symbols()        │  │ policy()         │
│ ✓ Check opp      │  │                  │  │                  │
│   score >= 0.5   │  │ ✓ Use top 10     │  │ ✓ Set AGGRESSIVE │
│                  │  │   ranked symbols │  │   if many high-  │
│ ✓ Block if low   │  │                  │  │   opp symbols    │
│                  │  │ ✓ Prioritize by  │  │                  │
│                  │  │   score          │  │ ✓ Set DEFENSIVE  │
│                  │  │                  │  │   if few good    │
│                  │  │                  │  │   symbols        │
└──────────────────┘  └──────────────────┘  └──────────────────┘
          │                   │                      │
          └───────────────────┴──────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          TRADING EXECUTION                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│  • Only high-opportunity symbols get traded                                  │
│  • Low-quality symbols automatically filtered                                │
│  • Risk adjusted based on opportunity landscape                              │
│  • Focus on top 10-15 symbols dynamically                                    │
└──────────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                           REST API LAYER                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│  GET  /api/opportunities/rankings          → All current rankings           │
│  GET  /api/opportunities/rankings/top?n=10 → Top N symbols                  │
│  GET  /api/opportunities/rankings/{symbol} → Specific symbol score          │
│  GET  /api/opportunities/rankings/{symbol}/details → Metric breakdown       │
│  POST /api/opportunities/refresh           → Manual update trigger          │
└──────────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                        BACKGROUND SCHEDULER                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  Every 15 minutes:                                                           │
│    1. Fetch latest market data for all symbols                              │
│    2. Compute 7 metrics per symbol                                           │
│    3. Aggregate scores                                                       │
│    4. Sort and filter                                                        │
│    5. Update OpportunityStore                                                │
│    6. Log results                                                            │
└──────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                              DATA FLOW EXAMPLE
═══════════════════════════════════════════════════════════════════════════════

Step 1: Background Scheduler triggers update
        ↓
Step 2: OpportunityRanker fetches market data for BTCUSDT
        ↓
Step 3: Compute metrics:
        • Trend Strength: 0.85 (strong uptrend)
        • Volatility Quality: 0.92 (optimal ATR)
        • Liquidity: 1.0 (excellent volume)
        • Spread: 0.98 (tight spread)
        • Winrate: 0.68 (68% historical)
        • Regime: 1.0 (aligned with BULL)
        • Noise: 0.75 (clean price action)
        ↓
Step 4: Aggregate: 0.25×0.85 + 0.20×0.92 + ... = 0.87
        ↓
Step 5: Store: {"BTCUSDT": 0.87} in Redis
        ↓
Step 6: Orchestrator checks new signal for BTCUSDT
        ↓
Step 7: Retrieves score: 0.87 >= 0.5 threshold
        ↓
Step 8: Trade ALLOWED (high opportunity)


═══════════════════════════════════════════════════════════════════════════════
                            KEY BENEFITS
═══════════════════════════════════════════════════════════════════════════════

✅ Objective Symbol Quality Assessment
   → No more guessing which symbols to trade

✅ Automatic Low-Quality Filtering
   → System avoids choppy, illiquid, high-spread symbols

✅ Dynamic Symbol Selection
   → Adapts to changing market conditions every 15 minutes

✅ Multi-Factor Analysis
   → Combines 7 different quality metrics

✅ Regime-Aware
   → Prioritizes symbols aligned with global market regime

✅ System-Wide Intelligence
   → Orchestrator, Strategy Engine, MSC AI all benefit

✅ Measurable Impact
   → Track correlation between opportunity score and profitability


═══════════════════════════════════════════════════════════════════════════════
                         INTEGRATION SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Required Dependencies:
  ✓ MarketDataClient (Binance/CCXT)
  ✓ TradeLogRepository (PostgreSQL)
  ✓ RegimeDetector (existing service)
  ✓ OpportunityStore (Redis)

Configuration:
  ✓ Add to config.py
  ✓ Set TRADEABLE_SYMBOLS list
  ✓ Set update interval (15 min recommended)
  ✓ Set min score threshold (0.5 recommended)

Startup:
  ✓ Initialize OpportunityRanker
  ✓ Compute initial rankings
  ✓ Start background scheduler

Integration Points:
  ✓ Orchestrator: Filter by opportunity score
  ✓ Strategy Engine: Use top-ranked symbols
  ✓ MSC AI: Adjust policy based on opportunity landscape

Monitoring:
  ✓ REST API for current rankings
  ✓ Logs for periodic updates
  ✓ Metrics for score distribution
  ✓ Correlation tracking (score vs profitability)


═══════════════════════════════════════════════════════════════════════════════
