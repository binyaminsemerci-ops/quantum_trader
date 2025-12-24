# EVENT FLOW MAP — Redis Streams Architecture
**Audit Date**: December 24, 2025 05:02 UTC

## REDIS STREAMS INVENTORY

**Total Streams**: 21 active  
**Total Messages**: 10,000+ events per major stream  
**Consumer Groups**: quantum:group:execution:trade.intent (LAG: 10,014 ⚠️)

---

## STREAM CATALOG

### Market Data Streams (5)
`
quantum:stream:market_data           [Producer: market_publisher]
quantum:stream:market.tick           [Producer: market_publisher]
quantum:stream:market.klines         [Producer: market_publisher]
quantum:stream:exchange.raw          [Producer: cross_exchange]
quantum:stream:exchange.normalized   [Producer: cross_exchange]
`

### AI & Signals (2)
`
quantum:stream:ai.signal_generated   [Producer: ai_engine]
quantum:stream:ai.decision.made      [Producer: ai_engine]
`

### Trading & Execution (4)
`
quantum:stream:trade.intent          [Producer: trading_bot] ← ILF metadata here
  ├─ Consumers: quantum:group:execution:trade.intent (34 consumers, LAG 10,014)
  └─ Latest event sample:
     {
        symbol: NEARUSDT,
       side: BUY,
       confidence: 0.72,
       position_size_usd: 200.0,
       leverage: 1,
       atr_value: 0.02,
       volatility_factor: 0.55,
       exchange_divergence: 0.0,
       funding_rate: 0.0,
       regime: unknown
     }

quantum:stream:execution.result      [Producer: execution_adapter]
quantum:stream:trade.closed          [Producer: execution/exitbrain]
quantum:stream:sizing.decided        [Producer: rl_sizing_agent]
`

### Portfolio Management (2)
`
quantum:stream:portfolio.snapshot_updated    [Producer: portfolio_intelligence]
quantum:stream:portfolio.exposure_updated    [Producer: portfolio_intelligence]
`

### Governance & Policy (2)
`
quantum:stream:policy.updated        [Producer: governance_brains]
quantum:stream:meta.regime          [Producer: meta_regime detector]
`

### Learning & Training (6)
`
quantum:stream:rl_v3.training.started       [Producer: rl_optimizer]
quantum:stream:rl_v3.training.completed     [Producer: rl_optimizer]
quantum:stream:learning.retraining.started  [Producer: clm]
quantum:stream:learning.retraining.completed [Producer: clm]
quantum:stream:learning.retraining.failed   [Producer: clm]
quantum:stream:model.retrain                [Producer: model_supervisor]
`

---

## EVENT FLOW DIAGRAM (ORDER LIFECYCLE)

`
┌─────────────────────┐
│ Market Publisher    │ → quantum:stream:market_data
│ (market feed)       │ → quantum:stream:market.tick
└─────────────────────┘ → quantum:stream:market.klines
           │
           ▼
┌─────────────────────┐
│ AI Engine           │ ← reads market_data
│ (ML predictions)    │ → quantum:stream:ai.signal_generated
└─────────────────────┘ → quantum:stream:ai.decision.made
           │
           ▼
┌─────────────────────┐
│ Trading Bot         │ ← reads ai.signal_generated
│ (signal processor)  │ ← calculates ILF metadata
│                     │ ← calls RL Sizing Agent
│                     │ → quantum:stream:trade.intent (WITH ILF)
└─────────────────────┘ → quantum:stream:sizing.decided
           │
           ▼
┌─────────────────────┐
│ ⚠️ MISSING CONSUMER │ ← should read trade.intent
│ Trade Executor      │ ← should forward ILF to ExitBrain
│ (execution layer)   │ ← should open positions
│ STATUS: NO PROCESS  │ → quantum:stream:execution.result
└─────────────────────┘
           │
           ▼ (IF EXECUTED)
┌─────────────────────┐
│ Position Monitor    │ ← reads execution.result
│ ExitBrain v3        │ ← monitors positions
│ (TP/SL management)  │ → quantum:stream:trade.closed
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Trade Journal       │ ← reads trade.closed
│ CLM / RL Optimizer  │ ← records PnL
│ (learning feedback) │ → quantum:stream:learning.retraining.*
└─────────────────────┘ → quantum:stream:model.retrain
`

---

## CRITICAL FINDINGS

### 🚨 CONSUMER LAG DETECTED
**Stream**: quantum:stream:trade.intent  
**Consumer Group**: quantum:group:execution:trade.intent  
**LAG**: 10,014 unprocessed messages  
**Impact**: Trade intents are published but NOT consumed → NO TRADES EXECUTED

**Evidence**:
`
$ docker exec quantum_redis redis-cli XINFO GROUPS 'quantum:stream:trade.intent'
name: quantum:group:execution:trade.intent
consumers: 34
pending: 1
last-delivered-id: 1766219478937-0
entries-read: 45409
lag: 10014  ← 10,014 events not processed!
`

### ✅ ILF METADATA PRESENT
Trade.intent events contain complete ILF metadata:
- atr_value: 0.02
- volatility_factor: 0.5-0.6
- exchange_divergence: 0.0
- funding_rate: 0.0
- regime: unknown

**But**: No consumer is reading this metadata and using it!

---

## PRODUCER/CONSUMER MAP

### Producers
- 	rading_bot → trade.intent, sizing.decided
- i_engine → ai.signal_generated, ai.decision.made
- market_publisher → market_data, market.tick, market.klines
- cross_exchange → exchange.raw, exchange.normalized
- l_optimizer → rl_v3.training.*
- clm → learning.retraining.*
- portfolio_intelligence → portfolio.snapshot_updated, portfolio.exposure_updated

### Consumers
- quantum:group:execution:trade.intent → 34 consumers registered (STALE)
- **GAP**: No active consumer processing trade.intent stream
- **GAP**: ExitBrain not receiving ILF metadata
- **GAP**: Adaptive leverage never calculated

---

## REDIS HEALTH

**Memory Usage**: 91MB  
**Commands Processed**: High (stats saved in raw/redis_info_stats.txt)  
**Client Connections**: Multiple (list saved in raw/redis_client_list.txt)  
**Slowlog**: No critical slow queries (saved in raw/redis_slowlog.txt)  
**Status**: ✅ HEALTHY (but consumers are not processing events)

---

**Audit Conclusion**: Event infrastructure is OPERATIONAL, but execution layer is DISCONNECTED.
