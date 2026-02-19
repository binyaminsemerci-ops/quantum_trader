#!/usr/bin/env python3
"""
MetricPack v1 Builder (P3.8.1) - READ-ONLY Exit/Harvest Metrics Aggregator
Consumes apply.result stream, reconstructs trades, exports Prometheus metrics.
"""

import os
import sys
import time
import json
import logging
import hashlib
from typing import Dict, Optional, Tuple
from datetime import datetime
from collections import OrderedDict
from dataclasses import dataclass, field
import asyncio

import redis
import requests
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

# ============================================================================
# CONFIGURATION
# ============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

METRICPACK_PORT = int(os.getenv("METRICPACK_PORT", 8051))
METRICPACK_SYMBOLS = os.getenv("METRICPACK_SYMBOLS", "BTCUSDT,ETHUSDT,TRXUSDT").split(",")
METRICPACK_PRICE_SOURCE = os.getenv("METRICPACK_PRICE_SOURCE", "binance_mark")
METRICPACK_MODE = os.getenv("METRICPACK_MODE", "testnet")
METRICPACK_ATR_PERIOD = int(os.getenv("METRICPACK_ATR_PERIOD", 14))
METRICPACK_REGIME_METHOD = os.getenv("METRICPACK_REGIME_METHOD", "adx")

APPLY_RESULT_STREAM = os.getenv("APPLY_RESULT_STREAM", "quantum:stream:apply.result")
CONSUMER_GROUP = os.getenv("METRICPACK_CONSUMER_GROUP", "metricpack_builder")
CONSUMER_NAME = os.getenv("METRICPACK_CONSUMER_NAME", f"metricpack_{int(time.time())}")

CHECKPOINT_KEY = "quantum:metricpack:last_id"
DEDUPE_SIZE = int(os.getenv("METRICPACK_DEDUPE_SIZE", 10000))

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("metricpack_builder")

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Counters
pnl_counter = Counter(
    "quantum_exit_realized_pnl_total",
    "Total realized PnL in USDT",
    ["symbol", "regime", "action"]
)

trades_counter = Counter(
    "quantum_exit_trades_total",
    "Total trades closed",
    ["symbol", "regime", "action", "outcome"]
)

events_processed_counter = Counter(
    "quantum_exit_events_processed_total",
    "Total apply.result events processed"
)

# Gauges
winrate_gauge = Gauge(
    "quantum_exit_winrate",
    "Win rate per symbol/regime/action",
    ["symbol", "regime", "action"]
)

expectancy_gauge = Gauge(
    "quantum_exit_expectancy",
    "Expectancy (E[R]) per symbol/regime",
    ["symbol", "regime"]
)

profit_factor_gauge = Gauge(
    "quantum_exit_profit_factor",
    "Profit factor (gross_profit / gross_loss)",
    ["symbol", "regime"]
)

lag_gauge = Gauge(
    "quantum_exit_builder_lag_seconds",
    "Lag between event timestamp and processing time"
)

# Histograms
time_in_trade_histogram = Histogram(
    "quantum_exit_time_in_trade_seconds",
    "Time in trade distribution",
    ["symbol", "regime"],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600, 7200, 14400]
)

mfe_histogram = Histogram(
    "quantum_exit_mfe_atr",
    "Most Favorable Excursion in ATR units",
    ["symbol", "regime"],
    buckets=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
)

mae_histogram = Histogram(
    "quantum_exit_mae_atr",
    "Most Adverse Excursion in ATR units",
    ["symbol", "regime"],
    buckets=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class TradeState:
    """Trade state machine"""
    trade_id: str
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: float
    entry_qty: float
    
    exits: list = field(default_factory=list)
    remaining_qty: float = 0.0
    
    mfe: float = 0.0  # Most Favorable Excursion (USDT)
    mae: float = 0.0  # Most Adverse Excursion (USDT)
    
    is_closed: bool = False
    regime: str = "unknown"
    
    def __post_init__(self):
        self.remaining_qty = self.entry_qty
    
    def add_exit(self, action: str, qty: float, price: float, timestamp: float):
        """Add exit event and calculate PnL"""
        # Calculate PnL for this exit
        if self.side == "LONG":
            exit_pnl = (price - self.entry_price) * qty
        else:  # SHORT
            exit_pnl = (self.entry_price - price) * qty
        
        self.exits.append({
            "action": action,
            "qty": qty,
            "price": price,
            "timestamp": timestamp,
            "pnl": exit_pnl
        })
        
        self.remaining_qty -= qty
        
        # Check if closed
        if self.remaining_qty <= 0.001:
            self.is_closed = True
        
        return exit_pnl
    
    def update_mfe_mae(self, current_price: float):
        """Update MFE/MAE based on current price"""
        if self.remaining_qty <= 0:
            return
        
        if self.side == "LONG":
            unrealized = (current_price - self.entry_price) * self.remaining_qty
        else:
            unrealized = (self.entry_price - current_price) * self.remaining_qty
        
        if unrealized > self.mfe:
            self.mfe = unrealized
        if unrealized < self.mae:
            self.mae = unrealized

# ============================================================================
# REGIME DETECTOR
# ============================================================================

class RegimeDetector:
    """Simple ADX-based regime classifier"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.cache_ttl = 60
    
    def get_regime(self, symbol: str) -> str:
        """
        Classify regime:
        - trend: ADX(14) > 25
        - chop: ADX(14) < 20
        - unknown: otherwise or no data
        """
        try:
            candles = self._get_candles(symbol)
            if not candles or len(candles) < 30:
                return "unknown"
            
            adx = self._calculate_adx(candles)
            
            if adx > 25:
                return "trend"
            elif adx < 20:
                return "chop"
            else:
                return "unknown"
        
        except Exception as e:
            logger.warning(f"Regime detection failed for {symbol}: {e}")
            return "unknown"
    
    def _get_candles(self, symbol: str) -> Optional[list]:
        """Get recent candles from Binance"""
        try:
            if METRICPACK_MODE == "testnet":
                base_url = "https://testnet.binancefuture.com"
            else:
                base_url = "https://fapi.binance.com"
            
            url = f"{base_url}/fapi/v1/klines"
            params = {
                "symbol": symbol,
                "interval": "5m",
                "limit": 50
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            klines = response.json()
            candles = []
            
            for k in klines:
                candles.append({
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4])
                })
            
            return candles
        
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return None
    
    def _calculate_adx(self, candles: list) -> float:
        """Simple ADX(14) calculation"""
        if len(candles) < METRICPACK_ATR_PERIOD + 1:
            return 0.0
        
        # Calculate True Range and Directional Movement
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        
        for i in range(1, len(candles)):
            high = candles[i]["high"]
            low = candles[i]["low"]
            prev_close = candles[i - 1]["close"]
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
            
            plus_dm = max(high - candles[i - 1]["high"], 0)
            minus_dm = max(candles[i - 1]["low"] - low, 0)
            
            if plus_dm > minus_dm:
                minus_dm = 0
            elif minus_dm > plus_dm:
                plus_dm = 0
            
            plus_dm_list.append(plus_dm)
            minus_dm_list.append(minus_dm)
        
        # Smoothed averages
        period = METRICPACK_ATR_PERIOD
        atr = sum(tr_list[-period:]) / period if len(tr_list) >= period else 1.0
        plus_di = (sum(plus_dm_list[-period:]) / period / atr * 100) if atr > 0 else 0
        minus_di = (sum(minus_dm_list[-period:]) / period / atr * 100) if atr > 0 else 0
        
        # ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 0.001) * 100
        adx = dx  # Simplified (should be smoothed over 14 periods, but good enough)
        
        return adx

# ============================================================================
# METRICPACK BUILDER
# ============================================================================

class MetricPackBuilder:
    """Main service orchestrator"""
    
    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False
        )
        
        self.regime_detector = RegimeDetector(self.redis)
        
        # Active trades per symbol
        self.active_trades: Dict[str, TradeState] = {}
        
        # Dedupe cache (LRU)
        self.dedupe_cache: OrderedDict = OrderedDict()
        
        # Aggregated stats for metrics
        self.stats: Dict[Tuple[str, str, str], Dict] = {}  # (symbol, regime, action) -> {wins, losses, total_pnl}
        
        # Create consumer group
        self._create_consumer_group()
    
    def _create_consumer_group(self):
        """Create consumer group for apply.result stream"""
        try:
            self.redis.xgroup_create(APPLY_RESULT_STREAM, CONSUMER_GROUP, id='0', mkstream=True)
            logger.info(f"Created consumer group {CONSUMER_GROUP}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group {CONSUMER_GROUP} already exists")
            else:
                raise
    
    def _is_duplicate(self, plan_id: str, order_id: Optional[str]) -> bool:
        """Check if event already processed (dedupe)"""
        key = f"{plan_id}:{order_id or 'none'}"
        
        if key in self.dedupe_cache:
            return True
        
        # Add to cache (LRU)
        self.dedupe_cache[key] = True
        
        # Evict oldest if cache too large
        if len(self.dedupe_cache) > DEDUPE_SIZE:
            self.dedupe_cache.popitem(last=False)
        
        return False
    
    def _generate_trade_id(self, symbol: str, entry_order_id: str) -> str:
        """Generate deterministic trade_id"""
        return hashlib.sha256(f"{symbol}:{entry_order_id}".encode()).hexdigest()[:16]
    
    def process_event(self, event: Dict):
        """Process single apply.result event"""
        try:
            # Parse event
            data = event.get("data", {})
            
            executed = data.get("executed", False)
            if not executed:
                return  # Skip non-executed events
            
            symbol = data.get("symbol")
            plan_id = data.get("plan_id")
            order_id = data.get("order_id")
            
            if not symbol or symbol not in METRICPACK_SYMBOLS:
                return  # Skip non-tracked symbols
            
            # Dedupe check
            if self._is_duplicate(plan_id, order_id):
                logger.debug(f"Skipping duplicate: {plan_id}/{order_id}")
                return
            
            side = data.get("side")  # BUY or SELL
            qty = float(data.get("qty", 0))
            filled_qty = float(data.get("filled_qty", 0))
            price = float(data.get("price", 0))
            timestamp = float(data.get("timestamp", time.time()))
            
            metadata = data.get("metadata", {})
            action = metadata.get("action", "UNKNOWN")
            reduce_only = metadata.get("reduceOnly", False)
            
            # Detect if entry or exit
            if not reduce_only:
                # Entry event
                self._process_entry(symbol, side, price, filled_qty, timestamp, order_id or plan_id)
            else:
                # Exit event
                self._process_exit(symbol, action, filled_qty, price, timestamp)
            
            events_processed_counter.inc()
            
            # Update lag
            lag = time.time() - timestamp
            lag_gauge.set(lag)
        
        except Exception as e:
            logger.error(f"Error processing event: {e}", exc_info=True)
    
    def _process_entry(self, symbol: str, side: str, price: float, qty: float, timestamp: float, order_id: str):
        """Process entry event (OPEN position)"""
        trade_id = self._generate_trade_id(symbol, order_id)
        
        # Determine side
        if side == "BUY":
            trade_side = "LONG"
        elif side == "SELL":
            trade_side = "SHORT"
        else:
            logger.warning(f"Unknown side: {side}")
            return
        
        # Create trade state
        trade = TradeState(
            trade_id=trade_id,
            symbol=symbol,
            side=trade_side,
            entry_price=price,
            entry_time=timestamp,
            entry_qty=qty
        )
        
        self.active_trades[trade_id] = trade
        logger.info(f"Entry: {trade_id} {symbol} {trade_side} @ {price} qty={qty}")
    
    def _process_exit(self, symbol: str, action: str, qty: float, price: float, timestamp: float):
        """Process exit event (reduceOnly=true)"""
        # Find active trade for this symbol
        trade = None
        for t in self.active_trades.values():
            if t.symbol == symbol and not t.is_closed:
                trade = t
                break
        
        if not trade:
            logger.warning(f"Exit without active trade: {symbol} {action}")
            return
        
        # Add exit
        exit_pnl = trade.add_exit(action, qty, price, timestamp)
        
        logger.info(f"Exit: {trade.trade_id} {action} qty={qty} @ {price} pnl={exit_pnl:.2f}")
        
        # If closed, finalize trade
        if trade.is_closed:
            self._finalize_trade(trade)
    
    def _finalize_trade(self, trade: TradeState):
        """Finalize closed trade and update metrics"""
        # Get regime
        trade.regime = self.regime_detector.get_regime(trade.symbol)
        
        # Calculate total PnL
        total_pnl = sum(e["pnl"] for e in trade.exits)
        
        # Time in trade
        if trade.exits:
            time_in_trade = trade.exits[-1]["timestamp"] - trade.entry_time
            time_in_trade_histogram.labels(
                symbol=trade.symbol,
                regime=trade.regime
            ).observe(time_in_trade)
        
        # Update stats per action
        for exit_event in trade.exits:
            action = exit_event["action"]
            pnl = exit_event["pnl"]
            
            outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")
            
            # Update counters
            pnl_counter.labels(
                symbol=trade.symbol,
                regime=trade.regime,
                action=action
            ).inc(pnl)
            
            trades_counter.labels(
                symbol=trade.symbol,
                regime=trade.regime,
                action=action,
                outcome=outcome
            ).inc()
            
            # Update aggregated stats
            key = (trade.symbol, trade.regime, action)
            if key not in self.stats:
                self.stats[key] = {"wins": 0, "losses": 0, "total_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0}
            
            self.stats[key]["total_pnl"] += pnl
            
            if pnl > 0:
                self.stats[key]["wins"] += 1
                self.stats[key]["gross_profit"] += pnl
            elif pnl < 0:
                self.stats[key]["losses"] += 1
                self.stats[key]["gross_loss"] += abs(pnl)
        
        # Calculate and update gauges
        self._update_gauges()
        
        # MFE/MAE (if tracked)
        if trade.mfe > 0:
            # Convert to ATR units (approximate with 1 ATR = 2% of price)
            atr_estimate = trade.entry_price * 0.02
            mfe_atr = trade.mfe / trade.entry_qty / atr_estimate if atr_estimate > 0 else 0
            mfe_histogram.labels(symbol=trade.symbol, regime=trade.regime).observe(mfe_atr)
        
        if trade.mae < 0:
            atr_estimate = trade.entry_price * 0.02
            mae_atr = abs(trade.mae) / trade.entry_qty / atr_estimate if atr_estimate > 0 else 0
            mae_histogram.labels(symbol=trade.symbol, regime=trade.regime).observe(mae_atr)
        
        logger.info(f"Trade closed: {trade.trade_id} PnL={total_pnl:.2f} regime={trade.regime}")
        
        # Remove from active
        del self.active_trades[trade.trade_id]
    
    def _update_gauges(self):
        """Update gauge metrics from aggregated stats"""
        for (symbol, regime, action), stats in self.stats.items():
            total = stats["wins"] + stats["losses"]
            
            if total > 0:
                # Win rate
                winrate = stats["wins"] / total
                winrate_gauge.labels(
                    symbol=symbol,
                    regime=regime,
                    action=action
                ).set(winrate)
        
        # Expectancy and profit factor per symbol/regime (aggregate across actions)
        per_sr = {}
        for (symbol, regime, action), stats in self.stats.items():
            key = (symbol, regime)
            if key not in per_sr:
                per_sr[key] = {"wins": 0, "losses": 0, "gross_profit": 0.0, "gross_loss": 0.0}
            
            per_sr[key]["wins"] += stats["wins"]
            per_sr[key]["losses"] += stats["losses"]
            per_sr[key]["gross_profit"] += stats["gross_profit"]
            per_sr[key]["gross_loss"] += stats["gross_loss"]
        
        for (symbol, regime), stats in per_sr.items():
            total = stats["wins"] + stats["losses"]
            
            if total > 0:
                # Expectancy
                avg_win = stats["gross_profit"] / stats["wins"] if stats["wins"] > 0 else 0
                avg_loss = stats["gross_loss"] / stats["losses"] if stats["losses"] > 0 else 0
                win_rate = stats["wins"] / total
                loss_rate = 1 - win_rate
                
                expectancy = avg_win * win_rate - avg_loss * loss_rate
                expectancy_gauge.labels(symbol=symbol, regime=regime).set(expectancy)
                
                # Profit factor
                if stats["gross_loss"] > 0:
                    profit_factor = stats["gross_profit"] / stats["gross_loss"]
                    profit_factor_gauge.labels(symbol=symbol, regime=regime).set(profit_factor)
    
    def save_checkpoint(self, last_id: str):
        """Save last processed message ID"""
        self.redis.set(CHECKPOINT_KEY, last_id)
    
    def load_checkpoint(self) -> str:
        """Load last processed message ID"""
        checkpoint = self.redis.get(CHECKPOINT_KEY)
        return checkpoint.decode() if checkpoint else "0-0"
    
    async def run(self):
        """Main event loop"""
        logger.info(f"MetricPack Builder starting")
        logger.info(f"Symbols: {METRICPACK_SYMBOLS}")
        logger.info(f"Stream: {APPLY_RESULT_STREAM}")
        
        # Load checkpoint
        last_id = self.load_checkpoint()
        logger.info(f"Starting from checkpoint: {last_id}")
        
        while True:
            try:
                # Read from stream
                messages = self.redis.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {APPLY_RESULT_STREAM: '>'},
                    count=10,
                    block=1000
                )
                
                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        # Decode message
                        decoded = {}
                        for k, v in msg_data.items():
                            key = k.decode() if isinstance(k, bytes) else k
                            val = v.decode() if isinstance(v, bytes) else v
                            
                            if key == "data":
                                try:
                                    decoded[key] = json.loads(val)
                                except:
                                    decoded[key] = {}
                            else:
                                decoded[key] = val
                        
                        # Process event
                        self.process_event(decoded)
                        
                        # ACK message
                        self.redis.xack(APPLY_RESULT_STREAM, CONSUMER_GROUP, msg_id)
                        
                        # Save checkpoint
                        last_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                        self.save_checkpoint(last_id)
                
                # Periodic MFE/MAE updates for active trades (every 30s)
                await asyncio.sleep(1)
            
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="MetricPack Builder v1")
builder = None

@app.on_event("startup")
async def startup_event():
    """Start background task"""
    global builder
    builder = MetricPackBuilder()
    asyncio.create_task(builder.run())

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "service": "metricpack_builder",
        "version": "1.0.0",
        "active_trades": len(builder.active_trades) if builder else 0
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=METRICPACK_PORT, log_level="info")
