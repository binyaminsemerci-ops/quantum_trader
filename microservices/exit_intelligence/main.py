#!/usr/bin/env python3
"""
Exit Intelligence Layer - READ-ONLY Observability Microservice
Measures exit quality, PnL, MFE, MAE, regime-tagged performance.
NEVER publishes to trading streams. Pure telemetry.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import pandas as pd
import ta

# ============================================================================
# CONFIGURATION
# ============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "quantum")
POSTGRES_USER = os.getenv("POSTGRES_USER", "quantum")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

METRICS_PORT = int(os.getenv("EXIT_INTELLIGENCE_METRICS_PORT", 9109))

APPLY_RESULT_STREAM = os.getenv("APPLY_RESULT_STREAM", "quantum:stream:apply.result")
EXECUTION_RESULT_STREAM = os.getenv("EXECUTION_RESULT_STREAM", "quantum:stream:execution.result")

CONSUMER_GROUP = os.getenv("EXIT_INTELLIGENCE_CONSUMER_GROUP", "exit_intelligence")
CONSUMER_NAME = os.getenv("EXIT_INTELLIGENCE_CONSUMER_NAME", f"exit_intelligence_{int(time.time())}")

CANDLE_INTERVAL = os.getenv("CANDLE_INTERVAL", "1m")
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", 100))

REGIME_ADX_TREND_THRESHOLD = float(os.getenv("REGIME_ADX_TREND_THRESHOLD", 25))
REGIME_ADX_CHOP_THRESHOLD = float(os.getenv("REGIME_ADX_CHOP_THRESHOLD", 20))
REGIME_EMA_SPREAD_THRESHOLD = float(os.getenv("REGIME_EMA_SPREAD_THRESHOLD", 0.0015))

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("exit_intelligence")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ExitEvent:
    """Single exit (partial or full)"""
    exit_type: str  # "partial_25" / "partial_50" / "partial_75" / "full"
    qty: float
    price: float
    time: float
    fees: float = 0.0
    pnl: float = 0.0

@dataclass
class TradeLifecycle:
    """Complete trade from entry to final exit"""
    trade_id: str
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    entry_time: float
    entry_qty: float
    
    exits: List[ExitEvent] = field(default_factory=list)
    
    mfe: float = 0.0  # Most Favorable Excursion (USDT)
    mae: float = 0.0  # Most Adverse Excursion (USDT)
    pnl_realized: float = 0.0
    pnl_percent: float = 0.0
    time_in_trade: float = 0.0
    
    regime: str = "unknown"  # "trend" / "chop" / "unknown"
    
    is_closed: bool = False
    remaining_qty: float = 0.0
    
    def __post_init__(self):
        self.remaining_qty = self.entry_qty
    
    def add_exit(self, exit_event: ExitEvent):
        """Add exit and recalculate metrics"""
        self.exits.append(exit_event)
        self.remaining_qty -= exit_event.qty
        
        # Calculate PnL for this exit
        if self.side == "LONG":
            exit_pnl = (exit_event.price - self.entry_price) * exit_event.qty
        else:  # SHORT
            exit_pnl = (self.entry_price - exit_event.price) * exit_event.qty
        
        exit_pnl -= exit_event.fees
        exit_event.pnl = exit_pnl
        self.pnl_realized += exit_pnl
        
        # Time in trade (use last exit time)
        self.time_in_trade = exit_event.time - self.entry_time
        
        # PnL percent (based on entry notional)
        entry_notional = self.entry_price * self.entry_qty
        self.pnl_percent = (self.pnl_realized / entry_notional) * 100 if entry_notional > 0 else 0.0
        
        # Check if fully closed
        if self.remaining_qty <= 0.001:  # Allow small rounding errors
            self.is_closed = True
    
    def update_mfe_mae(self, current_price: float):
        """Update MFE and MAE based on current price"""
        if self.side == "LONG":
            unrealized = (current_price - self.entry_price) * self.remaining_qty
        else:  # SHORT
            unrealized = (self.entry_price - current_price) * self.remaining_qty
        
        if unrealized > self.mfe:
            self.mfe = unrealized
        if unrealized < self.mae:
            self.mae = unrealized

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Gauges for latest values
pnl_gauge = Gauge(
    "quantum_exit_pnl_usdt",
    "Realized PnL in USDT",
    ["symbol", "regime", "side"]
)

mfe_gauge = Gauge(
    "quantum_exit_mfe_usdt",
    "Most Favorable Excursion in USDT",
    ["symbol", "regime", "side"]
)

mae_gauge = Gauge(
    "quantum_exit_mae_usdt",
    "Most Adverse Excursion in USDT",
    ["symbol", "regime", "side"]
)

time_in_trade_gauge = Gauge(
    "quantum_exit_time_seconds",
    "Time in trade (seconds)",
    ["symbol", "regime", "side"]
)

partial_winrate_gauge = Gauge(
    "quantum_exit_partial_winrate",
    "Partial exit win rate",
    ["symbol", "regime", "exit_type"]
)

expectancy_gauge = Gauge(
    "quantum_exit_expectancy",
    "Expectancy (avg_win * win_rate - avg_loss * loss_rate)",
    ["symbol", "regime"]
)

exit_efficiency_gauge = Gauge(
    "quantum_exit_efficiency",
    "Exit price / MFE price ratio",
    ["symbol", "regime", "exit_type"]
)

# Counters
trades_closed_counter = Counter(
    "quantum_exit_trades_closed_total",
    "Total trades closed",
    ["symbol", "regime", "side"]
)

exits_total_counter = Counter(
    "quantum_exit_exits_total",
    "Total exits (partial + full)",
    ["symbol", "exit_type"]
)

# Histograms
pnl_histogram = Histogram(
    "quantum_exit_pnl_histogram",
    "PnL distribution",
    ["symbol", "regime"],
    buckets=[-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000]
)

# ============================================================================
# REGIME ENGINE
# ============================================================================

class RegimeEngine:
    """Simple regime classification using ADX and EMA"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.cache_ttl = 60  # Cache candles for 60 seconds
    
    def get_regime(self, symbol: str) -> str:
        """
        Determine market regime:
        - trend: ADX(14) > 25 AND |EMA20 - EMA50| / price > 0.0015
        - chop: ADX(14) < 20 OR Bollinger Width < threshold
        - unknown: otherwise or insufficient data
        """
        try:
            df = self._get_candles(symbol)
            if df is None or len(df) < 50:
                return "unknown"
            
            # Calculate indicators
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            df['adx'] = adx.adx()
            
            df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            
            bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_width'] = (bollinger.bollinger_hband() - bollinger.bollinger_lband()) / df['close']
            
            # Latest values
            latest = df.iloc[-1]
            adx_val = latest['adx']
            ema_spread = abs(latest['ema20'] - latest['ema50']) / latest['close']
            bb_width = latest['bb_width']
            
            # Classification
            if adx_val > REGIME_ADX_TREND_THRESHOLD and ema_spread > REGIME_EMA_SPREAD_THRESHOLD:
                return "trend"
            elif adx_val < REGIME_ADX_CHOP_THRESHOLD or bb_width < 0.01:
                return "chop"
            else:
                return "unknown"
        
        except Exception as e:
            logger.warning(f"Regime calculation failed for {symbol}: {e}")
            return "unknown"
    
    def _get_candles(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get candles from Redis cache or fetch from Binance"""
        cache_key = f"quantum:candles:{symbol}:{CANDLE_INTERVAL}"
        
        # Try cache first
        cached = self.redis.get(cache_key)
        if cached:
            try:
                data = json.loads(cached)
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            except Exception as e:
                logger.warning(f"Cache parse failed for {symbol}: {e}")
        
        # Fetch from Binance (fallback)
        try:
            import requests
            url = f"https://fapi.binance.com/fapi/v1/klines"
            params = {
                "symbol": symbol,
                "interval": CANDLE_INTERVAL,
                "limit": CANDLE_LIMIT
            }
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            klines = response.json()
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Cache for next time
            cache_data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict(orient='records')
            self.redis.setex(cache_key, self.cache_ttl, json.dumps(cache_data))
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return None

# ============================================================================
# STORAGE LAYER
# ============================================================================

class StorageLayer:
    """Dual storage: Postgres (preferred) + Redis (fallback)"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.postgres_conn = None
        self.use_postgres = self._init_postgres()
    
    def _init_postgres(self) -> bool:
        """Initialize Postgres connection if available"""
        if not POSTGRES_HOST or not POSTGRES_PASSWORD:
            logger.info("Postgres not configured, using Redis only")
            return False
        
        try:
            self.postgres_conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD
            )
            self.postgres_conn.autocommit = True
            logger.info("Postgres connection established")
            
            # Create tables
            self._create_postgres_tables()
            return True
        
        except Exception as e:
            logger.warning(f"Postgres connection failed: {e}. Using Redis only.")
            return False
    
    def _create_postgres_tables(self):
        """Create Postgres tables if they don't exist"""
        with self.postgres_conn.cursor() as cur:
            # Trades table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS exit_metrics_trades (
                    trade_id VARCHAR(100) PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    entry_price NUMERIC NOT NULL,
                    entry_time NUMERIC NOT NULL,
                    entry_qty NUMERIC NOT NULL,
                    
                    pnl_realized NUMERIC DEFAULT 0,
                    pnl_percent NUMERIC DEFAULT 0,
                    mfe NUMERIC DEFAULT 0,
                    mae NUMERIC DEFAULT 0,
                    time_in_trade NUMERIC DEFAULT 0,
                    
                    regime VARCHAR(20) DEFAULT 'unknown',
                    is_closed BOOLEAN DEFAULT FALSE,
                    remaining_qty NUMERIC DEFAULT 0,
                    
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Partials table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS exit_metrics_partials (
                    id SERIAL PRIMARY KEY,
                    trade_id VARCHAR(100) NOT NULL,
                    exit_type VARCHAR(20) NOT NULL,
                    qty NUMERIC NOT NULL,
                    price NUMERIC NOT NULL,
                    time NUMERIC NOT NULL,
                    fees NUMERIC DEFAULT 0,
                    pnl NUMERIC DEFAULT 0,
                    
                    created_at TIMESTAMP DEFAULT NOW(),
                    
                    FOREIGN KEY (trade_id) REFERENCES exit_metrics_trades(trade_id)
                )
            """)
            
            logger.info("Postgres tables created/verified")
    
    def save_trade(self, trade: TradeLifecycle):
        """Save trade to Postgres and/or Redis"""
        if self.use_postgres:
            self._save_trade_postgres(trade)
        
        # Always save to Redis as backup
        self._save_trade_redis(trade)
    
    def _save_trade_postgres(self, trade: TradeLifecycle):
        """Save to Postgres"""
        try:
            with self.postgres_conn.cursor() as cur:
                # Upsert trade
                cur.execute("""
                    INSERT INTO exit_metrics_trades (
                        trade_id, symbol, side, entry_price, entry_time, entry_qty,
                        pnl_realized, pnl_percent, mfe, mae, time_in_trade,
                        regime, is_closed, remaining_qty, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (trade_id) DO UPDATE SET
                        pnl_realized = EXCLUDED.pnl_realized,
                        pnl_percent = EXCLUDED.pnl_percent,
                        mfe = EXCLUDED.mfe,
                        mae = EXCLUDED.mae,
                        time_in_trade = EXCLUDED.time_in_trade,
                        regime = EXCLUDED.regime,
                        is_closed = EXCLUDED.is_closed,
                        remaining_qty = EXCLUDED.remaining_qty,
                        updated_at = NOW()
                """, (
                    trade.trade_id, trade.symbol, trade.side,
                    trade.entry_price, trade.entry_time, trade.entry_qty,
                    trade.pnl_realized, trade.pnl_percent,
                    trade.mfe, trade.mae, trade.time_in_trade,
                    trade.regime, trade.is_closed, trade.remaining_qty
                ))
                
                # Insert exits (if new)
                for exit_event in trade.exits:
                    cur.execute("""
                        INSERT INTO exit_metrics_partials (
                            trade_id, exit_type, qty, price, time, fees, pnl
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        trade.trade_id, exit_event.exit_type,
                        exit_event.qty, exit_event.price, exit_event.time,
                        exit_event.fees, exit_event.pnl
                    ))
        
        except Exception as e:
            logger.error(f"Postgres save failed for {trade.trade_id}: {e}")
    
    def _save_trade_redis(self, trade: TradeLifecycle):
        """Save to Redis as JSON"""
        try:
            key = f"quantum:metrics:exit:{trade.trade_id}"
            trade_data = asdict(trade)
            self.redis.set(key, json.dumps(trade_data), ex=86400 * 30)  # 30 days TTL
        except Exception as e:
            logger.error(f"Redis save failed for {trade.trade_id}: {e}")
    
    def get_trade(self, trade_id: str) -> Optional[TradeLifecycle]:
        """Get trade from Postgres or Redis"""
        if self.use_postgres:
            trade = self._get_trade_postgres(trade_id)
            if trade:
                return trade
        
        return self._get_trade_redis(trade_id)
    
    def _get_trade_postgres(self, trade_id: str) -> Optional[TradeLifecycle]:
        """Get from Postgres"""
        try:
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM exit_metrics_trades WHERE trade_id = %s
                """, (trade_id,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                # Get exits
                cur.execute("""
                    SELECT * FROM exit_metrics_partials 
                    WHERE trade_id = %s 
                    ORDER BY time
                """, (trade_id,))
                
                exits_rows = cur.fetchall()
                exits = [
                    ExitEvent(
                        exit_type=e['exit_type'],
                        qty=float(e['qty']),
                        price=float(e['price']),
                        time=float(e['time']),
                        fees=float(e['fees']),
                        pnl=float(e['pnl'])
                    )
                    for e in exits_rows
                ]
                
                trade = TradeLifecycle(
                    trade_id=row['trade_id'],
                    symbol=row['symbol'],
                    side=row['side'],
                    entry_price=float(row['entry_price']),
                    entry_time=float(row['entry_time']),
                    entry_qty=float(row['entry_qty']),
                    exits=exits,
                    mfe=float(row['mfe']),
                    mae=float(row['mae']),
                    pnl_realized=float(row['pnl_realized']),
                    pnl_percent=float(row['pnl_percent']),
                    time_in_trade=float(row['time_in_trade']),
                    regime=row['regime'],
                    is_closed=row['is_closed'],
                    remaining_qty=float(row['remaining_qty'])
                )
                
                return trade
        
        except Exception as e:
            logger.error(f"Postgres get failed for {trade_id}: {e}")
            return None
    
    def _get_trade_redis(self, trade_id: str) -> Optional[TradeLifecycle]:
        """Get from Redis"""
        try:
            key = f"quantum:metrics:exit:{trade_id}"
            data = self.redis.get(key)
            if not data:
                return None
            
            trade_dict = json.loads(data)
            
            # Reconstruct exits
            exits = [
                ExitEvent(**exit_dict)
                for exit_dict in trade_dict.get('exits', [])
            ]
            trade_dict['exits'] = exits
            
            return TradeLifecycle(**trade_dict)
        
        except Exception as e:
            logger.error(f"Redis get failed for {trade_id}: {e}")
            return None

# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    """Calculate aggregate KPIs and update Prometheus metrics"""
    
    def __init__(self, storage: StorageLayer):
        self.storage = storage
    
    def calculate_kpis(self, trades: List[TradeLifecycle]):
        """Calculate and export KPIs to Prometheus"""
        if not trades:
            return
        
        # Group by symbol and regime
        by_symbol_regime = defaultdict(list)
        by_symbol_exit_type = defaultdict(list)
        
        for trade in trades:
            if not trade.is_closed:
                continue
            
            key = (trade.symbol, trade.regime, trade.side)
            by_symbol_regime[key].append(trade)
            
            for exit_event in trade.exits:
                exit_key = (trade.symbol, trade.regime, exit_event.exit_type)
                by_symbol_exit_type[exit_key].append((trade, exit_event))
        
        # Calculate per symbol/regime
        for (symbol, regime, side), trade_list in by_symbol_regime.items():
            # PnL
            avg_pnl = sum(t.pnl_realized for t in trade_list) / len(trade_list)
            pnl_gauge.labels(symbol=symbol, regime=regime, side=side).set(avg_pnl)
            
            # MFE / MAE
            avg_mfe = sum(t.mfe for t in trade_list) / len(trade_list)
            avg_mae = sum(t.mae for t in trade_list) / len(trade_list)
            mfe_gauge.labels(symbol=symbol, regime=regime, side=side).set(avg_mfe)
            mae_gauge.labels(symbol=symbol, regime=regime, side=side).set(avg_mae)
            
            # Time in trade
            avg_time = sum(t.time_in_trade for t in trade_list) / len(trade_list)
            time_in_trade_gauge.labels(symbol=symbol, regime=regime, side=side).set(avg_time)
            
            # Expectancy
            winners = [t for t in trade_list if t.pnl_realized > 0]
            losers = [t for t in trade_list if t.pnl_realized <= 0]
            
            if len(trade_list) > 0:
                win_rate = len(winners) / len(trade_list)
                loss_rate = 1 - win_rate
                
                avg_win = sum(t.pnl_realized for t in winners) / len(winners) if winners else 0
                avg_loss = abs(sum(t.pnl_realized for t in losers) / len(losers)) if losers else 0
                
                expectancy = avg_win * win_rate - avg_loss * loss_rate
                expectancy_gauge.labels(symbol=symbol, regime=regime).set(expectancy)
            
            # PnL histogram
            for trade in trade_list:
                pnl_histogram.labels(symbol=symbol, regime=regime).observe(trade.pnl_realized)
        
        # Partial win rates
        for (symbol, regime, exit_type), exit_list in by_symbol_exit_type.items():
            winning_exits = sum(1 for _, exit_event in exit_list if exit_event.pnl > 0)
            total_exits = len(exit_list)
            
            if total_exits > 0:
                partial_wr = winning_exits / total_exits
                partial_winrate_gauge.labels(
                    symbol=symbol,
                    regime=regime,
                    exit_type=exit_type
                ).set(partial_wr)
                
                # Exit efficiency (exit price vs MFE)
                efficiencies = []
                for trade, exit_event in exit_list:
                    if trade.mfe > 0:
                        # Calculate MFE price
                        if trade.side == "LONG":
                            mfe_price = trade.entry_price + (trade.mfe / trade.entry_qty)
                        else:
                            mfe_price = trade.entry_price - (trade.mfe / trade.entry_qty)
                        
                        efficiency = exit_event.price / mfe_price if mfe_price > 0 else 0
                        efficiencies.append(efficiency)
                
                if efficiencies:
                    avg_efficiency = sum(efficiencies) / len(efficiencies)
                    exit_efficiency_gauge.labels(
                        symbol=symbol,
                        regime=regime,
                        exit_type=exit_type
                    ).set(avg_efficiency)

# ============================================================================
# EXIT INTELLIGENCE SERVICE
# ============================================================================

class ExitIntelligenceService:
    """Main service orchestrator"""
    
    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False
        )
        
        self.regime_engine = RegimeEngine(self.redis)
        self.storage = StorageLayer(self.redis)
        self.analytics = AnalyticsEngine(self.storage)
        
        self.active_trades: Dict[str, TradeLifecycle] = {}
        
        # Create consumer groups
        self._create_consumer_groups()
    
    def _create_consumer_groups(self):
        """Create consumer groups for streams"""
        for stream in [APPLY_RESULT_STREAM]:
            try:
                self.redis.xgroup_create(stream, CONSUMER_GROUP, id='0', mkstream=True)
                logger.info(f"Created consumer group {CONSUMER_GROUP} on {stream}")
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    logger.info(f"Consumer group {CONSUMER_GROUP} already exists on {stream}")
                else:
                    raise
    
    def process_apply_result(self, message: Dict):
        """Process apply.result message (exit fills)"""
        try:
            data = message.get('data', {})
            
            # Extract fields
            symbol = data.get('symbol')
            side = data.get('side')  # BUY or SELL
            price = float(data.get('price', 0))
            qty = float(data.get('qty', 0))
            fees = float(data.get('fees', 0))
            timestamp = float(data.get('timestamp', time.time()))
            
            # Detect exit type from metadata
            metadata = data.get('metadata', {})
            exit_type = metadata.get('exit_type', 'unknown')  # "partial_25" / "full" etc
            
            # Get or create trade
            trade_id = data.get('trade_id') or f"{symbol}_{timestamp}"
            
            if trade_id not in self.active_trades:
                # Try to load from storage
                trade = self.storage.get_trade(trade_id)
                
                if not trade:
                    # Create new trade (assume this is entry fill)
                    # In real system, you'd correlate with execution.result
                    logger.info(f"Creating new trade {trade_id} from exit fill (missing entry)")
                    trade = TradeLifecycle(
                        trade_id=trade_id,
                        symbol=symbol,
                        side="LONG" if side == "SELL" else "SHORT",  # Exit SELL means LONG position
                        entry_price=price,  # Approximate
                        entry_time=timestamp - 3600,  # Guess 1hr ago
                        entry_qty=qty
                    )
                
                self.active_trades[trade_id] = trade
            
            trade = self.active_trades[trade_id]
            
            # Add exit
            exit_event = ExitEvent(
                exit_type=exit_type,
                qty=qty,
                price=price,
                time=timestamp,
                fees=fees
            )
            trade.add_exit(exit_event)
            
            # Update regime
            trade.regime = self.regime_engine.get_regime(symbol)
            
            # Save to storage
            self.storage.save_trade(trade)
            
            # Update counters
            exits_total_counter.labels(symbol=symbol, exit_type=exit_type).inc()
            
            if trade.is_closed:
                trades_closed_counter.labels(
                    symbol=symbol,
                    regime=trade.regime,
                    side=trade.side
                ).inc()
                
                logger.info(
                    f"Trade {trade_id} closed: "
                    f"PnL={trade.pnl_realized:.2f} USDT "
                    f"({trade.pnl_percent:.2f}%), "
                    f"MFE={trade.mfe:.2f}, MAE={trade.mae:.2f}, "
                    f"Time={trade.time_in_trade:.0f}s, "
                    f"Regime={trade.regime}"
                )
                
                # Remove from active
                del self.active_trades[trade_id]
        
        except Exception as e:
            logger.error(f"Error processing apply.result: {e}", exc_info=True)
    
    def update_mfe_mae(self):
        """Periodically update MFE/MAE for active trades"""
        for trade_id, trade in list(self.active_trades.items()):
            try:
                # Fetch current price
                df = self.regime_engine._get_candles(trade.symbol)
                if df is not None and len(df) > 0:
                    current_price = float(df.iloc[-1]['close'])
                    trade.update_mfe_mae(current_price)
            except Exception as e:
                logger.error(f"Failed to update MFE/MAE for {trade_id}: {e}")
    
    def run_analytics(self):
        """Run periodic analytics and update Prometheus"""
        try:
            # Get last 20 closed trades
            closed_trades = []
            
            if self.storage.use_postgres:
                with self.storage.postgres_conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM exit_metrics_trades 
                        WHERE is_closed = TRUE 
                        ORDER BY updated_at DESC 
                        LIMIT 20
                    """)
                    rows = cur.fetchall()
                    
                    for row in rows:
                        trade = self.storage._get_trade_postgres(row['trade_id'])
                        if trade:
                            closed_trades.append(trade)
            else:
                # Fallback: scan Redis (limited)
                pattern = "quantum:metrics:exit:*"
                for key in self.redis.scan_iter(match=pattern, count=100):
                    trade = self.storage._get_trade_redis(key.decode().split(':')[-1])
                    if trade and trade.is_closed:
                        closed_trades.append(trade)
                    
                    if len(closed_trades) >= 20:
                        break
            
            # Calculate KPIs
            self.analytics.calculate_kpis(closed_trades)
            
        except Exception as e:
            logger.error(f"Analytics run failed: {e}", exc_info=True)
    
    def run(self):
        """Main event loop"""
        logger.info(f"Exit Intelligence Service starting on :{METRICS_PORT}")
        logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
        logger.info(f"Postgres: {'ENABLED' if self.storage.use_postgres else 'DISABLED'}")
        
        # Start Prometheus metrics server
        start_http_server(METRICS_PORT)
        logger.info(f"Prometheus metrics available at http://localhost:{METRICS_PORT}/metrics")
        
        last_analytics = time.time()
        last_mfe_mae_update = time.time()
        
        while True:
            try:
                # Read from apply.result stream
                messages = self.redis.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {APPLY_RESULT_STREAM: '>'},
                    count=10,
                    block=1000
                )
                
                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        # Decode message (same pattern as metricpack_builder)
                        decoded = {}
                        for k, v in msg_data.items():
                            key = k.decode() if isinstance(k, bytes) else k
                            val = v.decode() if isinstance(v, bytes) else v
                            
                            # Only parse 'data' field as JSON
                            if key == "data":
                                try:
                                    decoded[key] = json.loads(val)
                                except:
                                    decoded[key] = {}
                            else:
                                decoded[key] = val
                        
                        self.process_apply_result(decoded)
                        
                        # ACK message
                        self.redis.xack(APPLY_RESULT_STREAM, CONSUMER_GROUP, msg_id)
                
                # Periodic MFE/MAE updates (every 10 seconds)
                if time.time() - last_mfe_mae_update > 10:
                    self.update_mfe_mae()
                    last_mfe_mae_update = time.time()
                
                # Periodic analytics (every 30 seconds)
                if time.time() - last_analytics > 30:
                    self.run_analytics()
                    last_analytics = time.time()
            
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    service = ExitIntelligenceService()
    service.run()
