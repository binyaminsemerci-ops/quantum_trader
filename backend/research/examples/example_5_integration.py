"""
Example 5: Integration with Quantum Trader.

Shows how to connect SG AI with existing Quantum Trader services.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from backend.research import (
    StrategyBacktester,
    StrategySearchEngine,
    StrategyConfig,
    EntryType
)
from backend.research.repositories import StrategyRepository, MarketDataClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================================
# STEP 1: Implement MarketDataClient
# ========================================

class BinanceMarketDataClient(MarketDataClient):
    """
    Connects to Binance futures API for historical data.
    
    In production, reuse existing Binance client from Quantum Trader.
    """
    
    def __init__(self, binance_client):
        """
        Args:
            binance_client: binance.client.Client instance
        """
        self.client = binance_client
    
    def get_history(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            timeframe: Interval (e.g., "15m", "1h", "4h")
            start: Start timestamp
            end: End timestamp
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"Fetching {symbol} {timeframe} data: {start} to {end}")
        
        # Convert to Binance interval format
        interval_map = {
            "1m": self.client.KLINE_INTERVAL_1MINUTE,
            "5m": self.client.KLINE_INTERVAL_5MINUTE,
            "15m": self.client.KLINE_INTERVAL_15MINUTE,
            "1h": self.client.KLINE_INTERVAL_1HOUR,
            "4h": self.client.KLINE_INTERVAL_4HOUR,
            "1d": self.client.KLINE_INTERVAL_1DAY,
        }
        
        interval = interval_map.get(timeframe, self.client.KLINE_INTERVAL_15MINUTE)
        
        # Fetch klines
        klines = self.client.futures_klines(
            symbol=symbol,
            interval=interval,
            startTime=int(start.timestamp() * 1000),
            endTime=int(end.timestamp() * 1000),
            limit=1500
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Parse types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Return only required columns
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


# ========================================
# STEP 2: Implement StrategyRepository
# ========================================

class PostgresStrategyRepository(StrategyRepository):
    """
    Stores strategies in PostgreSQL.
    
    In production, use existing database connection from Quantum Trader.
    """
    
    def __init__(self, connection_string: str):
        """
        Args:
            connection_string: PostgreSQL connection string
        """
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        self.conn = psycopg2.connect(connection_string)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # Create tables if not exist
        self._create_tables()
    
    def _create_tables(self):
        """Create strategies and stats tables"""
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                entry_type TEXT NOT NULL,
                regime_filter TEXT NOT NULL,
                min_confidence REAL NOT NULL,
                take_profit_pct REAL NOT NULL,
                stop_loss_pct REAL NOT NULL,
                trailing_stop_enabled BOOLEAN NOT NULL,
                risk_per_trade_pct REAL NOT NULL,
                leverage REAL NOT NULL,
                generation INTEGER NOT NULL,
                parent_ids TEXT[],
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_stats (
                id SERIAL PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                source TEXT NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                gross_profit REAL NOT NULL,
                gross_loss REAL NOT NULL,
                total_pnl REAL NOT NULL,
                win_rate REAL NOT NULL,
                profit_factor REAL NOT NULL,
                fitness_score REAL NOT NULL,
                period_start TIMESTAMP NOT NULL,
                period_end TIMESTAMP NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
            )
        """)
        
        self.conn.commit()
    
    def save_strategy(self, config: StrategyConfig):
        """Save or update strategy"""
        
        self.cursor.execute("""
            INSERT INTO strategies VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (strategy_id) DO UPDATE SET
                status = EXCLUDED.status,
                updated_at = EXCLUDED.updated_at
        """, (
            config.strategy_id,
            config.name,
            config.status.value,
            config.entry_type.value,
            config.regime_filter.value,
            config.min_confidence,
            config.take_profit_pct,
            config.stop_loss_pct,
            config.trailing_stop_enabled,
            config.risk_per_trade_pct,
            config.leverage,
            config.generation,
            config.parent_ids,
            config.created_at,
            datetime.utcnow()
        ))
        
        self.conn.commit()
    
    # ... implement other methods (get_strategies_by_status, save_stats, etc.)


# ========================================
# STEP 3: Enhanced Backtester with Ensemble
# ========================================

class EnsembleBacktester(StrategyBacktester):
    """
    Backtest with real ensemble predictions.
    
    Connects to Quantum Trader's 4-model ensemble.
    """
    
    def __init__(self, market_data: MarketDataClient, ensemble):
        """
        Args:
            market_data: Market data client
            ensemble: Quantum Trader ensemble (from backend.services.ensemble_orchestrator)
        """
        super().__init__(market_data)
        self.ensemble = ensemble
    
    def _check_entry(
        self,
        config: StrategyConfig,
        df: pd.DataFrame,
        idx: int
    ) -> Optional[bool]:
        """
        Check entry using real ensemble predictions.
        
        Args:
            config: Strategy configuration
            df: Market data
            idx: Current bar index
        
        Returns:
            True for long, False for short, None for no signal
        """
        # Get ensemble prediction for current bar
        current_data = df.iloc[:idx+1]
        
        try:
            # Call ensemble (assumes it has a predict method)
            prediction = self.ensemble.predict(current_data)
            
            if prediction is None:
                return None
            
            # Extract signal and confidence
            signal = prediction.get('signal')  # 'BUY', 'SELL', 'HOLD'
            confidence = prediction.get('confidence', 0.0)
            
            # Check confidence threshold
            if confidence < config.min_confidence:
                return None
            
            # Entry logic by type
            if config.entry_type == EntryType.ENSEMBLE_CONSENSUS:
                # Pure ensemble signal
                if signal == 'BUY':
                    return True
                elif signal == 'SELL':
                    return False
                else:
                    return None
            
            elif config.entry_type == EntryType.MOMENTUM:
                # Momentum confirmation
                if len(df) < idx + 20:
                    return None
                
                returns = df['close'].pct_change(20).iloc[idx]
                
                if signal == 'BUY' and returns > 0:
                    return True
                elif signal == 'SELL' and returns < 0:
                    return False
                else:
                    return None
            
            elif config.entry_type == EntryType.MEAN_REVERSION:
                # Mean reversion confirmation
                if len(df) < idx + 20:
                    return None
                
                ma = df['close'].rolling(20).mean().iloc[idx]
                current_price = df['close'].iloc[idx]
                deviation = (current_price - ma) / ma
                
                if signal == 'BUY' and deviation < -0.02:  # 2% below MA
                    return True
                elif signal == 'SELL' and deviation > 0.02:  # 2% above MA
                    return False
                else:
                    return None
        
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return None


# ========================================
# STEP 4: Full Integration Example
# ========================================

def main():
    """
    Complete integration with Quantum Trader.
    
    Prerequisites:
    1. Binance API credentials configured
    2. PostgreSQL database running
    3. Ensemble models trained and loaded
    """
    
    logger.info("=" * 60)
    logger.info("EXAMPLE 5: Quantum Trader Integration")
    logger.info("=" * 60)
    
    # Import Quantum Trader services
    # (In real system, these would be imported from backend.services)
    try:
        from binance.client import Client
        from backend.services.ensemble_orchestrator import EnsembleOrchestrator
    except ImportError:
        logger.warning("Binance/Ensemble not available, using stubs")
        
        # Use stubs for demo
        from example_1_first_generation import (
            InMemoryStrategyRepository,
            StubMarketDataClient
        )
        
        repo = InMemoryStrategyRepository()
        market_data = StubMarketDataClient()
        backtester = StrategyBacktester(market_data)
    else:
        # Real integration
        logger.info("\n🔌 Connecting to Quantum Trader services...\n")
        
        # Connect to Binance
        client = Client(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET")
        )
        
        market_data = BinanceMarketDataClient(client)
        
        # Connect to database
        db_url = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/quantum")
        repo = PostgresStrategyRepository(db_url)
        
        # Load ensemble
        ensemble = EnsembleOrchestrator()
        ensemble.load_models()
        
        backtester = EnsembleBacktester(market_data, ensemble)
        
        logger.info("✅ Connected to Quantum Trader services\n")
    
    # Create search engine
    search = StrategySearchEngine(backtester, repo)
    
    # Run generation with real data
    logger.info("🧬 Generating strategies with real ensemble...\n")
    
    strategies = search.run_generation(
        population_size=10,
        generation=1,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        start_date=datetime.utcnow() - timedelta(days=30),
        end_date=datetime.utcnow(),
        parent_strategies=None
    )
    
    # Display results
    logger.info("\n📊 Results:\n")
    
    for i, config in enumerate(strategies[:5], 1):
        stats = repo.get_stats(config.strategy_id, source="BACKTEST")[0]
        logger.info(
            f"{i}. {config.name}\n"
            f"   Fitness: {stats.fitness_score:.3f} | "
            f"PF: {stats.profit_factor:.2f} | "
            f"WR: {stats.win_rate:.1%}\n"
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Integration complete!")
    logger.info("=" * 60)
    
    logger.info("""
    
    NEXT STEPS:
    
    1. Deploy continuous evolution loop (docker-compose)
    2. Set up shadow testing (cron or async)
    3. Configure deployment thresholds
    4. Add monitoring/alerting
    5. Integrate with live trading system
    
    """)


if __name__ == "__main__":
    import os
    main()
