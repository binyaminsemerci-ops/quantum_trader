"""
Shadow testing for strategies in live forward-test mode.

Runs strategies on live data without placing real orders, tracking
simulated performance for promotion evaluation.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from .models import StrategyConfig, StrategyStats, StrategyStatus
from .repositories import StrategyRepository, MarketDataClient

logger = logging.getLogger(__name__)


class StrategyShadowTester:
    """
    Forward-tests strategies in paper mode on live data.
    
    Monitors SHADOW status strategies, simulates their execution on
    real-time market data, and tracks performance without risking capital.
    """
    
    def __init__(
        self,
        repository: StrategyRepository,
        market_data: MarketDataClient,
        initial_capital: float = 10000.0
    ):
        """
        Initialize shadow tester.
        
        Args:
            repository: Strategy storage
            market_data: Live market data client
            initial_capital: Virtual capital for simulation
        """
        self.repository = repository
        self.market_data = market_data
        self.initial_capital = initial_capital
        self._running = False
    
    async def run_shadow_loop(self, interval_minutes: int = 15) -> None:
        """
        Continuous shadow testing loop.
        
        Runs indefinitely, testing shadow strategies at regular intervals.
        
        Args:
            interval_minutes: Time between test iterations
        """
        logger.info(f"🔮 Starting shadow testing loop (interval={interval_minutes}m)")
        self._running = True
        
        while self._running:
            try:
                self.run_once()
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"Shadow loop error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Brief pause before retry
    
    def stop(self) -> None:
        """Stop the shadow testing loop"""
        logger.info("⏸️  Stopping shadow testing loop")
        self._running = False
    
    def run_once(self) -> None:
        """
        Single shadow test iteration.
        
        Can be called directly for scheduled execution (e.g. via cron).
        """
        logger.info("🔮 Running shadow test iteration")
        
        # Get all SHADOW strategies
        shadow_strategies = self.repository.get_strategies_by_status(
            StrategyStatus.SHADOW
        )
        
        if not shadow_strategies:
            logger.info("No SHADOW strategies to test")
            return
        
        logger.info(f"Testing {len(shadow_strategies)} SHADOW strategies")
        
        for config in shadow_strategies:
            try:
                self._test_strategy_shadow(config)
            except Exception as e:
                logger.error(
                    f"Failed to shadow test {config.strategy_id}: {e}",
                    exc_info=True
                )
    
    def _test_strategy_shadow(self, config: StrategyConfig) -> None:
        """
        Test a single strategy in shadow mode.
        
        Simulates execution on recent live data and saves stats.
        """
        logger.info(f"  🔮 Shadow testing: {config.name}")
        
        # Get recent data (last 7 days)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        # Simulate trades for each symbol
        all_trades: list[dict] = []
        
        for symbol in config.symbols:
            try:
                df = self.market_data.get_history(
                    symbol=symbol,
                    timeframe=config.timeframes[0],
                    start=start_date,
                    end=end_date
                )
                
                if df.empty:
                    continue
                
                # Simplified simulation (in production, use full strategy engine)
                trades = self._simulate_strategy(config, symbol, df)
                all_trades.extend(trades)
                
            except Exception as e:
                logger.warning(f"    Failed to get data for {symbol}: {e}")
                continue
        
        # Calculate stats
        stats = self._calculate_shadow_stats(
            config,
            all_trades,
            start_date,
            end_date
        )
        
        # Save stats
        self.repository.save_stats(stats)
        
        logger.info(
            f"    ✅ Shadow stats: Trades={stats.total_trades}, "
            f"PF={stats.profit_factor:.2f}, WR={stats.win_rate:.1%}"
        )
    
    def _simulate_strategy(
        self,
        config: StrategyConfig,
        symbol: str,
        df: "pd.DataFrame"
    ) -> list[dict]:
        """
        Simulate strategy execution on recent data.
        
        Simplified implementation - in production, integrate with
        actual strategy execution engine.
        """
        import numpy as np
        
        # Add basic indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['roc'] = df['close'].pct_change(10)
        df = df.dropna()
        
        trades: list[dict] = []
        in_position = False
        entry_price = 0.0
        position_side = ""
        
        for i in range(len(df)):
            bar = df.iloc[i]
            
            if not in_position:
                # Check entry (simplified)
                if config.entry_type == "MOMENTUM" and abs(bar['roc']) > 0.02:
                    in_position = True
                    entry_price = bar['close']
                    position_side = "BUY" if bar['roc'] > 0 else "SELL"
                
            else:
                # Check exit (time-based for simplicity)
                if i >= len(df) - 1 or (i % 20 == 0):  # Exit after 20 bars
                    exit_price = bar['close']
                    
                    if position_side == "BUY":
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price
                    
                    trades.append({
                        "symbol": symbol,
                        "pnl": pnl_pct * self.initial_capital * 0.02,
                        "pnl_pct": pnl_pct
                    })
                    
                    in_position = False
        
        return trades
    
    def _calculate_shadow_stats(
        self,
        config: StrategyConfig,
        trades: list[dict],
        start: datetime,
        end: datetime
    ) -> StrategyStats:
        """Calculate statistics from shadow trades"""
        
        stats = StrategyStats(
            strategy_id=config.strategy_id,
            source="SHADOW",
            start_date=start,
            end_date=end,
        )
        
        if not trades:
            return stats
        
        stats.total_trades = len(trades)
        
        for trade in trades:
            if trade['pnl'] > 0:
                stats.winning_trades += 1
                stats.gross_profit += trade['pnl']
            else:
                stats.losing_trades += 1
                stats.gross_loss += trade['pnl']
        
        stats.total_pnl = stats.gross_profit + stats.gross_loss
        
        # Calculate derived metrics
        stats.calculate_derived_metrics()
        
        return stats
