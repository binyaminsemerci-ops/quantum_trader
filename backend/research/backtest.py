"""
Strategy backtesting engine.

Simulates strategy execution on historical data to evaluate performance.
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np

from .models import StrategyConfig, StrategyStats, RegimeFilter
from .repositories import MarketDataClient

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Simulates strategy execution on historical data.
    
    Takes a StrategyConfig and historical market data, applies entry/exit
    rules, and returns performance statistics.
    """
    
    def __init__(
        self,
        market_data: MarketDataClient,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0004  # 0.04% per side
    ):
        """
        Initialize backtester.
        
        Args:
            market_data: Client for fetching historical data
            initial_capital: Starting equity for backtest
            commission_rate: Trading fee per side
        """
        self.market_data = market_data
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
    
    def backtest(
        self,
        config: StrategyConfig,
        symbols: list[str],
        start: datetime,
        end: datetime
    ) -> StrategyStats:
        """
        Run backtest and return performance stats.
        
        Args:
            config: Strategy configuration
            symbols: List of symbols to trade
            start: Backtest start date
            end: Backtest end date
            
        Returns:
            Performance statistics
        """
        logger.info(
            f"🔬 Backtesting strategy {config.strategy_id} "
            f"on {len(symbols)} symbols from {start} to {end}"
        )
        
        all_trades: list[dict[str, Any]] = []
        equity_curve: list[float] = [self.initial_capital]
        
        for symbol in symbols:
            try:
                symbol_trades = self._backtest_symbol(config, symbol, start, end)
                all_trades.extend(symbol_trades)
            except Exception as e:
                logger.warning(f"Failed to backtest {symbol}: {e}")
                continue
        
        # Calculate statistics
        stats = self._calculate_stats(config, all_trades, start, end, equity_curve)
        
        logger.info(
            f"✅ Backtest complete: {stats.total_trades} trades, "
            f"PF={stats.profit_factor:.2f}, WR={stats.win_rate:.1%}, "
            f"Fitness={stats.fitness_score:.1f}"
        )
        
        return stats
    
    def _backtest_symbol(
        self,
        config: StrategyConfig,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> list[dict[str, Any]]:
        """Backtest a single symbol"""
        
        # Get historical data
        df = self.market_data.get_history(
            symbol=symbol,
            timeframe=config.timeframes[0],  # Use first timeframe
            start=start,
            end=end
        )
        
        if df.empty:
            return []
        
        # Add indicators/features (simplified for example)
        df = self._add_indicators(df)
        
        # Simulate trades
        trades = self._simulate_trades(config, symbol, df)
        
        return trades
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators needed for entry/exit logic.
        
        Simplified implementation - in production, use comprehensive indicators.
        """
        # Example: Simple trend indicator
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['trend'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        
        # Volatility
        df['atr'] = df['high'] - df['low']  # Simplified ATR
        df['atr_20'] = df['atr'].rolling(20).mean()
        
        # Momentum
        df['roc_10'] = df['close'].pct_change(10)
        
        return df.dropna()
    
    def _simulate_trades(
        self,
        config: StrategyConfig,
        symbol: str,
        df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """
        Simulate strategy execution on bar data.
        
        Simplified implementation - checks entry conditions and simulates
        exits via TP/SL levels.
        """
        trades: list[dict[str, Any]] = []
        in_position = False
        entry_price = 0.0
        entry_bar = 0
        position_side = ""
        tp_price = 0.0
        sl_price = 0.0
        
        for i in range(len(df)):
            bar = df.iloc[i]
            
            if not in_position:
                # Check entry
                signal = self._check_entry(config, df, i)
                
                if signal in ["BUY", "SELL"]:
                    in_position = True
                    entry_price = bar['close']
                    entry_bar = i
                    position_side = signal
                    
                    # Set TP/SL
                    if signal == "BUY":
                        tp_price = entry_price * (1 + config.tp_percent)
                        sl_price = entry_price * (1 - config.sl_percent)
                    else:  # SELL
                        tp_price = entry_price * (1 - config.tp_percent)
                        sl_price = entry_price * (1 + config.sl_percent)
            
            else:
                # Check exit
                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""
                
                if position_side == "BUY":
                    if bar['high'] >= tp_price:
                        exit_triggered = True
                        exit_price = tp_price
                        exit_reason = "TP"
                    elif bar['low'] <= sl_price:
                        exit_triggered = True
                        exit_price = sl_price
                        exit_reason = "SL"
                else:  # SELL
                    if bar['low'] <= tp_price:
                        exit_triggered = True
                        exit_price = tp_price
                        exit_reason = "TP"
                    elif bar['high'] >= sl_price:
                        exit_triggered = True
                        exit_price = sl_price
                        exit_reason = "SL"
                
                if exit_triggered:
                    # Calculate PnL
                    if position_side == "BUY":
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price
                    
                    # Account for commission
                    pnl_pct -= (2 * self.commission_rate)  # Entry + exit
                    
                    pnl_dollars = self.initial_capital * pnl_pct * config.max_risk_per_trade
                    
                    trades.append({
                        "symbol": symbol,
                        "side": position_side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "entry_bar": entry_bar,
                        "exit_bar": i,
                        "bars_held": i - entry_bar,
                        "pnl": pnl_dollars,
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                    })
                    
                    in_position = False
        
        return trades
    
    def _check_entry(
        self,
        config: StrategyConfig,
        df: pd.DataFrame,
        idx: int
    ) -> str:
        """
        Check if entry conditions are met.
        
        Returns: "BUY", "SELL", or "NONE"
        
        Simplified logic - in production, integrate with actual ensemble.
        """
        if idx < 50:  # Need enough history
            return "NONE"
        
        bar = df.iloc[idx]
        
        # Example: Simple trend-following entry
        if config.entry_type == "ENSEMBLE_CONSENSUS":
            # Simulate ensemble consensus
            confidence = 0.5 + (bar['roc_10'] * 5)  # Fake confidence
            confidence = max(0.4, min(0.9, confidence))
            
            if confidence >= config.min_confidence:
                if bar['trend'] > 0:
                    return "BUY"
                elif bar['trend'] < 0:
                    return "SELL"
        
        elif config.entry_type == "MOMENTUM":
            # Momentum breakout
            if bar['roc_10'] > 0.02:  # 2% momentum
                return "BUY"
            elif bar['roc_10'] < -0.02:
                return "SELL"
        
        elif config.entry_type == "MEAN_REVERSION":
            # Mean reversion
            deviation = (bar['close'] - bar['sma_20']) / bar['sma_20']
            if deviation < -0.03:  # 3% below mean
                return "BUY"
            elif deviation > 0.03:
                return "SELL"
        
        return "NONE"
    
    def _calculate_stats(
        self,
        config: StrategyConfig,
        trades: list[dict[str, Any]],
        start: datetime,
        end: datetime,
        equity_curve: list[float]
    ) -> StrategyStats:
        """Calculate performance statistics from trade list"""
        
        stats = StrategyStats(
            strategy_id=config.strategy_id,
            source="BACKTEST",
            start_date=start,
            end_date=end,
        )
        
        if not trades:
            return stats
        
        stats.total_trades = len(trades)
        
        # Classify trades
        for trade in trades:
            if trade['pnl'] > 0:
                stats.winning_trades += 1
                stats.gross_profit += trade['pnl']
            else:
                stats.losing_trades += 1
                stats.gross_loss += trade['pnl']
        
        stats.total_pnl = stats.gross_profit + stats.gross_loss
        
        # Calculate drawdown
        equity = self.initial_capital
        peak = equity
        max_dd = 0.0
        
        for trade in trades:
            equity += trade['pnl']
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        
        stats.max_drawdown = max_dd
        if peak > 0:
            stats.max_drawdown_pct = max_dd / peak
        
        # Sharpe ratio (simplified)
        returns = [t['pnl_pct'] for t in trades]
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                stats.sharpe_ratio = (mean_return / std_return) * np.sqrt(252)  # Annualized
        
        # Average bars in trade
        stats.avg_bars_in_trade = np.mean([t['bars_held'] for t in trades])
        
        # Calculate derived metrics and fitness
        stats.calculate_derived_metrics()
        
        return stats
