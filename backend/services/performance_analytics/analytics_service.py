"""
Performance & Analytics Layer (PAL) - Analytics Service

Main service providing comprehensive performance analytics for Quantum Trader.
"""

import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

from .models import (
    Trade, MarketRegime, VolatilityLevel, RiskMode,
    PerformanceSummary, DrawdownPeriod
)
from .repositories import (
    TradeRepository, StrategyStatsRepository, SymbolStatsRepository,
    MetricsRepository, EventLogRepository
)

logger = logging.getLogger(__name__)


class PerformanceAnalyticsService:
    """
    Centralized analytics service for Quantum Trader.
    
    Provides read-only access to performance metrics, strategy analytics,
    symbol analytics, regime analysis, risk metrics, and system health data.
    """
    
    def __init__(
        self,
        trades: TradeRepository,
        strategies: StrategyStatsRepository,
        symbols: SymbolStatsRepository,
        metrics: MetricsRepository,
        events: EventLogRepository,
    ):
        """
        Args:
            trades: Trade history repository
            strategies: Strategy statistics repository
            symbols: Symbol statistics repository
            metrics: Global metrics repository
            events: Event log repository
        """
        self.trades = trades
        self.strategies = strategies
        self.symbols = symbols
        self.metrics = metrics
        self.events = events
        self.logger = logging.getLogger(f"{__name__}.PerformanceAnalyticsService")
    
    # ==================== 1. Global Performance ====================
    
    def get_global_equity_curve(
        self,
        days: int = 365
    ) -> list[tuple[datetime, float]]:
        """
        Get global equity curve.
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of (timestamp, equity) tuples
        """
        equity_points = self.metrics.get_equity_curve(days)
        return [(point.timestamp, point.equity) for point in equity_points]
    
    def get_global_performance_summary(
        self,
        days: int = 365
    ) -> dict:
        """
        Get comprehensive global performance summary.
        
        Args:
            days: Number of days to look back
        
        Returns:
            Dict with performance metrics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get trades
        trades = self.trades.get_trades(start=start_date, end=end_date)
        
        if not trades:
            return self._empty_summary(start_date, end_date, days)
        
        # Calculate metrics
        total_pnl = sum(t.pnl for t in trades)
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]
        
        initial_balance = self.metrics.get_initial_balance()
        current_equity = self.metrics.get_current_equity()
        pnl_pct = (current_equity - initial_balance) / initial_balance if initial_balance > 0 else 0.0
        
        # Drawdown
        dd_curve = self.metrics.get_drawdown_curve(days)
        max_drawdown = min((dd for _, dd in dd_curve), default=0.0)
        
        # R-multiples
        r_multiples = [t.r_multiple for t in trades if t.r_multiple is not None]
        avg_r_multiple = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0
        
        # Best/worst
        best_trade = max(trades, key=lambda t: t.pnl)
        worst_trade = min(trades, key=lambda t: t.pnl)
        
        # Daily PnL
        daily_pnl = self._calculate_daily_pnl(trades)
        best_day = max(daily_pnl.values()) if daily_pnl else 0.0
        worst_day = min(daily_pnl.values()) if daily_pnl else 0.0
        
        # Sharpe (simplified)
        sharpe = self._calculate_sharpe(daily_pnl)
        
        # Win rate
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Streaks
        win_streak, loss_streak, current_streak, current_type = self._calculate_streaks(trades)
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
            "balance": {
                "initial": initial_balance,
                "current": current_equity,
                "pnl_total": total_pnl,
                "pnl_pct": round(pnl_pct, 4),
            },
            "trades": {
                "total": len(trades),
                "winning": len(winning_trades),
                "losing": len(losing_trades),
                "win_rate": round(win_rate, 4),
            },
            "risk": {
                "max_drawdown": round(max_drawdown, 4),
                "sharpe_ratio": round(sharpe, 2),
                "profit_factor": round(profit_factor, 2),
                "avg_r_multiple": round(avg_r_multiple, 2),
            },
            "best_worst": {
                "best_trade_pnl": round(best_trade.pnl, 2),
                "worst_trade_pnl": round(worst_trade.pnl, 2),
                "best_day_pnl": round(best_day, 2),
                "worst_day_pnl": round(worst_day, 2),
            },
            "streaks": {
                "longest_win_streak": win_streak,
                "longest_loss_streak": loss_streak,
                "current_streak": current_streak,
                "current_streak_type": current_type,
            },
            "costs": {
                "total_commission": round(sum(t.commission for t in trades), 2),
                "total_slippage": round(sum(t.slippage for t in trades), 2),
            },
        }
    
    # ==================== 2. Strategy Analytics ====================
    
    def get_strategy_performance(
        self,
        strategy_id: str,
        days: int = 365
    ) -> dict:
        """
        Get comprehensive performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            days: Number of days to look back
        
        Returns:
            Dict with strategy performance metrics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get trades for this strategy
        trades = self.trades.get_trades(
            start=start_date,
            end=end_date,
            strategy_id=strategy_id
        )
        
        if not trades:
            return {
                "strategy_id": strategy_id,
                "error": "No trades found for this strategy",
            }
        
        # Calculate metrics
        total_pnl = sum(t.pnl for t in trades)
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # R-multiples
        r_multiples = [t.r_multiple for t in trades]
        avg_r = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0
        
        # Equity curve for this strategy
        equity_curve = self._calculate_strategy_equity_curve(trades)
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown_from_equity(equity_curve)
        
        # Per-symbol breakdown
        by_symbol = self._calculate_per_symbol_breakdown(trades)
        
        # Per-regime breakdown
        by_regime = self._calculate_per_regime_breakdown(trades)
        
        return {
            "strategy_id": strategy_id,
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
            "performance": {
                "pnl_total": round(total_pnl, 2),
                "win_rate": round(win_rate, 4),
                "profit_factor": round(profit_factor, 2),
                "avg_r_multiple": round(avg_r, 2),
            },
            "trades": {
                "total": len(trades),
                "winning": len(winning_trades),
                "losing": len(losing_trades),
            },
            "risk": {
                "max_drawdown": round(max_dd, 4),
            },
            "equity_curve": equity_curve,
            "by_symbol": by_symbol,
            "by_regime": by_regime,
        }
    
    def get_top_strategies(
        self,
        days: int = 365,
        limit: int = 10
    ) -> list[dict]:
        """
        Get top performing strategies ranked by PnL.
        
        Args:
            days: Number of days to look back
            limit: Max number of strategies to return
        
        Returns:
            List of strategy summaries sorted by PnL descending
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all trades
        trades = self.trades.get_trades(start=start_date, end=end_date)
        
        # Group by strategy
        by_strategy = defaultdict(list)
        for trade in trades:
            by_strategy[trade.strategy_id].append(trade)
        
        # Calculate summary for each
        summaries = []
        for strategy_id, strategy_trades in by_strategy.items():
            total_pnl = sum(t.pnl for t in strategy_trades)
            winning = [t for t in strategy_trades if t.is_winner]
            win_rate = len(winning) / len(strategy_trades) if strategy_trades else 0.0
            
            summaries.append({
                "strategy_id": strategy_id,
                "pnl_total": round(total_pnl, 2),
                "trade_count": len(strategy_trades),
                "win_rate": round(win_rate, 4),
            })
        
        # Sort by PnL descending
        summaries.sort(key=lambda x: x["pnl_total"], reverse=True)
        
        return summaries[:limit]
    
    # ==================== 3. Symbol Analytics ====================
    
    def get_symbol_performance(
        self,
        symbol: str,
        days: int = 365
    ) -> dict:
        """
        Get comprehensive performance metrics for a symbol.
        
        Args:
            symbol: Trading symbol
            days: Number of days to look back
        
        Returns:
            Dict with symbol performance metrics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get trades for this symbol
        trades = self.trades.get_trades(
            start=start_date,
            end=end_date,
            symbol=symbol
        )
        
        if not trades:
            return {
                "symbol": symbol,
                "error": "No trades found for this symbol",
            }
        
        # Calculate metrics
        total_pnl = sum(t.pnl for t in trades)
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Volume
        total_volume = sum(t.entry_size * t.entry_price for t in trades)
        
        # Per-strategy breakdown
        by_strategy = self._calculate_per_strategy_breakdown(trades)
        
        # Per-regime breakdown
        by_regime = self._calculate_per_regime_breakdown(trades)
        
        return {
            "symbol": symbol,
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
            "performance": {
                "pnl_total": round(total_pnl, 2),
                "win_rate": round(win_rate, 4),
                "profit_factor": round(profit_factor, 2),
            },
            "trades": {
                "total": len(trades),
                "winning": len(winning_trades),
                "losing": len(losing_trades),
            },
            "volume": {
                "total": round(total_volume, 2),
                "avg_per_trade": round(total_volume / len(trades), 2) if trades else 0.0,
            },
            "by_strategy": by_strategy,
            "by_regime": by_regime,
        }
    
    def get_top_symbols(
        self,
        days: int = 365,
        limit: int = 10
    ) -> list[dict]:
        """
        Get top performing symbols ranked by PnL.
        
        Args:
            days: Number of days to look back
            limit: Max number of symbols to return
        
        Returns:
            List of symbol summaries sorted by PnL descending
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all trades
        trades = self.trades.get_trades(start=start_date, end=end_date)
        
        # Group by symbol
        by_symbol = defaultdict(list)
        for trade in trades:
            by_symbol[trade.symbol].append(trade)
        
        # Calculate summary for each
        summaries = []
        for symbol, symbol_trades in by_symbol.items():
            total_pnl = sum(t.pnl for t in symbol_trades)
            winning = [t for t in symbol_trades if t.is_winner]
            win_rate = len(winning) / len(symbol_trades) if symbol_trades else 0.0
            
            summaries.append({
                "symbol": symbol,
                "pnl_total": round(total_pnl, 2),
                "trade_count": len(symbol_trades),
                "win_rate": round(win_rate, 4),
            })
        
        # Sort by PnL descending
        summaries.sort(key=lambda x: x["pnl_total"], reverse=True)
        
        return summaries[:limit]
    
    # ==================== 4. Regime Analytics ====================
    
    def get_regime_performance(
        self,
        days: int = 365
    ) -> dict:
        """
        Get performance breakdown by market regime and volatility.
        
        Args:
            days: Number of days to look back
        
        Returns:
            Dict with regime-based performance metrics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        trades = self.trades.get_trades(start=start_date, end=end_date)
        
        # Group by regime
        by_regime = defaultdict(list)
        for trade in trades:
            by_regime[trade.regime_at_entry].append(trade)
        
        # Group by volatility
        by_volatility = defaultdict(list)
        for trade in trades:
            by_volatility[trade.volatility_at_entry].append(trade)
        
        # Calculate metrics for each regime
        regime_stats = {}
        for regime, regime_trades in by_regime.items():
            total_pnl = sum(t.pnl for t in regime_trades)
            winning = [t for t in regime_trades if t.is_winner]
            win_rate = len(winning) / len(regime_trades) if regime_trades else 0.0
            
            regime_stats[regime.value] = {
                "trade_count": len(regime_trades),
                "pnl_total": round(total_pnl, 2),
                "win_rate": round(win_rate, 4),
            }
        
        # Calculate metrics for each volatility level
        volatility_stats = {}
        for vol, vol_trades in by_volatility.items():
            total_pnl = sum(t.pnl for t in vol_trades)
            winning = [t for t in vol_trades if t.is_winner]
            win_rate = len(winning) / len(vol_trades) if vol_trades else 0.0
            
            volatility_stats[vol.value] = {
                "trade_count": len(vol_trades),
                "pnl_total": round(total_pnl, 2),
                "win_rate": round(win_rate, 4),
            }
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
            "by_regime": regime_stats,
            "by_volatility": volatility_stats,
            "total_trades": len(trades),
        }
    
    # ==================== 5. Risk & Drawdown ====================
    
    def get_r_distribution(
        self,
        days: int = 365
    ) -> dict:
        """
        Get distribution of R-multiples.
        
        Args:
            days: Number of days to look back
        
        Returns:
            Dict with R-multiple distribution
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        trades = self.trades.get_trades(start=start_date, end=end_date)
        
        r_multiples = [t.r_multiple for t in trades]
        
        if not r_multiples:
            return {"error": "No R-multiple data available"}
        
        # Calculate distribution
        positive = [r for r in r_multiples if r > 0]
        negative = [r for r in r_multiples if r <= 0]
        
        # Buckets
        buckets = {
            "< -2R": len([r for r in r_multiples if r < -2]),
            "-2R to -1R": len([r for r in r_multiples if -2 <= r < -1]),
            "-1R to 0R": len([r for r in r_multiples if -1 <= r < 0]),
            "0R to 1R": len([r for r in r_multiples if 0 <= r < 1]),
            "1R to 2R": len([r for r in r_multiples if 1 <= r < 2]),
            "2R to 3R": len([r for r in r_multiples if 2 <= r < 3]),
            "> 3R": len([r for r in r_multiples if r >= 3]),
        }
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
            "summary": {
                "avg_r": round(sum(r_multiples) / len(r_multiples), 2),
                "median_r": round(sorted(r_multiples)[len(r_multiples) // 2], 2),
                "max_r": round(max(r_multiples), 2),
                "min_r": round(min(r_multiples), 2),
                "positive_count": len(positive),
                "negative_count": len(negative),
            },
            "buckets": buckets,
        }
    
    def get_drawdown_stats(
        self,
        days: int = 365
    ) -> dict:
        """
        Get drawdown statistics and periods.
        
        Args:
            days: Number of days to look back
        
        Returns:
            Dict with drawdown metrics
        """
        dd_curve = self.metrics.get_drawdown_curve(days)
        
        if not dd_curve:
            return {"error": "No drawdown data available"}
        
        # Find max drawdown
        max_dd = min((dd for _, dd in dd_curve), default=0.0)
        max_dd_point = min(dd_curve, key=lambda x: x[1])
        
        # Count drawdown periods (simplified)
        dd_periods = self._identify_drawdown_periods(dd_curve)
        
        return {
            "max_drawdown": round(max_dd, 4),
            "max_drawdown_date": max_dd_point[0].isoformat(),
            "drawdown_periods": len(dd_periods),
            "avg_drawdown": round(sum(dd for _, dd in dd_curve) / len(dd_curve), 4) if dd_curve else 0.0,
            "current_drawdown": round(dd_curve[-1][1], 4) if dd_curve else 0.0,
        }
    
    # ==================== 6. Events & Safety ====================
    
    def get_emergency_stop_history(
        self,
        days: int = 365
    ) -> list[dict]:
        """
        Get history of emergency stop events.
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of emergency stop events
        """
        events = self.events.get_emergency_events(days)
        
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "severity": event.severity,
                "description": event.description,
                "equity_at_event": event.equity_at_event,
                "drawdown_at_event": event.drawdown_at_event,
                "active_positions": event.active_positions,
                "details": event.details,
            }
            for event in events
        ]
    
    def get_system_health_timeline(
        self,
        days: int = 365
    ) -> list[dict]:
        """
        Get timeline of system health events.
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of health events
        """
        events = self.events.get_health_events(days)
        
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "severity": event.severity,
                "description": event.description,
                "details": event.details,
            }
            for event in events
        ]
    
    # ==================== Helper Methods ====================
    
    def _empty_summary(self, start_date: datetime, end_date: datetime, days: int) -> dict:
        """Return empty summary when no trades"""
        initial_balance = self.metrics.get_initial_balance()
        current_equity = self.metrics.get_current_equity()
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days,
            },
            "balance": {
                "initial": initial_balance,
                "current": current_equity,
                "pnl_total": 0.0,
                "pnl_pct": 0.0,
            },
            "trades": {
                "total": 0,
                "winning": 0,
                "losing": 0,
                "win_rate": 0.0,
            },
            "risk": {
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_r_multiple": 0.0,
            },
            "best_worst": {
                "best_trade_pnl": 0.0,
                "worst_trade_pnl": 0.0,
                "best_day_pnl": 0.0,
                "worst_day_pnl": 0.0,
            },
            "streaks": {
                "longest_win_streak": 0,
                "longest_loss_streak": 0,
                "current_streak": 0,
                "current_streak_type": "none",
            },
            "costs": {
                "total_commission": 0.0,
                "total_slippage": 0.0,
            },
        }
    
    def _calculate_daily_pnl(self, trades: list[Trade]) -> dict[str, float]:
        """Calculate PnL per day"""
        daily = defaultdict(float)
        for trade in trades:
            day = trade.exit_timestamp.date().isoformat()
            daily[day] += trade.pnl
        return dict(daily)
    
    def _calculate_sharpe(self, daily_pnl: dict[str, float]) -> float:
        """Calculate simplified Sharpe ratio"""
        if not daily_pnl:
            return 0.0
        
        returns = list(daily_pnl.values())
        avg_return = sum(returns) / len(returns)
        
        if len(returns) < 2:
            return 0.0
        
        variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        return (avg_return / std_dev) * (252 ** 0.5)  # Annualized
    
    def _calculate_streaks(self, trades: list[Trade]) -> tuple[int, int, int, str]:
        """Calculate win/loss streaks"""
        if not trades:
            return 0, 0, 0, "NONE"
        
        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 1
        current_type = "WIN" if trades[0].is_winner else "LOSS"
        
        temp_streak = 1
        temp_type = "WIN" if trades[0].is_winner else "LOSS"
        
        for i in range(1, len(trades)):
            if trades[i].is_winner == trades[i-1].is_winner:
                temp_streak += 1
            else:
                if temp_type == "WIN":
                    max_win_streak = max(max_win_streak, temp_streak)
                else:
                    max_loss_streak = max(max_loss_streak, temp_streak)
                
                temp_streak = 1
                temp_type = "WIN" if trades[i].is_winner else "LOSS"
        
        # Final streak
        if temp_type == "WIN":
            max_win_streak = max(max_win_streak, temp_streak)
        else:
            max_loss_streak = max(max_loss_streak, temp_streak)
        
        current_streak = temp_streak
        current_type = temp_type
        
        return max_win_streak, max_loss_streak, current_streak, current_type
    
    def _calculate_strategy_equity_curve(self, trades: list[Trade]) -> list[tuple[str, float]]:
        """Calculate equity curve from trades"""
        sorted_trades = sorted(trades, key=lambda t: t.exit_timestamp)
        equity = 0.0
        curve = []
        
        for trade in sorted_trades:
            equity += trade.pnl
            curve.append((trade.exit_timestamp.isoformat(), round(equity, 2)))
        
        return curve
    
    def _calculate_max_drawdown_from_equity(self, equity_curve: list[tuple[str, float]]) -> float:
        """Calculate max drawdown from equity curve"""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0][1]
        max_dd = 0.0
        
        for _, equity in equity_curve:
            if equity > peak:
                peak = equity
            
            dd = (equity - peak) / peak if peak > 0 else 0.0
            max_dd = min(max_dd, dd)
        
        return max_dd
    
    def _calculate_per_symbol_breakdown(self, trades: list[Trade]) -> dict:
        """Calculate per-symbol breakdown"""
        by_symbol = defaultdict(list)
        for trade in trades:
            by_symbol[trade.symbol].append(trade)
        
        result = {}
        for symbol, symbol_trades in by_symbol.items():
            total_pnl = sum(t.pnl for t in symbol_trades)
            winning = [t for t in symbol_trades if t.is_winner]
            win_rate = len(winning) / len(symbol_trades) if symbol_trades else 0.0
            
            result[symbol] = {
                "trade_count": len(symbol_trades),
                "pnl_total": round(total_pnl, 2),
                "win_rate": round(win_rate, 4),
            }
        
        return result
    
    def _calculate_per_strategy_breakdown(self, trades: list[Trade]) -> dict:
        """Calculate per-strategy breakdown"""
        by_strategy = defaultdict(list)
        for trade in trades:
            by_strategy[trade.strategy_id].append(trade)
        
        result = {}
        for strategy_id, strategy_trades in by_strategy.items():
            total_pnl = sum(t.pnl for t in strategy_trades)
            winning = [t for t in strategy_trades if t.is_winner]
            win_rate = len(winning) / len(strategy_trades) if strategy_trades else 0.0
            
            result[strategy_id] = {
                "trade_count": len(strategy_trades),
                "pnl_total": round(total_pnl, 2),
                "win_rate": round(win_rate, 4),
            }
        
        return result
    
    def _calculate_per_regime_breakdown(self, trades: list[Trade]) -> dict:
        """Calculate per-regime breakdown"""
        by_regime = defaultdict(list)
        for trade in trades:
            by_regime[trade.regime_at_entry].append(trade)
        
        result = {}
        for regime, regime_trades in by_regime.items():
            total_pnl = sum(t.pnl for t in regime_trades)
            winning = [t for t in regime_trades if t.is_winner]
            win_rate = len(winning) / len(regime_trades) if regime_trades else 0.0
            
            result[regime.value] = {
                "trade_count": len(regime_trades),
                "pnl_total": round(total_pnl, 2),
                "win_rate": round(win_rate, 4),
            }
        
        return result
    
    def _identify_drawdown_periods(self, dd_curve: list[tuple[datetime, float]]) -> list[dict]:
        """Identify distinct drawdown periods (simplified)"""
        periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, (ts, dd) in enumerate(dd_curve):
            if dd < -0.01 and not in_drawdown:  # Entered drawdown
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:  # Recovered
                periods.append({
                    "start": dd_curve[start_idx][0].isoformat(),
                    "end": ts.isoformat(),
                    "max_dd": min(dd_curve[start_idx:i+1], key=lambda x: x[1])[1],
                })
                in_drawdown = False
        
        return periods
