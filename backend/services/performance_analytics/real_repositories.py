"""
Real Repository Implementations for Performance & Analytics Layer

These implementations connect PAL to the Quantum Trader database,
providing access to real trading data for analytics.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import TradeLog
from backend.services.performance_analytics import (
    Trade, TradeDirection, TradeExitReason,
    MarketRegime, VolatilityLevel, RiskMode,
    EventLog, EventType, StrategyStats, SymbolStats, EquityPoint, DrawdownPeriod,
    TradeRepository, StrategyStatsRepository, SymbolStatsRepository,
    MetricsRepository, EventLogRepository,
)


class DatabaseTradeRepository(TradeRepository):
    """
    Real implementation of TradeRepository backed by Quantum Trader database.
    
    Maps TradeLog ORM model to PAL Trade dataclass.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_trades(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """
        Fetch trades from database with optional filters.
        """
        # Build query
        query = self.db.query(TradeLog)
        
        # Apply filters
        if start:
            query = query.filter(TradeLog.timestamp >= start)
        if end:
            query = query.filter(TradeLog.timestamp <= end)
        if symbol:
            query = query.filter(TradeLog.symbol == symbol)
        if strategy_id:
            query = query.filter(TradeLog.strategy_id == strategy_id)
        
        # Apply limit
        if limit:
            query = query.limit(limit)
        
        # Execute and convert to Trade dataclasses
        trades = []
        for trade_log in query.all():
            # Map TradeLog to Trade dataclass
            trade = Trade(
                id=str(trade_log.id),
                timestamp=trade_log.timestamp or datetime.now(timezone.utc),
                symbol=trade_log.symbol or "UNKNOWN",
                strategy_id=trade_log.strategy_id or "UNKNOWN",
                direction=TradeDirection(trade_log.side) if trade_log.side in ['LONG', 'SHORT'] else TradeDirection.LONG,
                entry_price=float(trade_log.entry_price or 0),
                entry_timestamp=trade_log.timestamp or datetime.now(timezone.utc),
                entry_size=float(trade_log.qty or 0),
                exit_price=float(trade_log.exit_price or 0),
                exit_timestamp=trade_log.timestamp or datetime.now(timezone.utc),
                exit_reason=self._map_exit_reason(trade_log.reason),
                pnl=float(trade_log.realized_pnl or 0),
                pnl_pct=float(trade_log.realized_pnl_pct or 0),
                r_multiple=0.0,  # Not in schema
                regime_at_entry=MarketRegime.UNKNOWN,  # Not in schema
                volatility_at_entry=VolatilityLevel.MEDIUM,  # Not in schema
                risk_mode=RiskMode.NORMAL,  # Not in schema
                confidence=0.0,  # Not in schema
                commission=0.0,  # Not in schema
                slippage=0.0,  # Not in schema
            )
            trades.append(trade)
        
        return trades
    
    def get_trade_count(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> int:
        """
        Count trades matching filters.
        """
        query = self.db.query(func.count(TradeLog.id))
        
        if start:
            query = query.filter(TradeLog.timestamp >= start)
        if end:
            query = query.filter(TradeLog.timestamp <= end)
        
        return query.scalar() or 0
    
    def _map_exit_reason(self, reason: Optional[str]) -> TradeExitReason:
        """Map database exit reason to enum."""
        if not reason:
            return TradeExitReason.UNKNOWN
        
        reason_upper = reason.upper()
        if "TP" in reason_upper or "TAKE_PROFIT" in reason_upper:
            return TradeExitReason.TAKE_PROFIT
        elif "SL" in reason_upper or "STOP_LOSS" in reason_upper:
            return TradeExitReason.STOP_LOSS
        elif "TRAIL" in reason_upper:
            return TradeExitReason.TRAILING_STOP
        elif "SIGNAL" in reason_upper or "AI" in reason_upper:
            return TradeExitReason.SIGNAL_REVERSAL
        elif "TIME" in reason_upper or "TIMEOUT" in reason_upper:
            return TradeExitReason.TIME_LIMIT
        elif "MANUAL" in reason_upper:
            return TradeExitReason.MANUAL_CLOSE
        elif "EMERGENCY" in reason_upper or "ESS" in reason_upper:
            return TradeExitReason.EMERGENCY_STOP
        else:
            return TradeExitReason.UNKNOWN
    
    def _map_regime(self, regime: Optional[str]) -> MarketRegime:
        """Map database regime to enum."""
        if not regime:
            return MarketRegime.UNKNOWN
        
        regime_upper = regime.upper()
        if regime_upper == "BULL":
            return MarketRegime.BULL
        elif regime_upper == "BEAR":
            return MarketRegime.BEAR
        elif regime_upper == "CHOPPY" or regime_upper == "SIDEWAYS":
            return MarketRegime.CHOPPY
        else:
            return MarketRegime.UNKNOWN
    
    def _map_volatility(self, volatility: Optional[str]) -> VolatilityLevel:
        """Map database volatility to enum."""
        if not volatility:
            return VolatilityLevel.MEDIUM
        
        vol_upper = volatility.upper()
        if vol_upper == "LOW":
            return VolatilityLevel.LOW
        elif vol_upper == "MEDIUM":
            return VolatilityLevel.MEDIUM
        elif vol_upper == "HIGH":
            return VolatilityLevel.HIGH
        else:
            return VolatilityLevel.MEDIUM
    
    def _map_risk_mode(self, risk_mode: Optional[str]) -> RiskMode:
        """Map database risk mode to enum."""
        if not risk_mode:
            return RiskMode.NORMAL
        
        mode_upper = risk_mode.upper()
        if mode_upper == "AGGRESSIVE":
            return RiskMode.AGGRESSIVE
        elif mode_upper == "DEFENSIVE":
            return RiskMode.DEFENSIVE
        else:
            return RiskMode.NORMAL
    
    def _calculate_holding_time(
        self, 
        entry_time: Optional[datetime], 
        exit_time: Optional[datetime]
    ) -> int:
        """Calculate holding time in minutes."""
        if not entry_time or not exit_time:
            return 0
        
        delta = exit_time - entry_time
        return int(delta.total_seconds() / 60)


class DatabaseStrategyStatsRepository(StrategyStatsRepository):
    """
    Strategy statistics repository backed by aggregated trade data.
    
    Computes strategy performance on-the-fly from TradeLog.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_strategy_stats(
        self,
        strategy_id: str,
        days: int = 90,
    ) -> List[StrategyStats]:
        """
        Compute strategy statistics from trades in date range.
        """
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Query trades for this strategy
        trades = self.db.query(TradeLog).filter(
            TradeLog.strategy_id == strategy_id,
            TradeLog.timestamp >= start_date,
            TradeLog.timestamp <= end_date,
        ).all()
        
        if not trades:
            return []
        
        # Compute stats
        total_pnl = sum(float(t.realized_pnl or 0) for t in trades)
        winning_trades = [t for t in trades if (t.realized_pnl or 0) > 0]
        losing_trades = [t for t in trades if (t.realized_pnl or 0) < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = sum(float(t.realized_pnl or 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = abs(sum(float(t.realized_pnl or 0) for t in losing_trades) / len(losing_trades)) if losing_trades else 0
        profit_factor = abs(sum(float(t.realized_pnl or 0) for t in winning_trades) / sum(float(t.realized_pnl or 0) for t in losing_trades)) if losing_trades else 0
        
        # Return list with single stats point
        return [StrategyStats(
            strategy_id=strategy_id,
            date=end_date.date(),
            total_pnl=total_pnl,
            trade_count=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown_pct=0.0,  # Would need equity curve calculation
            sharpe_ratio=0.0,       # Would need returns time series
        )]
    
    def get_all_strategy_ids(self) -> List[str]:
        """
        Get list of all unique strategy IDs.
        """
        result = self.db.query(TradeLog.strategy).distinct().all()
        return [r[0] for r in result if r[0]]


class DatabaseSymbolStatsRepository(SymbolStatsRepository):
    """
    Symbol statistics repository backed by aggregated trade data.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_symbol_stats(
        self,
        symbol: str,
        days: int = 90,
    ) -> List[SymbolStats]:
        """
        Compute symbol statistics from trades in date range.
        """
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        trades = self.db.query(TradeLog).filter(
            TradeLog.symbol == symbol,
            TradeLog.timestamp >= start_date,
            TradeLog.timestamp <= end_date,
        ).all()
        
        if not trades:
            return []
        
        # Compute stats
        total_pnl = sum(float(t.realized_pnl or 0) for t in trades)
        total_volume = sum(float(t.qty or 0) * float(t.price or 0) for t in trades)
        winning_trades = [t for t in trades if (t.realized_pnl or 0) > 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = sum(float(t.pnl or 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        
        # Return list with single stats point
        return [SymbolStats(
            symbol=symbol,
            date=end_date.date(),
            total_pnl=total_pnl,
            trade_count=len(trades),
            win_rate=win_rate,
            total_volume=total_volume,
            avg_profit_per_trade=total_pnl / len(trades) if trades else 0,
        )]
    
    def get_all_symbols(self) -> List[str]:
        """
        Get list of all unique symbols.
        """
        result = self.db.query(TradeLog.symbol).distinct().all()
        return [r[0] for r in result if r[0]]


class DatabaseMetricsRepository(MetricsRepository):
    """
    Metrics repository for equity curves and drawdown data.
    
    Computes equity curve from cumulative PnL.
    """
    
    def __init__(self, db_session: Session, initial_balance: float = 10000.0):
        self.db = db_session
        self.initial_balance = initial_balance
    
    def get_equity_curve(
        self,
        days: int = 365,
    ) -> List[EquityPoint]:
        """
        Compute equity curve from trade history.
        """
        # Calculate date range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Get all trades in chronological order
        trades = self.db.query(TradeLog).filter(
            TradeLog.timestamp >= start_time,
            TradeLog.timestamp <= end_time,
        ).order_by(TradeLog.timestamp).all()
        
        equity = self.initial_balance
        equity_curve = [EquityPoint(timestamp=start_time, equity=equity)]
        
        for trade in trades:
            if trade.timestamp and trade.realized_pnl:
                equity += float(trade.realized_pnl)
                equity_curve.append(
                    EquityPoint(timestamp=trade.timestamp, equity=equity)
                )
        
        return equity_curve
    
    def get_drawdown_curve(
        self,
        days: int = 365,
    ) -> List[tuple[datetime, float]]:
        """
        Compute drawdown curve from equity curve.
        """
        equity_curve = self.get_equity_curve(days=days)
        
        drawdown_curve = []
        peak = self.initial_balance
        
        for point in equity_curve:
            if point.equity > peak:
                peak = point.equity
            
            dd_pct = ((peak - point.equity) / peak * 100) if peak > 0 else 0
            drawdown_curve.append((point.timestamp, dd_pct))
        
        return drawdown_curve
    
    def get_current_equity(self) -> float:
        """
        Get current account equity.
        """
        # Sum all PnL
        total_pnl = self.db.query(func.sum(TradeLog.realized_pnl)).scalar() or 0
        return self.initial_balance + float(total_pnl)
    
    def get_initial_balance(self) -> float:
        """Get initial balance."""
        return self.initial_balance


class DatabaseEventLogRepository(EventLogRepository):
    """
    Event log repository for system events.
    
    NOTE: This is a stub implementation. A real implementation would
    connect to a dedicated event_log table.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def get_emergency_events(
        self,
        days: int = 365,
    ) -> List[EventLog]:
        """
        Get emergency stop events.
        
        TODO: Implement when event_log table is available.
        """
        return []
    
    def get_health_events(
        self,
        days: int = 365,
    ) -> List[EventLog]:
        """
        Get health warning/recovery events.
        
        TODO: Implement when event_log table is available.
        """
        return []
    
    def get_all_events(
        self,
        days: int = 365,
        event_types: Optional[list[str]] = None,
    ) -> List[EventLog]:
        """
        Get all events of specified type.
        
        TODO: Implement when event_log table is available.
        """
        return []
