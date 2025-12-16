"""
Performance & Analytics Layer (PAL) - Fake Repositories

Fake implementations for testing and demonstration purposes.
"""

import random
from datetime import datetime, timedelta
from typing import Optional

from .models import (
    Trade, TradeDirection, TradeExitReason, MarketRegime,
    VolatilityLevel, RiskMode, StrategyStats, SymbolStats,
    EventLog, EventType, EquityPoint
)


class FakeTradeRepository:
    """Fake trade repository with synthetic data"""
    
    def __init__(self, seed: int = 42):
        """Initialize with seed for reproducibility"""
        random.seed(seed)
        self.trades = self._generate_fake_trades(500)
    
    def _generate_fake_trades(self, count: int) -> list[Trade]:
        """Generate fake trades"""
        trades = []
        base_date = datetime.now() - timedelta(days=365)
        
        strategies = ["TREND_V3", "MEAN_REVERT_V2", "BREAKOUT_V1", "GRID_V4"]
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        regimes = list(MarketRegime)
        volatilities = list(VolatilityLevel)
        risk_modes = list(RiskMode)
        
        for i in range(count):
            entry_time = base_date + timedelta(hours=i * 4)
            exit_time = entry_time + timedelta(hours=random.randint(1, 24))
            
            entry_price = random.uniform(100, 5000)
            exit_price = entry_price * random.uniform(0.95, 1.05)
            size = random.uniform(10, 100)
            
            direction = random.choice(list(TradeDirection))
            
            # Calculate PnL
            if direction == TradeDirection.LONG:
                pnl = (exit_price - entry_price) * size
            else:
                pnl = (entry_price - exit_price) * size
            
            pnl_pct = pnl / (entry_price * size)
            
            # Commission & slippage
            commission = entry_price * size * 0.001
            slippage = abs(pnl) * 0.005
            pnl -= (commission + slippage)
            
            trade = Trade(
                id=f"TRADE_{i:06d}",
                timestamp=exit_time,
                symbol=random.choice(symbols),
                strategy_id=random.choice(strategies),
                direction=direction,
                entry_price=entry_price,
                entry_timestamp=entry_time,
                entry_size=size,
                exit_price=exit_price,
                exit_timestamp=exit_time,
                exit_reason=random.choice(list(TradeExitReason)),
                pnl=pnl,
                pnl_pct=pnl_pct,
                r_multiple=random.uniform(-2, 3),
                regime_at_entry=random.choice(regimes),
                volatility_at_entry=random.choice(volatilities),
                risk_mode=random.choice(risk_modes),
                confidence=random.uniform(0.5, 0.95),
                commission=commission,
                slippage=slippage,
                model_version=f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
            )
            
            trades.append(trade)
        
        return sorted(trades, key=lambda t: t.timestamp)
    
    def get_trades(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        symbol: str | None = None,
        strategy_id: str | None = None,
        limit: int | None = None,
    ) -> list[Trade]:
        """Filter trades"""
        filtered = self.trades
        
        if start:
            filtered = [t for t in filtered if t.timestamp >= start]
        
        if end:
            filtered = [t for t in filtered if t.timestamp <= end]
        
        if symbol:
            filtered = [t for t in filtered if t.symbol == symbol]
        
        if strategy_id:
            filtered = [t for t in filtered if t.strategy_id == strategy_id]
        
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_trade_count(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """Get trade count"""
        return len(self.get_trades(start=start, end=end))


class FakeStrategyStatsRepository:
    """Fake strategy stats repository"""
    
    def get_strategy_stats(
        self,
        strategy_id: str,
        days: int = 90
    ) -> list[StrategyStats]:
        """Generate fake stats"""
        stats = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            stats.append(StrategyStats(
                strategy_id=strategy_id,
                timestamp=date,
                total_trades=random.randint(5, 20),
                winning_trades=random.randint(3, 12),
                losing_trades=random.randint(2, 8),
                total_pnl=random.uniform(-100, 300),
                win_rate=random.uniform(0.45, 0.65),
                max_drawdown=random.uniform(-0.15, -0.02),
                avg_r_multiple=random.uniform(-0.5, 1.5),
                profit_factor=random.uniform(0.8, 2.5),
                sharpe_ratio=random.uniform(0.5, 2.0),
                total_volume=random.uniform(1000, 10000),
                avg_trade_size=random.uniform(50, 500),
                active=random.random() > 0.1,
                signals_generated=random.randint(10, 50),
                signals_accepted=random.randint(5, 20),
            ))
        
        return stats
    
    def get_all_strategy_ids(self) -> list[str]:
        """Get strategy IDs"""
        return ["TREND_V3", "MEAN_REVERT_V2", "BREAKOUT_V1", "GRID_V4"]


class FakeSymbolStatsRepository:
    """Fake symbol stats repository"""
    
    def get_symbol_stats(
        self,
        symbol: str,
        days: int = 90
    ) -> list[SymbolStats]:
        """Generate fake stats"""
        stats = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            stats.append(SymbolStats(
                symbol=symbol,
                timestamp=date,
                total_trades=random.randint(5, 15),
                winning_trades=random.randint(3, 10),
                losing_trades=random.randint(2, 5),
                total_pnl=random.uniform(-50, 200),
                win_rate=random.uniform(0.45, 0.65),
                max_drawdown=random.uniform(-0.12, -0.02),
                avg_r_multiple=random.uniform(-0.3, 1.2),
                profit_factor=random.uniform(0.9, 2.2),
                total_volume=random.uniform(5000, 50000),
                regime_distribution={
                    MarketRegime.BULL: random.randint(0, 5),
                    MarketRegime.BEAR: random.randint(0, 3),
                    MarketRegime.CHOPPY: random.randint(0, 7),
                    MarketRegime.UNKNOWN: random.randint(0, 1),
                },
                volatility_distribution={
                    VolatilityLevel.LOW: random.randint(0, 5),
                    VolatilityLevel.MEDIUM: random.randint(0, 7),
                    VolatilityLevel.HIGH: random.randint(0, 3),
                },
            ))
        
        return stats
    
    def get_all_symbols(self) -> list[str]:
        """Get symbols"""
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


class FakeMetricsRepository:
    """Fake metrics repository"""
    
    def __init__(self, initial_balance: float = 10_000.0):
        """Initialize with initial balance"""
        self.initial_balance = initial_balance
        self.current_equity = initial_balance * random.uniform(1.1, 1.5)
    
    def get_equity_curve(
        self,
        days: int = 365
    ) -> list[EquityPoint]:
        """Generate fake equity curve"""
        points = []
        base_date = datetime.now() - timedelta(days=days)
        equity = self.initial_balance
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            # Random walk
            change = random.uniform(-100, 150)
            equity += change
            equity = max(equity, self.initial_balance * 0.8)  # Floor
            
            points.append(EquityPoint(
                timestamp=date,
                equity=equity,
                balance=equity * random.uniform(0.9, 1.0),
                unrealized_pnl=equity * random.uniform(-0.05, 0.05),
            ))
        
        return points
    
    def get_drawdown_curve(
        self,
        days: int = 365
    ) -> list[tuple[datetime, float]]:
        """Generate fake drawdown curve"""
        equity_points = self.get_equity_curve(days)
        
        peak = equity_points[0].equity
        curve = []
        
        for point in equity_points:
            if point.equity > peak:
                peak = point.equity
            
            dd = (point.equity - peak) / peak if peak > 0 else 0.0
            curve.append((point.timestamp, dd))
        
        return curve
    
    def get_current_equity(self) -> float:
        """Get current equity"""
        return self.current_equity
    
    def get_initial_balance(self) -> float:
        """Get initial balance"""
        return self.initial_balance


class FakeEventLogRepository:
    """Fake event log repository"""
    
    def __init__(self, seed: int = 42):
        """Initialize with seed"""
        random.seed(seed)
        self.events = self._generate_fake_events(50)
    
    def _generate_fake_events(self, count: int) -> list[EventLog]:
        """Generate fake events"""
        events = []
        base_date = datetime.now() - timedelta(days=365)
        
        event_types = list(EventType)
        severities = ["INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for i in range(count):
            date = base_date + timedelta(days=random.randint(0, 365))
            event_type = random.choice(event_types)
            
            events.append(EventLog(
                id=f"EVENT_{i:06d}",
                timestamp=date,
                event_type=event_type,
                severity=random.choice(severities),
                description=f"{event_type.value} - Random event {i}",
                details={
                    "reason": f"Simulated {event_type.value}",
                    "component": random.choice(["MSC", "ESS", "RiskGuard", "Runtime"]),
                },
                equity_at_event=random.uniform(9000, 15000),
                drawdown_at_event=random.uniform(-0.15, 0.0),
                active_positions=random.randint(0, 5),
            ))
        
        return sorted(events, key=lambda e: e.timestamp)
    
    def get_emergency_events(
        self,
        days: int = 365
    ) -> list[EventLog]:
        """Get emergency events"""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            e for e in self.events
            if e.timestamp >= cutoff and e.event_type in [
                EventType.EMERGENCY_STOP,
                EventType.EMERGENCY_STOP_CLEARED
            ]
        ]
    
    def get_health_events(
        self,
        days: int = 365
    ) -> list[EventLog]:
        """Get health events"""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            e for e in self.events
            if e.timestamp >= cutoff and e.event_type in [
                EventType.HEALTH_WARNING,
                EventType.HEALTH_RECOVERED
            ]
        ]
    
    def get_all_events(
        self,
        days: int = 365,
        event_types: list[str] | None = None,
    ) -> list[EventLog]:
        """Get all events"""
        cutoff = datetime.now() - timedelta(days=days)
        filtered = [e for e in self.events if e.timestamp >= cutoff]
        
        if event_types:
            filtered = [e for e in filtered if e.event_type.value in event_types]
        
        return filtered
