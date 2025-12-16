"""
Replay Context - Maintains state during replay session
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Position:
    """Open position during replay"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    size: float
    entry_price: float
    entry_timestamp: datetime
    current_price: float
    unrealized_pnl: float = 0.0
    strategy_id: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayContext:
    """
    Maintains current state during replay session.
    
    Tracks timestamp, balance, positions, and metrics at each step.
    """
    # Current time
    timestamp: datetime
    
    # Account state
    balance: float  # Available cash
    equity: float   # Total account value (cash + positions)
    initial_balance: float
    
    # Positions
    open_positions: dict[str, Position] = field(default_factory=dict)
    
    # Trade history
    closed_trades: list[dict[str, Any]] = field(default_factory=list)
    
    # Current prices (for mark-to-market)
    current_prices: dict[str, float] = field(default_factory=dict)
    
    # Performance tracking
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    
    # System state
    emergency_stop_active: bool = False
    policy_mode: str = "NORMAL"  # AGGRESSIVE / NORMAL / DEFENSIVE
    
    # Counters
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Metrics
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    def __post_init__(self):
        """Initialize derived values"""
        if self.peak_equity == 0.0:
            self.peak_equity = self.equity
    
    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices and recalculate unrealized PnL"""
        self.current_prices.update(prices)
        
        # Recalculate unrealized PnL
        self.unrealized_pnl = 0.0
        for symbol, position in self.open_positions.items():
            if symbol in self.current_prices:
                current_price = self.current_prices[symbol]
                position.current_price = current_price
                
                if position.side == "LONG":
                    pnl = (current_price - position.entry_price) * position.size
                else:  # SHORT
                    pnl = (position.entry_price - current_price) * position.size
                
                position.unrealized_pnl = pnl
                self.unrealized_pnl += pnl
        
        # Update equity
        self.equity = self.balance + self.unrealized_pnl
        
        # Update drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
    
    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        strategy_id: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> None:
        """Open a new position"""
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_timestamp=self.timestamp,
            current_price=entry_price,
            strategy_id=strategy_id,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.open_positions[symbol] = position
        
        # Update balance (cost of position)
        cost = size * entry_price
        self.balance -= cost
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = "MANUAL",
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> Optional[dict[str, Any]]:
        """Close an existing position and return trade record"""
        if symbol not in self.open_positions:
            return None
        
        position = self.open_positions.pop(symbol)
        
        # Calculate PnL
        if position.side == "LONG":
            gross_pnl = (exit_price - position.entry_price) * position.size
        else:  # SHORT
            gross_pnl = (position.entry_price - exit_price) * position.size
        
        net_pnl = gross_pnl - commission - slippage
        pnl_pct = (net_pnl / (position.entry_price * position.size)) * 100
        
        # Update balance
        proceeds = position.size * exit_price + net_pnl
        self.balance += proceeds
        
        # Update counters
        self.total_trades += 1
        self.realized_pnl += net_pnl
        self.total_commission += commission
        self.total_slippage += slippage
        
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Create trade record
        trade_record = {
            "symbol": symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": position.size,
            "pnl": net_pnl,
            "pnl_pct": pnl_pct,
            "strategy_id": position.strategy_id,
            "entry_timestamp": position.entry_timestamp,
            "exit_timestamp": self.timestamp,
            "exit_reason": exit_reason,
            "commission": commission,
            "slippage": slippage,
            "bars_held": 0,  # Will be calculated later
        }
        
        self.closed_trades.append(trade_record)
        
        return trade_record
    
    def has_position(self, symbol: str) -> bool:
        """Check if symbol has an open position"""
        return symbol in self.open_positions
    
    def get_position_count(self) -> int:
        """Get number of open positions"""
        return len(self.open_positions)
    
    def get_total_exposure(self) -> float:
        """Get total USD value of all positions"""
        total = 0.0
        for position in self.open_positions.values():
            total += position.size * position.current_price
        return total
    
    def get_available_balance(self) -> float:
        """Get available cash for new positions"""
        return self.balance
    
    def check_stop_loss(self) -> list[str]:
        """Check which positions hit stop loss"""
        symbols_to_close = []
        
        for symbol, position in self.open_positions.items():
            if position.stop_loss is None:
                continue
            
            current_price = self.current_prices.get(symbol, position.current_price)
            
            if position.side == "LONG":
                if current_price <= position.stop_loss:
                    symbols_to_close.append(symbol)
            else:  # SHORT
                if current_price >= position.stop_loss:
                    symbols_to_close.append(symbol)
        
        return symbols_to_close
    
    def check_take_profit(self) -> list[str]:
        """Check which positions hit take profit"""
        symbols_to_close = []
        
        for symbol, position in self.open_positions.items():
            if position.take_profit is None:
                continue
            
            current_price = self.current_prices.get(symbol, position.current_price)
            
            if position.side == "LONG":
                if current_price >= position.take_profit:
                    symbols_to_close.append(symbol)
            else:  # SHORT
                if current_price <= position.take_profit:
                    symbols_to_close.append(symbol)
        
        return symbols_to_close
