"""
Trade Replay Engine - Main orchestrator for replaying historical trades
"""

import logging
from datetime import datetime
from typing import Protocol, Optional
import time

from .replay_config import ReplayConfig, ReplayMode
from .replay_context import ReplayContext, Position
from .replay_result import ReplayResult, TradeRecord, EventRecord, SymbolStats, StrategyStats
from .replay_market_data import ReplayMarketDataSource
from .exchange_simulator import ExchangeSimulator

logger = logging.getLogger(__name__)


# ==================== Component Protocols ====================

class StrategyRuntimeEngine(Protocol):
    """Protocol for strategy runtime engine"""
    def generate_candidates(self, symbol: str, candle: dict) -> list[dict]:
        """Generate candidate signals"""
        ...


class SignalOrchestrator(Protocol):
    """Protocol for signal orchestrator"""
    def filter_signals(self, candidates: list[dict], context: dict) -> list[dict]:
        """Filter and prioritize signals"""
        ...


class RiskGuard(Protocol):
    """Protocol for risk guard"""
    def validate_trade(self, signal: dict, context: dict) -> tuple[bool, str]:
        """Validate trade against risk rules"""
        ...


class PortfolioBalancer(Protocol):
    """Protocol for portfolio balancer"""
    def check_limits(self, signal: dict, context: dict) -> tuple[bool, str]:
        """Check portfolio limits"""
        ...


class SafetyGovernor(Protocol):
    """Protocol for safety governor"""
    def approve_trade(self, signal: dict, context: dict) -> tuple[bool, str]:
        """Final approval for trade"""
        ...


class PolicyStore(Protocol):
    """Protocol for MSC policy store"""
    def get_current_policy(self) -> dict:
        """Get current trading policy"""
        ...
    
    def update_policy(self, market_conditions: dict) -> None:
        """Update policy based on market conditions"""
        ...


class EmergencyStopSystem(Protocol):
    """Protocol for emergency stop system"""
    def check_conditions(self, context: dict) -> tuple[bool, str]:
        """Check if emergency stop should activate"""
        ...


# ==================== Trade Replay Engine ====================

class TradeReplayEngine:
    """
    Main orchestrator for replaying historical trades through full trading system.
    
    Coordinates all components:
    - Market data loading
    - Strategy signal generation
    - Signal orchestration
    - Risk validation
    - Portfolio balancing
    - Safety governance
    - Order execution simulation
    - Position management
    - Performance tracking
    """
    
    def __init__(
        self,
        market_data_source: ReplayMarketDataSource,
        exchange_simulator: ExchangeSimulator,
        runtime_engine: Optional[StrategyRuntimeEngine] = None,
        orchestrator: Optional[SignalOrchestrator] = None,
        risk_guard: Optional[RiskGuard] = None,
        portfolio_balancer: Optional[PortfolioBalancer] = None,
        safety_governor: Optional[SafetyGovernor] = None,
        policy_store: Optional[PolicyStore] = None,
        emergency_stop_system: Optional[EmergencyStopSystem] = None,
    ):
        """
        Args:
            market_data_source: Source for historical data
            exchange_simulator: Simulator for order execution
            runtime_engine: Strategy runtime engine (optional)
            orchestrator: Signal orchestrator (optional)
            risk_guard: Risk validation (optional)
            portfolio_balancer: Portfolio limits (optional)
            safety_governor: Final safety approval (optional)
            policy_store: MSC policy store (optional)
            emergency_stop_system: ESS (optional)
        """
        self.market_data_source = market_data_source
        self.exchange_simulator = exchange_simulator
        self.runtime_engine = runtime_engine
        self.orchestrator = orchestrator
        self.risk_guard = risk_guard
        self.portfolio_balancer = portfolio_balancer
        self.safety_governor = safety_governor
        self.policy_store = policy_store
        self.emergency_stop_system = emergency_stop_system
        
        self.logger = logging.getLogger(f"{__name__}.TradeReplayEngine")
    
    def run(self, config: ReplayConfig) -> ReplayResult:
        """
        Run replay session with given configuration.
        
        Args:
            config: Replay configuration
        
        Returns:
            ReplayResult with comprehensive metrics
        """
        self.logger.info(f"Starting replay: {config.start} to {config.end}")
        self.logger.info(f"Mode: {config.mode.name}, Symbols: {config.symbols}")
        
        start_time = time.time()
        
        # Load historical data
        try:
            data = self.market_data_source.load(config)
        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            raise
        
        # Initialize context
        context = ReplayContext(
            timestamp=config.start,
            balance=config.initial_balance,
            equity=config.initial_balance,
            initial_balance=config.initial_balance,
        )
        
        # Storage for results
        events: list[EventRecord] = []
        equity_curve: list[tuple[datetime, float]] = []
        
        # Log start
        events.append(EventRecord(
            timestamp=config.start,
            event_type="REPLAY_START",
            description=f"Replay started with {len(config.symbols)} symbols",
            details={"mode": config.mode.name, "initial_balance": config.initial_balance}
        ))
        
        # Main replay loop
        step_count = 0
        for timestamp in self.market_data_source.iter_time_steps(data):
            if timestamp < config.start or timestamp > config.end:
                continue
            
            step_count += 1
            context.timestamp = timestamp
            
            # Get current candles and prices
            candles = self.market_data_source.get_candles_at_time(data, timestamp)
            prices = self.market_data_source.get_price_snapshot(data, timestamp)
            
            # Update context with current prices
            context.update_prices(prices)
            
            # Record equity
            equity_curve.append((timestamp, context.equity))
            
            # Update MSC policy if enabled
            if config.include_msc and self.policy_store:
                try:
                    self.policy_store.update_policy({
                        "timestamp": timestamp,
                        "prices": prices,
                        "equity": context.equity,
                        "drawdown": context.current_drawdown,
                    })
                    policy = self.policy_store.get_current_policy()
                    context.policy_mode = policy.get("mode", "NORMAL")
                except Exception as e:
                    self.logger.warning(f"MSC update failed: {e}")
            
            # Check ESS if enabled
            if config.include_ess and self.emergency_stop_system:
                try:
                    ess_active, ess_reason = self.emergency_stop_system.check_conditions({
                        "equity": context.equity,
                        "drawdown": context.current_drawdown,
                        "balance": context.balance,
                    })
                    
                    if ess_active and not context.emergency_stop_active:
                        context.emergency_stop_active = True
                        events.append(EventRecord(
                            timestamp=timestamp,
                            event_type="EMERGENCY_STOP",
                            description=f"ESS activated: {ess_reason}",
                            details={"reason": ess_reason, "equity": context.equity}
                        ))
                        self.logger.warning(f"Emergency stop activated: {ess_reason}")
                        
                        # Close all positions
                        for symbol in list(context.open_positions.keys()):
                            if symbol in prices:
                                trade = context.close_position(
                                    symbol=symbol,
                                    exit_price=prices[symbol],
                                    exit_reason="ESS",
                                    commission=0.0,
                                    slippage=0.0,
                                )
                                events.append(EventRecord(
                                    timestamp=timestamp,
                                    event_type="POSITION_CLOSED",
                                    description=f"ESS forced close: {symbol}",
                                    details={"pnl": trade["pnl"]}
                                ))
                        
                        continue  # Skip signal generation
                    
                    elif not ess_active and context.emergency_stop_active:
                        context.emergency_stop_active = False
                        events.append(EventRecord(
                            timestamp=timestamp,
                            event_type="EMERGENCY_STOP_CLEARED",
                            description="ESS cleared, resuming trading",
                            details={"equity": context.equity}
                        ))
                
                except Exception as e:
                    self.logger.warning(f"ESS check failed: {e}")
            
            # Skip signal generation if emergency stop active
            if context.emergency_stop_active:
                continue
            
            # Check TP/SL exits
            sl_triggers = context.check_stop_loss()
            tp_triggers = context.check_take_profit()
            
            for symbol in sl_triggers:
                if symbol in prices:
                    trade = context.close_position(
                        symbol=symbol,
                        exit_price=prices[symbol],
                        exit_reason="STOP_LOSS",
                        commission=config.commission_rate * context.open_positions[symbol].size * prices[symbol],
                        slippage=0.0,
                    )
                    events.append(EventRecord(
                        timestamp=timestamp,
                        event_type="STOP_LOSS",
                        description=f"Stop loss hit: {symbol}",
                        details={"pnl": trade["pnl"], "pnl_pct": trade["pnl_pct"]}
                    ))
            
            for symbol in tp_triggers:
                if symbol in prices:
                    trade = context.close_position(
                        symbol=symbol,
                        exit_price=prices[symbol],
                        exit_reason="TAKE_PROFIT",
                        commission=config.commission_rate * context.open_positions[symbol].size * prices[symbol],
                        slippage=0.0,
                    )
                    events.append(EventRecord(
                        timestamp=timestamp,
                        event_type="TAKE_PROFIT",
                        description=f"Take profit hit: {symbol}",
                        details={"pnl": trade["pnl"], "pnl_pct": trade["pnl_pct"]}
                    ))
            
            # Generate signals if in strategy mode
            if config.mode in [ReplayMode.FULL, ReplayMode.STRATEGY_ONLY] and self.runtime_engine:
                for symbol, candle in candles.items():
                    # Skip if we already have a position
                    if context.has_position(symbol):
                        continue
                    
                    # Skip if max trades per bar exceeded
                    if config.max_trades_per_bar and context.get_position_count() >= config.max_trades_per_bar:
                        continue
                    
                    try:
                        # Generate candidates
                        candidates = self.runtime_engine.generate_candidates(symbol, candle)
                        
                        for candidate in candidates:
                            # Filter through orchestrator
                            if self.orchestrator:
                                filtered = self.orchestrator.filter_signals([candidate], {
                                    "timestamp": timestamp,
                                    "balance": context.balance,
                                    "equity": context.equity,
                                    "open_positions": len(context.open_positions),
                                })
                                if not filtered:
                                    continue
                                signal = filtered[0]
                            else:
                                signal = candidate
                            
                            # Validate with risk guard
                            if self.risk_guard:
                                valid, reason = self.risk_guard.validate_trade(signal, {
                                    "balance": context.balance,
                                    "equity": context.equity,
                                    "drawdown": context.current_drawdown,
                                })
                                if not valid:
                                    events.append(EventRecord(
                                        timestamp=timestamp,
                                        event_type="SIGNAL_REJECTED",
                                        description=f"Risk guard rejected: {symbol}",
                                        details={"reason": reason}
                                    ))
                                    continue
                            
                            # Check portfolio limits
                            if self.portfolio_balancer:
                                within_limits, reason = self.portfolio_balancer.check_limits(signal, {
                                    "balance": context.balance,
                                    "open_positions": context.get_position_count(),
                                    "total_exposure": context.get_total_exposure(),
                                })
                                if not within_limits:
                                    events.append(EventRecord(
                                        timestamp=timestamp,
                                        event_type="SIGNAL_REJECTED",
                                        description=f"Portfolio limits: {symbol}",
                                        details={"reason": reason}
                                    ))
                                    continue
                            
                            # Final safety approval
                            if self.safety_governor:
                                approved, reason = self.safety_governor.approve_trade(signal, {
                                    "balance": context.balance,
                                    "equity": context.equity,
                                    "emergency_stop": context.emergency_stop_active,
                                })
                                if not approved:
                                    events.append(EventRecord(
                                        timestamp=timestamp,
                                        event_type="SIGNAL_REJECTED",
                                        description=f"Safety governor rejected: {symbol}",
                                        details={"reason": reason}
                                    ))
                                    continue
                            
                            # Execute trade
                            execution = self.exchange_simulator.execute(
                                symbol=signal.get("symbol", symbol),
                                side=signal.get("side", "LONG"),
                                size=signal.get("size", 100.0),
                                price=candle["close"],
                                volume=candle.get("volume"),
                            )
                            
                            if execution.executed:
                                # Open position
                                context.open_position(
                                    symbol=symbol,
                                    side=execution.side,
                                    size=execution.size,
                                    entry_price=execution.filled_price,
                                    strategy_id=signal.get("strategy_id"),
                                    stop_loss=signal.get("stop_loss"),
                                    take_profit=signal.get("take_profit"),
                                )
                                
                                events.append(EventRecord(
                                    timestamp=timestamp,
                                    event_type="POSITION_OPENED",
                                    description=f"Opened {execution.side} position: {symbol}",
                                    details={
                                        "size": execution.size,
                                        "price": execution.filled_price,
                                        "fee": execution.fee,
                                        "slippage": execution.slippage,
                                    }
                                ))
                            else:
                                events.append(EventRecord(
                                    timestamp=timestamp,
                                    event_type="EXECUTION_FAILED",
                                    description=f"Execution failed: {symbol}",
                                    details={"reason": execution.reason}
                                ))
                    
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol} at {timestamp}: {e}")
                        continue
            
            # Speed control (if speed > 0)
            if config.speed > 0:
                time.sleep(config.speed)
        
        # Close all remaining positions at end
        final_prices = self.market_data_source.get_price_snapshot(data, config.end)
        for symbol in list(context.open_positions.keys()):
            if symbol in final_prices:
                trade = context.close_position(
                    symbol=symbol,
                    exit_price=final_prices[symbol],
                    exit_reason="REPLAY_END",
                    commission=0.0,
                    slippage=0.0,
                )
                events.append(EventRecord(
                    timestamp=config.end,
                    event_type="POSITION_CLOSED",
                    description=f"Replay end, closed: {symbol}",
                    details={"pnl": trade["pnl"]}
                ))
        
        # Calculate final metrics
        end_time = time.time()
        duration_seconds = end_time - start_time
        
        self.logger.info(f"Replay completed: {step_count} steps, {duration_seconds:.2f}s")
        self.logger.info(f"Final balance: ${context.balance:.2f}, Equity: ${context.equity:.2f}")
        self.logger.info(f"Total trades: {context.total_trades}, Win rate: {context.winning_trades/max(context.total_trades,1)*100:.1f}%")
        
        # Build result
        result = self._build_result(config, context, equity_curve, events, duration_seconds)
        
        return result
    
    def _build_result(
        self,
        config: ReplayConfig,
        context: ReplayContext,
        equity_curve: list[tuple[datetime, float]],
        events: list[EventRecord],
        duration_seconds: float,
    ) -> ReplayResult:
        """Build ReplayResult from context and events"""
        
        # Calculate per-symbol stats
        per_symbol: dict[str, SymbolStats] = {}
        for trade in context.closed_trades:
            symbol = trade["symbol"]
            if symbol not in per_symbol:
                per_symbol[symbol] = SymbolStats(
                    symbol=symbol,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    total_pnl=0.0,
                    win_rate=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    profit_factor=0.0,
                    max_win=0.0,
                    max_loss=0.0,
                )
            
            stats = per_symbol[symbol]
            stats.total_trades += 1
            stats.total_pnl += trade["pnl"]
            
            if trade["pnl"] > 0:
                stats.winning_trades += 1
                stats.max_win = max(stats.max_win, trade["pnl"])
            else:
                stats.losing_trades += 1
                stats.max_loss = min(stats.max_loss, trade["pnl"])
        
        # Calculate derived metrics
        for stats in per_symbol.values():
            if stats.total_trades > 0:
                stats.win_rate = stats.winning_trades / stats.total_trades
            
            if stats.winning_trades > 0:
                wins = [t["pnl"] for t in context.closed_trades if t["symbol"] == stats.symbol and t["pnl"] > 0]
                stats.avg_win = sum(wins) / len(wins)
            
            if stats.losing_trades > 0:
                losses = [t["pnl"] for t in context.closed_trades if t["symbol"] == stats.symbol and t["pnl"] <= 0]
                stats.avg_loss = sum(losses) / len(losses)
            
            if stats.avg_loss != 0:
                stats.profit_factor = abs(stats.avg_win * stats.winning_trades / (stats.avg_loss * stats.losing_trades))
        
        # Calculate per-strategy stats (simplified)
        per_strategy: dict[str, StrategyStats] = {}
        
        # Build result
        return ReplayResult(
            config=config,
            start_time=config.start,
            end_time=config.end,
            duration_seconds=duration_seconds,
            initial_balance=config.initial_balance,
            final_balance=context.balance,
            final_equity=context.equity,
            total_pnl=context.realized_pnl,
            max_drawdown=context.max_drawdown,
            total_trades=context.total_trades,
            winning_trades=context.winning_trades,
            losing_trades=context.losing_trades,
            win_rate=context.winning_trades / max(context.total_trades, 1),
            avg_pnl=context.realized_pnl / max(context.total_trades, 1),
            total_commission=context.total_commission,
            total_slippage=context.total_slippage,
            sharpe_ratio=0.0,  # TODO: Calculate properly
            profit_factor=0.0,  # TODO: Calculate properly
            equity_curve=equity_curve,
            trades=context.closed_trades,
            events=events,
            per_symbol_stats=per_symbol,
            per_strategy_stats=per_strategy,
            emergency_stops=len([e for e in events if e.event_type == "EMERGENCY_STOP"]),
            policy_changes=0,  # TODO: Track policy changes
            risk_breaches=0,  # TODO: Track risk breaches
        )
