"""
Scenario Executor - Runs full system simulation under stress
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Protocol, Any

from .scenario_models import Scenario, ScenarioResult, TradeRecord, ExecutionResult
from .exchange_simulator import ExchangeSimulator, TradeDecision

logger = logging.getLogger(__name__)


# Protocol definitions for system components
class StrategyRuntimeEngine(Protocol):
    """Generates trading signals"""
    def generate_signals(self, bar: dict, context: dict) -> list[Any]: ...


class Orchestrator(Protocol):
    """Filters and validates signals"""
    def evaluate_signal(self, signal: Any, context: dict) -> TradeDecision | None: ...


class RiskGuard(Protocol):
    """Pre-trade risk validation"""
    def validate_trade(self, decision: TradeDecision, context: dict) -> tuple[bool, str]: ...


class PortfolioBalancer(Protocol):
    """Portfolio-level constraints"""
    def check_constraints(self, decision: TradeDecision, positions: list) -> bool: ...


class SafetyGovernor(Protocol):
    """Emergency stop system"""
    def check_safety(self, equity: float, drawdown: float) -> tuple[bool, str]: ...


class MetaStrategyController(Protocol):
    """Top-level policy controller"""
    def get_current_policy(self) -> dict[str, Any]: ...
    def update_policy(self, metrics: dict) -> None: ...


class PolicyStore(Protocol):
    """Policy storage"""
    def get(self, key: str) -> Any: ...
    def update(self, updates: dict) -> None: ...


class EventBus(Protocol):
    """Event distribution"""
    def publish(self, event: str, data: dict) -> None: ...


class ScenarioExecutor:
    """
    Executes full system simulation under stress conditions.
    
    Steps through market data bar-by-bar, feeding all system components,
    simulating trades, tracking PnL, and recording system behavior.
    """
    
    def __init__(
        self,
        runtime_engine: StrategyRuntimeEngine | None = None,
        orchestrator: Orchestrator | None = None,
        risk_guard: RiskGuard | None = None,
        portfolio_balancer: PortfolioBalancer | None = None,
        safety_governor: SafetyGovernor | None = None,
        msc: MetaStrategyController | None = None,
        policy_store: PolicyStore | None = None,
        exchange_simulator: ExchangeSimulator | None = None,
        event_bus: EventBus | None = None,
        initial_capital: float = 100000.0
    ):
        """
        Initialize executor with system components.
        
        Args:
            runtime_engine: Strategy signal generator
            orchestrator: Signal filter/validator
            risk_guard: Pre-trade risk checker
            portfolio_balancer: Portfolio constraint manager
            safety_governor: Emergency stop system
            msc: Meta strategy controller
            policy_store: Global policy storage
            exchange_simulator: Order execution simulator
            event_bus: Event distribution
            initial_capital: Starting capital
        """
        self.runtime_engine = runtime_engine
        self.orchestrator = orchestrator
        self.risk_guard = risk_guard
        self.portfolio_balancer = portfolio_balancer
        self.safety_governor = safety_governor
        self.msc = msc
        self.policy_store = policy_store
        self.exchange_simulator = exchange_simulator or ExchangeSimulator()
        self.event_bus = event_bus
        self.initial_capital = initial_capital
        
        logger.info("[SST] ScenarioExecutor initialized")
    
    def run(self, df: pd.DataFrame, scenario: Scenario) -> ScenarioResult:
        """
        Execute scenario simulation.
        
        Args:
            df: Market data (potentially stressed)
            scenario: Scenario definition
            
        Returns:
            ScenarioResult with complete simulation output
        """
        logger.info(f"[SST] Starting simulation: {scenario.name}")
        start_time = datetime.utcnow()
        
        # Initialize result
        result = ScenarioResult(
            scenario_name=scenario.name,
            start_time=start_time
        )
        
        # Initialize simulation state
        state = {
            "equity": self.initial_capital,
            "cash": self.initial_capital,
            "positions": [],
            "closed_trades": [],
            "emergency_stopped": False,
            "bar_index": 0
        }
        
        result.equity_curve.append(self.initial_capital)
        result.pnl_curve.append(0.0)
        
        # Get symbols
        symbols = df["symbol"].unique()
        
        # Simulate bar by bar
        for idx in range(len(df)):
            # Get current bars for all symbols
            current_bars = {}
            for symbol in symbols:
                symbol_df = df[df["symbol"] == symbol]
                if idx < len(symbol_df):
                    bar = symbol_df.iloc[idx].to_dict()
                    current_bars[symbol] = bar
            
            if not current_bars:
                continue
            
            state["bar_index"] = idx
            
            # Update positions with current prices
            self._update_positions(state, current_bars)
            
            # Check safety governor
            if self.safety_governor:
                current_dd = self._calculate_drawdown(result.equity_curve)
                is_safe, reason = self.safety_governor.check_safety(
                    state["equity"], current_dd
                )
                
                if not is_safe:
                    result.emergency_stops += 1
                    result.notes.append(f"Bar {idx}: ESS triggered - {reason}")
                    state["emergency_stopped"] = True
                    logger.warning(f"[SST] Emergency stop at bar {idx}: {reason}")
            
            # Skip trading if emergency stopped
            if state["emergency_stopped"]:
                # Still update equity curve
                result.equity_curve.append(state["equity"])
                result.pnl_curve.append(state["equity"] - self.initial_capital)
                continue
            
            # Update MSC policy
            if self.msc and idx % 30 == 0:  # Every 30 bars
                try:
                    metrics = self._collect_metrics(state, result)
                    self.msc.update_policy(metrics)
                    
                    # Record policy transition
                    policy = self.msc.get_current_policy()
                    result.policy_transitions.append({
                        "bar": idx,
                        "policy": policy
                    })
                except Exception as e:
                    logger.error(f"[SST] MSC update failed: {e}")
                    result.failed_models.append("MSC")
            
            # Generate signals
            signals = self._generate_signals(current_bars, state)
            
            # Process each signal
            for signal in signals:
                # Check if trading allowed
                if len(state["positions"]) >= 10:  # Max positions
                    continue
                
                # Orchestrator decision
                decision = self._evaluate_signal(signal, state, current_bars)
                if not decision:
                    continue
                
                # Risk guard validation
                if self.risk_guard:
                    is_valid, reason = self.risk_guard.validate_trade(decision, state)
                    if not is_valid:
                        result.notes.append(f"Bar {idx}: Trade blocked by RiskGuard - {reason}")
                        continue
                
                # Portfolio balancer check
                if self.portfolio_balancer:
                    if not self.portfolio_balancer.check_constraints(decision, state["positions"]):
                        result.notes.append(f"Bar {idx}: Trade blocked by PortfolioBalancer")
                        continue
                
                # Execute trade
                bar = current_bars.get(decision.symbol, {})
                execution = self.exchange_simulator.execute_order(decision, bar, scenario)
                
                if not execution.success:
                    result.execution_failures += 1
                    result.notes.append(f"Bar {idx}: Execution failed - {execution.error_reason}")
                    continue
                
                # Open position
                self._open_position(state, decision, execution, idx)
            
            # Check for data quality issues
            for symbol, bar in current_bars.items():
                if bar.get("stressed") and "corruption" in bar.get("stress_type", ""):
                    result.data_quality_issues += 1
            
            # Update equity curve
            result.equity_curve.append(state["equity"])
            result.pnl_curve.append(state["equity"] - self.initial_capital)
        
        # Finalize result
        result.end_time = datetime.utcnow()
        result.duration_seconds = (result.end_time - start_time).total_seconds()
        
        # Close all remaining positions
        self._close_all_positions(state, df, result)
        
        # Calculate final metrics
        self._calculate_final_metrics(result, state)
        
        logger.info(
            f"[SST] Simulation complete: {result.total_trades} trades, "
            f"{result.max_drawdown:.2f}% max DD, "
            f"{result.emergency_stops} ESS activations"
        )
        
        return result
    
    def _generate_signals(self, bars: dict, state: dict) -> list[Any]:
        """Generate trading signals from runtime engine"""
        if not self.runtime_engine:
            return []
        
        try:
            signals = []
            for symbol, bar in bars.items():
                sig = self.runtime_engine.generate_signals(bar, state)
                if sig:
                    signals.extend(sig)
            return signals
        except Exception as e:
            logger.error(f"[SST] Signal generation failed: {e}")
            return []
    
    def _evaluate_signal(
        self,
        signal: Any,
        state: dict,
        bars: dict
    ) -> TradeDecision | None:
        """Evaluate signal through orchestrator"""
        if not self.orchestrator:
            # Create mock decision if no orchestrator
            return None
        
        try:
            decision = self.orchestrator.evaluate_signal(signal, state)
            return decision
        except Exception as e:
            logger.error(f"[SST] Orchestrator evaluation failed: {e}")
            return None
    
    def _open_position(
        self,
        state: dict,
        decision: TradeDecision,
        execution: ExecutionResult,
        bar_idx: int
    ):
        """Open a new position"""
        position = {
            "symbol": decision.symbol,
            "side": decision.side,
            "entry_price": execution.filled_price,
            "qty": execution.filled_qty,
            "entry_bar": bar_idx,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "strategy": getattr(decision, "strategy", "unknown"),
            "current_price": execution.filled_price
        }
        
        state["positions"].append(position)
        
        # Update cash
        cost = execution.filled_price * execution.filled_qty
        if decision.side == "BUY":
            state["cash"] -= cost
        else:
            state["cash"] += cost
    
    def _update_positions(self, state: dict, bars: dict):
        """Update positions with current prices and check exit conditions"""
        to_close = []
        
        for i, pos in enumerate(state["positions"]):
            symbol = pos["symbol"]
            if symbol not in bars:
                continue
            
            current_price = bars[symbol]["close"]
            pos["current_price"] = current_price
            
            # Check TP/SL
            if pos["side"] == "BUY":
                pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
                
                if pos["stop_loss"] and current_price <= pos["stop_loss"]:
                    to_close.append((i, "SL", current_price))
                elif pos["take_profit"] and current_price >= pos["take_profit"]:
                    to_close.append((i, "TP", current_price))
            
            else:  # SELL
                pnl_pct = (pos["entry_price"] - current_price) / pos["entry_price"]
                
                if pos["stop_loss"] and current_price >= pos["stop_loss"]:
                    to_close.append((i, "SL", current_price))
                elif pos["take_profit"] and current_price <= pos["take_profit"]:
                    to_close.append((i, "TP", current_price))
        
        # Close positions
        for idx, reason, exit_price in reversed(to_close):
            self._close_position(state, idx, exit_price, reason)
        
        # Update equity
        position_value = sum(
            pos["current_price"] * pos["qty"] if pos["side"] == "BUY"
            else pos["entry_price"] * pos["qty"] - (pos["current_price"] - pos["entry_price"]) * pos["qty"]
            for pos in state["positions"]
        )
        state["equity"] = state["cash"] + position_value
    
    def _close_position(
        self,
        state: dict,
        position_idx: int,
        exit_price: float,
        reason: str
    ):
        """Close a position"""
        pos = state["positions"].pop(position_idx)
        
        # Calculate PnL
        if pos["side"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["qty"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["qty"]
        
        # Update cash
        if pos["side"] == "BUY":
            state["cash"] += exit_price * pos["qty"]
        else:
            state["cash"] -= exit_price * pos["qty"]
        
        # Record trade
        trade = TradeRecord(
            symbol=pos["symbol"],
            side=pos["side"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            qty=pos["qty"],
            pnl=pnl,
            strategy=pos["strategy"],
            entry_time=datetime.utcnow(),  # Mock
            exit_time=datetime.utcnow(),
            closed=True,
            exit_reason=reason
        )
        
        state["closed_trades"].append(trade)
    
    def _close_all_positions(self, state: dict, df: pd.DataFrame, result: ScenarioResult):
        """Close all remaining positions at end of simulation"""
        for pos in state["positions"]:
            # Get last price
            symbol_df = df[df["symbol"] == pos["symbol"]]
            if len(symbol_df) > 0:
                last_price = symbol_df.iloc[-1]["close"]
                self._close_position(state, 0, last_price, "end_of_simulation")
        
        state["positions"].clear()
    
    def _calculate_drawdown(self, equity_curve: list[float]) -> float:
        """Calculate current drawdown percentage"""
        if len(equity_curve) < 2:
            return 0.0
        
        peak = max(equity_curve)
        current = equity_curve[-1]
        
        if peak == 0:
            return 0.0
        
        dd = ((peak - current) / peak) * 100
        return dd
    
    def _collect_metrics(self, state: dict, result: ScenarioResult) -> dict:
        """Collect metrics for MSC"""
        return {
            "equity": state["equity"],
            "open_positions": len(state["positions"]),
            "total_trades": len(state["closed_trades"]),
            "recent_pnl": result.pnl_curve[-10:] if len(result.pnl_curve) >= 10 else result.pnl_curve
        }
    
    def _calculate_final_metrics(self, result: ScenarioResult, state: dict):
        """Calculate final performance metrics"""
        # Collect trades
        result.trades = state["closed_trades"]
        result.total_trades = len(result.trades)
        
        # Win/loss counts
        result.winning_trades = sum(1 for t in result.trades if t.pnl > 0)
        result.losing_trades = sum(1 for t in result.trades if t.pnl <= 0)
        
        # Calculate max drawdown
        if len(result.equity_curve) > 0:
            peak = result.equity_curve[0]
            max_dd = 0.0
            max_dd_duration = 0
            current_dd_duration = 0
            
            for equity in result.equity_curve:
                if equity > peak:
                    peak = equity
                    current_dd_duration = 0
                else:
                    current_dd_duration += 1
                    dd = ((peak - equity) / peak) * 100
                    if dd > max_dd:
                        max_dd = dd
                        max_dd_duration = current_dd_duration
            
            result.max_drawdown = max_dd
            result.max_drawdown_duration = max_dd_duration
        
        # Calculate Sharpe ratio
        if len(result.pnl_curve) > 1:
            returns = np.diff(result.pnl_curve)
            if returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(24 * 365)  # Hourly to annual
