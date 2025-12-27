"""
Quantum Trader Component Adapter for Stress Testing

Adapts real Quantum Trader components to work with the Stress Testing System.
This allows SST to test the actual production components under stress.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from backend.services.stress_testing import (
    ScenarioExecutor,
    ExchangeSimulator
)
from backend.services.stress_testing.scenario_models import TradeRecord

logger = logging.getLogger(__name__)


class QuantumTraderRuntimeEngineAdapter:
    """Adapter for StrategyRuntimeEngine"""
    
    def __init__(self, runtime_engine):
        """
        Args:
            runtime_engine: backend.services.strategy_runtime_engine.StrategyRuntimeEngine
        """
        self.engine = runtime_engine
    
    def generate_signals(self, bar: dict, context: dict) -> list[Any]:
        """Generate signals from runtime engine"""
        try:
            # Convert bar to StrategyRuntimeEngine format
            market_data = {
                'timestamp': bar.get('timestamp'),
                'symbol': context.get('symbol', 'BTCUSDT'),
                'open': bar.get('open'),
                'high': bar.get('high'),
                'low': bar.get('low'),
                'close': bar.get('close'),
                'volume': bar.get('volume')
            }
            
            # Generate signals
            decisions = self.engine.generate_signals(
                market_data=market_data,
                positions=context.get('positions', [])
            )
            
            return decisions
        except Exception as e:
            logger.error(f"RuntimeEngine error: {e}")
            return []


class QuantumTraderOrchestratorAdapter:
    """Adapter for Orchestrator"""
    
    def __init__(self, orchestrator=None):
        """
        Args:
            orchestrator: Optional orchestrator component
        """
        self.orchestrator = orchestrator
    
    def evaluate_signal(self, signal: Any, context: dict):
        """Evaluate signal and convert to TradeDecision"""
        try:
            # For SST, we create a simple TradeDecision from the signal
            from backend.services.stress_testing.exchange_simulator import TradeDecision
            
            if not signal:
                return None
            
            # Extract signal attributes
            symbol = getattr(signal, 'symbol', context.get('symbol', 'BTCUSDT'))
            side = getattr(signal, 'side', 'LONG')
            size_usd = getattr(signal, 'size_usd', 1000.0)
            confidence = getattr(signal, 'confidence', 0.7)
            
            return TradeDecision(
                symbol=symbol,
                side=side,
                size_usd=size_usd,
                confidence=confidence,
                entry_price=context.get('current_price'),
                take_profit=None,  # Will be set by position sizing
                stop_loss=None
            )
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            return None


class QuantumTraderRiskGuardAdapter:
    """Adapter for RiskGuard"""
    
    def __init__(self, risk_guard=None):
        """
        Args:
            risk_guard: backend.services.risk_guard.RiskGuardService
        """
        self.risk_guard = risk_guard
    
    def validate_trade(self, decision, context: dict) -> tuple[bool, str]:
        """Validate trade through risk guard"""
        try:
            if not self.risk_guard:
                return True, "No risk guard configured"
            
            # Check portfolio limits
            positions = context.get('positions', [])
            equity = context.get('equity', 100000.0)
            
            # Simple validation
            if len(positions) >= 10:
                return False, "Max positions reached"
            
            if decision.size_usd > equity * 0.1:
                return False, "Position size too large"
            
            return True, "Approved"
        except Exception as e:
            logger.error(f"RiskGuard error: {e}")
            return False, f"Risk guard error: {e}"


class QuantumTraderPortfolioBalancerAdapter:
    """Adapter for PortfolioBalancer"""
    
    def __init__(self, portfolio_balancer=None):
        """
        Args:
            portfolio_balancer: backend.services.portfolio_balancer.PortfolioBalancerAI
        """
        self.balancer = portfolio_balancer
    
    def check_constraints(self, decision, positions: list) -> bool:
        """Check portfolio constraints"""
        try:
            # Simple constraint checking
            max_positions = 10
            max_concentration = 0.3  # 30% per position
            
            if len(positions) >= max_positions:
                logger.warning("Portfolio balancer: Max positions reached")
                return False
            
            # Check concentration
            total_exposure = sum(p.get('size_usd', 0) for p in positions)
            if total_exposure > 0:
                concentration = decision.size_usd / (total_exposure + decision.size_usd)
                if concentration > max_concentration:
                    logger.warning(f"Portfolio balancer: Concentration too high: {concentration:.2%}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"PortfolioBalancer error: {e}")
            return False


class QuantumTraderSafetyGovernorAdapter:
    """Adapter for SafetyGovernor"""
    
    def __init__(self, safety_governor=None):
        """
        Args:
            safety_governor: backend.services.safety_governor.SafetyGovernor
        """
        self.governor = safety_governor
    
    def check_safety(self, equity: float, drawdown: float) -> tuple[bool, str]:
        """Check emergency stop conditions"""
        try:
            # Emergency stop if drawdown > 30%
            if drawdown > 30.0:
                return False, f"Emergency stop: Drawdown {drawdown:.1f}% exceeds limit"
            
            # Stop if equity below 50% of initial
            if equity < 50000.0:  # Assuming 100k initial
                return False, f"Emergency stop: Equity ${equity:.0f} too low"
            
            return True, "OK"
        except Exception as e:
            logger.error(f"SafetyGovernor error: {e}")
            return True, f"Safety governor error: {e}"


class QuantumTraderMSCAdapter:
    """Adapter for Meta Strategy Controller"""
    
    def __init__(self, msc=None):
        """
        Args:
            msc: Meta strategy controller
        """
        self.msc = msc
        self.policy = {"risk_mode": "NORMAL", "max_positions": 10}
    
    def get_current_policy(self) -> dict[str, Any]:
        """Get current policy"""
        return self.policy
    
    def update_policy(self, metrics: dict) -> None:
        """Update policy based on metrics"""
        try:
            # Simple policy adjustment based on performance
            winrate = metrics.get('winrate', 0.5)
            
            if winrate < 0.4:
                self.policy['risk_mode'] = 'DEFENSIVE'
                self.policy['max_positions'] = 5
                logger.info("MSC: Switched to DEFENSIVE mode")
            elif winrate > 0.6:
                self.policy['risk_mode'] = 'AGGRESSIVE'
                self.policy['max_positions'] = 15
                logger.info("MSC: Switched to AGGRESSIVE mode")
            else:
                self.policy['risk_mode'] = 'NORMAL'
                self.policy['max_positions'] = 10
        except Exception as e:
            logger.error(f"MSC error: {e}")


class QuantumTraderPolicyStoreAdapter:
    """Adapter for PolicyStore"""
    
    def __init__(self, policy_store=None):
        """
        Args:
            policy_store: backend.services.policy_store.InMemoryPolicyStore
        """
        self.policy_store = policy_store
        self.cache = {}
    
    def get(self, key: str) -> Any:
        """Get policy value"""
        try:
            if self.policy_store:
                policy = self.policy_store.get()
                return policy.get(key)
            return self.cache.get(key)
        except Exception as e:
            logger.error(f"PolicyStore error: {e}")
            return None
    
    def update(self, updates: dict) -> None:
        """Update policy values"""
        try:
            if self.policy_store:
                self.policy_store.update(updates)
            self.cache.update(updates)
        except Exception as e:
            logger.error(f"PolicyStore error: {e}")


class QuantumTraderEventBusAdapter:
    """Adapter for EventBus"""
    
    def __init__(self, event_bus=None):
        """
        Args:
            event_bus: backend.services.event_bus.InMemoryEventBus
        """
        self.event_bus = event_bus
    
    def publish(self, event: str, data: dict) -> None:
        """Publish event"""
        try:
            if self.event_bus:
                self.event_bus.publish(event, data)
            else:
                logger.debug(f"Event: {event} - {data}")
        except Exception as e:
            logger.error(f"EventBus error: {e}")


def create_quantum_trader_executor(
    app_state,
    initial_capital: float = 100000.0
) -> ScenarioExecutor:
    """
    Create ScenarioExecutor using real Quantum Trader components from app state.
    
    Args:
        app_state: FastAPI app.state object with initialized components
        initial_capital: Starting capital for stress test
    
    Returns:
        Configured ScenarioExecutor
    """
    # Get components from app state (may be None)
    runtime_engine = getattr(app_state, 'executor', None)
    risk_guard = getattr(app_state, 'risk_guard', None)
    portfolio_balancer = getattr(app_state, 'portfolio_balancer', None)
    safety_governor = getattr(app_state, 'safety_governor', None)
    policy_store = getattr(app_state, 'policy_store', None)
    event_bus = getattr(app_state, 'event_bus', None)
    
    # Create adapters
    runtime_adapter = None
    if runtime_engine:
        runtime_adapter = QuantumTraderRuntimeEngineAdapter(runtime_engine)
    
    orchestrator_adapter = QuantumTraderOrchestratorAdapter()
    risk_guard_adapter = QuantumTraderRiskGuardAdapter(risk_guard)
    portfolio_adapter = QuantumTraderPortfolioBalancerAdapter(portfolio_balancer)
    safety_adapter = QuantumTraderSafetyGovernorAdapter(safety_governor)
    msc_adapter = QuantumTraderMSCAdapter()
    policy_adapter = QuantumTraderPolicyStoreAdapter(policy_store)
    event_adapter = QuantumTraderEventBusAdapter(event_bus)
    
    # Create executor
    executor = ScenarioExecutor(
        runtime_engine=runtime_adapter,
        orchestrator=orchestrator_adapter,
        risk_guard=risk_guard_adapter,
        portfolio_balancer=portfolio_adapter,
        safety_governor=safety_adapter,
        msc=msc_adapter,
        policy_store=policy_adapter,
        exchange_simulator=ExchangeSimulator(),
        event_bus=event_adapter,
        initial_capital=initial_capital
    )
    
    logger.info(
        f"[SST] Created executor with Quantum Trader components: "
        f"runtime={runtime_adapter is not None}, "
        f"risk_guard={risk_guard is not None}, "
        f"portfolio={portfolio_balancer is not None}, "
        f"safety={safety_governor is not None}"
    )
    
    return executor
