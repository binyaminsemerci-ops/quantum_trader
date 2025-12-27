"""
PolicyStore Integration with Existing Quantum Trader Components

Shows how to wire PolicyStore into the current system architecture.
"""

from typing import Protocol, Any
from policy_store import InMemoryPolicyStore, GlobalPolicy


# ============================================================================
# MOCK EXISTING COMPONENTS (for demonstration)
# ============================================================================

class Signal:
    """Mock signal from ensemble."""
    def __init__(self, symbol: str, strategy_id: str, confidence: float, 
                 direction: str, entry_price: float):
        self.symbol = symbol
        self.strategy_id = strategy_id
        self.confidence = confidence
        self.direction = direction
        self.entry_price = entry_price


class Position:
    """Mock position."""
    def __init__(self, symbol: str, size: float):
        self.symbol = symbol
        self.size = size


# ============================================================================
# INTEGRATION 1: ORCHESTRATOR POLICY
# ============================================================================

class OrchestratorPolicy:
    """
    Orchestrator now reads from PolicyStore for signal approval.
    
    Previously: Hard-coded thresholds
    Now: Dynamic thresholds from central policy
    """
    
    def __init__(self, policy_store: InMemoryPolicyStore):
        self.policy_store = policy_store
    
    def should_execute(self, signal: Signal) -> tuple[bool, str]:
        """
        Determine if signal should be executed.
        
        Returns:
            (should_execute: bool, reason: str)
        """
        # Read current policy
        policy = self.policy_store.get()
        
        # Check 1: Strategy allowed?
        if signal.strategy_id not in policy['allowed_strategies']:
            return False, f"Strategy {signal.strategy_id} not in allowed list"
        
        # Check 2: Symbol allowed?
        if signal.symbol not in policy['allowed_symbols']:
            return False, f"Symbol {signal.symbol} not in allowed list"
        
        # Check 3: Confidence threshold
        if signal.confidence < policy['global_min_confidence']:
            return False, f"Confidence {signal.confidence:.3f} below threshold {policy['global_min_confidence']:.3f}"
        
        # Check 4: Symbol ranking (if available)
        if signal.symbol in policy['opp_rankings']:
            ranking = policy['opp_rankings'][signal.symbol]
            if ranking < 0.7:  # Could also be configurable
                return False, f"Symbol ranking {ranking:.3f} too low"
        
        return True, "APPROVED"


# ============================================================================
# INTEGRATION 2: RISK GUARD
# ============================================================================

class RiskGuard:
    """
    RiskGuard now reads max risk from PolicyStore.
    
    Previously: Config file
    Now: Dynamic from PolicyStore (can change in real-time)
    """
    
    def __init__(self, policy_store: InMemoryPolicyStore, account_balance: float):
        self.policy_store = policy_store
        self.account_balance = account_balance
    
    def validate_trade(self, signal: Signal, position_size: float) -> tuple[bool, str]:
        """
        Validate proposed trade against risk limits.
        
        Args:
            signal: The trade signal
            position_size: Proposed position size in USD
            
        Returns:
            (is_valid: bool, reason: str)
        """
        policy = self.policy_store.get()
        
        # Calculate risk as fraction of account
        risk_fraction = position_size / self.account_balance
        max_allowed = policy['max_risk_per_trade']
        
        if risk_fraction > max_allowed:
            return False, f"Risk {risk_fraction:.4f} exceeds max {max_allowed:.4f}"
        
        return True, "Risk within limits"


# ============================================================================
# INTEGRATION 3: PORTFOLIO BALANCER
# ============================================================================

class PortfolioBalancer:
    """
    Portfolio Balancer checks capacity from PolicyStore.
    
    Previously: Hard-coded MAX_POSITIONS = 10
    Now: Dynamic from PolicyStore (MSC AI can adjust)
    """
    
    def __init__(self, policy_store: InMemoryPolicyStore):
        self.policy_store = policy_store
        self.open_positions: list[Position] = []
    
    def can_add_position(self) -> tuple[bool, str]:
        """Check if we can add another position."""
        policy = self.policy_store.get()
        
        current_count = len(self.open_positions)
        max_allowed = policy['max_positions']
        
        if current_count >= max_allowed:
            return False, f"At capacity: {current_count}/{max_allowed}"
        
        return True, f"Capacity available: {current_count}/{max_allowed}"
    
    def add_position(self, position: Position):
        """Add a position to the portfolio."""
        self.open_positions.append(position)
    
    def remove_position(self, symbol: str):
        """Remove a position."""
        self.open_positions = [p for p in self.open_positions if p.symbol != symbol]


# ============================================================================
# INTEGRATION 4: ENSEMBLE MANAGER
# ============================================================================

class EnsembleManager:
    """
    Ensemble Manager reads active model versions from PolicyStore.
    
    CLM updates model versions after retraining.
    Ensemble uses those versions for predictions.
    """
    
    def __init__(self, policy_store: InMemoryPolicyStore):
        self.policy_store = policy_store
        self.model_registry = {
            "xgboost": {"v13": "path/to/xgb_v13.pkl", "v14": "path/to/xgb_v14.pkl"},
            "lightgbm": {"v10": "path/to/lgb_v10.pkl", "v11": "path/to/lgb_v11.pkl"},
        }
    
    def get_active_models(self) -> dict[str, str]:
        """
        Get paths to currently active model versions.
        
        Returns:
            Dict mapping model_name -> model_path
        """
        policy = self.policy_store.get()
        active_versions = policy['model_versions']
        
        active_paths = {}
        for model_name, version in active_versions.items():
            if model_name in self.model_registry:
                if version in self.model_registry[model_name]:
                    active_paths[model_name] = self.model_registry[model_name][version]
        
        return active_paths
    
    def make_prediction(self, symbol: str, features: dict) -> float:
        """Make prediction using active models."""
        active_models = self.get_active_models()
        
        # Load and predict with active models
        predictions = []
        for model_name, model_path in active_models.items():
            # pred = load_model(model_path).predict(features)
            pred = 0.75  # Mock prediction
            predictions.append(pred)
        
        # Ensemble average
        return sum(predictions) / len(predictions) if predictions else 0.5


# ============================================================================
# INTEGRATION 5: SAFETY GOVERNOR
# ============================================================================

class SafetyGovernor:
    """
    Safety Governor can trigger emergency DEFENSIVE mode.
    
    Writes to PolicyStore when emergency conditions detected.
    """
    
    def __init__(self, policy_store: InMemoryPolicyStore):
        self.policy_store = policy_store
        self.original_policy = None
    
    def check_circuit_breaker(self, current_drawdown: float, max_allowed_dd: float) -> bool:
        """
        Check if circuit breaker should trigger.
        
        If yes, switch to DEFENSIVE mode immediately.
        """
        if current_drawdown > max_allowed_dd:
            print(f"⚠️  CIRCUIT BREAKER: DD {current_drawdown:.2%} > {max_allowed_dd:.2%}")
            
            # Save current policy
            self.original_policy = self.policy_store.get()
            
            # Switch to emergency DEFENSIVE mode
            self.policy_store.patch({
                "risk_mode": "DEFENSIVE",
                "max_risk_per_trade": 0.002,  # Very conservative
                "max_positions": 3,
                "global_min_confidence": 0.80,
                "allowed_strategies": [],  # Stop new entries
            })
            
            return True
        
        return False
    
    def restore_normal_operation(self):
        """Restore policy after emergency."""
        if self.original_policy:
            print("✅ Restoring normal operation")
            self.policy_store.update(self.original_policy)
            self.original_policy = None


# ============================================================================
# INTEGRATION 6: EXECUTOR
# ============================================================================

class Executor:
    """
    Main executor loop that coordinates all components.
    
    All components now share the same PolicyStore instance.
    """
    
    def __init__(self, policy_store: InMemoryPolicyStore, account_balance: float):
        self.policy_store = policy_store
        
        # Initialize all components with shared policy store
        self.orchestrator = OrchestratorPolicy(policy_store)
        self.risk_guard = RiskGuard(policy_store, account_balance)
        self.portfolio = PortfolioBalancer(policy_store)
        self.ensemble = EnsembleManager(policy_store)
        self.safety = SafetyGovernor(policy_store)
    
    def process_signal(self, signal: Signal) -> str:
        """
        Process a trading signal through the full pipeline.
        
        All stages read from the same PolicyStore.
        """
        print(f"\n📊 Processing signal: {signal.symbol} {signal.direction}")
        print(f"   Strategy: {signal.strategy_id}, Confidence: {signal.confidence:.3f}")
        
        # Stage 1: Orchestrator approval
        approved, reason = self.orchestrator.should_execute(signal)
        if not approved:
            print(f"   ❌ Orchestrator rejected: {reason}")
            return "REJECTED_ORCHESTRATOR"
        print(f"   ✅ Orchestrator approved")
        
        # Stage 2: Portfolio capacity
        can_add, capacity_msg = self.portfolio.can_add_position()
        if not can_add:
            print(f"   ❌ Portfolio full: {capacity_msg}")
            return "REJECTED_CAPACITY"
        print(f"   ✅ Capacity available: {capacity_msg}")
        
        # Stage 3: Risk validation
        position_size = 10000  # Mock: $10k position
        risk_ok, risk_msg = self.risk_guard.validate_trade(signal, position_size)
        if not risk_ok:
            print(f"   ❌ Risk check failed: {risk_msg}")
            return "REJECTED_RISK"
        print(f"   ✅ Risk approved: {risk_msg}")
        
        # Execute
        print(f"   🚀 EXECUTING TRADE")
        position = Position(signal.symbol, position_size)
        self.portfolio.add_position(position)
        
        return "EXECUTED"


# ============================================================================
# FULL INTEGRATION DEMO
# ============================================================================

def demo_full_integration():
    """Demonstrate complete system integration."""
    print("=" * 70)
    print("QUANTUM TRADER - PolicyStore Integration Demo")
    print("=" * 70)
    
    # Initialize shared policy store
    store = InMemoryPolicyStore()
    
    # Set initial policy (would be done by MSC AI)
    print("\n🔧 Initializing system policy...")
    store.update({
        "risk_mode": "NORMAL",
        "allowed_strategies": ["MOMENTUM_1", "MEANREV_2", "BREAKOUT_3"],
        "allowed_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "max_risk_per_trade": 0.01,
        "max_positions": 10,
        "global_min_confidence": 0.65,
        "opp_rankings": {
            "BTCUSDT": 0.92,
            "ETHUSDT": 0.88,
            "SOLUSDT": 0.85,
        },
        "model_versions": {
            "xgboost": "v14",
            "lightgbm": "v11",
        },
    })
    print("✅ Policy loaded")
    
    # Create executor (wires all components)
    executor = Executor(store, account_balance=100000)
    
    # Scenario 1: Good signal
    print("\n" + "=" * 70)
    print("SCENARIO 1: Valid Signal")
    print("=" * 70)
    signal1 = Signal(
        symbol="BTCUSDT",
        strategy_id="MOMENTUM_1",
        confidence=0.75,
        direction="LONG",
        entry_price=45000,
    )
    result = executor.process_signal(signal1)
    print(f"\n📈 Result: {result}")
    
    # Scenario 2: Low confidence
    print("\n" + "=" * 70)
    print("SCENARIO 2: Low Confidence Signal")
    print("=" * 70)
    signal2 = Signal(
        symbol="ETHUSDT",
        strategy_id="MEANREV_2",
        confidence=0.55,  # Below threshold
        direction="SHORT",
        entry_price=3000,
    )
    result = executor.process_signal(signal2)
    print(f"\n📉 Result: {result}")
    
    # Scenario 3: Symbol not allowed
    print("\n" + "=" * 70)
    print("SCENARIO 3: Non-Allowed Symbol")
    print("=" * 70)
    signal3 = Signal(
        symbol="DOGEUSDT",  # Not in allowed list
        strategy_id="MOMENTUM_1",
        confidence=0.80,
        direction="LONG",
        entry_price=0.10,
    )
    result = executor.process_signal(signal3)
    print(f"\n🚫 Result: {result}")
    
    # Scenario 4: MSC AI changes mode mid-session
    print("\n" + "=" * 70)
    print("SCENARIO 4: MSC AI Changes Risk Mode")
    print("=" * 70)
    print("🤖 MSC AI detected high volatility regime")
    print("   Switching to DEFENSIVE mode...")
    
    store.patch({
        "risk_mode": "DEFENSIVE",
        "max_risk_per_trade": 0.005,
        "max_positions": 5,
        "global_min_confidence": 0.75,
    })
    print("✅ Policy updated")
    
    # Try same signal as scenario 1 again
    print("\nRetrying previous signal with new policy:")
    result = executor.process_signal(signal1)
    print(f"\n📊 Result: {result} (rejected due to lower threshold)")
    
    # Scenario 5: Circuit breaker
    print("\n" + "=" * 70)
    print("SCENARIO 5: Emergency Circuit Breaker")
    print("=" * 70)
    current_dd = 0.12  # 12% drawdown
    max_dd = 0.10      # 10% limit
    
    if executor.safety.check_circuit_breaker(current_dd, max_dd):
        print("🛑 All trading stopped")
        
        policy = store.get()
        print(f"   Risk mode: {policy['risk_mode']}")
        print(f"   Max positions: {policy['max_positions']}")
        print(f"   Allowed strategies: {policy['allowed_strategies']}")
    
    print("\n" + "=" * 70)
    print("Demo completed - PolicyStore orchestrating all components!")
    print("=" * 70)


if __name__ == "__main__":
    demo_full_integration()
