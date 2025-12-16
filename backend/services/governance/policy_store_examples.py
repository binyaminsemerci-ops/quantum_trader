"""
PolicyStore Usage Examples

Demonstrates real-world usage patterns for the PolicyStore module.
Shows how different AI components interact with the central policy.
"""

from policy_store import (
    InMemoryPolicyStore,
    PolicyDefaults,
    GlobalPolicy,
    PolicyStoreFactory,
    RiskMode,
)


# ============================================================================
# EXAMPLE 1: Basic Setup and Initialization
# ============================================================================

def example_basic_setup():
    """Basic store initialization and first policy setup."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Setup")
    print("=" * 70)
    
    # Create store with default policy
    store = InMemoryPolicyStore()
    
    # View initial state
    print("\nInitial policy:")
    policy = store.get()
    print(f"  Risk Mode: {policy['risk_mode']}")
    print(f"  Max Positions: {policy['max_positions']}")
    print(f"  Max Risk/Trade: {policy['max_risk_per_trade']}")
    print(f"  Min Confidence: {policy['global_min_confidence']}")
    
    # Set up initial trading policy
    store.update({
        "risk_mode": "NORMAL",
        "allowed_strategies": ["MOMENTUM_LONG_1", "MEANREV_SHORT_3"],
        "allowed_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "max_risk_per_trade": 0.0075,
        "max_positions": 8,
        "global_min_confidence": 0.68,
        "opp_rankings": {
            "BTCUSDT": 0.92,
            "ETHUSDT": 0.87,
            "SOLUSDT": 0.85,
        },
        "model_versions": {
            "xgboost": "v14",
            "lightgbm": "v11",
            "nhits": "v9",
            "patchtst": "v7",
        },
    })
    
    print("\nAfter initialization:")
    policy = store.get()
    print(f"  Allowed Strategies: {policy['allowed_strategies']}")
    print(f"  Allowed Symbols: {policy['allowed_symbols']}")
    print(f"  Symbol Rankings: {policy['opp_rankings']}")
    print(f"  Model Versions: {policy['model_versions']}")


# ============================================================================
# EXAMPLE 2: MSC AI Managing Risk Modes
# ============================================================================

def example_msc_ai_risk_management():
    """Simulate Meta Strategy Controller changing risk modes."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: MSC AI Risk Management")
    print("=" * 70)
    
    store = InMemoryPolicyStore()
    
    # Initial setup
    store.update({
        "risk_mode": "NORMAL",
        "allowed_strategies": ["STRAT_1", "STRAT_2", "STRAT_3"],
        "max_risk_per_trade": 0.01,
        "max_positions": 10,
        "global_min_confidence": 0.65,
    })
    
    print("\n📊 Market conditions detected: High volatility + Strong trend")
    print("   MSC AI Decision: Switch to AGGRESSIVE mode")
    
    # MSC AI switches to aggressive
    store.patch({
        "risk_mode": "AGGRESSIVE",
        "max_risk_per_trade": 0.015,
        "max_positions": 12,
        "global_min_confidence": 0.58,
        "allowed_strategies": ["STRAT_1", "STRAT_2", "STRAT_3", "STRAT_7"],
    })
    
    policy = store.get()
    print(f"   New Risk Mode: {policy['risk_mode']}")
    print(f"   Max Risk/Trade: {policy['max_risk_per_trade']} (increased)")
    print(f"   Max Positions: {policy['max_positions']} (increased)")
    print(f"   Strategies: {len(policy['allowed_strategies'])} active")
    
    print("\n📊 Market conditions detected: Choppy + High correlation")
    print("   MSC AI Decision: Switch to DEFENSIVE mode")
    
    # MSC AI switches to defensive
    store.patch({
        "risk_mode": "DEFENSIVE",
        "max_risk_per_trade": 0.005,
        "max_positions": 5,
        "global_min_confidence": 0.75,
        "allowed_strategies": ["STRAT_1"],  # Only best strategy
    })
    
    policy = store.get()
    print(f"   New Risk Mode: {policy['risk_mode']}")
    print(f"   Max Risk/Trade: {policy['max_risk_per_trade']} (decreased)")
    print(f"   Max Positions: {policy['max_positions']} (decreased)")
    print(f"   Strategies: {len(policy['allowed_strategies'])} active (conservative)")


# ============================================================================
# EXAMPLE 3: Opportunity Ranker Updates
# ============================================================================

def example_opportunity_ranker():
    """Simulate OpportunityRanker updating symbol rankings."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Opportunity Ranker Updates")
    print("=" * 70)
    
    store = InMemoryPolicyStore()
    
    print("\n🔍 OppRank analyzing 50 symbols...")
    print("   Computing: Trend strength, volatility, liquidity, performance")
    
    # OppRank updates rankings
    store.patch({
        "opp_rankings": {
            "BTCUSDT": 0.94,
            "ETHUSDT": 0.91,
            "SOLUSDT": 0.88,
            "AVAXUSDT": 0.85,
            "MATICUSDT": 0.82,
        },
        "allowed_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT"],
    })
    
    policy = store.get()
    print("\n   Top 5 Opportunities:")
    for symbol, score in sorted(policy['opp_rankings'].items(), 
                                 key=lambda x: x[1], reverse=True):
        print(f"     {symbol:12s} Score: {score:.3f}")
    
    print("\n🔄 Market shift detected - reranking...")
    
    # Later: Rankings change
    store.patch({
        "opp_rankings": {
            "SOLUSDT": 0.96,   # SOL surging
            "BTCUSDT": 0.89,   # BTC cooling
            "ETHUSDT": 0.88,
            "AVAXUSDT": 0.84,
            "LINKUSDT": 0.83,  # LINK enters top 5
        },
        "allowed_symbols": ["SOLUSDT", "BTCUSDT", "ETHUSDT", "AVAXUSDT", "LINKUSDT"],
    })
    
    policy = store.get()
    print("\n   Updated Top 5:")
    for symbol, score in sorted(policy['opp_rankings'].items(), 
                                 key=lambda x: x[1], reverse=True):
        print(f"     {symbol:12s} Score: {score:.3f}")


# ============================================================================
# EXAMPLE 4: CLM Model Version Management
# ============================================================================

def example_clm_model_updates():
    """Simulate Continuous Learning Manager updating model versions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: CLM Model Version Management")
    print("=" * 70)
    
    store = InMemoryPolicyStore()
    
    # Initial model versions
    store.update({
        "model_versions": {
            "xgboost": "v12",
            "lightgbm": "v9",
            "nhits": "v8",
            "patchtst": "v6",
        },
    })
    
    print("\n📚 Current Model Versions:")
    policy = store.get()
    for model, version in policy['model_versions'].items():
        print(f"     {model:12s} {version}")
    
    print("\n🔬 CLM: Training XGBoost v13 with last 3 months data...")
    print("   Shadow evaluation: Sharpe +15%, DD improved")
    print("   Decision: Promote v13 to production")
    
    # CLM promotes new XGBoost version
    store.patch({
        "model_versions": {
            "xgboost": "v13",  # Updated
        }
    })
    
    print("\n🔬 CLM: Training LightGBM v10...")
    print("   Shadow evaluation: Sharpe -5%, more false signals")
    print("   Decision: Keep v9 in production")
    
    # v10 failed shadow eval, no update
    
    print("\n📚 Updated Model Versions:")
    policy = store.get()
    for model, version in policy['model_versions'].items():
        print(f"     {model:12s} {version}")


# ============================================================================
# EXAMPLE 5: Reading Policy in Trading Components
# ============================================================================

def example_component_reads():
    """Show how various components read policy."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Components Reading Policy")
    print("=" * 70)
    
    store = InMemoryPolicyStore()
    store.update({
        "risk_mode": "NORMAL",
        "allowed_strategies": ["MOMENTUM_1", "MEANREV_2"],
        "allowed_symbols": ["BTCUSDT", "ETHUSDT"],
        "max_risk_per_trade": 0.01,
        "max_positions": 10,
        "global_min_confidence": 0.65,
        "opp_rankings": {"BTCUSDT": 0.9, "ETHUSDT": 0.85},
    })
    
    # Component 1: RiskGuard
    print("\n🛡️  RiskGuard checking trade...")
    policy = store.get()
    proposed_risk = 0.008
    if proposed_risk <= policy['max_risk_per_trade']:
        print(f"   ✓ Risk {proposed_risk:.3f} within limit {policy['max_risk_per_trade']:.3f}")
    else:
        print(f"   ✗ Risk {proposed_risk:.3f} exceeds limit {policy['max_risk_per_trade']:.3f}")
    
    # Component 2: Orchestrator Policy
    print("\n🎯 Orchestrator evaluating signal...")
    signal_confidence = 0.68
    signal_symbol = "BTCUSDT"
    signal_strategy = "MOMENTUM_1"
    
    policy = store.get()
    
    checks = []
    checks.append(("Confidence", signal_confidence >= policy['global_min_confidence']))
    checks.append(("Symbol allowed", signal_symbol in policy['allowed_symbols']))
    checks.append(("Strategy allowed", signal_strategy in policy['allowed_strategies']))
    
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"   {status} {check_name}")
    
    if all(check[1] for check in checks):
        print("   → Signal APPROVED")
    else:
        print("   → Signal REJECTED")
    
    # Component 3: Portfolio Balancer
    print("\n⚖️  Portfolio Balancer checking capacity...")
    current_positions = 8
    policy = store.get()
    
    if current_positions < policy['max_positions']:
        print(f"   ✓ Can add position ({current_positions}/{policy['max_positions']})")
    else:
        print(f"   ✗ At capacity ({current_positions}/{policy['max_positions']})")


# ============================================================================
# EXAMPLE 6: Strategy Generator Integration
# ============================================================================

def example_strategy_generator():
    """Show Strategy Generator reading and updating policy."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Strategy Generator Workflow")
    print("=" * 70)
    
    store = InMemoryPolicyStore()
    store.update({
        "allowed_strategies": ["STRAT_1", "STRAT_5"],
        "risk_mode": "NORMAL",
    })
    
    print("\n🧬 SG AI: Generated 10 new strategy variations")
    print("   Backtesting on 2 years historical data...")
    print("   3 strategies passed minimum thresholds")
    print("   Starting shadow mode for STRAT_23, STRAT_24, STRAT_25")
    
    print("\n📊 After 1 week of shadow trading:")
    print("   STRAT_23: Sharpe 2.1, DD 8%  ✓ Excellent")
    print("   STRAT_24: Sharpe 1.3, DD 15% ✓ Acceptable")
    print("   STRAT_25: Sharpe 0.4, DD 22% ✗ Poor")
    
    print("\n🎯 SG AI Decision: Promote STRAT_23 and STRAT_24 to production")
    
    # SG AI adds new strategies
    policy = store.get()
    current_strategies = policy['allowed_strategies']
    new_strategies = current_strategies + ["STRAT_23", "STRAT_24"]
    
    store.patch({
        "allowed_strategies": new_strategies,
    })
    
    policy = store.get()
    print(f"\n   Active strategies: {policy['allowed_strategies']}")
    
    print("\n📉 After 2 weeks:")
    print("   STRAT_5: Sharpe degraded to 0.6, underperforming")
    print("   SG AI Decision: Demote STRAT_5")
    
    # Remove underperforming strategy
    policy = store.get()
    updated_strategies = [s for s in policy['allowed_strategies'] if s != "STRAT_5"]
    
    store.patch({
        "allowed_strategies": updated_strategies,
    })
    
    policy = store.get()
    print(f"   Active strategies: {policy['allowed_strategies']}")


# ============================================================================
# EXAMPLE 7: Complete Day-in-the-Life
# ============================================================================

def example_full_day_simulation():
    """Simulate a full day of trading with policy updates."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Full Day Simulation")
    print("=" * 70)
    
    store = InMemoryPolicyStore()
    
    print("\n🌅 06:00 UTC - System startup")
    store.update({
        "risk_mode": "NORMAL",
        "allowed_strategies": ["MOMENTUM_1", "MEANREV_2", "BREAKOUT_3"],
        "allowed_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "max_risk_per_trade": 0.01,
        "max_positions": 10,
        "global_min_confidence": 0.65,
        "opp_rankings": {"BTCUSDT": 0.9, "ETHUSDT": 0.87, "SOLUSDT": 0.84},
        "model_versions": {"xgboost": "v14", "lightgbm": "v11"},
    })
    print("   ✓ Policy loaded")
    
    print("\n📊 09:00 UTC - OppRank hourly update")
    store.patch({
        "opp_rankings": {"BTCUSDT": 0.93, "ETHUSDT": 0.89, "SOLUSDT": 0.86}
    })
    print("   ✓ Rankings refreshed")
    
    print("\n🎯 12:00 UTC - MSC AI regime analysis")
    print("   Detected: Strong uptrend + Low volatility")
    print("   Decision: Switch to AGGRESSIVE")
    store.patch({
        "risk_mode": "AGGRESSIVE",
        "max_risk_per_trade": 0.015,
        "max_positions": 12,
        "global_min_confidence": 0.60,
    })
    print("   ✓ Risk parameters adjusted")
    
    print("\n🔬 15:00 UTC - CLM completes training")
    print("   XGBoost v15 shadow eval: +18% improvement")
    store.patch({
        "model_versions": {"xgboost": "v15"}
    })
    print("   ✓ Model promoted to production")
    
    print("\n⚠️  18:00 UTC - Volatility spike detected")
    print("   VIX +40%, correlation breakdown")
    print("   MSC AI: Emergency DEFENSIVE mode")
    store.patch({
        "risk_mode": "DEFENSIVE",
        "max_risk_per_trade": 0.005,
        "max_positions": 5,
        "global_min_confidence": 0.75,
        "allowed_strategies": ["MOMENTUM_1"],  # Most stable
    })
    print("   ✓ Risk reduced")
    
    print("\n🌙 22:00 UTC - Market stabilized")
    print("   Return to NORMAL mode")
    store.patch({
        "risk_mode": "NORMAL",
        "max_risk_per_trade": 0.01,
        "max_positions": 10,
        "global_min_confidence": 0.65,
        "allowed_strategies": ["MOMENTUM_1", "MEANREV_2", "BREAKOUT_3"],
    })
    print("   ✓ Operations normalized")
    
    print("\n📈 Final Policy State:")
    policy = store.get_policy_object()
    print(f"   Risk Mode: {policy.risk_mode}")
    print(f"   Active Strategies: {len(policy.allowed_strategies)}")
    print(f"   Model Versions: {policy.model_versions}")
    print(f"   Max Risk/Trade: {policy.max_risk_per_trade}")


# ============================================================================
# EXAMPLE 8: Factory Pattern Usage
# ============================================================================

def example_factory_usage():
    """Demonstrate using factory for different backends."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Storage Backend Selection")
    print("=" * 70)
    
    # Development: In-memory
    print("\n🔧 Development mode: Using in-memory store")
    dev_store = PolicyStoreFactory.create("memory")
    dev_store.update({"risk_mode": "NORMAL", "max_positions": 10})
    print(f"   Policy loaded: {dev_store.get()['risk_mode']}")
    
    # Production options (stubs)
    print("\n🏭 Production mode: Backend options available")
    print("   - PostgreSQL: High concurrency, ACID guarantees")
    print("   - Redis: Sub-ms latency, pub/sub support")
    print("   - SQLite: Embedded, no external dependencies")
    
    # Example of switching backends
    print("\n💡 Switching backends via configuration:")
    print("   store = PolicyStoreFactory.create('postgres', connection_pool=pool)")
    print("   store = PolicyStoreFactory.create('redis', redis_client=client)")
    print("   store = PolicyStoreFactory.create('sqlite', db_path='policy.db')")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

def run_all_examples():
    """Run all usage examples."""
    example_basic_setup()
    example_msc_ai_risk_management()
    example_opportunity_ranker()
    example_clm_model_updates()
    example_component_reads()
    example_strategy_generator()
    example_full_day_simulation()
    example_factory_usage()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
