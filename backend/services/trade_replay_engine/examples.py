"""
Examples - Fake implementations and usage demonstrations
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import random

from .replay_config import ReplayConfig, ReplayMode
from .replay_market_data import ReplayMarketDataSource
from .exchange_simulator import ExchangeSimulator
from .trade_replay_engine import TradeReplayEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Fake Market Data Client ====================

class FakeMarketDataClient:
    """Fake market data client that generates synthetic OHLCV data"""
    
    def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Generate fake OHLCV data"""
        logger.info(f"Generating fake data for {symbol} from {start} to {end}")
        
        # Parse timeframe (e.g., "1h" -> 1 hour)
        if timeframe.endswith("h"):
            delta = timedelta(hours=int(timeframe[:-1]))
        elif timeframe.endswith("m"):
            delta = timedelta(minutes=int(timeframe[:-1]))
        elif timeframe.endswith("d"):
            delta = timedelta(days=int(timeframe[:-1]))
        else:
            delta = timedelta(hours=1)
        
        # Generate timestamps
        timestamps = []
        current = start
        while current <= end:
            timestamps.append(current)
            current += delta
        
        # Generate prices (random walk)
        base_price = random.uniform(100, 500)
        prices = [base_price]
        
        for _ in range(len(timestamps) - 1):
            change_pct = random.uniform(-0.03, 0.03)  # ±3%
            new_price = prices[-1] * (1 + change_pct)
            prices.append(new_price)
        
        # Build OHLCV
        data = []
        for i, ts in enumerate(timestamps):
            price = prices[i]
            high = price * random.uniform(1.001, 1.02)
            low = price * random.uniform(0.98, 0.999)
            open_price = random.uniform(low, high)
            close_price = random.uniform(low, high)
            volume = random.uniform(1000, 10000)
            
            data.append({
                "timestamp": ts,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume,
            })
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        return df


# ==================== Fake Components ====================

class FakeStrategyRuntimeEngine:
    """Fake runtime engine that generates random signals"""
    
    def generate_candidates(self, symbol: str, candle: dict) -> list[dict]:
        """Generate fake signals with 20% probability"""
        if random.random() < 0.2:  # 20% chance of signal
            return [{
                "symbol": symbol,
                "side": random.choice(["LONG", "SHORT"]),
                "size": random.uniform(50, 200),
                "confidence": random.uniform(0.6, 0.95),
                "strategy_id": "fake_strategy",
                "stop_loss": candle["close"] * (0.95 if random.random() > 0.5 else 1.05),
                "take_profit": candle["close"] * (1.05 if random.random() > 0.5 else 0.95),
            }]
        return []


class FakeOrchestrator:
    """Fake orchestrator that accepts all signals"""
    
    def filter_signals(self, candidates: list[dict], context: dict) -> list[dict]:
        """Accept all signals"""
        return candidates


class FakeRiskGuard:
    """Fake risk guard that rejects signals during high drawdown"""
    
    def validate_trade(self, signal: dict, context: dict) -> tuple[bool, str]:
        """Reject if drawdown > 15%"""
        if context.get("drawdown", 0) > 0.15:
            return False, "Drawdown too high"
        return True, "OK"


class FakePortfolioBalancer:
    """Fake portfolio balancer with max 5 positions"""
    
    def check_limits(self, signal: dict, context: dict) -> tuple[bool, str]:
        """Max 5 positions"""
        if context.get("open_positions", 0) >= 5:
            return False, "Max positions reached"
        return True, "OK"


class FakeSafetyGovernor:
    """Fake safety governor that approves all trades"""
    
    def approve_trade(self, signal: dict, context: dict) -> tuple[bool, str]:
        """Approve all trades unless emergency stop"""
        if context.get("emergency_stop", False):
            return False, "Emergency stop active"
        return True, "OK"


class FakePolicyStore:
    """Fake MSC policy store"""
    
    def __init__(self):
        self.policy = {"mode": "NORMAL", "risk_level": "MEDIUM"}
    
    def get_current_policy(self) -> dict:
        return self.policy
    
    def update_policy(self, market_conditions: dict) -> None:
        """Switch to DEFENSIVE if drawdown > 10%"""
        dd = market_conditions.get("drawdown", 0)
        if dd > 0.10:
            self.policy["mode"] = "DEFENSIVE"
        else:
            self.policy["mode"] = "NORMAL"


class FakeEmergencyStopSystem:
    """Fake ESS that triggers on 20% drawdown"""
    
    def check_conditions(self, context: dict) -> tuple[bool, str]:
        """Trigger on 20% drawdown"""
        dd = context.get("drawdown", 0)
        if dd > 0.20:
            return True, "Drawdown exceeded 20%"
        return False, "OK"


# ==================== Example Usage ====================

def example_full_replay():
    """
    Example: Full replay with all components
    """
    logger.info("=" * 60)
    logger.info("EXAMPLE: Full System Replay")
    logger.info("=" * 60)
    
    # Configuration
    config = ReplayConfig(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 31),
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        timeframe="1h",
        mode=ReplayMode.FULL,
        initial_balance=10_000.0,
        speed=0.0,  # Fast as possible
        include_msc=True,
        include_ess=True,
        slippage_model="realistic",
        commission_rate=0.001,
    )
    
    logger.info(f"Duration: {config.duration_days()} days")
    logger.info(f"Estimated candles: {config.estimated_candles()}")
    
    # Create components
    market_data_client = FakeMarketDataClient()
    market_data_source = ReplayMarketDataSource(market_data_client)
    exchange_simulator = ExchangeSimulator(
        commission_rate=config.commission_rate,
        slippage_model=config.slippage_model,
    )
    
    # Create fake Q-Trader components
    runtime_engine = FakeStrategyRuntimeEngine()
    orchestrator = FakeOrchestrator()
    risk_guard = FakeRiskGuard()
    portfolio_balancer = FakePortfolioBalancer()
    safety_governor = FakeSafetyGovernor()
    policy_store = FakePolicyStore()
    emergency_stop_system = FakeEmergencyStopSystem()
    
    # Create replay engine
    engine = TradeReplayEngine(
        market_data_source=market_data_source,
        exchange_simulator=exchange_simulator,
        runtime_engine=runtime_engine,
        orchestrator=orchestrator,
        risk_guard=risk_guard,
        portfolio_balancer=portfolio_balancer,
        safety_governor=safety_governor,
        policy_store=policy_store,
        emergency_stop_system=emergency_stop_system,
    )
    
    # Run replay
    logger.info("\nStarting replay...\n")
    result = engine.run(config)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("REPLAY RESULTS")
    logger.info("=" * 60)
    print(result.summary())
    
    # Detailed breakdown
    logger.info("\n" + "-" * 60)
    logger.info("PER-SYMBOL PERFORMANCE")
    logger.info("-" * 60)
    
    best_symbol = result.get_best_symbol()
    if best_symbol:
        stats = result.per_symbol_stats[best_symbol]
        logger.info(f"Best symbol: {best_symbol}")
        logger.info(f"  Trades: {stats.total_trades}")
        logger.info(f"  PnL: ${stats.total_pnl:.2f}")
        logger.info(f"  Win rate: {stats.win_rate*100:.1f}%")
    
    # Event summary
    logger.info("\n" + "-" * 60)
    logger.info("KEY EVENTS")
    logger.info("-" * 60)
    
    event_types = {}
    for event in result.events:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    
    for event_type, count in sorted(event_types.items(), key=lambda x: -x[1]):
        logger.info(f"  {event_type}: {count}")
    
    return result


def example_strategy_only():
    """
    Example: Strategy-only replay (no execution)
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Strategy-Only Replay")
    logger.info("=" * 60)
    
    config = ReplayConfig(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 7),
        symbols=["BTCUSDT"],
        timeframe="4h",
        mode=ReplayMode.STRATEGY_ONLY,
        initial_balance=5_000.0,
        include_msc=False,
        include_ess=False,
    )
    
    # Minimal setup
    market_data_client = FakeMarketDataClient()
    market_data_source = ReplayMarketDataSource(market_data_client)
    exchange_simulator = ExchangeSimulator()
    runtime_engine = FakeStrategyRuntimeEngine()
    
    engine = TradeReplayEngine(
        market_data_source=market_data_source,
        exchange_simulator=exchange_simulator,
        runtime_engine=runtime_engine,
    )
    
    logger.info("\nStarting strategy-only replay...\n")
    result = engine.run(config)
    
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY ANALYSIS")
    logger.info("=" * 60)
    print(result.summary())
    
    return result


def example_model_validation():
    """
    Example: Model-only replay for validation
    """
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Model Validation Replay")
    logger.info("=" * 60)
    
    config = ReplayConfig(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 3),
        symbols=["ETHUSDT", "SOLUSDT"],
        timeframe="1h",
        mode=ReplayMode.MODEL_ONLY,
        initial_balance=10_000.0,
    )
    
    market_data_client = FakeMarketDataClient()
    market_data_source = ReplayMarketDataSource(market_data_client)
    exchange_simulator = ExchangeSimulator()
    
    engine = TradeReplayEngine(
        market_data_source=market_data_source,
        exchange_simulator=exchange_simulator,
    )
    
    logger.info("\nStarting model validation...\n")
    result = engine.run(config)
    
    logger.info("\n" + "=" * 60)
    logger.info("MODEL PERFORMANCE")
    logger.info("=" * 60)
    print(result.summary())
    
    return result


if __name__ == "__main__":
    # Run all examples
    result1 = example_full_replay()
    result2 = example_strategy_only()
    result3 = example_model_validation()
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL EXAMPLES COMPLETED")
    logger.info("=" * 60)
