"""
Scenario Library - Pre-defined stress test scenarios
"""

from datetime import datetime, timedelta
from .scenario_models import Scenario, ScenarioType


class ScenarioLibrary:
    """
    Collection of pre-defined stress test scenarios.
    
    Includes:
    - Historical market crashes
    - Synthetic extreme conditions
    - Multi-stress combinations
    """
    
    @staticmethod
    def get_all() -> list[Scenario]:
        """Get all library scenarios"""
        return [
            ScenarioLibrary.btc_flash_crash_2021(),
            ScenarioLibrary.eth_liquidity_collapse(),
            ScenarioLibrary.sol_volatility_explosion(),
            ScenarioLibrary.mixed_regime_chaos(),
            ScenarioLibrary.model_failure_simulation(),
            ScenarioLibrary.data_corruption_shock(),
            ScenarioLibrary.execution_crisis(),
            ScenarioLibrary.correlation_breakdown_test(),
            ScenarioLibrary.pump_and_dump_mania(),
            ScenarioLibrary.black_swan_combo()
        ]
    
    @staticmethod
    def btc_flash_crash_2021() -> Scenario:
        """
        Simulate May 2021 BTC flash crash (-30% in hours).
        """
        return Scenario(
            name="BTC Flash Crash May 2021",
            type=ScenarioType.FLASH_CRASH,
            parameters={
                "drop_pct": 0.30,
                "duration_bars": 5,
                "recovery_bars": 15
            },
            symbols=["BTCUSDT"],
            description="Recreates the May 2021 crash where BTC dropped 30% in hours"
        )
    
    @staticmethod
    def eth_liquidity_collapse() -> Scenario:
        """
        Extreme liquidity drought scenario.
        """
        return Scenario(
            name="ETH Liquidity Collapse",
            type=ScenarioType.LIQUIDITY_DROP,
            parameters={
                "drop_pct": 0.95,
                "duration_bars": 20
            },
            symbols=["ETHUSDT"],
            description="95% volume drop simulating extreme illiquidity"
        )
    
    @staticmethod
    def sol_volatility_explosion() -> Scenario:
        """
        Massive volatility spike.
        """
        return Scenario(
            name="SOL Volatility Explosion",
            type=ScenarioType.VOLATILITY_SPIKE,
            parameters={
                "multiplier": 8.0,
                "duration_bars": 30
            },
            symbols=["SOLUSDT"],
            description="8x volatility increase for extended period"
        )
    
    @staticmethod
    def mixed_regime_chaos() -> Scenario:
        """
        Multiple stress conditions simultaneously.
        """
        return Scenario(
            name="Mixed Regime Chaos",
            type=ScenarioType.MIXED_CUSTOM,
            parameters={
                "flash_crash_drop_pct": 0.20,
                "volatility_multiplier": 5.0,
                "spread_mult": 10.0,
                "liquidity_drop_pct": 0.80
            },
            symbols=["BTCUSDT", "ETHUSDT"],
            description="Flash crash + vol spike + spread explosion + liquidity drain"
        )
    
    @staticmethod
    def model_failure_simulation() -> Scenario:
        """
        Test system behavior when models fail.
        """
        return Scenario(
            name="Model Failure Simulation",
            type=ScenarioType.MODEL_FAILURE,
            parameters={
                "failure_rate": 0.30,
                "affected_models": ["xgboost", "lightgbm"]
            },
            symbols=["BTCUSDT", "ETHUSDT"],
            description="30% of model predictions fail or timeout"
        )
    
    @staticmethod
    def data_corruption_shock() -> Scenario:
        """
        Data feed corruption and gaps.
        """
        return Scenario(
            name="Data Corruption Shock",
            type=ScenarioType.DATA_CORRUPTION,
            parameters={
                "corruption_pct": 0.15,
                "corruption_types": ["nan", "spike", "duplicate"]
            },
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            description="15% of data corrupted with NaNs, spikes, duplicates"
        )
    
    @staticmethod
    def execution_crisis() -> Scenario:
        """
        High execution failure rate.
        """
        return Scenario(
            name="Execution Crisis",
            type=ScenarioType.EXECUTION_FAILURE,
            parameters={
                "failure_rate": 0.40,
                "latency_spike_mult": 10.0
            },
            symbols=["BTCUSDT", "ETHUSDT"],
            description="40% execution failure rate with 10x latency"
        )
    
    @staticmethod
    def correlation_breakdown_test() -> Scenario:
        """
        Multi-asset correlation breakdown.
        """
        return Scenario(
            name="Correlation Breakdown",
            type=ScenarioType.CORRELATION_BREAKDOWN,
            parameters={
                "duration_bars": 40
            },
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            description="BTC/ETH/BNB correlations break down completely"
        )
    
    @staticmethod
    def pump_and_dump_mania() -> Scenario:
        """
        Coordinated pump and dump.
        """
        return Scenario(
            name="Pump & Dump Mania",
            type=ScenarioType.PUMP_DUMP,
            parameters={
                "pump_pct": 1.20,  # 120% pump
                "pump_duration": 8,
                "dump_duration": 4
            },
            symbols=["DOGEUSDT"],
            description="Extreme 120% pump followed by rapid dump"
        )
    
    @staticmethod
    def black_swan_combo() -> Scenario:
        """
        Worst-case scenario: everything breaks at once.
        """
        return Scenario(
            name="Black Swan Combo",
            type=ScenarioType.MIXED_CUSTOM,
            parameters={
                "flash_crash_drop_pct": 0.40,
                "volatility_multiplier": 10.0,
                "spread_mult": 15.0,
                "liquidity_drop_pct": 0.95,
                "corruption_pct": 0.20
            },
            symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            description="Catastrophic multi-failure scenario"
        )
    
    @staticmethod
    def get_by_name(name: str) -> Scenario | None:
        """Get scenario by name"""
        scenarios = {s.name: s for s in ScenarioLibrary.get_all()}
        return scenarios.get(name)
    
    @staticmethod
    def get_by_type(scenario_type: ScenarioType) -> list[Scenario]:
        """Get all scenarios of a specific type"""
        return [s for s in ScenarioLibrary.get_all() if s.type == scenario_type]
