"""
Scenario Loader - Loads and prepares data for stress testing
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Protocol

from .scenario_models import Scenario, ScenarioType

logger = logging.getLogger(__name__)


class MarketDataClient(Protocol):
    """Interface for fetching market data"""
    
    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        ...


class ScenarioLoader:
    """
    Loads market data for scenarios.
    
    For historical replays: fetches real data from exchange
    For synthetic scenarios: generates appropriate baseline data
    """
    
    def __init__(self, market_data_client: MarketDataClient | None = None):
        """
        Initialize loader.
        
        Args:
            market_data_client: Client for fetching historical data
        """
        self.market_data = market_data_client
        logger.info("[SST] ScenarioLoader initialized")
    
    def load_data(self, scenario: Scenario) -> pd.DataFrame:
        """
        Load or generate market data for scenario.
        
        Args:
            scenario: Scenario definition
            
        Returns:
            DataFrame with OHLCV data (multi-symbol)
            
        Raises:
            ValueError: If data cannot be loaded
        """
        logger.info(f"[SST] Loading data for scenario: {scenario.name}")
        
        if scenario.type == ScenarioType.HISTORIC_REPLAY:
            return self._load_historical_data(scenario)
        else:
            return self._generate_synthetic_data(scenario)
    
    def _load_historical_data(self, scenario: Scenario) -> pd.DataFrame:
        """
        Load real historical data for replay scenarios.
        
        Args:
            scenario: Historical replay scenario
            
        Returns:
            Multi-symbol OHLCV DataFrame
        """
        if not self.market_data:
            logger.warning("[SST] No market data client, using synthetic data")
            return self._generate_synthetic_data(scenario)
        
        if not scenario.start or not scenario.end:
            raise ValueError("Historical replay requires start and end dates")
        
        symbols = scenario.symbols or ["BTCUSDT", "ETHUSDT"]
        dfs = []
        
        for symbol in symbols:
            try:
                df = self.market_data.get_historical_klines(
                    symbol=symbol,
                    interval="1h",
                    start=scenario.start,
                    end=scenario.end
                )
                df["symbol"] = symbol
                dfs.append(df)
                logger.info(f"[SST] Loaded {len(df)} bars for {symbol}")
            except Exception as e:
                logger.error(f"[SST] Failed to load {symbol}: {e}")
                # Generate fallback data
                dfs.append(self._generate_symbol_data(
                    symbol, scenario.start, scenario.end
                ))
        
        if not dfs:
            raise ValueError("Failed to load any historical data")
        
        # Combine all symbols
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"[SST] Loaded {len(combined)} total bars across {len(symbols)} symbols")
        return combined
    
    def _generate_synthetic_data(self, scenario: Scenario) -> pd.DataFrame:
        """
        Generate synthetic baseline data for stress scenarios.
        
        Args:
            scenario: Synthetic scenario
            
        Returns:
            Multi-symbol OHLCV DataFrame with realistic properties
        """
        symbols = scenario.symbols or ["BTCUSDT", "ETHUSDT"]
        
        # Use scenario dates if provided, otherwise generate 30 days
        if scenario.start and scenario.end:
            start = scenario.start
            end = scenario.end
        else:
            end = datetime.utcnow()
            start = end - timedelta(days=30)
        
        dfs = []
        for symbol in symbols:
            df = self._generate_symbol_data(symbol, start, end, scenario.seed)
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"[SST] Generated {len(combined)} synthetic bars")
        return combined
    
    def _generate_symbol_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        seed: int | None = None
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic OHLCV data for a symbol.
        
        Uses geometric brownian motion with volatility clusters.
        
        Args:
            symbol: Trading pair
            start: Start datetime
            end: End datetime
            seed: Random seed
            
        Returns:
            OHLCV DataFrame
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate hourly timestamps
        timestamps = pd.date_range(start=start, end=end, freq="1H")
        n = len(timestamps)
        
        # Symbol-specific parameters
        if "BTC" in symbol:
            base_price = 40000.0
            volatility = 0.015  # 1.5% per hour
        elif "ETH" in symbol:
            base_price = 2500.0
            volatility = 0.02  # 2% per hour
        else:
            base_price = 100.0
            volatility = 0.025
        
        # Generate price using GBM
        returns = np.random.normal(0, volatility, n)
        
        # Add volatility clustering (GARCH-like)
        vol_cluster = np.zeros(n)
        vol_cluster[0] = volatility
        for i in range(1, n):
            # Volatility depends on previous volatility and shock
            vol_cluster[i] = (
                0.1 * volatility +
                0.85 * vol_cluster[i-1] +
                0.05 * abs(returns[i-1])
            )
        
        returns = returns * (vol_cluster / volatility)
        
        # Generate prices
        log_prices = np.log(base_price) + np.cumsum(returns)
        closes = np.exp(log_prices)
        
        # Generate OHLC from closes
        highs = closes * (1 + np.abs(np.random.normal(0, volatility/2, n)))
        lows = closes * (1 - np.abs(np.random.normal(0, volatility/2, n)))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        
        # Generate volumes (correlated with volatility)
        base_volume = 1000.0 if "BTC" in symbol else 5000.0
        volumes = base_volume * (1 + vol_cluster * 20) * np.random.lognormal(0, 0.5, n)
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "symbol": symbol,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate loaded data quality.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns
        required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")
        
        # Check for NaN values
        if df[["open", "high", "low", "close", "volume"]].isna().any().any():
            issues.append("Contains NaN values")
        
        # Check OHLC logic
        invalid_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        if invalid_ohlc.any():
            issues.append(f"Invalid OHLC logic in {invalid_ohlc.sum()} bars")
        
        # Check for negative prices/volumes
        if (df["close"] <= 0).any() or (df["volume"] < 0).any():
            issues.append("Contains negative or zero prices/volumes")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"[SST] Data validation failed: {issues}")
        else:
            logger.info("[SST] Data validation passed")
        
        return is_valid, issues
