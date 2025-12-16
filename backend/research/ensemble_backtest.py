"""
Ensemble-integrated backtester.

Uses real Quantum Trader ensemble predictions for strategy evaluation.
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

from .backtest import StrategyBacktester
from .models import StrategyConfig
from .repositories import MarketDataClient

logger = logging.getLogger(__name__)


class EnsembleBacktester(StrategyBacktester):
    """
    Backtester that uses real ensemble predictions.
    
    Integrates with Quantum Trader's 4-model ensemble for realistic
    entry signal generation during backtesting.
    """
    
    def __init__(
        self,
        market_data_client: MarketDataClient,
        ensemble_manager,
        commission_rate: float = 0.0004
    ):
        """
        Initialize ensemble backtester.
        
        Args:
            market_data_client: Market data source
            ensemble_manager: EnsembleManager instance from ai_engine
            commission_rate: Trading commission (default: 0.04%)
        """
        super().__init__(market_data_client, commission_rate)
        self.ensemble = ensemble_manager
        self._prediction_cache = {}
    
    def _check_entry(
        self,
        config: StrategyConfig,
        df: pd.DataFrame,
        idx: int
    ) -> Optional[bool]:
        """
        Check entry conditions using ensemble predictions.
        
        Args:
            config: Strategy configuration
            df: Market data (OHLCV)
            idx: Current bar index
        
        Returns:
            True for long entry, False for short entry, None for no signal
        """
        try:
            # Get historical data up to current bar (avoid look-ahead bias)
            historical_data = df.iloc[:idx+1].copy()
            
            if len(historical_data) < 50:
                # Need minimum data for ensemble
                return None
            
            # Generate cache key
            symbol = config.symbols[0] if config.symbols else "UNKNOWN"
            timestamp = historical_data['timestamp'].iloc[-1]
            cache_key = f"{symbol}:{timestamp}"
            
            # Check cache
            if cache_key not in self._prediction_cache:
                # Get ensemble prediction
                try:
                    # Call ensemble with historical data
                    # Note: This is a simplified interface - adjust based on actual ensemble API
                    prediction = self.ensemble.predict(
                        data=historical_data,
                        symbol=symbol
                    )
                    
                    self._prediction_cache[cache_key] = prediction
                    
                except Exception as e:
                    logger.warning(f"Ensemble prediction failed: {e}")
                    return None
            
            prediction = self._prediction_cache[cache_key]
            
            if not prediction:
                return None
            
            # Extract signal and confidence
            # Adjust these keys based on actual ensemble return format
            signal = prediction.get('signal', 'HOLD')  # 'BUY'/'SELL'/'HOLD'
            confidence = prediction.get('confidence', 0.0)
            
            # Check confidence threshold
            if confidence < config.min_confidence:
                return None
            
            # Entry logic by type
            if config.entry_type == "ENSEMBLE_CONSENSUS":
                # Pure ensemble signal
                if signal == 'BUY':
                    return True
                elif signal == 'SELL':
                    return False
                else:
                    return None
            
            elif config.entry_type == "MOMENTUM":
                # Ensemble + momentum confirmation
                if len(df) < idx + 20:
                    return None
                
                # Calculate 20-bar momentum
                returns_20 = df['close'].pct_change(20).iloc[idx]
                
                if signal == 'BUY' and returns_20 > 0:
                    return True
                elif signal == 'SELL' and returns_20 < 0:
                    return False
                else:
                    return None
            
            elif config.entry_type == "MEAN_REVERSION":
                # Ensemble + mean reversion confirmation
                if len(df) < idx + 20:
                    return None
                
                # Calculate deviation from 20-bar MA
                ma_20 = df['close'].rolling(20).mean().iloc[idx]
                current_price = df['close'].iloc[idx]
                deviation = (current_price - ma_20) / ma_20
                
                # Mean reversion logic: buy when below MA, sell when above
                if signal == 'BUY' and deviation < -0.02:  # 2% below MA
                    return True
                elif signal == 'SELL' and deviation > 0.02:  # 2% above MA
                    return False
                else:
                    return None
            
            else:
                logger.warning(f"Unknown entry type: {config.entry_type}")
                return None
        
        except Exception as e:
            logger.error(f"Error checking entry: {e}")
            return None
    
    def clear_prediction_cache(self):
        """Clear cached ensemble predictions"""
        self._prediction_cache.clear()
        logger.info("Ensemble prediction cache cleared")


class SimplifiedEnsembleBacktester(StrategyBacktester):
    """
    Simplified backtester for when ensemble is not available.
    
    Uses technical indicators as proxies for ensemble signals.
    Good for testing and development.
    """
    
    def _check_entry(
        self,
        config: StrategyConfig,
        df: pd.DataFrame,
        idx: int
    ) -> Optional[bool]:
        """
        Check entry using simplified technical indicators.
        
        This is a fallback when real ensemble is not available.
        """
        try:
            if len(df) < idx + 50:
                return None
            
            # Calculate simple indicators
            close = df['close'].iloc[:idx+1]
            
            # Simple moving averages
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            current_price = close.iloc[-1]
            
            # RSI-like momentum
            returns = close.pct_change(14)
            avg_gain = returns[returns > 0].mean()
            avg_loss = abs(returns[returns < 0].mean())
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Generate pseudo-confidence based on indicator strength
            confidence = 0.5
            
            # Price above both MAs = bullish
            if current_price > sma_20 and sma_20 > sma_50:
                confidence += 0.2
            
            # RSI conditions
            if 30 < rsi < 70:
                confidence += 0.1
            
            # Check threshold
            if confidence < config.min_confidence:
                return None
            
            # Entry logic
            if config.entry_type == "ENSEMBLE_CONSENSUS":
                # Use MA crossover
                if current_price > sma_20 and sma_20 > sma_50:
                    return True
                elif current_price < sma_20 and sma_20 < sma_50:
                    return False
            
            elif config.entry_type == "MOMENTUM":
                # Momentum-based
                momentum_20 = close.pct_change(20).iloc[-1]
                if momentum_20 > 0.02:  # 2% gain
                    return True
                elif momentum_20 < -0.02:  # 2% loss
                    return False
            
            elif config.entry_type == "MEAN_REVERSION":
                # Mean reversion
                deviation = (current_price - sma_20) / sma_20
                if deviation < -0.03:  # 3% below MA
                    return True
                elif deviation > 0.03:  # 3% above MA
                    return False
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking entry: {e}")
            return None
