"""
RL v3 Live Feature Adapter
Builds observations from live market data for RLv3Manager.predict()
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np

try:
    from backend.core.event_bus import EventBusV2
    from backend.core.policy_store import PolicyStore
    from backend.services.execution.execution import BinanceFuturesExecutionAdapter
except ImportError:
    EventBusV2 = Any
    PolicyStore = Any
    BinanceFuturesExecutionAdapter = Any


class RLv3LiveFeatureAdapter:
    """Adapter for building live observations for RL v3 agent."""
    
    def __init__(
        self,
        event_bus: EventBusV2,
        policy_store: PolicyStore,
        execution_adapter: BinanceFuturesExecutionAdapter,
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.event_bus = event_bus
        self.policy_store = policy_store
        self.execution_adapter = execution_adapter
        self.logger = logger_instance or logging.getLogger(__name__)
        
        self._price_cache: Dict[str, list] = {}
        self._last_cache_update: Dict[str, datetime] = {}
        self._cache_ttl_seconds = 60
        
    async def build_observation(self, symbol: str, trace_id: str) -> Dict[str, Any]:
        """Build observation dict for RL v3 from live data."""
        try:
            obs_dict = {}
            
            # Get market data
            prices = await self._get_recent_prices(symbol, trace_id)
            if not prices or len(prices) < 15:
                self.logger.warning(
                    "[live_adapter] Insufficient price data",
                    symbol=symbol,
                    price_count=len(prices) if prices else 0,
                    trace_id=trace_id,
                )
                return self._get_fallback_observation(symbol)
            
            # Price changes
            obs_dict["price_change_1m"] = self._calculate_return(prices, minutes=1)
            obs_dict["price_change_5m"] = self._calculate_return(prices, minutes=5)
            obs_dict["price_change_15m"] = self._calculate_return(prices, minutes=15)
            
            # Volatility
            obs_dict["volatility"] = self._calculate_volatility(prices)
            
            # RSI
            obs_dict["rsi"] = self._calculate_rsi(prices)
            
            # MACD
            macd_value, macd_signal = self._calculate_macd(prices)
            obs_dict["macd"] = macd_value
            obs_dict["macd_signal"] = macd_signal
            obs_dict["macd_histogram"] = macd_value - macd_signal
            
            # Trend strength
            obs_dict["trend_strength"] = self._calculate_trend_strength(prices)
            
            # Volume ratio (placeholder - requires volume data)
            obs_dict["volume_ratio"] = 1.0
            
            # Bid-ask spread (placeholder - requires orderbook)
            obs_dict["bid_ask_spread"] = 0.001
            
            # Position information
            positions = await self._get_positions(symbol, trace_id)
            if positions:
                position = positions[0]
                obs_dict["position_size"] = float(position.get("positionAmt", 0))
                obs_dict["position_side"] = self._encode_position_side(position.get("positionAmt", 0))
                obs_dict["unrealized_pnl"] = float(position.get("unRealizedProfit", 0))
                obs_dict["entry_price"] = float(position.get("entryPrice", 0))
                obs_dict["mark_price"] = float(position.get("markPrice", prices[-1]))
                obs_dict["leverage"] = float(position.get("leverage", 1))
            else:
                obs_dict["position_size"] = 0.0
                obs_dict["position_side"] = 0.0
                obs_dict["unrealized_pnl"] = 0.0
                obs_dict["entry_price"] = 0.0
                obs_dict["mark_price"] = prices[-1] if prices else 0.0
                obs_dict["leverage"] = 1.0
            
            # Account information
            balance_info = await self._get_account_balance(trace_id)
            obs_dict["balance"] = balance_info.get("balance", 10000.0)
            obs_dict["equity"] = balance_info.get("equity", 10000.0)
            obs_dict["margin_ratio"] = balance_info.get("margin_ratio", 0.0)
            
            # Regime detection (placeholder - requires regime detector)
            obs_dict["regime"] = self._detect_regime(prices)
            obs_dict["regime_confidence"] = 0.5
            
            # Time features
            now = datetime.utcnow()
            obs_dict["hour_of_day"] = now.hour / 24.0
            obs_dict["day_of_week"] = now.weekday() / 7.0
            
            # Additional features to reach 64 dimensions
            obs_dict["price_std_1m"] = self._calculate_std(prices, minutes=1)
            obs_dict["price_std_5m"] = self._calculate_std(prices, minutes=5)
            obs_dict["price_std_15m"] = self._calculate_std(prices, minutes=15)
            
            obs_dict["price_max_1m"] = self._calculate_max_deviation(prices, minutes=1)
            obs_dict["price_min_1m"] = self._calculate_min_deviation(prices, minutes=1)
            
            obs_dict["momentum_1m"] = self._calculate_momentum(prices, minutes=1)
            obs_dict["momentum_5m"] = self._calculate_momentum(prices, minutes=5)
            obs_dict["momentum_15m"] = self._calculate_momentum(prices, minutes=15)
            
            # Rate of change features
            obs_dict["roc_1m"] = self._calculate_roc(prices, minutes=1)
            obs_dict["roc_5m"] = self._calculate_roc(prices, minutes=5)
            obs_dict["roc_15m"] = self._calculate_roc(prices, minutes=15)
            
            # Moving average features
            obs_dict["sma_5"] = self._calculate_sma(prices, periods=5)
            obs_dict["sma_10"] = self._calculate_sma(prices, periods=10)
            obs_dict["sma_15"] = self._calculate_sma(prices, periods=15)
            obs_dict["ema_5"] = self._calculate_ema(prices, periods=5)
            obs_dict["ema_10"] = self._calculate_ema(prices, periods=10)
            obs_dict["ema_15"] = self._calculate_ema(prices, periods=15)
            
            # Price distance from moving averages
            current_price = prices[-1]
            obs_dict["distance_sma_5"] = (current_price - obs_dict["sma_5"]) / current_price if obs_dict["sma_5"] > 0 else 0.0
            obs_dict["distance_sma_10"] = (current_price - obs_dict["sma_10"]) / current_price if obs_dict["sma_10"] > 0 else 0.0
            obs_dict["distance_ema_5"] = (current_price - obs_dict["ema_5"]) / current_price if obs_dict["ema_5"] > 0 else 0.0
            obs_dict["distance_ema_10"] = (current_price - obs_dict["ema_10"]) / current_price if obs_dict["ema_10"] > 0 else 0.0
            
            # Bollinger bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)
            obs_dict["bb_upper"] = bb_upper
            obs_dict["bb_middle"] = bb_middle
            obs_dict["bb_lower"] = bb_lower
            obs_dict["bb_position"] = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
            
            # Additional momentum indicators
            obs_dict["rsi_change"] = self._calculate_rsi_change(prices)
            obs_dict["macd_change"] = self._calculate_macd_change(prices)
            
            # Risk metrics
            obs_dict["sharpe_estimate"] = self._estimate_sharpe(prices)
            obs_dict["max_drawdown_estimate"] = self._estimate_max_drawdown(prices)
            
            # Market microstructure (placeholders)
            obs_dict["order_flow_imbalance"] = 0.0
            obs_dict["liquidity_score"] = 1.0
            
            # Pad remaining features to reach exactly 64
            current_feature_count = len(obs_dict)
            for i in range(current_feature_count, 64):
                obs_dict[f"feature_{i}"] = 0.0
            
            self.logger.info(
                "[live_adapter] Observation built",
                symbol=symbol,
                feature_count=len(obs_dict),
                price=current_price,
                position_side=obs_dict["position_side"],
                trace_id=trace_id,
            )
            
            return obs_dict
            
        except Exception as e:
            self.logger.error(
                "[live_adapter] Failed to build observation",
                symbol=symbol,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return self._get_fallback_observation(symbol)
    
    async def _get_recent_prices(self, symbol: str, trace_id: str) -> list:
        """Get recent price data with caching."""
        now = datetime.utcnow()
        
        # Check cache
        if symbol in self._price_cache and symbol in self._last_cache_update:
            if (now - self._last_cache_update[symbol]).total_seconds() < self._cache_ttl_seconds:
                return self._price_cache[symbol]
        
        try:
            # Fetch klines from Binance (1m interval, last 30 minutes)
            klines = await self.execution_adapter.get_klines(
                symbol=symbol,
                interval="1m",
                limit=30
            )
            
            if klines:
                prices = [float(k[4]) for k in klines]  # Close prices
                self._price_cache[symbol] = prices
                self._last_cache_update[symbol] = now
                return prices
            else:
                self.logger.warning(
                    "[live_adapter] No klines returned",
                    symbol=symbol,
                    trace_id=trace_id,
                )
                return []
                
        except Exception as e:
            self.logger.error(
                "[live_adapter] Failed to fetch klines",
                symbol=symbol,
                error=str(e),
                trace_id=trace_id,
            )
            return self._price_cache.get(symbol, [])
    
    async def _get_positions(self, symbol: str, trace_id: str) -> list:
        """Get current positions."""
        try:
            positions = await self.execution_adapter.get_positions()
            return [p for p in positions if p.get("symbol") == symbol]
        except Exception as e:
            self.logger.error(
                "[live_adapter] Failed to fetch positions",
                symbol=symbol,
                error=str(e),
                trace_id=trace_id,
            )
            return []
    
    async def _get_account_balance(self, trace_id: str) -> Dict[str, float]:
        """Get account balance information."""
        try:
            cash_balance = await self.execution_adapter.get_cash_balance()
            
            # Get equity (balance + unrealized PnL)
            positions = await self.execution_adapter.get_positions()
            unrealized_pnl = sum(float(p.get("unRealizedProfit", 0)) for p in positions)
            equity = cash_balance + unrealized_pnl
            
            # Calculate margin ratio
            margin_ratio = 0.0
            if equity > 0:
                used_margin = sum(
                    abs(float(p.get("positionAmt", 0)) * float(p.get("entryPrice", 0)))
                    for p in positions
                )
                margin_ratio = used_margin / equity if equity > 0 else 0.0
            
            return {
                "balance": cash_balance,
                "equity": equity,
                "margin_ratio": margin_ratio,
            }
        except Exception as e:
            self.logger.error(
                "[live_adapter] Failed to fetch balance",
                error=str(e),
                trace_id=trace_id,
            )
            return {"balance": 10000.0, "equity": 10000.0, "margin_ratio": 0.0}
    
    def _calculate_return(self, prices: list, minutes: int) -> float:
        """Calculate return over specified minutes."""
        if len(prices) < minutes + 1:
            return 0.0
        return (prices[-1] - prices[-minutes - 1]) / prices[-minutes - 1] if prices[-minutes - 1] != 0 else 0.0
    
    def _calculate_volatility(self, prices: list, window: int = 15) -> float:
        """Calculate rolling volatility."""
        if len(prices) < window:
            return 0.0
        returns = [prices[i] / prices[i - 1] - 1 for i in range(len(prices) - window + 1, len(prices))]
        return float(np.std(returns)) if returns else 0.0
    
    def _calculate_rsi(self, prices: list, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i - 1] for i in range(len(prices) - period, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: list) -> tuple:
        """Calculate MACD and signal line."""
        if len(prices) < 26:
            return 0.0, 0.0
        
        # Simplified MACD calculation
        ema_12 = self._calculate_ema(prices, periods=12)
        ema_26 = self._calculate_ema(prices, periods=26)
        macd = ema_12 - ema_26
        
        # Signal line (9-period EMA of MACD) - simplified
        macd_signal = macd * 0.9
        
        return macd, macd_signal
    
    def _calculate_trend_strength(self, prices: list) -> float:
        """Calculate trend strength using linear regression slope."""
        if len(prices) < 10:
            return 0.0
        
        recent_prices = prices[-10:]
        x = list(range(len(recent_prices)))
        y = recent_prices
        
        # Linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Normalize by price level
        avg_price = sum_y / n
        return slope / avg_price if avg_price != 0 else 0.0
    
    def _encode_position_side(self, position_amt: float) -> float:
        """Encode position side as -1 (short), 0 (flat), 1 (long)."""
        if position_amt > 0:
            return 1.0
        elif position_amt < 0:
            return -1.0
        else:
            return 0.0
    
    def _detect_regime(self, prices: list) -> int:
        """Detect market regime (0-3)."""
        if len(prices) < 15:
            return 0
        
        volatility = self._calculate_volatility(prices)
        trend = self._calculate_trend_strength(prices)
        
        # Simple regime classification
        if abs(trend) > 0.001 and volatility < 0.02:
            return 0  # Trending low volatility
        elif abs(trend) > 0.001 and volatility >= 0.02:
            return 1  # Trending high volatility
        elif abs(trend) <= 0.001 and volatility < 0.02:
            return 2  # Range-bound low volatility
        else:
            return 3  # Range-bound high volatility
    
    def _calculate_std(self, prices: list, minutes: int) -> float:
        """Calculate standard deviation over specified minutes."""
        if len(prices) < minutes:
            return 0.0
        recent = prices[-minutes:]
        return float(np.std(recent))
    
    def _calculate_max_deviation(self, prices: list, minutes: int) -> float:
        """Calculate max upward deviation over specified minutes."""
        if len(prices) < minutes:
            return 0.0
        recent = prices[-minutes:]
        base = recent[0]
        return max((p - base) / base for p in recent) if base != 0 else 0.0
    
    def _calculate_min_deviation(self, prices: list, minutes: int) -> float:
        """Calculate max downward deviation over specified minutes."""
        if len(prices) < minutes:
            return 0.0
        recent = prices[-minutes:]
        base = recent[0]
        return min((p - base) / base for p in recent) if base != 0 else 0.0
    
    def _calculate_momentum(self, prices: list, minutes: int) -> float:
        """Calculate momentum over specified minutes."""
        if len(prices) < minutes + 1:
            return 0.0
        return prices[-1] - prices[-minutes - 1]
    
    def _calculate_roc(self, prices: list, minutes: int) -> float:
        """Calculate rate of change over specified minutes."""
        if len(prices) < minutes + 1:
            return 0.0
        return (prices[-1] - prices[-minutes - 1]) / prices[-minutes - 1] if prices[-minutes - 1] != 0 else 0.0
    
    def _calculate_sma(self, prices: list, periods: int) -> float:
        """Calculate simple moving average."""
        if len(prices) < periods:
            return prices[-1] if prices else 0.0
        return sum(prices[-periods:]) / periods
    
    def _calculate_ema(self, prices: list, periods: int) -> float:
        """Calculate exponential moving average."""
        if len(prices) < periods:
            return prices[-1] if prices else 0.0
        
        multiplier = 2 / (periods + 1)
        ema = prices[-periods]
        
        for price in prices[-periods + 1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_bollinger_bands(self, prices: list, period: int = 20, std_dev: float = 2.0) -> tuple:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            price = prices[-1] if prices else 0.0
            return price, price, price
        
        sma = self._calculate_sma(prices, period)
        std = self._calculate_std(prices, period)
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper, sma, lower
    
    def _calculate_rsi_change(self, prices: list) -> float:
        """Calculate change in RSI."""
        if len(prices) < 30:
            return 0.0
        
        rsi_current = self._calculate_rsi(prices[-15:])
        rsi_previous = self._calculate_rsi(prices[-30:-15])
        
        return rsi_current - rsi_previous
    
    def _calculate_macd_change(self, prices: list) -> float:
        """Calculate change in MACD."""
        if len(prices) < 30:
            return 0.0
        
        macd_current, _ = self._calculate_macd(prices[-15:])
        macd_previous, _ = self._calculate_macd(prices[-30:-15])
        
        return macd_current - macd_previous
    
    def _estimate_sharpe(self, prices: list) -> float:
        """Estimate Sharpe ratio from recent returns."""
        if len(prices) < 15:
            return 0.0
        
        returns = [prices[i] / prices[i - 1] - 1 for i in range(len(prices) - 14, len(prices))]
        
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        std_return = float(np.std(returns))
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def _estimate_max_drawdown(self, prices: list) -> float:
        """Estimate max drawdown from recent prices."""
        if len(prices) < 2:
            return 0.0
        
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak if peak != 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _get_fallback_observation(self, symbol: str) -> Dict[str, Any]:
        """Get fallback observation with all zeros."""
        obs_dict = {f"feature_{i}": 0.0 for i in range(64)}
        obs_dict["symbol"] = symbol
        self.logger.warning("[live_adapter] Using fallback observation", symbol=symbol)
        return obs_dict
