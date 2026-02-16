#!/usr/bin/env python3
"""
Feature Publisher Service — PATH 2.3D Bridge

Purpose: Publishes market features to quantum:stream:features
         so ensemble_predictor can consume them.

Architecture:
    market.tick (existing)
        ↓ subscribe
    FeaturePublisher
        ↓ extract features (TA indicators)
        ↓ publish
    quantum:stream:features
        ↓ consumed by
    ensemble_predictor_service
        ↓ produces
    quantum:stream:signal.score

This service sits between market data and the ensemble predictor.
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import deque

import redis.asyncio as aioredis
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketFeatureExtractor:
    """
    Extracts technical features from market data.
    
    Now generates ALL 49 features required by LightGBM model:
    - Candlestick features (10): returns, body_size, wicks, patterns
    - Momentum & Oscillators (13): RSI, MACD, Stochastic, ROC, momentum
    - Moving Averages (12): EMAs, SMAs, distances
    - Bollinger Bands (5): upper, lower, middle, width, position
    - Volatility (4): ATR, volatility metrics
    - Trend (3): ADX, +DI, -DI
    - Volume (5): volume, ratios, OBV, VPT
    """
    
    def __init__(self, window_size: int = 250):  # Increased for EMA200
        self.window_size = window_size
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.high_history: Dict[str, deque] = {}  # For ATR, patterns
        self.low_history: Dict[str, deque] = {}   # For ATR, patterns
        self.obv_history: Dict[str, deque] = {}   # For OBV tracking
        
    def update_history(self, symbol: str, price: float, volume: float, high: float = None, low: float = None):
        """Update price, volume, high, low history for symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.window_size)
            self.volume_history[symbol] = deque(maxlen=self.window_size)
            self.high_history[symbol] = deque(maxlen=self.window_size)
            self.low_history[symbol] = deque(maxlen=self.window_size)
            self.obv_history[symbol] = deque(maxlen=self.window_size)
        
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        
        # If high/low not provided, use price as approximation
        self.high_history[symbol].append(high if high is not None else price * 1.001)
        self.low_history[symbol].append(low if low is not None else price * 0.999)
    
    def extract_features(self, symbol: str, price: float, volume: float, high: float = None, low: float = None) -> Dict[str, float]:
        """
        Extract ALL 49 features required by LightGBM model.
        
        Returns dict of feature_name -> value matching model expectations.
        """
        # Update history
        self.update_history(symbol, price, volume, high, low)
        
        prices = list(self.price_history.get(symbol, []))
        volumes = list(self.volume_history.get(symbol, []))
        highs = list(self.high_history.get(symbol, []))
        lows = list(self.low_history.get(symbol, []))
        
        if len(prices) < 5:
            # Not enough history, return neutral features
            return self._neutral_features()
        
        # Convert to numpy for efficiency
        prices_arr = np.array(prices)
        volumes_arr = np.array(volumes)
        highs_arr = np.array(highs)
        lows_arr = np.array(lows)
        
        features = {}
        
        # ===== 1. CANDLESTICK FEATURES (10) =====
        # Basic price metrics
        features['returns'] = float((prices_arr[-1] / prices_arr[-2] - 1) if len(prices_arr) >= 2 else 0.0)
        features['log_returns'] = float(np.log(prices_arr[-1] / prices_arr[-2]) if len(prices_arr) >= 2 and prices_arr[-2] > 0 else 0.0)
        features['price_range'] = float(highs_arr[-1] - lows_arr[-1])
        
        # Candlestick body and wicks
        open_price = prices_arr[-2] if len(prices_arr) >= 2 else price
        close_price = prices_arr[-1]
        high_price = highs_arr[-1]
        low_price = lows_arr[-1]
        
        features['body_size'] = float(abs(close_price - open_price))
        features['upper_wick'] = float(high_price - max(open_price, close_price))
        features['lower_wick'] = float(min(open_price, close_price) - low_price)
        
        # Candlestick patterns (boolean → 1.0/0.0)
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price if high_price > low_price else 0.001
        
        features['is_doji'] = 1.0 if (body_size / total_range < 0.1) else 0.0
        features['is_hammer'] = 1.0 if (features['lower_wick'] > 2 * body_size and features['upper_wick'] < body_size) else 0.0
        
        # Engulfing pattern (requires 2 candles)
        if len(prices_arr) >= 3:
            prev_open = prices_arr[-3]
            prev_close = prices_arr[-2]
            prev_body = abs(prev_close - prev_open)
            is_bullish_engulfing = close_price > open_price and prev_close < prev_open and body_size > prev_body
            is_bearish_engulfing = close_price < open_price and prev_close > prev_open and body_size > prev_body
            features['is_engulfing'] = 1.0 if (is_bullish_engulfing or is_bearish_engulfing) else 0.0
        else:
            features['is_engulfing'] = 0.0
        
        # Gap detection
        if len(prices_arr) >= 2 and len(highs_arr) >= 2 and len(lows_arr) >= 2:
            prev_high = highs_arr[-2]
            prev_low = lows_arr[-2]
            curr_low = lows_arr[-1]
            curr_high = highs_arr[-1]
            features['gap_up'] = 1.0 if (curr_low > prev_high) else 0.0
            features['gap_down'] = 1.0 if (curr_high < prev_low) else 0.0
        else:
            features['gap_up'] = 0.0
            features['gap_down'] = 0.0
        
        # ===== 2. RSI =====
        if len(prices_arr) >= 14:
            returns = np.diff(prices_arr[-14:])
            gains = returns[returns > 0].sum() if len(returns[returns > 0]) > 0 else 0.001
            losses = -returns[returns < 0].sum() if len(returns[returns < 0]) > 0 else 0.001
            rs = gains / losses
            features['rsi'] = float(100 - (100 / (1 + rs)))
        else:
            features['rsi'] = 50.0
        
        # ===== 3. MACD (12-26-9) =====
        if len(prices_arr) >= 26:
            ema_12 = self._ema(prices_arr, 12)
            ema_26 = self._ema(prices_arr, 26)
            macd_line = ema_12 - ema_26
            
            # MACD signal (9-day EMA of MACD)
            # Simplified: use recent MACD values if we tracked them
            # For now, approximate as 90% of MACD line
            macd_signal = macd_line * 0.9
            
            features['macd'] = float(macd_line)
            features['macd_signal'] = float(macd_signal)
            features['macd_hist'] = float(macd_line - macd_signal)
        else:
            features['macd'] = 0.0
            features['macd_signal'] = 0.0
            features['macd_hist'] = 0.0
        
        # ===== 4. STOCHASTIC OSCILLATOR =====
        if len(prices_arr) >= 14:
            low_14 = np.min(lows_arr[-14:])
            high_14 = np.max(highs_arr[-14:])
            if high_14 > low_14:
                stoch_k = 100 * (close_price - low_14) / (high_14 - low_14)
            else:
                stoch_k = 50.0
            
            # Stoch %D is 3-day SMA of %K (simplified: use current %K)
            stoch_d = stoch_k * 0.95  # Approximation
            
            features['stoch_k'] = float(stoch_k)
            features['stoch_d'] = float(stoch_d)
        else:
            features['stoch_k'] = 50.0
            features['stoch_d'] = 50.0
        
        # ===== 5. RATE OF CHANGE (ROC) =====
        if len(prices_arr) >= 12:
            roc = 100 * (prices_arr[-1] / prices_arr[-12] - 1)
            features['roc'] = float(roc)
        else:
            features['roc'] = 0.0
        
        # ===== 6. EMAs (9, 21, 50, 200) + Distances =====
        features['ema_9'] = float(self._ema(prices_arr, 9))
        features['ema_9_dist'] = float((price - features['ema_9']) / price if price > 0 else 0.0)
        
        features['ema_21'] = float(self._ema(prices_arr, 21))
        features['ema_21_dist'] = float((price - features['ema_21']) / price if price > 0 else 0.0)
        
        features['ema_50'] = float(self._ema(prices_arr, 50))
        features['ema_50_dist'] = float((price - features['ema_50']) / price if price > 0 else 0.0)
        
        features['ema_200'] = float(self._ema(prices_arr, 200))
        features['ema_200_dist'] = float((price - features['ema_200']) / price if price > 0 else 0.0)
        
        # ===== 7. SMAs (20, 50) =====
        features['sma_20'] = float(np.mean(prices_arr[-20:])) if len(prices_arr) >= 20 else price
        features['sma_50'] = float(np.mean(prices_arr[-50:])) if len(prices_arr) >= 50 else price
        
        # ===== 8. ADX (Average Directional Index) =====
        if len(prices_arr) >= 14:
            # Simplified ADX calculation
            high_diffs = np.diff(highs_arr[-15:])
            low_diffs = -np.diff(lows_arr[-15:])
            
            plus_dm = np.where((high_diffs > low_diffs) & (high_diffs > 0), high_diffs, 0)
            minus_dm = np.where((low_diffs > high_diffs) & (low_diffs > 0), low_diffs, 0)
            
            # True Range
            tr = np.maximum(highs_arr[-14:] - lows_arr[-14:], 
                           np.maximum(abs(highs_arr[-14:] - prices_arr[-15:-1]),
                                     abs(lows_arr[-14:] - prices_arr[-15:-1])))
            
            atr_14 = np.mean(tr) if len(tr) > 0 else 0.001
            
            plus_di = 100 * (np.mean(plus_dm) / atr_14) if atr_14 > 0 else 0.0
            minus_di = 100 * (np.mean(minus_dm) / atr_14) if atr_14 > 0 else 0.0
            
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0.0
            adx = dx * 100  # Simplified (should be EMA of DX)
            
            features['adx'] = float(adx)
            features['plus_di'] = float(plus_di)
            features['minus_di'] = float(minus_di)
        else:
            features['adx'] = 0.0
            features['plus_di'] = 0.0
            features['minus_di'] = 0.0
        
        # ===== 9. BOLLINGER BANDS =====
        if len(prices_arr) >= 20:
            sma_20 = np.mean(prices_arr[-20:])
            std_20 = np.std(prices_arr[-20:])
            
            features['bb_middle'] = float(sma_20)
            features['bb_upper'] = float(sma_20 + 2 * std_20)
            features['bb_lower'] = float(sma_20 - 2 * std_20)
            features['bb_width'] = float(4 * std_20)  # Upper - Lower
            features['bb_position'] = float((price - sma_20) / (2 * std_20) if std_20 > 0 else 0.0)
        else:
            features['bb_middle'] = price
            features['bb_upper'] = price * 1.02
            features['bb_lower'] = price * 0.98
            features['bb_width'] = price * 0.04
            features['bb_position'] = 0.0
        
        # ===== 10. ATR (Average True Range) =====
        if len(prices_arr) >= 14:
            # True Range already calculated for ADX
            tr = np.maximum(highs_arr[-14:] - lows_arr[-14:], 
                           np.maximum(abs(highs_arr[-14:] - prices_arr[-15:-1]),
                                     abs(lows_arr[-14:] - prices_arr[-15:-1])))
            atr = np.mean(tr)
            features['atr'] = float(atr)
            features['atr_pct'] = float(atr / price * 100 if price > 0 else 0.0)
        else:
            features['atr'] = 0.0
            features['atr_pct'] = 0.0
        
        # ===== 11. VOLATILITY =====
        if len(prices_arr) >= 20:
            volatility = np.std(np.diff(prices_arr[-20:]) / prices_arr[-21:-1])
            features['volatility'] = float(volatility)
        else:
            features['volatility'] = 0.0
        
        # ===== 12. VOLUME FEATURES =====
        features['volume_sma'] = float(np.mean(volumes_arr[-20:])) if len(volumes_arr) >= 20 else float(np.mean(volumes_arr)) if len(volumes_arr) > 0 else 0.0
        features['volume_ratio'] = float(volume / features['volume_sma'] if features['volume_sma'] > 0 else 1.0)
        
        # OBV (On-Balance Volume)
        if len(prices_arr) >= 2 and len(volumes_arr) >= 2:
            # Calculate OBV incrementally
            if symbol not in self.obv_history or len(self.obv_history[symbol]) == 0:
                # Initialize OBV
                obv = 0.0
                for i in range(1, len(prices_arr)):
                    if prices_arr[i] > prices_arr[i-1]:
                        obv += volumes_arr[i]
                    elif prices_arr[i] < prices_arr[i-1]:
                        obv -= volumes_arr[i]
            else:
                # Update OBV based on last value
                prev_obv = list(self.obv_history[symbol])[-1] if len(self.obv_history[symbol]) > 0 else 0.0
                if prices_arr[-1] > prices_arr[-2]:
                    obv = prev_obv + volumes_arr[-1]
                elif prices_arr[-1] < prices_arr[-2]:
                    obv = prev_obv - volumes_arr[-1]
                else:
                    obv = prev_obv
            
            # Store OBV
            if symbol in self.obv_history:
                self.obv_history[symbol].append(obv)
            
            features['obv'] = float(obv)
            
            # OBV EMA (10-period)
            obv_values = list(self.obv_history.get(symbol, [obv]))
            features['obv_ema'] = float(self._ema(np.array(obv_values), 10))
        else:
            features['obv'] = 0.0
            features['obv_ema'] = 0.0
        
        # VPT (Volume Price Trend)
        if len(prices_arr) >= 2 and len(volumes_arr) >= 2:
            price_change_pct = (prices_arr[-1] - prices_arr[-2]) / prices_arr[-2] if prices_arr[-2] > 0 else 0.0
            vpt = volumes_arr[-1] * price_change_pct
            features['vpt'] = float(vpt)
        else:
            features['vpt'] = 0.0
        
        # ===== 13. MOMENTUM FEATURES =====
        if len(prices_arr) >= 5:
            features['momentum_5'] = float(prices_arr[-1] / prices_arr[-5] - 1)
        else:
            features['momentum_5'] = 0.0
        
        if len(prices_arr) >= 10:
            features['momentum_10'] = float(prices_arr[-1] / prices_arr[-10] - 1)
        else:
            features['momentum_10'] = 0.0
        
        if len(prices_arr) >= 20:
            features['momentum_20'] = float(prices_arr[-1] / prices_arr[-20] - 1)
        else:
            features['momentum_20'] = 0.0
        
        # Acceleration (change in momentum)
        if len(prices_arr) >= 20:
            mom_10_now = features['momentum_10']
            mom_10_prev = (prices_arr[-11] / prices_arr[-20] - 1) if len(prices_arr) >= 20 else 0.0
            features['acceleration'] = float(mom_10_now - mom_10_prev)
        else:
            features['acceleration'] = 0.0
        
        # Relative Spread (high-low as % of price)
        features['relative_spread'] = float((high_price - low_price) / price if price > 0 else 0.0)
        
        # Ensure ALL values are native Python floats
        return {k: float(v) for k, v in features.items()}
    
    def _ema(self, values: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(values) < period:
            return float(np.mean(values))
        
        values_window = values[-period:]
        alpha = 2.0 / (period + 1)
        ema = values_window[0]
        for price in values_window[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return float(ema)
    
    def _neutral_features(self) -> Dict[str, float]:
        """Return neutral features when insufficient history - ALL 49 features."""
        return {
            # Candlestick (10)
            'returns': 0.0,
            'log_returns': 0.0,
            'price_range': 0.0,
            'body_size': 0.0,
            'upper_wick': 0.0,
            'lower_wick': 0.0,
            'is_doji': 0.0,
            'is_hammer': 0.0,
            'is_engulfing': 0.0,
            'gap_up': 0.0,
            'gap_down': 0.0,
            
            # Oscillators (7)
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_hist': 0.0,
            'stoch_k': 50.0,
            'stoch_d': 50.0,
            'roc': 0.0,
            
            # EMAs (8)
            'ema_9': 0.0,
            'ema_9_dist': 0.0,
            'ema_21': 0.0,
            'ema_21_dist': 0.0,
            'ema_50': 0.0,
            'ema_50_dist': 0.0,
            'ema_200': 0.0,
            'ema_200_dist': 0.0,
            
            # SMAs (2)
            'sma_20': 0.0,
            'sma_50': 0.0,
            
            # ADX (3)
            'adx': 0.0,
            'plus_di': 0.0,
            'minus_di': 0.0,
            
            # Bollinger Bands (5)
            'bb_middle': 0.0,
            'bb_upper': 0.0,
            'bb_lower': 0.0,
            'bb_width': 0.0,
            'bb_position': 0.0,
            
            # Volatility (3)
            'atr': 0.0,
            'atr_pct': 0.0,
            'volatility': 0.0,
            
            # Volume (5)
            'volume_sma': 0.0,
            'volume_ratio': 1.0,
            'obv': 0.0,
            'obv_ema': 0.0,
            'vpt': 0.0,
            
            # Momentum (5)
            'momentum_5': 0.0,
            'momentum_10': 0.0,
            'momentum_20': 0.0,
            'acceleration': 0.0,
            'relative_spread': 0.0
        }


class FeaturePublisherService:
    """
    Feature Publisher Service
    
    Subscribes to market.tick events, extracts features, publishes to quantum:stream:features.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        input_stream: str = "quantum:stream:exchange.normalized",
        output_stream: str = "quantum:stream:features"
    ):
        self.redis_url = redis_url
        self.input_stream = input_stream
        self.output_stream = output_stream
        
        self.redis = None
        self.feature_extractor = MarketFeatureExtractor(window_size=50)
        
        self.running = False
        self.features_published = 0
        
    async def start(self):
        """Start service."""
        logger.info("=" * 70)
        logger.info("🚀 FEATURE PUBLISHER SERVICE STARTING")
        logger.info("=" * 70)
        logger.info(f"Input: {self.input_stream}")
        logger.info(f"Output: {self.output_stream}")
        logger.info("")
        
        # Connect to Redis
        self.redis = await aioredis.from_url(
            self.redis_url,
            decode_responses=True
        )
        logger.info("✅ Connected to Redis")
        
        # Start subscription
        self.running = True
        await self.subscribe_market_tick()
    
    async def stop(self):
        """Stop service."""
        logger.info("🛑 Stopping Feature Publisher Service...")
        self.running = False
        
        if self.redis:
            await self.redis.close()
        
        logger.info(f"📊 Total features published: {self.features_published}")
        logger.info("✅ Stopped")
    
    async def subscribe_market_tick(self):
        """Subscribe to market.tick stream and process ticks."""
        # Create consumer group (idempotent)
        try:
            await self.redis.xgroup_create(
                self.input_stream,
                "feature_publisher",
                id='0',
                mkstream=True
            )
            logger.info("[FEATURE-PUBLISHER] Created consumer group: feature_publisher")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.warning(f"[FEATURE-PUBLISHER] Consumer group error: {e}")
        
        logger.info(f"[FEATURE-PUBLISHER] 🎧 Subscribed to {self.input_stream}")
        logger.info("[FEATURE-PUBLISHER] Waiting for market ticks...")
        
        last_log_time = datetime.now()
        
        while self.running:
            try:
                # Read from consumer group (blocking with timeout)
                events = await self.redis.xreadgroup(
                    "feature_publisher",
                    "publisher_v1",
                    {self.input_stream: '>'},
                    count=10,
                    block=5000  # 5s timeout
                )
                
                if not events:
                    # Log every 30s to show service is alive
                    if (datetime.now() - last_log_time) > timedelta(seconds=30):
                        logger.info(f"[FEATURE-PUBLISHER] Alive (published {self.features_published} features)")
                        last_log_time = datetime.now()
                    continue
                
                # Process each event
                for stream_name, messages in events:
                    for message_id, fields in messages:
                        await self._process_tick(message_id, fields)
                
            except asyncio.CancelledError:
                logger.info("[FEATURE-PUBLISHER] Subscription cancelled")
                break
            except Exception as e:
                logger.error(f"[FEATURE-PUBLISHER] Stream read error: {e}")
                await asyncio.sleep(5)
    
    async def _process_tick(self, message_id: str, fields: Dict[str, str]):
        """Process single market tick and publish features."""
        try:
            # Extract tick data (from exchange.normalized format)
            symbol = fields.get("symbol", "UNKNOWN")
            
            # Try avg_price first (aggregated), fallback to binance_price
            price_str = fields.get("avg_price") or fields.get("binance_price") or fields.get("close") or "0"
            price = float(price_str)
            
            # Get high/low if available (for candlestick patterns)
            high = float(fields.get("high", price))
            low = float(fields.get("low", price))
            
            # Get volume - check multiple fields
            volume_str = fields.get("volume") or fields.get("binance_volume") or "0"
            volume = float(volume_str)
            
            # If no volume, use price divergence as proxy
            if volume == 0:
                price_div = float(fields.get("price_divergence", "0"))
                volume = abs(price_div) * 1000  # Synthetic volume
            
            if price <= 0:
                logger.warning(f"[FEATURE-PUBLISHER] Invalid price for {symbol}: {price}")
                # ACK anyway to avoid blocking
                await self.redis.xack(self.input_stream, "feature_publisher", message_id)
                return
            
            # Extract ALL 49 features
            features = self.feature_extractor.extract_features(symbol, price, volume, high, low)
            
            # Add metadata
            features['symbol'] = symbol
            features['timestamp'] = datetime.utcnow().isoformat() + "Z"
            
            # Publish to output stream
            await self.redis.xadd(
                self.output_stream,
                features,
                maxlen=10000  # Keep last 10k features
            )
            
            self.features_published += 1
            
            # Log periodically (every 100 features)
            if self.features_published % 100 == 0:
                logger.info(
                    f"[FEATURE-PUBLISHER] ✅ {symbol} | "
                    f"price={price:.6f} rsi={features.get('rsi', 50):.1f} "
                    f"adx={features.get('adx', 0):.1f} | "
                    f"total={self.features_published}"
                )
            
            # ACK message
            await self.redis.xack(self.input_stream, "feature_publisher", message_id)
            
        except Exception as e:
            logger.error(f"[FEATURE-PUBLISHER] ❌ Error processing tick: {e}", exc_info=True)


async def main():
    """Main entry point."""
    service = FeaturePublisherService()
    
    # Setup signal handlers
    def signal_handler(sig):
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(service.stop())
    
    loop = asyncio.get_running_loop()
    for sig in [signal.SIGTERM, signal.SIGINT]:
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
