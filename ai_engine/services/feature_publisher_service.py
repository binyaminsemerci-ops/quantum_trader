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
    
    Features include:
    - Price-based: returns, volatility, momentum
    - Volume-based: volume ratio, volume momentum
    - Technical: RSI, MA crossovers, Bollinger bands
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        
    def update_history(self, symbol: str, price: float, volume: float):
        """Update price and volume history for symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.window_size)
            self.volume_history[symbol] = deque(maxlen=self.window_size)
        
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
    
    def extract_features(self, symbol: str, price: float, volume: float) -> Dict[str, float]:
        """
        Extract features for current tick.
        
        Returns dict of feature_name -> value.
        """
        # Update history
        self.update_history(symbol, price, volume)
        
        prices = list(self.price_history.get(symbol, []))
        volumes = list(self.volume_history.get(symbol, []))
        
        if len(prices) < 5:
            # Not enough history, return neutral features
            return self._neutral_features()
        
        # Convert to numpy for efficiency
        prices_arr = np.array(prices)
        volumes_arr = np.array(volumes)
        
        features = {}
        
        # Price-based features
        features['price'] = float(price)
        features['price_return_1'] = float((prices_arr[-1] / prices_arr[-2] - 1) if len(prices_arr) >= 2 else 0.0)
        features['price_return_5'] = float((prices_arr[-1] / prices_arr[-5] - 1) if len(prices_arr) >= 5 else 0.0)
        features['price_volatility_10'] = float(np.std(prices_arr[-10:])) if len(prices_arr) >= 10 else 0.0
        
        # price_change (LightGBM requirement - same as 1-tick return)
        features['price_change'] = features['price_return_1']
        
        # Moving averages
        features['ma_10'] = float(np.mean(prices_arr[-10:])) if len(prices_arr) >= 10 else price
        features['ma_20'] = float(np.mean(prices_arr[-20:])) if len(prices_arr) >= 20 else price
        features['ma_50'] = float(np.mean(prices_arr[-50:])) if len(prices_arr) >= 50 else price
        
        # MA crossover signals
        if len(prices_arr) >= 20:
            features['ma_cross_10_20'] = 1.0 if features['ma_10'] > features['ma_20'] else -1.0
        else:
            features['ma_cross_10_20'] = 0.0
        
        # RSI approximation (simplified)
        if len(prices_arr) >= 14:
            returns = np.diff(prices_arr[-14:])
            gains = returns[returns > 0].sum() if len(returns[returns > 0]) > 0 else 0.001
            losses = -returns[returns < 0].sum() if len(returns[returns < 0]) > 0 else 0.001
            rs = gains / losses
            features['rsi_14'] = float(100 - (100 / (1 + rs)))
        else:
            features['rsi_14'] = 50.0  # Neutral
        
        # MACD (12-26-9) - simplified version
        if len(prices_arr) >= 26:
            ema_12 = self._ema(prices_arr, 12)
            ema_26 = self._ema(prices_arr, 26)
            features['macd'] = float(ema_12 - ema_26)
        else:
            features['macd'] = 0.0
        
        # Volume features
        features['volume'] = float(volume)
        features['volume_ratio'] = float(volume / np.mean(volumes_arr) if len(volumes_arr) > 0 and np.mean(volumes_arr) > 0 else 1.0)
        
        # Bollinger bands
        if len(prices_arr) >= 20:
            ma_20 = np.mean(prices_arr[-20:])
            std_20 = np.std(prices_arr[-20:])
            features['bb_upper'] = ma_20 + 2 * std_20
            features['bb_lower'] = ma_20 - 2 * std_20
            features['bb_position'] = (price - ma_20) / (2 * std_20) if std_20 > 0 else 0.0
        else:
            features['bb_upper'] = price * 1.02
            features['bb_lower'] = price * 0.98
            features['bb_position'] = 0.0
        
        # Momentum
        if len(prices_arr) >= 10:
            features['momentum_10'] = float(prices_arr[-1] / prices_arr[-10] - 1)
        else:
            features['momentum_10'] = 0.0
        
        # Ensure ALL values are native Python floats (not numpy types)
        # This prevents Redis serialization issues like 'np.float64(...)'
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
        """Return neutral features when insufficient history."""
        return {
            'price': 0.0,
            'price_change': 0.0,  # LightGBM requirement
            'price_return_1': 0.0,
            'price_return_5': 0.0,
            'price_volatility_10': 0.0,
            'ma_10': 0.0,
            'ma_20': 0.0,
            'ma_50': 0.0,
            'ma_cross_10_20': 0.0,
            'rsi_14': 50.0,
            'macd': 0.0,  # LightGBM requirement
            'volume': 0.0,
            'volume_ratio': 1.0,
            'bb_upper': 0.0,
            'bb_lower': 0.0,
            'bb_position': 0.0,
            'momentum_10': 0.0
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
            price_str = fields.get("avg_price") or fields.get("binance_price") or "0"
            price = float(price_str)
            
            # Volume not in exchange.normalized, use price change as proxy
            price_div = float(fields.get("price_divergence", "0"))
            volume = abs(price_div) * 1000  # Synthetic volume from price divergence
            
            if price <= 0:
                logger.warning(f"[FEATURE-PUBLISHER] Invalid price for {symbol}: {price}")
                # ACK anyway to avoid blocking
                await self.redis.xack(self.input_stream, "feature_publisher", message_id)
                return
            
            # Extract features
            features = self.feature_extractor.extract_features(symbol, price, volume)
            
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
            
            # Log periodically
            if self.features_published % 100 == 0:
                logger.info(
                    f"[FEATURE-PUBLISHER] ✅ {symbol} | "
                    f"price={price:.2f} rsi={features['rsi_14']:.1f} | "
                    f"total={self.features_published}"
                )
            
            # ACK message
            await self.redis.xack(self.input_stream, "feature_publisher", message_id)
            
        except Exception as e:
            logger.error(f"[FEATURE-PUBLISHER] ❌ Error processing tick: {e}")


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
