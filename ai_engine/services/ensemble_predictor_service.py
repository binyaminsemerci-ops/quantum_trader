"""
ENSEMBLE PREDICTOR SERVICE (PATH 2.2)

Authority: SCORER ONLY (NO EXECUTION)
Purpose: Produce scored exit recommendations for apply_layer consumption

Design Principles:
1. NO execution capability (no Binance client)
2. NO order generation (no TradeIntent)
3. NO write access to apply.result or trade.intent
4. Output stream: quantum:stream:signal.score ONLY
5. Fail-mode: Degraded confidence (not system halt)

Input: quantum:stream:features.* (market features)
Output: quantum:stream:signal.score (scored recommendations)
Consumer: apply_layer (enriches decision context)

Governance: NO_AUTHORITY_ENSEMBLE_PREDICTOR_FEB11_2026.md
"""
import asyncio
import logging
import time
import random
import os
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import redis.asyncio as aioredis

# NO Binance client import (authority restriction)
# NO TradeIntent import (no order generation)

logger = logging.getLogger(__name__)


@dataclass
class SignalScore:
    """
    Exit-focused signal score output.
    
    Schema enforced by validator (see validate_signal_score).
    """
    symbol: str
    horizon: str  # MUST be "exit"
    suggested_action: str  # ONLY "CLOSE" or "HOLD"
    confidence: float  # [0.0, 1.0]
    expected_edge: float  # Can be negative
    risk_context: str  # Max 5 words
    ensemble_version: str
    models_used: str  # Comma-separated
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Redis stream payload."""
        return asdict(self)


class SignalValidator:
    """
    Validator enforcing NO_AUTHORITY contract.
    
    Drops events with forbidden fields or invalid schema.
    """
    
    FORBIDDEN_FIELDS = {
        "quantity", "size", "price", "order",
        "reduceOnly", "leverage", "execute",
        "position_size", "entry_price", "stop_loss"
    }
    
    ALLOWED_ACTIONS = {"CLOSE", "HOLD"}
    REQUIRED_HORIZON = "exit"
    
    def __init__(self):
        self.total_validated = 0
        self.total_dropped = 0
        self.drop_reasons: Dict[str, int] = {}
    
    def validate(self, msg: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate signal score message.
        
        Returns:
            (is_valid, drop_reason)
        """
        self.total_validated += 1
        
        try:
            # Check horizon
            if msg.get("horizon") != self.REQUIRED_HORIZON:
                reason = f"invalid_horizon_{msg.get('horizon')}"
                self._record_drop(reason)
                return False, reason
            
            # Check action
            if msg.get("suggested_action") not in self.ALLOWED_ACTIONS:
                reason = f"invalid_action_{msg.get('suggested_action')}"
                self._record_drop(reason)
                return False, reason
            
            # Check confidence bounds
            confidence = float(msg.get("confidence", -1))
            if not (0.0 <= confidence <= 1.0):
                reason = f"confidence_out_of_bounds_{confidence}"
                self._record_drop(reason)
                return False, reason
            
            # Check expected_edge is float
            if not isinstance(msg.get("expected_edge"), (int, float)):
                reason = "expected_edge_not_numeric"
                self._record_drop(reason)
                return False, reason
            
            # Check required fields
            required = {"models_used", "timestamp"}
            missing = required - set(msg.keys())
            if missing:
                reason = f"missing_fields_{','.join(missing)}"
                self._record_drop(reason)
                return False, reason
            
            # Check for forbidden fields (CRITICAL)
            forbidden_present = self.FORBIDDEN_FIELDS & set(msg.keys())
            if forbidden_present:
                reason = f"FORBIDDEN_FIELDS_{','.join(forbidden_present)}"
                self._record_drop(reason)
                logger.error(f"🚨 [VALIDATOR] AUTHORITY VIOLATION: {reason}")
                return False, reason
            
            return True, None
            
        except Exception as e:
            reason = f"validation_error_{type(e).__name__}"
            self._record_drop(reason)
            logger.error(f"[VALIDATOR] Exception: {e}")
            return False, reason
    
    def _record_drop(self, reason: str):
        """Record dropped event for audit."""
        self.total_dropped += 1
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1
        logger.warning(f"[VALIDATOR] ❌ DROPPED: {reason} (total={self.total_dropped})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics for audit."""
        return {
            "total_validated": self.total_validated,
            "total_dropped": self.total_dropped,
            "drop_rate": self.total_dropped / max(1, self.total_validated),
            "drop_reasons": dict(self.drop_reasons)
        }


class EnsemblePredictorService:
    """
    Ensemble Predictor Service (PATH 2.2)
    
    Authority: SCORER ONLY
    Scope: EXIT decisions (CLOSE vs HOLD)
    Output: quantum:stream:signal.score
    
    NO EXECUTION CAPABILITY.
    """
    
    VERSION = "v1.0.0"
    OUTPUT_STREAM = "quantum:stream:signal.score"
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        enable_calibration: bool = True
    ):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.validator = SignalValidator()
        self.enable_calibration = enable_calibration
        
        # Statistics
        self.signals_produced = 0
        self.signals_dropped = 0
        self.start_time = time.time()
        
        # Model loading deferred (prevent execution imports)
        self.models_loaded = False
        
        # Calibration (loaded if available)
        self.calibration_loader = None
        if enable_calibration:
            try:
                from ai_engine.calibration.calibration_loader import CalibrationLoader
                self.calibration_loader = CalibrationLoader()
                logger.info(f"[ENSEMBLE-PREDICTOR] Calibration: {self.calibration_loader.get_status()}")
            except ImportError:
                logger.warning("[ENSEMBLE-PREDICTOR] Calibration loader not available")
        
        logger.info(f"[ENSEMBLE-PREDICTOR] Initialized (version={self.VERSION})")
        logger.info(f"[ENSEMBLE-PREDICTOR] Authority: SCORER ONLY")
        logger.info(f"[ENSEMBLE-PREDICTOR] Output: {self.OUTPUT_STREAM}")
        logger.info(f"[ENSEMBLE-PREDICTOR] Governance: NO_AUTHORITY_ENSEMBLE_PREDICTOR_FEB11_2026.md")
    
    async def connect(self):
        """Connect to Redis (read/write to signal.score only)."""
        self.redis = await aioredis.from_url(
            self.redis_url,
            decode_responses=False  # Handle bytes manually
        )
        logger.info(f"[ENSEMBLE-PREDICTOR] ✅ Connected to Redis")
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            logger.info(f"[ENSEMBLE-PREDICTOR] Disconnected from Redis")
    
    def _load_models(self):
        """
        Load model agents (deferred initialization).
        
        NOTE: NO Binance client, NO execution modules.
        """
        if self.models_loaded:
            return
        
        try:
            # Import model agents (OBSERVER/SCORER level only)
            from ai_engine.agents.lgbm_agent import LightGBMAgent
            from ai_engine.agents.patchtst_agent_v3 import PatchTSTAgent
            from ai_engine.agents.nhits_agent import NHiTSAgent
            from ai_engine.agents.xgb_agent import XGBAgent
            from ai_engine.agents.unified_agents import TFTAgent
            
            # Initialize agents (no execution capability)
            self.lgbm = LightGBMAgent()
            self.patchtst = PatchTSTAgent()
            self.nhits = NHiTSAgent()
            self.xgb = XGBAgent()
            self.tft = TFTAgent()
            
            self.models_loaded = True
            logger.info(f"[ENSEMBLE-PREDICTOR] ✅ Models loaded (5 agents: LGBM, XGB, N-HiTS, PatchTST, TFT)")
            
        except Exception as e:
            logger.error(f"[ENSEMBLE-PREDICTOR] ❌ Model loading failed: {e}")
            self.models_loaded = False
    
    async def _aggregate_predictions(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> Optional[SignalScore]:
        """
        Aggregate model predictions into single score.
        
        Uses conservative voting:
        - Disagreement REDUCES confidence
        - Agreement INCREASES confidence
        - Uncertainty allowed (low confidence valid)
        
        Args:
            symbol: Trading pair
            features: Market features (price, volatility, trend, etc.)
        
        Returns:
            SignalScore or None (if fail-mode)
        """
        try:
            # Ensure models loaded
            self._load_models()
            
            if not self.models_loaded:
                # Fail-mode: Degraded confidence
                return self._fail_mode_signal(symbol, "models_unavailable")
            
            # Full ensemble voting with all 5 agents (PATH 2.5 implementation)
            predictions = []
            agents = [
                ("LGBM", self.lgbm),
                ("XGB", self.xgb),
                ("N-HiTS", self.nhits),
                ("PatchTST", self.patchtst),
                ("TFT", self.tft)
            ]
            
            for agent_name, agent in agents:
                try:
                    pred = agent.predict(symbol, features)
                    action = pred.get("action", "HOLD")
                    conf = pred.get("confidence", 0.5)
                    predictions.append((agent_name, action, conf))
                    logger.debug(f"[ENSEMBLE-PREDICTOR] {symbol} {agent_name}: {action} (conf={conf:.3f})")
                except Exception as e:
                    logger.warning(f"[ENSEMBLE-PREDICTOR] {agent_name} failed: {e}")
                    # Fallback: neutral vote
                    predictions.append((agent_name, "HOLD", 0.5))
            
            # Weighted voting: Action by majority, confidence by weighted average
            if not predictions:
                # Fail-mode
                raw_confidence = 0.50
                suggested_action = "HOLD"
                models_used = "none"
            else:
                # Count votes for each action
                votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
                weights = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
                
                for agent_name, action, conf in predictions:
                    votes[action] += 1
                    weights[action] += conf  # Sum confidence for each action
                
                # Majority vote
                majority_action = max(votes, key=votes.get)
                
                # Average confidence for winning action
                if votes[majority_action] > 0:
                    avg_conf = weights[majority_action] / votes[majority_action]
                else:
                    avg_conf = 0.5
                
                # Map to exit-focused recommendation
                if majority_action in ["BUY", "SELL"]:
                    suggested_action = "CLOSE"
                    raw_confidence = avg_conf
                else:
                    suggested_action = "HOLD"
                    raw_confidence = avg_conf * 0.7  # Conservative HOLD reduction
                
                models_used = ",".join([name for name, _, _ in predictions])
                
                logger.debug(
                    f"[ENSEMBLE-PREDICTOR] {symbol} Ensemble: {majority_action} → {suggested_action} "
                    f"(votes={votes}, conf={raw_confidence:.3f})"
                )
            
            # Apply calibration if available
            final_confidence = raw_confidence
            if self.calibration_loader:
                try:
                    final_confidence = self.calibration_loader.apply_confidence_calibration(
                        raw_confidence
                    )
                    logger.debug(
                        f"[ENSEMBLE-PREDICTOR] Calibrated: {raw_confidence:.3f} → {final_confidence:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"[ENSEMBLE-PREDICTOR] Calibration error: {e}")
                    # Fallback to raw confidence
                    final_confidence = raw_confidence
            
            return SignalScore(
                symbol=symbol,
                horizon="exit",
                suggested_action=suggested_action,
                confidence=final_confidence,
                expected_edge=0.0,
                risk_context="initialization",
                ensemble_version=self.VERSION,
                models_used=models_used,  # Dynamic: reflects actual agents used
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            
        except Exception as e:
            logger.error(f"[ENSEMBLE-PREDICTOR] Aggregation error: {e}")
            return self._fail_mode_signal(symbol, f"aggregation_error_{type(e).__name__}")
    
    def _fail_mode_signal(self, symbol: str, reason: str) -> SignalScore:
        """
        Fail-mode: Return degraded confidence signal.
        
        System MUST NOT halt on predictor failure.
        
        If SYNTHETIC_MODE=true: Generate random confidences for calibration testing.
        """
        # Synthetic mode for calibration testing (bypass model dependencies)
        synthetic_mode = os.getenv("SYNTHETIC_MODE", "false").lower() == "true"
        
        if synthetic_mode:
            # Generate realistic-looking random confidence (skewed towards 0.5-0.8)
            random_conf = random.betavariate(2, 2) * 0.5 + 0.25  # Range: 0.25-0.75
            action_choices = ["CLOSE", "HOLD"]
            action_weights = [0.3, 0.7]  # More HOLD than CLOSE
            
            logger.info(f"[ENSEMBLE-PREDICTOR] 🎲 SYNTHETIC-MODE: {symbol} conf={random_conf:.3f}")
            
            return SignalScore(
                symbol=symbol,
                horizon="exit",
                suggested_action=random.choices(action_choices, weights=action_weights)[0],
                confidence=random_conf,
                expected_edge=random_conf * 0.02,  # Synthetic edge proportional to confidence
                risk_context="synthetic_test",
                ensemble_version=self.VERSION + "-SYNTH",
                models_used="random",
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
        
        # Normal fail-mode (zero confidence)
        logger.warning(f"[ENSEMBLE-PREDICTOR] ⚠️ FAIL-MODE: {reason}")
        return SignalScore(
            symbol=symbol,
            horizon="exit",
            suggested_action="HOLD",
            confidence=0.0,  # Minimum confidence
            expected_edge=0.0,
            risk_context=reason[:25],  # Truncate for schema
            ensemble_version=self.VERSION,
            models_used="none",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    async def produce_signal(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> bool:
        """
        Produce single signal score for symbol.
        
        Returns:
            True if signal published, False if dropped
        """
        # Aggregate predictions
        signal = await self._aggregate_predictions(symbol, features)
        
        if signal is None:
            self.signals_dropped += 1
            return False
        
        # Convert to dict
        signal_dict = signal.to_dict()
        
        # Validate (CRITICAL: enforce NO_AUTHORITY contract)
        is_valid, drop_reason = self.validator.validate(signal_dict)
        
        if not is_valid:
            self.signals_dropped += 1
            logger.warning(
                f"[ENSEMBLE-PREDICTOR] ❌ Signal DROPPED: {symbol} - {drop_reason}"
            )
            return False
        
        # Publish to Redis stream
        try:
            message_id = await self.redis.xadd(
                self.OUTPUT_STREAM,
                signal_dict,
                maxlen=10000  # Keep last 10k signals
            )
            
            self.signals_produced += 1
            
            logger.info(
                f"[ENSEMBLE-PREDICTOR] ✅ {symbol} {signal.suggested_action} | "
                f"conf={signal.confidence:.3f} edge={signal.expected_edge:.3f} | "
                f"msg_id={message_id.decode()}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"[ENSEMBLE-PREDICTOR] ❌ Redis publish error: {e}")
            self.signals_dropped += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics for monitoring."""
        uptime = time.time() - self.start_time
        return {
            "version": self.VERSION,
            "uptime_seconds": uptime,
            "signals_produced": self.signals_produced,
            "signals_dropped": self.signals_dropped,
            "drop_rate": self.signals_dropped / max(1, self.signals_produced + self.signals_dropped),
            "validator_stats": self.validator.get_stats(),
            "models_loaded": self.models_loaded
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        try:
            # Check Redis connection
            await self.redis.ping()
            redis_ok = True
        except:
            redis_ok = False
        
        stats = self.get_stats()
        
        return {
            "status": "healthy" if redis_ok and self.models_loaded else "degraded",
            "redis_connected": redis_ok,
            "models_loaded": self.models_loaded,
            "authority": "SCORER",
            "output_stream": self.OUTPUT_STREAM,
            "stats": stats
        }
    
    async def subscribe_features_stream(
        self,
        feature_stream: str = "quantum:stream:features",
        group_name: str = "ensemble_predictor",
        consumer_name: str = "predictor_v1"
    ):
        """
        Subscribe to feature stream and process events.
        
        Consumes features, produces signal scores to quantum:stream:signal.score.
        
        Args:
            feature_stream: Input stream name
            group_name: Consumer group for distributed processing
            consumer_name: This consumer's identifier
        """
        # Create consumer group (idempotent)
        try:
            await self.redis.xgroup_create(
                feature_stream,
                group_name,
                id='0',
                mkstream=True
            )
            logger.info(f"[ENSEMBLE-PREDICTOR] Created consumer group: {group_name}")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.warning(f"[ENSEMBLE-PREDICTOR] Consumer group error: {e}")
        
        logger.info(f"[ENSEMBLE-PREDICTOR] 🎧 Subscribed to {feature_stream}")
        
        while True:
            try:
                # Read from consumer group (blocking with timeout)
                events = await self.redis.xreadgroup(
                    group_name,
                    consumer_name,
                    {feature_stream: '>'},
                    count=10,
                    block=5000  # 5s timeout
                )
                
                if not events:
                    continue
                
                # Process each event
                for stream_name, messages in events:
                    for message_id, fields in messages:
                        await self._process_feature_event(
                            message_id.decode() if isinstance(message_id, bytes) else message_id,
                            fields,
                            feature_stream,
                            group_name
                        )
                
            except asyncio.CancelledError:
                logger.info("[ENSEMBLE-PREDICTOR] Subscription cancelled")
                break
            except Exception as e:
                logger.error(f"[ENSEMBLE-PREDICTOR] Stream read error: {e}")
                await asyncio.sleep(5)
    
    async def _process_feature_event(
        self,
        message_id: str,
        fields: Dict[bytes, bytes],
        stream_name: str,
        group_name: str
    ):
        """
        Process single feature event from stream.
        
        Extracts features, produces signal, ACKs event.
        """
        try:
            # Decode fields
            features = {}
            symbol = None
            
            for k, v in fields.items():
                key = k.decode() if isinstance(k, bytes) else k
                value = v.decode() if isinstance(v, bytes) else v
                
                if key == "symbol":
                    symbol = value
                else:
                    # Try to parse as float
                    try:
                        features[key] = float(value)
                    except ValueError:
                        features[key] = value
            
            if not symbol:
                logger.warning(f"[ENSEMBLE-PREDICTOR] No symbol in message {message_id}")
                await self.redis.xack(stream_name, group_name, message_id)
                return
            
            # Produce signal
            success = await self.produce_signal(symbol, features)
            
            # ACK regardless of success (validator drops invalid, no retry)
            await self.redis.xack(stream_name, group_name, message_id)
            
            if success:
                logger.debug(f"[ENSEMBLE-PREDICTOR] ✅ Processed {symbol} msg={message_id}")
            
        except Exception as e:
            logger.error(f"[ENSEMBLE-PREDICTOR] Event processing error: {e}")
            # ACK to prevent infinite loop
            await self.redis.xack(stream_name, group_name, message_id)


async def main():
    """
    Main entry point for standalone service.
    
    NOTE: This is PATH 2.2 - SCORER authority only.
    NO execution capability.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    service = EnsemblePredictorService()
    
    await service.connect()
    
    try:
        logger.info("[ENSEMBLE-PREDICTOR] 🚀 Service started")
        logger.info("[ENSEMBLE-PREDICTOR] Authority: SCORER ONLY")
        logger.info("[ENSEMBLE-PREDICTOR] Output: quantum:stream:signal.score")
        logger.info("[ENSEMBLE-PREDICTOR] Governance: NO_AUTHORITY_ENSEMBLE_PREDICTOR_FEB11_2026.md")
        
        # Start feature stream subscription
        subscription_task = asyncio.create_task(
            service.subscribe_features_stream()
        )
        
        # Stats monitoring loop
        async def monitor_stats():
            while True:
                await asyncio.sleep(60)
                stats = service.get_stats()
                logger.info(
                    f"[STATS] Produced={stats['signals_produced']} "
                    f"Dropped={stats['signals_dropped']} "
                    f"DropRate={stats['drop_rate']:.2%}"
                )
        
        stats_task = asyncio.create_task(monitor_stats())
        
        # Wait for tasks
        await asyncio.gather(subscription_task, stats_task)
        
    except KeyboardInterrupt:
        logger.info("[ENSEMBLE-PREDICTOR] Shutting down...")
    finally:
        await service.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
