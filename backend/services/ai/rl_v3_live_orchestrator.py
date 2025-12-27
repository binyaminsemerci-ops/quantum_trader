"""
RL v3 Live Orchestrator
Combines RL v3 decisions with signals and manages trading modes.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore
from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.live_adapter_v3 import RLv3LiveFeatureAdapter
from backend.domains.learning.rl_v3.metrics_v3 import RLv3MetricsStore
from backend.services.risk.risk_guard import RiskGuardService
from backend.services.risk.safety_governor import SafetyGovernor


class RLv3LiveOrchestrator:
    """Orchestrator for RL v3 live trading decisions."""
    
    MODES = ["OFF", "SHADOW", "PRIMARY", "HYBRID"]
    ACTION_MAP = {
        0: "FLAT",
        1: "LONG_SMALL",
        2: "LONG_LARGE",
        3: "SHORT_SMALL",
        4: "SHORT_LARGE",
        5: "HOLD",
    }
    
    def __init__(
        self,
        event_bus: EventBus,
        rl_manager: RLv3Manager,
        feature_adapter: RLv3LiveFeatureAdapter,
        policy_store: PolicyStore,
        risk_guard: RiskGuardService,
        safety_governor: Optional[SafetyGovernor] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.event_bus = event_bus
        self.rl_manager = rl_manager
        self.feature_adapter = feature_adapter
        self.policy_store = policy_store
        self.risk_guard = risk_guard
        self.safety_governor = safety_governor
        self.logger = logger_instance or logging.getLogger(__name__)
        
        self.metrics_store = RLv3MetricsStore.instance()
        self._running = False
        self._config_cache = {}
        self._last_config_fetch = None
        self._config_ttl_seconds = 10
        
        self._trade_count_hour = 0
        self._last_trade_hour = None
        
        # [PRODUCTION] Track open intents per symbol to prevent double-intent
        self._open_intents = set()  # Set of symbols with active intents
        self._promotion_acked = False  # Promotion safety lock
    
    async def start(self):
        """Start the orchestrator and subscribe to events."""
        if self._running:
            self.logger.warning("[rl_v3_orchestrator] Already running")
            return
        
        self._running = True
        
        # Get initial config
        config = await self._get_config()
        mode = config.get("mode", "SHADOW")
        
        self.logger.info(
            "[rl_v3_orchestrator] Starting orchestrator",
            mode=mode,
            enabled=config.get("enabled", True),
        )
        
        # Subscribe to events if not in OFF mode
        if mode != "OFF":
            self.event_bus.subscribe("signal.generated", self._handle_signal_generated)
            self.logger.info("[rl_v3_orchestrator] Subscribed to signal.generated")
    
    async def stop(self):
        """Stop the orchestrator."""
        self._running = False
        self.logger.info("[rl_v3_orchestrator] Stopping orchestrator")
    
    async def _handle_signal_generated(self, payload: Dict[str, Any]):
        """Handle signal.generated event."""
        try:
            trace_id = payload.get("trace_id", str(uuid.uuid4()))
            symbol = payload.get("symbol", "BTCUSDT")
            signal_confidence = payload.get("confidence", 0.5)
            signal_action = payload.get("action", "HOLD")
            
            # Get config
            config = await self._get_config()
            mode = config.get("mode", "SHADOW")
            enabled = config.get("enabled", True)
            min_confidence = config.get("min_confidence", 0.6)
            max_trades_per_hour = config.get("max_trades_per_hour", 10)
            
            if not enabled or mode == "OFF":
                self.logger.debug(
                    "[rl_v3_orchestrator] Orchestrator disabled or OFF",
                    mode=mode,
                    enabled=enabled,
                    trace_id=trace_id,
                )
                return
            
            # Build observation
            obs_dict = await self.feature_adapter.build_observation(symbol, trace_id)
            
            # Get RL v3 prediction
            prediction = await asyncio.to_thread(self.rl_manager.predict, obs_dict)
            
            rl_action_idx = prediction.get("action", 5)
            rl_confidence = prediction.get("confidence", 0.0)
            rl_value = prediction.get("value", 0.0)
            
            rl_action = self.ACTION_MAP.get(rl_action_idx, "HOLD")
            
            self.logger.info(
                "[rl_v3_orchestrator] RL v3 prediction",
                symbol=symbol,
                mode=mode,
                rl_action=rl_action,
                rl_confidence=rl_confidence,
                rl_value=rl_value,
                signal_action=signal_action,
                signal_confidence=signal_confidence,
                trace_id=trace_id,
            )
            
            # Record decision in metrics
            self.metrics_store.record_live_decision({
                "symbol": symbol,
                "mode": mode,
                "action": rl_action,
                "action_idx": rl_action_idx,
                "confidence": rl_confidence,
                "value": rl_value,
                "signal_action": signal_action,
                "signal_confidence": signal_confidence,
                "published_trade_intent": mode not in ["SHADOW", "OFF"],
                "timestamp": datetime.utcnow().isoformat(),
                "trace_id": trace_id,
            })
            
            # [PRODUCTION GUARD] SHADOW mode - NEVER publish intents
            if mode == "SHADOW":
                self.logger.info(
                    "[rl_v3_orchestrator] 🔒 SHADOW mode - recording metrics only, NO trade intent published",
                    rl_action=rl_action,
                    rl_confidence=rl_confidence,
                    trace_id=trace_id,
                )
                # Record shadow decision for analysis
                self.metrics_store.record_trade_intent({
                    "symbol": symbol,
                    "side": rl_action,
                    "source": "RL_V3_SHADOW",
                    "confidence": rl_confidence,
                    "size_pct": 0.0,  # Shadow = no real size
                    "executed": False,
                    "shadow_only": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "trace_id": trace_id,
                })
                return  # CRITICAL: Exit here - no further processing
            
            # Check confidence threshold
            if rl_confidence < min_confidence:
                self.logger.info(
                    "[rl_v3_orchestrator] Confidence below threshold",
                    rl_confidence=rl_confidence,
                    min_confidence=min_confidence,
                    trace_id=trace_id,
                )
                return
            
            # [PRODUCTION GUARD A] Check open position for symbol (prevent double intent)
            if config.get("check_open_positions", True) and symbol in self._open_intents:
                self.logger.warning(
                    f"[rl_v3_orchestrator] ⚠️ GUARD A: Open intent already exists for {symbol}, trace_id: {trace_id}"
                )
                return
            
            # [PRODUCTION GUARD B] Rate limiting
            if not self._check_rate_limit(max_trades_per_hour):
                self.logger.warning(
                    f"[rl_v3_orchestrator] ⚠️ GUARD C: Rate limit exceeded: {max_trades_per_hour}/hour, current: {self._trade_count_hour}, trace_id: {trace_id}"
                )
                return
            
            # [PRODUCTION SAFETY] Promotion lock check
            if mode in ["PRIMARY", "HYBRID"] and config.get("promotion_requires_ack", True):
                if not self._promotion_acked:
                    self.logger.error(
                        f"[rl_v3_orchestrator] 🚨 PROMOTION BLOCKED: mode={mode} requires ACK from RiskGuard+ExitBrain, trace_id: {trace_id}"
                    )
                    return
            
            # Build trade intent based on mode
            if mode == "PRIMARY":
                trade_intent = await self._build_trade_intent_primary(
                    symbol, rl_action, rl_action_idx, rl_confidence, trace_id
                )
            elif mode == "HYBRID":
                trade_intent = await self._build_trade_intent_hybrid(
                    symbol, rl_action, rl_action_idx, rl_confidence,
                    signal_action, signal_confidence, trace_id
                )
            else:
                self.logger.warning(
                    "[rl_v3_orchestrator] Unknown mode",
                    mode=mode,
                    trace_id=trace_id,
                )
                return
            
            if not trade_intent:
                self.logger.info(
                    "[rl_v3_orchestrator] No trade intent generated",
                    mode=mode,
                    trace_id=trace_id,
                )
                return
            
            # Risk check
            can_execute, denial_reason = await self._check_risk_guard(trade_intent, trace_id)
            
            if not can_execute:
                self.logger.warning(
                    f"[rl_v3_orchestrator] RiskGuard denied trade: {denial_reason}, trace_id: {trace_id}"
                )
                return
            
            # [PRODUCTION] Mark symbol as having open intent
            symbol = trade_intent["symbol"]
            self._open_intents.add(symbol)
            
            # Publish trade intent
            await self.event_bus.publish(
                "trade.intent",
                trade_intent,
                trace_id=trace_id,
            )
            
            # Record trade intent
            self.metrics_store.record_trade_intent({
                "symbol": symbol,
                "side": trade_intent["side"],
                "source": trade_intent["source"],
                "confidence": trade_intent["confidence"],
                "size_pct": trade_intent["size_pct"],
                "executed": False,
                "timestamp": datetime.utcnow().isoformat(),
                "trace_id": trace_id,
            })
            
            self._increment_trade_count()
            
            self.logger.info(
                f"[rl_v3_orchestrator] ✅ Trade intent published: {symbol} {trade_intent['side']} {trade_intent['size_pct']:.2%}",
                mode=mode,
                trade_intent=trade_intent,
                trace_id=trace_id,
            )
            
            # [PRODUCTION] Schedule cleanup of open intent after 60s (timeout)
            asyncio.create_task(self._cleanup_open_intent(symbol, 60))
            
        except Exception as e:
            self.logger.error(
                "[rl_v3_orchestrator] Error handling signal",
                error=str(e),
                trace_id=payload.get("trace_id"),
                exc_info=True,
            )
    
    async def _build_trade_intent_primary(
        self,
        symbol: str,
        rl_action: str,
        rl_action_idx: int,
        rl_confidence: float,
        trace_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Build trade intent from RL v3 action only (PRIMARY mode)."""
        side, size_pct = self._decode_rl_action(rl_action, rl_action_idx)
        
        if side == "HOLD":
            return None
        
        return {
            "symbol": symbol,
            "side": side,
            "size_pct": size_pct,
            "source": "RL_V3_PRIMARY",
            "confidence": rl_confidence,
            "leverage": await self._get_leverage_from_policy(),
            "trace_id": trace_id,
        }
    
    async def _build_trade_intent_hybrid(
        self,
        symbol: str,
        rl_action: str,
        rl_action_idx: int,
        rl_confidence: float,
        signal_action: str,
        signal_confidence: float,
        trace_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Build trade intent combining RL v3 and signal (HYBRID mode)."""
        rl_side, rl_size_pct = self._decode_rl_action(rl_action, rl_action_idx)
        signal_side = self._normalize_signal_action(signal_action)
        
        # Strategy: Use highest confidence source
        if rl_confidence > signal_confidence:
            side = rl_side
            size_pct = rl_size_pct
            confidence = rl_confidence
            source = "RL_V3_HYBRID_RL_PRIMARY"
        else:
            side = signal_side
            size_pct = 0.15  # Default size for signal
            confidence = signal_confidence
            source = "RL_V3_HYBRID_SIGNAL_PRIMARY"
        
        # If both agree on direction, increase confidence
        if rl_side == signal_side and rl_side != "HOLD":
            confidence = min(1.0, (rl_confidence + signal_confidence) / 2 + 0.1)
            size_pct = min(0.3, size_pct * 1.2)
            source = "RL_V3_HYBRID_CONSENSUS"
        
        if side == "HOLD":
            return None
        
        return {
            "symbol": symbol,
            "side": side,
            "size_pct": size_pct,
            "source": source,
            "confidence": confidence,
            "leverage": await self._get_leverage_from_policy(),
            "rl_action": rl_action,
            "rl_confidence": rl_confidence,
            "signal_action": signal_action,
            "signal_confidence": signal_confidence,
            "trace_id": trace_id,
        }
    
    def _decode_rl_action(self, rl_action: str, rl_action_idx: int) -> tuple:
        """Decode RL action to side and size_pct."""
        if rl_action == "FLAT":
            return "FLAT", 0.0
        elif rl_action == "LONG_SMALL":
            return "LONG", 0.1
        elif rl_action == "LONG_LARGE":
            return "LONG", 0.2
        elif rl_action == "SHORT_SMALL":
            return "SHORT", 0.1
        elif rl_action == "SHORT_LARGE":
            return "SHORT", 0.2
        else:  # HOLD
            return "HOLD", 0.0
    
    def _normalize_signal_action(self, signal_action: str) -> str:
        """Normalize signal action to LONG/SHORT/HOLD."""
        signal_upper = signal_action.upper()
        if "LONG" in signal_upper or "BUY" in signal_upper:
            return "LONG"
        elif "SHORT" in signal_upper or "SELL" in signal_upper:
            return "SHORT"
        elif "FLAT" in signal_upper or "CLOSE" in signal_upper:
            return "FLAT"
        else:
            return "HOLD"
    
    async def _check_risk_guard(self, trade_intent: Dict[str, Any], trace_id: str) -> tuple:
        """[PRODUCTION] Delegate all risk checking to RiskGuard.
        
        CRITICAL: RL v3 does NO futures math. RiskGuard owns all risk calculations.
        """
        try:
            # [PRODUCTION GUARD B] Size limit check (policy-based)
            config = await self._get_config()
            max_size_pct = config.get("max_size_pct", 0.15)
            size_pct = trade_intent.get("size_pct", 0.0)
            
            if size_pct > max_size_pct:
                return (False, f"⚠️ GUARD B: size_pct {size_pct:.2%} exceeds policy limit {max_size_pct:.2%}")
            
            # Delegate to RiskGuard.evaluate_trade_intent
            can_execute, denial_reason = await self.risk_guard.evaluate_trade_intent(
                trade_intent,
                trace_id=trace_id,
            )
            
            if not can_execute:
                self.logger.warning(
                    f"[rl_v3_orchestrator] ❌ RiskGuard denied: {denial_reason}",
                    extra={"trace_id": trace_id, "trade_intent": trade_intent}
                )
            
            return can_execute, denial_reason
            
        except Exception as e:
            self.logger.error(
                "[rl_v3_orchestrator] Error checking RiskGuard",
                error=str(e),
                trace_id=trace_id,
            )
            return False, f"RiskGuard error: {str(e)}"
    
    async def _get_leverage_from_policy(self) -> int:
        """Get leverage from PolicyStore."""
        try:
            risk_profile = await self.policy_store.get_active_risk_profile()
            return risk_profile.max_leverage
        except Exception as e:
            self.logger.error(
                "[rl_v3_orchestrator] Failed to get leverage from policy",
                error=str(e),
            )
            return 10
    
    async def _get_config(self) -> Dict[str, Any]:
        """Get orchestrator config from PolicyStore with caching."""
        now = datetime.utcnow()
        
        # Check cache
        if self._last_config_fetch and self._config_cache:
            elapsed = (now - self._last_config_fetch).total_seconds()
            if elapsed < self._config_ttl_seconds:
                return self._config_cache
        
        try:
            policy = await self.policy_store.get_policy()
            
            # [PRODUCTION] Default config (futures-safe)
            config = {
                "enabled": True,
                "mode": "SHADOW",
                "min_confidence": 0.6,
                "max_trades_per_hour": 10,
                "max_size_pct": 0.15,
                "max_margin_alloc_pct": 0.25,
                "liq_buffer_pct": 0.20,
                "max_loss_per_trade_pct": 0.02,
                "check_open_positions": True,
                "promotion_requires_ack": True,
            }
            
            config_source = "default"
            
            # PRIORITY 1: Check environment variable (easiest deployment method)
            import os
            env_mode = os.getenv("RL_V3_MODE")
            if env_mode:
                config["mode"] = env_mode
                config_source = "env"
                self.logger.info(f"[rl_v3_orchestrator] Using mode from environment: {env_mode}")
            
            # PRIORITY 2: Try to get from policy JSON snapshot (if env not set)
            if config_source == "default":
                try:
                    import json
                    from pathlib import Path
                    
                    policy_path = Path("/app/data/policy_snapshot.json") if Path("/app/data").exists() else Path("data/policy_snapshot.json")
                    
                    if policy_path.exists():
                        with open(policy_path, "r") as f:
                            policy_json = json.load(f)
                        
                        if "rl_v3_live" in policy_json:
                            rl_v3_config = policy_json["rl_v3_live"]
                            config.update(rl_v3_config)  # Override with policy values
                            config_source = "policy"
                except Exception as e:
                    self.logger.warning(f"[rl_v3_orchestrator] Could not load from policy JSON: {e}")
            
            self.logger.info(
                f"[rl_v3_orchestrator] 📋 Config loaded from {config_source}: mode={config['mode']}, enabled={config['enabled']}, max_size_pct={config['max_size_pct']:.2%}"
            )
            
            # Cache config
            self._config_cache = config
            self._last_config_fetch = now
            
            return config
            
        except Exception as e:
            self.logger.error(
                f"[rl_v3_orchestrator] Failed to get config from PolicyStore: {e}"
            )
            # Return cached or default
            return self._config_cache if self._config_cache else {
                "enabled": True,
                "mode": "SHADOW",
                "min_confidence": 0.6,
                "max_trades_per_hour": 10,
            }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current config synchronously (returns cached)."""
        return self._config_cache if self._config_cache else {
            "enabled": True,
            "mode": "SHADOW",
            "min_confidence": 0.6,
            "max_trades_per_hour": 10,
        }
    
    def _check_rate_limit(self, max_trades_per_hour: int) -> bool:
        """Check if rate limit allows more trades."""
        now = datetime.utcnow()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        
        # Reset counter if new hour
        if self._last_trade_hour != current_hour:
            self._trade_count_hour = 0
            self._last_trade_hour = current_hour
        
        # Check limit
        if self._trade_count_hour >= max_trades_per_hour:
            return False
        
        return True
    
    def _increment_trade_count(self):
        """Increment trade count for rate limiting."""
        self._trade_count_hour += 1
    
    async def _cleanup_open_intent(self, symbol: str, timeout_seconds: int):
        """[PRODUCTION] Cleanup open intent after timeout to prevent deadlock."""
        await asyncio.sleep(timeout_seconds)
        if symbol in self._open_intents:
            self._open_intents.discard(symbol)
            self.logger.debug(f"[rl_v3_orchestrator] Cleaned up open intent for {symbol} after {timeout_seconds}s timeout")
    
    def clear_open_intent(self, symbol: str):
        """[PRODUCTION] Manually clear open intent (called by trade execution subscriber)."""
        if symbol in self._open_intents:
            self._open_intents.discard(symbol)
            self.logger.debug(f"[rl_v3_orchestrator] Manually cleared open intent for {symbol}")
    
    async def promote_to_live(self, mode: str = "PRIMARY") -> Tuple[bool, str]:
        """[PRODUCTION] Promote orchestrator from SHADOW to PRIMARY/HYBRID.
        
        Requires ACK from RiskGuard and ExitBrain before allowing live trading.
        
        Args:
            mode: Target mode ("PRIMARY" or "HYBRID")
            
        Returns:
            (success, message)
        """
        if mode not in ["PRIMARY", "HYBRID"]:
            return (False, f"Invalid mode: {mode}. Must be PRIMARY or HYBRID")
        
        current_config = await self._get_config()
        current_mode = current_config.get("mode", "SHADOW")
        
        if current_mode == mode:
            return (True, f"Already in {mode} mode")
        
        self.logger.warning(
            f"[rl_v3_orchestrator] 🚨 PROMOTION REQUEST: {current_mode} → {mode}"
        )
        
        # [PRODUCTION SAFETY] Check RiskGuard ACK
        try:
            # TODO: Implement actual ACK flow with RiskGuard
            # For now, require manual flag set
            if not self._promotion_acked:
                return (False, "❌ Promotion blocked: RiskGuard ACK required. Set _promotion_acked=True manually after verification.")
            
            # Update policy
            policy = await self.policy_store.get_policy()
            if hasattr(policy, "rl_v3_live"):
                policy.rl_v3_live["mode"] = mode
                await self.policy_store.update_policy(policy)
                
                # Clear cache to force reload
                self._config_cache = {}
                self._last_config_fetch = None
                
                self.logger.warning(
                    f"[rl_v3_orchestrator] ✅ PROMOTED: {current_mode} → {mode}"
                )
                
                return (True, f"Promoted to {mode} mode")
            else:
                return (False, "rl_v3_live config not found in policy")
                
        except Exception as e:
            self.logger.error(f"[rl_v3_orchestrator] Promotion failed: {e}")
            return (False, f"Promotion error: {str(e)}")
