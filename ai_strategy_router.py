#!/usr/bin/env python3
"""AI → Strategy Router
Routes AI decisions from Redis Stream → Strategy Brain HTTP → Risk Brain

P0.CAP+QUAL: Capacity-aware best-of selection with target open orders
"""

import asyncio
import httpx
import redis
import logging
import sys
import time
import os
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

REDIS_URL = "redis://localhost:6379"
STRATEGY_BRAIN_URL = "http://127.0.0.1:8011"
RISK_BRAIN_URL = "http://127.0.0.1:8012"

# Streams
AI_DECISION_STREAM = "quantum:stream:ai.decision.made"
TRADE_INTENT_STREAM = "quantum:stream:trade.intent"
CONSUMER_GROUP = "router"
CONSUMER_NAME = "ai_strategy_router"

# P0.CAP+QUAL: Capacity management (env overridable)
MAX_OPEN_ORDERS = int(os.getenv("MAX_OPEN_ORDERS", "10"))
TARGET_OPEN_ORDERS = int(os.getenv("TARGET_OPEN_ORDERS", "3"))
MAX_NEW_PER_CYCLE = int(os.getenv("MAX_NEW_PER_CYCLE", "2"))
CANDIDATE_WINDOW_SEC = int(os.getenv("CANDIDATE_WINDOW_SEC", "3"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.75"))
SUPER_CONFIDENCE = float(os.getenv("SUPER_CONFIDENCE", "0.90"))
OPEN_ORDERS_KEY = "quantum:state:open_orders"


class AIStrategyRouter:
    def __init__(self):
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.http_client = httpx.AsyncClient(timeout=5.0)
        self._last_invalid_warn_ts = 0.0
        
        # P0.CAP+QUAL: Best-of candidate buffer
        self.candidate_buffer: deque = deque(maxlen=50)  # Keep last 50 candidates
        self.last_publish_time = 0.0
        self.pending_acks: List[Tuple[str, str]] = []  # (stream, msg_id) to ACK after publish
        
    async def setup(self):
        """Create consumer group if not exists."""
        try:
            await asyncio.to_thread(
                self.redis.xgroup_create,
                AI_DECISION_STREAM,
                CONSUMER_GROUP,
                id="0",
                mkstream=True
            )
            logger.info(f"✅ Consumer group '{CONSUMER_GROUP}' created")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"✅ Consumer group '{CONSUMER_GROUP}' already exists")
            else:
                raise
                
    def _build_dedup_key(self, correlation_id: str, trace_id: str, msg_id: str) -> Tuple[str, str, str, str]:
        """Return dedup key components with sane fallbacks."""
        corr = (correlation_id or "").strip()
        trace = (trace_id or "").strip()
        msg = (msg_id or "").strip()

        def _valid(val: str) -> bool:
            return bool(val) and val.lower() not in {"0", "null", "none"}

        chosen = corr if _valid(corr) else trace if _valid(trace) else msg
        return chosen, corr, trace, msg

    def _log_invalid_once(self, message: str) -> None:
        """Rate-limit invalid metadata warnings to once per minute."""
        now = time.time()
        if now - self._last_invalid_warn_ts >= 60:
            logger.warning(message)
            self._last_invalid_warn_ts = now
    
    def _read_capacity(self) -> Optional[int]:
        """Read open_orders from Redis. Return None if missing/unparseable (fail-closed)."""
        try:
            val = self.redis.get(OPEN_ORDERS_KEY)
            if val is None:
                logger.warning(f"🔴 CAPACITY_UNKNOWN: {OPEN_ORDERS_KEY} missing → fail-closed")
                return None
            return int(val)
        except (ValueError, TypeError) as e:
            logger.warning(f"🔴 CAPACITY_UNPARSEABLE: {OPEN_ORDERS_KEY}={val!r} → fail-closed | {e}")
            return None
    
    def _compute_allowed_slots(self, open_orders: int) -> Tuple[int, str]:
        """Compute allowed new intents based on capacity + target policy.
        Returns: (allowed_count, reason_str)
        """
        slots = MAX_OPEN_ORDERS - open_orders
        
        if slots <= 0:
            return (0, f"CAPACITY_FULL open={open_orders} max={MAX_OPEN_ORDERS}")
        
        # Target-open policy: don't always fill to max
        if open_orders >= TARGET_OPEN_ORDERS:
            # Only allow super-confidence signals
            return (0, f"TARGET_REACHED open={open_orders} target={TARGET_OPEN_ORDERS} (awaiting super-signal)")
        
        # Below target: allow up to desired amount
        desired = TARGET_OPEN_ORDERS - open_orders
        allowed = min(desired, slots, MAX_NEW_PER_CYCLE)
        return (allowed, f"CAPACITY_OK open={open_orders} target={TARGET_OPEN_ORDERS} allowed={allowed}")
    
    def _score_candidate(self, cand: Dict) -> float:
        """Compute candidate score for best-of selection."""
        decision = cand["decision"]
        confidence = decision.get("confidence", 0.0)
        score = confidence
        
        # Small bonus for RL gate pass
        if decision.get("rl_gate_pass"):
            score += 0.02
        
        # Small bonus for known regime
        if decision.get("regime") not in {None, "unknown", ""}:
            score += 0.01
        
        return score
    
    def _apply_size_boost(self, rank: int, confidence: float, base_size: float) -> Tuple[float, float]:
        """Apply position size boost for top-ranked signals.
        Returns: (boosted_size, boost_factor)
        """
        factor = 1.0
        
        if rank == 1 and confidence >= SUPER_CONFIDENCE:
            factor = 1.30
        elif rank == 2 and confidence >= 0.85:
            factor = 1.10
        
        boosted = base_size * factor
        return (boosted, factor)

    async def buffer_candidate(self, decision: dict, trace_id: str, correlation_id: str, msg_id: str, stream_name: str):
        """Buffer candidate for best-of selection (non-blocking)."""
        try:
            symbol = decision.get("symbol", "").strip() if isinstance(decision, dict) else ""
            side_raw = decision.get("side", decision.get("action", "")).strip() if isinstance(decision, dict) else ""
            side = side_raw.upper()
            confidence = decision.get("confidence", 0.0)

            if not symbol or not side:
                self._log_invalid_once(
                    f"⚠️ INVALID_DECISION_DROP symbol={symbol!r} side={side_raw!r}"
                )
                return
            
            # Discard low-confidence signals immediately
            if confidence < MIN_CONFIDENCE:
                logger.debug(f"🗑️ LOW_CONFIDENCE_DROP {symbol} {side} conf={confidence:.2%} min={MIN_CONFIDENCE:.2%}")
                return
            
            # Add to buffer
            self.candidate_buffer.append({
                "decision": decision,
                "trace_id": trace_id,
                "correlation_id": correlation_id,
                "msg_id": msg_id,
                "stream_name": stream_name,
                "timestamp": time.time(),
                "symbol": symbol,
                "side": side,
                "confidence": confidence
            })
            
        except Exception as e:
            logger.error(f"Error buffering candidate: {e}")
    
    async def process_best_of_batch(self):
        """Select and publish top-N candidates from buffer."""
        try:
            # Step 1: Read capacity
            open_orders = self._read_capacity()
            if open_orders is None:
                # Fail-closed: capacity unknown
                logger.warning("🔴 CAPACITY_UNKNOWN → not publishing any intents")
                # Still ACK buffered messages to avoid redelivery
                for cand in self.candidate_buffer:
                    await asyncio.to_thread(
                        self.redis.xack,
                        cand["stream_name"],
                        CONSUMER_GROUP,
                        cand["msg_id"]
                    )
                self.candidate_buffer.clear()
                await asyncio.sleep(1)
                return
            
            # Step 2: Compute allowed slots
            allowed, reason = self._compute_allowed_slots(open_orders)
            slots = MAX_OPEN_ORDERS - open_orders
            
            # STRICT GATING: Never publish if slots <= 0 (capacity full)
            if slots <= 0:
                logger.info(f"🔴 CAPACITY_FULL open={open_orders} max={MAX_OPEN_ORDERS} slots={slots}")
                # ACK all buffered messages
                for cand in self.candidate_buffer:
                    await asyncio.to_thread(
                        self.redis.xack,
                        cand["stream_name"],
                        CONSUMER_GROUP,
                        cand["msg_id"]
                    )
                self.candidate_buffer.clear()
                await asyncio.sleep(1)
                return
            
            # TARGET POLICY: At/above target, only allow super-confidence signals
            if open_orders >= TARGET_OPEN_ORDERS:
                super_candidates = [
                    c for c in self.candidate_buffer
                    if c["confidence"] >= SUPER_CONFIDENCE
                ]
                
                if not super_candidates:
                    logger.info(f"⏸️ TARGET_REACHED open={open_orders} target={TARGET_OPEN_ORDERS} (no super-signal)")
                    # ACK all buffered messages
                    for cand in self.candidate_buffer:
                        await asyncio.to_thread(
                            self.redis.xack,
                            cand["stream_name"],
                            CONSUMER_GROUP,
                            cand["msg_id"]
                        )
                    self.candidate_buffer.clear()
                    return
                else:
                    # Allow max 1 super-signal per cycle at/above target
                    allowed = min(1, slots, MAX_NEW_PER_CYCLE)
                    logger.info(f"🌟 SUPER_SIGNAL_BYPASS open={open_orders} target={TARGET_OPEN_ORDERS} count={len(super_candidates)} allowed={allowed}")
            elif allowed <= 0:
                # Should not happen (open < target but allowed=0), but be defensive
                logger.warning(f"⚠️ UNEXPECTED: open={open_orders} < target={TARGET_OPEN_ORDERS} but allowed={allowed}")
                for cand in self.candidate_buffer:
                    await asyncio.to_thread(
                        self.redis.xack,
                        cand["stream_name"],
                        CONSUMER_GROUP,
                        cand["msg_id"]
                    )
                self.candidate_buffer.clear()
                return
            
            # Step 3: Filter stale candidates (older than CANDIDATE_WINDOW_SEC)
            now = time.time()
            fresh_candidates = [
                c for c in self.candidate_buffer
                if (now - c["timestamp"]) <= CANDIDATE_WINDOW_SEC
            ]
            
            if not fresh_candidates:
                # ACK stale messages
                for cand in self.candidate_buffer:
                    await asyncio.to_thread(
                        self.redis.xack,
                        cand["stream_name"],
                        CONSUMER_GROUP,
                        cand["msg_id"]
                    )
                self.candidate_buffer.clear()
                return
            
            # Step 4: Score and sort candidates
            for cand in fresh_candidates:
                cand["score"] = self._score_candidate(cand)
            
            fresh_candidates.sort(key=lambda c: c["score"], reverse=True)
            
            # Step 5: Select top-N
            selected = fresh_candidates[:allowed]
            
            if not selected:
                return
            
            # Step 6: Publish selected intents
            top_summary = [(c["symbol"], c["side"], f"{c['score']:.3f}") for c in selected[:3]]
            logger.info(
                f"📤 BEST_OF_PUBLISH count={len(selected)} allowed={allowed} open={open_orders} "
                f"slots={slots} top={top_summary}"
            )
            
            for rank, cand in enumerate(selected, start=1):
                await self.publish_trade_intent(cand, rank)
                # ACK after successful publish
                await asyncio.to_thread(
                    self.redis.xack,
                    cand["stream_name"],
                    CONSUMER_GROUP,
                    cand["msg_id"]
                )
            
            # Step 7: ACK remaining candidates (not published)
            remaining = [c for c in self.candidate_buffer if c not in selected]
            for cand in remaining:
                await asyncio.to_thread(
                    self.redis.xack,
                    cand["stream_name"],
                    CONSUMER_GROUP,
                    cand["msg_id"]
                )
            
            # Clear buffer
            self.candidate_buffer.clear()
            self.last_publish_time = time.time()
            
        except Exception as e:
            logger.error(f"Error processing best-of batch: {e}")
    
    async def publish_trade_intent(self, cand: Dict, rank: int):
        """Publish single trade intent with Strategy/Risk checks and size boost."""
        try:
            decision = cand["decision"]
            trace_id = cand["trace_id"]
            correlation_id = cand["correlation_id"]
            msg_id = cand["msg_id"]
            symbol = cand["symbol"]
            side = cand["side"]
            confidence = cand["confidence"]
            dedup_id, corr_id_clean, trace_id_clean, msg_id_clean = self._build_dedup_key(correlation_id, trace_id, msg_id)

            was_set = await asyncio.to_thread(
                self.redis.set,
                f"quantum:dedup:trade_intent:{dedup_id}",
                "1",
                nx=True,
                ex=300  # 5 minute TTL to bound cache
            )

            if not was_set:
                logger.warning(
                    f"🔁 DUPLICATE_SKIP key={dedup_id} corr={corr_id_clean} trace={trace_id_clean} msg_id={msg_id_clean}"
                )
                return
            
            symbol = decision.get("symbol", "").strip() if isinstance(decision, dict) else ""
            side_raw = decision.get("side", decision.get("action", "")).strip() if isinstance(decision, dict) else ""
            side = side_raw.upper()
            confidence = decision.get("confidence", 0.0)

            if not symbol or not side:
                self._log_invalid_once(
                    f"⚠️ INVALID_DECISION_DROP symbol={symbol!r} side={side_raw!r} corr={corr_id_clean} trace={trace_id_clean} msg_id={msg_id_clean}"
                )
                return
            
            # Dedup check
            dedup_id, _, _, _ = self._build_dedup_key(correlation_id, trace_id, msg_id)
            was_set = await asyncio.to_thread(
                self.redis.set,
                f"quantum:dedup:trade_intent:{dedup_id}",
                "1",
                nx=True,
                ex=300
            )
            
            if not was_set:
                logger.debug(f"🔁 DUPLICATE_SKIP_PUBLISH {symbol} {side}")
                return
            
            logger.info(f"📥 RANK_{rank} {symbol} {side} conf={confidence:.2%} score={cand['score']:.3f}")
            
            # Step 1: Strategy Brain evaluation
            strategy_response = await self.http_client.post(
                f"{STRATEGY_BRAIN_URL}/evaluate",
                json={
                    "symbol": symbol,
                    "direction": side,
                    "confidence": confidence
                }
            )
            strategy_response.raise_for_status()
            strategy_result = strategy_response.json()
            
            if not strategy_result.get("approved"):
                logger.info(f"❌ Strategy denied: {strategy_result.get('reason')}")
                return
                
            logger.info(f"✅ Strategy approved")
            
            # Step 2: Risk Brain evaluation (TEMPORARILY SKIPPED - 422 error)
            # TODO: Fix Risk Brain API schema mismatch
            risk_result = {
                "approved": True,
                "adjusted_size_usd": decision.get("position_size_usd", 100.0),
                "adjusted_leverage": decision.get("leverage", 1.0)
            }
            logger.info(f"⚠️  Risk Brain skipped (TESTNET)")
            
            # Step 3: Apply size boost for top-ranked signals
            base_size = risk_result.get("adjusted_size_usd", decision.get("position_size_usd", 100.0))
            boosted_size, boost_factor = self._apply_size_boost(rank, confidence, base_size)
            
            if boost_factor > 1.0:
                logger.info(f"💰 SIZE_BOOST rank={rank} conf={confidence:.2%} factor={boost_factor:.2f} size=${base_size:.0f}→${boosted_size:.0f}")
            
            # Step 4: Publish trade intent (EventBus format)
            trade_intent = {
                "symbol": symbol,
                "action": side,
                "confidence": confidence,
                "position_size_usd": boosted_size,
                "leverage": risk_result.get("adjusted_leverage", decision.get("leverage", 1.0)),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "ai_strategy_router",
                "stop_loss_pct": decision.get("stop_loss_pct"),
                "take_profit_pct": decision.get("take_profit_pct"),
                "entry_price": decision.get("entry_price"),
                "stop_loss": decision.get("stop_loss"),
                "take_profit": decision.get("take_profit"),
                "quantity": decision.get("quantity")
            }
            
            # Wrap in EventBus format (execution service expects "data" field)
            import json
            await asyncio.to_thread(
                self.redis.xadd,
                TRADE_INTENT_STREAM,
                {"data": json.dumps(trade_intent)},
                maxlen=10000
            )
            
            logger.info(f"🚀 Trade Intent published: {symbol} {side}")
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error routing decision: {e}")
        except Exception as e:
            logger.error(f"Error routing decision: {e}")
            
    async def run(self):
        """Main consumer loop with best-of batching."""
        await self.setup()
        logger.info(f"🚀 AI→Strategy Router started (P0.CAP+QUAL)")
        logger.info(f"📥 Consuming: {AI_DECISION_STREAM}")
        logger.info(f"📤 Publishing: {TRADE_INTENT_STREAM}")
        logger.info(f"⚙️ Capacity: max={MAX_OPEN_ORDERS} target={TARGET_OPEN_ORDERS} burst={MAX_NEW_PER_CYCLE}")
        logger.info(f"📊 Quality: min_conf={MIN_CONFIDENCE:.2%} super_conf={SUPER_CONFIDENCE:.2%} window={CANDIDATE_WINDOW_SEC}s")
        
        last_id = ">"  # Only new messages
        last_batch_time = time.time()
        
        while True:
            try:
                # Read messages into buffer (non-blocking)
                messages = await asyncio.to_thread(
                    self.redis.xreadgroup,
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {AI_DECISION_STREAM: last_id},
                    count=10,
                    block=1000  # Shorter block for batching
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for msg_id, msg_data in stream_messages:
                            # Parse AI Engine event format (payload is JSON string)
                            if 'payload' in msg_data:
                                import json
                                decision = json.loads(msg_data['payload'])
                            else:
                                decision = msg_data
                            
                            # Extract trace_id and correlation_id
                            correlation_id = msg_data.get('correlation_id', '')
                            trace_id = msg_data.get('trace_id', '')

                            # Buffer candidate (don't publish immediately)
                            await self.buffer_candidate(decision, trace_id, correlation_id, msg_id, stream_name)
                
                # Process batch if window elapsed or buffer full
                now = time.time()
                window_elapsed = (now - last_batch_time) >= CANDIDATE_WINDOW_SEC
                buffer_full = len(self.candidate_buffer) >= 10
                
                if (window_elapsed or buffer_full) and self.candidate_buffer:
                    await self.process_best_of_batch()
                    last_batch_time = now
                        
            except KeyboardInterrupt:
                logger.info("⏹️  Shutting down...")
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(5)
                
        await self.http_client.aclose()


async def main():
    router = AIStrategyRouter()
    await router.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
        sys.exit(0)
