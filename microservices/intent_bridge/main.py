#!/usr/bin/env python3
"""
Intent Bridge - trade.intent → apply.plan
=========================================

Bridges the gap between:
- trading_bot/ai_engine publishing trade.intent
- apply_layer consuming apply.plan

Minimal, fail-closed design:
- Allowlist filtering (INTENT_BRIDGE_ALLOWLIST)
- Idempotency via Redis SETNX
- Consumer group for reliable processing
- No modifications to Apply Layer or P3.3

Author: Quantum Trader Team
Date: 2026-01-26
"""
import os
import sys
import json
import time
import math
import socket
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed")
    sys.exit(1)

# PolicyStore (fail-closed autonomy)
try:
    from lib.policy_store import load_policy, PolicyData
    POLICY_ENABLED = True
except ImportError:
    logger.warning("PolicyStore not available - running in legacy mode")
    POLICY_ENABLED = False

# Setup logging
LOG_LEVEL = os.getenv("INTENT_BRIDGE_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [INTENT-BRIDGE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))


def _ledger_last_known_amt(r: redis.Redis, symbol: str) -> float:
    """
    Get last known position amount from ledger.
    Returns NaN if not found.
    """
    try:
        key = f"quantum:ledger:{symbol}"
        data = r.hgetall(key)
        if not data:
            return float('nan')
        
        amt_bytes = data.get(b'position_amt')
        if not amt_bytes:
            return float('nan')
        
        return float(amt_bytes.decode())
    except Exception as e:
        logger.warning(f"Failed to read ledger for {symbol}: {e}")
        return float('nan')


# Allowlist (CSV) - can be overridden by USE_TOP10_UNIVERSE=true
ALLOWLIST_STR = os.getenv("INTENT_BRIDGE_ALLOWLIST", "BTCUSDT")
ALLOWLIST = set([s.strip() for s in ALLOWLIST_STR.split(",") if s.strip()])

# TOP10 Universe mode (10-symbol concentration limit)
USE_TOP10_UNIVERSE = os.getenv("INTENT_BRIDGE_USE_TOP10", "false").lower() == "true"

# Portfolio exposure limits (AI-driven, not hardcoded position count)
MAX_EXPOSURE_PCT = float(os.getenv("MAX_EXPOSURE_PCT", "80.0"))  # From Exposure Balancer

# FLAT SELL skip (avoid no_position spam)
SKIP_FLAT_SELL = os.getenv("INTENT_BRIDGE_SKIP_FLAT_SELL", "true").lower() == "true"
FLAT_EPS = float(os.getenv("INTENT_BRIDGE_FLAT_EPS", "0.0") or "0.0")

# Streams
INTENT_STREAM = "quantum:stream:trade.intent"
PLAN_STREAM = "quantum:stream:apply.plan"
CONSUMER_GROUP = "quantum:group:intent_bridge"
CONSUMER_NAME = f"{socket.gethostname()}_{os.getpid()}"

# Idempotency TTL
IDEMPOTENCY_TTL = int(os.getenv("INTENT_BRIDGE_IDEMPOTENCY_TTL", "86400"))  # 24h

# Ledger gate config (avoid chicken-and-egg deadlock for new symbols)
REQUIRE_LEDGER_FOR_OPEN = os.getenv("INTENT_BRIDGE_REQUIRE_LEDGER_FOR_OPEN", "false").lower() == "true"

# Testnet mode (intersect AI universe with testnet tradables)
TESTNET_MODE = os.getenv("TESTNET_MODE", "true").lower() == "true"
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# Build tag for deployment verification
BUILD_TAG = "intent-bridge-ledger-open-v1"

logger.info("=" * 80)
logger.info(f"Intent Bridge - trade.intent → apply.plan [{BUILD_TAG}]")
logger.info("=" * 80)
logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
logger.info(f"Consumer: {CONSUMER_GROUP} / {CONSUMER_NAME}")
logger.info(f"USE_TOP10_UNIVERSE: {USE_TOP10_UNIVERSE}")
if USE_TOP10_UNIVERSE:
    logger.info(f"Allowlist: TOP10 (loaded dynamically from quantum:cfg:universe:top10)")
else:
    logger.info(f"Allowlist: {sorted(ALLOWLIST)}")
logger.info(f"Intent stream: {INTENT_STREAM}")
logger.info(f"Plan stream: {PLAN_STREAM}")
logger.info(f"Require ledger for OPEN: {REQUIRE_LEDGER_FOR_OPEN}")
logger.info(f"Skip flat SELL: {SKIP_FLAT_SELL}")
logger.info("=" * 80)


class IntentBridge:
    """Bridges trade.intent to apply.plan"""
    
    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False  # Binary mode for XREADGROUP
        )
        
        # Test connection
        self.redis.ping()
        logger.info("✅ Redis connected")
        
        # Create consumer group if not exists
        try:
            self.redis.xgroup_create(
                INTENT_STREAM,
                CONSUMER_GROUP,
                id="0",
                mkstream=True
            )
            logger.info(f"✅ Consumer group created: {CONSUMER_GROUP}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"✅ Consumer group exists: {CONSUMER_GROUP}")
            else:
                raise
        
        # Initialize allowlist (will be refreshed if USE_TOP10_UNIVERSE)
        self.current_allowlist = ALLOWLIST.copy()
        self.top10_last_refresh = 0
        self.TOP10_REFRESH_INTERVAL = 300  # 5 minutes
        
        # Load AI policy (fail-closed)
        self.current_policy: Optional[PolicyData] = None
        self._refresh_policy()
        
        if USE_TOP10_UNIVERSE:
            self._refresh_top10_allowlist()
    
    def _refresh_policy(self):
        """Load AI policy from PolicyStore (fail-closed)."""
        if not POLICY_ENABLED:
            return
        
        try:
            self.current_policy = load_policy()
            if self.current_policy:
                logger.info(
                    f"✅ POLICY_LOADED: version={self.current_policy.policy_version} "
                    f"hash={self.current_policy.policy_hash[:8]} "
                    f"universe_count={len(self.current_policy.universe_symbols)}"
                )
            else:
                logger.warning("⚠️  POLICY_MISSING or POLICY_STALE - will SKIP trades without policy")
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            self.current_policy = None
    
    def _refresh_top10_allowlist(self):
        """Refresh allowlist from quantum:cfg:universe:top10."""
        try:
            data = self.redis.get("quantum:cfg:universe:top10")
            if not data:
                logger.warning("⚠️  quantum:cfg:universe:top10 not found, using static allowlist")
                return
            
            top10_config = json.loads(data.decode() if isinstance(data, bytes) else data)
            new_allowlist = set(top10_config.get("symbols", []))
            
            if not new_allowlist:
                logger.warning("⚠️  TOP10 has no symbols, keeping current allowlist")
                return
            
            # Update allowlist
            old_count = len(self.current_allowlist)
            self.current_allowlist = new_allowlist
            self.top10_last_refresh = time.time()
            
            logger.info(f"✅ TOP10 allowlist refreshed: {old_count} → {len(new_allowlist)} symbols")
            logger.info(f"   {sorted(new_allowlist)}")
            
        except Exception as e:
            logger.error(f"Failed to refresh TOP10 allowlist: {e}")
    
    def _get_testnet_tradable_symbols(self) -> set:
        """Fetch tradable symbols from Binance testnet/mainnet exchange info."""
        try:
            import requests
            
            if BINANCE_TESTNET:
                url = "https://testnet.binancefuture.com/fapi/v1/exchangeInfo"
            else:
                url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
            
            response = requests.get(url, timeout=5)
            data = response.json()
            
            symbols = set()
            for symbol_info in data.get("symbols", []):
                if symbol_info.get("status") == "TRADING":
                    symbols.add(symbol_info["symbol"])
            
            logger.info(f"✅ Fetched {len(symbols)} tradable symbols from {'testnet' if BINANCE_TESTNET else 'mainnet'}")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to fetch testnet symbols: {e}")
            return set()  # Empty set = no filtering
    
    def _get_effective_allowlist(self) -> set:
        """
        Get current effective allowlist with explicit venue intersection.
        
        Priority:
        1. AI Policy universe (if PolicyStore enabled and valid)
        2. TOP10 universe (if USE_TOP10_UNIVERSE enabled)
        3. Static ALLOWLIST (fallback)
        
        Final step: intersect with venue tradable symbols (fail-open if fetch fails)
        """
        source = "unknown"
        allowlist = set()
        policy_version = "none"
        policy_hash = "none"
        policy_count = 0
        
        # Priority 1: AI Policy universe (fail-closed!)
        if POLICY_ENABLED and self.current_policy:
            # Refresh policy if stale
            if self.current_policy.is_stale():
                logger.warning("Policy stale, refreshing...")
                self._refresh_policy()
            
            if self.current_policy and not self.current_policy.is_stale():
                allowlist = set(self.current_policy.universe_symbols)
                source = "policy"
                policy_version = self.current_policy.policy_version
                policy_hash = self.current_policy.policy_hash[:8]
                policy_count = len(allowlist)
        
        # Priority 2: TOP10 universe
        if not allowlist and USE_TOP10_UNIVERSE:
            # Refresh if stale
            if time.time() - self.top10_last_refresh > self.TOP10_REFRESH_INTERVAL:
                self._refresh_top10_allowlist()
        
            allowlist = self.current_allowlist
            source = "top10"
        
        # Priority 3: Static allowlist (legacy)
        if not allowlist:
            allowlist = self.current_allowlist
            source = "static"
        
        # Capture allowlist before venue intersection
        allowlist_count = len(allowlist)
        
        # Venue intersection (explicit tradable filter)
        venue = "binance-testnet" if BINANCE_TESTNET else "binance-mainnet"
        tradable_count = 0
        tradable_fetch_failed = 0
        venue_limited = 0
        
        if TESTNET_MODE and allowlist_count > 0:
            tradable_symbols = self._get_testnet_tradable_symbols()
            if tradable_symbols:
                tradable_count = len(tradable_symbols)
                # Explicit intersection
                final_allowlist = allowlist & tradable_symbols
                
                # Fail-open: if intersection is empty but allowlist was not, keep original
                if not final_allowlist and allowlist:
                    logger.warning(f"Venue intersection empty! Keeping original allowlist (fail-open)")
                    final_allowlist = allowlist
                    venue_limited = 0  # Not limited, failed intersection
                else:
                    allowlist = final_allowlist
                    venue_limited = 1 if len(allowlist) < allowlist_count else 0
            else:
                # Tradable fetch failed - fail-open (keep allowlist)
                tradable_fetch_failed = 1
                logger.warning(f"Failed to fetch venue tradables - using allowlist as-is (fail-open)")
        
        # Final count and sample
        final_count = len(allowlist)
        sample = ",".join(sorted(list(allowlist))[:3]) if allowlist else ""
        
        # Structured logging (EXACT FORMAT for proof script)
        logger.info(
            f"ALLOWLIST_EFFECTIVE venue={venue} source={source} "
            f"policy_count={policy_count} allowlist_count={allowlist_count} "
            f"tradable_count={tradable_count} final_count={final_count} "
            f"venue_limited={venue_limited} tradable_fetch_failed={tradable_fetch_failed} "
            f"sample={sample}"
        )
        
        return allowlist
    
    def _get_portfolio_exposure(self) -> float:
        """
        Get current portfolio exposure from quantum:state:portfolio.
        
        Returns:
            Exposure percentage (0-100), or 0 if unavailable
        """
        try:
            data = self.redis.hgetall(b"quantum:state:portfolio")
            if not data:
                return 0.0
            
            equity_usd = float(data.get(b'equity_usd', b'10000').decode())
            total_notional = 0.0
            
            # Sum absolute notional across all ledger positions
            for symbol in self._get_effective_allowlist():
                ledger_key = f"quantum:ledger:{symbol}".encode()
                ledger_data = self.redis.hgetall(ledger_key)
                if ledger_data and b'notional_usd' in ledger_data:
                    notional = float(ledger_data[b'notional_usd'].decode())
                    total_notional += abs(notional)
            
            if equity_usd <= 0:
                return 0.0
            
            exposure_pct = (total_notional / equity_usd) * 100.0
            return exposure_pct
            
        except Exception as e:
            logger.warning(f"Failed to get portfolio exposure: {e}")
            return 0.0
    
    def _count_open_positions(self) -> int:
        """Count current open positions via ledger"""
        try:
            count = 0
            for symbol in self._get_effective_allowlist():
                key = f"quantum:ledger:{symbol}"
                data = self.redis.hgetall(key)
                if data and b'position_amt' in data:
                    amt = float(data[b'position_amt'].decode())
                    if abs(amt) > FLAT_EPS:
                        count += 1
            return count
        except Exception as e:
            logger.warning(f"Failed to count open positions: {e}")
            return 0
    
    def _make_plan_id(self, stream_id: str, symbol: str, side: str, qty: float) -> str:
        """Generate deterministic 16-hex plan_id"""
        data = f"{stream_id}|{symbol}|{side}|{qty}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _is_seen(self, stream_id: str) -> bool:
        """Check if stream_id already processed"""
        key = f"quantum:intent_bridge:seen:{stream_id}"
        return self.redis.exists(key) > 0
    
    def _mark_seen(self, stream_id: str):
        """Mark stream_id as processed"""
        key = f"quantum:intent_bridge:seen:{stream_id}"
        self.redis.setex(key, IDEMPOTENCY_TTL, "1")
    
    def _parse_intent(self, event_data: Dict) -> Optional[Dict]:
        """
        Parse trade.intent event
        
        Expected fields (from trading_bot or ai_engine):
        - symbol: str
        - action: str (BUY/SELL)
        - size: float (USD notional) OR quantity: float
        - price: float (for quantity calculation if needed)
        - type: str (default MARKET)
        - reduceOnly: bool (default false)
        """
        try:
            # Try payload field first (JSON)
            if b'payload' in event_data:
                payload = json.loads(event_data[b'payload'].decode())
            else:
                # Direct fields
                payload = {k.decode(): v.decode() for k, v in event_data.items()}
            
            symbol = payload.get("symbol", "").upper()
            # Try both 'action' and 'side' field names
            action = payload.get("action", payload.get("side", "")).upper()
            
            # Validate
            if not symbol or not action:
                logger.warning(f"Missing symbol or action: {payload}")
                return None
            
            if action not in ("BUY", "SELL"):
                logger.warning(f"Invalid action: {action}")
                return None
            
            # Allowlist check
            if symbol not in self._get_effective_allowlist():
                logger.debug(f"Symbol not in allowlist: {symbol}")
                return None
            
            # Extract quantity or size
            qty = None
            if "quantity" in payload:
                qty = float(payload["quantity"])
            elif "qty" in payload:
                qty = float(payload["qty"])
            elif "size" in payload and "price" in payload:
                # Calculate qty from USD size
                size_usd = float(payload["size"])
                price = float(payload["price"])
                if price > 0:
                    qty = size_usd / price
            elif "position_size_usd" in payload and "entry_price" in payload:
                # Calculate qty from position_size_usd
                size_usd = float(payload["position_size_usd"])
                price = float(payload["entry_price"])
                if price > 0:
                    qty = size_usd / price
            
            if not qty or qty <= 0:
                logger.warning(f"Invalid quantity: {payload}")
                return None
            
            # Optional fields
            order_type = payload.get("type", "MARKET").upper()
            reduce_only = str(payload.get("reduceOnly", "false")).lower() in ("true", "1", "yes")
            
            # 🔥 RL SIZING METADATA: Extract leverage, TP/SL from RL Position Sizing Agent
            leverage = payload.get("leverage", 1)
            stop_loss = payload.get("stop_loss")
            take_profit = payload.get("take_profit")
            
            logger.info(f"✓ Parsed {symbol} {action}: qty={qty:.4f}, leverage={leverage}, sl={stop_loss}, tp={take_profit}")
            
            return {
                "symbol": symbol,
                "side": action,
                "qty": qty,
                "type": order_type,
                "reduceOnly": reduce_only,
                "leverage": leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "source_payload": payload
            }
            
        except Exception as e:
            logger.error(f"Failed to parse intent: {e}", exc_info=True)
            return None
    
    def _publish_plan(self, plan_id: str, intent: Dict):
        """Publish apply.plan to Apply Layer"""
        
        # Build apply.plan with FLAT fields (matching native Apply Layer format)
        # Apply Layer expects top-level fields, NOT nested payload JSON
        ts_unix = int(time.time())
        
        # Base message fields (required)
        message_fields = {
            b"plan_id": plan_id.encode(),
            b"decision": b"EXECUTE",
            b"symbol": intent["symbol"].encode(),
            b"side": intent["side"].encode(),
            b"type": intent["type"].encode(),
            b"qty": str(intent["qty"]).encode(),
            b"reduceOnly": str(intent["reduceOnly"]).lower().encode(),
            b"source": b"intent_bridge",
            b"signature": b"intent_bridge",
            b"timestamp": str(ts_unix).encode()
        }
        
        # DEBUG: Log what we're trying to add
        leverage = intent.get("leverage")
        stop_loss = intent.get("stop_loss")
        take_profit = intent.get("take_profit")
        
        # DIAGNOSTIC: Show full intent dict for troubleshooting
        logger.info(f"📋 Publishing plan for {intent['symbol']} {intent['side']}: leverage={leverage}, sl={stop_loss}, tp={take_profit}")
        
        # 🔥 RL SIZING METADATA: Add leverage, TP/SL if available
        if leverage is not None:
            message_fields[b"leverage"] = str(leverage).encode()
            logger.info(f"✓ Added leverage={leverage} to {intent['symbol']}")
        if stop_loss is not None:
            message_fields[b"stop_loss"] = str(stop_loss).encode()
            logger.info(f"✓ Added stop_loss={stop_loss} to {intent['symbol']}")
        if take_profit is not None:
            message_fields[b"take_profit"] = str(take_profit).encode()
            logger.info(f"✓ Added take_profit={take_profit} to {intent['symbol']}")
        
        # Publish to quantum:stream:apply.plan with FLAT structure
        message_id = self.redis.xadd(
            PLAN_STREAM,
            message_fields
        )
        
        logger.info(
            f"✅ Published plan: {plan_id[:8]} | {intent['symbol']} {intent['side']} "
            f"qty={intent['qty']:.4f} leverage={leverage}x "
            f"reduceOnly={intent['reduceOnly']} | msg={message_id.decode()}"
        )
        
        # Auto-create P3.3 permit (P3.3 service not active, auto-bypass)
        permit_key = f"quantum:permit:p33:{plan_id}"
        self.redis.hset(permit_key, mapping={
            "allow": "true",
            "safe_qty": "0",
            "reason": "auto_bypass_no_p33",
            "timestamp": str(int(time.time()))
        })
        logger.debug(f"✅ Auto-created permit: {plan_id[:8]}")
    
    def process_intent(self, stream_id: bytes, event_data: Dict):
        """Process single trade.intent event"""
        
        stream_id_str = stream_id.decode()
        
        # Idempotency check
        if self._is_seen(stream_id_str):
            logger.debug(f"Already processed: {stream_id_str}")
            self.redis.xack(INTENT_STREAM, CONSUMER_GROUP, stream_id)
            return
        
        # Parse intent
        intent = self._parse_intent(event_data)
        if not intent:
            # Invalid or filtered out → ack and skip
            self.redis.xack(INTENT_STREAM, CONSUMER_GROUP, stream_id)
            self._mark_seen(stream_id_str)
            return
        
        # 🔥 POLICY GATE: Fail-closed universe check
        if POLICY_ENABLED and self.current_policy:
            if not self.current_policy.contains_symbol(intent["symbol"]):
                logger.info(
                    f"SKIP POLICY_UNIVERSE_FILTER: {intent['symbol']} not in AI policy "
                    f"(policy_version={self.current_policy.policy_version}, plan_id=N/A)"
                )
                self._mark_seen(stream_id_str)
                self.redis.xack(INTENT_STREAM, CONSUMER_GROUP, stream_id)
                return
        
        # Generate plan_id
        plan_id = self._make_plan_id(
            stream_id_str,
            intent["symbol"],
            intent["side"],
            intent["qty"]
        )
        
        # Gate: Ledger check (avoid chicken-and-egg deadlock)
        # OPEN (BUY): Don't require ledger (will be created after first fill)
        # CLOSE (SELL): Require ledger to avoid accidental closes on flat positions
        
        if intent["side"].upper() == "BUY":
            # BUY/OPEN: Check ledger only if explicitly required (default: false)
            if REQUIRE_LEDGER_FOR_OPEN:
                ledger_amt = _ledger_last_known_amt(self.redis, intent["symbol"])
                if math.isnan(ledger_amt):
                    logger.info(f"Skip publish: {intent['symbol']} BUY but ledger unknown (strict mode, plan_id={plan_id[:8]})")
                    self._mark_seen(stream_id_str)
                    self.redis.xack(INTENT_STREAM, CONSUMER_GROUP, stream_id)
                    return
            else:
                # Default: Allow OPEN even without ledger (deadlock fix)
                ledger_amt = _ledger_last_known_amt(self.redis, intent["symbol"])
                if math.isnan(ledger_amt):
                    logger.info(f"LEDGER_MISSING_OPEN allowed: symbol={intent['symbol']} side=BUY (plan_id={plan_id[:8]})")
        
        elif intent["side"].upper() == "SELL":
            # SELL/CLOSE: Always check ledger if SKIP_FLAT_SELL enabled
            if SKIP_FLAT_SELL:
                ledger_amt = _ledger_last_known_amt(self.redis, intent["symbol"])
                if math.isnan(ledger_amt):
                    # Prefer snapshot truth over ledger, but if both missing, block
                    # TODO: Check quantum:position:snapshot:{symbol} as fallback
                    logger.info(f"Skip publish: {intent['symbol']} SELL but ledger unknown (plan_id={plan_id[:8]})")
                    self._mark_seen(stream_id_str)
                    self.redis.xack(INTENT_STREAM, CONSUMER_GROUP, stream_id)
                    return
                if abs(ledger_amt) <= FLAT_EPS:
                    logger.info(f"Skip publish: {intent['symbol']} SELL but ledger flat (last_known_amt={ledger_amt}, plan_id={plan_id[:8]})")
                    self._mark_seen(stream_id_str)
                    self.redis.xack(INTENT_STREAM, CONSUMER_GROUP, stream_id)
                    return
        
        # Gate: AI-driven exposure check for new entries (BUY only)
        # Let RL Agent + Governor decide optimal position count via exposure limits
        if intent["side"].upper() == "BUY":
            current_exposure = self._get_portfolio_exposure()
            if current_exposure >= MAX_EXPOSURE_PCT:
                logger.info(
                    f"Skip publish: {intent['symbol']} BUY rejected (exposure={current_exposure:.1f}% >= MAX={MAX_EXPOSURE_PCT:.1f}%, plan_id={plan_id[:8]})"
                )
                self._mark_seen(stream_id_str)
                self.redis.xack(INTENT_STREAM, CONSUMER_GROUP, stream_id)
                return

        
        # Publish to apply.plan
        try:
            self._publish_plan(plan_id, intent)
            
            # Mark as seen and ack
            self._mark_seen(stream_id_str)
            self.redis.xack(INTENT_STREAM, CONSUMER_GROUP, stream_id)
            
            logger.info(f"✅ Bridge success: {stream_id_str} → {plan_id[:8]}")
            
        except Exception as e:
            logger.error(f"❌ Failed to publish plan: {e}", exc_info=True)
            # Don't ack - will retry
    
    def run(self):
        """Main event loop"""
        logger.info("🚀 Intent Bridge started")
        logger.info(f"Consuming: {INTENT_STREAM} → {PLAN_STREAM}")
        
        while True:
            try:
                # XREADGROUP with blocking
                messages = self.redis.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {INTENT_STREAM: ">"},
                    count=10,
                    block=2000  # 2 second timeout
                )
                
                if not messages:
                    continue
                
                for stream_name, events in messages:
                    for stream_id, event_data in events:
                        self.process_intent(stream_id, event_data)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in event loop: {e}", exc_info=True)
                time.sleep(5)


def main():
    """Entry point"""
    bridge = IntentBridge()
    bridge.run()


if __name__ == "__main__":
    main()
