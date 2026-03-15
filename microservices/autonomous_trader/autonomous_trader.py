"""
Autonomous Trader - Full autonomy trading service

Combines:
- Entry scanning
- Position monitoring
- Exit management
- RL-based sizing
- Intent execution
"""
import asyncio
import logging
import os
import sys
import time
from typing import Dict, List, Optional
import redis.asyncio as redis

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from microservices.autonomous_trader.position_tracker import PositionTracker, Position
from microservices.autonomous_trader.entry_scanner import EntryScanner, EntryOpportunity
from microservices.autonomous_trader.exit_manager import ExitManager, ExitDecision
from microservices.autonomous_trader.funding_rate_filter import get_filtered_symbols
from microservices.rl_sizing_agent.rl_agent import RLPositionSizingAgent
import json

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


async def get_authoritative_open_positions(redis_client) -> int:
    """
    Returns the count of currently open positions by scanning
    quantum:state:positions:{SYMBOL} keys directly from Redis.

    Rules:
    - Only counts keys where abs(float(quantity)) > 0.
    - Uses non-blocking SCAN with COUNT 200 per batch.
    - Hard-limits total key inspection to 10k to prevent runaway scans.
    - Logs a warning if truncated.

    Returns: int open_count
    Raises:  RuntimeError on scan failure (caller must handle).
    """
    count: int = 0
    keys_scanned: int = 0
    MAX_KEYS: int = 10_000
    SCAN_COUNT: int = 200
    cursor: int = 0
    truncated: bool = False

    try:
        while True:
            cursor, keys = await redis_client.scan(
                cursor, match="quantum:state:positions:*", count=SCAN_COUNT
            )
            for key in keys:
                keys_scanned += 1
                if keys_scanned > MAX_KEYS:
                    truncated = True
                    break

                qty_raw = await redis_client.hget(key, "quantity")
                if qty_raw is not None:
                    try:
                        if abs(float(qty_raw)) > 0.0:
                            count += 1
                    except (ValueError, TypeError):
                        pass  # unparseable quantity — skip

            if truncated or cursor == 0:
                break

        if truncated:
            logger.warning(
                f"[SLOT_FIX] SCAN truncated at {MAX_KEYS} keys — "
                f"authoritative_count={count} may be undercount"
            )

    except Exception as e:
        raise RuntimeError(
            f"get_authoritative_open_positions failed: {e}"
        ) from e

    return count


class AutonomousTrader:
    """
    Autonomous trading system
    
    Main loop:
    1. Scan for entry opportunities
    2. Monitor active positions
    3. Calculate position sizing (RL)
    4. Publish trade intents
    5. Learn from outcomes
    """
    
    def __init__(self):
        # Redis connection
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis = redis.from_url(redis_url, decode_responses=True)
        
        # Configuration
        self.max_positions = int(os.getenv("MAX_POSITIONS", "5"))
        self.max_exposure_usd = float(os.getenv("MAX_EXPOSURE_USD", "2500"))
        self.max_position_usd = float(os.getenv("MAX_POSITION_USD", "500"))
        self.scan_interval_sec = int(os.getenv("SCAN_INTERVAL_SEC", "30"))
        self.min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.65"))
        
        # Universe OS configuration
        self.use_universe_os = os.getenv("USE_UNIVERSE_OS", "false").lower() == "true"
        self.universe_max_symbols = int(os.getenv("UNIVERSE_MAX_SYMBOLS", "50"))
        
        # Components
        self.position_tracker = PositionTracker(self.redis)
        
        # Parse symbols from env (fallback if Universe OS disabled)
        symbols_str = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT")
        candidate_symbols = [s.strip() for s in symbols_str.split(",")]
        
        # NOTE: Actual symbol filtering happens async in start() method
        # Store candidates for now, will be filtered by funding rates on startup
        self.candidate_symbols = candidate_symbols
        
        # Placeholder - will be initialized in start() after funding filter
        self.entry_scanner = None
        self.exit_manager = ExitManager(self.redis, use_ai_exits=True)
        
        # RL Sizing Agent
        model_path = os.getenv("RL_MODEL_PATH", "/models/rl_sizing_agent_v3.pth")
        self.rl_agent = RLPositionSizingAgent(model_path=model_path)
        
        # Stream routing — override via env to redirect entries to shadow stream
        # STREAM_ENTRY_INTENT=quantum:stream:shadow.intent  → shadow mode (no live execution)
        # STREAM_ENTRY_INTENT=quantum:stream:trade.intent   → live mode (default)
        self.stream_entry_intent = os.getenv(
            "STREAM_ENTRY_INTENT", "quantum:stream:trade.intent"
        )

        # State
        self._running = False
        self._main_loop_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.cycle_count = 0
        self.entries_executed = 0
        self.exits_executed = 0
        
        logger.info("[AutonomousTrader] Initialized")
        logger.info(f"  Max positions: {self.max_positions}")
        logger.info(f"  Entry stream: {self.stream_entry_intent}")
        logger.info(f"  Max exposure: ${self.max_exposure_usd}")
        logger.info(f"  Max position size: ${self.max_position_usd}")
        logger.info(f"  Scan interval: {self.scan_interval_sec}s")
        logger.info(f"  USE_UNIVERSE_OS: {self.use_universe_os}")
        if self.use_universe_os:
            logger.info(f"  Universe max symbols: {self.universe_max_symbols}")
        else:
            logger.info(f"  Candidate symbols: {len(self.candidate_symbols)} (hardcoded from ENV)")
    
    async def _get_universe_symbols(self) -> List[str]:
        """
        Fetch symbols from Universe Service (Redis)
        
        Returns dynamic symbol list from quantum:cfg:universe:active
        Falls back to ENV symbols if Universe Service unavailable
        """
        try:
            universe_data = await self.redis.get("quantum:cfg:universe:active")
            if not universe_data:
                logger.warning("[Universe] No universe data in Redis, using ENV fallback")
                return self.candidate_symbols
            
            data = json.loads(universe_data)
            all_symbols = data.get("symbols", [])
            
            if not all_symbols:
                logger.warning("[Universe] Empty symbol list, using ENV fallback")
                return self.candidate_symbols
            
            # Limit to configured max
            symbols = all_symbols[:self.universe_max_symbols]
            
            logger.info(f"[Universe] Loaded {len(symbols)} symbols from Universe Service (total available: {len(all_symbols)})")
            return symbols
            
        except Exception as e:
            logger.error(f"[Universe] Failed to fetch universe: {e}, using ENV fallback")
            return self.candidate_symbols
    
    async def start(self):
        """Start autonomous trading"""
        logger.info("=" * 60)
        logger.info("🤖 AUTONOMOUS TRADER STARTING")
        logger.info("=" * 60)
        
        self._running = True
        
        # Get symbols - from Universe Service OR ENV
        if self.use_universe_os:
            logger.info("[AutonomousTrader] 🌐 UNIVERSE OS ENABLED - Fetching dynamic symbols...")
            candidate_symbols = await self._get_universe_symbols()
        else:
            logger.info("[AutonomousTrader] 📋 Using hardcoded ENV symbols")
            candidate_symbols = self.candidate_symbols
        
        # Filter symbols by funding rate BEFORE starting
        logger.info(f"[AutonomousTrader] Filtering {len(candidate_symbols)} symbols by funding rate...")
        safe_symbols = await get_filtered_symbols(candidate_symbols)
        
        if len(safe_symbols) < len(candidate_symbols):
            removed = len(candidate_symbols) - len(safe_symbols)
            logger.warning(f"[AutonomousTrader] 🛡️ Removed {removed} high-funding symbols")
        
        logger.info(f"[AutonomousTrader] Trading with {len(safe_symbols)} safe symbols")
        
        # Initialize EntryScanner with filtered symbols
        self.entry_scanner = EntryScanner(
            self.redis,
            min_confidence=self.min_confidence,
            max_positions=self.max_positions,
            symbols=safe_symbols
        )
        
        # Start position tracker background task
        await self.position_tracker.start()
        
        # Wait for initial position sync (prevent race condition)
        logger.info("[AutonomousTrader] Waiting for position sync...")
        await asyncio.sleep(3)
        positions = self.position_tracker.get_all_positions()
        logger.info(f"[AutonomousTrader] Position sync complete: {len(positions)} positions loaded")
        
        # Start main loop
        self._main_loop_task = asyncio.create_task(self._main_loop())
        
        logger.info("✅ Autonomous Trader STARTED")
    
    async def stop(self):
        """Stop autonomous trading"""
        logger.info("[AutonomousTrader] Stopping...")
        
        self._running = False
        
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        
        await self.position_tracker.stop()
        await self.exit_manager.close()
        await self.redis.close()
        
        logger.info("✅ Autonomous Trader STOPPED")
    
    async def _main_loop(self):
        """
        Main autonomous trading cycle
        
        Every 30 seconds:
        1. Monitor positions for exits
        2. Scan for new entries
        3. Execute decisions
        """
        logger.info("[AutonomousTrader] Main loop started")
        
        while self._running:
            try:
                self.cycle_count += 1
                cycle_start = time.time()
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 CYCLE #{self.cycle_count}")
                logger.info(f"{'='*60}")
                
                # Step 1: Monitor active positions
                await self._monitor_positions()
                
                # Step 2: Scan for entries
                await self._scan_entries()
                
                # Statistics
                cycle_duration = time.time() - cycle_start
                logger.info(f"✅ Cycle completed in {cycle_duration:.2f}s")
                logger.info(f"📊 Stats: {self.entries_executed} entries, {self.exits_executed} exits")
                
                # Wait for next cycle
                await asyncio.sleep(self.scan_interval_sec)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AutonomousTrader] Cycle error: {e}", exc_info=True)
                await asyncio.sleep(self.scan_interval_sec)
    
    async def _is_exit_ownership_suspended(self) -> bool:
        """PATCH-2 kill-switch reader.

        Returns True  if ``quantum:exit_agent:active_flag`` exists in Redis
                      (any non-None value), meaning exit_management_agent has
                      taken ownership and AutonomousTrader must stand down.
        Returns False if the key is absent OR if the Redis read raises.
                      Fail-open: preserve existing exit behaviour on Redis failure.
        """
        try:
            value = await self.redis.get("quantum:exit_agent:active_flag")
            return value is not None
        except Exception as exc:
            logger.warning(
                "[AutonomousTrader] KILL_SWITCH_READ_ERROR patch=PATCH-2 — "
                f"fail-open, continuing normal exit ownership: {exc}"
            )
            return False

    async def _monitor_positions(self):
        """Monitor active positions and execute exits"""
        positions = self.position_tracker.get_all_positions()

        if not positions:
            logger.info("[Monitor] No active positions")
            return

        # PATCH-2: exit-ownership kill-switch.
        # If quantum:exit_agent:active_flag exists, exit_management_agent owns
        # exit decisions — stand down and skip normal evaluation entirely.
        if await self._is_exit_ownership_suspended():
            logger.warning(
                "[AutonomousTrader] EXIT_OWNERSHIP_SUSPENDED "
                "patch=PATCH-2 flag=quantum:exit_agent:active_flag — "
                "skipping normal exit evaluation this cycle"
            )
            return

        logger.info(f"[Monitor] Checking {len(positions)} positions...")
        
        for position in positions:
            try:
                # Evaluate exit
                decision = await self.exit_manager.evaluate_position(position)
                
                logger.info(
                    f"  {position.symbol}: {decision.action} "
                    f"({decision.percentage:.0%}) "
                    f"R={position.R_net:.2f} "
                    f"PnL=${position.pnl_usd:.2f} "
                    f"hold={decision.hold_score} exit={decision.exit_score}"
                )
                
                # Execute exit if needed
                if decision.action in ["PARTIAL_CLOSE", "CLOSE"]:
                    await self._execute_exit(position, decision)
            
            except Exception as e:
                logger.error(f"[Monitor] Error evaluating {position.symbol}: {e}")
    
    async def _scan_entries(self):
        """Scan for entry opportunities and execute"""
        # Safety check - entry_scanner initialized in start()
        if self.entry_scanner is None:
            logger.error("[Scanner] EntryScanner not initialized - skipping scan")
            return
        
        positions = self.position_tracker.get_all_positions()
        current_count = len(positions)

        # ── SLOT_FIX: authoritative Redis-scan position count ─────────────────
        # Legacy count reads position_tracker in-memory dict (stream-sourced).
        # Authoritative count scans quantum:position:{SYMBOL} hashes directly,
        # filtering only non-zero quantity keys and skipping sub-namespaces.
        # Falls back to legacy_count on any error — fail-safe.
        legacy_count = current_count
        try:
            authoritative_count = await get_authoritative_open_positions(self.redis)
            logger.info(f"[SLOT_FIX] authoritative_count={authoritative_count}")
            logger.info(f"[SLOT_FIX] legacy_count={legacy_count}")
            logger.info(f"[SLOT_FIX] delta={legacy_count - authoritative_count}")
            current_count = authoritative_count
            logger.info(f"[SLOT_FIX] Using authoritative position count: {current_count}")
        except Exception as _slot_fix_err:
            logger.warning(f"[SLOT_FIX] Fallback to legacy slot logic — {_slot_fix_err}")
            current_count = legacy_count
        # ── END SLOT_FIX ──────────────────────────────────────────────────────

        if current_count >= self.max_positions:
            logger.info(f"[Scanner] Max positions reached ({current_count}/{self.max_positions})")
            return
        
        # Get current exposure
        current_exposure = self.position_tracker.get_total_exposure_usd()
        
        if current_exposure >= self.max_exposure_usd:
            logger.info(f"[Scanner] Max exposure reached (${current_exposure:.0f}/${self.max_exposure_usd:.0f})")
            return
        
        logger.info(f"[Scanner] Scanning for entries (slots: {self.max_positions - current_count})...")
        
        # Scan
        opportunities = await self.entry_scanner.scan_for_entries(
            current_position_count=current_count,
            max_exposure_usd=self.max_exposure_usd
        )
        
        if not opportunities:
            logger.info("[Scanner] No entry opportunities found")
            return
        
        # Execute entries
        for opp in opportunities:
            try:
                # Check if already have position
                if self.position_tracker.has_position(opp.symbol):
                    logger.info(f"  {opp.symbol}: SKIP (already in position)")
                    continue
                
                # Calculate position size using RL agent
                sizing = await self._calculate_position_size(opp)
                
                logger.info(
                    f"  {opp.symbol}: ENTRY {opp.side} "
                    f"conf={opp.confidence:.2f} "
                    f"size=${sizing['position_usd']:.2f} "
                    f"lev={sizing['leverage']:.1f}x"
                )
                
                # Execute entry
                await self._execute_entry(opp, sizing)
            
            except Exception as e:
                logger.error(f"[Scanner] Error executing entry {opp.symbol}: {e}")
    
    async def _calculate_position_size(self, opportunity: EntryOpportunity) -> Dict:
        """
        Calculate position size using RL agent
        
        Returns:
            Dict with: position_usd, leverage, tp_pct, sl_pct
        """
        try:
            # Build state vector for RL agent
            state = {
                "confidence": opportunity.confidence,
                "volatility": opportunity.volatility,
                "pnl_trend": 0.0,  # No history for new position
                "exch_divergence": 0.0,  # Not using yet
                "funding_rate": 0.0,  # Not using yet
                "margin_util": 0.0  # Not using yet
            }
            
            # Get RL sizing multiplier (returns float in [0.5, 1.5])
            # NOTE: get_position_size_multiplier() replaces the old get_action()
            #       which no longer exists on RLPositionSizingAgent.
            multiplier = self.rl_agent.get_position_size_multiplier(state)
            
            # Default sizing if RL fails
            position_usd = min(self.max_position_usd, 300.0)
            leverage = 2.0
            tp_pct = 2.0
            sl_pct = 1.0
            
            if multiplier:
                # Apply RL multiplier to base position size
                position_usd = min(self.max_position_usd, 300.0 * multiplier)
            
            return {
                "position_usd": position_usd,
                "leverage": leverage,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct
            }
        
        except Exception as e:
            logger.error(f"[AutonomousTrader] Sizing error: {e}")
            # Safe defaults
            return {
                "position_usd": 300.0,
                "leverage": 2.0,
                "tp_pct": 2.0,
                "sl_pct": 1.0
            }
    
    async def _execute_entry(
        self,
        opportunity: EntryOpportunity,
        sizing: Dict
    ):
        """Publish entry intent to Redis"""
        try:
            intent = {
                "intent_type": "AUTONOMOUS_ENTRY",
                "symbol": opportunity.symbol,
                "action": "BUY" if opportunity.side == "LONG" else "SELL",
                "side": opportunity.side,
                "position_usd": str(sizing["position_usd"]),
                "leverage": str(sizing["leverage"]),
                "tp_pct": str(sizing["tp_pct"]),
                "sl_pct": str(sizing["sl_pct"]),
                "confidence": str(opportunity.confidence),
                "regime": opportunity.regime,
                "reason": opportunity.reason,
                "reduceOnly": "false",  # 🔥 FIX: Entry intents open NEW positions (not close)
                "timestamp": str(int(time.time()))
            }
            
            await self.redis.xadd(self.stream_entry_intent, intent)

            self.entries_executed += 1
            logger.info(f"✅ Entry intent published: {opportunity.symbol} → {self.stream_entry_intent}")
        
        except Exception as e:
            logger.error(f"[AutonomousTrader] Execute entry error: {e}", exc_info=True)
    
    async def _execute_exit(
        self,
        position: Position,
        decision: ExitDecision
    ):
        """Publish exit intent to Redis"""
        try:
            intent = {
                "intent_type": "AUTONOMOUS_EXIT",
                "symbol": position.symbol,
                "action": decision.action,
                "percentage": str(decision.percentage),
                "reason": decision.reason,
                "hold_score": str(decision.hold_score),
                "exit_score": str(decision.exit_score),
                "R_net": str(position.R_net),
                "pnl_usd": str(position.pnl_usd),
                "entry_price": str(position.entry_price),  # For CLM
                "exit_price": str(position.current_price),  # For CLM
                "timestamp": str(int(time.time()))
            }
            
            await self.redis.xadd("quantum:stream:harvest.intent", intent)
            
            self.exits_executed += 1
            logger.info(f"✅ Exit intent published: {position.symbol} {decision.action}")
        
        except Exception as e:
            logger.error(f"[AutonomousTrader] Execute exit error: {e}", exc_info=True)


async def main():
    """Main entry point"""
    trader = AutonomousTrader()
    
    try:
        await trader.start()
        
        # Run indefinitely
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Shutdown signal received")
    finally:
        await trader.stop()


if __name__ == "__main__":
    asyncio.run(main())
