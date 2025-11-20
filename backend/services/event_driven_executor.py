"""
Event-driven execution engine: AI continuously monitors market and trades
when it detects strong signals, without fixed time intervals.
"""
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.services.ai_trading_engine import AITradingEngine
from backend.services.execution import run_portfolio_rebalance, build_execution_adapter
from backend.config.execution import load_execution_config
from backend.config.risk import load_risk_config
from backend.database import SessionLocal

logger = logging.getLogger(__name__)


class EventDrivenExecutor:
    """
    Continuously monitors market signals and executes trades when AI
    detects high-confidence opportunities, regardless of time.
    """

    def __init__(
        self,
        ai_engine: AITradingEngine,
        symbols: List[str],
        confidence_threshold: float = 0.72,
        check_interval_seconds: int = 30,
        cooldown_seconds: int = 300,
    ):
        self.ai_engine = ai_engine
        self.symbols = symbols
        self.confidence_threshold = confidence_threshold
        self.check_interval = check_interval_seconds
        self.cooldown = cooldown_seconds
        
        # OPTIMIZATION: Enable direct execution to bypass slow rebalancing
        self._direct_execute = os.getenv("QT_EVENT_DIRECT_EXECUTE", "1") == "1"
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_rebalance_time: Optional[datetime] = None
        
        # Load risk config for position sizing
        self._risk_config = load_risk_config()
        self._execution_config = load_execution_config()
        self._adapter = None
        
        logger.info(
            "Event-driven executor initialized: %d symbols, confidence >= %.2f, "
            "check every %ds, cooldown %ds",
            len(symbols), confidence_threshold, check_interval_seconds, cooldown_seconds
        )

    async def start(self):
        """Start the event-driven monitoring loop as a background task."""
        if self._running:
            logger.warning("Event-driven executor already running")
            return
        
        self._running = True
        # Create background task and immediately add it to a set to prevent garbage collection
        # This is the recommended asyncio pattern for long-lived background tasks
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._monitor_loop(), name="event-driven-monitor")
        logger.info("Event-driven trading mode active - monitoring market continuously")

    async def stop(self):
        """Stop the monitoring loop."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("EventDrivenExecutor stopped")

    async def _monitor_loop(self):
        """Main loop: continuously check market and execute on strong signals."""
        logger.info("Monitoring loop started")
        try:
            while self._running:
                try:
                    logger.info("Checking %d symbols for signals >= %.2f threshold", len(self.symbols), self.confidence_threshold)
                    await self._check_and_execute()
                    logger.info("Check complete, sleeping %d seconds", self.check_interval)
                except asyncio.CancelledError:
                    logger.info("Monitoring loop cancelled")
                    raise  # Re-raise to exit properly
                except Exception as e:
                    logger.error("Error in event-driven monitoring: %s", e, exc_info=True)
                
                # Wait before next check
                try:
                    await asyncio.sleep(self.check_interval)
                except asyncio.CancelledError:
                    logger.info("Sleep cancelled, exiting loop")
                    raise  # Re-raise to exit properly
        except asyncio.CancelledError:
            logger.info("Monitor loop task cancelled")
        except Exception as e:
            logger.error("FATAL error in monitor loop: %s", e, exc_info=True)
        finally:
            logger.info("Monitoring loop ended")

    async def _check_and_execute(self):
        """
        Check AI signals and trigger portfolio rebalancing if strong signals detected.
        """
        now = datetime.now(timezone.utc)
        
        # Enforce cooldown between rebalances
        if self._last_rebalance_time:
            time_since_last = (now - self._last_rebalance_time).total_seconds()
            if time_since_last < self.cooldown:
                return
        
        # Get AI signals for all symbols
        try:
            # get_trading_signals expects (symbols, current_positions)
            # For monitoring, we pass empty positions since we check portfolio during rebalance
            signals_list = await self.ai_engine.get_trading_signals(self.symbols, {})
            logger.info("Got %d AI signals from engine", len(signals_list))
        except Exception as e:
            logger.error("Failed to get AI signals: %s", e, exc_info=True)
            return
        
            # Check for high-confidence signals
        strong_signals = []
        for signal in signals_list:
            symbol = signal.get("symbol", "")
            confidence = abs(signal.get("confidence", 0.0))
            action = signal.get("action", "HOLD")
            model = signal.get("model", "unknown")
            
            # ✅ ACCEPT ALL SIGNAL SOURCES including rule_fallback_rsi
            # ML models (XGBoost, TFT) and rule-based signals are both valid
            # Confidence threshold filters weak signals regardless of source
            if action != "HOLD" and confidence >= self.confidence_threshold:
                strong_signals.append((symbol, action, confidence, model))
        
        logger.info("Found %d high-confidence signals (>= %.2f)", len(strong_signals), self.confidence_threshold)
        
        if not strong_signals:
            # Provide a quick snapshot of the best BUY/SELL confidences observed
            best_buy = 0.0
            best_sell = 0.0
            for s in signals_list:
                act = s.get("action", "HOLD")
                conf = abs(float(s.get("confidence", 0.0)))
                if act == "BUY":
                    best_buy = max(best_buy, conf)
                elif act == "SELL":
                    best_sell = max(best_sell, conf)
            logger.debug(
                "No strong signals (thr=%.2f). Best BUY=%.2f, SELL=%.2f across %d symbols",
                self.confidence_threshold, best_buy, best_sell, len(signals_list)
            )
            return
        
        # Sort by confidence (highest first) and take top opportunities
        strong_signals.sort(key=lambda x: x[2], reverse=True)
        
        # Limit to top 5 simultaneous positions to manage risk
        max_positions = 5
        top_signals = strong_signals[:max_positions]
        
        if len(strong_signals) > max_positions:
            logger.info(
                "📊 Found %d strong signals, selecting top %d by confidence",
                len(strong_signals), max_positions
            )
        
        # Strong signals detected - execute orders
        logger.info(
            "🎯 Strong signals: %s",
            ", ".join(f"{sym}={act}({conf:.2f},{mdl})" for sym, act, conf, mdl in top_signals)
        )
        
        # OPTIMIZATION: Use direct execution for faster order placement
        if self._direct_execute:
            logger.info("⚡ Direct execution mode - placing orders immediately")
            result = await self._execute_signals_direct(top_signals)
        else:
            # Fallback to traditional rebalancing (slower)
            logger.info("🔄 Using portfolio rebalancing (legacy mode)")
            with SessionLocal() as db:
                result = await run_portfolio_rebalance(db)
        
        # Log raw result for debugging
        logger.info("🔍 Execution result: %s", result)
        
        # Check for success (status can be "ok" or "success")
        status = result.get("status", "")
        if status in ("ok", "success"):
            # ✅ ALWAYS update cooldown, even if no orders submitted
            self._last_rebalance_time = now
            num_orders = result.get("orders_submitted", 0)
            orders_planned = result.get("orders_planned", 0)
            orders_skipped = result.get("orders_skipped", 0)
            orders_failed = result.get("orders_failed", 0)
            
            if num_orders > 0:
                logger.info(
                    "✅ Execution complete: planned=%d submitted=%d skipped=%d failed=%d",
                    orders_planned, num_orders, orders_skipped, orders_failed
                )
            elif orders_skipped > 0:
                logger.warning(
                    "⚠️ All orders skipped (planned=%d skipped=%d) - will retry in %ds",
                    orders_planned, orders_skipped, self.cooldown
                )
            else:
                logger.info("ℹ️ No orders to execute")
        else:
            error_msg = result.get("error", status or "unknown")
            logger.warning("⚠️ Execution failed: status=%s error=%s", status, error_msg)

    async def _execute_signals_direct(self, signals: List[tuple]) -> Dict[str, any]:
        """
        FAST PATH: Execute orders directly from signals without portfolio rebalancing.
        
        Args:
            signals: List of (symbol, action, confidence) tuples
            
        Returns:
            Dict with execution results
        """
        if self._adapter is None:
            self._adapter = build_execution_adapter(self._execution_config)
        
        try:
            # Get current positions to check available slots
            raw_positions = await self._adapter.get_positions()
            positions = {sym.upper(): abs(float(qty)) for sym, qty in raw_positions.items() if float(qty) != 0}
            open_positions = len(positions)
            max_positions = int(os.getenv("QT_MAX_POSITIONS", "4"))
            available_slots = max(0, max_positions - open_positions)
            
            logger.info(f"💼 Current positions: {open_positions}/{max_positions}, available: {available_slots}")
            
            if available_slots == 0:
                logger.warning("⚠️ Max positions reached, skipping new orders")
                return {
                    "status": "ok",
                    "orders_planned": len(signals),
                    "orders_submitted": 0,
                    "orders_skipped": len(signals),
                    "orders_failed": 0,
                    "reason": "max_positions_reached"
                }
            
            # Get account balance for position sizing
            cash = await self._adapter.get_cash_balance()
            max_notional = self._risk_config.max_notional_per_trade or 4000.0
            
            logger.info(f"💰 Cash: ${cash:.2f}, Max per trade: ${max_notional:.2f}")
            
            # Place orders for top signals (up to available slots)
            orders_to_place = signals[:available_slots]
            orders_submitted = 0
            orders_failed = 0
            orders_skipped = 0
            
            for symbol, action, confidence in orders_to_place:
                try:
                    # Get current price
                    ticker = await self._adapter.get_ticker(symbol)
                    if not ticker or "last" not in ticker:
                        logger.warning(f"⚠️ No price data for {symbol}, skipping")
                        orders_skipped += 1
                        continue
                    
                    price = float(ticker["last"])
                    
                    # LIQUIDITY CHECK: Skip low-volume symbols
                    # Market orders on low-liquidity coins result in partial fills
                    # Require minimum $100k notional to ensure full $4000 order fills
                    try:
                        from backend.api_bulletproof import binance_ohlcv
                        ohlcv_data = await binance_ohlcv(symbol=symbol, limit=1)
                        candles = ohlcv_data.get("candles", [])
                        if candles and len(candles) > 0:
                            last_candle = candles[-1] if isinstance(candles, list) else candles
                            volume_usdt = float(last_candle.get("volume_usdt", 0))
                            
                            # Skip if 24h volume < $1M (ensures good liquidity)
                            if volume_usdt < 1_000_000:
                                logger.warning(
                                    f"⚠️ Low liquidity for {symbol}: ${volume_usdt:.0f} 24h volume, "
                                    f"skipping to avoid partial fills"
                                )
                                orders_skipped += 1
                                continue
                            
                            logger.debug(f"✅ {symbol} liquidity OK: ${volume_usdt:.0f} 24h volume")
                    except Exception as e:
                        logger.debug(f"Could not check liquidity for {symbol}: {e}")
                        # Continue anyway if volume check fails
                    
                    # Check if we already have this position
                    if symbol.upper() in positions:
                        logger.info(f"⏭️ Already have {symbol} position, skipping")
                        orders_skipped += 1
                        continue
                    
                    # Calculate quantity
                    notional = min(max_notional, cash * 0.95)  # Use max 95% of cash per trade
                    quantity = notional / price
                    
                    # Determine side
                    side = "buy" if action == "BUY" else "sell"
                    
                    logger.info(
                        f"📤 Placing {side.upper()} order: {symbol} qty={quantity:.4f} @ ${price:.4f} "
                        f"(notional=${notional:.2f}, conf={confidence:.2%})"
                    )
                    
                    # Submit market order (use current price as limit for immediate fill)
                    order_id = await self._adapter.submit_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price  # Market-like execution with current price
                    )
                    
                    logger.info(f"✅ Order placed: {symbol} {side.upper()} - ID: {order_id}")
                    orders_submitted += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to place order for {symbol}: {e}", exc_info=True)
                    orders_failed += 1
            
            return {
                "status": "ok",
                "orders_planned": len(orders_to_place),
                "orders_submitted": orders_submitted,
                "orders_skipped": orders_skipped,
                "orders_failed": orders_failed
            }
            
        except Exception as e:
            logger.error(f"❌ Direct execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "orders_planned": 0,
                "orders_submitted": 0,
                "orders_skipped": 0,
                "orders_failed": 0
            }


# Global instance (initialized at startup)
_executor: Optional[EventDrivenExecutor] = None


async def start_event_driven_executor(
    ai_engine: AITradingEngine,
    symbols: List[str],
    confidence_threshold: float = 0.65,
    check_interval: int = 30,
    cooldown: int = 300,
) -> "EventDrivenExecutor":
    """Start the global event-driven executor. Returns executor instance.
    
    IMPORTANT: The executor starts a background task. The caller MUST keep
    a reference to executor._task to prevent garbage collection!
    """
    global _executor
    
    if _executor is not None:
        logger.warning("Event-driven executor already running")
        return _executor
    
    _executor = EventDrivenExecutor(
        ai_engine=ai_engine,
        symbols=symbols,
        confidence_threshold=confidence_threshold,
        check_interval_seconds=check_interval,
        cooldown_seconds=cooldown,
    )
    
    await _executor.start()
    
    # ⚠️ CRITICAL: Verify task was created and is running
    if _executor._task is None or _executor._task.done():
        logger.error("❌ CRITICAL: Event-driven executor task failed to start!")
        raise RuntimeError("Event-driven executor task not running")
    else:
        logger.info("✅ Event-driven executor task confirmed running: %s", _executor._task.get_name())
    
    return _executor


async def stop_event_driven_executor():
    """Stop the global event-driven executor."""
    global _executor
    
    if _executor is None:
        return
    
    await _executor.stop()
    _executor = None


def is_event_driven_active() -> bool:
    """Check if event-driven executor is running."""
    return _executor is not None and _executor._running
