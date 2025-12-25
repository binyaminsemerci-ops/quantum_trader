"""
Position Monitor - Automatic TP/SL Management
Monitors all open positions and ensures they have TP/SL protection
"""
import os
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List
from binance.client import Client
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# [NEW] TRADESTORE - SPRINT 1 D5: Import for trade persistence
try:
    from backend.core.trading import get_trade_store, TradeStatus
    TRADESTORE_AVAILABLE = True
    logger_tradestore = logging.getLogger(__name__ + ".tradestore")
    logger_tradestore.info("[OK] TradeStore available in PositionMonitor")
except ImportError as e:
    TRADESTORE_AVAILABLE = False
    logger_tradestore = logging.getLogger(__name__ + ".tradestore")
    logger_tradestore.warning(f"[WARNING] TradeStore not available: {e}")

# [NEW] SPRINT 1 - D6: Binance Rate Limiter Integration
try:
    from backend.integrations.binance import BinanceClientWrapper, create_binance_wrapper
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False

# Always configure a dedicated logger for rate limiter messages
logger_rate_limiter = logging.getLogger(__name__ + ".rate_limiter")

# [PHASE 1] Exit Order Gateway for observability
try:
    from backend.services.execution.exit_order_gateway import submit_exit_order
    EXIT_GATEWAY_AVAILABLE = True
except ImportError:
    EXIT_GATEWAY_AVAILABLE = False
    logger.warning("[EXIT_GATEWAY] Not available - will place orders directly")
    logger_rate_limiter = logging.getLogger(__name__ + ".rate_limiter")
    logger_rate_limiter.info("[OK] Binance rate limiter available in PositionMonitor")
except ImportError as e:
    RATE_LIMITER_AVAILABLE = False
    logger_rate_limiter = logging.getLogger(__name__ + ".rate_limiter")
    logger_rate_limiter.warning(f"[WARNING] Binance rate limiter not available: {e}")

# [TARGET] CRITICAL: Read Dynamic TP/SL from event_driven_executor
TPSL_STORAGE_PATH = Path("/app/tmp/quantum_tpsl.json")

# [NEW] TP v3 Enhancements: Dynamic trailing and performance tracking
try:
    from backend.services.monitoring.dynamic_trailing_rearm import DynamicTrailingManager
    from backend.services.monitoring.tp_performance_tracker import get_tp_performance_tracker
    DYNAMIC_TRAILING_AVAILABLE = True
    logger_trailing = logging.getLogger(__name__ + ".dynamic_trailing")
    logger_trailing.info("[OK] Dynamic Trailing Rearm available")
except ImportError as e:
    DYNAMIC_TRAILING_AVAILABLE = False
    logger_trailing = logging.getLogger(__name__ + ".dynamic_trailing")
    logger_trailing.warning(f"[WARNING] Dynamic Trailing not available: {e}")

# [NEW] EXIT BRAIN V3: Unified exit orchestrator
from backend.config.exit_mode import get_exit_mode, get_exit_executor_mode, is_exit_brain_live_fully_enabled
EXIT_MODE = get_exit_mode()
EXIT_BRAIN_V3_ENABLED = (EXIT_MODE == "EXIT_BRAIN_V3" or os.getenv("EXIT_BRAIN_V3_ENABLED", "false").lower() == "true")
try:
    from backend.domains.exits.exit_brain_v3 import ExitBrainV3
    from backend.domains.exits.exit_brain_v3.router import ExitRouter
    from backend.domains.exits.exit_brain_v3.integration import to_dynamic_tpsl, to_trailing_config
    from backend.domains.exits.exit_brain_v3.dynamic_executor import ExitBrainDynamicExecutor
    from backend.domains.exits.exit_brain_v3.adapter import ExitBrainAdapter
    EXIT_BRAIN_V3_AVAILABLE = True
    logger_exit_brain = logging.getLogger(__name__ + ".exit_brain_v3")
    logger_exit_brain.info(f"[OK] Exit Brain v3 available (mode={EXIT_MODE}, enabled={EXIT_BRAIN_V3_ENABLED})")
except ImportError as e:
    EXIT_BRAIN_V3_AVAILABLE = False
    EXIT_BRAIN_V3_ENABLED = False
    logger_exit_brain = logging.getLogger(__name__ + ".exit_brain_v3")
    logger_exit_brain.warning(f"[WARNING] Exit Brain v3 not available: {e}")

# [NEW] EXIT BRAIN V3.5: Next-gen exit orchestrator with ILFv2
EXIT_BRAIN_V35_ENABLED = os.getenv("EXIT_BRAIN_V35_ENABLED", "false").lower() == "true"
try:
    import sys
    sys.path.insert(0, '/app/microservices')
    from exitbrain_v3_5.exit_brain import ExitBrainV35, SignalContext
    EXIT_BRAIN_V35_AVAILABLE = True
    logger_exit_brain_v35 = logging.getLogger(__name__ + ".exit_brain_v35")
    logger_exit_brain_v35.info(f"✅ [OK] Exit Brain v3.5 available (enabled={EXIT_BRAIN_V35_ENABLED})")
except ImportError as e:
    EXIT_BRAIN_V35_AVAILABLE = False
    EXIT_BRAIN_V35_ENABLED = False
    logger_exit_brain_v35 = logging.getLogger(__name__ + ".exit_brain_v35")
    logger_exit_brain_v35.warning(f"⚠️ [WARNING] Exit Brain v3.5 not available: {e}")


class PositionMonitor:
    """
    Continuously monitors all open positions and ensures TP/SL protection.
    
    Features:
    - Detects positions without TP/SL orders
    - Automatically sets hybrid TP/SL strategy (partial exit + trailing)
    - Uses AI-generated TP/SL percentages when available
    - Runs independently from trade execution
    - [ALERT] FIX #3: Re-evaluates AI sentiment and warns if changed
    """
    
    def __init__(
        self,
        check_interval: int = 10,  # [TARGET] Check every 10 seconds for DYNAMIC TP/SL adjustment!
        ai_engine=None,  # [ALERT] NEW: Accept AI engine for re-evaluation
        app_state=None,  # [NEW] HEDGEFUND MODE: Accept app_state for SafetyGovernor directives
        event_bus=None,  # [CRITICAL FIX #3] Accept EventBus for model.promoted events
    ):
        self.check_interval = check_interval
        self.ai_engine = ai_engine  # [ALERT] NEW: Store AI engine reference
        self.app_state = app_state  # [NEW] Store app_state for SafetyGovernor access
        self.event_bus = event_bus  # [CRITICAL FIX #3] Store EventBus reference
        
        # [NEW] TRADESTORE - SPRINT 1 D5: Initialize trade persistence
        self.trade_store = None
        if TRADESTORE_AVAILABLE:
            # TradeStore will be initialized async in monitor_loop
            logger_tradestore.info("[OK] TradeStore will be initialized on monitor_loop start")
        
        # [CRITICAL FIX #3] Track model loading timestamp
        self.models_loaded_at = datetime.now(timezone.utc)
        
        # [FIX] Track which positions we've already set TP/SL for
        # This prevents re-setting orders every 10 seconds when Binance API doesn't return conditional orders
        self._protected_positions = {}  # {symbol: timestamp_when_protected}
        
        # [CRITICAL FIX #3] Subscribe to model promotion events
        if self.event_bus:
            self.event_bus.subscribe("model.promoted", self._handle_model_promotion)
            logger.info("[OK] Position Monitor subscribed to model.promoted events")
        
        # [NEW] META-STRATEGY SELECTOR: Load for reward updates
        try:
            from backend.services.meta_strategy_integration import get_meta_strategy_integration
            meta_integration = get_meta_strategy_integration()
            if meta_integration and meta_integration.enabled:
                self.meta_strategy = meta_integration
                logger.info("[OK] Meta-Strategy Selector loaded for reward updates")
            else:
                self.meta_strategy = None
                logger.info("[INFO] Meta-Strategy Selector disabled")
        except Exception as e:
            logger.warning(f"[WARNING] Meta-Strategy Selector not available: {e}")
            self.meta_strategy = None
        
        # Binance client - with testnet support
        # Check STAGING_MODE or BINANCE_TESTNET (same as execution adapter)
        use_testnet = os.getenv("STAGING_MODE", "false").lower() == "true" or os.getenv("BINANCE_TESTNET", "false").lower() == "true"
        
        if use_testnet:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            if not api_key or not api_secret:
                raise ValueError("Missing Binance TESTNET credentials (BINANCE_API_KEY, BINANCE_API_SECRET)")
            logger.info("[TEST_TUBE] Position Monitor: Using Binance Testnet API")
            self.client = Client(api_key, api_secret, testnet=True)
            self.client.API_URL = 'https://testnet.binancefuture.com'
        else:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            if not api_key or not api_secret:
                raise ValueError("Missing Binance credentials")
            logger.info("[RED_CIRCLE] Position Monitor: Using Binance Live API")
            self.client = Client(api_key, api_secret)
        
        # [NEW] SPRINT 1 - D6: Initialize Binance rate limiter wrapper
        if RATE_LIMITER_AVAILABLE:
            self._binance_wrapper = create_binance_wrapper()
            logger_rate_limiter.info("[OK] Binance rate limiter enabled for PositionMonitor")
        else:
            self._binance_wrapper = None
            logger_rate_limiter.warning("[WARNING] Binance rate limiter not available")
        
        # Configuration from environment
        self.tp_pct = float(os.getenv("QT_TP_PCT", "0.03"))
        self.sl_pct = float(os.getenv("QT_SL_PCT", "0.02"))
        self.trail_pct = float(os.getenv("QT_TRAIL_PCT", "0.015"))
        self.partial_tp = float(os.getenv("QT_PARTIAL_TP", "0.5"))
        
        # [FIX] CRITICAL: Get trail callback rate from config (must be [0.1, 5.0])
        self.trail_callback_rate = float(os.getenv("QT_TRAIL_CALLBACK_RATE", "1.5"))
        
        # [FIX] CRITICAL: Detect position mode (One-Way vs Hedge Mode)
        # In One-Way Mode: positionSide must be 'BOTH' or omitted
        # In Hedge Mode: positionSide must be 'LONG' or 'SHORT'
        self._position_mode = None  # Will be detected on first check
        self._is_hedge_mode = False
        
        # Track open positions to detect closures (for Meta-Strategy reward updates)
        self._previous_open_positions = {}  # {symbol: position_data}
        
        # [CRITICAL FIX #1] Flash crash detection - track previous equity for rapid drawdown monitoring
        self._previous_equity = None
        self._flash_crash_threshold_pct = -0.03  # -3% in single check = flash crash
        
        # [NEW] TP v3: Initialize Dynamic Trailing Manager and Performance Tracker
        self.trailing_manager = None
        self.tp_tracker = None
        if DYNAMIC_TRAILING_AVAILABLE:
            self.trailing_manager = DynamicTrailingManager()
            self.tp_tracker = get_tp_performance_tracker()
            logger_trailing.info("[OK] Dynamic Trailing Manager initialized")
        
        # [NEW] EXIT BRAIN V3: Initialize router and executor
        self.exit_router = None
        self.exit_brain_executor = None
        if EXIT_BRAIN_V3_ENABLED and EXIT_BRAIN_V3_AVAILABLE:
            self.exit_router = ExitRouter()
            logger_exit_brain.info("[OK] Exit Router initialized - Exit Brain v3 ACTIVE")
        
        # [NEW] EXIT BRAIN V3.5: Initialize next-gen exit brain with ILFv2
        self.exit_brain_v35 = None
        if EXIT_BRAIN_V35_ENABLED and EXIT_BRAIN_V35_AVAILABLE:
            try:
                self.exit_brain_v35 = ExitBrainV35()
                logger_exit_brain_v35.info("✅ [OK] Exit Brain v3.5 initialized with Intelligent Leverage Framework v2")
                if EXIT_BRAIN_V3_ENABLED:
                    logger_exit_brain_v35.warning("⚠️ [DUAL MODE] Both Exit Brain v3 and v3.5 enabled - v3.5 will override v3 decisions")
            except Exception as e:
                logger_exit_brain_v35.error(f"❌ [ERROR] Failed to initialize Exit Brain v3.5: {e}")
                self.exit_brain_v35 = None
            
            # Initialize Dynamic Executor for active TP/SL monitoring
            try:
                planner = ExitBrainV3()
                adapter = ExitBrainAdapter(planner=planner)
                
                # Create exit order gateway wrapper
                class ExitOrderGatewayWrapper:
                    def __init__(self, client):
                        self.client = client
                    
                    async def submit_exit_order(self, **kwargs):
                        if 'client' not in kwargs:
                            kwargs['client'] = self.client
                        if EXIT_GATEWAY_AVAILABLE:
                            return await submit_exit_order(**kwargs)
                        else:
                            # Fallback: direct API call
                            order_params = kwargs.get('order_params', {})
                            return self.client.futures_create_order(**order_params)
                
                exit_gateway = ExitOrderGatewayWrapper(self.client)
                loop_interval = float(os.getenv("EXIT_BRAIN_CHECK_INTERVAL_SEC", "10"))
                
                self.exit_brain_executor = ExitBrainDynamicExecutor(
                    adapter=adapter,
                    exit_order_gateway=exit_gateway,
                    position_source=self.client,
                    loop_interval_sec=loop_interval,
                    shadow_mode=False
                )
                
                logger_exit_brain.warning(
                    f"[EXIT_BRAIN_V3] ✅ Dynamic Executor INITIALIZED "
                    f"(interval={loop_interval}s, will start with monitor loop)"
                )
            except Exception as e:
                logger_exit_brain.error(f"[EXIT_BRAIN_V3] ❌ Failed to initialize Dynamic Executor: {e}", exc_info=True)
                self.exit_brain_executor = None
        
        logger.info(f"[SEARCH] Position Monitor initialized: TP={self.tp_pct*100:.1f}% SL={self.sl_pct*100:.1f}% Trail={self.trail_pct*100:.1f}% TrailCallback={self.trail_callback_rate:.1f}%")
        logger.info(f"[FLASH-CRASH] Real-time drawdown monitor active (threshold: {self._flash_crash_threshold_pct*100:.1f}%)")
        if EXIT_BRAIN_V3_ENABLED:
            logger.info(f"[EXIT BRAIN] Exit Brain v3 ENABLED - TP/SL orchestration active")
    
    async def _call_binance_api(self, method_name: str, *args, **kwargs):
        """
        Wrapper for all Binance API calls with rate limiting.
        
        [NEW] SPRINT 1 - D6: Rate limiter integration
        
        Args:
            method_name: Name of client method to call (e.g., 'futures_position_information')
            *args, **kwargs: Arguments to pass to method
        
        Returns:
            Result from Binance API call
        """
        # Get the actual method from client
        method = getattr(self.client, method_name)
        
        # Wrap with rate limiter if available
        if self._binance_wrapper and RATE_LIMITER_AVAILABLE:
            return await self._binance_wrapper.call_async(method, *args, **kwargs)
        else:
            # Fallback: call directly (sync method, run in executor)
            import asyncio
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: method(*args, **kwargs))
    
    def _cancel_all_orders_for_symbol(self, symbol: str) -> int:
        """
        Cancel ALL open orders for a symbol (comprehensive cleanup).
        Uses futures_cancel_all_open_orders to catch hidden/conditional orders.
        Returns number of orders cancelled.
        """
        cancelled_count = 0
        try:
            # METHOD 1: Use Binance's bulk cancel API (catches hidden orders)
            result = self.client.futures_cancel_all_open_orders(symbol=symbol)
            
            if result and result.get('code') == 200:
                logger.info(f"🗑️  Bulk cancelled all orders for {symbol} (including hidden)")
                # [CRITICAL] Wait for Binance backend to release trigger slots
                # Testnet needs ~3s, mainnet needs ~1-2s
                import time
                time.sleep(3.0)  # 3 second wait for trigger slot release
                return 1  # We don't know exact count, but assume at least 1
            
            # METHOD 2: Fallback to individual cancellation
            open_orders = self.client.futures_get_open_orders(symbol=symbol)
            if not open_orders:
                return 0
            
            logger.info(f"🗑️  Cancelling {len(open_orders)} visible orders for {symbol}")
            
            for order in open_orders:
                try:
                    order_type = order.get('type', 'UNKNOWN')
                    order_id = order.get('orderId')
                    self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
                    logger.info(f"   ✓ Cancelled {order_type} order {order_id}")
                    cancelled_count += 1
                except Exception as e:
                    logger.warning(f"   ✗ Failed to cancel order {order_id}: {e}")
            
            logger.info(f"[OK] Cancelled {cancelled_count}/{len(open_orders)} orders for {symbol}")
            
            # Wait for trigger slot release after individual cancels too
            if cancelled_count > 0:
                import time
                time.sleep(2.0)
                
        except Exception as e:
            # Check if error is "no orders to cancel" (which is OK)
            error_str = str(e)
            if 'No such open order' in error_str or 'Unknown order sent' in error_str:
                logger.debug(f"No orders to cancel for {symbol} (expected)")
                return 0
            logger.error(f"❌ Error cancelling orders for {symbol}: {e}")
        
        return cancelled_count
    
    async def _handle_model_promotion(self, event_data: dict) -> None:
        """
        Handle model.promoted event - reload models after promotion (CRITICAL FIX #3).
        
        Args:
            event_data: {
                "model_name": str,
                "model_type": str,
                "promoted_at": str,
                "performance_metrics": dict
            }
        """
        model_name = event_data.get("model_name", "unknown")
        model_type = event_data.get("model_type", "unknown")
        promoted_at = event_data.get("promoted_at")
        
        logger.warning(
            f"🔄 MODEL PROMOTION DETECTED: {model_name} ({model_type}) "
            f"promoted at {promoted_at} - reloading all models"
        )
        
        try:
            # Reload models in AI engine
            if self.ai_engine:
                await self._reload_models()
                self.models_loaded_at = datetime.now(timezone.utc)
                logger.info(
                    f"✅ Models reloaded successfully after {model_name} promotion "
                    f"(loaded_at={self.models_loaded_at.isoformat()})"
                )
            else:
                logger.warning("⚠️ AI engine not available - cannot reload models")
        
        except Exception as e:
            logger.error(f"❌ Failed to reload models after promotion: {e}")
    
    async def _reload_models(self) -> None:
        """
        Reload all ML models from disk (CRITICAL FIX #3).
        
        This ensures Position Monitor uses the latest promoted models
        for re-evaluation and sentiment checks.
        """
        if not self.ai_engine:
            logger.warning("AI engine not available for model reload")
            return
        
        try:
            # Check if AI engine has reload method
            if hasattr(self.ai_engine, 'reload_models'):
                await self.ai_engine.reload_models()
                logger.info("✅ AI engine models reloaded")
            elif hasattr(self.ai_engine, 'load_models'):
                await self.ai_engine.load_models()
                logger.info("✅ AI engine models loaded")
            else:
                logger.warning(
                    "AI engine does not have reload_models() or load_models() method - "
                    "skipping reload"
                )
        
        except Exception as e:
            logger.error(f"Failed to reload AI engine models: {e}")
            raise
    
    async def _update_meta_strategy_reward(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        side: str,
        pnl: float,
        duration_hours: float,
    ) -> None:
        """
        Update Meta-Strategy RL reward after position closes.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            exit_price: Exit price
            side: Position side ('LONG' or 'SHORT')
            pnl: Realized PnL
            duration_hours: Position duration in hours
        """
        if not self.meta_strategy or not self.meta_strategy.enabled:
            return
        
        try:
            # Get trade state to retrieve meta-strategy info
            from backend.services.execution.execution import TradeStateStore
            from pathlib import Path
            
            trade_state_path = Path("/app/data/trade_state.json")
            trade_store = TradeStateStore(trade_state_path)
            state = trade_store.get(symbol)
            
            if not state or "meta_strategy" not in state:
                logger.debug(f"[META-STRATEGY] No meta-strategy info for {symbol} - skipping reward update")
                return
            
            meta_info = state["meta_strategy"]
            stored_entry_price = meta_info.get("entry_price", entry_price)
            atr = meta_info.get("atr", stored_entry_price * 0.01)  # Fallback: 1% of price
            
            # Calculate realized R
            if side.upper() == "LONG":
                realized_r = (exit_price - stored_entry_price) / atr
            else:  # SHORT
                realized_r = (stored_entry_price - exit_price) / atr
            
            # Update RL reward
            await self.meta_strategy.update_strategy_reward(
                symbol=symbol,
                realized_r=realized_r,
                trade_meta={
                    "pnl": pnl,
                    "duration_hours": duration_hours,
                    "entry_price": stored_entry_price,
                    "exit_price": exit_price,
                    "side": side
                }
            )
            
            logger.info(
                f"[RL UPDATE] {symbol}: R={realized_r:+.2f} "
                f"(PnL=${pnl:+.2f}, Duration={duration_hours:.1f}h)"
            )
            
        except Exception as e:
            logger.error(f"[ERROR] Meta-Strategy reward update failed for {symbol}: {e}", exc_info=True)
    
    def _get_price_precision(self, symbol: str) -> int:
        """Get price precision for symbol"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                if price_filter:
                    tick_size = float(price_filter['tickSize'])
                    tick_str = f"{tick_size:.10f}".rstrip('0')
                    if '.' in tick_str:
                        return len(tick_str.split('.')[-1])
            return 5
        except:
            return 5
    
    def _get_quantity_precision(self, symbol: str) -> float:
        """Get quantity step size for symbol"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                lot_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if lot_filter:
                    return float(lot_filter['stepSize'])
            return 0.01
        except:
            return 0.01
    
    def _round_quantity(self, qty: float, step_size: float) -> float:
        """Round quantity to step size"""
        if step_size >= 1:
            return int(qty)
        step_str = f"{step_size:.10f}".rstrip('0')
        if '.' in step_str:
            precision = len(step_str.split('.')[-1])
        else:
            precision = 0
        return round(qty, precision)
    
    def _round_price(self, price: float, precision: int) -> float:
        """Round price to exact precision using Decimal for accuracy"""
        from decimal import Decimal, ROUND_DOWN
        if precision == 0:
            return int(price)
        # Use string formatting to ensure exact precision
        return float(f"{price:.{precision}f}")
    
    async def _adjust_tpsl_dynamically(self, position: Dict, ai_signals: List[Dict] = None) -> bool:
        """
        [TARGET] DYNAMISK TP/SL JUSTERING basert på profit, sentiment og markedsbevegelse
        
        [EXIT BRAIN V3] If enabled, delegates to Exit Brain orchestrator.
        Otherwise uses legacy dynamic adjustment logic.
        """
        symbol = position['symbol']
        amt = float(position['positionAmt'])
        entry_price = float(position['entryPrice'])
        mark_price = float(position.get('markPrice', entry_price))  # 🔧 Extract mark_price BEFORE using it
        
        # [EXIT BRAIN V3] ENABLED - Check if plan exists, create retroactively if missing
        if EXIT_BRAIN_V3_ENABLED and EXIT_BRAIN_V3_AVAILABLE and self.exit_router:
            # Check if Exit Brain plan exists for this position
            existing_plan = self.exit_router.get_active_plan(symbol)
            
            if not existing_plan:
                # Plan missing - position opened before Exit Brain was integrated or plan creation failed
                logger_exit_brain.warning(
                    f"[EXIT BRAIN V3] {symbol}: No exit plan found - creating retroactively"
                )
                try:
                    # Build minimal context for retroactive plan creation
                    rl_hints = {
                        "tp_target_pct": 0.03,  # Default 3% TP
                        "sl_target_pct": 0.02,  # Default 2% SL
                        "trail_callback_pct": 0.015,  # Default 1.5% trail
                        "confidence": 0.60,
                    }
                    
                    risk_context = {
                        "risk_mode": "NORMAL",
                        "daily_pnl_pct": 0.0,
                        "position_count": 1,
                        "max_positions": 10,
                    }
                    
                    market_data = {
                        "current_price": mark_price,
                        "volatility": 0.02,
                        "atr": mark_price * 0.01,
                        "regime": "NORMAL",
                    }
                    
                    # Create plan retroactively
                    plan = await self.exit_router.get_or_create_plan(
                        position=position,
                        rl_hints=rl_hints,
                        risk_context=risk_context,
                        market_data=market_data
                    )
                    
                    if plan and len(plan.legs) > 0:
                        logger_exit_brain.info(
                            f"[EXIT BRAIN V3] {symbol}: Retroactive plan created with {len(plan.legs)} legs"
                        )
                    else:
                        logger_exit_brain.error(
                            f"[EXIT BRAIN V3] {symbol}: Retroactive plan creation returned empty plan - using legacy logic"
                        )
                        # Fall through to legacy logic
                        
                except Exception as e:
                    logger_exit_brain.error(
                        f"[EXIT BRAIN V3] {symbol}: Retroactive plan creation failed: {e}",
                        exc_info=True
                    )
                    # Fall through to legacy logic
            else:
                logger_exit_brain.debug(
                    f"[EXIT BRAIN V3] {symbol}: Exit plan exists (strategy={existing_plan.strategy_id})"
                )
            
            # Always delegate to Exit Brain when enabled (plan now exists)
            logger_exit_brain.info(
                f"[EXIT BRAIN V3] {symbol}: Delegating TP/SL management to Exit Brain (profile-based)"
            )
            
            # [PHASE 2] Check if executor actually exists
            if hasattr(self, 'exit_brain_executor') and self.exit_brain_executor is not None:
                logger_exit_brain.info(
                    f"[EXIT_GUARD] ✅ Exit Brain Executor ACTIVE - {symbol} will be monitored "
                    f"for dynamic TP/SL execution every {self.exit_brain_executor.loop_interval_sec}s"
                )
            else:
                logger_exit_brain.warning(
                    f"[EXIT_GUARD] ⚠️  DELEGATION GAP: {symbol} delegated to Exit Brain v3, "
                    f"but Exit Brain Executor does NOT EXIST. Exit orders will NOT be placed. "
                    f"This is a known architecture gap - see AI_OS_INTEGRATION_SUMMARY.md Phase 2."
                )
            
            return False  # Don't adjust - Exit Brain will handle via executor
        
        # [LEGACY PATH] Original dynamic adjustment logic
        # [FIX] CRITICAL: Get LIVE mark price from market, NOT cached position data!
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            mark_price = float(ticker['price'])
            logger.debug(f"[LIVE] {symbol}: Mark price from market: ${mark_price:.6f}")
        except Exception as e:
            # Fallback to position markPrice only if live data fails
            mark_price = float(position['markPrice'])
            logger.warning(f"[STALE] {symbol}: Using cached mark price ${mark_price:.6f} - live data failed: {e}")
        
        # [FIX] CRITICAL: Recalculate unrealized PnL with LIVE price
        # For LONG: PnL = (mark_price - entry_price) * position_size
        # For SHORT: PnL = (entry_price - mark_price) * abs(position_size)
        is_long = amt > 0
        if is_long:
            unrealized_pnl = (mark_price - entry_price) * amt
        else:
            unrealized_pnl = (entry_price - mark_price) * abs(amt)
        
        leverage = float(position.get('leverage', 30))  # [TARGET] Using 30x leverage
        
        # Beregn profit i prosent av margin (ikke position value!)
        margin = abs(amt * entry_price) / leverage
        pnl_pct = (unrealized_pnl / margin) * 100 if margin > 0 else 0
        
        # Finn AI signal for dette symbolet
        ai_signal = None
        if ai_signals:
            ai_signal = next((s for s in ai_signals if s['symbol'] == symbol), None)
        
        # [TARGET] DYNAMISK LOGIKK FOR 30x LEVERAGE - Progressive partial profit taking:
        # Med 30x leverage beveger PnL seg MYE raskere!
        # Justerte thresholds for å matche høy leverage:
        # 1. I TAP: Hold original SL/TP
        # 2. Litt profit (5-10%): SL→breakeven, Partial TP #1 (25% @ 8%)
        # 3. God profit (10-20%): Lock 20%, Partial TP #2 (25% @ 12%)
        # 4. Veldig god (20-35%): Lock 40%, Partial TP #3 (25% @ 18%)
        # 5. Stor profit (35-60%): Lock 60%, Final TP (25% @ 30%+)
        # 6. Ekstrem (>60%): Lock 70%, Moon TP (remaining @ 50%)
        
        logger.debug(f"[CHART] {symbol}: PnL {pnl_pct:+.2f}% margin (+${unrealized_pnl:.2f} USDT), Mark ${mark_price:.6f}, Entry ${entry_price:.6f}")
        
        # Beregn ny SL OG multiple TP levels basert på profit
        new_sl_price = None
        partial_tp_levels = []  # List of (price, qty_pct, reason)
        adjustment_reason = ""
        
        if pnl_pct < -3.0:
            # I TAP mer enn 3% - hold original SL, men advare
            logger.warning(f"[WARNING] {symbol}: Losing {pnl_pct:.2f}% - holding SL/TP")
            if ai_signal and ai_signal.get('action') == 'HOLD' and ai_signal.get('confidence', 0) < 0.4:
                logger.warning(f"[ALERT] {symbol}: AI sentiment weak ({ai_signal.get('confidence', 0):.0%}) - consider closing!")
            return False
            
        elif -3.0 <= pnl_pct < 5.0:
            # Liten profit - la posisjonen vandre
            return False
            
        elif 5.0 <= pnl_pct < 10.0:
            # Moderat profit - move SL to breakeven (entry price) + small buffer for fees
            # [FIX] For LONG: SL moves UP to entry, for SHORT: SL moves DOWN to entry
            fee_buffer_pct = 0.001  # 0.1% buffer for trading fees
            if is_long:
                new_sl_price = entry_price * (1 + fee_buffer_pct)  # LONG: SL just above entry
            else:
                new_sl_price = entry_price * (1 - fee_buffer_pct)  # SHORT: SL just below entry
            # TP #1: 25% av position @ 8% margin profit
            tp1_pct = 0.08
            if is_long:
                tp1_price = entry_price * (1 + tp1_pct / leverage)
            else:
                tp1_price = entry_price * (1 - tp1_pct / leverage)
            partial_tp_levels.append((tp1_price, 0.25, "TP1@8%"))
            adjustment_reason = "lock profit: SL→breakeven+fees"
            
        elif 10.0 <= pnl_pct < 20.0:
            # God profit - trail SL to lock 50% of profit
            # [FIX] SL should follow price direction: UP for LONG, DOWN for SHORT
            profit_distance = abs(mark_price - entry_price)
            if is_long:
                new_sl_price = entry_price + (profit_distance * 0.50)  # LONG: SL moves UP
            else:
                new_sl_price = entry_price - (profit_distance * 0.50)  # SHORT: SL moves DOWN
            # TP #1: 25% @ 10%
            tp1_pct = 0.10
            tp1_price = entry_price * (1 + tp1_pct / leverage * (1 if is_long else -1))
            partial_tp_levels.append((tp1_price, 0.25, "TP1@10%"))
            # TP #2: 25% @ 15%
            tp2_pct = 0.15
            tp2_price = entry_price * (1 + tp2_pct / leverage * (1 if is_long else -1))
            partial_tp_levels.append((tp2_price, 0.25, "TP2@15%"))
            adjustment_reason = "lock 85%, partial TPs (25%+25%)"
            
        elif 20.0 <= pnl_pct < 35.0:
            # Veldig god - trail SL to lock 70% of profit
            # [FIX] SL trails in profit direction
            profit_distance = abs(mark_price - entry_price)
            if is_long:
                new_sl_price = entry_price + (profit_distance * 0.70)  # LONG: SL moves UP
            else:
                new_sl_price = entry_price - (profit_distance * 0.70)  # SHORT: SL moves DOWN
            # TP #1: 20% @ 12%
            tp1_pct = 0.12
            tp1_price = entry_price * (1 + tp1_pct / leverage * (1 if is_long else -1))
            partial_tp_levels.append((tp1_price, 0.20, "TP1@12%"))
            # TP #2: 20% @ 18%
            tp2_pct = 0.18
            tp2_price = entry_price * (1 + tp2_pct / leverage * (1 if is_long else -1))
            partial_tp_levels.append((tp2_price, 0.20, "TP2@18%"))
            # TP #3: 30% @ 25%
            tp3_pct = 0.25
            tp3_price = entry_price * (1 + tp3_pct / leverage * (1 if is_long else -1))
            partial_tp_levels.append((tp3_price, 0.30, "TP3@25%"))
            adjustment_reason = "lock 90%, 3x partial TPs"
            
        elif 35.0 <= pnl_pct < 60.0:
            # Enorm - trail SL to lock 85% of profit
            # [FIX] SL trails in profit direction
            profit_distance = abs(mark_price - entry_price)
            if is_long:
                new_sl_price = entry_price + (profit_distance * 0.85)  # LONG: SL moves UP
            else:
                new_sl_price = entry_price - (profit_distance * 0.85)  # SHORT: SL moves DOWN
            # TP #1: 20% @ 20%
            partial_tp_levels.append((entry_price * (1 + 0.20 / leverage * (1 if is_long else -1)), 0.20, "TP1@20%"))
            # TP #2: 20% @ 28%
            partial_tp_levels.append((entry_price * (1 + 0.28 / leverage * (1 if is_long else -1)), 0.20, "TP2@28%"))
            # TP #3: 30% @ 35%
            partial_tp_levels.append((entry_price * (1 + 0.35 / leverage * (1 if is_long else -1)), 0.30, "TP3@35%"))
            adjustment_reason = "lock 93%, aggressive partials"
            
        else:  # pnl_pct >= 60.0
            # EKSTREM - trail SL to lock 90% of profit
            # [FIX] SL trails in profit direction
            profit_distance = abs(mark_price - entry_price)
            if is_long:
                new_sl_price = entry_price + (profit_distance * 0.90)  # LONG: SL moves UP
            else:
                new_sl_price = entry_price - (profit_distance * 0.90)  # SHORT: SL moves DOWN
            # Let it ride to the moon with remaining position
            partial_tp_levels.append((entry_price * (1 + 0.30 / leverage * (1 if is_long else -1)), 0.15, "TP1@30%"))
            partial_tp_levels.append((entry_price * (1 + 0.40 / leverage * (1 if is_long else -1)), 0.15, "TP2@40%"))
            partial_tp_levels.append((entry_price * (1 + 0.50 / leverage * (1 if is_long else -1)), 0.20, "TP3@50%"))
            adjustment_reason = f"lock 95%, MOON targets! ({pnl_pct:.1f}%)"
        
        if not new_sl_price and not partial_tp_levels:
            return False
        
        # Hent price precision
        price_precision = self._get_price_precision(symbol)
        if new_sl_price:
            new_sl_price = self._round_price(new_sl_price, price_precision)
        
        # Round all TP prices
        for i in range(len(partial_tp_levels)):
            price, qty_pct, reason = partial_tp_levels[i]
            partial_tp_levels[i] = (self._round_price(price, price_precision), qty_pct, reason)
        
        # Sjekk eksisterende orders
        current_orders = self.client.futures_get_open_orders(symbol=symbol)
        existing_sl = None
        existing_tps = []
        
        for order in current_orders:
            if order['type'] == 'STOP_MARKET' and order.get('closePosition', False):
                existing_sl = float(order['stopPrice'])
            elif order['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT']:
                existing_tps.append(float(order.get('stopPrice', 0)))
        
        # Bare oppdater SL hvis ny SL er bedre
        should_update_sl = False
        if new_sl_price:
            if not existing_sl:
                should_update_sl = True
            elif is_long and new_sl_price > existing_sl:
                should_update_sl = True
            elif not is_long and new_sl_price < existing_sl:
                should_update_sl = True
        
        # Oppdater TPs hvis vi har nye levels
        should_update_tps = len(partial_tp_levels) > 0
        
        if not should_update_sl and not should_update_tps:
            logger.debug(f"⏸️ {symbol}: SL/TP already optimal")
            return False
        
        # Oppdater SL og/eller TP!
        try:
            logger.info(f"[TARGET] {symbol}: ADJUSTING - {adjustment_reason}")
            
            # Oppdater SL
            if should_update_sl:
                if existing_sl:
                    logger.info(f"   SL: ${existing_sl:.6f} → ${new_sl_price:.6f} (tightening stop)")
                else:
                    logger.info(f"   Setting SL: ${new_sl_price:.6f}")
                logger.info(f"   PnL: {pnl_pct:+.2f}% margin (${unrealized_pnl:+.2f})")
                
                # [COORDINATION] Try to place new SL first, only cancel old one if successful
                side = 'SELL' if is_long else 'BUY'
                position_side = 'LONG' if is_long else 'SHORT'  # [FIX] Define position_side for Binance Futures hedge mode
                
                # [SAFETY] Ensure SL price has minimum distance from current price (0.5%)
                min_distance_pct = 0.005  # 0.5% minimum distance
                if is_long:
                    min_sl_price = mark_price * (1 - min_distance_pct)
                    if new_sl_price > min_sl_price:
                        logger.warning(f"   [SKIP] New SL ${new_sl_price:.6f} too close to mark ${mark_price:.6f} - would trigger immediately")
                        new_sl_price = min_sl_price
                        logger.info(f"   [ADJUSTED] SL lowered to ${new_sl_price:.6f} for safety buffer")
                        new_sl_price = self._round_price(new_sl_price, price_precision)  # Re-round after adjustment
                else:  # SHORT
                    max_sl_price = mark_price * (1 + min_distance_pct)
                    if new_sl_price < max_sl_price:
                        logger.warning(f"   [SKIP] New SL ${new_sl_price:.6f} too close to mark ${mark_price:.6f} - would trigger immediately")
                        new_sl_price = max_sl_price
                        logger.info(f"   [ADJUSTED] SL raised to ${new_sl_price:.6f} for safety buffer")
                        new_sl_price = self._round_price(new_sl_price, price_precision)  # Re-round after adjustment
                
                try:
                    # Place new SL first
                    # 🔥 CRITICAL: Ensure SL price is correctly rounded for this symbol's precision
                    new_sl_price = self._round_price(new_sl_price, price_precision)
                    logger.info(f"   [DEBUG] Placing SL order @ ${new_sl_price} (precision={price_precision})")
                    
                    # [PHASE 1] Route through exit gateway for observability
                    order_params = {
                        'symbol': symbol,
                        'side': side,
                        'type': 'STOP_MARKET',
                        'stopPrice': new_sl_price,
                        'closePosition': True,
                        'workingType': 'MARK_PRICE',
                        'positionSide': position_side
                    }
                    if EXIT_GATEWAY_AVAILABLE:
                        await submit_exit_order(
                            module_name="position_monitor",
                            symbol=symbol,
                            order_params=order_params,
                            order_kind="sl",
                            client=self.client,
                            explanation="Dynamic SL adjustment from Exit Brain plan"
                        )
                    else:
                        self.client.futures_create_order(**order_params)
                    logger.info(f"   [OK] New SL @ ${new_sl_price:.6f} placed successfully")
                    
                    # Only NOW cancel old SL (new one is already protecting)
                    if existing_sl:
                        for order in current_orders:
                            if order['type'] == 'STOP_MARKET' and order.get('closePosition', False):
                                try:
                                    self.client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                                    logger.info(f"   🗑️  Cancelled old SL order {order['orderId']}")
                                except Exception as cancel_e:
                                    logger.warning(f"   Could not cancel old SL: {cancel_e}")
                except Exception as sl_e:
                    logger.error(f"   ❌ {symbol}: Failed to adjust SL - {sl_e}")
                    # Old SL remains in place - position still protected
                    should_update_sl = False  # Mark as failed so we don't claim success
                    
                    # 🛡️ CRITICAL FIX: If SL adjustment failed, DON'T touch TP orders!
                    # Keep existing TP to allow position to close at profit targets
                    logger.warning(f"   🛡️ {symbol}: Keeping existing TP orders since SL adjustment failed")
                    should_update_tps = False  # Prevent TP cancellation
            
            # Oppdater multiple partial TPs
            if should_update_tps:
                # Slett alle gamle TP orders først med comprehensive cleanup
                for order in current_orders:
                    if order['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT', 'LIMIT']:
                        try:
                            self.client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                            logger.info(f"   🗑️  Cancelled {order['type']} order {order['orderId']}")
                        except Exception as cancel_e:
                            logger.warning(f"   Could not cancel {order['type']}: {cancel_e}")
                
                # Plasser nye partial TP orders
                side = 'SELL' if is_long else 'BUY'
                position_side = 'LONG' if is_long else 'SHORT'  # [FIX] Define position_side for TP orders
                step_size = self._get_quantity_precision(symbol)
                
                logger.info(f"   [MONEY] Setting {len(partial_tp_levels)} partial TPs:")
                for tp_price, qty_pct, reason in partial_tp_levels:
                    qty = abs(amt) * qty_pct
                    qty = self._round_quantity(qty, step_size)
                    
                    # [PHASE 1] Route through exit gateway for observability
                    order_params = {
                        'symbol': symbol,
                        'side': side,
                        'type': 'TAKE_PROFIT_MARKET',
                        'quantity': qty,
                        'stopPrice': tp_price,
                        'workingType': 'MARK_PRICE',
                        'positionSide': position_side,
                        'reduceOnly': True  # 🔥 CRITICAL FIX: Required for One-Way mode conditional orders
                    }
                    if EXIT_GATEWAY_AVAILABLE:
                        await submit_exit_order(
                            module_name="position_monitor",
                            symbol=symbol,
                            order_params=order_params,
                            order_kind="partial_tp",
                            client=self.client,
                            explanation=f"{reason}: {qty_pct*100:.0f}% at ${tp_price:.6f}"
                        )
                    else:
                        self.client.futures_create_order(**order_params)
                    logger.info(f"      • {reason}: {qty_pct*100:.0f}% @ ${tp_price:.6f} (qty: {qty})")
            
            # [NEW] TP v3: Apply Dynamic Trailing Rearm if enabled
            if self.trailing_manager and pnl_pct >= 2.0:
                try:
                    optimal_callback = self.trailing_manager.calculate_optimal_callback(
                        entry_price=entry_price,
                        current_price=mark_price,
                        side="LONG" if is_long else "SHORT",
                        current_callback_rate=self.trail_callback_rate
                    )
                    
                    if optimal_callback is not None:
                        if self.trailing_manager.should_update_trailing_stop(symbol):
                            # Update trailing stop callback rate
                            logger_trailing.info(
                                f"[TP v3] {symbol}: Tightening trailing stop "
                                f"{self.trail_callback_rate:.2f}% → {optimal_callback:.2f}% "
                                f"(profit={pnl_pct:.1f}%)"
                            )
                            # Note: Actual trailing stop update would be done via order modification
                            # For now, log the recommendation for manual review
                except Exception as trail_e:
                    logger_trailing.warning(f"[TP v3] {symbol}: Trailing rearm failed - {trail_e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {symbol}: Failed to adjust SL - {e}")
            return False
    
    async def _set_tpsl_for_position(self, position: Dict) -> bool:
        """Set TP/SL orders for a single position using RL Agent"""
        symbol = position['symbol']
        amt = float(position['positionAmt'])
        entry_price = float(position['entryPrice'])
        
        if amt == 0:
            return False
        
        logger.info(f"[SHIELD] Setting TP/SL for {symbol}: amt={amt}, entry=${entry_price}")
        
        # 🔥 [RL-UNIFIED] Use RL Agent for consistent TP/SL across all systems
        # Same RL module used by event_driven_executor during trade placement
        from backend.services.ai.rl_position_sizing_agent import get_rl_sizing_agent
        
        tp_pct = self.tp_pct  # Fallback
        sl_pct = self.sl_pct  # Fallback
        trail_pct = self.trail_pct  # Fallback
        partial_tp = self.partial_tp  # Fallback
        
        # Try to use RL Agent for TP/SL
        try:
            rl_agent = get_rl_sizing_agent(enabled=True)
            
            if rl_agent:
                # Estimate parameters for RL decision
                estimated_confidence = 0.55  # Medium confidence fallback
                action = "BUY" if amt > 0 else "SELL"
                
                # Get market conditions for volatility estimate
                volatility_estimate = 0.02  # Default 2% volatility
                try:
                    # Get recent price data for volatility calculation
                    klines = self.client.futures_klines(symbol=symbol, interval='1h', limit=24)
                    if klines:
                        closes = [float(k[4]) for k in klines]
                        if len(closes) > 1:
                            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                            volatility_estimate = sum(abs(r) for r in returns) / len(returns)
                except Exception as vol_e:
                    logger.debug(f"Could not calculate volatility for {symbol}: {vol_e}")
                
                # Get account balance for exposure calculation
                account = self.client.futures_account()
                balance = float(account['totalWalletBalance'])
                
                # Estimate current exposure (rough)
                positions = self.client.futures_position_information()
                total_notional = sum(abs(float(p['notional'])) for p in positions if float(p['positionAmt']) != 0)
                exposure_pct = total_notional / (balance * 10) if balance > 0 else 0.0
                
                # Get RL decision
                rl_decision = rl_agent.decide_sizing(
                    symbol=symbol,
                    confidence=estimated_confidence,
                    atr_pct=volatility_estimate,
                    current_exposure_pct=exposure_pct,
                    equity_usd=balance,
                    adx=None,
                    trend_strength=None
                )
                
                tp_pct = rl_decision.tp_percent
                sl_pct = rl_decision.sl_percent
                partial_tp = rl_decision.partial_tp_size if rl_decision.partial_tp_enabled else 0.0
                
                logger.info(
                    f"   🤖 [RL-TPSL] {action} @ {estimated_confidence:.0%} confidence → "
                    f"TP={tp_pct*100:.1f}% SL={sl_pct*100:.1f}% Partial={partial_tp*100:.0f}%"
                )
            else:
                logger.warning(f"RL agent not available for {symbol}, using defaults")
            
        except Exception as e:
            logger.warning(f"Could not use RL Agent for {symbol}: {e}, using defaults")
            # Fallback to storage file if available
            try:
                if TPSL_STORAGE_PATH.exists():
                    with open(TPSL_STORAGE_PATH, 'r') as f:
                        tpsl_data = json.load(f)
                        if symbol in tpsl_data:
                            tp_pct = tpsl_data[symbol].get("tp_percent", tp_pct)
                            sl_pct = tpsl_data[symbol].get("sl_percent", sl_pct)
                            trail_pct = tpsl_data[symbol].get("trail_percent", trail_pct)
                            partial_tp = tpsl_data[symbol].get("partial_tp", partial_tp)
                            logger.info(f"[OK] Loaded TP/SL from storage for {symbol}: TP={tp_pct*100:.1f}% SL={sl_pct*100:.1f}%")
            except Exception as storage_e:
                logger.debug(f"Could not load TP/SL from storage: {storage_e}")
        
        # Get precision
        price_precision = self._get_price_precision(symbol)
        step_size = self._get_quantity_precision(symbol)
        
        # Get leverage for this position
        leverage = float(position.get('leverage', 30))
        
        # [AI-DRIVEN] Use AI Dynamic TP/SL as PRICE percentages, not margin percentages!
        # AI already calculates proper TP/SL for the leverage being used (20x)
        # DO NOT divide by leverage - the percentages are already price-based!
        price_tp_pct = tp_pct  # Already calculated for current leverage by AI
        price_sl_pct = sl_pct  # Already calculated for current leverage by AI
        
        logger.info(f"   🎯 [AI TP/SL] Using AI-calculated price targets: TP={price_tp_pct*100:.2f}% SL={price_sl_pct*100:.2f}% (leverage={leverage}x)")
        
        # Calculate TP/SL prices directly from AI percentages
        if amt > 0:  # LONG position
            tp_price = self._round_price(entry_price * (1 + price_tp_pct), price_precision)
            sl_price = self._round_price(entry_price * (1 - price_sl_pct), price_precision)
            side = 'SELL'
            # [FIX] Use correct positionSide based on account mode
            position_side = 'LONG' if self._is_hedge_mode else 'BOTH'
        else:  # SHORT position
            tp_price = self._round_price(entry_price * (1 - price_tp_pct), price_precision)
            sl_price = self._round_price(entry_price * (1 + price_sl_pct), price_precision)
            side = 'BUY'
            # [FIX] Use correct positionSide based on account mode
            position_side = 'SHORT' if self._is_hedge_mode else 'BOTH'
        
        # Calculate quantities using Dynamic partial_tp
        partial_qty = self._round_quantity(abs(amt) * partial_tp, step_size)
        remaining_qty = self._round_quantity(abs(amt) - partial_qty, step_size)
        
        logger.info(f"   TP: ${tp_price:.8f}, SL: ${sl_price:.8f}")
        logger.info(f"   Partial: {partial_qty}, Trailing: {remaining_qty}")
        
        # [FIX] Check if we JUST protected this position (within last 5 minutes)
        # This prevents re-setting orders when Binance API doesn't return conditional orders
        if symbol in self._protected_positions:
            time_since_protected = (datetime.now(timezone.utc) - self._protected_positions[symbol]).total_seconds()
            if time_since_protected < 300:  # 5 minutes
                logger.info(f"   ✅ [OK] {symbol} was protected {time_since_protected:.0f}s ago - skipping check")
                return True
        
        # [FIX] Check if BOTH TP and SL exist - if so, SKIP replacement (position fully protected)
        # CRITICAL: Binance stores conditional orders (STOP_MARKET, TAKE_PROFIT_MARKET, TRAILING_STOP_MARKET)
        # in a SEPARATE system from regular orders! We must check position risk directly.
        try:
            # Method 1: Check positionRisk endpoint which includes stop orders info
            position_risk = self.client.futures_position_information(symbol=symbol)
            
            # Method 2: Also get regular open orders as backup
            existing_orders = self.client.futures_get_open_orders(symbol=symbol)
            
            # Method 3: Count orders we just created (simplest check!)
            # If position has ANY open orders, assume it's protected (we just set them!)
            order_count = len(existing_orders)
            
            # SIMPLE HEURISTIC: If there are 2+ orders for this symbol, it's protected
            # (1 TP + 1 SL minimum, possibly more with trailing/backup)
            if order_count >= 2:
                logger.info(f"   ✅ [OK] {symbol} has {order_count} orders - position appears protected")
                logger.info(f"   [SKIP] Not replacing orders to preserve existing protection")
                return True  # Position is already protected
            
            # FALLBACK: Parse order types if count is low
            has_stop_loss = any(
                order.get('type') in ['STOP_MARKET', 'STOP', 'STOP_LOSS', 'STOP_LOSS_LIMIT', 'TRAILING_STOP_MARKET'] 
                for order in existing_orders
            )
            has_take_profit = any(
                order.get('type') in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT', 'LIMIT'] 
                for order in existing_orders
            )
            
            if has_stop_loss and has_take_profit:
                logger.info(f"   ✅ [OK] TP and SL already exist for {symbol} - position fully protected")
                logger.info(f"   [SKIP] Not replacing orders to preserve existing protection")
                return True  # Position is already fully protected, don't interfere
            
            # If we get here, position might be unprotected or only partially protected
            if order_count > 0:
                logger.info(f"   ⚠️ [PARTIAL] {symbol} has {order_count} orders but incomplete protection")
                logger.info(f"   [DEBUG] Order types: {[o.get('type') for o in existing_orders]}")
            
        except Exception as check_e:
            logger.warning(f"   Could not check existing orders: {check_e}")
        
        # 🛡️ CRITICAL FIX: Save existing orders before cancellation
        # If TP placement fails, we MUST restore SL to prevent unprotected position!
        existing_orders_backup = []
        try:
            existing_orders_backup = self.client.futures_get_open_orders(symbol=symbol)
        except Exception as backup_e:
            logger.warning(f"   Could not backup existing orders: {backup_e}")
        
        # Cancel existing orders first - use comprehensive cleanup
        try:
            cancelled = self._cancel_all_orders_for_symbol(symbol)
            if cancelled > 0:
                logger.info(f"   Cleaned up {cancelled} existing orders, trigger slots should be released")
                # [CRITICAL] Additional safety wait for trigger slot garbage collection
                # The wait in _cancel_all_orders_for_symbol() happens DURING the cancel
                # But testnet may need MORE time for slots to fully release before new orders
                import time
                logger.info(f"   ⏳ Waiting extra 2 seconds for trigger slot garbage collection...")
                time.sleep(2.0)
        except Exception as e:
            logger.warning(f"   Could not cancel orders: {e}")
        
        # 🔍 DEBUG: Log parameters before order placement
        logger.info(f"   [DEBUG] About to place orders:")
        logger.info(f"   [DEBUG] TP: {partial_qty} @ ${tp_price:.8f}, side={side}")
        logger.info(f"   [DEBUG] SL: {abs(amt)} @ ${sl_price:.8f}")
        logger.info(f"   [DEBUG] Trail callback: {self.trail_callback_rate:.2f}%")
        
        try:
            # 1. Partial TP order
            # 🔥 CRITICAL: Use positionSide='LONG'/'SHORT' for hedge mode!
            logger.info(f"   [DEBUG] Calling futures_create_order for TP...")
            
            # [PHASE 1] Route through exit gateway for observability
            tp_order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'TAKE_PROFIT_MARKET',
                'stopPrice': tp_price,
                'quantity': partial_qty,
                'workingType': 'MARK_PRICE',
                'positionSide': position_side,
                'reduceOnly': True  # 🔥 CRITICAL FIX: Required for One-Way mode conditional orders
            }
            if EXIT_GATEWAY_AVAILABLE:
                tp_order = await submit_exit_order(
                    module_name="position_monitor",
                    symbol=symbol,
                    order_params=tp_order_params,
                    order_kind="partial_tp",
                    client=self.client,
                    explanation=f"Initial partial TP {partial_qty} at ${tp_price:.8f}"
                )
            else:
                tp_order = self.client.futures_create_order(**tp_order_params)
            logger.info(f"   ✅ [OK] TP: {partial_qty} @ ${tp_price:.8f}")
            
            # 2. Trailing stop for remaining (use validated callback rate from config)
            # [FIX] CRITICAL: Validate callback rate is in [0.1, 5.0] range
            # NOTE: Trailing stop REPLACES fixed SL because it provides better protection
            # (locks profits while still having SL protection via callback rate)
            callback_rate = self.trail_callback_rate
            trailing_set = False
            
            if callback_rate < 0.1 or callback_rate > 5.0:
                logger.warning(f"   [WARNING] Invalid trail callback rate {callback_rate:.2f}% - using fixed SL instead")
            else:
                try:
                    # 🔥 CRITICAL: Use positionSide='LONG'/'SHORT' for hedge mode!
                    
                    # [PHASE 1] Route through exit gateway for observability
                    trail_order_params = {
                        'symbol': symbol,
                        'side': side,
                        'type': 'TRAILING_STOP_MARKET',
                        'quantity': remaining_qty,
                        'callbackRate': callback_rate,
                        'workingType': 'MARK_PRICE',
                        'positionSide': position_side,
                        'reduceOnly': True  # 🔥 CRITICAL FIX: Required for One-Way mode conditional orders
                    }
                    if EXIT_GATEWAY_AVAILABLE:
                        trail_order = await submit_exit_order(
                            module_name="position_monitor",
                            symbol=symbol,
                            order_params=trail_order_params,
                            order_kind="trailing",
                            client=self.client,
                            explanation=f"Trailing stop {remaining_qty} @ {callback_rate:.1f}% callback"
                        )
                    else:
                        trail_order = self.client.futures_create_order(**trail_order_params)
                    logger.info(f"   ✅ [OK] Trailing: {remaining_qty} @ {callback_rate:.1f}%")
                    trailing_set = True
                except Exception as trail_e:
                    logger.warning(f"   ⚠️ [WARNING] Could not set trailing stop: {trail_e}")
                    logger.info(f"   [FALLBACK] Will use fixed SL instead")
            
            # 3. Fixed Stop Loss (only if trailing stop failed or invalid)
            # NOTE: We use EITHER trailing stop OR fixed SL, NOT BOTH
            # (Binance doesn't allow multiple closePosition orders in same direction)
            if not trailing_set:
                # Use STOP_MARKET with closePosition=True for full protection
                # 🔥 CRITICAL: Use positionSide='LONG'/'SHORT' for hedge mode!
                
                # [PHASE 1] Route through exit gateway for observability
                sl_order_params = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'STOP_MARKET',
                    'stopPrice': sl_price,
                    'workingType': 'MARK_PRICE',
                    'positionSide': position_side,
                    'reduceOnly': True  # 🔥 CRITICAL FIX: Required for One-Way mode conditional orders
                }
                if EXIT_GATEWAY_AVAILABLE:
                    sl_order = await submit_exit_order(
                        module_name="position_monitor",
                        symbol=symbol,
                        order_params=sl_order_params,
                        order_kind="sl",
                        client=self.client,
                        explanation=f"Fixed SL (fallback from trailing) at ${sl_price:.8f}"
                    )
                else:
                    sl_order = self.client.futures_create_order(**sl_order_params)
                logger.info(f"   ✅ [OK] SL (STOP_MARKET): Full position @ ${sl_price:.8f}")
            else:
                logger.info(f"   ✅ [OK] SL protection via Trailing Stop (no fixed SL needed)")
            
            logger.info(f"   ✅ [SUCCESS] {symbol} fully protected with TP/SL")
            
            # [FIX] Remember that we protected this position
            self._protected_positions[symbol] = datetime.now(timezone.utc)
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ CRITICAL: Failed to set TP/SL for {symbol}: {e}")
            logger.error(f"   Exception type: {type(e).__name__}, Details: {str(e)}")
            
            # [FIX] Special handling for -4045 (order limit) error
            if '-4045' in str(e):
                logger.error(f"   ⚠️  Binance order limit reached (global account issue)")
                logger.error(f"   This is typically a Binance testnet bug with ghost orders")
                logger.error(f"   Recommendation: Wait 60 seconds or switch to production API")
                
                # Try simplified approach: ONE order only (closePosition SL)
                logger.info(f"   🔧 Attempting emergency fallback: simple closePosition SL only...")
                try:
                    # 🔥 CRITICAL: When using closePosition=True, do NOT include reduceOnly
                    # (Binance API error: -1106 "Parameter 'reduceonly' sent when not required")
                    emergency_sl_params = {
                        'symbol': symbol,
                        'side': side,
                        'type': 'STOP_MARKET',
                        'stopPrice': sl_price,
                        'closePosition': True,
                        'workingType': 'MARK_PRICE',
                        'positionSide': position_side
                        # NOTE: NO reduceOnly when using closePosition=True
                    }
                    
                    emergency_order = self.client.futures_create_order(**emergency_sl_params)
                    logger.info(f"   ✅ [EMERGENCY] Minimal SL protection set: closePosition @ ${sl_price}")
                    self._protected_positions[symbol] = datetime.now(timezone.utc)
                    return True
                    
                except Exception as emergency_e:
                    logger.critical(f"   ❌ Emergency SL also failed: {emergency_e}")
                    # Fall through to backup restoration
            
            # 🛡️ EMERGENCY FALLBACK: Restore SL protection if we have backup!
            if existing_orders_backup:
                logger.warning(f"   🚨 [EMERGENCY] Attempting to restore SL protection from backup...")
                try:
                    # Find and restore SL orders from backup
                    sl_orders = [o for o in existing_orders_backup if o.get('type') in ['STOP_MARKET', 'STOP', 'TRAILING_STOP_MARKET']]
                    if sl_orders:
                        for sl_order in sl_orders:
                            try:
                                # Recreate SL order
                                order_type = sl_order.get('type')
                                if order_type == 'STOP_MARKET':
                                    # [PHASE 1] Route through exit gateway for observability
                                    restore_order_params = {
                                        'symbol': symbol,
                                        'side': sl_order['side'],
                                        'type': 'STOP_MARKET',
                                        'stopPrice': float(sl_order['stopPrice']),
                                        'closePosition': True,
                                        'workingType': 'MARK_PRICE'
                                    }
                                    if EXIT_GATEWAY_AVAILABLE:
                                        await submit_exit_order(
                                            module_name="position_monitor",
                                            symbol=symbol,
                                            order_params=restore_order_params,
                                            order_kind="sl",
                                            client=self.client,
                                            explanation=f"Emergency SL restore from backup at ${sl_order['stopPrice']}"
                                        )
                                    else:
                                        self.client.futures_create_order(**restore_order_params)
                                    logger.info(f"   🛡️ [RESTORED] SL protection @ ${sl_order['stopPrice']}")
                                    break  # At least one SL restored - position protected!
                            except Exception as restore_e:
                                logger.error(f"   Failed to restore SL: {restore_e}")
                        
                        logger.warning(f"   ⚠️ {symbol} has SL protection but MISSING TP - manual intervention needed")
                    else:
                        logger.critical(f"   🚨 CRITICAL: {symbol} COMPLETELY UNPROTECTED - NO SL BACKUP FOUND!")
                except Exception as restore_e:
                    logger.critical(f"   🚨 CRITICAL: Could not restore SL protection: {restore_e}")
            else:
                logger.critical(f"   🚨 CRITICAL: {symbol} COMPLETELY UNPROTECTED - NO ORDER BACKUP!")
            
            return False
    
    async def _check_flash_crash(self) -> bool:
        """[CRITICAL FIX #1] Detect flash crashes via rapid equity drops.
        
        Returns True if flash crash detected, False otherwise.
        Publishes market.flash_crash_detected event on detection.
        """
        try:
            # Get current account equity
            account = self.client.futures_account()
            current_equity = float(account.get('totalWalletBalance', 0))
            
            if self._previous_equity is None:
                # First check - initialize baseline
                self._previous_equity = current_equity
                return False
            
            # Calculate drawdown since last check
            if self._previous_equity > 0:
                dd_pct = (current_equity - self._previous_equity) / self._previous_equity
                
                if dd_pct <= self._flash_crash_threshold_pct:
                    logger.critical(
                        f"🚨 FLASH CRASH DETECTED: Equity dropped {dd_pct*100:.2f}% in single check "
                        f"(${self._previous_equity:.2f} → ${current_equity:.2f})"
                    )
                    
                    # Publish flash crash event
                    if self.event_bus:
                        await self.event_bus.publish({
                            'type': 'market.flash_crash_detected',
                            'equity_before': self._previous_equity,
                            'equity_after': current_equity,
                            'drawdown_pct': dd_pct,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
                    
                    # Update baseline for next check
                    self._previous_equity = current_equity
                    return True
            
            # Update baseline for next check
            self._previous_equity = current_equity
            return False
            
        except Exception as e:
            logger.error(f"[FLASH-CRASH] Detection error: {e}")
            return False
    
    async def check_all_positions(self) -> Dict:
        """Check all positions and set TP/SL if missing
        
        [FULL AI-OS DEPLOYMENT MODE]
        - Position Intelligence Layer (PIL) classification
        - Profit Amplification Layer (PAL) recommendations
        - Self-Healing emergency interventions
        - AI sentiment re-evaluation
        - Meta-Strategy RL reward updates on position close
        
        [EXIT BRAIN V3] When Exit Brain Executor is active in LIVE mode,
        this function delegates ALL exit management to the executor.
        Position Monitor only handles PIL/PAL analytics and flash crash detection.
        """
        try:
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE -1: EXIT BRAIN EXECUTOR GUARD (CRITICAL!)
            # ═══════════════════════════════════════════════════════════════════════
            # [CRITICAL] When Exit Brain Executor is running in LIVE mode,
            # Position Monitor MUST NOT interfere with exit order placement.
            # Exit Brain Executor is the SINGLE MUSCLE for all exit decisions.
            
            exit_brain_is_active = (
                self.exit_brain_executor is not None 
                and is_exit_brain_live_fully_enabled()
            )
            
            if exit_brain_is_active:
                logger_exit_brain.info(
                    "[EXIT_GUARD] ✅ Exit Brain Executor ACTIVE in LIVE mode - "
                    "Position Monitor will ONLY handle analytics (PIL/PAL/flash crash), "
                    "NOT TP/SL placement. Exit Brain Executor owns all exit decisions."
                )
            
            logger.info("[DEBUG] check_all_positions() started")
            
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE 0: FLASH CRASH DETECTION (CRITICAL FIX #1)
            # ═══════════════════════════════════════════════════════════════════════
            
            flash_crash_detected = await self._check_flash_crash()
            logger.info(f"[DEBUG] Flash crash check complete: {flash_crash_detected}")
            if flash_crash_detected:
                logger.critical("[FLASH-CRASH] Rapid drawdown detected - AI-HFOS will respond")
            
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE 1: FETCH OPEN POSITIONS
            # ═══════════════════════════════════════════════════════════════════════
            
            logger.info("[DEBUG] Fetching positions...")
            # Get all positions (with rate limiting)
            positions = await self._call_binance_api('futures_position_information')
            logger.info(f"[DEBUG] Got {len(positions)} total positions")
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            logger.info(f"[DEBUG] Found {len(open_positions)} open positions")
            
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE 0.5: DETECT CLOSED POSITIONS (for Meta-Strategy RL updates)
            # ═══════════════════════════════════════════════════════════════════════
            
            # Get current open symbols
            current_open_symbols = {p['symbol'] for p in open_positions}
            
            # Find symbols that were open but are now closed
            closed_positions = []
            for symbol, prev_data in self._previous_open_positions.items():
                if symbol not in current_open_symbols:
                    closed_positions.append({
                        'symbol': symbol,
                        'previous_data': prev_data
                    })
            
            # [FIX] Clean up orphaned orders for closed positions
            if closed_positions:
                for closed_pos in closed_positions:
                    symbol = closed_pos['symbol']
                    try:
                        cancelled = self._cancel_all_orders_for_symbol(symbol)
                        if cancelled > 0:
                            logger.info(f"🗑️  Cleaned up {cancelled} orphaned orders for closed position {symbol}")
                    except Exception as cleanup_e:
                        logger.warning(f"Could not cleanup orders for {symbol}: {cleanup_e}")
            
            # Update Meta-Strategy rewards for closed positions
            if closed_positions:
                logger.info(f"[RL] Detected {len(closed_positions)} closed positions - updating Meta-Strategy rewards")
                
                for closed_pos in closed_positions:
                    symbol = closed_pos['symbol']
                    prev_data = closed_pos['previous_data']
                    
                    try:
                        # Get trade state with entry price and meta-strategy info
                        from pathlib import Path as PathLib
                        from backend.services.execution.execution import TradeStateStore
                        trade_state_path = PathLib("/app/data/trade_state.json")
                        trade_store = TradeStateStore(trade_state_path)
                        state = trade_store.get(symbol)
                        
                        if not state:
                            logger.debug(f"[RL] No trade state for {symbol} - skipping reward update")
                            continue
                        
                        # Get position close details from Binance (get last filled order)
                        recent_trades = await self._call_binance_api('futures_account_trades', symbol=symbol, limit=10)
                        
                        # Find the most recent closing trade
                        exit_price = None
                        realized_pnl = 0.0
                        
                        for trade in reversed(recent_trades):  # Most recent first
                            if trade.get('realizedPnl'):  # This was a closing trade
                                exit_price = float(trade['price'])
                                realized_pnl += float(trade['realizedPnl'])
                                break
                        
                        if not exit_price:
                            # Fallback to mark price from previous data
                            exit_price = float(prev_data.get('markPrice', prev_data.get('entryPrice', 0)))
                        
                        entry_price = float(prev_data['entryPrice'])
                        position_side = "LONG" if float(prev_data['positionAmt']) > 0 else "SHORT"
                        
                        # Calculate duration
                        opened_at = datetime.now(timezone.utc)
                        if "opened_at" in state:
                            opened_at = datetime.fromisoformat(state["opened_at"])
                        duration_hours = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600
                        
                        # Update Meta-Strategy reward
                        if self.meta_strategy and self.meta_strategy.enabled:
                            await self._update_meta_strategy_reward(
                                symbol=symbol,
                                entry_price=entry_price,
                                exit_price=exit_price,
                                side=position_side,
                                pnl=realized_pnl,
                                duration_hours=duration_hours
                            )
                            logger.info(
                                f"[RL] ✅ Updated Meta-Strategy reward for {symbol}: "
                                f"{position_side} entry=${entry_price:.4f} exit=${exit_price:.4f} "
                                f"PnL=${realized_pnl:+.2f} duration={duration_hours:.1f}h"
                            )
                        
                        # Update RL Position Sizing agent
                        try:
                            from backend.services.ai.rl_position_sizing_agent import get_rl_sizing_agent
                            rl_agent = get_rl_sizing_agent(enabled=True)
                            
                            if rl_agent and state.get('rl_state_key') and state.get('rl_action_key'):
                                # Calculate PnL percentage
                                notional = float(prev_data['positionAmt']) * entry_price
                                pnl_pct = (realized_pnl / abs(notional)) * 100 if notional != 0 else 0.0
                                
                                # Estimate max drawdown from unrealized PnL history
                                max_drawdown_pct = abs(pnl_pct) * 0.5  # Conservative estimate
                                
                                # Update RL agent
                                rl_agent.update_from_outcome(
                                    state_key=state['rl_state_key'],
                                    action_key=state['rl_action_key'],
                                    pnl_pct=pnl_pct,
                                    duration_hours=duration_hours,
                                    max_drawdown_pct=max_drawdown_pct
                                )
                                
                                logger.info(
                                    f"[RL-SIZING] 📈 Updated for {symbol}: "
                                    f"PnL={pnl_pct:+.2f}% duration={duration_hours:.1f}h"
                                )
                        except Exception as e:
                            logger.debug(f"[RL-SIZING] Could not update for {symbol}: {e}")
                        
                        # 💡 [SMART-SIZER] UPDATE TRADE OUTCOME FOR WIN RATE TRACKING
                        try:
                            from backend.services.execution.smart_position_sizer import get_smart_position_sizer
                            
                            smart_sizer = get_smart_position_sizer()
                            
                            # Determine if trade was a win
                            win = realized_pnl > 0
                            
                            # Update trade outcome
                            smart_sizer.update_trade_outcome(
                                symbol=symbol,
                                win=win,
                                pnl_usd=realized_pnl
                            )
                            
                            # Remove from open positions tracking
                            smart_sizer.remove_open_position(symbol)
                            
                            logger.info(
                                f"💡 [SMART-SIZER] Trade outcome recorded for {symbol}: "
                                f"{'WIN' if win else 'LOSS'} ${realized_pnl:+.2f}"
                            )
                        except Exception as smart_e:
                            logger.debug(f"[SMART-SIZER] Could not update for {symbol}: {smart_e}")
                        
                        # Clean up trade state
                        trade_store.delete(symbol)
                        
                        # [NEW] TRADESTORE - SPRINT 1 D5: Mark trade as closed in persistent storage
                        if self.trade_store and TRADESTORE_AVAILABLE:
                            try:
                                # Find trade by symbol (most recent open trade)
                                open_trades = await self.trade_store.get_open_trades(symbol=symbol)
                                if open_trades:
                                    trade = open_trades[0]  # Get most recent
                                    
                                    # Determine close reason
                                    if realized_pnl > 0:
                                        close_reason = "take_profit"
                                    elif realized_pnl < 0:
                                        close_reason = "stop_loss"
                                    else:
                                        close_reason = "manual_close"
                                    
                                    # Mark trade as closed
                                    await self.trade_store.mark_trade_closed(
                                        trade_id=trade.trade_id,
                                        exit_price=exit_price,
                                        exit_time=datetime.now(timezone.utc),
                                        close_reason=close_reason,
                                        exit_fee_usd=0.0,  # TODO: Get actual exit fee
                                        realized_pnl_usd=realized_pnl
                                    )
                                    
                                    logger_tradestore.info(
                                        f"[OK] Trade marked closed: {symbol} {trade.side.value} "
                                        f"Entry=${trade.entry_price:.2f} Exit=${exit_price:.2f} "
                                        f"PnL=${realized_pnl:+.2f} Reason={close_reason}"
                                    )
                                else:
                                    logger_tradestore.debug(f"[INFO] No open trade found in TradeStore for {symbol}")
                            except Exception as e:
                                logger_tradestore.error(f"[ERROR] Failed to mark trade closed for {symbol}: {e}", exc_info=True)
                        
                        # [FIX] Call TradeLifecycleManager.close_trade() to remove from active_trades
                        try:
                            from backend.services.risk_management.trade_lifecycle_manager import TradeLifecycleManager
                            from backend.services.risk_management.trade_lifecycle_manager import ExitDecision, ExitSignal
                            
                            # Get lifecycle manager instance from app_state
                            if self.app_state and hasattr(self.app_state, 'trade_lifecycle_manager'):
                                lifecycle_mgr = self.app_state.trade_lifecycle_manager
                                
                                # Find trade_id for this symbol in active_trades
                                trade_id = None
                                for tid, trade in lifecycle_mgr.active_trades.items():
                                    if trade.symbol == symbol:
                                        trade_id = tid
                                        break
                                
                                if trade_id:
                                    # Create exit decision
                                    exit_signal = ExitSignal.TAKE_PROFIT if realized_pnl > 0 else ExitSignal.STOP_LOSS
                                    exit_decision = ExitDecision(
                                        exit_signal=exit_signal,
                                        reason=f"Position closed (detected by monitor) - PnL: ${realized_pnl:+.2f}",
                                        exit_quantity=None  # Full close
                                    )
                                    
                                    # Call close_trade to remove from active_trades
                                    lifecycle_mgr.close_trade(
                                        trade_id=trade_id,
                                        exit_decision=exit_decision,
                                        actual_exit_price=exit_price
                                    )
                                    logger.info(f"✅ Removed {trade_id} from active_trades (position closed)")
                                else:
                                    logger.debug(f"No active trade found for {symbol} in lifecycle manager")
                        except Exception as close_e:
                            logger.warning(f"Could not call close_trade for {symbol}: {close_e}")
                        
                    except Exception as e:
                        logger.warning(f"[RL] Could not update reward for closed position {symbol}: {e}")
            
            # Update tracking dict with current open positions
            self._previous_open_positions = {
                p['symbol']: {
                    'symbol': p['symbol'],
                    'positionAmt': p['positionAmt'],
                    'entryPrice': p['entryPrice'],
                    'markPrice': p['markPrice'],
                    'unRealizedProfit': p['unRealizedProfit']
                }
                for p in open_positions
            }
            
            # [FIX] Clean up protection tracking for closed positions
            current_symbols = {p['symbol'] for p in open_positions}
            closed_symbols = set(self._protected_positions.keys()) - current_symbols
            for symbol in closed_symbols:
                del self._protected_positions[symbol]
                logger.debug(f"[CLEANUP] Removed protection tracking for closed position {symbol}")
            
            if not open_positions:
                return {"checked": 0, "newly_protected": 0, "adjusted": 0}
            
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE 2: AI-OS POSITION INTELLIGENCE & AMPLIFICATION
            # ═══════════════════════════════════════════════════════════════════════
            
            # [NEW] PIL: Classify all positions
            position_classifications = {}
            try:
                from backend.services.system_services import get_ai_services
                ai_services = get_ai_services()
                
                if (ai_services._initialized and 
                    ai_services.pil and 
                    ai_services.config.pil_enabled):
                    
                    for position in open_positions:
                        symbol = position['symbol']
                        amt = float(position['positionAmt'])
                        entry_price = float(position['entryPrice'])
                        mark_price = float(position['markPrice'])
                        unrealized_pnl = float(position['unRealizedProfit'])
                        
                        # Classify position
                        classification = ai_services.pil.classify_position(
                            symbol=symbol,
                            unrealized_pnl=unrealized_pnl,
                            current_price=mark_price,
                            entry_price=entry_price,
                            position_age_hours=0,  # TODO: Calculate from entry time
                            recent_momentum=0.0    # TODO: Calculate from price history
                        )
                        
                        if classification:
                            position_classifications[symbol] = classification
                            logger.info(
                                f"[PIL] {symbol}: {classification.category.value} "
                                f"(R={classification.current_R:.2f}, "
                                f"Rec={classification.recommendation.value})"
                            )
                            
            except Exception as e:
                logger.error(f"[ERROR] PIL classification failed: {e}", exc_info=True)
            
            # [NEW] PAL: Check for amplification opportunities
            pal_instructions = []
            try:
                if (ai_services._initialized and 
                    ai_services.pal and 
                    ai_services.config.pal_enabled):
                    
                    # Get PAL recommendations
                    pal_analysis = await post_trade_amplification_check(open_positions)
                    if pal_analysis:
                        pal_instructions = pal_analysis.get("instructions", [])
                        
                        if pal_instructions:
                            logger.info(
                                f"[PAL] Generated {len(pal_instructions)} amplification instructions"
                            )
                            for instr in pal_instructions:
                                logger.info(
                                    f"[PAL] {instr.get('symbol')}: "
                                    f"{instr.get('action')} "
                                    f"(reason: {instr.get('reason')})"
                                )
                                
            except Exception as e:
                logger.error(f"[ERROR] PAL analysis failed: {e}", exc_info=True)
            
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE 3: EMERGENCY INTERVENTIONS
            # ═══════════════════════════════════════════════════════════════════════
            
            # [ALERT] EMERGENCY CHECK: Close positions where SL should have triggered but didn't
            emergency_closed = 0
            for position in open_positions:
                symbol = position['symbol']
                amt = float(position['positionAmt'])
                entry_price = float(position['entryPrice'])
                mark_price = float(position['markPrice'])
                unrealized_pnl = float(position['unRealizedProfit'])
                leverage = float(position.get('leverage', 30))
                
                # Calculate PnL percentage correctly
                # For LONG: PnL% = (current_price - entry_price) / entry_price * 100
                # For SHORT: PnL% = (entry_price - current_price) / entry_price * 100
                if amt > 0:  # LONG position
                    price_change_pct = ((mark_price - entry_price) / entry_price) * 100
                else:  # SHORT position
                    price_change_pct = ((entry_price - mark_price) / entry_price) * 100
                
                pnl_pct = price_change_pct
                
                # [ALERT] EMERGENCY: If losing more than 10% (unleveraged), force close immediately
                if pnl_pct < -10.0:
                    logger.error(f"[ALERT] EMERGENCY: {symbol} losing {pnl_pct:.2f}% - SL FAILED! Force closing...")
                    try:
                        side_close = 'SELL' if amt > 0 else 'BUY'
                        position_side = 'LONG' if amt > 0 else 'SHORT'
                        
                        # Get position duration for reward update
                        from datetime import datetime, timezone
                        from backend.services.execution.execution import TradeStateStore
                        from pathlib import Path
                        
                        trade_state_path = Path("/app/data/trade_state.json")
                        trade_store = TradeStateStore(trade_state_path)
                        state = trade_store.get(symbol)
                        
                        opened_at = datetime.now(timezone.utc)
                        if state and "opened_at" in state:
                            try:
                                opened_at = datetime.fromisoformat(state["opened_at"].replace('Z', '+00:00'))
                            except:
                                pass
                        
                        duration_hours = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600
                        
                        # Cancel all orders first
                        self._cancel_all_orders_for_symbol(symbol)
                        # Force close with market order
                        
                        # [PHASE 1] Route through exit gateway for observability
                        emergency_order_params = {
                            'symbol': symbol,
                            'side': side_close,
                            'type': 'MARKET',
                            'quantity': abs(amt),
                            'positionSide': position_side
                        }
                        if EXIT_GATEWAY_AVAILABLE:
                            await submit_exit_order(
                                module_name="position_monitor",
                                symbol=symbol,
                                order_params=emergency_order_params,
                                order_kind="other_exit",
                                client=self.client,
                                explanation=f"Emergency close: duration {duration_hours:.1f}h, loss {pnl_pct:.2f}%"
                            )
                        else:
                            self.client.futures_create_order(**emergency_order_params)
                        logger.error(f"[OK] EMERGENCY CLOSE: {symbol} closed at ${mark_price:.6f} (loss: {pnl_pct:.2f}%)")
                        
                        # [NEW] META-STRATEGY REWARD UPDATE: Update Q-table with realized loss
                        await self._update_meta_strategy_reward(
                            symbol=symbol,
                            entry_price=entry_price,
                            exit_price=mark_price,
                            side=position_side,
                            pnl=unrealized_pnl,
                            duration_hours=duration_hours
                        )
                        
                        emergency_closed += 1
                        continue  # Skip to next position
                    except Exception as e:
                        logger.error(f"❌ EMERGENCY CLOSE FAILED for {symbol}: {e}")
            
            if emergency_closed > 0:
                logger.error(f"[ALERT] Emergency closed {emergency_closed} positions with failed SL")
                return {"status": "ok", "positions": 0, "protected": 0, "unprotected": 0}
            
            # [FIX] Initialize ai_signals to empty list before try block
            ai_signals = []
            
            # [ALERT] FIX #3: Re-evaluate AI sentiment for open positions
            if hasattr(self, 'ai_engine') and self.ai_engine:
                symbols = [p['symbol'] for p in open_positions]
                try:
                    current_positions_map = {p['symbol']: float(p['positionAmt']) for p in open_positions}
                    ai_signals = await self.ai_engine.get_trading_signals(symbols, current_positions_map)  # [TARGET] RENAMED from 'signals'
                    
                    for signal in ai_signals:
                        symbol = signal['symbol']
                        ai_action = signal['action']
                        ai_confidence = signal['confidence']
                        
                        # Find matching position
                        pos = next((p for p in open_positions if p['symbol'] == symbol), None)
                        if pos:
                            amt = float(pos['positionAmt'])
                            current_direction = 'BUY' if amt > 0 else 'SELL'
                            
                            # Check if AI disagrees or is weak
                            if ai_action == 'HOLD' and ai_confidence < 0.5:
                                logger.warning(
                                    f"[WARNING] {symbol}: AI sentiment weak (HOLD {ai_confidence:.0%}) - consider closing"
                                )
                            elif ai_action != current_direction and ai_action != 'HOLD':
                                logger.warning(
                                    f"[ALERT] {symbol}: AI changed from {current_direction} to {ai_action} "
                                    f"({ai_confidence:.0%}) - consider closing!"
                                )
                except Exception as e:
                    logger.debug(f"Could not re-evaluate AI sentiment: {e}")
            
            # [TARGET] STEG 1: DYNAMISK JUSTERING for alle posisjoner (hver 10 sek med check_interval=10)
            adjusted_count = 0
            adjusted_symbols = set()  # [TARGET] Track which symbols we adjusted
            
            # [NEW] PIL & PAL Integration - Classify and analyze positions
            try:
                from fastapi import FastAPI
                app = FastAPI()
                
                # Position Intelligence Layer (PIL) - Classify positions
                if hasattr(app.state, "position_intelligence"):
                    pil = app.state.position_intelligence
                    
                    for position in open_positions:
                        symbol = position['symbol']
                        amt = float(position['positionAmt'])
                        unrealized_pnl = float(position['unRealizedProfit'])
                        
                        # Classify position
                        classification = pil.classify_position(
                            symbol=symbol,
                            unrealized_pnl=unrealized_pnl,
                            current_price=float(position.get('markPrice', 0)),
                            entry_price=float(position.get('entryPrice', 0))
                        )
                        
                        if classification:
                            logger.info(
                                f"📊 [PIL] {symbol}: {classification.category.value} - "
                                f"R={classification.current_R:.2f} (Peak R={classification.peak_R:.2f})"
                            )
                
                # Profit Amplification Layer (PAL) - Check for amplification opportunities
                if hasattr(app.state, "profit_amplification"):
                    pal = app.state.profit_amplification
                    
                    # Convert positions to snapshots (simplified for now)
                    position_snapshots = []
                    
                    for position in open_positions:
                        try:
                            from backend.services.profit_amplification import PositionSnapshot, TrendStrength
                            from datetime import datetime, timezone
                            
                            symbol = position['symbol']
                            amt = float(position['positionAmt'])
                            side = "LONG" if amt > 0 else "SHORT"
                            unrealized_pnl = float(position['unRealizedProfit'])
                            
                            # Create snapshot (stub data for now)
                            snapshot = PositionSnapshot(
                                symbol=symbol,
                                side=side,
                                current_R=unrealized_pnl / 100,  # Rough estimate
                                peak_R=unrealized_pnl / 100 * 1.2,  # Assume some peak
                                unrealized_pnl=unrealized_pnl,
                                unrealized_pnl_pct=0.05,  # Stub
                                drawdown_from_peak_R=0.0,
                                drawdown_from_peak_pnl_pct=0.0,
                                current_leverage=20,
                                position_size_usd=abs(amt) * float(position.get('markPrice', 0)),
                                risk_pct=2.0,
                                hold_time_hours=24,  # Stub
                                entry_time=datetime.now(timezone.utc).isoformat(),
                                pil_classification="WINNER" if unrealized_pnl > 0 else "LOSER",
                                trend_strength=TrendStrength.STRONG,
                                volatility_regime="NORMAL",
                                symbol_rank=1,
                                symbol_category="CORE"
                            )
                            
                            position_snapshots.append(snapshot)
                        
                        except Exception as e:
                            logger.debug(f"[PAL] Could not create snapshot for {symbol}: {e}")
                    
                    # Analyze positions for amplification
                    if position_snapshots:
                        # [NEW] Get SafetyGovernor directives
                        safety_directives = None
                        if self.app_state and hasattr(self.app_state, "safety_governor_directives"):
                            safety_directives = self.app_state.safety_governor_directives
                        
                        recommendations = pal.analyze_positions(
                            positions=position_snapshots,
                            safety_governor_directives=safety_directives  # [NEW] Pass SafetyGovernor
                        )
                        
                        if recommendations:
                            logger.info(f"💰 [PAL] Found {len(recommendations)} amplification opportunities:")
                            for rec in recommendations[:3]:  # Show top 3
                                logger.info(
                                    f"💰 [PAL] {rec.position.symbol}: {rec.action.value} - "
                                    f"{rec.details}"
                                )
            
            except Exception as e:
                logger.debug(f"[PIL/PAL] Integration error: {e}")
            
            # Continue with dynamic adjustments
            for position in open_positions:
                try:
                    if await self._adjust_tpsl_dynamically(position, ai_signals):
                        adjusted_count += 1
                        adjusted_symbols.add(position['symbol'])  # [TARGET] Remember this symbol
                except Exception as e:
                    logger.debug(f"Could not adjust {position['symbol']}: {type(e).__name__}: {e}")
            
            if adjusted_count > 0:
                logger.info(f"🔄 Dynamically adjusted TP/SL for {adjusted_count} positions")
            
            # Get all open orders
            all_orders = self.client.futures_get_open_orders()
            
            # Group orders by symbol
            orders_by_symbol = {}
            for order in all_orders:
                symbol = order['symbol']
                if symbol not in orders_by_symbol:
                    orders_by_symbol[symbol] = []
                orders_by_symbol[symbol].append(order)
            
            # [TARGET] STEG 2: BASIC BESKYTTELSE - Check each position for TP/SL existence
            # [EXIT BRAIN GUARD] Skip protection checks if Exit Brain Executor is in LIVE mode
            if exit_brain_is_active:
                logger_exit_brain.info(
                    "[EXIT_GUARD] ⏭️  Skipping TP/SL protection checks - "
                    "Exit Brain Executor handles all exit orders"
                )
                protected = len(open_positions)  # All positions protected by Exit Brain
                unprotected = 0
                newly_protected = 0
            else:
                # [LEGACY PATH] Position Monitor places TP/SL orders
                protected = 0
                unprotected = 0
                newly_protected = 0
                
                for position in open_positions:
                    symbol = position['symbol']
                    
                    # [TARGET] SKIP hvis vi nettopp justerte denne symbolet dynamisk!
                    if symbol in adjusted_symbols:
                        protected += 1
                        logger.debug(f"[OK] {symbol} dynamically adjusted (skipping re-check)")
                        continue
                    
                    orders = orders_by_symbol.get(symbol, [])
                    
                    # Check if position has TP/SL
                    has_tp = any(o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] for o in orders)
                    # Check for STOP_MARKET with closePosition=True (the real safety net)
                    has_sl = any(o['type'] == 'STOP_MARKET' and o.get('closePosition', False) for o in orders)
                    
                    if has_tp and has_sl:
                        protected += 1
                        logger.debug(f"[OK] {symbol} already protected")
                    else:
                        unprotected += 1
                        logger.warning(f"[WARNING] {symbol} UNPROTECTED - setting TP/SL now...")
                        
                        # [FIX] Add delay to avoid overwhelming Binance API
                        if newly_protected > 0:
                            await asyncio.sleep(1.0)  # 1 second delay between positions
                        
                        # Set TP/SL asynchronously
                        success = await self._set_tpsl_for_position(position)
                        if success:
                            newly_protected += 1
            
            logger.info(
                f"[CHART] Position check: {len(open_positions)} total, "
                f"{protected} protected, {unprotected} unprotected, "
                f"{newly_protected} newly protected"
            )
            
            # ⚠️ BUG #6 FIX: DISABLED orphaned order cleanup due to false positives
            # Previous implementation caused catastrophic losses by deleting TP/SL orders
            # when API temporarily returned empty positions or had transient failures.
            # This resulted in:
            # - Positions left UNPROTECTED during critical price movements
            # - Force closures at loss due to missing stop losses
            # - Re-entries at worse prices
            # 
            # Example: APRUSDT lost -$370 USDT, AIOTUSDT position reduced 86%
            # 
            # TODO: Implement safer orphaned order cleanup with:
            # 1. Multiple confirmation checks (3+ consecutive cycles)
            # 2. Cross-validation with position history
            # 3. Manual confirmation requirement for order deletion
            # 4. Never delete TP/SL orders without confirmed position closure
            
            # ORIGINAL BROKEN CODE (DISABLED):
            # try:
            #     all_open_orders = self.client.futures_get_open_orders()
            #     if all_open_orders:
            #         open_symbols = {p['symbol'] for p in open_positions}
            #         orphaned_symbols = set()
            #         
            #         for order in all_open_orders:
            #             symbol = order['symbol']
            #             if symbol not in open_symbols:
            #                 orphaned_symbols.add(symbol)
            #         
            #         if orphaned_symbols:
            #             logger.warning(f"🗑️  Found orphaned orders for {len(orphaned_symbols)} symbols with no position: {orphaned_symbols}")
            #             total_cancelled = 0
            #             for symbol in orphaned_symbols:
            #                 cancelled = self._cancel_all_orders_for_symbol(symbol)
            #                 total_cancelled += cancelled
            #             logger.info(f"[OK] Cleaned up {total_cancelled} orphaned orders")
            # except Exception as cleanup_exc:
            #     logger.warning(f"Could not clean orphaned orders: {cleanup_exc}")
            
            return {
                "status": "ok",
                "positions": len(open_positions),
                "protected": protected,
                "unprotected": unprotected,
                "newly_protected": newly_protected
            }
            
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
            return {"status": "error", "error": str(e)}
    
    async def monitor_loop(self) -> None:
        """Main monitoring loop"""
        logger.info(f"[SEARCH] Starting Position Monitor (interval: {self.check_interval}s)")
        
        # [NEW] Start Exit Brain Dynamic Executor if enabled
        if self.exit_brain_executor:
            try:
                await self.exit_brain_executor.start()
                logger_exit_brain.warning(
                    f"[EXIT_BRAIN_V3] ✅ Dynamic Executor STARTED "
                    f"(monitoring interval: {self.exit_brain_executor.loop_interval_sec}s)"
                )
                logger_exit_brain.warning("[EXIT_BRAIN_V3] 🎯 Active monitoring for:")
                logger_exit_brain.warning("[EXIT_BRAIN_V3]   • Dynamic ATR-based stop-loss")
                logger_exit_brain.warning("[EXIT_BRAIN_V3]   • Trailing profit protection")
                logger_exit_brain.warning("[EXIT_BRAIN_V3]   • Automatic partial profit harvesting")
                logger_exit_brain.warning("[EXIT_BRAIN_V3]   • SL ratcheting after TP hits")
            except Exception as e:
                logger_exit_brain.error(f"[EXIT_BRAIN_V3] ❌ Failed to start Dynamic Executor: {e}", exc_info=True)
        
        # [NEW] Announce Exit Brain v3.5 status
        if self.exit_brain_v35:
            logger_exit_brain_v35.warning("🚀 [EXIT_BRAIN_V3.5] ✅ ACTIVE - Intelligent Leverage Framework v2")
            logger_exit_brain_v35.warning("[EXIT_BRAIN_V3.5] 🎯 Enhanced features:")
            logger_exit_brain_v35.warning("[EXIT_BRAIN_V3.5]   • Adaptive leverage scaling with ILFv2")
            logger_exit_brain_v35.warning("[EXIT_BRAIN_V3.5]   • Cross-exchange intelligence")
            logger_exit_brain_v35.warning("[EXIT_BRAIN_V3.5]   • PnL feedback integration")
            logger_exit_brain_v35.warning("[EXIT_BRAIN_V3.5]   • Advanced exit plan optimization")
        
        # [FIX] CRITICAL: Detect position mode on startup (One-Way vs Hedge Mode)
        try:
            import hmac
            import hashlib
            import time
            import requests
            
            timestamp = int(time.time() * 1000)
            params = {'timestamp': timestamp}
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.client.API_SECRET.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
            
            headers = {'X-MBX-APIKEY': self.client.API_KEY}
            response = requests.get(
                f"{self.client.API_URL}/fapi/v1/positionSide/dual",
                headers=headers,
                params=params,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                dual_side = data.get('dualSidePosition', False)
                self._is_hedge_mode = dual_side
                self._position_mode = "HEDGE" if dual_side else "ONE-WAY"
                
                if dual_side:
                    logger.info(f"[POSITION_MODE] ✅ HEDGE MODE detected - will use positionSide='LONG'/'SHORT'")
                else:
                    logger.info(f"[POSITION_MODE] ✅ ONE-WAY MODE detected - will use positionSide='BOTH'")
            else:
                logger.warning(f"[POSITION_MODE] ⚠️ Could not detect mode (HTTP {response.status_code}), defaulting to ONE-WAY")
                self._is_hedge_mode = False
                self._position_mode = "ONE-WAY"
        except Exception as e:
            logger.warning(f"[POSITION_MODE] ⚠️ Could not detect mode: {e}, defaulting to ONE-WAY")
            self._is_hedge_mode = False
            self._position_mode = "ONE-WAY"
        
        # [NEW] TRADESTORE - SPRINT 1 D5: Initialize async trade persistence
        if TRADESTORE_AVAILABLE and self.trade_store is None:
            try:
                # Get Redis client from app_state if available
                redis_client = None
                if self.app_state and hasattr(self.app_state, 'redis'):
                    redis_client = self.app_state.redis
                
                self.trade_store = await get_trade_store(redis_client=redis_client)
                logger_tradestore.info(
                    f"[OK] TradeStore initialized in PositionMonitor: {self.trade_store.backend_name} backend"
                )
            except Exception as e:
                logger_tradestore.error(f"[ERROR] TradeStore initialization failed: {e}", exc_info=True)
                self.trade_store = None
        
        while True:
            try:
                result = await self.check_all_positions()
                if result.get('newly_protected', 0) > 0:
                    logger.info(f"[OK] Protected {result['newly_protected']} positions")
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    def run(self) -> None:
        """Run the position monitor"""
        asyncio.run(self.monitor_loop())


def main():
    """Entry point for standalone execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    monitor = PositionMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
