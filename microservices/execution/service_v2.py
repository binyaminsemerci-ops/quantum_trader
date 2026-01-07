"""
Execution Service - REFACTORED V2

🎯 Clean microservice with NO monolith dependencies

Dependencies:
- ✅ EventBus (backend.core.event_bus) - isolated
- ✅ RiskStub (local) - minimal validation
- ✅ BinanceAdapter (local) - exchange integration
- ❌ NO TradeStore (replaced with local trade tracking)
- ❌ NO ExecutionSafetyGuard (replaced with RiskStub)
- ❌ NO GlobalRateLimiter (simple local rate limiter)
"""
import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from collections import deque

# EventBus (only core dependency)
try:
    from backend.core.event_bus import EventBus
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from backend.core.event_bus import EventBus

# Local imports
from .models import (
    AIDecisionEvent, OrderRequest, OrderResponse, OrderStatus,
    Position, Trade, ServiceHealth, ComponentHealth, OrderSide, TradeStatus
)
from .config import Settings
from .risk_stub import RiskStub
from .binance_adapter import BinanceAdapter, ExecutionMode

# [EXIT BRAIN V3] Advanced exit strategy orchestration
try:
    from .exit_brain_v3.router import ExitRouter
    from .exit_brain_v3.integration import build_context_from_position
    EXIT_BRAIN_V3_AVAILABLE = True
except ImportError as e:
    EXIT_BRAIN_V3_AVAILABLE = False

# [SIMPLE CLM] Continuous Learning Manager
try:
    from .simple_clm import SimpleCLM
    SIMPLE_CLM_AVAILABLE = True
except ImportError as e:
    SIMPLE_CLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class SimpleRateLimiter:
    """Simple token bucket rate limiter"""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_rpm = max_requests_per_minute
        self.requests = deque()
        logger.info(f"[RATE-LIMITER] Initialized: {max_requests_per_minute} rpm")
    
    async def acquire(self):
        """Wait until request can proceed"""
        now = time.time()
        
        # Remove requests older than 1 minute
        while self.requests and self.requests[0] < now - 60:
            self.requests.popleft()
        
        # Check if at limit
        if len(self.requests) >= self.max_rpm:
            # Wait until oldest request expires
            sleep_time = 60 - (now - self.requests[0])
            logger.warning(f"[RATE-LIMITER] Limit reached, waiting {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)
            self.requests.popleft()
        
        self.requests.append(now)


class ExecutionService:
    """
    Execution Service V2
    
    Responsibilities:
    - ✅ Receive trade signals from EventBus
    - ✅ Validate with RiskStub (minimal checks)
    - ✅ Execute orders via BinanceAdapter
    - ✅ Publish execution results to EventBus
    - ✅ Track active positions
    
    NOT responsible for:
    - ❌ Complex risk management (use Risk-Safety Service later)
    - ❌ Portfolio optimization
    - ❌ Persistent trade storage (optional add later)
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Core components
        self.event_bus: Optional[EventBus] = None
        self.risk_stub: Optional[RiskStub] = None
        self.binance: Optional[BinanceAdapter] = None
        self.rate_limiter: Optional[SimpleRateLimiter] = None
        
        # State tracking
        self._running = False
        self._event_task: Optional[asyncio.Task] = None
        
        # Trade tracking (in-memory for now)
        self.trades: Dict[str, Dict] = {}  # trade_id -> trade_data
        self.positions: Dict[str, Dict] = {}  # symbol -> position_data
        self.order_counter = 1
        
        # [GOVERNOR GATE] Redis client for kill switch and governor checks
        self.redis_client: Optional[Any] = None
        
        # [EXIT BRAIN V3] Initialize Exit Router for exit strategy orchestration
        self.exit_router = None
        self.exit_brain_enabled = os.getenv("EXIT_BRAIN_V3_ENABLED", "true").lower() == "true"
        if EXIT_BRAIN_V3_AVAILABLE and self.exit_brain_enabled:
            try:
                self.exit_router = ExitRouter()
                logger.info("[EXECUTION-V2] ✅ Exit Brain V3 initialized")
            except Exception as e:
                logger.error(f"[EXECUTION-V2] Exit Brain V3 init failed: {e}", exc_info=True)
                self.exit_router = None
        elif not self.exit_brain_enabled:
            logger.info("[EXECUTION-V2] Exit Brain V3 disabled")
        else:
            logger.warning("[EXECUTION-V2] Exit Brain V3 not available")
        
        # [SIMPLE CLM] Initialize Continuous Learning Manager
        self.clm = None
        self.clm_enabled = os.getenv("CLM_ENABLED", "true").lower() == "true"
        if SIMPLE_CLM_AVAILABLE and self.clm_enabled:
            try:
                ai_engine_url = os.getenv("AI_ENGINE_URL", "http://ai-engine:8001")
                retraining_hours = int(os.getenv("CLM_RETRAINING_HOURS", "168"))  # 7 days
                min_samples = int(os.getenv("CLM_MIN_SAMPLES", "100"))
                
                self.clm = SimpleCLM(
                    ai_engine_url=ai_engine_url,
                    retraining_interval_hours=retraining_hours,
                    min_samples_required=min_samples,
                    event_bus=None  # Will set after EventBus init
                )
                logger.info("[EXECUTION-V2] ✅ SimpleCLM initialized")
            except Exception as e:
                logger.error(f"[EXECUTION-V2] SimpleCLM init failed: {e}", exc_info=True)
                self.clm = None
        elif not self.clm_enabled:
            logger.info("[EXECUTION-V2] SimpleCLM disabled")
        else:
            logger.warning("[EXECUTION-V2] SimpleCLM not available")
        
        logger.info("[EXECUTION-V2] Service initialized")
    
    async def start(self):
        """Start the service"""
        if self._running:
            logger.warning("[EXECUTION-V2] Already running")
            return
        
        try:
            # 1. Initialize EventBus
            logger.info("[EXECUTION-V2] Connecting to Redis EventBus...")
            
            # Create Redis client
            from redis.asyncio import Redis
            redis_client = Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                password=self.settings.REDIS_PASSWORD or None,
                db=self.settings.REDIS_DB,
                decode_responses=True
            )
            
            # Create EventBus
            self.event_bus = EventBus(
                redis_client=redis_client,
                service_name="execution"
            )
            await self.event_bus.initialize()
            logger.info("[EXECUTION-V2] ✅ EventBus connected")
            
            # [GOVERNOR GATE] Store Redis client for governor checks
            self.redis_client = redis_client
            logger.info("[EXECUTION-V2] ✅ Governor Gate enabled - will check quantum:kill before each order")
            
            # 2. Initialize RiskStub
            logger.info("[EXECUTION-V2] Initializing RiskStub...")
            self.risk_stub = RiskStub(
                max_position_usd=self.settings.MAX_POSITION_USD,
                max_leverage=self.settings.MAX_LEVERAGE
            )
            logger.info(
                f"[EXECUTION-V2] ✅ RiskStub ready "
                f"(max_position: ${self.settings.MAX_POSITION_USD}, "
                f"max_leverage: {self.settings.MAX_LEVERAGE}x)"
            )
            
            # 3. Initialize Binance Adapter
            logger.info(f"[EXECUTION-V2] Initializing Binance ({self.settings.EXECUTION_MODE})...")
            
            mode = ExecutionMode(self.settings.EXECUTION_MODE)
            api_key = self.settings.BINANCE_API_KEY if mode != ExecutionMode.PAPER else None
            api_secret = self.settings.BINANCE_API_SECRET if mode != ExecutionMode.PAPER else None
            
            self.binance = BinanceAdapter(
                mode=mode,
                api_key=api_key,
                api_secret=api_secret
            )
            await self.binance.connect()
            logger.info("[EXECUTION-V2] ✅ Binance connected")
            
            # 4. Initialize Rate Limiter
            self.rate_limiter = SimpleRateLimiter(self.settings.BINANCE_RATE_LIMIT_RPM)
            logger.info("[EXECUTION-V2] ✅ Rate limiter ready")
            
            # 5. Subscribe to trade signals
            print("[DEBUG] About to subscribe to trade.intent events")
            logger.info("[EXECUTION-V2] Subscribing to 'trade.intent' events...")
            self.event_bus.subscribe("trade.intent", self._handle_trade_signal)
            print("[DEBUG] Subscribed! Handler added.")
            logger.info("[EXECUTION-V2] ✅ Event subscription active")
            
            # 6. Start EventBus consumer loop (stop first to ensure clean state)
            print("[DEBUG] Stopping EventBus if already running...")
            await self.event_bus.stop()
            print("[DEBUG] EventBus stopped. Now starting fresh...")
            await self.event_bus.start()
            print("[DEBUG] EventBus.start() completed!")
            logger.info("[EXECUTION-V2] ✅ EventBus consumer loop started")
            
            # 7. Start SimpleCLM if enabled
            if self.clm:
                self.clm.event_bus = self.event_bus  # Give CLM access to EventBus
                await self.clm.start()
                logger.info("[EXECUTION-V2] ✅ SimpleCLM started")
            
            self._running = True
            logger.info("=" * 60)
            logger.info("[EXECUTION-V2] 🚀 SERVICE STARTED")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"[EXECUTION-V2] ❌ Startup failed: {e}", exc_info=True)
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the service"""
        logger.info("[EXECUTION-V2] Stopping service...")
        self._running = False
        
        # Stop SimpleCLM
        if self.clm:
            await self.clm.stop()
        
        # Stop event consumer
        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
        
        # Close connections
        if self.binance:
            await self.binance.close()
        
        if self.event_bus:
            await self.event_bus.stop()
        
        logger.info("[EXECUTION-V2] ✅ Service stopped")
    
    async def _handle_trade_signal(self, event_data: Dict[str, Any]):
        """
        Handle incoming trade.intent event from AI Engine
        
        Event format:
        {
            "symbol": "BTCUSDT",
            "side": "BUY",  # or SELL
            "confidence": 0.85,
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "take_profit": 52000.0,
            "position_size_usd": 500.0,
            "leverage": 5,
            "timestamp": "2024-01-15T10:30:00Z"
        }
        """
        print(f"[DEBUG] _handle_trade_signal CALLED! event_data={event_data}")
        try:
            logger.info(f"[EXECUTION-V2] 📥 Received trade signal")
            logger.info(f"[EXECUTION-V2] 🔍 Raw event_data type: {type(event_data)}")
            logger.info(f"[EXECUTION-V2] 🔍 Raw event_data: {event_data}")
            logger.info(f"[EXECUTION-V2] 🔍 event_data keys: {list(event_data.keys()) if isinstance(event_data, dict) else 'NOT A DICT'}")
            
            # Parse signal
            signal = AIDecisionEvent(**event_data)
            
            # Generate trade ID
            trade_id = f"T{self.order_counter:06d}"
            self.order_counter += 1
            
            # 1. Risk validation
            logger.info(f"[EXECUTION-V2] {trade_id} - Validating risk...")
            risk_result = await self.risk_stub.validate_trade(
                symbol=signal.symbol,
                side=signal.side,
                size=signal.position_size_usd,
                leverage=signal.leverage or 1,
                price_estimate=signal.entry_price
            )
            
            if not risk_result["allowed"]:
                logger.warning(
                    f"[EXECUTION-V2] {trade_id} - ❌ Risk check FAILED: "
                    f"{risk_result['reason']}"
                )
                await self._publish_execution_result({
                    "trade_id": trade_id,
                    "status": "REJECTED",
                    "reason": risk_result["reason"],
                    "signal": signal.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                })
                return
            
            logger.info(f"[EXECUTION-V2] {trade_id} - ✅ Risk check passed")
            
            # 2. Governor Gate - Check kill switch BEFORE execution
            try:
                kill_switch = await self.redis_client.get("quantum:kill")
                mode = await self.redis_client.get("quantum:mode")
                governor_enabled = await self.redis_client.get("quantum:governor:execution")
                
                # Convert to int/str with defaults
                kill_switch = int(kill_switch) if kill_switch else 0
                mode = str(mode) if mode else "UNKNOWN"
                governor_enabled = str(governor_enabled) if governor_enabled else "DISABLED"
                
                logger.info(f"[GOVERNOR] Checking... kill={kill_switch}, mode={mode}, enabled={governor_enabled}")
                
                if governor_enabled.upper() == "ENABLED" and kill_switch == 1:
                    logger.warning(
                        f"[GOVERNOR] 🛑 BLOCKED {trade_id} - quantum:kill=1 (KILL MODE)"
                    )
                    await self._publish_execution_result({
                        "trade_id": trade_id,
                        "status": "BLOCKED",
                        "reason": "Governor kill switch active (quantum:kill=1)",
                        "signal": signal.dict(),
                        "governor_status": {
                            "kill": kill_switch,
                            "mode": mode,
                            "enabled": governor_enabled
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    return
                
                logger.info(f"[GOVERNOR] ✅ PASSED {trade_id} - kill={kill_switch}, mode={mode}")
                
            except Exception as gov_error:
                # Fail-safe: if governor check fails, BLOCK execution
                logger.error(f"[GOVERNOR] ❌ CHECK FAILED for {trade_id}: {gov_error}", exc_info=True)
                await self._publish_execution_result({
                    "trade_id": trade_id,
                    "status": "BLOCKED",
                    "reason": f"Governor check failed: {gov_error}",
                    "signal": signal.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                })
                return
            
            # 3. Rate limiting
            await self.rate_limiter.acquire()
            
            # 4. Calculate quantity
            # For futures: quantity = position_size_usd / entry_price
            quantity = signal.position_size_usd / signal.entry_price
            
            # Note: Quantity precision will be handled by BinanceAdapter based on symbol rules
            
            logger.info(
                f"[EXECUTION-V2] {trade_id} - Executing: {signal.symbol} {signal.side} "
                f"{quantity:.6f} @ ~{signal.entry_price} (leverage: {signal.leverage}x)"
            )
            
            # 5. Execute order
            order_result = await self.binance.place_market_order(
                symbol=signal.symbol,
                side=signal.side.upper(),
                quantity=quantity,
                leverage=signal.leverage or 1
            )
            
            # 6. Store trade
            trade_data = {
                "trade_id": trade_id,
                "order_id": order_result["order_id"],
                "symbol": signal.symbol,
                "side": signal.side,
                "quantity": quantity,
                "entry_price": order_result["price"] or signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "leverage": signal.leverage or 1,
                "status": order_result["status"],
                "confidence": signal.confidence,
                "created_at": datetime.utcnow().isoformat(),
                "mode": self.settings.EXECUTION_MODE
            }
            
            self.trades[trade_id] = trade_data
            
            # 7. Update position tracking and create exit plan
            # Note: For testnet, order_result["price"] may be 0, so use signal.entry_price as fallback
            actual_price = order_result["price"] if order_result["price"] and order_result["price"] > 0 else signal.entry_price
            order_is_filled = order_result["status"] in ["FILLED", "NEW", "PARTIALLY_FILLED"] and order_result["order_id"]
            
            if order_is_filled:
                self._update_position(signal.symbol, signal.side, quantity, actual_price)
                
                logger.info(
                    f"[EXECUTION-V2] T{self.order_counter:06d} - ✅ Order executed: "
                    f"{signal.symbol} {signal.side} {quantity} @ {actual_price}"
                )
                
                # [EXIT BRAIN V3] Create exit plan for position protection
                if self.exit_router and EXIT_BRAIN_V3_AVAILABLE:
                    try:
                        # Build position dict matching Binance API format for build_context_from_position
                        position_amt = quantity if signal.side.upper() in ['BUY', 'LONG'] else -quantity
                        position_for_brain = {
                            'symbol': signal.symbol,
                            'positionAmt': str(position_amt),
                            'entryPrice': str(actual_price),
                            'markPrice': str(actual_price),
                            'leverage': str(signal.leverage or 1),
                            'unrealizedProfit': '0.0',
                            'side': 'LONG' if signal.side.upper() in ['BUY', 'LONG'] else 'SHORT'
                        }
                        
                        # Build RL hints for Exit Brain
                        rl_hints = {
                            'tp_pct': ((signal.take_profit / actual_price) - 1) * 100 if signal.take_profit and actual_price else 2.0,
                            'sl_pct': ((actual_price / signal.stop_loss) - 1) * 100 if signal.stop_loss and actual_price else 1.0,
                            'confidence': signal.confidence
                        }
                        
                        # Create exit plan via router
                        plan = await self.exit_router.get_or_create_plan(
                            position=position_for_brain,
                            rl_hints=rl_hints,
                            risk_context={'max_leverage': signal.leverage or 1},
                            market_data={'atr': actual_price * 0.01}
                        )
                        
                        if plan and len(plan.legs) > 0:
                            logger.info(
                                f"[EXIT BRAIN V3] {signal.symbol}: Created exit plan with {len(plan.legs)} legs"
                            )
                            
                            # 🔥 NEW: Actually place the exit orders on Binance!
                            try:
                                # Extract TP and SL prices from plan
                                tp_prices = []
                                tp_quantities = []
                                sl_price = None
                                
                                for leg in plan.legs:
                                    if leg.kind.name == "STOP_LOSS":
                                        # Stop loss: trigger_pct is negative (e.g., -2%)
                                        sl_price = actual_price * (1 + leg.trigger_pct / 100)
                                        logger.info(
                                            f"  ↳ SL: size={leg.size_pct*100:.0f}%, "
                                            f"trigger={leg.trigger_pct*100:+.2f}%, price={sl_price:.6f}"
                                        )
                                    elif leg.kind.name in ["TAKE_PROFIT_1", "TAKE_PROFIT_2", "TAKE_PROFIT_3"]:
                                        # Take profit: trigger_pct is positive (e.g., +1.95%)
                                        tp_price = actual_price * (1 + leg.trigger_pct / 100)
                                        tp_qty = quantity * leg.size_pct
                                        tp_prices.append(tp_price)
                                        tp_quantities.append(tp_qty)
                                        logger.info(
                                            f"  ↳ TP{len(tp_prices)}: size={leg.size_pct*100:.0f}%, "
                                            f"trigger={leg.trigger_pct*100:+.2f}%, price={tp_price:.6f}, qty={tp_qty:.4f}"
                                        )
                                
                                # Place exit orders on Binance
                                if sl_price and len(tp_prices) > 0:
                                    exit_result = await self.binance.place_exit_orders(
                                        symbol=signal.symbol,
                                        side=signal.side,
                                        quantity=quantity,
                                        stop_loss_price=sl_price,
                                        take_profit_prices=tp_prices,
                                        take_profit_quantities=tp_quantities
                                    )
                                    
                                    if exit_result["status"] == "SUCCESS":
                                        logger.info(
                                            f"[EXIT BRAIN V3] ✅ {signal.symbol}: All exit orders placed! "
                                            f"SL @ {sl_price:.6f}, {len(tp_prices)} TPs"
                                        )
                                    elif exit_result["status"] == "PARTIAL":
                                        logger.warning(
                                            f"[EXIT BRAIN V3] ⚠️ {signal.symbol}: Partial exit order placement"
                                        )
                                    else:
                                        logger.error(
                                            f"[EXIT BRAIN V3] ❌ {signal.symbol}: Failed to place exit orders"
                                        )
                                else:
                                    logger.warning(
                                        f"[EXIT BRAIN V3] {signal.symbol}: Incomplete exit plan - "
                                        f"SL={sl_price}, TPs={len(tp_prices)}"
                                    )
                                    
                            except Exception as order_exc:
                                logger.error(
                                    f"[EXIT BRAIN V3] {signal.symbol}: Order placement failed: {order_exc}",
                                    exc_info=True
                                )
                        else:
                            logger.warning(f"[EXIT BRAIN V3] {signal.symbol}: Plan has no legs")
                            
                    except Exception as brain_exc:
                        logger.error(
                            f"[EXIT BRAIN V3] {signal.symbol}: Plan creation failed: {brain_exc}",
                            exc_info=True
                        )
            
            else:
                logger.error(
                    f"[EXECUTION-V2] {trade_id} - ❌ Order FAILED: {order_result.get('status', 'Unknown')}"
                )
            
            # 7. Publish result
            await self._publish_execution_result({
                "trade_id": trade_id,
                "status": "FILLED" if order_is_filled else order_result["status"],
                "order_id": order_result["order_id"],
                "symbol": signal.symbol,
                "side": signal.side,
                "quantity": quantity,
                "price": order_result["price"],
                "timestamp": datetime.utcnow().isoformat(),
                "mode": self.settings.EXECUTION_MODE
            })
            
            if order_result["status"] == "FILLED":
                logger.info(
                    f"[EXECUTION-V2] {trade_id} - ✅ Order FILLED at {order_result['price']}"
                )
            else:
                logger.error(
                    f"[EXECUTION-V2] {trade_id} - ❌ Order FAILED: {order_result.get('error', 'Unknown')}"
                )
                
        except Exception as e:
            logger.error(f"[EXECUTION-V2] Error handling trade signal: {e}", exc_info=True)
    
    def _update_position(self, symbol: str, side: str, quantity: float, price: float):
        """Update position tracking"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "quantity": 0.0,
                "side": None,
                "avg_entry": 0.0,
                "unrealized_pnl": 0.0
            }
        
        pos = self.positions[symbol]
        
        if side.upper() in ["BUY", "LONG"]:
            pos["quantity"] += quantity
            pos["side"] = "LONG"
        else:
            pos["quantity"] -= quantity
            if pos["quantity"] < 0:
                pos["side"] = "SHORT"
            elif pos["quantity"] == 0:
                pos["side"] = None
        
        # Simplified avg entry calc
        pos["avg_entry"] = price
        
        logger.info(f"[EXECUTION-V2] Position updated: {symbol} {pos['side']} {pos['quantity']}")
    
    async def _publish_execution_result(self, result: Dict[str, Any]):
        """Publish execution result to EventBus"""
        try:
            await self.event_bus.publish("execution.result", result)
            logger.info(f"[EXECUTION-V2] Published execution result: {result['trade_id']}")
        except Exception as e:
            logger.error(f"[EXECUTION-V2] Failed to publish result: {e}")
    
    async def get_health(self) -> ServiceHealth:
        """Get service health status"""
        components = []
        
        # EventBus health
        if self.event_bus and self.event_bus.redis:
            try:
                await self.event_bus.redis.ping()
                components.append(ComponentHealth(
                    name="eventbus",
                    status="OK",
                    latency_ms=0.5
                ))
            except Exception as e:
                components.append(ComponentHealth(
                    name="eventbus",
                    status="DOWN",
                    error=str(e)
                ))
        else:
            components.append(ComponentHealth(name="eventbus", status="NOT_INITIALIZED"))
        
        # Binance health
        if self.binance:
            status = "OK" if self.binance.mode == ExecutionMode.PAPER else "CONNECTED"
            components.append(ComponentHealth(
                name="binance",
                status=status,
                message=f"Mode: {self.binance.mode.value}"
            ))
        else:
            components.append(ComponentHealth(name="binance", status="NOT_INITIALIZED"))
        
        # RiskStub health
        if self.risk_stub:
            components.append(ComponentHealth(
                name="risk_stub",
                status="OK",
                message=f"{len(self.risk_stub.allowed_symbols)} symbols allowed"
            ))
        
        # Exit Brain V3 health
        if self.exit_router:
            components.append(ComponentHealth(
                name="exit_brain_v3",
                status="OK",
                message="Exit strategy orchestration active"
            ))
        
        # SimpleCLM health
        if self.clm:
            clm_status = self.clm.get_status()
            components.append(ComponentHealth(
                name="clm",
                status="OK" if clm_status["running"] else "STOPPED",
                message=f"Next retraining: {clm_status.get('next_retraining', 'N/A')}"
            ))
        
        # Overall status
        all_ok = all(c.status in ["OK", "CONNECTED"] for c in components)
        
        return ServiceHealth(
            service="execution",
            status="OK" if all_ok else "DEGRADED",
            version="2.0.0",
            components=components,
            active_trades=len(self.trades),
            active_positions=len([p for p in self.positions.values() if p["quantity"] != 0]),
            mode=self.settings.EXECUTION_MODE
        )
    
    async def get_positions(self) -> List[Dict]:
        """Get active positions"""
        return [
            pos for pos in self.positions.values()
            if pos["quantity"] != 0
        ]
    
    async def get_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades"""
        trades = sorted(
            self.trades.values(),
            key=lambda t: t["created_at"],
            reverse=True
        )
        return trades[:limit]
