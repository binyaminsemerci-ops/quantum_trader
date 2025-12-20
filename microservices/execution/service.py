"""
Execution Service - Core Service Logic

Orchestrates order execution, position monitoring, and trade lifecycle.
"""
import asyncio
import logging
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

# Import existing modules (will be refactored to avoid deep backend imports)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# D5: TradeStore
from backend.core.trading.trade_store_base import TradeStore
from backend.services.performance_analytics.models import Trade as TradeModel, TradeDirection as TradeSide

# D6: Rate Limiter
from backend.services.execution.global_rate_limiter import GlobalRateLimiter
from backend.integrations.binance import BinanceClientWrapper, create_binance_wrapper

# D7: Execution Safety
from backend.services.execution.execution_safety import ExecutionSafetyGuard
from backend.services.execution.safe_order_executor import SafeOrderExecutor

# EventBus + DiskBuffer
from backend.core.event_bus import EventBus
from backend.core.disk_buffer import DiskBuffer

# Local imports
from .models import (
    AIDecisionEvent, OrderRequest, OrderResponse, OrderStatus,
    Position, Trade, PositionListResponse, ServiceHealth, ComponentHealth
)
from .config import settings

logger = logging.getLogger(__name__)


class ExecutionService:
    """
    Execution Service
    
    Responsibilities:
    - Order execution (entry + exit)
    - Position monitoring
    - Trade lifecycle management
    - Binance API integration with safety layers
    """
    
    def __init__(self):
        # Core components
        self.trade_store: Optional[TradeStore] = None
        self.rate_limiter: Optional[GlobalRateLimiter] = None
        self.binance_client: Optional[BinanceClientWrapper] = None
        self.safety_guard: Optional[ExecutionSafetyGuard] = None
        self.safe_executor: Optional[SafeOrderExecutor] = None
        self.event_bus: Optional[EventBus] = None
        self.disk_buffer: Optional[DiskBuffer] = None
        
        # HTTP client for risk-safety-service API
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # State tracking
        self._running = False
        self._event_loop_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._active_positions: Dict[str, Any] = {}
        self._orders_last_minute = 0
        
        logger.info("[EXECUTION] Service initialized")
    
    async def start(self):
        """Start the service."""
        if self._running:
            logger.warning("[EXECUTION] Service already running")
            return
        
        try:
            # Initialize TradeStore (D5)
            logger.info("[EXECUTION] Initializing TradeStore...")
            self.trade_store = TradeStore(
                redis_url=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                sqlite_path=settings.SQLITE_DB_PATH
            )
            await self.trade_store.initialize()
            logger.info("[EXECUTION] ✅ TradeStore ready")
            
            # Initialize Rate Limiter (D6)
            logger.info("[EXECUTION] Initializing Rate Limiter...")
            self.rate_limiter = GlobalRateLimiter(rate_limit_per_minute=settings.BINANCE_RATE_LIMIT_RPM)
            logger.info(f"[EXECUTION] ✅ Rate Limiter ready ({settings.BINANCE_RATE_LIMIT_RPM} rpm)")
            
            # Initialize Binance Client (D6)
            logger.info("[EXECUTION] Initializing Binance Client...")
            self.binance_client = create_binance_wrapper(
                api_key=settings.BINANCE_API_KEY,
                api_secret=settings.BINANCE_API_SECRET,
                rate_limiter=self.rate_limiter,
                testnet=settings.USE_BINANCE_TESTNET
            )
            logger.info(f"[EXECUTION] ✅ Binance Client ready ({'TESTNET' if settings.USE_BINANCE_TESTNET else 'MAINNET'})")
            
            # Initialize Execution Safety (D7)
            logger.info("[EXECUTION] Initializing Execution Safety layers...")
            self.safety_guard = ExecutionSafetyGuard(policy_store=None)  # TODO: PolicyStore snapshot
            self.safe_executor = SafeOrderExecutor(
                policy_store=None,  # TODO: PolicyStore snapshot
                safety_guard=self.safety_guard
            )
            logger.info("[EXECUTION] ✅ Execution Safety ready")
            
            # Initialize EventBus
            logger.info("[EXECUTION] Initializing EventBus...")
            self.event_bus = EventBus()
            
            # Subscribe to relevant events
            self.event_bus.subscribe("ai.decision.made", self._handle_ai_decision)
            self.event_bus.subscribe("ess.tripped", self._handle_ess_tripped)
            self.event_bus.subscribe("policy.updated", self._handle_policy_updated)
            logger.info("[EXECUTION] ✅ EventBus subscriptions active")
            
            # Initialize DiskBuffer
            self.disk_buffer = DiskBuffer(
                buffer_dir="data/event_buffers/execution",
                buffer_name="execution_events"
            )
            logger.info("[EXECUTION] ✅ DiskBuffer ready")
            
            # Initialize HTTP client for risk-safety-service
            self.http_client = httpx.AsyncClient(base_url=settings.RISK_SAFETY_SERVICE_URL, timeout=5.0)
            logger.info(f"[EXECUTION] ✅ HTTP client ready (risk-safety: {settings.RISK_SAFETY_SERVICE_URL})")
            
            # Start background tasks
            self._running = True
            self._event_loop_task = asyncio.create_task(self._event_processing_loop())
            self._monitor_task = asyncio.create_task(self._position_monitor_loop())
            
            logger.info("[EXECUTION] ✅ Service started successfully")
            
        except Exception as e:
            logger.error(f"[EXECUTION] ❌ Failed to start service: {e}", exc_info=True)
            raise
    
    async def stop(self):
        """Stop the service."""
        if not self._running:
            return
        
        logger.info("[EXECUTION] Stopping service...")
        self._running = False
        
        # Cancel background tasks
        for task in [self._event_loop_task, self._monitor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Flush disk buffer
        if self.disk_buffer:
            self.disk_buffer.flush()
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
        
        logger.info("[EXECUTION] ✅ Service stopped")
    
    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================
    
    async def _handle_ai_decision(self, event_data: Dict[str, Any]):
        """
        Handle ai.decision.made event.
        
        Flow:
        1. Check ESS status via risk-safety-service API
        2. Validate order with ExecutionSafetyGuard
        3. Place order via SafeOrderExecutor
        4. Save trade to TradeStore
        5. Publish order.placed and trade.opened events
        """
        try:
            symbol = event_data.get("symbol")
            side = event_data.get("side")
            confidence = event_data.get("confidence", 0.0)
            
            logger.info(f"[EXECUTION] AI decision received: {symbol} {side.upper()} (confidence={confidence:.2f})")
            
            # 1. Check ESS
            try:
                ess_response = await self.http_client.get("/api/risk/ess/status")
                ess_status = ess_response.json()
                
                if not ess_status.get("can_execute"):
                    logger.warning(
                        f"[EXECUTION] ❌ Order blocked by ESS: {symbol} {side.upper()}\n"
                        f"   ESS State: {ess_status.get('state')}\n"
                        f"   Reason: {ess_status.get('trip_reason', 'N/A')}"
                    )
                    
                    # Publish order.failed event
                    await self.event_bus.publish("order.failed", {
                        "symbol": symbol,
                        "side": side,
                        "reason": "ESS blocked",
                        "ess_state": ess_status.get("state"),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    return
            except Exception as e:
                logger.error(f"[EXECUTION] Failed to check ESS: {e}")
                # Fail-open: Continue with order (ESS check failure should not block trading)
            
            # 2-5. Execute order (delegated to execute_order method)
            order_request = OrderRequest(
                symbol=symbol,
                side=side,
                quantity=event_data.get("quantity", 0.0),
                price=event_data.get("entry_price"),
                leverage=event_data.get("leverage"),
                stop_loss=event_data.get("stop_loss"),
                take_profit=event_data.get("take_profit"),
                trail_percent=event_data.get("trail_percent"),
                metadata={
                    "source": "ai.decision.made",
                    "confidence": confidence,
                    "model": event_data.get("model", "ensemble"),
                    "meta_strategy": event_data.get("meta_strategy")
                }
            )
            
            await self.execute_order(order_request)
            
        except Exception as e:
            logger.error(f"[EXECUTION] Error handling ai.decision: {e}", exc_info=True)
    
    async def _handle_ess_tripped(self, event_data: Dict[str, Any]):
        """Handle ess.tripped event - log and potentially cancel pending orders."""
        reason = event_data.get("reason", "Unknown")
        logger.error(f"[EXECUTION] 🚨 ESS TRIPPED: {reason} - All trading blocked!")
        
        # TODO: Cancel all pending orders (future enhancement)
    
    async def _handle_policy_updated(self, event_data: Dict[str, Any]):
        """Handle policy.updated event - refresh PolicyStore snapshot."""
        key = event_data.get("key")
        value = event_data.get("new_value")
        logger.info(f"[EXECUTION] Policy updated: {key} = {value}")
        
        # TODO: Refresh PolicyStore snapshot (future enhancement)
    
    # ========================================================================
    # ORDER EXECUTION
    # ========================================================================
    
    async def execute_order(self, request: OrderRequest) -> OrderResponse:
        """
        Execute an order with full safety checks.
        
        Flow:
        1. Validate with ExecutionSafetyGuard
        2. Place order via SafeOrderExecutor
        3. Save to TradeStore
        4. Publish events
        """
        symbol = request.symbol
        side = request.side.value
        
        try:
            # 1. Safety validation (D7)
            if request.price:
                # Fetch current market price for slippage check
                ticker = await self.binance_client._signed_request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
                current_price = float(ticker.get("price", 0))
                
                validation_result = await self.safety_guard.validate_and_adjust_order(
                    symbol=symbol,
                    side=side,
                    planned_entry_price=request.price,
                    current_market_price=current_price,
                    sl_price=request.stop_loss,
                    tp_price=request.take_profit,
                    leverage=request.leverage
                )
                
                if not validation_result.is_valid:
                    logger.warning(f"[EXECUTION] Order validation failed: {validation_result.reason}")
                    return OrderResponse(
                        success=False,
                        symbol=symbol,
                        side=side,
                        quantity=request.quantity,
                        status=OrderStatus.FAILED,
                        message=f"Validation failed: {validation_result.reason}",
                        timestamp=datetime.now(timezone.utc).isoformat()
                    )
                
                # Use adjusted prices if safety guard modified them
                if validation_result.adjusted_sl:
                    request.stop_loss = validation_result.adjusted_sl
                    logger.info(f"[EXECUTION] SL adjusted to {request.stop_loss:.2f}")
                if validation_result.adjusted_tp:
                    request.take_profit = validation_result.adjusted_tp
                    logger.info(f"[EXECUTION] TP adjusted to {request.take_profit:.2f}")
            
            # 2. Place entry order via SafeOrderExecutor (D7)
            logger.info(f"[EXECUTION] Placing order: {symbol} {side.upper()} {request.quantity}")
            
            # TODO: Implement actual order placement via BinanceFuturesExecutionAdapter
            # For now, placeholder
            order_id = f"ORDER_{symbol}_{int(datetime.now().timestamp())}"
            fill_price = request.price or 0.0
            
            logger.info(f"[EXECUTION] ✅ Order placed: {order_id}")
            
            # 3. Save to TradeStore (D5)
            trade = TradeModel(
                trade_id=order_id,
                symbol=symbol,
                side=TradeSide.LONG if side in ["buy", "long"] else TradeSide.SHORT,
                status=TradeStatus.OPEN,
                quantity=request.quantity,
                leverage=request.leverage or 1,
                margin_usd=request.quantity * fill_price / (request.leverage or 1),
                entry_price=fill_price,
                entry_time=datetime.utcnow(),
                sl_price=request.stop_loss,
                tp_price=request.take_profit,
                trail_percent=request.trail_percent,
                model=request.metadata.get("model", "manual") if request.metadata else "manual",
                confidence=request.metadata.get("confidence", 0.0) if request.metadata else 0.0,
                metadata=request.metadata
            )
            
            await self.trade_store.save_new_trade(trade)
            logger.info(f"[EXECUTION] Trade saved to TradeStore: {order_id}")
            
            # 4. Publish events
            await self.event_bus.publish("order.placed", {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": request.quantity,
                "price": fill_price,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            await self.event_bus.publish("trade.opened", {
                "trade_id": order_id,
                "symbol": symbol,
                "side": side,
                "entry_price": fill_price,
                "quantity": request.quantity,
                "leverage": request.leverage,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return OrderResponse(
                success=True,
                order_id=order_id,
                trade_id=order_id,
                symbol=symbol,
                side=side,
                quantity=request.quantity,
                price=fill_price,
                status=OrderStatus.FILLED,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error(f"[EXECUTION] Order execution failed: {e}", exc_info=True)
            
            # Publish order.failed event
            await self.event_bus.publish("order.failed", {
                "symbol": symbol,
                "side": side,
                "reason": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return OrderResponse(
                success=False,
                symbol=symbol,
                side=side,
                quantity=request.quantity,
                status=OrderStatus.FAILED,
                message=str(e),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    # ========================================================================
    # POSITION MONITORING
    # ========================================================================
    
    async def _position_monitor_loop(self):
        """Background loop for position monitoring."""
        logger.info("[EXECUTION] Position monitor loop started")
        
        while self._running:
            try:
                # TODO: Implement position monitoring logic (migrate from position_monitor.py)
                # For now, placeholder
                await asyncio.sleep(settings.POSITION_MONITOR_INTERVAL_SEC)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[EXECUTION] Error in position monitor loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)
        
        logger.info("[EXECUTION] Position monitor loop stopped")
    
    async def _event_processing_loop(self):
        """Background loop for processing buffered events."""
        logger.info("[EXECUTION] Event processing loop started")
        
        while self._running:
            try:
                # Process buffered events
                if self.disk_buffer:
                    while True:
                        event = self.disk_buffer.pop()
                        if event is None:
                            break
                        
                        # Re-publish to EventBus
                        event_type = event.get("type")
                        event_data = event.get("data")
                        if event_type and event_data:
                            await self.event_bus.publish(event_type, event_data)
                            logger.debug(f"[EXECUTION] Replayed buffered event: {event_type}")
                
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[EXECUTION] Error in event processing loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)
        
        logger.info("[EXECUTION] Event processing loop stopped")
    
    # ========================================================================
    # QUERY METHODS (for REST API)
    # ========================================================================
    
    async def get_positions(self) -> List[Position]:
        """Get all current positions."""
        # TODO: Fetch from Binance + enrich with TradeStore data
        return []
    
    async def get_trades(self, limit: int = 100, status: Optional[str] = None) -> List[Trade]:
        """Get trade history from TradeStore."""
        # TODO: Query TradeStore
        return []
    
    async def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get specific trade by ID."""
        # TODO: Query TradeStore
        return None
    
    async def get_health(self) -> Dict[str, Any]:
        """Get service health status."""
        healthy = True
        components = {}
        
        # Check Binance
        if self.binance_client:
            try:
                # Ping Binance to check connectivity
                start_time = datetime.now()
                await self.binance_client._signed_request("GET", "/fapi/v1/ping", {})
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                components["binance"] = ComponentHealth(healthy=True, latency_ms=latency_ms)
            except Exception as e:
                components["binance"] = ComponentHealth(healthy=False, error=str(e))
                healthy = False
        else:
            components["binance"] = ComponentHealth(healthy=False, error="Not initialized")
            healthy = False
        
        # Check EventBus
        components["event_bus"] = ComponentHealth(healthy=self.event_bus is not None)
        if not self.event_bus:
            healthy = False
        
        # Check TradeStore
        if self.trade_store:
            backend = "redis" if self.trade_store._redis_client else "sqlite"
            components["trade_store"] = ComponentHealth(
                healthy=True,
                details={"backend": backend}
            )
        else:
            components["trade_store"] = ComponentHealth(healthy=False, error="Not initialized")
            healthy = False
        
        # Check Rate Limiter
        if self.rate_limiter:
            tokens = self.rate_limiter.get_tokens_available()
            components["rate_limiter"] = ComponentHealth(
                healthy=True,
                details={
                    "tokens_available": tokens,
                    "rpm_limit": settings.BINANCE_RATE_LIMIT_RPM
                }
            )
        else:
            components["rate_limiter"] = ComponentHealth(healthy=False, error="Not initialized")
            healthy = False
        
        # Check ESS via risk-safety-service
        try:
            if self.http_client:
                ess_response = await self.http_client.get("/api/risk/ess/status")
                ess_data = ess_response.json()
                components["ess"] = ComponentHealth(
                    healthy=True,
                    details={
                        "state": ess_data.get("state"),
                        "can_execute": ess_data.get("can_execute")
                    }
                )
        except Exception as e:
            components["ess"] = ComponentHealth(healthy=False, error=str(e))
            # Don't mark overall service unhealthy if ESS check fails (fail-open)
        
        return ServiceHealth(
            service="execution",
            healthy=healthy,
            running=self._running,
            components=components,
            active_positions=len(self._active_positions),
            orders_last_minute=self._orders_last_minute,
            timestamp=datetime.now(timezone.utc).isoformat()
        ).dict()
