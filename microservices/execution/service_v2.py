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
            logger.info("[EXECUTION-V2] Subscribing to 'trade.intent' events...")
            self.event_bus.subscribe("trade.intent", self._handle_trade_signal)
            logger.info("[EXECUTION-V2] ✅ Event subscription active")
            
            # 6. Start EventBus consumer loop
            await self.event_bus.start()
            logger.info("[EXECUTION-V2] ✅ EventBus consumer loop started")
            
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
        try:
            logger.info(f"[EXECUTION-V2] 📥 Received trade signal: {event_data}")
            
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
            
            # 2. Rate limiting
            await self.rate_limiter.acquire()
            
            # 3. Calculate quantity
            # For futures: quantity = position_size_usd / entry_price
            quantity = signal.position_size_usd / signal.entry_price
            
            # Round to appropriate precision (Binance requires specific precision)
            # For now, round to 3 decimals
            quantity = round(quantity, 3)
            
            logger.info(
                f"[EXECUTION-V2] {trade_id} - Executing: {signal.symbol} {signal.side} "
                f"{quantity} @ ~{signal.entry_price} (leverage: {signal.leverage}x)"
            )
            
            # 4. Execute order
            order_result = await self.binance.place_market_order(
                symbol=signal.symbol,
                side=signal.side.upper(),
                quantity=quantity,
                leverage=signal.leverage or 1
            )
            
            # 5. Store trade
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
            
            # 6. Update position tracking
            if order_result["status"] == "FILLED":
                self._update_position(signal.symbol, signal.side, quantity, order_result["price"])
            
            # 7. Publish result
            await self._publish_execution_result({
                "trade_id": trade_id,
                "status": order_result["status"],
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
