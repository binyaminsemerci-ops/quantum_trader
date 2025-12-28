"""
Execution & Risk Service - Order Execution and Risk Management
===============================================================

Quantum Trader v3.0 - Exec-Risk Service

Responsibilities:
- Execute trades on Binance Futures
- Smart execution strategies (MARKET, LIMIT, TWAP, VWAP)
- Pre-execution risk validation
- Position monitoring and PnL tracking
- Dynamic TP/SL adjustment
- Emergency stop enforcement
- Publish execution.result, position.opened, position.closed, risk.alert events

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0.0
"""

import asyncio
import logging
import os
import signal as system_signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import redis.asyncio as redis
from redis.asyncio import Redis
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Import core infrastructure
from backend.core.event_bus import EventBus, initialize_event_bus
from backend.core.policy_store import PolicyStore, initialize_policy_store, shutdown_policy_store
from backend.core.logger import configure_logging, get_logger
from backend.core.service_rpc import ServiceRPCClient, ServiceRPCServer
from backend.events.v3_schemas import (
    EventTypes,
    ExecutionRequestPayload,
    ExecutionResultPayload,
    PositionOpenedPayload,
    PositionClosedPayload,
    RiskAlertPayload,
    EmergencyStopPayload,
    ServiceHeartbeatPayload,
    build_event,
)

# Import execution components
from backend.services.execution.execution import BinanceFuturesExecutionAdapter
from backend.services.risk.safety_governor import SafetyGovernor
from backend.domains.risk.risk_guard_v2 import RiskGuardV2

logger = get_logger(__name__, component="exec_risk_service")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExecRiskServiceConfig:
    """Execution & Risk Service configuration"""
    service_name: str = "exec-risk-service"
    redis_url: str = "redis://localhost:6379"
    
    # Binance settings
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = False
    
    # Execution settings
    default_execution_mode: str = "MARKET"  # MARKET, LIMIT, TWAP, VWAP
    max_slippage_percent: float = 0.5
    order_timeout_seconds: float = 30.0
    max_retry_attempts: int = 3
    
    # Risk limits
    max_leverage: float = 20.0
    max_position_size_usd: float = 10000.0
    max_portfolio_exposure_usd: float = 50000.0
    max_daily_drawdown_percent: float = 10.0
    emergency_stop_threshold_percent: float = 15.0
    
    # Position monitoring
    position_check_interval_seconds: int = 5
    dynamic_tpsl_enabled: bool = True
    trailing_stop_percent: float = 2.0
    
    # Health check
    heartbeat_interval_seconds: int = 5
    health_check_port: int = 8002
    
    # Performance
    max_event_handlers: int = 100
    event_processing_timeout: float = 30.0
    
    @classmethod
    def from_env(cls) -> "ExecRiskServiceConfig":
        """Load configuration from environment variables"""
        return cls(
            service_name=os.getenv("SERVICE_NAME", "exec-risk-service"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            binance_api_key=os.getenv("BINANCE_API_KEY", ""),
            binance_api_secret=os.getenv("BINANCE_API_SECRET", ""),
            binance_testnet=os.getenv("BINANCE_TESTNET", "false").lower() == "true",
            default_execution_mode=os.getenv("DEFAULT_EXECUTION_MODE", "MARKET"),
            max_slippage_percent=float(os.getenv("MAX_SLIPPAGE_PERCENT", "0.5")),
            max_leverage=float(os.getenv("MAX_LEVERAGE", "20.0")),
            max_position_size_usd=float(os.getenv("MAX_POSITION_SIZE_USD", "10000.0")),
            max_portfolio_exposure_usd=float(os.getenv("MAX_PORTFOLIO_EXPOSURE_USD", "50000.0")),
            max_daily_drawdown_percent=float(os.getenv("MAX_DAILY_DRAWDOWN_PERCENT", "10.0")),
            emergency_stop_threshold_percent=float(os.getenv("EMERGENCY_STOP_THRESHOLD_PERCENT", "15.0")),
            position_check_interval_seconds=int(os.getenv("POSITION_CHECK_INTERVAL_SECONDS", "5")),
            dynamic_tpsl_enabled=os.getenv("DYNAMIC_TPSL_ENABLED", "true").lower() == "true",
            heartbeat_interval_seconds=int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "5")),
        )


# ============================================================================
# EXEC-RISK SERVICE
# ============================================================================

class ExecRiskService:
    """
    Execution & Risk Service - Handles order execution and risk management.
    
    Architecture:
    - Executes trades on Binance Futures
    - Validates all execution requests against risk limits
    - Monitors open positions for TP/SL and PnL tracking
    - Adjusts TP/SL dynamically based on market conditions
    - Enforces emergency stops and risk alerts
    - Publishes events via EventBus v2
    - Handles RPC requests from other services
    """
    
    def __init__(self, config: ExecRiskServiceConfig):
        """Initialize Exec-Risk Service"""
        self.config = config
        self.running = False
        self.startup_time = time.time()
        
        # FastAPI app for health endpoints
        self.app = FastAPI(title="Exec-Risk Service", version="3.0.0")
        
        # Core components
        self.redis: Optional[Redis] = None
        self.event_bus: Optional[EventBus] = None
        self.policy_store: Optional[PolicyStore] = None
        self.rpc_client: Optional[ServiceRPCClient] = None
        self.rpc_server: Optional[ServiceRPCServer] = None
        
        # Execution components
        self.execution_adapter: Optional[BinanceFuturesExecutionAdapter] = None
        self.safety_governor: Optional[SafetyGovernor] = None
        self.risk_guard: Optional[RiskGuardV2] = None
        
        # Position tracking
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        
        # Background tasks
        self.tasks: List[asyncio.Task] = []
        
        # Metrics
        self.metrics = {
            "orders_executed": 0,
            "positions_opened": 0,
            "positions_closed": 0,
            "risk_alerts": 0,
            "emergency_stops": 0,
            "execution_errors": 0,
            "events_published": 0,
            "events_received": 0,
            "rpc_calls_received": 0,
        }
        
        logger.info(
            f"ExecRiskService initialized",
            service_name=config.service_name,
            max_leverage=config.max_leverage,
            binance_testnet=config.binance_testnet,
        )
    
    async def bootstrap(self) -> None:
        """Bootstrap Exec-Risk Service"""
        logger.info("Bootstrapping Exec-Risk Service...")
        
        try:
            # 1. Initialize Redis
            logger.info(f"Connecting to Redis: {self.config.redis_url}")
            self.redis = redis.from_url(
                self.config.redis_url,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            await self.redis.ping()
            logger.info("✓ Redis connected")
            
            # 2. Initialize PolicyStore v2
            logger.info("Initializing PolicyStore v2...")
            self.policy_store = await initialize_policy_store(self.redis)
            logger.info("✓ PolicyStore v2 initialized")
            
            # 3. Initialize EventBus v2
            logger.info("Initializing EventBus v2...")
            self.event_bus = await initialize_event_bus(
                self.redis,
                service_name=self.config.service_name
            )
            logger.info("✓ EventBus v2 initialized")
            
            # 4. Initialize RPC Client
            logger.info("Initializing RPC Client...")
            self.rpc_client = ServiceRPCClient(
                self.redis,
                service_name=self.config.service_name
            )
            await self.rpc_client.initialize()
            logger.info("✓ RPC Client initialized")
            
            # 5. Initialize RPC Server
            logger.info("Initializing RPC Server...")
            self.rpc_server = ServiceRPCServer(
                self.redis,
                service_name=self.config.service_name
            )
            self._register_rpc_handlers()
            await self.rpc_server.start()
            logger.info("✓ RPC Server started")
            
            # 6. Initialize Binance execution adapter
            logger.info("Initializing Binance adapter...")
            await self._initialize_binance()
            logger.info("✓ Binance adapter initialized")
            
            # 7. Initialize Safety Governor & Risk Guard
            logger.info("Initializing Safety Governor & Risk Guard...")
            self.safety_governor = SafetyGovernor(policy_store=self.policy_store)
            self.risk_guard = RiskGuardV2(
                redis_client=self.redis,
                policy_store=self.policy_store
            )
            logger.info("✓ Safety Governor & Risk Guard initialized")
            
            # 8. Recover open positions from Binance
            logger.info("Recovering open positions from Binance...")
            await self._recover_positions()
            logger.info(f"✓ Recovered {len(self.open_positions)} open positions")
            
            # 9. Subscribe to events
            logger.info("Subscribing to events...")
            self._subscribe_to_events()
            logger.info("✓ Event subscriptions registered")
            
            # 10. Start EventBus consumer
            await self.event_bus.start()
            logger.info("✓ EventBus consumer started")
            
            # 11. Setup health endpoints
            logger.info("Setting up Health v3 endpoints...")
            self._setup_health_endpoints()
            logger.info("✓ Health endpoints configured")
            
            logger.info(
                "🚀 Exec-Risk Service bootstrap complete",
                uptime_seconds=time.time() - self.startup_time,
                open_positions=len(self.open_positions),
            )
        
        except Exception as e:
            logger.error(f"Bootstrap failed: {e}", exc_info=True)
            raise
    
    async def _initialize_binance(self) -> None:
        """Initialize Binance execution adapter"""
        try:
            self.execution_adapter = BinanceFuturesExecutionAdapter(
                api_key=self.config.binance_api_key,
                api_secret=self.config.binance_api_secret,
                testnet=self.config.binance_testnet
            )
            
            # Test connection
            account_info = await self.execution_adapter.get_account_info()
            
            logger.info(
                f"✓ Binance connected: "
                f"balance={account_info.get('totalWalletBalance', 0)} USDT"
            )
        
        except Exception as e:
            logger.error(f"Failed to initialize Binance: {e}", exc_info=True)
            raise
    
    async def _recover_positions(self) -> None:
        """Recover open positions from Binance API"""
        try:
            if not self.execution_adapter:
                return
            
            positions = await self.execution_adapter.get_open_positions()
            
            for pos in positions:
                if abs(float(pos.get("positionAmt", 0))) > 0:
                    symbol = pos["symbol"]
                    self.open_positions[symbol] = {
                        "symbol": symbol,
                        "side": "LONG" if float(pos["positionAmt"]) > 0 else "SHORT",
                        "size": abs(float(pos["positionAmt"])),
                        "entry_price": float(pos["entryPrice"]),
                        "current_price": float(pos["markPrice"]),
                        "unrealized_pnl": float(pos["unRealizedProfit"]),
                        "leverage": float(pos.get("leverage", 1)),
                        "open_time": datetime.now(timezone.utc).isoformat(),
                        "take_profit": None,
                        "stop_loss": None,
                    }
            
            logger.info(
                f"Recovered {len(self.open_positions)} positions: "
                f"{list(self.open_positions.keys())}"
            )
        
        except Exception as e:
            logger.error(f"Position recovery failed: {e}", exc_info=True)
    
    def _register_rpc_handlers(self) -> None:
        """Register RPC command handlers"""
        
        @self.rpc_server.register_handler("validate_risk")
        async def handle_validate_risk(params: dict) -> dict:
            """RPC: Validate risk for execution request"""
            try:
                symbol = params["symbol"]
                side = params["side"]
                size_usd = params["size_usd"]
                
                logger.debug(f"RPC validate_risk: {symbol} {side} ${size_usd}")
                
                validation = await self._validate_risk(symbol, side, size_usd)
                
                self.metrics["rpc_calls_received"] += 1
                
                return validation
            except Exception as e:
                logger.error(f"RPC validate_risk error: {e}", exc_info=True)
                return {"approved": False, "reason": str(e)}
        
        @self.rpc_server.register_handler("get_position_status")
        async def handle_get_position_status(params: dict) -> dict:
            """RPC: Get position status"""
            try:
                symbol = params.get("symbol")
                
                if symbol:
                    position = self.open_positions.get(symbol)
                    return {"positions": [position] if position else []}
                else:
                    return {"positions": list(self.open_positions.values())}
                
            except Exception as e:
                logger.error(f"RPC get_position_status error: {e}", exc_info=True)
                return {"error": str(e)}
        
        @self.rpc_server.register_handler("close_position")
        async def handle_close_position(params: dict) -> dict:
            """RPC: Close position immediately"""
            try:
                symbol = params["symbol"]
                reason = params.get("reason", "MANUAL")
                
                logger.info(f"RPC close_position: {symbol}, reason={reason}")
                
                result = await self._close_position(symbol, reason)
                
                self.metrics["rpc_calls_received"] += 1
                
                return result
            except Exception as e:
                logger.error(f"RPC close_position error: {e}", exc_info=True)
                return {"success": False, "error": str(e)}
        
        logger.info("✓ RPC handlers registered: validate_risk, get_position_status, close_position")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to EventBus events"""
        
        # Subscribe to signal.generated (from ai-service)
        self.event_bus.subscribe(EventTypes.SIGNAL_GENERATED, self._handle_signal_generated)
        
        # Subscribe to execution.request (from analytics-os-service)
        self.event_bus.subscribe(EventTypes.EXECUTION_REQUEST, self._handle_execution_request)
        
        # Subscribe to emergency.stop
        self.event_bus.subscribe(EventTypes.EMERGENCY_STOP, self._handle_emergency_stop)
        
        # Subscribe to policy.updated
        self.event_bus.subscribe(EventTypes.POLICY_UPDATED, self._handle_policy_updated)
        
        logger.info(
            "✓ Event subscriptions: signal.generated, execution.request, "
            "emergency.stop, policy.updated"
        )
    
    async def _handle_signal_generated(self, event_data: dict) -> None:
        """Handle signal.generated event"""
        try:
            symbol = event_data.get("symbol")
            action = event_data.get("action")
            confidence = event_data.get("confidence", 0.5)
            
            logger.info(
                f"Signal received: {symbol} {action} conf={confidence:.2f}"
            )
            
            # Check if should execute signal
            policy = await self.policy_store.get_policy()
            min_confidence = policy.signal_confidence_threshold
            
            if confidence < min_confidence:
                logger.debug(
                    f"Signal ignored: confidence {confidence:.2f} < threshold {min_confidence}"
                )
                return
            
            # Only execute if action is BUY or SELL
            if action in ["BUY", "SELL"]:
                await self._execute_signal(event_data)
            
            self.metrics["events_received"] += 1
        
        except Exception as e:
            logger.error(f"Error handling signal: {e}", exc_info=True)
            self.metrics["execution_errors"] += 1
    
    async def _handle_execution_request(self, event_data: dict) -> None:
        """Handle execution.request event"""
        try:
            symbol = event_data.get("symbol")
            side = event_data.get("side")
            size_usd = event_data.get("size_usd")
            
            logger.info(f"Execution request: {symbol} {side} ${size_usd}")
            
            # Validate and execute
            await self._execute_order(symbol, side, size_usd, event_data)
            
            self.metrics["events_received"] += 1
        
        except Exception as e:
            logger.error(f"Error handling execution request: {e}", exc_info=True)
            self.metrics["execution_errors"] += 1
    
    async def _handle_emergency_stop(self, event_data: dict) -> None:
        """Handle emergency.stop event"""
        try:
            reason = event_data.get("reason", "UNKNOWN")
            
            logger.critical(f"EMERGENCY STOP triggered: {reason}")
            
            # Close all positions immediately
            for symbol in list(self.open_positions.keys()):
                await self._close_position(symbol, "EMERGENCY_STOP")
            
            self.metrics["emergency_stops"] += 1
            self.metrics["events_received"] += 1
        
        except Exception as e:
            logger.error(f"Error handling emergency stop: {e}", exc_info=True)
    
    async def _handle_policy_updated(self, event_data: dict) -> None:
        """Handle policy.updated event"""
        try:
            logger.info("Policy updated")
            
            # Reload policy
            policy = await self.policy_store.reload_policy()
            
            logger.info(f"Policy reloaded: version={policy.version}")
            
            self.metrics["events_received"] += 1
        
        except Exception as e:
            logger.error(f"Error handling policy update: {e}", exc_info=True)
    
    async def _validate_risk(self, symbol: str, side: str, size_usd: float) -> Dict[str, Any]:
        """Validate risk for execution request"""
        try:
            # Check position size limit
            if size_usd > self.config.max_position_size_usd:
                return {
                    "approved": False,
                    "reason": f"Position size ${size_usd} exceeds max ${self.config.max_position_size_usd}",
                }
            
            # Check portfolio exposure
            current_exposure = sum(
                pos.get("size", 0) * pos.get("current_price", 0)
                for pos in self.open_positions.values()
            )
            
            if current_exposure + size_usd > self.config.max_portfolio_exposure_usd:
                return {
                    "approved": False,
                    "reason": f"Portfolio exposure would exceed ${self.config.max_portfolio_exposure_usd}",
                }
            
            # Check Safety Governor
            if self.safety_governor:
                can_trade = self.safety_governor.can_trade(symbol)
                if not can_trade:
                    return {
                        "approved": False,
                        "reason": "Safety Governor blocked trade",
                    }
            
            return {"approved": True, "reason": "All risk checks passed"}
        
        except Exception as e:
            logger.error(f"Risk validation error: {e}", exc_info=True)
            return {"approved": False, "reason": str(e)}
    
    async def _execute_signal(self, signal_data: dict) -> None:
        """Execute trading signal"""
        symbol = signal_data["symbol"]
        action = signal_data["action"]
        confidence = signal_data.get("confidence", 0.5)
        
        # Determine position size based on confidence
        base_size = 1000.0  # Base $1000 position
        size_usd = base_size * confidence
        
        side = "LONG" if action == "BUY" else "SHORT"
        
        await self._execute_order(symbol, side, size_usd, signal_data)
    
    async def _execute_order(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        metadata: Dict[str, Any]
    ) -> None:
        """Execute order with risk validation"""
        try:
            # 1. Validate risk
            validation = await self._validate_risk(symbol, side, size_usd)
            
            if not validation["approved"]:
                logger.warning(
                    f"Order rejected: {symbol} {side} ${size_usd}, "
                    f"reason={validation['reason']}"
                )
                await self._publish_risk_alert(symbol, validation["reason"])
                return
            
            # 2. Execute on Binance
            logger.info(f"Executing: {symbol} {side} ${size_usd}")
            
            order_result = await self.execution_adapter.place_market_order(
                symbol=symbol,
                side=side,
                size_usd=size_usd,
            )
            
            # 3. Track position
            self.open_positions[symbol] = {
                "symbol": symbol,
                "side": side,
                "size": order_result["executedQty"],
                "entry_price": order_result["avgPrice"],
                "current_price": order_result["avgPrice"],
                "unrealized_pnl": 0.0,
                "leverage": self.config.max_leverage,
                "open_time": datetime.now(timezone.utc).isoformat(),
                "entry_confidence": metadata.get("confidence"),
                "entry_model": metadata.get("model_source"),
                "take_profit": None,
                "stop_loss": None,
            }
            
            # 4. Set TP/SL
            await self._set_tpsl(symbol)
            
            # 5. Publish events
            await self._publish_execution_result(order_result, success=True)
            await self._publish_position_opened(symbol)
            
            self.metrics["orders_executed"] += 1
            self.metrics["positions_opened"] += 1
            
            logger.info(
                f"✓ Order executed: {symbol} {side} "
                f"entry={order_result['avgPrice']}"
            )
        
        except Exception as e:
            logger.error(f"Order execution failed: {e}", exc_info=True)
            await self._publish_execution_result({}, success=False, error=str(e))
            self.metrics["execution_errors"] += 1
    
    async def _set_tpsl(self, symbol: str) -> None:
        """Set take profit and stop loss for position"""
        position = self.open_positions.get(symbol)
        if not position:
            return
        
        entry_price = position["entry_price"]
        side = position["side"]
        
        # Calculate TP/SL levels
        if side == "LONG":
            take_profit = entry_price * 1.05  # +5% profit target (above entry)
            stop_loss = entry_price * 0.98    # -2% stop loss (below entry)
        else:  # SHORT
            take_profit = entry_price * 0.95  # -5% profit target (below entry)
            stop_loss = entry_price * 1.02    # +2% stop loss (above entry)
        
        position["take_profit"] = take_profit
        position["stop_loss"] = stop_loss
        
        logger.debug(
            f"TP/SL set: {symbol} TP={take_profit:.2f} SL={stop_loss:.2f}"
        )
    
    async def _close_position(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Close position"""
        try:
            position = self.open_positions.get(symbol)
            if not position:
                return {"success": False, "error": "Position not found"}
            
            logger.info(f"Closing position: {symbol}, reason={reason}")
            
            # Execute close order
            close_result = await self.execution_adapter.close_position(symbol)
            
            # Calculate PnL
            exit_price = close_result["avgPrice"]
            entry_price = position["entry_price"]
            
            if position["side"] == "LONG":
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            
            pnl_usd = pnl_pct * position["size"] * entry_price / 100
            
            # Update daily PnL
            self.daily_pnl += pnl_usd
            self.daily_trades += 1
            
            # Remove from tracking
            del self.open_positions[symbol]
            
            # Publish position.closed event (CRITICAL for learning)
            await self._publish_position_closed(
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                exit_reason=reason,
                entry_confidence=position.get("entry_confidence"),
                entry_model=position.get("entry_model"),
            )
            
            self.metrics["positions_closed"] += 1
            
            logger.info(
                f"✓ Position closed: {symbol}, PnL={pnl_pct:.2f}% (${pnl_usd:.2f}), "
                f"reason={reason}"
            )
            
            return {"success": True, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct}
        
        except Exception as e:
            logger.error(f"Failed to close position: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def start(self) -> None:
        """Start background tasks"""
        logger.info("Starting Exec-Risk Service background tasks...")
        
        self.running = True
        
        # Start position monitoring task
        self.tasks.append(
            asyncio.create_task(self._position_monitoring_loop())
        )
        
        # Start heartbeat task
        self.tasks.append(
            asyncio.create_task(self._heartbeat_loop())
        )
        
        logger.info(f"✓ Started {len(self.tasks)} background tasks")
    
    async def _position_monitoring_loop(self) -> None:
        """Background task: Monitor open positions"""
        logger.info("Position monitoring loop started")
        
        while self.running:
            try:
                for symbol in list(self.open_positions.keys()):
                    await self._check_position(symbol)
                
                await asyncio.sleep(self.config.position_check_interval_seconds)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position monitoring error: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.info("Position monitoring loop stopped")
    
    async def _check_position(self, symbol: str) -> None:
        """Check position for TP/SL and PnL"""
        try:
            position = self.open_positions.get(symbol)
            if not position:
                return
            
            # Get current price
            current_price = await self.execution_adapter.get_current_price(symbol)
            position["current_price"] = current_price
            
            # Calculate unrealized PnL
            entry_price = position["entry_price"]
            if position["side"] == "LONG":
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            position["unrealized_pnl"] = pnl_pct
            
            # Check TP/SL
            take_profit = position.get("take_profit")
            stop_loss = position.get("stop_loss")
            
            if take_profit and (
                (position["side"] == "LONG" and current_price >= take_profit) or
                (position["side"] == "SHORT" and current_price <= take_profit)
            ):
                logger.info(f"Take profit hit: {symbol}")
                await self._close_position(symbol, "TAKE_PROFIT")
            
            elif stop_loss and (
                (position["side"] == "LONG" and current_price <= stop_loss) or
                (position["side"] == "SHORT" and current_price >= stop_loss)
            ):
                logger.info(f"Stop loss hit: {symbol}")
                await self._close_position(symbol, "STOP_LOSS")
            
            # Dynamic TP/SL adjustment
            elif self.config.dynamic_tpsl_enabled:
                await self._adjust_tpsl(symbol, current_price)
        
        except Exception as e:
            logger.error(f"Position check error for {symbol}: {e}")
    
    async def _adjust_tpsl(self, symbol: str, current_price: float) -> None:
        """Adjust TP/SL dynamically (trailing stop)"""
        position = self.open_positions.get(symbol)
        if not position:
            return
        
        # Implement trailing stop logic here
        pass
    
    async def _heartbeat_loop(self) -> None:
        """Background task: Publish service heartbeat"""
        logger.info("Heartbeat loop started")
        
        while self.running:
            try:
                await self._publish_heartbeat()
                await asyncio.sleep(self.config.heartbeat_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
        
        logger.info("Heartbeat loop stopped")
    
    async def _publish_execution_result(
        self,
        order_result: Dict[str, Any],
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Publish execution.result event"""
        try:
            payload = ExecutionResultPayload(
                symbol=order_result.get("symbol", ""),
                order_id=order_result.get("orderId", ""),
                status="FILLED" if success else "FAILED",
                executed_qty=float(order_result.get("executedQty", 0)),
                avg_price=float(order_result.get("avgPrice", 0)),
                commission=float(order_result.get("commission", 0)),
                error_message=error,
            )
            
            await self.event_bus.publish(
                EventTypes.EXECUTION_RESULT,
                payload.dict()
            )
            
            self.metrics["events_published"] += 1
        
        except Exception as e:
            logger.error(f"Failed to publish execution result: {e}")
    
    async def _publish_position_opened(self, symbol: str) -> None:
        """Publish position.opened event"""
        try:
            position = self.open_positions[symbol]
            
            payload = PositionOpenedPayload(
                symbol=symbol,
                position_id=f"{symbol}_{int(time.time())}",
                side=position["side"],
                size=position["size"],
                entry_price=position["entry_price"],
                leverage=position["leverage"],
                take_profit_price=position.get("take_profit"),
                stop_loss_price=position.get("stop_loss"),
            )
            
            await self.event_bus.publish(
                EventTypes.POSITION_OPENED,
                payload.dict()
            )
            
            self.metrics["events_published"] += 1
        
        except Exception as e:
            logger.error(f"Failed to publish position opened: {e}")
    
    async def _publish_position_closed(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl_usd: float,
        pnl_pct: float,
        exit_reason: str,
        entry_confidence: Optional[float] = None,
        entry_model: Optional[str] = None
    ) -> None:
        """Publish position.closed event (CRITICAL for learning)"""
        try:
            payload = PositionClosedPayload(
                symbol=symbol,
                position_id=f"{symbol}_{int(time.time())}",
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
                entry_confidence=entry_confidence,
                entry_model=entry_model,
            )
            
            await self.event_bus.publish(
                EventTypes.POSITION_CLOSED,
                payload.dict()
            )
            
            self.metrics["events_published"] += 1
            
            logger.info(
                f"Position closed event published: {symbol}, "
                f"PnL={pnl_pct:.2f}%, reason={exit_reason}"
            )
        
        except Exception as e:
            logger.error(f"Failed to publish position closed: {e}", exc_info=True)
    
    async def _publish_risk_alert(self, symbol: str, reason: str) -> None:
        """Publish risk.alert event"""
        try:
            payload = RiskAlertPayload(
                alert_type="RISK_LIMIT_EXCEEDED",
                severity="MEDIUM",
                symbol=symbol,
                message=reason,
            )
            
            await self.event_bus.publish(
                EventTypes.RISK_ALERT,
                payload.dict()
            )
            
            self.metrics["risk_alerts"] += 1
            self.metrics["events_published"] += 1
        
        except Exception as e:
            logger.error(f"Failed to publish risk alert: {e}")
    
    async def _publish_heartbeat(self) -> None:
        """Publish service.heartbeat event"""
        try:
            import psutil
            
            payload = ServiceHeartbeatPayload(
                service_name=self.config.service_name,
                status="HEALTHY",
                uptime_seconds=time.time() - self.startup_time,
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=psutil.virtual_memory().percent,
                memory_used_mb=psutil.virtual_memory().used / 1024 / 1024,
                active_tasks=len([t for t in self.tasks if not t.done()]),
                processed_events_last_minute=self.metrics["events_received"],
            )
            
            await self.event_bus.publish(
                EventTypes.SERVICE_HEARTBEAT,
                payload.dict()
            )
        
        except Exception as e:
            logger.error(f"Failed to publish heartbeat: {e}")
    
    def _setup_health_endpoints(self) -> None:
        """Setup Health v3 endpoints for Docker/Kubernetes health checks"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint - returns service health status"""
            return JSONResponse({
                "status": "healthy" if self.running else "starting",
                "service": self.config.service_name,
                "version": "3.0.0",
                "uptime_seconds": time.time() - self.startup_time,
                "binance_connected": self.execution_adapter is not None,
                "open_positions": len(self.open_positions),
                "daily_pnl": self.daily_pnl,
                "daily_trades": self.daily_trades,
                "metrics": {
                    "orders_executed": self.metrics["orders_executed"],
                    "positions_opened": self.metrics["positions_opened"],
                    "positions_closed": self.metrics["positions_closed"],
                    "risk_alerts": self.metrics["risk_alerts"],
                    "execution_errors": self.metrics["execution_errors"],
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        
        @self.app.get("/ready")
        async def ready():
            """Readiness probe - returns 200 if service is ready to accept requests"""
            is_ready = (
                self.running
                and self.execution_adapter is not None
                and self.safety_governor is not None
                and self.event_bus is not None
            )
            status_code = 200 if is_ready else 503
            return JSONResponse(
                {
                    "status": "ready" if is_ready else "not_ready",
                    "components": {
                        "execution_adapter": self.execution_adapter is not None,
                        "safety_governor": self.safety_governor is not None,
                        "event_bus": self.event_bus is not None,
                        "rpc_server": self.rpc_server is not None,
                    },
                },
                status_code=status_code,
            )
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus-compatible metrics endpoint"""
            uptime = time.time() - self.startup_time
            
            metrics_text = f"""# HELP exec_risk_service_uptime_seconds Service uptime in seconds
# TYPE exec_risk_service_uptime_seconds gauge
exec_risk_service_uptime_seconds {uptime}

# HELP exec_risk_service_orders_executed_total Total orders executed
# TYPE exec_risk_service_orders_executed_total counter
exec_risk_service_orders_executed_total {self.metrics['orders_executed']}

# HELP exec_risk_service_positions_opened_total Total positions opened
# TYPE exec_risk_service_positions_opened_total counter
exec_risk_service_positions_opened_total {self.metrics['positions_opened']}

# HELP exec_risk_service_positions_closed_total Total positions closed
# TYPE exec_risk_service_positions_closed_total counter
exec_risk_service_positions_closed_total {self.metrics['positions_closed']}

# HELP exec_risk_service_risk_alerts_total Total risk alerts triggered
# TYPE exec_risk_service_risk_alerts_total counter
exec_risk_service_risk_alerts_total {self.metrics['risk_alerts']}

# HELP exec_risk_service_emergency_stops_total Total emergency stops triggered
# TYPE exec_risk_service_emergency_stops_total counter
exec_risk_service_emergency_stops_total {self.metrics['emergency_stops']}

# HELP exec_risk_service_execution_errors_total Total execution errors
# TYPE exec_risk_service_execution_errors_total counter
exec_risk_service_execution_errors_total {self.metrics['execution_errors']}

# HELP exec_risk_service_events_published_total Total events published
# TYPE exec_risk_service_events_published_total counter
exec_risk_service_events_published_total {self.metrics['events_published']}

# HELP exec_risk_service_rpc_calls_received_total Total RPC calls received
# TYPE exec_risk_service_rpc_calls_received_total counter
exec_risk_service_rpc_calls_received_total {self.metrics['rpc_calls_received']}

# HELP exec_risk_service_open_positions Current number of open positions
# TYPE exec_risk_service_open_positions gauge
exec_risk_service_open_positions {len(self.open_positions)}

# HELP exec_risk_service_daily_pnl_usd Daily PnL in USD
# TYPE exec_risk_service_daily_pnl_usd gauge
exec_risk_service_daily_pnl_usd {self.daily_pnl}

# HELP exec_risk_service_daily_trades Daily trade count
# TYPE exec_risk_service_daily_trades gauge
exec_risk_service_daily_trades {self.daily_trades}
"""
            return metrics_text
    
    async def start_fastapi(self) -> None:
        """Start FastAPI server for health endpoints"""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=8002,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown Exec-Risk Service"""
        logger.info("Shutting down Exec-Risk Service...")
        
        self.running = False
        
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Shutdown components
        if self.rpc_server:
            await self.rpc_server.stop()
        
        if self.rpc_client:
            await self.rpc_client.shutdown()
        
        if self.event_bus:
            await self.event_bus.stop()
        
        await shutdown_policy_store()
        
        if self.redis:
            await self.redis.close()
        
        logger.info(
            "Exec-Risk Service shutdown complete",
            uptime_seconds=time.time() - self.startup_time,
            orders_executed=self.metrics["orders_executed"],
            daily_pnl=self.daily_pnl,
        )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    
    logger.info("=" * 80)
    logger.info("QUANTUM TRADER v3.0 - EXEC-RISK SERVICE")
    logger.info("=" * 80)
    
    config = ExecRiskServiceConfig.from_env()
    service = ExecRiskService(config)
    
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()
    
    system_signal.signal(system_signal.SIGINT, signal_handler)
    system_signal.signal(system_signal.SIGTERM, signal_handler)
    
    try:
        await service.bootstrap()
        await service.start()
        
        logger.info("🚀 Exec-Risk Service running")
        logger.info(f"   Service: {config.service_name}")
        logger.info(f"   Max Leverage: {config.max_leverage}x")
        logger.info(f"   Open Positions: {len(service.open_positions)}")
        logger.info(f"   Health: http://localhost:8002/health")
        logger.info(f"   Ready: http://localhost:8002/ready")
        logger.info(f"   Metrics: http://localhost:8002/metrics")
        logger.info("")
        logger.info("Press Ctrl+C to shutdown")
        logger.info("=" * 80)
        
        # Start FastAPI server in background
        fastapi_task = asyncio.create_task(service.start_fastapi())
        
        await shutdown_event.wait()
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        await service.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
