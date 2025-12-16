"""
Portfolio Intelligence Service - Core Logic
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import httpx

from microservices.portfolio_intelligence.config import settings
from microservices.portfolio_intelligence.models import (
    PositionInfo, PortfolioSnapshot, PnLBreakdown, ExposureBreakdown, DrawdownMetrics,
    PortfolioSnapshotUpdatedEvent, PortfolioPnLUpdatedEvent,
    PortfolioDrawdownUpdatedEvent, PortfolioExposureUpdatedEvent,
    ComponentHealth
)
from backend.core.event_bus import EventBus
from backend.core.event_buffer import EventBuffer

logger = logging.getLogger(__name__)


class PortfolioIntelligenceService:
    """
    Portfolio Intelligence Core Service
    
    Responsibilities:
    - Aggregate portfolio data from TradeStore + Binance
    - Calculate equity, PnL, exposure, drawdown
    - Publish portfolio.* events
    - Expose API endpoints for dashboard
    """
    
    def __init__(self):
        self.event_bus: Optional[EventBus] = None
        self.event_buffer: Optional[EventBuffer] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # TradeStore client (will be initialized in start())
        self.trade_store = None
        
        # Internal state
        self._running = False
        self._snapshot_task: Optional[asyncio.Task] = None
        self._event_loop_task: Optional[asyncio.Task] = None
        
        # Cached data
        self._current_snapshot: Optional[PortfolioSnapshot] = None
        self._exchange_positions: List = []  # Cache for Binance positions
        self._peak_equity_today: float = 0.0
        self._peak_equity_week: float = 0.0
        self._equity_history: List[Dict] = []  # For drawdown calculation
        
        # Metrics
        self._snapshots_generated: int = 0
        self._last_snapshot_time: Optional[datetime] = None
        
        logger.info("[PORTFOLIO-INTELLIGENCE] Service initialized")
    
    async def start(self):
        """Start the service."""
        try:
            logger.info("[PORTFOLIO-INTELLIGENCE] Starting service...")
            
            # Initialize Redis client
            import redis.asyncio as redis_async
            redis_client = await redis_async.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Initialize EventBus
            self.event_bus = EventBus(
                redis_client=redis_client,
                service_name="portfolio_intelligence"
            )
            await self.event_bus.initialize()
            
            # Subscribe to events
            self.event_bus.subscribe("trade.opened", self._handle_trade_opened)
            self.event_bus.subscribe("trade.closed", self._handle_trade_closed)
            self.event_bus.subscribe("order.executed", self._handle_order_executed)
            self.event_bus.subscribe("ess.tripped", self._handle_ess_tripped)
            self.event_bus.subscribe("market.tick", self._handle_market_tick)
            logger.info("[PORTFOLIO-INTELLIGENCE] EventBus subscriptions active")
            
            # EventBuffer not critical for now - skip it
            # self.event_buffer = EventBuffer(...)
            logger.info("[PORTFOLIO-INTELLIGENCE] EventBuffer skipped (not critical)")
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                base_url=settings.RISK_SAFETY_SERVICE_URL,
                timeout=5.0
            )
            logger.info(f"[PORTFOLIO-INTELLIGENCE]  HTTP client ready (risk-safety: {settings.RISK_SAFETY_SERVICE_URL})")
            
            # Initialize TradeStore
            await self._init_trade_store()
            
            # Generate initial snapshot
            await self._generate_snapshot()
            
            # Start background tasks
            self._running = True
            self._snapshot_task = asyncio.create_task(self._snapshot_update_loop())
            self._event_loop_task = asyncio.create_task(self._event_processing_loop())
            self._position_sync_task = asyncio.create_task(self._position_sync_loop())
            
            logger.info("[PORTFOLIO-INTELLIGENCE]  Service started successfully")
            
        except Exception as e:
            logger.error(f"[PORTFOLIO-INTELLIGENCE]  Failed to start service: {e}", exc_info=True)
            raise
    
    async def stop(self):
        """Stop the service."""
        if not self._running:
            return
        
        logger.info("[PORTFOLIO-INTELLIGENCE] Stopping service...")
        self._running = False
        
        # Cancel background tasks
        for task in [self._snapshot_task, self._event_loop_task, self._position_sync_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Flush event buffer
        if self.event_buffer:
            self.event_buffer.flush()
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
        
        logger.info("[PORTFOLIO-INTELLIGENCE]  Service stopped")
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    async def _init_trade_store(self):
        """Initialize TradeStore client."""
        logger.info("[PORTFOLIO-INTELLIGENCE] Initializing TradeStore...")
        
        try:
            if settings.TRADE_STORE_TYPE == "sqlite":
                from backend.core.trading.trade_store_sqlite import TradeStoreSQLite
                self.trade_store = TradeStoreSQLite(db_path=settings.TRADE_STORE_DB_PATH)
                await self.trade_store.initialize()
            elif settings.TRADE_STORE_TYPE == "redis":
                from backend.core.trading.trade_store_redis import TradeStoreRedis
                self.trade_store = TradeStoreRedis(redis_url=settings.TRADE_STORE_REDIS_URL or settings.REDIS_URL)
                await self.trade_store.initialize()
            else:
                raise ValueError(f"Unknown TRADE_STORE_TYPE: {settings.TRADE_STORE_TYPE}")
            
            logger.info(f"[PORTFOLIO-INTELLIGENCE]  TradeStore ready ({settings.TRADE_STORE_TYPE})")
            
        except Exception as e:
            logger.error(f"[PORTFOLIO-INTELLIGENCE]  Failed to initialize TradeStore: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================
    
    async def _handle_trade_opened(self, event_data: dict):
        """Handle trade.opened event."""
        logger.info(f"[PORTFOLIO-INTELLIGENCE] trade.opened: {event_data.get('symbol')}")
        # Trigger snapshot rebuild
        await self._generate_snapshot()
    
    async def _handle_trade_closed(self, event_data: dict):
        """Handle trade.closed event."""
        logger.info(f"[PORTFOLIO-INTELLIGENCE] trade.closed: {event_data.get('symbol')}, PnL: {event_data.get('realized_pnl')}")
        # Trigger snapshot rebuild + PnL event
        await self._generate_snapshot()
        await self._publish_pnl_updated()
    
    async def _handle_order_executed(self, event_data: dict):
        """Handle order.executed event."""
        logger.debug(f"[PORTFOLIO-INTELLIGENCE] order.executed: {event_data.get('symbol')}")
        # Trigger snapshot rebuild
        await self._generate_snapshot()
    
    async def _handle_ess_tripped(self, event_data: dict):
        """Handle ess.tripped event."""
        logger.warning(f"[PORTFOLIO-INTELLIGENCE] ESS tripped: {event_data.get('reason')}")
        # Recalculate risk metrics
        await self._publish_drawdown_updated()
    
    async def _handle_market_tick(self, event_data: dict):
        """Handle market.tick event - update unrealized PnL."""
        # Only update if we have open positions in this symbol
        symbol = event_data.get("symbol")
        if symbol and self._current_snapshot:
            # Check if we have open positions in this symbol
            has_position = any(pos.symbol == symbol for pos in self._current_snapshot.positions)
            if has_position:
                logger.debug(f"[PORTFOLIO-INTELLIGENCE] market.tick for {symbol}, updating unrealized PnL")
                await self._generate_snapshot()
    
    # ========================================================================
    # CORE SNAPSHOT GENERATION
    # ========================================================================
    
    async def _generate_snapshot(self) -> PortfolioSnapshot:
        """
        Build complete portfolio snapshot.
        
        Data sources:
        - TradeStore OR Binance: Open trades/positions
        - Binance: Cash balance
        - Market prices: Current prices for unrealized PnL
        """
        try:
            logger.debug("[PORTFOLIO-INTELLIGENCE] Generating snapshot...")
            
            # 1. Get open trades - prefer Binance positions if available
            if self._exchange_positions:
                # Use real positions from Binance
                logger.debug(f"[PORTFOLIO-INTELLIGENCE] Using {len(self._exchange_positions)} positions from Binance")
                open_trades = [
                    {
                        "symbol": p.symbol,
                        "side": p.side.value if hasattr(p.side, 'value') else str(p.side),
                        "quantity": float(p.quantity),
                        "entry_price": float(p.entry_price),
                        "leverage": p.leverage,
                        "category": "EXPANSION"  # Default category
                    }
                    for p in self._exchange_positions
                ]
            elif self.trade_store:
                # Fallback to TradeStore
                open_trades = await self.trade_store.get_open_trades()
            else:
                open_trades = []
            
            # 2. Get cash balance
            cash_balance = await self._get_cash_balance()
            
            # 3. Build position info list
            positions: List[PositionInfo] = []
            total_exposure = 0.0
            unrealized_pnl = 0.0
            
            for trade in open_trades:
                # Get current price (placeholder - would get from price service)
                current_price = await self._get_current_price(trade.get("symbol", ""))
                
                # [SPRINT 5 - PATCH #5] Use Decimal for PnL precision
                entry_price = Decimal(str(trade.get("entry_price", 0.0)))
                current_price_dec = Decimal(str(current_price))
                size = Decimal(str(trade.get("quantity", 0.0)))
                side = trade.get("side", "LONG")
                leverage = trade.get("leverage", 1.0)
                
                # Calculate unrealized PnL with Decimal precision
                if side.upper() == "LONG":
                    pnl_dec = (current_price_dec - entry_price) * size
                    pnl_pct = float((current_price_dec - entry_price) / entry_price) if entry_price > 0 else 0.0
                else:  # SHORT
                    pnl_dec = (entry_price - current_price_dec) * size
                    pnl_pct = float((entry_price - current_price_dec) / entry_price) if entry_price > 0 else 0.0
                
                # Round to 2 decimal places (USDT precision)
                pnl = float(pnl_dec.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                exposure = float((size * current_price_dec).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                
                positions.append(PositionInfo(
                    symbol=trade.get("symbol", ""),
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl=pnl,
                    unrealized_pnl_pct=pnl_pct,
                    exposure=exposure,
                    leverage=leverage,
                    category=trade.get("category", "EXPANSION")
                ))
                
                total_exposure += exposure
                unrealized_pnl += pnl
            
            # 4. Get realized PnL today
            realized_pnl_today = await self._get_realized_pnl_today()
            
            # 5. Calculate total equity
            total_equity = cash_balance + unrealized_pnl
            
            # 6. Calculate daily PnL
            daily_pnl = realized_pnl_today + unrealized_pnl
            
            # 7. Calculate daily drawdown
            if total_equity > self._peak_equity_today:
                self._peak_equity_today = total_equity
            
            daily_drawdown_pct = 0.0
            if self._peak_equity_today > 0:
                daily_drawdown_pct = ((self._peak_equity_today - total_equity) / self._peak_equity_today) * 100
            
            # 8. Build snapshot
            snapshot = PortfolioSnapshot(
                total_equity=total_equity,
                cash_balance=cash_balance,
                total_exposure=total_exposure,
                num_positions=len(positions),
                positions=positions,
                unrealized_pnl=unrealized_pnl,
                realized_pnl_today=realized_pnl_today,
                daily_pnl=daily_pnl,
                daily_drawdown_pct=daily_drawdown_pct,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            # Update cache
            self._current_snapshot = snapshot
            self._last_snapshot_time = datetime.now(timezone.utc)
            self._snapshots_generated += 1
            
            # Store equity history for drawdown calculation
            self._equity_history.append({
                "timestamp": datetime.now(timezone.utc),
                "equity": total_equity
            })
            # Keep only last 7 days
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            self._equity_history = [h for h in self._equity_history if h["timestamp"] > cutoff]
            
            # Publish event
            await self._publish_snapshot_updated(snapshot)
            
            logger.debug(f"[PORTFOLIO-INTELLIGENCE] Snapshot generated: equity={total_equity:.2f}, positions={len(positions)}, unrealized_pnl={unrealized_pnl:.2f}")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"[PORTFOLIO-INTELLIGENCE] Error generating snapshot: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    async def _get_cash_balance(self) -> float:
        """Get cash balance from Binance."""
        try:
            # Get Binance adapter
            from backend.integrations.exchanges.binance_client import get_binance_adapter
            adapter = await get_binance_adapter()
            balances = await adapter.get_balances()
            
            # Sum USDT/USDC balances
            total_cash = 0.0
            for balance in balances:
                if balance.asset in ["USDT", "USDC", "BUSD"]:
                    # Convert Decimal to float
                    total_cash += float(balance.free)
            
            logger.debug(f"[PORTFOLIO-INTELLIGENCE] Real cash balance from Binance: {total_cash:.2f}")
            return total_cash
            
        except Exception as e:
            logger.warning(f"[PORTFOLIO-INTELLIGENCE] Failed to get cash balance from Binance: {e}")
            # Fallback to default
            return 10000.0
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol from Binance."""
        try:
            # Get price from cached Binance positions if available
            if self._exchange_positions:
                for pos in self._exchange_positions:
                    if pos.symbol == symbol:
                        return float(pos.mark_price)
            
            # Fallback: fetch from Binance
            from backend.integrations.exchanges.binance_client import get_binance_adapter
            adapter = await get_binance_adapter()
            ticker = await adapter.get_ticker(symbol)
            return float(ticker.last_price)
            
        except Exception as e:
            logger.warning(f"[PORTFOLIO-INTELLIGENCE] Failed to get current price for {symbol}: {e}")
            # Return placeholder
            return 100.0
    
    async def _get_realized_pnl_today(self) -> float:
        """Get realized PnL for today from closed trades."""
        try:
            if not self.trade_store:
                return 0.0
            
            # TradeStoreSQLite doesn't have get_closed_trades_since method yet
            # Return 0.0 for now until implemented
            logger.debug("[PORTFOLIO-INTELLIGENCE] Realized PnL today: 0.0 (not implemented)")
            return 0.0
            
        except Exception as e:
            logger.error(f"[PORTFOLIO-INTELLIGENCE] Error getting realized PnL today: {e}")
            return 0.0
            return total_pnl
            
        except Exception as e:
            logger.error(f"[PORTFOLIO-INTELLIGENCE] Error getting realized PnL today: {e}")
            return 0.0
    
    # ========================================================================
    # EVENT PUBLISHING
    # ========================================================================
    
    async def _publish_snapshot_updated(self, snapshot: PortfolioSnapshot):
        """Publish portfolio.snapshot_updated event."""
        event = PortfolioSnapshotUpdatedEvent(
            total_equity=snapshot.total_equity,
            cash_balance=snapshot.cash_balance,
            total_exposure=snapshot.total_exposure,
            num_positions=snapshot.num_positions,
            unrealized_pnl=snapshot.unrealized_pnl,
            realized_pnl_today=snapshot.realized_pnl_today,
            daily_drawdown_pct=snapshot.daily_drawdown_pct,
            timestamp=snapshot.timestamp
        )
        await self.event_bus.publish("portfolio.snapshot_updated", event.model_dump())
        logger.debug("[PORTFOLIO-INTELLIGENCE] Published portfolio.snapshot_updated")
    
    async def _publish_pnl_updated(self):
        """Publish portfolio.pnl_updated event."""
        if not self._current_snapshot:
            return
        
        event = PortfolioPnLUpdatedEvent(
            realized_pnl_today=self._current_snapshot.realized_pnl_today,
            realized_pnl_total=0.0,  # TODO: Calculate total realized PnL
            unrealized_pnl=self._current_snapshot.unrealized_pnl,
            daily_pnl=self._current_snapshot.daily_pnl,
            weekly_pnl=0.0,  # TODO: Calculate weekly PnL
            monthly_pnl=0.0,  # TODO: Calculate monthly PnL
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        await self.event_bus.publish("portfolio.pnl_updated", event.model_dump())
        logger.debug("[PORTFOLIO-INTELLIGENCE] Published portfolio.pnl_updated")
    
    async def _publish_drawdown_updated(self):
        """Publish portfolio.drawdown_updated event."""
        if not self._current_snapshot:
            return
        
        # Calculate weekly drawdown
        weekly_peak = self._peak_equity_week
        if self._current_snapshot.total_equity > weekly_peak:
            weekly_peak = self._current_snapshot.total_equity
            self._peak_equity_week = weekly_peak
        
        weekly_drawdown_pct = 0.0
        if weekly_peak > 0:
            weekly_drawdown_pct = ((weekly_peak - self._current_snapshot.total_equity) / weekly_peak) * 100
        
        event = PortfolioDrawdownUpdatedEvent(
            daily_drawdown_pct=self._current_snapshot.daily_drawdown_pct,
            weekly_drawdown_pct=weekly_drawdown_pct,
            max_drawdown_pct=max(self._current_snapshot.daily_drawdown_pct, weekly_drawdown_pct),
            peak_equity=self._peak_equity_today,
            current_equity=self._current_snapshot.total_equity,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        await self.event_bus.publish("portfolio.drawdown_updated", event.model_dump())
        logger.debug("[PORTFOLIO-INTELLIGENCE] Published portfolio.drawdown_updated")
    
    async def _publish_exposure_updated(self):
        """Publish portfolio.exposure_updated event."""
        if not self._current_snapshot:
            return
        
        # Calculate exposure by symbol
        exposure_by_symbol = {}
        long_exposure = 0.0
        short_exposure = 0.0
        
        for pos in self._current_snapshot.positions:
            exposure_by_symbol[pos.symbol] = pos.exposure
            if pos.side == "LONG":
                long_exposure += pos.exposure
            else:
                short_exposure += pos.exposure
        
        net_exposure = long_exposure - short_exposure
        
        event = PortfolioExposureUpdatedEvent(
            total_exposure=self._current_snapshot.total_exposure,
            exposure_by_symbol=exposure_by_symbol,
            exposure_by_sector={},  # TODO: Add sector mapping
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            net_exposure=net_exposure,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        await self.event_bus.publish("portfolio.exposure_updated", event.model_dump())
        logger.debug("[PORTFOLIO-INTELLIGENCE] Published portfolio.exposure_updated")
    
    # ========================================================================
    # BACKGROUND TASKS
    # ========================================================================
    
    async def _snapshot_update_loop(self):
        """Background task: Periodically update snapshot."""
        logger.info("[PORTFOLIO-INTELLIGENCE] Snapshot update loop started")
        
        while self._running:
            try:
                await self._generate_snapshot()
                await self._publish_exposure_updated()
                await asyncio.sleep(settings.SNAPSHOT_UPDATE_INTERVAL_SEC)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[PORTFOLIO-INTELLIGENCE] Error in snapshot update loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)
        
        logger.info("[PORTFOLIO-INTELLIGENCE] Snapshot update loop stopped")
    
    async def _event_processing_loop(self):
        """Background loop for processing buffered events."""
        logger.info("[PORTFOLIO-INTELLIGENCE] Event processing loop started")
        
        while self._running:
            try:
                # Process buffered events
                if self.event_buffer:
                    while True:
                        event = self.event_buffer.pop()
                        if event is None:
                            break
                        
                        event_type = event.get("type")
                        event_data = event.get("data")
                        if event_type and event_data:
                            await self.event_bus.publish(event_type, event_data)
                            logger.debug(f"[PORTFOLIO-INTELLIGENCE] Replayed buffered event: {event_type}")
                
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[PORTFOLIO-INTELLIGENCE] Error in event processing loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)
        
        logger.info("[PORTFOLIO-INTELLIGENCE] Event processing loop stopped")
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def get_current_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Get current cached snapshot."""
        return self._current_snapshot
    
    async def get_pnl_breakdown(self) -> PnLBreakdown:
        """Get PnL breakdown."""
        if not self._current_snapshot:
            await self._generate_snapshot()
        
        # TODO: Calculate win rate, profit factor from TradeStore
        return PnLBreakdown(
            realized_pnl_total=0.0,  # TODO
            realized_pnl_today=self._current_snapshot.realized_pnl_today,
            realized_pnl_week=0.0,  # TODO
            realized_pnl_month=0.0,  # TODO
            unrealized_pnl=self._current_snapshot.unrealized_pnl,
            total_pnl=self._current_snapshot.realized_pnl_today + self._current_snapshot.unrealized_pnl,
            best_trade_pnl=0.0,  # TODO
            worst_trade_pnl=0.0,  # TODO
            win_rate=0.0,  # TODO
            profit_factor=0.0,  # TODO
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    async def get_exposure_breakdown(self) -> ExposureBreakdown:
        """Get exposure breakdown."""
        if not self._current_snapshot:
            await self._generate_snapshot()
        
        exposure_by_symbol = {}
        long_exposure = 0.0
        short_exposure = 0.0
        
        for pos in self._current_snapshot.positions:
            exposure_by_symbol[pos.symbol] = pos.exposure
            if pos.side == "LONG":
                long_exposure += pos.exposure
            else:
                short_exposure += pos.exposure
        
        net_exposure = long_exposure - short_exposure
        exposure_pct = (self._current_snapshot.total_exposure / self._current_snapshot.total_equity * 100) if self._current_snapshot.total_equity > 0 else 0.0
        
        return ExposureBreakdown(
            total_exposure=self._current_snapshot.total_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            net_exposure=net_exposure,
            exposure_by_symbol=exposure_by_symbol,
            exposure_by_sector={},  # TODO
            exposure_pct_of_equity=exposure_pct,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    async def get_drawdown_metrics(self) -> DrawdownMetrics:
        """Get drawdown metrics."""
        if not self._current_snapshot:
            await self._generate_snapshot()
        
        # Calculate recovery progress
        recovery_progress_pct = 0.0
        if self._peak_equity_today > 0 and self._current_snapshot.daily_drawdown_pct > 0:
            recovery_progress_pct = (1 - (self._current_snapshot.daily_drawdown_pct / 100)) * 100
        
        return DrawdownMetrics(
            daily_drawdown_pct=self._current_snapshot.daily_drawdown_pct,
            weekly_drawdown_pct=0.0,  # TODO
            max_drawdown_pct=self._current_snapshot.daily_drawdown_pct,
            peak_equity=self._peak_equity_today,
            current_equity=self._current_snapshot.total_equity,
            days_since_peak=0,  # TODO
            recovery_progress_pct=recovery_progress_pct,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    async def get_health(self) -> Dict[str, Any]:
        """Get service health."""
        
        health_data = {
            "service": "portfolio-intelligence",
            "status": "healthy" if self._running else "stopped",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - self._last_snapshot_time).total_seconds() if self._last_snapshot_time else 0,
            "snapshots_generated": self._snapshots_generated,
            "last_snapshot_time": self._last_snapshot_time.isoformat() if self._last_snapshot_time else None,
            "current_positions": len(self._current_snapshot.positions) if self._current_snapshot else 0,
            "current_equity": self._current_snapshot.total_equity if self._current_snapshot else 0.0,
            "components": []
        }
        
        # Check components
        if self.event_bus:
            health_data["components"].append(ComponentHealth(name="event_bus", status="healthy", uptime_seconds=0).dict())
        
        return health_data
    
    # ========================================================================
    # BACKGROUND POSITION SYNC FROM BINANCE
    # ========================================================================
    
    async def _position_sync_loop(self):
        """
        Periodically sync positions from Binance.
        Runs every 30 seconds to ensure TradeStore has latest exchange state.
        """
        logger.info("[PORTFOLIO-INTELLIGENCE] Position sync loop started")
        
        while self._running:
            try:
                await asyncio.sleep(30)  # Sync every 30 seconds
                
                if not self._running:
                    break
                
                # Sync positions from Binance
                await self._sync_positions_from_exchange()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[PORTFOLIO-INTELLIGENCE] Error in position sync loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait before retry
        
        logger.info("[PORTFOLIO-INTELLIGENCE] Position sync loop stopped")
    
    async def _sync_positions_from_exchange(self):
        """Sync open positions from Binance into TradeStore."""
        try:
            logger.debug("[PORTFOLIO-INTELLIGENCE] Syncing positions from Binance...")
            
            # Get Binance adapter
            from backend.integrations.exchanges.binance_client import get_binance_adapter
            adapter = await get_binance_adapter()
            
            # Fetch positions from Binance
            exchange_positions = await adapter.get_open_positions()
            
            # Filter only positions with non-zero quantity
            active_positions = [p for p in exchange_positions if p.quantity != 0]
            
            logger.info(
                f"[PORTFOLIO-INTELLIGENCE] Synced {len(active_positions)} active positions from Binance",
                extra={
                    "exchange": "binance",
                    "positions_count": len(active_positions),
                    "symbols": [p.symbol for p in active_positions]
                }
            )
            
            # Cache for use in snapshot generation
            self._exchange_positions = active_positions
            
        except Exception as e:
            logger.warning(f"[PORTFOLIO-INTELLIGENCE] Failed to sync positions from Binance: {e}")
        components = [
            ComponentHealth(
                name="EventBus",
                status="healthy" if self.event_bus else "unhealthy",
                message="Connected" if self.event_bus else "Not initialized"
            ),
            ComponentHealth(
                name="TradeStore",
                status="healthy" if self.trade_store else "unhealthy",
                message=f"Type: {settings.TRADE_STORE_TYPE}" if self.trade_store else "Not initialized"
            ),
            ComponentHealth(
                name="EventBuffer",
                status="healthy" if self.event_buffer else "unhealthy",
                message="Ready" if self.event_buffer else "Not initialized"
            ),
        ]
        
        return {
            "service": settings.SERVICE_NAME,
            "version": settings.VERSION,
            "status": "healthy" if all(c.status == "healthy" for c in components) else "degraded",
            "components": [c.model_dump() for c in components],
            "snapshots_generated": self._snapshots_generated,
            "last_snapshot_update": self._last_snapshot_time.isoformat() if self._last_snapshot_time else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
