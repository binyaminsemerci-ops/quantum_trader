"""
Analytics Service - aggregates metrics and provides performance insights.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.services.eventbus import InMemoryEventBus, Event
from backend.services.analytics.models import (
    StrategyMetrics,
    SystemMetrics,
    ModelMetrics,
    TradeMetrics,
)
from backend.services.analytics.repository import InMemoryMetricsRepository

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Analytics Service that tracks system-wide metrics.
    
    Subscribes to all events and maintains performance metrics.
    """
    
    def __init__(
        self,
        eventbus: InMemoryEventBus,
        repository: Optional[InMemoryMetricsRepository] = None,
    ):
        self.eventbus = eventbus
        self.repository = repository or InMemoryMetricsRepository()
        self._start_time = datetime.now()
        self._running = False
        
        # In-memory tracking
        self._strategy_metrics: Dict[str, StrategyMetrics] = {}
        self._model_metrics: Dict[str, ModelMetrics] = {}
        self._system_metrics = SystemMetrics()
    
    def subscribe_to_events(self):
        """Subscribe to all relevant events."""
        self.eventbus.subscribe("policy.updated", self.on_policy_updated)
        self.eventbus.subscribe("strategy.promoted", self.on_strategy_promoted)
        self.eventbus.subscribe("model.promoted", self.on_model_promoted)
        self.eventbus.subscribe("health.status_changed", self.on_health_changed)
        self.eventbus.subscribe("opportunities.updated", self.on_opportunities_updated)
        self.eventbus.subscribe("trade.executed", self.on_trade_executed)
        
        logger.info("AnalyticsService subscribed to all events")
    
    async def on_policy_updated(self, event: Event):
        """Handle policy update events."""
        try:
            payload = event.payload
            self._system_metrics.current_risk_mode = payload.get("risk_mode", "NORMAL")
            self._system_metrics.policy_changes_count += 1
            self._system_metrics.last_policy_change = event.timestamp
            
            await self.repository.save_system_metrics(self._system_metrics)
            logger.debug(f"Policy updated: {payload.get('risk_mode')}")
        except Exception as e:
            logger.error(f"Error handling policy update: {e}")
    
    async def on_strategy_promoted(self, event: Event):
        """Handle strategy promotion events."""
        try:
            payload = event.payload
            strategy_id = payload.get("strategy_id")
            
            if strategy_id not in self._strategy_metrics:
                self._strategy_metrics[strategy_id] = StrategyMetrics(strategy_id=strategy_id)
            
            metrics = self._strategy_metrics[strategy_id]
            metrics.stage = payload.get("to_stage", "LIVE")
            metrics.promoted_at = event.timestamp
            
            await self.repository.save_strategy_metrics(metrics)
            logger.info(f"Strategy {strategy_id} promoted to {metrics.stage}")
        except Exception as e:
            logger.error(f"Error handling strategy promotion: {e}")
    
    async def on_model_promoted(self, event: Event):
        """Handle model promotion events."""
        try:
            payload = event.payload
            model_name = payload.get("model_name")
            new_version = payload.get("new_version")
            
            key = f"{model_name}:{new_version}"
            if key not in self._model_metrics:
                self._model_metrics[key] = ModelMetrics(
                    model_name=model_name,
                    version=new_version,
                )
            
            metrics = self._model_metrics[key]
            metrics.stage = "LIVE"
            metrics.promoted_at = event.timestamp
            
            # Update with shadow performance if available
            shadow_perf = payload.get("shadow_performance", {})
            if shadow_perf:
                metrics.accuracy = shadow_perf.get("accuracy", metrics.accuracy)
                metrics.precision = shadow_perf.get("precision", metrics.precision)
                metrics.recall = shadow_perf.get("recall", metrics.recall)
            
            await self.repository.save_model_metrics(metrics)
            logger.info(f"Model {model_name} v{new_version} promoted to LIVE")
        except Exception as e:
            logger.error(f"Error handling model promotion: {e}")
    
    async def on_health_changed(self, event: Event):
        """Handle health status change events."""
        try:
            payload = event.payload
            new_status = payload.get("new_status", "UNKNOWN")
            self._system_metrics.health_status = new_status
            
            await self.repository.save_system_metrics(self._system_metrics)
            logger.debug(f"Health status: {new_status}")
        except Exception as e:
            logger.error(f"Error handling health change: {e}")
    
    async def on_opportunities_updated(self, event: Event):
        """Handle opportunities update events."""
        try:
            # Track event processing
            self._system_metrics.events_processed += 1
            await self.repository.save_system_metrics(self._system_metrics)
        except Exception as e:
            logger.error(f"Error handling opportunities update: {e}")
    
    async def on_trade_executed(self, event: Event):
        """Handle trade execution events."""
        try:
            payload = event.payload
            
            # Map event payload to TradeMetrics
            trade = TradeMetrics(
                trade_id=payload.get("order_id", ""),
                symbol=payload.get("symbol", ""),
                strategy_id=payload.get("strategy_id", ""),
                side=payload.get("side", "BUY"),
                entry_price=payload.get("price", 0.0),
                exit_price=None,  # We'll get this from pnl
                quantity=payload.get("size", 0.0),
            )
            
            # If PnL is provided, this is a closed trade
            pnl = payload.get("pnl")
            if pnl is not None:
                trade.pnl = pnl
                trade.closed_at = event.timestamp
                
                # Estimate exit price from PnL
                if trade.quantity > 0:
                    if trade.side == "BUY":  # Closed long
                        trade.exit_price = trade.entry_price + (pnl / trade.quantity)
                    else:  # Closed short
                        trade.exit_price = trade.entry_price - (pnl / trade.quantity)
                
                trade.pnl_percent = trade.calculate_pnl_percent()
                
                # Update strategy metrics
                strategy_id = trade.strategy_id
                if strategy_id not in self._strategy_metrics:
                    self._strategy_metrics[strategy_id] = StrategyMetrics(strategy_id=strategy_id)
                
                metrics = self._strategy_metrics[strategy_id]
                metrics.total_trades += 1
                metrics.total_pnl += trade.pnl
                
                if trade.pnl > 0:
                    metrics.winning_trades += 1
                    metrics.avg_win = (
                        (metrics.avg_win * (metrics.winning_trades - 1) + trade.pnl)
                        / metrics.winning_trades
                    )
                else:
                    metrics.losing_trades += 1
                    metrics.avg_loss = (
                        (metrics.avg_loss * (metrics.losing_trades - 1) + trade.pnl)
                        / metrics.losing_trades
                    )
                
                metrics.win_rate = metrics.calculate_win_rate()
                metrics.profit_factor = metrics.calculate_profit_factor()
                metrics.last_trade_at = event.timestamp
                
                await self.repository.save_strategy_metrics(metrics)
                
                # Update system metrics
                self._system_metrics.closed_positions += 1
                self._system_metrics.total_pnl += trade.pnl
            else:
                self._system_metrics.open_positions += 1
            
            self._system_metrics.total_positions += 1
            
            await self.repository.save_trade_metrics(trade)
            await self.repository.save_system_metrics(self._system_metrics)
            
            logger.debug(f"Trade executed: {trade.symbol} {trade.side}")
        except Exception as e:
            logger.error(f"Error handling trade execution: {e}")
    
    async def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get metrics for a specific strategy."""
        return await self.repository.get_strategy_metrics(strategy_id)
    
    async def get_all_strategy_metrics(self) -> List[StrategyMetrics]:
        """Get metrics for all strategies."""
        return await self.repository.get_all_strategy_metrics()
    
    async def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get metrics for a specific model."""
        return await self.repository.get_model_metrics(model_name)
    
    async def get_all_model_metrics(self) -> List[ModelMetrics]:
        """Get metrics for all models."""
        return await self.repository.get_all_model_metrics()
    
    async def get_system_metrics(self) -> SystemMetrics:
        """Get overall system metrics."""
        # Update uptime
        self._system_metrics.uptime_seconds = (
            datetime.now() - self._start_time
        ).total_seconds()
        
        # Get EventBus stats
        bus_stats = self.eventbus.get_stats()
        self._system_metrics.events_published = bus_stats.get("published", 0)
        self._system_metrics.event_errors = bus_stats.get("errors", 0)
        
        await self.repository.save_system_metrics(self._system_metrics)
        return self._system_metrics
    
    async def get_trade_history(
        self,
        strategy_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[TradeMetrics]:
        """Get trade history."""
        return await self.repository.get_trade_history(strategy_id, limit)
    
    async def generate_performance_report(self) -> Dict:
        """Generate a comprehensive performance report."""
        system_metrics = await self.get_system_metrics()
        all_strategies = await self.get_all_strategy_metrics()
        all_models = await self.get_all_model_metrics()
        
        return {
            "system": system_metrics.to_dict(),
            "strategies": [s.to_dict() for s in all_strategies],
            "models": [m.to_dict() for m in all_models],
            "summary": {
                "total_strategies": len(all_strategies),
                "live_strategies": sum(1 for s in all_strategies if s.stage == "LIVE"),
                "total_models": len(all_models),
                "live_models": sum(1 for m in all_models if m.stage == "LIVE"),
                "overall_win_rate": (
                    sum(s.win_rate * s.total_trades for s in all_strategies) /
                    sum(s.total_trades for s in all_strategies)
                    if sum(s.total_trades for s in all_strategies) > 0 else 0
                ),
            }
        }
    
    async def run_forever(self, interval_seconds: int = 60):
        """Continuously update metrics."""
        self._running = True
        logger.info("AnalyticsService started")
        
        while self._running:
            try:
                # Update system metrics
                await self.get_system_metrics()
                
                # Wait for next update
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("AnalyticsService cancelled")
                break
            except Exception as e:
                logger.error(f"Error in analytics loop: {e}")
                await asyncio.sleep(interval_seconds)
        
        logger.info("AnalyticsService stopped")
    
    def stop(self):
        """Stop the analytics loop."""
        self._running = False
