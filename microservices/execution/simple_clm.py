"""
Simple Continuous Learning Manager (CLM) - Lightweight retraining orchestrator.

Simplified version that runs without database/PolicyStore dependencies.
Periodically retrains AI models based on collected data.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleCLM:
    """
    Lightweight CLM for periodic model retraining.
    
    Features:
    - Scheduled retraining (default: every 7 days)
    - Triggers retraining via AI Engine API
    - No database dependencies
    - EventBus notifications for retraining events
    """
    
    def __init__(
        self,
        ai_engine_url: str = "http://ai-engine:8001",
        retraining_interval_hours: int = 168,  # 7 days
        min_samples_required: int = 100,
        event_bus = None
    ):
        self.ai_engine_url = ai_engine_url
        self.retraining_interval_hours = retraining_interval_hours
        self.min_samples_required = min_samples_required
        self.event_bus = event_bus
        
        self.running = False
        self.last_retraining: Optional[datetime] = None
        self._task: Optional[asyncio.Task] = None
        
        logger.info(
            f"[SIMPLE-CLM] Initialized: retraining every {retraining_interval_hours}h, "
            f"min_samples={min_samples_required}"
        )
    
    async def start(self):
        """Start CLM background loop."""
        if self.running:
            logger.warning("[SIMPLE-CLM] Already running")
            return
        
        self.running = True
        self._task = asyncio.create_task(self._retraining_loop())
        logger.info("[SIMPLE-CLM] ✅ Started")
    
    async def stop(self):
        """Stop CLM background loop."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[SIMPLE-CLM] ✅ Stopped")
    
    async def _retraining_loop(self):
        """Main retraining loop - runs every interval."""
        try:
            while self.running:
                try:
                    # Wait until next retraining time
                    if self.last_retraining:
                        next_retraining = self.last_retraining + timedelta(hours=self.retraining_interval_hours)
                        wait_seconds = (next_retraining - datetime.utcnow()).total_seconds()
                        
                        if wait_seconds > 0:
                            logger.info(
                                f"[SIMPLE-CLM] Next retraining in {wait_seconds/3600:.1f}h "
                                f"({next_retraining.isoformat()})"
                            )
                            await asyncio.sleep(wait_seconds)
                    else:
                        # First run - wait 1 hour to collect some data
                        logger.info("[SIMPLE-CLM] First run - waiting 1h before initial retraining")
                        await asyncio.sleep(3600)
                    
                    # Trigger retraining
                    await self._trigger_retraining()
                    
                except Exception as e:
                    logger.error(f"[SIMPLE-CLM] Error in retraining loop: {e}", exc_info=True)
                    # Wait 1 hour before retry
                    await asyncio.sleep(3600)
        
        except asyncio.CancelledError:
            logger.info("[SIMPLE-CLM] Retraining loop cancelled")
    
    async def _trigger_retraining(self):
        """Trigger retraining via AI Engine API."""
        try:
            import aiohttp
            
            logger.info("[SIMPLE-CLM] 🔄 Triggering model retraining...")
            
            # Call AI Engine retraining endpoint
            url = f"{self.ai_engine_url}/api/ai/retrain"
            params = {"min_samples": self.min_samples_required}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(
                            f"[SIMPLE-CLM] ✅ Retraining completed: "
                            f"version={result.get('version_id')}, "
                            f"accuracy={result.get('validation_accuracy', 0):.2%}"
                        )
                        
                        # Update last retraining time
                        self.last_retraining = datetime.utcnow()
                        
                        # Publish event if EventBus available
                        if self.event_bus:
                            await self.event_bus.publish("learning.retraining.completed", {
                                "timestamp": self.last_retraining.isoformat(),
                                "version_id": result.get('version_id'),
                                "validation_accuracy": result.get('validation_accuracy'),
                                "model_type": "ensemble"
                            })
                    
                    elif resp.status == 400:
                        error = await resp.json()
                        logger.warning(f"[SIMPLE-CLM] ⚠️ Retraining skipped: {error.get('detail')}")
                    
                    else:
                        logger.error(f"[SIMPLE-CLM] ❌ Retraining failed: HTTP {resp.status}")
        
        except Exception as e:
            logger.error(f"[SIMPLE-CLM] ❌ Failed to trigger retraining: {e}", exc_info=True)
    
    async def trigger_manual_retraining(self):
        """Manually trigger retraining (can be called via API)."""
        logger.info("[SIMPLE-CLM] 🔄 Manual retraining triggered")
        await self._trigger_retraining()
    
    def get_status(self) -> dict:
        """Get CLM status."""
        return {
            "running": self.running,
            "last_retraining": self.last_retraining.isoformat() if self.last_retraining else None,
            "next_retraining": (
                (self.last_retraining + timedelta(hours=self.retraining_interval_hours)).isoformat()
                if self.last_retraining else "First run pending"
            ),
            "retraining_interval_hours": self.retraining_interval_hours,
            "min_samples_required": self.min_samples_required
        }
