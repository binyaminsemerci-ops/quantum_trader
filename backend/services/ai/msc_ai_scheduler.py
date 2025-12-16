"""
Meta Strategy Controller (MSC AI) Scheduler

Background scheduler that runs MSC AI evaluation periodically.

Configuration:
- Runs every 30 minutes by default (configurable via MSC_EVALUATION_INTERVAL_MINUTES)
- Can be disabled via MSC_ENABLED=false
- Logs evaluation results to file and Prometheus

Author: Quantum Trader Team
Date: 2025-11-30
"""

import logging
import os
import asyncio
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from backend.services.msc_ai_integration import run_msc_evaluation

logger = logging.getLogger(__name__)


class MSCScheduler:
    """
    Scheduler for periodic MSC AI evaluation.
    """
    
    def __init__(self, policy_store=None):
        """Initialize MSC AI scheduler.
        
        Args:
            policy_store: PolicyStore instance for writing policy updates
        """
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.enabled = os.getenv("MSC_ENABLED", "true").lower() == "true"
        self.interval_minutes = int(os.getenv("MSC_EVALUATION_INTERVAL_MINUTES", "30"))
        self.last_run: Optional[datetime] = None
        self.last_result: Optional[dict] = None
        self.policy_store = policy_store
        
        if self.policy_store:
            logger.info(f"[MSC Scheduler] Initialized with PolicyStore (enabled={self.enabled}, interval={self.interval_minutes}m)")
        else:
            logger.info(f"[MSC Scheduler] Initialized WITHOUT PolicyStore (enabled={self.enabled}, interval={self.interval_minutes}m)")
    
    def start(self):
        """
        Start the MSC AI scheduler.
        """
        if not self.enabled:
            logger.info("[MSC Scheduler] MSC AI is disabled (MSC_ENABLED=false)")
            return
        
        if self.scheduler:
            logger.warning("[MSC Scheduler] Scheduler already running")
            return
        
        try:
            self.scheduler = AsyncIOScheduler()
            
            # Add periodic evaluation job
            self.scheduler.add_job(
                self._run_evaluation,
                trigger=IntervalTrigger(minutes=self.interval_minutes),
                id="msc_ai_evaluation",
                name="MSC AI Policy Evaluation",
                replace_existing=True,
                max_instances=1  # Prevent overlapping executions
            )
            
            # Run immediately on startup
            self.scheduler.add_job(
                self._run_evaluation,
                id="msc_ai_startup",
                name="MSC AI Initial Evaluation",
                replace_existing=True
            )
            
            self.scheduler.start()
            logger.info(f"[MSC Scheduler] Started - will run every {self.interval_minutes} minutes")
            
        except Exception as e:
            logger.error(f"[MSC Scheduler] Failed to start: {e}", exc_info=True)
    
    def stop(self):
        """
        Stop the MSC AI scheduler.
        """
        if self.scheduler:
            try:
                self.scheduler.shutdown(wait=False)
                self.scheduler = None
                logger.info("[MSC Scheduler] Stopped")
            except Exception as e:
                logger.error(f"[MSC Scheduler] Error stopping scheduler: {e}")
    
    async def _run_evaluation(self):
        """
        Execute MSC AI evaluation (async wrapper for scheduler).
        """
        try:
            logger.info("[MSC Scheduler] Starting scheduled evaluation")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_msc_evaluation)
            
            # Write policy to PolicyStore if available
            if self.policy_store and result.get('status') == 'success':
                try:
                    policy = result['policy']
                    self.policy_store.patch({
                        'risk_mode': policy['risk_mode'],
                        'max_risk_per_trade': policy['max_risk_per_trade'],
                        'max_positions': policy['max_positions'],
                        'global_min_confidence': policy['global_min_confidence'],
                        'allowed_strategies': policy.get('allowed_strategies', [])
                    })
                    logger.info(f"[MSC Scheduler] ✅ Policy written to PolicyStore: {policy['risk_mode']}")
                except Exception as e:
                    logger.error(f"[MSC Scheduler] ❌ Failed to write to PolicyStore: {e}")
            
            result = await loop.run_in_executor(None, run_msc_evaluation)
            
            self.last_run = datetime.now(timezone.utc)
            self.last_result = result
            
            if result["status"] == "success":
                policy = result["policy"]
                logger.info(
                    f"[MSC Scheduler] Evaluation completed successfully - "
                    f"Mode: {policy['risk_mode']}, "
                    f"Strategies: {len(policy['allowed_strategies'])}, "
                    f"Duration: {result['duration_seconds']:.2f}s"
                )
            else:
                logger.error(f"[MSC Scheduler] Evaluation failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"[MSC Scheduler] Evaluation error: {e}", exc_info=True)
            self.last_result = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_status(self) -> dict:
        """
        Get scheduler status.
        
        Returns:
            Dictionary with scheduler state
        """
        if not self.enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "message": "MSC AI scheduler is disabled"
            }
        
        if not self.scheduler or not self.scheduler.running:
            return {
                "enabled": True,
                "status": "stopped",
                "message": "MSC AI scheduler is not running"
            }
        
        jobs = self.scheduler.get_jobs()
        next_run = None
        
        for job in jobs:
            if job.id == "msc_ai_evaluation" and job.next_run_time:
                next_run = job.next_run_time.isoformat()
                break
        
        return {
            "enabled": True,
            "status": "running",
            "interval_minutes": self.interval_minutes,
            "next_run": next_run,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_result": self.last_result,
            "active_jobs": len(jobs)
        }


# Global scheduler instance
_scheduler: Optional[MSCScheduler] = None


def get_msc_scheduler(policy_store=None) -> MSCScheduler:
    """
    Get or create global MSC scheduler instance.
    
    Args:
        policy_store: PolicyStore instance for writing policy updates
    
    Returns:
        MSCScheduler instance
    """
    global _scheduler
    
    if _scheduler is None:
        _scheduler = MSCScheduler(policy_store=policy_store)
    elif policy_store and not _scheduler.policy_store:
        # Update existing scheduler with policy_store
        _scheduler.policy_store = policy_store
        logger.info("[MSC Scheduler] PolicyStore attached to existing scheduler")
    
    return _scheduler


def start_msc_scheduler(policy_store=None):
    """
    Start the global MSC AI scheduler.
    
    Args:
        policy_store: PolicyStore instance for writing policy updates
    
    Call this during application startup.
    """
    scheduler = get_msc_scheduler(policy_store=policy_store)
    scheduler.start()


def stop_msc_scheduler():
    """
    Stop the global MSC AI scheduler.
    
    Call this during application shutdown.
    """
    scheduler = get_msc_scheduler()
    scheduler.stop()
