"""
Self-Healing System Service
Monitors and auto-recovers from failures
"""
import asyncio
import logging
from fastapi import FastAPI
from backend.services.monitoring.self_healing import SelfHealingSystem
from backend.services.common.health_check import HealthChecker
from backend.services.common.feature_flags import is_enabled, get_mode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Self-Healing System", version="1.0.0")

# Initialize
self_healing = None
health_checker = HealthChecker("self_healing")


@app.on_event("startup")
async def startup():
    """Initialize Self-Healing System on startup"""
    global self_healing
    
    if not is_enabled("self_healing"):
        logger.warning("Self-Healing is DISABLED via feature flag")
        return
    
    mode = get_mode("self_healing")
    logger.info(f"🚀 Starting Self-Healing System (mode: {mode.value})...")
    self_healing = SelfHealingSystem()
    
    # Start background monitoring
    asyncio.create_task(monitoring_loop())
    logger.info(f"✅ Self-Healing System started in {mode.value} mode")


async def monitoring_loop():
    """Background monitoring and recovery"""
    while True:
        try:
            if self_healing:
                # Run health checks on all subsystems
                issues = self_healing.check_all_subsystems()
                
                if issues:
                    logger.warning(f"Found {len(issues)} issues")
                    
                    mode = get_mode("self_healing")
                    if mode.value in ["ENFORCE", "AGGRESSIVE"]:
                        # Attempt recovery
                        for issue in issues:
                            recovery_result = self_healing.attempt_recovery(issue)
                            logger.info(f"Recovery attempt: {recovery_result}")
        except Exception as e:
            health_checker.record_error(f"Monitoring failed: {e}")
            logger.error(f"Self-healing error: {e}")
        
        await asyncio.sleep(120)  # 2 minutes


@app.get("/health")
async def health():
    """Health check endpoint"""
    result = health_checker.check_health()
    return result.to_dict()


@app.get("/system_health")
async def get_system_health():
    """Get health status of all subsystems"""
    if not self_healing:
        return {"error": "Self-Healing not initialized"}
    
    try:
        health_status = self_healing.get_system_health()
        return health_status
    except Exception as e:
        health_checker.record_error(f"System health check failed: {e}")
        return {"error": str(e)}


@app.post("/trigger_recovery/{subsystem}")
async def trigger_recovery(subsystem: str):
    """Manually trigger recovery for a subsystem"""
    if not self_healing:
        return {"error": "Self-Healing not initialized"}
    
    try:
        result = self_healing.recover_subsystem(subsystem)
        return {"status": "success", "result": result}
    except Exception as e:
        health_checker.record_error(f"Manual recovery failed: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
