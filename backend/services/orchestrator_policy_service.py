"""
Orchestrator Policy Service
Dynamic policy management based on market conditions
"""
import asyncio
import logging
from fastapi import FastAPI
from backend.services.orchestrator_policy import OrchestratorPolicy
from backend.services.common.health_check import HealthChecker
from backend.services.common.feature_flags import is_enabled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orchestrator Policy", version="1.0.0")

# Initialize
orchestrator_policy = None
health_checker = HealthChecker("orchestrator_policy")


@app.on_event("startup")
async def startup():
    """Initialize Orchestrator Policy on startup"""
    global orchestrator_policy
    
    if not is_enabled("orchestrator_policy"):
        logger.warning("Orchestrator Policy is DISABLED via feature flag")
        return
    
    logger.info("🚀 Starting Orchestrator Policy...")
    orchestrator_policy = OrchestratorPolicy()
    
    # Start background policy updates
    asyncio.create_task(policy_update_loop())
    logger.info("✅ Orchestrator Policy started successfully")


async def policy_update_loop():
    """Background task to update policy periodically"""
    while True:
        try:
            if orchestrator_policy:
                policy = orchestrator_policy.update_policy()
                logger.info(f"Policy updated: allow_trades={policy.get('allow_trades')}, min_confidence={policy.get('min_confidence')}")
        except Exception as e:
            health_checker.record_error(f"Policy update failed: {e}")
            logger.error(f"Policy update error: {e}")
        
        await asyncio.sleep(60)  # 1 minute


@app.get("/health")
async def health():
    """Health check endpoint"""
    result = health_checker.check_health()
    return result.to_dict()


@app.get("/policy")
async def get_policy():
    """Get current trading policy"""
    if not orchestrator_policy:
        return {"error": "Orchestrator Policy not initialized"}
    
    try:
        policy = orchestrator_policy.get_current_policy()
        return policy
    except Exception as e:
        health_checker.record_error(f"Policy retrieval failed: {e}")
        return {"error": str(e)}


@app.post("/update_policy")
async def update_policy():
    """Manually trigger policy update"""
    if not orchestrator_policy:
        return {"error": "Orchestrator Policy not initialized"}
    
    try:
        policy = orchestrator_policy.update_policy()
        return {"status": "success", "policy": policy}
    except Exception as e:
        health_checker.record_error(f"Policy update failed: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8014)
