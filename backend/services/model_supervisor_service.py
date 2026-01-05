"""
Model Supervisor Service
Monitors AI model performance and bias
"""
import asyncio
import logging
from fastapi import FastAPI
from backend.services.ai.model_supervisor import ModelSupervisor
from backend.services.common.health_check import HealthChecker
from backend.services.common.feature_flags import is_enabled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Model Supervisor", version="1.0.0")

# Initialize
model_supervisor = None
health_checker = HealthChecker("model_supervisor")


@app.on_event("startup")
async def startup():
    """Initialize Model Supervisor on startup"""
    global model_supervisor
    
    if not is_enabled("model_supervisor"):
        logger.warning("Model Supervisor is DISABLED via feature flag")
        return
    
    logger.info("🚀 Starting Model Supervisor...")
    model_supervisor = ModelSupervisor()
    
    # Start background monitoring
    asyncio.create_task(monitoring_loop())
    logger.info("✅ Model Supervisor started successfully")


async def monitoring_loop():
    """Background task to monitor models periodically"""
    while True:
        try:
            if model_supervisor:
                # Note: analyze_models() requires signal_logs, skipping periodic analysis
                # Real-time monitoring via monitor_loop() handles observation
                logger.info("Model supervisor in observation mode (use /report endpoint for analysis)")
        except Exception as e:
            health_checker.record_error(f"Supervision monitoring failed: {e}")
            logger.error(f"Model supervision error: {e}")
        
        await asyncio.sleep(1800)  # 30 minutes


@app.get("/health")
async def health():
    """Health check endpoint"""
    result = health_checker.check_health()
    return result.to_dict()


@app.get("/report")
async def get_report():
    """Get latest supervision report"""
    if not model_supervisor:
        return {"error": "Model Supervisor not initialized"}
    
    try:
        # Return basic status - full analysis requires signal logs via analyze_models()
        return {
            "status": "observing",
            "message": "Model supervisor running in observation mode",
            "mode": model_supervisor.mode,
            "analysis_window_days": model_supervisor.analysis_window
        }
    except Exception as e:
        health_checker.record_error(f"Report generation failed: {e}")
        return {"error": str(e)}


@app.post("/trigger")
async def trigger_supervision():
    """Manually trigger model supervision"""
    if not model_supervisor:
        return {"error": "Model Supervisor not initialized"}
    
    try:
        # Model supervisor runs via real-time observation (monitor_loop)
        # Full analysis requires signal_logs from Redis/database
        return {
            "status": "success",
            "message": "Model supervisor observation active",
            "note": "Use analyze_models() with signal logs for full analysis"
        }
    except Exception as e:
        health_checker.record_error(f"Manual trigger failed: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
