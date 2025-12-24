"""
AI Engine Service - Main Entry Point

Responsibilities:
- AI model inference (XGBoost, LightGBM, N-HiTS, PatchTST)
- Ensemble voting and signal aggregation
- Meta-strategy selection (RL-based)
- RL position sizing
- Market regime detection
- Memory state management
- Trade intent generation

Port: 8001
Events IN: market.tick, market.klines, trade.closed, policy.updated
Events OUT: ai.decision.made, ai.signal_generated, strategy.selected, sizing.decided, trade.intent
"""
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST

from .service import AIEngineService
from . import api
from .config import settings

# [EPIC-OBS-001] Initialize observability (tracing, metrics, structured logging)
try:
    from backend.infra.observability import (
        init_observability, 
        get_logger, 
        get_tracer, 
        instrument_fastapi,
        add_metrics_middleware,
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    # Fallback to basic logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'{settings.LOG_DIR}/ai_engine_service.log')
        ]
    )

# Initialize observability at module level (before service starts)
if OBSERVABILITY_AVAILABLE:
    init_observability(
        service_name="ai-engine",
        log_level=settings.LOG_LEVEL,
        enable_tracing=True,
        enable_metrics=True,
    )
    logger = get_logger(__name__)
else:
    logger = logging.getLogger(__name__)

# Global service instance
service: AIEngineService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global service
    
    # Startup
    logger.info("=" * 60)
    logger.info("🤖 AI ENGINE SERVICE STARTING")
    logger.info(f"Version: {settings.VERSION}")
    logger.info(f"Port: {settings.PORT}")
    logger.info(f"Ensemble Models: {settings.ENSEMBLE_MODELS}")
    logger.info("=" * 60)
    
    try:
        service = AIEngineService()
        await service.start()
        logger.info("✅ AI Engine Service STARTED")
        
        # Inject service into API module
        api.set_service(service)
        logger.info("✅ Service injected into API module")
        
        # Register signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))
        
        yield
        
    finally:
        # Shutdown
        logger.info("🛑 AI ENGINE SERVICE SHUTTING DOWN...")
        if service:
            await service.stop()
        logger.info("✅ AI Engine Service STOPPED")


async def shutdown(sig):
    """Graceful shutdown on signal."""
    logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")
    # FastAPI will call lifespan's finally block


# Create FastAPI app
app = FastAPI(
    title="AI Engine Service",
    version=settings.VERSION,
    description="AI model inference, ensemble voting, meta-strategy selection, and RL position sizing",
    lifespan=lifespan
)

# [EPIC-OBS-001] Instrument FastAPI with OpenTelemetry tracing
if OBSERVABILITY_AVAILABLE:
    instrument_fastapi(app)
    add_metrics_middleware(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api.router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with service info."""
    return {
        "service": "ai-engine",
        "version": settings.VERSION,
        "status": "running" if service and service._running else "stopped",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", tags=["health"])
async def health_check():
    """
    Legacy health check endpoint (backward compatible).
    
    Returns full health status. Use /health/ready for K8s readiness probes.
    """
    if not service:
        return {"healthy": False, "error": "Service not initialized"}
    
    return await service.get_health()


@app.get("/health/live", tags=["health"])
async def health_live():
    """
    Liveness probe endpoint.
    
    Returns 200 if the process is alive and responsive.
    Used by Kubernetes/Docker to detect if container needs restart.
    """
    return {
        "status": "ok",
        "service": "ai-engine",
        "version": settings.VERSION,
    }


@app.get("/health/ready", tags=["health"])
async def health_ready():
    """
    Readiness probe endpoint.
    
    Returns 200 if service is ready to accept traffic (dependencies healthy).
    Returns 503 if service is not ready (still initializing or deps down).
    """
    if not service or not service._running:
        return Response(
            content='{"status": "not_ready", "reason": "Service not initialized"}',
            status_code=503,
            media_type="application/json"
        )
    
    # Get full health status from service
    health = await service.get_health()
    
    # If service reports unhealthy, return 503
    if not health.get("healthy", False):
        return Response(
            content=f'{{"status": "not_ready", "health": {health}}}',
            status_code=503,
            media_type="application/json"
        )
    
    return {
        "status": "ready",
        "service": "ai-engine",
        "healthy": True,
    }


@app.get("/metrics", tags=["observability"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes all service metrics in Prometheus text format for scraping.
    """
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# PHASE 3C: SYSTEM HEALTH ENDPOINTS
# ============================================================================

@app.get("/health/detailed", tags=["health", "phase_3c"])
async def health_detailed():
    """
    Phase 3C: Comprehensive health status with module breakdown
    
    Returns:
        - Overall health score and status
        - Individual module health (2B, 2D, 3A, 3B, ensemble)
        - Performance metrics
        - Active alerts
    """
    if not service.health_monitor:
        return {
            "error": "Health monitor not initialized",
            "service": "ai-engine",
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    current_health = service.health_monitor.get_current_health()
    
    if not current_health:
        # No health check performed yet, trigger one
        current_health = await service.health_monitor.perform_health_check()
    
    return current_health.to_dict()


@app.get("/health/alerts", tags=["health", "phase_3c"])
async def health_alerts(hours: int = 24):
    """
    Phase 3C: Get recent health alerts
    
    Args:
        hours: Number of hours of alert history (default: 24)
    
    Returns:
        List of alerts with severity, module, message, and recommendations
    """
    if not service.health_monitor:
        return {
            "error": "Health monitor not initialized",
            "alerts": []
        }
    
    alerts = service.health_monitor.get_recent_alerts(hours=hours)
    active_alerts = service.health_monitor.get_active_alerts()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "active_alerts_count": len(active_alerts),
        "total_alerts_count": len(alerts),
        "active_alerts": [alert.to_dict() for alert in active_alerts],
        "recent_alerts": [alert.to_dict() for alert in alerts],
    }


@app.get("/health/history", tags=["health", "phase_3c"])
async def health_history(hours: int = 24):
    """
    Phase 3C: Get health metrics history
    
    Args:
        hours: Number of hours of history (default: 24)
    
    Returns:
        Time series of health metrics for trending analysis
    """
    if not service.health_monitor:
        return {
            "error": "Health monitor not initialized",
            "history": []
        }
    
    history = service.health_monitor.get_health_history(hours=hours)
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "history_points": len(history),
        "history": [h.to_dict() for h in history],
    }


# ========================================================================
# PHASE 3C-2: PERFORMANCE BENCHMARKING ENDPOINTS
# ========================================================================

@app.get("/performance/current", tags=["performance", "phase_3c_2"])
async def performance_current():
    """
    Phase 3C-2: Get current performance benchmarks
    
    Returns:
        Current performance metrics for all modules
    """
    if not service.performance_benchmarker:
        return {"error": "Performance benchmarker not initialized"}
    
    benchmarks = service.performance_benchmarker.get_current_benchmarks()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "module_count": len(benchmarks),
        "benchmarks": {k: v.to_dict() for k, v in benchmarks.items()}
    }


@app.get("/performance/comparison", tags=["performance", "phase_3c_2"])
async def performance_comparison():
    """
    Phase 3C-2: Compare module performances
    
    Returns:
        Comparative analysis of all modules
    """
    if not service.performance_benchmarker:
        return {"error": "Performance benchmarker not initialized"}
    
    try:
        comparison = service.performance_benchmarker.compare_modules()
        return comparison.to_dict()
    except ValueError as e:
        return {"error": str(e)}


@app.get("/performance/report", tags=["performance", "phase_3c_2"])
async def performance_report(hours: int = 24):
    """
    Phase 3C-2: Generate comprehensive performance report
    
    Args:
        hours: Time window for analysis (default: 24)
    
    Returns:
        Detailed performance report with recommendations
    """
    if not service.performance_benchmarker:
        return {"error": "Performance benchmarker not initialized"}
    
    report = await service.performance_benchmarker.generate_performance_report(hours)
    return report.to_dict()


@app.post("/performance/baseline/reset", tags=["performance", "phase_3c_2"])
async def reset_performance_baseline():
    """
    Phase 3C-2: Reset performance baseline
    
    Returns:
        Confirmation message
    """
    if not service.performance_benchmarker:
        return {"error": "Performance benchmarker not initialized"}
    
    service.performance_benchmarker.reset_baseline()
    return {
        "status": "success",
        "message": "Performance baseline reset",
        "timestamp": datetime.utcnow().isoformat()
    }


# ========================================================================
# PHASE 3C-3: ADAPTIVE THRESHOLD ENDPOINTS
# ========================================================================

@app.get("/thresholds/current", tags=["thresholds", "phase_3c_3"])
async def thresholds_current():
    """
    Phase 3C-3: Get current thresholds
    
    Returns:
        All current threshold values
    """
    if not service.adaptive_threshold_manager:
        return {"error": "Adaptive threshold manager not initialized"}
    
    thresholds = service.adaptive_threshold_manager.get_all_thresholds()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "thresholds": {
            module_type: {
                metric: threshold.to_dict()
                for metric, threshold in metrics.items()
            }
            for module_type, metrics in thresholds.items()
        }
    }


@app.get("/thresholds/adjustments", tags=["thresholds", "phase_3c_3"])
async def threshold_adjustments(hours: int = 24):
    """
    Phase 3C-3: Get threshold adjustment history
    
    Args:
        hours: Number of hours of history (default: 24)
    
    Returns:
        Recent threshold adjustments
    """
    if not service.adaptive_threshold_manager:
        return {"error": "Adaptive threshold manager not initialized"}
    
    adjustments = service.adaptive_threshold_manager.get_adjustment_history(hours)
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "adjustment_count": len(adjustments),
        "adjustments": [adj.to_dict() for adj in adjustments]
    }


@app.get("/thresholds/weights", tags=["thresholds", "phase_3c_3"])
async def health_weights():
    """
    Phase 3C-3: Get current health score weights
    
    Returns:
        Module weights for health score calculation
    """
    if not service.adaptive_threshold_manager:
        return {"error": "Adaptive threshold manager not initialized"}
    
    weights = service.adaptive_threshold_manager.get_health_weights()
    return weights.to_dict()


@app.post("/thresholds/override", tags=["thresholds", "phase_3c_3"])
async def override_threshold(
    module_type: str,
    metric_name: str,
    warning: float,
    error: float,
    critical: float
):
    """
    Phase 3C-3: Manually override threshold
    
    Args:
        module_type: Module type (e.g., "phase_2b")
        metric_name: Metric name (e.g., "latency_ms")
        warning: Warning threshold value
        error: Error threshold value
        critical: Critical threshold value
    
    Returns:
        Confirmation message
    """
    if not service.adaptive_threshold_manager:
        return {"error": "Adaptive threshold manager not initialized"}
    
    try:
        service.adaptive_threshold_manager.override_threshold(
            module_type, metric_name, warning, error, critical
        )
        return {
            "status": "success",
            "message": f"Threshold overridden for {module_type}.{metric_name}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except ValueError as e:
        return {"error": str(e)}


@app.get("/thresholds/predictive", tags=["thresholds", "phase_3c_3"])
async def predictive_alerts():
    """
    Phase 3C-3: Get predictive alerts
    
    Returns:
        Alerts predicting future issues based on trends
    """
    if not service.adaptive_threshold_manager:
        return {"error": "Adaptive threshold manager not initialized"}
    
    alerts = await service.adaptive_threshold_manager.generate_predictive_alerts()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "alert_count": len(alerts),
        "alerts": [alert.to_dict() for alert in alerts]
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )
