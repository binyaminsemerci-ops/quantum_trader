"""
QuantumFond Hedge Fund OS - Main API Entry Point
Orchestrates all backend routers and middleware
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import httpx

from routers import (
    overview_router,
    trades_router,
    risk_router,
    ai_router,
    strategy_router,
    performance_router,
    system_router,
    admin_router,
    incident_router,
    auth_router,
    journal_router,
    replay_router
)

# Import AI-Trading integration
try:
    from routers import ai_trading_link
    AI_TRADING_ENABLED = True
except ImportError:
    AI_TRADING_ENABLED = False

app = FastAPI(
    title="QuantumFond Hedge Fund OS API",
    description="Enterprise-grade hedge fund operating system with AI prediction engine",
    version="1.0.0"
)

# CORS Configuration
origins = [
    "https://app.quantumfond.com",
    "https://api.quantumfond.com",
    "https://quantumfond.com",
    "http://localhost:5173",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include all routers
app.include_router(auth_router.router)
app.include_router(overview_router.router)
app.include_router(trades_router.router)
app.include_router(risk_router.router)
app.include_router(ai_router.router)
app.include_router(strategy_router.router)
app.include_router(performance_router.router)
app.include_router(system_router.router)
app.include_router(admin_router.router)
app.include_router(incident_router.router)
app.include_router(journal_router.router)
app.include_router(replay_router.router)

# Include export router for reports
try:
    from routers import export_router
    app.include_router(export_router.router)
except ImportError:
    pass

# Include AI-Trading integration if available
if AI_TRADING_ENABLED:
    app.include_router(ai_trading_link.router)

# Background task for auto-triggering AI predictions
async def auto_trigger():
    """
    Continuously trigger AI prediction → trade execution
    Runs every 15 seconds in background
    """
    await asyncio.sleep(10)  # Wait for startup
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            try:
                await client.post("http://localhost:8026/ai-trading/trigger")
            except Exception as e:
                print(f"Auto-trigger error: {e}")
            await asyncio.sleep(15)

@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks on application startup"""
    if AI_TRADING_ENABLED:
        asyncio.create_task(auto_trigger())
        print("✅ AI Auto-Trigger Loop Started (15s interval)")

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "OK",
        "service": "QuantumFond API",
        "version": "1.0.0"
    }

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "QuantumFond Hedge Fund OS",
        "status": "operational",
        "docs": "/docs"
    }
