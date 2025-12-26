from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.connection import Base, engine
from routers import ai_router, portfolio_router, risk_router, system_router, stream_router, ai_insights_router

app = FastAPI(title="Quantum Trader Dashboard API", version="0.1.0")

# CORS configuration - Production settings for quantumfond.com
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.quantumfond.com",
        "http://localhost:5173",  # Local development
        "http://localhost:8889",  # VPS testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize database tables on startup
@app.on_event("startup")
async def startup_event():
    """Create database tables on application startup"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")
        print("   Backend will run in standalone mode without database")
        # Don't crash - continue running without database

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "OK", "message": "Quantum Trader Dashboard API is running"}

@app.get("/version")
async def version():
    """Version endpoint"""
    return {"version": "0.1.0"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Quantum Trader Dashboard API"}

# Register routers
app.include_router(ai_router.router)
app.include_router(portfolio_router.router)
app.include_router(risk_router.router)
app.include_router(system_router.router)
app.include_router(stream_router.router)  # WebSocket stream for real-time updates
app.include_router(ai_insights_router.router)  # AI ensemble analytics and drift detection

# Integration router for direct access to Quantum services
from routers import integrations_router
app.include_router(integrations_router.router)
