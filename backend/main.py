from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.utils.logging import configure_logging, get_logger
from backend.utils.metrics import router as metrics_router, add_metrics_middleware
from backend.routes import (
    trades,
    stats,
    chart,
    settings,
    binance,
    signals,
    prices,
    candles,
    stress,
)

configure_logging()
logger = get_logger(__name__)

app = FastAPI()
add_metrics_middleware(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # eller ["*"] for enkel test
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    logger.debug("health ping")
    return {"message": "Quantum Trader API is running"}


# inkluder routere uten trailing slash-problemer
app.include_router(trades.router, prefix="/trades")
app.include_router(stats.router, prefix="/stats")
app.include_router(chart.router, prefix="/chart")
app.include_router(settings.router, prefix="/settings")
app.include_router(binance.router, prefix="/binance")
app.include_router(signals.router, prefix="/signals")
app.include_router(prices.router, prefix="/prices")
app.include_router(candles.router, prefix="/candles")
app.include_router(stress.router, prefix="/stress")
app.include_router(metrics_router, prefix="/metrics")
