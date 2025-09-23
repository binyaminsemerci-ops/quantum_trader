from fastapi import FastAPI
from backend.routes import trades, stats, chart, settings, binance

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Quantum Trader API is running"}


# inkluder routere uten trailing slash-problemer
app.include_router(trades.router, prefix="/trades")
app.include_router(stats.router, prefix="/stats")
app.include_router(chart.router, prefix="/chart")
app.include_router(settings.router, prefix="/settings")
app.include_router(binance.router, prefix="/binance")
