from fastapi import FastAPI
from backend.routes import trades, stats, chart

app = FastAPI(title="Quantum Trader API")

app.include_router(trades.router, prefix="/api/trades", tags=["Trades"])
app.include_router(stats.router, prefix="/api/stats", tags=["Stats"])
app.include_router(chart.router, prefix="/api/chart", tags=["Chart"])


@app.get("/")
def root():
    return {"message": "Quantum Trader API is running 🚀"}
