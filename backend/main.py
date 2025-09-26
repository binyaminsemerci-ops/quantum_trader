from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import trades, stats, chart, settings, binance, signals

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # eller ["*"] for enkel test
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Quantum Trader API is running"}


# inkluder routere uten trailing slash-problemer
app.include_router(trades.router, prefix="/trades")
app.include_router(stats.router, prefix="/stats")
app.include_router(chart.router, prefix="/chart")
app.include_router(settings.router, prefix="/settings")
app.include_router(binance.router, prefix="/binance")
app.include_router(signals.router, prefix="/signals")
