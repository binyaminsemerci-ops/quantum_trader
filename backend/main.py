from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import trades, stats, chart, settings, binance, signals, prices, candles

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
app.include_router(prices.router, prefix="/prices")
app.include_router(candles.router, prefix="/candles")


# Safety wrapper: some CI/test environments can fail to register routers
# (or tests may import the app before routers are fully initialized). To be
# resilient, provide a thin top-level path that delegates to the candles
# handler. This guarantees `/candles/` exists for tests that call it
# directly.
@app.get("/candles/")
async def _candles_fallback(symbol: str, limit: int = 100):
    # Import inside the handler to avoid import-time cycles in some test
    # harnesses. Use the router's function directly so we keep a single
    # implementation of the endpoint logic.
    from backend.routes import candles as _candles_mod

    # The implementation in backend.routes.candles is synchronous and
    # returns a dict, so call and return it directly.
    return _candles_mod.get_candles(symbol=symbol, limit=limit)
