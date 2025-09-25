from fastapi import FastAPI
import importlib
from backend.utils.startup import log_startup_info

# Import route modules defensively — some modules may not exist in the rebased
# subset. This prevents circular import errors during test collection.
def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

trades = _try_import("backend.routes.trades")
stats = _try_import("backend.routes.stats")
chart = _try_import("backend.routes.chart")
settings = _try_import("backend.routes.settings")
external_data = _try_import("backend.routes.external_data")
health = _try_import("backend.routes.health")


app = FastAPI()


@app.on_event("startup")
async def _on_startup() -> None:
    # Non-blocking startup logging; keep lightweight for CI and tests
    await log_startup_info()


@app.get("/")
async def root():
    return {"message": "Quantum Trader API is running"}


# include routers
if trades is not None:
    app.include_router(trades.router, prefix="/trades")
if stats is not None:
    app.include_router(stats.router, prefix="/stats")
if chart is not None:
    app.include_router(chart.router, prefix="/chart")
if settings is not None:
    app.include_router(settings.router, prefix="/settings")
if external_data is not None:
    app.include_router(external_data.router, prefix="/external_data")
if health is not None:
    app.include_router(health.router, prefix="")
