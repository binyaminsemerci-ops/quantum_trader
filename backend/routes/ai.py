from fastapi import APIRouter, HTTPException, Response, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Any
import logging
import os
import pickle  # pragma: nosec B403 - pickle used to load internal model artifacts only
import json
import numpy as np
from ai_engine.agents.xgb_agent import make_default_agent
from ai_engine.train_and_save import train_and_save
from backend.database import create_training_task, update_training_task, get_db  # type: ignore[attr-defined]

# backend.database exports ORM symbols dynamically; narrow-ignore attr-defined for now
from backend.database import TrainingTask  # type: ignore[attr-defined]

router = APIRouter()
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    prediction: float


# Simple lazy loader for the model file under ai_engine/models/
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "..",
    "ai_engine",
    "models",
    "xgb_model.pkl",
)
_MODEL: Optional[Any] = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    # Try couple of likely locations relative to repo root
    candidates = [
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "ai_engine",
            "models",
            "xgb_model.pkl",
        ),
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "..",
            "ai_engine",
            "models",
            "xgb_model.pkl",
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    # Loading models from repository-tracked artifacts. These
                    # files are created/controlled by our build/training
                    # processes and are trusted in this context. Suppress
                    # bandit warning about pickle usage here.
                    _MODEL = pickle.load(f)  # pragma: nosec B301
                    return _MODEL
            except Exception as e:
                # Log and fall through to try other locations / fallbacks
                logger.debug("failed to load pickled model from %s: %s", p, e)
    # Try a JSON-based lightweight model spec as a tiny example model.
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "..",
        "ai_engine",
        "models",
        "xgb_model.json",
    )
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                spec = json.load(f)

            # Simple wrapper: provide a .predict() that accepts a 2D numpy array
            class JSONModel:
                def __init__(self, spec):
                    # spec may contain a scale factor or other simple rules
                    self.scale = float(spec.get("scale", 1.0))

                def predict(self, arr):
                    # arr expected shape (n_samples, n_features)
                    # We'll return mean(feature_row) * scale for each row
                    arr = np.asarray(arr, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    means = np.nanmean(arr, axis=1)
                    return means * self.scale

            _MODEL = JSONModel(spec)
            return _MODEL
        except Exception as e:
            logger.debug("failed to load JSON model spec %s: %s", json_path, e)
    # If model not found, raise so callers can fall back
    raise FileNotFoundError("Model file not found in ai_engine/models/xgb_model.pkl")


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Return a single numeric prediction for a small feature vector.

    This endpoint is intentionally minimal so it works as an MVP demo.
    If the pickled model is missing, we fall back to a trivial heuristic.
    """
    try:
        model = _load_model()
    except FileNotFoundError:
        # Simple fallback: return the mean of features as a mock prediction
        if not req.features:
            raise HTTPException(
                status_code=400, detail="features must be a non-empty list"
            )
        pred = float(np.mean(req.features))
        return PredictResponse(prediction=pred)

    try:
        arr = np.array([req.features], dtype=float)
        # Try common estimator interfaces
        if hasattr(model, "predict"):
            out = model.predict(arr)
            # return first element as scalar
            return PredictResponse(prediction=float(out[0]))
        # otherwise try sklearn API compatibility
        raise HTTPException(status_code=500, detail="Loaded model has no predict()")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}")


# Some browsers send an OPTIONS preflight request before POSTing JSON. While
# CORSMiddleware normally handles this, some dev setups (proxies, preview
# servers) can result in an OPTIONS -> 405. Provide an explicit OPTIONS handler
# to ensure preflight returns 200 for both `/predict` and the trailing-slash
# variant.
@router.options("/predict")
@router.options("/predict/")
async def predict_options():
    return Response(status_code=200)


class ScanRequest(BaseModel):
    # optional mapping of symbol -> list of candle rows (list of dicts)
    symbols: Optional[List[str]] = None
    reload_model: Optional[bool] = False


class TrainRequest(BaseModel):
    symbols: Optional[List[str]] = None
    limit: Optional[int] = 600


class ReloadResponse(BaseModel):
    loaded_model: bool
    loaded_scaler: bool


@router.post("/scan")
async def scan(req: ScanRequest):
    """Scan symbols by volume and return BUY/SELL/HOLD suggestions.

    If `symbols` is omitted, the agent expects the caller to provide a set of
    symbols from e.g. recent market list. For now we return an error if no
    symbols are provided.
    """
    if not req.symbols:
        raise HTTPException(status_code=400, detail="symbols list required")

    # In this minimal implementation we attempt to fetch OHLCV from a simple
    # in-memory source - callers should provide full OHLCV for best results.
    # We'll create a synthetic DataFrame per symbol when pandas is available;
    # otherwise we build plain Python lists-of-dicts and use a lightweight
    # heuristic so the endpoint remains usable in constrained environments.
    # Avoid shadowing the pandas module name with None; use pandas_mod so
    # mypy doesn't infer a Module | None assignment which can cause errors.
    pandas_mod: Optional[Any] = None
    try:
        import pandas as pd  # type: ignore

        pandas_mod = pd
    except Exception as e:
        # leave pandas_mod as None when pandas isn't available
        logger.debug("pandas not available: %s", e)

    from ai_engine.agents.xgb_agent import make_default_agent

    symbol_ohlcv = {}

    if pandas_mod is not None:
        # build pandas DataFrames (agent requires pandas for feature engineering)
        agent = make_default_agent()
        for s in req.symbols:
            now = pandas_mod.Timestamp.utcnow()
            rows = []
            price = 40000.0
            for i in range(120):
                ts = now - pandas_mod.Timedelta(minutes=(120 - i))
                open_p = price
                close_p = price + ((i % 5) - 2) * 10
                high_p = max(open_p, close_p) + 5
                low_p = min(open_p, close_p) - 5
                volume = 1000 + (i % 10) * 10
                rows.append(
                    {
                        "timestamp": ts.isoformat() + "Z",
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "volume": volume,
                    }
                )
                price = close_p
            df = pandas_mod.DataFrame(rows)
            symbol_ohlcv[s] = df

        # Optionally reload model artifacts before running the scan
        if getattr(req, "reload_model", False):
            try:
                agent.reload()
            except Exception as e:
                logger.debug("agent.reload() failed: %s", e)

        try:
            results = agent.scan_symbols(symbol_ohlcv, top_n=min(10, len(req.symbols)))
            return results
        except Exception as e:
            # fall through to fallback heuristic below
            logger.debug("agent.scan_symbols failed: %s", e)
    else:
        # pandas is not available: create plain lists-of-dicts using stdlib datetime
        import datetime

        for s in req.symbols:
            now = datetime.datetime.now(datetime.timezone.utc)
            rows = []
            price = 40000.0
            for i in range(120):
                ts = now - datetime.timedelta(minutes=(120 - i))
                open_p = price
                close_p = price + ((i % 5) - 2) * 10
                high_p = max(open_p, close_p) + 5
                low_p = min(open_p, close_p) - 5
                volume = 1000 + (i % 10) * 10
                rows.append(
                    {
                        "timestamp": ts.isoformat() + "Z",
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "volume": volume,
                    }
                )
                price = close_p
            symbol_ohlcv[s] = rows

    # Fallback heuristic (pure Python) if agent or pandas isn't usable
    out = {}
    for s, df in symbol_ohlcv.items():
        try:
            # df may be list-of-dicts or pandas DataFrame; handle both
            if isinstance(df, list):
                if len(df) >= 2:
                    last = float(df[-1]["close"])
                    prev = float(df[-2]["close"])
                else:
                    last = float(df[-1]["close"])
                    prev = last
            else:
                # pandas DataFrame path
                last = float(df["close"].iloc[-1])
                prev = float(df["close"].iloc[-2]) if len(df) >= 2 else last
            change = (last - prev) / prev if prev != 0 else 0
            # simple thresholds
            if change > 0.002:
                action = "BUY"
                score = min(0.99, change * 100)
            elif change < -0.002:
                action = "SELL"
                score = min(0.99, abs(change) * 100)
            else:
                action = "HOLD"
                score = 0.0
        except Exception as e:
            logger.debug("fallback heuristic error for symbol %s: %s", s, e)
            action = "HOLD"
            score = 0.0
        out[s] = {"action": action, "score": float(score)}
    return out


@router.post("/reload", response_model=ReloadResponse)
async def reload_model_endpoint():
    """Reload model and scaler artifacts for the default agent."""
    agent = make_default_agent()
    agent.reload()
    return ReloadResponse(
        loaded_model=(agent.model is not None), loaded_scaler=(agent.scaler is not None)
    )


@router.post("/train")
async def train_endpoint(req: TrainRequest, background: BackgroundTasks):
    """Schedule a background training run. Returns 202 accepted immediately."""
    # Use USDC as the spot quote by default; futures/cross-margin can still use USDT
    from config.config import DEFAULT_QUOTE

    symbols = req.symbols or [f"BTC{DEFAULT_QUOTE}", f"ETH{DEFAULT_QUOTE}"]
    limit = req.limit or 600
    # create a DB task record
    # get a session for creating the task synchronously
    db = next(get_db())
    task = create_training_task(db, ",".join(symbols), limit)

    # background wrapper to run training and update task status
    def _bg_train(task_id: int, symbols_list: list, limit_val: int):
        db2 = next(get_db())
        try:
            update_training_task(db2, task_id, "running")
            train_and_save(symbols=symbols_list, limit=limit_val)
            update_training_task(db2, task_id, "completed", details="ok")
        except Exception as e:
            try:
                update_training_task(db2, task_id, "failed", details=str(e))
            except Exception:
                pass

    # schedule background training
    background.add_task(_bg_train, task.id, symbols, limit)
    return {
        "status": "scheduled",
        "task_id": task.id,
        "symbols": symbols,
        "limit": limit,
    }


@router.get("/tasks")
async def list_tasks(limit: int = 50, offset: int = 0):
    """Return a paginated list of recent training tasks."""
    db = next(get_db())
    try:
        q = (
            db.query(TrainingTask)
            .order_by(TrainingTask.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        items = []
        for t in q.all():
            items.append(
                {
                    "id": t.id,
                    "symbols": t.symbols,
                    "limit": t.limit,
                    "status": t.status,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "completed_at": (
                        t.completed_at.isoformat() if t.completed_at else None
                    ),
                    "details": t.details,
                }
            )
        return {"tasks": items, "count": len(items)}
    finally:
        db.close()


@router.get("/tasks/{task_id}")
async def get_task(task_id: int):
    db = next(get_db())
    try:
        t = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if not t:
            raise HTTPException(status_code=404, detail="task not found")
        return {
            "id": t.id,
            "symbols": t.symbols,
            "limit": t.limit,
            "status": t.status,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            "details": t.details,
        }
    finally:
        db.close()


@router.get("/status")
async def status_endpoint():
    agent = make_default_agent()
    meta = None
    try:
        meta = agent.get_metadata()
    except Exception as e:
        logger.debug("agent.get_metadata failed: %s", e)
        meta = None
    return {
        "loaded_model": agent.model is not None,
        "loaded_scaler": agent.scaler is not None,
        "metadata": meta,
    }
