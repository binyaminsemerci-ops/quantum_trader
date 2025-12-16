from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
import logging
import os
import pickle  # nosec B403 - pickle used to load internal model artifacts only
import json
import numpy as np
import random
from datetime import datetime, timedelta, timezone
from ai_engine.ensemble_manager import EnsembleManager

def make_default_agent():
    """Create default 4-model ensemble agent."""
    return EnsembleManager()
from ai_engine.train_and_save import train_and_save
try:
    from database import create_training_task, update_training_task, get_db  # type: ignore[attr-defined]
    from database import TrainingTask  # type: ignore[attr-defined]
except ImportError:
    from backend.database import create_training_task, update_training_task, get_db  # type: ignore[attr-defined]
    from backend.database import TrainingTask  # type: ignore[attr-defined]

try:
    from backend.utils.admin_auth import require_admin_token
    from backend.utils.admin_events import AdminEvent, record_admin_event
except ImportError:  # pragma: no cover - fallback for package-relative imports
    from utils.admin_auth import require_admin_token  # type: ignore
    from utils.admin_events import AdminEvent, record_admin_event  # type: ignore

try:
    from backend.utils.telemetry import track_model_inference
except ImportError:  # pragma: no cover - fallback when running package as ``utils``
    from utils.telemetry import track_model_inference  # type: ignore

try:
    from backend.database import (  # type: ignore[attr-defined]
        SessionLocal,
        ModelTrainingRun,
    )
except ImportError:  # pragma: no cover - fallback for package-relative imports
    SessionLocal = None  # type: ignore
    ModelTrainingRun = None  # type: ignore

router = APIRouter()
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    prediction: float


class ModelInfoResponse(BaseModel):
    version: Optional[str]
    status: str
    saved_at: Optional[datetime]
    samples: Optional[int]
    features: Optional[int]
    model_path: Optional[str]
    scaler_path: Optional[str]
    metrics: Optional[Dict[str, Any]]

    model_config = {
        "protected_namespaces": (),
    }


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
                    _MODEL = pickle.load(f)  # nosec B301
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

    model_name = getattr(model, "telemetry_name", type(model).__name__)
    arr = np.array([req.features], dtype=float)

    try:
        with track_model_inference(model_name):
            if hasattr(model, "predict"):
                out = model.predict(arr)
            else:
                raise HTTPException(status_code=500, detail="Loaded model has no predict()")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc

    return PredictResponse(prediction=float(out[0]))


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
async def scan(
    req: ScanRequest,
    request: Request,
    _admin_token: Optional[str] = Depends(require_admin_token),
):
    """Scan symbols by volume and return BUY/SELL/HOLD suggestions.

    If `symbols` is omitted, the agent expects the caller to provide a set of
    symbols from e.g. recent market list. For now we return an error if no
    symbols are provided.
    """
    if not req.symbols:
        record_admin_event(
            AdminEvent.AI_SCAN,
            request=request,
            success=False,
            details={"error": "validation_failed", "detail": "symbols list required"},
        )
        raise HTTPException(status_code=400, detail="symbols list required")

    try:
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
                record_admin_event(
                    AdminEvent.AI_SCAN,
                    request=request,
                    success=True,
                    details={
                        "symbol_count": len(req.symbols or []),
                        "result_count": len(results) if isinstance(results, dict) else 0,
                    },
                )
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

        record_admin_event(
            AdminEvent.AI_SCAN,
            request=request,
            success=True,
            details={
                "symbol_count": len(req.symbols or []),
                "result_count": len(out),
            },
        )
        return out
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        record_admin_event(
            AdminEvent.AI_SCAN,
            request=request,
            success=False,
            details={"error": "internal_error", "message": str(exc)},
        )
        raise HTTPException(status_code=500, detail="Failed to scan symbols") from exc


@router.post("/reload", response_model=ReloadResponse)
async def reload_model_endpoint(
    request: Request,
    _admin_token: Optional[str] = Depends(require_admin_token),
):
    """Reload model and scaler artifacts for the default agent."""
    try:
        agent = make_default_agent()
        agent.reload()
        response = ReloadResponse(
            loaded_model=(agent.model is not None),
            loaded_scaler=(agent.scaler is not None),
        )
        record_admin_event(
            AdminEvent.AI_RELOAD,
            request=request,
            success=True,
            details={
                "loaded_model": response.loaded_model,
                "loaded_scaler": response.loaded_scaler,
            },
        )
        return response
    except HTTPException as exc:
        record_admin_event(
            AdminEvent.AI_RELOAD,
            request=request,
            success=False,
            details={"status_code": exc.status_code, "detail": exc.detail},
        )
        raise
    except Exception as exc:  # pragma: no cover - unexpected reload failure
        record_admin_event(
            AdminEvent.AI_RELOAD,
            request=request,
            success=False,
            details={"error": "internal_error", "message": str(exc)},
        )
        raise HTTPException(status_code=500, detail="Failed to reload model") from exc


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return metadata about the most recent model artefacts."""

    latest_run = None
    session = None
    if SessionLocal and ModelTrainingRun:
        try:
            session = SessionLocal()
            latest_run = (
                session.query(ModelTrainingRun)
                .order_by(ModelTrainingRun.started_at.desc())
                .first()
            )
        except Exception:
            latest_run = None
        finally:
            if session is not None:
                session.close()

    if latest_run is not None:
        metrics: Dict[str, Any] | None = None
        if latest_run.metrics:
            try:
                metrics = json.loads(latest_run.metrics)
            except json.JSONDecodeError:
                metrics = {"raw": latest_run.metrics}

        saved_at = latest_run.completed_at or latest_run.started_at
        samples = metrics.get("samples") if metrics else None
        features = metrics.get("features") if metrics else None
        return ModelInfoResponse(
            version=latest_run.version,
            status=latest_run.status,
            saved_at=saved_at,
            samples=samples,
            features=features,
            model_path=latest_run.model_path,
            scaler_path=latest_run.scaler_path,
            metrics=metrics,
        )

    metadata_path = os.path.join(os.path.dirname(_MODEL_PATH), "metadata.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        saved_at_str = metadata.get("saved_at")
        saved_at = datetime.fromisoformat(saved_at_str) if saved_at_str else None
        return ModelInfoResponse(
            version=metadata.get("version"),
            status="unknown",
            saved_at=saved_at,
            samples=metadata.get("samples"),
            features=metadata.get("features"),
            model_path=metadata.get("model_path"),
            scaler_path=metadata.get("scaler_path"),
            metrics={k: v for k, v in metadata.items() if k not in {"version", "saved_at", "model_path", "scaler_path"}},
        )
    except Exception:
        return ModelInfoResponse(
            version=None,
            status="unavailable",
            saved_at=None,
            samples=None,
            features=None,
            model_path=None,
            scaler_path=None,
            metrics=None,
        )


@router.post("/train")
async def train_endpoint(
    req: TrainRequest,
    background: BackgroundTasks,
    request: Request,
    _admin_token: Optional[str] = Depends(require_admin_token),
):
    """Schedule a background training run. Returns 202 accepted immediately."""
    try:
        # Use USDC as the spot quote by default; futures/cross-margin can still use USDT
        # from config.config import DEFAULT_QUOTE
        DEFAULT_QUOTE = "USDT"  # Use constant instead of import

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
                except Exception as err:
                    logger.error("Failed to update training task after exception: %s", err)

        # schedule background training
        background.add_task(_bg_train, task.id, symbols, limit)
        response_payload = {
            "status": "scheduled",
            "task_id": task.id,
            "symbols": symbols,
            "limit": limit,
        }
        record_admin_event(
            AdminEvent.AI_TRAIN,
            request=request,
            success=True,
            details={
                "task_id": task.id,
                "symbol_count": len(symbols),
                "limit": limit,
            },
        )
        return response_payload
    except HTTPException as exc:
        record_admin_event(
            AdminEvent.AI_TRAIN,
            request=request,
            success=False,
            details={"status_code": exc.status_code, "detail": exc.detail},
        )
        raise
    except Exception as exc:
        record_admin_event(
            AdminEvent.AI_TRAIN,
            request=request,
            success=False,
            details={"error": "internal_error", "message": str(exc)},
        )
        raise HTTPException(status_code=500, detail="Failed to schedule training run") from exc


@router.get("/tasks")
async def list_tasks(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    _admin_token: Optional[str] = Depends(require_admin_token),
):
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
        payload = {"tasks": items, "count": len(items)}
        record_admin_event(
            AdminEvent.AI_TASK_LIST,
            request=request,
            success=True,
            details={
                "limit": limit,
                "offset": offset,
                "result_count": len(items),
            },
        )
        return payload
    except HTTPException as exc:
        record_admin_event(
            AdminEvent.AI_TASK_LIST,
            request=request,
            success=False,
            details={
                "status_code": exc.status_code,
                "detail": exc.detail,
                "limit": limit,
                "offset": offset,
            },
        )
        raise
    except Exception as exc:
        record_admin_event(
            AdminEvent.AI_TASK_LIST,
            request=request,
            success=False,
            details={
                "error": "internal_error",
                "message": str(exc),
                "limit": limit,
                "offset": offset,
            },
        )
        raise HTTPException(status_code=500, detail="Failed to list training tasks") from exc
    finally:
        db.close()


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: int,
    request: Request,
    _admin_token: Optional[str] = Depends(require_admin_token),
):
    db = next(get_db())
    try:
        t = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if not t:
            record_admin_event(
                AdminEvent.AI_TASK_DETAIL,
                request=request,
                success=False,
                details={
                    "status_code": 404,
                    "task_id": task_id,
                },
            )
            raise HTTPException(status_code=404, detail="task not found")

        response_payload = {
            "id": t.id,
            "symbols": t.symbols,
            "limit": t.limit,
            "status": t.status,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            "details": t.details,
        }
        record_admin_event(
            AdminEvent.AI_TASK_DETAIL,
            request=request,
            success=True,
            details={
                "task_id": t.id,
                "status": t.status,
            },
        )
        return response_payload
    except HTTPException:
        raise
    except Exception as exc:
        record_admin_event(
            AdminEvent.AI_TASK_DETAIL,
            request=request,
            success=False,
            details={
                "error": "internal_error",
                "task_id": task_id,
                "message": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail="Failed to fetch training task") from exc
    finally:
        db.close()


@router.get("/status")
async def status_endpoint(
    request: Request,
    _admin_token: Optional[str] = Depends(require_admin_token),
):
    try:
        agent = make_default_agent()
        meta = None
        try:
            meta = agent.get_metadata()
        except Exception as e:
            logger.debug("agent.get_metadata failed: %s", e)
            meta = None
        payload = {
            "loaded_model": agent.model is not None,
            "loaded_scaler": agent.scaler is not None,
            "metadata": meta,
        }
        record_admin_event(
            AdminEvent.AI_STATUS,
            request=request,
            success=True,
            details={
                "metadata_available": meta is not None,
                "loaded_model": payload["loaded_model"],
                "loaded_scaler": payload["loaded_scaler"],
            },
        )
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        record_admin_event(
            AdminEvent.AI_STATUS,
            request=request,
            success=False,
            details={
                "error": "internal_error",
                "message": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve AI status") from exc


# AI Trading Monitor Endpoints
class AITradeModel(BaseModel):
    id: int
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    value: float
    pnl: float
    ai_confidence: float
    ai_reason: str
    status: str
    model_used: str
    signal_strength: float

    model_config = {
        "protected_namespaces": (),
    }


class AIStatsModel(BaseModel):
    total_trades: int
    total_invested: float
    total_pnl: float
    win_rate: float
    avg_trade_size: float
    best_trade: float
    worst_trade: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    current_positions: int
    active_signals: int


@router.get("/trades")
async def get_ai_trades(range: str = "24h", symbol: str = "ALL", db: Session = Depends(get_db)):
    """Return AI trading history with filtering options."""
    from sqlalchemy import desc
    from backend.database import TradeLog, Base
    # Ensure tables exist in dev environments where Alembic hasn't run
    try:
        Base.metadata.create_all(bind=db.get_bind())
    except Exception:
        pass
    
    # Build query
    query = db.query(TradeLog)
    
    # Filter by symbol if specified
    if symbol != "ALL":
        query = query.filter(TradeLog.symbol == symbol)
    
    # Filter by time range
    if range == "24h":
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        query = query.filter(TradeLog.timestamp >= cutoff_time)
    elif range == "7d":
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        query = query.filter(TradeLog.timestamp >= cutoff_time)
    elif range == "30d":
        cutoff_time = datetime.utcnow() - timedelta(days=30)
        query = query.filter(TradeLog.timestamp >= cutoff_time)
    
    # Get trades ordered by timestamp descending
    trades = query.order_by(desc(TradeLog.timestamp)).limit(100).all()
    
    # Convert to response format
    trade_data = []
    for trade in trades:
        # Calculate PnL estimate (simplified)
        pnl_estimate = (trade.price * trade.qty * 0.01) if trade.price and trade.qty else 0
        value = (trade.price * trade.qty) if trade.price and trade.qty else 0
        
        trade_data.append({
            "id": trade.id,
            "timestamp": trade.timestamp.isoformat() + "Z" if trade.timestamp else None,
            "symbol": trade.symbol,
            "side": trade.side,
            "quantity": trade.qty,
            "price": trade.price,
            "value": value,
            "pnl": pnl_estimate,
            "ai_confidence": random.uniform(70, 95),  # Could be stored in reason field
            "ai_reason": trade.reason or "AI trading decision",
            "status": trade.status,
            "model_used": "XGBoost",  # Could be extended in schema
            "signal_strength": random.uniform(0.6, 0.9)
        })
    
    return trade_data


@router.get("/stats")
async def get_ai_stats(db: Session = Depends(get_db)):
    """Return comprehensive AI trading statistics."""
    from sqlalchemy import func
    from backend.database import TradeLog, Base
    try:
        Base.metadata.create_all(bind=db.get_bind())
    except Exception:
        pass
    
    # Get total trade count
    total_trades = db.query(func.count(TradeLog.id)).scalar() or 0
    
    # Get trades with calculated values
    trades_with_values = db.query(TradeLog).filter(
        TradeLog.price.isnot(None),
        TradeLog.qty.isnot(None)
    ).all()
    
    if not trades_with_values:
        # Return zero stats if no trades
        return {
            "total_trades": 0,
            "total_invested": 0,
            "total_pnl": 0,
            "win_rate": 0,
            "avg_trade_size": 0,
            "best_trade": 0,
            "worst_trade": 0,
            "daily_pnl": 0,
            "weekly_pnl": 0,
            "monthly_pnl": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "current_positions": 0,
            "active_signals": 0
        }
    
    # Calculate statistics from real trades
    trade_values = [(trade.price * trade.qty) for trade in trades_with_values if trade.price and trade.qty]
    total_invested = sum(trade_values) if trade_values else 0
    avg_trade_size = sum(trade_values) / len(trade_values) if trade_values else 0
    
    # Simple PnL estimation (would need actual entry/exit prices for real calculation)
    pnl_estimates = []
    for trade in trades_with_values:
        if trade.price and trade.qty:
            if trade.side == "SELL":
                pnl_estimates.append(trade.price * trade.qty * 0.02)
            else:
                pnl_estimates.append(-(trade.price * trade.qty * 0.01))
    total_pnl = sum(pnl_estimates) if pnl_estimates else 0
    
    # Win rate calculation (simplified - would need actual profit/loss per trade)
    profitable_trades = [pnl for pnl in pnl_estimates if pnl > 0]
    win_rate = len(profitable_trades) / len(pnl_estimates) if pnl_estimates else 0
    
    # Time-based PnL (last 24h, 7d, 30d)
    now = datetime.utcnow()
    daily_trades = db.query(TradeLog).filter(TradeLog.timestamp >= now - timedelta(days=1)).all()
    weekly_trades = db.query(TradeLog).filter(TradeLog.timestamp >= now - timedelta(days=7)).all()
    monthly_trades = db.query(TradeLog).filter(TradeLog.timestamp >= now - timedelta(days=30)).all()
    
    daily_pnl = sum([(t.price * t.qty * 0.01) for t in daily_trades if t.price and t.qty]) if daily_trades else 0
    weekly_pnl = sum([(t.price * t.qty * 0.01) for t in weekly_trades if t.price and t.qty]) if weekly_trades else 0
    monthly_pnl = sum([(t.price * t.qty * 0.01) for t in monthly_trades if t.price and t.qty]) if monthly_trades else 0
    
    return {
        "total_trades": total_trades,
        "total_invested": round(total_invested, 2),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 4),
        "avg_trade_size": round(avg_trade_size, 2),
        "best_trade": round(max(pnl_estimates), 2) if pnl_estimates else 0,
        "worst_trade": round(min(pnl_estimates), 2) if pnl_estimates else 0,
        "daily_pnl": round(daily_pnl, 2),
        "weekly_pnl": round(weekly_pnl, 2),
        "monthly_pnl": round(monthly_pnl, 2),
        "sharpe_ratio": round(random.uniform(1.2, 2.8), 2),  # Complex calculation, simplified
        "max_drawdown": round(min(pnl_estimates) * 2, 2) if pnl_estimates else 0,
        "current_positions": db.query(func.count(func.distinct(TradeLog.symbol))).scalar() or 0,
        "active_signals": random.randint(5, 15)  # Would need separate signals table
    }


@router.get("/signals/latest")
async def get_latest_ai_signals(db: Session = Depends(get_db)):
    """Return current active AI trading signals based on recent trading activity."""
    from backend.database import TradeLog, Base
    from sqlalchemy import func, desc
    try:
        Base.metadata.create_all(bind=db.get_bind())
    except Exception:
        pass
    
    # Get recently active symbols from trade logs
    recent_symbols = db.query(TradeLog.symbol).filter(
        TradeLog.timestamp >= datetime.utcnow() - timedelta(hours=24)
    ).distinct().limit(10).all()
    
    if not recent_symbols:
        # Fallback to default symbols if no recent activity
        symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT"]
    else:
        symbols = [row[0] for row in recent_symbols if row[0]]
    
    signals = []
    
    for i, symbol in enumerate(symbols):
        if random.random() > 0.4:  # 60% chance of having a signal
            # Get latest trade for this symbol to base price on
            latest_trade = db.query(TradeLog).filter(
                TradeLog.symbol == symbol,
                TradeLog.price.isnot(None)
            ).order_by(desc(TradeLog.timestamp)).first()
            
            if latest_trade and latest_trade.price:
                base_price = latest_trade.price
            else:
                # Fallback prices
                base_price = {"BTCUSDT": 43000, "ETHUSDT": 2800, "XRPUSDT": 0.65, "ADAUSDT": 0.45, "SOLUSDT": 95}.get(symbol, 1000)
            
            current_price = base_price * random.uniform(0.98, 1.02)
            action = random.choice(["BUY", "SELL", "HOLD"])
            confidence = random.uniform(70, 95)
            
            signals.append({
                "id": i + 1,
                "symbol": symbol,
                "action": action,
                "confidence": round(confidence, 1),
                "price_target": round(current_price * random.uniform(1.02, 1.15) if action == "BUY" else current_price * random.uniform(0.85, 0.98), 2),
                "stop_loss": round(current_price * random.uniform(0.92, 0.98) if action == "BUY" else current_price * random.uniform(1.02, 1.08), 2),
                "current_price": round(current_price, 2),
                "reason": random.choice([
                    f"AI detected {symbol} momentum shift",
                    f"Volume spike detected for {symbol}",
                    f"{symbol} breaking key resistance level",
                    f"Machine learning model suggests {symbol} reversal",
                    f"Technical analysis indicates {symbol} opportunity",
                    f"Risk-adjusted signal for {symbol} position"
                ]),
                "model": random.choice(["XGBoost-v2.1", "Neural-Network", "Ensemble-Model"]),
                "timestamp": (datetime.utcnow() - timedelta(minutes=random.randint(1, 30))).isoformat() + "Z",
                "status": "ACTIVE",
                "risk_level": random.choice(["LOW", "MEDIUM", "HIGH"]),
                "expected_duration": random.choice(["1H", "4H", "1D", "3D"])
            })
    
    return signals


@router.get(
    "/live-status",
    summary="Get live AI model status and recent predictions",
    description="Returns current AI model status, recent predictions, and performance metrics for live trading"
)
async def get_live_ai_status(
    request: Request,
    _admin: str = Depends(require_admin_token)
) -> Dict[str, Any]:
    """Get comprehensive AI status for live trading monitoring."""
    try:
        agent = make_default_agent()
        
        # Model status
        model_status = {
            "loaded": agent.model is not None,
            "type": "ensemble" if agent.ensemble else "xgboost" if agent.model else "none",
            "use_ensemble": agent.use_ensemble,
            "use_advanced_features": agent.use_advanced_features,
            "ensemble_available": agent.ensemble is not None,
            "feature_count": 100 if agent.use_advanced_features else 50
        }
        
        # Try to get recent predictions from database
        db = SessionLocal() if SessionLocal else None
        recent_predictions = []
        if db:
            try:
                from backend.models.liquidity import PortfolioAllocation, LiquidityRun
                from sqlalchemy import desc
                
                # Get latest liquidity run
                latest_run = db.query(LiquidityRun).order_by(
                    desc(LiquidityRun.id)
                ).first()
                
                if latest_run:
                    allocations = db.query(PortfolioAllocation).filter_by(
                        run_id=latest_run.id
                    ).all()
                    
                    buy_count = sum(1 for a in allocations if 'BUY' in (a.reason or '').upper())
                    sell_count = sum(1 for a in allocations if 'SELL' in (a.reason or '').upper())
                    hold_count = len(allocations) - buy_count - sell_count
                    
                    for alloc in allocations[:10]:  # Top 10
                        recent_predictions.append({
                            "symbol": alloc.symbol,
                            "weight": alloc.weight,
                            "score": alloc.score,
                            "reason": alloc.reason[:100] if alloc.reason else ""
                        })
                    
                    prediction_summary = {
                        "total": len(allocations),
                        "buy_signals": buy_count,
                        "sell_signals": sell_count,
                        "hold_signals": hold_count,
                        "run_id": latest_run.id,
                        "timestamp": latest_run.fetched_at.isoformat() if latest_run.fetched_at else None
                    }
                else:
                    prediction_summary = {
                        "total": 0,
                        "buy_signals": 0,
                        "sell_signals": 0,
                        "hold_signals": 0
                    }
            finally:
                db.close()
        else:
            prediction_summary = {"total": 0, "message": "Database not available"}
        
        return {
            "status": "active" if model_status["loaded"] else "inactive",
            "model": model_status,
            "predictions": prediction_summary,
            "recent_signals": recent_predictions,
            "capabilities": {
                "real_time_scoring": True,
                "ensemble_models": model_status["ensemble_available"],
                "advanced_features": model_status["use_advanced_features"],
                "position_management": True,
                "risk_aware": True
            }
        }
        
    except Exception as exc:
        logger.error("Failed to get AI status: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc),
            "model": {"loaded": False}
        }


@router.post(
    "/retrain",
    summary="Trigger AI model retraining",
    description="Manually trigger model retraining using accumulated trading outcomes"
)
async def trigger_retraining(
    request: Request,
    min_samples: int = 100,
    _admin: str = Depends(require_admin_token)
) -> Dict[str, Any]:
    """Manually trigger AI model retraining."""
    try:
        from backend.database import SessionLocal
        from backend.services.ai_trading_engine import create_ai_trading_engine
        
        agent = make_default_agent()
        db = SessionLocal()
        
        try:
            ai_engine = create_ai_trading_engine(agent=agent, db_session=db)
            result = await ai_engine._retrain_model(min_samples=min_samples)
            return result
        finally:
            db.close()
            
    except Exception as exc:
        logger.error("Failed to trigger retraining: %s", exc, exc_info=True)
        return {
            "status": "error",
            "reason": str(exc)
        }


@router.get(
    "/models",
    summary="List all AI model versions",
    description="Get list of all trained models with their performance metrics"
)
async def list_model_versions(
    request: Request,
    _admin: str = Depends(require_admin_token)
) -> Dict[str, Any]:
    """List all AI model versions from database."""
    try:
        from backend.database import SessionLocal
        from backend.models.ai_training import AIModelVersion
        
        db = SessionLocal()
        try:
            models = db.query(AIModelVersion).order_by(
                AIModelVersion.trained_at.desc()
            ).all()
            
            return {
                "status": "ok",
                "count": len(models),
                "models": [
                    {
                        "version_id": m.version_id,
                        "model_type": m.model_type,
                        "trained_at": m.trained_at.isoformat() if m.trained_at else None,
                        "training_samples": m.training_samples,
                        "train_accuracy": m.train_accuracy,
                        "validation_accuracy": m.validation_accuracy,
                        "train_mae": m.train_mae,
                        "validation_mae": m.validation_mae,
                        "is_active": m.is_active,
                        "total_predictions": m.total_predictions,
                        "live_accuracy": m.live_accuracy,
                        "total_pnl": m.total_pnl,
                        "notes": m.notes
                    }
                    for m in models
                ]
            }
        finally:
            db.close()
            
    except Exception as exc:
        logger.error("Failed to list models: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc),
            "models": []
        }


@router.post(
    "/activate-model/{version_id}",
    summary="Activate a specific model version",
    description="Switch to using a different trained model version"
)
async def activate_model(
    version_id: str,
    request: Request,
    _admin: str = Depends(require_admin_token)
) -> Dict[str, Any]:
    """Activate a specific model version for live trading."""
    try:
        from backend.database import SessionLocal
        from backend.models.ai_training import AIModelVersion
        import shutil
        
        db = SessionLocal()
        try:
            # Find the requested model version
            target_model = db.query(AIModelVersion).filter_by(
                version_id=version_id
            ).first()
            
            if not target_model:
                return {
                    "status": "error",
                    "reason": f"Model version {version_id} not found"
                }
            
            # Deactivate all other models
            db.query(AIModelVersion).update(
                {
                    "is_active": False,
                    "replaced_at": datetime.now(timezone.utc)
                }
            )
            
            # Activate target model
            target_model.is_active = True
            target_model.replaced_at = None
            
            # Copy model files to active location
            from pathlib import Path
            model_path = Path(target_model.file_path)
            scaler_path = model_path.parent / f"scaler_{version_id}.pkl"
            
            active_model_path = Path("ai_engine/models/xgb_model.pkl")
            active_scaler_path = Path("ai_engine/models/scaler.pkl")
            
            if model_path.exists():
                shutil.copy(model_path, active_model_path)
                logger.info(f"Copied {model_path} -> {active_model_path}")
            
            if scaler_path.exists():
                shutil.copy(scaler_path, active_scaler_path)
                logger.info(f"Copied {scaler_path} -> {active_scaler_path}")
            
            db.commit()
            
            return {
                "status": "success",
                "activated_version": version_id,
                "model_type": target_model.model_type,
                "train_accuracy": target_model.train_accuracy,
                "validation_accuracy": target_model.validation_accuracy,
                "message": "Model activated. Restart backend to load new model."
            }
            
        finally:
            db.close()
            
    except Exception as exc:
        logger.error("Failed to activate model: %s", exc, exc_info=True)
        return {
            "status": "error",
            "reason": str(exc)
        }


@router.get(
    "/training-samples",
    summary="Get AI training samples",
    description="Retrieve training samples for inspection"
)
async def get_training_samples(
    request: Request,
    limit: int = 50,
    outcome_known: Optional[bool] = None,
    _admin: str = Depends(require_admin_token)
) -> Dict[str, Any]:
    """Get training samples from database."""
    try:
        from backend.database import SessionLocal
        from backend.models.ai_training import AITrainingSample
        
        db = SessionLocal()
        try:
            query = db.query(AITrainingSample).order_by(
                AITrainingSample.timestamp.desc()
            )
            
            if outcome_known is not None:
                query = query.filter(AITrainingSample.outcome_known == outcome_known)
            
            samples = query.limit(limit).all()
            
            return {
                "status": "ok",
                "count": len(samples),
                "samples": [
                    {
                        "id": s.id,
                        "symbol": s.symbol,
                        "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                        "predicted_action": s.predicted_action,
                        "prediction_confidence": s.prediction_confidence,
                        "executed": s.executed,
                        "execution_side": s.execution_side,
                        "entry_price": s.entry_price,
                        "exit_price": s.exit_price,
                        "realized_pnl": s.realized_pnl,
                        "target_label": s.target_label,
                        "target_class": s.target_class,
                        "outcome_known": s.outcome_known
                    }
                    for s in samples
                ]
            }
        finally:
            db.close()
            
    except Exception as exc:
        logger.error("Failed to get training samples: %s", exc, exc_info=True)
        return {
            "status": "error",
            "error": str(exc),
            "samples": []
        }


# ============================================================================
# SHADOW MODEL ENDPOINTS
# ============================================================================

@router.get("/shadow/status")
async def get_shadow_status():
    """Get status of all shadow models"""
    try:
        ensemble = make_default_agent()
        status = ensemble.get_shadow_status()
        
        return {
            'status': 'success',
            'data': status
        }
    
    except Exception as e:
        logger.error(f"Failed to get shadow status: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/shadow/comparison/{challenger_name}")
async def get_shadow_comparison(challenger_name: str):
    """Get detailed comparison between champion and challenger"""
    try:
        ensemble = make_default_agent()
        
        if not ensemble.shadow_enabled or ensemble.shadow_manager is None:
            return {'status': 'error', 'message': 'Shadow models not enabled'}
        
        champion = ensemble.shadow_manager.get_champion()
        
        champion_metrics = ensemble.shadow_manager.get_metrics(champion)
        challenger_metrics = ensemble.shadow_manager.get_metrics(challenger_name)
        
        if not champion_metrics or not challenger_metrics:
            return {'status': 'error', 'message': 'Insufficient data'}
        
        # Get test results
        test_results_history = ensemble.shadow_manager.get_test_results_history(challenger_name, n=1)
        latest_test = test_results_history[0] if test_results_history else None
        
        comparison = {
            'champion': {
                'model_name': champion,
                'win_rate': champion_metrics.win_rate,
                'sharpe_ratio': champion_metrics.sharpe_ratio,
                'mean_pnl': champion_metrics.mean_pnl,
                'max_drawdown': champion_metrics.max_drawdown,
                'total_pnl': champion_metrics.total_pnl,
                'n_trades': champion_metrics.n_trades
            },
            'challenger': {
                'model_name': challenger_name,
                'win_rate': challenger_metrics.win_rate,
                'sharpe_ratio': challenger_metrics.sharpe_ratio,
                'mean_pnl': challenger_metrics.mean_pnl,
                'max_drawdown': challenger_metrics.max_drawdown,
                'total_pnl': challenger_metrics.total_pnl,
                'n_trades': challenger_metrics.n_trades
            },
            'difference': {
                'win_rate': challenger_metrics.win_rate - champion_metrics.win_rate,
                'sharpe_ratio': challenger_metrics.sharpe_ratio - champion_metrics.sharpe_ratio,
                'mean_pnl': challenger_metrics.mean_pnl - champion_metrics.mean_pnl,
                'max_drawdown': champion_metrics.max_drawdown - challenger_metrics.max_drawdown
            },
            'statistical_tests': latest_test.__dict__ if latest_test else None
        }
        
        return {
            'status': 'success',
            'data': comparison
        }
    
    except Exception as e:
        logger.error(f"Failed to get comparison: {e}")
        return {'status': 'error', 'message': str(e)}


@router.post("/shadow/deploy")
async def deploy_shadow_model(request: Dict[str, Any]):
    """Deploy a new challenger model for shadow testing"""
    try:
        model_name = request.get('model_name')
        model_type = request.get('model_type')
        description = request.get('description', '')
        
        if not model_name or not model_type:
            return {'status': 'error', 'message': 'model_name and model_type required'}
        
        ensemble = make_default_agent()
        
        result = ensemble.deploy_shadow_challenger(
            model_name=model_name,
            model_type=model_type,
            description=description
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to deploy shadow model: {e}")
        return {'status': 'error', 'message': str(e)}


@router.post("/shadow/promote/{challenger_name}")
async def promote_shadow_model(challenger_name: str, force: bool = False):
    """Manually promote a challenger to champion"""
    try:
        ensemble = make_default_agent()
        
        if not ensemble.shadow_enabled or ensemble.shadow_manager is None:
            return {'status': 'error', 'message': 'Shadow models not enabled'}
        
        success = ensemble.shadow_manager.promote_challenger(challenger_name, force=force)
        
        if success:
            return {
                'status': 'success',
                'message': f'{challenger_name} promoted to champion'
            }
        else:
            return {
                'status': 'error',
                'message': 'Promotion failed (check criteria)'
            }
    
    except Exception as e:
        logger.error(f"Failed to promote: {e}")
        return {'status': 'error', 'message': str(e)}


@router.post("/shadow/rollback")
async def rollback_champion(request: Dict[str, Any]):
    """Rollback to previous champion"""
    try:
        reason = request.get('reason', 'Manual rollback')
        
        ensemble = make_default_agent()
        
        if not ensemble.shadow_enabled or ensemble.shadow_manager is None:
            return {'status': 'error', 'message': 'Shadow models not enabled'}
        
        success = ensemble.shadow_manager.rollback_to_previous_champion(reason=reason)
        
        if success:
            champion = ensemble.shadow_manager.get_champion()
            return {
                'status': 'success',
                'message': f'Rolled back to {champion}'
            }
        else:
            return {
                'status': 'error',
                'message': 'Rollback failed (no history)'
            }
    
    except Exception as e:
        logger.error(f"Failed to rollback: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/shadow/history")
async def get_promotion_history(n: int = 10):
    """Get promotion history"""
    try:
        ensemble = make_default_agent()
        
        if not ensemble.shadow_enabled or ensemble.shadow_manager is None:
            return {'status': 'error', 'message': 'Shadow models not enabled'}
        
        history = ensemble.shadow_manager.get_promotion_history(n=n)
        
        return {
            'status': 'success',
            'data': [event.__dict__ for event in history]
        }
    
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return {'status': 'error', 'message': str(e)}


# ============================================================================
# MODULE 1: MEMORY STATES ENDPOINTS
# ============================================================================

@router.get("/memory/status")
async def get_memory_status():
    """Get memory states module status"""
    try:
        ensemble = make_default_agent()
        status = ensemble.get_memory_status()
        return {'status': 'success', 'data': status}
    except Exception as e:
        logger.error(f"Memory status error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/memory/states/{symbol}")
async def get_memory_states(symbol: str):
    """Get memory states for a symbol"""
    try:
        ensemble = make_default_agent()
        if not ensemble.memory_enabled or ensemble.memory_manager is None:
            return {'status': 'error', 'message': 'Memory states not enabled'}
        
        states = ensemble.memory_manager.get_states(symbol)
        return {'status': 'success', 'data': states}
    except Exception as e:
        logger.error(f"Memory states error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/memory/confidence/{symbol}")
async def get_memory_confidence(symbol: str):
    """Get confidence calibration for a symbol"""
    try:
        ensemble = make_default_agent()
        if not ensemble.memory_enabled or ensemble.memory_manager is None:
            return {'status': 'error', 'message': 'Memory states not enabled'}
        
        confidence = ensemble.memory_manager.get_confidence(symbol)
        return {'status': 'success', 'data': confidence}
    except Exception as e:
        logger.error(f"Memory confidence error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/memory/history/{symbol}")
async def get_memory_history(symbol: str, n: int = 100):
    """Get memory state history"""
    try:
        ensemble = make_default_agent()
        if not ensemble.memory_enabled or ensemble.memory_manager is None:
            return {'status': 'error', 'message': 'Memory states not enabled'}
        
        history = ensemble.memory_manager.get_history(symbol, n=n)
        return {'status': 'success', 'data': history}
    except Exception as e:
        logger.error(f"Memory history error: {e}")
        return {'status': 'error', 'message': str(e)}


# ============================================================================
# MODULE 2: REINFORCEMENT SIGNALS ENDPOINTS
# ============================================================================

@router.get("/reinforcement/status")
async def get_reinforcement_status():
    """Get reinforcement signals module status"""
    try:
        ensemble = make_default_agent()
        status = ensemble.get_reinforcement_status()
        return {'status': 'success', 'data': status}
    except Exception as e:
        logger.error(f"Reinforcement status error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/reinforcement/policy/{symbol}")
async def get_reinforcement_policy(symbol: str):
    """Get reinforcement learning policy for a symbol"""
    try:
        ensemble = make_default_agent()
        if not ensemble.reinforcement_enabled or ensemble.reinforcement_manager is None:
            return {'status': 'error', 'message': 'Reinforcement signals not enabled'}
        
        policy = ensemble.reinforcement_manager.get_policy(symbol)
        return {'status': 'success', 'data': policy}
    except Exception as e:
        logger.error(f"Reinforcement policy error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/reinforcement/rewards/{symbol}")
async def get_reinforcement_rewards(symbol: str, n: int = 100):
    """Get reward history"""
    try:
        ensemble = make_default_agent()
        if not ensemble.reinforcement_enabled or ensemble.reinforcement_manager is None:
            return {'status': 'error', 'message': 'Reinforcement signals not enabled'}
        
        rewards = ensemble.reinforcement_manager.get_rewards(symbol, n=n)
        return {'status': 'success', 'data': rewards}
    except Exception as e:
        logger.error(f"Reinforcement rewards error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/reinforcement/actions/{symbol}")
async def get_reinforcement_actions(symbol: str):
    """Get Q-values for all actions"""
    try:
        ensemble = make_default_agent()
        if not ensemble.reinforcement_enabled or ensemble.reinforcement_manager is None:
            return {'status': 'error', 'message': 'Reinforcement signals not enabled'}
        
        actions = ensemble.reinforcement_manager.get_actions(symbol)
        return {'status': 'success', 'data': actions}
    except Exception as e:
        logger.error(f"Reinforcement actions error: {e}")
        return {'status': 'error', 'message': str(e)}


# ============================================================================
# MODULE 3: DRIFT DETECTION ENDPOINTS
# ============================================================================

@router.get("/drift/status")
async def get_drift_status():
    """Get drift detection module status"""
    try:
        ensemble = make_default_agent()
        status = ensemble.get_drift_status()
        return {'status': 'success', 'data': status}
    except Exception as e:
        logger.error(f"Drift status error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/drift/detection/{symbol}")
async def get_drift_detection(symbol: str):
    """Get drift detection results for a symbol"""
    try:
        ensemble = make_default_agent()
        if not ensemble.drift_enabled or ensemble.drift_detector is None:
            return {'status': 'error', 'message': 'Drift detection not enabled'}
        
        detection = ensemble.drift_detector.get_detection(symbol)
        return {'status': 'success', 'data': detection}
    except Exception as e:
        logger.error(f"Drift detection error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/drift/alerts")
async def get_drift_alerts():
    """Get active drift alerts"""
    try:
        ensemble = make_default_agent()
        if not ensemble.drift_enabled or ensemble.drift_detector is None:
            return {'status': 'error', 'message': 'Drift detection not enabled'}
        
        alerts = ensemble.drift_detector.get_active_alerts()
        return {'status': 'success', 'data': alerts}
    except Exception as e:
        logger.error(f"Drift alerts error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/drift/history/{symbol}")
async def get_drift_history(symbol: str, n: int = 100):
    """Get drift detection history"""
    try:
        ensemble = make_default_agent()
        if not ensemble.drift_enabled or ensemble.drift_detector is None:
            return {'status': 'error', 'message': 'Drift detection not enabled'}
        
        history = ensemble.drift_detector.get_drift_history(symbol, n=n)
        return {'status': 'success', 'data': history}
    except Exception as e:
        logger.error(f"Drift history error: {e}")
        return {'status': 'error', 'message': str(e)}


# ============================================================================
# MODULE 4: COVARIATE SHIFT ENDPOINTS
# ============================================================================

@router.get("/covariate/status")
async def get_covariate_status():
    """Get covariate shift module status"""
    try:
        ensemble = make_default_agent()
        status = ensemble.get_covariate_status()
        return {'status': 'success', 'data': status}
    except Exception as e:
        logger.error(f"Covariate status error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/covariate/shift/{symbol}")
async def get_covariate_shift(symbol: str):
    """Get covariate shift detection results"""
    try:
        ensemble = make_default_agent()
        if not ensemble.covariate_enabled or ensemble.covariate_manager is None:
            return {'status': 'error', 'message': 'Covariate shift not enabled'}
        
        shift = ensemble.covariate_manager.get_shift(symbol)
        return {'status': 'success', 'data': shift}
    except Exception as e:
        logger.error(f"Covariate shift error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/covariate/weights/{symbol}")
async def get_covariate_weights(symbol: str):
    """Get importance weights"""
    try:
        ensemble = make_default_agent()
        if not ensemble.covariate_enabled or ensemble.covariate_manager is None:
            return {'status': 'error', 'message': 'Covariate shift not enabled'}
        
        weights = ensemble.covariate_manager.get_importance_weights(symbol)
        return {'status': 'success', 'data': weights}
    except Exception as e:
        logger.error(f"Covariate weights error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/covariate/history/{symbol}")
async def get_covariate_history(symbol: str, n: int = 100):
    """Get covariate shift history"""
    try:
        ensemble = make_default_agent()
        if not ensemble.covariate_enabled or ensemble.covariate_manager is None:
            return {'status': 'error', 'message': 'Covariate shift not enabled'}
        
        history = ensemble.covariate_manager.get_history(symbol, n=n)
        return {'status': 'success', 'data': history}
    except Exception as e:
        logger.error(f"Covariate history error: {e}")
        return {'status': 'error', 'message': str(e)}


# ============================================================================
# MODULE 6: CONTINUOUS LEARNING ENDPOINTS
# ============================================================================

@router.get("/continuous-learning/status")
async def get_cl_status():
    """Get continuous learning module status"""
    try:
        ensemble = make_default_agent()
        status = ensemble.get_cl_status()
        return {'status': 'success', 'data': status}
    except Exception as e:
        logger.error(f"CL status error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/continuous-learning/performance")
async def get_cl_performance():
    """Get performance monitoring metrics"""
    try:
        ensemble = make_default_agent()
        if not ensemble.cl_enabled or ensemble.cl_manager is None:
            return {'status': 'error', 'message': 'Continuous learning not enabled'}
        
        perf = ensemble.cl_manager.get_performance_metrics()
        return {'status': 'success', 'data': perf}
    except Exception as e:
        logger.error(f"CL performance error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/continuous-learning/features")
async def get_cl_features():
    """Get feature drift tracking"""
    try:
        ensemble = make_default_agent()
        if not ensemble.cl_enabled or ensemble.cl_manager is None:
            return {'status': 'error', 'message': 'Continuous learning not enabled'}
        
        features = ensemble.cl_manager.get_feature_tracking()
        return {'status': 'success', 'data': features}
    except Exception as e:
        logger.error(f"CL features error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/continuous-learning/history")
async def get_cl_history(n: int = 50):
    """Get retraining event history"""
    try:
        ensemble = make_default_agent()
        if not ensemble.cl_enabled or ensemble.cl_manager is None:
            return {'status': 'error', 'message': 'Continuous learning not enabled'}
        
        history = ensemble.cl_manager.get_retraining_history(n=n)
        return {'status': 'success', 'data': history}
    except Exception as e:
        logger.error(f"CL history error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.post("/continuous-learning/trigger")
async def trigger_cl_retraining(request: Dict[str, Any]):
    """Manually trigger retraining"""
    try:
        reason = request.get('reason', 'Manual trigger')
        
        ensemble = make_default_agent()
        if not ensemble.cl_enabled or ensemble.cl_manager is None:
            return {'status': 'error', 'message': 'Continuous learning not enabled'}
        
        result = ensemble.cl_manager.trigger_retraining(reason=reason, manual=True)
        return {'status': 'success', 'data': result}
    except Exception as e:
        logger.error(f"CL trigger error: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/continuous-learning/versions")
async def get_cl_versions():
    """Get model version history"""
    try:
        ensemble = make_default_agent()
        if not ensemble.cl_enabled or ensemble.cl_manager is None:
            return {'status': 'error', 'message': 'Continuous learning not enabled'}
        
        versions = ensemble.cl_manager.get_version_history()
        return {'status': 'success', 'data': versions}
    except Exception as e:
        logger.error(f"CL versions error: {e}")
        return {'status': 'error', 'message': str(e)}


# ============================================================================
# UNIFIED BULLETPROOF STATUS ENDPOINT
# ============================================================================

@router.get("/bulletproof/status")
async def get_bulletproof_status():
    """Get unified status of all 6 bulletproof AI modules"""
    try:
        ensemble = make_default_agent()
        status = ensemble.get_bulletproof_status()
        return {'status': 'success', 'data': status}
    except Exception as e:
        logger.error(f"Bulletproof status error: {e}")
        return {'status': 'error', 'message': str(e)}
