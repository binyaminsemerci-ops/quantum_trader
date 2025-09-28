from fastapi import APIRouter, Query
from typing import List, Dict, Literal, Annotated, Optional
import datetime
import logging

# Note: mock_signals is test/demo-only. Import it lazily inside the
# generator function so production imports of this module don't pull
# testing/demo code into the production import graph.
from pydantic import BaseModel, Field

router = APIRouter()

# module logger for Bandit-friendly error handling
logger = logging.getLogger(__name__)


class SignalDetails(BaseModel):
    source: str
    note: Optional[str]


class Signal(BaseModel):
    id: str
    timestamp: datetime.datetime
    symbol: str
    side: Literal["buy", "sell"]
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    details: SignalDetails


class PaginatedSignals(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[Signal]


def _generate_mock_signals(count: int, profile: Literal["left", "right", "mixed"]) -> List[Dict]:
    # Delegate to testing/mock_signals.py which contains the deterministic
    # mock generator. Import lazily so production installs that don't include
    # test/demo helpers won't fail at module import time.
    # Prefer the implementation under tests/ (so test helpers aren't part of
    # the production import graph). Fall back to the legacy location for
    # compatibility, and if neither is available return an empty list.
    # Prefer test helpers under backend.tests, but be robust: import the
    # module and check for the attribute rather than assuming it exists.
    try:
        import importlib

        mod = importlib.import_module("backend.tests.utils.mock_signals")
        if hasattr(mod, "generate_mock_signals"):
            return getattr(mod, "generate_mock_signals")(count, profile)
    except (ImportError, AttributeError) as e:
        # Tests/demo-only module may be absent in production installs; log
        # at debug level so Bandit doesn't flag a silent pass.
        logger.debug("tests.mock_signals not available: %s", e)

    try:
        import importlib

        legacy = importlib.import_module("backend.testing.mock_signals")
        if hasattr(legacy, "generate_mock_signals"):
            return getattr(legacy, "generate_mock_signals")(count, profile)
    except (ImportError, AttributeError) as e:
        logger.debug("legacy mock_signals not available: %s", e)

    return []


@router.get("/recent", response_model=List[Dict])
def recent_signals(
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    profile: Annotated[Literal["left", "right", "mixed"], Query()] = "mixed",
) -> List[Dict]:
    """Legacy endpoint kept for tests and frontend stubs. Returns a list of
    mock signals as dicts (same shape as new typed endpoints).
    """
    return _generate_mock_signals(limit, profile)


@router.get("/", response_model=PaginatedSignals)
def list_signals(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=200)] = 20,
    profile: Annotated[Literal["left", "right", "mixed"], Query()] = "mixed",
    symbol: Optional[str] = None,
):
    """Return paginated mock signals. This is a deterministic generator for
    frontend/demo use and tests; in production this would query a database.
    """
    total_available = 500
    all_signals = _generate_mock_signals(total_available, profile)

    if symbol:
        all_signals = [s for s in all_signals if s["symbol"] == symbol]

    total = len(all_signals)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = all_signals[start:end]

    # Convert timestamps to datetime for Pydantic model parsing
    for it in page_items:
        if isinstance(it["timestamp"], datetime.datetime):
            continue
        it["timestamp"] = datetime.datetime.fromisoformat(it["timestamp"])

    # Convert dicts to Signal instances so the PaginatedSignals items list
    # has the expected type: List[Signal]. This satisfies mypy and ensures
    # response_model validation uses the Signal model.
    signal_items = [Signal(**it) for it in page_items]

    return PaginatedSignals(
        total=total, page=page, page_size=page_size, items=signal_items
    )
