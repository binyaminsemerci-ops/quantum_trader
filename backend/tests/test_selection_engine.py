import math
from dataclasses import replace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config.liquidity import LiquidityConfig
from backend.models.liquidity import PortfolioAllocation
from backend.services.liquidity import LiquidityRecord, persist_liquidity_run
from backend.services.selection_engine import blend_liquidity_and_model
from backend.database import Base


@pytest.fixture()
def in_memory_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionMaker = sessionmaker(bind=engine)
    session = SessionMaker()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _record(symbol: str, aggregate: float, quote_volume: float = 1_000_000.0) -> LiquidityRecord:
    return LiquidityRecord(
        symbol=symbol,
        price=100.0,
        change_percent=2.5,
        base_volume=500_000.0,
        quote_volume=quote_volume,
        market_cap=1_000_000.0,
        liquidity_score=aggregate / 2,
        momentum_score=0.12,
        aggregate_score=aggregate,
        providers={},
    )


def test_blend_liquidity_and_model_applies_agent_signals():
    config = LiquidityConfig(selection_min=1, selection_max=2, max_per_base=1)
    selection = [
        _record("BTCUSDT", aggregate=12.0),
        _record("ETHUSDT", aggregate=9.0),
        _record("BTCUSDC", aggregate=7.5),
    ]
    signals = {
        "BTCUSDT": {"action": "BUY", "score": 0.8},
        "ETHUSDT": {"action": "SELL", "score": 0.7},
        "BTCUSDC": {"action": "HOLD", "score": 0.3},
    }

    blended = blend_liquidity_and_model(selection, signals, config)

    assert len(blended) == 2
    bases = {item.symbol[:-4] for item in blended}
    assert len(bases) == 2  # diversification respected

    leader = blended[0]
    assert leader.symbol == "BTCUSDT"
    assert leader.model_action == "BUY"
    assert math.isclose(leader.model_score, 0.8, rel_tol=1e-6)
    assert leader.allocation_score > blended[1].allocation_score


def test_persist_liquidity_run_uses_allocation_scores(in_memory_session: Session):
    db = in_memory_session
    records = [_record("BTCUSDT", 10.0), _record("ETHUSDT", 8.0)]
    selection = [
        _record("BTCUSDT", 10.0),
        _record("ETHUSDT", 8.0),
    ]
    selection[0] = replace(
        selection[0], allocation_score=0.75, model_action="BUY", model_score=0.9
    )
    selection[1] = replace(
        selection[1], allocation_score=0.25, model_action="HOLD", model_score=0.2
    )
    signals = {"BTCUSDT": {"action": "BUY", "score": 0.9}}

    run = persist_liquidity_run(
        db,
        records=records,
        selection=selection,
        provider_primary="binance",
        signals=signals,
    )

    allocations = db.query(PortfolioAllocation).order_by(PortfolioAllocation.symbol).all()
    assert len(allocations) == 2
    assert math.isclose(sum(a.weight for a in allocations), 1.0, rel_tol=1e-6)
    btc = next(a for a in allocations if a.symbol == "BTCUSDT")
    eth = next(a for a in allocations if a.symbol == "ETHUSDT")
    assert btc.weight > eth.weight
    assert "model=BUY" in btc.reason
    assert "blend=0.750" in btc.reason
    assert run.selection_size == 2

