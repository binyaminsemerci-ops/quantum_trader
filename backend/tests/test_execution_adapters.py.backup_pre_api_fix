import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Dict

from backend.config.execution import ExecutionConfig, load_execution_config
from backend.config.risk import load_risk_config
from backend.database import Base
from backend.models.liquidity import LiquidityRun, LiquiditySnapshot, PortfolioAllocation
from backend.services import execution
from backend.services.execution.execution import run_portfolio_rebalance
from backend.services.risk.risk_guard import RiskGuardService


def test_build_execution_adapter_defaults_to_paper():
    adapter = execution.build_execution_adapter(ExecutionConfig())
    assert isinstance(adapter, execution.PaperExchangeAdapter)


def test_build_execution_adapter_prefers_binance_when_ready(monkeypatch):
    class DummyAdapter:
        ready = True

        async def get_positions(self):  # pragma: no cover - interface stub
            return {}

        async def get_cash_balance(self):  # pragma: no cover - interface stub
            return 0.0

        async def submit_order(self, symbol, side, quantity, price):  # pragma: no cover
            return "dummy"

    captured_kwargs = {}

    def factory(**kwargs):
        captured_kwargs.update(kwargs)
        return DummyAdapter()

    monkeypatch.setattr(execution, "BinanceExecutionAdapter", factory)

    cfg = ExecutionConfig(exchange="binance")
    adapter = execution.build_execution_adapter(cfg)

    assert isinstance(adapter, DummyAdapter)
    assert captured_kwargs.get("quote_asset") == cfg.quote_asset
    assert captured_kwargs.get("testnet") is False


def test_build_execution_adapter_falls_back_when_not_ready(monkeypatch):
    class DummyAdapter:
        ready = False

        async def get_positions(self):  # pragma: no cover - interface stub
            return {}

        async def get_cash_balance(self):  # pragma: no cover - interface stub
            return 0.0

        async def submit_order(self, symbol, side, quantity, price):  # pragma: no cover
            return "dummy"

    def factory(**kwargs):
        return DummyAdapter()

    monkeypatch.setattr(execution, "BinanceExecutionAdapter", factory)

    cfg = ExecutionConfig(exchange="binance")
    adapter = execution.build_execution_adapter(cfg)

    assert isinstance(adapter, execution.PaperExchangeAdapter)
 

def test_load_execution_config_reads_testnet_flag(monkeypatch):
    monkeypatch.setenv("QT_EXECUTION_BINANCE_TESTNET", "1")
    cfg = load_execution_config()
    assert cfg.binance_testnet is True


@pytest.mark.asyncio
async def test_run_portfolio_rebalance_respects_exposure_limits(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    run = LiquidityRun(universe_size=1, selection_size=1, provider_primary="test")
    session.add(run)
    session.flush()

    snapshot = LiquiditySnapshot(
        run_id=run.id,
        rank=1,
        symbol="BTCUSDT",
        price=100.0,
        liquidity_score=1.0,
        aggregate_score=1.0,
    )
    allocation = PortfolioAllocation(
        run_id=run.id,
        symbol="BTCUSDT",
        weight=1.0,
        score=1.0,
    )
    session.add_all([snapshot, allocation])
    session.commit()

    class StubAdapter:
        def __init__(self) -> None:
            self.submissions = 0

        async def get_positions(self):
            return {}

        async def get_cash_balance(self):
            return 1000.0

        async def submit_order(self, *args, **kwargs):  # pragma: no cover - should not run
            self.submissions += 1
            return "stub-order"

    adapter = StubAdapter()

    monkeypatch.setenv("QT_MAX_POSITION_PER_SYMBOL", "50")
    monkeypatch.setenv("QT_MAX_GROSS_EXPOSURE", "50")

    risk_guard = RiskGuardService(load_risk_config())

    result = await run_portfolio_rebalance(
        session,
        adapter=adapter,
        execution_config=ExecutionConfig(
            min_notional=10.0,
            max_orders=5,
            cash_buffer=0.0,
            allow_partial=True,
            exchange="paper",
            quote_asset="USDT",
        ),
        risk_guard=risk_guard,
    )

    assert result["orders_submitted"] == 0
    assert result["orders_skipped"] >= 1
    assert adapter.submissions == 0

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_run_portfolio_rebalance_uses_binance_adapter(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    run = LiquidityRun(universe_size=1, selection_size=1, provider_primary="test")
    session.add(run)
    session.flush()

    snapshot = LiquiditySnapshot(
        run_id=run.id,
        rank=1,
        symbol="BTCUSDT",
        price=100.0,
        liquidity_score=1.0,
        aggregate_score=1.0,
    )
    allocation = PortfolioAllocation(
        run_id=run.id,
        symbol="BTCUSDT",
        weight=1.0,
        score=1.0,
    )
    session.add_all([snapshot, allocation])
    session.commit()

    class DummyAdapter:
        def __init__(self, **kwargs) -> None:
            self.ready = True
            self.kwargs = kwargs
            self.submissions: list = []

        async def get_positions(self):
            return {}

        async def get_cash_balance(self):
            return 1000.0

        async def submit_order(self, symbol, side, quantity, price):
            self.submissions.append((symbol, side, quantity, price))
            return "dummy-binance-order"

    captured: Dict[str, DummyAdapter] = {}

    def factory(**kwargs):
        adapter = DummyAdapter(**kwargs)
        captured["adapter"] = adapter
        return adapter

    monkeypatch.setattr(execution, "BinanceExecutionAdapter", factory)

    risk_guard = RiskGuardService(load_risk_config())

    result = await run_portfolio_rebalance(
        session,
        execution_config=ExecutionConfig(
            min_notional=10.0,
            max_orders=5,
            cash_buffer=0.0,
            allow_partial=True,
            exchange="binance",
            quote_asset="USDT",
            binance_testnet=True,
        ),
        risk_guard=risk_guard,
    )

    adapter = captured["adapter"]
    assert adapter.kwargs.get("testnet") is True
    assert adapter.kwargs.get("quote_asset") == "USDT"
    assert adapter.submissions, "Expected at least one order submission"
    assert result["orders_submitted"] == len(adapter.submissions) == 1

    session.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_run_portfolio_rebalance_normalizes_symbols_for_adapter(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    run = LiquidityRun(universe_size=1, selection_size=1, provider_primary="test")
    session.add(run)
    session.flush()

    snapshot = LiquiditySnapshot(
        run_id=run.id,
        rank=1,
        symbol="BTCUSDC",
        price=100.0,
        liquidity_score=1.0,
        aggregate_score=1.0,
    )
    allocation = PortfolioAllocation(
        run_id=run.id,
        symbol="BTCUSDC",
        weight=1.0,
        score=1.0,
    )
    session.add_all([snapshot, allocation])
    session.commit()

    class NormalizingAdapter:
        def __init__(self) -> None:
            self.submissions: list[str] = []

        async def get_positions(self):
            return {}

        async def get_cash_balance(self):
            return 1000.0

        def normalize_symbol(self, symbol: str) -> str:
            return symbol.replace("USDC", "USDT")

        async def submit_order(self, symbol, side, quantity, price):
            self.submissions.append(symbol)
            return "normalized-order"

    adapter = NormalizingAdapter()
    risk_guard = RiskGuardService(load_risk_config())

    result = await run_portfolio_rebalance(
        session,
        adapter=adapter,
        execution_config=ExecutionConfig(
            min_notional=10.0,
            max_orders=5,
            cash_buffer=0.0,
            allow_partial=True,
            exchange="paper",
            quote_asset="USDC",
        ),
        risk_guard=risk_guard,
    )

    assert adapter.submissions == ["BTCUSDT"]
    assert result["orders_submitted"] == 1

    session.close()
    engine.dispose()
