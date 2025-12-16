import math

from backend.config.execution import ExecutionConfig
from backend.models.liquidity import PortfolioAllocation
from backend.services.execution.execution import compute_target_orders


def _alloc(symbol: str, weight: float, *, run_id: int = 1, score: float = 1.0):
    return PortfolioAllocation(run_id=run_id, symbol=symbol, weight=weight, score=score)


def test_compute_target_orders_creates_intents_sorted():
    allocations = [
        _alloc("BTCUSDT", 0.6),
        _alloc("ETHUSDT", 0.4),
    ]
    prices = {"BTCUSDT": 20000.0, "ETHUSDT": 1000.0}
    positions = {"BTCUSDT": 0.1, "ETHUSDT": 0.5}
    config = ExecutionConfig(min_notional=100.0, max_orders=10, cash_buffer=100.0, allow_partial=True)

    orders = compute_target_orders(
        allocations,
        prices,
        positions=positions,
        total_equity=25000.0,
        config=config,
    )

    assert len(orders) == 2
    # Orders should be sorted by notional descending (ETH lower than BTC)
    assert orders[0].symbol == "BTCUSDT"
    assert orders[0].side == "BUY"
    assert orders[1].symbol == "ETHUSDT"
    assert orders[1].side == "BUY"
    assert orders[0].notional >= orders[1].notional
    # Ensure quantities align with target delta within rounding tolerance
    btc_target_notional = 0.6 * (25000.0 - 100.0)
    btc_current = positions["BTCUSDT"] * prices["BTCUSDT"]
    expected_btc_qty = (btc_target_notional - btc_current) / prices["BTCUSDT"]
    assert math.isclose(orders[0].quantity, abs(expected_btc_qty), rel_tol=1e-9)


def test_compute_target_orders_respects_min_notional():
    allocations = [_alloc("BTCUSDT", 0.1)]
    prices = {"BTCUSDT": 20000.0}
    positions = {"BTCUSDT": 0.0}
    config = ExecutionConfig(min_notional=5000.0, max_orders=10, cash_buffer=0.0, allow_partial=True)

    orders = compute_target_orders(
        allocations,
        prices,
        positions=positions,
        total_equity=10000.0,
        config=config,
    )

    assert orders == []


def test_compute_target_orders_applies_max_orders():
    allocations = [
        _alloc("BTCUSDT", 0.4),
        _alloc("ETHUSDT", 0.3),
        _alloc("SOLUSDT", 0.2),
        _alloc("XRPUSDT", 0.1),
    ]
    prices = {
        "BTCUSDT": 20000.0,
        "ETHUSDT": 1500.0,
        "SOLUSDT": 30.0,
        "XRPUSDT": 0.5,
    }
    positions = {}
    config = ExecutionConfig(min_notional=10.0, max_orders=2, cash_buffer=0.0, allow_partial=True)

    orders = compute_target_orders(
        allocations,
        prices,
        positions=positions,
        total_equity=50000.0,
        config=config,
    )

    assert len(orders) == 2
    assert {order.symbol for order in orders} <= {"BTCUSDT", "ETHUSDT"}
    assert orders[0].notional >= orders[1].notional


def test_compute_target_orders_forces_full_exit_when_partial_disabled():
    allocations = [_alloc("BTCUSDT", 0.1)]
    prices = {"BTCUSDT": 20000.0}
    positions = {"BTCUSDT": 1.0}
    config = ExecutionConfig(min_notional=10.0, max_orders=5, cash_buffer=0.0, allow_partial=False)

    orders = compute_target_orders(
        allocations,
        prices,
        positions=positions,
        total_equity=20000.0,
        config=config,
    )

    assert len(orders) == 1
    order = orders[0]
    assert order.symbol == "BTCUSDT"
    assert order.side == "SELL"
    assert math.isclose(order.quantity, 1.0, rel_tol=1e-9)
