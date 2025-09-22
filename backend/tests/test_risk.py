from backend.utils.risk import RiskManager


def test_valid_order():
    rm = RiskManager(max_position=0.1, max_loss_pct=0.02)
    valid, reason = rm.validate_order(
        balance=10000, qty=0.05, price=20000, stop_loss=19000
    )
    assert valid is True
    assert reason == "Order is valid"


def test_too_large_position():
    rm = RiskManager(max_position=0.1)
    valid, reason = rm.validate_order(balance=1000, qty=1, price=20000, stop_loss=19000)
    assert valid is False
    assert "exceeds" in reason


def test_bad_stop_loss():
    rm = RiskManager(max_loss_pct=0.02)
    valid, reason = rm.validate_order(
        balance=10000, qty=0.05, price=20000, stop_loss=21000
    )
    assert valid is False
    assert "Invalid stop-loss" in reason
