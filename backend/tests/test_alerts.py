from backend.alerts.evaluator import evaluate_alert_for_symbol

# Basic smoke test for evaluator (relies on demo candles)


def test_evaluator_smoke():
    # create a fake Alert-like object
    class FakeAlert:
        def __init__(self):
            self.id = None
            self.symbol = "BTCUSDT"
            self.condition = "price_above"
            self.threshold = -1.0
            self.enabled = 1

    a = FakeAlert()
    # Should not raise
    evaluate_alert_for_symbol(a)
