def test_adapter_factory_smoke():
    """A tiny smoke test for the exchange adapter factory.

    This test ensures the factory function exists and can be called with a
    supported exchange name. The implementation should mock network calls in
    real tests; this skeleton is a fast-running placeholder for sprint-1.
    """
    from backend.utils import exchanges

    adapter = exchanges.get_exchange_client("binance")
    assert adapter is not None
