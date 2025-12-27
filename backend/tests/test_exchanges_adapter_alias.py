import pytest


def test_get_adapter_alias_exists_and_deprecated():
    """Ensure the deprecated alias `get_adapter` emits a DeprecationWarning and
    returns a client-like object.
    """
    from backend.utils import exchanges

    # The alias should warn that it's deprecated
    with pytest.warns(DeprecationWarning):
        adapter_from_alias = exchanges.get_adapter("binance")

    adapter_from_factory = exchanges.get_exchange_client("binance")

    assert adapter_from_alias is not None
    assert adapter_from_factory is not None

    # They may be distinct instances, but both should provide the expected methods
    for obj in (adapter_from_alias, adapter_from_factory):
        assert hasattr(obj, "fetch_recent_trades")
        assert hasattr(obj, "spot_balance")
        assert hasattr(obj, "create_order")
