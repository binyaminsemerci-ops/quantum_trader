from types import SimpleNamespace

import pytest

from backend.utils.cache import load_json, save_json
from backend.utils.telemetry import CACHE_LOOKUPS


@pytest.fixture(autouse=True)
def reset_cache_metrics():
    # Prometheus counters keep cumulative state; record baseline so assertions use deltas.
    baseline = {}

    def get_value(cache_name: str, result: str) -> float:
        return CACHE_LOOKUPS.labels(cache_name=cache_name, result=result)._value.get() or 0.0

    for name in ("sample", "twitter_recent_search", "cryptopanic_posts"):
        for result in ("hit", "miss", "write"):
            key = (name, result)
            baseline[key] = get_value(*key)

    yield baseline

    # Ensure module-level caches do not leak across tests
    from backend.utils import twitter_client, cryptopanic_client

    twitter_client._CACHE.clear()
    cryptopanic_client._CACHE.clear()


def metric_delta(baseline, cache_name: str, result: str) -> float:
    return (
        CACHE_LOOKUPS.labels(cache_name=cache_name, result=result)._value.get() or 0.0
    ) - baseline[(cache_name, result)]


def test_disk_cache_records_hits_and_misses(tmp_path, reset_cache_metrics):
    path = tmp_path / "sample.json"
    baseline = reset_cache_metrics

    # Cache miss on absent file
    assert load_json(path) is None
    assert metric_delta(baseline, "sample", "miss") == pytest.approx(1.0)

    # Write payload and confirm write metric increments
    payload = {"value": 42}
    save_json(path, payload)
    assert metric_delta(baseline, "sample", "write") == pytest.approx(1.0)

    # Subsequent load returns value and records a hit
    loaded = load_json(path)
    assert loaded == payload
    assert metric_delta(baseline, "sample", "hit") == pytest.approx(1.0)


def test_inmemory_caches_emit_metrics(monkeypatch, reset_cache_metrics):
    baseline = reset_cache_metrics

    monkeypatch.setattr(
        "backend.utils.twitter_client.load_config",
        lambda: SimpleNamespace(x_bearer_token=None),
        raising=False,
    )
    monkeypatch.setattr(
        "backend.utils.cryptopanic_client.load_config",
        lambda: SimpleNamespace(cryptopanic_key=None),
        raising=False,
    )

    from backend.utils.twitter_client import TwitterClient, _CACHE as TW_CACHE
    from backend.utils.cryptopanic_client import CryptoPanicClient, _CACHE as CP_CACHE

    TW_CACHE.clear()
    CP_CACHE.clear()

    twitter_client = TwitterClient()
    cache_key = "tw:BTC:10"

    assert twitter_client._cached(cache_key) is None
    assert metric_delta(baseline, "twitter_recent_search", "miss") == pytest.approx(1.0)

    twitter_client._set_cache(cache_key, {"score": 0.5})
    assert metric_delta(baseline, "twitter_recent_search", "write") == pytest.approx(1.0)
    assert twitter_client._cached(cache_key) == {"score": 0.5}
    assert metric_delta(baseline, "twitter_recent_search", "hit") == pytest.approx(1.0)

    crypto_client = CryptoPanicClient()
    cache_key = "cp:btc:5"

    assert crypto_client._cached(cache_key) is None
    assert metric_delta(baseline, "cryptopanic_posts", "miss") == pytest.approx(1.0)

    crypto_client._set_cache(cache_key, ["item"])
    assert metric_delta(baseline, "cryptopanic_posts", "write") == pytest.approx(1.0)
    assert crypto_client._cached(cache_key) == ["item"]
    assert metric_delta(baseline, "cryptopanic_posts", "hit") == pytest.approx(1.0)
