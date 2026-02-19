import pytest

from backend.utils import telemetry


@pytest.fixture(autouse=True)
def reset_admin_metrics():
    telemetry.reset_admin_event_metrics()
    yield
    telemetry.reset_admin_event_metrics()


def test_record_cache_helpers_increment_counters():
    hit_metric = telemetry.CACHE_LOOKUPS.labels(cache_name="orders", result="hit")
    miss_metric = telemetry.CACHE_LOOKUPS.labels(cache_name="orders", result="miss")
    write_metric = telemetry.CACHE_LOOKUPS.labels(cache_name="orders", result="write")
    before_hit = hit_metric._value.get()
    before_miss = miss_metric._value.get()
    before_write = write_metric._value.get()

    telemetry.record_cache_hit("orders")
    telemetry.record_cache_miss("orders")
    telemetry.record_cache_write("orders")

    assert write_metric._value.get() == before_write + 1
    assert hit_metric._value.get() == before_hit + 1
    assert miss_metric._value.get() == before_miss + 1


def test_record_provider_counters_and_gauge():
    failure_metric = telemetry.PROVIDER_FAILURES.labels(provider="binance")
    success_metric = telemetry.PROVIDER_SUCCESSES.labels(provider="binance")
    circuit_gauge = telemetry.PROVIDER_CIRCUIT_OPEN.labels(provider="binance")

    start_failures = failure_metric._value.get()
    start_success = success_metric._value.get()

    telemetry.record_provider_failure("binance", circuit_open=True)
    assert failure_metric._value.get() == start_failures + 1
    assert circuit_gauge._value.get() == 1.0

    telemetry.record_provider_success("binance")
    assert success_metric._value.get() == start_success + 1
    assert circuit_gauge._value.get() == 0.0


def test_record_admin_event_metric_tracks_outcome():
    metric = telemetry.ADMIN_EVENTS_TOTAL.labels(event="scheduler.status", category="scheduler", severity="info", success="true")
    baseline = metric._value.get()

    telemetry.record_admin_event_metric(event="scheduler.status", category="scheduler", severity="info", success=True)

    assert metric._value.get() == baseline + 1


def test_track_model_inference_success_and_error():
    success_hist = telemetry.MODEL_INFERENCE_DURATION.labels(model_name="signals", outcome="success")
    error_hist = telemetry.MODEL_INFERENCE_DURATION.labels(model_name="signals", outcome="error")
    success_count_before = sum(bucket.get() for bucket in success_hist._buckets)
    success_sum_before = success_hist._sum.get()
    error_count_before = sum(bucket.get() for bucket in error_hist._buckets)

    with telemetry.track_model_inference("signals"):
        pass

    assert sum(bucket.get() for bucket in success_hist._buckets) == success_count_before + 1
    assert success_hist._sum.get() > success_sum_before

    with pytest.raises(RuntimeError):
        with telemetry.track_model_inference("signals"):
            raise RuntimeError("boom")

    assert sum(bucket.get() for bucket in error_hist._buckets) == error_count_before + 1


def test_record_scheduler_run_updates_metrics():
    duration_hist = telemetry.SCHEDULER_RUN_DURATION.labels(job_id="liquidity", status="ok")
    total_counter = telemetry.SCHEDULER_RUN_TOTAL.labels(job_id="liquidity", status="ok")
    duration_count_before = sum(bucket.get() for bucket in duration_hist._buckets)
    total_before = total_counter._value.get()

    telemetry.record_scheduler_run("liquidity", "ok", 1.5)

    assert sum(bucket.get() for bucket in duration_hist._buckets) == duration_count_before + 1
    assert total_counter._value.get() == total_before + 1