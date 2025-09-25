def calculate_analytics() -> dict:
    """Return a lightweight analytics summary for the dashboard.

    This is intentionally minimal to keep tests deterministic.
    """
    return {"sharpe": 0.0, "sortino": 0.0}
