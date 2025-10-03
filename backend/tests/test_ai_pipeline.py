import json
from pathlib import Path

import pytest

from ai_engine import train_and_save


@pytest.fixture(autouse=True)
def stub_symbol_fetch(monkeypatch):
    async def _fake_fetch(symbol: str, limit: int = 600):
        candles = []
        price = 100.0
        for i in range(limit):
            price += 0.2
            candles.append(
                {
                    "timestamp": f"t{i}",
                    "open": price - 0.3,
                    "high": price + 0.4,
                    "low": price - 0.4,
                    "close": price,
                    "volume": 100 + i,
                }
            )
        sentiment = [0.1] * limit
        news = [1.0 if i % 10 == 0 else 0.0 for i in range(limit)]
        return candles, sentiment, news

    monkeypatch.setattr(train_and_save, "_fetch_symbol_data", _fake_fetch)


def test_training_pipeline_creates_artifacts(tmp_path: Path):
    summary = train_and_save.train_and_save(
        symbols=["TEST"], limit=50, model_dir=tmp_path, backtest=True
    )
    model_path = tmp_path / "xgb_model.pkl"
    fallback_path = tmp_path / "xgb_model.json"
    scaler_path = tmp_path / "scaler.pkl"
    assert model_path.exists() or fallback_path.exists()
    assert scaler_path.exists()

    report_path = tmp_path / train_and_save.REPORT_FILENAME
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["metrics"]["rmse"] >= 0
    if "backtest" in report:
        assert report["backtest"]["trades"] >= 0

    assert summary["metrics"]["mae"] >= 0
    assert summary["backtest"] is not None


def test_run_backtest_only_uses_saved_artifacts(tmp_path: Path):
    train_and_save.train_and_save(
        symbols=["TEST"], limit=30, model_dir=tmp_path, backtest=True
    )
    result = train_and_save.run_backtest_only(
        symbols=["TEST"], limit=20, model_dir=tmp_path, entry_threshold=0.001
    )
    assert result["metrics"]["mae"] >= 0
    assert result["backtest"]["trades"] >= 0

    report = train_and_save.load_report(model_dir=tmp_path)
    assert report is not None
