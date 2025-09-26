import asyncio
import os

import pytest

from tests._helpers import train_and_save_stub, external_data_stub


class DummyExternalData:
    async def binance_ohlcv(self, symbol: str, limit: int = 600):
        # Return deterministic candles: list of dicts with open/high/low/close/volume
        candles = []
        price = 100.0
        for i in range(limit):
            close_val = float(price + (i % 3 - 1) * 0.1)
            candles.append(
                {
                    "timestamp": f"t{i}",
                    "open": float(price),
                    "high": float(price + 1),
                    "low": float(price - 1),
                    "close": close_val,
                    "volume": 100 + i,
                }
            )
        from typing import Any, cast as _cast

        price = float(_cast(Any, candles[-1]["close"]))
        return {"candles": candles}

    async def twitter_sentiment(self, symbol: str):
        return {"score": 0.1, "label": "positive", "source": "mock"}

    async def cryptopanic_news(self, symbol: str, limit: int = 200):
        return {"news": [{"id": "n1"}]}


@pytest.mark.asyncio
async def test_train_and_save_creates_artifacts(monkeypatch, tmp_path):
    # Ensure MODEL_DIR inside the test helper is a temp dir for test isolation
    test_model_dir = tmp_path / "models"
    test_model_dir.mkdir()
    monkeypatch.setattr(train_and_save_stub, "MODEL_DIR", str(test_model_dir))
    try:
        # Monkeypatch the internal external_data helpers used by train_and_save
        monkeypatch.setattr(
            "backend.routes.external_data.binance_ohlcv",
            external_data_stub.binance_ohlcv,
        )
        monkeypatch.setattr(
            "backend.routes.external_data.twitter_sentiment",
            external_data_stub.twitter_sentiment,
        )
        monkeypatch.setattr(
            "backend.routes.external_data.cryptopanic_news",
            external_data_stub.cryptopanic_news,
        )

        # Call the test helper training function directly to create artifacts
        await asyncio.to_thread(train_and_save_stub.train_and_save, ["TEST1"], 50)

        # Check artifacts exist in ai_engine/models
        files = os.listdir(str(test_model_dir))
        assert "xgb_model.pkl" in files or "xgb_model.json" in files
        assert "scaler.pkl" in files
        # metadata should be present
        assert "metadata.json" in files

    finally:
        # nothing to restore; test helper MODEL_DIR was set via monkeypatch
        pass
