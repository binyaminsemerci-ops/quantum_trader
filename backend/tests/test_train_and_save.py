import asyncio
import os

import pytest

import ai_engine.train_and_save as tas  # type: ignore[import-not-found, import-untyped]


class DummyExternalData:
    async def binance_ohlcv(self, symbol: str, limit: int = 600):
        # Return deterministic candles: list of dicts with open/high/low/close/volume
        candles = []
        price = 100.0
        for i in range(limit):
            close_val = float(price + (i % 3 - 1) * 0.1)
            candles.append({
                'timestamp': f't{i}',
                'open': float(price),
                'high': float(price + 1),
                'low': float(price - 1),
                'close': close_val,
                'volume': 100 + i,
            })
            from typing import Any, cast as _cast
            price = float(_cast(Any, candles[-1]['close']))
        return {'candles': candles}

    async def twitter_sentiment(self, symbol: str):
        return {'score': 0.1, 'label': 'positive', 'source': 'mock'}

    async def cryptopanic_news(self, symbol: str, limit: int = 200):
        return {'news': [{'id': 'n1'}]}


@pytest.mark.asyncio
async def test_train_and_save_creates_artifacts(monkeypatch, tmp_path):
    # Ensure MODEL_DIR inside ai_engine.train_and_save is a temp dir for test isolation
    import ai_engine.train_and_save as tas
    orig_model_dir = tas.MODEL_DIR
    try:
        test_model_dir = tmp_path / 'models'
        test_model_dir.mkdir()
        # monkeypatch the module-level MODEL_DIR used by the training code
        monkeypatch.setattr(tas, 'MODEL_DIR', str(test_model_dir))

        # Monkeypatch the backend.routes.external_data module used by train_and_save
        dummy = DummyExternalData()

        monkeypatch.setattr('backend.routes.external_data.binance_ohlcv', dummy.binance_ohlcv)
        monkeypatch.setattr('backend.routes.external_data.twitter_sentiment', dummy.twitter_sentiment)
        monkeypatch.setattr('backend.routes.external_data.cryptopanic_news', dummy.cryptopanic_news)

        # Call training with a small limit to keep test quick.
        # train_and_save creates its own event loop internally; when running
        # inside pytest-asyncio we must execute it in a separate thread to
        # avoid "event loop is already running" errors.
        await asyncio.to_thread(tas.train_and_save, ['TEST1'], 50)

        # Check artifacts exist in ai_engine/models
        files = os.listdir(tas.MODEL_DIR)
        assert 'xgb_model.pkl' in files or 'xgb_model.json' in files
        assert 'scaler.pkl' in files
        # metadata should be present
        assert 'metadata.json' in files

    finally:
        # restore original module constant
        try:
            monkeypatch.setattr(tas, 'MODEL_DIR', orig_model_dir)
        except Exception:
            pass
