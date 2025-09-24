import asyncio
import os
import json
import tempfile

import pytest

from ai_engine.train_and_save import train_and_save, MODEL_DIR


class DummyExternalData:
    async def binance_ohlcv(self, symbol: str, limit: int = 600):
        # Return deterministic candles: list of dicts with open/high/low/close/volume
        candles = []
        price = 100.0
        for i in range(limit):
            candles.append({
                'timestamp': f't{i}',
                'open': price,
                'high': price + 1,
                'low': price - 1,
                'close': price + (i % 3 - 1) * 0.1,
                'volume': 100 + i,
            })
            price = candles[-1]['close']
        return {'candles': candles}

    async def twitter_sentiment(self, symbol: str):
        return {'score': 0.1, 'label': 'positive', 'source': 'mock'}

    async def cryptopanic_news(self, symbol: str, limit: int = 200):
        return {'news': [{'id': 'n1'}]}


@pytest.mark.asyncio
async def test_train_and_save_creates_artifacts(monkeypatch, tmp_path):
    # Ensure MODEL_DIR is a temp dir for test isolation
    orig_model_dir = os.environ.get('MODEL_DIR')
    try:
        # patch internal MODEL_DIR constant by setting env var used by module
        # ai_engine.train_and_save uses MODEL_DIR from its module variable; ensure
        # we write into a temp folder by creating it and monkeypatching
        test_model_dir = tmp_path / 'models'
        test_model_dir.mkdir()

        # Monkeypatch the backend.routes.external_data module used by train_and_save
        dummy = DummyExternalData()
        import backend.routes.external_data as external_data

        monkeypatch.setattr('backend.routes.external_data.binance_ohlcv', dummy.binance_ohlcv)
        monkeypatch.setattr('backend.routes.external_data.twitter_sentiment', dummy.twitter_sentiment)
        monkeypatch.setattr('backend.routes.external_data.cryptopanic_news', dummy.cryptopanic_news)

        # Call training with a small limit to keep test quick.
        # train_and_save creates its own event loop internally; when running
        # inside pytest-asyncio we must execute it in a separate thread to
        # avoid "event loop is already running" errors.
        await asyncio.to_thread(train_and_save, ['TEST1'], 50)

        # Check artifacts exist in ai_engine/models
        files = os.listdir(MODEL_DIR)
        assert 'xgb_model.pkl' in files or 'xgb_model.json' in files
        assert 'scaler.pkl' in files
        # metadata should be present
        assert 'metadata.json' in files

    finally:
        if orig_model_dir is not None:
            os.environ['MODEL_DIR'] = orig_model_dir
