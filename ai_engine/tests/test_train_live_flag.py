

def test_prepare_live_dataset_monkeypatch(monkeypatch, tmp_path):
    # monkeypatch the live feature helper to return a small deterministic dataset
    def fake_fetch(sym, limit=200, lags=5, horizon=1):
        import numpy as np

        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([0.01])
        names = ["close_lag_1", "vol_lag_1", "rtn_lag_1"]
        return X, y, names

    monkeypatch.setattr("ai_engine.data.live_features.fetch_features_for_sklearn", fake_fetch)

    from ai_engine.train_and_save import _prepare_live_dataset

    ds = _prepare_live_dataset(["BTCUSDT"], limit=50)
    assert ds.features.shape[0] == 1
    assert ds.target.shape[0] == 1
    assert ds.feature_names[0].startswith("close_lag_")
