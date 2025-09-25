import os
import json
import pickle
from typing import Iterable

# Minimal test helper that mirrors the previous test shim
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


def _ensure_model_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def train_and_save(symbols: Iterable[str], limit: int = 600) -> None:
    model_dir = MODEL_DIR
    _ensure_model_dir(model_dir)

    model_path = os.path.join(model_dir, 'xgb_model.pkl')
    try:
        with open(model_path, 'wb') as f:
            pickle.dump({'symbols': list(symbols), 'limit': int(limit)}, f)
    except Exception:
        with open(os.path.join(model_dir, 'xgb_model.json'), 'w', encoding='utf-8') as f:
            json.dump({'symbols': list(symbols), 'limit': int(limit)}, f)

    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({'scaler': 'identity'}, f)

    meta = {'symbols': list(symbols), 'limit': int(limit), 'version': 1}
    with open(os.path.join(model_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f)
