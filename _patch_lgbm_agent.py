#!/usr/bin/env python3
"""Patch lgbm_agent.py to support lgb.Booster native format and pickle fallback."""
import re

path = "/opt/quantum/ai_engine/agents/lgbm_agent.py"
with open(path, "r") as f:
    src = f.read()

# 1. Insert _LGBMBoosterWrapper class after imports (after 'logger = ...' line)
wrapper_class = '''

class _LGBMBoosterWrapper:
    """Wraps lgb.Booster to expose predict_proba() so ensemble code works."""
    def __init__(self, booster):
        self._booster = booster

    def predict_proba(self, X):
        import numpy as np
        probs = self._booster.predict(X)
        if probs.ndim == 1:
            return np.column_stack([1 - probs, probs])
        return probs  # shape (n_samples, n_classes)

    def predict(self, X):
        return self._booster.predict(X)

    @property
    def feature_importances_(self):
        return self._booster.feature_importance()

'''

if "_LGBMBoosterWrapper" not in src:
    src = src.replace(
        "logger = logging.getLogger(__name__)\n",
        "logger = logging.getLogger(__name__)\n" + wrapper_class
    )
    print("Inserted _LGBMBoosterWrapper")
else:
    print("Wrapper already present")

# 2. Replace the load block inside _load_model
old_load = '''            if model_file.exists():
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"✅ LightGBM model loaded from {model_file.name}")'''

new_load = '''            if model_file.exists():
                import lightgbm as _lgb
                try:
                    with open(model_file, 'rb') as f:
                        loaded = pickle.load(f)
                    if isinstance(loaded, _lgb.Booster):
                        self.model = _LGBMBoosterWrapper(loaded)
                    else:
                        self.model = loaded
                except Exception as pkl_err:
                    logger.warning(f"[LGBM] pickle load failed ({pkl_err!r}), trying native lgb.Booster")
                    self.model = _LGBMBoosterWrapper(_lgb.Booster(model_file=str(model_file)))
                logger.info(f"✅ LightGBM model loaded from {model_file.name}")'''

if old_load in src:
    src = src.replace(old_load, new_load)
    print("Patched _load_model load block")
else:
    print("ERROR: old_load block not found - check indentation")

with open(path, "w") as f:
    f.write(src)
print("Done")
