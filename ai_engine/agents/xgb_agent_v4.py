import os
import glob
import json
import joblib
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class XGBAgent:
    """
    XGBoost Agent v4
    - Automatisk last av modell (.pkl) + scaler (.pkl)
    - Kompatibel med v2, v3, v4
    - Robust mot manglende filer og feil versjoner
    """

    def __init__(self, use_ensemble=False, model_path=None, scaler_path=None, use_advanced_features=True):
        """Initialize XGBAgent with backward compatibility.
        
        Args:
            use_ensemble: Legacy parameter (ignored)
            model_path: Optional path to model file
            scaler_path: Optional path to scaler file
            use_advanced_features: Legacy parameter (ignored)
        """
        self.model = None
        self.scaler = None
        self.model_version = None
        self.feature_order = None
        self.model_path = model_path  # Will be overridden by _find_latest_model if None
        self.scaler_path = scaler_path  # Will be overridden by _find_latest_model if None

        try:
            self._load_model_and_scaler()
        except Exception as e:
            logger.error(f"[XGB-Agent] ❌ Error loading model: {e}")
            self.model = None

    # =====================================================
    # === MODEL DISCOVERY ===
    # =====================================================
    def _find_latest_model(self, directory: str, pattern: str):
        files = glob.glob(os.path.join(directory, pattern))
        if not files:
            return None
        latest = max(files, key=os.path.getmtime)
        return latest

    # =====================================================
    # === MODEL LOADING ===
    # =====================================================
    def _load_model_and_scaler(self):
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        retraining_dir = "/app/models" if os.path.exists("/app/models") else base_dir

        # Use provided paths or find latest
        if self.model_path and os.path.exists(self.model_path):
            model_path = self.model_path
        else:
            # Prioriter v4, deretter v3, så v2
            model_path = (
                self._find_latest_model(retraining_dir, "xgb_v*_v4.pkl")
                or self._find_latest_model(retraining_dir, "xgb_v*_v3.pkl")
                or self._find_latest_model(retraining_dir, "xgb_v*_v2.pkl")
            )

        if not model_path:
            raise FileNotFoundError("No XGBoost model (.pkl) found in models directory")

        # Finn matchende scaler (samme timestamp)
        prefix = model_path.replace(".pkl", "")
        scaler_path = self.scaler_path if self.scaler_path and os.path.exists(self.scaler_path) else f"{prefix}_scaler.pkl"
        meta_path = f"{prefix}_meta.json"

        # Last modell
        self.model = joblib.load(model_path)
        self.model_path = model_path
        self.model_version = "v4" if "_v4" in model_path else ("v3" if "_v3" in model_path else "v2")

        # Last scaler (dersom eksisterer)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.scaler_path = scaler_path
            logger.info(f"[XGB-Agent] ✅ Scaler loaded: {os.path.basename(scaler_path)}")
        else:
            logger.warning(f"[XGB-Agent] ⚠️ No scaler found for model {os.path.basename(model_path)}")

        # Les feature order fra metadata om mulig
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.feature_order = meta.get("features", [])
                logger.info(f"[XGB-Agent] Feature order loaded ({len(self.feature_order)} features)")
        else:
            logger.warning("[XGB-Agent] ⚠️ Metadata not found — using dynamic feature extraction")

        logger.info(f"[XGB-Agent] ✅ Model loaded: {os.path.basename(model_path)} ({self.model_version})")

    # =====================================================
    # === FEATURE PREPROCESSING ===
    # =====================================================
    def _prepare_input(self, features: dict):
        # Hvis feature_order kjent → behold rekkefølge
        if self.feature_order:
            x = np.array([[features.get(f, 0.0) for f in self.feature_order]])
        else:
            x = np.array([list(features.values())])

        # Bruk scaler hvis tilgjengelig
        if self.scaler is not None:
            x = self.scaler.transform(x)

        return x

    # =====================================================
    # === PREDICT ===
    # =====================================================
    def predict(self, symbol: str, features: dict):
        if self.model is None:
            return "HOLD", 0.0, "xgb_model_not_loaded"

        x = self._prepare_input(features)
        try:
            # XGBoost Booster requires DMatrix
            import xgboost as xgb
            dmatrix = xgb.DMatrix(x)
            y_pred_prob = self.model.predict(dmatrix)
            
            if y_pred_prob.ndim == 1:
                y_pred_prob = np.expand_dims(y_pred_prob, axis=0)
            class_idx = np.argmax(y_pred_prob, axis=1)[0]

            mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
            action = mapping.get(class_idx, "HOLD")

            conf = float(np.max(y_pred_prob))
            conf_std = float(np.std(y_pred_prob))

            # Fallback hvis modellen er degenerert
            if conf_std < 0.02:
                logger.warning(f"[XGB-Agent] ⚠️ Degenerate output: conf_std={conf_std:.4f} < 0.02")
                return "HOLD", conf, "xgb_degenerate_output"

            logger.debug(f"[XGB-Agent] {symbol}: {action} {conf:.2%} (conf_std={conf_std:.4f})")
            return action, conf, f"xgb_{self.model_version}"
        except Exception as e:
            logger.error(f"[XGB-Agent] ❌ Prediction error: {e}")
            return "HOLD", 0.0, "xgb_prediction_error"

    # =====================================================
    # === COMPATIBILITY WITH OLD INTERFACE ===
    # =====================================================
    def is_ready(self):
        """Check if model is loaded and ready."""
        return self.model is not None
    
    def predict_for_symbol(self, ohlcv):
        """Compatibility method for existing code that expects dict return."""
        # This is a placeholder - in production you'd need to extract features from ohlcv
        # For now, return safe fallback
        return {
            "action": "HOLD",
            "score": 0.5,
            "confidence": 0.5,
            "model": "xgboost_v4"
        }

    # =====================================================
    # === DEBUG INFO ===
    # =====================================================
    def info(self):
        return {
            "version": self.model_version,
            "model": os.path.basename(self.model_path) if self.model_path else None,
            "scaler": os.path.basename(self.scaler_path) if self.scaler_path else None,
            "features": len(self.feature_order) if self.feature_order else None,
        }


# === Manual test ===
if __name__ == "__main__":
    agent = XGBAgent()
    print(agent.info())

    # Dummy features for test
    test_features = {f"feat_{i}": np.random.randn() for i in range(23)}
    action, conf, msg = agent.predict("BTCUSDT", test_features)
    print(f"Prediction: {action} ({conf:.2%}) | {msg}")
