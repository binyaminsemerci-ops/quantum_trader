import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


class XGBAgent:
    """
    Unified XGBoost Agent v5
    - Fully backward compatible with ensemble_manager
    - Supports model_path, scaler_path, use_ensemble args
    - Auto feature alignment between prod & model
    - Proper 3-class classification (SELL / HOLD / BUY)
    """

    def __init__(self, use_ensemble=True, model_path=None, scaler_path=None):
        self.model = None
        self.scaler = None
        self.expected_features = []
        self.model_version = "unknown"
        self.ready = False
        self.use_ensemble = use_ensemble

        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.model_dir = os.path.join(base_dir, "models")

        print("[XGB-Agent] Initializing unified v5 agent...")

        try:
            self._load_model_and_scaler(model_path, scaler_path)
            self.ready = True
            print(
                f"[XGB-Agent] ✅ Model ready (version={self.model_version}, "
                f"features={len(self.expected_features)})"
            )
        except Exception as e:
            print(f"[XGB-Agent] ❌ Initialization failed: {e}")

    # ----------------------------------------------------------------------
    def _find_latest_file(self, prefix: str):
        """Find the newest file in model directory with given prefix."""
        files = [
            os.path.join(self.model_dir, f)
            for f in os.listdir(self.model_dir)
            if f.startswith(prefix) and f.endswith(".pkl")
        ]
        return max(files, key=os.path.getmtime) if files else None

    # ----------------------------------------------------------------------
    def _load_model_and_scaler(self, model_path=None, scaler_path=None):
        """Load model, scaler, and meta (features)"""
        if not model_path:
            model_path = self._find_latest_file("xgb_v")
        if not model_path:
            raise FileNotFoundError("No XGBoost model file found in models directory.")

        if not scaler_path:
            scaler_path = model_path.replace(".pkl", "_scaler.pkl")

        meta_path = model_path.replace(".pkl", "_meta.json")

        print(f"[XGB-Agent] Loading model: {model_path}")
        print(f"[XGB-Agent] Loading scaler: {scaler_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.model_version = os.path.basename(model_path)

        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            self.expected_features = meta.get("features", [])
            print(f"[XGB-Agent] Meta loaded ({len(self.expected_features)} features).")
        else:
            print("[XGB-Agent] ⚠️ No meta found, inferring feature count from scaler.")
            self.expected_features = [f"f{i}" for i in range(self.scaler.n_features_in_)]

    # ----------------------------------------------------------------------
    def _align_features(self, features: dict):
        """Align input feature dict to expected model features safely."""
        df = pd.DataFrame([features])
        incoming = set(df.columns)
        expected = self.expected_features

        dropped = [c for c in incoming if c not in expected]
        if dropped:
            print(f"[XGB-Agent] Dropping unknown features: {dropped}")

        df = df[[f for f in expected if f in df.columns]]

        missing = [f for f in expected if f not in df.columns]
        for f in missing:
            df[f] = 0.0

        df = df[expected]
        return df, dropped, missing

    # ----------------------------------------------------------------------
    def predict(self, symbol: str, features: dict):
        """Main prediction entry point for ensemble manager."""
        if not self.is_ready():
            raise RuntimeError("XGBoost agent not ready or model not loaded.")

        aligned_df, dropped, missing = self._align_features(features)

        X_scaled = self.scaler.transform(aligned_df)
        y_pred_prob = self.model.predict_proba(X_scaled)

        class_idx = int(np.argmax(y_pred_prob, axis=1)[0])
        mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
        action = mapping.get(class_idx, "HOLD")

        confidence = float(np.max(y_pred_prob))
        confidence_std = float(np.std(y_pred_prob))

        # Ensemble logging
        print(
            f"[XGB-Agent] {symbol} → {action} "
            f"(conf={confidence:.4f}, std={confidence_std:.4f}) | "
            f"used={len(aligned_df.columns)}, dropped={len(dropped)}, missing={len(missing)}"
        )

        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "confidence_std": confidence_std,
            "used_features": len(aligned_df.columns),
            "dropped_features": dropped,
            "missing_features": missing,
            "version": self.model_version,
        }

    # ----------------------------------------------------------------------
    def is_ready(self):
        return self.ready and self.model is not None and self.scaler is not None
