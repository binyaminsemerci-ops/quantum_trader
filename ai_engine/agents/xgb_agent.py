from typing import List, Dict, Any, Optional, Mapping
import os
import pickle
import logging
import asyncio
import numpy as np
import time
import hashlib

try:
    from backend.utils.twitter_client import TwitterClient
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    TwitterClient = None

logger = logging.getLogger(__name__)

# Rate limiter for confidence forensics logging
_last_forensic_log_time = 0
_FORENSIC_LOG_INTERVAL = 30  # seconds


class XGBAgent:
    """Enhanced agent wrapper with ensemble models and advanced features.
    
    Now supports:
    - Ensemble models (6 models combined)
    - Advanced features (100+)
    - Confidence scoring from model agreement
    - Fallback to single XGBoost if ensemble unavailable

    The implementation avoids hard failures when optional deps (pandas, xgboost)
    are missing and uses small fallbacks so endpoints remain responsive.
    """

    def __init__(
        self, 
        model_path: Optional[str] = None, 
        scaler_path: Optional[str] = None,
        use_ensemble: bool = True,
        use_advanced_features: bool = True
    ):
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
        # 🔥 USE LATEST TIMESTAMPED MODEL (not old hardcoded names)
        # Retraining saves to /app/models/, agents default to ai_engine/models
        retraining_dir = "/app/models" if os.path.exists("/app/models") else base
        # Try new naming first (xgb_v*_v2.pkl), fall back to old (xgboost_v*_v2.pkl)
        latest_model = self._find_latest_model(retraining_dir, "xgb_v*_v2.pkl") or \
                       self._find_latest_model(retraining_dir, "xgboost_v*_v2.pkl")
        latest_scaler = self._find_latest_model(retraining_dir, "xgboost_scaler_v*_v2.pkl")
        
        self.model_path = model_path or latest_model or os.path.join(base, "xgb_model.pkl")
        self.scaler_path = scaler_path or latest_scaler or os.path.join(base, "scaler.pkl")
        self.ensemble_path = os.path.join(base, "ensemble_model.pkl")
        
        self.model = None
        self.scaler = None
        self.ensemble = None
        self.use_ensemble = use_ensemble
        self.use_advanced_features = use_advanced_features
        
        # 🔒 FAIL-CLOSED: Degeneracy detection (testnet only)
        self._prediction_history = []  # Rolling window of (action, confidence)
        self._degeneracy_window = 100
        
        # helper clients
        self.twitter: Optional[TwitterClient] = None
        if TWITTER_AVAILABLE:
            try:
                self.twitter = TwitterClient()
            except Exception as e:
                logger.debug("Failed to init Twitter client: %s", e)
                self.twitter = None
        self._load()
        
        # 🔒 FAIL-CLOSED: Log model metadata for diagnostics
        self._log_model_metadata()

    def _find_latest_model(self, base_dir: str, pattern: str) -> Optional[str]:
        """Find the latest timestamped model file matching pattern."""
        try:
            import glob
            model_files = glob.glob(os.path.join(base_dir, pattern))
            if model_files:
                # Sort by filename (timestamp is in filename)
                latest = sorted(model_files)[-1]
                logger.info(f"🔍 Found latest model: {os.path.basename(latest)}")
                return latest
        except Exception as e:
            logger.warning(f"Failed to find latest model with pattern {pattern}: {e}")
        return None

    def _load(self) -> None:
        """Load model and scaler from disk if present. Swallow errors and log them."""
        # Try loading ensemble first if requested
        if self.use_ensemble and os.path.exists(self.ensemble_path):
            try:
                from ai_engine.model_ensemble import EnsemblePredictor
                self.ensemble = EnsemblePredictor()
                self.ensemble.load("ensemble_model.pkl")
                logger.info("[OK] Loaded ensemble model from %s", self.ensemble_path)
            except Exception as e:
                logger.warning("Failed to load ensemble: %s. Falling back to single model.", e)
                self.ensemble = None
        
        # Load single model as fallback
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                    logger.info("✅ Loaded XGBoost model from %s", os.path.basename(self.model_path))
                    
                    # 🔬 FORENSICS: Log model metadata for diagnostics
                    if hasattr(self.model, 'classes_'):
                        logger.info(f"[XGB-FORENSIC] Model classes: {self.model.classes_} (n={len(self.model.classes_)})")
                    elif hasattr(self.model, 'n_classes_'):
                        logger.info(f"[XGB-FORENSIC] Model n_classes: {self.model.n_classes_}")
                    else:
                        logger.info("[XGB-FORENSIC] Model type: no classes_ attribute (may be regressor)")
            else:
                logger.warning("[XGB] Model file not found at: %s", self.model_path)
                self.model = None
        except Exception as e:
            logger.warning("[XGB] Failed to load model from %s: %s", self.model_path, e)
            self.model = None

        try:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                    logger.info("✅ Loaded XGBoost scaler from %s", os.path.basename(self.scaler_path))
            else:
                logger.warning("[XGB] Scaler file not found at: %s", self.scaler_path)
                self.scaler = None
        except Exception as e:
            logger.warning("[XGB] Failed to load scaler from %s: %s", self.scaler_path, e)
            self.scaler = None
    
    def _log_model_metadata(self) -> None:
        """🔒 FAIL-CLOSED: Log model metadata at initialization"""
        if self.model is None:
            logger.warning("[XGB-INIT] ⚠️  Model not loaded - predictions will fail (FAIL-CLOSED)")
            return
        
        # Log model file info
        logger.info(f"[XGB-INIT] Model file: {os.path.basename(self.model_path)}")
        logger.info(f"[XGB-INIT] Scaler file: {os.path.basename(self.scaler_path) if self.scaler else 'None'}")
        
        # Try to log expected feature dimension
        feature_names_path = os.path.join(
            os.path.dirname(self.model_path), 
            'xgboost_features.pkl'
        )
        if os.path.exists(feature_names_path):
            try:
                with open(feature_names_path, 'rb') as f:
                    expected_features = pickle.load(f)
                logger.info(f"[XGB-INIT] Expected feature_dim: {len(expected_features)}")
            except Exception as e:
                logger.warning(f"[XGB-INIT] Could not read feature names: {e}")
        else:
            logger.info("[XGB-INIT] Expected feature_dim: 9 (SPOT model, no features.pkl)")
        
        # Compute SHA256 fingerprint of model file
        try:
            with open(self.model_path, 'rb') as f:
                model_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            logger.info(f"[XGB-INIT] Model SHA256: {model_hash}...")
        except Exception as e:
            logger.warning(f"[XGB-INIT] Could not compute model hash: {e}")

    def _features_from_ohlcv(self, df) -> Any:
        """Turn raw OHLCV (DataFrame or list-of-dicts) into model features (last row).
        
        BULLETPROOF: Never raises, always returns valid features or None.
        Uses advanced features (100+) when available.
        """
        try:
            import pandas as _pd  # type: ignore[import-untyped]

            # local import for feature engineer; mypy in CI may not resolve ai_engine package here
            from ai_engine.feature_engineer import compute_all_indicators  # type: ignore[import-not-found, import-untyped]
            from ai_engine.feature_engineer import add_sentiment_features as _add_sentiment_features  # type: ignore[import-not-found, import-untyped]
        except Exception as e:
            logger.error("CRITICAL: Pandas or feature_engineer not available: %s", e)
            return None

        # BULLETPROOF: Validate input data before processing
        if df is None:
            logger.error("CRITICAL: df is None in _features_from_ohlcv")
            return None
            
        # Accept both DataFrame and list-of-dicts
        try:
            if isinstance(df, list):
                if not df:  # Empty list
                    logger.error("CRITICAL: Empty list passed to _features_from_ohlcv")
                    return None
                df = _pd.DataFrame(df)
        except Exception as e:
            logger.error("CRITICAL: Failed to convert list to DataFrame: %s", e)
            return None

        # Normalize column casing to predictable names
        try:
            df.columns = [str(c).lower() for c in df.columns]
        except Exception as e:
            logger.error("CRITICAL: Failed to normalize column names: %s", e)
            return None

        # BULLETPROOF: ensure we have required columns (open, high, low, close, volume)
        # Use forward-fill strategy for missing values instead of NA
        try:
            required_cols = ["open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    # Try to use close price as fallback for OHLC
                    if "close" in df.columns and col in ["open", "high", "low"]:
                        df[col] = df["close"]
                    elif "volume" in df.columns and col == "volume":
                        df[col] = df["volume"]
                    else:
                        df[col] = 0.0  # Safe numeric fallback
                        logger.warning("Missing column %s, using 0.0", col)
            
            # Validate we have at least some data
            if len(df) < 2:
                logger.error("CRITICAL: Insufficient data rows: %d", len(df))
                return None
        except Exception as e:
            logger.error("CRITICAL: Failed to ensure required columns: %s", e)
            return None

        # prepare DataFrame expected by feature engineer (capitalized names sometimes expected)
        df_norm = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        # BULLETPROOF: Feature computation using UNIFIED feature engineering
        # This ensures training and inference use IDENTICAL features
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from backend.shared.unified_features import get_feature_engineer
            
            # Use unified feature engineer (same as training)
            engineer = get_feature_engineer()
            feat = engineer.compute_features(df)  # Use lowercase df directly
            
            # Validate features were generated
            if feat is None or feat.shape[0] == 0:
                logger.error("CRITICAL: Unified feature engineer returned empty features")
                return None
                
            logger.info(f"✅ Computed {len(feat.columns)} unified features")
        except Exception as e:
            logger.error("CRITICAL: Unified feature computation failed: %s", e)
            # FALLBACK: Try old feature engineer
            try:
                feat = compute_all_indicators(df_norm, use_advanced=self.use_advanced_features)
                if feat is None or feat.shape[0] == 0:
                    logger.error("CRITICAL: Fallback feature computation failed")
                    return None
            except Exception as e2:
                logger.error("CRITICAL: All feature computation failed: %s", e2)
                return None

        # BULLETPROOF: Add sentiment features with error handling
        try:
            sentiment_series = None
            news_counts = None
            if "sentiment" in df.columns:
                sentiment_series = df["sentiment"]
            if "news_count" in df.columns:
                news_counts = df["news_count"]

            if sentiment_series is not None or news_counts is not None:
                feat = _add_sentiment_features(
                    feat, sentiment_series=sentiment_series, news_counts=news_counts
                )
        except Exception as e:
            logger.warning("Failed to add sentiment features (non-critical): %s", e)
            # Continue without sentiment features

        # BULLETPROOF: Safe return with validation
        try:
            if feat.shape[0] == 0:
                logger.error("CRITICAL: No features after processing")
                return None
            return feat.iloc[-1:]
        except Exception as e:
            logger.error("CRITICAL: Failed to return feature row: %s", e)
            return None
    
    def _safe_rsi(self, series, period: int = 14):
        """BULLETPROOF RSI calculation that never fails"""
        try:
            import pandas as pd
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Neutral RSI for NaN values
        except Exception as e:
            logger.error("RSI calculation failed: %s", e)
            import pandas as pd
            return pd.Series([50] * len(series))  # Return neutral RSI

    def is_ready(self) -> bool:
        """Check if model and scaler are loaded and ready for predictions."""
        return self.model is not None and self.scaler is not None

    def predict(self, symbol: str, features: Dict[str, float]) -> tuple[str, float, str]:
        """
        Prediction interface for ensemble compatibility.
        Supports both SPOT features (9) and FUTURES features (22).
        
        🔒 FAIL-CLOSED: Validates features, detects degeneracy, raises on error.
        
        Args:
            symbol: Trading pair (unused but kept for interface compatibility)
            features: Dict of technical indicators
        
        Returns:
            (action, confidence, model_name)
        """
        try:
            if self.model is None:
                raise RuntimeError("[XGB] QSC FAIL-CLOSED: Model not loaded. Cannot predict without model.")
            
            if features is None or not features:
                raise ValueError("[XGB] QSC FAIL-CLOSED: Features dict is None or empty.")
            
            # Load feature names from model training if available
            import os
            feature_names_path = os.path.join(
                os.path.dirname(self.model_path), 
                'xgboost_features.pkl'
            )
            
            if os.path.exists(feature_names_path):
                # FUTURES model - use exact feature order from training
                with open(feature_names_path, 'rb') as f:
                    expected_features = pickle.load(f)
                
                # Build feature array in exact order
                feature_list = [features.get(name, 0.0) for name in expected_features]
            else:
                # SPOT model - use original 9-feature order
                feature_list = [
                    features.get('rsi_14', 50),
                    features.get('macd', 0),
                    features.get('macd_signal', 0),
                    features.get('ema_10', 0),
                    features.get('ema_20', 0),
                    features.get('ema_50', 0),
                    features.get('price_change', 0),
                    features.get('volume_change', 0),
                    features.get('volatility_20', 0.01)
                ]
            
            # 🔒 FAIL-CLOSED: Validate feature array before scaling
            import numpy as np
            feature_array = np.array(feature_list).reshape(1, -1)
            
            # Check for NaN/Inf
            if np.any(np.isnan(feature_array)):
                raise ValueError(f"[XGB] QSC FAIL-CLOSED: Feature array contains NaN values for {symbol}")
            if np.any(np.isinf(feature_array)):
                raise ValueError(f"[XGB] QSC FAIL-CLOSED: Feature array contains Inf values for {symbol}")
            
            # Check dimension match if scaler available
            if self.scaler and hasattr(self.scaler, 'n_features_in_'):
                expected_dim = self.scaler.n_features_in_
                actual_dim = feature_array.shape[1]
                if actual_dim != expected_dim:
                    raise ValueError(
                        f"[XGB] QSC FAIL-CLOSED: Feature dimension mismatch for {symbol}. "
                        f"Expected {expected_dim}, got {actual_dim}. "
                        f"Feature engineering must produce correct dimension."
                    )
            
            # Scale if scaler available
            if self.scaler:
                try:
                    import pandas as pd
                    if hasattr(self.scaler, "feature_names_in_"):
                        cols = list(self.scaler.feature_names_in_)
                        target_len = len(cols)
                        vec = feature_array
                        if vec.shape[1] != target_len:
                            if vec.shape[1] > target_len:
                                vec = vec[:, :target_len]
                            else:
                                pad = np.zeros((vec.shape[0], target_len - vec.shape[1]))
                                vec = np.concatenate([vec, pad], axis=1)
                        feature_array = pd.DataFrame(vec, columns=cols)
                    feature_array = self.scaler.transform(feature_array)
                except Exception:
                    feature_array = self.scaler.transform(feature_array)
            else:
                import numpy as np
                feature_array = np.array(feature_list).reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(feature_array)[0]
            proba = self.model.predict_proba(feature_array)[0]
            
            # Get top 2 probabilities for forensics and calibration
            sorted_proba = sorted(proba, reverse=True)
            proba_top1 = sorted_proba[0]
            proba_top2 = sorted_proba[1] if len(sorted_proba) > 1 else 0.0
            margin = proba_top1 - proba_top2
            
            # Confidence mode: "max" (default) or "margin" (calibrated)
            conf_mode = os.getenv('XGB_CONF_MODE', 'max').lower()
            
            if conf_mode == 'margin':
                # Margin-based calibration: [0, 1] → [0.50, 0.95]
                confidence = float(0.50 + min(margin, 1.0) * 0.45)
            else:
                # Original: max probability (may be overconfident)
                confidence = float(max(proba))
            
            # Map prediction to action
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action = action_map.get(prediction, 'HOLD')
            
            # 🔬 FORENSICS: Rate-limited debug logging (max once per 30s)
            global _last_forensic_log_time
            now = time.time()
            if now - _last_forensic_log_time >= _FORENSIC_LOG_INTERVAL:
                _last_forensic_log_time = now
                
                # Feature vector fingerprint (length only, no raw data)
                feat_hash = hashlib.md5(str(feature_array.shape).encode()).hexdigest()[:8]
                
                logger.info(
                    f"[XGB-FORENSIC] {symbol} | action={action} pred={prediction} | "
                    f"top1={proba_top1:.4f} top2={proba_top2:.4f} margin={margin:.4f} | "
                    f"max_proba={max(proba):.4f} conf_final={confidence:.4f} | "
                    f"mode={conf_mode} feat_dim={feature_array.shape[1]} hash={feat_hash}"
                )
            
            # DEBUG: Log every 10th prediction (existing code)
            import random
            if random.random() < 0.1:  # 10% sampling
                logger.info(f"XGB {symbol}: {action} {confidence:.2%} (pred={prediction})")
            
            # 🔒 FAIL-CLOSED: Degeneracy detection (testnet/collection only)
            is_testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
            if is_testnet:
                self._prediction_history.append((action, confidence))
                if len(self._prediction_history) > self._degeneracy_window:
                    self._prediction_history.pop(0)
                
                # Check for degeneracy: >95% same action + low confidence variance
                if len(self._prediction_history) >= self._degeneracy_window:
                    actions = [a for a, c in self._prediction_history]
                    confidences = [c for a, c in self._prediction_history]
                    
                    # Count most common action
                    from collections import Counter
                    action_counts = Counter(actions)
                    most_common_action, most_common_count = action_counts.most_common(1)[0]
                    action_pct = (most_common_count / len(actions)) * 100
                    
                    # Calculate confidence std
                    conf_std = np.std(confidences)
                    
                    # FAIL-CLOSED: If >95% same action AND low variance
                    if action_pct > 95 and conf_std < 0.02:
                        raise RuntimeError(
                            f"[XGB] QSC FAIL-CLOSED: Degenerate output detected. "
                            f"Action '{most_common_action}' occurs {action_pct:.1f}% of time "
                            f"with confidence_std={conf_std:.6f} < 0.02. "
                            f"Model is not producing varied predictions - likely OOD input or collapsed weights."
                        )
            
            return (action, confidence, 'xgboost')
            
        except Exception as e:
            logger.error(f"XGBoost predict failed: {e} - FAIL-CLOSED (no fallback)")
            raise  # FAIL-CLOSED: propagate error instead of returning HOLD 0.5

    def predict_for_symbol(self, ohlcv) -> Dict[str, Any]:
        """BULLETPROOF prediction - ALWAYS returns valid response, NEVER raises.
        
        Returns {'action':..., 'score':..., 'confidence':..., 'model':...}
        Includes confidence scoring from ensemble model agreement.
        Falls back to rule-based logic if ML fails.
        """
        # BULLETPROOF: Validate input
        if ohlcv is None:
            logger.error("CRITICAL: ohlcv is None in predict_for_symbol")
            return {
                "action": "HOLD", 
                "score": 0.0, 
                "confidence": 0.0, 
                "model": "error_null_input"
            }
        
        try:
            feat = self._features_from_ohlcv(ohlcv)
            if feat is None:
                logger.error("CRITICAL: Feature extraction returned None")
                # Try rule-based fallback with raw data
                return self._emergency_fallback(ohlcv)
        except Exception as e:
            logger.error("CRITICAL: Feature extraction crashed: %s", e)
            return self._emergency_fallback(ohlcv)

        # BULLETPROOF: Cast and select numeric features with validation
        from typing import Any as _Any, cast as _cast

        feat_any = _cast(_Any, feat)

        # CRITICAL FIX: Select ONLY the 14 features the model was trained on
        # This matches bootstrap data features exactly
        required_features = [
            "Close", "Volume", "EMA_10", "EMA_50", "RSI_14",
            "MACD", "MACD_signal", "BB_upper", "BB_middle", "BB_lower",
            "ATR", "volume_sma_20", "price_change_pct", "high_low_range"
        ]
        
        # Map from actual column names to expected names
        feature_mapping = {
            "RSI_14": "RSI",  # Feature engineer uses RSI_14, model expects RSI
            # Add other mappings if needed
        }
        
        # select numeric features with multiple fallback strategies
        try:
            # Create DataFrame with only the required features
            import pandas as pd
            feature_dict = {}
            
            for feat_name in required_features:
                # Check if feature exists (with or without mapping)
                actual_name = feat_name
                if feat_name not in feat_any.columns:
                    # Try mapped name
                    for mapped_from, mapped_to in feature_mapping.items():
                        if mapped_to == feat_name and mapped_from in feat_any.columns:
                            actual_name = mapped_from
                            break
                
                if actual_name in feat_any.columns:
                    feature_dict[feat_name if feat_name not in feature_mapping.values() else list(feature_mapping.keys())[list(feature_mapping.values()).index(feat_name)]] = float(feat_any[actual_name].iloc[0])
                else:
                    # Use safe default for missing features
                    logger.warning(f"Missing feature {feat_name}, using default")
                    feature_dict[feat_name] = 0.0
            
            # Convert to numpy array in correct order
            X = np.array([[feature_dict.get(f, 0.0) for f in required_features]])
            
            # Validate result
            if X.size == 0:
                logger.error("CRITICAL: No numeric features extracted")
                return self._emergency_fallback(ohlcv)
            
            # Replace inf/nan with safe values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
        except Exception as e:
            logger.error("CRITICAL: Failed to select numeric features: %s", e)
            return self._emergency_fallback(ohlcv)

        # BULLETPROOF: Ensure 2D with validation
        try:
            if X.ndim == 1:
                X = X.reshape(1, -1)
            elif X.ndim == 0:
                logger.error("CRITICAL: X is 0-dimensional")
                return self._emergency_fallback(ohlcv)
        except Exception as e:
            logger.error("CRITICAL: Failed to reshape X: %s", e)
            return self._emergency_fallback(ohlcv)

        # BULLETPROOF: Apply scaler with error handling
        if self.scaler is not None:
            try:
                Xs = self.scaler.transform(X)
                # Validate scaling didn't produce garbage
                if np.any(np.isnan(Xs)) or np.any(np.isinf(Xs)):
                    logger.warning("Scaler produced nan/inf, using unscaled features")
                    Xs = X
            except Exception as e:
                logger.warning("Scaler.transform failed: %s, using unscaled features", e)
                Xs = X
        else:
            Xs = X

        # ============================================================
        # ENSEMBLE PREDICTION (if available)
        # ============================================================
        if self.ensemble is not None:
            try:
                prediction, confidence = self.ensemble.predict_with_confidence(Xs)
                v = float(prediction[0])
                conf = float(confidence[0])
                
                # interpret numeric prediction: positive -> buy, negative -> sell
                if v > 0.01:
                    return {
                        "action": "BUY", 
                        "score": min(0.99, float(v)),
                        "confidence": conf,
                        "model": "ensemble",
                        "features": feat_any.to_dict('records')[0] if hasattr(feat_any, 'to_dict') else {}
                    }
                if v < -0.01:
                    return {
                        "action": "SELL", 
                        "score": min(0.99, float(abs(v))),
                        "confidence": conf,
                        "model": "ensemble",
                        "features": feat_any.to_dict('records')[0] if hasattr(feat_any, 'to_dict') else {}
                    }
                return {
                    "action": "HOLD", 
                    "score": float(abs(v)),
                    "confidence": conf,
                    "model": "ensemble",
                    "features": feat_any.to_dict('records')[0] if hasattr(feat_any, 'to_dict') else {}
                }
            except Exception as e:
                logger.warning("Ensemble prediction failed: %s. Falling back to single model.", e)
                # Fall through to single model

        # ============================================================
        # SINGLE MODEL PREDICTION (fallback)
        # ============================================================
        
        # If no trained model, use simple EMA heuristic if available
        if self.model is None:
            try:
                if "EMA_10" in feat_any.columns and "Close" in feat_any.columns:
                    last = float(feat_any["Close"].iloc[0])
                    ema = float(feat_any["EMA_10"].iloc[0])
                    if last > ema * 1.002:
                        return {"action": "BUY", "score": 0.6, "confidence": 0.5, "model": "heuristic"}
                    if last < ema * 0.998:
                        return {"action": "SELL", "score": 0.6, "confidence": 0.5, "model": "heuristic"}
            except Exception as e:
                logger.debug("EMA heuristic failed: %s", e)
            return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "model": "none"}

        # Use model predict or predict_proba when available
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(Xs)
                # choose positive class probability if 2-class
                score = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])

                # Make BUY/SELL probability thresholds configurable to ease tuning
                import os as _os
                try:
                    buy_thr = float(_os.getenv("QT_PROBA_BUY", "0.52"))
                except Exception:
                    buy_thr = 0.52
                try:
                    sell_thr = float(_os.getenv("QT_PROBA_SELL", "0.48"))
                except Exception:
                    sell_thr = 0.48

                # Clamp to sensible range and ensure separation
                buy_thr = max(0.5, min(0.99, buy_thr))
                sell_thr = min(0.5, max(0.01, sell_thr))

                if score >= buy_thr:
                    action = "BUY"
                elif score <= sell_thr:
                    action = "SELL"
                else:
                    action = "HOLD"

                confidence = abs(score - 0.5) * 2
                
                # If model has very low confidence, use rule-based fallback
                # Lowered from 0.55 to 0.35 to give ML model more chances
                if confidence < 0.35:
                    return self._rule_based_fallback(feat_any)

                return {
                    "action": action,
                    "score": score,
                    "confidence": confidence,
                    "model": "xgboost",
                    "features": feat_any.to_dict('records')[0] if hasattr(feat_any, 'to_dict') else {}
                }

            preds = self.model.predict(Xs)
            v = float(preds[0])
            confidence = min(0.99, abs(v))  # Use prediction magnitude as confidence

            # interpret numeric prediction: positive -> buy, negative -> sell
            # Make threshold configurable via env for easier tuning without code changes
            import os
            try:
                threshold = float(os.getenv("QT_XGB_THRESHOLD", "0.001"))
                if threshold <= 0:
                    threshold = 0.001
            except Exception:
                threshold = 0.001

            # ADJUSTED THRESHOLDS: Now configurable; default 0.001
            # Example: set QT_XGB_THRESHOLD=0.0005 to increase sensitivity
            if v > threshold:
                return {
                    "action": "BUY", 
                    "score": min(0.99, float(v)),
                    "confidence": confidence,
                    "model": "xgboost",
                    "features": feat_any.to_dict('records')[0] if hasattr(feat_any, 'to_dict') else {}
                }
            if v < -threshold:
                return {
                    "action": "SELL", 
                    "score": min(0.99, float(abs(v))),
                    "confidence": confidence,
                    "model": "xgboost",
                    "features": feat_any.to_dict('records')[0] if hasattr(feat_any, 'to_dict') else {}
                }
            
            # If predictions are too close to zero or uncertain, use rules
            # Changed from 0.01 to 0.55 to be more aggressive with rule-based trading
            if confidence < 0.55:
                return self._rule_based_fallback(feat_any)
            
            return {
                "action": "HOLD", 
                "score": float(abs(v)),
                "confidence": confidence,
                "model": "xgboost",
                "features": feat_any.to_dict('records')[0] if hasattr(feat_any, 'to_dict') else {}
            }
        except Exception as e:
            logger.debug("Model prediction failed: %s", e)
            # Model error - use rule-based fallback
            try:
                return self._rule_based_fallback(feat_any)
            except Exception:
                return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "model": "error"}

    def predict_direction(self, features: Dict[str, float]) -> tuple[str, float]:
        """Simplified prediction method for compatibility with HybridAgent.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (action, confidence) where action is BUY/SELL/HOLD
        """
        try:
            import pandas as pd
            
            # Convert features dict to DataFrame (single row)
            df = pd.DataFrame([features])
            
            # Use existing predict_for_symbol
            result = self.predict_for_symbol(df)
            
            action = result.get('action', 'HOLD')
            confidence = result.get('confidence', 0.5)
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"XGBAgent predict_direction failed: {e}")
            return "HOLD", 0.5

    def scan_symbols(
        self, symbol_ohlcv: Mapping[str, Any], top_n: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """Pick top_n symbols by recent volume and return predictions."""
        volumes = []
        for s, df in symbol_ohlcv.items():
            try:
                # support pandas DataFrame and list-of-dicts
                if hasattr(df, "columns"):
                    # normalize to lowercase access
                    cols = [c.lower() for c in df.columns]
                    if "volume" in cols:
                        vol = (
                            float(df["volume"].iloc[-1])
                            if "volume" in df.columns
                            else float(df["Volume"].iloc[-1])
                        )
                    else:
                        vol = 0.0
                else:
                    vol = float(df[-1].get("volume", 0.0))
            except Exception:
                vol = 0.0
            volumes.append((s, vol))

        volumes.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in volumes[:top_n]]
        results: Dict[str, Dict[str, Any]] = {}
        for s in selected:
            try:
                res = self.predict_for_symbol(symbol_ohlcv[s])
            except Exception as e:
                logger.debug("predict_for_symbol failed for %s: %s", s, e)
                res = {"action": "HOLD", "score": 0.0}
            results[s] = res
        return results

    def _emergency_fallback(self, ohlcv) -> Dict[str, Any]:
        """EMERGENCY fallback using raw OHLCV data when features fail.
        This ALWAYS works and NEVER raises.
        """
        try:
            import pandas as pd
            
            # Try to extract close prices
            if isinstance(ohlcv, pd.DataFrame):
                if 'close' in ohlcv.columns:
                    close_prices = ohlcv['close'].values
                elif 'Close' in ohlcv.columns:
                    close_prices = ohlcv['Close'].values
                else:
                    return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "model": "emergency_no_data"}
            elif isinstance(ohlcv, list) and len(ohlcv) > 0:
                close_prices = [row.get('close', row.get('Close', 0)) for row in ohlcv]
            else:
                return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "model": "emergency_bad_format"}
            
            # Simple momentum check
            if len(close_prices) < 2:
                return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "model": "emergency_insufficient"}
            
            recent = close_prices[-5:] if len(close_prices) >= 5 else close_prices
            momentum = (recent[-1] / recent[0] - 1.0) * 100 if recent[0] > 0 else 0
            
            # Simple momentum-based signal
            if momentum > 1.0:  # 1% up
                return {"action": "BUY", "score": 0.3, "confidence": 0.15, "model": "emergency_momentum"}
            elif momentum < -1.0:  # 1% down
                return {"action": "SELL", "score": 0.3, "confidence": 0.15, "model": "emergency_momentum"}
            else:
                return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "model": "emergency_neutral"}
        except Exception as e:
            logger.error("CRITICAL: Even emergency fallback failed: %s", e)
            # ABSOLUTE LAST RESORT
            return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "model": "emergency_crashed"}
    
    def _rule_based_fallback(self, feat) -> Dict[str, Any]:
        """Simple rule-based trading logic when ML model has no confidence.
        
        Uses momentum and trend indicators to generate actionable signals.
        """
        try:
            # Extract key features
            close = float(feat["Close"].iloc[0]) if "Close" in feat.columns else 0.0
            if close <= 0:
                return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "model": "rule_fallback"}
            
            # Try to use EMAs if available
            ema_10 = float(feat["EMA_10"].iloc[0]) if "EMA_10" in feat.columns else close
            ema_50 = float(feat["EMA_50"].iloc[0]) if "EMA_50" in feat.columns else close
            
            # Calculate EMA divergence percentage
            ema_div = ((ema_10 / ema_50) - 1.0) * 100 if ema_50 > 0 else 0.0
            
            # Try RSI ONLY for EXTREME oversold/overbought - CONSERVATIVE THRESHOLDS
            if "RSI_14" in feat.columns:
                rsi = float(feat["RSI_14"].iloc[0])
                
                # CONSERVATIVE RSI thresholds - only extreme signals
                if rsi < 30:  # Oversold - BUY signal
                    conf = 0.55 + (30 - rsi) / 60  # Higher confidence when more oversold
                    return {
                        "action": "BUY",
                        "score": 0.70 + (30 - rsi) / 100,
                        "confidence": min(0.75, conf),
                        "model": "rule_fallback_rsi",
                        "features": feat.to_dict('records')[0] if hasattr(feat, 'to_dict') else {}
                    }
                elif rsi > 70:  # Overbought - SELL signal
                    conf = 0.55 + (rsi - 70) / 60  # Higher confidence when more overbought
                    return {
                        "action": "SELL",
                        "score": 0.70 + (rsi - 70) / 100,
                        "confidence": min(0.75, conf),
                        "model": "rule_fallback_rsi",
                        "features": feat.to_dict('records')[0] if hasattr(feat, 'to_dict') else {}
                    }
            
            # EMA momentum: CONSERVATIVE thresholds
            if ema_div > 1.5:  # 10 EMA > 50 EMA by 1.5% (strong trend)
                conf = 0.55 + min(0.20, abs(ema_div) / 10)
                return {
                    "action": "BUY",
                    "score": 0.70,
                    "confidence": conf,
                    "model": "rule_fallback_ema_cross",
                    "features": feat.to_dict('records')[0] if hasattr(feat, 'to_dict') else {}
                }
            elif ema_div < -1.5:  # 10 EMA < 50 EMA by 1.5% (strong downtrend)
                conf = 0.55 + min(0.20, abs(ema_div) / 10)
                return {
                    "action": "SELL",
                    "score": 0.70,
                    "confidence": conf,
                    "model": "rule_fallback_ema_cross",
                    "features": feat.to_dict('records')[0] if hasattr(feat, 'to_dict') else {}
                }
            
            # No clear signal
            return {
                "action": "HOLD", 
                "score": 0.5, 
                "confidence": 0.05, 
                "model": "rule_fallback",
                "features": feat.to_dict('records')[0] if hasattr(feat, 'to_dict') else {}
            }
            
        except Exception as e:
            logger.debug("Rule-based fallback failed: %s", e)
            return {"action": "HOLD", "score": 0.0, "confidence": 0.0, "model": "rule_fallback_error"}

    def reload(self) -> None:
        """Reload model/scaler artifacts from disk."""
        self._load()

    def get_metadata(self) -> Optional[dict]:
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        meta_path = os.path.join(base, "metadata.json")
        if not os.path.exists(meta_path):
            return None
        try:
            import json

            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug("Failed to read metadata: %s", e)
            return None

    async def scan_top_by_volume_from_api(
        self, symbols: List[str], top_n: int = 10, limit: int = 240
    ) -> Dict[str, Dict[str, Any]]:
        """BULLETPROOF: Fetch OHLCV and scan symbols. ALWAYS returns valid dict, NEVER raises.

        Uses bounded concurrency and timeouts at multiple levels.
        Returns empty dict on total failure, partial results on partial failure.
        """
        # BULLETPROOF: Validate inputs
        if not symbols or not isinstance(symbols, list):
            logger.error("CRITICAL: Invalid symbols input: %s", symbols)
            return {}
        
        if top_n <= 0:
            top_n = 10
        if limit <= 0:
            limit = 240
            
        try:
            # Import external_data module directly with fallback
            try:
                import backend.routes.external_data as external_data
            except ImportError:
                from backend.routes import external_data
        except Exception as e:
            logger.error("CRITICAL: external_data not importable: %s", e)
            # Return empty results instead of raising
            return {}

        # OPTIMIZATION: Increased from 6 to 30 concurrent requests
        # Binance Futures can handle 50 requests/second with weight limits
        # 30 parallel requests reduces 222 symbols from 3+ min to ~30 seconds
        sem = asyncio.Semaphore(30)

        async def _fetch(s: str):
            async with sem:
                try:
                    # Timeout each symbol fetch to 10 seconds
                    resp = await asyncio.wait_for(
                        external_data.binance_ohlcv(symbol=s, limit=limit),
                        timeout=10.0
                    )
                    candles = resp.get("candles", [])
                except asyncio.TimeoutError:
                    logger.warning("Timeout fetching candles for %s", s)
                    candles = []
                except Exception as e:
                    logger.debug("Failed to fetch candles for %s: %s", s, e)
                    candles = []

                # OPTIMIZATION: Skip sentiment API calls during fast scanning
                # Sentiment has minimal impact on 65%+ confidence signals
                # Use neutral sentiment (0.0) for all symbols to speed up scanning
                sent_score = 0.0
                
                # DISABLED: Per-symbol sentiment calls cause 3+ min delays
                # TODO: Implement batch sentiment cache (fetch once, reuse for all symbols)
                """
                try:
                    tw = await asyncio.wait_for(
                        external_data.twitter_sentiment(symbol=s),
                        timeout=5.0
                    )
                    sent_score = (
                        tw.get("score", tw.get("sentiment", {}).get("score", 0.0))
                        if isinstance(tw, dict)
                        else 0.0
                    )
                except asyncio.TimeoutError:
                    logger.debug("twitter_sentiment timeout for %s", s)
                    sent_score = 0.0
                except Exception:
                    logger.debug("twitter_sentiment lookup failed for %s", s)
                    sent_score = 0.0
                """

                # Use CoinGecko trending coins as news proxy
                # DISABLED: CoinGecko rate limiting causes timeouts
                # TODO: Re-enable with caching or alternative API
                try:
                    # Skip trending check to avoid rate limiting
                    news_items = []
                    """
                    # Import CoinGecko functions
                    from backend.routes.coingecko_data import get_trending_coins

                    trending = await asyncio.wait_for(
                        get_trending_coins(),
                        timeout=5.0
                    )
                    # Check if symbol is trending (simplified news proxy)
                    trending_coins = trending[:10] if isinstance(trending, list) else []
                    is_trending = any(
                        coin.get("symbol", "").upper()
                        == s.replace("USDT", "").replace("BTC", "").upper()
                        for coin in trending_coins
                    )
                    news_items = [{"trending": True}] if is_trending else []
                    """
                except asyncio.TimeoutError:
                    logger.debug("trending coins timeout for %s", s)
                    news_items = []
                except Exception:
                    logger.debug("trending coins lookup failed for %s", s)
                    news_items = []

                # expand sentiment/news into arrays aligned to candles
                n = len(candles)
                sentiment_series = [float(sent_score)] * n
                news_series = [0] * n
                nc = len(news_items)
                if n > 0 and nc > 0:
                    step = max(1, n // nc)
                    placed = 0
                    for i in range(0, n, step):
                        if placed >= nc:
                            break
                        news_series[i] = 1
                        placed += 1

                # attach when list-of-dicts
                if isinstance(candles, list):
                    for idx, row in enumerate(candles):
                        try:
                            row["sentiment"] = sentiment_series[idx]
                            row["news_count"] = news_series[idx]
                        except Exception:
                            logger.debug(
                                "failed to attach sentiment/news to candle idx=%s for %s",
                                idx,
                                s,
                            )

                return s, candles

        tasks = [_fetch(s) for s in symbols]
        # CRITICAL FIX: Add total timeout to prevent execution cycle from hanging
        # Even with per-symbol timeouts, gather can hang if too many symbols
        # Increased to 300s to handle 200+ symbols with API rate limits
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300.0  # Maximum 300 seconds for all symbols (was 180)
            )
        except asyncio.TimeoutError:
            logger.error("Total timeout (300s) exceeded for scan_top_by_volume_from_api")
            # Return empty results to allow execution to continue
            return {}
        
        # BULLETPROOF: Filter and validate results
        symbol_ohlcv = {}
        for result in results:
            try:
                if isinstance(result, Exception):
                    logger.debug("Result is exception: %s", result)
                    continue
                if isinstance(result, tuple) and len(result) == 2:
                    s, candles = result
                    if candles and isinstance(candles, (list, dict)) and len(candles) > 0:
                        symbol_ohlcv[s] = candles
            except Exception as e:
                logger.debug("Failed to process result: %s", e)
                continue

        # BULLETPROOF: Ensure we always return something
        try:
            if not symbol_ohlcv:
                logger.warning("No valid symbol data fetched, returning empty dict")
                return {}
            return self.scan_symbols(symbol_ohlcv, top_n=top_n)
        except Exception as e:
            logger.error("CRITICAL: scan_symbols failed: %s", e)
            # Return empty dict instead of crashing
            return {}


def make_default_agent():
    """Create AI agent based on AI_MODEL environment variable.
    
    Options:
    - 'xgb' or 'xgboost': XGBoost only (fast, proven)
    - 'tft': Temporal Fusion Transformer only (temporal patterns)
    - 'hybrid': TFT + XGBoost ensemble (best performance, default)
    
    Returns: Agent instance (XGBAgent, TFTAgent, or HybridAgent)
    """
    import os
    
    model_mode = os.getenv('AI_MODEL', 'hybrid').lower()
    
    if model_mode in ['tft', 'temporal']:
        logger.info("🔮 Using TFT Agent (Temporal Fusion Transformer)")
        from ai_engine.agents.tft_agent import TFTAgent
        return TFTAgent()
    
    elif model_mode in ['hybrid', 'ensemble']:
        logger.info("🤖 Using Hybrid Agent (TFT + XGBoost Ensemble)")
        try:
            from ai_engine.agents.hybrid_agent import HybridAgent
            return HybridAgent()
        except Exception as e:
            logger.warning(f"[WARNING] Hybrid Agent unavailable ({e}), falling back to XGBoost")
            return XGBAgent()
    
    else:  # 'xgb', 'xgboost', or default
        logger.info("[CHART] Using XGBoost Agent")
        return XGBAgent()
