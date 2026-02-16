"""
PatchTST AGENT v3 - 49-Feature Support (Feb 2026)
Supports both v3 (49-feature SimplePatchTST) and legacy (23/8-feature) architectures.
Prioritizes v3 models with automatic scaler loading and metadata-driven feature extraction.
Updated to match LightGBM/XGBoost/N-HiTS feature engineering uniformity.
"""
import os
import json
import joblib
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SimplePatchTST(nn.Module):
    """
    SimplePatchTST for v3 models (49 features - Feb 2026 update).
    Matches training architecture from retrain_patchtst_v3.py
    """
    def __init__(self, num_features=49, d_model=128, num_heads=4, num_layers=2, num_classes=3):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_features)
        Returns:
            (batch, num_classes)
        """
        x = self.input_proj(x).unsqueeze(1)  # (batch, 1, d_model)
        x = self.encoder(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        x = self.head(x)  # (batch, num_classes)
        return x


class PatchTSTAgent:
    """
    Trading agent with v3 model support (23 features + scaler).
    """
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self.scaler = None
        self.features = []
        self.model_path = None
        
        # Discover latest model (v3 > v2)
        if model_path:
            self.model_path = model_path
        else:
            retraining_dir = Path("/app/models") if Path("/app/models").exists() else Path("ai_engine/models")
            self.model_path = self._find_latest_model(retraining_dir)
        
        if self.model_path:
            logger.info(f"[PatchTST] Found latest model: {self.model_path}")
        else:
            logger.warning("[PatchTST] No model found!")
    
    def _find_latest_model(self, base_dir: Path) -> Optional[str]:
        """Find latest model (v3 prioritized)."""
        try:
            all_files = list(base_dir.glob("patchtst_v*.pth"))
            if not all_files:
                return None
            
            v3_models = [f for f in all_files if f.stem.endswith("_v3")]
            v2_models = [f for f in all_files if f.stem.endswith("_v2")]
            
            candidates = v3_models or v2_models
            if candidates:
                latest = sorted(candidates)[-1]
                logger.info(f"[PatchTST] Found latest model: {latest.name}")
                return str(latest)
        except Exception as e:
            logger.warning(f"[PatchTST] Model search failed: {e}")
        return None
    
    def _ensure_model_loaded(self):
        """Load model + scaler + metadata."""
        if self.model is not None:
            return
        
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(f"[PatchTST] Model file not found: {self.model_path}")
        
        try:
            model_path = Path(self.model_path)
            
            # Load metadata to get feature list
            meta_path = model_path.with_name(model_path.stem + "_meta.json")
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.features = meta.get("features", [])
                num_features = len(self.features) or 49
                logger.info(f"[PatchTST] Loaded feature schema ({num_features} features).")
            else:
                logger.warning("[PatchTST] No metadata found, using 49-feature default (Feb 2026 schema).")
                num_features = 49
                # Complete 49-feature schema (matches LightGBM/XGBoost/N-HiTS)
                self.features = [
                    'returns', 'log_returns', 'price_range', 'body_size', 'upper_wick', 'lower_wick',
                    'is_doji', 'is_hammer', 'is_engulfing', 'gap_up', 'gap_down',
                    'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d', 'roc',
                    'ema_9', 'ema_9_dist', 'ema_21', 'ema_21_dist', 'ema_50', 'ema_50_dist', 'ema_200', 'ema_200_dist',
                    'sma_20', 'sma_50',
                    'adx', 'plus_di', 'minus_di',
                    'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                    'atr', 'atr_pct', 'volatility',
                    'volume_sma', 'volume_ratio', 'obv', 'obv_ema', 'vpt',
                    'momentum_5', 'momentum_10', 'momentum_20', 'acceleration', 'relative_spread'
                ]
            
            # Initialize SimplePatchTST with correct num_features
            logger.info(f"[PatchTST] Initializing model with {num_features} features.")
            self.model = SimplePatchTST(num_features=num_features)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                # Check scaler/model dimension match (bypass if mismatch)
                if hasattr(self.scaler, 'n_features_in_'):
                    expected_dim = self.scaler.n_features_in_
                    if expected_dim != num_features:
                        logger.warning(
                            f"[PatchTST] Scaler dimension mismatch: "
                            f"scaler expects {expected_dim}, model has {num_features}. "
                            f"Bypassing scaler (graceful degradation)."
                        )
                        self.scaler = None
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            logger.info(f"[PatchTST] ✅ Model loaded successfully: {model_path.name}")
            
            # Load scaler
            scaler_path = model_path.with_name(model_path.stem + "_scaler.pkl")
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"[PatchTST] ✅ Scaler loaded: {scaler_path.name}")
                # Bypass scaler if dimension mismatch
                expected_dim = self.scaler.n_features_in_
                if expected_dim != num_features:
                    logger.warning(f"[PatchTST] Scaler expects {expected_dim} features but got {num_features}. Bypassing scaler.")
                    self.scaler = None
            else:
                logger.warning("[PatchTST] No scaler found")
                self.scaler = None

        except Exception as e:
            logger.error(f"[PatchTST] Model loading failed: {e}")
            raise RuntimeError(f"[PatchTST] Failed to load model: {e}")
            
    def predict(self, symbol: str, features: Dict) -> Tuple[str, float, str]:
        """
        Generate trading prediction.
        
        Args:
            symbol: Trading symbol
            features: Dict of market features
            
        Returns:
            Tuple of (action, confidence, model_name)
        """
        try:
            self._ensure_model_loaded()
            
            # Extract features in correct order
            if self.features:
                feature_keys = self.features
            else:
                # Fallback: use sorted keys from features dict
                feature_keys = sorted(features.keys())
            
            # Build feature vector
            vector = np.array([[features.get(k, 0.0) for k in feature_keys]])
            
            # Apply scaler if loaded
            if self.scaler is not None:
                vector = self.scaler.transform(vector)
            
            # Convert to tensor
            x = torch.tensor(vector, dtype=torch.float32, device=self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Map to action
            action_map = ["sell", "hold", "buy"]
            action_idx = np.argmax(probs)
            action = action_map[action_idx]
            confidence = float(probs[action_idx])
            
            return action, confidence, "patchtst_model_v3"
        
        except Exception as e:
            logger.error(f"[PatchTST] Prediction error: {e}", exc_info=True)
            raise RuntimeError(f"[PatchTST] Prediction failed for {symbol}: {e}")
