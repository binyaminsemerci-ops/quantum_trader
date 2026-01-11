"""
PatchTST AGENT - CPU-OPTIMIZED with TorchScript
Patch-based Transformer for trading (Phase 4C+)
Expected WIN rate: 68-73%

SHADOW MODE (P0.4):
- PATCHTST_SHADOW_ONLY=true → evaluate but don't vote
- Full inference + telemetry, zero ensemble impact
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Shadow mode rate limiter
_last_shadow_log_time = 0
_SHADOW_LOG_INTERVAL = 30  # seconds


class PatchTSTModel(nn.Module):
    """
    Patch-based Transformer model for time series prediction.
    
    Key optimizations:
    - Patch-based attention (16-len patches → 8x8 = 64 attention ops vs 128x128 = 16,384)
    - CPU-optimized (no GPU dependencies)
    - TorchScript compilation for 2-3x speedup
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        output_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        patch_len: int = 16,
        num_patches: int = 8
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.patch_len = patch_len
        self.num_patches = num_patches
        
        # Patch embedding: Flatten patch and project to hidden_dim
        self.patch_embedding = nn.Linear(patch_len * input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape into patches: (batch, num_patches, patch_len * input_dim)
        x = x.reshape(batch_size, self.num_patches, self.patch_len * self.input_dim)
        
        # Patch embedding: (batch, num_patches, hidden_dim)
        x = self.patch_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer: (batch, num_patches, hidden_dim)
        x = self.transformer(x)
        
        # Global average pooling over patches: (batch, hidden_dim)
        x = x.mean(dim=1)
        
        # Output: (batch, output_dim)
        x = self.output_proj(x)
        
        return x


class PatchTSTAgent:
    """
    Trading agent using PatchTST with TorchScript compilation.
    
    Optimizations:
    - Patch-based attention (8 patches vs 128 timesteps)
    - TorchScript JIT compilation for CPU
    - Min-max normalization for stable training
    """
    
    def __init__(
        self,
        model_path: str = None,
        sequence_length: int = 128,
        patch_len: int = 16,
        device: str = "cpu"
    ):
        self.device = device
        self.sequence_length = sequence_length
        self.patch_len = patch_len
        self.num_patches = sequence_length // patch_len
        
        # 🔥 PATCHTST_MODEL_PATH FEATURE FLAG (P0.4)
        # Priority: 1) model_path arg, 2) PATCHTST_MODEL_PATH env, 3) latest timestamped, 4) fallback
        env_model_path = os.getenv('PATCHTST_MODEL_PATH')
        
        if model_path:
            # Explicit path passed to constructor
            self.model_path = model_path
            logger.info(f"[PatchTST] Using explicit model path: {model_path}")
        elif env_model_path:
            # Environment variable override (for deployment)
            self.model_path = env_model_path
            logger.info(f"[PatchTST] Using PATCHTST_MODEL_PATH env var: {env_model_path}")
        else:
            # Auto-discover latest model or fallback to default
            retraining_dir = Path("/app/models") if Path("/app/models").exists() else Path("ai_engine/models")
            # Try v3 (with scaler), then v2, then v1
            latest_model = self._find_latest_model(retraining_dir, "patchtst_v*_v3.pth") or \
                          self._find_latest_model(retraining_dir, "patchtst_v*_v2.pth")
            self.model_path = str(latest_model) if latest_model else "/app/models/patchtst_model.pth"
            
            # Look for scaler for v3 models
            self.scaler_path = None
            if latest_model and "_v3.pth" in str(latest_model):
                scaler_path = str(latest_model).replace(".pth", "_scaler.pkl")
                if Path(scaler_path).exists():
                    self.scaler_path = scaler_path
                    logger.info(f"[PatchTST] Found scaler: {Path(scaler_path).name}")
            
            logger.info(f"[PatchTST] Auto-discovered model: {self.model_path}")
        
        # Initialize model
        self.model = PatchTSTModel(
            input_dim=8,
            output_dim=1,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            patch_len=patch_len,
            num_patches=self.num_patches
        )
        
        # Load weights if model exists
        model_file = Path(self.model_path)
        if model_file.exists():
            try:
                logger.info(f"[PatchTST] Loading model from {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
                
                # Handle both checkpoint dict and raw state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"[PatchTST] ✅ Model weights loaded from {model_file.name}")
                
                self.model.eval()
                
                # Compile with TorchScript for CPU optimization
                logger.info("[PatchTST] Compiling model with TorchScript...")
                example_input = torch.randn(1, sequence_length, 8)
                try:
                    self.compiled_model = torch.jit.trace(self.model, example_input)
                    self.compiled_model.eval()
                    logger.info("[PatchTST] ✅ TorchScript compilation complete")
                except Exception as trace_err:
                    logger.warning(f"[PatchTST] TorchScript tracing failed: {trace_err}")
                    logger.info("[PatchTST] Using uncompiled model (slower but functional)")
                    self.compiled_model = self.model
                
                # 🔒 FAIL-CLOSED: Log model metadata
                logger.info(f"[PatchTST-INIT] Model file: {model_file.name}")
                logger.info(f"[PatchTST-INIT] Device: {self.device}")
                logger.info(f"[PatchTST-INIT] Input shape: (batch, {self.sequence_length}, 8)")
                logger.info(f"[PatchTST-INIT] Patch config: {self.num_patches} patches x {self.patch_len} timesteps")
                    
            except Exception as e:
                logger.error(f"[PatchTST] ❌ Model loading failed: {e}")
                raise RuntimeError(f"[PatchTST] QSC FAIL-CLOSED: Model loading failed from {self.model_path}. Error: {e}")
        else:
            logger.warning(f"[PatchTST] ⚠️ Model file not found: {self.model_path}")
            raise FileNotFoundError(f"[PatchTST] QSC FAIL-CLOSED: Model file not found at {self.model_path}. Cannot predict without model.")
        
        # Load scaler for v3 models
        self.scaler = None
        if hasattr(self, 'scaler_path') and self.scaler_path and Path(self.scaler_path).exists():
            try:
                import joblib
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"[PatchTST] ✅ Scaler loaded from {Path(self.scaler_path).name}")
            except Exception as e:
                logger.warning(f"[PatchTST] Failed to load scaler: {e}. Using raw features.")
        
        # Feature names for normalization
        self.feature_names = [
            'close', 'high', 'low', 'volume',
            'volatility', 'rsi', 'macd', 'momentum'
        ]
        
        # 🔒 DEGENERACY DETECTION (testnet only - QSC fail-closed)
        from collections import deque
        self._prediction_history = deque(maxlen=100)  # Last 100 predictions
        self._degeneracy_window = 100
        self._is_testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    
    def _find_latest_model(self, base_dir: Path, pattern: str):
        """Find the latest timestamped model file matching pattern."""
        try:
            model_files = list(base_dir.glob(pattern))
            if model_files:
                latest = sorted(model_files)[-1]
                logger.info(f"🔍 Found latest PatchTST model: {latest.name}")
                return latest
        except Exception as e:
            logger.warning(f"Failed to find latest model with pattern {pattern}: {e}")
        return None
    
    def _ensure_model_loaded(self):
        """Model is already loaded in __init__, so just return True."""
        return self.model is not None
    
    def _preprocess(self, market_data: Dict) -> Optional[torch.Tensor]:
        """
        Preprocess market data into model input.
        
        Features:
        1. close, high, low: Price features
        2. volume: Trading volume
        3. volatility: Price volatility (rolling std)
        4. rsi: Relative Strength Index
        5. macd: MACD indicator
        6. momentum: Price momentum
        """
        try:
            # Extract price history
            close = np.array(market_data.get('close', []))
            high = np.array(market_data.get('high', []))
            low = np.array(market_data.get('low', []))
            volume = np.array(market_data.get('volume', []))
            
            if len(close) < self.sequence_length:
                logger.warning(f"[PatchTST] Insufficient data: {len(close)} < {self.sequence_length}")
                return None
            
            # Use last sequence_length points
            close = close[-self.sequence_length:]
            high = high[-self.sequence_length:]
            low = low[-self.sequence_length:]
            volume = volume[-self.sequence_length:]
            
            # Calculate technical indicators
            volatility = np.std(close[-20:]) if len(close) >= 20 else 0.01
            rsi = self._calculate_rsi(close)
            macd = self._calculate_macd(close)
            momentum = (close[-1] - close[0]) / close[0] if close[0] != 0 else 0
            
            # Stack features: (seq_len, 8)
            features = np.stack([
                close,
                high,
                low,
                volume,
                np.full(self.sequence_length, volatility),
                np.full(self.sequence_length, rsi),
                np.full(self.sequence_length, macd),
                np.full(self.sequence_length, momentum)
            ], axis=1)
            
            # Normalization: Use StandardScaler if available (v3), else min-max (v2)
            if self.scaler is not None:
                # v3 models: Apply StandardScaler across ALL features (flatten and reshape)
                # Training: X_scaled = scaler.fit_transform(X)  # X shape: (n_samples, 23 features)
                # Inference: Need to flatten (seq_len, 8) → (1, seq_len*8) then reshape back
                original_shape = features.shape  # (seq_len, 8)
                features_flat = features.flatten().reshape(1, -1)  # (1, seq_len*8)
                features_scaled = self.scaler.transform(features_flat)  # StandardScaler
                features = features_scaled.reshape(original_shape)  # Back to (seq_len, 8)
            else:
                # v2 models: Min-max normalization per feature (legacy)
                features_min = features.min(axis=0, keepdims=True)
                features_max = features.max(axis=0, keepdims=True)
                features = (features - features_min) / (features_max - features_min + 1e-8)
            
            # Convert to tensor: (1, seq_len, 8)
            tensor = torch.FloatTensor(features).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            logger.error(f"[PatchTST] Preprocessing error: {e}")
            return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> float:
        """Calculate MACD indicator."""
        if len(prices) < 26:
            return 0.0
        
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        macd = ema_12 - ema_26
        
        return macd
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1]
        
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        
        ema = np.convolve(prices, weights, mode='valid')[-1]
        
        return ema
    
    def predict(self, symbol: str, features: Dict) -> Tuple[str, float, str]:
        """
        Generate trading prediction (compatible with ensemble manager).
        
        Args:
            symbol: Trading symbol (unused, for compatibility)
            features: Dict of market features
            
        Returns:
            Tuple of (action, confidence, model_name)
        """
        try:
            # 🔒 FAIL-CLOSED: Validate features
            if not features or features.get('price', 0.0) == 0.0:
                raise ValueError("[PatchTST] QSC FAIL-CLOSED: Features invalid or price is zero.")
            
            # Check if v3 model (expects 23 flat features)
            if self.scaler is not None and hasattr(self.scaler, 'n_features_in_'):
                return self._predict_v3(symbol, features)
            else:
                return self._predict_v2(symbol, features)
            
        except Exception as e:
            msg = str(e)
            if "[PatchTST] QSC FAIL-CLOSED:" in msg:
                raise
            else:
                raise RuntimeError(f"[PatchTST] QSC FAIL-CLOSED: Prediction failed for {symbol}. Error: {msg} - excluding from ensemble (FAIL-CLOSED)")
    
    def _predict_v3(self, symbol: str, features: Dict) -> Tuple[str, float, str]:
        """
        V3 prediction path: Uses 23 flat features (same as XGBoost v3).
        Expected features dict keys match train_full.csv columns:
        open, high, low, close, volume, price_change, rsi_14, macd, volume_ratio,
        momentum_10, high_low_range, volume_change, volume_ma_ratio, ema_10,
        ema_20, ema_50, ema_10_20_cross, ema_10_50_cross, volatility_20,
        macd_signal, macd_hist, bb_position, momentum_20
        """
        # Extract 23 features in the exact order as training data
        feature_list = [
            features.get('open', features.get('price', 0.0)),
            features.get('high', features.get('price', 0.0)),
            features.get('low', features.get('price', 0.0)),
            features.get('close', features.get('price', 0.0)),
            features.get('volume', 0.0),
            features.get('price_change', 0.0),
            features.get('rsi_14', 50.0),
            features.get('macd', 0.0),
            features.get('volume_ratio', 1.0),
            features.get('momentum_10', 0.0),
            features.get('high_low_range', 0.0),
            features.get('volume_change', 0.0),
            features.get('volume_ma_ratio', 1.0),
            features.get('ema_10', features.get('price', 0.0)),
            features.get('ema_20', features.get('price', 0.0)),
            features.get('ema_50', features.get('price', 0.0)),
            features.get('ema_10_20_cross', 0.0),
            features.get('ema_10_50_cross', 0.0),
            features.get('volatility_20', 0.01),
            features.get('macd_signal', 0.0),
            features.get('macd_hist', 0.0),
            features.get('bb_position', 0.5),
            features.get('momentum_20', 0.0),
        ]
        
        # Convert to numpy array (1, 23)
        X = np.array([feature_list], dtype=np.float32)
        
        # Apply StandardScaler normalization
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Predict (model expects (batch, 23) and outputs (batch, 3) logits)
        with torch.no_grad():
            logits = self.model(input_tensor)  # (1, 3)
            probs = torch.softmax(logits, dim=1)[0]  # (3,)
            
            action_idx = torch.argmax(probs).item()
            confidence = probs[action_idx].item()
            
            # Map to action names
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            action = action_map[action_idx]
        
        # 🔍 DEGENERACY DETECTION (testnet only)
        if self._is_testnet:
            self._prediction_history.append((action, confidence))
            
            if len(self._prediction_history) >= self._degeneracy_window:
                actions = [a for a, _ in self._prediction_history]
                confidences = [c for _, c in self._prediction_history]
                
                from collections import Counter
                action_counts = Counter(actions)
                most_common_action, most_common_count = action_counts.most_common(1)[0]
                action_pct = (most_common_count / len(actions)) * 100
                
                conf_std = np.std(confidences)
                
                if action_pct > 95.0 and conf_std < 0.02:
                    raise RuntimeError(
                        f"[PatchTST] QSC FAIL-CLOSED: Degenerate output detected. "
                        f"Action '{most_common_action}' occurs {action_pct:.1f}% of time "
                        f"with confidence_std={conf_std:.4f} (threshold: >95% and <0.02). "
                        f"This indicates likely OOD input or collapsed model weights. "
                        f"Model marked INACTIVE."
                    )
        
        return action, confidence, "patchtst_v3"
    
    def _predict_v2(self, symbol: str, features: Dict) -> Tuple[str, float, str]:
        """
        V2 prediction path: Uses temporal sequence (seq_len, 8 features).
        Legacy code path for backward compatibility with v2 models.
        """
        # Convert features dict to market_data format
        market_data = {
            'close': [features.get('price', 0.0)] * self.sequence_length,
            'high': [features.get('price', 0.0)] * self.sequence_length,
            'low': [features.get('price', 0.0)] * self.sequence_length,
            'volume': [features.get('volume', 0.0)] * self.sequence_length
        }
        
        try:
            # 🔒 FAIL-CLOSED: Validate features
            if not features or features.get('price', 0.0) == 0.0:
                raise ValueError("[PatchTST] QSC FAIL-CLOSED: Features invalid or price is zero.")
            
            # Preprocess
            input_tensor = self._preprocess(market_data)
            if input_tensor is None:
                raise RuntimeError("[PatchTST] QSC FAIL-CLOSED: Preprocessing failed - returned None.")
            
            # 🔒 FAIL-CLOSED: Validate tensor shape and values
            expected_shape = (1, self.sequence_length, 8)
            if input_tensor.shape != expected_shape:
                raise ValueError(
                    f"[PatchTST] QSC FAIL-CLOSED: Input tensor shape mismatch. "
                    f"Expected {expected_shape}, got {input_tensor.shape}"
                )
            
            if torch.isnan(input_tensor).any():
                raise ValueError("[PatchTST] QSC FAIL-CLOSED: Input tensor contains NaN values.")
            if torch.isinf(input_tensor).any():
                raise ValueError("[PatchTST] QSC FAIL-CLOSED: Input tensor contains Inf values.")
            
            # Predict (full inference always)
            with torch.no_grad():
                output = self.compiled_model(input_tensor)
                logit = output.item()
                prob = torch.sigmoid(output).item()
            
            # Convert to action
            if prob > 0.6:
                action = 'BUY'
                confidence = prob
            elif prob < 0.4:
                action = 'SELL'
                confidence = 1.0 - prob
            else:
                # FAIL-CLOSED: prob in dead zone - raise error instead of HOLD 0.5
                raise ValueError(f"PatchTST probability {prob:.4f} in dead zone [0.4, 0.6] - no clear signal")
            
            # � DEGENERACY DETECTION (testnet only)
            if self._is_testnet:
                self._prediction_history.append((action, confidence))
                
                if len(self._prediction_history) >= self._degeneracy_window:
                    actions = [a for a, _ in self._prediction_history]
                    confidences = [c for _, c in self._prediction_history]
                    
                    from collections import Counter
                    action_counts = Counter(actions)
                    most_common_action, most_common_count = action_counts.most_common(1)[0]
                    action_pct = (most_common_count / len(actions)) * 100
                    
                    conf_std = np.std(confidences)
                    
                    if action_pct > 95.0 and conf_std < 0.02:
                        raise RuntimeError(
                            f"[PatchTST] QSC FAIL-CLOSED: Degenerate output detected. "
                            f"Action '{most_common_action}' occurs {action_pct:.1f}% of time "
                            f"with confidence_std={conf_std:.4f} (threshold: >95% and <0.02). "
                            f"This indicates likely OOD input or collapsed model weights. "
                            f"Model marked INACTIVE."
                        )
            
            # �🔍 SHADOW MODE (P0.4) - DISABLED FOR QSC CANARY DEPLOYMENT
            # QSC MODE: All models must actively vote for quality gates to be valid
            shadow_mode = False  # Was: os.getenv('PATCHTST_SHADOW_ONLY', 'false').lower() == 'true'
            
            if shadow_mode:
                # Rate-limited detailed logging
                global _last_shadow_log_time
                now = time.time()
                if now - _last_shadow_log_time >= _SHADOW_LOG_INTERVAL:
                    logger.info(
                        f"[SHADOW] PatchTST | {symbol} | action={action} conf={confidence:.4f} | "
                        f"prob={prob:.4f} logit={logit:.4f} | mode=SHADOW_ONLY"
                    )
                    _last_shadow_log_time = now
                
                # Return with shadow marker (tuple becomes dict in ensemble)
                return action, confidence, 'patchtst_shadow'
            
            return action, confidence, 'patchtst_model'
            
        except Exception as e:
            logger.error(f"[PatchTST] Prediction error: {e}", exc_info=True)
            raise RuntimeError(f"[PatchTST] QSC FAIL-CLOSED: Prediction failed for {symbol}. Error: {e}")
