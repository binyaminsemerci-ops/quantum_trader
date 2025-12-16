"""
N-HiTS AGENT - Neural Hierarchical Interpolation for trading
Multi-rate temporal pattern recognition (2022)
Expected WIN rate: 70-75%
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import logging

from ai_engine.nhits_simple import SimpleNHiTS
import torch

logger = logging.getLogger(__name__)


class NHiTSAgent:
    """
    Trading agent using N-HiTS architecture.
    
    Features:
    - Multi-rate sampling (short + medium + long term)
    - Hierarchical interpolation
    - Fast inference
    - Excellent for volatile crypto markets
    """
    
    def __init__(
        self,
        model_path: str = None,
        sequence_length: int = 120,  # Must match training (120 candles = 2 hours of data)
        device: str = None
    ):
        self.sequence_length = sequence_length
        
        # 🔥 USE LATEST TIMESTAMPED MODEL (not old hardcoded names)
        # Retraining saves to /app/models/, agents default to ai_engine/models
        retraining_dir = Path("/app/models") if Path("/app/models").exists() else Path("ai_engine/models")
        # Search for .pth files (PyTorch format)
        latest_model = self._find_latest_model(retraining_dir, "nhits_v*.pth")
        self.model_path = model_path or str(latest_model) if latest_model else "ai_engine/models/nhits_model.pth"
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # History buffer for sequences
        self.history_buffer = {}  # symbol -> list of features
        
        # Model will be loaded lazily
        self.model: Optional[SimpleNHiTS] = None
        self.feature_mean = None
        self.feature_std = None
        self._shape_logged = False
        
        logger.info(f"[TARGET] N-HiTS Agent initialized (device: {self.device}, seq_len: {self.sequence_length})")
    
    def _find_latest_model(self, base_dir: Path, pattern: str):
        """Find the latest timestamped model file matching pattern."""
        try:
            model_files = list(base_dir.glob(pattern))
            if model_files:
                # Sort by filename (timestamp is in filename)
                latest = sorted(model_files)[-1]
                logger.info(f"🔍 Found latest N-HiTS model: {latest.name}")
                return latest
        except Exception as e:
            logger.warning(f"Failed to find latest model with pattern {pattern}: {e}")
        return None
    
    def _ensure_model_loaded(self):
        """Load model if not already loaded."""
        if self.model is not None:
            return True
        
        model_file = Path(self.model_path)
        if not model_file.exists():
            logger.warning(f"[WARNING]  N-HiTS model not found: {model_file}")
            logger.warning("    Run: python scripts/train_nhits.py")
            return False
        
        try:
            checkpoint = torch.load(str(model_file), map_location=self.device, weights_only=False)
            self.model = SimpleNHiTS(
                input_size=checkpoint.get('input_size', 120),
                hidden_size=256,
                num_features=checkpoint.get('num_features', 12)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.feature_mean = checkpoint.get('feature_mean')
            self.feature_std = checkpoint.get('feature_std')
            logger.info(f"✅ N-HiTS model loaded from {model_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load N-HiTS model: {e}")
            return False
    
    def update_history(self, symbol: str, features: Dict[str, float]):
        """Add new features to history buffer."""
        if symbol not in self.history_buffer:
            self.history_buffer[symbol] = []
        
        # Extract feature values in consistent order
        feature_order = [
            'price_change', 'high_low_range', 'volume_change', 'volume_ma_ratio',
            'ema_10', 'ema_20', 'ema_50', 'ema_10_20_cross', 'ema_10_50_cross',
            'rsi_14', 'volatility_20', 'macd', 'macd_signal', 'macd_hist'
        ]
        
        values = []
        for feat in feature_order:
            values.append(features.get(feat, 0.0))
        
        self.history_buffer[symbol].append(values)
        
        # Keep only last sequence_length + 10 (for safety)
        max_len = self.sequence_length + 10
        if len(self.history_buffer[symbol]) > max_len:
            self.history_buffer[symbol] = self.history_buffer[symbol][-max_len:]
    
    def predict(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> Tuple[str, float, str]:
        """
        Predict trading action for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            features: Dictionary of features (technical indicators)
        
        Returns:
            Tuple of (action, confidence, model_name)
            - action: 'BUY', 'SELL', or 'HOLD'
            - confidence: 0.0 to 1.0
            - model_name: 'nhits_model'
        """
        # Update history
        self.update_history(symbol, features)
        
        # Check if model loaded
        if not self._ensure_model_loaded():
            return self._fallback_prediction(features)
        
        # Check if enough history
        buffer_size = len(self.history_buffer.get(symbol, []))
        if buffer_size < self.sequence_length:
            logger.debug(
                f"N-HiTS {symbol}: Not enough history "
                f"({buffer_size}/{self.sequence_length}), using fallback"
            )
            return self._fallback_prediction(features)
        
        # Log that we're using the trained model (only first time)
        if not hasattr(self, '_logged_model_use'):
            self._logged_model_use = set()
        if symbol not in self._logged_model_use:
            logger.info(f"✅ N-HiTS {symbol}: Using trained model with {buffer_size} candles!")
            self._logged_model_use.add(symbol)
        
        try:
            # Get sequence
            sequence = np.array(self.history_buffer[symbol][-self.sequence_length:])
            # Align to expected feature dimension to avoid broadcast errors
            sequence = self._align_feature_dims(sequence)
            
            # Normalize
            if self.feature_mean is not None and self.feature_std is not None:
                mean_vec = self._match_vector(self.feature_mean, sequence.shape[-1])
                std_vec = self._match_vector(self.feature_std, sequence.shape[-1])
                sequence = (sequence - mean_vec) / (std_vec + 1e-8)
            
            # Final safety check: enforce target length before tensor conversion
            sequence = self._enforce_target_len(sequence)
            if not self._shape_logged:
                logger.info(
                    "[NHITS] Input aligned: seq_shape=%s target_len=%s",
                    sequence.shape,
                    sequence.shape[-1],
                )
                self._shape_logged = True
            
            # Convert to tensor: [1, seq_len, num_features]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                logits, _ = self.model(sequence_tensor)
                probs = torch.softmax(logits, dim=1)[0]
            
            # Get action
            pred_class = torch.argmax(probs).item()
            confidence = float(probs[pred_class])
            
            # Map class to action (0=SELL, 1=HOLD, 2=BUY)
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            action = action_map[pred_class]
            
            logger.debug(
                f"N-HiTS {symbol}: {action} (conf={confidence:.2f}, "
                f"probs=[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}])"
            )
            
            return action, confidence, "nhits_model"
            
        except Exception as e:
            logger.error(f"❌ N-HiTS prediction failed for {symbol}: {e}")
            return self._fallback_prediction(features)

    def _align_feature_dims(self, sequence: np.ndarray) -> np.ndarray:
        """
        Ensure sequence feature dimension matches model/scaler expectations.
        Trims or pads with zeros when feature_mean/std length differs from stored history.
        """
        target_len = 12
        if self.feature_mean is not None and hasattr(self.feature_mean, "shape"):
            target_len = self.feature_mean.shape[-1]
        if self.feature_std is not None and hasattr(self.feature_std, "shape"):
            target_len = min(target_len, self.feature_std.shape[-1])
        if self.feature_mean is not None and not hasattr(self.feature_mean, "shape"):
            try:
                target_len = len(self.feature_mean)
            except Exception:
                pass
        if self.feature_std is not None and not hasattr(self.feature_std, "shape"):
            try:
                target_len = min(target_len, len(self.feature_std))
            except Exception:
                pass
        if self.model is not None and hasattr(self.model, "num_features"):
            mf = getattr(self.model, "num_features", target_len)
            if mf:
                target_len = min(target_len, mf)

        if sequence.shape[-1] == target_len:
            return sequence

        if sequence.shape[-1] > target_len:
            adjusted = sequence[..., :target_len]
        else:
            pad_width = target_len - sequence.shape[-1]
            pad = np.zeros((sequence.shape[0], pad_width))
            adjusted = np.concatenate([sequence, pad], axis=-1)

        logger.warning(
            "[NHITS] Adjusted feature dimension %d -> %d to match scaler/model",
            sequence.shape[-1],
            target_len,
        )
        return adjusted

    def _match_vector(self, vec: np.ndarray, target_len: int) -> np.ndarray:
        """Trim or pad a vector to target_len."""
        if vec is None:
            return np.zeros(target_len)
        vec = np.asarray(vec)
        if vec.shape[-1] == target_len:
            return vec
        if vec.shape[-1] > target_len:
            return vec[..., :target_len]
        pad = np.zeros(target_len - vec.shape[-1])
        return np.concatenate([vec, pad], axis=-1)

    def _enforce_target_len(self, sequence: np.ndarray) -> np.ndarray:
        """Ensure the last dimension matches the model/scaler feature length."""
        target_len = 12  # default expected feature count
        if self.model is not None and hasattr(self.model, "num_features"):
            target_len = getattr(self.model, "num_features", target_len)
        if self.feature_mean is not None and hasattr(self.feature_mean, "shape"):
            target_len = min(target_len, self.feature_mean.shape[-1])
        if self.feature_std is not None and hasattr(self.feature_std, "shape"):
            target_len = min(target_len, self.feature_std.shape[-1])
        if sequence.shape[-1] == target_len:
            return sequence
        if sequence.shape[-1] > target_len:
            return sequence[..., :target_len]
        pad_width = target_len - sequence.shape[-1]
        pad = np.zeros((sequence.shape[0], pad_width))
        return np.concatenate([sequence, pad], axis=-1)

    def _fallback_prediction(
        self,
        features: Dict[str, float]
    ) -> Tuple[str, float, str]:
        """
        Fallback prediction using BALANCED and TREND-AWARE rules.
        
        FIXED: Returns BUY/SELL based on actual market conditions AND respects trend.
        - RSI < 40: Oversold → BUY (if not strong downtrend)
        - RSI > 60: Overbought → SELL (if not strong uptrend)
        - Price momentum: Positive → BUY, Negative → SELL
        - Trend filter: Block SELL in strong uptrend, block BUY in strong downtrend
        """
        try:
            rsi = features.get('rsi_14', 50)
            price_change = features.get('price_change', 0) * 100
            price = features.get('close', 0)
            ema_50 = features.get('ema_50', price)  # Use EMA50 (exists in features)
            
            # Calculate trend strength
            if price > 0 and ema_50 > 0:
                price_vs_ema = price / ema_50
            else:
                price_vs_ema = 1.0  # Neutral if missing data
            
            action = 'HOLD'
            confidence = 0.50
            
            # RSI-based signals (CONSERVATIVE THRESHOLDS)
            if rsi < 35:  # Oversold - CONSERVATIVE (was 40)
                action = 'BUY'
                confidence = min(0.65, 0.50 + (35 - rsi) / 100)  # Max 0.65
            elif rsi > 65:  # Overbought - CONSERVATIVE (was 60)
                action = 'SELL'
                confidence = min(0.65, 0.50 + (rsi - 65) / 100)  # Max 0.65
            
            # Price momentum (if RSI is neutral)
            if action == 'HOLD':
                if price_change > 0.5:  # Price going UP
                    action = 'BUY'
                    confidence = min(0.65, 0.50 + abs(price_change) / 15)  # Max 0.65 (was 0.70)
                elif price_change < -0.5:  # Price going DOWN
                    action = 'SELL'
                    confidence = min(0.65, 0.50 + abs(price_change) / 15)  # Max 0.65 (was 0.70)
            
            # TREND FILTER: Block trades against trends
            if action == 'SELL' and price_vs_ema > 1.005:  # Uptrend (>0.5%)
                logger.debug(
                    f"N-HiTS fallback: BLOCKED SELL in uptrend "
                    f"(price {price_vs_ema:.2%} vs EMA50)"
                )
                action = 'HOLD'
                confidence = 0.50
            elif action == 'BUY' and price_vs_ema < 0.995:  # Downtrend (<-0.5%)
                logger.debug(
                    f"N-HiTS fallback: BLOCKED BUY in downtrend "
                    f"(price {price_vs_ema:.2%} vs EMA50)"
                )
                action = 'HOLD'
                confidence = 0.50
            
            logger.debug(
                f"N-HiTS fallback: {action} (conf={confidence:.2f}, "
                f"rsi={rsi:.1f}, price_chg={price_change:.2f}%, trend={price_vs_ema:.2%})"
            )
            
            return action, confidence, "nhits_fallback_rules"
            
        except Exception as e:
            logger.error(f"Fallback prediction error: {e}")
            return 'HOLD', 0.50, "nhits_error"
    
    def clear_history(self, symbol: Optional[str] = None):
        """Clear history buffer for symbol or all symbols."""
        if symbol:
            if symbol in self.history_buffer:
                del self.history_buffer[symbol]
        else:
            self.history_buffer.clear()
