"""
PatchTST AGENT v3 / v6 - 49-Feature Support (Feb 2026)
=======================================================
v3 (SimplePatchTST): single candle input — legacy, kept for backward compat
v6 (SequencePatchTST): 30-candle sequences -> 6 patches -> proper temporal learning
  - Detected automatically from checkpoint key 'seq_len'
  - Per-sequence z-score normalization (no global scaler -> no OOD)
  - Requires 30-candle history buffer before predicting (same pattern as NHiTS)
"""
import os
import json
import joblib
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


class SimplePatchTST(nn.Module):
    """v3 legacy: single-candle transformer (kept for backward compat)."""
    def __init__(self, num_features=49, d_model=128, num_heads=4, num_layers=2, num_classes=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.dropout    = nn.Dropout(dropout)
        self.encoder    = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                       dim_feedforward=d_model*2, dropout=dropout, batch_first=True),
            num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.encoder(self.dropout(x)).squeeze(1)
        return self.head(x)


class SequencePatchTST(nn.Module):
    """
    v6: True patch-based Transformer.
    Input:  (batch, seq_len, n_features)
    Patches: seq_len // patch_size tokens -> Transformer -> GlobalAvgPool -> classify
    Caller applies per-sequence z-score normalisation before passing (no global scaler).
    """
    def __init__(self, n_features=49, patch_size=5, d_model=128, num_heads=4,
                 num_layers=3, dim_ff=256, dropout=0.15, num_classes=3):
        super().__init__()
        self.patch_size  = patch_size
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_size * n_features, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_emb = nn.Embedding(64, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                       dim_feedforward=dim_ff, dropout=dropout,
                                       batch_first=True, norm_first=True),
            num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        n_patches = T // self.patch_size
        x = x.reshape(B, n_patches, self.patch_size * F)
        x = self.patch_embed(x)
        x = x + self.pos_emb(torch.arange(n_patches, device=x.device)).unsqueeze(0)
        x = self.encoder(self.dropout(x))
        return self.head(x.mean(dim=1))


class PatchTSTAgent:
    """
    Trading agent supporting:
      - v3 SimplePatchTST (single candle, legacy)
      - v6 SequencePatchTST (30-candle history buffer, per-sequence z-score norm)
    Model version is auto-detected from checkpoint 'seq_len' key.
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device     = device
        self.model      = None
        self.scaler     = None
        self.features: List[str] = []
        self.model_path = None
        self.is_v6      = False    # True if SequencePatchTST loaded
        self.seq_len    = 1        # 1 for v3, 30 for v6
        self.patch_size = 5
        self.history_buffer: Dict[str, deque] = {}

        if model_path:
            self.model_path = model_path
        else:
            base_dir = Path("/app/models") if Path("/app/models").exists() else Path("ai_engine/models")
            self.model_path = self._find_latest_model(base_dir)

        if self.model_path:
            logger.info(f"[PatchTST] Found model: {Path(self.model_path).name}")
        else:
            logger.warning("[PatchTST] No model found!")

    def _find_latest_model(self, base_dir: Path) -> Optional[str]:
        """Prefer v6 > v3 > v2 by version, then by timestamp."""
        try:
            all_files = list(base_dir.glob("patchtst_v*.pth"))
            if not all_files:
                return None
            v6 = sorted([f for f in all_files if "patchtst_v6_" in f.name])
            v3 = sorted([f for f in all_files if f.stem.endswith("_v3") or "patchtst_v3_" in f.name])
            v2 = sorted([f for f in all_files if f.stem.endswith("_v2") or "patchtst_v2_" in f.name])
            candidates = v6 or v3 or v2 or sorted(all_files)
            latest = candidates[-1]
            logger.info(f"[PatchTST] Autodiscovered: {latest.name}")
            return str(latest)
        except Exception as e:
            logger.warning(f"[PatchTST] Model search error: {e}")
        return None
    
    _DEFAULT_FEATURES = [
        'returns','log_returns','price_range','body_size','upper_wick','lower_wick',
        'is_doji','is_hammer','is_engulfing','gap_up','gap_down',
        'rsi','macd','macd_signal','macd_hist','stoch_k','stoch_d','roc',
        'ema_9','ema_9_dist','ema_21','ema_21_dist','ema_50','ema_50_dist','ema_200','ema_200_dist',
        'sma_20','sma_50','adx','plus_di','minus_di',
        'bb_middle','bb_upper','bb_lower','bb_width','bb_position',
        'atr','atr_pct','volatility','volume_sma','volume_ratio',
        'obv','obv_ema','vpt','momentum_5','momentum_10','momentum_20',
        'acceleration','relative_spread',
    ]

    def _ensure_model_loaded(self):
        """Load model + scaler + detect v3 vs v6."""
        if self.model is not None:
            return

        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(f"[PatchTST] Model not found: {self.model_path}")

        try:
            model_path = Path(self.model_path)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            cfg = checkpoint.get("model_config", {})

            # Feature list (checkpoint > metadata file > default 49)
            self.features = checkpoint.get("features", [])
            if not self.features:
                meta_path = model_path.with_name(model_path.stem + "_metadata.json")
                if not meta_path.exists():
                    meta_path = model_path.with_name(model_path.stem + "_meta.json")
                if meta_path.exists():
                    with open(meta_path) as f:
                        self.features = json.load(f).get("features", [])
            if not self.features:
                self.features = list(self._DEFAULT_FEATURES)
            num_features = len(self.features)

            # Detect v6 by 'seq_len' in checkpoint
            stored_seq_len = int(checkpoint.get("seq_len", cfg.get("seq_len", 0)))
            if stored_seq_len > 1:
                # ── v6: SequencePatchTST ──────────────────────────────────
                self.is_v6      = True
                self.seq_len    = stored_seq_len
                self.patch_size = int(checkpoint.get("patch_size", cfg.get("patch_size", 5)))
                self.model = SequencePatchTST(
                    n_features  = cfg.get("n_features",  num_features),
                    patch_size  = cfg.get("patch_size",  self.patch_size),
                    d_model     = cfg.get("d_model",     128),
                    num_heads   = cfg.get("num_heads",   4),
                    num_layers  = cfg.get("num_layers",  3),
                    dim_ff      = cfg.get("dim_ff",      256),
                    dropout     = cfg.get("dropout",     0.15),
                    num_classes = cfg.get("num_classes", 3),
                )
                self.scaler = None  # per-sequence z-score — no global scaler needed
                logger.info(f"[PatchTST-v6] SequencePatchTST: seq_len={self.seq_len} patch={self.patch_size}")
            else:
                # ── v3: SimplePatchTST (legacy single-candle) ────────────
                self.is_v6   = False
                self.seq_len = 1
                self.model   = SimplePatchTST(
                    num_features = num_features,
                    d_model      = cfg.get("d_model",    128),
                    num_heads    = cfg.get("num_heads",  4),
                    num_layers   = cfg.get("num_layers", 2),
                )
                scaler_path = model_path.with_name(model_path.stem + "_scaler.pkl")
                if scaler_path.exists():
                    sc = joblib.load(scaler_path)
                    if getattr(sc, "n_features_in_", num_features) == num_features:
                        self.scaler = sc
                        logger.info(f"[PatchTST-v3] Scaler loaded: {scaler_path.name}")
                    else:
                        logger.warning("[PatchTST-v3] Scaler dimension mismatch — bypassing")
                        self.scaler = None
                else:
                    self.scaler = None
                logger.info(f"[PatchTST-v3] SimplePatchTST (legacy): {num_features} features")

            sd = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(sd, strict=True)
            self.model.eval()
            logger.info(f"[PatchTST] Loaded: {model_path.name}  is_v6={self.is_v6}")

        except Exception as e:
            logger.error(f"[PatchTST] Load failed: {e}", exc_info=True)
            raise RuntimeError(f"[PatchTST] Failed to load: {e}")

    # ── History helpers for v6 ────────────────────────────────────────────────

    def _update_history(self, symbol: str, feat_vec: np.ndarray):
        """Append feat_vec to symbol's rolling history buffer."""
        if symbol not in self.history_buffer:
            self.history_buffer[symbol] = deque(maxlen=self.seq_len + 10)
        self.history_buffer[symbol].append(feat_vec.copy())

    def _get_sequence_tensor(self, symbol: str) -> Optional[torch.Tensor]:
        """
        Build normalised (1, seq_len, n_features) tensor for v6.
        Returns None if not enough history.
        Normalisation: per-sequence z-score + ±5 clip (eliminates OOD).
        """
        buf = list(self.history_buffer.get(symbol, []))
        if len(buf) < self.seq_len:
            return None
        seq = np.array(buf[-self.seq_len:], dtype=np.float32)   # (seq_len, n_features)
        m = seq.mean(axis=0, keepdims=True)
        s = seq.std( axis=0, keepdims=True) + 1e-6
        seq = np.clip((seq - m) / s, -5.0, 5.0)
        return torch.FloatTensor(seq).unsqueeze(0).to(self.device)  # (1, seq_len, n_features)
            
    def predict(self, symbol: str, features: Dict) -> Dict:
        """
        Generate trading prediction.
        v3: single candle -> instant prediction
        v6: accumulates 30-candle history, then uses SequencePatchTST
        """
        try:
            self._ensure_model_loaded()

            # Build ordered feature vector (49 values)
            feat_keys = self.features or self._DEFAULT_FEATURES
            feat_vec  = np.array([features.get(k, 0.0) for k in feat_keys], dtype=np.float32)

            # ── v6: sequence path ─────────────────────────────────────────────
            if self.is_v6:
                self._update_history(symbol, feat_vec)
                buf_size = len(self.history_buffer.get(symbol, []))
                if buf_size < self.seq_len:
                    logger.debug(f"[PatchTST-v6] {symbol}: filling history ({buf_size}/{self.seq_len}), fallback")
                    return self._fallback_prediction(symbol, features)

                x = self._get_sequence_tensor(symbol)
                if x is None:
                    return self._fallback_prediction(symbol, features)

                with torch.no_grad():
                    logits = self.model(x)
                    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

                action_map = ["SELL", "HOLD", "BUY"]
                idx        = int(np.argmax(probs))
                action     = action_map[idx]
                confidence = float(probs[idx])
                logger.info(
                    f"PatchTST {symbol}: {action} (conf={confidence:.3f}, "
                    f"probs=[{probs[0]:.2f},{probs[1]:.2f},{probs[2]:.2f}])"
                )
                return {"action": action, "confidence": confidence, "model": "patchtst_v6"}

            # ── v3: single-candle path (legacy) ──────────────────────────────
            vector = feat_vec.reshape(1, -1)
            if self.scaler is not None:
                vector = self.scaler.transform(vector)
            x = torch.tensor(vector, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

            action_map = ["SELL", "HOLD", "BUY"]
            idx        = int(np.argmax(probs))
            action     = action_map[idx]
            confidence = float(probs[idx])
            logger.info(
                f"PatchTST {symbol}: {action} (conf={confidence:.3f}, "
                f"probs=[{probs[0]:.2f},{probs[1]:.2f},{probs[2]:.2f}])"
            )
            return {"action": action, "confidence": confidence, "model": "patchtst_model_v3"}

        except Exception as e:
            logger.error(f"[PatchTST] Prediction error for {symbol}: {e}", exc_info=True)
            raise RuntimeError(f"[PatchTST] Prediction failed: {e}")

    def _fallback_prediction(self, symbol: str, features: Dict) -> Dict:
        """Simple RSI/momentum fallback while history buffer fills (v6 warmup only)."""
        rsi    = float(features.get('rsi', 50))
        ret    = float(features.get('returns', 0)) * 100
        action = 'HOLD'; conf = 0.50
        if rsi < 35:
            action = 'BUY';  conf = min(0.65, 0.50 + (35 - rsi) / 100)
        elif rsi > 65:
            action = 'SELL'; conf = min(0.65, 0.50 + (rsi - 65) / 100)
        elif ret >  0.5:
            action = 'BUY';  conf = min(0.65, 0.50 + abs(ret) / 15)
        elif ret < -0.5:
            action = 'SELL'; conf = min(0.65, 0.50 + abs(ret) / 15)
        logger.debug(f"[PatchTST-v6] {symbol} fallback: {action} rsi={rsi:.1f} ret={ret:.2f}%")
        return {"action": action, "confidence": conf, "model": "patchtst_v6_fallback"}
