"""
Calibration Runtime Integration (PATH 2.4A)

Applies learned calibration to ensemble_predictor_service in production.

Flow:
1. Ensemble produces raw_confidence
2. Calibrator maps raw → calibrated
3. Calibrated confidence published to signal.score

Authority: STILL SCORER ONLY (no execution)
"""
import pickle
import logging
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


class CalibrationLoader:
    """
    Loads pre-trained calibration function and applies at runtime.
    
    This is injected into EnsemblePredictorService after calibration is proven.
    """
    
    def __init__(
        self,
        calibration_path: str = "/home/qt/quantum_trader/ai_engine/calibration/calibrator_v1.pkl",
        fallback_mode: str = "passthrough"  # What to do if calibrator fails
    ):
        """
        Initialize calibration loader.
        
        Args:
            calibration_path: Path to saved calibrator pickle
            fallback_mode: "passthrough" (use raw) or "conservative" (reduce confidence)
        """
        self.calibration_path = Path(calibration_path)
        self.fallback_mode = fallback_mode
        self.calibrator = None
        self.is_loaded = False
        
        # Try to load calibrator
        self._load_calibrator()
    
    def _load_calibrator(self):
        """Load calibrator from disk."""
        if not self.calibration_path.exists():
            logger.warning(f"[CALIBRATION] Calibrator not found: {self.calibration_path}")
            logger.warning(f"[CALIBRATION] Using fallback_mode={self.fallback_mode}")
            return
        
        try:
            with open(self.calibration_path, 'rb') as f:
                self.calibrator = pickle.load(f)
            
            self.is_loaded = True
            logger.info(f"[CALIBRATION] ✅ Loaded calibrator from {self.calibration_path.name}")
            
        except Exception as e:
            logger.error(f"[CALIBRATION] Failed to load calibrator: {e}")
            logger.warning(f"[CALIBRATION] Using fallback_mode={self.fallback_mode}")
    
    def apply_confidence_calibration(self, raw_confidence: float) -> float:
        """
        Apply calibration to raw confidence.
        
        Args:
            raw_confidence: Raw confidence from ensemble [0.0, 1.0]
        
        Returns:
            Calibrated confidence (semantically true)
        """
        if not self.is_loaded or self.calibrator is None:
            # Fallback strategies
            if self.fallback_mode == "passthrough":
                return raw_confidence
            elif self.fallback_mode == "conservative":
                # Reduce confidence by 20% when uncalibrated
                return max(0.0, raw_confidence * 0.8)
            else:
                return raw_confidence
        
        try:
            # Apply calibrator
            if NUMPY_AVAILABLE and self.calibrator is not None:
                calibrated = self.calibrator.predict([raw_confidence])[0]
                # CRITICAL: Clip to [0,1] - isotonic regression can extrapolate
                calibrated =float(np.clip(calibrated, 0.0, 1.0))
                logger.debug(f"[CALIBRATION] Applied: {raw_confidence:.3f} → {calibrated:.3f}")
                return calibrated
            else:
                logger.warning("[CALIBRATION] NumPy not available or calibrator None, using passthrough")
                return raw_confidence
                
        except Exception as e:
            logger.error(f"[CALIBRATION] Calibration error: {e}")
            return raw_confidence if self.fallback_mode == "passthrough" else raw_confidence * 0.8
    
    def get_status(self) -> dict:
        """Get calibration status for monitoring."""
        return {
            "loaded": self.is_loaded,
            "calibration_path": str(self.calibration_path),
            "fallback_mode": self.fallback_mode,
            "active": self.is_loaded and self.calibrator is not None
        }


def save_calibrator(calibrator, output_path: str, metadata: dict = None):
    """
    Save trained calibrator to disk.
    
    Args:
        calibrator: Fitted IsotonicRegression or similar
        output_path: Where to save
        metadata: Optional metadata (training stats, etc.)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save calibrator
    with open(output_path, 'wb') as f:
        pickle.dump(calibrator, f)
    
    logger.info(f"[CALIBRATION] Saved calibrator to {output_path}")
    
    # Save metadata
    if metadata:
        import json
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"[CALIBRATION] Saved metadata to {metadata_path}")
