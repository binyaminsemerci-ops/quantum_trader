"""
Calibration Loader - Integrate Calibration-Only Learning with AI Engine

This module loads calibration configuration and provides methods to:
1. Apply confidence calibration to prediction confidence scores
2. Override ensemble weights if calibration is deployed

Used by:
- AI Engine at startup
- EnsembleManager during predictions

Philosophy: "Vi forbedrer beslutningskvalitet uten å endre modeller"
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CalibrationLoader:
    """
    Loads and applies calibration-only learning configuration.
    
    This is the ONLY place where calibration config is read into AI Engine.
    """
    
    DEFAULT_CONFIG_PATH = "/home/qt/quantum_trader/config/calibration.json"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or self.DEFAULT_CONFIG_PATH)
        
        # Calibration state
        self.calibration_loaded = False
        self.confidence_calibration_enabled = False
        self.confidence_mapping: Dict[float, float] = {}
        self.ensemble_weights: Optional[Dict[str, float]] = None
        
        # Metadata
        self.version: Optional[str] = None
        self.created_at: Optional[datetime] = None
        self.based_on_trades: Optional[int] = None
        
        # Attempt load at initialization
        self.reload()
    
    def reload(self) -> bool:
        """
        Reload calibration configuration from disk.
        
        Returns:
            True if calibration loaded successfully, False otherwise
        """
        if not self.config_path.exists():
            logger.info(f"[CalibrationLoader] No config at {self.config_path}")
            logger.info(f"[CalibrationLoader] Using baseline ensemble weights")
            self.calibration_loaded = False
            return False
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Parse metadata
            self.version = config.get('version')
            created_at_str = config.get('created_at')
            if created_at_str:
                self.created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            self.based_on_trades = config.get('based_on_trades')
            
            # Parse confidence calibration
            conf_cal = config.get('confidence_calibration', {})
            self.confidence_calibration_enabled = conf_cal.get('enabled', False)
            
            # Convert string keys to float
            mapping_raw = conf_cal.get('mapping', {})
            self.confidence_mapping = {float(k): float(v) for k, v in mapping_raw.items()}
            
            # Parse ensemble weights
            weights_cal = config.get('ensemble_weights', {})
            if weights_cal.get('enabled', False):
                self.ensemble_weights = {
                    'xgb': weights_cal.get('weights', {}).get('xgb'),
                    'lgbm': weights_cal.get('weights', {}).get('lgbm'),
                    'nhits': weights_cal.get('weights', {}).get('nhits'),
                    'patchtst': weights_cal.get('weights', {}).get('patchtst')
                }
                
                # Validate weights (should sum to ~1.0)
                weight_sum = sum(w for w in self.ensemble_weights.values() if w is not None)
                if abs(weight_sum - 1.0) > 0.01:
                    logger.warning(f"[CalibrationLoader] Weights don't sum to 1.0: {weight_sum:.3f}")
                
            else:
                self.ensemble_weights = None
            
            self.calibration_loaded = True
            
            logger.info(f"[CalibrationLoader] ✅ Loaded calibration: {self.version}")
            logger.info(f"[CalibrationLoader]    Based on {self.based_on_trades} trades")
            logger.info(f"[CalibrationLoader]    Confidence calibration: {'✅ ENABLED' if self.confidence_calibration_enabled else '⏸ DISABLED'}")
            logger.info(f"[CalibrationLoader]    Ensemble weights: {'✅ OVERRIDE' if self.ensemble_weights else '⏸ BASELINE'}")
            
            if self.confidence_calibration_enabled:
                logger.info(f"[CalibrationLoader]    Confidence mapping: {len(self.confidence_mapping)} bins")
            
            if self.ensemble_weights:
                logger.info(f"[CalibrationLoader]    Weights: {self.ensemble_weights}")
            
            return True
            
        except Exception as e:
            logger.error(f"[CalibrationLoader] ❌ Failed to load config: {e}")
            self.calibration_loaded = False
            return False
    
    def apply_confidence_calibration(self, raw_confidence: float) -> float:
        """
        Apply isotonic regression calibration to a confidence score.
        
        This maps predicted confidence → actual win rate.
        
        Args:
            raw_confidence: Raw confidence from model (0.0-1.0)
        
        Returns:
            Calibrated confidence (0.0-1.0)
        """
        if not self.confidence_calibration_enabled:
            return raw_confidence
        
        if not self.confidence_mapping:
            logger.warning(f"[CalibrationLoader] Confidence calibration enabled but no mapping!")
            return raw_confidence
        
        # Interpolate from lookup table (binned mapping)
        import numpy as np
        
        bins = sorted(self.confidence_mapping.keys())
        values = [self.confidence_mapping[b] for b in bins]
        
        # Linear interpolation
        calibrated = np.interp(raw_confidence, bins, values)
        calibrated = float(np.clip(calibrated, 0.0, 1.0))
        
        # Log only if difference is significant (>2%)
        diff = abs(calibrated - raw_confidence)
        if diff > 0.02:
            logger.debug(f"[CalibrationLoader] Confidence: {raw_confidence:.3f} → {calibrated:.3f} (Δ{calibrated-raw_confidence:+.3f})")
        
        return calibrated
    
    def get_ensemble_weights(self, baseline: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Get ensemble weights (calibrated or baseline).
        
        Args:
            baseline: Default weights if no calibration loaded
        
        Returns:
            Dict of {'xgb': weight, 'lgbm': weight, ...}
        """
        if self.calibration_loaded and self.ensemble_weights:
            return self.ensemble_weights.copy()
        
        # Return baseline
        if baseline:
            return baseline.copy()
        
        # Hard fallback
        return {
            'xgb': 0.30,
            'lgbm': 0.30,
            'nhits': 0.20,
            'patchtst': 0.20
        }
    
    def get_status(self) -> Dict:
        """
        Get calibration status for logging/monitoring.
        
        Returns:
            Dict with calibration status information
        """
        return {
            'loaded': self.calibration_loaded,
            'version': self.version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'based_on_trades': self.based_on_trades,
            'confidence_enabled': self.confidence_calibration_enabled,
            'weights_override': self.ensemble_weights is not None,
            'config_path': str(self.config_path),
            'config_exists': self.config_path.exists()
        }


# Global singleton instance for AI Engine consumption
_calibration_loader: Optional[CalibrationLoader] = None


def get_calibration_loader() -> CalibrationLoader:
    """
    Get global CalibrationLoader instance (singleton pattern).
    
    This is the main entry point for AI Engine integration.
    """
    global _calibration_loader
    
    if _calibration_loader is None:
        _calibration_loader = CalibrationLoader()
    
    return _calibration_loader


def reload_calibration() -> bool:
    """
    Reload calibration from disk.
    
    Call this when calibration.json is updated.
    
    Returns:
        True if reload successful
    """
    loader = get_calibration_loader()
    return loader.reload()
