"""
Adaptive Confidence Calibrator
Replaces hardcoded consensus multipliers with learned weights based on historical performance.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveConfidenceCalibrator:
    """
    Learns optimal confidence multipliers from actual trading outcomes.
    
    Key Principles:
    - NO HARDCODED multipliers - all learned from data
    - Adjusts based on actual PnL outcomes
    - Tracks per-consensus-type performance
    - Saves state to disk for persistence
    """
    
    def __init__(self, state_file: str = "/app/data/confidence_weights.json"):
        self.state_file = state_file
        self.weights = self._load_weights()
        self.performance_history = self._load_history()
        
        logger.info(
            f"[ADAPTIVE-CONFIDENCE] Initialized with weights: "
            f"unanimous={self.weights['unanimous']:.3f}, "
            f"strong={self.weights['strong']:.3f}, "
            f"split={self.weights['split']:.3f}, "
            f"weak={self.weights['weak']:.3f}"
        )
    
    def _load_weights(self) -> Dict[str, float]:
        """
        Load learned weights from disk or use neutral defaults.
        
        ✅ AI-DRIVEN: Starts neutral (1.0) and learns from outcomes
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"[ADAPTIVE-CONFIDENCE] Loaded weights from {self.state_file}")
                    return data.get('weights', self._default_weights())
        except Exception as e:
            logger.warning(f"[ADAPTIVE-CONFIDENCE] Failed to load weights: {e}")
        
        # Start neutral - let AI learn optimal values
        return self._default_weights()
    
    def _default_weights(self) -> Dict[str, float]:
        """
        ✅ AI-DRIVEN: All consensus types start equal (1.0)
        Let the system learn which consensus patterns are most reliable
        """
        return {
            'unanimous': 1.0,  # All 4 models agree
            'strong': 1.0,     # 3/4 models agree
            'split': 1.0,      # 2/4 models agree
            'weak': 1.0        # Only 1 model (should rarely be used)
        }
    
    def _load_history(self) -> Dict[str, list]:
        """Load performance history for each consensus type"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return data.get('history', self._default_history())
        except:
            pass
        return self._default_history()
    
    def _default_history(self) -> Dict[str, list]:
        """Initialize empty performance history"""
        return {
            'unanimous': [],
            'strong': [],
            'split': [],
            'weak': []
        }
    
    def get_multiplier(self, consensus_count: int, total_models: int = 4) -> Tuple[float, str]:
        """
        Get confidence multiplier for given consensus count.
        
        ✅ AI-DRIVEN: Returns learned weights, not hardcoded values
        
        Args:
            consensus_count: Number of models agreeing
            total_models: Total number of models voting
        
        Returns:
            (multiplier, consensus_label)
        """
        # Determine consensus type
        if consensus_count >= total_models:
            consensus_type = 'unanimous'
        elif consensus_count >= (total_models * 0.75):
            consensus_type = 'strong'
        elif consensus_count >= (total_models * 0.5):
            consensus_type = 'split'
        else:
            consensus_type = 'weak'
        
        multiplier = self.weights[consensus_type]
        
        logger.debug(
            f"[ADAPTIVE-CONFIDENCE] {consensus_count}/{total_models} consensus → "
            f"{consensus_type} → multiplier={multiplier:.3f}"
        )
        
        return multiplier, consensus_type
    
    def update_from_outcome(
        self,
        consensus_type: str,
        pnl_pct: float,
        confidence_used: float
    ):
        """
        Update weights based on actual trade outcome.
        
        ✅ AI-DRIVEN: Learn from real results, adjust weights adaptively
        
        Args:
            consensus_type: 'unanimous', 'strong', 'split', or 'weak'
            pnl_pct: Actual PnL percentage (e.g., 0.05 for +5%)
            confidence_used: Confidence that was used for this trade
        """
        # Record outcome in history
        self.performance_history[consensus_type].append({
            'pnl_pct': pnl_pct,
            'confidence': confidence_used
        })
        
        # Keep only last 100 outcomes per type
        if len(self.performance_history[consensus_type]) > 100:
            self.performance_history[consensus_type] = self.performance_history[consensus_type][-100:]
        
        # Adaptive learning rate based on outcome magnitude
        base_learning_rate = 0.02  # 2% adjustment per trade
        
        if pnl_pct > 0:  # Profitable trade
            # Increase weight for this consensus type
            adjustment = 1.0 + (base_learning_rate * abs(pnl_pct) * 10)  # Scale by PnL magnitude
            self.weights[consensus_type] *= adjustment
            
            logger.info(
                f"[ADAPTIVE-CONFIDENCE] ✅ {consensus_type} won +{pnl_pct:.2%} → "
                f"weight increased to {self.weights[consensus_type]:.3f}"
            )
        else:  # Losing trade
            # Decrease weight for this consensus type
            adjustment = 1.0 - (base_learning_rate * abs(pnl_pct) * 10)
            self.weights[consensus_type] *= adjustment
            
            logger.info(
                f"[ADAPTIVE-CONFIDENCE] ❌ {consensus_type} lost {pnl_pct:.2%} → "
                f"weight decreased to {self.weights[consensus_type]:.3f}"
            )
        
        # Clamp weights to reasonable range [0.5, 1.5]
        self.weights[consensus_type] = np.clip(self.weights[consensus_type], 0.5, 1.5)
        
        # Save updated weights
        self._save_weights()
        
        # Log statistics
        recent_trades = len(self.performance_history[consensus_type])
        if recent_trades >= 10:
            win_rate = sum(1 for t in self.performance_history[consensus_type][-10:] if t['pnl_pct'] > 0) / 10
            avg_pnl = np.mean([t['pnl_pct'] for t in self.performance_history[consensus_type][-10:]])
            
            logger.info(
                f"[ADAPTIVE-CONFIDENCE] {consensus_type} stats (last 10): "
                f"Win rate={win_rate:.0%}, Avg PnL={avg_pnl:.2%}"
            )
    
    def _save_weights(self):
        """Persist learned weights to disk"""
        try:
            # Ensure directory exists
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'weights': self.weights,
                'history': self.performance_history
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"[ADAPTIVE-CONFIDENCE] Saved weights to {self.state_file}")
        except Exception as e:
            logger.error(f"[ADAPTIVE-CONFIDENCE] Failed to save weights: {e}")
    
    def get_stats(self) -> Dict:
        """Get current calibrator statistics"""
        stats = {
            'weights': self.weights,
            'trade_counts': {
                k: len(v) for k, v in self.performance_history.items()
            }
        }
        
        # Calculate win rates for types with sufficient data
        for consensus_type, history in self.performance_history.items():
            if len(history) >= 10:
                recent = history[-10:]
                stats[f'{consensus_type}_win_rate'] = sum(1 for t in recent if t['pnl_pct'] > 0) / len(recent)
                stats[f'{consensus_type}_avg_pnl'] = np.mean([t['pnl_pct'] for t in recent])
        
        return stats


# Global instance - shared across predictions
_calibrator_instance = None


def get_calibrator() -> AdaptiveConfidenceCalibrator:
    """
    Get singleton calibrator instance.
    
    ✅ AI-DRIVEN: Single source of truth for learned confidence weights
    """
    global _calibrator_instance
    if _calibrator_instance is None:
        _calibrator_instance = AdaptiveConfidenceCalibrator()
    return _calibrator_instance
