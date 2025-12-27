"""
PHASE 4: ADAPTIVE POLICY REINFORCEMENT LAYER (APRL)
====================================================
Real-time risk optimization through continuous learning.

This module:
- Monitors P&L, drawdown, and volatility in real-time
- Adjusts Safety Governor and Risk Brain parameters automatically
- Optimizes risk exposure (VaR, max position, leverage)
- Improves stability through continuous reinforcement learning
"""

import json
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AdaptivePolicyReinforcement:
    """
    Adaptive Policy Reinforcement Layer (APRL)
    
    Continuously monitors system performance and adjusts risk policies
    to optimize for:
    - Maximum returns within risk constraints
    - Drawdown minimization
    - Volatility management
    - Leverage optimization
    """
    
    def __init__(
        self,
        governor: Optional[Any] = None,
        risk_brain: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        max_window: int = 1000
    ):
        """
        Initialize Adaptive Policy Reinforcement.
        
        Args:
            governor: SafetyGovernor instance (optional for standalone testing)
            risk_brain: RiskBrain instance (optional for standalone testing)
            event_bus: EventBus instance (optional for standalone testing)
            max_window: Maximum number of P&L samples to retain
        """
        self.governor = governor
        self.risk_brain = risk_brain
        self.bus = event_bus
        self.performance_window: List[float] = []
        self.max_window = max_window
        
        # Policy adjustment thresholds
        self.drawdown_threshold_high = -0.05  # 5% drawdown triggers defensive mode
        self.drawdown_threshold_low = -0.02   # 2% drawdown allows normal operation
        self.volatility_threshold_high = 0.02  # 2% std triggers risk reduction
        self.volatility_threshold_low = 0.01   # 1% std allows aggressive mode
        self.performance_threshold_good = 0.01  # 1% mean return allows leverage increase
        
        # Policy state
        self.current_mode = "NORMAL"  # DEFENSIVE, NORMAL, AGGRESSIVE
        self.policy_update_count = 0
        
        logger.info("[PHASE 4] Adaptive Policy Reinforcement initialized")
        logger.info(f"[APRL] Performance window: {max_window} samples")
        logger.info(f"[APRL] Thresholds: DD={self.drawdown_threshold_high:.2%}, VOL={self.volatility_threshold_high:.2%}")
    
    def record_pnl(self, pnl: float) -> None:
        """
        Record a P&L observation.
        
        Args:
            pnl: Profit/Loss value (relative, e.g., 0.01 = 1% gain)
        """
        self.performance_window.append(pnl)
        if len(self.performance_window) > self.max_window:
            self.performance_window.pop(0)
        
        logger.debug(f"[APRL] Recorded P&L: {pnl:.4f} (window size: {len(self.performance_window)})")
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute performance metrics from P&L window.
        
        Returns:
            Dictionary with mean, std, drawdown, sharpe
        """
        if not self.performance_window:
            return {
                "mean": 0.0,
                "std": 0.0,
                "drawdown": 0.0,
                "sharpe": 0.0,
                "sample_count": 0
            }
        
        pnl = np.array(self.performance_window)
        mean = float(np.mean(pnl))
        std = float(np.std(pnl))
        
        # Compute maximum drawdown
        cumsum = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = float(np.min(cumsum - running_max))
        
        # Compute Sharpe ratio (annualized, assuming hourly data)
        sharpe = (mean / std * np.sqrt(8760)) if std > 0 else 0.0
        
        return {
            "mean": mean,
            "std": std,
            "drawdown": drawdown,
            "sharpe": float(sharpe),
            "sample_count": len(self.performance_window)
        }
    
    def determine_mode(self, metrics: Dict[str, float]) -> str:
        """
        Determine operational mode based on metrics.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Mode string: DEFENSIVE, NORMAL, or AGGRESSIVE
        """
        mean = metrics["mean"]
        std = metrics["std"]
        drawdown = metrics["drawdown"]
        
        # DEFENSIVE: High drawdown or high volatility
        if drawdown < self.drawdown_threshold_high or std > self.volatility_threshold_high:
            return "DEFENSIVE"
        
        # AGGRESSIVE: Good performance and low volatility
        if mean > self.performance_threshold_good and drawdown > self.drawdown_threshold_low and std < self.volatility_threshold_low:
            return "AGGRESSIVE"
        
        # NORMAL: Everything else
        return "NORMAL"
    
    def adjust_policy(self) -> Dict[str, Any]:
        """
        Adjust risk policies based on current performance metrics.
        
        Returns:
            Dictionary with adjustment details
        """
        metrics = self.compute_metrics()
        mean, std, dd = metrics["mean"], metrics["std"], metrics["drawdown"]
        
        logger.info(f"[POLICY] Metrics - mean={mean:.4f} std={std:.4f} drawdown={dd:.4f} sharpe={metrics['sharpe']:.2f}")
        
        # Determine new mode
        new_mode = self.determine_mode(metrics)
        mode_changed = new_mode != self.current_mode
        
        if mode_changed:
            logger.warning(f"[ADAPT] Mode transition: {self.current_mode} → {new_mode}")
            self.current_mode = new_mode
        
        adjustments = {
            "timestamp": datetime.now().isoformat(),
            "mode": new_mode,
            "mode_changed": mode_changed,
            "metrics": metrics,
            "policy_changes": []
        }
        
        # Apply mode-specific policy adjustments
        if new_mode == "DEFENSIVE":
            if self.governor:
                self.governor.update_policy("max_leverage", 0.5)
                self.governor.update_policy("max_position_size", 0.5)
                adjustments["policy_changes"].append("max_leverage=0.5x (DEFENSIVE)")
                adjustments["policy_changes"].append("max_position_size=50% (DEFENSIVE)")
                logger.warning("[ADAPT] DEFENSIVE MODE: Reducing leverage to 0.5x, position size to 50%")
            
            if dd < self.drawdown_threshold_high:
                logger.error(f"[ADAPT] CRITICAL DRAWDOWN: {dd:.2%} (threshold: {self.drawdown_threshold_high:.2%})")
            if std > self.volatility_threshold_high:
                logger.error(f"[ADAPT] HIGH VOLATILITY: {std:.2%} (threshold: {self.volatility_threshold_high:.2%})")
        
        elif new_mode == "AGGRESSIVE":
            if self.governor:
                self.governor.update_policy("max_leverage", 1.5)
                self.governor.update_policy("max_position_size", 0.8)
                adjustments["policy_changes"].append("max_leverage=1.5x (AGGRESSIVE)")
                adjustments["policy_changes"].append("max_position_size=80% (AGGRESSIVE)")
                logger.info("[ADAPT] AGGRESSIVE MODE: Increasing leverage to 1.5x, position size to 80%")
        
        else:  # NORMAL
            if self.governor:
                self.governor.update_policy("max_leverage", 1.0)
                self.governor.update_policy("max_position_size", 0.7)
                adjustments["policy_changes"].append("max_leverage=1.0x (NORMAL)")
                adjustments["policy_changes"].append("max_position_size=70% (NORMAL)")
                logger.info("[ADAPT] NORMAL MODE: Standard leverage 1.0x, position size 70%")
        
        # Update Risk Brain with volatility constraint
        if self.risk_brain:
            # Risk Brain expects exposure limit (0-1) and volatility estimate
            exposure_limit = 0.5 if new_mode == "DEFENSIVE" else (0.9 if new_mode == "AGGRESSIVE" else 0.7)
            try:
                self.risk_brain.update_limits(exposure=exposure_limit, vol=std)
                adjustments["policy_changes"].append(f"risk_brain_exposure={exposure_limit:.1%}")
                logger.info(f"[ADAPT] Risk Brain updated: exposure={exposure_limit:.1%}, vol={std:.4f}")
            except Exception as e:
                logger.warning(f"[ADAPT] Could not update Risk Brain: {e}")
        
        # Publish policy update event
        if self.bus:
            try:
                self.bus.publish("policy_update", adjustments)
                logger.debug("[APRL] Published policy_update event to EventBus")
            except Exception as e:
                logger.warning(f"[APRL] Could not publish event: {e}")
        
        self.policy_update_count += 1
        adjustments["update_count"] = self.policy_update_count
        
        return adjustments
    
    def run_continuous(self, interval_seconds: int = 3600) -> None:
        """
        Run continuous policy adjustment loop.
        
        Args:
            interval_seconds: Time between adjustments (default: 3600 = 1 hour)
        """
        logger.info(f"[PHASE 4] Adaptive Policy Reinforcement loop started (interval: {interval_seconds}s)")
        
        while True:
            try:
                adjustments = self.adjust_policy()
                
                if adjustments["policy_changes"]:
                    logger.info(f"[APRL] Applied {len(adjustments['policy_changes'])} policy changes")
                else:
                    logger.info("[APRL] No policy changes required")
                
                # Sleep until next adjustment cycle
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("[APRL] Shutting down (KeyboardInterrupt)")
                break
            except Exception as e:
                logger.error(f"[ERROR] Reinforcement loop failure: {e}", exc_info=True)
                # Continue running even on error, but wait before retry
                time.sleep(60)
        
        logger.info("[PHASE 4] Adaptive Policy Reinforcement loop stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current APRL status.
        
        Returns:
            Dictionary with status information
        """
        metrics = self.compute_metrics()
        
        return {
            "active": True,
            "mode": self.current_mode,
            "policy_updates": self.policy_update_count,
            "performance_samples": len(self.performance_window),
            "current_metrics": metrics,
            "thresholds": {
                "drawdown_defensive": self.drawdown_threshold_high,
                "volatility_defensive": self.volatility_threshold_high,
                "performance_aggressive": self.performance_threshold_good
            }
        }


# Standalone execution for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create standalone APRL (no dependencies)
    aprl = AdaptivePolicyReinforcement()
    
    # Simulate some P&L data
    logger.info("[TEST] Simulating P&L data...")
    
    # Good performance
    for i in range(50):
        aprl.record_pnl(np.random.normal(0.001, 0.005))
    
    aprl.adjust_policy()
    logger.info(f"[TEST] Mode after good performance: {aprl.current_mode}")
    
    # High volatility
    for i in range(50):
        aprl.record_pnl(np.random.normal(0.0, 0.03))
    
    aprl.adjust_policy()
    logger.info(f"[TEST] Mode after high volatility: {aprl.current_mode}")
    
    # Large drawdown
    for i in range(50):
        aprl.record_pnl(np.random.normal(-0.002, 0.005))
    
    aprl.adjust_policy()
    logger.info(f"[TEST] Mode after drawdown: {aprl.current_mode}")
    
    # Print status
    status = aprl.get_status()
    logger.info(f"[TEST] Final status: {json.dumps(status, indent=2)}")
    
    logger.info("[PHASE 4 TEST] Adaptive Policy Reinforcement test complete ✓")
