"""Orchestrator - Wires CEO Brain, Strategy Brain, and Risk Brain together.

Phase 2.2: Wire CEO Brain Orchestration
This orchestrator coordinates decisions between:
- CEO Brain (port 8010): Operating mode decisions (EXPANSION/PRESERVATION/EMERGENCY)
- Strategy Brain (port 8011): Signal evaluation and strategy recommendations
- Risk Brain (port 8012): Position sizing and risk assessment

Usage:
    orchestrator = Orchestrator()
    decision = await orchestrator.evaluate_signal(signal)
"""

from __future__ import annotations

import logging
import httpx
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SignalEvaluation:
    """Result of signal evaluation by orchestrator."""
    
    # CEO Brain decision
    operating_mode: str                  # EXPANSION/PRESERVATION/EMERGENCY
    ceo_confidence: float                # 0-1
    
    # Strategy evaluation
    strategy_approved: bool              # Should we take this signal?
    strategy_reason: str                 # Why/why not?
    
    # Risk assessment
    position_size: float                 # USD amount
    leverage: float                      # Leverage to use
    risk_score: float                    # 0-100
    max_loss: float                      # USD max loss for this trade
    
    # Combined decision
    final_decision: str                  # "EXECUTE", "SKIP", "DELAY"
    decision_reason: str                 # Why this decision?
    timestamp: datetime


class CEOBrainClient:
    """HTTP client for CEO Brain service."""
    
    def __init__(self, base_url: str = "http://ceo-brain:8010"):
        """Initialize CEO Brain client.
        
        Args:
            base_url: Base URL for CEO Brain service (default: http://ceo-brain:8010)
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=5.0)
        logger.info(f"CEOBrainClient initialized: {self.base_url}")
    
    async def get_mode(self) -> tuple[str, float]:
        """Get current operating mode from CEO Brain.
        
        Returns:
            Tuple of (mode, confidence)
            - mode: "EXPANSION", "PRESERVATION", or "EMERGENCY"
            - confidence: 0.0 to 1.0
        """
        try:
            response = await self._client.get(f"{self.base_url}/status")
            response.raise_for_status()
            data = response.json()
            
            mode = data.get("mode", "EXPANSION")
            confidence = data.get("confidence", 0.8)
            
            logger.debug(f"CEO Brain mode: {mode} (confidence={confidence:.2f})")
            return mode, confidence
        
        except Exception as e:
            logger.warning(f"Failed to get CEO Brain mode: {e}. Defaulting to EXPANSION")
            return "EXPANSION", 0.5  # Safe default
    
    async def decide(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Request CEO decision based on system state.
        
        Args:
            system_state: Dictionary with current system metrics
        
        Returns:
            CEO decision with mode, actions, and reasoning
        """
        try:
            response = await self._client.post(
                f"{self.base_url}/decide",
                json=system_state
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"CEO Brain decide failed: {e}")
            return {
                "operating_mode": "EXPANSION",
                "decision": "maintain_current_operations",
                "error": str(e)
            }


class StrategyBrainClient:
    """HTTP client for Strategy Brain service."""
    
    def __init__(self, base_url: str = "http://strategy-brain:8011"):
        """Initialize Strategy Brain client.
        
        Args:
            base_url: Base URL for Strategy Brain service
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=5.0)
        logger.info(f"StrategyBrainClient initialized: {self.base_url}")
    
    async def evaluate(self, signal: Dict[str, Any]) -> tuple[bool, str]:
        """Evaluate trading signal through strategy analysis.
        
        Args:
            signal: Trading signal to evaluate
        
        Returns:
            Tuple of (approved, reason)
        """
        try:
            response = await self._client.post(
                f"{self.base_url}/evaluate",
                json=signal
            )
            response.raise_for_status()
            data = response.json()
            
            approved = data.get("approved", True)
            reason = data.get("reason", "Strategy evaluation passed")
            
            logger.debug(f"Strategy Brain: approved={approved}, reason={reason}")
            return approved, reason
        
        except Exception as e:
            logger.warning(f"Strategy Brain evaluation failed: {e}. Defaulting to approved=True")
            return True, "Strategy Brain unavailable - default approve"


class RiskBrainClient:
    """HTTP client for Risk Brain service."""
    
    def __init__(self, base_url: str = "http://risk-brain:8012"):
        """Initialize Risk Brain client.
        
        Args:
            base_url: Base URL for Risk Brain service
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=5.0)
        logger.info(f"RiskBrainClient initialized: {self.base_url}")
    
    async def evaluate(
        self, 
        signal: Dict[str, Any], 
        operating_mode: str
    ) -> tuple[float, float, float, float]:
        """Evaluate risk and determine position sizing.
        
        Args:
            signal: Trading signal to evaluate
            operating_mode: Current CEO Brain operating mode
        
        Returns:
            Tuple of (position_size, leverage, risk_score, max_loss)
        """
        try:
            response = await self._client.post(
                f"{self.base_url}/evaluate",
                json={
                    "signal": signal,
                    "operating_mode": operating_mode
                }
            )
            response.raise_for_status()
            data = response.json()
            
            position_size = data.get("position_size", 100.0)
            leverage = data.get("leverage", 1.0)
            risk_score = data.get("risk_score", 50.0)
            max_loss = data.get("max_loss", position_size * 0.02)
            
            logger.debug(
                f"Risk Brain: size=${position_size:.2f}, "
                f"leverage={leverage:.1f}x, risk={risk_score:.1f}"
            )
            
            return position_size, leverage, risk_score, max_loss
        
        except Exception as e:
            logger.warning(f"Risk Brain evaluation failed: {e}. Using conservative defaults")
            # Conservative defaults when Risk Brain unavailable
            return 100.0, 1.0, 50.0, 2.0


class Orchestrator:
    """
    Master orchestrator that coordinates CEO Brain, Strategy Brain, and Risk Brain.
    
    This is the Phase 2.2 implementation that wires together:
    - CEO Brain: Operating mode decisions
    - Strategy Brain: Signal evaluation
    - Risk Brain: Position sizing and risk assessment
    
    The orchestrator ensures signals flow through proper evaluation pipeline:
    1. Get operating mode from CEO Brain
    2. If EXPANSION mode, evaluate signal with Strategy Brain
    3. Get position sizing from Risk Brain
    4. Make final execution decision
    """
    
    def __init__(
        self,
        ceo_brain_url: Optional[str] = None,
        strategy_brain_url: Optional[str] = None,
        risk_brain_url: Optional[str] = None,
    ):
        """Initialize orchestrator with brain clients.
        
        Args:
            ceo_brain_url: CEO Brain service URL (default: env or http://ceo-brain:8010)
            strategy_brain_url: Strategy Brain URL (default: env or http://strategy-brain:8011)
            risk_brain_url: Risk Brain URL (default: env or http://risk-brain:8012)
        """
        # Use environment variables if available, otherwise defaults
        ceo_url = ceo_brain_url or os.getenv("CEO_BRAIN_URL", "http://ceo-brain:8010")
        strategy_url = strategy_brain_url or os.getenv("STRATEGY_BRAIN_URL", "http://strategy-brain:8011")
        risk_url = risk_brain_url or os.getenv("RISK_BRAIN_URL", "http://risk-brain:8012")
        
        # Initialize brain clients
        self.ceo_brain = CEOBrainClient(ceo_url)
        self.strategy_brain = StrategyBrainClient(strategy_url)
        self.risk_brain = RiskBrainClient(risk_url)
        
        # Configuration
        self._enabled = os.getenv("ENABLE_ORCHESTRATION", "true").lower() == "true"
        self._bypass_ceo = os.getenv("BYPASS_CEO_BRAIN", "false").lower() == "true"
        
        logger.info(
            f"🧠 Orchestrator initialized: "
            f"enabled={self._enabled}, "
            f"bypass_ceo={self._bypass_ceo}"
        )
    
    async def evaluate_signal(self, signal: Dict[str, Any]) -> SignalEvaluation:
        """
        Evaluate trading signal through full orchestration pipeline.
        
        Args:
            signal: Trading signal with symbol, direction, confidence, etc.
        
        Returns:
            SignalEvaluation with final decision and all intermediate results
        """
        timestamp = datetime.utcnow()
        
        # If orchestration disabled, approve with defaults
        if not self._enabled:
            logger.debug("Orchestration disabled - auto-approving signal")
            return SignalEvaluation(
                operating_mode="EXPANSION",
                ceo_confidence=1.0,
                strategy_approved=True,
                strategy_reason="Orchestration disabled",
                position_size=100.0,
                leverage=1.0,
                risk_score=50.0,
                max_loss=2.0,
                final_decision="EXECUTE",
                decision_reason="Orchestration disabled - default execute",
                timestamp=timestamp,
            )
        
        symbol = signal.get("symbol", "UNKNOWN")
        confidence = signal.get("confidence", 0.0)
        
        logger.info(f"🧠 Orchestrating signal: {symbol} (confidence={confidence:.2f})")
        
        # Step 1: Get operating mode from CEO Brain
        operating_mode, ceo_confidence = await self.ceo_brain.get_mode()
        
        logger.info(f"  ├─ CEO Brain: mode={operating_mode}, confidence={ceo_confidence:.2f}")
        
        # Step 2: Check if we should trade based on mode
        if operating_mode == "EMERGENCY":
            logger.warning(f"  ├─ EMERGENCY mode - skipping signal {symbol}")
            return SignalEvaluation(
                operating_mode=operating_mode,
                ceo_confidence=ceo_confidence,
                strategy_approved=False,
                strategy_reason="EMERGENCY mode active",
                position_size=0.0,
                leverage=0.0,
                risk_score=100.0,
                max_loss=0.0,
                final_decision="SKIP",
                decision_reason="CEO Brain in EMERGENCY mode - no new positions",
                timestamp=timestamp,
            )
        
        # Step 3: Evaluate signal with Strategy Brain (if EXPANSION mode)
        strategy_approved = True
        strategy_reason = "PRESERVATION mode - conservative approach"
        
        if operating_mode == "EXPANSION":
            strategy_approved, strategy_reason = await self.strategy_brain.evaluate(signal)
            logger.info(f"  ├─ Strategy Brain: approved={strategy_approved}, reason={strategy_reason}")
        else:
            logger.info(f"  ├─ Strategy Brain: skipped (PRESERVATION mode)")
        
        # If strategy rejected, skip signal
        if not strategy_approved:
            return SignalEvaluation(
                operating_mode=operating_mode,
                ceo_confidence=ceo_confidence,
                strategy_approved=False,
                strategy_reason=strategy_reason,
                position_size=0.0,
                leverage=0.0,
                risk_score=100.0,
                max_loss=0.0,
                final_decision="SKIP",
                decision_reason=f"Strategy Brain rejected: {strategy_reason}",
                timestamp=timestamp,
            )
        
        # Step 4: Get position sizing from Risk Brain
        position_size, leverage, risk_score, max_loss = await self.risk_brain.evaluate(
            signal, operating_mode
        )
        
        logger.info(
            f"  ├─ Risk Brain: size=${position_size:.2f}, "
            f"leverage={leverage:.1f}x, risk={risk_score:.1f}"
        )
        
        # Step 5: Make final decision
        final_decision = "EXECUTE"
        decision_reason = f"All checks passed - {operating_mode} mode"
        
        # Additional safety checks
        if position_size <= 0:
            final_decision = "SKIP"
            decision_reason = "Position size <= 0 from Risk Brain"
        elif risk_score > 90:
            final_decision = "SKIP"
            decision_reason = f"Risk score too high: {risk_score:.1f}"
        elif confidence < 0.45:  # Below ILF threshold
            final_decision = "SKIP"
            decision_reason = f"Signal confidence too low: {confidence:.2f}"
        
        logger.info(f"  └─ Final: {final_decision} - {decision_reason}")
        
        return SignalEvaluation(
            operating_mode=operating_mode,
            ceo_confidence=ceo_confidence,
            strategy_approved=strategy_approved,
            strategy_reason=strategy_reason,
            position_size=position_size,
            leverage=leverage,
            risk_score=risk_score,
            max_loss=max_loss,
            final_decision=final_decision,
            decision_reason=decision_reason,
            timestamp=timestamp,
        )


# Singleton instance for easy import
_orchestrator_instance: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get or create singleton orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance
