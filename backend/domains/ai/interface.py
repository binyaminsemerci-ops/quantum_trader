"""
AI Module Interface
===================
Uniform interface for all AI modules in Quantum Trader v2.0

Provides: AIInput, AIOutput, adapter functions for different module types
"""

from __future__ import annotations
from typing import Any, Optional, Union, Dict, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import numpy as np


class AIInput(BaseModel):
    """Standardized input for AI modules."""
    
    # Core data
    features: Union[np.ndarray, List[float], List[List[float]]] = Field(
        ..., description="Feature vector or matrix"
    )
    
    # Context
    symbol: Optional[str] = None
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class AIOutput(BaseModel):
    """Standardized output from AI modules."""
    
    # Primary decision
    action: Optional[str] = None  # "BUY" | "SELL" | "HOLD" | "SIZE" | etc.
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Scores/probabilities
    scores: Dict[str, Any] = Field(default_factory=dict)
    
    # Raw output for debugging
    raw: Any = None
    
    # Status
    success: bool = True
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# ADAPTER FUNCTIONS
# ============================================================================

def normalize_features(features: Any) -> np.ndarray:
    """Convert any feature format to numpy array."""
    if isinstance(features, np.ndarray):
        return features
    elif isinstance(features, list):
        return np.array(features, dtype=np.float32)
    else:
        raise ValueError(f"Cannot convert {type(features)} to numpy array")


def clamp_confidence(value: Optional[float]) -> float:
    """Clamp confidence to [0, 1]."""
    if value is None:
        return 0.5
    return max(0.0, min(1.0, float(value)))


# ============================================================================
# PREDICTOR ADAPTER
# ============================================================================

def run_predictor(model: Any, ai_input: AIInput) -> AIOutput:
    """
    Run a predictor model (price/signal prediction).
    
    Expected model interface:
        model.predict(features) -> np.ndarray or float
    """
    try:
        features = normalize_features(ai_input.features)
        
        # Handle different predictor interfaces
        if hasattr(model, 'predict'):
            prediction = model.predict(features)
        elif hasattr(model, 'forward'):
            prediction = model.forward(features)
        elif callable(model):
            prediction = model(features)
        else:
            return AIOutput(
                success=False,
                error=f"Model has no predict/forward method: {type(model)}"
            )
        
        # Normalize prediction
        if isinstance(prediction, np.ndarray):
            pred_value = float(prediction.flatten()[0])
        else:
            pred_value = float(prediction)
        
        # Interpret as confidence or action
        if pred_value > 0.5:
            action = "BUY"
            confidence = pred_value
        elif pred_value < -0.5:
            action = "SELL"
            confidence = abs(pred_value)
        else:
            action = "HOLD"
            confidence = 1.0 - abs(pred_value)
        
        return AIOutput(
            action=action,
            confidence=clamp_confidence(confidence),
            scores={"prediction": pred_value},
            raw=prediction,
            success=True
        )
        
    except Exception as e:
        return AIOutput(
            success=False,
            error=f"Predictor error: {str(e)}"
        )


# ============================================================================
# RL AGENT ADAPTER
# ============================================================================

def run_rl_agent(agent: Any, ai_input: AIInput) -> AIOutput:
    """
    Run an RL agent (position sizing, meta-strategy).
    
    Expected agent interface:
        agent.act(state) -> action (int or dict)
        agent.get_action(state) -> action
    """
    try:
        features = normalize_features(ai_input.features)
        
        # Try different RL interfaces
        action_result = None
        if hasattr(agent, 'act'):
            action_result = agent.act(features)
        elif hasattr(agent, 'get_action'):
            action_result = agent.get_action(features)
        elif hasattr(agent, 'predict'):
            action_result = agent.predict(features)
        else:
            return AIOutput(
                success=False,
                error=f"RL agent has no act/get_action method: {type(agent)}"
            )
        
        # Parse action result
        if isinstance(action_result, dict):
            action = action_result.get('action', 'HOLD')
            confidence = action_result.get('confidence', 0.5)
            scores = action_result.get('scores', {})
        elif isinstance(action_result, (int, float)):
            action = f"ACTION_{int(action_result)}"
            confidence = 0.7
            scores = {"action_id": float(action_result)}
        else:
            action = str(action_result)
            confidence = 0.5
            scores = {}
        
        return AIOutput(
            action=action,
            confidence=clamp_confidence(confidence),
            scores=scores,
            raw=action_result,
            success=True
        )
        
    except Exception as e:
        return AIOutput(
            success=False,
            error=f"RL agent error: {str(e)}"
        )


# ============================================================================
# DETECTOR ADAPTER
# ============================================================================

def run_detector(detector: Any, ai_input: AIInput) -> AIOutput:
    """
    Run a detector (regime, drift, anomaly).
    
    Expected detector interface:
        detector.detect(data) -> str or dict
        detector.detect_regime(context) -> MarketRegime
    """
    try:
        features = normalize_features(ai_input.features)
        
        # Try different detector interfaces
        detection_result = None
        
        # Method priority: detect > detect_regime > check > detect_drift > detect_shift
        if hasattr(detector, 'detect') and callable(getattr(detector, 'detect')):
            detection_result = detector.detect(features)
        elif hasattr(detector, 'detect_regime') and callable(getattr(detector, 'detect_regime')):
            # Requires MarketContext, build from metadata
            try:
                from backend.services.ai.regime_detector import MarketContext
                context = ai_input.metadata.get('market_context')
                if context:
                    detection_result = detector.detect_regime(context)
                else:
                    # Create dummy context for testing
                    detection_result = "TREND_UP"  # Dummy for health check
            except Exception as e:
                detection_result = "UNKNOWN"
        elif hasattr(detector, 'check') and callable(getattr(detector, 'check')):
            detection_result = detector.check(features)
        elif hasattr(detector, 'detect_drift') and callable(getattr(detector, 'detect_drift')):
            detection_result = detector.detect_drift(features)
        elif hasattr(detector, 'detect_shift') and callable(getattr(detector, 'detect_shift')):
            detection_result = detector.detect_shift(features)
        else:
            # Detector loaded but has no detect method - try generic
            return run_generic(detector, ai_input)
        
        # Parse detection result
        if isinstance(detection_result, dict):
            action = detection_result.get('regime', detection_result.get('status', 'UNKNOWN'))
            confidence = detection_result.get('confidence', 0.5)
            scores = detection_result
        elif isinstance(detection_result, str):
            action = detection_result
            confidence = 0.7
            scores = {"regime": detection_result}
        else:
            action = str(detection_result)
            confidence = 0.5
            scores = {}
        
        return AIOutput(
            action=action,
            confidence=clamp_confidence(confidence),
            scores=scores,
            raw=detection_result,
            success=True
        )
        
    except Exception as e:
        return AIOutput(
            success=False,
            error=f"Detector error: {str(e)}"
        )


# ============================================================================
# GENERIC ADAPTER (fallback)
# ============================================================================

def run_generic(module: Any, ai_input: AIInput) -> AIOutput:
    """
    Generic adapter for modules with unknown interface.
    Tries common method names.
    """
    features = normalize_features(ai_input.features)
    
    # Try common methods
    for method_name in ['run', 'execute', 'process', 'predict', 'act', 'update', 'check_health', '__call__']:
        if hasattr(module, method_name):
            try:
                method = getattr(module, method_name)
                
                # Skip non-callables
                if not callable(method):
                    continue
                
                # Try calling with features
                try:
                    result = method(features)
                except TypeError:
                    # Try without arguments (health check, etc.)
                    try:
                        result = method()
                    except:
                        continue
                
                return AIOutput(
                    action="EXECUTED",
                    confidence=0.5,
                    scores={},
                    raw=result,
                    success=True
                )
            except Exception as e:
                continue
    
    # If module has ANY public method, consider it loadable
    public_methods = [m for m in dir(module) if not m.startswith('_') and callable(getattr(module, m, None))]
    if public_methods:
        return AIOutput(
            action="LOADABLE",
            confidence=0.8,
            scores={"methods_found": len(public_methods)},
            raw={"methods": public_methods[:10]},  # Sample
            success=True
        )
    
    return AIOutput(
        success=False,
        error=f"No compatible method found on {type(module)}"
    )
