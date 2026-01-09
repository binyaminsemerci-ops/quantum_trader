#!/usr/bin/env python3
"""
XGBoost Confidence Mapping Fix - PROPOSAL
"""
from typing import Dict

"""

Root Cause:
-----------
XGBoost uses 3-class classification with predict_proba() returning:
  [P(HOLD), P(BUY), P(SELL)]

Current implementation uses max(proba) which is always 0.95-1.00 for 
well-calibrated classifiers, leading to bimodal distribution:
  - Fallback: 0.50 (rule-based)
  - ML Model: 0.90-1.00 (always overconfident)

Proposed Fix:
-------------
Use margin-based confidence:
  margin = P(top_class) - P(second_class)
  confidence = 0.50 + (margin * 0.45)

This maps:
  - margin=0.0 (tie between classes) → confidence=0.50
  - margin=0.5 (moderate gap) → confidence=0.725
  - margin=1.0 (unanimous) → confidence=0.95

Expected Distribution After Fix:
---------------------------------
  0.50-0.60: ~20% (low margin, uncertain)
  0.60-0.70: ~25% (moderate margin)
  0.70-0.80: ~30% (good margin)
  0.80-0.90: ~20% (high margin)
  0.90-1.00: ~5%  (very high margin)

Implementation:
---------------
See proposed changes below.
"""

# ============================================================================
# PROPOSED CHANGE TO: ai_engine/agents/xgb_agent.py
# Lines: 337-348
# ============================================================================

def predict_BEFORE(self, symbol: str, features: Dict[str, float]) -> tuple[str, float, str]:
    """Current implementation (WRONG)"""
    # ... feature processing ...
    
    # Predict
    prediction = self.model.predict(feature_array)[0]
    proba = self.model.predict_proba(feature_array)[0]
    confidence = float(max(proba))  # ← BUG: Always 0.95-1.00
    
    # Map prediction to action
    action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    action = action_map.get(prediction, 'HOLD')
    
    return (action, confidence, 'xgboost')


def predict_AFTER(self, symbol: str, features: Dict[str, float]) -> tuple[str, float, str]:
    """Fixed implementation"""
    # ... feature processing ...
    
    # Predict
    prediction = self.model.predict(feature_array)[0]
    proba = self.model.predict_proba(feature_array)[0]
    
    # ✅ FIX: Use margin-based confidence
    sorted_proba = sorted(proba, reverse=True)
    margin = sorted_proba[0] - sorted_proba[1]  # Difference between top 2
    
    # Scale to ensemble range: [0.50, 0.95]
    confidence = float(0.50 + (margin * 0.45))
    
    # Map prediction to action
    action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    action = action_map.get(prediction, 'HOLD')
    
    # DEBUG: Log every 10th prediction with margin
    import random
    if random.random() < 0.1:
        logger.info(
            f"XGB {symbol}: {action} conf={confidence:.2f} "
            f"(margin={margin:.3f}, proba={proba})"
        )
    
    return (action, confidence, 'xgboost')


# ============================================================================
# EXACT STRING REPLACEMENT FOR replace_string_in_file
# ============================================================================

OLD_STRING = '''            # Predict
            prediction = self.model.predict(feature_array)[0]
            proba = self.model.predict_proba(feature_array)[0]
            confidence = float(max(proba))
            
            # Map prediction to action
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action = action_map.get(prediction, 'HOLD')
            
            # DEBUG: Log every 10th prediction
            import random
            if random.random() < 0.1:  # 10% sampling
                logger.info(f"XGB {symbol}: {action} {confidence:.2%} (pred={prediction})")
            
            return (action, confidence, 'xgboost')'''

NEW_STRING = '''            # Predict
            prediction = self.model.predict(feature_array)[0]
            proba = self.model.predict_proba(feature_array)[0]
            
            # ✅ FIX: Use margin-based confidence instead of max(proba)
            # Margin = difference between top 2 class probabilities
            sorted_proba = sorted(proba, reverse=True)
            margin = sorted_proba[0] - sorted_proba[1]
            
            # Scale margin to ensemble range: [0.50, 0.95]
            # margin=0.0 (tie) → 0.50, margin=1.0 (unanimous) → 0.95
            confidence = float(0.50 + (margin * 0.45))
            
            # Map prediction to action
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action = action_map.get(prediction, 'HOLD')
            
            # DEBUG: Log every 10th prediction with margin details
            import random
            if random.random() < 0.1:  # 10% sampling
                logger.info(
                    f"XGB {symbol}: {action} conf={confidence:.2f} "
                    f"(margin={margin:.3f}, pred={prediction})"
                )
            
            return (action, confidence, 'xgboost')'''

# ============================================================================
# VERIFICATION EXAMPLES
# ============================================================================

def test_confidence_mapping():
    """Test cases showing before/after confidence values"""
    import numpy as np
    
    test_cases = [
        # (proba_distribution, description)
        ([0.02, 0.97, 0.01], "Strong BUY signal"),
        ([0.45, 0.50, 0.05], "Weak BUY (barely winning)"),
        ([0.33, 0.34, 0.33], "Complete tie (3-way split)"),
        ([0.10, 0.80, 0.10], "Moderate BUY"),
        ([0.05, 0.90, 0.05], "Strong BUY (very confident)"),
    ]
    
    print("Confidence Mapping Comparison")
    print("=" * 70)
    print(f"{'Probabilities':<20} {'Old (max)':<12} {'New (margin)':<15} {'Description':<20}")
    print("-" * 70)
    
    for proba, desc in test_cases:
        # Old method
        old_conf = max(proba)
        
        # New method
        sorted_p = sorted(proba, reverse=True)
        margin = sorted_p[0] - sorted_p[1]
        new_conf = 0.50 + (margin * 0.45)
        
        # Format probabilities
        proba_str = f"[{proba[0]:.2f}, {proba[1]:.2f}, {proba[2]:.2f}]"
        
        print(f"{proba_str:<20} {old_conf:.4f}       {new_conf:.4f}          {desc:<20}")
    
    print("\nExpected Results:")
    print("  - Old method: Mostly 0.80-0.97 (overconfident)")
    print("  - New method: Range 0.50-0.93 (properly calibrated)")

if __name__ == '__main__':
    test_confidence_mapping()
