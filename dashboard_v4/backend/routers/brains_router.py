from fastapi import APIRouter
import random
import time

router = APIRouter(prefix="/brains", tags=["Brains & Strategy"])

def generate_brain_state():
    """Generate simulated brain state with mode, confidence, and transition log"""
    modes = ["OPTIMIZE", "EXPAND", "STABILIZE", "EMERGENCY"]
    confidence = round(random.uniform(0.7, 0.99), 3)
    mode = random.choice(modes)
    transition_log = [
        {"timestamp": time.time()-3600, "from": "OPTIMIZE", "to": "EXPAND"},
        {"timestamp": time.time()-1800, "from": "EXPAND", "to": "STABILIZE"}
    ]
    return {"mode": mode, "confidence": confidence, "transition_log": transition_log}

@router.get("/state")
def get_brain_states():
    """
    Get current operational state for CEO, Strategy, and Risk Brains
    
    Returns:
        Brain states with mode, confidence levels, and recent transitions
    """
    ceo = generate_brain_state()
    strategy = generate_brain_state()
    risk = generate_brain_state()
    return {
        "CEO_Brain": ceo,
        "Strategy_Brain": strategy,
        "Risk_Brain": risk,
        "system_timestamp": time.time()
    }
