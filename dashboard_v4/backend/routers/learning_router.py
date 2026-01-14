from fastapi import APIRouter
import random
import time

router = APIRouter(prefix="/learning", tags=["Continuous Learning"])

def simulate_training_log():
    """Simulate training statistics for a model"""
    return {
        "model": random.choice(["XGB", "LGBM", "N-HiTS", "TFT"]),
        "version": f"v{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}",
        "last_loss": round(random.uniform(0.010, 0.035), 4),
        "validation_score": round(random.uniform(0.65, 0.85), 3),
        "last_trained": time.time() - random.randint(3600, 86400),
        "training_time": random.randint(600, 2400),
    }

@router.get("/status")
async def get_learning_status():
    """
    Get continuous learning manager status
    
    Returns:
        Model training status with retraining recommendations based on:
        - Validation score < 0.70
        - Time since last training > 12 hours
    """
    log = simulate_training_log()
    
    # Basic retrain suggestion logic
    retrain_needed = log["validation_score"] < 0.70 or (time.time() - log["last_trained"]) > 43200
    suggestion = "Retrain recommended" if retrain_needed else "Stable"
    
    return {
        "timestamp": time.time(),
        "model": log["model"],
        "version": log["version"],
        "last_loss": log["last_loss"],
        "validation_score": log["validation_score"],
        "last_trained": log["last_trained"],
        "training_time": log["training_time"],
        "suggestion": suggestion
    }
