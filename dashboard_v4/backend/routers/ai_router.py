from fastapi import APIRouter
from schemas import AIStatus

router = APIRouter(prefix="/ai", tags=["AI"])

@router.get("/status", response_model=AIStatus)
def get_ai_status():
    """Get AI engine status with model performance metrics"""
    return AIStatus(
        accuracy=0.72,
        sharpe=1.14,
        latency=184,
        models=["XGB", "LGBM", "N-HiTS", "TFT"]
    )
