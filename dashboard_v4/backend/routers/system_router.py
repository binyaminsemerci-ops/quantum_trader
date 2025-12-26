from fastapi import APIRouter
from schemas import SystemHealth

router = APIRouter(prefix="/system", tags=["System"])

@router.get("/health", response_model=SystemHealth)
def get_system_health():
    """Get system health metrics including CPU, RAM, and container status"""
    return SystemHealth(
        cpu=47.5,
        ram=62.3,
        uptime=109384,
        containers=6
    )
