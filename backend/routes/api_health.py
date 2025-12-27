"""
API Health Monitoring Endpoint
Exposes health status of all external API connections
"""

from fastapi import APIRouter
from typing import Dict, Any
import logging

try:
    from backend.api_bulletproof import get_all_api_health
except ImportError:
    from api_bulletproof import get_all_api_health

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/external-apis/health")
async def get_external_apis_health() -> Dict[str, Any]:
    """
    Get health status of all external API connections
    
    Returns detailed statistics including:
    - Success/failure rates
    - Circuit breaker status
    - Average response times
    - Recent errors
    - Overall system health
    """
    try:
        health = get_all_api_health()
        return {
            "status": "success",
            "data": health
        }
    except Exception as e:
        logger.error(f"Failed to get API health: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
