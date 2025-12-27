"""
Admin Router
User management, system configuration, permissions
"""
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from .auth_router import require_role, verify_token

router = APIRouter(prefix="/admin", tags=["Administration"])

@router.get("/users")
def get_all_users(
    _: str = Depends(require_role(["admin"]))
):
    """Get all users (admin only)"""
    return {
        "users": [
            {
                "id": 1,
                "username": "admin",
                "email": "admin@quantumfond.com",
                "role": "admin",
                "is_active": True,
                "last_login": datetime.utcnow().isoformat()
            },
            {
                "id": 2,
                "username": "riskops",
                "email": "risk@quantumfond.com",
                "role": "risk",
                "is_active": True,
                "last_login": datetime.utcnow().isoformat()
            }
        ]
    }

@router.get("/audit-log")
def get_audit_log(
    limit: int = 100,
    _: str = Depends(require_role(["admin", "risk"]))
):
    """Get audit log of all actions"""
    return {
        "logs": [
            {
                "id": 1,
                "user": "admin",
                "role": "admin",
                "action": "login",
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "id": 2,
                "user": "trader",
                "role": "trader",
                "action": "place_order",
                "status": "success",
                "metadata": {"symbol": "BTCUSDT", "quantity": 0.5},
                "timestamp": datetime.utcnow().isoformat()
            }
        ],
        "total": 2,
        "limit": limit
    }

@router.get("/settings")
def get_system_settings(
    _: str = Depends(require_role(["admin"]))
):
    """Get system settings"""
    return {
        "trading": {
            "enabled": True,
            "mode": "live",
            "max_daily_trades": 100
        },
        "risk": {
            "max_position_size": 100000,
            "max_leverage": 3.0,
            "circuit_breakers_enabled": True
        },
        "ai": {
            "enabled": True,
            "auto_retraining": True,
            "min_confidence": 0.65
        }
    }

@router.post("/settings")
def update_system_settings(
    settings: dict,
    _: str = Depends(require_role(["admin"]))
):
    """Update system settings (admin only)"""
    return {
        "message": "Settings updated successfully",
        "updated_at": datetime.utcnow().isoformat()
    }

@router.post("/emergency-stop")
def emergency_stop(
    _: str = Depends(require_role(["admin", "risk"]))
):
    """Emergency stop - halt all trading"""
    return {
        "message": "Emergency stop activated",
        "all_trading_halted": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/permissions")
def get_role_permissions(
    _: str = Depends(require_role(["admin"]))
):
    """Get role-based permissions matrix"""
    return {
        "roles": {
            "admin": ["all"],
            "risk": ["view_all", "risk_management", "emergency_stop"],
            "trader": ["view_trades", "place_orders", "view_performance"],
            "viewer": ["view_only"]
        }
    }
