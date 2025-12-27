"""System Control Endpoints - Protected by JWT Authentication"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from auth.auth_router import verify_token, TokenData
from db.connection import get_db
from db.models import ControlLog
from datetime import datetime

router = APIRouter(prefix="/control", tags=["Control & Management"])


def log_action(db: Session, user: str, role: str, action: str, status_msg: str = "success", details: str = None):
    """Log control action to database"""
    try:
        log_entry = ControlLog(
            action=action,
            user=user,
            role=role,
            status=status_msg,
            details=details,
            timestamp=datetime.utcnow()
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        return log_entry
    except Exception as e:
        print(f"⚠️ Failed to log action: {e}")
        db.rollback()
        return None


@router.post("/retrain")
def retrain(token_data: TokenData = Depends(verify_token), db: Session = Depends(get_db)):
    """Initiate model retraining - Requires admin or analyst role"""
    if token_data.role not in ["admin", "analyst"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized. Admin or analyst role required."
        )
    
    # Log action to database
    log_action(db, token_data.username, token_data.role, "retrain", "triggered")
    
    return {
        "status": "success",
        "message": "Model retraining initiated",
        "user": token_data.username,
        "role": token_data.role
    }


@router.post("/heal")
def heal(token_data: TokenData = Depends(verify_token), db: Session = Depends(get_db)):
    """Trigger system healing - Requires admin role only"""
    if token_data.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    # Log action to database
    log_action(db, token_data.username, token_data.role, "heal", "executed")
    
    return {
        "status": "success",
        "message": "System healing initiated",
        "user": token_data.username
    }


@router.post("/mode")
def switch_mode(mode: str, token_data: TokenData = Depends(verify_token), db: Session = Depends(get_db)):
    """Switch trading mode - Requires admin or analyst role"""
    if token_data.role not in ["admin", "analyst"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized. Admin or analyst role required."
        )
    
    # Validate mode
    valid_modes = ["NORMAL", "SAFE", "OPTIMIZE"]
    if mode not in valid_modes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid mode. Must be one of: {', '.join(valid_modes)}"
        )
    
    # Log action to database
    log_action(db, token_data.username, token_data.role, "mode_switch", "changed", details=mode)
    
    return {
        "status": "success",
        "message": f"System mode changed to {mode}",
        "user": token_data.username,
        "role": token_data.role,
        "mode": mode
    }


@router.get("/status")
def control_status(token_data: TokenData = Depends(verify_token)):
    """Get control system status - All authenticated users"""
    return {
        "control_status": "operational",
        "authenticated_user": token_data.username,
        "role": token_data.role
    }


@router.get("/logs")
def get_control_logs(limit: int = 50, token_data: TokenData = Depends(verify_token), db: Session = Depends(get_db)):
    """Get recent control action logs - All authenticated users"""
    try:
        logs = db.query(ControlLog).order_by(ControlLog.timestamp.desc()).limit(limit).all()
        return {
            "total": len(logs),
            "logs": [
                {
                    "id": log.id,
                    "action": log.action,
                    "user": log.user,
                    "role": log.role,
                    "status": log.status,
                    "details": log.details,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None
                }
                for log in logs
            ]
        }
    except Exception as e:
        return {"error": "Database not connected", "logs": []}
