"""System Control Endpoints - Protected by JWT Authentication"""
from fastapi import APIRouter, Depends, HTTPException, status
from auth.auth_router import verify_token, TokenData

router = APIRouter(prefix="/control", tags=["Control & Management"])


@router.post("/retrain")
def retrain(token_data: TokenData = Depends(verify_token)):
    """Initiate model retraining - Requires admin or analyst role"""
    if token_data.role not in ["admin", "analyst"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized. Admin or analyst role required."
        )
    
    return {
        "status": "success",
        "message": "Model retraining initiated",
        "user": token_data.username,
        "role": token_data.role
    }


@router.post("/heal")
def heal(token_data: TokenData = Depends(verify_token)):
    """Trigger system healing - Requires admin role only"""
    if token_data.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return {
        "status": "success",
        "message": "System healing initiated",
        "user": token_data.username
    }


@router.post("/mode")
def switch_mode(token_data: TokenData = Depends(verify_token)):
    """Switch trading mode - Requires admin or analyst role"""
    if token_data.role not in ["admin", "analyst"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized. Admin or analyst role required."
        )
    
    return {
        "status": "success",
        "message": "Trading mode switch initiated",
        "user": token_data.username,
        "role": token_data.role
    }


@router.get("/status")
def control_status(token_data: TokenData = Depends(verify_token)):
    """Get control system status - All authenticated users"""
    return {
        "control_status": "operational",
        "authenticated_user": token_data.username,
        "role": token_data.role
    }
