"""System Control Endpoints - Protected by JWT Authentication"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi_jwt_auth import AuthJWT

router = APIRouter(prefix="/control", tags=["Control & Management"])


@router.post("/retrain")
def retrain(Authorize: AuthJWT = Depends()):
    """Initiate model retraining - Requires admin or analyst role"""
    Authorize.jwt_required()
    claims = Authorize.get_raw_jwt()
    
    if claims.get("role") not in ["admin", "analyst"]:
        raise HTTPException(status_code=403, detail="Not authorized. Admin or analyst role required.")
    
    return {
        "status": "success",
        "message": "Model retraining initiated",
        "user": Authorize.get_jwt_subject(),
        "role": claims.get("role")
    }


@router.post("/heal")
def heal(Authorize: AuthJWT = Depends()):
    """Trigger system healing - Requires admin role only"""
    Authorize.jwt_required()
    claims = Authorize.get_raw_jwt()
    
    if claims.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    return {
        "status": "success",
        "message": "System healing initiated",
        "user": Authorize.get_jwt_subject()
    }


@router.post("/mode")
def switch_mode(Authorize: AuthJWT = Depends()):
    """Switch trading mode - Requires admin or analyst role"""
    Authorize.jwt_required()
    claims = Authorize.get_raw_jwt()
    
    if claims.get("role") not in ["admin", "analyst"]:
        raise HTTPException(status_code=403, detail="Not authorized. Admin or analyst role required.")
    
    return {
        "status": "success",
        "message": "Trading mode switch initiated",
        "user": Authorize.get_jwt_subject(),
        "role": claims.get("role")
    }


@router.get("/status")
def control_status(Authorize: AuthJWT = Depends()):
    """Get control system status - All authenticated users"""
    Authorize.jwt_required()
    
    return {
        "control_status": "operational",
        "authenticated_user": Authorize.get_jwt_subject(),
        "role": Authorize.get_raw_jwt().get("role")
    }
