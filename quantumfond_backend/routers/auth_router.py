"""
Authentication Router
Handles JWT-based authentication and role-based access control
"""
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import jwt

router = APIRouter(prefix="/auth", tags=["Authentication"])

# JWT Configuration
JWT_SECRET = "QUANTUMFOND_SECRET_987"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

security = HTTPBearer()

# Request/Response Models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

# Simplified static users (will link to database later)
USERS_DB = {
    "admin": {
        "password": "AdminPass123",
        "role": "admin",
        "email": "admin@quantumfond.com"
    },
    "riskops": {
        "password": "Risk123",
        "role": "risk",
        "email": "risk@quantumfond.com"
    },
    "trader": {
        "password": "Trade123",
        "role": "trader",
        "email": "trader@quantumfond.com"
    },
    "viewer": {
        "password": "View123",
        "role": "viewer",
        "email": "viewer@quantumfond.com"
    }
}

def create_access_token(username: str, role: str, email: str) -> str:
    """Create JWT access token"""
    expiration = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "sub": username,
        "role": role,
        "email": email,
        "exp": expiration
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify and decode JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@router.post("/login", response_model=LoginResponse)
def login(user: LoginRequest):
    """
    Login endpoint - returns JWT token
    
    Roles:
    - admin: Full system access
    - risk: Risk management and monitoring
    - trader: Trading operations
    - viewer: Read-only access
    """
    # Validate credentials
    if user.username not in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    user_data = USERS_DB[user.username]
    if user_data["password"] != user.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Create access token with user claims
    access_token = create_access_token(
        username=user.username,
        role=user_data["role"],
        email=user_data["email"]
    )
    
    return LoginResponse(
        access_token=access_token,
        role=user_data["role"],
        username=user.username
    )

@router.get("/me")
def get_current_user(payload: dict = Depends(verify_token)):
    """Get current user information"""
    return {
        "username": payload.get("sub"),
        "role": payload.get("role"),
        "email": payload.get("email")
    }

@router.post("/logout")
def logout(payload: dict = Depends(verify_token)):
    """Logout - invalidate token (client should remove token)"""
    return {"message": "Successfully logged out"}

# Helper function for role-based access control
def require_role(required_roles: list):
    """Dependency to check user role"""
    def role_checker(payload: dict = Depends(verify_token)):
        user_role = payload.get("role")
        
        if user_role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {user_role} not authorized for this action"
            )
        return user_role
    return role_checker
