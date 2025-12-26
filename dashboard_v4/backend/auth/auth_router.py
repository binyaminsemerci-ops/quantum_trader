"""Authentication and Authorization Module"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from passlib.hash import bcrypt
from fastapi_jwt_auth import AuthJWT
from fastapi_jwt_auth.exceptions import AuthJWTException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/auth", tags=["Auth & Security"])

# Example in-memory user store (in production, use a database)
USERS = {
    "admin": {"password": bcrypt.hash("AdminPass123"), "role": "admin"},
    "analyst": {"password": bcrypt.hash("AnalystPass456"), "role": "analyst"},
    "viewer": {"password": bcrypt.hash("ViewerPass789"), "role": "viewer"},
}


class User(BaseModel):
    username: str
    password: str


class Settings(BaseModel):
    authjwt_secret_key: str = "QuantumSuperSecretKeyReplaceLater"
    authjwt_access_token_expires: int = 7200  # 2 hours


@AuthJWT.load_config
def get_config():
    return Settings()


@router.exception_handler(AuthJWTException)
def authjwt_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@router.post("/login")
def login(user: User, Authorize: AuthJWT = Depends()):
    """Login endpoint - returns JWT token and user role"""
    if user.username not in USERS or not bcrypt.verify(user.password, USERS[user.username]["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = Authorize.create_access_token(
        subject=user.username,
        user_claims={"role": USERS[user.username]["role"]}
    )
    
    return {
        "access_token": access_token,
        "role": USERS[user.username]["role"],
        "username": user.username
    }


@router.get("/whoami")
def whoami(Authorize: AuthJWT = Depends()):
    """Get current user information from JWT token"""
    Authorize.jwt_required()
    current_user = Authorize.get_jwt_subject()
    role = Authorize.get_raw_jwt()["role"]
    return {"user": current_user, "role": role}


@router.post("/logout")
def logout():
    """Logout endpoint - client should remove token from storage"""
    return {"message": "Logged out successfully. Remove token from client storage."}
