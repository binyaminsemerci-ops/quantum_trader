"""Authentication and Authorization Module"""
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from passlib.hash import bcrypt
from jose import JWTError, jwt

router = APIRouter(prefix="/auth", tags=["Auth & Security"])
security = HTTPBearer()

# JWT Configuration
SECRET_KEY = "QuantumSuperSecretKeyReplaceLater"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

# Example in-memory user store (in production, use a database)
# Note: Passwords are pre-hashed for bcrypt compatibility
USERS = {
    "admin": {
        "password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYIxKTHLG/C",  # AdminPass123
        "role": "admin"
    },
    "analyst": {
        "password": "$2b$12$X2wJF/7KtJq9r5RlXzKvFuGVqyJ7z3YvRrXE8Rm5Z0KmQ3x4P8z2G",  # AnalystPass456
        "role": "analyst"
    },
    "viewer": {
        "password": "$2b$12$JnL0C8KpYvHxN5ZqYvWxYeY3Z6T9RxM1WzJ7E3Qx4Y5P6L8K9M0N2",  # ViewerPass789
        "role": "viewer"
    },
}


class User(BaseModel):
    username: str
    password: str


class TokenData(BaseModel):
    username: str
    role: str


def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Verify JWT token and extract user data"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        
        if username is None or role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return TokenData(username=username, role=role)
    
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/login")
def login(user: User):
    """Login endpoint - returns JWT token and user role"""
    if user.username not in USERS or not bcrypt.verify(user.password, USERS[user.username]["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    access_token = create_access_token(
        data={"sub": user.username, "role": USERS[user.username]["role"]}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": USERS[user.username]["role"],
        "username": user.username
    }


@router.get("/whoami")
def whoami(token_data: TokenData = Depends(verify_token)):
    """Get current user information from JWT token"""
    return {"user": token_data.username, "role": token_data.role}


@router.post("/logout")
def logout():
    """Logout endpoint - client should remove token from storage"""
    return {"message": "Logged out successfully. Remove token from client storage."}
