"""
JWT Authentication System for Quantum Trader v2.0

Implements:
- JWT token generation and validation
- API key authentication
- Role-based access control (RBAC)
- Token refresh mechanism
- Rate limiting per user/API key
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
import secrets
from functools import wraps

from fastapi import Depends, HTTPException, status, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import redis.asyncio as redis

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer(auto_error=False)

# Redis for rate limiting and token blacklist
redis_client: Optional[redis.Redis] = None


class TokenData(BaseModel):
    """JWT token payload."""
    username: str
    role: str = "user"
    api_key: Optional[str] = None


class User(BaseModel):
    """User model."""
    username: str
    role: str = "user"
    disabled: bool = False


async def init_auth_redis():
    """Initialize Redis connection for auth."""
    global redis_client
    try:
        redis_client = await redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
    except Exception as e:
        print(f"⚠️ Redis not available for rate limiting: {e}")


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)


async def verify_token(token: str) -> TokenData:
    """Verify JWT token and return token data."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "user")
        
        if username is None:
            raise credentials_exception
        
        # Check if token is blacklisted (logged out)
        if redis_client:
            blacklisted = await redis_client.get(f"blacklist:{token}")
            if blacklisted:
                raise credentials_exception
        
        token_data = TokenData(username=username, role=role)
        return token_data
    except JWTError:
        raise credentials_exception


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> TokenData:
    """Get current authenticated user from JWT token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return await verify_token(credentials.credentials)


async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """Get current active user (not disabled)."""
    # In production, check user.disabled from database
    return current_user


async def require_admin(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """Require admin role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def verify_api_key(
    x_api_key: Optional[str] = Header(None)
) -> Optional[TokenData]:
    """Verify API key authentication (alternative to JWT)."""
    if not x_api_key:
        return None
    
    # Check against environment or database
    valid_api_keys = {
        os.getenv("API_KEY_ADMIN"): "admin",
        os.getenv("API_KEY_USER"): "user",
    }
    
    role = valid_api_keys.get(x_api_key)
    if role:
        return TokenData(username="api_key_user", role=role, api_key=x_api_key)
    
    return None


async def get_current_user_or_api_key(
    jwt_user: Optional[TokenData] = Depends(get_current_user),
    api_key_user: Optional[TokenData] = Depends(verify_api_key)
) -> TokenData:
    """Accept either JWT token or API key."""
    if jwt_user:
        return jwt_user
    if api_key_user:
        return api_key_user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required (JWT or API key)"
    )


async def rate_limit_user(
    request: Request,
    user: TokenData = Depends(get_current_user_or_api_key),
    limit: int = 100,
    window: int = 60
) -> TokenData:
    """Rate limit requests per user."""
    if not redis_client:
        return user  # Skip rate limiting if Redis unavailable
    
    key = f"rate_limit:{user.username}:{request.url.path}"
    
    try:
        current = await redis_client.incr(key)
        if current == 1:
            await redis_client.expire(key, window)
        
        if current > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests per {window} seconds"
            )
    except Exception as e:
        print(f"⚠️ Rate limiting error: {e}")
    
    return user


# Middleware for optional authentication
class OptionalAuth:
    """Optional authentication - returns None if not authenticated."""
    
    async def __call__(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        api_key: Optional[str] = Header(None, alias="X-API-Key")
    ) -> Optional[TokenData]:
        """Try to authenticate, return None if not authenticated."""
        try:
            if credentials:
                return await verify_token(credentials.credentials)
            if api_key:
                return await verify_api_key(api_key)
        except HTTPException:
            pass
        return None


optional_auth = OptionalAuth()


# Authentication endpoints will be added to main.py
def create_auth_endpoints():
    """Create authentication endpoints for FastAPI."""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/auth", tags=["authentication"])
    
    class LoginRequest(BaseModel):
        username: str
        password: str
    
    class TokenResponse(BaseModel):
        access_token: str
        refresh_token: str
        token_type: str = "bearer"
    
    @router.post("/login", response_model=TokenResponse)
    async def login(request: LoginRequest):
        """Login and receive JWT tokens."""
        # In production, verify against database
        # For now, accept demo credentials
        demo_users = {
            "admin": ("admin_password_hash", "admin"),
            "user": ("user_password_hash", "user"),
        }
        
        user_data = demo_users.get(request.username)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        # In production: verify_password(request.password, user_data[0])
        # For demo: accept any password
        
        access_token = create_access_token(
            data={"sub": request.username, "role": user_data[1]}
        )
        refresh_token = create_refresh_token(
            data={"sub": request.username, "role": user_data[1]}
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token
        )
    
    @router.post("/refresh", response_model=TokenResponse)
    async def refresh_token(refresh_token: str):
        """Refresh access token using refresh token."""
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != "refresh":
                raise HTTPException(status_code=401, detail="Invalid token type")
            
            username = payload.get("sub")
            role = payload.get("role", "user")
            
            access_token = create_access_token(
                data={"sub": username, "role": role}
            )
            new_refresh_token = create_refresh_token(
                data={"sub": username, "role": role}
            )
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=new_refresh_token
            )
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    
    @router.post("/logout")
    async def logout(user: TokenData = Depends(get_current_user)):
        """Logout (blacklist token)."""
        # In production, blacklist the token in Redis
        return {"message": "Successfully logged out"}
    
    return router
