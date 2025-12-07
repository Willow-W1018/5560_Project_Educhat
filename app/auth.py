"""
User Authentication Module

Features:
- Simple user registration and login
- JWT token-based authentication
- API key authentication for programmatic access
- Rate limiting support
"""

import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel
import json

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Simple JWT secret (in production, use environment variable)
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# In-memory user storage (in production, use a database)
# Format: {username: {password_hash, api_key, created_at, usage_count}}
users_db: Dict[str, Dict] = {}

# File path for persistent storage
USERS_FILE = "data/users.json"


def load_users():
    """Load users from file"""
    global users_db
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                users_db = json.load(f)
        except Exception:
            users_db = {}


def save_users():
    """Save users to file"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users_db, f, indent=2)


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a random API key"""
    return f"edu_{secrets.token_urlsafe(32)}"


def generate_token(username: str) -> str:
    """
    Generate a simple JWT-like token
    
    In production, use a proper JWT library like python-jose
    """
    import base64
    import json
    import hmac
    
    # Create payload
    payload = {
        "sub": username,
        "exp": (datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)).isoformat(),
        "iat": datetime.utcnow().isoformat()
    }
    
    # Encode payload
    payload_bytes = json.dumps(payload).encode()
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode().rstrip('=')
    
    # Create signature
    signature = hmac.new(
        JWT_SECRET.encode(),
        payload_b64.encode(),
        hashlib.sha256
    ).hexdigest()[:32]
    
    return f"{payload_b64}.{signature}"


def verify_token(token: str) -> Optional[str]:
    """
    Verify token and return username if valid
    
    Returns None if token is invalid
    """
    import base64
    import json
    import hmac
    
    try:
        parts = token.split('.')
        if len(parts) != 2:
            return None
        
        payload_b64, signature = parts
        
        # Verify signature
        expected_signature = hmac.new(
            JWT_SECRET.encode(),
            payload_b64.encode(),
            hashlib.sha256
        ).hexdigest()[:32]
        
        if not hmac.compare_digest(signature, expected_signature):
            return None
        
        # Decode payload
        padding = 4 - len(payload_b64) % 4
        payload_b64 += '=' * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        
        # Check expiration
        exp = datetime.fromisoformat(payload['exp'])
        if datetime.utcnow() > exp:
            return None
        
        return payload['sub']
    except Exception:
        return None


# Pydantic models for request/response
class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRATION_HOURS * 3600
    api_key: Optional[str] = None


class UserInfo(BaseModel):
    username: str
    created_at: str
    usage_count: int
    has_api_key: bool


def register_user(user_data: UserRegister) -> TokenResponse:
    """
    Register a new user
    
    Args:
        user_data: Username and password
        
    Returns:
        Token response with access token and API key
        
    Raises:
        HTTPException if username already exists
    """
    load_users()
    
    if user_data.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    if len(user_data.username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be at least 3 characters"
        )
    
    if len(user_data.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters"
        )
    
    # Create user
    api_key = generate_api_key()
    users_db[user_data.username] = {
        "password_hash": hash_password(user_data.password),
        "api_key": api_key,
        "created_at": datetime.utcnow().isoformat(),
        "usage_count": 0
    }
    
    save_users()
    
    # Generate token
    token = generate_token(user_data.username)
    
    return TokenResponse(
        access_token=token,
        api_key=api_key
    )


def login_user(user_data: UserLogin) -> TokenResponse:
    """
    Login user and return token
    
    Args:
        user_data: Username and password
        
    Returns:
        Token response
        
    Raises:
        HTTPException if credentials are invalid
    """
    load_users()
    
    if user_data.username not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    user = users_db[user_data.username]
    
    if user["password_hash"] != hash_password(user_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Generate token
    token = generate_token(user_data.username)
    
    return TokenResponse(access_token=token)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    api_key: str = Depends(api_key_header)
) -> Optional[str]:
    """
    Get current user from token or API key
    
    This is a dependency for protected endpoints
    
    Returns:
        Username if authenticated, None otherwise
    """
    load_users()
    
    # Try bearer token first
    if credentials:
        username = verify_token(credentials.credentials)
        if username:
            # Update usage count
            if username in users_db:
                users_db[username]["usage_count"] += 1
                save_users()
            return username
    
    # Try API key
    if api_key:
        for username, user_data in users_db.items():
            if user_data.get("api_key") == api_key:
                users_db[username]["usage_count"] += 1
                save_users()
                return username
    
    return None


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    api_key: str = Depends(api_key_header)
) -> str:
    """
    Require authentication - raises exception if not authenticated
    
    Use this dependency for protected endpoints
    """
    username = get_current_user(credentials, api_key)
    
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return username


def get_user_info(username: str) -> UserInfo:
    """Get user information"""
    load_users()
    
    if username not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user = users_db[username]
    
    return UserInfo(
        username=username,
        created_at=user["created_at"],
        usage_count=user["usage_count"],
        has_api_key=bool(user.get("api_key"))
    )


def regenerate_api_key(username: str) -> str:
    """Regenerate API key for user"""
    load_users()
    
    if username not in users_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    new_key = generate_api_key()
    users_db[username]["api_key"] = new_key
    save_users()
    
    return new_key


# Initialize - load users on module import
load_users()
