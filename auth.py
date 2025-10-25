import os
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models import UserRole, SubscriptionTier, UserStatus, generate_password
from database import db

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-this-in-production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()

class AuthManager:
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM
        self.token_expire_minutes = ACCESS_TOKEN_EXPIRE_MINUTES

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        # Truncate password if it's longer than 72 bytes (bcrypt limit)
        if len(password.encode('utf-8')) > 72:
            password = password[:72]
        return pwd_context.hash(password)

    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user with username and password"""
        user = await db.get_user_by_username(username)
        if not user:
            return None
        
        if not self.verify_password(password, user["hashed_password"]):
            return None
        
        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        if user.get("status") != UserStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is suspended"
            )
        
        return user

    async def create_user_account(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user account (admin only)"""
        # Check if username already exists
        existing_user = await db.get_user_by_username(user_data["username"])
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Check if email already exists
        existing_email = await db.get_user_by_email(user_data["email"])
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
        
        # Generate password if not provided
        password = user_data.get("password")
        if not password:
            password = generate_password()
        
        # Hash password
        hashed_password = self.get_password_hash(password)
        
        # Create user document
        user_doc = {
            "username": user_data["username"],
            "email": user_data["email"],
            "hashed_password": hashed_password,
            "subscription_tier": user_data.get("subscription_tier", SubscriptionTier.FREE),
            "api_calls_limit": user_data.get("api_calls_limit", 100),
            "api_calls_used": 0,
            "company_name": user_data.get("company_name"),
            "contact_person": user_data.get("contact_person"),
            "status": UserStatus.ACTIVE,
            "role": UserRole.USER,
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        
        # Save to database
        user_id = await db.create_user(user_doc)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user account"
            )
        
        return {
            "user_id": user_id,
            "username": user_data["username"],
            "email": user_data["email"],
            "password": password,  # Return plain password for email
            "subscription_tier": user_data.get("subscription_tier", SubscriptionTier.FREE),
            "api_calls_limit": user_data.get("api_calls_limit", 100)
        }

    async def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """Login a user and return JWT token"""
        user = await self.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update last login
        await db.update_user_login(user["id"])
        
        # Create access token
        token_data = {
            "sub": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user.get("role", UserRole.USER),
            "subscription_tier": user.get("subscription_tier", SubscriptionTier.FREE)
        }
        access_token = self.create_access_token(token_data)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.token_expire_minutes * 60,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "subscription_tier": user.get("subscription_tier", SubscriptionTier.FREE),
                "api_calls_limit": user.get("api_calls_limit", 100),
                "api_calls_used": user.get("api_calls_used", 0),
                "company_name": user.get("company_name"),
                "contact_person": user.get("contact_person"),
                "status": user.get("status", UserStatus.ACTIVE),
                "created_at": user.get("created_at"),
                "last_login": datetime.utcnow(),
                "is_active": user.get("is_active", True)
            }
        }

    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        """Get current authenticated user from JWT token"""
        token = credentials.credentials
        payload = self.verify_token(token)
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = await db.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        return user

    async def get_current_admin(self, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        """Get current user and verify admin role"""
        if current_user.get("role") != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        return current_user

    async def check_user_permissions(self, user: Dict[str, Any], required_tier: Optional[SubscriptionTier] = None) -> bool:
        """Check if user has required permissions"""
        if not user.get("is_active", True):
            return False
        
        if user.get("status") != UserStatus.ACTIVE:
            return False
        
        if required_tier:
            user_tier = user.get("subscription_tier", SubscriptionTier.FREE)
            tier_hierarchy = {
                SubscriptionTier.FREE: 0,
                SubscriptionTier.BASIC: 1,
                SubscriptionTier.PREMIUM: 2,
                SubscriptionTier.ENTERPRISE: 3
            }
            return tier_hierarchy.get(user_tier, 0) >= tier_hierarchy.get(required_tier, 0)
        
        return True

# Global auth manager instance
auth_manager = AuthManager()

# Dependency functions for FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """FastAPI dependency to get current user"""
    return await auth_manager.get_current_user(credentials)

async def get_current_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """FastAPI dependency to get current admin user"""
    return await auth_manager.get_current_admin(current_user)

async def require_subscription_tier(tier: SubscriptionTier):
    """FastAPI dependency factory for subscription tier requirements"""
    async def check_tier(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if not await auth_manager.check_user_permissions(current_user, tier):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Subscription tier '{tier.value}' or higher required"
            )
        return current_user
    return check_tier
