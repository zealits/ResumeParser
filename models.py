from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import secrets
import string

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class SubscriptionTier(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

# Pydantic Models for API
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: Optional[str] = None  # If None, will be auto-generated
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    api_calls_limit: int = 100  # Monthly limit
    company_name: Optional[str] = None
    contact_person: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    subscription_tier: SubscriptionTier
    api_calls_limit: int
    api_calls_used: int
    company_name: Optional[str]
    contact_person: Optional[str]
    status: UserStatus
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class UsageStats(BaseModel):
    user_id: str
    date: datetime
    api_calls: int
    total_processing_time: float  # in seconds
    successful_calls: int
    failed_calls: int
    file_sizes: List[int]  # in bytes

class AnalyticsResponse(BaseModel):
    total_users: int
    active_users: int
    total_api_calls: int
    total_processing_time: float
    average_response_time: float
    usage_by_tier: Dict[str, int]
    daily_usage: List[Dict[str, Any]]
    top_users: List[Dict[str, Any]]

class RateLimitResponse(BaseModel):
    limit: int
    remaining: int
    reset_time: datetime

# MongoDB Document Models (for Motor/PyMongo)
class UserDocument:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.username = kwargs.get('username')
        self.email = kwargs.get('email')
        self.hashed_password = kwargs.get('hashed_password')
        self.subscription_tier = kwargs.get('subscription_tier', SubscriptionTier.FREE)
        self.api_calls_limit = kwargs.get('api_calls_limit', 100)
        self.api_calls_used = kwargs.get('api_calls_used', 0)
        self.company_name = kwargs.get('company_name')
        self.contact_person = kwargs.get('contact_person')
        self.status = kwargs.get('status', UserStatus.ACTIVE)
        self.role = kwargs.get('role', UserRole.USER)
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.last_login = kwargs.get('last_login')
        self.is_active = kwargs.get('is_active', True)
        self.password_reset_token = kwargs.get('password_reset_token')
        self.password_reset_expires = kwargs.get('password_reset_expires')

    def to_dict(self):
        return {
            'username': self.username,
            'email': self.email,
            'hashed_password': self.hashed_password,
            'subscription_tier': self.subscription_tier,
            'api_calls_limit': self.api_calls_limit,
            'api_calls_used': self.api_calls_used,
            'company_name': self.company_name,
            'contact_person': self.contact_person,
            'status': self.status,
            'role': self.role,
            'created_at': self.created_at,
            'last_login': self.last_login,
            'is_active': self.is_active,
            'password_reset_token': self.password_reset_token,
            'password_reset_expires': self.password_reset_expires
        }

class UsageDocument:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.user_id = kwargs.get('user_id')
        self.date = kwargs.get('date', datetime.utcnow().date())
        self.api_calls = kwargs.get('api_calls', 0)
        self.total_processing_time = kwargs.get('total_processing_time', 0.0)
        self.successful_calls = kwargs.get('successful_calls', 0)
        self.failed_calls = kwargs.get('failed_calls', 0)
        self.file_sizes = kwargs.get('file_sizes', [])
        self.endpoints_used = kwargs.get('endpoints_used', {})  # {"parse-resume": 5, "parse-resume-text": 3}
        self.created_at = kwargs.get('created_at', datetime.utcnow())

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'date': self.date,
            'api_calls': self.api_calls,
            'total_processing_time': self.total_processing_time,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'file_sizes': self.file_sizes,
            'endpoints_used': self.endpoints_used,
            'created_at': self.created_at
        }

class APICallDocument:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.user_id = kwargs.get('user_id')
        self.endpoint = kwargs.get('endpoint')
        self.method = kwargs.get('method')
        self.status_code = kwargs.get('status_code')
        self.response_time = kwargs.get('response_time')  # in seconds
        self.file_size = kwargs.get('file_size')  # in bytes
        self.ip_address = kwargs.get('ip_address')
        self.user_agent = kwargs.get('user_agent')
        self.timestamp = kwargs.get('timestamp', datetime.utcnow())
        self.error_message = kwargs.get('error_message')

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'endpoint': self.endpoint,
            'method': self.method,
            'status_code': self.status_code,
            'response_time': self.response_time,
            'file_size': self.file_size,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp,
            'error_message': self.error_message
        }

# Utility Functions
def generate_password(length: int = 12) -> str:
    """Generate a secure random password"""
    characters = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(characters) for _ in range(length))
    return password

def generate_username_from_email(email: str) -> str:
    """Generate username from email"""
    username = email.split('@')[0]
    # Remove special characters and make lowercase
    username = ''.join(c for c in username if c.isalnum()).lower()
    return username

# Subscription Tier Limits
SUBSCRIPTION_LIMITS = {
    SubscriptionTier.FREE: {
        "api_calls_limit": 100,
        "file_size_limit": 5 * 1024 * 1024,  # 5MB
        "daily_limit": 10
    },
    SubscriptionTier.BASIC: {
        "api_calls_limit": 1000,
        "file_size_limit": 10 * 1024 * 1024,  # 10MB
        "daily_limit": 50
    },
    SubscriptionTier.PREMIUM: {
        "api_calls_limit": 5000,
        "file_size_limit": 25 * 1024 * 1024,  # 25MB
        "daily_limit": 200
    },
    SubscriptionTier.ENTERPRISE: {
        "api_calls_limit": -1,  # Unlimited
        "file_size_limit": 50 * 1024 * 1024,  # 50MB
        "daily_limit": -1  # Unlimited
    }
}

