import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from database import db
from email_service import email_service
from models import SUBSCRIPTION_LIMITS, SubscriptionTier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UsageTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track API usage and enforce rate limits"""
    
    def __init__(self, app, track_endpoints: list = None):
        super().__init__(app)
        self.track_endpoints = track_endpoints or ["/parse-resume", "/parse-resume-text"]
        self.rate_limit_cache = {}  # Simple in-memory cache for rate limiting
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request first to get user from authentication
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Get user from request state (set by FastAPI dependency)
            user = getattr(request.state, 'user', None)
            user_id = user.get('id') if user else None
            
            # Log API call if it's a tracked endpoint and user is authenticated
            if request.url.path in self.track_endpoints and user_id:
                await self.log_api_call(
                    user_id=user_id,
                    request=request,
                    response=response,
                    processing_time=processing_time,
                    success=True
                )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Get user from request state even for failed requests
            user = getattr(request.state, 'user', None)
            user_id = user.get('id') if user else None
            
            # Log failed API call
            if request.url.path in self.track_endpoints and user_id:
                await self.log_api_call(
                    user_id=user_id,
                    request=request,
                    response=None,
                    processing_time=processing_time,
                    success=False,
                    error_message=str(e)
                )
            
            raise e
    
    async def check_rate_limit(self, user_id: str, request: Request) -> Dict[str, Any]:
        """Check if user has exceeded rate limits"""
        try:
            # Get user data
            user = await db.get_user_by_id(user_id)
            if not user:
                return {"allowed": False, "reason": "User not found"}
            
            # Check monthly limit
            api_calls_used = user.get("api_calls_used", 0)
            api_calls_limit = user.get("api_calls_limit", 100)
            
            if api_calls_used >= api_calls_limit:
                # Send usage alert if at 80% or more
                if api_calls_used >= api_calls_limit * 0.8:
                    await self.send_usage_alert(user, api_calls_used / api_calls_limit * 100)
                
                return {
                    "allowed": False,
                    "reason": "Monthly API limit exceeded",
                    "limit": api_calls_limit,
                    "used": api_calls_used
                }
            
            # Check daily limit
            subscription_tier = user.get("subscription_tier", SubscriptionTier.FREE)
            daily_limit = SUBSCRIPTION_LIMITS.get(subscription_tier, {}).get("daily_limit", 10)
            
            if daily_limit != -1:  # -1 means unlimited
                today = datetime.utcnow().date().isoformat()
                cache_key = f"{user_id}:{today}"
                
                # Get today's usage from cache or database
                today_usage = self.rate_limit_cache.get(cache_key, 0)
                
                if today_usage >= daily_limit:
                    return {
                        "allowed": False,
                        "reason": "Daily API limit exceeded",
                        "limit": daily_limit,
                        "used": today_usage
                    }
                
                # Increment today's usage
                self.rate_limit_cache[cache_key] = today_usage + 1
            
            return {"allowed": True}
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return {"allowed": False, "reason": "Error checking limits"}
    
    async def log_api_call(self, user_id: str, request: Request, response, 
                          processing_time: float, success: bool = True, 
                          error_message: str = None):
        """Log API call for analytics"""
        try:
            # Get file size from content-length header
            file_size = 0
            content_length = request.headers.get("content-length")
            if content_length:
                file_size = int(content_length)
            
            # Prepare API call data
            from datetime import datetime
            api_call_data = {
                "user_id": user_id,
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": response.status_code if response else 500,
                "response_time": processing_time,
                "file_size": file_size,
                "ip_address": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "timestamp": datetime.utcnow(),
                "error_message": error_message
            }
            
            # Log to database
            await db.log_api_call(api_call_data)
            
            # Update daily usage stats
            await db.update_daily_usage(
                user_id=user_id,
                endpoint=request.url.path,
                processing_time=processing_time,
                file_size=file_size,
                success=success
            )
            
            # Increment user's API calls count
            if success:
                await db.increment_api_calls(user_id)
            
            # Check if user needs usage alert
            user = await db.get_user_by_id(user_id)
            if user:
                usage_percentage = (user.get("api_calls_used", 0) / user.get("api_calls_limit", 100)) * 100
                if usage_percentage >= 80:  # Alert at 80% usage
                    await self.send_usage_alert(user, usage_percentage)
            
        except Exception as e:
            logger.error(f"Error logging API call: {e}")
    
    async def send_usage_alert(self, user: Dict[str, Any], usage_percentage: float):
        """Send usage alert to user"""
        try:
            # Only send alert once per day
            today = datetime.utcnow().date().isoformat()
            alert_key = f"alert:{user['id']}:{today}"
            
            if alert_key not in self.rate_limit_cache:
                await email_service.send_usage_alert(user, usage_percentage)
                self.rate_limit_cache[alert_key] = True
                
        except Exception as e:
            logger.error(f"Error sending usage alert: {e}")

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log requests for monitoring"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")
        
        response = await call_next(request)
        
        # Log response
        processing_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} in {processing_time:.3f}s")
        
        return response

class FileSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce file size limits"""
    
    def __init__(self, app, max_file_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_file_size = max_file_size
    
    async def dispatch(self, request: Request, call_next):
        # Check file size for file uploads
        if request.method == "POST" and "multipart/form-data" in request.headers.get("content-type", ""):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_file_size:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": "File too large",
                        "message": f"File size exceeds {self.max_file_size / (1024*1024):.1f}MB limit",
                        "max_size": self.max_file_size,
                        "received_size": int(content_length)
                    }
                )
        
        return await call_next(request)

class CORSHeadersMiddleware(BaseHTTPMiddleware):
    """Enhanced CORS middleware with security"""
    
    def __init__(self, app, allowed_origins: list = None, allowed_methods: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    async def dispatch(self, request: Request, call_next):
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = JSONResponse(content={})
        else:
            response = await call_next(request)
        
        # Add CORS headers
        origin = request.headers.get("origin")
        if origin in self.allowed_origins or "*" in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "86400"
        
        return response
