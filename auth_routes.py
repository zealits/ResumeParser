from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from models import (
    UserCreate, UserLogin, UserResponse, TokenResponse, 
    AnalyticsResponse, RateLimitResponse, SubscriptionTier, UserStatus
)
from auth import auth_manager, get_current_user, get_current_admin
from database import db
from email_service import email_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login", response_model=TokenResponse, summary="User Login")
async def login(user_credentials: UserLogin):
    """
    Authenticate user and return JWT token.
    
    - **username**: User's username
    - **password**: User's password
    """
    try:
        result = await auth_manager.login_user(
            username=user_credentials.username,
            password=user_credentials.password
        )
        
        logger.info(f"User {user_credentials.username} logged in successfully")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error for user {user_credentials.username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )

@router.post("/admin/create-user", response_model=Dict[str, Any], summary="Create User Account (Admin Only)")
async def create_user(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    current_admin: Dict[str, Any] = Depends(get_current_admin)
):
    """
    Create a new user account (Admin only).
    
    - **username**: Unique username for the user
    - **email**: User's email address
    - **password**: Optional password (auto-generated if not provided)
    - **subscription_tier**: User's subscription tier
    - **api_calls_limit**: Monthly API calls limit
    - **company_name**: User's company name
    - **contact_person**: Contact person name
    """
    try:
        # Create user account
        result = await auth_manager.create_user_account(user_data.dict())
        
        # Send credentials via email in background
        background_tasks.add_task(
            email_service.send_user_credentials,
            result
        )
        
        # Send admin notification
        background_tasks.add_task(
            email_service.send_admin_notification,
            "New User Account Created",
            f"New user account created for {result['username']} ({result['email']}) with {result['subscription_tier']} subscription."
        )
        
        logger.info(f"Admin {current_admin['username']} created user account: {result['username']}")
        
        return {
            "success": True,
            "message": "User account created successfully. Credentials sent via email.",
            "user_id": result["user_id"],
            "username": result["username"],
            "email": result["email"],
            "subscription_tier": result["subscription_tier"],
            "api_calls_limit": result["api_calls_limit"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )

@router.get("/profile", response_model=UserResponse, summary="Get User Profile")
async def get_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current user's profile information.
    """
    try:
        return UserResponse(
            id=current_user["id"],
            username=current_user["username"],
            email=current_user["email"],
            subscription_tier=current_user.get("subscription_tier", SubscriptionTier.FREE),
            api_calls_limit=current_user.get("api_calls_limit", 100),
            api_calls_used=current_user.get("api_calls_used", 0),
            company_name=current_user.get("company_name"),
            contact_person=current_user.get("contact_person"),
            status=current_user.get("status", UserStatus.ACTIVE),
            created_at=current_user.get("created_at", datetime.utcnow()),
            last_login=current_user.get("last_login"),
            is_active=current_user.get("is_active", True)
        )
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )

@router.get("/usage-stats", summary="Get User Usage Statistics")
async def get_user_usage_stats(
    days: int = 30,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get user's usage statistics for the last N days.
    
    - **days**: Number of days to retrieve stats for (default: 30)
    """
    try:
        if days > 365:  # Limit to 1 year
            days = 365
        
        usage_stats = await db.get_user_usage_stats(current_user["id"], days)
        
        # Calculate totals
        total_calls = sum(stat.get("api_calls", 0) for stat in usage_stats)
        total_processing_time = sum(stat.get("total_processing_time", 0) for stat in usage_stats)
        successful_calls = sum(stat.get("successful_calls", 0) for stat in usage_stats)
        failed_calls = sum(stat.get("failed_calls", 0) for stat in usage_stats)
        
        return {
            "user_id": current_user["id"],
            "username": current_user["username"],
            "subscription_tier": current_user.get("subscription_tier", SubscriptionTier.FREE),
            "period_days": days,
            "total_api_calls": total_calls,
            "total_processing_time": round(total_processing_time, 2),
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": round((successful_calls / max(total_calls, 1)) * 100, 2),
            "average_response_time": round(total_processing_time / max(total_calls, 1), 3),
            "daily_breakdown": usage_stats,
            "remaining_calls": current_user.get("api_calls_limit", 100) - current_user.get("api_calls_used", 0)
        }
        
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get usage statistics"
        )

@router.get("/rate-limit", response_model=RateLimitResponse, summary="Check Rate Limits")
async def check_rate_limits(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Check current user's rate limits and usage.
    """
    try:
        rate_limit_result = await db.check_rate_limit(current_user["id"])
        
        if not rate_limit_result.get("allowed", True):
            return RateLimitResponse(
                limit=rate_limit_result.get("limit", 0),
                remaining=0,
                reset_time=datetime.utcnow() + timedelta(days=1)  # Reset at midnight
            )
        
        # Calculate remaining calls
        api_calls_used = current_user.get("api_calls_used", 0)
        api_calls_limit = current_user.get("api_calls_limit", 100)
        remaining = max(0, api_calls_limit - api_calls_used)
        
        return RateLimitResponse(
            limit=api_calls_limit,
            remaining=remaining,
            reset_time=datetime.utcnow() + timedelta(days=30)  # Monthly reset
        )
        
    except Exception as e:
        logger.error(f"Error checking rate limits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check rate limits"
        )

# Admin-only endpoints
@router.get("/admin/users", summary="Get All Users (Admin Only)")
async def get_all_users(
    skip: int = 0,
    limit: int = 100,
    current_admin: Dict[str, Any] = Depends(get_current_admin)
):
    """
    Get all users with pagination (Admin only).
    
    - **skip**: Number of users to skip
    - **limit**: Maximum number of users to return
    """
    try:
        users = await db.get_all_users(skip=skip, limit=limit)
        
        return {
            "users": users,
            "total": len(users),
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting all users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get users list"
        )

@router.get("/admin/analytics", response_model=AnalyticsResponse, summary="Get Analytics (Admin Only)")
async def get_analytics(
    days: int = 30,
    current_admin: Dict[str, Any] = Depends(get_current_admin)
):
    """
    Get system analytics and usage statistics (Admin only).
    
    - **days**: Number of days to analyze (default: 30)
    """
    try:
        if days > 365:  # Limit to 1 year
            days = 365
        
        analytics_data = await db.get_analytics_data(days)
        
        return AnalyticsResponse(**analytics_data)
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analytics data"
        )

@router.put("/admin/users/{user_id}/suspend", summary="Suspend User (Admin Only)")
async def suspend_user(
    user_id: str,
    current_admin: Dict[str, Any] = Depends(get_current_admin)
):
    """
    Suspend a user account (Admin only).
    """
    try:
        # Update user status to suspended
        success = await db.update_user(user_id, {"status": UserStatus.SUSPENDED})
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"Admin {current_admin['username']} suspended user {user_id}")
        
        return {
            "success": True,
            "message": "User account suspended successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error suspending user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to suspend user"
        )

@router.put("/admin/users/{user_id}/activate", summary="Activate User (Admin Only)")
async def activate_user(
    user_id: str,
    current_admin: Dict[str, Any] = Depends(get_current_admin)
):
    """
    Activate a user account (Admin only).
    """
    try:
        # Update user status to active
        success = await db.update_user(user_id, {"status": UserStatus.ACTIVE})
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"Admin {current_admin['username']} activated user {user_id}")
        
        return {
            "success": True,
            "message": "User account activated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate user"
        )

@router.put("/admin/users/{user_id}/upgrade", summary="Upgrade User Subscription (Admin Only)")
async def upgrade_user_subscription(
    user_id: str,
    subscription_tier: SubscriptionTier,
    api_calls_limit: int,
    current_admin: Dict[str, Any] = Depends(get_current_admin)
):
    """
    Upgrade user's subscription tier (Admin only).
    
    - **subscription_tier**: New subscription tier
    - **api_calls_limit**: New monthly API calls limit
    """
    try:
        # Update user subscription
        success = await db.update_user(user_id, {
            "subscription_tier": subscription_tier,
            "api_calls_limit": api_calls_limit
        })
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"Admin {current_admin['username']} upgraded user {user_id} to {subscription_tier}")
        
        return {
            "success": True,
            "message": f"User subscription upgraded to {subscription_tier.value}",
            "subscription_tier": subscription_tier,
            "api_calls_limit": api_calls_limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error upgrading user subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upgrade user subscription"
        )

@router.delete("/admin/users/{user_id}", summary="Delete User (Admin Only)")
async def delete_user(
    user_id: str,
    current_admin: Dict[str, Any] = Depends(get_current_admin)
):
    """
    Delete a user account (Admin only).
    """
    try:
        # Get user info before deletion
        user = await db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Delete user from database
        from bson import ObjectId
        result = await db.users_collection.delete_one({"_id": ObjectId(user_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"Admin {current_admin['username']} deleted user {user_id} ({user['username']})")
        
        return {
            "success": True,
            "message": "User account deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


