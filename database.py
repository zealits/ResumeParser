import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime, timedelta
from models import UserDocument, UsageDocument, APICallDocument, UserRole, SubscriptionTier, UserStatus

class MongoDB:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.users_collection = None
        self.usage_collection = None
        self.api_calls_collection = None
        self.candidates_collection = None
        self.projects_collection = None
        self.credit_tx_collection = None

    async def connect(self):
        """Connect to MongoDB"""
        try:
            # Get MongoDB connection string from environment
            mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            database_name = os.getenv("MONGODB_DATABASE", "resume_parser")
            
            self.client = AsyncIOMotorClient(mongodb_url)
            self.database = self.client[database_name]
            
            # Initialize collections
            self.users_collection = self.database.users
            self.usage_collection = self.database.usage_stats
            self.api_calls_collection = self.database.api_calls
            self.credit_tx_collection = self.database.credit_transactions
            
            # Initialize candidates collection (from environment or default)
            candidates_col_name = os.getenv("MONGO_COL", "candidates")
            self.candidates_collection = self.database[candidates_col_name]
            
            # Initialize projects collection (from environment or default)
            projects_col_name = os.getenv("MONGO_PROJECT_COL", "projects")
            self.projects_collection = self.database[projects_col_name]
            
            # Create indexes for better performance
            await self.create_indexes()
            
            print(f"✅ Connected to MongoDB: {database_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            return False

    async def create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Users collection indexes
            await self.users_collection.create_index("username", unique=True)
            await self.users_collection.create_index("email", unique=True)
            await self.users_collection.create_index("subscription_tier")
            await self.users_collection.create_index("status")
            
            # Usage collection indexes
            await self.usage_collection.create_index([("user_id", 1), ("date", 1)], unique=True)
            await self.usage_collection.create_index("date")
            await self.usage_collection.create_index("user_id")
            
            # API calls collection indexes
            await self.api_calls_collection.create_index("user_id")
            await self.api_calls_collection.create_index("timestamp")
            await self.api_calls_collection.create_index("endpoint")
            await self.api_calls_collection.create_index([("user_id", 1), ("timestamp", 1)])
            
            # Credit transaction ledger indexes
            await self.credit_tx_collection.create_index([("user_id", 1), ("created_at", -1)])
            await self.credit_tx_collection.create_index("type")
            # Square payment ids are unique per charge; sparse so ledger rows
            # without one (debits, refunds, grants) are not forced to collide.
            await self.credit_tx_collection.create_index(
                "square_payment_id", unique=True, sparse=True
            )

            # Candidates collection indexes
            await self.candidates_collection.create_index("_id", unique=True)
            await self.candidates_collection.create_index("created_at")
            
            # Projects collection indexes
            await self.projects_collection.create_index("_id", unique=True)
            await self.projects_collection.create_index("created_at")
            
            print("✅ Database indexes created successfully")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not create indexes: {e}")

    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            print("✅ Disconnected from MongoDB")

    # User Management Methods
    async def create_user(self, user_data: Dict[str, Any]) -> Optional[str]:
        """Create a new user"""
        try:
            result = await self.users_collection.insert_one(user_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            return None

    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        try:
            user = await self.users_collection.find_one({"username": username})
            if user:
                user["id"] = str(user["_id"])
                del user["_id"]
            return user
        except Exception as e:
            print(f"❌ Error getting user by username: {e}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            user = await self.users_collection.find_one({"email": email})
            if user:
                user["id"] = str(user["_id"])
                del user["_id"]
            return user
        except Exception as e:
            print(f"❌ Error getting user by email: {e}")
            return None

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            from bson import ObjectId
            user = await self.users_collection.find_one({"_id": ObjectId(user_id)})
            if user:
                user["id"] = str(user["_id"])
                del user["_id"]
            return user
        except Exception as e:
            print(f"❌ Error getting user by ID: {e}")
            return None

    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user data"""
        try:
            from bson import ObjectId
            result = await self.users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"❌ Error updating user: {e}")
            return False

    async def update_user_login(self, user_id: str) -> bool:
        """Update user's last login time"""
        try:
            from bson import ObjectId
            result = await self.users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"❌ Error updating user login: {e}")
            return False

    async def increment_api_calls(self, user_id: str) -> bool:
        """Increment user's API calls count"""
        try:
            from bson import ObjectId
            result = await self.users_collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$inc": {"api_calls_used": 1}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"❌ Error incrementing API calls: {e}")
            return False

    async def get_all_users(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all users with pagination.

        Secrets are projected out at the query, so password hashes and reset
        tokens can never reach an API response by accident.
        """
        try:
            cursor = (
                self.users_collection.find(
                    {},
                    {
                        "hashed_password": 0,
                        "password_reset_token": 0,
                        "password_reset_expires": 0,
                    },
                )
                .sort("created_at", -1)
                .skip(skip)
                .limit(limit)
            )
            users = []
            async for user in cursor:
                user["id"] = str(user["_id"])
                del user["_id"]
                users.append(user)
            return users
        except Exception as e:
            print(f"❌ Error getting all users: {e}")
            return []

    # Usage Tracking Methods
    async def log_api_call(self, api_call_data: Dict[str, Any]) -> Optional[str]:
        """Log an API call"""
        try:
            result = await self.api_calls_collection.insert_one(api_call_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Error logging API call: {e}")
            return None

    async def update_daily_usage(self, user_id: str, endpoint: str, processing_time: float, 
                                file_size: int, success: bool = True) -> bool:
        """Update daily usage statistics"""
        try:
            today = datetime.utcnow().date().isoformat()  # Convert to string
            
            # Update or create daily usage record
            await self.usage_collection.update_one(
                {"user_id": user_id, "date": today},
                {
                    "$inc": {
                        "api_calls": 1,
                        "total_processing_time": processing_time,
                        "successful_calls": 1 if success else 0,
                        "failed_calls": 0 if success else 1,
                        f"endpoints_used.{endpoint}": 1
                    },
                    "$push": {"file_sizes": file_size},
                    "$setOnInsert": {
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            print(f"❌ Error updating daily usage: {e}")
            return False

    async def get_user_usage_stats(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get user's usage statistics for the last N days"""
        try:
            start_date = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
            cursor = self.usage_collection.find({
                "user_id": user_id,
                "date": {"$gte": start_date}
            }).sort("date", 1)
            
            stats = []
            async for stat in cursor:
                stat["id"] = str(stat["_id"])
                del stat["_id"]
                stats.append(stat)
            return stats
        except Exception as e:
            print(f"❌ Error getting user usage stats: {e}")
            return []

    async def get_analytics_data(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics data for admin dashboard"""
        try:
            start_date = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
            
            # Total users
            total_users = await self.users_collection.count_documents({})
            active_users = await self.users_collection.count_documents({"status": UserStatus.ACTIVE})
            
            # Usage statistics
            usage_pipeline = [
                {"$match": {"date": {"$gte": start_date}}},
                {"$group": {
                    "_id": None,
                    "total_calls": {"$sum": "$api_calls"},
                    "total_processing_time": {"$sum": "$total_processing_time"},
                    "successful_calls": {"$sum": "$successful_calls"},
                    "failed_calls": {"$sum": "$failed_calls"}
                }}
            ]
            
            usage_stats = await self.usage_collection.aggregate(usage_pipeline).to_list(1)
            usage_data = usage_stats[0] if usage_stats else {
                "total_calls": 0, "total_processing_time": 0, 
                "successful_calls": 0, "failed_calls": 0
            }
            
            # Usage by subscription tier
            tier_pipeline = [
                {"$group": {
                    "_id": "$subscription_tier",
                    "total_calls": {"$sum": "$api_calls_used"}
                }}
            ]
            tier_stats = await self.users_collection.aggregate(tier_pipeline).to_list(None)
            # A user saved without a tier groups under null, which is not a
            # valid key for the Dict[str, int] response model.
            usage_by_tier = {
                (stat["_id"] or "unknown"): stat["total_calls"] for stat in tier_stats
            }
            
            # Daily usage for the last N days
            daily_pipeline = [
                {"$match": {"date": {"$gte": start_date}}},
                {"$group": {
                    "_id": "$date",
                    "total_calls": {"$sum": "$api_calls"},
                    "total_processing_time": {"$sum": "$total_processing_time"}
                }},
                {"$sort": {"_id": 1}}
            ]
            daily_rows = await self.usage_collection.aggregate(daily_pipeline).to_list(None)
            # Expose the grouping key as "date" rather than Mongo's "_id".
            daily_usage = [
                {
                    "date": row.pop("_id"),
                    **row,
                }
                for row in daily_rows
            ]
            
            # Top users by API calls
            top_users_pipeline = [
                {"$sort": {"api_calls_used": -1}},
                {"$limit": 10},
                {"$project": {
                    "username": 1,
                    "email": 1,
                    "subscription_tier": 1,
                    "api_calls_used": 1,
                    "company_name": 1
                }}
            ]
            top_rows = await self.users_collection.aggregate(top_users_pipeline).to_list(None)
            # $project keeps _id, and a raw ObjectId is not JSON-serializable,
            # which made this whole endpoint 500.
            top_users = []
            for row in top_rows:
                row["id"] = str(row.pop("_id"))
                top_users.append(row)
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "total_api_calls": usage_data["total_calls"],
                "total_processing_time": usage_data["total_processing_time"],
                "average_response_time": usage_data["total_processing_time"] / max(usage_data["total_calls"], 1),
                "usage_by_tier": usage_by_tier,
                "daily_usage": daily_usage,
                "top_users": top_users
            }
            
        except Exception as e:
            print(f"❌ Error getting analytics data: {e}")
            # Return a valid empty shape: an {} here would fail to unpack into
            # AnalyticsResponse and turn a read error into a 500.
            return {
                "total_users": 0,
                "active_users": 0,
                "total_api_calls": 0,
                "total_processing_time": 0.0,
                "average_response_time": 0.0,
                "usage_by_tier": {},
                "daily_usage": [],
                "top_users": [],
            }

    async def check_rate_limit(self, user_id: str) -> Dict[str, Any]:
        """Check if user has exceeded rate limits"""
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return {"allowed": False, "reason": "User not found"}
            
            # Check monthly limit
            if user["api_calls_used"] >= user["api_calls_limit"]:
                return {
                    "allowed": False, 
                    "reason": "Monthly API limit exceeded",
                    "limit": user["api_calls_limit"],
                    "used": user["api_calls_used"]
                }
            
            # Check daily limit
            today = datetime.utcnow().date()
            daily_usage = await self.usage_collection.find_one({
                "user_id": user_id,
                "date": today
            })
            
            daily_limit = 10  # Default daily limit
            if user["subscription_tier"] == SubscriptionTier.BASIC:
                daily_limit = 50
            elif user["subscription_tier"] == SubscriptionTier.PREMIUM:
                daily_limit = 200
            elif user["subscription_tier"] == SubscriptionTier.ENTERPRISE:
                daily_limit = -1  # Unlimited
            
            if daily_limit != -1 and daily_usage and daily_usage["api_calls"] >= daily_limit:
                return {
                    "allowed": False,
                    "reason": "Daily API limit exceeded",
                    "limit": daily_limit,
                    "used": daily_usage["api_calls"]
                }
            
            return {"allowed": True}
            
        except Exception as e:
            print(f"❌ Error checking rate limit: {e}")
            return {"allowed": False, "reason": "Error checking limits"}

    # ------------------------------------------------------------------
    # Credit Billing Methods
    # ------------------------------------------------------------------
    async def _log_credit_tx(self, doc: Dict[str, Any]) -> Optional[str]:
        """Append a row to the credit ledger."""
        try:
            doc.setdefault("created_at", datetime.utcnow())
            result = await self.credit_tx_collection.insert_one(doc)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error writing credit transaction: {e}")
            return None

    async def get_credit_balance(self, user_id: str) -> int:
        """Current credit balance for a user, or 0 if unknown."""
        try:
            from bson import ObjectId
            user = await self.users_collection.find_one(
                {"_id": ObjectId(user_id)}, {"credits_balance": 1}
            )
            return int(user.get("credits_balance", 0)) if user else 0
        except Exception as e:
            print(f"Error reading credit balance: {e}")
            return 0

    async def deduct_credits(self, user_id: str, amount: int, operation: str) -> bool:
        """
        Atomically charge `amount` credits.

        The balance guard lives in the query filter, so MongoDB only applies
        the decrement when the funds are actually there. Two concurrent calls
        against a balance of 1 can therefore never both succeed.
        """
        if amount <= 0:
            return True
        try:
            from bson import ObjectId
            updated = await self.users_collection.find_one_and_update(
                {"_id": ObjectId(user_id), "credits_balance": {"$gte": amount}},
                {
                    "$inc": {"credits_balance": -amount, "credits_used": amount},
                    "$set": {"updated_at": datetime.utcnow()},
                },
                projection={"credits_balance": 1},
                return_document=True,
            )
            if not updated:
                return False

            await self._log_credit_tx({
                "user_id": user_id,
                "type": "debit",
                "operation": operation,
                "credits": -amount,
                "balance_after": int(updated.get("credits_balance", 0)),
            })
            return True
        except Exception as e:
            print(f"Error deducting credits: {e}")
            return False

    async def refund_credits(
        self, user_id: str, amount: int, operation: str, reason: str = ""
    ) -> bool:
        """Return credits taken for a call that did not succeed."""
        if amount <= 0:
            return True
        try:
            from bson import ObjectId
            updated = await self.users_collection.find_one_and_update(
                {"_id": ObjectId(user_id)},
                {
                    "$inc": {"credits_balance": amount, "credits_used": -amount},
                    "$set": {"updated_at": datetime.utcnow()},
                },
                projection={"credits_balance": 1},
                return_document=True,
            )
            if not updated:
                return False

            await self._log_credit_tx({
                "user_id": user_id,
                "type": "refund",
                "operation": operation,
                "reason": reason,
                "credits": amount,
                "balance_after": int(updated.get("credits_balance", 0)),
            })
            return True
        except Exception as e:
            print(f"Error refunding credits: {e}")
            return False

    async def add_credits(
        self,
        user_id: str,
        amount: int,
        source: str,
        plan_key: Optional[str] = None,
        square_payment_id: Optional[str] = None,
        amount_cents: Optional[int] = None,
        note: Optional[str] = None,
    ) -> bool:
        """
        Credit an account after a purchase, a signup grant, or an admin grant.

        When `square_payment_id` is supplied the ledger row carries it, and the
        unique sparse index on that field makes a replayed payment fail rather
        than silently granting credits twice.
        """
        if amount <= 0:
            return False
        try:
            from bson import ObjectId

            if square_payment_id:
                existing = await self.credit_tx_collection.find_one(
                    {"square_payment_id": square_payment_id}
                )
                if existing:
                    print(f"Payment {square_payment_id} already credited; skipping")
                    return False

            updated = await self.users_collection.find_one_and_update(
                {"_id": ObjectId(user_id)},
                {
                    "$inc": {"credits_balance": amount},
                    "$set": {"updated_at": datetime.utcnow()},
                },
                projection={"credits_balance": 1},
                return_document=True,
            )
            if not updated:
                return False

            await self._log_credit_tx({
                "user_id": user_id,
                "type": "purchase" if square_payment_id else source,
                "source": source,
                "plan_key": plan_key,
                "square_payment_id": square_payment_id,
                "amount_cents": amount_cents,
                "note": note,
                "credits": amount,
                "balance_after": int(updated.get("credits_balance", 0)),
            })
            return True
        except Exception as e:
            print(f"Error adding credits: {e}")
            return False

    async def get_credit_transactions(
        self, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Most recent ledger rows for a user, newest first."""
        try:
            rows = []
            cursor = (
                self.credit_tx_collection.find({"user_id": user_id})
                .sort("created_at", -1)
                .limit(max(1, min(limit, 200)))
            )
            async for row in cursor:
                row["id"] = str(row.pop("_id"))
                rows.append(row)
            return rows
        except Exception as e:
            print(f"Error reading credit transactions: {e}")
            return []

    async def get_revenue_summary(self, days: int = 30) -> Dict[str, Any]:
        """Purchase totals for the admin dashboard."""
        try:
            since = datetime.utcnow() - timedelta(days=days)
            pipeline = [
                {"$match": {"type": "purchase", "created_at": {"$gte": since}}},
                {
                    "$group": {
                        "_id": None,
                        "gross_cents": {"$sum": "$amount_cents"},
                        "credits_sold": {"$sum": "$credits"},
                        "payments": {"$sum": 1},
                    }
                },
            ]
            agg = await self.credit_tx_collection.aggregate(pipeline).to_list(length=1)
            row = agg[0] if agg else {}
            return {
                "period_days": days,
                "gross_cents": int(row.get("gross_cents") or 0),
                "credits_sold": int(row.get("credits_sold") or 0),
                "payments": int(row.get("payments") or 0),
            }
        except Exception as e:
            print(f"Error building revenue summary: {e}")
            return {
                "period_days": days,
                "gross_cents": 0,
                "credits_sold": 0,
                "payments": 0,
            }


# Global database instance
db = MongoDB()

