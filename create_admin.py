#!/usr/bin/env python3
"""
Create (or repair) the System Administrator account for the Resume Parser API.

Reads ADMIN_USERNAME / ADMIN_EMAIL / ADMIN_PASSWORD from .env, removes any
stale placeholder admin accounts, and upserts the real admin with role=admin.

Usage:
    python create_admin.py
"""

import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from auth import auth_manager
from config import settings
from database import db
from models import SubscriptionTier, UserRole, UserStatus

# Placeholder accounts shipped with the sample config; removed if present.
STALE_ADMIN_EMAILS = ["admin@yourdomain.com", "admin@yourcompany.com"]


async def remove_stale_admins(keep_email: str) -> None:
    """Delete leftover placeholder admin accounts."""
    for email in STALE_ADMIN_EMAILS:
        if email == keep_email:
            continue
        existing = await db.users_collection.find_one({"email": email})
        if not existing:
            print(f"   nothing to delete for {email}")
            continue
        await db.users_collection.delete_one({"_id": existing["_id"]})
        print(f"   deleted {email} (username={existing.get('username')}, _id={existing['_id']})")


async def upsert_admin() -> None:
    username = settings.ADMIN_USERNAME
    email = settings.ADMIN_EMAIL
    password = settings.ADMIN_PASSWORD

    admin_doc = {
        "username": username,
        "email": email,
        "hashed_password": auth_manager.get_password_hash(password),
        "subscription_tier": SubscriptionTier.ENTERPRISE.value,
        "api_calls_limit": -1,  # unlimited
        "api_calls_used": 0,
        "company_name": "System Admin",
        "contact_person": "System Administrator",
        "status": UserStatus.ACTIVE.value,
        "role": UserRole.ADMIN.value,
        "is_active": True,
    }

    existing = await db.users_collection.find_one(
        {"$or": [{"username": username}, {"email": email}]}
    )

    if existing:
        await db.users_collection.update_one({"_id": existing["_id"]}, {"$set": admin_doc})
        print(f"✅ Updated existing account to System Administrator: {email} (_id={existing['_id']})")
    else:
        admin_doc["created_at"] = datetime.utcnow()
        result = await db.users_collection.insert_one(admin_doc)
        print(f"✅ Created System Administrator: {email} (_id={result.inserted_id})")


async def main() -> int:
    print("🔐 Resume Parser API - System Administrator setup")
    print("=" * 55)
    print(f"   database : {os.getenv('MONGODB_DATABASE', 'resume_parser')}")
    print(f"   username : {settings.ADMIN_USERNAME}")
    print(f"   email    : {settings.ADMIN_EMAIL}")

    if not await db.connect():
        print("❌ Could not connect to MongoDB - check MONGODB_URL in .env")
        return 1

    try:
        print("\n🧹 Removing placeholder admin accounts...")
        await remove_stale_admins(settings.ADMIN_EMAIL)

        print("\n👤 Writing System Administrator account...")
        await upsert_admin()

        admin = await db.users_collection.find_one({"email": settings.ADMIN_EMAIL})
        print("\n📋 Verification:")
        print(f"   username          : {admin.get('username')}")
        print(f"   email             : {admin.get('email')}")
        print(f"   role              : {admin.get('role')}")
        print(f"   status            : {admin.get('status')}")
        print(f"   subscription_tier : {admin.get('subscription_tier')}")
        print(f"   api_calls_limit   : {admin.get('api_calls_limit')}")
        print(f"   contact_person    : {admin.get('contact_person')}")

        remaining = await db.users_collection.count_documents({})
        print(f"\n   total users in collection: {remaining}")
        print("\n✅ Done. Log in at POST /auth/login with the username above.")
        return 0
    finally:
        await db.disconnect()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
