"""
Billing and self-serve signup endpoints.

Flow for a paid plan:
    1. Browser tokenizes the card in Square's iframe -> single-use source_id.
    2. POST /api/billing/signup with the plan and that token.
    3. Square is charged FIRST. Only on a completed payment is the account
       created and the credits granted, so a declined card never leaves a
       half-made user behind.
    4. The response carries a JWT, so the browser lands on the dashboard
       already logged in.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from auth import auth_manager, get_current_admin, get_current_user
from config import settings
from credits import (
    CREDIT_COSTS,
    DEFAULT_PLAN_KEY,
    cost_table,
    get_plan,
    public_plans,
)
from database import db
from email_service import email_service
from models import (
    AdminGrantCreditsRequest,
    SignupRequest,
    SubscriptionTier,
    TopUpRequest,
    UserRole,
    UserStatus,
)
from payments import PaymentError, square_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/billing", tags=["Billing"])


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------
@router.get("/plans", summary="List credit plans and per-call costs")
async def list_plans() -> Dict[str, Any]:
    """Everything the pricing section needs. No auth required."""
    return {
        "plans": public_plans(),
        "credit_costs": cost_table(),
        "currency": settings.SQUARE_CURRENCY,
        "payments_enabled": settings.square_configured(),
        "sales_email": settings.ADMIN_EMAIL or settings.FROM_EMAIL,
    }


@router.get("/config", summary="Public Square config for the browser SDK")
async def public_config() -> Dict[str, Any]:
    """
    Application id and location id are *public* values - the Web Payments SDK
    needs them client-side. The access token is secret and never sent here.
    """
    return {
        "square": {
            "application_id": settings.SQUARE_APPLICATION_ID,
            "location_id": settings.SQUARE_LOCATION_ID,
            "environment": settings.SQUARE_ENVIRONMENT,
            "enabled": settings.square_configured(),
        },
        "currency": settings.SQUARE_CURRENCY,
    }


# ---------------------------------------------------------------------------
# Signup
# ---------------------------------------------------------------------------
async def _assert_identity_available(username: str, email: str) -> None:
    if await db.get_user_by_username(username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="That username is already taken.",
        )
    if await db.get_user_by_email(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account already exists for that email.",
        )


async def _provision_user(payload: SignupRequest, plan: Dict[str, Any]) -> str:
    """Create the account. Credits are granted separately so they get a ledger row."""
    user_doc = {
        "username": payload.username.strip(),
        "email": payload.email.strip().lower(),
        "hashed_password": auth_manager.get_password_hash(payload.password),
        "subscription_tier": plan["tier"],
        "plan_key": plan["key"],
        # Legacy monthly counters are kept in sync so existing analytics and
        # the older middleware keep working; credits are the real meter.
        "api_calls_limit": plan["credits"],
        "api_calls_used": 0,
        "credits_balance": 0,
        "credits_used": 0,
        "company_name": payload.company_name,
        "contact_person": payload.contact_person,
        "status": UserStatus.ACTIVE.value,
        "role": UserRole.USER.value,
        "created_at": datetime.utcnow(),
        "is_active": True,
    }
    user_id = await db.create_user(user_doc)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create your account. Please try again.",
        )
    return user_id


@router.post("/signup", summary="Create an account and buy a credit plan")
async def signup(payload: SignupRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    plan = get_plan(payload.plan_key)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown plan."
        )
    if plan.get("contact_sales"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Enterprise is sold through sales. Contact us for a custom quote.",
        )

    await _assert_identity_available(payload.username.strip(), payload.email.strip().lower())

    payment: Dict[str, Any] = {}
    is_free = plan.get("price_cents") == 0

    if not is_free:
        if not payload.source_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Card details are required for a paid plan.",
            )
        # Charge before creating anything. A decline raises and we are done.
        try:
            payment = await square_client.create_payment(
                source_id=payload.source_id,
                amount_cents=plan["price_cents"],
                idempotency_key=payload.idempotency_key or str(uuid.uuid4()),
                buyer_email=payload.email,
                note="{} plan - {} credits".format(plan["name"], plan["credits"]),
                reference_id=payload.username.strip(),
            )
        except PaymentError as exc:
            logger.info("Signup payment declined for %s: %s", payload.email, exc.message)
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=exc.message
            )

    user_id = await _provision_user(payload, plan)

    granted = await db.add_credits(
        user_id=user_id,
        amount=plan["credits"],
        source="signup_purchase" if not is_free else "signup_grant",
        plan_key=plan["key"],
        square_payment_id=payment.get("payment_id"),
        amount_cents=plan["price_cents"] if not is_free else 0,
        note="{} plan signup".format(plan["name"]),
    )
    if not granted:
        # The account exists and the card was charged; surface it loudly rather
        # than pretending the purchase worked.
        logger.error(
            "Credits not granted after signup for user %s (payment %s)",
            user_id,
            payment.get("payment_id"),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Your payment went through but credits could not be applied. "
                "Contact support with this reference: {}".format(
                    payment.get("payment_id") or user_id
                )
            ),
        )

    if not is_free:
        background_tasks.add_task(
            email_service.send_admin_notification,
            "New paid signup",
            "{} ({}) bought the {} plan for {} credits.".format(
                payload.username, payload.email, plan["name"], plan["credits"]
            ),
        )

    # Log them straight in so the browser can redirect to the dashboard.
    token = await auth_manager.login_user(
        username=payload.username.strip(), password=payload.password
    )

    return {
        "success": True,
        "message": "Welcome aboard. {} credits added.".format(plan["credits"]),
        "plan": plan["key"],
        "credits_granted": plan["credits"],
        "receipt_url": payment.get("receipt_url"),
        "auth": token,
    }


# ---------------------------------------------------------------------------
# Top-ups
# ---------------------------------------------------------------------------
@router.post("/topup", summary="Buy more credits for the current account")
async def topup(
    payload: TopUpRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    plan = get_plan(payload.plan_key)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown plan."
        )
    if plan.get("contact_sales"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Enterprise is sold through sales. Contact us for a custom quote.",
        )
    if not plan.get("price_cents"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The free plan is a one-time signup grant, not a top-up.",
        )

    try:
        payment = await square_client.create_payment(
            source_id=payload.source_id,
            amount_cents=plan["price_cents"],
            idempotency_key=payload.idempotency_key or str(uuid.uuid4()),
            buyer_email=current_user.get("email"),
            note="Top-up: {} plan - {} credits".format(plan["name"], plan["credits"]),
            reference_id=str(current_user["id"]),
        )
    except PaymentError as exc:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED, detail=exc.message
        )

    granted = await db.add_credits(
        user_id=current_user["id"],
        amount=plan["credits"],
        source="topup",
        plan_key=plan["key"],
        square_payment_id=payment.get("payment_id"),
        amount_cents=plan["price_cents"],
        note="{} top-up".format(plan["name"]),
    )
    if not granted:
        logger.error(
            "Top-up credits not applied for %s (payment %s)",
            current_user["id"],
            payment.get("payment_id"),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Your payment went through but credits could not be applied. "
                "Contact support with this reference: {}".format(payment.get("payment_id"))
            ),
        )

    # Keep the tier in step with the largest bundle the user has bought.
    tier_rank = {
        SubscriptionTier.FREE.value: 0,
        SubscriptionTier.BASIC.value: 1,
        SubscriptionTier.PREMIUM.value: 2,
        SubscriptionTier.ENTERPRISE.value: 3,
    }
    current_tier = current_user.get("subscription_tier", SubscriptionTier.FREE.value)
    if tier_rank.get(plan["tier"], 0) > tier_rank.get(current_tier, 0):
        await db.update_user(
            current_user["id"],
            {"subscription_tier": plan["tier"], "plan_key": plan["key"]},
        )

    balance = await db.get_credit_balance(current_user["id"])
    return {
        "success": True,
        "message": "{} credits added.".format(plan["credits"]),
        "credits_granted": plan["credits"],
        "credits_balance": balance,
        "receipt_url": payment.get("receipt_url"),
    }


# ---------------------------------------------------------------------------
# Account-facing reads
# ---------------------------------------------------------------------------
@router.get("/me", summary="Credit balance and spend for the current account")
async def my_credits(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    balance = await db.get_credit_balance(current_user["id"])
    return {
        "user_id": current_user["id"],
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "role": current_user.get("role", UserRole.USER.value),
        "plan_key": current_user.get("plan_key"),
        "subscription_tier": current_user.get(
            "subscription_tier", SubscriptionTier.FREE.value
        ),
        "credits_balance": balance,
        "credits_used": int(current_user.get("credits_used", 0) or 0),
        "credit_costs": cost_table(),
    }


@router.get("/transactions", summary="Credit ledger for the current account")
async def my_transactions(
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    rows = await db.get_credit_transactions(current_user["id"], limit=limit)
    return {"transactions": rows, "total": len(rows)}


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
@router.post("/admin/users/{user_id}/credits", summary="Grant credits (Admin Only)")
async def grant_credits(
    user_id: str,
    payload: AdminGrantCreditsRequest,
    current_admin: Dict[str, Any] = Depends(get_current_admin),
) -> Dict[str, Any]:
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    granted = await db.add_credits(
        user_id=user_id,
        amount=payload.credits,
        source="admin_grant",
        note=payload.note or "Granted by {}".format(current_admin["username"]),
    )
    if not granted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not grant credits",
        )

    balance = await db.get_credit_balance(user_id)
    logger.info(
        "Admin %s granted %s credits to %s",
        current_admin["username"],
        payload.credits,
        user.get("username"),
    )
    return {
        "success": True,
        "message": "{} credits granted to {}.".format(payload.credits, user.get("username")),
        "credits_balance": balance,
    }


@router.get("/admin/revenue", summary="Revenue summary (Admin Only)")
async def revenue(
    days: int = 30,
    current_admin: Dict[str, Any] = Depends(get_current_admin),
) -> Dict[str, Any]:
    days = max(1, min(days, 365))
    summary = await db.get_revenue_summary(days)
    summary["gross_display"] = "${:,.2f}".format(summary["gross_cents"] / 100)
    summary["currency"] = settings.SQUARE_CURRENCY
    return summary


@router.get("/admin/transactions", summary="Recent credit ledger (Admin Only)")
async def all_transactions(
    limit: int = 100,
    current_admin: Dict[str, Any] = Depends(get_current_admin),
) -> Dict[str, Any]:
    limit = max(1, min(limit, 500))
    rows: List[Dict[str, Any]] = []
    cursor = db.credit_tx_collection.find({}).sort("created_at", -1).limit(limit)
    async for row in cursor:
        row["id"] = str(row.pop("_id"))
        rows.append(row)

    # Attach usernames so the admin table is readable.
    usernames: Dict[str, str] = {}
    for row in rows:
        uid = row.get("user_id")
        if uid and uid not in usernames:
            user = await db.get_user_by_id(uid)
            usernames[uid] = user.get("username", "(deleted)") if user else "(deleted)"
        row["username"] = usernames.get(row.get("user_id"), "(unknown)")

    return {"transactions": rows, "total": len(rows)}


@router.get("/admin/credit-costs", summary="Per-endpoint credit costs (Admin Only)")
async def admin_credit_costs(
    current_admin: Dict[str, Any] = Depends(get_current_admin),
) -> Dict[str, Any]:
    return {"credit_costs": cost_table(), "raw": CREDIT_COSTS}
