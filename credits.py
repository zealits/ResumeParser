"""
Credit-based billing for the Resume Parser API.

Every metered endpoint costs a fixed number of credits. Credits are bought in
bundles (see CREDIT_PLANS) and deducted atomically before the work is done, so
two concurrent requests can never overspend a balance. If the endpoint then
fails, CreditRefundMiddleware puts the credits back.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from auth import get_current_user
from config import settings
from database import db
from models import SubscriptionTier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
# price_cents is what Square charges. credits is what lands in the balance.
# Edit freely - nothing below reads these values positionally.
CREDIT_PLANS: Dict[str, Dict[str, Any]] = {
    "starter": {
        "key": "starter",
        "name": "Starter",
        "tier": SubscriptionTier.FREE.value,
        "price_cents": 200,
        "credits": 50,
        "tagline": "Try it on a real shortlist",
        "features": [
            "50 credits",
            "Resume parsing & GitHub analysis",
            "5 MB max upload",
            "Community support",
        ],
    },
    "basic": {
        "key": "basic",
        "name": "Basic",
        "tier": SubscriptionTier.BASIC.value,
        "price_cents": 2900,
        "credits": 1000,
        "tagline": "For small hiring teams",
        "features": [
            "1,000 credits",
            "All parsing & ranking endpoints",
            "10 MB max upload",
            "Email support",
        ],
    },
    "premium": {
        "key": "premium",
        "name": "Premium",
        "tier": SubscriptionTier.PREMIUM.value,
        "price_cents": 9900,
        "credits": 5000,
        "tagline": "For scaling recruitment",
        "popular": True,
        "features": [
            "5,000 credits",
            "Test generation & evaluation",
            "25 MB max upload",
            "Priority support",
        ],
    },
    "enterprise": {
        "key": "enterprise",
        "name": "Enterprise",
        "tier": SubscriptionTier.ENTERPRISE.value,
        "price_cents": 34900,
        "credits": 25000,
        "tagline": "High volume, best rate per credit",
        "features": [
            "25,000 credits",
            "Every endpoint, no daily cap",
            "50 MB max upload",
            "Dedicated support",
        ],
    },
}

# Cheapest plan, used as the default when /signup is opened without ?plan=.
# There is no $0 plan any more, so every signup takes a card.
DEFAULT_PLAN_KEY = "starter"


def get_plan(plan_key: str) -> Optional[Dict[str, Any]]:
    """Look up a plan by key, or None if it isn't a real plan."""
    return CREDIT_PLANS.get((plan_key or "").strip().lower())


def public_plans() -> List[Dict[str, Any]]:
    """Plans shaped for the pricing section, cheapest first."""
    out = []
    for plan in sorted(CREDIT_PLANS.values(), key=lambda p: p["price_cents"]):
        cents = plan["price_cents"]
        credits = plan["credits"]
        out.append(
            {
                **plan,
                "price_display": "Free" if cents == 0 else "${:,.0f}".format(cents / 100),
                "per_credit": None if cents == 0 else round(cents / credits / 100, 4),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Per-endpoint credit costs
# ---------------------------------------------------------------------------
# Keyed by a stable internal name, not by URL path, so route changes and path
# parameters (/candidate/{id}/evaluate) don't silently stop metering.
CREDIT_COSTS: Dict[str, int] = {
    "parse_resume": settings.CREDIT_COST_PARSE_RESUME,
    "parse_resume_text": settings.CREDIT_COST_PARSE_RESUME_TEXT,
    "github_analysis": settings.CREDIT_COST_GITHUB_ANALYSIS,
    "ranked_candidates": settings.CREDIT_COST_RANKED_CANDIDATES,
    "generate_test": settings.CREDIT_COST_GENERATE_TEST,
    "evaluate_test": settings.CREDIT_COST_EVALUATE_TEST,
}

# Human labels for the dashboard cost table.
CREDIT_COST_LABELS: Dict[str, str] = {
    "parse_resume": "Parse resume (PDF)",
    "parse_resume_text": "Parse resume (text)",
    "github_analysis": "Analyze GitHub profile",
    "ranked_candidates": "Rank candidates for project",
    "generate_test": "Generate candidate test",
    "evaluate_test": "Evaluate test submission",
}


def cost_table() -> List[Dict[str, Any]]:
    """Credit costs shaped for display, cheapest first."""
    rows = [
        {"key": key, "label": CREDIT_COST_LABELS.get(key, key), "credits": cost}
        for key, cost in CREDIT_COSTS.items()
    ]
    return sorted(rows, key=lambda row: (row["credits"], row["label"]))


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------
def require_credits(operation: str):
    """
    Build a dependency that charges `operation`'s credit cost before the
    endpoint runs.

    Returns the authenticated user, so it is a drop-in replacement for
    `Depends(get_current_user)` on any metered route. Admins are never charged.
    """
    if operation not in CREDIT_COSTS:
        raise KeyError("Unknown metered operation: {}".format(operation))

    async def dependency(
        request: Request,
        current_user: Dict[str, Any] = Depends(get_current_user),
    ) -> Dict[str, Any]:
        # Existing routes rely on this for UsageTrackingMiddleware.
        request.state.user = current_user

        cost = CREDIT_COSTS[operation]

        # Admins use the API for support and testing; never bill them.
        if current_user.get("role") == "admin":
            request.state.credit_charge = None
            return current_user

        charged = await db.deduct_credits(
            user_id=current_user["id"],
            amount=cost,
            operation=operation,
        )

        if not charged:
            balance = await db.get_credit_balance(current_user["id"])
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "Insufficient credits",
                    "message": (
                        "This call costs {} credit(s) but your balance is {}. "
                        "Top up from your dashboard to continue.".format(cost, balance)
                    ),
                    "operation": operation,
                    "credits_required": cost,
                    "credits_balance": balance,
                },
            )

        # Recorded so a failed response can be refunded by the middleware.
        request.state.credit_charge = {
            "user_id": current_user["id"],
            "amount": cost,
            "operation": operation,
        }
        return current_user

    return dependency


class CreditRefundMiddleware(BaseHTTPMiddleware):
    """
    Refund credits when a charged request did not succeed.

    Credits are taken up front so concurrent calls cannot overspend, which
    means a 4xx/5xx would otherwise bill the user for nothing.
    """

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception:
            await self._refund(request, reason="unhandled_error")
            raise

        if response.status_code >= 400:
            await self._refund(request, reason="http_{}".format(response.status_code))

        return response

    @staticmethod
    async def _refund(request: Request, reason: str) -> None:
        charge = getattr(request.state, "credit_charge", None)
        if not charge:
            return
        # Clear first so nothing can double-refund the same charge.
        request.state.credit_charge = None
        try:
            await db.refund_credits(
                user_id=charge["user_id"],
                amount=charge["amount"],
                operation=charge["operation"],
                reason=reason,
            )
            logger.info(
                "Refunded %s credit(s) to %s (%s: %s)",
                charge["amount"],
                charge["user_id"],
                charge["operation"],
                reason,
            )
        except Exception as exc:
            logger.error("Failed to refund credits for %s: %s", charge["user_id"], exc)
