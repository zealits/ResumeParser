"""
Square payments for credit purchases.

Card details are tokenized in Square's own iframe by the Web Payments SDK in
the browser, so this server only ever receives a single-use `source_id` token.
Raw card numbers never touch our process, our logs, or our database.

Talks to the Square Connect REST API directly over httpx, which is already a
dependency - no extra SDK to keep in version lockstep.
"""

import logging
import uuid
from typing import Any, Dict, Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)


class PaymentError(Exception):
    """A payment could not be completed. `message` is safe to show a user."""

    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class SquareClient:
    """Thin wrapper over the Square Connect API endpoints we actually use."""

    def __init__(self) -> None:
        self.timeout = httpx.Timeout(30.0, connect=10.0)

    @property
    def base_url(self) -> str:
        return settings.square_api_base()

    def _headers(self) -> Dict[str, str]:
        if not settings.SQUARE_ACCESS_TOKEN:
            raise PaymentError("Payments are not configured on this server.")
        return {
            "Authorization": "Bearer {}".format(settings.SQUARE_ACCESS_TOKEN),
            "Square-Version": settings.SQUARE_API_VERSION,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _first_error(payload: Dict[str, Any]) -> str:
        """Pull a human-usable message out of a Square error envelope."""
        errors = payload.get("errors") or []
        if errors:
            first = errors[0]
            return first.get("detail") or first.get("code") or "Payment declined."
        return "Payment declined."

    async def create_payment(
        self,
        source_id: str,
        amount_cents: int,
        idempotency_key: Optional[str] = None,
        buyer_email: Optional[str] = None,
        note: Optional[str] = None,
        reference_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Charge a tokenized card.

        `idempotency_key` is what stops a double-submit or a network retry from
        charging twice - Square returns the original payment for a repeated key.
        """
        if not settings.square_configured():
            raise PaymentError(
                "Payments are not configured. Set SQUARE_ACCESS_TOKEN, "
                "SQUARE_APPLICATION_ID and SQUARE_LOCATION_ID in .env."
            )
        if amount_cents <= 0:
            raise PaymentError("Nothing to charge for this plan.")

        body: Dict[str, Any] = {
            "source_id": source_id,
            "idempotency_key": idempotency_key or str(uuid.uuid4()),
            "amount_money": {
                "amount": int(amount_cents),
                "currency": settings.SQUARE_CURRENCY,
            },
            "location_id": settings.SQUARE_LOCATION_ID,
            "autocomplete": True,
        }
        if buyer_email:
            body["buyer_email_address"] = buyer_email
        if note:
            body["note"] = note[:500]
        if reference_id:
            body["reference_id"] = reference_id[:40]

        url = "{}/v2/payments".format(self.base_url)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=self._headers(), json=body)
        except httpx.HTTPError as exc:
            logger.error("Square request failed: %s", exc)
            raise PaymentError(
                "Could not reach the payment processor. Please try again."
            ) from exc

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if response.status_code >= 400:
            # Square error payloads carry no card data, so they are safe to log.
            logger.warning(
                "Square declined payment (%s): %s", response.status_code, payload
            )
            raise PaymentError(self._first_error(payload), detail=payload.get("errors"))

        payment = payload.get("payment") or {}
        payment_status = payment.get("status")

        if payment_status not in ("COMPLETED", "APPROVED"):
            logger.warning("Square payment not completed: %s", payment_status)
            raise PaymentError(
                "Payment was not completed (status: {}).".format(payment_status or "unknown")
            )

        return {
            "payment_id": payment.get("id"),
            "status": payment_status,
            "amount_cents": (payment.get("amount_money") or {}).get("amount"),
            "currency": (payment.get("amount_money") or {}).get("currency"),
            "receipt_url": payment.get("receipt_url"),
            "card_brand": ((payment.get("card_details") or {}).get("card") or {}).get(
                "card_brand"
            ),
            "last_4": ((payment.get("card_details") or {}).get("card") or {}).get("last_4"),
            "created_at": payment.get("created_at"),
        }


square_client = SquareClient()
