"""Minimal Square API client — checkout links + webhook signature verification.

Pure standard library (urllib + hmac/hashlib), no Square SDK, to stay light on
Render's free tier. Two jobs:

  1. create_payment_link(...) — call Square's Checkout API to mint a UNIQUE hosted
     checkout page per unlock, stamping our unlock reference into the order so the
     webhook can match the payment back to the exact unlock.
  2. verify_webhook_signature(...) — validate that an incoming webhook really came
     from Square (HMAC-SHA256 over notification_url + raw body, constant-time).

Everything degrades gracefully: if Square isn't configured, create_payment_link
raises SquareNotConfigured and callers fall back to the static link.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import urllib.request
import urllib.error

from .config import (
    SQUARE_ACCESS_TOKEN, SQUARE_LOCATION_ID, SQUARE_WEBHOOK_SIGNATURE_KEY,
    SQUARE_WEBHOOK_URL, SQUARE_REDIRECT_URL, square_api_base, square_api_configured,
)

SQUARE_VERSION = "2026-05-20"  # pin the API version we built against

# We prefix our unlock reference so the webhook can tell the two flows apart.
#   "pet:<id>"    -> a ContactUnlock (finder paid to see the owner)
#   "finder:<id>" -> a FinderUnlock  (owner paid to see the finders)
def make_ref(kind: str, unlock_id: int) -> str:
    return f"{kind}:{unlock_id}"


def parse_ref(ref: str) -> tuple[str, int] | None:
    """('pet'|'finder', unlock_id) from a reference string, or None if unusable."""
    if not ref or ":" not in ref:
        return None
    kind, _, num = ref.partition(":")
    if kind not in ("pet", "finder"):
        return None
    try:
        return kind, int(num)
    except ValueError:
        return None


class SquareError(RuntimeError):
    pass


class SquareNotConfigured(SquareError):
    pass


def create_payment_link(kind: str, unlock_id: int, amount_usd: float,
                        name: str, description: str = "") -> dict:
    """Create a unique Square hosted checkout page for one unlock.

    Returns {"url": <checkout url>, "payment_link_id": <id>, "order_id": <id|"">}.
    The unlock reference (make_ref) is stored as the order's reference_id AND in
    the payment_note, so the webhook can recover it from whichever Square sends.
    Raises SquareNotConfigured if the API creds are missing.
    """
    if not square_api_configured():
        raise SquareNotConfigured("Square API not configured")

    ref = make_ref(kind, unlock_id)
    amount_cents = int(round(float(amount_usd) * 100))
    # Idempotency key ties a retry to the same unlock so we never double-create.
    idem = f"fmp-{ref}"
    # payment_note is the key field: Square copies it onto the resulting Payment,
    # and the payment webhook includes that note — that's how we match the payment
    # back to this exact unlock. (quick_pay auto-builds the order, so we can't set
    # the order's reference_id directly; payment_note is the reliable carrier.)
    body = {
        "idempotency_key": idem,
        "quick_pay": {
            "name": name[:255],
            "price_money": {"amount": amount_cents, "currency": "USD"},
            "location_id": SQUARE_LOCATION_ID,
        },
        "checkout_options": {
            "redirect_url": SQUARE_REDIRECT_URL,
            "ask_for_shipping_address": False,
        },
        "description": description or ref,   # for our own dashboard readability
        "payment_note": ref,                 # <- matched by the webhook
    }

    data = _post("/v2/online-checkout/payment-links", body)
    link = data.get("payment_link", {}) or {}
    return {
        "url": link.get("url", ""),
        "payment_link_id": link.get("id", ""),
        "order_id": link.get("order_id", ""),
        "ref": ref,
    }


def get_order_reference(order_id: str) -> str:
    """Fetch an order and return its reference_id (our make_ref), or ''.

    Used by the webhook: a payment event carries an order_id; we look up the order
    to read the reference we stamped on it.
    """
    if not order_id or not square_api_configured():
        return ""
    try:
        data = _post(f"/v2/orders/batch-retrieve", {"order_ids": [order_id]})
        orders = data.get("orders", []) or []
        if orders:
            return orders[0].get("reference_id", "") or ""
    except SquareError:
        return ""
    return ""


def verify_webhook_signature(raw_body: bytes, signature_header: str,
                             notification_url: str | None = None) -> bool:
    """True if the webhook signature is valid for our signature key.

    Square computes HMAC-SHA256 over (notification_url + raw_request_body) using
    the subscription's signature key, base64-encodes it, and sends it in the
    x-square-hmacsha256-signature header. We recompute and compare in constant time.
    """
    key = SQUARE_WEBHOOK_SIGNATURE_KEY
    if not key or not signature_header:
        return False
    url = notification_url or SQUARE_WEBHOOK_URL
    mac = hmac.new(key.encode("utf-8"), digestmod=hashlib.sha256)
    mac.update(url.encode("utf-8"))
    mac.update(raw_body)
    expected = base64.b64encode(mac.digest()).decode("utf-8")
    return hmac.compare_digest(expected, signature_header)


# --------------------------------------------------------------------- internals
def _post(path: str, body: dict) -> dict:
    url = square_api_base() + path
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Authorization", f"Bearer {SQUARE_ACCESS_TOKEN}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Square-Version", SQUARE_VERSION)
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            pass
        raise SquareError(f"Square API {e.code} on {path}: {detail}") from e
    except Exception as e:  # network/timeout
        raise SquareError(f"Square API call failed on {path}: {e}") from e
