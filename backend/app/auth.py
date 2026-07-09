"""Minimal email/password auth with signed session tokens (no extra deps)."""
from __future__ import annotations

import hashlib
import hmac
import os
import base64
import json
import time

from .config import SECRET_KEY

_ITERATIONS = 120_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"pbkdf2${_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def _sign(data: bytes) -> str:
    sig = hmac.new(SECRET_KEY.encode(), data, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode().rstrip("=")


def make_token(user_id: int, days: int = 30) -> str:
    payload = {"uid": user_id, "exp": int(time.time()) + days * 86400}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{body}.{_sign(body.encode())}"


def read_token(token: str) -> int | None:
    """Return user_id if the token is valid and unexpired, else None."""
    try:
        body, sig = token.split(".")
        if not hmac.compare_digest(sig, _sign(body.encode())):
            return None
        pad = "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(body + pad))
        if payload["exp"] < time.time():
            return None
        return int(payload["uid"])
    except Exception:
        return None


# --- Password reset tokens ---------------------------------------------------
# Stateless & signed (no DB row needed). We bind the token to the user's current
# password hash, so once the password changes the token can't be reused.

def make_reset_token(user_id: int, password_hash: str, minutes: int = 30) -> str:
    """A short-lived, single-use-ish token emailed to the user."""
    # Tie the token to a fingerprint of the current password hash. After a
    # successful reset the hash changes, invalidating any older reset links.
    fp = hmac.new(SECRET_KEY.encode(), password_hash.encode(), hashlib.sha256).hexdigest()[:16]
    payload = {"uid": user_id, "typ": "reset", "fp": fp,
               "exp": int(time.time()) + minutes * 60}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{body}.{_sign(body.encode())}"


def read_reset_token(token: str, password_hash: str) -> int | None:
    """Return user_id if this reset token is valid for the given current hash."""
    try:
        body, sig = token.split(".")
        if not hmac.compare_digest(sig, _sign(body.encode())):
            return None
        pad = "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(body + pad))
        if payload.get("typ") != "reset" or payload["exp"] < time.time():
            return None
        fp = hmac.new(SECRET_KEY.encode(), password_hash.encode(), hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(payload.get("fp", ""), fp):
            return None  # password already changed since this link was issued
        return int(payload["uid"])
    except Exception:
        return None
