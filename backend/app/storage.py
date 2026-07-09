"""Photo storage — Cloudinary in production, local files as a fallback.

If CLOUDINARY_URL is set (Render env), photos are uploaded to Cloudinary and
`save_photo` returns a full https:// URL that survives redeploys. Otherwise
photos are written to the local UPLOAD_DIR and `save_photo` returns a bare
filename (served by the /photos static mount).

`photo_ref_to_url` turns whatever was stored back into something the browser can
load: a full URL is returned as-is; a bare filename becomes "/photos/<name>".
"""
from __future__ import annotations

import uuid
from pathlib import Path

from .config import UPLOAD_DIR, CLOUDINARY_URL

_cloudinary_ready = False

if CLOUDINARY_URL:
    try:
        import cloudinary
        import cloudinary.uploader  # noqa: F401  (registers uploader)

        # cloudinary reads CLOUDINARY_URL from the environment automatically.
        cloudinary.config(secure=True)
        _cloudinary_ready = True
    except Exception as e:  # pragma: no cover - only hit on misconfig
        print(f"[storage] Cloudinary configured but failed to init: {e}")
        _cloudinary_ready = False


def storage_backend() -> str:
    """Return 'cloudinary' or 'local' — handy for the /api/health check."""
    return "cloudinary" if _cloudinary_ready else "local"


def save_photo(raw: bytes, ext: str) -> str:
    """Persist image bytes and return a reference to store in the DB.

    Cloudinary  -> full secure URL (https://res.cloudinary.com/...).
    Local       -> bare filename (served at /photos/<filename>).
    """
    if _cloudinary_ready:
        import cloudinary.uploader

        public_id = uuid.uuid4().hex
        result = cloudinary.uploader.upload(
            raw,
            folder="findmypet",
            public_id=public_id,
            resource_type="image",
        )
        return result["secure_url"]

    # Local fallback
    fname = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / fname).write_bytes(raw)
    return fname


def photo_ref_to_url(ref: str | None) -> str | None:
    """Turn a stored photo reference into a browser-loadable URL."""
    if not ref:
        return None
    if ref.startswith("http://") or ref.startswith("https://"):
        return ref
    return f"/photos/{ref}"
